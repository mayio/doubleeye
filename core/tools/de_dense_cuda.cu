// Dense MASDA with the image plane on the GPU and the graph on the CPU.
//
// This is the split 10-architecture.md proposed: the GPU owns everything that is
// per-pixel and regular -- census, the graded cost, the recursive edge-aware
// filter, the running top-2 -- and hands the CPU a compact candidate buffer
// (two scored disparities per pixel), which is exactly what the MASDA solver
// consumes. The 46 ms the CPU spent producing candidates and merging them
// becomes GPU work; the solver itself is unchanged, shared through
// doubleeye/dense_solve.hpp.
//
// EVERY device computation replicates the CPU tool's integer arithmetic exactly:
// the census bit order, the Q14 score tables with C++ truncating division, the
// filter's int32-carry-int16-store recurrences, and the top-2's ascending-k
// strictly-greater insertion -- which on the GPU is a per-pixel scan and therefore
// DETERMINISTIC. So the whole binary is verified by cmp against de_dense
// --threads 1, not by eyeballing maps: same descriptors, same scores, same
// filtered planes, same candidates, same solve, same bytes out.
//
// TX2 only in practice (sm_62, CUDA 10.0); the Makefile builds this only where
// nvcc exists, so the desktop build is untouched.

#include "doubleeye/preproc.hpp"
#include "doubleeye/dense_solve.hpp"

#include <cuda_runtime.h>

#include <time.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace doubleeye;

namespace {

double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) * 1e3 + double(ts.tv_nsec) / 1e6;
}

#define CK(call)                                                          \
  do {                                                                    \
    cudaError_t e_ = (call);                                              \
    if (e_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,         \
                   __FILE__, __LINE__, cudaGetErrorString(e_));           \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// Q14 like the CPU. Kept as an enum so device code sees a compile-time constant.
enum { SCORE_Q = 14, SCORE_ONE = 1 << SCORE_Q };

__constant__ int16_t c_tbl[64];     // Census score per Hamming level, 49 used
__constant__ int16_t c_adt[256];    // truncated-absolute-difference score
__constant__ uint16_t c_rf[256];    // filter coefficient per byte difference

// --- census, 7x7 -> 48 bits, identical bit order to census_rows -----------------
//
// The CPU builds bit b in scan order dy = -3..3, dx = -3..3 skipping the centre,
// with bit b landing at position b. One thread per pixel does the same 48
// compares; the window reads hit L1 for the warp's neighbours.
__global__ void k_census(const uint8_t* img, uint64_t* out, int W, int H) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  const size_t i = size_t(y) * W + x;
  if (x < 3 || x >= W - 3 || y < 3 || y >= H - 3) { out[i] = 0; return; }
  const uint8_t c = img[i];
  uint64_t v = 0;
  int bit = 0;
  for (int dy = -3; dy <= 3; ++dy)
    for (int dx = -3; dx <= 3; ++dx) {
      if (dx == 0 && dy == 0) continue;
      v |= uint64_t(img[i + size_t(dy) * W + dx] < c ? 1u : 0u) << bit;
      ++bit;
    }
  out[i] = v;
}

// --- recursive-filter coefficients from raw bytes, same LUT as rf_coeffs --------
__global__ void k_rf_coeffs(const uint8_t* raw, uint16_t* ax, uint16_t* ay,
                            int W, int H) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  const size_t i = size_t(y) * W + x;
  ax[i] = x > 0 ? c_rf[abs(int(raw[i]) - int(raw[i - 1]))] : c_rf[0];
  ay[i] = y > 0 ? c_rf[abs(int(raw[i]) - int(raw[i - size_t(W)]))] : c_rf[0];
}

// --- graded cost: one thread per pixel, ALL disparities, transposed output ------
//
// The grid.z-per-disparity version re-read every census descriptor D times --
// 195 MB of the 450 MB the kernel moved -- and was measured at 15 ms. One thread
// per pixel keeps cl[i] and Ld[i] in registers and walks cr/Rd backwards through
// memory the warp has just used, so the reads collapse to one pass over each
// input.
//
// It writes the volume TRANSPOSED ([k][x][y]): the store at fixed k is consecutive
// in y for consecutive threads if threads map warp-major along y, and producing
// the transposed layout here deletes the 93 MB pre-transpose the filter otherwise
// needs before its horizontal passes. Outside the valid window the plane holds 0,
// exactly the value the CPU's plane clear leaves, because the filter propagates
// whatever it finds.
__global__ void k_score_all(const uint64_t* cl, const uint64_t* cr,
                            const uint8_t* Ld, const uint8_t* Rd,
                            int16_t* volT, int W, int H, int D, int dmin,
                            int32_t wq) {
  // 32x32 tile per plane, staged through shared memory like k_transpose: the
  // census/image reads are coalesced along x, the transposed volume writes are
  // coalesced along y, and neither the first version of this kernel (grid.z per
  // plane, reads coalesced, re-reads cl D times) nor the second (thread per
  // pixel, transposed writes, reads strided) had both -- measured 15 and 21 ms
  // against ~2 of raw traffic. The tile has both.
  __shared__ int16_t tile[32][33];
  const int k = blockIdx.z;
  const int d = dmin + k;
  const size_t WH = size_t(W) * H;
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  for (int j = 0; j < 32; j += 8) {
    const int yy = y + j;
    int16_t v = 0;
    if (x < W && yy < H && yy >= 3 && yy < H - 3 && x >= 3 + d && x < W - 3) {
      const size_t i = size_t(yy) * W + x;
      const int32_t c = c_tbl[__popcll(cl[i] ^ cr[i - size_t(d)])];
      const int32_t a = c_adt[abs(int(Ld[i]) - int(Rd[i - size_t(d)]))];
      v = int16_t((c * (1024 - wq) + a * wq) >> 10);
    }
    tile[threadIdx.y + j][threadIdx.x] = v;
  }
  __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x;                     // transposed coordinates
  y = blockIdx.x * 32 + threadIdx.y;
  int16_t* dst = volT + size_t(k) * WH;
  for (int j = 0; j < 32; j += 8)
    if (x < H && y + j < W) dst[size_t(y + j) * H + x] = tile[threadIdx.x][threadIdx.y + j];
}

// --- the recursive filter's horizontal passes, via transposition ----------------
//
// A thread per row is the natural mapping and it was measured at 403 ms: a warp of
// threads on consecutive rows strides W*2 bytes per lane, so every access is its
// own transaction and the sequential reuse thrashes out of the TX2's 512 KB L2
// before it pays. Transposing costs two extra passes of pure data movement -- no
// arithmetic, so bit-identity is untouched -- and turns the horizontal recurrence
// into the coalesced column-parallel form the vertical passes already have.
//
// The recurrence itself is exactly rf_horiz_i16: the carry is the int32, NOT a
// reload of the int16 store, and the backward pass reads the coefficient at i+1 --
// which in the transposed frame is one row down.
template <typename T>
__global__ void k_transpose(const T* in, T* out, int W, int H) {
  // 32x32 tiles through shared memory, +1 column to dodge bank conflicts.
  __shared__ T tile[32][33];
  const int k = blockIdx.z;
  const T* src = in + size_t(k) * W * H;
  T* dst = out + size_t(k) * W * H;
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  for (int j = 0; j < 32; j += 8)
    if (x < W && y + j < H)
      tile[threadIdx.y + j][threadIdx.x] = src[size_t(y + j) * W + x];
  __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x;   // transposed coordinates
  y = blockIdx.x * 32 + threadIdx.y;
  for (int j = 0; j < 32; j += 8)
    if (x < H && y + j < W)
      dst[size_t(y + j) * H + x] = tile[threadIdx.x][threadIdx.y + j];
}

// Runs on the TRANSPOSED volume: plane layout is [x][y], threads walk x (the
// original row direction) with consecutive threads on consecutive y -- coalesced.
__global__ void k_rf_horizT(int16_t* volT, const uint16_t* axT, int W, int H,
                            int ncols) {
  const int c0 = blockIdx.x * blockDim.x + threadIdx.x;
  if (c0 >= ncols) return;
  const int y = c0 % H;          // original row, now the minor axis
  const int k = c0 / H;
  int16_t* C = volT + size_t(k) * W * H;
  int32_t f = C[y];              // original (y, x = 0)
  for (int x = 1; x < W; ++x) {
    const size_t i = size_t(x) * H + y;
    const int32_t c = C[i];
    f = c + ((int32_t(axT[i]) * (f - c) + (1 << 14)) >> 15);
    C[i] = int16_t(f);
  }
  f = C[size_t(W - 1) * H + y];
  for (int x = W - 2; x >= 0; --x) {
    const size_t i = size_t(x) * H + y;
    const int32_t c = C[i];
    f = c + ((int32_t(axT[i + size_t(H)]) * (f - c) + (1 << 14)) >> 15);
    C[i] = int16_t(f);
  }
}

// --- vertical passes: one thread per (column, plane), coalesced -----------------
//
// The CPU's vertical pass reads the PREVIOUS row's already-updated int16 value,
// so the carry here is through the stored int16, not an int32 register. Copied
// faithfully; this is the kind of half-truth that would survive a visual check
// and die under cmp.
__global__ void k_rf_vert(int16_t* vol, const uint16_t* ay, int W, int H,
                          int ncols) {
  const int c0 = blockIdx.x * blockDim.x + threadIdx.x;   // column within all planes
  if (c0 >= ncols) return;
  const int x = c0 % W;
  const int k = c0 / W;
  int16_t* C = vol + size_t(k) * W * H;
  // The CPU reads the previous row's already-stored int16 -- but that value is
  // this thread's own previous store, so carrying it in a register is the SAME
  // int16, verified by cmp. It also removes a dependent global load per step, so
  // the recurrence's latency chain is arithmetic instead of memory: this one
  // change took the filter block from 37 ms to what the timing line now shows.
  int16_t prev = C[x];
  for (int y = 1; y < H; ++y) {
    const size_t i = size_t(y) * W + x;
    const int32_t c = C[i];
    prev = int16_t(c + ((int32_t(ay[i]) * (int32_t(prev) - c) + (1 << 14)) >> 15));
    C[i] = prev;
  }
  for (int y = H - 2; y >= 0; --y) {
    const size_t i = size_t(y) * W + x;
    const int32_t c = C[i];
    prev = int16_t(c + ((int32_t(ay[i + size_t(W)]) * (int32_t(prev) - c) +
                         (1 << 14)) >> 15));
    C[i] = prev;
  }
}

// --- top-2 per pixel over the filtered volume ------------------------------------
//
// Ascending k with strictly-greater displacement: the exact insertion order and
// tie rule of the CPU's single-threaded plane loop, which is what makes this
// deterministic where the CPU's six-thread version is not. Reads are coalesced
// (consecutive pixels, same plane). The window test mirrors the CPU insert:
// a pixel only ever sees disparities with x - d >= 3.
__global__ void k_top2(const int16_t* vol, float* cs, int* cd, int* cn,
                       int W, int H, int D, int dmin) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t WH = size_t(W) * H;
  if (i >= WH) return;
  const int x = int(i % W), y = int(i / W);
  int16_t b0 = -32768, b1 = -32768;
  int i0 = -1, i1 = -1;
  if (y >= 3 && y < H - 3 && x >= 3 + dmin && x < W - 3) {
    const int kmax = min(D - 1, x - 3 - dmin);
    for (int k = 0; k <= kmax; ++k) {
      const int16_t v = vol[size_t(k) * WH + i];
      if (v <= b1) continue;
      if (v > b0) { b1 = b0; i1 = i0; b0 = v; i0 = k; }
      else        { b1 = v;  i1 = k; }
    }
  }
  const float q = 1.f / float(SCORE_ONE);
  int n = 0;
  if (i0 >= 0) { cs[i * 2] = float(b0) * q; cd[i * 2] = i0; n = 1; }
  if (i1 >= 0) { cs[i * 2 + 1] = float(b1) * q; cd[i * 2 + 1] = i1; n = 2; }
  cn[i] = n;
}

float event_ms(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
        "usage: %s LEFT.y8 RIGHT.y8 W H [--dmax N] [--iters N] [--threads N]\n"
        "          [--min-margin F] [--sigma-s F] [--sigma-r F] [--ad F]\n"
        "          [--ad-trunc N] [--out disp.f32]\n"
        "  GPU census/cost/filter/top-2, CPU MASDA solve. Bit-identical to\n"
        "  de_dense --threads 1 by construction; verified with cmp, not maps.\n",
        argv[0]);
    return 2;
  }
  const std::string lp = argv[1], rp = argv[2];
  const int W = std::atoi(argv[3]), H = std::atoi(argv[4]);
  Cfg cfg;
  std::string outp;
  int frames = 1;
  for (int i = 5; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has = i + 1 < argc;
    if (a == "--dmax" && has) cfg.dmax = std::atoi(argv[++i]);
    else if (a == "--iters" && has) cfg.iters = std::atoi(argv[++i]);
    else if (a == "--threads" && has) cfg.threads = std::atoi(argv[++i]);
    else if (a == "--min-margin" && has) cfg.min_margin = float(std::atof(argv[++i]));
    else if (a == "--sigma-s" && has) cfg.sigma_s = float(std::atof(argv[++i]));
    else if (a == "--sigma-r" && has) cfg.sigma_r = float(std::atof(argv[++i]));
    else if (a == "--ad" && has) cfg.ad = float(std::atof(argv[++i]));
    else if (a == "--ad-trunc" && has) cfg.ad_trunc = std::atoi(argv[++i]);
    else if (a == "--out" && has) outp = argv[++i];
    else if (a == "--frames" && has) frames = std::max(1, std::atoi(argv[++i]));
    else if (a == "--agg" && has) ++i;   // accepted for flag-compat; rf ignores it
    else {
      std::fprintf(stderr, "unknown or unsupported flag: %s\n", a.c_str());
      return 2;
    }
  }
  Image8 L, R;
  if (!load_raw_y8(lp, W, H, &L) || !load_raw_y8(rp, W, H, &R)) {
    std::fprintf(stderr, "failed to load %s / %s at %dx%d\n",
                 lp.c_str(), rp.c_str(), W, H);
    return 1;
  }
  const int D = cfg.dmax - cfg.dmin + 1;
  const size_t WH = size_t(W) * H;
  int nthreads = cfg.threads > 0 ? cfg.threads
                                 : int(std::thread::hardware_concurrency());
  if (nthreads < 1) nthreads = 1;

  // --- host-built tables, identical expressions to de_dense -----------------------
  int16_t tbl[64] = {0};
  for (int h = 0; h <= 48; ++h) tbl[h] = int16_t(((24 - h) * SCORE_ONE) / 24);
  const int32_t wq = int32_t(std::max(0.f, std::min(1.f, cfg.ad)) * 1024.f);
  const int T = std::max(1, cfg.ad_trunc);
  int16_t adt[256];
  for (int v = 0; v < 256; ++v)
    adt[v] = int16_t(SCORE_ONE - (2 * SCORE_ONE * std::min(v, T)) / T);
  uint16_t rft[256];
  {
    const float kk = -std::sqrt(2.f) / cfg.sigma_s;
    const float rs = cfg.sigma_s / cfg.sigma_r;
    for (int d = 0; d < 256; ++d) {
      const float c = std::exp(kk * (1.f + rs * (float(d) / 255.f)));
      rft[d] = uint16_t(std::min(32767.f, c * 32768.f + 0.5f));
    }
  }

  const double t0 = now_ms();
  CK(cudaSetDevice(0));
  CK(cudaMemcpyToSymbol(c_tbl, tbl, sizeof(tbl)));
  CK(cudaMemcpyToSymbol(c_adt, adt, sizeof(adt)));
  CK(cudaMemcpyToSymbol(c_rf, rft, sizeof(rft)));

  uint8_t *dL, *dR;
  uint64_t *dcl, *dcr;
  uint16_t *dax, *day, *daxT;
  int16_t *dvol, *dvolT;
  CK(cudaMalloc(&dL, WH));
  CK(cudaMalloc(&dR, WH));
  CK(cudaMalloc(&dcl, WH * 8));
  CK(cudaMalloc(&dcr, WH * 8));
  CK(cudaMalloc(&dax, WH * 2));
  CK(cudaMalloc(&day, WH * 2));
  CK(cudaMalloc(&daxT, WH * 2));
  CK(cudaMalloc(&dvol, WH * size_t(D) * 2));
  CK(cudaMalloc(&dvolT, WH * size_t(D) * 2));
  // The candidates the CPU solver reads must be ORDINARY PAGEABLE MEMORY. The TX2
  // has no I/O coherency, so every cudaHostAlloc flavour -- Mapped AND Default --
  // is CPU-uncached there, and the solver on uncached candidates was measured
  // twice at ~300 ms against ~40 on cached (once via zero-copy, once via "cached
  // pinned" that Tegra does not actually have). Pageable D2H costs a staged
  // ~4 ms copy per frame, synchronous, paid at the pipeline's sync point where
  // the GPU is between frames anyway. Two host buffers so the pipelined mode can
  // solve frame t while frame t+1 computes.
  std::vector<float> cs[2] = {std::vector<float>(WH * 2),
                              std::vector<float>(WH * 2)};
  std::vector<int> cd[2] = {std::vector<int>(WH * 2), std::vector<int>(WH * 2)};
  std::vector<int> cn[2] = {std::vector<int>(WH), std::vector<int>(WH)};
  float* dcs;
  int *dcd, *dcn;
  CK(cudaMalloc(&dcs, WH * 2 * sizeof(float)));
  CK(cudaMalloc(&dcd, WH * 2 * sizeof(int)));
  CK(cudaMalloc(&dcn, WH * sizeof(int)));
  // Pageable host-to-device copies on Tegra were 7.4 ms for 0.8 MB of images.
  // cudaHostRegister is not supported here (CUDA 10.0 on Tegra returns
  // "operation not supported"), so the images go through pinned staging buffers
  // instead -- copied once at setup, uploaded from pinned at frame time.
  uint8_t *hL, *hR;
  CK(cudaHostAlloc(&hL, WH, cudaHostAllocDefault));
  CK(cudaHostAlloc(&hR, WH, cudaHostAllocDefault));
  std::memcpy(hL, L.data.data(), WH);
  std::memcpy(hR, R.data.data(), WH);
  const double t_alloc = now_ms() - t0;

  cudaEvent_t e0, e1, e2, e3, e4;
  for (cudaEvent_t* e : {&e0, &e1, &e2, &e3, &e4}) CK(cudaEventCreate(e));

  const double tg0 = now_ms();
  CK(cudaMemcpyAsync(dL, hL, WH, cudaMemcpyHostToDevice));
  CK(cudaMemcpyAsync(dR, hR, WH, cudaMemcpyHostToDevice));
  CK(cudaEventRecord(e0));

  const dim3 b2(32, 8);
  const dim3 g2((W + 31) / 32, (H + 7) / 8);
  k_census<<<g2, b2>>>(dL, dcl, W, H);
  k_census<<<g2, b2>>>(dR, dcr, W, H);
  k_rf_coeffs<<<g2, b2>>>(dL, dax, day, W, H);
  CK(cudaEventRecord(e1));

  const dim3 bs(32, 8);
  const dim3 gs((W + 31) / 32, (H + 31) / 32, D);
  k_score_all<<<gs, bs>>>(dcl, dcr, dL, dR, dvolT, W, H, D, cfg.dmin, wq);
  CK(cudaEventRecord(e2));

  {
    const dim3 tb(32, 8);
    k_transpose<uint16_t><<<dim3((W + 31) / 32, (H + 31) / 32, 1), tb>>>(dax, daxT,
                                                                         W, H);
    const int colsT = H * D;
    k_rf_horizT<<<(colsT + 127) / 128, 128>>>(dvolT, daxT, W, H, colsT);
    // transpose back: the source is now H x W per plane
    k_transpose<int16_t><<<dim3((H + 31) / 32, (W + 31) / 32, D), tb>>>(dvolT, dvol,
                                                                        H, W);
    const int cols = W * D;
    k_rf_vert<<<(cols + 127) / 128, 128>>>(dvol, day, W, H, cols);
  }
  CK(cudaEventRecord(e3));

  const size_t px = WH;
  k_top2<<<int((px + 255) / 256), 256>>>(dvol, dcs, dcd, dcn, W, H, D, cfg.dmin);
  CK(cudaEventRecord(e4));
  CK(cudaMemcpy(cs[0].data(), dcs, WH * 2 * sizeof(float),
                cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(cd[0].data(), dcd, WH * 2 * sizeof(int), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(cn[0].data(), dcn, WH * sizeof(int), cudaMemcpyDeviceToHost));
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  const double t_gpu = now_ms() - tg0;

  // --- the CPU keeps the graph -----------------------------------------------------
  std::vector<float> disp(WH, std::nanf("")), margin(WH, 0.f);
  auto solve_all = [&](const float* pcs, const int* pcd, const int* pcn) {
    std::vector<std::thread> pool;
    for (int t = 0; t < nthreads; ++t) pool.emplace_back([&, t]() {
      const int K = 2;
      std::vector<float> sbeta(size_t(W) * K), srho(size_t(W) * K);
      std::vector<int> bstart(size_t(W) + 1), bitems(size_t(W) * K), cursor(W);
      std::vector<float> best(W);
      std::vector<int> bestk(W), order(W);
      std::vector<char> taken(W);
      for (int y = t; y < H; y += nthreads)
        solve_row_sparse(W, D, K, cfg, nullptr,
                         const_cast<float*>(pcs) + size_t(y) * W * 2,
                         const_cast<int*>(pcd) + size_t(y) * W * 2,
                         const_cast<int*>(pcn) + size_t(y) * W,
                         sbeta.data(), srho.data(),
                         &disp[size_t(y) * W], &margin[size_t(y) * W], 3,
                         bstart.data(), bitems.data(), cursor.data(),
                         best.data(), bestk.data(), order.data(), taken.data());
    });
    for (auto& th : pool) th.join();
  };
  const double t1 = now_ms();
  solve_all(cs[0].data(), cd[0].data(), cn[0].data());
  const double t_solve = now_ms() - t1;

  // --- pipelined steady state: GPU on frame t+1 while the CPU solves frame t ------
  //
  // The production shape from 10-architecture.md, measured instead of claimed.
  // The same pair is fed N times; what is being measured is the overlap, and the
  // per-frame steady state is total wall over N with the first frame's fill-in
  // amortised. Every frame's output is identical by construction, and frame N-1's
  // solve result is what --out writes, so a broken overlap would break the
  // identity check rather than pass silently.
  double t_pipe = 0;
  if (frames > 1) {
    const double tp0 = now_ms();
    // frame 0's GPU pass
    auto gpu_pass = [&]() {
      CK(cudaMemcpyAsync(dL, hL, WH, cudaMemcpyHostToDevice));
      CK(cudaMemcpyAsync(dR, hR, WH, cudaMemcpyHostToDevice));
      k_census<<<g2, b2>>>(dL, dcl, W, H);
      k_census<<<g2, b2>>>(dR, dcr, W, H);
      k_rf_coeffs<<<g2, b2>>>(dL, dax, day, W, H);
      k_score_all<<<gs, bs>>>(dcl, dcr, dL, dR, dvolT, W, H, D, cfg.dmin, wq);
      const dim3 tb(32, 8);
      k_transpose<uint16_t><<<dim3((W + 31) / 32, (H + 31) / 32, 1), tb>>>(
          dax, daxT, W, H);
      const int colsT = H * D;
      k_rf_horizT<<<(colsT + 127) / 128, 128>>>(dvolT, daxT, W, H, colsT);
      k_transpose<int16_t><<<dim3((H + 31) / 32, (W + 31) / 32, D), tb>>>(
          dvolT, dvol, H, W);
      const int cols = W * D;
      k_rf_vert<<<(cols + 127) / 128, 128>>>(dvol, day, W, H, cols);
      k_top2<<<int((WH + 255) / 256), 256>>>(dvol, dcs, dcd, dcn, W, H, D,
                                             cfg.dmin);
    };
    auto fetch = [&](int buf) {
      CK(cudaMemcpy(cs[buf].data(), dcs, WH * 2 * sizeof(float),
                    cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(cd[buf].data(), dcd, WH * 2 * sizeof(int),
                    cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(cn[buf].data(), dcn, WH * sizeof(int),
                    cudaMemcpyDeviceToHost));
    };
    gpu_pass();
    fetch(0);                                // implicit sync: pageable D2H
    for (int f = 1; f < frames; ++f) {
      gpu_pass();                            // async: GPU works on frame f
      solve_all(cs[(f - 1) % 2].data(), cd[(f - 1) % 2].data(),
                cn[(f - 1) % 2].data());     // CPU solves frame f-1 meanwhile
      fetch(f % 2);                          // syncs the stream, then copies
    }
    solve_all(cs[(frames - 1) % 2].data(), cd[(frames - 1) % 2].data(),
              cn[(frames - 1) % 2].data());
    t_pipe = (now_ms() - tp0) / frames;
  }

  size_t filled = 0;
  for (float v : disp) if (std::isfinite(v)) ++filled;
  std::printf("%dx%d  D=%d  iters=%d  threads=%d  (GPU cost, CPU solve)\n",
              W, H, D, cfg.iters, nthreads);
  std::printf("  gpu breakdown: upload+readback %.1f  census+coeffs %.1f  "
              "score %.1f  filter %.1f  top2 %.1f ms   (alloc %.1f, once)\n",
              t_gpu - event_ms(e0, e4), event_ms(e0, e1), event_ms(e1, e2),
              event_ms(e2, e3), event_ms(e3, e4), t_alloc);
  if (frames > 1)
    std::printf("  pipelined over %d frames: %.1f ms/frame steady state (%.1f Hz)\n",
                frames, t_pipe, 1000.0 / t_pipe);
  std::printf("census %.1f ms  cost %.1f ms  solve %.1f ms  total %.1f ms  "
              "agg=%d iters=%d  filled %.1f%%\n",
              event_ms(e0, e1), event_ms(e1, e4), t_solve,
              t_gpu + t_solve, cfg.agg, cfg.iters,
              100.0 * double(filled) / double(disp.size()));
  if (!outp.empty()) {
    FILE* f = std::fopen(outp.c_str(), "wb");
    if (f) { std::fwrite(disp.data(), sizeof(float), disp.size(), f); std::fclose(f); }
  }
  return 0;
}
