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

#include <pthread.h>
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

// --- the k-minor volume: vol[y][x][k], Dpad = 64 so every pixel's disparity run
// is one 128-byte-aligned line ------------------------------------------------------
//
// This is the [x][d] layout the CPU work rejected three times, and on the GPU it is
// the right answer -- the resolution of that whole saga is that the two machines
// want OPPOSITE layouts. Every access below is either a coalesced k-run (32
// consecutive disparities per warp) or a warp broadcast, and the score fuses with
// the horizontal-forward pass so the scored-but-unfiltered volume never makes a
// round trip through DRAM.
//
// A warp is 32 DISPARITIES of one image row (ReS2tAC's lane assignment, third
// appearance in this project). The census reads for 32 consecutive d form one
// 256-byte window that slides one element per x step, so it lives in L1; cl and
// ax are the same address for every lane, a hardware broadcast.
//
// The alternative that was measured first and recorded as a negative: a
// warp-serial shuffle scan of the recurrence (lanes take turns) is bit-exact and
// coalesced and 55.7 ms, WORSE than the transposed thread-per-row version,
// because 31 of 32 lanes burn issue slots on every serial step. Instruction
// throughput, not bandwidth. Block-parallel reassociation (Nehab et al.) fixes
// that for float filters and cannot be bit-exact for this integer recurrence.
//
// The recurrences themselves are exactly rf_horiz_i16 / rf_filter_i16: int32
// carry, int16 store, +-(1<<14) rounding, backward coefficient at +1.

// The k-run padding is RUNTIME: Dpad = D rounded up to 64, so a warp of k-pairs
// covers each 64-wide group and a pixel's run stays line-aligned. The first
// version hard-coded 64 and was caught by the eight-scene identity check: six of
// the eight Middlebury scenes have D > 64, their upper disparities were never
// scored, and the top-2 read into the next pixel's run. teddy and cones -- the
// two scenes everything gets tuned on -- happen to have D = 60 and passed.

// score + horizontal FORWARD pass, fused. Grid: (H, D/32). Warp w of a block
// covers row y = blockIdx.x * (blockDim.x/32/...) -- one row per block of 2 warps.
__global__ void k_score_hfwd(const uint64_t* cl, const uint64_t* cr,
                             const uint8_t* Ld, const uint8_t* Rd,
                             const uint16_t* ax,
                             int16_t* vol, int W, int H, int D, int Dpad,
                             int dmin, int32_t wq) {
  // The tables are indexed DIVERGENTLY -- 32 different Hamming distances per
  // warp -- and constant memory serializes on divergent access: one replay per
  // distinct address. Shared memory has no such rule; copying the tables once
  // per block was measured at 11.2 -> 6.0 ms for this kernel.
  __shared__ int16_t stbl[64];
  __shared__ int16_t sadt[256];
  for (int t = threadIdx.x; t < 64; t += blockDim.x) stbl[t] = c_tbl[t];
  for (int t = threadIdx.x; t < 256; t += blockDim.x) sadt[t] = c_adt[t];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int kg = threadIdx.x >> 5;          // k-group within this 64-wide block
  const int y = blockIdx.x;
  const int k = blockIdx.y * 64 + kg * 32 + lane;
  const int d = dmin + k;
  const bool live = k < D;
  const bool row_ok = y >= 3 && y < H - 3;
  const uint64_t* clr = cl + size_t(y) * W;
  const uint64_t* crr = cr + size_t(y) * W;
  const uint8_t* Lr = Ld + size_t(y) * W;
  const uint8_t* Rr = Rd + size_t(y) * W;
  const uint16_t* ar = ax + size_t(y) * W;
  int16_t* out = vol + size_t(y) * W * Dpad + k;
  int32_t f = 0;                             // x = 0 score is 0 by the window
  for (int x = 0; x < W; ++x) {
    int32_t v = 0;
    if (live && row_ok && x >= 3 + d && x < W - 3) {
      const int32_t c = stbl[__popcll(__ldg(clr + x) ^ __ldg(crr + x - d))];
      const int32_t a = sadt[abs(int(__ldg(Lr + x)) - int(__ldg(Rr + x - d)))];
      v = (c * (1024 - wq) + a * wq) >> 10;
    }
    // forward recurrence, seeded with the x = 0 value exactly like the CPU
    if (x == 0) f = v;
    else        f = v + ((int32_t(ar[x]) * (f - v) + (1 << 14)) >> 15);
    if (live) out[size_t(x) * Dpad] = int16_t(f);
  }
}

// horizontal BACKWARD pass over the forward result. Same warp shape.
__global__ void k_hbwd(int16_t* vol, const uint16_t* ax, int W, int H,
                       int Dpad) {
  // A k-pair per lane: one int32 load/store per step covers a 64-wide k group
  // per warp -- full 128-byte transactions. grid.y walks the groups.
  const int lane = threadIdx.x & 31;
  const int y = blockIdx.x;
  const int k2 = blockIdx.y * 64 + lane * 2;
  const uint16_t* ar = ax + size_t(y) * W;
  int32_t* row = (int32_t*)(vol + size_t(y) * W * Dpad + k2);
  const size_t xs = size_t(Dpad) / 2;        // x stride in int32 units
  int32_t packed = row[size_t(W - 1) * xs];
  int16_t f0 = int16_t(packed & 0xFFFF), f1 = int16_t(packed >> 16);
  for (int x = W - 2; x >= 0; --x) {
    const int32_t cc = row[size_t(x) * xs];
    const int32_t a = ar[x + 1];
    const int32_t c0 = int16_t(cc & 0xFFFF), c1 = int16_t(cc >> 16);
    f0 = int16_t(c0 + ((a * (int32_t(f0) - c0) + (1 << 14)) >> 15));
    f1 = int16_t(c1 + ((a * (int32_t(f1) - c1) + (1 << 14)) >> 15));
    row[size_t(x) * xs] = (int32_t(uint16_t(f1)) << 16) | uint16_t(f0);
  }
}

// vertical passes: thread per (x, k), warp = 32 consecutive k of one x. The
// coefficient ay[y][x] is one address for the whole warp -- a broadcast -- and the
// volume access is a coalesced k-run. Register carry, exactly like the CPU's
// stored-int16 carry in value.
// vertical FORWARD: a thread owns a PAIR of adjacent k (one int32 load/store), a
// warp owns all 64 padded k of one x -- full 128-byte transactions.
template <int G>                             // 64-wide k groups; 1 covers D <= 64
__global__ void k_vert_fwd(int16_t* vol, const uint16_t* ay, int W, int H,
                           int Dpad, int nthreads_total) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nthreads_total) return;
  const int lane = t & 31;
  const int x = t >> 5;
  const size_t rs = size_t(W) * Dpad / 2;    // row stride in int32 units
  int32_t* col0 = (int32_t*)(vol + size_t(x) * Dpad) + lane;
  int16_t p0[4], p1[4];
  for (int g = 0; g < G; ++g) {
    const int32_t packed = col0[g * 32];
    p0[g] = int16_t(packed & 0xFFFF);
    p1[g] = int16_t(packed >> 16);
  }
  for (int y = 1; y < H; ++y) {
    const int32_t a = ay[size_t(y) * W + x];   // one address per warp: broadcast
    for (int g = 0; g < G; ++g) {
      const int32_t cc = col0[size_t(y) * rs + g * 32];
      const int32_t c0 = int16_t(cc & 0xFFFF), c1 = int16_t(cc >> 16);
      p0[g] = int16_t(c0 + ((a * (int32_t(p0[g]) - c0) + (1 << 14)) >> 15));
      p1[g] = int16_t(c1 + ((a * (int32_t(p1[g]) - c1) + (1 << 14)) >> 15));
      col0[size_t(y) * rs + g * 32] =
          (int32_t(uint16_t(p1[g])) << 16) | uint16_t(p0[g]);
    }
  }
}

// vertical BACKWARD with the top-2 fused in as a warp reduction, and NO stores of
// the filtered values: after this pass nothing reads the volume again, so the
// only output is the candidates. This deleted a separate 10.9 ms top-2 kernel
// whose per-pixel k-runs strided 128 bytes across the warp, plus ~50 MB of
// backward stores.
//
// A warp holds ALL of pixel (y, x)'s disparities (32 lanes x 2 each), so the
// exact sequential top-2 -- ascending k, strictly greater displaces, first of
// equals kept -- is reproduced by a lane-local top-2 in k order followed by a
// shuffle merge that prefers the smaller k on equal values. Top-2 of a multiset
// with stable-by-k tie order is associative, which is what makes the reduction
// exact rather than approximately right; cmp against the CPU is the referee.
template <int G>
// `cnb`, when non-null, receives the filtered cost one disparity either side of the
// winning candidate, for the host's sub-pixel fit. On the GPU this is exact and
// nearly free: the whole disparity range for this pixel is live in registers across
// the warp at the moment the argmax is known, so both neighbours are one broadcast
// and one shuffle away. The CPU tool works much harder for the same answer -- it
// streams planes and keeps a 3-wide window around the running best.
//
// Bit-identity survives --subpixel, which was not obvious and is the reason it was
// checked rather than assumed. The CPU's window falls back to integer at the edges
// of a thread's chunk, so a MULTI-threaded CPU run genuinely has fewer fits than
// this kernel -- but the reference is `de_dense --threads 1`, where one thread walks
// every chunk consecutively and never loses a neighbour. Verified on teddy and Art:
// coverage identical, every fitted pixel identical, and the 4.3% / 8.2% the CPU
// leaves integer are the legitimate refusals (no vertex, or k at the range edge)
// which this kernel refuses too.
__global__ void k_vert_bwd_top2(const int16_t* vol, const uint16_t* ay,
                                float* cs, int* cd, int* cn, int16_t* cnb,
                                int W, int H, int D, int Dpad, int dmin,
                                int nthreads_total) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nthreads_total) return;
  const int lane = t & 31;
  const int x = t >> 5;
  const int32_t* col0 = (const int32_t*)(vol + size_t(x) * Dpad) + lane;
  const size_t rs = size_t(W) * Dpad / 2;
  const float q = 1.f / float(SCORE_ONE);
  // rows the backward pass never visits still need cn = 0 for the solver
  if (lane == 0) cn[size_t(H - 1) * W + x] = 0;
  int16_t p0[4], p1[4];
  for (int g = 0; g < G; ++g) {
    const int32_t packed = col0[size_t(H - 1) * rs + g * 32];
    p0[g] = int16_t(packed & 0xFFFF);
    p1[g] = int16_t(packed >> 16);
  }
  for (int y = H - 2; y >= 0; --y) {
    const int32_t a = ay[size_t(y + 1) * W + x];
    const int kmax = (y >= 3 && y < H - 3 && x >= 3 + dmin && x < W - 3)
                         ? min(D - 1, x - 3 - dmin) : -1;
    // lane-local top-2 as packed (value, k) ints: value+32768 in the high bits,
    // 255-k in the low 8, so plain integer > is (value desc, k asc) and the
    // whole reduction is order-independent. 8 bits of k because D can reach 220.
    int32_t pk0 = 0, pk1 = 0;
    for (int g = 0; g < G; ++g) {
      const int32_t cc = col0[size_t(y) * rs + g * 32];
      const int32_t c0 = int16_t(cc & 0xFFFF), c1 = int16_t(cc >> 16);
      p0[g] = int16_t(c0 + ((a * (int32_t(p0[g]) - c0) + (1 << 14)) >> 15));
      p1[g] = int16_t(c1 + ((a * (int32_t(p1[g]) - c1) + (1 << 14)) >> 15));
      const int k2 = g * 64 + lane * 2;
      if (k2 <= kmax) {
        const int32_t c = ((int32_t(p0[g]) + 32768) << 8) | (255 - k2);
        if (c > pk0)      { pk1 = pk0; pk0 = c; }
        else if (c > pk1) { pk1 = c; }
      }
      if (k2 + 1 <= kmax) {
        const int32_t c = ((int32_t(p1[g]) + 32768) << 8) | (255 - k2 - 1);
        if (c > pk0)      { pk1 = pk0; pk0 = c; }
        else if (c > pk1) { pk1 = c; }
      }
    }
    // The merge order across the shfl_down tree is arbitrary -- lane 0 merges
    // lane 16 before lane 1 -- which is exactly why the comparison carries k
    // explicitly in the packing: the first version assumed "other side has
    // larger k", produced 10 wrong pixels in 407k (all exact ties), and a
    // 5-million-run host simulation of the tree pinned it (top2sim). Packed,
    // the merge is order-independent and 2 shuffles per round.
    for (int off = 16; off > 0; off >>= 1) {
      const int32_t o0 = __shfl_down_sync(0xffffffffu, pk0, off);
      const int32_t o1 = __shfl_down_sync(0xffffffffu, pk1, off);
      if (o0 > pk0) {
        pk1 = pk0 > o1 ? pk0 : o1;
        pk0 = o0;
      } else if (o0 > pk1) {
        pk1 = o0;
      }
    }
    const int16_t b0 = int16_t((pk0 >> 8) - 32768);
    const int i0 = pk0 ? 255 - (pk0 & 255) : -1;
    const int16_t b1 = int16_t((pk1 >> 8) - 32768);
    const int i1 = pk1 ? 255 - (pk1 & 255) : -1;
    if (cnb) {
      // The reduction leaves the answer in lane 0, so the winner has to come back
      // out before the other lanes can say whether they hold its neighbours.
      const int kw = __shfl_sync(0xffffffffu, i0, 0);
      const int km = kw - 1, kp = kw + 1;
      int16_t mym = -32768, myp = -32768;
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const int k2 = g * 64 + lane * 2;
        if (k2 <= kmax) {
          if (k2 == km) mym = p0[g];
          if (k2 == kp) myp = p0[g];
        }
        if (k2 + 1 <= kmax) {
          if (k2 + 1 == km) mym = p1[g];
          if (k2 + 1 == kp) myp = p1[g];
        }
      }
      // Exactly one lane owns a given k, and which one is arithmetic rather than a
      // search: k's lane is (k % 64) / 2 whatever group it fell in. So this is two
      // shuffles, not two more reduction trees.
      const int16_t vm = int16_t(__shfl_sync(0xffffffffu, int(mym), (km & 63) >> 1));
      const int16_t vp = int16_t(__shfl_sync(0xffffffffu, int(myp), (kp & 63) >> 1));
      if (lane == 0) {
        const size_t i = size_t(y) * W + x;
        const bool ok = (kw >= 0);
        // Left in Q14 and dequantised by the solve thread. The bus is the scarce
        // thing here, not the multiply: this is a pageable staged copy on a board
        // with no I/O coherency, and float doubled it for no added information --
        // the device never had more than int16 to give.
        cnb[i * 2]     = (ok && km >= 0) ? vm : int16_t(-32768);
        cnb[i * 2 + 1] = ok ? vp : int16_t(-32768);
      }
    }
    if (lane == 0) {
      const size_t i = size_t(y) * W + x;
      int n = 0;
      if (i0 >= 0) { cs[i * 2] = float(b0) * q; cd[i * 2] = i0; n = 1; }
      if (i1 >= 0) { cs[i * 2 + 1] = float(b1) * q; cd[i * 2 + 1] = i1; n = 2; }
      cn[i] = n;
    }
  }
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
  std::string outp, kpp;
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
    // lambda = clutter (a LEFT pixel left unmatched), gamma = misdetection (a RIGHT
    // pixel left unclaimed). Hand-set at -0.1 since the solver was written and never
    // swept; they are the two parameters that price coverage directly, which is the
    // axis SGM still leads on. Exposed so that can be measured rather than assumed.
    else if (a == "--lambda" && has) cfg.lambda = float(std::atof(argv[++i]));
    else if (a == "--gamma" && has) cfg.gamma = float(std::atof(argv[++i]));
    else if (a == "--damping" && has) cfg.damping = float(std::atof(argv[++i]));
    else if (a == "--out" && has) outp = argv[++i];
    else if (a == "--keypoints" && has) kpp = argv[++i];
    else if (a == "--frames" && has) frames = std::max(1, std::atoi(argv[++i]));
    else if (a == "--subpixel") cfg.subpixel = true;   // now the default
    else if (a == "--no-subpixel") cfg.subpixel = false;
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
  uint16_t *dax, *day;
  int16_t* dvol;
  CK(cudaMalloc(&dL, WH));
  CK(cudaMalloc(&dR, WH));
  CK(cudaMalloc(&dcl, WH * 8));
  CK(cudaMalloc(&dcr, WH * 8));
  CK(cudaMalloc(&dax, WH * 2));
  CK(cudaMalloc(&day, WH * 2));
  const int Dpad = (D + 63) & ~63;             // k-run padding, 64-aligned
  CK(cudaMalloc(&dvol, WH * size_t(Dpad) * 2));   // k-minor, padded runs
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
  // Same double-buffering and the same pageable-memory rule as the candidates:
  // the solver reads this on the CPU, so it must not be cudaHostAlloc'd.
  std::vector<int16_t> cnb[2] = {std::vector<int16_t>(cfg.subpixel ? WH * 2 : 0),
                                 std::vector<int16_t>(cfg.subpixel ? WH * 2 : 0)};
  float* dcs;
  int *dcd, *dcn;
  CK(cudaMalloc(&dcs, WH * 2 * sizeof(float)));
  CK(cudaMalloc(&dcd, WH * 2 * sizeof(int)));
  CK(cudaMalloc(&dcn, WH * sizeof(int)));
  int16_t* dcnb = nullptr;
  if (cfg.subpixel) CK(cudaMalloc(&dcnb, WH * 2 * sizeof(int16_t)));
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

  k_score_hfwd<<<dim3(H, Dpad / 64), 64>>>(dcl, dcr, dL, dR, dax, dvol, W, H, D,
                                           Dpad, cfg.dmin, wq);
  CK(cudaEventRecord(e2));

  cudaEvent_t f1;
  CK(cudaEventCreate(&f1));
  const int vthreads = W * 32;               // one warp per x, k pairs per lane
  const int vblocks = (vthreads + 255) / 256;
  // G as a template parameter so the per-y group loop unrolls; the runtime
  // version cost 1.3 ms/frame of the 30 Hz margin at G = 1.
  auto vert_fwd = [&]() {
    switch (Dpad >> 6) {
      case 1: k_vert_fwd<1><<<vblocks, 256>>>(dvol, day, W, H, Dpad, vthreads); break;
      case 2: k_vert_fwd<2><<<vblocks, 256>>>(dvol, day, W, H, Dpad, vthreads); break;
      case 3: k_vert_fwd<3><<<vblocks, 256>>>(dvol, day, W, H, Dpad, vthreads); break;
      default: k_vert_fwd<4><<<vblocks, 256>>>(dvol, day, W, H, Dpad, vthreads);
    }
  };
  auto vert_bwd_top2 = [&]() {
    switch (Dpad >> 6) {
      case 1: k_vert_bwd_top2<1><<<vblocks, 256>>>(dvol, day, dcs, dcd, dcn, dcnb, W, H,
                                                   D, Dpad, cfg.dmin, vthreads); break;
      case 2: k_vert_bwd_top2<2><<<vblocks, 256>>>(dvol, day, dcs, dcd, dcn, dcnb, W, H,
                                                   D, Dpad, cfg.dmin, vthreads); break;
      case 3: k_vert_bwd_top2<3><<<vblocks, 256>>>(dvol, day, dcs, dcd, dcn, dcnb, W, H,
                                                   D, Dpad, cfg.dmin, vthreads); break;
      default: k_vert_bwd_top2<4><<<vblocks, 256>>>(dvol, day, dcs, dcd, dcn, dcnb, W, H,
                                                    D, Dpad, cfg.dmin, vthreads);
    }
  };
  {
    k_hbwd<<<dim3(H, Dpad / 64), 32>>>(dvol, dax, W, H, Dpad);
    CK(cudaEventRecord(f1));
    vert_fwd();
  }
  CK(cudaEventRecord(e3));

  vert_bwd_top2();
  CK(cudaEventRecord(e4));
  CK(cudaMemcpy(cs[0].data(), dcs, WH * 2 * sizeof(float),
                cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(cd[0].data(), dcd, WH * 2 * sizeof(int), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(cn[0].data(), dcn, WH * sizeof(int), cudaMemcpyDeviceToHost));
  if (cfg.subpixel)
    CK(cudaMemcpy(cnb[0].data(), dcnb, WH * 2 * sizeof(int16_t),
                  cudaMemcpyDeviceToHost));
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  const double t_gpu = now_ms() - tg0;

  // --- the CPU keeps the graph -----------------------------------------------------
  std::vector<float> disp(WH, std::nanf("")), margin(WH, 0.f);
  // Solve threads are PINNED to the A57 cluster (cores 0, 3, 4, 5 on the TX2).
  // Unpinned, the scheduler mixes in the two Denver cores and the solve wanders
  // 30-45 ms run to run; pinned it sits near its minimum. The Denvers are left
  // to the CUDA driver and the fetch thread.
  static const int A57[] = {0, 3, 4, 5};
  auto solve_all = [&](const float* pcs, const int* pcd, const int* pcn,
                       const int16_t* pcnb) {
    std::vector<std::thread> pool;
    for (int t = 0; t < nthreads; ++t) pool.emplace_back([&, t]() {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(A57[t & 3], &set);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
      const int K = 2;
      std::vector<float> sbeta(size_t(W) * K), srho(size_t(W) * K);
      std::vector<int> bstart(size_t(W) + 1), bitems(size_t(W) * K), cursor(W);
      std::vector<float> best(W);
      std::vector<int> bestk(W), order(W);
      std::vector<char> taken(W);
      std::vector<float> nb(pcnb ? size_t(W) * 2 : 0);
      const float qs = 1.f / float(SCORE_ONE);
      for (int y = t; y < H; y += nthreads) {
        if (pcnb) {
          const int16_t* src = pcnb + size_t(y) * W * 2;
          for (int j = 0; j < W * 2; ++j)
            nb[j] = (src[j] == int16_t(-32768)) ? -1e30f : float(src[j]) * qs;
        }
        solve_row_sparse(W, D, K, cfg, nullptr,
                         const_cast<float*>(pcs) + size_t(y) * W * 2,
                         const_cast<int*>(pcd) + size_t(y) * W * 2,
                         const_cast<int*>(pcn) + size_t(y) * W,
                         sbeta.data(), srho.data(),
                         &disp[size_t(y) * W], &margin[size_t(y) * W], 3,
                         bstart.data(), bitems.data(), cursor.data(),
                         best.data(), bestk.data(), order.data(), taken.data(),
                         pcnb ? nb.data() : nullptr);
      }
    });
    for (auto& th : pool) th.join();
  };
  const double t1 = now_ms();
  solve_all(cs[0].data(), cd[0].data(), cn[0].data(),
            cfg.subpixel ? cnb[0].data() : nullptr);
  const double t_solve = now_ms() - t1;

  // --- pipelined steady state: GPU on frame t+1 while the CPU solves frame t ------
  //
  // The production shape from 10-architecture.md, measured instead of claimed.
  // The same pair is fed N times; what is being measured is the overlap, and the
  // per-frame steady state is total wall over N with the first frame's fill-in
  // amortised. Every frame's output is identical by construction, and frame N-1's
  // solve result is what --out writes, so a broken overlap would break the
  // identity check rather than pass silently.
  DetectorConfig dcfg;
  std::vector<Keypoint> kps_live;
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
      k_score_hfwd<<<dim3(H, Dpad / 64), 64>>>(dcl, dcr, dL, dR, dax, dvol, W,
                                               H, D, Dpad, cfg.dmin, wq);
      k_hbwd<<<dim3(H, Dpad / 64), 32>>>(dvol, dax, W, H, Dpad);
      vert_fwd();
      vert_bwd_top2();
    };
    auto fetch = [&](int buf) {
      CK(cudaMemcpy(cs[buf].data(), dcs, WH * 2 * sizeof(float),
                    cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(cd[buf].data(), dcd, WH * 2 * sizeof(int),
                    cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(cn[buf].data(), dcn, WH * sizeof(int),
                    cudaMemcpyDeviceToHost));
      if (cfg.subpixel)
        CK(cudaMemcpy(cnb[buf].data(), dcnb, WH * 2 * sizeof(int16_t),
                      cudaMemcpyDeviceToHost));
    };
    gpu_pass();
    fetch(0);                                // implicit sync: pageable D2H
    for (int f = 1; f < frames; ++f) {
      gpu_pass();                            // async: GPU works on frame f
      // The fetch runs on ITS OWN thread: cudaMemcpy serializes with the stream,
      // so it waits out frame f's kernels and copies while the solve of frame
      // f-1 is still on the A57s -- the 4 ms staged copy disappears into the
      // solve instead of sitting serially in the loop.
      std::thread fetcher([&, f]() { fetch(f % 2); });
      // Detection, when the sparse feature set is wanted, runs as a THIRD thread
      // beside the solve. It is the open budget question in 0.4: detection is ~29 ms
      // of one core against 26 ms of kernels, so whether keypoints are free depends
      // entirely on this overlap and not on anything that can be argued. Inside the
      // loop it is measured; bolted on after it, it was not.
      std::thread detector;
      if (!kpp.empty())
        detector = std::thread([&]() { kps_live = detect_keypoints_fast(L, dcfg); });
      solve_all(cs[(f - 1) % 2].data(), cd[(f - 1) % 2].data(),
                cn[(f - 1) % 2].data(),
                cfg.subpixel ? cnb[(f - 1) % 2].data() : nullptr);     // CPU solves frame f-1 meanwhile
      if (detector.joinable()) detector.join();
      fetcher.join();
    }
    solve_all(cs[(frames - 1) % 2].data(), cd[(frames - 1) % 2].data(),
              cn[(frames - 1) % 2].data(),
              cfg.subpixel ? cnb[(frames - 1) % 2].data() : nullptr);
    t_pipe = (now_ms() - tp0) / frames;
  }

  size_t filled = 0;
  for (float v : disp) if (std::isfinite(v)) ++filled;
  std::printf("%dx%d  D=%d  iters=%d  threads=%d  (GPU cost, CPU solve)\n",
              W, H, D, cfg.iters, nthreads);
  std::printf("  gpu breakdown: upload+readback %.1f  census+coeffs %.1f  "
              "score+hfwd %.1f  hbwd+vfwd %.1f (hbwd %.1f, vfwd %.1f)  "
              "vbwd+top2 %.1f ms   (alloc %.1f, once)\n",
              t_gpu - event_ms(e0, e4), event_ms(e0, e1), event_ms(e1, e2),
              event_ms(e2, e3), event_ms(e2, f1), event_ms(f1, e3),
              event_ms(e3, e4), t_alloc);
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

  // --- the sparse feature set, read out of the dense map ---------------------------
  //
  // Section 0 wants ONE producer for the sparse features, and 0.4 measured that the
  // dense map beats the sparse matcher at its own keypoints on every axis -- 0.853
  // precision against 0.706, with 57% more correct matches, and no matcher to run.
  // So the keypoints do not need matching, only detecting and looking up.
  //
  // Detection is the whole added cost and it is CPU. It is timed separately and
  // printed, because whether it fits alongside the solve is the open budget question
  // in 0.4 and nobody should have to guess at it.
  if (!kpp.empty()) {
    const double tk0 = now_ms();
    const std::vector<Keypoint> kl =
        kps_live.empty() ? detect_keypoints_fast(L, dcfg) : kps_live;
    const double tk1 = now_ms();
    FILE* f = std::fopen(kpp.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", kpp.c_str()); return 1; }
    std::fprintf(f, "x,y,disparity,margin\n");
    int n = 0;
    for (const Keypoint& k : kl) {
      const int x = std::min(std::max(int(k.x + 0.5f), 0), W - 1);
      const int y = std::min(std::max(int(k.y + 0.5f), 0), H - 1);
      const float d = disp[size_t(y) * size_t(W) + size_t(x)];
      if (!(d > 0.f)) continue;          // the map has a hole here; emit nothing
      std::fprintf(f, "%.2f,%.2f,%.4f,%.4f\n", k.x, k.y, d,
                   margin[size_t(y) * size_t(W) + size_t(x)]);
      ++n;
    }
    std::fclose(f);
    std::printf("keypoints: %zu detected in %.1f ms, %d carried a disparity "
                "(%.1f%%) -> %s\n", kl.size(), tk1 - tk0, n,
                kl.empty() ? 0.0 : 100.0 * double(n) / double(kl.size()),
                kpp.c_str());
  }
  return 0;
}
