// Dense MASDA in C++: every pixel, every disparity, threaded over rows.
//
// The NumPy version takes 21 s per 450x375 pair. Almost all of that is scatter --
// np.maximum.at over a 9.15M-entry edge list, which is a gather/scatter with an
// indirection per element. The structure does not need any of that.
//
// On a dense disparity grid the problem is regular. Index scores as
//
//     s[(y*W + x)*D + d]        left pixel (y,x) against right pixel (y, x-d)
//
// and both max-sum reductions become constant-stride walks:
//
//   rho update, "max over this LEFT pixel's other candidates": for fixed (y,x)
//   the alternatives are the other d, so it is a contiguous run of D floats.
//
//   beta update, "max over this RIGHT pixel's other claimants": for fixed
//   (y,xr) the claimants are left pixels x = xr+d, i.e. index
//   (y*W + xr + d)*D + d = base + d*(D+1). Stride D+1, still regular.
//
// So there is no edge list, no indirection, and every inner loop is a strided
// top-2 the compiler can vectorise. Rows are independent -- correspondences lie
// on one row -- so they are handed to a thread pool with no synchronisation at
// all beyond the join.
//
//   de_dense LEFT.y8 RIGHT.y8 W H [--dmax N] [--iters N] [--threads N]
//            [--min-margin F] [--out disp.f32]

#include "doubleeye/preproc.hpp"
#include "doubleeye/simd_score.hpp"

#include <time.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

using namespace doubleeye;

namespace {

double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) * 1e3 + double(ts.tv_nsec) / 1e6;
}

// Census over the whole image, one uint64 per pixel. 7x7 -> 48 bits.
//
// Offset outer, x inner, accumulating into SIX BYTE planes which are packed into
// the uint64 once per row. Measured 5.9x faster than the natural form, with
// byte-for-byte identical descriptors.
//
// The obvious version puts x outermost and builds each pixel's word with
// `v |= (n < c) << bit` over 48 neighbours. That is a 48-long serial OR chain per
// pixel, every neighbour read recomputes `y * width`, and nothing vectorises,
// because the only wide axis available is x and x is the outer loop.
//
// Swapping the loops so x is innermost makes the shift constant within a pass and
// the compare a wide byte operation. **On its own that is 3.7x WORSE** (40.0 ms
// against 10.8), because a uint64 accumulator fits only four lanes in a 256-bit
// register and 48 passes over a row is a lot of loop overhead to pay for four
// lanes.
//
// It pays once the accumulator is a byte: six 8-bit planes give 32 lanes per
// register for the compare-and-or, and the packing costs one pass per row. Same
// bit assignment -- bit b lands at (b>>3)*8 + (b&7) = b -- so the descriptors are
// unchanged, which is verified rather than assumed.
//
// Worth keeping the middle result: loop order and lane width looked like two
// independent tweaks and are not. Reordering without widening loses, and the
// difference between the two reordered forms is 22x.
//
// The window must be a COMPILE-TIME constant for any of this to pay. Measured with
// hw/hh as runtime arguments the reordered form is 17.6 ms against the original's
// 8.1 -- twice as slow -- because the 48 passes cannot be unrolled, so the shift
// mask and the group pointer are recomputed every pass and the compiler will not
// widen the compare. The same code with the window fixed at 3x3 is 5.9x faster
// than the original. The transformation and the specialisation are one change, not
// two: the first measurement of it was taken in a microbenchmark where W, hw and hh
// happened to be file-scope constants, and that difference alone inverted the
// result.
double g_cen_clear = 0, g_cen_cmp = 0, g_cen_pack = 0;
double g_pool_spawn = 0, g_pool_join = 0;

template <int HW, int HH>
void census_rows(const Image8& img, uint64_t* out, int t, int nth) {
  const int W = img.width, H = img.height;
  const int ngroups = ((2 * HW + 1) * (2 * HH + 1) - 1 + 7) / 8;
  std::vector<uint8_t> g(size_t(ngroups) * W);
  double tc = 0, tm = 0, tp = 0;
  for (int y = HH + t; y < H - HH; y += nth) {
    double t0 = now_ms();
    std::fill(g.begin(), g.end(), 0);
    tc += now_ms() - t0; t0 = now_ms();
    const uint8_t* cen = img.data.data() + size_t(y) * W;
    int bit = 0;
    for (int dy = -HH; dy <= HH; ++dy)
      for (int dx = -HW; dx <= HW; ++dx) {
        if (dx == 0 && dy == 0) continue;
        const uint8_t* n = img.data.data() + size_t(y + dy) * W + dx;
        uint8_t* a = &g[size_t(bit >> 3) * W];
        const uint8_t one = uint8_t(1u << (bit & 7));
        for (int x = HW; x < W - HW; ++x)
          a[x] |= n[x] < cen[x] ? one : uint8_t(0);
        ++bit;
      }
    tm += now_ms() - t0; t0 = now_ms();
    uint64_t* o = out + size_t(y) * W;
    for (int x = HW; x < W - HW; ++x) {
      uint64_t v = 0;
      for (int j = 0; j < ngroups; ++j)
        v |= uint64_t(g[size_t(j) * W + x]) << (8 * j);
      o[x] = v;
    }
    tp += now_ms() - t0;
  }
  static std::mutex m;
  std::lock_guard<std::mutex> lk(m);
  g_cen_clear += tc; g_cen_cmp += tm; g_cen_pack += tp;
}

// Centre-symmetric census: compare each pixel against its point reflection through
// the centre rather than against the centre itself. A 7x7 window has 24 symmetric
// pairs instead of 48 neighbours, so the descriptor halves to 24 bits and fits
// uint32.
//
// The saving is NOT in the transform, which is already 2.3 ms and vectorised. It is
// in the score loop, which reads both descriptor planes once per disparity slice and
// is the second largest item on the TX2 at 57.5 ms: halving the descriptor halves
// those loads and doubles the NEON lanes per register.
//
// The risk is the opposite direction, and is the reason this is measured rather than
// assumed: it halves the Hamming range from 49 levels to 25, and 09-matching already
// records Census's coarse quantisation as the likeliest cause of the residual error
// in flat regions. Fewer levels is the wrong way for that. It also removes the
// centre pixel as a single point of failure, which is the usual argument in its
// favour.
template <int HW, int HH>
void census_cs_rows(const Image8& img, uint32_t* out, int t, int nth) {
  const int W = img.width, H = img.height;
  const int nbits = ((2 * HW + 1) * (2 * HH + 1) - 1) / 2;
  const int ngroups = (nbits + 7) / 8;
  std::vector<uint8_t> g(size_t(ngroups) * W);
  for (int y = HH + t; y < H - HH; y += nth) {
    std::fill(g.begin(), g.end(), 0);
    int bit = 0;
    for (int dy = -HH; dy <= HH; ++dy)
      for (int dx = -HW; dx <= HW; ++dx) {
        if (dy < 0 || (dy == 0 && dx <= 0)) continue;   // each pair once
        const uint8_t* a = img.data.data() + size_t(y + dy) * W + dx;
        const uint8_t* b = img.data.data() + size_t(y - dy) * W - dx;
        uint8_t* acc = &g[size_t(bit >> 3) * W];
        const uint8_t one = uint8_t(1u << (bit & 7));
        for (int x = HW; x < W - HW; ++x)
          acc[x] |= a[x] < b[x] ? one : uint8_t(0);
        ++bit;
      }
    uint32_t* o = out + size_t(y) * W;
    for (int x = HW; x < W - HW; ++x) {
      uint32_t v = 0;
      for (int j = 0; j < ngroups; ++j)
        v |= uint32_t(g[size_t(j) * W + x]) << (8 * j);
      o[x] = v;
    }
  }
}

std::vector<uint32_t> census_cs(const Image8& img, int nth) {
  std::vector<uint32_t> out(size_t(img.width) * img.height, 0);
  std::vector<std::thread> pool;
  for (int t = 0; t < nth; ++t)
    pool.emplace_back([&, t]() { census_cs_rows<3, 3>(img, out.data(), t, nth); });
  for (auto& th : pool) th.join();
  return out;
}

std::vector<uint64_t> census_image(const Image8& img, int hw, int hh, int nth) {
  const int W = img.width, H = img.height;
  std::vector<uint64_t> out(size_t(W) * H, 0);
  if (hw == 3 && hh == 3) {
    std::vector<std::thread> pool;
    for (int t = 0; t < nth; ++t)
      pool.emplace_back([&, t]() { census_rows<3, 3>(img, out.data(), t, nth); });
    for (auto& th : pool) th.join();
    return out;
  }
  // Generic fallback for any other window, kept correct rather than fast.
  const int nbits = (2 * hw + 1) * (2 * hh + 1) - 1;
  const int ngroups = (nbits + 7) / 8;
  std::vector<std::thread> pool;
  for (int t = 0; t < nth; ++t) pool.emplace_back([&, t]() {
  std::vector<uint8_t> g(size_t(ngroups) * W);
  for (int y = hh + t; y < H - hh; y += nth) {
    std::fill(g.begin(), g.end(), 0);
    const uint8_t* cen = img.data.data() + size_t(y) * W;
    int bit = 0;
    for (int dy = -hh; dy <= hh; ++dy)
      for (int dx = -hw; dx <= hw; ++dx) {
        if (dx == 0 && dy == 0) continue;
        const uint8_t* n = img.data.data() + size_t(y + dy) * W + dx;
        uint8_t* a = &g[size_t(bit >> 3) * W];
        const uint8_t one = uint8_t(1u << (bit & 7));
        for (int x = hw; x < W - hw; ++x)
          a[x] |= n[x] < cen[x] ? one : uint8_t(0);
        ++bit;
      }
    uint64_t* o = out.data() + size_t(y) * W;
    for (int x = hw; x < W - hw; ++x) {
      uint64_t v = 0;
      for (int j = 0; j < ngroups; ++j)
        v |= uint64_t(g[size_t(j) * W + x]) << (8 * j);
      o[x] = v;
    }
  }
  });
  for (auto& th : pool) th.join();
  return out;
}

inline int popcnt64(uint64_t v) { return __builtin_popcountll(v); }

// Largest and second largest of a strided run, plus the index of the largest.
// The "max excluding one element" that max-sum needs is then O(1).
struct Top2 {
  float b1 = -1e30f, b2 = -1e30f;
  int i1 = -1;
  inline void push(float v, int i) {
    if (v > b1) { b2 = b1; b1 = v; i1 = i; }
    else if (v > b2) { b2 = v; }
  }
  inline float excl(int i) const { return i == i1 ? b2 : b1; }
};


// Horizontal running sums for L rows at once.
//
// A running sum is a serial dependency chain: every add waits on the previous
// one, so the loop advances at float-add LATENCY -- four cycles per element on
// this core -- no matter how many adders are idle. It also cannot vectorise,
// for the same reason. Measured, this one pass was 74% of the whole box filter
// (95.6 ms of 122.6 over 240 calls).
//
// The fix for a non-vectorisable reduction is to run many independent ones side
// by side. L rows give L independent chains that the same adders interleave,
// and it needs no transpose: the L rows are L sequential streams, which is what
// the prefetcher is for.
//
// L = 8 measured best. The curve is the one two adders at four cycles of
// latency predict -- 1.55x at L=2, 2.41x at 4, 2.45x at 8, then 2.38x at 16 as
// the accumulators start spilling.
//
// The `x >= span` test is hoisted out by splitting the loop at x = span rather
// than asking on every element of every chain. Each chain still performs the
// same adds on the same values in the same order, so the result is
// bit-identical -- verified against the previous implementation, not assumed.
template <int L>
inline void horiz_rows(const float* in, float* const* op, int W, int r,
                       int y, int n) {
  const int span = 2 * r + 1;
  float a[L];
  const float* ip[L];
  for (int l = 0; l < n; ++l) { a[l] = 0.f; ip[l] = in + size_t(y + l) * W; }
  const int x1 = std::min(W, span);
  for (int x = 0; x < x1; ++x) {
    const int ox = x - r > 0 ? x - r : 0;
    for (int l = 0; l < n; ++l) { a[l] += ip[l][x]; op[l][ox] = a[l]; }
  }
  for (int x = x1; x < W; ++x) {
    const int ox = x - r;
    for (int l = 0; l < n; ++l) {
      a[l] += ip[l][x]; a[l] -= ip[l][x - span]; op[l][ox] = a[l];
    }
  }
  for (int x = std::max(0, W - r); x < W; ++x)
    for (int l = 0; l < n; ++l) op[l][x] = a[l];
}

// Separable box filter with running sums: O(N) regardless of radius.
void box_filter(const float* in, float* out, float* tmp, int W, int H, int r) {
  if (r <= 0) { std::copy(in, in + size_t(W) * H, out); return; }
  const int CH = 8;
  for (int y = 0; y < H; y += CH) {
    const int n = std::min(CH, H - y);
    float* op[CH];
    for (int l = 0; l < n; ++l) op[l] = tmp + size_t(y + l) * W;
    horiz_rows<CH>(in, op, W, r, y, n);
  }
  // Vertical pass, y outer and x inner.
  //
  // The obvious form is x outer, one scalar accumulator, stepping y -- and it is
  // doubly bad: the inner loop strides by W so every access is a fresh cache
  // line, and the accumulator is a serial dependency that cannot vectorise.
  //
  // Carrying a whole ROW of accumulators and stepping y instead keeps the serial
  // dependency in y, where it is unavoidable, while every inner loop runs over x
  // contiguously with W independent lanes. Same arithmetic, same result, but the
  // compiler can vectorise all three inner loops.
  // The normalisation is folded into these stores rather than run as its own
  // pass over W*H, which was a full read-modify-write of 675 KB for one flop an
  // element. Every store already had the value in a register. Overwriting a row
  // several times (which the clamped borders do) is still correct, because each
  // store writes acc*norm rather than scaling in place.
  const float norm = 1.f / float((2 * r + 1) * (2 * r + 1));
  std::vector<float> acc(size_t(W), 0.f);
  for (int y = 0; y < H; ++y) {
    const float* add = tmp + size_t(y) * W;
    for (int x = 0; x < W; ++x) acc[x] += add[x];
    if (y > 2 * r) {
      const float* sub = tmp + size_t(y - 2 * r - 1) * W;
      for (int x = 0; x < W; ++x) acc[x] -= sub[x];
    }
    float* dst = out + size_t(std::max(0, y - r)) * W;
    for (int x = 0; x < W; ++x) dst[x] = acc[x] * norm;
  }
  for (int y = std::max(0, H - r); y < H; ++y) {
    float* dst = out + size_t(y) * W;
    for (int x = 0; x < W; ++x) dst[x] = acc[x] * norm;
  }
}


// Decimate by an integer factor (nearest). Cheap and adequate: the guided
// filter's coefficient planes are smooth, which is the whole premise of the
// fast variant.
void downsample(const float* in, float* out, int W, int H, int Ws, int Hs, int f) {
  for (int y = 0; y < Hs; ++y) {
    const float* ip = in + size_t(std::min(H - 1, y * f)) * W;
    float* op = out + size_t(y) * Ws;
    for (int x = 0; x < Ws; ++x) op[x] = ip[std::min(W - 1, x * f)];
  }
}

// Bilinear upsample of a coefficient plane back to full resolution.
void upsample(const float* in, float* out, int W, int H, int Ws, int Hs, int f) {
  const float sx = float(Ws) / float(W), sy = float(Hs) / float(H);
  for (int y = 0; y < H; ++y) {
    const float fy = std::min(float(Hs) - 1.f, (float(y) + 0.5f) * sy - 0.5f);
    const int y0 = std::max(0, int(fy)), y1 = std::min(Hs - 1, y0 + 1);
    const float wy = fy - float(y0);
    const float* r0 = in + size_t(y0) * Ws;
    const float* r1 = in + size_t(y1) * Ws;
    float* op = out + size_t(y) * W;
    for (int x = 0; x < W; ++x) {
      const float fx = std::min(float(Ws) - 1.f, (float(x) + 0.5f) * sx - 0.5f);
      const int x0 = std::max(0, int(fx)), x1 = std::min(Ws - 1, x0 + 1);
      const float wx = fx - float(x0);
      const float a = r0[x0] * (1.f - wx) + r0[x1] * wx;
      const float b = r1[x0] * (1.f - wx) + r1[x1] * wx;
      op[x] = a * (1.f - wy) + b * wy;
    }
  }
  (void)f;
}

// Domain-transform recursive edge-aware filter, in place, no temporaries.
//
// Why replace the guided filter: measured, the cost stage is bound by how many
// times it walks memory, not by the arithmetic in the walks -- a 15x larger
// aggregation window is free, and threads saturate at two. The guided filter
// touches 29 whole planes per disparity slice and needs eight temporaries. This
// touches 12 and needs none, so the per-thread working set drops from ~6 MB to
// one plane and the shared coefficients.
//
// Gastal and Oliveira 2011, doi:10.1145/2010324.1964964. Two 1-D passes per
// axis, each a normalised IIR:
//
//     F[x] = F[x] + a[x] * (F[x-1] - F[x])
//
// with a[x] in [0,1) shrinking towards 0 across an intensity edge, which is what
// makes it edge-aware. a depends only on the GUIDE image, so like the guided
// filter's mean_I and var_I it is computed once for the whole volume rather than
// per disparity -- that is what makes an edge-aware filter affordable here at
// all.
//
// The horizontal passes are a serial dependency in x, the same problem the box
// filter's running sum had, and get the same fix: L rows at once for L
// independent chains. The vertical passes carry a whole row and vectorise over x
// with no help.
template <int L>
inline void rf_horiz(float* C, const uint16_t* ax, int W, int y, int n) {
  float f[L];
  for (int l = 0; l < n; ++l) f[l] = C[size_t(y + l) * W];
  for (int x = 1; x < W; ++x)
    for (int l = 0; l < n; ++l) {
      const size_t i = size_t(y + l) * W + x;
      f[l] = C[i] + float(ax[i]) * (1.f / 32768.f) * (f[l] - C[i]);
      C[i] = f[l];
    }
  for (int l = 0; l < n; ++l) f[l] = C[size_t(y + l) * W + W - 1];
  for (int x = W - 2; x >= 0; --x)
    for (int l = 0; l < n; ++l) {
      const size_t i = size_t(y + l) * W + x;
      f[l] = C[i] + float(ax[i + 1]) * (1.f / 32768.f) * (f[l] - C[i]);
      C[i] = f[l];
    }
}

// Q14 fixed point: the score is (24 - popcount)/24 in [-1,1], which maps exactly
// onto +-16384 and leaves a factor of two of headroom in int16. The intermediate is
// int32 -- the largest product is 32767 * 32768, just inside it -- and the shift
// rounds to nearest rather than truncating, because four passes of a downward bias
// would walk the whole plane.
//
// Why int16 here specifically: the filter is four passes of pure float work and is
// 33% of the cost stage on the desktop and ~50% on the TX2. NEON is 128-bit against
// AVX2's 256, so halving the element width buys 8 lanes against 4 exactly where the
// platform that matters is weakest.
static const int SCORE_Q = 14;
static const int SCORE_ONE = 1 << SCORE_Q;

template <int L>
inline void rf_horiz_i16(int16_t* C, const uint16_t* ax, int W, int y, int n) {
  int32_t f[L];
  for (int l = 0; l < n; ++l) f[l] = C[size_t(y + l) * W];
  for (int x = 1; x < W; ++x)
    for (int l = 0; l < n; ++l) {
      const size_t i = size_t(y + l) * W + x;
      const int32_t c = C[i];
      f[l] = c + ((int32_t(ax[i]) * (f[l] - c) + (1 << 14)) >> 15);
      C[i] = int16_t(f[l]);
    }
  for (int l = 0; l < n; ++l) f[l] = C[size_t(y + l) * W + W - 1];
  for (int x = W - 2; x >= 0; --x)
    for (int l = 0; l < n; ++l) {
      const size_t i = size_t(y + l) * W + x;
      const int32_t c = C[i];
      f[l] = c + ((int32_t(ax[i + 1]) * (f[l] - c) + (1 << 14)) >> 15);
      C[i] = int16_t(f[l]);
    }
}

// The recursive filter over a sub-rectangle, for masked planes. The IIRs start at
// the rectangle's edges instead of the image's, so aggregation support is truncated
// there -- an approximation the full-sweep path does not make, which is why the
// default path still calls the whole-plane version and why the mask's accuracy is
// scored with dense_bench rather than argued. Bounds: rows [y0, y1), cols [x0, x1).
template <int LN>
inline void rf_horiz_i16_rect(int16_t* C, const uint16_t* ax, int W, int y, int n,
                              int x0, int x1) {
  int32_t f[LN];
  for (int l = 0; l < n; ++l) f[l] = C[size_t(y + l) * W + x0];
  for (int x = x0 + 1; x < x1; ++x)
    for (int l = 0; l < n; ++l) {
      const size_t i = size_t(y + l) * W + x;
      const int32_t c = C[i];
      f[l] = c + ((int32_t(ax[i]) * (f[l] - c) + (1 << 14)) >> 15);
      C[i] = int16_t(f[l]);
    }
  for (int l = 0; l < n; ++l) f[l] = C[size_t(y + l) * W + x1 - 1];
  for (int x = x1 - 2; x >= x0; --x)
    for (int l = 0; l < n; ++l) {
      const size_t i = size_t(y + l) * W + x;
      const int32_t c = C[i];
      f[l] = c + ((int32_t(ax[i + 1]) * (f[l] - c) + (1 << 14)) >> 15);
      C[i] = int16_t(f[l]);
    }
}

void rf_filter_i16_rect(int16_t* C, const uint16_t* ax, const uint16_t* ay,
                        int W, int y0, int y1, int x0, int x1) {
  if (y1 - y0 < 1 || x1 - x0 < 2) return;
  const int CH = 8;
  for (int y = y0; y < y1; y += CH)
    rf_horiz_i16_rect<CH>(C, ax, W, y, std::min(CH, y1 - y), x0, x1);
  for (int y = y0 + 1; y < y1; ++y) {
    int16_t* cur = C + size_t(y) * W;
    const int16_t* prv = C + size_t(y - 1) * W;
    const uint16_t* a = ay + size_t(y) * W;
    for (int x = x0; x < x1; ++x) {
      const int32_t c = cur[x];
      cur[x] = int16_t(c + ((int32_t(a[x]) * (prv[x] - c) + (1 << 14)) >> 15));
    }
  }
  for (int y = y1 - 2; y >= y0; --y) {
    int16_t* cur = C + size_t(y) * W;
    const int16_t* nxt = C + size_t(y + 1) * W;
    const uint16_t* a = ay + size_t(y + 1) * W;
    for (int x = x0; x < x1; ++x) {
      const int32_t c = cur[x];
      cur[x] = int16_t(c + ((int32_t(a[x]) * (nxt[x] - c) + (1 << 14)) >> 15));
    }
  }
}

void rf_filter_i16(int16_t* C, const uint16_t* ax, const uint16_t* ay,
                   int W, int H) {
  const int CH = 8;
  for (int y = 0; y < H; y += CH)
    rf_horiz_i16<CH>(C, ax, W, y, std::min(CH, H - y));
  for (int y = 1; y < H; ++y) {
    int16_t* cur = C + size_t(y) * W;
    const int16_t* prv = C + size_t(y - 1) * W;
    const uint16_t* a = ay + size_t(y) * W;
    for (int x = 0; x < W; ++x) {
      const int32_t c = cur[x];
      cur[x] = int16_t(c + ((int32_t(a[x]) * (prv[x] - c) + (1 << 14)) >> 15));
    }
  }
  for (int y = H - 2; y >= 0; --y) {
    int16_t* cur = C + size_t(y) * W;
    const int16_t* nxt = C + size_t(y + 1) * W;
    const uint16_t* a = ay + size_t(y + 1) * W;
    for (int x = 0; x < W; ++x) {
      const int32_t c = cur[x];
      cur[x] = int16_t(c + ((int32_t(a[x]) * (nxt[x] - c) + (1 << 14)) >> 15));
    }
  }
}

void rf_filter(float* C, const uint16_t* ax, const uint16_t* ay, int W, int H) {
  const int CH = 8;
  for (int y = 0; y < H; y += CH) rf_horiz<CH>(C, ax, W, y, std::min(CH, H - y));
  for (int y = 1; y < H; ++y) {
    float* cur = C + size_t(y) * W;
    const float* prv = C + size_t(y - 1) * W;
    const uint16_t* a = ay + size_t(y) * W;
    for (int x = 0; x < W; ++x)
      cur[x] += float(a[x]) * (1.f / 32768.f) * (prv[x] - cur[x]);
  }
  for (int y = H - 2; y >= 0; --y) {
    float* cur = C + size_t(y) * W;
    const float* nxt = C + size_t(y + 1) * W;
    const uint16_t* a = ay + size_t(y + 1) * W;
    for (int x = 0; x < W; ++x)
      cur[x] += float(a[x]) * (1.f / 32768.f) * (nxt[x] - cur[x]);
  }
}

// The coefficient planes. ax[i] governs the step into pixel i from its left
// neighbour, ay[i] the step from the row above.
void rf_coeffs(const uint8_t* raw, uint16_t* ax, uint16_t* ay, int W, int H,
               float sigma_s, float sigma_r, int nth) {
  const float k = -std::sqrt(2.f) / sigma_s;
  const float rs = sigma_s / sigma_r;
  // The coefficient depends only on the neighbour difference of two bytes, which
  // has 256 possible values, so both exponentials collapse into one 512-byte
  // table. Measured before the change: ~700k transcendentals took 10 ms of the
  // cost-stage prologue on the TX2 with all six threads on it. The table is built
  // from the byte difference directly rather than from |a/255 - b/255|, which is
  // the same quantity up to the last float ulp -- NOT bit-identical to the old
  // path, so this checkpoint is scored with dense_bench, not cmp.
  uint16_t tbl[256];
  for (int d = 0; d < 256; ++d) {
    const float c = std::exp(k * (1.f + rs * (float(d) / 255.f)));
    tbl[d] = uint16_t(std::min(32767.f, c * 32768.f + 0.5f));
  }
  std::vector<std::thread> pool;
  for (int t = 0; t < nth; ++t) pool.emplace_back([&, t]() {
  for (int y = t; y < H; y += nth) {
    const uint8_t* r = raw + size_t(y) * W;
    const uint8_t* u = raw + size_t(y > 0 ? y - 1 : 0) * W;
    uint16_t* axr = ax + size_t(y) * W;
    uint16_t* ayr = ay + size_t(y) * W;
    axr[0] = tbl[0];
    for (int x = 1; x < W; ++x) axr[x] = tbl[std::abs(int(r[x]) - int(r[x - 1]))];
    if (y == 0) for (int x = 0; x < W; ++x) ayr[x] = tbl[0];
    else        for (int x = 0; x < W; ++x) ayr[x] = tbl[std::abs(int(r[x]) - int(u[x]))];
  }
  });
  for (auto& th : pool) th.join();
}


struct Cfg {
  int dmin = 1, dmax = 60, iters = 2, agg = 3;
  bool rf = true;              // recursive edge-aware filter; --guided for the old one
  float sigma_s = 12.f, sigma_r = 0.20f;   // sigma_r swept: 0.2 is the peak
  bool subpixel = false;   // measured: hurts slightly, see below
  bool guided = true;      // edge-aware aggregation
  float eps = 0.0025f;     // guided-filter regularisation, I in [0,1]
  int fgf = 1;             // fast guided filter: measured NOT worth it, see below
  int topk = 2;                // candidates per pixel; 2 measured best, 0 = dense
  int band = 2;                // prior-guided: search +-band around the prior
  bool csct = false;           // centre-symmetric census: 24 bits instead of 48
  float ad = 0.15f;            // graded cost weight; 0.15 with trunc 10 measured best
  int ad_trunc = 10;           // truncation for the absolute difference, 8-bit
  float lambda = -0.1f, gamma = -0.1f, damping = 0.4f, min_margin = 0.f;
  int threads = 0;
  bool simd = false;           // NEON score kernel, disparity in the lanes
  bool c2f = false;            // half-res coarse pass, prior as a mask (--band)
};

// One image row. Rows share no left or right pixel, so this needs no locking.
void solve_row(int y, int W, int D, const Cfg& cfg,
               const float* vol,
               float* s, float* beta, float* rho,
               float* out_disp, float* out_margin, int hw,
               float* gath, float* svals, int* idxs, int* k0, int* k1,
               float* best, int* bestk, int* order, char* taken) {
  const int dmin = cfg.dmin;
  const float keep = 1.f - cfg.damping;

  // --- scores, the valid range, and the margin, in one pass ---
  //
  // The valid disparities for pixel x are exactly those with x-d inside the
  // census border, which is a contiguous interval. Storing [k0, k1) per pixel
  // removes the sentinel comparison from every subsequent inner loop -- the
  // difference between a branch per element and none.
  for (int x = 0; x < W; ++x) {
    float* sp = s + size_t(x) * D;
    int lo = 0, hi = 0;
    if (x >= hw && x < W - hw) {
      lo = std::max(0, x - (W - hw) + 1 - dmin + 1);
      hi = std::min(D, x - hw - dmin + 1);
      if (hi < lo) hi = lo;
    }
    k0[x] = lo; k1[x] = hi;
    Top2 t;
    for (int k = lo; k < hi; ++k) {
      const float v = vol[size_t(x) * D + k];
      sp[k] = v;
      if (v > -1e29f) t.push(v, k);
    }
    const float alt = (t.b2 < -1e29f) ? cfg.lambda : std::max(t.b2, cfg.lambda);
    out_margin[x] = (t.i1 < 0) ? 0.f : t.b1 - alt;
  }

  std::fill(beta, beta + size_t(W) * D, 0.f);
  std::fill(rho, rho + size_t(W) * D, 0.f);

  for (int it = 0; it < cfg.iters; ++it) {
    // rho_ij = s - max(lambda, max over this LEFT pixel's other disparities of beta)
    // Contiguous run of D per pixel.
    for (int x = 0; x < W; ++x) {
      const float* bp = beta + size_t(x) * D;
      const float* sp = s + size_t(x) * D;
      float* rp = rho + size_t(x) * D;
      const int lo = k0[x], hi = k1[x];
      if (hi <= lo) continue;
      Top2 t;
      for (int k = lo; k < hi; ++k) t.push(bp[k], k);
      for (int k = lo; k < hi; ++k) {
        const float tgt = sp[k] - std::max(cfg.lambda, t.excl(k));
        rp[k] = keep * tgt + cfg.damping * rp[k];
      }
    }
    // beta_ij = s - max(gamma, max over this RIGHT pixel's other claimants of rho)
    //
    // Claimants of right pixel xr are the left pixels xr+d, so the walk is
    // index (xr+dmin)*D + k*(D+1): stride D+1 floats, a cache miss per element.
    // Gathering the diagonal into a contiguous scratch buffer first turns two
    // strided passes (top-2, then update) into one strided gather plus two
    // contiguous passes, and the scratch is 60 floats -- one cache line's worth
    // of work per pixel instead of sixty misses.
    for (int xr = 0; xr < W; ++xr) {
      const int kmax = std::min(D, W - xr - dmin);
      if (kmax <= 0) break;
      const size_t base = (size_t(xr) + size_t(dmin)) * D;
      const size_t step = size_t(D) + 1;
      Top2 t;
      int nv = 0;
      for (int k = 0; k < kmax; ++k) {
        const int x = xr + dmin + k;
        if (k < k0[x] || k >= k1[x]) { idxs[k] = -1; continue; }
        const size_t id2 = base + size_t(k) * step;
        gath[k] = rho[id2]; svals[k] = s[id2]; idxs[k] = 1;
        t.push(gath[k], k); ++nv;
      }
      if (nv == 0) continue;
      for (int k = 0; k < kmax; ++k) {
        if (idxs[k] < 0) continue;
        const size_t id2 = base + size_t(k) * step;
        beta[id2] = keep * (svals[k] - std::max(cfg.gamma, t.excl(k)))
                    + cfg.damping * beta[id2];
      }
    }
  }

  // --- decision: best belief per left pixel, then uniqueness by greedy claim ---
  for (int x = 0; x < W; ++x) {
    best[x] = -1e30f; bestk[x] = -1; taken[x] = 0; order[x] = x;
    const float* sp = s + size_t(x) * D;
    for (int k = k0[x]; k < k1[x]; ++k) {
      if (sp[k] <= cfg.lambda) continue;
      const float bel = beta[size_t(x) * D + k] + rho[size_t(x) * D + k] - sp[k];
      if (bel > best[x]) { best[x] = bel; bestk[x] = k; }
    }
  }
  std::sort(order, order + W, [&](int a, int b) { return best[a] > best[b]; });
  for (int oi = 0; oi < W; ++oi) {
    const int x = order[oi];
    out_disp[x] = std::nanf("");
    if (bestk[x] < 0) continue;
    if (cfg.min_margin > 0.f && out_margin[x] < cfg.min_margin) continue;
    const int k = bestk[x];
    const int d = cfg.dmin + k;
    const int xr = x - d;
    if (xr < 0 || taken[xr]) continue;
    taken[xr] = 1;

    // Sub-pixel by parabola through the cost at k-1, k, k+1. The output was
    // integer disparity, which forfeits up to half a pixel before any matching
    // error -- and the accuracy metric has a one-pixel threshold, so that
    // quantisation is charged directly against the score. SGM interpolates to
    // 1/16 px for exactly this reason.
    //
    // Clamped to +-0.5 and skipped where the three samples are not a maximum.
    //
    // OFF BY DEFAULT: measured, it makes things slightly WORSE -- 8.8% bad
    // against 8.6% at matched coverage. The window-aggregated Census cost is a
    // mean of 49-level Hamming scores over a 7x7 block, which is not locally
    // parabolic enough for a three-point fit to find a real vertex. The theory
    // that integer quantisation was costing accuracy was reasonable and wrong.
    float off = 0.f;
    const float* sp2 = s + size_t(x) * D;
    if (cfg.subpixel && k > k0[x] && k + 1 < k1[x]) {
      const float cm = sp2[k - 1], c0 = sp2[k], cp = sp2[k + 1];
      const float den = 2.f * c0 - cm - cp;
      if (den > 1e-6f) {
        off = 0.5f * (cp - cm) / den;
        if (off > 0.5f) off = 0.5f;
        if (off < -0.5f) off = -0.5f;
      }
    }
    out_disp[x] = float(d) + off;
  }
  (void)y;
}

// MASDA on a pruned candidate list instead of the full disparity range.
//
// Measured before building it (article/topk_recall.py): the true disparity is
// within 1 px of one of the top 8 candidates by aggregated cost for 82.8% of
// known-ground-truth pixels, while the shipping pipeline delivers 67.9% correct.
// Fifteen points of headroom, so pruning to k=8 is 7.5x fewer edges at no
// accuracy the benchmark can resolve.
//
// Pruning is by RANK WITHIN A PIXEL, never a global threshold on score. Census
// costs are absolute Hamming fractions, so a textureless region scores uniformly
// mediocre and its correct match scores mediocre too; a magnitude cutoff would
// delete precisely the flat regions where half the remaining error sits.
//
// The rho update keeps its shape -- a contiguous run of k per left pixel. The
// beta update is the interesting one: "this right pixel's other claimants" was a
// stride-(D+1) diagonal walk through the volume, which pruning makes irregular.
// A counting sort into buckets by xr = x - d rebuilds the claimant lists in
// O(k*W) per row, entirely within the row and cache-resident at 28 KB. That
// diagonal walk was the one genuinely awkward thing the dense layout imposed, and
// this removes it rather than reorganising it.
void solve_row_sparse(int W, int D, int K, const Cfg& cfg, const float* vol,
                      float* cs, int* cd, int* cn,
                      float* beta, float* rho,
                      float* out_disp, float* out_margin, int hw,
                      int* bstart, int* bitems, int* cursor,
                      float* best, int* bestj, int* order, char* taken) {
  if (W < 1 || K < 1) return;      // states the obvious to the compiler, which
  const size_t KS = size_t(K);      // otherwise cannot bound the fills below and
  const int dmin = cfg.dmin;        // warns that they may exceed any object size
  const float keep = 1.f - cfg.damping;

  // --- select the top K per pixel, sorted best first ---
  //
  // Skipped entirely when vol is null: the blockwise cost stage has already
  // produced the candidates, because with K = 2 the running top-2 IS the volume
  // and there is nothing left to reduce.
  //
  // One pass over D holding a sorted list of K by insertion. The alternative,
  // K selection passes over D, is 8x60 comparisons per pixel; this is 60 plus
  // the few insertions that actually beat the running Kth best.
  if (vol) for (int x = 0; x < W; ++x) {
    float* S = cs + size_t(x) * KS;
    int* Dd = cd + size_t(x) * KS;
    int n = 0;
    int lo = 0, hi = 0;
    if (x >= hw && x < W - hw) {
      lo = std::max(0, x - (W - hw) + 1 - dmin + 1);
      hi = std::min(D, x - hw - dmin + 1);
      if (hi < lo) hi = lo;
    }
    const float* V = vol + size_t(x) * D;
    for (int k = lo; k < hi; ++k) {
      const float v = V[k];
      if (v <= -1e29f) continue;
      if (n < K) {
        int j = n++;
        for (; j > 0 && S[j - 1] < v; --j) { S[j] = S[j - 1]; Dd[j] = Dd[j - 1]; }
        S[j] = v; Dd[j] = k;
      } else if (v > S[K - 1]) {
        int j = K - 1;
        for (; j > 0 && S[j - 1] < v; --j) { S[j] = S[j - 1]; Dd[j] = Dd[j - 1]; }
        S[j] = v; Dd[j] = k;
      }
    }
    cn[x] = n;
  }

  // Identical to the dense margin: best minus the better of the runner-up and
  // lambda, which for K >= 2 sees the same two values it did before.
  for (int x = 0; x < W; ++x) {
    const float* S = cs + size_t(x) * KS;
    const int n = cn[x];
    const float alt = (n < 2) ? cfg.lambda : std::max(S[1], cfg.lambda);
    out_margin[x] = (n == 0) ? 0.f : S[0] - alt;
  }

  // --- claimant lists per right pixel, by counting sort ---
  std::fill(cursor, cursor + W, 0);
  for (int x = 0; x < W; ++x)
    for (int j = 0; j < cn[x]; ++j) {
      const int xr = x - (dmin + cd[size_t(x) * KS + j]);
      if (xr >= 0 && xr < W) ++cursor[xr];
    }
  bstart[0] = 0;
  for (int xr = 0; xr < W; ++xr) bstart[xr + 1] = bstart[xr] + cursor[xr];
  for (int xr = 0; xr < W; ++xr) cursor[xr] = bstart[xr];
  for (int x = 0; x < W; ++x)
    for (int j = 0; j < cn[x]; ++j) {
      const int xr = x - (dmin + cd[size_t(x) * KS + j]);
      if (xr >= 0 && xr < W) bitems[cursor[xr]++] = int(size_t(x) * KS + j);
    }

  std::fill(beta, beta + size_t(W) * KS, 0.f);
  std::fill(rho, rho + size_t(W) * KS, 0.f);

  for (int it = 0; it < cfg.iters; ++it) {
    for (int x = 0; x < W; ++x) {
      const int n = cn[x];
      if (n == 0) continue;
      const size_t b = size_t(x) * KS;
      Top2 t;
      for (int j = 0; j < n; ++j) t.push(beta[b + j], j);
      for (int j = 0; j < n; ++j) {
        const float tgt = cs[b + j] - std::max(cfg.lambda, t.excl(j));
        rho[b + j] = keep * tgt + cfg.damping * rho[b + j];
      }
    }
    for (int xr = 0; xr < W; ++xr) {
      const int lo = bstart[xr], hi = bstart[xr + 1];
      if (hi <= lo) continue;
      Top2 t;
      for (int p = lo; p < hi; ++p) t.push(rho[bitems[p]], p - lo);
      for (int p = lo; p < hi; ++p) {
        const int idx = bitems[p];
        beta[idx] = keep * (cs[idx] - std::max(cfg.gamma, t.excl(p - lo)))
                    + cfg.damping * beta[idx];
      }
    }
  }

  for (int x = 0; x < W; ++x) {
    best[x] = -1e30f; bestj[x] = -1; taken[x] = 0; order[x] = x;
    const size_t b = size_t(x) * KS;
    for (int j = 0; j < cn[x]; ++j) {
      if (cs[b + j] <= cfg.lambda) continue;
      const float bel = beta[b + j] + rho[b + j] - cs[b + j];
      if (bel > best[x]) { best[x] = bel; bestj[x] = j; }
    }
  }
  std::sort(order, order + W, [&](int a, int b) { return best[a] > best[b]; });
  for (int oi = 0; oi < W; ++oi) {
    const int x = order[oi];
    out_disp[x] = std::nanf("");
    if (bestj[x] < 0) continue;
    if (cfg.min_margin > 0.f && out_margin[x] < cfg.min_margin) continue;
    const int d = dmin + cd[size_t(x) * KS + bestj[x]];
    const int xr = x - d;
    if (xr < 0 || taken[xr]) continue;
    taken[xr] = 1;
    out_disp[x] = float(d);
  }
}

}  // namespace

// The whole dense pipeline -- census, cost volume, solve -- as one callable, so a
// coarse pass can run it at half resolution and a fine pass can run it again under
// the resulting prior. Extracted unchanged from main(); the timing struct exists
// because the prints stay in main and dense_bench parses them.
struct LevelTimes {
  double census = 0, cost = 0, solve = 0;
  double c_alloc = 0, c_fill = 0, c_score = 0, c_filt = 0, c_ins = 0;
  double c_span = 0, span_max = 0;
  double prologue = 0, tp_alloc = 0, tp_norm = 0, tp_coeff = 0, after = 0;
};

void run_dense(const Image8& L, const Image8& R, const Cfg& cfg, int nthreads,
               const std::vector<float>& prior, const std::string& volp,
               std::vector<float>* disp_out, std::vector<float>* margin_out,
               LevelTimes* T) {
  const int W = L.width, H = L.height;
  const int D = cfg.dmax - cfg.dmin + 1;
  const double t0 = now_ms();
  std::vector<uint64_t> cl, cr;
  std::vector<uint32_t> cl32, cr32;
  if (cfg.csct) { cl32 = census_cs(L, nthreads); cr32 = census_cs(R, nthreads); }
  else { cl = census_image(L, 3, 3, nthreads); cr = census_image(R, 3, 3, nthreads); }
  T->census = now_ms() - t0;

  // --- cost volume, aggregated over a window ---
  //
  // A single Census comparison is 48 binary tests quantised to 49 Hamming levels.
  // Measured against SGM, that -- not the absence of a smoothness prior -- is
  // where the accuracy goes: SGM stripped of smoothness AND post-filtering still
  // reaches 12.7% bad against this matcher's 28.1%, and its cost is an absolute
  // difference aggregated over a 5x5 block.
  //
  // So aggregate. Summing the score over a window at fixed disparity is the same
  // trick, costs two separable passes, and turns 49 levels into a graded score.
  // Built disparity-major so each slice is a contiguous H*W plane the box filter
  // can walk, then transposed once into the [x][k] layout the solver wants.
  //
  // Parallel over DISPARITY here, not over rows. Each slice is an independent
  // H*W plane -- score, box filter, scatter -- so the only shared state is the
  // output volume, and slices write disjoint elements of it. Measured before
  // this change the cost volume was 211 ms of a 273 ms total and did not move
  // with thread count at all, because it was the one serial stage left.
  //
  // Both axes of parallelism in this program are over an index that partitions
  // the work with no sharing: disparity here, rows in the solver. That is also
  // what makes the whole thing a GPU kernel later rather than a rewrite.
  // In-situ accumulators, summed across threads. Timing the real loops with their
  // real parameters, because a microbenchmark twice today measured code the
  // compiler could specialise in ways the shipping function cannot be.
  std::vector<double> ts_fill(nthreads, 0), ts_score(nthreads, 0);
  std::vector<double> ts_filt(nthreads, 0), ts_ins(nthreads, 0);
  std::vector<double> ts_alloc(nthreads, 0);
  // Per-thread WALL span, so idle cores can be attributed. The four in-situ timers
  // give busy time; span minus busy is time the thread existed and was not working,
  // and comparing spans across threads separates imbalance (spans differ) from
  // contention or preemption (spans equal, busy short). Occupancy has been reasoned
  // about twice today and both guesses were wrong.
  std::vector<double> ts_span(nthreads, 0);
  const double tc0 = now_ms();
  // With K = 2 the running top-2 per pixel IS the reduced volume, so the 40 MB
  // array is never allocated. It is still needed for --dump-vol and for the k
  // sweep, which is why both paths exist.
  //
  // Why this is a traffic win and not a cache-capacity one, which was measured:
  // staging plus transposing the volume moved ~120 MB, and the solver then read
  // 40 MB back. The blockwise form reads each filtered slice once (40 MB total)
  // and compares against a 675 KB runner-up plane that stays resident across all
  // D slices, so the common path is one streaming read and a cached compare.
  const bool blockwise = (cfg.topk == 2) && volp.empty();
  const size_t WH = size_t(W) * H;
  // --- the prior as a MASK on absolute-disparity planes ---------------------------
  //
  // The offset-indexed version of this -- planes indexed by offset from the prior --
  // is a measured negative with a known mechanism: aggregation needs
  // constant-disparity planes, and an offset plane mixes disparities everywhere the
  // prior varies, which cost 4 points of bad-1.0 at every band width (09-matching.md,
  // "the ceiling was real and this construction cannot reach it"). So the planes
  // stay ABSOLUTE and the prior only decides which pixels of which planes are worth
  // computing: plane d is evaluated where |d - prior| <= band, as a per-row interval
  // [mx0, mx1] plus a per-plane bounding box, and rows where nothing needs d are
  // skipped outright. Scores outside the interval are simply never inserted, so a
  // wrong prior costs candidates, never wrong values.
  const bool mask = !prior.empty();
  std::vector<int16_t> mx0, mx1;                     // [y][k] row intervals
  std::vector<int> pylo, pyhi, pxlo, pxhi;           // per-plane bounding boxes
  if (mask) {
    mx0.assign(size_t(H) * D, int16_t(W));
    mx1.assign(size_t(H) * D, int16_t(-1));
    pylo.assign(D, H); pyhi.assign(D, -1);
    pxlo.assign(D, W); pxhi.assign(D, -1);
    std::vector<std::thread> ipool;
    for (int t = 0; t < nthreads; ++t) ipool.emplace_back([&, t]() {
      for (int y = t; y < H; y += nthreads) {
        int16_t* r0 = &mx0[size_t(y) * D];
        int16_t* r1 = &mx1[size_t(y) * D];
        const float* pr = &prior[size_t(y) * W];
        for (int x = 0; x < W; ++x) {
          // ONE band center per pixel, deliberately. A second band around the
          // coarse runner-up was built and measured -- the idea being that a thin
          // structure vanishes at half resolution and the runner-up is the other
          // surface -- and it made every scene WORSE (pooled 10.6% -> 11.2%,
          // teddy 7.1 -> 8.0, Laundry 20.2 -> 21.9): where the coarse level is
          // ambiguous its runner-up is usually the wrong period of a repetitive
          // texture, and a band around it hands the fine solver a wrong surface
          // it can aggregate into a confident answer. The thin-structure loss is
          // real (Art 12.7 -> 14.9) but the cure was worse than the disease.
          const int dc = int(std::lround(pr[x]));
          const int klo = std::max(cfg.dmin, dc - cfg.band) - cfg.dmin;
          const int khi = std::min(cfg.dmax, dc + cfg.band) - cfg.dmin;
          for (int k = klo; k <= khi; ++k) {
            if (x < r0[k]) r0[k] = int16_t(x);
            if (x > r1[k]) r1[k] = int16_t(x);
          }
        }
      }
    });
    for (auto& th : ipool) th.join();
    for (int y = 0; y < H; ++y)
      for (int k = 0; k < D; ++k) {
        const int a = mx0[size_t(y) * D + k], b = mx1[size_t(y) * D + k];
        if (b < a) continue;
        if (y < pylo[k]) pylo[k] = y;
        if (y > pyhi[k]) pyhi[k] = y;
        if (a < pxlo[k]) pxlo[k] = a;
        if (b > pxhi[k]) pxhi[k] = b;
      }
  }
  // How far outside the interval the filter still needs real scores: its influence
  // decays by exp(-sqrt(2)/sigma_s) per pixel on flat image, 0.89 at sigma_s 12, so
  // 16 pixels is a factor of ~0.15. Untuned; a sweep belongs to the measurement.
  const int MPAD = 8;
  {
    const double t = now_ms();
    (void)t;
  }
  const double tpa = now_ms();
  std::vector<float> vol(blockwise ? 0 : WH * size_t(D), -1e30f);
  T->tp_alloc = now_ms() - tpa;
  // The merged-candidate arrays that used to be allocated here are gone: the 2-of-2n
  // selection now happens inside the solve, row by row, where the candidates are
  // consumed. Measured before the change: filling them was 5-15 ms of serial
  // prologue and the merge pass another ~30 ms outside any pool, against a stage
  // wall of ~180 -- the two biggest pieces of the Amdahl loss this stage had.
  //
  // Guided-filter aggregation instead of a box.
  //
  // A box averages the cost over a square regardless of what is in it, so at a
  // depth discontinuity it mixes two surfaces. Measured, that is where the error
  // is: bad-1.0 is 6.7% on near-fronto-parallel pixels and 50%+ where the
  // ground-truth disparity gradient exceeds 0.6 px/px.
  //
  // The guided filter weights the support by agreement with the LEFT IMAGE, so a
  // strong intensity edge stops the aggregation. It is still O(N) -- box filters
  // all the way down -- and the guidance statistics (mean_I, var_I) do not depend
  // on disparity, so they are computed once for the whole volume.
  //
  // Fast guided filter: the four box passes per disparity slice happen at 1/f^2
  // of the pixels, and only the two coefficient planes come back to full
  // resolution. The coefficients are smooth, which is what makes this nearly
  // free in quality -- He and Sun's observation.
  const int F = cfg.fgf;
  const int Ws = (F == 1) ? W : std::max(1, W / F);
  const int Hs = (F == 1) ? H : std::max(1, H / F);
  const int rs = (F == 1) ? cfg.agg : std::max(1, cfg.agg / F);
  std::vector<float> I(size_t(W) * H);
  const size_t GS = cfg.rf ? 0 : size_t(Ws) * Hs;   // guided filter's guidance only
  std::vector<float> Is(GS), mIs(GS), vIs(GS);
  // Shared, read-only, disparity-independent: 1.35 MB for both, against the
  // guided filter's eight per-thread temporaries.
  std::vector<uint16_t> axv, ayv;
  const uint16_t *axp = nullptr, *ayp = nullptr;
  if (!cfg.rf) {
    // Only the guided filter reads I. The recursive path's coefficients come from
    // the raw bytes, so normalising 407k floats for it was a pure prologue cost.
    const double t = now_ms();
    for (size_t i = 0; i < I.size(); ++i) I[i] = float(L.data[i]) / 255.f;
    T->tp_norm = now_ms() - t;
  }
  // mean_I and var_I exist only for the guided filter. Under the recursive filter
  // they were still being computed -- a downsample, two box filters and a variance
  // pass over the whole image, serially, for a result nothing read.
  if (!cfg.rf) {
    std::vector<float> t1(size_t(Ws) * Hs), IIs(size_t(Ws) * Hs);
    downsample(I.data(), Is.data(), W, H, Ws, Hs, F);
    for (size_t i = 0; i < Is.size(); ++i) IIs[i] = Is[i] * Is[i];
    box_filter(Is.data(), mIs.data(), t1.data(), Ws, Hs, rs);
    box_filter(IIs.data(), vIs.data(), t1.data(), Ws, Hs, rs);
    for (size_t i = 0; i < vIs.size(); ++i) vIs[i] -= mIs[i] * mIs[i];
  }
  if (cfg.rf) {
    const double t = now_ms();
    axv.resize(size_t(W) * H); ayv.resize(size_t(W) * H);
    rf_coeffs(L.data.data(), axv.data(), ayv.data(), W, H, cfg.sigma_s,
              cfg.sigma_r, nthreads);
    axp = axv.data(); ayp = ayv.data();
    T->tp_coeff = now_ms() - t;
  }
  // The per-thread top-2 planes outlive the cost stage: each worker saw a disjoint
  // set of disparities, and the SOLVE now does the 2-of-2n selection per row instead
  // of a separate merge pass writing 8 MB of merged candidates nothing else read.
  std::vector<std::vector<int16_t>> as0(nthreads), as1(nthreads);
  std::vector<std::vector<int16_t>> ad0(nthreads), ad1(nthreads);
  {
    const int r = std::max(0, cfg.agg);
    // Disparity is processed in BLOCKS, and each thread owns whole blocks.
    //
    // The scatter into the volume writes vol[(y*W+x)*D + k], which for a fixed k
    // walks stride D floats -- 240 bytes, so one useful float per cache line, over
    // a 40 MB array, sixty times. That write amplification was the dominant memory
    // cost, and the work is bandwidth bound.
    //
    // Computing KB consecutive disparities before transposing means each cache
    // line receives KB consecutive k values instead of one. KB = 16 fills a
    // 64-byte line exactly.
    // Blockwise needs no transpose, so KB exists only to batch work. At KB=16 and
    // D=60 that is exactly four blocks for four threads -- no balancing at all, and
    // higher disparities do less work because the valid x range shrinks with d. So
    // one disparity per unit of work, handed out dynamically.
    //
    // Measured before changing it: 159.86 ms of CPU over 56.5 ms wall is 2.83 of
    // four threads busy, at a constant 3.36 GHz. The stage is not memory-bound --
    // stalls on L3 misses are 2-5% of cycles and DRAM traffic is 8% of this
    // machine's bandwidth -- it was simply idle.
    //
    // The vector score kernel needs SIMD_G disparities in hand at once, so it takes
    // its work in groups of that size and the ragged last group falls back to the
    // scalar path. At D=60 and SIMD_G=8 that is seven full groups and a group of
    // four, over six threads -- worse balance than one disparity at a time, and the
    // reason the wall-clock win is smaller than the score-loop win.
    const int KB = blockwise ? (cfg.simd ? SIMD_G : 1) : 16;
    const int nblocks = (D + KB - 1) / KB;
    std::atomic<int> next_block(0);
    std::vector<std::thread> cpool;
    g_pool_spawn = now_ms();
    for (int t = 0; t < nthreads; ++t) cpool.emplace_back([&, t]() {
    // Timed because it is on the critical path and no other timer covers it. Every
    // one of the four in-situ timers below starts after the scratch exists, so a
    // regression that lives in allocation is invisible in the breakdown and shows
    // up only as lost occupancy -- which is exactly how it presented.
    const double ta0 = now_ms();
    struct SpanGuard {
      double t0; double* out;
      ~SpanGuard() { *out = now_ms() - t0; }
    } span_guard{ta0, &ts_span[t]};
    // Only the chosen filter path's scratch is allocated. Every thread used to
    // take all eighteen planes -- 12 MB each, 46 MB across four threads -- and
    // std::vector zero-fills them, so most of a memset of 46 MB was being paid on
    // the critical path for buffers the recursive filter never touches.
    const size_t FN = cfg.rf ? 0 : size_t(W) * H;
    const bool i16 = cfg.rf && blockwise;
    std::vector<int16_t> islice(i16 ? size_t(W) * H : 0);
    // SIMD_G planes live at once instead of one: 2.7 MB a thread at 450x375, against
    // 2 MB of L2 shared across the A57 cluster. So a plane can no longer stay
    // resident from score through filter to insert, which is the one real cost of
    // this change and the reason it is measured rather than argued. The counters say
    // the stage had 1.6 GB/s of 18.9 and a core and a half idle, so there is room --
    // but that is a prediction, and traffic predictions here have been wrong by 7x.
    const bool simd_grp = cfg.simd && i16;
    // NOT a std::vector: it would zero-fill 2.7 MB a thread on construction and the
    // per-group std::fill immediately zeroes it again. Measured at 18.8 ms of
    // thread-summed CPU for the redundant pass -- the fourth time in this file that
    // vector's zero-fill has been the thing on the critical path, and the first time
    // any timer was watching the allocation itself.
    //
    // The fill that remains is not redundant: rf_filter_i16 runs over the whole
    // plane including the border and the left strip the score loop never writes, and
    // it propagates what it finds there into the valid window.
    const size_t GN = simd_grp ? size_t(SIMD_G) * W * H : 0;
    std::unique_ptr<int16_t[]> gbuf(simd_grp ? new int16_t[GN] : nullptr);
    int16_t* const gslice = gbuf.get();
    (void)gslice;
    std::vector<float> slice(i16 ? 0 : size_t(W) * H), tmp(FN);
    std::vector<float> ip(FN), mp(FN), mip(FN);
    std::vector<float> ab(FN), bb(FN);
    std::vector<float> ma(FN), mb(FN);
    std::vector<float> blk(blockwise ? 0 : size_t(KB) * W * H);
    // Thread-local running top-2, structure of arrays so the reject test touches
    // ONE plane. Separate arrays matter: a packed struct would pull 12 bytes per
    // pixel to answer a question that needs 4, and the reject is the common case.
    if (blockwise) {
      as0[t].assign(WH, -32768); as1[t].assign(WH, -32768);
      ad0[t].assign(WH, -1);     ad1[t].assign(WH, -1);
    }
    int16_t* ts0 = blockwise ? as0[t].data() : nullptr;
    int16_t* ts1 = blockwise ? as1[t].data() : nullptr;
    int16_t* td0 = blockwise ? ad0[t].data() : nullptr;
    int16_t* td1 = blockwise ? ad1[t].data() : nullptr;
    const size_t NS = cfg.rf ? 0 : ((F == 1) ? size_t(W) * H : size_t(Ws) * Hs);
    std::vector<float> ps(NS), ips(NS), mps(NS), mips(NS), ts(NS);
    std::vector<float> abs_(NS), bbs(NS), mas(NS), mbs(NS);
    ts_alloc[t] += now_ms() - ta0;
    for (int b = next_block.fetch_add(1); b < nblocks;
         b = next_block.fetch_add(1)) {
    const int klo = b * KB, khi = std::min(D, klo + KB);
#ifdef DE_HAVE_NEON
    // Vector path: score SIMD_G disparities with disparity in the lanes, then run
    // the unchanged filter and insert over each of the SIMD_G planes it produced.
    // A ragged final group falls through to the scalar loop below.
    if (simd_grp && khi - klo == SIMD_G) {
      double tm = now_ms();
      // Zero only what the score kernel does not write. Clearing the whole buffer
      // costs 7.8x here what it costs in the scalar path for the same byte count --
      // 14.0 ms against 1.8 -- because one 337 KB plane refilled sixty times stays
      // in L2 and eight planes at 2.7 MB cannot. The kernel covers x in [3+d, W-3)
      // for every y in [3, H-3), so only the border and the left strip are left.
      for (int g = 0; g < SIMD_G; ++g) {
        int16_t* pl = gslice + size_t(g) * WH;
        const int lft = std::min(3 + cfg.dmin + klo + g, W);
        std::fill(pl, pl + size_t(3) * W, int16_t(0));
        for (int y = 3; y < H - 3; ++y) {
          int16_t* rw = pl + size_t(y) * W;
          std::fill(rw, rw + lft, int16_t(0));
          std::fill(rw + std::max(lft, W - 3), rw + W, int16_t(0));
        }
        std::fill(pl + size_t(H - 3) * W, pl + WH, int16_t(0));
      }
      ts_fill[t] += now_ms() - tm; tm = now_ms();
      int16_t tbl[64] = {0};
      const int half = 24;
      for (int h = 0; h <= 2 * half; ++h)
        tbl[h] = int16_t(((half - h) * SCORE_ONE) / half);
      const int32_t wq = int32_t(std::max(0.f, std::min(1.f, cfg.ad)) * 1024.f);
      const int Tt = std::max(1, cfg.ad_trunc);
      int16_t adt[256];
      for (int v = 0; v < 256; ++v)
        adt[v] = int16_t(SCORE_ONE - (2 * SCORE_ONE * std::min(v, Tt)) / Tt);
      score_group_neon(cl.data(), cr.data(), L.data.data(), R.data.data(),
                       gslice, WH, W, H, cfg.dmin + klo, tbl, adt, wq, Tt);
      ts_score[t] += now_ms() - tm;
      for (int g = 0; g < SIMD_G; ++g) {
        int16_t* pl = gslice + size_t(g) * WH;
        tm = now_ms();
        rf_filter_i16(pl, axp, ayp, W, H);
        ts_filt[t] += now_ms() - tm; tm = now_ms();
        const int dg = cfg.dmin + klo + g;
        const int16_t kk = int16_t(klo + g);
        for (int y = 3; y < H - 3; ++y) {
          const size_t rw = size_t(y) * W;
          for (int x = 3 + dg; x < W - 3; ++x) {
            const size_t i = rw + x;
            const int16_t v = pl[i];
            if (v <= ts1[i]) continue;
            if (v > ts0[i]) {
              ts1[i] = ts0[i]; td1[i] = td0[i];
              ts0[i] = v;      td0[i] = kk;
            } else {
              ts1[i] = v;      td1[i] = kk;
            }
          }
        }
        ts_ins[t] += now_ms() - tm;
      }
      continue;
    }
#endif
    for (int k = klo; k < khi; ++k) {
      const int d = cfg.dmin + k;
      // Masked: this plane's rectangle, or nothing at all. The score is computed
      // MPAD beyond the interval so the filter aggregates real neighbours, the
      // filter runs over the padded rectangle, and the insert stays strictly inside
      // the interval -- pixels outside it never see d as a candidate.
      int rylo = 3, ryhi = H - 3, rxlo = 3 + d, rxhi = W - 3;
      if (mask) {
        if (pyhi[k] < pylo[k]) continue;             // no pixel wants this plane
        rylo = std::max(rylo, pylo[k] - MPAD);
        ryhi = std::min(ryhi, pyhi[k] + 1 + MPAD);
        rxlo = std::max(rxlo, pxlo[k] - MPAD);
        rxhi = std::min(rxhi, pxhi[k] + 1 + MPAD);
        if (ryhi - rylo < 1 || rxhi - rxlo < 2) continue;
      }
      double tm = now_ms();
      if (mask && i16) {
        // Zero the rectangle only: the filter reads nothing outside it.
        for (int y = rylo; y < ryhi; ++y)
          std::fill(islice.begin() + size_t(y) * W + rxlo,
                    islice.begin() + size_t(y) * W + rxhi, int16_t(0));
      }
      else if (i16) std::fill(islice.begin(), islice.end(), int16_t(0));
      else std::fill(slice.begin(), slice.end(), 0.f);
      ts_fill[t] += now_ms() - tm; tm = now_ms();
      // Multiply by the reciprocal, do not divide.
      //
      // This inner loop runs D * H * W times -- 9.6M for a 450x375 pair with 60
      // disparities -- and a float division is roughly twenty cycles against four
      // for a multiply. The compiler cannot make the substitution itself: 1/24 is
      // not exactly representable, so turning `x / 24.f` into `x * (1/24.f)`
      // changes the last bit and is forbidden without -freciprocal-math.
      //
      // Measured: 66.7 -> 59.5 ms in the score-and-scatter stage, 11% of it.
      // Worth having and not the whole story -- what remains in that 59.5 ms is
      // still unattributed, and reasoning about it has a poor record here.
      const float inv_half = 1.f / 24.f;
      if (i16) {
        // 49 possible Hamming distances, so the scale conversion is a table
        // lookup rather than an integer divide per pixel-disparity.
        int16_t tbl[49];
        const int half = cfg.csct ? 12 : 24;
        for (int h = 0; h <= 2 * half; ++h)
          tbl[h] = int16_t(((half - h) * SCORE_ONE) / half);
        // Graded cost: Census plus a truncated absolute difference.
        //
        // Census gives 49 quantisation levels (25 under --csct) and is invariant to
        // monotonic intensity change, which is what makes it robust. Its weakness is
        // that a flat region produces near-identical descriptors and the score has
        // nothing to separate candidates with. An absolute difference is continuous
        // and has exactly the complementary failure -- it breaks under gain and
        // exposure differences where Census does not. Combining them is standard and
        // is the level-count argument running the other way from --csct, which
        // measurably lost accuracy by halving the levels.
        //
        // Truncated so that an occlusion or a specular pixel cannot dominate.
        const int32_t wq = int32_t(std::max(0.f, std::min(1.f, cfg.ad)) * 1024.f);
        int16_t adt[256];
        if (wq) {
          const int T = std::max(1, cfg.ad_trunc);
          for (int v = 0; v < 256; ++v)
            adt[v] = int16_t(SCORE_ONE - (2 * SCORE_ONE * std::min(v, T)) / T);
        }
        if (wq && !cfg.csct) {
          const uint8_t* Ld = L.data.data();
          const uint8_t* Rd = R.data.data();
          for (int y = rylo; y < ryhi; ++y)
            for (int x = rxlo; x < rxhi; ++x) {
              const size_t i = size_t(y) * W + x;
              const int32_t c = tbl[popcnt64(cl[i] ^ cr[i - size_t(d)])];
              const int32_t a = adt[std::abs(int(Ld[i]) - int(Rd[i - size_t(d)]))];
              islice[i] = int16_t((c * (1024 - wq) + a * wq) >> 10);
            }
        } else if (cfg.csct) {
          for (int y = 3; y < H - 3; ++y)
            for (int x = 3 + d; x < W - 3; ++x)
              islice[size_t(y) * W + x] = tbl[__builtin_popcount(
                  cl32[size_t(y) * W + x] ^ cr32[size_t(y) * W + x - d])];
        } else
        for (int y = rylo; y < ryhi; ++y)
          for (int x = rxlo; x < rxhi; ++x)
            islice[size_t(y) * W + x] =
                tbl[popcnt64(cl[size_t(y) * W + x] ^ cr[size_t(y) * W + x - d])];
        ts_score[t] += now_ms() - tm; tm = now_ms();
        if (mask) rf_filter_i16_rect(islice.data(), axp, ayp, W,
                                     rylo, ryhi, rxlo, rxhi);
        else      rf_filter_i16(islice.data(), axp, ayp, W, H);
        ts_filt[t] += now_ms() - tm; tm = now_ms();
        const int16_t kk = int16_t(k);
        for (int y = mask ? std::max(rylo, pylo[k]) : rylo;
             y < (mask ? std::min(ryhi, pyhi[k] + 1) : ryhi); ++y) {
          const size_t row = size_t(y) * W;
          // Strictly inside the interval when masked: a pixel outside it never
          // sees d as a candidate, however good the padded score looks.
          const int xa = mask ? std::max(rxlo + 0, int(mx0[size_t(y) * D + k])) : rxlo;
          const int xb = mask ? std::min(rxhi, int(mx1[size_t(y) * D + k]) + 1) : rxhi;
          for (int x = xa; x < xb; ++x) {
            const size_t i = row + x;
            // Deliberately NO per-pixel membership test: the row interval is the
            // union over pixels, so a pixel between two band regions can receive
            // candidates its own band would exclude. Tightening this to strict
            // |d - prior| <= band was measured at a full point worse (10.6% ->
            // 11.6% pooled): the slack admits extra candidates, and everything
            // measured on this matcher says candidates decide the outcome.
            const int16_t v = islice[i];
            if (v <= ts1[i]) continue;
            if (v > ts0[i]) {
              ts1[i] = ts0[i]; td1[i] = td0[i];
              ts0[i] = v;      td0[i] = kk;
            } else {
              ts1[i] = v;      td1[i] = kk;
            }
          }
        }
        ts_ins[t] += now_ms() - tm;
        continue;
      }
      // Left in index form deliberately. Hoisting row pointers out of this loop --
      // the change that was worth 1.62x in the census transform -- measures 14%
      // SLOWER here (48.4 ms against 42.4, interleaved best-of-6). GCC already
      // eliminates the common subexpression, and the explicit pointers defeat
      // something it was doing on top. Same transformation, opposite sign, two
      // loops apart.
      for (int y = 3; y < H - 3; ++y)
        for (int x = 3 + d; x < W - 3; ++x)
          slice[size_t(y) * W + x] =
              (24.f - float(popcnt64(cl[size_t(y) * W + x] ^
                                     cr[size_t(y) * W + x - d]))) * inv_half;
      ts_score[t] += now_ms() - tm; tm = now_ms();
      // The block buffer is filled here, before the filter, so the filter's
      // final combine can write into it directly instead of writing a full
      // plane and having it copied straight back out again.
      float* dst = blockwise ? nullptr : &blk[size_t(k - klo) * W * H];
      if (dst) std::fill(dst, dst + size_t(W) * H, -1e30f);
      bool staged = false;
      if (cfg.rf) {
        rf_filter(slice.data(), axp, ayp, W, H);
      } else if (r > 0 && cfg.guided && F == 1) {
        // Direct full-resolution guided filter. Kept as a separate path because
        // routing F == 1 through the subsample/upsample machinery costs a
        // redundant copy and two bilinear passes for nothing: 207 ms against
        // 155 ms in the cost stage.
        for (size_t i = 0; i < ps.size(); ++i) ips[i] = Is[i] * slice[i];
        box_filter(slice.data(), mps.data(), ts.data(), W, H, r);
        box_filter(ips.data(), mips.data(), ts.data(), W, H, r);
        for (size_t i = 0; i < abs_.size(); ++i) {
          const float cov = mips[i] - mIs[i] * mps[i];
          abs_[i] = cov / (vIs[i] + cfg.eps);
          bbs[i] = mps[i] - abs_[i] * mIs[i];
        }
        box_filter(abs_.data(), ma.data(), ts.data(), W, H, r);
        box_filter(bbs.data(), mb.data(), ts.data(), W, H, r);
        // Combine and stage in one pass. The previous form wrote ma*I+mb across
        // the whole plane and then copied the valid window of it into the block
        // buffer -- a 675 KB write, read and write again per slice, 81 MB over
        // the volume, for values that were already in registers. Only the valid
        // window is ever read downstream, so computing it straight into place is
        // the same arithmetic on the same elements.
        if (dst) {
          for (int y = 3; y < H - 3; ++y)
            for (int x = 3 + d; x < W - 3; ++x) {
              const size_t i = size_t(y) * W + x;
              dst[i] = ma[i] * I[i] + mb[i];
            }
          staged = true;
        } else {
          for (size_t i = 0; i < slice.size(); ++i) slice[i] = ma[i] * I[i] + mb[i];
        }
      } else if (r > 0 && cfg.guided) {
        // Fast variant. Measured: 2.4 points of bad-1.0 worse at F=4 for 84 ms,
        // because a stereo cost slice is not smooth the way an image is.
        downsample(slice.data(), ps.data(), W, H, Ws, Hs, F);
        for (size_t i = 0; i < ps.size(); ++i) ips[i] = Is[i] * ps[i];
        box_filter(ps.data(), mps.data(), ts.data(), Ws, Hs, rs);
        box_filter(ips.data(), mips.data(), ts.data(), Ws, Hs, rs);
        for (size_t i = 0; i < abs_.size(); ++i) {
          const float cov = mips[i] - mIs[i] * mps[i];
          abs_[i] = cov / (vIs[i] + cfg.eps);
          bbs[i] = mps[i] - abs_[i] * mIs[i];
        }
        box_filter(abs_.data(), mas.data(), ts.data(), Ws, Hs, rs);
        box_filter(bbs.data(), mbs.data(), ts.data(), Ws, Hs, rs);
        upsample(mas.data(), ma.data(), W, H, Ws, Hs, F);
        upsample(mbs.data(), mb.data(), W, H, Ws, Hs, F);
        for (size_t i = 0; i < slice.size(); ++i) slice[i] = ma[i] * I[i] + mb[i];
      } else if (r > 0) {
        box_filter(slice.data(), mp.data(), tmp.data(), W, H, r);
        slice.swap(mp);
      }
      ts_filt[t] += now_ms() - tm; tm = now_ms();
      if (blockwise) {
        // Insert this disparity into the running top-2, valid window only.
        //
        // The first test rejects almost everything and reads only ts1, which is
        // the same 675 KB plane on every one of the D slices and therefore stays
        // in cache. So the common cost per pixel is one streaming slice read plus
        // one cached compare -- which is what replaces staging and transposing
        // 120 MB through a 40 MB volume.
        const int16_t kk = int16_t(k);
        for (int y = 3; y < H - 3; ++y) {
          const size_t row = size_t(y) * W;
          for (int x = 3 + d; x < W - 3; ++x) {
            const size_t i = row + x;
            const float v = slice[i];
            if (v <= ts1[i]) continue;
            if (v > ts0[i]) {
              ts1[i] = ts0[i]; td1[i] = td0[i];
              ts0[i] = v;      td0[i] = kk;
            } else {
              ts1[i] = v;      td1[i] = kk;
            }
          }
        }
        ts_ins[t] += now_ms() - tm;
        continue;
      }
      // The other filter paths still produce a plane, so they stage as before.
      if (!staged)
        for (int y = 3; y < H - 3; ++y)
          for (int x = 3 + d; x < W - 3; ++x)
            dst[size_t(y) * W + x] = slice[size_t(y) * W + x];
    }
    // Blocked transpose: KB consecutive k values per output cache line.
    if (!blockwise) {
      const int kn = khi - klo;
      for (size_t px = 0; px < size_t(W) * H; ++px) {
        float* out = &vol[px * D + klo];
        for (int j = 0; j < kn; ++j) out[j] = blk[size_t(j) * W * H + px];
      }
    }
    }
    });
    for (auto& th : cpool) th.join();
    g_pool_join = now_ms();

  }
  T->cost = now_ms() - tc0;
  T->prologue = g_pool_spawn - tc0;
  T->after = now_ms() - g_pool_join;
  double c_fill = 0, c_score = 0, c_filt = 0, c_ins = 0, c_alloc = 0;
  double c_span = 0, span_max = 0;
  for (int t = 0; t < nthreads; ++t) {
    c_fill += ts_fill[t]; c_score += ts_score[t];
    c_filt += ts_filt[t]; c_ins += ts_ins[t];
    c_alloc += ts_alloc[t];
    c_span += ts_span[t];
    span_max = std::max(span_max, ts_span[t]);
  }

  // The aggregated volume, for measuring how many candidates per pixel are
  // actually needed. Diagnostic only; 40 MB for a 450x375 pair at D=60.
  if (!volp.empty()) {
    FILE* f = std::fopen(volp.c_str(), "wb");
    if (f) { std::fwrite(vol.data(), sizeof(float), vol.size(), f); std::fclose(f); }
  }

  std::vector<float> disp(size_t(W) * H, std::nanf(""));
  std::vector<float> margin(size_t(W) * H, 0.f);

  const double t1 = now_ms();
  std::vector<std::thread> pool;
  for (int t = 0; t < nthreads; ++t) {
    pool.emplace_back([&, t]() {
      // Per-thread scratch for whichever path runs, and only that one.
      //
      // The dense planes are W*D floats -- 108 KB each at 450x60, so 324 KB a
      // thread -- and std::vector zero-fills them. Allocated unconditionally they
      // were 2 MB of memset across six threads for buffers the sparse path never
      // reads. Measured on the TX2 this was most of the solve stage; it barely
      // showed on the desktop, which is the whole reason for measuring on the
      // target.
      const int K = cfg.topk > 0 ? std::min(cfg.topk, D) : 0;
      const size_t DN = K > 0 ? 0 : size_t(W) * D;       // dense path only
      const size_t GN = K > 0 ? 0 : size_t(D) + 2;
      const size_t WN = K > 0 ? 0 : size_t(W);
      // Blockwise now fills the same per-thread row buffers the vol path uses,
      // via the fused per-row merge below, so CN is needed in both modes.
      const size_t CN = K > 0 ? size_t(W) * K : 0;
      std::vector<float> s(DN), beta(DN), rho(DN);
      std::vector<float> gath(GN), svals(GN);
      std::vector<int> idxs(GN), k0(WN), k1(WN), bestk(W), order(W);
      std::vector<float> best(W);
      std::vector<char> taken(W);
      std::vector<float> cs(CN);
      std::vector<float> sbeta(size_t(W) * std::max(1, K));
      std::vector<float> srho(size_t(W) * std::max(1, K));
      std::vector<int> cd(CN), cn(CN ? size_t(W) : 0);
      std::vector<int> bstart(size_t(W) + 1), bitems(size_t(W) * std::max(1, K));
      std::vector<int> cursor(W);
      for (int y = t; y < H; y += nthreads) {
        if (K > 0) {
          if (blockwise) {
            // The 2-of-2n selection, fused into the solve. This used to be a
            // separate pass over 8 MB of merged-candidate arrays, run in its own
            // thread pool between the cost join and the solve spawn: ~30 ms of
            // stage wall at 848x480 in which the six cost workers -- measured at
            // 5.99 of 6 busy while alive -- were all dead. Row-local, the
            // candidates are written hot into the same 28 KB the solver reads
            // back, and the pass disappears into the solve's own pool.
            const float q = 1.f / float(SCORE_ONE);
            const size_t row = size_t(y) * W;
            for (int x = 0; x < W; ++x) {
              const size_t i = row + x;
              int16_t b0 = -32768, b1 = -32768;
              int i0 = -1, i1 = -1;
              for (int tt = 0; tt < nthreads; ++tt) {
                const int16_t v0 = as0[tt][i], v1 = as1[tt][i];
                if (v0 > b0)      { b1 = b0; i1 = i0; b0 = v0; i0 = ad0[tt][i]; }
                else if (v0 > b1) { b1 = v0; i1 = ad0[tt][i]; }
                if (v1 > b0)      { b1 = b0; i1 = i0; b0 = v1; i0 = ad1[tt][i]; }
                else if (v1 > b1) { b1 = v1; i1 = ad1[tt][i]; }
              }
              int n = 0;
              // Back to float here: the solver's lambda/gamma comparisons and the
              // margin stay in float, and it is a few ms, so there is nothing to
              // win by quantising it and a real risk in doing so.
              if (i0 >= 0) { cs[size_t(x) * 2] = float(b0) * q; cd[size_t(x) * 2] = i0; n = 1; }
              if (i1 >= 0) { cs[size_t(x) * 2 + 1] = float(b1) * q; cd[size_t(x) * 2 + 1] = i1; n = 2; }
              cn[x] = n;
            }
          }
          solve_row_sparse(W, D, K, cfg,
                           blockwise ? nullptr : &vol[size_t(y) * W * D],
                           cs.data(), cd.data(), cn.data(),
                           sbeta.data(), srho.data(),
                           &disp[size_t(y) * W], &margin[size_t(y) * W], 3,
                           bstart.data(), bitems.data(), cursor.data(),
                           best.data(), bestk.data(), order.data(), taken.data());
          continue;
        }
        solve_row(y, W, D, cfg, &vol[size_t(y) * W * D],
                  s.data(), beta.data(), rho.data(),
                  &disp[size_t(y) * W], &margin[size_t(y) * W], 3,
                  gath.data(), svals.data(), idxs.data(), k0.data(), k1.data(),
                  best.data(), bestk.data(), order.data(), taken.data());
      }
    });
  }
  for (auto& th : pool) th.join();
  T->solve = now_ms() - t1;
  disp_out->swap(disp);
  margin_out->swap(margin);
  for (int t = 0; t < nthreads; ++t) {
    T->c_alloc += ts_alloc[t]; T->c_fill += ts_fill[t]; T->c_score += ts_score[t];
    T->c_filt += ts_filt[t];   T->c_ins += ts_ins[t];
    T->c_span += ts_span[t];
    T->span_max = std::max(T->span_max, ts_span[t]);
  }
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
        "usage: %s LEFT.y8 RIGHT.y8 W H [--dmax N] [--iters N] [--threads N]\n"
        "          [--min-margin F] [--out disp.f32] [--simd]\n"
        "  --simd  NEON score kernel, AArch64 only, bit-identical, off by default:\n"
        "          1.6x on the score loop and 0.92x on the stage at D=60. See\n"
        "          doc/09-matching.md.\n", argv[0]);
    return 2;
  }
  const std::string lp = argv[1], rp = argv[2];
  const int W = std::atoi(argv[3]), H = std::atoi(argv[4]);
  Cfg cfg;
  std::string outp, volp, priorp;
  for (int i = 5; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has = i + 1 < argc;
    if (a == "--dmax" && has) cfg.dmax = std::atoi(argv[++i]);
    else if (a == "--iters" && has) cfg.iters = std::atoi(argv[++i]);
    else if (a == "--agg" && has) cfg.agg = std::atoi(argv[++i]);
    else if (a == "--subpixel") cfg.subpixel = true;
    else if (a == "--box") cfg.guided = false;
    else if (a == "--rf") cfg.rf = true;
    else if (a == "--guided") cfg.rf = false;
    else if (a == "--sigma-s" && has) cfg.sigma_s = float(std::atof(argv[++i]));
    else if (a == "--sigma-r" && has) cfg.sigma_r = float(std::atof(argv[++i]));
    else if (a == "--eps" && has) cfg.eps = float(std::atof(argv[++i]));
    else if (a == "--fgf" && has) cfg.fgf = std::max(1, std::atoi(argv[++i]));
    else if (a == "--threads" && has) cfg.threads = std::atoi(argv[++i]);
    else if (a == "--min-margin" && has) cfg.min_margin = float(std::atof(argv[++i]));
    else if (a == "--out" && has) outp = argv[++i];
    else if (a == "--dump-vol" && has) volp = argv[++i];
    else if (a == "--topk" && has) cfg.topk = std::atoi(argv[++i]);
    else if (a == "--band" && has) cfg.band = std::atoi(argv[++i]);
    else if (a == "--csct") cfg.csct = true;
    else if (a == "--ad" && has) cfg.ad = float(std::atof(argv[++i]));
    else if (a == "--ad-trunc" && has) cfg.ad_trunc = std::atoi(argv[++i]);
    else if (a == "--prior" && has) priorp = argv[++i];
    else if (a == "--simd") cfg.simd = true;
    else if (a == "--c2f") cfg.c2f = true;
  }
  Image8 L, R;
  if (!load_raw_y8(lp, W, H, &L) || !load_raw_y8(rp, W, H, &R)) {
    std::fprintf(stderr, "failed to load %s / %s at %dx%d\n",
                 lp.c_str(), rp.c_str(), W, H);
    return 1;
  }
  // --- optional prior, for a coarse-to-fine search ---------------------------
  //
  // The volume is indexed by OFFSET FROM THE PRIOR rather than by absolute
  // disparity, so there are 2*band+1 whole planes instead of D. That is what keeps
  // the recursive filter usable: it still sees one plane per index and aggregates
  // at constant offset, which inside a smooth region is constant disparity -- the
  // assumption window aggregation already makes. At a prior discontinuity the plane
  // does mix disparities, but the filter is edge-aware and prior discontinuities
  // sit on intensity edges, so it is better aligned here than a fixed-disparity
  // plane is, not worse.
  //
  // Holes must already be filled by the caller. A hole means no search band at
  // all, and measuring the ceiling with holes left in understated it by 13 points
  // (see 09-matching.md) -- so this refuses a prior it cannot use rather than
  // silently scoring one.
  std::vector<float> prior;
  if (!priorp.empty()) {
    FILE* f = std::fopen(priorp.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open prior %s\n", priorp.c_str()); return 1; }
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    const size_t n = size_t(bytes) / sizeof(float);
    std::vector<float> raw(n);
    if (std::fread(raw.data(), sizeof(float), n, f) != n) {
      std::fprintf(stderr, "short read on prior %s\n", priorp.c_str());
      std::fclose(f); return 1;
    }
    std::fclose(f);
    prior.assign(size_t(W) * H, std::nanf(""));
    if (n == size_t(W) * H) {
      prior = raw;
    } else if (n == size_t(W / 2) * (H / 2)) {
      // Half resolution: nearest upsample, and a disparity of d at half scale is
      // 2d at full scale.
      const int W2 = W / 2, H2 = H / 2;
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
          prior[size_t(y) * W + x] =
              2.f * raw[size_t(std::min(H2 - 1, y / 2)) * W2 + std::min(W2 - 1, x / 2)];
    } else {
      std::fprintf(stderr,
                   "prior %s has %zu floats, expected %zu (full) or %zu (half)\n",
                   priorp.c_str(), n, size_t(W) * H, size_t(W / 2) * (H / 2));
      return 1;
    }
    size_t holes = 0;
    for (float v : prior) if (!std::isfinite(v)) ++holes;
    if (holes) {
      std::fprintf(stderr, "prior has %zu holes of %zu; fill them first\n",
                   holes, prior.size());
      return 1;
    }
    if (cfg.topk != 2) {
      std::fprintf(stderr, "--prior requires --topk 2\n");
      return 1;
    }
  }

  if ((cfg.c2f || !priorp.empty()) &&
      !(cfg.rf && cfg.topk == 2 && volp.empty() && !cfg.csct && !cfg.simd)) {
    // The mask is wired through exactly one path: recursive filter, top-2,
    // blockwise, scalar graded cost. Everything else refuses rather than silently
    // running a full sweep under a flag that says otherwise.
    std::fprintf(stderr, "--c2f/--prior require the default path (--rf, --topk 2, "
                         "no --dump-vol, no --csct, no --simd)\n");
    return 1;
  }
  if (cfg.c2f && !priorp.empty()) {
    std::fprintf(stderr, "--c2f computes its own prior; drop --prior\n");
    return 1;
  }
  if (cfg.csct && !(cfg.rf && cfg.topk == 2 && volp.empty())) {
    // The other paths still read the 64-bit descriptors, which are not built in
    // this mode. Refusing beats reading an empty vector.
    std::fprintf(stderr, "--csct requires the default path (--rf, --topk 2, no "
                         "--dump-vol)\n");
    return 1;
  }

  // --simd refuses rather than falling back. A flag that silently does nothing is
  // indistinguishable downstream from one that did something and bought nothing,
  // which is the whole of rule 1 -- and this flag exists to be timed.
  if (cfg.simd) {
#ifndef DE_HAVE_NEON
    std::fprintf(stderr, "--simd is the AArch64 NEON kernel; this is not an "
                         "aarch64 build\n");
    return 1;
#else
    if (cfg.csct) {
      std::fprintf(stderr, "--simd reads the 64-bit descriptors; not --csct\n");
      return 1;
    }
    if (!(cfg.rf && cfg.topk == 2 && volp.empty() && priorp.empty())) {
      std::fprintf(stderr, "--simd requires the default path (--rf, --topk 2, no "
                           "--dump-vol, no --prior)\n");
      return 1;
    }
    if (cfg.ad_trunc < 1 || cfg.ad_trunc > 63) {
      std::fprintf(stderr, "--simd needs 1 <= --ad-trunc <= 63 so the clamped "
                           "difference is a legal 64-entry table index (got %d)\n",
                   cfg.ad_trunc);
      return 1;
    }
#endif
  }

  const int D = cfg.dmax - cfg.dmin + 1;
  int nthreads = cfg.threads > 0 ? cfg.threads
                                 : int(std::thread::hardware_concurrency());
  if (nthreads < 1) nthreads = 1;

  // --- the coarse pass: half resolution, half the disparities, UNGATED ------------
  //
  // D/2 over a quarter of the pixels is D/8 of the work, and the result is only a
  // search band, so it is run with min_margin 0: the first ceiling measurement was
  // wrong by 13 points precisely because a gated prior leaves holes, and a hole is
  // a pixel with no search band at all (09-matching.md). Holes that remain because
  // the solver was undecided are filled from row neighbours, then columns.
  LevelTimes TC, TF;
  double t_prep = 0;
  if (cfg.c2f) {
    const double tt0 = now_ms();
    const int W2 = W / 2, H2 = H / 2;
    Image8 L2, R2;
    L2.width = R2.width = W2; L2.height = R2.height = H2;
    L2.data.resize(size_t(W2) * H2); R2.data.resize(size_t(W2) * H2);
    for (int y = 0; y < H2; ++y)
      for (int x = 0; x < W2; ++x) {
        const size_t a = size_t(2 * y) * W + 2 * x;
        L2.data[size_t(y) * W2 + x] = uint8_t((int(L.data[a]) + L.data[a + 1] +
                                               L.data[a + W] + L.data[a + W + 1] + 2) / 4);
        R2.data[size_t(y) * W2 + x] = uint8_t((int(R.data[a]) + R.data[a + 1] +
                                               R.data[a + W] + R.data[a + W + 1] + 2) / 4);
      }
    Cfg cc = cfg;
    cc.c2f = false;
    cc.dmin = std::max(1, cfg.dmin / 2);
    cc.dmax = (cfg.dmax + 1) / 2;
    cc.min_margin = 0.f;
    // One iteration, not cfg.iters: the coarse answer only needs to land within
    // +-band of the truth, not converge. Measured: pooled accuracy moves < 0.1
    // either way, and the coarse solve is the third-largest item on the TX2.
    cc.iters = 1;
    t_prep += now_ms() - tt0;
    std::vector<float> dh, mh;
    run_dense(L2, R2, cc, nthreads, {}, "", &dh, &mh, &TC);
    const double tt1 = now_ms();
    // Row-nearest hole fill, then column-nearest for rows with nothing at all.
    for (int y = 0; y < H2; ++y) {
      float* r = &dh[size_t(y) * W2];
      float last = std::nanf("");
      for (int x = 0; x < W2; ++x) {
        if (std::isfinite(r[x])) last = r[x];
        else if (std::isfinite(last)) r[x] = last;
      }
      last = std::nanf("");
      for (int x = W2 - 1; x >= 0; --x) {
        if (std::isfinite(r[x])) last = r[x];
        else if (std::isfinite(last)) r[x] = last;
      }
    }
    for (int x = 0; x < W2; ++x) {
      float last = std::nanf("");
      for (int y = 0; y < H2; ++y) {
        float& v = dh[size_t(y) * W2 + x];
        if (std::isfinite(v)) last = v;
        else if (std::isfinite(last)) v = last;
      }
      last = std::nanf("");
      for (int y = H2 - 1; y >= 0; --y) {
        float& v = dh[size_t(y) * W2 + x];
        if (std::isfinite(v)) last = v;
        else if (std::isfinite(last)) v = last;
      }
    }
    size_t holes = 0;
    for (float v : dh) if (!std::isfinite(v)) ++holes;
    if (holes) {
      // A whole image with no finite coarse disparity. Nothing to mask with, so
      // fall back to the full sweep rather than deliver an empty answer.
      std::fprintf(stderr, "c2f: coarse level empty (%zu holes), full sweep\n",
                   holes);
    } else {
      prior.assign(size_t(W) * H, 0.f);
      // Nearest upsample; a disparity of d at half scale is 2d at full scale.
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
          prior[size_t(y) * W + x] =
              2.f * dh[size_t(std::min(H2 - 1, y / 2)) * W2 +
                       std::min(W2 - 1, x / 2)];
    }
    t_prep += now_ms() - tt1;
  }

  std::vector<float> disp, margin;
  run_dense(L, R, cfg, nthreads, prior, volp, &disp, &margin, &TF);

  size_t filled = 0;
  for (float v : disp) if (std::isfinite(v)) ++filled;
  std::printf("%dx%d  D=%d  iters=%d  threads=%d\n", W, H, D, cfg.iters, nthreads);
  std::printf("  census breakdown (thread-summed, both images): clear %.2f  "
              "48 compare-or passes %.2f  pack to uint64 %.2f ms\n",
              g_cen_clear, g_cen_cmp, g_cen_pack);
  std::printf("  cost breakdown (thread-summed): alloc %.1f  clear %.1f  score %.1f  "
              "filter %.1f  insert %.1f ms\n",
              TF.c_alloc, TF.c_fill, TF.c_score, TF.c_filt, TF.c_ins);
  {
    const double busy = TF.c_alloc + TF.c_fill + TF.c_score + TF.c_filt + TF.c_ins;
    std::printf("  occupancy: busy %.1f ms / spans %.1f ms = %.2f of %d threads busy "
                "while alive; longest span %.1f of %.1f ms stage wall\n",
                busy, TF.c_span, TF.c_span > 0 ? busy / TF.c_span * nthreads : 0.0,
                nthreads, TF.span_max, TF.cost);
    std::printf("  outside the pool: prologue %.1f ms (alloc %.1f, normalize %.1f, "
                "rf_coeffs %.1f, spawn %.1f) + after-pool %.1f ms of %.1f stage wall\n",
                TF.prologue, TF.tp_alloc, TF.tp_norm, TF.tp_coeff,
                TF.prologue - TF.tp_alloc - TF.tp_norm - TF.tp_coeff, TF.after,
                TF.cost);
  }
  if (cfg.c2f)
    // "sum", not "total": dense_bench and tx2_ab take the LAST line containing
    // "total" as the whole run, and this line is one level of it.
    std::printf("  coarse %dx%d D=%d: census %.1f  cost %.1f  solve %.1f  "
                "prior prep %.1f ms, sum %.1f\n",
                W / 2, H / 2, (cfg.dmax + 1) / 2 - std::max(1, cfg.dmin / 2) + 1,
                TC.census, TC.cost, TC.solve, t_prep,
                TC.census + TC.cost + TC.solve + t_prep);
  std::printf("census %.1f ms  cost %.1f ms  solve %.1f ms  total %.1f ms  "
              "agg=%d iters=%d  filled %.1f%%\n",
              TF.census, TF.cost, TF.solve,
              TC.census + TC.cost + TC.solve + t_prep +
                  TF.census + TF.cost + TF.solve,
              cfg.agg, cfg.iters,
              100.0 * double(filled) / double(disp.size()));
  if (!outp.empty()) {
    FILE* f = std::fopen(outp.c_str(), "wb");
    if (f) { std::fwrite(disp.data(), sizeof(float), disp.size(), f); std::fclose(f); }
  }
  return 0;
}
