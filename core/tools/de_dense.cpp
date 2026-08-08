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

// Census over the whole image, one uint64 per pixel. 7x7 -> 48 bits.
std::vector<uint64_t> census_image(const Image8& img, int hw, int hh, int nth) {
  const int W = img.width, H = img.height;
  std::vector<uint64_t> out(size_t(W) * H, 0);
  std::vector<std::thread> pool;
  for (int t = 0; t < nth; ++t) pool.emplace_back([&, t]() {
  for (int y = hh + t; y < H - hh; y += nth) {
    for (int x = hw; x < W - hw; ++x) {
      const int c = img.at(x, y);
      uint64_t v = 0;
      int bit = 0;
      for (int dy = -hh; dy <= hh; ++dy)
        for (int dx = -hw; dx <= hw; ++dx) {
          if (dx == 0 && dy == 0) continue;
          v |= uint64_t(img.at(x + dx, y + dy) < c) << bit;
          ++bit;
        }
      out[size_t(y) * W + x] = v;
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

struct Cfg {
  int dmin = 1, dmax = 60, iters = 2, agg = 3;
  bool subpixel = false;   // measured: hurts slightly, see below
  bool guided = true;      // edge-aware aggregation
  float eps = 0.0025f;     // guided-filter regularisation, I in [0,1]
  int fgf = 1;             // fast guided filter: measured NOT worth it, see below
  float lambda = -0.1f, gamma = -0.1f, damping = 0.4f, min_margin = 0.f;
  int threads = 0;
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
        "usage: %s LEFT.y8 RIGHT.y8 W H [--dmax N] [--iters N] [--threads N]\n"
        "          [--min-margin F] [--out disp.f32]\n", argv[0]);
    return 2;
  }
  const std::string lp = argv[1], rp = argv[2];
  const int W = std::atoi(argv[3]), H = std::atoi(argv[4]);
  Cfg cfg;
  std::string outp;
  for (int i = 5; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has = i + 1 < argc;
    if (a == "--dmax" && has) cfg.dmax = std::atoi(argv[++i]);
    else if (a == "--iters" && has) cfg.iters = std::atoi(argv[++i]);
    else if (a == "--agg" && has) cfg.agg = std::atoi(argv[++i]);
    else if (a == "--subpixel") cfg.subpixel = true;
    else if (a == "--box") cfg.guided = false;
    else if (a == "--eps" && has) cfg.eps = float(std::atof(argv[++i]));
    else if (a == "--fgf" && has) cfg.fgf = std::max(1, std::atoi(argv[++i]));
    else if (a == "--threads" && has) cfg.threads = std::atoi(argv[++i]);
    else if (a == "--min-margin" && has) cfg.min_margin = float(std::atof(argv[++i]));
    else if (a == "--out" && has) outp = argv[++i];
  }
  Image8 L, R;
  if (!load_raw_y8(lp, W, H, &L) || !load_raw_y8(rp, W, H, &R)) {
    std::fprintf(stderr, "failed to load %s / %s at %dx%d\n",
                 lp.c_str(), rp.c_str(), W, H);
    return 1;
  }
  const int D = cfg.dmax - cfg.dmin + 1;
  int nthreads = cfg.threads > 0 ? cfg.threads
                                 : int(std::thread::hardware_concurrency());
  if (nthreads < 1) nthreads = 1;

  const double t0 = now_ms();
  const std::vector<uint64_t> cl = census_image(L, 3, 3, nthreads);
  const std::vector<uint64_t> cr = census_image(R, 3, 3, nthreads);
  const double t_census = now_ms() - t0;

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
  const double tc0 = now_ms();
  std::vector<float> vol(size_t(W) * H * D, -1e30f);
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
  std::vector<float> Is(size_t(Ws) * Hs), mIs(size_t(Ws) * Hs), vIs(size_t(Ws) * Hs);
  {
    for (size_t i = 0; i < I.size(); ++i) I[i] = float(L.data[i]) / 255.f;
    std::vector<float> t1(size_t(Ws) * Hs), IIs(size_t(Ws) * Hs);
    downsample(I.data(), Is.data(), W, H, Ws, Hs, F);
    for (size_t i = 0; i < Is.size(); ++i) IIs[i] = Is[i] * Is[i];
    box_filter(Is.data(), mIs.data(), t1.data(), Ws, Hs, rs);
    box_filter(IIs.data(), vIs.data(), t1.data(), Ws, Hs, rs);
    for (size_t i = 0; i < vIs.size(); ++i) vIs[i] -= mIs[i] * mIs[i];
  }
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
    const int KB = 16;
    const int nblocks = (D + KB - 1) / KB;
    std::vector<std::thread> cpool;
    for (int t = 0; t < nthreads; ++t) cpool.emplace_back([&, t]() {
    std::vector<float> slice(size_t(W) * H), tmp(size_t(W) * H);
    std::vector<float> ip(size_t(W) * H), mp(size_t(W) * H), mip(size_t(W) * H);
    std::vector<float> ab(size_t(W) * H), bb(size_t(W) * H);
    std::vector<float> ma(size_t(W) * H), mb(size_t(W) * H);
    std::vector<float> blk(size_t(KB) * W * H);
    const size_t NS = (F == 1) ? size_t(W) * H : size_t(Ws) * Hs;
    std::vector<float> ps(NS), ips(NS), mps(NS), mips(NS), ts(NS);
    std::vector<float> abs_(NS), bbs(NS), mas(NS), mbs(NS);
    for (int b = t; b < nblocks; b += nthreads) {
    const int klo = b * KB, khi = std::min(D, klo + KB);
    for (int k = klo; k < khi; ++k) {
      const int d = cfg.dmin + k;
      std::fill(slice.begin(), slice.end(), 0.f);
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
      for (int y = 3; y < H - 3; ++y)
        for (int x = 3 + d; x < W - 3; ++x)
          slice[size_t(y) * W + x] =
              (24.f - float(popcnt64(cl[size_t(y) * W + x] ^
                                     cr[size_t(y) * W + x - d]))) * inv_half;
      // The block buffer is filled here, before the filter, so the filter's
      // final combine can write into it directly instead of writing a full
      // plane and having it copied straight back out again.
      float* dst = &blk[size_t(k - klo) * W * H];
      std::fill(dst, dst + size_t(W) * H, -1e30f);
      bool staged = false;
      if (r > 0 && cfg.guided && F == 1) {
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
        for (int y = 3; y < H - 3; ++y)
          for (int x = 3 + d; x < W - 3; ++x) {
            const size_t i = size_t(y) * W + x;
            dst[i] = ma[i] * I[i] + mb[i];
          }
        staged = true;
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
      // The other filter paths still produce a plane, so they stage as before.
      if (!staged)
        for (int y = 3; y < H - 3; ++y)
          for (int x = 3 + d; x < W - 3; ++x)
            dst[size_t(y) * W + x] = slice[size_t(y) * W + x];
    }
    // Blocked transpose: KB consecutive k values per output cache line.
    const int kn = khi - klo;
    for (size_t px = 0; px < size_t(W) * H; ++px) {
      float* out = &vol[px * D + klo];
      for (int j = 0; j < kn; ++j) out[j] = blk[size_t(j) * W * H + px];
    }
    }
    });
    for (auto& th : cpool) th.join();
  }
  const double t_cost = now_ms() - tc0;

  std::vector<float> disp(size_t(W) * H, std::nanf(""));
  std::vector<float> margin(size_t(W) * H, 0.f);

  const double t1 = now_ms();
  std::vector<std::thread> pool;
  for (int t = 0; t < nthreads; ++t) {
    pool.emplace_back([&, t]() {
      // Per-thread scratch, allocated once: three W*D float planes.
      std::vector<float> s(size_t(W) * D), beta(size_t(W) * D), rho(size_t(W) * D);
      std::vector<float> gath(size_t(D) + 2), svals(size_t(D) + 2);
      std::vector<int> idxs(size_t(D) + 2), k0(W), k1(W), bestk(W), order(W);
      std::vector<float> best(W);
      std::vector<char> taken(W);
      for (int y = t; y < H; y += nthreads) {
        solve_row(y, W, D, cfg, &vol[size_t(y) * W * D],
                  s.data(), beta.data(), rho.data(),
                  &disp[size_t(y) * W], &margin[size_t(y) * W], 3,
                  gath.data(), svals.data(), idxs.data(), k0.data(), k1.data(),
                  best.data(), bestk.data(), order.data(), taken.data());
      }
    });
  }
  for (auto& th : pool) th.join();
  const double t_solve = now_ms() - t1;

  size_t filled = 0;
  for (float v : disp) if (std::isfinite(v)) ++filled;
  std::printf("%dx%d  D=%d  iters=%d  threads=%d\n", W, H, D, cfg.iters, nthreads);
  std::printf("census %.1f ms  cost %.1f ms  solve %.1f ms  total %.1f ms  "
              "agg=%d iters=%d  filled %.1f%%\n",
              t_census, t_cost, t_solve, t_census + t_cost + t_solve,
              cfg.agg, cfg.iters,
              100.0 * double(filled) / double(disp.size()));
  if (!outp.empty()) {
    FILE* f = std::fopen(outp.c_str(), "wb");
    if (f) { std::fwrite(disp.data(), sizeof(float), disp.size(), f); std::fclose(f); }
  }
  return 0;
}
