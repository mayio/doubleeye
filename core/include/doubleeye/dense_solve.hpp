// The dense MASDA row solver and its configuration, shared between the CPU tool
// (de_dense.cpp) and the CUDA tool (de_dense_cuda.cu). The GPU owns the image
// plane -- census, cost, aggregation, top-2 candidates -- and this is the part the
// CPU keeps: the graph. Header-only so core/ still builds with plain make and no
// dependencies on either machine.
//
// Everything here is extracted verbatim from de_dense.cpp; the comments carry the
// measurements they were written with.

#ifndef DOUBLEEYE_DENSE_SOLVE_HPP
#define DOUBLEEYE_DENSE_SOLVE_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace doubleeye {

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


struct Top2 {
  float b1 = -1e30f, b2 = -1e30f;
  int i1 = -1;
  inline void push(float v, int i) {
    if (v > b1) { b2 = b1; b1 = v; i1 = i; }
    else if (v > b2) { b2 = v; }
  }
  inline float excl(int i) const { return i == i1 ? b2 : b1; }
};


inline

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


}  // namespace doubleeye

#endif  // DOUBLEEYE_DENSE_SOLVE_HPP
