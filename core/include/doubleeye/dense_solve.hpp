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
  // iters = 0 since 2026-08-11: the message passing does not pay in the DENSE path.
  // Measured on both benchmarks -- v3 25.18 against 26.04 bad-1.0, dense_bench 9.4%
  // against 9.8% -- each at ~0.4 points less coverage, which the gate buys back for
  // free, so ~0.5 points better at matched coverage. And it is 12 ms of the 23 ms
  // CPU solve at 848x480, which is most of what the keypoint wiring is short by.
  //
  // This does NOT turn MASDA off. The uniqueness constraint and the score margin --
  // the two things that distinguish this from a block matcher, and what makes the
  // dense map beat SGM at keypoints -- are unchanged. Only the loopy BP on top of a
  // TOP-2 candidate set is dropped, which is the case where it has least to decide.
  //
  // Untested where it might still matter: projected-dot IR, where descriptors are
  // 3.3x degenerate, and temporal association, where the candidate set is large and
  // 2-D. Neither has ground truth here. `--iters 2` restores the old behaviour.
  int dmin = 1, dmax = 60, iters = 0, agg = 3;
  bool rf = true;              // recursive edge-aware filter; --guided for the old one
  // sigma_r swept: 0.2 is the peak. sigma_s was NOT swept until 2026-08-10, and
  // 12 turned out to be past the optimum on both axes -- Middlebury v3, all 15
  // training scenes, official scoring, --threads 1:
  //   sigma_s  4    6      8      10     12     20     30
  //   bad-1.0  --   25.96  26.04  26.41  26.94  31.91  33.49
  //   coverage --   80.4%  80.1%  79.7%  79.3%  78.2%  77.3%
  // 8 is inside the plateau and clear of the degradation below 6; 6 measured
  // marginally better and the difference is inside scene-to-scene noise. Worth
  // 0.9 points of bad-1.0 and 0.8 of coverage over the old 12, and the filter is
  // an IIR whose cost does not depend on sigma, so it is free.
  // sigma_r 0.30 measured 0.15 better than 0.20 at sigma_s 8 -- inside noise, and
  // left alone rather than tuned on a difference that small.
  float sigma_s = 8.f, sigma_r = 0.20f;
  // Which estimator refines the winning disparity. A parabola assumes the cost is
  // locally quadratic; the graded cost's truncated-absolute-difference term is
  // piecewise LINEAR, which makes the surface locally V-shaped and is measured in
  // 2.2a as the thing that degrades the fit. The equiangular estimator is the
  // standard one for a V, and costs the same arithmetic.
  // ON by default since 2026-08-11: 25.18 -> 24.47 bad-1.0 on Middlebury v3 at
  // identical coverage, for the same arithmetic. Neutral (9.2 either way) on the
  // eight-scene native-resolution benchmark, whose one-pixel tolerance cannot
  // resolve a sub-pixel estimator at all -- so the two benchmarks do not disagree,
  // one of them is blind to the change. --fit-parabola restores the old estimator.
  bool fit_eq = true;
  bool subpixel = true;    // ON by default since 2026-08-10: worth 15.0 points
                           // of bad-1.0 on Middlebury v3. --no-subpixel disables.
  bool guided = true;      // edge-aware aggregation
  float eps = 0.0025f;     // guided-filter regularisation, I in [0,1]
  int fgf = 1;             // fast guided filter: measured NOT worth it, see below
  int topk = 2;                // candidates per pixel; 2 measured best, 0 = dense
  int band = 2;                // prior-guided: search +-band around the prior
  bool csct = false;           // centre-symmetric census: 24 bits instead of 48
  float ad = 0.15f;            // graded cost weight; 0.15 with trunc 10 measured best
  int ad_trunc = 10;           // truncation for the absolute difference, 8-bit
  float lambda = -0.1f, gamma = -0.1f, damping = 0.4f, min_margin = 0.f;
  // An alternative gate on the SAME two scores, as a ratio of costs rather than a
  // difference of similarities. Both are standard confidence measures -- the margin
  // is Hu and Mordohai's maximum-margin-naive, this is their peak-ratio-naive.
  //
  // OFF because the two benchmarks disagree, and article/confidence.py says why.
  // At bad-1.0 on the eight 450x375 scenes the ratio is better at every matched
  // coverage, by 0.4 points at 76% coverage rising to 1.3 at 39%. On Middlebury v3,
  // whose quarter-resolution threshold is 0.25 px, it is level or 0.2 worse. The
  // ratio ranks GROSS mismatches better and says almost nothing more than the
  // margin about sub-pixel accuracy, so the tolerance decides which benchmark
  // sees a win. `--min-ratio 1.0` is inert; 1.10 keeps ~61% of pixels.
  float min_ratio = 0.f;
  // The reverse match, for left-right consistency. Every score already says how well
  // left pixel x fits right pixel x - d; the forward path reduces those over d for
  // each x, and this reduces the SAME scores over x for each x - d. So it is a second
  // running maximum in the loop that already holds the score, not a second matching
  // pass, and it needs one best rather than a top-2: the only question asked of a
  // right pixel is which left pixel it would have chosen.
  bool lrc = false;
  int threads = 0;
  bool simd = false;           // NEON score kernel, disparity in the lanes
  bool c2f = false;            // half-res coarse pass, prior as a mask (--band)
  bool noblock = false;        // keep the full volume even at --topk 2. Slow and
                               // memory-hungry, and the only way to A/B a change
                               // that needs neighbouring disparities against the
                               // shipping top-2 algorithm rather than the dense one.
};


// P(this pixel is within 1 disparity of correct), from the two candidate scores the
// solver has already ranked and, when it is available, the reverse match.
//
// ONE fitted feature, the peak ratio, and the arithmetic is two floats:
//
//   alt   = max(s2, lambda)             the runner-up, floored where there is none
//   x     = log((1 - alt) / (1 - s1))   the two costs as a ratio, as a ratio should be
//   logit = 0.692684 + 7.936751 * x
//   P     = 1 / (1 + exp(-logit))
//
// Worked at the eight-scene mean, s1 = 0.682 and alt = 0.537:
//   1 - s1 = 0.318, x = log(0.463 / 0.318) = 0.3757
//   logit = 0.692684 + 7.936751 * 0.3757 = 3.674,  P = 0.975
//
// Fitted by leave-one-scene-out logistic regression on the eight Middlebury scenes
// with `article/confidence.py --fit`: an area under the sparsification curve of
// 0.0301 alone, 0.0288 with the cap below, against 0.1041 for no confidence at all
// and 0.0062 for an oracle that removes the wrong pixels first.
//
// THE CAP, and why it is a cap and not a fitted weight. `lrc` is the reverse match:
// how far this pixel's answer is from what the right pixel it claimed would itself
// have chosen. Where the two disagree by more than a disparity, P is capped at 0.35,
// which is the measured correctness of that population -- 31.8% over the eight
// scenes, against 89.6% for the rest.
//
// A fitted offset calibrates marginally better on Middlebury (0.0088 against 0.0133)
// and is not usable, because Middlebury contains no example of the case this exists
// for. Flagged pixels there occur ONLY where the ratio is already weak: of the two
// strongest ratio quintiles, exactly zero are flagged. A linear model handed
// "strong ratio, reverse match disagrees" therefore extrapolates into an empty
// region and returns 0.998. That combination is precisely obstacle 24a's ghost --
// a match whose true partner is off the sensor has no competitor, so it scores a
// confident ratio, and the reverse match is the only cue that objects. The cap
// refuses to extrapolate where the fit has no evidence.
//
// Pass lrc = NaN, or anything under the tolerance, to leave P uncapped.
//
// WHY NOT MORE FEATURES. Adding the margin and the winning score reaches 0.0290, and
// both richer models are WRONG in the case that matters most here. A region with no
// texture has a constant Census descriptor, so every disparity matches perfectly:
// s1 = s2, the margin is 0 and the ratio is 1, and both cues correctly say "no idea".
// A model that also reads the winning score sees a perfect score and returns 0.94.
// It cannot say "a high score only counts when the margin is not zero", because it is
// linear, and it was never shown such a pixel -- Middlebury is textured almost
// everywhere. Caught on a synthetic frame whose black padding scored HIGHER than the
// real image. This model returns 0.667 there, below anything a real match reaches.
//
// The two-feature version is worse than it looks for a different reason: its weights
// come out +15.73 and -16.31, a cancellation between two collinear cues, with the
// margin entering NEGATIVELY. It fits and it is not a statement about the world.
//
// WHAT IS NOT ESTABLISHED. The ranking is measured, on Middlebury and on this camera:
// against a flat wall it closes 74% of the distance to an oracle, against 76% there.
// The absolute number is not. It is fitted on visible-light Middlebury while this
// camera is projected-dot infrared, it promises 0.86 on that wall and delivers 0.94,
// and it is already 0.08 out on the hardest of the eight scenes it WAS trained on.
// Treat it as an ordering that carries a probability's units until the wall check in
// TODO 0.55 validates it here.
inline float confidence(float s1, float s2, float lambda,
                        float lrc = 0.f, float lrc_tol = 1.f) {
  if (!(s1 > -1e29f)) return 0.f;
  const float alt = (s2 > -1e29f) ? std::max(s2, lambda) : lambda;
  const float x = std::log(std::max((1.f - alt) / std::max(1.f - s1, 1e-6f), 1e-6f));
  const float logit = 0.692684f + 7.936751f * x;
  const float p = 1.f / (1.f + std::exp(-std::max(-30.f, std::min(30.f, logit))));
  return (lrc > lrc_tol) ? std::min(p, 0.35f) : p;
}

inline uint8_t confidence_u8(float s1, float s2, float lambda,
                             float lrc = 0.f, float lrc_tol = 1.f) {
  return uint8_t(std::lround(confidence(s1, s2, lambda, lrc, lrc_tol) * 255.f));
}


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
// `cnb`, when given, is two floats per pixel: the aggregated cost one disparity
// either side of candidate 0, or -1e30 where that neighbour does not exist. It is
// what the blockwise cost stage can supply and the full volume cannot be asked for,
// and it exists only to feed the sub-pixel fit below. Null is fine; with a volume
// the fit reads the volume directly and covers every candidate rather than just
// the first.
void solve_row_sparse(int W, int D, int K, const Cfg& cfg, const float* vol,
                      float* cs, int* cd, int* cn,
                      float* beta, float* rho,
                      float* out_disp, float* out_margin, int hw,
                      int* bstart, int* bitems, int* cursor,
                      float* best, int* bestj, int* order, char* taken,
                      const float* cnb = nullptr, float* out_lrc = nullptr) {
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

  // --- the reverse match, from the buckets the counting sort has just built -------
  //
  // Left-right consistency asks what the right pixel this one claimed would itself
  // have chosen. The exact answer needs every disparity and therefore the cost
  // volume, which the blockwise path never stores; this asks the same question of
  // the candidates that survived, which are already sorted into per-right-pixel
  // buckets three lines above because the beta update needs exactly that.
  //
  // Measured against the exact version on the eight scenes, at matched removal it
  // catches 23.7% of the wrong pixels in the off-sensor border against the exact
  // 45.5% and the peak ratio's 11.9%. Half the benefit, no volume, no second pass,
  // and it wins over the ratio on 8 scenes of 8. What it misses is a right pixel
  // whose true best claimant did not keep it among ITS top two, which then has no
  // one to be beaten by.
  //
  // `cursor` is reused rather than another W of scratch: the counting sort finished
  // with it two loops ago and it is dead until the next row.
  if (out_lrc) {
    for (int xr = 0; xr < W; ++xr) {
      int bd = -1;
      float bv = -1e30f;
      for (int p = bstart[xr]; p < bstart[xr + 1]; ++p) {
        const int idx = bitems[p];
        if (cs[idx] > bv) { bv = cs[idx]; bd = cd[idx]; }
      }
      cursor[xr] = bd;
    }
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
      // With no message passing beta and rho are zero, and the max-sum belief
      // beta + rho - cs degenerates to MINUS the score -- which picks the WORST
      // candidate, not the best. That made --iters 0 read as a catastrophic 67.3%
      // bad-1.0 and look like evidence that message passing was worth 41 points.
      // It is not an ablation, it is an inverted objective. The honest degenerate
      // case is winner-take-all on the score, which is what --iters 0 now means.
      const float bel = (cfg.iters > 0) ? (beta[b + j] + rho[b + j] - cs[b + j])
                                        : cs[b + j];
      if (bel > best[x]) { best[x] = bel; bestj[x] = j; }
    }
  }
  std::sort(order, order + W, [&](int a, int b) { return best[a] > best[b]; });
  for (int oi = 0; oi < W; ++oi) {
    const int x = order[oi];
    out_disp[x] = std::nanf("");
    if (bestj[x] < 0) continue;
    if (cfg.min_margin > 0.f && out_margin[x] < cfg.min_margin) continue;
    if (cfg.min_ratio > 0.f) {
      // On the two candidates as RANKED BY SCORE, not on whichever one the belief
      // chose: the peak ratio is a statement about the shape of the cost curve,
      // and it is the same statement whether or not message passing ran.
      const size_t b = size_t(x) * KS;
      const float s2 = (cn[x] > 1) ? std::max(cs[b + 1], cfg.lambda) : cfg.lambda;
      // Similarity to cost, so both sides are non-negative and the ratio is >= 1.
      const float c1 = std::max(1.f - cs[b], 1e-6f);
      if ((1.f - s2) / c1 < cfg.min_ratio) continue;
    }
    const int d = dmin + cd[size_t(x) * KS + bestj[x]];
    const int xr = x - d;
    if (xr < 0 || taken[xr]) continue;
    taken[xr] = 1;

    // Sub-pixel by parabola through the aggregated cost at k-1, k, k+1.
    //
    // Integer output forfeits up to half a pixel before any matching error. That
    // is invisible at bad-1.0 on 450x375 -- which is why an earlier measurement
    // called this useless -- and decisive under Middlebury v3's own metric, which
    // is one pixel of FULL resolution and therefore a quarter pixel of what we
    // compute at Q. Measured there, on the dense path which already had this fit:
    // 42.22 -> 26.03 bad-1.0 at identical coverage, 16.2 points, taking the error
    // rate over filled pixels from 52.2% to 32.2% against SGM's 31.8%.
    // See doc/TODO.md 2.2.
    float off = 0.f;
    if (cfg.subpixel) {
      const int k = cd[size_t(x) * KS + bestj[x]];
      const float c0 = cs[size_t(x) * KS + bestj[x]];
      float cm = -1e30f, cp = -1e30f;
      if (vol) {
        // The same valid window the selection above used; outside it the volume
        // holds the -1e30 sentinel, and fitting through that is meaningless.
        int lo = 0, hi = 0;
        if (x >= hw && x < W - hw) {
          lo = std::max(0, x - (W - hw) + 1 - dmin + 1);
          hi = std::min(D, x - hw - dmin + 1);
          if (hi < lo) hi = lo;
        }
        if (k - 1 >= lo) cm = vol[size_t(x) * D + k - 1];
        if (k + 1 < hi)  cp = vol[size_t(x) * D + k + 1];
      } else if (cnb && bestj[x] == 0) {
        cm = cnb[size_t(x) * 2];
        cp = cnb[size_t(x) * 2 + 1];
      }
      if (cm > -1e29f && cp > -1e29f) {
        // Both estimators want the same thing -- where the peak really sits between
        // three samples -- and differ in what they assume the shape between them is.
        // Only where the triple genuinely brackets a maximum; a flat or inverted one
        // has no vertex and the clamp would invent one.
        if (cfg.fit_eq) {
          const float lower = cm < cp ? cm : cp;
          const float den = c0 - lower;              // height above the lower flank
          if (den > 1e-6f) {
            off = 0.5f * (cp - cm) / den;
            if (off > 0.5f) off = 0.5f;
            if (off < -0.5f) off = -0.5f;
          }
        } else {
          const float den = 2.f * c0 - cm - cp;
          if (den > 1e-6f) {
            off = 0.5f * (cp - cm) / den;
            if (off > 0.5f) off = 0.5f;
            if (off < -0.5f) off = -0.5f;
          }
        }
      }
    }
    out_disp[x] = float(d) + off;
  }

  // Now that every pixel has its answer: follow it to the right pixel it claimed and
  // compare with what that pixel would have chosen. Rounded, because the reverse
  // match is an integer candidate and the forward one carries a sub-pixel offset.
  // NaN where this pixel has no answer, or where the right pixel it points at had no
  // claimant at all -- neither is a disagreement, and calling them one would make the
  // border of every frame look inconsistent.
  if (out_lrc) {
    for (int x = 0; x < W; ++x) {
      out_lrc[x] = std::nanf("");
      const float dl = out_disp[x];
      if (!(dl == dl)) continue;
      const int xr = x - int(std::lround(dl));
      if (xr >= 0 && xr < W && cursor[xr] >= 0)
        out_lrc[x] = std::fabs(dl - float(dmin + cursor[xr]));
    }
  }
}


}  // namespace doubleeye

#endif  // DOUBLEEYE_DENSE_SOLVE_HPP
