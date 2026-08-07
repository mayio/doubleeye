#include "doubleeye/match.hpp"

#include <time.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <utility>

namespace doubleeye {
namespace {

// One candidate association. The graph is sparse: an epipolar band plus a
// disparity range plus a per-row cap on candidates, which is where the plan's
// coarse-to-fine reduction of k from ~100-200 down to ~8-16 will eventually act.
struct Edge {
  int i;        // left index
  int j;        // right index
  float s;      // score
  float beta;
  float rho;
};

// Largest and second-largest, so "max excluding one element" is O(1) per edge.
struct Top2 {
  float best = -1e30f;
  float second = -1e30f;
  int best_idx = -1;

  void push(float v, int idx) {
    if (v > best) {
      second = best;
      best = v;
      best_idx = idx;
    } else if (v > second) {
      second = v;
    }
  }
  // Max over the set with element `idx` removed.
  float excluding(int idx) const {
    return (idx == best_idx) ? second : best;
  }
};

std::vector<Edge> build_candidates(const std::vector<Keypoint>& left,
                                  const std::vector<uint64_t>& dl,
                                  const std::vector<Keypoint>& right,
                                  const std::vector<uint64_t>& dr,
                                  int census_bits, const MatchConfig& cfg) {
  // Sort right keypoints by y so the epipolar band is a contiguous slice rather
  // than a scan of every right keypoint for every left one.
  std::vector<int> order(right.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return right[a].y < right[b].y;
  });

  std::vector<Edge> edges;
  edges.reserve(left.size() * size_t(std::max(1, cfg.max_candidates)));
  std::vector<std::pair<float, int>> scratch;

  for (size_t i = 0; i < left.size(); ++i) {
    const Keypoint& l = left[i];
    const auto lo = std::lower_bound(
        order.begin(), order.end(), l.y - cfg.max_dy,
        [&](int idx, float v) { return right[idx].y < v; });
    const auto hi = std::upper_bound(
        order.begin(), order.end(), l.y + cfg.max_dy,
        [&](float v, int idx) { return v < right[idx].y; });

    scratch.clear();
    for (auto it = lo; it != hi; ++it) {
      const int j = *it;
      const float d = l.x - right[j].x;   // positive disparity: right sees it left
      if (d < cfg.min_disparity || d > cfg.max_disparity) continue;
      scratch.emplace_back(
          pair_score(l, right[j], dl[i], dr[j], census_bits, cfg, int(i)), j);
    }
    if (scratch.empty()) continue;

    const size_t keep = std::min(scratch.size(),
                                 size_t(std::max(1, cfg.max_candidates)));
    if (scratch.size() > keep) {
      std::partial_sort(scratch.begin(), scratch.begin() + keep, scratch.end(),
                        [](const std::pair<float, int>& a,
                           const std::pair<float, int>& b) {
                          return a.first > b.first;
                        });
    }
    for (size_t k = 0; k < keep; ++k) {
      Edge e;
      e.i = int(i);
      e.j = scratch[k].second;
      e.s = scratch[k].first;
      e.beta = 0.f;
      e.rho = 0.f;
      edges.push_back(e);
    }
  }
  return edges;
}

// Best-minus-second-best s(i,j) per left keypoint, in one pass over the edge
// list. Where a keypoint has a single candidate the only alternative is being
// called clutter, so the margin is measured against lambda instead.
//
// This is the confidence worth exporting. Over eight Middlebury scenes with
// ground truth, precision by margin quartile is 0.169 / 0.286 / 0.391 / 0.659.
// It is nearly free here, and free outright inside the solver, whose message
// update is already a top-2 reduction over exactly these rows.
static std::vector<float> row_margins(const std::vector<Edge>& edges, int n_left,
                                      const MatchConfig& cfg) {
  const float kNeg = -1e30f;
  std::vector<float> b1(size_t(n_left), kNeg), b2(size_t(n_left), kNeg);
  for (const Edge& e : edges) {
    if (e.s > b1[size_t(e.i)]) {
      b2[size_t(e.i)] = b1[size_t(e.i)];
      b1[size_t(e.i)] = e.s;
    } else if (e.s > b2[size_t(e.i)]) {
      b2[size_t(e.i)] = e.s;
    }
  }
  std::vector<float> out(size_t(n_left), 0.f);
  for (int i = 0; i < n_left; ++i) {
    if (b1[size_t(i)] == kNeg) continue;   // no candidates at all
    const float alt = (b2[size_t(i)] == kNeg) ? cfg.lambda
                                             : std::max(b2[size_t(i)], cfg.lambda);
    out[size_t(i)] = b1[size_t(i)] - alt;
  }
  return out;
}

std::vector<Match> emit(const std::vector<Edge>& edges,
                        const std::vector<Keypoint>& left,
                        const std::vector<Keypoint>& right,
                        const std::vector<float>& belief,
                        int n_left, int n_right, const MatchConfig& cfg) {
  // Accept an edge only if it is simultaneously the best option for its row and
  // for its column, and beats the unmatched alternative. Max-sum ought to make
  // these agree; requiring it explicitly guarantees a valid one-to-one matching
  // even when the messages have not fully converged.
  //
  // Two different questions, which an earlier version of this conflated:
  //
  //   WHICH candidate?   The belief. It measures an edge's advantage over the
  //                      others competing for the same row and column, so it is
  //                      the right ordering -- but its SIGN is not a decision.
  //                      When candidates are near-tied, no edge has an advantage
  //                      and every belief is <= 0, even though matching is
  //                      clearly better than not matching. Gating on belief > 0
  //                      returned zero matches on exactly-tied problems whose
  //                      objective wanted everything matched.
  //   MATCH AT ALL?      s(i,j) against lambda. That is what lambda means: the
  //                      score of leaving a left keypoint unmatched.
  //
  // So: order by belief, decide by lambda, and require mutual agreement between
  // row and column so the result is a valid one-to-one matching even when the
  // messages have not fully converged. Near-tie behaviour matters here rather
  // than being a corner case: projected IR dots make descriptors ~3.3x
  // degenerate, which is the plan's cited condition for BP's guarantee lapsing
  // (a non-unique LP optimum, Bayati/Shah/Sharma).
  const float kNeg = -1e30f;
  std::vector<int> best_row(n_left, -1), best_col(n_right, -1);
  std::vector<float> bl_row(n_left, kNeg), bl_col(n_right, kNeg);
  for (size_t e = 0; e < edges.size(); ++e) {
    if (edges[e].s <= cfg.lambda) continue;
    if (best_row[edges[e].i] < 0 || belief[e] > bl_row[edges[e].i]) {
      best_row[edges[e].i] = int(e);
      bl_row[edges[e].i] = belief[e];
    }
    if (best_col[edges[e].j] < 0 || belief[e] > bl_col[edges[e].j]) {
      best_col[edges[e].j] = int(e);
      bl_col[edges[e].j] = belief[e];
    }
  }

  std::vector<Match> out;
  std::vector<char> row_taken(n_left, 0), col_taken(n_right, 0);
  const std::vector<float> margin = row_margins(edges, n_left, cfg);

  auto accept = [&](int e) {
    Match m;
    m.left = edges[e].i;
    m.right = edges[e].j;
    m.score = edges[e].s;
    m.belief = belief[e];
    m.margin = margin[size_t(m.left)];
    m.disparity = left[m.left].x - right[m.right].x;
    m.dy = left[m.left].y - right[m.right].y;
    out.push_back(m);
    row_taken[m.left] = 1;
    col_taken[m.right] = 1;
  };

  // Pass 1: edges where row and column agree. These are the confident ones.
  for (int i = 0; i < n_left; ++i) {
    const int e = best_row[i];
    if (e < 0) continue;
    if (best_col[edges[e].j] != e) continue;
    accept(e);
  }

  // Pass 2: greedy completion in belief order over what remains.
  //
  // Necessary, not cosmetic. Under near-ties every row's best belief points at
  // the same column, so pass 1 resolves exactly ONE pair and leaves the rest
  // unmatched -- on a problem whose optimum matches everything. BP has supplied
  // the ordering it can; the mutual-agreement test simply cannot commit when
  // nothing has an advantage.
  //
  // Safe by construction: each accepted edge has s > lambda and both endpoints
  // free, so it strictly raises the objective. Taking them in belief order means
  // the strongest available association is committed first.
  std::vector<int> rest;
  rest.reserve(edges.size());
  for (size_t e = 0; e < edges.size(); ++e) {
    if (edges[e].s <= cfg.lambda) continue;
    if (row_taken[edges[e].i] || col_taken[edges[e].j]) continue;
    rest.push_back(int(e));
  }
  std::sort(rest.begin(), rest.end(), [&](int a, int b) {
    if (belief[a] != belief[b]) return belief[a] > belief[b];
    return edges[a].s > edges[b].s;   // deterministic tie-break
  });
  for (int e : rest) {
    if (row_taken[edges[e].i] || col_taken[edges[e].j]) continue;
    accept(e);
  }
  return out;
}

}  // namespace

bool DisparityPrior::lookup(float x, float y, float* d, float* sd) const {
  if (cols <= 0 || rows <= 0) return false;
  const int cx = std::min(cols - 1, std::max(0, int(x) / cell));
  const int cy = std::min(rows - 1, std::max(0, int(y) / cell));
  const size_t i = size_t(cy) * cols + cx;
  if (!valid[i]) return false;
  if (d) *d = disparity[i];
  if (sd) *sd = sigma[i];
  return true;
}

float DisparityPrior::coverage() const {
  if (valid.empty()) return 0.f;
  size_t n = 0;
  for (uint8_t v : valid) n += v ? 1 : 0;
  return float(n) / float(valid.size());
}

DisparityPrior build_disparity_prior(const std::vector<Match>& coarse,
                                     const std::vector<Keypoint>& coarse_left,
                                     int full_width, int full_height, int scale,
                                     int cell, int min_count, float sigma_floor) {
  DisparityPrior pr;
  pr.cell = std::max(8, cell);
  pr.cols = (full_width + pr.cell - 1) / pr.cell;
  pr.rows = (full_height + pr.cell - 1) / pr.cell;
  const size_t n = size_t(pr.cols) * pr.rows;
  pr.disparity.assign(n, 0.f);
  pr.sigma.assign(n, 0.f);
  pr.valid.assign(n, 0);
  if (coarse.empty() || scale < 1) return pr;

  std::vector<std::vector<float>> bucket(n);
  for (const Match& m : coarse) {
    if (m.left < 0 || size_t(m.left) >= coarse_left.size()) continue;
    // Coarse coordinates and disparity both scale up by the same factor.
    const float x = coarse_left[m.left].x * float(scale);
    const float y = coarse_left[m.left].y * float(scale);
    const int cx = std::min(pr.cols - 1, std::max(0, int(x) / pr.cell));
    const int cy = std::min(pr.rows - 1, std::max(0, int(y) / pr.cell));
    bucket[size_t(cy) * pr.cols + cx].push_back(m.disparity * float(scale));
  }

  for (size_t i = 0; i < n; ++i) {
    std::vector<float>& v = bucket[i];
    if (int(v.size()) < min_count) continue;   // too little evidence: no prior
    std::sort(v.begin(), v.end());
    const float med = v[v.size() / 2];
    // Median absolute deviation, scaled to a Gaussian-equivalent sigma. This is
    // what makes the prior self-widening: a cell spanning a depth discontinuity
    // has a large MAD and therefore a weak prior, which is precisely where the
    // plan wants the full disparity range kept.
    std::vector<float> dev(v.size());
    for (size_t k = 0; k < v.size(); ++k) dev[k] = std::fabs(v[k] - med);
    std::sort(dev.begin(), dev.end());
    const float mad = dev[dev.size() / 2];
    pr.disparity[i] = med;
    pr.sigma[i] = std::max(sigma_floor, 1.4826f * mad);
    pr.valid[i] = 1;
  }
  return pr;
}

float pair_score(const Keypoint& l, const Keypoint& r, uint64_t dl, uint64_t dr,
                 int census_bits, const MatchConfig& cfg, int left_index) {
  // Descriptor term: bits agree by chance half the time, so hamming == bits/2
  // scores 0 and a perfect match scores +1. That puts lambda == 0 at the
  // interpretable place of "no better than chance".
  const float half = 0.5f * float(census_bits);
  const float desc = (half - float(hamming(dl, dr))) / half;

  // y-residual. The plan asks for this as a unary term regardless, since median
  // |dy| over matched pairs doubles as a free online monitor for extrinsic drift.
  const float dy = (l.y - r.y) / (cfg.sigma_y > 0 ? cfg.sigma_y : 1.f);
  float score = desc - cfg.w_y * dy * dy;

  // Soft coarse-disparity prior. A penalty, never a veto: the plan is explicit
  // that hard pruning at the coarse level is how thin structures get lost.
  if (cfg.prior) {
    float d_pred = 0.f, sd = 1.f;
    if (cfg.prior->lookup(l.x, l.y, &d_pred, &sd) && sd > 0.f) {
      const float e = ((l.x - r.x) - d_pred) / sd;
      score -= cfg.w_prior * e * e;
    }
  }

  // Smoothness prior from a previous pass, also a penalty and never a veto.
  if (cfg.smooth && left_index >= 0 &&
      size_t(left_index) < cfg.smooth->valid.size() &&
      cfg.smooth->valid[left_index]) {
    const float sd = cfg.smooth->sigma[left_index];
    if (sd > 0.f) {
      const float e = ((l.x - r.x) - cfg.smooth->disparity[left_index]) / sd;
      score -= cfg.w_smooth * e * e;
    }
  }
  return score;
}

std::vector<Match> match_masda(const std::vector<Keypoint>& left,
                               const std::vector<uint64_t>& desc_left,
                               const std::vector<Keypoint>& right,
                               const std::vector<uint64_t>& desc_right,
                               const MatchConfig& cfg, MatchStats* stats) {
  const int census_bits = 48;
  std::vector<Edge> edges = build_candidates(left, desc_left, right, desc_right,
                                            census_bits, cfg);
  if (stats) {
    stats->left = int(left.size());
    stats->right = int(right.size());
    stats->candidates = int(edges.size());
    stats->iterations_run = 0;
    stats->converged = false;
    stats->final_max_delta = 0.f;
    stats->oscillating = 0;
  }
  if (edges.empty()) return {};

  // Edge indices grouped by row and by column, so each reduction touches each
  // edge once.
  std::vector<std::vector<int>> by_row(left.size()), by_col(right.size());
  for (size_t e = 0; e < edges.size(); ++e) {
    by_row[edges[e].i].push_back(int(e));
    by_col[edges[e].j].push_back(int(e));
  }

  const float d = std::min(std::max(cfg.damping, 0.f), 0.99f);
  float prev_delta = 1e30f;
  int iter = 0;
  for (; iter < cfg.iterations; ++iter) {
    float max_delta = 0.f;

    // rho_ij = s_ij - max( lambda, max_{k != j} beta_ik )   -- reduce over rows
    for (size_t i = 0; i < by_row.size(); ++i) {
      Top2 t;
      for (int e : by_row[i]) t.push(edges[e].beta, e);
      for (int e : by_row[i]) {
        const float competitor = std::max(cfg.lambda, t.excluding(e));
        const float target = edges[e].s - competitor;
        const float updated = (1.f - d) * target + d * edges[e].rho;
        max_delta = std::max(max_delta, std::fabs(updated - edges[e].rho));
        edges[e].rho = updated;
      }
    }

    // beta_ij = s_ij - max( gamma, max_{k != i} rho_kj )  -- reduce over cols
    for (size_t j = 0; j < by_col.size(); ++j) {
      Top2 t;
      for (int e : by_col[j]) t.push(edges[e].rho, e);
      for (int e : by_col[j]) {
        const float competitor = std::max(cfg.gamma, t.excluding(e));
        const float target = edges[e].s - competitor;
        const float updated = (1.f - d) * target + d * edges[e].beta;
        max_delta = std::max(max_delta, std::fabs(updated - edges[e].beta));
        edges[e].beta = updated;
      }
    }

    // The plan warns that adding loopy pairwise factors destroys the
    // bipartite-matching convergence guarantee and to watch for oscillation. Count
    // iterations where the change grew, so the warning has a number attached even
    // before those factors exist.
    if (stats && max_delta > prev_delta) ++stats->oscillating;
    prev_delta = max_delta;
    if (stats) {
      stats->iterations_run = iter + 1;
      stats->final_max_delta = max_delta;
    }
    if (max_delta < cfg.converge_eps) {
      if (stats) stats->converged = true;
      break;
    }
  }

  // Max-sum belief. beta and rho each carry s plus the opposite side's incoming
  // message, so their sum counts s twice.
  std::vector<float> belief(edges.size());
  for (size_t e = 0; e < edges.size(); ++e)
    belief[e] = edges[e].beta + edges[e].rho - edges[e].s;

  return emit(edges, left, right, belief, int(left.size()), int(right.size()),
              cfg);
}

std::vector<Match> match_mutual_nn(const std::vector<Keypoint>& left,
                                   const std::vector<uint64_t>& desc_left,
                                   const std::vector<Keypoint>& right,
                                   const std::vector<uint64_t>& desc_right,
                                   const MatchConfig& cfg, float ratio,
                                   MatchStats* stats) {
  const int census_bits = 48;
  std::vector<Edge> edges = build_candidates(left, desc_left, right, desc_right,
                                            census_bits, cfg);
  if (stats) {
    stats->left = int(left.size());
    stats->right = int(right.size());
    stats->candidates = int(edges.size());
    stats->iterations_run = 0;
    stats->converged = true;
  }
  if (edges.empty()) return {};

  const std::vector<float> margin = row_margins(edges, int(left.size()), cfg);

  // Best and runner-up per row, for the ratio test; best per column, for mutuality.
  std::vector<int> best_row(left.size(), -1), second_row(left.size(), -1);
  std::vector<int> best_col(right.size(), -1);
  for (size_t e = 0; e < edges.size(); ++e) {
    const int i = edges[e].i, j = edges[e].j;
    if (best_row[i] < 0 || edges[e].s > edges[best_row[i]].s) {
      second_row[i] = best_row[i];
      best_row[i] = int(e);
    } else if (second_row[i] < 0 || edges[e].s > edges[second_row[i]].s) {
      second_row[i] = int(e);
    }
    if (best_col[j] < 0 || edges[e].s > edges[best_col[j]].s) best_col[j] = int(e);
  }

  std::vector<Match> out;
  for (size_t i = 0; i < left.size(); ++i) {
    const int e = best_row[i];
    if (e < 0) continue;
    if (edges[e].s <= cfg.lambda) continue;          // worse than not matching
    if (best_col[edges[e].j] != e) continue;        // not mutual
    if (second_row[i] >= 0) {
      // Ratio test on the score, shifted so lambda is the origin. On degenerate
      // projected-dot texture the runner-up is often a near-tie, which is exactly
      // the regime this test throws away and MASDA does not have to.
      const float b = edges[e].s - cfg.lambda;
      const float s2 = edges[second_row[i]].s - cfg.lambda;
      if (b > 0.f && s2 > 0.f && s2 / b > ratio) continue;
    }
    Match m;
    m.left = edges[e].i;
    m.right = edges[e].j;
    m.score = edges[e].s;
    m.belief = edges[e].s;
    m.margin = margin[size_t(m.left)];
    m.disparity = left[m.left].x - right[m.right].x;
    m.dy = left[m.left].y - right[m.right].y;
    out.push_back(m);
  }
  return out;
}

std::vector<Match> match_brute_force(const std::vector<Keypoint>& left,
                                     const std::vector<uint64_t>& desc_left,
                                     const std::vector<Keypoint>& right,
                                     const std::vector<uint64_t>& desc_right,
                                     const MatchConfig& cfg) {
  const int census_bits = 48;
  const std::vector<Edge> edges = build_candidates(left, desc_left, right,
                                                  desc_right, census_bits, cfg);
  const int n = int(left.size()), m = int(right.size());
  // Dense score table with "absent" marked, so the search can skip non-edges.
  std::vector<float> s(size_t(n) * m, 0.f);
  std::vector<char> present(size_t(n) * m, 0);
  for (const Edge& e : edges) {
    s[size_t(e.i) * m + e.j] = e.s;
    present[size_t(e.i) * m + e.j] = 1;
  }

  const std::vector<float> margin = row_margins(edges, n, cfg);

  std::vector<int> assign(n, -1), best_assign(n, -1);
  std::vector<char> used(m, 0);
  float best_value = 0.f;
  for (int i = 0; i < n; ++i) best_value += cfg.lambda;  // all unmatched

  // Depth-first over rows: each may take lambda, or any free column it has an
  // edge to. Factorial, hence tests only.
  std::function<void(int, float)> rec = [&](int i, float acc) {
    if (i == n) {
      // Columns left over contribute gamma.
      float total = acc;
      for (int j = 0; j < m; ++j)
        if (!used[j]) total += cfg.gamma;
      if (total > best_value) {
        best_value = total;
        best_assign = assign;
      }
      return;
    }
    assign[i] = -1;
    rec(i + 1, acc + cfg.lambda);              // leave i unmatched
    for (int j = 0; j < m; ++j) {
      if (used[j] || !present[size_t(i) * m + j]) continue;
      used[j] = 1;
      assign[i] = j;
      rec(i + 1, acc + s[size_t(i) * m + j]);
      assign[i] = -1;
      used[j] = 0;
    }
  };
  rec(0, 0.f);

  std::vector<Match> out;
  for (int i = 0; i < n; ++i) {
    if (best_assign[i] < 0) continue;
    Match mt;
    mt.left = i;
    mt.right = best_assign[i];
    mt.score = s[size_t(i) * m + best_assign[i]];
    mt.belief = mt.score;
    mt.margin = margin[size_t(i)];
    mt.disparity = left[i].x - right[best_assign[i]].x;
    mt.dy = left[i].y - right[best_assign[i]].y;
    out.push_back(mt);
  }
  return out;
}

CoarseToFineResult match_coarse_to_fine(const Image8& left_img,
                                        const Image8& right_img,
                                        const DetectorConfig& det,
                                        const CensusConfig& census,
                                        const MatchConfig& cfg,
                                        int scale, int prior_cell) {
  CoarseToFineResult out;
  const auto clock_ms = []() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
  };

  // --- coarse pass ---
  const double t0 = clock_ms();
  const Image8 cl = downsample(left_img, scale);
  const Image8 cr = downsample(right_img, scale);
  if (!cl.empty() && !cr.empty()) {
    // The coarse level is small, so the detector's border and grid must shrink
    // with it or it finds almost nothing.
    DetectorConfig cdet = det;
    cdet.cell = std::max(4, det.cell / scale);
    cdet.border = std::max(6, det.border / 2);

    const std::vector<Keypoint> kl = detect_keypoints_fast(cl, cdet);
    const std::vector<Keypoint> kr = detect_keypoints_fast(cr, cdet);
    std::vector<uint64_t> dl(kl.size()), dr(kr.size());
    for (size_t k = 0; k < kl.size(); ++k)
      dl[k] = census_at(cl, int(kl[k].x + 0.5f), int(kl[k].y + 0.5f), census);
    for (size_t k = 0; k < kr.size(); ++k)
      dr[k] = census_at(cr, int(kr[k].x + 0.5f), int(kr[k].y + 0.5f), census);

    // Disparity range scales down with resolution; the epipolar band does not
    // shrink below a pixel, since rectification residual is absolute.
    MatchConfig ccfg = cfg;
    ccfg.prior = nullptr;
    ccfg.min_disparity = std::max(0.5f, cfg.min_disparity / float(scale));
    ccfg.max_disparity = cfg.max_disparity / float(scale);
    ccfg.max_dy = std::max(1.0f, cfg.max_dy / float(scale));
    ccfg.sigma_y = std::max(0.5f, cfg.sigma_y / float(scale));

    out.coarse_matches = match_masda(kl, dl, kr, dr, ccfg, &out.coarse_stats);
    out.prior = build_disparity_prior(out.coarse_matches, kl, left_img.width,
                                     left_img.height, scale, prior_cell);
  }
  out.coarse_ms = clock_ms() - t0;

  // --- fine pass, with the prior folded into s(i,j) ---
  const double t1 = clock_ms();
  const std::vector<Keypoint> kl = detect_keypoints_fast(left_img, det);
  const std::vector<Keypoint> kr = detect_keypoints_fast(right_img, det);
  std::vector<uint64_t> dl(kl.size()), dr(kr.size());
  for (size_t k = 0; k < kl.size(); ++k)
    dl[k] = census_at(left_img, int(kl[k].x + 0.5f), int(kl[k].y + 0.5f), census);
  for (size_t k = 0; k < kr.size(); ++k)
    dr[k] = census_at(right_img, int(kr[k].x + 0.5f), int(kr[k].y + 0.5f), census);

  MatchConfig fine = cfg;
  fine.prior = out.prior.cols > 0 ? &out.prior : nullptr;
  out.matches = match_masda(kl, dl, kr, dr, fine, &out.fine_stats);
  out.fine_ms = clock_ms() - t1;
  return out;
}

SmoothPrior fit_smooth_prior(const std::vector<Match>& matches,
                             const std::vector<Keypoint>& left,
                             int k_neighbours, float sigma_floor,
                             int irls_iterations) {
  SmoothPrior pr;
  pr.disparity.assign(left.size(), 0.f);
  pr.sigma.assign(left.size(), 0.f);
  pr.valid.assign(left.size(), 0);
  const int k = std::max(4, k_neighbours);
  if (int(matches.size()) < k + 1) return pr;

  // Matched keypoints only: they are the ones with a disparity to fit.
  struct Pt { float x, y, d; };
  std::vector<Pt> pts;
  pts.reserve(matches.size());
  for (const Match& m : matches) {
    if (m.left < 0 || size_t(m.left) >= left.size()) continue;
    pts.push_back({left[m.left].x, left[m.left].y, m.disparity});
  }
  if (int(pts.size()) < k + 1) return pr;

  std::vector<std::pair<float, int>> near;
  for (size_t i = 0; i < left.size(); ++i) {
    const float px = left[i].x, py = left[i].y;
    near.clear();
    near.reserve(pts.size());
    for (size_t q = 0; q < pts.size(); ++q) {
      const float dx = pts[q].x - px, dy = pts[q].y - py;
      near.emplace_back(dx * dx + dy * dy, int(q));
    }
    const size_t take = std::min(near.size(), size_t(k));
    std::partial_sort(near.begin(), near.begin() + take, near.end());

    // IRLS with Tukey weights. A neighbourhood spanning a depth discontinuity
    // would otherwise pull the plane between the two surfaces; redescending
    // weights let the far surface drop out instead.
    std::vector<float> w(take, 1.f);
    float a0 = 0.f, b0 = 0.f, c0 = 0.f, scale = 1.f;
    bool ok = false;
    for (int it = 0; it < std::max(1, irls_iterations); ++it) {
      // Weighted normal equations for d = a*x + b*y + c, centred on the query
      // point so the system stays well conditioned.
      double sxx = 0, sxy = 0, sx = 0, syy = 0, sy = 0, sw = 0;
      double sdx = 0, sdy = 0, sd = 0;
      for (size_t n = 0; n < take; ++n) {
        const Pt& q = pts[near[n].second];
        const double x = q.x - px, y = q.y - py, dd = q.d, ww = w[n];
        sxx += ww * x * x; sxy += ww * x * y; sx += ww * x;
        syy += ww * y * y; sy += ww * y;      sw += ww;
        sdx += ww * dd * x; sdy += ww * dd * y; sd += ww * dd;
      }
      // 3x3 symmetric solve by cofactors. No Eigen needed for this size.
      const double m11 = sxx, m12 = sxy, m13 = sx;
      const double m22 = syy, m23 = sy,  m33 = sw;
      const double c11 = m22 * m33 - m23 * m23;
      const double c12 = m13 * m23 - m12 * m33;
      const double c13 = m12 * m23 - m13 * m22;
      const double det = m11 * c11 + m12 * c12 + m13 * c13;
      if (std::fabs(det) < 1e-9) break;
      const double c22 = m11 * m33 - m13 * m13;
      const double c23 = m12 * m13 - m11 * m23;
      const double c33 = m11 * m22 - m12 * m12;
      a0 = float((c11 * sdx + c12 * sdy + c13 * sd) / det);
      b0 = float((c12 * sdx + c22 * sdy + c23 * sd) / det);
      c0 = float((c13 * sdx + c23 * sdy + c33 * sd) / det);
      ok = true;

      std::vector<float> res(take);
      for (size_t n = 0; n < take; ++n) {
        const Pt& q = pts[near[n].second];
        res[n] = q.d - (a0 * (q.x - px) + b0 * (q.y - py) + c0);
      }
      std::vector<float> absr(take);
      for (size_t n = 0; n < take; ++n) absr[n] = std::fabs(res[n]);
      std::sort(absr.begin(), absr.end());
      scale = std::max(sigma_floor, 1.4826f * absr[absr.size() / 2]);
      const float cut = 4.685f * scale;   // Tukey, 95% efficiency
      for (size_t n = 0; n < take; ++n) {
        const float u = res[n] / cut;
        w[n] = (std::fabs(u) < 1.f) ? (1.f - u * u) * (1.f - u * u) : 0.f;
      }
    }
    if (!ok) continue;
    // c0 is the fitted disparity AT the query point, since the fit was centred.
    pr.disparity[i] = c0;
    pr.sigma[i] = scale;
    pr.valid[i] = 1;
  }
  return pr;
}

SmoothResult match_iterated_smoothness(const std::vector<Keypoint>& left,
                                      const std::vector<uint64_t>& desc_left,
                                      const std::vector<Keypoint>& right,
                                      const std::vector<uint64_t>& desc_right,
                                      const MatchConfig& cfg, int passes,
                                      int k_neighbours) {
  SmoothResult out;
  MatchConfig base = cfg;      // for scoring the objective without smoothness
  base.smooth = nullptr;
  base.w_smooth = 0.f;

  MatchConfig cur = cfg;
  cur.smooth = nullptr;
  SmoothPrior prior;
  std::vector<Match> prev;

  for (int pass = 0; pass < std::max(1, passes); ++pass) {
    const std::vector<Match> m =
        match_masda(left, desc_left, right, desc_right, cur);

    // Re-score every accepted match with the BASE score so passes are comparable.
    float base_obj = 0.f;
    std::vector<double> ady;
    for (const Match& x : m) {
      base_obj += pair_score(left[x.left], right[x.right], desc_left[x.left],
                             desc_right[x.right], 48, base, -1);
      ady.push_back(std::fabs(double(x.dy)));
    }
    base_obj += base.lambda * float(int(left.size()) - int(m.size()));
    base_obj += base.gamma * float(int(right.size()) - int(m.size()));

    std::sort(ady.begin(), ady.end());
    out.base_objective.push_back(base_obj);
    out.counts.push_back(int(m.size()));
    out.median_abs_dy.push_back(ady.empty() ? 0.f
                                            : float(ady[ady.size() / 2]));
    size_t cov = 0;
    for (uint8_t v : prior.valid) cov += v ? 1 : 0;
    out.prior_coverage.push_back(
        prior.valid.empty() ? 0.f : float(cov) / float(prior.valid.size()));

    // How much did the assignment actually move? If nothing changes, further
    // passes are wasted.
    int changed = 0;
    if (!prev.empty()) {
      std::vector<int> before(left.size(), -1);
      for (const Match& x : prev) before[x.left] = x.right;
      for (const Match& x : m)
        if (before[x.left] != x.right) ++changed;
    }
    out.changed.push_back(changed);

    if (pass == 0) out.first_pass = m;
    out.matches = m;
    prev = m;

    if (pass + 1 < passes) {
      prior = fit_smooth_prior(m, left, k_neighbours);
      cur = cfg;
      cur.smooth = &prior;
    }
  }
  return out;
}


void refine_disparity(const Image8& left, const Image8& right,
                      const std::vector<Keypoint>& kl,
                      const std::vector<Keypoint>& kr,
                      std::vector<Match>* matches, int half) {
  if (!matches || left.empty() || right.empty()) return;
  const int W = left.width, H = left.height;

  // SAD between the left window at (xl, y) and the right window at (xl - d, y).
  // The pair is rectified, so both windows sit on the same row by construction.
  const auto cost = [&](int xl, int y, int d) -> long {
    const int xr = xl - d;
    if (xl - half < 0 || xl + half >= W) return -1;
    if (xr - half < 0 || xr + half >= W) return -1;
    if (y - half < 0 || y + half >= H) return -1;
    long acc = 0;
    for (int v = -half; v <= half; ++v) {
      for (int u = -half; u <= half; ++u) {
        acc += std::abs(int(left.at(xl + u, y + v)) -
                        int(right.at(xr + u, y + v)));
      }
    }
    return acc;
  };

  for (Match& m : *matches) {
    if (m.left < 0 || m.right < 0) continue;
    if (size_t(m.left) >= kl.size() || size_t(m.right) >= kr.size()) continue;
    const int xl = int(kl[size_t(m.left)].x + 0.5f);
    const int y = int(kl[size_t(m.left)].y + 0.5f);
    const int d0 = int(std::lround(m.disparity));
    const long c0 = cost(xl, y, d0);
    const long cm = cost(xl, y, d0 - 1);
    const long cp = cost(xl, y, d0 + 1);
    if (c0 < 0 || cm < 0 || cp < 0) continue;      // too near a border

    const double denom = double(cm) - 2.0 * double(c0) + double(cp);
    if (denom <= 0.0) continue;   // not a minimum here; leave it alone
    double off = 0.5 * (double(cm) - double(cp)) / denom;
    if (off > 0.5) off = 0.5;
    if (off < -0.5) off = -0.5;
    m.disparity = float(double(d0) + off);
  }
}

float matching_objective(const std::vector<Match>& matches,
                         const MatchConfig& cfg, int n_left, int n_right) {
  float total = 0.f;
  for (const Match& m : matches) total += m.score;
  total += cfg.lambda * float(n_left - int(matches.size()));
  total += cfg.gamma * float(n_right - int(matches.size()));
  return total;
}

}  // namespace doubleeye
