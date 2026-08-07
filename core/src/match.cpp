#include "doubleeye/match.hpp"

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
      scratch.emplace_back(pair_score(l, right[j], dl[i], dr[j], census_bits, cfg),
                           j);
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
  //   MATCH AT ALL?      s(i,j) against gamma. That is what gamma means: the
  //                      score of leaving a left keypoint unmatched.
  //
  // So: order by belief, decide by gamma, and require mutual agreement between
  // row and column so the result is a valid one-to-one matching even when the
  // messages have not fully converged. Near-tie behaviour matters here rather
  // than being a corner case: projected IR dots make descriptors ~3.3x
  // degenerate, which is the plan's cited condition for BP's guarantee lapsing
  // (a non-unique LP optimum, Bayati/Shah/Sharma).
  const float kNeg = -1e30f;
  std::vector<int> best_row(n_left, -1), best_col(n_right, -1);
  std::vector<float> bl_row(n_left, kNeg), bl_col(n_right, kNeg);
  for (size_t e = 0; e < edges.size(); ++e) {
    if (edges[e].s <= cfg.gamma) continue;
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

  auto accept = [&](int e) {
    Match m;
    m.left = edges[e].i;
    m.right = edges[e].j;
    m.score = edges[e].s;
    m.belief = belief[e];
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
  // Safe by construction: each accepted edge has s > gamma and both endpoints
  // free, so it strictly raises the objective. Taking them in belief order means
  // the strongest available association is committed first.
  std::vector<int> rest;
  rest.reserve(edges.size());
  for (size_t e = 0; e < edges.size(); ++e) {
    if (edges[e].s <= cfg.gamma) continue;
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

float pair_score(const Keypoint& l, const Keypoint& r, uint64_t dl, uint64_t dr,
                 int census_bits, const MatchConfig& cfg) {
  // Descriptor term: bits agree by chance half the time, so hamming == bits/2
  // scores 0 and a perfect match scores +1. That puts gamma == 0 at the
  // interpretable place of "no better than chance".
  const float half = 0.5f * float(census_bits);
  const float desc = (half - float(hamming(dl, dr))) / half;

  // y-residual. The plan asks for this as a unary term regardless, since median
  // |dy| over matched pairs doubles as a free online monitor for extrinsic drift.
  const float dy = (l.y - r.y) / (cfg.sigma_y > 0 ? cfg.sigma_y : 1.f);
  return desc - cfg.w_y * dy * dy;
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

    // rho_ij = s_ij - max( gamma, max_{k != j} beta_ik )   -- reduce over rows
    for (size_t i = 0; i < by_row.size(); ++i) {
      Top2 t;
      for (int e : by_row[i]) t.push(edges[e].beta, e);
      for (int e : by_row[i]) {
        const float competitor = std::max(cfg.gamma, t.excluding(e));
        const float target = edges[e].s - competitor;
        const float updated = (1.f - d) * target + d * edges[e].rho;
        max_delta = std::max(max_delta, std::fabs(updated - edges[e].rho));
        edges[e].rho = updated;
      }
    }

    // beta_ij = s_ij - max( lambda, max_{k != i} rho_kj )  -- reduce over cols
    for (size_t j = 0; j < by_col.size(); ++j) {
      Top2 t;
      for (int e : by_col[j]) t.push(edges[e].rho, e);
      for (int e : by_col[j]) {
        const float competitor = std::max(cfg.lambda, t.excluding(e));
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
    if (edges[e].s <= cfg.gamma) continue;          // worse than not matching
    if (best_col[edges[e].j] != e) continue;        // not mutual
    if (second_row[i] >= 0) {
      // Ratio test on the score, shifted so gamma is the origin. On degenerate
      // projected-dot texture the runner-up is often a near-tie, which is exactly
      // the regime this test throws away and MASDA does not have to.
      const float b = edges[e].s - cfg.gamma;
      const float s2 = edges[second_row[i]].s - cfg.gamma;
      if (b > 0.f && s2 > 0.f && s2 / b > ratio) continue;
    }
    Match m;
    m.left = edges[e].i;
    m.right = edges[e].j;
    m.score = edges[e].s;
    m.belief = edges[e].s;
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

  std::vector<int> assign(n, -1), best_assign(n, -1);
  std::vector<char> used(m, 0);
  float best_value = 0.f;
  for (int i = 0; i < n; ++i) best_value += cfg.gamma;  // all unmatched

  // Depth-first over rows: each may take gamma, or any free column it has an
  // edge to. Factorial, hence tests only.
  std::function<void(int, float)> rec = [&](int i, float acc) {
    if (i == n) {
      // Columns left over contribute lambda.
      float total = acc;
      for (int j = 0; j < m; ++j)
        if (!used[j]) total += cfg.lambda;
      if (total > best_value) {
        best_value = total;
        best_assign = assign;
      }
      return;
    }
    assign[i] = -1;
    rec(i + 1, acc + cfg.gamma);              // leave i unmatched
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
    mt.disparity = left[i].x - right[best_assign[i]].x;
    mt.dy = left[i].y - right[best_assign[i]].y;
    out.push_back(mt);
  }
  return out;
}

float matching_objective(const std::vector<Match>& matches,
                         const MatchConfig& cfg, int n_left, int n_right) {
  float total = 0.f;
  for (const Match& m : matches) total += m.score;
  total += cfg.gamma * float(n_left - int(matches.size()));
  total += cfg.lambda * float(n_right - int(matches.size()));
  return total;
}

}  // namespace doubleeye
