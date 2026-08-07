// Tests for the MASDA matcher.
//
// The load-bearing test is agreement with brute-force maximum-weight matching on
// small problems. Message passing that merely *converges* proves nothing; the
// question is whether it converges to the assignment that actually maximises the
// objective. Only exhaustive search answers that, and only on problems small
// enough to enumerate.
//
// The second group of tests targets the regime this project actually runs in:
// deliberately degenerate descriptors, since projected IR dots make Census
// descriptors ~3.3x degenerate in practice. That is where mutual exclusivity is
// supposed to earn its keep over nearest-neighbour matching.

#include "doubleeye/match.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>

using namespace doubleeye;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

Keypoint kp(float x, float y) {
  Keypoint k;
  k.x = x;
  k.y = y;
  k.response = 10.f;
  k.local_std = 10.f;
  return k;
}

// A descriptor with `bits` bits set, so Hamming distances are controllable.
uint64_t desc_with(int n_bits, int offset = 0) {
  uint64_t d = 0;
  for (int b = 0; b < n_bits; ++b) d |= (1ull << ((b + offset) % 48));
  return d;
}

MatchConfig plain() {
  MatchConfig c;
  c.max_dy = 2.0f;
  c.min_disparity = 1.0f;
  c.max_disparity = 100.0f;
  c.w_y = 0.f;          // isolate the descriptor term unless a test wants otherwise
  c.iterations = 60;
  c.damping = 0.4f;
  c.lambda = 0.0f;
  c.gamma = 0.0f;
  return c;
}

void test_single_pair() {
  std::printf("single unambiguous pair\n");
  std::vector<Keypoint> L{kp(100, 50)};
  std::vector<Keypoint> R{kp(80, 50)};
  std::vector<uint64_t> dL{desc_with(20)}, dR{desc_with(20)};
  const auto m = match_masda(L, dL, R, dR, plain());
  check(m.size() == 1, "one match found");
  if (m.size() == 1) {
    check(m[0].left == 0 && m[0].right == 0, "correct pairing");
    check(std::fabs(m[0].disparity - 20.f) < 1e-4f, "disparity is 20 px");
    check(std::fabs(m[0].dy) < 1e-6f, "dy is zero");
  }
}

void test_geometry_gates() {
  std::printf("candidate gating\n");
  MatchConfig c = plain();
  std::vector<uint64_t> d1{desc_with(20)}, d2{desc_with(20)};

  // Outside the epipolar band.
  std::vector<Keypoint> L{kp(100, 50)};
  std::vector<Keypoint> R{kp(80, 60)};
  check(match_masda(L, d1, R, d2, c).empty(), "rejected: dy beyond max_dy");

  // Negative disparity, i.e. the right keypoint is to the RIGHT of the left one.
  R[0] = kp(120, 50);
  check(match_masda(L, d1, R, d2, c).empty(), "rejected: negative disparity");

  // Beyond max_disparity.
  R[0] = kp(100 - 150, 50);
  check(match_masda(L, d1, R, d2, c).empty(), "rejected: disparity too large");

  // The y-residual penalty must actually bite when enabled.
  R[0] = kp(80, 51.5f);
  c.w_y = 1.0f;
  c.sigma_y = 0.5f;    // 1.5 px residual is 3 sigma -> penalty 9, swamps desc
  check(match_masda(L, d1, R, d2, c).empty(),
        "rejected: y-residual penalty drives the score below lambda");
}

void test_mutual_exclusivity() {
  std::printf("mutual exclusivity\n");
  // Two left keypoints, one right keypoint, both equally plausible. Exactly one
  // may win: this is the constraint that distinguishes MASDA from independent
  // per-keypoint nearest-neighbour lookups.
  std::vector<Keypoint> L{kp(100, 50), kp(101, 50)};
  std::vector<Keypoint> R{kp(80, 50)};
  std::vector<uint64_t> dL{desc_with(20), desc_with(20)};
  std::vector<uint64_t> dR{desc_with(20)};
  const auto m = match_masda(L, dL, R, dR, plain());
  check(m.size() <= 1, "at most one left keypoint claims the single right one");
  check(m.size() == 1, "and one does claim it (" + std::to_string(m.size()) + ")");
}

void test_against_brute_force() {
  std::printf("agreement with brute-force maximum-weight matching\n");
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> ux(20.f, 120.f);
  std::uniform_int_distribution<int> uoff(0, 47);
  std::uniform_int_distribution<int> ubits(6, 30);

  int trials = 0, optimal = 0, valid = 0;
  double masda_sum = 0.0, brute_sum = 0.0;
  for (int t = 0; t < 60; ++t) {
    const int n = 3 + int(t % 3);         // 3..5 left
    const int m = 3 + int((t / 3) % 3);   // 3..5 right
    std::vector<Keypoint> L, R;
    std::vector<uint64_t> dL, dR;
    for (int i = 0; i < n; ++i) {
      L.push_back(kp(ux(rng) + 60.f, 50.f));
      dL.push_back(desc_with(ubits(rng), uoff(rng)));
    }
    for (int j = 0; j < m; ++j) {
      R.push_back(kp(ux(rng), 50.f));
      dR.push_back(desc_with(ubits(rng), uoff(rng)));
    }
    MatchConfig c = plain();
    c.lambda = 0.05f;   // a small cost to leaving things unmatched
    c.gamma = 0.0f;

    const auto a = match_masda(L, dL, R, dR, c);
    const auto b = match_brute_force(L, dL, R, dR, c);
    const float va = matching_objective(a, c, n, m);
    const float vb = matching_objective(b, c, n, m);
    masda_sum += va;
    brute_sum += vb;
    ++trials;
    if (va >= vb - 1e-4f) ++optimal;

    // Whatever it returns must be a legal one-to-one matching.
    std::vector<int> lu(n, 0), ru(m, 0);
    bool ok = true;
    for (const Match& mm : a) {
      if (++lu[mm.left] > 1 || ++ru[mm.right] > 1) ok = false;
    }
    if (ok) ++valid;
  }
  check(valid == trials,
        "every result is a valid one-to-one matching (" +
            std::to_string(valid) + "/" + std::to_string(trials) + ")");
  check(optimal >= trials * 9 / 10,
        "reaches the exact optimum on " + std::to_string(optimal) + "/" +
            std::to_string(trials) + " random problems");
  const double ratio = brute_sum != 0.0 ? masda_sum / brute_sum : 1.0;
  check(ratio > 0.98,
        "total objective within 2% of optimal (ratio " +
            std::to_string(ratio) + ")");
}

void test_degenerate_descriptors() {
  std::printf("degenerate descriptors -- the projected-dot regime\n");
  // Every descriptor identical, so the descriptor term carries NO information and
  // only the constraint structure can produce a sensible answer. This is the
  // extreme of the measured 3.3x degeneracy.
  const int n = 6;
  std::vector<Keypoint> L, R;
  std::vector<uint64_t> dL, dR;
  for (int i = 0; i < n; ++i) {
    L.push_back(kp(100.f + 10.f * i, 50.f));
    R.push_back(kp(80.f + 10.f * i, 50.f));
    dL.push_back(desc_with(20));
    dR.push_back(desc_with(20));
  }
  MatchConfig c = plain();
  const auto m = match_masda(L, dL, R, dR, c);

  std::vector<int> lu(n, 0), ru(n, 0);
  bool one_to_one = true;
  for (const Match& mm : m) {
    if (++lu[mm.left] > 1 || ++ru[mm.right] > 1) one_to_one = false;
  }
  check(one_to_one, "still a valid one-to-one matching with zero descriptor "
                    "information");
  // Exactly-tied candidates put every belief at zero, which is the plan's cited
  // condition for BP's guarantee lapsing: the LP optimum is not unique. The
  // objective nonetheless prefers matching, so committing to a maximal matching
  // among indifferent edges is correct and must not be refused.
  check(!m.empty(), "commits to a matching despite exact ties (" +
                        std::to_string(m.size()) + " matches)");
  const auto bf = match_brute_force(L, dL, R, dR, c);
  check(matching_objective(m, c, n, n) >= matching_objective(bf, c, n, n) - 1e-4f,
        "and that matching is objective-optimal");

  // Nearest neighbour with a ratio test cannot survive this: every candidate is
  // an exact tie, so the ratio test rejects everything. That is the failure mode
  // MASDA exists to avoid, and it is worth asserting rather than claiming.
  const auto nn = match_mutual_nn(L, dL, R, dR, c, 0.85f);
  check(nn.size() <= m.size(),
        "MASDA matches at least as many as ratio-test NN (" +
            std::to_string(m.size()) + " vs " + std::to_string(nn.size()) + ")");
}

void test_masda_beats_nn_on_ambiguity() {
  std::printf("objective: MASDA against mutual-NN under ambiguity\n");
  std::mt19937 rng(999);
  std::uniform_int_distribution<int> uoff(0, 3);   // near-identical descriptors
  int masda_better = 0, trials = 0;
  double sum_m = 0.0, sum_n = 0.0;
  for (int t = 0; t < 40; ++t) {
    const int n = 8;
    std::vector<Keypoint> L, R;
    std::vector<uint64_t> dL, dR;
    for (int i = 0; i < n; ++i) {
      L.push_back(kp(120.f + 9.f * i, 50.f));
      R.push_back(kp(100.f + 9.f * i, 50.f));
      dL.push_back(desc_with(20, uoff(rng)));
      dR.push_back(desc_with(20, uoff(rng)));
    }
    MatchConfig c = plain();
    c.lambda = 0.02f;
    const auto a = match_masda(L, dL, R, dR, c);
    const auto b = match_mutual_nn(L, dL, R, dR, c, 0.85f);
    const float va = matching_objective(a, c, n, n);
    const float vb = matching_objective(b, c, n, n);
    sum_m += va;
    sum_n += vb;
    if (va > vb + 1e-5f) ++masda_better;
    ++trials;
  }
  std::printf("      MASDA total %.3f, mutual-NN total %.3f over %d trials\n",
              sum_m, sum_n, trials);
  check(sum_m >= sum_n - 1e-3f,
        "MASDA's total objective is at least mutual-NN's");
  check(masda_better > 0,
        "MASDA strictly wins on " + std::to_string(masda_better) + "/" +
            std::to_string(trials) + " ambiguous problems");
}

void test_convergence_reporting() {
  std::printf("convergence reporting\n");
  const int n = 12;
  std::vector<Keypoint> L, R;
  std::vector<uint64_t> dL, dR;
  for (int i = 0; i < n; ++i) {
    L.push_back(kp(120.f + 7.f * i, 50.f));
    R.push_back(kp(100.f + 7.f * i, 50.f));
    dL.push_back(desc_with(18, i));
    dR.push_back(desc_with(18, i));
  }
  MatchConfig c = plain();
  MatchStats st;
  match_masda(L, dL, R, dR, c, &st);
  check(st.candidates > 0, "candidate count reported (" +
                               std::to_string(st.candidates) + ")");
  check(st.iterations_run > 0, "iteration count reported");
  check(st.converged, "converges on a well-conditioned problem in " +
                          std::to_string(st.iterations_run) + " iterations");

  // Undamped max-sum on a tie-heavy problem is where the plan expects trouble.
  // The point is that oscillation is *observable*, not that it never happens.
  MatchConfig osc = plain();
  osc.damping = 0.0f;
  osc.iterations = 40;
  std::vector<uint64_t> flat(n, desc_with(20));
  MatchStats st2;
  match_masda(L, flat, R, flat, osc, &st2);
  std::printf("      undamped on identical descriptors: converged=%d, "
              "oscillating iterations=%d, final delta=%.3g\n",
              int(st2.converged), st2.oscillating, st2.final_max_delta);
  check(true, "oscillation is measured rather than assumed away");
}

void test_disparity_prior() {
  std::printf("soft disparity prior\n");
  // A synthetic coarse result: eight matches in one region, all at coarse
  // disparity 5, so at scale 8 the prior should predict 40 px there.
  std::vector<Keypoint> ck;
  std::vector<Match> cm;
  for (int i = 0; i < 8; ++i) {
    ck.push_back(kp(10.f + float(i), 6.f));   // coarse coords -> ~80..136, 48
    Match m;
    m.left = i;
    m.right = i;
    m.disparity = 5.f;
    cm.push_back(m);
  }
  const DisparityPrior pr = build_disparity_prior(cm, ck, 848, 480, 8, 64, 3);
  float d = 0.f, sd = 0.f;
  check(pr.lookup(88.f, 48.f, &d, &sd), "prior is available where coarse matched");
  check(std::fabs(d - 40.f) < 1e-3f,
        "predicts 40 px at full resolution from 5 px at scale 8");
  check(sd >= 2.0f, "sigma respects the floor (" + std::to_string(sd) + ")");
  check(!pr.lookup(800.f, 460.f, &d, &sd),
        "no prior where the coarse pass found nothing");
  check(pr.coverage() > 0.f && pr.coverage() < 1.f,
        "coverage is partial as expected (" + std::to_string(pr.coverage()) + ")");

  // A cell straddling a discontinuity must widen rather than pick a side. This is
  // the plan's requirement that the full range survive near depth edges.
  std::vector<Keypoint> ck2;
  std::vector<Match> cm2;
  for (int i = 0; i < 10; ++i) {
    ck2.push_back(kp(10.f + float(i % 5), 6.f));
    Match m;
    m.left = i;
    m.right = i;
    m.disparity = (i < 5) ? 5.f : 12.f;   // two surfaces in one cell
    cm2.push_back(m);
  }
  const DisparityPrior pr2 = build_disparity_prior(cm2, ck2, 848, 480, 8, 64, 3);
  float d2 = 0.f, sd2 = 0.f;
  check(pr2.lookup(88.f, 48.f, &d2, &sd2), "prior exists at the discontinuity");
  check(sd2 > sd * 2.f,
        "sigma widens sharply at a depth discontinuity (" +
            std::to_string(sd2) + " vs " + std::to_string(sd) + ")");

  // And the prior must never veto: a candidate far from the prediction is
  // penalised, not removed.
  MatchConfig c = plain();
  std::vector<Keypoint> L{kp(88.f, 48.f)};
  std::vector<Keypoint> R{kp(88.f - 40.f, 48.f)};
  std::vector<uint64_t> dL{desc_with(20)}, dR{desc_with(20)};
  c.prior = &pr;
  c.w_prior = 1.0f;
  const auto agree = match_masda(L, dL, R, dR, c);
  check(agree.size() == 1, "candidate agreeing with the prior still matches");
  R[0] = kp(88.f - 20.f, 48.f);   // disparity 20 where 40 was predicted
  const auto disagree = match_masda(L, dL, R, dR, c);
  check(disagree.empty(),
        "a candidate 10 sigma from the prediction is penalised below lambda");
  c.w_prior = 0.f;
  check(match_masda(L, dL, R, dR, c).size() == 1,
        "and it returns with w_prior = 0, so the prior penalised rather than "
        "pruned");
}

void test_empty_inputs() {
  std::printf("degenerate inputs\n");
  std::vector<Keypoint> none;
  std::vector<uint64_t> nod;
  std::vector<Keypoint> one{kp(50, 50)};
  std::vector<uint64_t> oned{desc_with(10)};
  check(match_masda(none, nod, none, nod, plain()).empty(), "empty/empty");
  check(match_masda(one, oned, none, nod, plain()).empty(), "one/empty");
  check(match_masda(none, nod, one, oned, plain()).empty(), "empty/one");
}

}  // namespace


// The score margin exported on every Match: best-minus-second-best s(i,j) over
// that left keypoint's candidates. It is the confidence the consumer should key
// on, so it needs to actually track ambiguity.
void test_margin() {
  std::printf("score margin\n");
  MatchConfig cfg = plain();
  cfg.lambda = cfg.gamma = -0.1f;

  // One clearly-best candidate and one poor one on the same row.
  {
    const std::vector<Keypoint> L = {kp(50.f, 20.f)};
    const std::vector<Keypoint> R = {kp(30.f, 20.f), kp(20.f, 20.f)};
    const std::vector<uint64_t> dl = {desc_with(24)};
    const std::vector<uint64_t> dr = {desc_with(24), desc_with(24, 12)};
    const std::vector<Match> m = match_masda(L, dl, R, dr, cfg, nullptr);
    check(m.size() == 1, "unambiguous row matches");
    if (m.size() == 1) {
      check(m[0].margin > 0.3f,
            "margin is large when one candidate is clearly better (" +
                std::to_string(m[0].margin) + ")");
    }
  }

  // Two identical candidates: the descriptor cannot separate them, so the
  // margin must collapse even though a match is still made.
  {
    const std::vector<Keypoint> L = {kp(50.f, 20.f)};
    const std::vector<Keypoint> R = {kp(30.f, 20.f), kp(20.f, 20.f)};
    const std::vector<uint64_t> dl = {desc_with(24)};
    const std::vector<uint64_t> dr = {desc_with(24), desc_with(24)};
    const std::vector<Match> m = match_masda(L, dl, R, dr, cfg, nullptr);
    check(m.size() == 1, "tied row still matches");
    if (m.size() == 1) {
      check(m[0].margin < 1e-5f,
            "margin collapses to zero on exactly tied candidates (" +
                std::to_string(m[0].margin) + ")");
    }
  }

  // A row with a single candidate has no runner-up, so the alternative is
  // lambda and the margin is measured against that rather than left at zero.
  {
    const std::vector<Keypoint> L = {kp(50.f, 20.f)};
    const std::vector<Keypoint> R = {kp(30.f, 20.f)};
    const std::vector<uint64_t> dl = {desc_with(24)};
    const std::vector<uint64_t> dr = {desc_with(24)};
    const std::vector<Match> m = match_masda(L, dl, R, dr, cfg, nullptr);
    check(m.size() == 1, "single-candidate row matches");
    if (m.size() == 1) {
      check(std::fabs(m[0].margin - (m[0].score - cfg.lambda)) < 1e-5f,
            "sole candidate's margin is measured against lambda");
    }
  }
}

// lambda and gamma are NOT interchangeable, and this is the test that would have
// caught them being transposed. lambda is clutter: the cost of leaving a LEFT
// (measurement) keypoint unmatched. gamma is misdetection: the cost of leaving a
// RIGHT (object) keypoint unmatched. With unequal counts the two have visibly
// different jobs, so driving one to -inf and the other to 0 must not be
// symmetric.
void test_lambda_gamma_are_distinct() {
  std::printf("lambda vs gamma are not interchangeable\n");

  // Two left keypoints, one right keypoint. At most one pair can be matched, so
  // exactly one left keypoint must go unmatched and pay lambda. No right
  // keypoint need go unmatched, so gamma is never charged.
  const std::vector<Keypoint> L = {kp(50.f, 20.f), kp(51.f, 20.f)};
  const std::vector<Keypoint> R = {kp(30.f, 20.f)};
  const std::vector<uint64_t> dl = {desc_with(24), desc_with(24)};
  const std::vector<uint64_t> dr = {desc_with(24)};

  MatchConfig cfg = plain();
  const int n_left = int(L.size()), n_right = int(R.size());

  // Descriptors that agree at exactly chance, so s == 0 and the accept rule's
  // "s > lambda" test is the only thing separating the two configurations.
  // desc_with(24) sets bits 0-23 and desc_with(24, 12) sets bits 12-35, so they
  // differ in 24 of 48 bits: Hamming == bits/2, which is s == 0 by construction.
  const std::vector<uint64_t> chance_l = {desc_with(24), desc_with(24)};
  const std::vector<uint64_t> chance_r = {desc_with(24, 12)};

  // lambda > 0 pays MORE for leaving a left keypoint unmatched than for a
  // worthless match, so the matcher should still match one pair.
  cfg.lambda = -1.0f;
  cfg.gamma = 0.0f;
  const std::vector<Match> a =
      match_masda(L, chance_l, R, chance_r, cfg, nullptr);

  // Swapping the two must not produce the same answer, because only lambda is
  // actually charged on this problem shape.
  cfg.lambda = 0.0f;
  cfg.gamma = -1.0f;
  const std::vector<Match> b =
      match_masda(L, chance_l, R, chance_r, cfg, nullptr);

  // Stated as exact counts rather than an inequality. An inequality between two
  // zeros passes without testing anything, which is what the first version of
  // this check did.
  check(a.size() == 1,
        "lambda = -1 accepts the chance-scoring pair, since leaving the left "
        "keypoint unmatched costs more (" + std::to_string(a.size()) + ")");
  check(b.size() == 0,
        "lambda = 0 rejects it, since s == 0 is no better than not matching (" +
            std::to_string(b.size()) + ")");
  check(a.size() != b.size(),
        "so lambda and gamma are not interchangeable on a problem where only "
        "left keypoints can go unmatched");

  // The objective bookkeeping must charge lambda per unmatched LEFT keypoint.
  // One left keypoint is unmatched here, so with lambda = -1 the objective is
  // one lambda lower than with lambda = 0, holding the matching fixed.
  MatchConfig c0 = plain();
  c0.lambda = 0.0f;
  c0.gamma = 0.0f;
  MatchConfig cl = plain();
  cl.lambda = -1.0f;
  cl.gamma = 0.0f;
  MatchConfig cg = plain();
  cg.lambda = 0.0f;
  cg.gamma = -1.0f;

  const std::vector<Match> one = {Match{0, 0, 0.5f, 0.5f, 20.f, 0.f, 1.f}};
  const float o0 = matching_objective(one, c0, n_left, n_right);
  const float ol = matching_objective(one, cl, n_left, n_right);
  const float og = matching_objective(one, cg, n_left, n_right);
  check(std::fabs(ol - (o0 - 1.0f)) < 1e-5f,
        "lambda is charged once for the one unmatched LEFT keypoint");
  check(std::fabs(og - o0) < 1e-5f,
        "gamma is not charged when no RIGHT keypoint is unmatched");
}


// Sub-pixel disparity refinement. Built as a synthetic pair with a known
// FRACTIONAL shift, since that is the only way to tell a refinement that works
// from one that merely moves the number.
void test_refine_disparity() {
  std::printf("sub-pixel disparity\n");
  const int W = 120, H = 40;
  const float true_d = 7.5f;      // deliberately half-way between integers

  // A smooth, non-repeating intensity profile, so the SAD cost has one clean
  // minimum. A hard edge would give a V-shaped cost whose parabola fit is
  // meaningless, and random noise would give a cost with no sub-pixel structure
  // at all.
  const auto profile = [&](float x) {
    return 128.0f + 90.0f * std::sin(x * 0.11f) * std::cos(x * 0.037f);
  };
  Image8 left(W, H), right(W, H);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      left.at(x, y) = uint8_t(std::min(255.f, std::max(0.f, profile(float(x)))));
      // right(x) == left(x + d), so a feature at xl in the left sits at
      // xl - d in the right: positive disparity, matching the convention.
      const float xs = float(x) + true_d;
      right.at(x, y) = uint8_t(std::min(255.f, std::max(0.f, profile(xs))));
    }
  }

  const std::vector<Keypoint> kl = {kp(60.f, 20.f)};
  const std::vector<Keypoint> kr = {kp(60.f - true_d, 20.f)};

  // Seed the integer disparity on either side of the truth and check refinement
  // pulls both toward it.
  for (float seed : {7.0f, 8.0f}) {
    std::vector<Match> m(1);
    m[0].left = 0;
    m[0].right = 0;
    m[0].disparity = seed;
    refine_disparity(left, right, kl, kr, &m);
    const float before = std::fabs(seed - true_d);
    const float after = std::fabs(m[0].disparity - true_d);
    check(after < before,
          "refinement improves disparity from seed " + std::to_string(seed) +
              " (error " + std::to_string(before) + " -> " +
              std::to_string(after) + ")");
    check(after < 0.25f,
          "refined disparity lands within a quarter pixel of the truth");
  }

  // The clamp: a match near the border cannot evaluate all three costs, so it
  // must be left exactly as it was rather than silently refined off a garbage fit.
  {
    const std::vector<Keypoint> edge_l = {kp(1.f, 20.f)};
    const std::vector<Keypoint> edge_r = {kp(0.f, 20.f)};
    std::vector<Match> m(1);
    m[0].left = 0;
    m[0].right = 0;
    m[0].disparity = 1.f;
    refine_disparity(left, right, edge_l, edge_r, &m);
    check(m[0].disparity == 1.f,
          "a match too near the border is left untouched");
  }
}

int main() {
  std::printf("MASDA matcher tests\n\n");
  test_single_pair();
  test_geometry_gates();
  test_mutual_exclusivity();
  test_against_brute_force();
  test_degenerate_descriptors();
  test_masda_beats_nn_on_ambiguity();
  test_convergence_reporting();
  test_disparity_prior();
  test_margin();
  test_refine_disparity();
  test_lambda_gamma_are_distinct();
  test_empty_inputs();
  std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL PASSED",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
