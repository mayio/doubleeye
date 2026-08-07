// MASDA as the matcher for sparse stereo correspondence.
//
// Max-sum loopy belief propagation for bipartite data association with
// mutual-exclusivity constraints, plus clutter and misdetection terms. Left
// keypoints are "measurements", right keypoints are "objects".
//
// The message updates, from the plan:
//
//     beta_ij  = s(i,j) - max_{k != i} rho_kj
//     rho_ij   = s(i,j) - max_{k != j} beta_ik
//
// with the misdetection and clutter options folded into those maxima, since
// "leave i unmatched" and "j is clutter" are competing explanations:
//
//     rho_ij   = s(i,j) - max( gamma, max_{k != j} beta_ik )
//     beta_ij  = s(i,j) - max( lambda, max_{k != i} rho_kj )
//
// Both maxima exclude one element, which is why each iteration is two
// top-2 reductions -- one over rows, one over columns -- rather than a
// quadratic scan. That is what keeps the plan's estimate of ~10^5-10^6 ops per
// frame honest.
//
// WHY THIS PROBLEM WANTS MASDA, measured rather than assumed: with the IR
// projector on, Census descriptors at keypoints are ~3.3x degenerate -- 338
// distinct descriptors serve 1115 keypoints (see doc/06-preprocessing.md). Every
// projected dot looks locally identical, so a nearest-neighbour matcher is
// choosing between near-ties. Mutual exclusivity is the information that breaks
// them, which is exactly the plan's argument for using MASDA here.
//
// A mutual-nearest-neighbour baseline is provided alongside, because that claim
// is only worth anything if the two are compared on the same data.

#ifndef DOUBLEEYE_MATCH_HPP
#define DOUBLEEYE_MATCH_HPP

#include "doubleeye/preproc.hpp"

#include <cstdint>
#include <vector>

namespace doubleeye {

struct Match {
  int left = -1;
  int right = -1;
  float score = 0.f;      // s(i,j)
  float belief = 0.f;     // max-sum belief at the accepted edge
  float disparity = 0.f;  // xL - xR, sub-pixel
  float dy = 0.f;         // yL - yR, the free rectification health metric
};

// A soft, spatially varying expectation of disparity, built from a coarse pass.
//
// The plan is explicit that this must be SOFT: hard pruning at the coarse level
// is how thin structures get lost, and an apartment is full of chair and table
// legs. So it enters s(i,j) as a penalty and never removes a candidate.
//
// Confidence comes out of the data rather than being configured. Each cell's
// sigma is a robust spread (MAD) of the coarse disparities that landed in it, so
// a cell straddling a depth discontinuity automatically gets a wide, weak prior,
// and a cell with too few coarse matches is marked invalid and imposes none at
// all. That is the plan's "keep the full range where coarse confidence is low or
// near a coarse disparity discontinuity", obtained for free.
struct DisparityPrior {
  int cell = 64;              // full-resolution pixels per cell
  int cols = 0;
  int rows = 0;
  std::vector<float> disparity;   // predicted, full-resolution px
  std::vector<float> sigma;       // robust spread, full-resolution px
  std::vector<uint8_t> valid;

  bool lookup(float x, float y, float* d, float* sd) const;
  float coverage() const;         // fraction of cells with a usable prediction
};

// Build the prior from coarse matches. `scale` is the downsample factor, so a
// coarse disparity of 4.7 px at scale 8 predicts 37.6 px at full resolution.
DisparityPrior build_disparity_prior(const std::vector<Match>& coarse,
                                     const std::vector<Keypoint>& coarse_left,
                                     int full_width, int full_height, int scale,
                                     int cell = 64, int min_count = 3,
                                     float sigma_floor = 2.0f);

struct MatchConfig {
  // --- candidate generation ---
  // The pair is rectified (verified: zero distortion, identity rotation), so a
  // true match lies on the same image row up to calibration residual.
  float max_dy = 2.0f;
  float min_disparity = 1.0f;
  float max_disparity = 220.0f;   // 220 px ~ 0.1 m at f*B = 21.5 px*m
  int max_candidates = 24;        // per left keypoint, best by Hamming

  // --- scoring ---
  // s(i,j) is a stand-in for a calibrated log-likelihood ratio. The plan flags
  // calibrating this properly (its candidate: a small MLP over descriptor
  // distance, y-residual, coarse-disparity residual, response ratio) as an open
  // question, so this is deliberately simple and interpretable rather than
  // tuned: descriptor agreement scaled to roughly [-1, 1], minus a quadratic
  // y-residual penalty.
  float sigma_y = 1.0f;           // px, scale of the y-residual penalty
  float w_y = 1.0f;               // weight on that penalty

  // --- MASDA ---
  // Soft disparity prior. Null means no prior, i.e. a single-level search.
  const DisparityPrior* prior = nullptr;
  float w_prior = 1.0f;           // weight on the (d - d_pred)/sigma penalty

  int iterations = 20;
  float damping = 0.4f;           // plan expects 0.3-0.5 for stereo
  float gamma = 0.0f;             // misdetection: score of leaving a left kp unmatched
  float lambda = 0.0f;            // clutter: score of leaving a right kp unmatched
  float converge_eps = 1e-4f;     // stop when the largest message change is below
};

struct MatchStats {
  int left = 0;
  int right = 0;
  int candidates = 0;         // edges in the association graph
  int iterations_run = 0;
  bool converged = false;
  float final_max_delta = 0.f;
  int oscillating = 0;        // iterations where the change grew rather than fell
};

// s(i,j) for one candidate pair. Exposed so tests and tools can reason about the
// scale that gamma and lambda are compared against.
float pair_score(const Keypoint& l, const Keypoint& r, uint64_t dl, uint64_t dr,
                 int census_bits, const MatchConfig& cfg);

// MASDA. Descriptors must be parallel to their keypoint vectors.
std::vector<Match> match_masda(const std::vector<Keypoint>& left,
                               const std::vector<uint64_t>& desc_left,
                               const std::vector<Keypoint>& right,
                               const std::vector<uint64_t>& desc_right,
                               const MatchConfig& cfg,
                               MatchStats* stats = nullptr);

// Mutual nearest neighbour with a Lowe-style ratio test, over the same
// candidate graph and the same score. The point of comparison.
std::vector<Match> match_mutual_nn(const std::vector<Keypoint>& left,
                                   const std::vector<uint64_t>& desc_left,
                                   const std::vector<Keypoint>& right,
                                   const std::vector<uint64_t>& desc_right,
                                   const MatchConfig& cfg,
                                   float ratio = 0.85f,
                                   MatchStats* stats = nullptr);

// Two-level search: match at 1/scale resolution, build a soft disparity prior from
// the result, then match at full resolution with that prior folded into s(i,j).
//
// The plan calls this the biggest single lever, for two compounding reasons: the
// candidate count per keypoint falls, and -- more importantly -- false candidates
// from repetitive structure disappear before they can create the near-ties that
// drive BP oscillation on this graph.
struct CoarseToFineResult {
  std::vector<Match> matches;
  std::vector<Match> coarse_matches;
  DisparityPrior prior;
  MatchStats fine_stats;
  MatchStats coarse_stats;
  double coarse_ms = 0.0;
  double fine_ms = 0.0;
};

CoarseToFineResult match_coarse_to_fine(const Image8& left_img,
                                        const Image8& right_img,
                                        const DetectorConfig& det,
                                        const CensusConfig& census,
                                        const MatchConfig& cfg,
                                        int scale = 8,
                                        int prior_cell = 64);

// Exact maximum-weight matching by exhaustive search. Only for tests on tiny
// problems -- it is factorial. Having ground truth to compare against is the
// only way to know whether the message passing is converging to the right thing
// rather than merely converging.
std::vector<Match> match_brute_force(const std::vector<Keypoint>& left,
                                     const std::vector<uint64_t>& desc_left,
                                     const std::vector<Keypoint>& right,
                                     const std::vector<uint64_t>& desc_right,
                                     const MatchConfig& cfg);

// Sum of s(i,j) over a matching, the quantity all three methods are trying to
// maximise (subject to gamma/lambda for leaving nodes unmatched).
float matching_objective(const std::vector<Match>& matches,
                         const MatchConfig& cfg, int n_left, int n_right);

}  // namespace doubleeye

#endif  // DOUBLEEYE_MATCH_HPP
