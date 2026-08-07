// Portable sparse-stereo preprocessing: Census descriptors and well-distributed
// keypoints.
//
// This is the third top-level component the plan's portability rule calls for.
// It depends on NOTHING but the C++14 standard library -- no librealsense, no
// OpenCV, no CUDA, no Jetson headers. That is what keeps a future Orin Nano
// migration a weekend rather than a rewrite, and it also means the whole thing
// compiles and runs on the desktop against recorded bags.
//
// Descriptor choice follows the plan: Census over a small window, Hamming
// distance. With the IR projector on, every projected dot looks locally
// identical, so discriminability lives in the *constellation* of dots inside the
// window rather than in any single dot's appearance. Census reads exactly that,
// and is invariant to any monotonic intensity mapping, which absorbs the
// measured 2.6 DN exposure mismatch between the two sensors for free.

#ifndef DOUBLEEYE_PREPROC_HPP
#define DOUBLEEYE_PREPROC_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace doubleeye {

// ---------------------------------------------------------------------------
// Image

struct Image8 {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> data;

  Image8() = default;
  Image8(int w, int h) : width(w), height(h), data(size_t(w) * h, 0) {}

  bool empty() const { return width <= 0 || height <= 0; }
  size_t pixels() const { return size_t(width) * size_t(height); }
  uint8_t at(int x, int y) const { return data[size_t(y) * width + x]; }
  uint8_t& at(int x, int y) { return data[size_t(y) * width + x]; }
};

// Raw Y8 is exactly one byte per pixel with no header -- what rs_ir_capture
// writes. Size is therefore not self-describing and must be supplied.
bool load_raw_y8(const std::string& path, int width, int height, Image8* out);

// PGM so intermediate images can be inspected without an image library.
bool save_pgm(const std::string& path, const Image8& img);

// Box-average downsample by an integer factor, for the coarse level of the
// coarse-to-fine search. Averaging rather than decimating on purpose: dropping
// pixels aliases the projected dot pattern, which is exactly the high-frequency
// content that would then fabricate false coarse matches.
Image8 downsample(const Image8& img, int factor);

// ---------------------------------------------------------------------------
// Census

struct CensusConfig {
  // 7x7 (half 3,3 -> 48 bits) or 9x7 (half 4,3 -> 62 bits). Both fit a uint64,
  // which is what keeps Hamming distance a single popcount.
  int half_w = 3;
  int half_h = 3;

  int bits() const { return (2 * half_w + 1) * (2 * half_h + 1) - 1; }
};

struct CensusImage {
  int width = 0;
  int height = 0;
  std::vector<uint64_t> bits;   // descriptor per pixel
  std::vector<uint8_t> valid;   // 0 inside the border where the window is cut
  CensusConfig config;

  uint64_t at(int x, int y) const { return bits[size_t(y) * width + x]; }
  bool is_valid(int x, int y) const { return valid[size_t(y) * width + x] != 0; }
};

// Classic Census: each window neighbour compared against the centre pixel.
//
// A "modified Census" comparing against the window *mean* is measurably more
// robust, because it does not stake all 48 bits on one possibly-noisy centre
// pixel. Kept classic here to match the plan; worth revisiting once matching
// quality is measurable, which is the point at which the comparison is cheap.
CensusImage census_transform(const Image8& img, CensusConfig cfg);

// Census at a single point.
//
// This, not the dense transform, is the hot path. Sparse stereo needs
// descriptors only at keypoints -- roughly 1100 of them against 407040 pixels.
// Measured: the dense transform cost 61 ms per frame while ~0.3% of its output
// was ever read. Prefer this; census_transform() remains for whole-image
// analysis and for the tests.
//
// Returns 0 and sets *ok=false when the window would cross the border.
uint64_t census_at(const Image8& img, int x, int y, CensusConfig cfg,
                   bool* ok = nullptr);

inline int hamming(uint64_t a, uint64_t b) {
  return __builtin_popcountll(a ^ b);
}

// ---------------------------------------------------------------------------
// Local contrast
//
// Per-pixel standard deviation over a square window, via integral images so it
// is O(1) per pixel regardless of window size.
//
// This is not decoration. Measured on real recordings: with the projector off,
// 57% of the image has 7x7 local std below 2 DN, i.e. is textureless at the
// Census window scale; with it on, 25%. Detecting keypoints in those regions
// produces descriptors made of sensor noise, which is worse than no keypoint at
// all because it manufactures confident wrong matches.
std::vector<float> local_std(const Image8& img, int window);

// Local standard deviation at a single point, computed directly over the window.
// Same reasoning as census_at: the texture test is only needed at candidate
// keypoints that already survived thresholding and non-maximum suppression, so
// evaluating it densely does hundreds of times more work than required.
float local_std_at(const Image8& img, int x, int y, int window);

// ---------------------------------------------------------------------------
// Keypoints

struct Keypoint {
  float x = 0.f;          // sub-pixel
  float y = 0.f;
  float response = 0.f;   // Shi-Tomasi min-eigenvalue
  float local_std = 0.f;  // texture energy at this point
};

struct DetectorConfig {
  int grad_window = 3;         // structure-tensor box window
  int cell = 32;               // grid cell for spatial distribution
  int per_cell = 3;            // keep top-N per cell
  float min_response = 1.0f;
  float min_local_std = 2.0f;  // the measured noise floor
  int border = 8;              // must exceed the Census half-window
  bool subpixel = true;
  // 8 DN, chosen by measurement: 96% of the dense detector's keypoints and 98%
  // of its cell coverage at 1.45x the speed. Below ~5 the candidate count grows
  // until the sparse path costs MORE than the dense scan it replaces.
  int fast_threshold = 8;
  int nms_radius = 3;          // candidate suppression radius, FAST path only
};

// Shi-Tomasi (minimum eigenvalue of the structure tensor) rather than Harris.
// Same cost, but the response is an eigenvalue in intensity units instead of a
// determinant-minus-k-trace-squared blend, so a threshold means something and
// transfers between scenes. k in Harris is one more magic number to tune.
std::vector<float> shi_tomasi_response(const Image8& img, int grad_window);

// Detect, then enforce spatial distribution by grid-bucketing and keeping the
// top-N per cell.
//
// The bucketing is essential, not a refinement: without it every keypoint piles
// onto the highest-texture region. On these recordings that means the projected
// dots on the nearest cardboard box, while the far wall -- where disparity
// precision is worst and most needs measurements -- gets nothing.
std::vector<Keypoint> detect_keypoints(const Image8& img,
                                       const DetectorConfig& cfg);

// ---------------------------------------------------------------------------
// FAST-based detection — the cheap path
//
// Profiling said the dense Shi-Tomasi response costs 49 ms per frame on the TX2,
// which is most of the 299%-of-budget figure. But almost all of it is thrown
// away: ~1100 keypoints survive out of 407040 pixels.
//
// This is the same mistake the dense Census transform made, and the same fix
// applies. FAST is a handful of comparisons per pixel and cheap enough to run
// densely; the structure tensor is then only evaluated at the few thousand pixels
// FAST nominates. Shi-Tomasi still does the scoring and sub-pixel work, so
// keypoint *quality* is unchanged -- FAST only decides where to look.

// FAST-9 on the 16-pixel Bresenham circle of radius 3: a corner needs 9
// contiguous circle pixels all brighter than centre+t or all darker than
// centre-t.
bool is_fast_corner(const Image8& img, int x, int y, int threshold);

// Shi-Tomasi minimum eigenvalue at one point, structure tensor summed directly
// over the window. The sparse counterpart of shi_tomasi_response().
float shi_tomasi_at(const Image8& img, int x, int y, int grad_window);

// FAST for candidates, Shi-Tomasi at candidates only. Same output contract as
// detect_keypoints(): thresholded, suppressed, grid-bucketed, sub-pixel refined.
std::vector<Keypoint> detect_keypoints_fast(const Image8& img,
                                            const DetectorConfig& cfg);

// Fraction of grid cells that ended up with at least one keypoint. A single
// number for "is the coverage actually spread out", which is the property the
// bucketing exists to produce and therefore the one to verify.
float cell_occupancy(const std::vector<Keypoint>& kps, int width, int height,
                     int cell);

}  // namespace doubleeye

#endif  // DOUBLEEYE_PREPROC_HPP
