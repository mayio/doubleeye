// Unit tests for the preprocessing core. No framework: this component has no
// dependencies and it is not worth acquiring one for a handful of assertions.
//
// Each test targets a property the design actually relies on, rather than
// restating the implementation. The Census monotonic-invariance test is the most
// important one here, because that invariance is the whole reason the plan picks
// Census over an intensity-difference measure.

#include "doubleeye/preproc.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace doubleeye;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

void check_near(double got, double want, double tol, const std::string& what) {
  const bool ok = std::fabs(got - want) <= tol;
  std::printf("  [%s] %s (got %.6f, want %.6f +-%.6f)\n", ok ? "PASS" : "FAIL",
              what.c_str(), got, want, tol);
  if (!ok) ++g_failures;
}

Image8 uniform(int w, int h, uint8_t v) {
  Image8 img(w, h);
  for (auto& p : img.data) p = v;
  return img;
}

// Distinct, well-separated values so a strictly monotonic remap cannot create
// ties that would legitimately change Census bits.
Image8 stepped(int w, int h) {
  Image8 img(w, h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      img.at(x, y) = uint8_t(8 * ((x * 3 + y * 5) % 16) + 4);
  return img;
}

void test_census_basics() {
  std::printf("census basics\n");
  CensusConfig c7;
  check(c7.bits() == 48, "7x7 window yields 48 bits");
  CensusConfig c97;
  c97.half_w = 4;
  c97.half_h = 3;
  check(c97.bits() == 62, "9x7 window yields 62 bits (fits uint64)");

  const Image8 flat = uniform(32, 32, 100);
  const CensusImage cen = census_transform(flat, c7);
  check(cen.is_valid(16, 16), "interior pixel is valid");
  check(!cen.is_valid(0, 0), "border pixel is invalid");
  check(cen.at(16, 16) == 0ull, "uniform image gives all-zero descriptor");
  check(hamming(0ull, 0ull) == 0, "hamming of identical is 0");
  check(hamming(0ull, 0xFull) == 4, "hamming counts differing bits");
}

void test_census_monotonic_invariance() {
  std::printf("census invariance to monotonic intensity mapping\n");
  const Image8 a = stepped(40, 40);
  Image8 b = a;
  // Strictly increasing on the values present, and models exactly the case that
  // matters: the two IR sensors differ by a gain and an offset (measured at
  // 2.6 DN of mean level between ir1 and ir2).
  for (auto& p : b.data) p = uint8_t(p / 2 + 20);

  const CensusImage ca = census_transform(a, CensusConfig());
  const CensusImage cb = census_transform(b, CensusConfig());

  int compared = 0, differing = 0;
  for (int y = 4; y < 36; ++y) {
    for (int x = 4; x < 36; ++x) {
      if (!ca.is_valid(x, y) || !cb.is_valid(x, y)) continue;
      ++compared;
      if (ca.at(x, y) != cb.at(x, y)) ++differing;
    }
  }
  check(compared > 500, "enough interior pixels compared");
  check(differing == 0,
        "descriptors identical after gain+offset (" +
            std::to_string(differing) + " of " + std::to_string(compared) +
            " differ)");
}

void test_local_std() {
  std::printf("local standard deviation\n");
  const Image8 flat = uniform(32, 32, 77);
  const std::vector<float> s0 = local_std(flat, 7);
  check_near(s0[16 * 32 + 16], 0.0, 1e-4, "uniform image has zero local std");

  // Vertical stripes alternating 0 and 200: a centred 7-wide window covers 4 of
  // one value and 3 of the other, so the expected std follows from that split
  // rather than being the full 100 of a balanced one.
  Image8 stripes(32, 32);
  for (int y = 0; y < 32; ++y)
    for (int x = 0; x < 32; ++x) stripes.at(x, y) = (x % 2 == 0) ? 0 : 200;
  const std::vector<float> s1 = local_std(stripes, 7);
  const double n = 49.0, n_hi = 7.0 * 3.0;  // 3 of 7 columns are the 200s
  const double mean = 200.0 * n_hi / n;
  const double var = 200.0 * 200.0 * n_hi / n - mean * mean;
  check_near(s1[16 * 32 + 16], std::sqrt(var), 1.0,
             "striped image local std matches analytic value");
}

void test_shi_tomasi_ordering() {
  std::printf("shi-tomasi response ordering\n");
  // Flat region, a straight edge, and a corner in one image, well separated.
  Image8 img = uniform(64, 64, 30);
  for (int y = 8; y < 24; ++y)            // vertical edge at x=40
    for (int x = 40; x < 56; ++x) img.at(x, y) = 220;
  for (int y = 40; y < 56; ++y)          // filled square -> corners
    for (int x = 8; x < 24; ++x) img.at(x, y) = 220;

  const std::vector<float> r = shi_tomasi_response(img, 3);
  const float flat = r[size_t(32) * 64 + 32];
  const float edge = r[size_t(16) * 64 + 40];   // mid-height of the edge
  const float corner = r[size_t(40) * 64 + 8];  // square's top-left corner

  check(flat < 1e-3f, "flat region has ~zero response");
  check(corner > edge, "corner responds more strongly than a straight edge");
  check(edge >= flat, "edge responds at least as much as flat");
}

void test_detect_and_bucketing() {
  std::printf("detection, grid bucketing and sub-pixel\n");
  // A grid of small bright squares, so corners exist all over the image and
  // bucketing has something to spread across.
  Image8 img = uniform(256, 128, 20);
  for (int cy = 16; cy < 128; cy += 32)
    for (int cx = 16; cx < 256; cx += 32)
      for (int y = cy - 4; y <= cy + 4; ++y)
        for (int x = cx - 4; x <= cx + 4; ++x) img.at(x, y) = 200;

  DetectorConfig cfg;
  cfg.cell = 32;
  cfg.per_cell = 2;
  cfg.min_local_std = 0.5f;
  const std::vector<Keypoint> kps = detect_keypoints(img, cfg);

  const int cols = (256 + 31) / 32, rows = (128 + 31) / 32;
  check(!kps.empty(), "keypoints found");
  check(kps.size() <= size_t(cols * rows * cfg.per_cell),
        "per-cell cap respected (" + std::to_string(kps.size()) + " <= " +
            std::to_string(cols * rows * cfg.per_cell) + ")");

  const float occ = cell_occupancy(kps, 256, 128, 32);
  check(occ > 0.5f,
        "coverage spread over cells (occupancy " + std::to_string(occ) + ")");

  bool in_bounds = true, subpixel_seen = false;
  for (const Keypoint& kp : kps) {
    if (kp.x < 0 || kp.x > 256 || kp.y < 0 || kp.y > 128) in_bounds = false;
    if (std::fabs(kp.x - std::floor(kp.x) - 0.5f) > 1e-6f &&
        kp.x != std::floor(kp.x))
      subpixel_seen = true;
  }
  check(in_bounds, "all keypoints inside the image");
  check(subpixel_seen, "sub-pixel refinement moved at least one keypoint");

  DetectorConfig strict = cfg;
  strict.min_local_std = 250.f;  // unreachable for 8-bit data
  check(detect_keypoints(img, strict).empty(),
        "texture floor suppresses everything when set above the range");
}

void test_load_rejects_wrong_size() {
  std::printf("raw loader guards geometry\n");
  const std::string path = "/tmp/de_test_frame.raw";
  Image8 img = stepped(20, 10);
  FILE* fh = std::fopen(path.c_str(), "wb");
  if (!fh) { check(false, "could not create temp file"); return; }
  std::fwrite(img.data.data(), 1, img.pixels(), fh);
  std::fclose(fh);

  Image8 ok_img, bad_img;
  check(load_raw_y8(path, 20, 10, &ok_img), "correct geometry loads");
  check(ok_img.width == 20 && ok_img.height == 10, "geometry preserved");
  // Wrong geometry must fail rather than silently produce garbage, which is how
  // a resolution mismatch would otherwise reach the matcher unnoticed.
  check(!load_raw_y8(path, 848, 480, &bad_img), "too-large geometry rejected");
  check(!load_raw_y8(path, 10, 10, &bad_img), "too-small geometry rejected");
  std::remove(path.c_str());
}

}  // namespace

int main() {
  std::printf("preprocessing core tests\n\n");
  test_census_basics();
  test_census_monotonic_invariance();
  test_local_std();
  test_shi_tomasi_ordering();
  test_detect_and_bucketing();
  test_load_rejects_wrong_size();
  std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL PASSED",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
