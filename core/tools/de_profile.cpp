// Where does the frame budget go, and what does dropping resolution buy?
//
// The pipeline does not close at 30 Hz: preprocessing is 26.54 ms per stereo pair
// against a 33.3 ms budget, and detection is 20.98 of the 21.24 ms per frame. Two
// questions follow, and both were being answered by guesswork.
//
//   1. Which detector stage is the 21 ms? The dense FAST scan touches every pixel
//      while everything after it is sparse, so stage 1 should dominate -- but
//      "should" is not a measurement, and if the cost is actually in the sparse
//      stages then the fix is completely different.
//
//   2. What does dropping resolution buy? Preprocessing ought to be close to
//      pixel-bound, so halving the pixels ought to halve it. Ought to.
//
// Resolution is varied by box-downsampling the captured 848x480 frames rather
// than re-capturing. That measures the algorithm's scaling honestly; it does not
// measure what the sensor would deliver at a native lower mode, which differs in
// noise and in how much real texture survives. The keypoint counts here are
// therefore indicative and the timings are the point.
//
//   de_profile BAG_DIR [--limit N]

#include "doubleeye/match.hpp"
#include "doubleeye/preproc.hpp"

#include <dirent.h>
#include <time.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace doubleeye;

namespace {

double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) * 1e3 + double(ts.tv_nsec) / 1e6;
}

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

std::vector<std::string> frame_numbers(const std::string& bag) {
  std::vector<std::string> nums;
  const std::string d = bag + "/frames";
  if (DIR* dir = opendir(d.c_str())) {
    while (struct dirent* e = readdir(dir)) {
      const std::string n = e->d_name;
      if (n.size() > 8 && n.compare(0, 4, "ir1_") == 0 &&
          n.compare(n.size() - 4, 4, ".raw") == 0) {
        nums.push_back(n.substr(4, n.size() - 8));
      }
    }
    closedir(dir);
  }
  std::sort(nums.begin(), nums.end());
  return nums;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s BAG_DIR [--limit N]\n", argv[0]);
    return 2;
  }
  const std::string bag = argv[1];
  int limit = 30;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--limit" && i + 1 < argc) limit = std::atoi(argv[++i]);
  }

  int W = 848, H = 480;
  if (FILE* f = std::fopen((bag + "/run.txt").c_str(), "r")) {
    char line[256];
    while (std::fgets(line, sizeof(line), f)) {
      int w = 0, h = 0;
      if (std::sscanf(line, "resolution %dx%d", &w, &h) == 2 && w > 0) {
        W = w; H = h;
      }
    }
    std::fclose(f);
  }
  const std::vector<std::string> nums = frame_numbers(bag);
  if (nums.empty()) {
    std::fprintf(stderr, "no ir1_*.raw frames under %s/frames\n", bag.c_str());
    return 1;
  }
  const int n = std::min(int(nums.size()), limit > 0 ? limit : int(nums.size()));
  std::printf("bag %s  %dx%d  %d frames of %zu\n\n", bag.c_str(), W, H, n,
              nums.size());

  DetectorConfig det;
  CensusConfig ccfg;
  MatchConfig cfg;
  cfg.lambda = cfg.gamma = -0.1f;
  cfg.iterations = 20;
  cfg.damping = 0.4f;

  std::printf("%-11s %7s %8s %8s %8s %8s %8s %8s %8s %9s %8s\n",
              "resolution", "kp/img", "cand", "fast", "nms", "refine",
              "census", "match", "pair*", "%budget", "Mpx/s");
  for (int factor : {1, 2, 3}) {
    std::vector<double> fast, nms, refine, cens, match, kps, cands;
    for (int i = 0; i < n; ++i) {
      Image8 a, b;
      if (!load_raw_y8(bag + "/frames/ir1_" + nums[i] + ".raw", W, H, &a)) continue;
      if (!load_raw_y8(bag + "/frames/ir2_" + nums[i] + ".raw", W, H, &b)) continue;
      if (factor > 1) { a = downsample(a, factor); b = downsample(b, factor); }

      DetectProfile pa, pb;
      const std::vector<Keypoint> kl = detect_keypoints_fast(a, det, &pa);
      const std::vector<Keypoint> kr = detect_keypoints_fast(b, det, &pb);

      const double t0 = now_ms();
      std::vector<uint64_t> dl(kl.size()), dr(kr.size());
      for (size_t k = 0; k < kl.size(); ++k)
        dl[k] = census_at(a, int(kl[k].x + 0.5f), int(kl[k].y + 0.5f), ccfg);
      for (size_t k = 0; k < kr.size(); ++k)
        dr[k] = census_at(b, int(kr[k].x + 0.5f), int(kr[k].y + 0.5f), ccfg);
      const double t1 = now_ms();

      // Disparity scales with resolution, so the search range must too or the
      // comparison silently changes the problem as well as its size.
      MatchConfig mc = cfg;
      mc.max_disparity = cfg.max_disparity / float(factor);
      const std::vector<Match> m = match_masda(kl, dl, kr, dr, mc, nullptr);
      const double t2 = now_ms();
      (void)m;

      // Per IMAGE for the detector stages, since that is what a thread does.
      fast.push_back(0.5 * (pa.fast_ms + pb.fast_ms));
      nms.push_back(0.5 * (pa.nms_ms + pb.nms_ms));
      refine.push_back(0.5 * (pa.refine_ms + pb.refine_ms));
      cens.push_back(0.5 * (t1 - t0));
      match.push_back(t2 - t1);
      kps.push_back(0.5 * double(kl.size() + kr.size()));
      cands.push_back(0.5 * double(pa.candidates + pb.candidates));
    }
    if (fast.empty()) continue;
    const int w = W / factor, h = H / factor;
    const double det_img = median(fast) + median(nms) + median(refine)
                           + median(cens);
    // What a real pipeline gets: the two images detected concurrently on
    // separate cores, then one match over the pair.
    const double pair = det_img + median(match);
    const double mpx = (double(w) * double(h) * 2.0) / (pair * 1e3);
    std::printf("%-11s %7.0f %8.0f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.1f%% %8.1f\n",
                (std::to_string(w) + "x" + std::to_string(h)).c_str(),
                median(kps), median(cands), median(fast), median(nms),
                median(refine), median(cens), median(match), pair,
                100.0 * pair / 33.3, mpx);
  }
  std::printf("\n  * pair = the two images detected concurrently (so per-image\n"
              "    detector cost, not doubled) plus one match over the pair.\n"
              "    %%budget is against 33.3 ms for 30 Hz.\n");
  return 0;
}
