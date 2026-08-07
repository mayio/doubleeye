// Run the preprocessing stage over a recorded bag.
//
// Reads the raw Y8 frames a bag saved, detects well-distributed keypoints,
// computes a Census descriptor at each, and writes them to keypoints.csv for the
// matcher (and for desktop/view_keypoints.py) to consume.
//
// It also times each stage per frame. That is deliberate and follows the plan's
// instruction to profile before touching CUDA: the plan's estimate is that MASDA
// itself is sub-millisecond and that preprocessing plus memory bandwidth are the
// real cost. This is where that gets checked rather than assumed -- and the
// numbers it prints are for THIS machine, so run it on the TX2 before drawing
// any conclusion about on-vehicle budget.

#include "doubleeye/preproc.hpp"

#include <dirent.h>
#include <time.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace doubleeye;

namespace {

double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

struct Args {
  std::string bag;
  int width = 848;
  int height = 480;
  int census_half_w = 3;
  int census_half_h = 3;
  DetectorConfig det;
  int limit = 0;   // 0 = all pairs
  bool dump = false;
};

void usage(const char* argv0) {
  std::printf(
      "usage: %s BAG_DIR [options]\n"
      "  --width N --height N     frame geometry (default from run.txt)\n"
      "  --census WxH             7x7 or 9x7 (default 7x7)\n"
      "  --cell N                 grid cell size, px (default 32)\n"
      "  --per-cell N             keypoints kept per cell (default 3)\n"
      "  --min-response F         Shi-Tomasi threshold (default 1.0)\n"
      "  --min-local-std F        texture floor in DN (default 2.0)\n"
      "  --limit N                only process the first N pairs\n"
      "  --dump                   also write a PGM of frame 1 for eyeballing\n",
      argv0);
}

// run.txt is "key value" per line.
std::map<std::string, std::string> read_meta(const std::string& bag) {
  std::map<std::string, std::string> meta;
  FILE* fh = std::fopen((bag + "/run.txt").c_str(), "r");
  if (!fh) return meta;
  char line[512];
  while (std::fgets(line, sizeof(line), fh)) {
    char key[128], val[256];
    if (std::sscanf(line, "%127s %255[^\n]", key, val) == 2) meta[key] = val;
  }
  std::fclose(fh);
  return meta;
}

// Group frames/ir{1,2}_NNNNNNNN.raw by frame number, keeping only complete pairs.
std::vector<std::string> paired_frame_numbers(const std::string& bag) {
  std::map<std::string, int> seen;
  DIR* dir = opendir((bag + "/frames").c_str());
  if (!dir) return {};
  while (struct dirent* e = readdir(dir)) {
    const std::string name = e->d_name;
    if (name.size() < 12 || name.compare(name.size() - 4, 4, ".raw") != 0)
      continue;
    const std::string stem = name.substr(0, name.size() - 4);
    const size_t us = stem.find('_');
    if (us == std::string::npos) continue;
    const std::string idx = stem.substr(0, us);
    const std::string num = stem.substr(us + 1);
    if (idx == "ir1") seen[num] |= 1;
    else if (idx == "ir2") seen[num] |= 2;
  }
  closedir(dir);

  std::vector<std::string> out;
  for (const auto& kv : seen)
    if (kv.second == 3) out.push_back(kv.first);
  std::sort(out.begin(), out.end());
  return out;
}

struct Stats {
  std::vector<double> t_detect, t_census, kp_count, occupancy;
};

void summarise(const char* label, std::vector<double> v, const char* unit) {
  if (v.empty()) return;
  std::sort(v.begin(), v.end());
  const double sum = [&] { double s = 0; for (double x : v) s += x; return s; }();
  std::printf("  %-22s mean %8.3f   median %8.3f   min %8.3f   max %8.3f  %s\n",
              label, sum / v.size(), v[v.size() / 2], v.front(), v.back(), unit);
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  if (argc < 2 || argv[1][0] == '-') { usage(argv[0]); return 2; }
  a.bag = argv[1];

  const auto meta = read_meta(a.bag);
  auto it = meta.find("resolution");
  if (it != meta.end()) {
    int w = 0, h = 0;
    if (std::sscanf(it->second.c_str(), "%dx%d", &w, &h) == 2 && w > 0 && h > 0) {
      a.width = w;
      a.height = h;
    }
  }

  for (int i = 2; i < argc; ++i) {
    const std::string s = argv[i];
    const bool has = (i + 1 < argc);
    if (s == "--width" && has) a.width = std::atoi(argv[++i]);
    else if (s == "--height" && has) a.height = std::atoi(argv[++i]);
    else if (s == "--census" && has) {
      int cw = 7, ch = 7;
      if (std::sscanf(argv[++i], "%dx%d", &cw, &ch) == 2) {
        a.census_half_w = cw / 2;
        a.census_half_h = ch / 2;
      }
    }
    else if (s == "--cell" && has) a.det.cell = std::atoi(argv[++i]);
    else if (s == "--per-cell" && has) a.det.per_cell = std::atoi(argv[++i]);
    else if (s == "--min-response" && has) a.det.min_response = float(std::atof(argv[++i]));
    else if (s == "--min-local-std" && has) a.det.min_local_std = float(std::atof(argv[++i]));
    else if (s == "--limit" && has) a.limit = std::atoi(argv[++i]);
    else if (s == "--dump") a.dump = true;
    else { std::fprintf(stderr, "unknown argument '%s'\n", s.c_str()); usage(argv[0]); return 2; }
  }

  CensusConfig ccfg;
  ccfg.half_w = a.census_half_w;
  ccfg.half_h = a.census_half_h;
  if (ccfg.bits() > 64) {
    std::fprintf(stderr, "census window needs %d bits, max 64\n", ccfg.bits());
    return 2;
  }
  // The Census window must fit inside the detector border or descriptors at
  // edge keypoints would be silently invalid.
  a.det.border = std::max(a.det.border, std::max(ccfg.half_w, ccfg.half_h) + 1);

  const std::vector<std::string> numbers = paired_frame_numbers(a.bag);
  if (numbers.empty()) {
    std::fprintf(stderr,
                 "no complete L/R raw frame pairs in %s/frames\n"
                 "Record with --save-every N (N>0); --save-every 0 saves none.\n",
                 a.bag.c_str());
    return 1;
  }

  std::printf("bag              %s\n", a.bag.c_str());
  std::printf("geometry         %dx%d\n", a.width, a.height);
  std::printf("emitter          %s\n",
              meta.count("emitter") ? meta.at("emitter").c_str() : "?");
  std::printf("census           %dx%d (%d bits)\n", 2 * ccfg.half_w + 1,
              2 * ccfg.half_h + 1, ccfg.bits());
  std::printf("grid             %d px cells, top %d each\n", a.det.cell,
              a.det.per_cell);
  std::printf("thresholds       response >= %.2f, local std >= %.2f DN\n",
              a.det.min_response, a.det.min_local_std);
  std::printf("pairs            %zu\n\n", numbers.size());

  const std::string out_path = a.bag + "/keypoints.csv";
  FILE* out = std::fopen(out_path.c_str(), "w");
  if (!out) {
    std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
    return 1;
  }
  std::fprintf(out, "frame,stream,x,y,response,local_std,census_hex\n");

  Stats stats;
  size_t processed = 0, kp_total = 0;

  for (const std::string& num : numbers) {
    if (a.limit > 0 && processed >= size_t(a.limit)) break;
    for (int stream = 1; stream <= 2; ++stream) {
      char path[512];
      std::snprintf(path, sizeof(path), "%s/frames/ir%d_%s.raw", a.bag.c_str(),
                    stream, num.c_str());
      Image8 img;
      if (!load_raw_y8(path, a.width, a.height, &img)) {
        std::fprintf(stderr, "skip %s (size mismatch or unreadable)\n", path);
        continue;
      }

      const double t0 = now_ms();
      const std::vector<Keypoint> kps = detect_keypoints(img, a.det);
      const double t1 = now_ms();
      // Sparse: one descriptor per keypoint, not one per pixel.
      std::vector<uint64_t> descs(kps.size(), 0ull);
      for (size_t k = 0; k < kps.size(); ++k) {
        const int xi = int(kps[k].x + 0.5f), yi = int(kps[k].y + 0.5f);
        descs[k] = census_at(img, xi, yi, ccfg);
      }
      const double t2 = now_ms();

      stats.t_detect.push_back(t1 - t0);
      stats.t_census.push_back(t2 - t1);
      stats.kp_count.push_back(double(kps.size()));
      stats.occupancy.push_back(
          cell_occupancy(kps, a.width, a.height, a.det.cell));
      kp_total += kps.size();

      for (size_t k = 0; k < kps.size(); ++k) {
        const Keypoint& kp = kps[k];
        std::fprintf(out, "%s,%d,%.3f,%.3f,%.4f,%.3f,%016llx\n", num.c_str(),
                     stream, kp.x, kp.y, kp.response, kp.local_std,
                     static_cast<unsigned long long>(descs[k]));
      }

      if (a.dump && processed == 0 && stream == 1)
        save_pgm(a.bag + "/preproc_frame.pgm", img);
    }
    ++processed;
  }
  std::fclose(out);

  std::printf("wrote %s (%zu keypoints over %zu pairs)\n\n", out_path.c_str(),
              kp_total, processed);

  std::printf("per-frame statistics\n");
  summarise("keypoints", stats.kp_count, "");
  summarise("cell occupancy", stats.occupancy, "fraction of cells");
  summarise("detect time", stats.t_detect, "ms");
  summarise("census time", stats.t_census, "ms");

  if (!stats.t_detect.empty()) {
    double d = 0, c = 0;
    for (double x : stats.t_detect) d += x;
    for (double x : stats.t_census) c += x;
    const double per_frame = (d + c) / double(stats.t_detect.size());
    std::printf("\n  combined %.2f ms per frame -> %.2f ms per stereo pair\n",
                per_frame, 2.0 * per_frame);
    std::printf("  at 30 Hz a stereo pair has a 33.3 ms budget, so this uses "
                "%.1f%% of it\n", 100.0 * 2.0 * per_frame / 33.333);
    std::printf("  (measured on THIS machine -- rerun on the TX2 for the\n"
                "   on-vehicle number; the plan expects preprocessing and\n"
                "   memory bandwidth to dominate, not MASDA)\n");
  }
  return 0;
}
