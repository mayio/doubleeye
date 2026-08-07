// Match a recorded bag with MASDA, and compare against mutual-NN.
//
// The comparison is the point. MASDA's claimed advantage is on ambiguous
// descriptors, and the projected-dot regime is measurably ambiguous (~3.3x
// degenerate), so running both on the same frames is the only way the claim
// means anything.
//
// Reports, per method: match count, the objective being maximised, median |dy|
// (the plan's free rectification-health metric), disparity distribution, and
// timing. Writes matches.csv for the desktop tools.

#include "doubleeye/match.hpp"
#include "doubleeye/preproc.hpp"

#include <dirent.h>
#include <time.h>

#include <algorithm>
#include <cmath>
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

std::map<std::string, std::string> read_meta(const std::string& bag) {
  std::map<std::string, std::string> meta;
  FILE* fh = std::fopen((bag + "/run.txt").c_str(), "r");
  if (!fh) return meta;
  char line[512];
  while (std::fgets(line, sizeof(line), fh)) {
    char k[128], v[256];
    if (std::sscanf(line, "%127s %255[^\n]", k, v) == 2) meta[k] = v;
  }
  std::fclose(fh);
  return meta;
}

std::vector<std::string> pairs_in(const std::string& bag) {
  std::map<std::string, int> seen;
  DIR* dir = opendir((bag + "/frames").c_str());
  if (!dir) return {};
  while (struct dirent* e = readdir(dir)) {
    const std::string n = e->d_name;
    if (n.size() < 12 || n.compare(n.size() - 4, 4, ".raw") != 0) continue;
    const std::string stem = n.substr(0, n.size() - 4);
    const size_t us = stem.find('_');
    if (us == std::string::npos) continue;
    if (stem.substr(0, us) == "ir1") seen[stem.substr(us + 1)] |= 1;
    else if (stem.substr(0, us) == "ir2") seen[stem.substr(us + 1)] |= 2;
  }
  closedir(dir);
  std::vector<std::string> out;
  for (const auto& kv : seen) if (kv.second == 3) out.push_back(kv.first);
  std::sort(out.begin(), out.end());
  return out;
}

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct Agg {
  std::vector<double> count, objective, abs_dy, ms, disp;
  void add(const std::vector<Match>& m, double t, int nl, int nr,
           const MatchConfig& c) {
    count.push_back(double(m.size()));
    objective.push_back(matching_objective(m, c, nl, nr));
    ms.push_back(t);
    for (const Match& x : m) {
      abs_dy.push_back(std::fabs(double(x.dy)));
      disp.push_back(double(x.disparity));
    }
  }
  void report(const char* name) const {
    if (count.empty()) { std::printf("  %-10s no frames\n", name); return; }
    double c = 0, o = 0, t = 0;
    for (double x : count) c += x;
    for (double x : objective) o += x;
    for (double x : ms) t += x;
    std::printf("  %-10s %7.1f matches  objective %8.2f  median|dy| %6.3f px  "
                "median d %6.2f px  %6.2f ms\n",
                name, c / count.size(), o / objective.size(), median(abs_dy),
                median(disp), t / ms.size());
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
        "usage: %s BAG_DIR [--limit N] [--damping F] [--iterations N]\n"
        "       [--gamma F] [--lambda F] [--max-dy F] [--max-candidates N]\n"
        "       [--sigma-y F] [--w-y F] [--nn-ratio F]\n", argv[0]);
    return 2;
  }
  const std::string bag = argv[1];
  MatchConfig cfg;
  DetectorConfig det;
  CensusConfig ccfg;
  int limit = 0;
  float nn_ratio = 0.85f;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has = i + 1 < argc;
    if (a == "--limit" && has) limit = std::atoi(argv[++i]);
    else if (a == "--damping" && has) cfg.damping = float(std::atof(argv[++i]));
    else if (a == "--iterations" && has) cfg.iterations = std::atoi(argv[++i]);
    else if (a == "--gamma" && has) cfg.gamma = float(std::atof(argv[++i]));
    else if (a == "--lambda" && has) cfg.lambda = float(std::atof(argv[++i]));
    else if (a == "--max-dy" && has) cfg.max_dy = float(std::atof(argv[++i]));
    else if (a == "--max-candidates" && has) cfg.max_candidates = std::atoi(argv[++i]);
    else if (a == "--sigma-y" && has) cfg.sigma_y = float(std::atof(argv[++i]));
    else if (a == "--w-y" && has) cfg.w_y = float(std::atof(argv[++i]));
    else if (a == "--nn-ratio" && has) nn_ratio = float(std::atof(argv[++i]));
    else { std::fprintf(stderr, "unknown argument '%s'\n", a.c_str()); return 2; }
  }

  const auto meta = read_meta(bag);
  int w = 848, h = 480;
  auto it = meta.find("resolution");
  if (it != meta.end()) std::sscanf(it->second.c_str(), "%dx%d", &w, &h);

  const std::vector<std::string> nums = pairs_in(bag);
  if (nums.empty()) {
    std::fprintf(stderr, "no complete L/R pairs in %s/frames\n", bag.c_str());
    return 1;
  }

  std::printf("bag            %s  (%dx%d)\n", bag.c_str(), w, h);
  std::printf("emitter        %s\n",
              meta.count("emitter") ? meta.at("emitter").c_str() : "?");
  std::printf("pairs          %zu%s\n", nums.size(),
              limit ? "  (limited)" : "");
  std::printf("MASDA          %d iterations, damping %.2f, gamma %.3f, "
              "lambda %.3f\n", cfg.iterations, cfg.damping, cfg.gamma, cfg.lambda);
  std::printf("candidates     epipolar band +-%.1f px, disparity [%.0f, %.0f], "
              "<=%d per keypoint\n\n", cfg.max_dy, cfg.min_disparity,
              cfg.max_disparity, cfg.max_candidates);

  FILE* out = std::fopen((bag + "/matches.csv").c_str(), "w");
  if (out)
    std::fprintf(out, "frame,method,left,right,xl,yl,xr,yr,disparity,dy,score,"
                      "belief\n");

  Agg masda, nn;
  std::vector<double> cand, iters, conv, osc, left_counts;
  size_t done = 0;
  for (const std::string& num : nums) {
    if (limit && done >= size_t(limit)) break;
    Image8 a, b;
    char p1[512], p2[512];
    std::snprintf(p1, sizeof(p1), "%s/frames/ir1_%s.raw", bag.c_str(), num.c_str());
    std::snprintf(p2, sizeof(p2), "%s/frames/ir2_%s.raw", bag.c_str(), num.c_str());
    if (!load_raw_y8(p1, w, h, &a) || !load_raw_y8(p2, w, h, &b)) continue;

    const std::vector<Keypoint> kl = detect_keypoints_fast(a, det);
    const std::vector<Keypoint> kr = detect_keypoints_fast(b, det);
    std::vector<uint64_t> dl(kl.size()), dr(kr.size());
    for (size_t k = 0; k < kl.size(); ++k)
      dl[k] = census_at(a, int(kl[k].x + 0.5f), int(kl[k].y + 0.5f), ccfg);
    for (size_t k = 0; k < kr.size(); ++k)
      dr[k] = census_at(b, int(kr[k].x + 0.5f), int(kr[k].y + 0.5f), ccfg);

    MatchStats st;
    const double t0 = now_ms();
    const auto m1 = match_masda(kl, dl, kr, dr, cfg, &st);
    const double t1 = now_ms();
    const auto m2 = match_mutual_nn(kl, dl, kr, dr, cfg, nn_ratio);
    const double t2 = now_ms();

    masda.add(m1, t1 - t0, int(kl.size()), int(kr.size()), cfg);
    nn.add(m2, t2 - t1, int(kl.size()), int(kr.size()), cfg);
    cand.push_back(double(st.candidates));
    left_counts.push_back(double(kl.size()));
    iters.push_back(double(st.iterations_run));
    conv.push_back(st.converged ? 1.0 : 0.0);
    osc.push_back(double(st.oscillating));

    if (out) {
      for (int pass = 0; pass < 2; ++pass) {
        const auto& mm = pass ? m2 : m1;
        const char* name = pass ? "nn" : "masda";
        for (const Match& x : mm)
          std::fprintf(out, "%s,%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f\n",
                       num.c_str(), name, x.left, x.right,
                       kl[x.left].x, kl[x.left].y, kr[x.right].x, kr[x.right].y,
                       x.disparity, x.dy, x.score, x.belief);
      }
    }
    ++done;
  }
  if (out) std::fclose(out);

  double cs = 0, is = 0, cv = 0, os = 0;
  for (double x : cand) cs += x;
  for (double x : iters) is += x;
  for (double x : conv) cv += x;
  for (double x : osc) os += x;
  const double nf = double(std::max<size_t>(1, cand.size()));

  double kl_sum = 0;
  for (double x : left_counts) kl_sum += x;
  std::printf("association graph  %.0f candidate edges per pair, %.1f per left "
              "keypoint (k in the plan's terms)\n", cs / nf,
              kl_sum > 0 ? cs / kl_sum : 0.0);
  std::printf("MASDA convergence  %.1f iterations, converged on %.0f%% of pairs, "
              "%.2f oscillating iterations per pair\n\n",
              is / nf, 100.0 * cv / nf, os / nf);

  std::printf("results, averaged over %zu pairs\n", done);
  masda.report("MASDA");
  nn.report("mutual-NN");

  if (!masda.objective.empty() && !nn.objective.empty()) {
    double om = 0, on = 0, cm = 0, cn = 0;
    for (double x : masda.objective) om += x;
    for (double x : nn.objective) on += x;
    for (double x : masda.count) cm += x;
    for (double x : nn.count) cn += x;
    std::printf("\n  MASDA finds %.1f%% %s matches and scores %.1f%% %s on the "
                "objective\n",
                std::fabs(100.0 * (cm - cn) / std::max(1.0, cn)),
                cm >= cn ? "more" : "fewer",
                std::fabs(100.0 * (om - on) / std::max(1e-9, std::fabs(on))),
                om >= on ? "higher" : "lower");
    std::printf("  (objective is the quantity both are maximising; median |dy| is\n"
                "   the independent quality check, since a wrong match has no\n"
                "   reason to land on the same image row)\n");
  }
  std::printf("\nwrote %s/matches.csv\n", bag.c_str());
  return 0;
}
