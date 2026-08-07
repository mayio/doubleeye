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
  std::vector<double> count, objective, abs_dy, ms, disp, margin;
  void add(const std::vector<Match>& m, double t, int nl, int nr,
           const MatchConfig& c) {
    count.push_back(double(m.size()));
    objective.push_back(matching_objective(m, c, nl, nr));
    ms.push_back(t);
    for (const Match& x : m) {
      abs_dy.push_back(std::fabs(double(x.dy)));
      disp.push_back(double(x.disparity));
      margin.push_back(double(x.margin));
    }
  }
  void report(const char* name) const {
    if (count.empty()) { std::printf("  %-10s no frames\n", name); return; }
    double c = 0, o = 0, t = 0;
    for (double x : count) c += x;
    for (double x : objective) o += x;
    for (double x : ms) t += x;
    // Median margin and the low-margin share are reported because the margin is
    // the per-match confidence the consumer should weight by: over eight
    // Middlebury scenes, precision by margin quartile runs 0.169 to 0.659. On
    // real data there is no ground truth here, so this is the closest thing to a
    // quality signal the tool can print.
    std::vector<double> mg = margin;
    double weak = 0;
    for (double x : mg) if (x < 0.2) weak += 1;
    std::printf("  %-10s %7.1f matches  objective %8.2f  median|dy| %6.3f px  "
                "median d %6.2f px  median margin %6.3f  margin<0.2 %4.1f%%  "
                "%6.2f ms\n",
                name, c / count.size(), o / objective.size(), median(abs_dy),
                median(disp), median(mg),
                mg.empty() ? 0.0 : 100.0 * weak / double(mg.size()),
                t / ms.size());
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
        "usage: %s BAG_DIR [--limit N] [--damping F] [--iterations N]\n"
        "       [--lambda F] [--gamma F] [--max-dy F] [--max-candidates N]\n"
        "       [--sigma-y F] [--w-y F] [--nn-ratio F]\n", argv[0]);
    return 2;
  }
  const std::string bag = argv[1];
  MatchConfig cfg;
  DetectorConfig det;
  CensusConfig ccfg;
  int limit = 0;
  float nn_ratio = 0.85f;
  int scale = 0;          // 0 = single level; else coarse-to-fine factor
  int prior_cell = 64;
  int smooth_passes = 0;  // 0/1 = off
  int knn = 10;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has = i + 1 < argc;
    if (a == "--limit" && has) limit = std::atoi(argv[++i]);
    else if (a == "--damping" && has) cfg.damping = float(std::atof(argv[++i]));
    else if (a == "--iterations" && has) cfg.iterations = std::atoi(argv[++i]);
    else if (a == "--lambda" && has) cfg.lambda = float(std::atof(argv[++i]));
    else if (a == "--gamma" && has) cfg.gamma = float(std::atof(argv[++i]));
    else if (a == "--max-dy" && has) cfg.max_dy = float(std::atof(argv[++i]));
    else if (a == "--max-candidates" && has) cfg.max_candidates = std::atoi(argv[++i]);
    else if (a == "--sigma-y" && has) cfg.sigma_y = float(std::atof(argv[++i]));
    else if (a == "--w-y" && has) cfg.w_y = float(std::atof(argv[++i]));
    else if (a == "--nn-ratio" && has) nn_ratio = float(std::atof(argv[++i]));
    else if (a == "--coarse-to-fine" && has) scale = std::atoi(argv[++i]);
    else if (a == "--w-prior" && has) cfg.w_prior = float(std::atof(argv[++i]));
    else if (a == "--prior-cell" && has) prior_cell = std::atoi(argv[++i]);
    else if (a == "--smooth-passes" && has) smooth_passes = std::atoi(argv[++i]);
    else if (a == "--w-smooth" && has) cfg.w_smooth = float(std::atof(argv[++i]));
    else if (a == "--knn" && has) knn = std::atoi(argv[++i]);
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
  std::printf("MASDA          %d iterations, damping %.2f, lambda %.3f, "
              "gamma %.3f\n", cfg.iterations, cfg.damping, cfg.lambda, cfg.gamma);
  std::printf("candidates     epipolar band +-%.1f px, disparity [%.0f, %.0f], "
              "<=%d per keypoint\n\n", cfg.max_dy, cfg.min_disparity,
              cfg.max_disparity, cfg.max_candidates);

  FILE* out = std::fopen((bag + "/matches.csv").c_str(), "w");
  if (out)
    std::fprintf(out, "frame,method,left,right,xl,yl,xr,yr,disparity,dy,score,"
                      "belief,margin\n");

  Agg masda, nn, c2f;
  std::vector<double> cand, iters, conv, osc, left_counts;
  std::vector<double> c2f_cand, c2f_cov, c2f_ms, c2f_iters, c2f_osc;
  std::vector<std::vector<float>> sm_obj, sm_dy, sm_cov;
  std::vector<std::vector<int>> sm_cnt, sm_chg;
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

    if (scale > 1) {
      CoarseToFineResult r = match_coarse_to_fine(a, b, det, ccfg, cfg, scale,
                                                  prior_cell);
      c2f.add(r.matches, r.coarse_ms + r.fine_ms, r.fine_stats.left,
              r.fine_stats.right, cfg);
      c2f_cand.push_back(double(r.fine_stats.candidates));
      c2f_cov.push_back(double(r.prior.coverage()));
      c2f_ms.push_back(r.coarse_ms);
      c2f_iters.push_back(double(r.fine_stats.iterations_run));
      c2f_osc.push_back(double(r.fine_stats.oscillating));
    }

    if (smooth_passes > 1) {
      const double ts = now_ms();
      SmoothResult r = match_iterated_smoothness(kl, dl, kr, dr, cfg,
                                                 smooth_passes, knn);
      sm_obj.push_back(r.base_objective);
      sm_dy.push_back(r.median_abs_dy);
      sm_cov.push_back(r.prior_coverage);
      sm_cnt.push_back(r.counts);
      sm_chg.push_back(r.changed);
      (void)ts;
    }

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
          std::fprintf(out,
                       "%s,%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                       num.c_str(), name, x.left, x.right,
                       kl[x.left].x, kl[x.left].y, kr[x.right].x, kr[x.right].y,
                       x.disparity, x.dy, x.score, x.belief, x.margin);
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
  if (!sm_obj.empty()) {
    const size_t np = sm_obj[0].size();
    std::printf("\n  iterated smoothness, %zu passes, k=%d neighbours\n",
                np, knn);
    std::printf("    %-6s %-9s %-16s %-11s %-9s %s\n", "pass", "matches",
                "base objective", "median|dy|", "changed", "prior cov");
    for (size_t q = 0; q < np; ++q) {
      double o = 0, c = 0, d = 0, ch = 0, cv = 0;
      for (size_t f = 0; f < sm_obj.size(); ++f) {
        if (q >= sm_obj[f].size()) continue;
        o += sm_obj[f][q]; c += sm_cnt[f][q]; d += sm_dy[f][q];
        ch += sm_chg[f][q]; cv += sm_cov[f][q];
      }
      const double n = double(sm_obj.size());
      std::printf("    %-6zu %-9.1f %-16.2f %-11.3f %-9.1f %.0f%%\n",
                  q + 1, c / n, o / n, d / n, ch / n, 100.0 * cv / n);
    }
    std::printf("    (base objective excludes the smoothness term, so passes are\n"
                "     comparable; median |dy| is the independent check, being\n"
                "     independent of disparity entirely)\n");
  }
  if (!c2f.count.empty()) {
    c2f.report("MASDA c2f");
    double cc = 0, cv = 0, cms = 0, ci = 0, co = 0;
    for (double x : c2f_cand) cc += x;
    for (double x : c2f_cov) cv += x;
    for (double x : c2f_ms) cms += x;
    for (double x : c2f_iters) ci += x;
    for (double x : c2f_osc) co += x;
    const double n2 = double(c2f_cand.size());
    double kl2 = 0;
    for (double x : left_counts) kl2 += x;
    std::printf("\n  coarse-to-fine at 1/%d: prior covers %.0f%% of cells, "
                "%.0f fine candidate edges (%.2f per keypoint),\n"
                "  %.1f iterations, %.2f oscillating, coarse pass %.2f ms\n",
                scale, 100.0 * cv / n2, cc / n2,
                kl2 > 0 ? cc / kl2 : 0.0, ci / n2, co / n2, cms / n2);
  }

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
