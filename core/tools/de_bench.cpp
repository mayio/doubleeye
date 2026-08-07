// Ground-truth benchmark for the C++ matcher, on the Middlebury scenes.
//
// Every accuracy number for this matcher so far came from either (a) a Python
// reimplementation on Middlebury, which has ground truth but a different detector,
// or (b) the C++ path on my own IR bags, which have the right detector and no
// ground truth. Neither combination can tell you whether the C++ matcher is
// getting matches RIGHT, so this closes the gap: C++ matcher, C++ detector,
// Middlebury ground truth.
//
// It exists specifically to check whether the recommended configuration --
// over-propose in the right image, then gate on score margin -- reproduces the
// gain it showed in Python. It should not be turned on by default until it does.
//
// Input is written by article/export_middlebury.py: left.y8, right.y8, disp.f32
// and meta.txt per scene.
//
//   de_bench DIR [--right-density N] [--min-margin F] [--sweep]

#include "doubleeye/match.hpp"
#include "doubleeye/preproc.hpp"

#include <dirent.h>
#include <time.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace doubleeye;

namespace {

const float kTol = 1.0f;   // Middlebury's standard bad-pixel threshold

struct Scene {
  std::string name;
  int width = 0, height = 0;
  float dmax = 60.f;
  Image8 left, right;
  std::vector<float> disp;      // 0 == unknown, matching Middlebury
};

bool load_scene(const std::string& dir, const std::string& name, Scene* s) {
  const std::string d = dir + "/" + name;
  FILE* f = std::fopen((d + "/meta.txt").c_str(), "r");
  if (!f) return false;
  int w = 0, h = 0;
  float dmax = 0.f;
  const int got = std::fscanf(f, "%d %d %f", &w, &h, &dmax);
  std::fclose(f);
  if (got != 3 || w <= 0 || h <= 0) return false;
  s->name = name;
  s->width = w;
  s->height = h;
  s->dmax = dmax;
  if (!load_raw_y8(d + "/left.y8", w, h, &s->left)) return false;
  if (!load_raw_y8(d + "/right.y8", w, h, &s->right)) return false;
  s->disp.assign(size_t(w) * size_t(h), 0.f);
  f = std::fopen((d + "/disp.f32").c_str(), "rb");
  if (!f) return false;
  const size_t n = std::fread(s->disp.data(), sizeof(float), s->disp.size(), f);
  std::fclose(f);
  return n == s->disp.size();
}

struct Eval {
  int matches = 0, scorable = 0, tp = 0, fp = 0, unscorable = 0, matchable = 0;
  double prec() const { return scorable ? double(tp) / scorable : 0.0; }
  double recall() const { return matchable ? double(tp) / matchable : 0.0; }
};

// Same rules as masda_middlebury.evaluate_real, deliberately, so the two sides
// are comparable rather than merely similar:
//   - a match on unknown ground truth has no correct answer, so it is counted
//     separately and excluded from precision rather than scored wrong;
//   - a left keypoint is matchable only if its ground truth is known AND a right
//     keypoint was actually detected within tol of the true correspondence.
Eval evaluate(const Scene& s, const std::vector<Keypoint>& kl,
              const std::vector<Keypoint>& kr,
              const std::vector<Match>& m) {
  Eval e;
  e.matches = int(m.size());
  const auto gt_at = [&](const Keypoint& k) {
    const int x = std::min(std::max(int(k.x + 0.5f), 0), s.width - 1);
    const int y = std::min(std::max(int(k.y + 0.5f), 0), s.height - 1);
    return s.disp[size_t(y) * size_t(s.width) + size_t(x)];
  };
  for (const Match& x : m) {
    const float d_true = gt_at(kl[x.left]);
    if (d_true <= 0.f) { ++e.unscorable; continue; }
    const float d_est = kl[x.left].x - kr[x.right].x;
    if (std::fabs(d_est - d_true) <= kTol) ++e.tp; else ++e.fp;
  }
  e.scorable = e.tp + e.fp;

  // Sort right keypoints by y so the "was the partner detected?" test is a band
  // scan rather than a full sweep per left keypoint.
  std::vector<int> order(kr.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = int(i);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return kr[a].y < kr[b].y; });
  for (const Keypoint& l : kl) {
    const float d_true = gt_at(l);
    if (d_true <= 0.f) continue;
    const float want_x = l.x - d_true;
    const auto lo = std::lower_bound(
        order.begin(), order.end(), l.y - kTol,
        [&](int idx, float v) { return kr[idx].y < v; });
    const auto hi = std::upper_bound(
        order.begin(), order.end(), l.y + kTol,
        [&](float v, int idx) { return v < kr[idx].y; });
    for (auto it = lo; it != hi; ++it) {
      if (std::fabs(kr[*it].x - want_x) <= kTol) { ++e.matchable; break; }
    }
  }
  return e;
}

double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) * 1e3 + double(ts.tv_nsec) / 1e6;
}

struct Row {
  Eval e;
  double ms = 0;
  int kp_l = 0, kp_r = 0, edges = 0;
};

Row run_one(const Scene& s, DetectorConfig det, int right_density,
            float min_margin, MatchConfig cfg) {
  cfg.max_disparity = s.dmax;
  cfg.min_disparity = 1.f;
  CensusConfig ccfg;

  DetectorConfig det_r = det;
  if (right_density > 0) det_r.per_cell = right_density;
  const std::vector<Keypoint> kl = detect_keypoints_fast(s.left, det);
  const std::vector<Keypoint> kr = detect_keypoints_fast(s.right, det_r);

  std::vector<uint64_t> dl(kl.size()), dr(kr.size());
  for (size_t k = 0; k < kl.size(); ++k)
    dl[k] = census_at(s.left, int(kl[k].x + 0.5f), int(kl[k].y + 0.5f), ccfg);
  for (size_t k = 0; k < kr.size(); ++k)
    dr[k] = census_at(s.right, int(kr[k].x + 0.5f), int(kr[k].y + 0.5f), ccfg);

  MatchStats st;
  const double t0 = now_ms();
  std::vector<Match> m = match_masda(kl, dl, kr, dr, cfg, &st);
  const double t1 = now_ms();
  if (min_margin > 0.f) {
    m.erase(std::remove_if(m.begin(), m.end(),
                           [&](const Match& x) { return x.margin < min_margin; }),
            m.end());
  }
  Row r;
  r.e = evaluate(s, kl, kr, m);
  r.ms = t1 - t0;
  r.kp_l = int(kl.size());
  r.kp_r = int(kr.size());
  r.edges = st.candidates;
  return r;
}

Row total(const std::vector<Row>& rows) {
  Row t;
  for (const Row& r : rows) {
    t.e.matches += r.e.matches; t.e.scorable += r.e.scorable;
    t.e.tp += r.e.tp; t.e.fp += r.e.fp;
    t.e.unscorable += r.e.unscorable; t.e.matchable += r.e.matchable;
    t.ms += r.ms; t.kp_l += r.kp_l; t.kp_r += r.kp_r; t.edges += r.edges;
  }
  return t;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
        "usage: %s DIR [--right-density N] [--min-margin F] [--sweep]\n"
        "\n"
        "DIR holds one subdirectory per scene, as written by\n"
        "article/export_middlebury.py. --sweep runs the density x margin grid\n"
        "that the Python experiment ran, so the two can be compared directly.\n",
        argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  int right_density = 0;
  float min_margin = 0.f;
  bool sweep = false;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has = i + 1 < argc;
    if (a == "--right-density" && has) right_density = std::atoi(argv[++i]);
    else if (a == "--min-margin" && has) min_margin = float(std::atof(argv[++i]));
    else if (a == "--sweep") sweep = true;
    else { std::fprintf(stderr, "unknown argument '%s'\n", a.c_str()); return 2; }
  }

  // Discover scenes rather than hard-coding the list, so adding one to the
  // exporter is enough.
  std::vector<std::string> names;
  if (DIR* d = opendir(dir.c_str())) {
    while (struct dirent* e = readdir(d)) {
      const std::string n = e->d_name;
      if (n == "." || n == "..") continue;
      FILE* f = std::fopen((dir + "/" + n + "/meta.txt").c_str(), "r");
      if (f) { std::fclose(f); names.push_back(n); }
    }
    closedir(d);
  }
  std::sort(names.begin(), names.end());
  if (names.empty()) {
    std::fprintf(stderr, "no scenes with meta.txt under %s\n", dir.c_str());
    return 1;
  }

  std::vector<Scene> scenes;
  for (const std::string& n : names) {
    Scene s;
    if (!load_scene(dir, n, &s)) {
      std::fprintf(stderr, "failed to load scene %s\n", n.c_str());
      return 1;
    }
    scenes.push_back(std::move(s));
  }
  std::printf("%zu scenes from %s\n", scenes.size(), dir.c_str());

  DetectorConfig det;
  MatchConfig cfg;
  cfg.lambda = cfg.gamma = -0.1f;
  cfg.iterations = 20;
  cfg.damping = 0.4f;

  const std::vector<int> densities =
      sweep ? std::vector<int>{0, 6, 9, 12} : std::vector<int>{right_density};
  const std::vector<float> margins =
      sweep ? std::vector<float>{0.f, 0.05f, 0.10f, 0.20f, 0.30f}
            : std::vector<float>{min_margin};

  std::printf("\n%-9s %-7s %8s %8s %8s %9s %8s %9s %9s\n",
              "right/c", "margin", "kp_r", "edges", "matches", "correct",
              "prec", "recall", "ms/scene");
  for (int rd : densities) {
    for (float mm : margins) {
      std::vector<Row> rows;
      for (const Scene& s : scenes)
        rows.push_back(run_one(s, det, rd, mm, cfg));
      const Row t = total(rows);
      std::printf("%-9d %-7.2f %8d %8d %8d %9d %8.3f %9.3f %9.2f\n",
                  rd > 0 ? rd : det.per_cell, mm, t.kp_r, t.edges,
                  t.e.matches, t.e.tp, t.e.prec(), t.e.recall(),
                  t.ms / double(scenes.size()));
    }
  }
  return 0;
}
