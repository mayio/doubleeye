// Dense MASDA in C++: every pixel, every disparity, threaded over rows.
//
// The NumPy version takes 21 s per 450x375 pair. Almost all of that is scatter --
// np.maximum.at over a 9.15M-entry edge list, which is a gather/scatter with an
// indirection per element. The structure does not need any of that.
//
// On a dense disparity grid the problem is regular. Index scores as
//
//     s[(y*W + x)*D + d]        left pixel (y,x) against right pixel (y, x-d)
//
// and both max-sum reductions become constant-stride walks:
//
//   rho update, "max over this LEFT pixel's other candidates": for fixed (y,x)
//   the alternatives are the other d, so it is a contiguous run of D floats.
//
//   beta update, "max over this RIGHT pixel's other claimants": for fixed
//   (y,xr) the claimants are left pixels x = xr+d, i.e. index
//   (y*W + xr + d)*D + d = base + d*(D+1). Stride D+1, still regular.
//
// So there is no edge list, no indirection, and every inner loop is a strided
// top-2 the compiler can vectorise. Rows are independent -- correspondences lie
// on one row -- so they are handed to a thread pool with no synchronisation at
// all beyond the join.
//
//   de_dense LEFT.y8 RIGHT.y8 W H [--dmax N] [--iters N] [--threads N]
//            [--min-margin F] [--out disp.f32]

#include "doubleeye/preproc.hpp"

#include <time.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace doubleeye;

namespace {

double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) * 1e3 + double(ts.tv_nsec) / 1e6;
}

// Census over the whole image, one uint64 per pixel. 7x7 -> 48 bits.
std::vector<uint64_t> census_image(const Image8& img, int hw, int hh) {
  const int W = img.width, H = img.height;
  std::vector<uint64_t> out(size_t(W) * H, 0);
  for (int y = hh; y < H - hh; ++y) {
    for (int x = hw; x < W - hw; ++x) {
      const int c = img.at(x, y);
      uint64_t v = 0;
      int bit = 0;
      for (int dy = -hh; dy <= hh; ++dy)
        for (int dx = -hw; dx <= hw; ++dx) {
          if (dx == 0 && dy == 0) continue;
          v |= uint64_t(img.at(x + dx, y + dy) < c) << bit;
          ++bit;
        }
      out[size_t(y) * W + x] = v;
    }
  }
  return out;
}

inline int popcnt64(uint64_t v) { return __builtin_popcountll(v); }

// Largest and second largest of a strided run, plus the index of the largest.
// The "max excluding one element" that max-sum needs is then O(1).
struct Top2 {
  float b1 = -1e30f, b2 = -1e30f;
  int i1 = -1;
  inline void push(float v, int i) {
    if (v > b1) { b2 = b1; b1 = v; i1 = i; }
    else if (v > b2) { b2 = v; }
  }
  inline float excl(int i) const { return i == i1 ? b2 : b1; }
};

struct Cfg {
  int dmin = 1, dmax = 60, iters = 2, agg = 3;
  bool subpixel = false;   // measured: hurts slightly, see below
  float lambda = -0.1f, gamma = -0.1f, damping = 0.4f, min_margin = 0.f;
  int threads = 0;
};

// One image row. Rows share no left or right pixel, so this needs no locking.
void solve_row(int y, int W, int D, const Cfg& cfg,
               const float* vol,
               float* s, float* beta, float* rho,
               float* out_disp, float* out_margin, int hw,
               float* gath, float* svals, int* idxs, int* k0, int* k1,
               float* best, int* bestk, int* order, char* taken) {
  const int dmin = cfg.dmin;
  const float keep = 1.f - cfg.damping;

  // --- scores, the valid range, and the margin, in one pass ---
  //
  // The valid disparities for pixel x are exactly those with x-d inside the
  // census border, which is a contiguous interval. Storing [k0, k1) per pixel
  // removes the sentinel comparison from every subsequent inner loop -- the
  // difference between a branch per element and none.
  for (int x = 0; x < W; ++x) {
    float* sp = s + size_t(x) * D;
    int lo = 0, hi = 0;
    if (x >= hw && x < W - hw) {
      lo = std::max(0, x - (W - hw) + 1 - dmin + 1);
      hi = std::min(D, x - hw - dmin + 1);
      if (hi < lo) hi = lo;
    }
    k0[x] = lo; k1[x] = hi;
    Top2 t;
    for (int k = lo; k < hi; ++k) {
      const float v = vol[size_t(x) * D + k];
      sp[k] = v;
      if (v > -1e29f) t.push(v, k);
    }
    const float alt = (t.b2 < -1e29f) ? cfg.lambda : std::max(t.b2, cfg.lambda);
    out_margin[x] = (t.i1 < 0) ? 0.f : t.b1 - alt;
  }

  std::fill(beta, beta + size_t(W) * D, 0.f);
  std::fill(rho, rho + size_t(W) * D, 0.f);

  for (int it = 0; it < cfg.iters; ++it) {
    // rho_ij = s - max(lambda, max over this LEFT pixel's other disparities of beta)
    // Contiguous run of D per pixel.
    for (int x = 0; x < W; ++x) {
      const float* bp = beta + size_t(x) * D;
      const float* sp = s + size_t(x) * D;
      float* rp = rho + size_t(x) * D;
      const int lo = k0[x], hi = k1[x];
      if (hi <= lo) continue;
      Top2 t;
      for (int k = lo; k < hi; ++k) t.push(bp[k], k);
      for (int k = lo; k < hi; ++k) {
        const float tgt = sp[k] - std::max(cfg.lambda, t.excl(k));
        rp[k] = keep * tgt + cfg.damping * rp[k];
      }
    }
    // beta_ij = s - max(gamma, max over this RIGHT pixel's other claimants of rho)
    //
    // Claimants of right pixel xr are the left pixels xr+d, so the walk is
    // index (xr+dmin)*D + k*(D+1): stride D+1 floats, a cache miss per element.
    // Gathering the diagonal into a contiguous scratch buffer first turns two
    // strided passes (top-2, then update) into one strided gather plus two
    // contiguous passes, and the scratch is 60 floats -- one cache line's worth
    // of work per pixel instead of sixty misses.
    for (int xr = 0; xr < W; ++xr) {
      const int kmax = std::min(D, W - xr - dmin);
      if (kmax <= 0) break;
      const size_t base = (size_t(xr) + size_t(dmin)) * D;
      const size_t step = size_t(D) + 1;
      Top2 t;
      int nv = 0;
      for (int k = 0; k < kmax; ++k) {
        const int x = xr + dmin + k;
        if (k < k0[x] || k >= k1[x]) { idxs[k] = -1; continue; }
        const size_t id2 = base + size_t(k) * step;
        gath[k] = rho[id2]; svals[k] = s[id2]; idxs[k] = 1;
        t.push(gath[k], k); ++nv;
      }
      if (nv == 0) continue;
      for (int k = 0; k < kmax; ++k) {
        if (idxs[k] < 0) continue;
        const size_t id2 = base + size_t(k) * step;
        beta[id2] = keep * (svals[k] - std::max(cfg.gamma, t.excl(k)))
                    + cfg.damping * beta[id2];
      }
    }
  }

  // --- decision: best belief per left pixel, then uniqueness by greedy claim ---
  for (int x = 0; x < W; ++x) {
    best[x] = -1e30f; bestk[x] = -1; taken[x] = 0; order[x] = x;
    const float* sp = s + size_t(x) * D;
    for (int k = k0[x]; k < k1[x]; ++k) {
      if (sp[k] <= cfg.lambda) continue;
      const float bel = beta[size_t(x) * D + k] + rho[size_t(x) * D + k] - sp[k];
      if (bel > best[x]) { best[x] = bel; bestk[x] = k; }
    }
  }
  std::sort(order, order + W, [&](int a, int b) { return best[a] > best[b]; });
  for (int oi = 0; oi < W; ++oi) {
    const int x = order[oi];
    out_disp[x] = std::nanf("");
    if (bestk[x] < 0) continue;
    if (cfg.min_margin > 0.f && out_margin[x] < cfg.min_margin) continue;
    const int k = bestk[x];
    const int d = cfg.dmin + k;
    const int xr = x - d;
    if (xr < 0 || taken[xr]) continue;
    taken[xr] = 1;

    // Sub-pixel by parabola through the cost at k-1, k, k+1. The output was
    // integer disparity, which forfeits up to half a pixel before any matching
    // error -- and the accuracy metric has a one-pixel threshold, so that
    // quantisation is charged directly against the score. SGM interpolates to
    // 1/16 px for exactly this reason.
    //
    // Clamped to +-0.5 and skipped where the three samples are not a maximum.
    //
    // OFF BY DEFAULT: measured, it makes things slightly WORSE -- 8.8% bad
    // against 8.6% at matched coverage. The window-aggregated Census cost is a
    // mean of 49-level Hamming scores over a 7x7 block, which is not locally
    // parabolic enough for a three-point fit to find a real vertex. The theory
    // that integer quantisation was costing accuracy was reasonable and wrong.
    float off = 0.f;
    const float* sp2 = s + size_t(x) * D;
    if (cfg.subpixel && k > k0[x] && k + 1 < k1[x]) {
      const float cm = sp2[k - 1], c0 = sp2[k], cp = sp2[k + 1];
      const float den = 2.f * c0 - cm - cp;
      if (den > 1e-6f) {
        off = 0.5f * (cp - cm) / den;
        if (off > 0.5f) off = 0.5f;
        if (off < -0.5f) off = -0.5f;
      }
    }
    out_disp[x] = float(d) + off;
  }
  (void)y;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
        "usage: %s LEFT.y8 RIGHT.y8 W H [--dmax N] [--iters N] [--threads N]\n"
        "          [--min-margin F] [--out disp.f32]\n", argv[0]);
    return 2;
  }
  const std::string lp = argv[1], rp = argv[2];
  const int W = std::atoi(argv[3]), H = std::atoi(argv[4]);
  Cfg cfg;
  std::string outp;
  for (int i = 5; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has = i + 1 < argc;
    if (a == "--dmax" && has) cfg.dmax = std::atoi(argv[++i]);
    else if (a == "--iters" && has) cfg.iters = std::atoi(argv[++i]);
    else if (a == "--agg" && has) cfg.agg = std::atoi(argv[++i]);
    else if (a == "--subpixel") cfg.subpixel = true;
    else if (a == "--threads" && has) cfg.threads = std::atoi(argv[++i]);
    else if (a == "--min-margin" && has) cfg.min_margin = float(std::atof(argv[++i]));
    else if (a == "--out" && has) outp = argv[++i];
  }
  Image8 L, R;
  if (!load_raw_y8(lp, W, H, &L) || !load_raw_y8(rp, W, H, &R)) {
    std::fprintf(stderr, "failed to load %s / %s at %dx%d\n",
                 lp.c_str(), rp.c_str(), W, H);
    return 1;
  }
  const int D = cfg.dmax - cfg.dmin + 1;
  int nthreads = cfg.threads > 0 ? cfg.threads
                                 : int(std::thread::hardware_concurrency());
  if (nthreads < 1) nthreads = 1;

  const double t0 = now_ms();
  const std::vector<uint64_t> cl = census_image(L, 3, 3);
  const std::vector<uint64_t> cr = census_image(R, 3, 3);
  const double t_census = now_ms() - t0;

  // --- cost volume, aggregated over a window ---
  //
  // A single Census comparison is 48 binary tests quantised to 49 Hamming levels.
  // Measured against SGM, that -- not the absence of a smoothness prior -- is
  // where the accuracy goes: SGM stripped of smoothness AND post-filtering still
  // reaches 12.7% bad against this matcher's 28.1%, and its cost is an absolute
  // difference aggregated over a 5x5 block.
  //
  // So aggregate. Summing the score over a window at fixed disparity is the same
  // trick, costs two separable passes, and turns 49 levels into a graded score.
  // Built disparity-major so each slice is a contiguous H*W plane the box filter
  // can walk, then transposed once into the [x][k] layout the solver wants.
  const double tc0 = now_ms();
  std::vector<float> vol(size_t(W) * H * D, -1e30f);
  {
    std::vector<float> slice(size_t(W) * H), tmp(size_t(W) * H);
    const int r = std::max(0, cfg.agg);
    for (int k = 0; k < D; ++k) {
      const int d = cfg.dmin + k;
      std::fill(slice.begin(), slice.end(), 0.f);
      for (int y = 3; y < H - 3; ++y)
        for (int x = 3 + d; x < W - 3; ++x)
          slice[size_t(y) * W + x] =
              (24.f - float(popcnt64(cl[size_t(y) * W + x] ^
                                     cr[size_t(y) * W + x - d]))) / 24.f;
      if (r > 0) {
        // separable box, running sum
        for (int y = 0; y < H; ++y) {
          float acc = 0.f;
          const float* in = &slice[size_t(y) * W];
          float* out = &tmp[size_t(y) * W];
          for (int x = 0; x < W; ++x) {
            acc += in[x];
            if (x > 2 * r) acc -= in[x - 2 * r - 1];
            out[std::max(0, x - r)] = acc;
          }
        }
        for (int x = 0; x < W; ++x) {
          float acc = 0.f;
          for (int y = 0; y < H; ++y) {
            acc += tmp[size_t(y) * W + x];
            if (y > 2 * r) acc -= tmp[size_t(y - 2 * r - 1) * W + x];
            slice[size_t(std::max(0, y - r)) * W + x] = acc;
          }
        }
      }
      const float norm = 1.f / float((2 * r + 1) * (2 * r + 1));
      for (int y = 3; y < H - 3; ++y)
        for (int x = 3 + d; x < W - 3; ++x)
          vol[(size_t(y) * W + x) * D + k] = slice[size_t(y) * W + x] * norm;
    }
  }
  const double t_cost = now_ms() - tc0;

  std::vector<float> disp(size_t(W) * H, std::nanf(""));
  std::vector<float> margin(size_t(W) * H, 0.f);

  const double t1 = now_ms();
  std::vector<std::thread> pool;
  for (int t = 0; t < nthreads; ++t) {
    pool.emplace_back([&, t]() {
      // Per-thread scratch, allocated once: three W*D float planes.
      std::vector<float> s(size_t(W) * D), beta(size_t(W) * D), rho(size_t(W) * D);
      std::vector<float> gath(size_t(D) + 2), svals(size_t(D) + 2);
      std::vector<int> idxs(size_t(D) + 2), k0(W), k1(W), bestk(W), order(W);
      std::vector<float> best(W);
      std::vector<char> taken(W);
      for (int y = t; y < H; y += nthreads) {
        solve_row(y, W, D, cfg, &vol[size_t(y) * W * D],
                  s.data(), beta.data(), rho.data(),
                  &disp[size_t(y) * W], &margin[size_t(y) * W], 3,
                  gath.data(), svals.data(), idxs.data(), k0.data(), k1.data(),
                  best.data(), bestk.data(), order.data(), taken.data());
      }
    });
  }
  for (auto& th : pool) th.join();
  const double t_solve = now_ms() - t1;

  size_t filled = 0;
  for (float v : disp) if (std::isfinite(v)) ++filled;
  std::printf("%dx%d  D=%d  iters=%d  threads=%d\n", W, H, D, cfg.iters, nthreads);
  std::printf("census %.1f ms  cost %.1f ms  solve %.1f ms  total %.1f ms  "
              "agg=%d iters=%d  filled %.1f%%\n",
              t_census, t_cost, t_solve, t_census + t_cost + t_solve,
              cfg.agg, cfg.iters,
              100.0 * double(filled) / double(disp.size()));
  if (!outp.empty()) {
    FILE* f = std::fopen(outp.c_str(), "wb");
    if (f) { std::fwrite(disp.data(), sizeof(float), disp.size(), f); std::fclose(f); }
  }
  return 0;
}
