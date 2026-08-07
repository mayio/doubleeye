#include "doubleeye/preproc.hpp"

#include <algorithm>
#include <time.h>

#include <cmath>
#include <cstdio>

namespace doubleeye {
namespace {

// Box-sum of a float plane over a clamped (2r+1)^2 window, separably.
//
// Deliberately NOT an integral image. An integral image needs a double
// accumulator (a float mantissa cannot hold the running total of squared Sobel
// products across a 848x480 plane) and touches an extra 3.3 MB buffer per call.
// On a board whose actual bottleneck is memory bandwidth -- 58.4 GB/s shared
// between CPU and GPU, per the plan -- that is the wrong trade for the r=1
// windows this code actually uses.
//
// Two 1D passes with a running sum instead. Composing two clamped 1D sums gives
// exactly the clamped 2D sum, so edge behaviour is unchanged, and float is safe
// because a local window total never exceeds a few times 1e7.
std::vector<float> box_sum(const std::vector<float>& src, int w, int h, int r) {
  std::vector<float> tmp(size_t(w) * h, 0.f);
  for (int y = 0; y < h; ++y) {
    const float* in = &src[size_t(y) * w];
    float* out = &tmp[size_t(y) * w];
    float run = 0.f;
    const int first = std::min(r, w - 1);
    for (int x = 0; x <= first; ++x) run += in[x];
    out[0] = run;
    for (int x = 1; x < w; ++x) {
      const int add = x + r, drop = x - r - 1;
      if (add < w) run += in[add];
      if (drop >= 0) run -= in[drop];
      out[x] = run;
    }
  }

  std::vector<float> res(size_t(w) * h, 0.f);
  std::vector<float> run(size_t(w), 0.f);
  const int first_row = std::min(r, h - 1);
  for (int y = 0; y <= first_row; ++y)
    for (int x = 0; x < w; ++x) run[x] += tmp[size_t(y) * w + x];
  for (int x = 0; x < w; ++x) res[x] = run[x];
  for (int y = 1; y < h; ++y) {
    const int add = y + r, drop = y - r - 1;
    for (int x = 0; x < w; ++x) {
      if (add < h) run[x] += tmp[size_t(add) * w + x];
      if (drop >= 0) run[x] -= tmp[size_t(drop) * w + x];
      res[size_t(y) * w + x] = run[x];
    }
  }
  return res;
}

}  // namespace

// ---------------------------------------------------------------------------

bool load_raw_y8(const std::string& path, int width, int height, Image8* out) {
  if (!out || width <= 0 || height <= 0) return false;
  FILE* fh = std::fopen(path.c_str(), "rb");
  if (!fh) return false;
  *out = Image8(width, height);
  const size_t want = out->pixels();
  const size_t got = std::fread(out->data.data(), 1, want, fh);
  // A short read means the geometry does not match the file, which silently
  // produces garbage further down. Fail instead.
  bool ok = (got == want) && (std::fgetc(fh) == EOF);
  std::fclose(fh);
  if (!ok) *out = Image8();
  return ok;
}

bool save_pgm(const std::string& path, const Image8& img) {
  if (img.empty()) return false;
  FILE* fh = std::fopen(path.c_str(), "wb");
  if (!fh) return false;
  std::fprintf(fh, "P5\n%d %d\n255\n", img.width, img.height);
  const size_t n = std::fwrite(img.data.data(), 1, img.pixels(), fh);
  std::fclose(fh);
  return n == img.pixels();
}

Image8 downsample(const Image8& img, int factor) {
  if (img.empty() || factor < 1) return Image8();
  if (factor == 1) return img;
  const int w = img.width / factor, h = img.height / factor;
  if (w < 1 || h < 1) return Image8();
  Image8 out(w, h);
  const int n = factor * factor;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      unsigned sum = 0;
      for (int dy = 0; dy < factor; ++dy) {
        const uint8_t* row = &img.data[size_t(y * factor + dy) * img.width
                                      + x * factor];
        for (int dx = 0; dx < factor; ++dx) sum += row[dx];
      }
      out.at(x, y) = uint8_t((sum + n / 2) / n);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------

CensusImage census_transform(const Image8& img, CensusConfig cfg) {
  CensusImage out;
  out.width = img.width;
  out.height = img.height;
  out.config = cfg;
  if (img.empty()) return out;
  out.bits.assign(img.pixels(), 0ull);
  out.valid.assign(img.pixels(), 0);
  if (cfg.bits() > 64) return out;  // would not fit a uint64

  for (int y = cfg.half_h; y < img.height - cfg.half_h; ++y) {
    for (int x = cfg.half_w; x < img.width - cfg.half_w; ++x) {
      const uint8_t centre = img.at(x, y);
      uint64_t bits = 0ull;
      int bit = 0;
      for (int dy = -cfg.half_h; dy <= cfg.half_h; ++dy) {
        for (int dx = -cfg.half_w; dx <= cfg.half_w; ++dx) {
          if (dx == 0 && dy == 0) continue;
          if (img.at(x + dx, y + dy) < centre) bits |= (1ull << bit);
          ++bit;
        }
      }
      const size_t i = size_t(y) * img.width + x;
      out.bits[i] = bits;
      out.valid[i] = 1;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------

uint64_t census_at(const Image8& img, int x, int y, CensusConfig cfg,
                   bool* ok) {
  const bool inside = x >= cfg.half_w && y >= cfg.half_h &&
                      x < img.width - cfg.half_w &&
                      y < img.height - cfg.half_h;
  if (ok) *ok = inside;
  if (!inside || cfg.bits() > 64) return 0ull;

  const uint8_t centre = img.at(x, y);
  uint64_t bits = 0ull;
  int bit = 0;
  for (int dy = -cfg.half_h; dy <= cfg.half_h; ++dy) {
    const uint8_t* row = &img.data[size_t(y + dy) * img.width + x];
    for (int dx = -cfg.half_w; dx <= cfg.half_w; ++dx) {
      if (dx == 0 && dy == 0) continue;
      if (row[dx] < centre) bits |= (1ull << bit);
      ++bit;
    }
  }
  return bits;
}

float local_std_at(const Image8& img, int x, int y, int window) {
  if (img.empty() || window < 2) return 0.f;
  const int r = window / 2;
  const int x0 = std::max(0, x - r), x1 = std::min(img.width - 1, x + r);
  const int y0 = std::max(0, y - r), y1 = std::min(img.height - 1, y + r);
  double s1 = 0.0, s2 = 0.0;
  for (int yy = y0; yy <= y1; ++yy) {
    const uint8_t* row = &img.data[size_t(yy) * img.width];
    for (int xx = x0; xx <= x1; ++xx) {
      const double v = row[xx];
      s1 += v;
      s2 += v * v;
    }
  }
  const double n = double((y1 - y0 + 1) * (x1 - x0 + 1));
  const double mean = s1 / n;
  const double var = s2 / n - mean * mean;
  return var > 0.0 ? float(std::sqrt(var)) : 0.f;
}

std::vector<float> local_std(const Image8& img, int window) {
  const int w = img.width, h = img.height;
  std::vector<float> out(img.pixels(), 0.f);
  if (img.empty() || window < 2) return out;
  const int r = window / 2;

  std::vector<float> v(img.pixels()), v2(img.pixels());
  for (size_t i = 0; i < img.pixels(); ++i) {
    const float p = static_cast<float>(img.data[i]);
    v[i] = p;
    v2[i] = p * p;
  }
  const std::vector<float> s1 = box_sum(v, w, h, r);
  const std::vector<float> s2 = box_sum(v2, w, h, r);

  for (int y = 0; y < h; ++y) {
    const int y0 = std::max(0, y - r), y1 = std::min(h - 1, y + r);
    for (int x = 0; x < w; ++x) {
      const int x0 = std::max(0, x - r), x1 = std::min(w - 1, x + r);
      // Actual window area, so edge pixels are not biased by a clipped window.
      const float n = float((y1 - y0 + 1) * (x1 - x0 + 1));
      const size_t i = size_t(y) * w + x;
      const float mean = s1[i] / n;
      const float var = s2[i] / n - mean * mean;
      out[i] = var > 0.f ? std::sqrt(var) : 0.f;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------

std::vector<float> shi_tomasi_response(const Image8& img, int grad_window) {
  const int w = img.width, h = img.height;
  std::vector<float> out(img.pixels(), 0.f);
  if (w < 3 || h < 3) return out;

  std::vector<float> ixx(img.pixels(), 0.f), iyy(img.pixels(), 0.f),
      ixy(img.pixels(), 0.f);
  // Row pointers rather than img.at(x, y). The readable form calls at() twelve
  // times per pixel and each call recomputes y * width + x, which is 12
  // multiplies per pixel purely to address memory the compiler cannot prove is
  // contiguous. Hoisting the three rows also gives the vectoriser something it
  // can actually work with.
  for (int y = 1; y < h - 1; ++y) {
    const uint8_t* rm = &img.data[size_t(y - 1) * w];
    const uint8_t* r0 = &img.data[size_t(y) * w];
    const uint8_t* rp = &img.data[size_t(y + 1) * w];
    float* oxx = &ixx[size_t(y) * w];
    float* oyy = &iyy[size_t(y) * w];
    float* oxy = &ixy[size_t(y) * w];
    for (int x = 1; x < w - 1; ++x) {
      // Sobel 3x3, scaled by 1/8 so the response stays in intensity units and
      // min_response thresholds transfer between scenes.
      const float gx = (float(rm[x + 1]) - float(rm[x - 1])
                        + 2.f * (float(r0[x + 1]) - float(r0[x - 1]))
                        + float(rp[x + 1]) - float(rp[x - 1])) * 0.125f;
      const float gy = (float(rp[x - 1]) - float(rm[x - 1])
                        + 2.f * (float(rp[x]) - float(rm[x]))
                        + float(rp[x + 1]) - float(rm[x + 1])) * 0.125f;
      oxx[x] = gx * gx;
      oyy[x] = gy * gy;
      oxy[x] = gx * gy;
    }
  }

  const int r = std::max(1, grad_window / 2);
  const std::vector<float> sxx = box_sum(ixx, w, h, r);
  const std::vector<float> syy = box_sum(iyy, w, h, r);
  const std::vector<float> sxy = box_sum(ixy, w, h, r);
  const float norm = 1.f / float((2 * r + 1) * (2 * r + 1));

  for (size_t i = 0; i < img.pixels(); ++i) {
    const float a = sxx[i] * norm, b = syy[i] * norm, c = sxy[i] * norm;
    const float half_tr = 0.5f * (a + b);
    const float disc = std::sqrt(std::max(0.f, 0.25f * (a - b) * (a - b) + c * c));
    out[i] = half_tr - disc;  // smaller eigenvalue
  }
  return out;
}

std::vector<Keypoint> detect_keypoints(const Image8& img,
                                       const DetectorConfig& cfg) {
  std::vector<Keypoint> result;
  if (img.empty()) return result;
  const int w = img.width, h = img.height;

  // Response must be dense: non-maximum suppression compares neighbours, so
  // there is no way to know which pixels matter without computing all of them.
  // The texture test does not have that constraint and is evaluated sparsely
  // below, once a pixel has already survived thresholding and NMS.
  const std::vector<float> resp = shi_tomasi_response(img, cfg.grad_window);
  const int kTextureWindow = 7;

  const int border = std::max(cfg.border, 2);
  const int cell = std::max(4, cfg.cell);
  const int cols = (w + cell - 1) / cell;
  const int rows = (h + cell - 1) / cell;
  std::vector<std::vector<Keypoint>> buckets(size_t(cols) * rows);

  for (int y = border; y < h - border; ++y) {
    for (int x = border; x < w - border; ++x) {
      const size_t i = size_t(y) * w + x;
      const float r = resp[i];
      if (r < cfg.min_response) continue;

      // 3x3 non-maximum suppression. Strictly greater to the left/up and
      // greater-or-equal to the right/down, so a plateau yields exactly one
      // keypoint rather than none or several.
      bool is_max = true;
      for (int dy = -1; dy <= 1 && is_max; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0) continue;
          const float n = resp[size_t(y + dy) * w + (x + dx)];
          const bool before = (dy < 0) || (dy == 0 && dx < 0);
          if (before ? (n >= r) : (n > r)) { is_max = false; break; }
        }
      }
      if (!is_max) continue;

      // Cheap to reach here (a few thousand pixels per frame), so the texture
      // test costs almost nothing evaluated pointwise.
      const float texture = local_std_at(img, x, y, kTextureWindow);
      if (texture < cfg.min_local_std) continue;

      Keypoint kp;
      kp.x = float(x);
      kp.y = float(y);
      kp.response = r;
      kp.local_std = texture;

      if (cfg.subpixel) {
        // Parabola through the three response samples in each axis. Depth
        // accuracy hinges on sub-pixel precision, so this is not optional:
        // at 2 m, 0.1 px of disparity is ~2 cm of range.
        const float l = resp[i - 1], c = r, rr = resp[i + 1];
        const float dxx = l - 2.f * c + rr;
        if (std::fabs(dxx) > 1e-12f) {
          const float dx = 0.5f * (l - rr) / dxx;
          if (std::fabs(dx) <= 0.5f) kp.x += dx;
        }
        const float u = resp[i - w], d = resp[i + w];
        const float dyy = u - 2.f * c + d;
        if (std::fabs(dyy) > 1e-12f) {
          const float dy = 0.5f * (u - d) / dyy;
          if (std::fabs(dy) <= 0.5f) kp.y += dy;
        }
      }

      buckets[size_t(y / cell) * cols + (x / cell)].push_back(kp);
    }
  }

  const size_t keep = size_t(std::max(1, cfg.per_cell));
  for (auto& bucket : buckets) {
    if (bucket.size() > keep) {
      std::partial_sort(bucket.begin(), bucket.begin() + keep, bucket.end(),
                        [](const Keypoint& a, const Keypoint& b) {
                          return a.response > b.response;
                        });
      bucket.resize(keep);
    }
    result.insert(result.end(), bucket.begin(), bucket.end());
  }
  return result;
}

// ---------------------------------------------------------------------------
// FAST

namespace {

// Bresenham circle of radius 3, starting at the top and going clockwise. Order
// matters: contiguity is defined around this ring.
const int kCircleX[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
const int kCircleY[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};

// Is there a circular run of at least n set bits in the low 16 bits?
inline bool circular_run(unsigned mask, int n) {
  const unsigned doubled = mask | (mask << 16);
  const unsigned want = (n >= 32) ? 0xFFFFFFFFu : ((1u << n) - 1u);
  for (int i = 0; i < 16; ++i) {
    if (((doubled >> i) & want) == want) return true;
  }
  return false;
}

}  // namespace

bool is_fast_corner(const Image8& img, int x, int y, int threshold) {
  if (x < 3 || y < 3 || x >= img.width - 3 || y >= img.height - 3) return false;
  const int centre = img.at(x, y);
  const int hi = centre + threshold;
  const int lo = centre - threshold;

  // High-speed rejection on the four compass points, which are 4 apart on the
  // ring. A run of 9 consecutive indices out of 16 must contain at least TWO of
  // them -- the worst case, indices 1..9, contains only 4 and 8. Requiring three
  // is the correct test for FAST-12, not FAST-9, and using it here silently
  // discarded real corners: a square's corner pixel has exactly two dark compass
  // points, so every one of them was rejected.
  int bright = 0, dark = 0;
  for (int k = 0; k < 16; k += 4) {
    const int v = img.at(x + kCircleX[k], y + kCircleY[k]);
    if (v > hi) ++bright;
    else if (v < lo) ++dark;
  }
  if (bright < 2 && dark < 2) return false;

  unsigned bright_mask = 0, dark_mask = 0;
  for (int k = 0; k < 16; ++k) {
    const int v = img.at(x + kCircleX[k], y + kCircleY[k]);
    if (v > hi) bright_mask |= (1u << k);
    else if (v < lo) dark_mask |= (1u << k);
  }
  return circular_run(bright_mask, 9) || circular_run(dark_mask, 9);
}

float shi_tomasi_at(const Image8& img, int x, int y, int grad_window) {
  const int r = std::max(1, grad_window / 2);
  // Gradients are needed over the window, and each Sobel needs one pixel beyond.
  if (x - r - 1 < 0 || y - r - 1 < 0 ||
      x + r + 1 >= img.width || y + r + 1 >= img.height)
    return 0.f;

  float sxx = 0.f, syy = 0.f, sxy = 0.f;
  for (int dy = -r; dy <= r; ++dy) {
    for (int dx = -r; dx <= r; ++dx) {
      const int px = x + dx, py = y + dy;
      const uint8_t* rm = &img.data[size_t(py - 1) * img.width];
      const uint8_t* r0 = &img.data[size_t(py) * img.width];
      const uint8_t* rp = &img.data[size_t(py + 1) * img.width];
      const float gx = (float(rm[px + 1]) - float(rm[px - 1])
                        + 2.f * (float(r0[px + 1]) - float(r0[px - 1]))
                        + float(rp[px + 1]) - float(rp[px - 1])) * 0.125f;
      const float gy = (float(rp[px - 1]) - float(rm[px - 1])
                        + 2.f * (float(rp[px]) - float(rm[px]))
                        + float(rp[px + 1]) - float(rm[px + 1])) * 0.125f;
      sxx += gx * gx;
      syy += gy * gy;
      sxy += gx * gy;
    }
  }
  const float n = 1.f / float((2 * r + 1) * (2 * r + 1));
  const float a = sxx * n, b = syy * n, c = sxy * n;
  const float half_tr = 0.5f * (a + b);
  const float disc = std::sqrt(std::max(0.f, 0.25f * (a - b) * (a - b) + c * c));
  return half_tr - disc;
}

namespace {
double prof_now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) * 1e3 + double(ts.tv_nsec) / 1e6;
}
}  // namespace

std::vector<Keypoint> detect_keypoints_fast(const Image8& img,
                                            const DetectorConfig& cfg,
                                            DetectProfile* prof) {
  std::vector<Keypoint> result;
  if (img.empty()) return result;
  const double t_start = prof ? prof_now_ms() : 0.0;
  const int w = img.width, h = img.height;
  const int border = std::max(cfg.border, 5);  // FAST needs 3, Sobel one more

  // 1. FAST nominates candidates, densely but cheaply.
  std::vector<Keypoint> cand;
  cand.reserve(4096);
  for (int y = border; y < h - border; ++y) {
    for (int x = border; x < w - border; ++x) {
      if (!is_fast_corner(img, x, y, cfg.fast_threshold)) continue;
      Keypoint kp;
      kp.x = float(x);
      kp.y = float(y);
      kp.response = shi_tomasi_at(img, x, y, cfg.grad_window);
      if (kp.response < cfg.min_response) continue;
      cand.push_back(kp);
    }
  }
  if (cand.empty()) return result;

  const double t_fast = prof ? prof_now_ms() : 0.0;
  if (prof) { prof->fast_ms = t_fast - t_start; prof->candidates = int(cand.size()); }

  // 2. Suppress within nms_radius, strongest first. Sparse candidates make this
  // a sort plus an occupancy stamp rather than a neighbourhood scan per pixel.
  std::sort(cand.begin(), cand.end(),
            [](const Keypoint& a, const Keypoint& b) {
              return a.response > b.response;
            });
  const int rad = std::max(1, cfg.nms_radius);
  std::vector<uint8_t> taken(img.pixels(), 0);
  std::vector<Keypoint> kept;
  kept.reserve(cand.size());
  for (const Keypoint& kp : cand) {
    const int xi = int(kp.x), yi = int(kp.y);
    bool blocked = false;
    for (int dy = -rad; dy <= rad && !blocked; ++dy) {
      const int py = yi + dy;
      if (py < 0 || py >= h) continue;
      for (int dx = -rad; dx <= rad; ++dx) {
        const int px = xi + dx;
        if (px < 0 || px >= w) continue;
        if (taken[size_t(py) * w + px]) { blocked = true; break; }
      }
    }
    if (blocked) continue;
    taken[size_t(yi) * w + xi] = 1;
    kept.push_back(kp);
  }

  const double t_nms = prof ? prof_now_ms() : 0.0;
  if (prof) { prof->nms_ms = t_nms - t_fast; prof->suppressed = int(kept.size()); }

  // 3. Texture floor, then sub-pixel, then grid bucketing -- all sparse.
  const int cell = std::max(4, cfg.cell);
  const int cols = (w + cell - 1) / cell;
  const int rows = (h + cell - 1) / cell;
  std::vector<std::vector<Keypoint>> buckets(size_t(cols) * rows);
  for (Keypoint kp : kept) {
    const int xi = int(kp.x), yi = int(kp.y);
    kp.local_std = local_std_at(img, xi, yi, 7);
    if (kp.local_std < cfg.min_local_std) continue;

    if (cfg.subpixel) {
      // Parabola through three sparse response samples per axis.
      const float c = kp.response;
      const float l = shi_tomasi_at(img, xi - 1, yi, cfg.grad_window);
      const float r = shi_tomasi_at(img, xi + 1, yi, cfg.grad_window);
      const float dxx = l - 2.f * c + r;
      if (std::fabs(dxx) > 1e-12f) {
        const float d = 0.5f * (l - r) / dxx;
        if (std::fabs(d) <= 0.5f) kp.x += d;
      }
      const float u = shi_tomasi_at(img, xi, yi - 1, cfg.grad_window);
      const float dn = shi_tomasi_at(img, xi, yi + 1, cfg.grad_window);
      const float dyy = u - 2.f * c + dn;
      if (std::fabs(dyy) > 1e-12f) {
        const float d = 0.5f * (u - dn) / dyy;
        if (std::fabs(d) <= 0.5f) kp.y += d;
      }
    }
    buckets[size_t(yi / cell) * cols + (xi / cell)].push_back(kp);
  }

  const size_t keep = size_t(std::max(1, cfg.per_cell));
  for (auto& bucket : buckets) {
    if (bucket.size() > keep) {
      std::partial_sort(bucket.begin(), bucket.begin() + keep, bucket.end(),
                        [](const Keypoint& a, const Keypoint& b) {
                          return a.response > b.response;
                        });
      bucket.resize(keep);
    }
    result.insert(result.end(), bucket.begin(), bucket.end());
  }
  if (prof) prof->refine_ms = prof_now_ms() - t_start - prof->fast_ms
                             - prof->nms_ms;
  return result;
}

// ---------------------------------------------------------------------------

float cell_occupancy(const std::vector<Keypoint>& kps, int width, int height,
                     int cell) {
  if (width <= 0 || height <= 0) return 0.f;
  cell = std::max(4, cell);
  const int cols = (width + cell - 1) / cell;
  const int rows = (height + cell - 1) / cell;
  std::vector<uint8_t> hit(size_t(cols) * rows, 0);
  for (const Keypoint& kp : kps) {
    const int cx = int(kp.x) / cell, cy = int(kp.y) / cell;
    if (cx >= 0 && cx < cols && cy >= 0 && cy < rows)
      hit[size_t(cy) * cols + cx] = 1;
  }
  size_t n = 0;
  for (uint8_t v : hit) n += v;
  return float(n) / float(hit.size());
}

}  // namespace doubleeye
