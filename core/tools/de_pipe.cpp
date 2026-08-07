// Streaming matcher: DEIR frames in on stdin, matches out on stdout.
//
// This is the live path. rs_ir_stream on the Jetson already emits raw IR frames as
// DEIR packets, so the missing piece was something that consumes them, matches, and
// emits a result compact enough to cross a network. Detection and matching run
// wherever this runs -- put it on the Jetson and the numbers are the real pipeline's;
// put it on the laptop and it is only a viewer.
//
// Deliberately not a ROS node. The capture path stays ROS-free because
// preprocessing is the binding constraint on memory bandwidth, and JetPack 4.2 is
// Ubuntu 18.04 where every ROS 2 release is past end of life. So: compact binary
// here, and a small bridge on the laptop turns it into topics.
//
// Input, per frame (14-byte header, from rs_ir_stream):
//   "DEIR" | width u16 | height u16 | stream u8 (1|2) | flags u8 | frame u32 | Y8
//
// Output, per matched PAIR (16-byte header):
//   "DEMR" | width u16 | height u16 | frame u32 | n_matches u32
//   | width*height bytes of Y8, the LEFT image
//   | n_matches * 16 bytes: xl f32, yl f32, disparity f32, margin f32
//
// Left and right are paired by frame_number. A frame whose partner never arrives is
// dropped rather than held, because holding it would grow a queue forever on a
// stream that has lost sync; the drop count goes to stderr.
//
//   rs_ir_stream --streams both | de_pipe [--right-density N] [--min-margin F]

#include "doubleeye/match.hpp"
#include "doubleeye/preproc.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace doubleeye;

namespace {

bool read_all(void* dst, size_t n) {
  auto* p = static_cast<unsigned char*>(dst);
  while (n > 0) {
    const ssize_t got = ::read(STDIN_FILENO, p, n);
    if (got <= 0) return false;
    p += got;
    n -= size_t(got);
  }
  return true;
}

bool write_all(const void* src, size_t n) {
  const auto* p = static_cast<const unsigned char*>(src);
  while (n > 0) {
    const ssize_t put = ::write(STDOUT_FILENO, p, n);
    if (put <= 0) return false;
    p += put;
    n -= size_t(put);
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  DetectorConfig det;
  MatchConfig cfg;
  cfg.lambda = cfg.gamma = -0.1f;
  cfg.iterations = 20;
  cfg.damping = 0.4f;
  int right_density = 0;
  float min_margin = 0.f;
  bool subpixel = true;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has = i + 1 < argc;
    if (a == "--right-density" && has) right_density = std::atoi(argv[++i]);
    else if (a == "--min-margin" && has) min_margin = float(std::atof(argv[++i]));
    else if (a == "--fast-threshold" && has) det.fast_threshold = std::atoi(argv[++i]);
    else if (a == "--no-subpixel") subpixel = false;
    else if (a == "--max-disparity" && has) cfg.max_disparity = float(std::atof(argv[++i]));
    else if (a == "--min-disparity" && has) cfg.min_disparity = float(std::atof(argv[++i]));
    else if (a == "--cell" && has) det.cell = std::atoi(argv[++i]);
    else if (a == "--per-cell" && has) det.per_cell = std::atoi(argv[++i]);
    else if (a == "--max-candidates" && has) cfg.max_candidates = std::atoi(argv[++i]);
    else {
      std::fprintf(stderr,
          "usage: %s [--right-density N] [--min-margin F] [--fast-threshold N]\n"
          "          [--min-disparity F] [--max-disparity F] [--no-subpixel]\n"
          "          [--cell N] [--per-cell N] [--max-candidates N]\n"
          "\n"
          "--cell and --per-cell set keypoint density: the detector keeps the\n"
          "top per-cell responses in each cell x cell block. The default 32/3 is\n"
          "about 1000 keypoints on an 848x480 frame, which is one per 400 pixels\n"
          "-- enough to feed geometry, far too sparse to LOOK like a depth map.\n"
          "12/2 gives roughly 5x more if you want to see surfaces.\n"
          "\n"
          "The disparity gate matters more than it looks. f*B is 21.48 px*m, so\n"
          "the default [1, 220] px spans 0.10 m to 21.5 m. In a room that admits\n"
          "five times more range than exists, and wrong matches pile up against\n"
          "both limits: a quarter of them come out nearer than 0.19 m. Gate to\n"
          "the depth you actually have.\n"
          "reads DEIR frames on stdin, writes DEMR match packets on stdout\n",
          argv[0]);
      return 2;
    }
  }
  const float kFB = 430.551f * 0.049883f;   // 21.48 px*m, factory calibration
  std::fprintf(stderr, "de_pipe: cell %d, per_cell %d, fast_threshold %d, right/cell %d, "
               "min_margin %.2f, subpixel %s, disparity [%.1f, %.1f] px "
               "= depth [%.2f, %.2f] m\n", det.cell, det.per_cell, det.fast_threshold,
               right_density > 0 ? right_density : det.per_cell, min_margin,
               subpixel ? "on" : "off", cfg.min_disparity, cfg.max_disparity,
               kFB / cfg.max_disparity, kFB / cfg.min_disparity);

  CensusConfig ccfg;
  DetectorConfig det_r = det;
  if (right_density > 0) det_r.per_cell = right_density;

  // Pending frames waiting for their partner, keyed by frame number.
  struct Pending { Image8 img; int stream; };
  std::map<uint32_t, Pending> pending;
  long pairs = 0, dropped = 0;

  for (;;) {
    unsigned char hdr[14];
    if (!read_all(hdr, sizeof(hdr))) break;
    if (std::memcmp(hdr, "DEIR", 4) != 0) {
      std::fprintf(stderr, "de_pipe: lost sync (bad magic), stopping\n");
      return 1;
    }
    uint16_t w, h;
    uint32_t num;
    std::memcpy(&w, hdr + 4, 2);
    std::memcpy(&h, hdr + 6, 2);
    const int stream = hdr[8];
    std::memcpy(&num, hdr + 10, 4);

    // Braces, not parentheses: Image8 img(int(w), int(h)) is the most vexing
    // parse -- the compiler reads it as a function declaration.
    Image8 img{int(w), int(h)};
    if (!read_all(img.data.data(), img.data.size())) break;

    auto it = pending.find(num);
    if (it == pending.end()) {
      pending[num] = Pending{std::move(img), stream};
      // Anything older than a few frames has lost its partner for good. Holding
      // it would grow this map without bound on a stream that has desynchronised.
      while (pending.size() > 8) {
        pending.erase(pending.begin());
        ++dropped;
      }
      continue;
    }
    if (it->second.stream == stream) {   // duplicate index, not a pair
      it->second.img = std::move(img);
      continue;
    }
    const Image8& left = (stream == 1) ? img : it->second.img;
    const Image8& right = (stream == 1) ? it->second.img : img;

    const std::vector<Keypoint> kl = detect_keypoints_fast(left, det);
    const std::vector<Keypoint> kr = detect_keypoints_fast(right, det_r);
    std::vector<uint64_t> dl(kl.size()), dr(kr.size());
    for (size_t k = 0; k < kl.size(); ++k)
      dl[k] = census_at(left, int(kl[k].x + 0.5f), int(kl[k].y + 0.5f), ccfg);
    for (size_t k = 0; k < kr.size(); ++k)
      dr[k] = census_at(right, int(kr[k].x + 0.5f), int(kr[k].y + 0.5f), ccfg);

    std::vector<Match> m = match_masda(kl, dl, kr, dr, cfg, nullptr);
    if (min_margin > 0.f)
      m.erase(std::remove_if(m.begin(), m.end(),
                             [&](const Match& x) { return x.margin < min_margin; }),
              m.end());
    if (subpixel) refine_disparity(left, right, kl, kr, &m);

    unsigned char out[16];
    std::memcpy(out, "DEMR", 4);
    std::memcpy(out + 4, &w, 2);
    std::memcpy(out + 6, &h, 2);
    std::memcpy(out + 8, &num, 4);
    const uint32_t n = uint32_t(m.size());
    std::memcpy(out + 12, &n, 4);
    if (!write_all(out, sizeof(out))) break;
    if (!write_all(left.data.data(), left.data.size())) break;
    std::vector<float> rec;
    rec.reserve(m.size() * 4);
    for (const Match& x : m) {
      rec.push_back(kl[size_t(x.left)].x);
      rec.push_back(kl[size_t(x.left)].y);
      rec.push_back(x.disparity);
      rec.push_back(x.margin);
    }
    if (!rec.empty() && !write_all(rec.data(), rec.size() * sizeof(float))) break;

    pending.erase(it);
    if (++pairs % 30 == 0)
      std::fprintf(stderr, "de_pipe: %ld pairs, %ld unpaired dropped, "
                   "%zu matches last\n", pairs, dropped, m.size());
  }
  std::fprintf(stderr, "de_pipe: %ld pairs, %ld unpaired dropped\n", pairs, dropped);
  return 0;
}
