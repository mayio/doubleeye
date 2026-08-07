// Stream IR frames to stdout so the desktop can show a live view.
//
// This exists for one reason: you cannot aim a calibration target at an IR
// camera you cannot see. Holding a checkerboard in front of a blind sensor and
// hoping the poses were varied enough is not a workable procedure, and the
// failure only shows up after the session.
//
// Measured link throughput to the desktop is ~22 MB/s over WiFi, so full
// 848x480 Y8 on both channels at 10 Hz (8 MB/s) fits comfortably. No compression
// and no encoding: raw bytes down a pipe is the least that can go wrong, and the
// TX2 has no cycles to spare for encoding anyway.
//
// Unlike rs_ir_capture this DOES use rs2::pipeline. For a preview the syncer's
// frameset pairing is exactly what is wanted -- matched left/right for display.
// The capture tool avoids it because the syncer discards frames, which is the
// signal that tool measures; here it is a feature.
//
// Protocol, little-endian, one packet per frame:
//   magic "DEIR"  4 bytes
//   width         uint16
//   height        uint16
//   stream_index  uint8   (1 or 2)
//   flags         uint8   (reserved, 0)
//   frame_number  uint32
//   payload       width * height bytes of Y8
//
// Diagnostics go to stderr so they cannot corrupt the stream.

#include "rs_common.hpp"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace doubleeye;

namespace {

struct Options {
  int width = kWidth;
  int height = kHeight;
  int fps = 30;        // sensor rate
  double out_fps = 10; // rate actually emitted
  int exposure_us = 1500;
  int gain = 96;       // higher default than capture: preview is often emitter-off
  std::string emitter = "off";  // calibration wants the projector off
  bool both = true;
};

void usage(const char* argv0) {
  std::fprintf(stderr,
      "usage: %s [options]   (binary stream on stdout, logs on stderr)\n"
      "  --out-fps F      frames/s emitted per channel (default 10)\n"
      "  --exposure-us N  (default 1500)\n"
      "  --gain N         (default 96)\n"
      "  --emitter MODE   on | off   (default off, for calibration)\n"
      "  --single         send only ir1\n"
      "  --width/--height/--fps\n", argv0);
}

bool write_all(const void* p, size_t n) {
  const char* c = static_cast<const char*>(p);
  while (n) {
    const ssize_t w = ::write(STDOUT_FILENO, c, n);
    if (w <= 0) return false;
    c += w;
    n -= size_t(w);
  }
  return true;
}

bool send_frame(const rs2::video_frame& f, int index) {
  unsigned char hdr[14];
  std::memcpy(hdr, "DEIR", 4);
  const uint16_t w = uint16_t(f.get_width()), h = uint16_t(f.get_height());
  std::memcpy(hdr + 4, &w, 2);
  std::memcpy(hdr + 6, &h, 2);
  hdr[8] = static_cast<unsigned char>(index);
  hdr[9] = 0;
  const uint32_t num = uint32_t(f.get_frame_number());
  std::memcpy(hdr + 10, &num, 4);
  if (!write_all(hdr, sizeof(hdr))) return false;
  return write_all(f.get_data(), size_t(w) * h);
}

void set_option(rs2::sensor& s, rs2_option opt, float v) {
  if (!s.supports(opt)) return;
  rs2::option_range r = s.get_option_range(opt);
  const float c = v < r.min ? r.min : (v > r.max ? r.max : v);
  try {
    s.set_option(opt, c);
    std::fprintf(stderr, "  %-24s = %g\n", rs2_option_to_string(opt), c);
  } catch (const rs2::error& e) {
    std::fprintf(stderr, "  %-24s set failed: %s\n", rs2_option_to_string(opt),
                 e.what());
  }
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has = (i + 1 < argc);
    if (a == "--out-fps" && has) o.out_fps = std::atof(argv[++i]);
    else if (a == "--exposure-us" && has) o.exposure_us = std::atoi(argv[++i]);
    else if (a == "--gain" && has) o.gain = std::atoi(argv[++i]);
    else if (a == "--emitter" && has) o.emitter = argv[++i];
    else if (a == "--single") o.both = false;
    else if (a == "--width" && has) o.width = std::atoi(argv[++i]);
    else if (a == "--height" && has) o.height = std::atoi(argv[++i]);
    else if (a == "--fps" && has) o.fps = std::atoi(argv[++i]);
    else { usage(argv[0]); return 2; }
  }

  try {
    report_power_state(read_power_state());

    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    if (devices.size() == 0) {
      std::fprintf(stderr, "No RealSense device found.\n");
      return 1;
    }
    rs2::device dev = devices[0];

    rs2::depth_sensor sensor = dev.first<rs2::depth_sensor>();
    std::fprintf(stderr, "sensor configuration:\n");
    set_option(sensor, RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0.f);
    set_option(sensor, RS2_OPTION_EXPOSURE, float(o.exposure_us));
    set_option(sensor, RS2_OPTION_GAIN, float(o.gain));
    set_option(sensor, RS2_OPTION_EMITTER_ENABLED,
               o.emitter == "on" ? 1.f : 0.f);

    rs2::pipeline pipe(ctx);
    rs2::config cfg;
    cfg.enable_device(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, o.width, o.height,
                      RS2_FORMAT_Y8, o.fps);
    if (o.both)
      cfg.enable_stream(RS2_STREAM_INFRARED, 2, o.width, o.height,
                        RS2_FORMAT_Y8, o.fps);
    pipe.start(cfg);
    std::fprintf(stderr, "streaming %dx%d, %s, emitting %.1f fps/channel\n",
                 o.width, o.height, o.both ? "ir1+ir2" : "ir1 only", o.out_fps);

    const double period = o.out_fps > 0 ? 1000.0 / o.out_fps : 0.0;
    double next = monotonic_seconds() * 1000.0;

    while (true) {
      rs2::frameset fs = pipe.wait_for_frames(5000);
      const double now = monotonic_seconds() * 1000.0;
      if (now < next) continue;   // throttle: drop rather than queue
      next = now + period;

      if (!send_frame(fs.get_infrared_frame(1), 1)) break;
      if (o.both && !send_frame(fs.get_infrared_frame(2), 2)) break;
    }
    pipe.stop();
  } catch (const rs2::error& e) {
    std::fprintf(stderr, "librealsense error in %s: %s\n",
                 e.get_failed_function().c_str(), e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
