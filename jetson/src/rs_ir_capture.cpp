// Bring-up step 1: IR capture with timestamp instrumentation.
//
// Records both D435 IR channels and logs, per frame, every clock the stack
// exposes. The point is not the images -- it is the CSV. Before the IMU can be
// fused against this camera you need to know which timestamp domain is
// trustworthy, and whether frames are being dropped under real load.
//
// Why the low-level sensor API instead of rs2::pipeline:
//   pipeline aggregates streams through a syncer, which *discards* frames it
//   cannot match into a frameset. That discard is precisely the signal being
//   measured, so the pipeline would hide the answer. sensor::open/start
//   delivers every frame exactly as it arrives, unsynced and uncounted, and
//   left/right pairing is then reconstructed offline from frame numbers.
//
// Depends on librealsense2 only. No OpenCV -- raw Y8 goes to disk and is
// decoded on the desktop.

#include "rs_common.hpp"

#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace doubleeye;

namespace {

std::atomic<bool> g_interrupted(false);
void on_sigint(int) { g_interrupted.store(true); }

struct Options {
  std::string outdir;
  double seconds = 60.0;
  int exposure_us = 1500;
  int gain = 64;
  std::string emitter = "on";  // on | off | alternate
  int save_every = 30;
  // Overridable so the delivered-rate problem can be bisected across
  // resolution and rate without editing and rebuilding.
  int width = kWidth;
  int height = kHeight;
  int fps = kFps;
};

struct FrameRec {
  int stream;
  unsigned long long number;
  double ts_ms;
  rs2_timestamp_domain domain;
  double host_monotonic_s;
  double host_realtime_s;
  std::vector<long long> md;
  std::vector<char> md_ok;
};

bool make_dir(const std::string& path) {
  if (mkdir(path.c_str(), 0755) == 0) return true;
  return errno == EEXIST;
}

void usage(const char* argv0) {
  std::printf(
      "usage: %s OUTDIR [options]\n"
      "  --seconds F        recording length (default 60)\n"
      "  --exposure-us N    fixed exposure; 1000-2000 keeps motion blur under\n"
      "                     a pixel at RC speeds (default 1500)\n"
      "  --gain N           raise to compensate the short exposure (default 64)\n"
      "  --emitter MODE     on | off | alternate (default on)\n"
      "  --save-every N     write raw Y8 every Nth frame, 0 disables.\n"
      "                     Saving every frame will itself cause drops.\n"
      "  --width N          (default %d)\n"
      "  --height N         (default %d)\n"
      "  --fps N            (default %d)\n",
      argv0, kWidth, kHeight, kFps);
}

bool parse_args(int argc, char** argv, Options* opt) {
  if (argc < 2) return false;
  opt->outdir = argv[1];
  if (opt->outdir.empty() || opt->outdir[0] == '-') return false;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has_next = (i + 1 < argc);
    if (a == "--seconds" && has_next) {
      opt->seconds = std::atof(argv[++i]);
    } else if (a == "--exposure-us" && has_next) {
      opt->exposure_us = std::atoi(argv[++i]);
    } else if (a == "--gain" && has_next) {
      opt->gain = std::atoi(argv[++i]);
    } else if (a == "--emitter" && has_next) {
      opt->emitter = argv[++i];
      if (opt->emitter != "on" && opt->emitter != "off" &&
          opt->emitter != "alternate") {
        std::fprintf(stderr, "bad --emitter '%s'\n", opt->emitter.c_str());
        return false;
      }
    } else if (a == "--save-every" && has_next) {
      opt->save_every = std::atoi(argv[++i]);
    } else if (a == "--width" && has_next) {
      opt->width = std::atoi(argv[++i]);
    } else if (a == "--height" && has_next) {
      opt->height = std::atoi(argv[++i]);
    } else if (a == "--fps" && has_next) {
      opt->fps = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr, "unknown argument '%s'\n", a.c_str());
      return false;
    }
  }
  return opt->seconds > 0.0;
}

// Set an option, clamping into the advertised range rather than throwing.
bool set_option(rs2::sensor& sensor, rs2_option opt, float value) {
  const char* name = rs2_option_to_string(opt);
  if (!sensor.supports(opt)) {
    std::printf("  %-24s unsupported -- skipped\n", name);
    return false;
  }
  rs2::option_range r = sensor.get_option_range(opt);
  const float clamped = std::min(std::max(value, r.min), r.max);
  if (clamped != value) {
    std::printf("  %-24s %g out of range [%g, %g] -- using %g\n", name, value,
                r.min, r.max, clamped);
  }
  try {
    sensor.set_option(opt, clamped);
  } catch (const rs2::error& e) {
    std::printf("  %-24s set failed: %s\n", name, e.what());
    return false;
  }
  std::printf("  %-24s = %g\n", name, clamped);
  return true;
}

void configure_sensor(rs2::sensor& sensor, const Options& opt) {
  std::printf("sensor configuration:\n");
  // Auto-exposure off first: it overrides any manual exposure written after it.
  set_option(sensor, RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0.0f);
  set_option(sensor, RS2_OPTION_EXPOSURE, static_cast<float>(opt.exposure_us));
  set_option(sensor, RS2_OPTION_GAIN, static_cast<float>(opt.gain));
  if (opt.emitter == "on") {
    set_option(sensor, RS2_OPTION_EMITTER_ENABLED, 1.0f);
  } else if (opt.emitter == "off") {
    set_option(sensor, RS2_OPTION_EMITTER_ENABLED, 0.0f);
  } else {
    set_option(sensor, RS2_OPTION_EMITTER_ENABLED, 1.0f);
    if (!set_option(sensor, RS2_OPTION_EMITTER_ON_OFF, 1.0f)) {
      std::printf("  !! per-frame alternation unavailable on this firmware\n");
    }
  }
}

// Pick the two IR profiles we intend to stream, in index order.
std::vector<rs2::stream_profile> select_profiles(const rs2::sensor& sensor,
                                                 const Options& opt) {
  std::map<int, rs2::stream_profile> by_index;
  for (const rs2::stream_profile& p : sensor.get_stream_profiles()) {
    if (p.stream_type() != RS2_STREAM_INFRARED) continue;
    if (p.format() != RS2_FORMAT_Y8) continue;
    auto vp = p.as<rs2::video_stream_profile>();
    if (!vp) continue;
    if (vp.width() != opt.width || vp.height() != opt.height ||
        vp.fps() != opt.fps)
      continue;
    if (p.stream_index() == 1 || p.stream_index() == 2)
      by_index[p.stream_index()] = p;
  }
  std::vector<rs2::stream_profile> out;
  for (const auto& kv : by_index) out.push_back(kv.second);
  return out;
}

void write_csv(const Options& opt, const std::vector<FrameRec>& rows) {
  const std::string path = opt.outdir + "/frames.csv";
  FILE* fh = std::fopen(path.c_str(), "w");
  if (!fh) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    return;
  }
  std::fprintf(fh, "stream,frame_number,ts_ms,ts_domain,host_monotonic_s,"
                   "host_realtime_s");
  for (const auto& f : metadata_fields()) std::fprintf(fh, ",%s", f.column);
  std::fprintf(fh, "\n");

  for (const FrameRec& r : rows) {
    std::fprintf(fh, "%d,%llu,%.6f,%s,%.9f,%.9f", r.stream, r.number, r.ts_ms,
                 rs2_timestamp_domain_to_string(r.domain), r.host_monotonic_s,
                 r.host_realtime_s);
    for (size_t i = 0; i < r.md.size(); ++i) {
      if (r.md_ok[i]) {
        std::fprintf(fh, ",%lld", r.md[i]);
      } else {
        std::fprintf(fh, ",");
      }
    }
    std::fprintf(fh, "\n");
  }
  std::fclose(fh);
  std::printf("wrote %s (%zu rows)\n", path.c_str(), rows.size());
}

void write_run_meta(const Options& opt, const rs2::device& dev,
                    const std::vector<rs2::stream_profile>& profiles,
                    double duration, const std::string& usb) {
  const std::string path = opt.outdir + "/run.txt";
  FILE* fh = std::fopen(path.c_str(), "w");
  if (!fh) return;

  std::fprintf(fh, "serial %s\n",
               device_info(dev, RS2_CAMERA_INFO_SERIAL_NUMBER).c_str());
  std::fprintf(fh, "firmware %s\n",
               device_info(dev, RS2_CAMERA_INFO_FIRMWARE_VERSION).c_str());
  std::fprintf(fh, "usb %s\n", usb.c_str());
  std::fprintf(fh, "librealsense %d.%d.%d\n", RS2_API_MAJOR_VERSION,
               RS2_API_MINOR_VERSION, RS2_API_PATCH_VERSION);
  std::fprintf(fh, "resolution %dx%d @ %d\n", opt.width, opt.height, opt.fps);
  std::fprintf(fh, "exposure_us %d\n", opt.exposure_us);
  std::fprintf(fh, "gain %d\n", opt.gain);
  std::fprintf(fh, "emitter %s\n", opt.emitter.c_str());
  std::fprintf(fh, "duration_s %.3f\n", duration);
  std::fprintf(fh, "frame_bytes %d\n", opt.width * opt.height);

  if (profiles.size() == 2) {
    auto ir1 = profiles[0].as<rs2::video_stream_profile>();
    auto ir2 = profiles[1].as<rs2::video_stream_profile>();
    rs2_intrinsics in = ir1.get_intrinsics();
    std::fprintf(fh, "fx %.6f\nfy %.6f\ncx %.6f\ncy %.6f\n", in.fx, in.fy,
                 in.ppx, in.ppy);
    try {
      rs2_extrinsics ex = ir1.get_extrinsics_to(ir2);
      std::fprintf(fh, "baseline_m %.9f\n", std::fabs(ex.translation[0]));
    } catch (const rs2::error&) {
      std::fprintf(fh, "baseline_m unavailable\n");
    }
  }
  std::fclose(fh);
}

// Coarse in-situ check. Real analysis is desktop/capture_report.py.
void quick_summary(const std::vector<FrameRec>& rows, double duration) {
  std::printf("\n-- quick summary (full analysis: desktop/capture_report.py) --\n");
  std::set<unsigned long long> numbers[3];
  for (const FrameRec& r : rows) {
    if (r.stream == 1 || r.stream == 2) numbers[r.stream].insert(r.number);
  }
  for (int idx = 1; idx <= 2; ++idx) {
    const std::set<unsigned long long>& s = numbers[idx];
    if (s.empty()) {
      std::printf("  ir%d: NO FRAMES\n", idx);
      continue;
    }
    const unsigned long long span = *s.rbegin() - *s.begin() + 1;
    const unsigned long long missing = span - s.size();
    std::printf("  ir%d: %zu frames, span %llu, %llu missing (%.3f%%), %.2f fps\n",
                idx, s.size(), span, missing,
                100.0 * static_cast<double>(missing) / static_cast<double>(span),
                duration > 0 ? s.size() / duration : 0.0);
  }
  size_t paired = 0;
  for (unsigned long long n : numbers[1]) {
    if (numbers[2].count(n)) ++paired;
  }
  std::printf("  paired: %zu   ir1-only: %zu   ir2-only: %zu\n", paired,
              numbers[1].size() - paired, numbers[2].size() - paired);

  // The domain decides whether the recording can answer the clock-skew
  // question at all, so surface it here rather than only in the report.
  std::set<std::string> domains;
  for (const FrameRec& r : rows)
    domains.insert(rs2_timestamp_domain_to_string(r.domain));
  for (const std::string& d : domains) std::printf("  ts domain: %s\n", d.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!parse_args(argc, argv, &opt)) {
    usage(argv[0]);
    return 2;
  }

  if (!make_dir(opt.outdir) || !make_dir(opt.outdir + "/frames")) {
    std::fprintf(stderr, "cannot create %s\n", opt.outdir.c_str());
    return 1;
  }

  std::signal(SIGINT, on_sigint);

  try {
    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    if (devices.size() == 0) {
      std::fprintf(stderr, "No RealSense device found.\n");
      return 1;
    }
    rs2::device dev = devices[0];
    const std::string usb =
        device_info(dev, RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR, "");
    std::printf("device %s  fw %s  usb %s\n",
                device_info(dev, RS2_CAMERA_INFO_SERIAL_NUMBER).c_str(),
                device_info(dev, RS2_CAMERA_INFO_FIRMWARE_VERSION).c_str(),
                usb.c_str());
    if (!usb3_link(usb)) {
      std::fprintf(stderr,
                   "!! USB '%s' -- run rs_probe and fix the link first.\n",
                   usb.c_str());
      return 1;
    }

    rs2::depth_sensor sensor = dev.first<rs2::depth_sensor>();
    std::vector<rs2::stream_profile> profiles = select_profiles(sensor, opt);
    if (profiles.size() != 2) {
      std::fprintf(stderr,
                   "!! could not select both IR profiles at %dx%d@%d y8 "
                   "(found %zu). Run rs_probe.\n",
                   opt.width, opt.height, opt.fps, profiles.size());
      return 1;
    }

    sensor.open(profiles);
    configure_sensor(sensor, opt);

    const size_t n_md = metadata_fields().size();
    std::vector<FrameRec> rows;
    rows.reserve(static_cast<size_t>(opt.seconds * opt.fps * 2 * 1.2) + 64);
    std::mutex mu;
    size_t saved = 0;
    size_t callback_errors = 0;

    // Runs on a librealsense delivery thread. Keep it short: the frame buffer
    // is recycled as soon as this returns, and stalling here drops frames.
    auto callback = [&](rs2::frame f) {
      const double mono = monotonic_seconds();
      const double real = realtime_seconds();
      try {
        const rs2::stream_profile p = f.get_profile();
        if (p.stream_type() != RS2_STREAM_INFRARED) return;

        FrameRec rec;
        rec.stream = p.stream_index();
        rec.number = f.get_frame_number();
        rec.ts_ms = f.get_timestamp();
        rec.domain = f.get_frame_timestamp_domain();
        rec.host_monotonic_s = mono;
        rec.host_realtime_s = real;
        rec.md.resize(n_md, 0);
        rec.md_ok.resize(n_md, 0);
        for (size_t i = 0; i < n_md; ++i) {
          const rs2_frame_metadata_value key = metadata_fields()[i].key;
          if (f.supports_frame_metadata(key)) {
            rec.md[i] = static_cast<long long>(f.get_frame_metadata(key));
            rec.md_ok[i] = 1;
          }
        }

        const bool save = opt.save_every > 0 &&
                          (rec.number % static_cast<unsigned long long>(
                                            opt.save_every) == 0);
        {
          std::lock_guard<std::mutex> lock(mu);
          rows.push_back(std::move(rec));
          if (save) ++saved;
        }
        if (save) {
          auto vf = f.as<rs2::video_frame>();
          if (vf) {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/frames/ir%d_%08llu.raw",
                          opt.outdir.c_str(), p.stream_index(),
                          f.get_frame_number());
            FILE* fh = std::fopen(path, "wb");
            if (fh) {
              std::fwrite(vf.get_data(), 1,
                          static_cast<size_t>(vf.get_width()) * vf.get_height(),
                          fh);
              std::fclose(fh);
            }
          }
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(mu);
        ++callback_errors;
      }
    };

    std::printf("\nrecording %.1f s ...\n", opt.seconds);
    const double t0 = monotonic_seconds();
    sensor.start(callback);
    while (monotonic_seconds() - t0 < opt.seconds && !g_interrupted.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    sensor.stop();
    sensor.close();
    const double duration = monotonic_seconds() - t0;
    if (g_interrupted.load()) std::printf("interrupted\n");

    std::vector<FrameRec> snapshot;
    {
      std::lock_guard<std::mutex> lock(mu);
      snapshot = rows;
    }
    std::sort(snapshot.begin(), snapshot.end(),
              [](const FrameRec& a, const FrameRec& b) {
                if (a.stream != b.stream) return a.stream < b.stream;
                return a.number < b.number;
              });

    write_csv(opt, snapshot);
    write_run_meta(opt, dev, profiles, duration, usb);
    std::printf("saved %zu raw frames", saved);
    if (callback_errors) std::printf(", %zu callback errors", callback_errors);
    std::printf("\n");
    quick_summary(snapshot, duration);
  } catch (const rs2::error& e) {
    std::fprintf(stderr, "\nlibrealsense error in %s(%s): %s\n",
                 e.get_failed_function().c_str(), e.get_failed_args().c_str(),
                 e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "\nerror: %s\n", e.what());
    return 1;
  }
  return 0;
}
