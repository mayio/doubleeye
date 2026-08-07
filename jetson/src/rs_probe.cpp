// Device health probe -- run this first, before any capture.
//
// Answers the questions that silently ruin everything downstream:
//   * Is the camera on a USB3 link, or did it fall back to USB2?
//   * Does the 848x480 Y8 IR profile exist on both stream indices?
//   * What does factory calibration claim for f and the IR baseline?
//   * Which timestamp metadata does this kernel actually expose?
//
// That last one is the real reason this program exists. librealsense can only
// surface the camera's hardware clock if uvcvideo is patched to expose UVC
// metadata nodes. On a stock JetPack kernel it is not, and get_timestamp()
// then silently reports host arrival time instead. Everything the plan says
// about camera-vs-Jetson clock skew depends on knowing which of those two you
// actually have, so the probe reports it explicitly rather than leaving it to
// be discovered later during IMU fusion.

#include "rs_common.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>

using namespace doubleeye;

namespace {

void section(const char* title) {
  std::printf("\n== %s ", title);
  int pad = 68 - static_cast<int>(std::strlen(title));
  for (int i = 0; i < pad; ++i) std::putchar('=');
  std::putchar('\n');
}

void probe_device(const rs2::device& dev) {
  section("device");
  const std::pair<const char*, rs2_camera_info> infos[] = {
      {"name", RS2_CAMERA_INFO_NAME},
      {"serial_number", RS2_CAMERA_INFO_SERIAL_NUMBER},
      {"firmware_version", RS2_CAMERA_INFO_FIRMWARE_VERSION},
      {"recommended_firmware", RS2_CAMERA_INFO_RECOMMENDED_FIRMWARE_VERSION},
      {"usb_type_descriptor", RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR},
      {"physical_port", RS2_CAMERA_INFO_PHYSICAL_PORT},
  };
  for (const auto& kv : infos) {
    if (dev.supports(kv.second)) {
      std::printf("  %-24s %s\n", kv.first, dev.get_info(kv.second));
    }
  }

  const std::string usb =
      device_info(dev, RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR, "");
  if (usb.empty()) {
    std::printf("\n  ?? no USB descriptor reported\n");
  } else if (!usb3_link(usb)) {
    std::printf("\n  !! USB link is '%s', not 3.x.\n", usb.c_str());
    std::printf("     Two IR streams at %dx%d@%d will not fit. Fix the cable\n",
                kWidth, kHeight, kFps);
    std::printf("     or port before trusting any measurement.\n");
  } else {
    std::printf("\n  USB link %s -- OK\n", usb.c_str());
  }
}

// Collect every Y8 infrared profile, keyed by geometry, valued by stream index.
bool probe_profiles(const rs2::device& dev) {
  section("ir profiles (stereo module)");
  std::map<std::tuple<int, int, int>, std::set<int>> found;
  for (const rs2::sensor& sensor : dev.query_sensors()) {
    for (const rs2::stream_profile& p : sensor.get_stream_profiles()) {
      if (p.stream_type() != RS2_STREAM_INFRARED) continue;
      if (p.format() != RS2_FORMAT_Y8) continue;
      auto vp = p.as<rs2::video_stream_profile>();
      if (!vp) continue;
      found[std::make_tuple(vp.width(), vp.height(), vp.fps())].insert(
          p.stream_index());
    }
  }

  const auto want = std::make_tuple(kWidth, kHeight, kFps);
  for (const auto& kv : found) {
    std::printf("  %4dx%-4d @ %3d Hz  y8  indices", std::get<0>(kv.first),
                std::get<1>(kv.first), std::get<2>(kv.first));
    for (int idx : kv.second) std::printf(" %d", idx);
    std::printf("%s\n", kv.first == want ? "   <-- requested" : "");
  }

  auto it = found.find(want);
  if (it == found.end()) {
    std::printf("\n  !! %dx%d@%d y8 not offered. Pick from the list above.\n",
                kWidth, kHeight, kFps);
    return false;
  }
  if (it->second.count(1) == 0 || it->second.count(2) == 0) {
    std::printf("\n  !! requested profile lacks both stream indices 1 and 2.\n");
    return false;
  }
  return true;
}

void probe_options(const rs2::device& dev) {
  section("stereo module options");
  rs2::depth_sensor sensor = dev.first<rs2::depth_sensor>();
  const rs2_option opts[] = {
      RS2_OPTION_EMITTER_ENABLED, RS2_OPTION_EMITTER_ON_OFF,
      RS2_OPTION_LASER_POWER,     RS2_OPTION_ENABLE_AUTO_EXPOSURE,
      RS2_OPTION_EXPOSURE,        RS2_OPTION_GAIN,
  };
  for (rs2_option opt : opts) {
    const char* name = rs2_option_to_string(opt);
    if (!sensor.supports(opt)) {
      std::printf("  %-22s unsupported\n", name);
      continue;
    }
    rs2::option_range r = sensor.get_option_range(opt);
    std::printf("  %-22s value=%-9g range=[%g, %g] step=%g default=%g\n", name,
                sensor.get_option(opt), r.min, r.max, r.step, r.def);
  }
  if (sensor.supports(RS2_OPTION_EMITTER_ON_OFF)) {
    std::printf(
        "\n  emitter_on_off present -> per-frame projector alternation is\n"
        "  available for the on/off A/B evaluation.\n");
  }
}

void probe_stream(const rs2::device& dev, const char* save_prefix) {
  section("live stream check");
  rs2::pipeline pipe;
  rs2::config cfg;
  cfg.enable_device(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
  // Depth deliberately not enabled: saves USB bandwidth and ASIC power.
  cfg.enable_stream(RS2_STREAM_INFRARED, 1, kWidth, kHeight, RS2_FORMAT_Y8, kFps);
  cfg.enable_stream(RS2_STREAM_INFRARED, 2, kWidth, kHeight, RS2_FORMAT_Y8, kFps);

  rs2::pipeline_profile profile = pipe.start(cfg);
  try {
    auto ir1 = profile.get_stream(RS2_STREAM_INFRARED, 1)
                   .as<rs2::video_stream_profile>();
    auto ir2 = profile.get_stream(RS2_STREAM_INFRARED, 2)
                   .as<rs2::video_stream_profile>();

    rs2_intrinsics intr = ir1.get_intrinsics();
    std::printf("  ir1 intrinsics  fx=%.3f fy=%.3f cx=%.3f cy=%.3f\n", intr.fx,
                intr.fy, intr.ppx, intr.ppy);
    std::printf("  ir1 distortion  [");
    for (int i = 0; i < 5; ++i) std::printf("%s%.5f", i ? ", " : "", intr.coeffs[i]);
    std::printf("]\n");

    rs2_extrinsics extr = ir1.get_extrinsics_to(ir2);
    const double tx = extr.translation[0];
    const double ty = extr.translation[1];
    const double tz = extr.translation[2];
    std::printf("  ir1->ir2 t      [%.6f, %.6f, %.6f] m  -> baseline %.3f mm\n",
                tx, ty, tz, std::fabs(tx) * 1000.0);
    std::printf("  f*B             %.2f px*m  (plan expects ~21)\n",
                intr.fx * std::fabs(tx));
    // On a rectified pair the off-axis translation and the rotation should
    // both be essentially zero. If they are not, the factory calibration has
    // drifted and step 3 is not optional.
    std::printf("  off-axis |t|    %.1f um (expect ~0 for a rectified pair)\n",
                std::max(std::fabs(ty), std::fabs(tz)) * 1e6);
    double rot_dev = 0.0;
    for (int i = 0; i < 9; ++i) {
      const double expect = (i % 4 == 0) ? 1.0 : 0.0;
      rot_dev = std::max(rot_dev, std::fabs(extr.rotation[i] - expect));
    }
    std::printf("  rotation dev    %.2e from identity\n", rot_dev);

    rs2::frameset fs;
    for (int i = 0; i < kFps; ++i) fs = pipe.wait_for_frames(5000);
    rs2::video_frame f1 = fs.get_infrared_frame(1);
    rs2::video_frame f2 = fs.get_infrared_frame(2);

    const bool matched = f1.get_frame_number() == f2.get_frame_number();
    std::printf("\n  frame numbers   ir1=%llu ir2=%llu %s\n",
                f1.get_frame_number(), f2.get_frame_number(),
                matched ? "(matched)" : "(MISMATCH -- check hardware sync)");
    std::printf("  get_timestamp   %.3f ms   domain=%s\n", f1.get_timestamp(),
                rs2_timestamp_domain_to_string(f1.get_frame_timestamp_domain()));
    std::printf("  |dt| ir1-ir2    %.6f ms\n",
                std::fabs(f1.get_timestamp() - f2.get_timestamp()));

    std::printf("\n  metadata availability (ir1):\n");
    int supported = 0;
    for (const auto& field : metadata_fields()) {
      const char* name = rs2_frame_metadata_to_string(field.key);
      if (f1.supports_frame_metadata(field.key)) {
        std::printf("    %-28s %lld\n", name,
                    static_cast<long long>(f1.get_frame_metadata(field.key)));
        ++supported;
      } else {
        std::printf("    %-28s UNSUPPORTED\n", name);
      }
    }

    const rs2_timestamp_domain domain = f1.get_frame_timestamp_domain();
    std::printf("\n  verdict:\n");
    if (domain == RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK) {
      std::printf("    timestamps are on the CAMERA clock. Clock-skew\n"
                  "    measurement against the Jetson is meaningful.\n");
    } else {
      std::printf("    timestamps are NOT on the camera clock (domain above).\n"
                  "    uvcvideo is almost certainly unpatched, so there is no\n"
                  "    UVC metadata node and librealsense is falling back to\n"
                  "    host arrival time. Camera-vs-host skew CANNOT be\n"
                  "    measured in this state, and the IMU time offset from\n"
                  "    bring-up step 3 will absorb an unknown USB latency.\n"
                  "    Fix: run librealsense's patch-realsense-ubuntu script\n"
                  "    for this kernel, or accept arrival-time semantics and\n"
                  "    document it.\n");
    }
    if (supported == 0) {
      std::printf("    no per-frame metadata at all -- consistent with the\n"
                  "    unpatched-kernel diagnosis above.\n");
    }

    if (save_prefix && *save_prefix) {
      for (int idx = 1; idx <= 2; ++idx) {
        const rs2::video_frame& f = (idx == 1) ? f1 : f2;
        char path[512];
        std::snprintf(path, sizeof(path), "%s_ir%d_%dx%d.raw", save_prefix, idx,
                      kWidth, kHeight);
        FILE* fh = std::fopen(path, "wb");
        if (!fh) {
          std::printf("\n  !! cannot write %s\n", path);
          continue;
        }
        std::fwrite(f.get_data(), 1,
                    static_cast<size_t>(f.get_width()) * f.get_height(), fh);
        std::fclose(fh);
        std::printf("\n  wrote %s\n", path);
      }
    }
  } catch (...) {
    pipe.stop();
    throw;
  }
  pipe.stop();
}

}  // namespace

int main(int argc, char** argv) {
  const char* save_prefix = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--save-prefix") == 0 && i + 1 < argc) {
      save_prefix = argv[++i];
    } else {
      std::printf("usage: %s [--save-prefix PREFIX]\n", argv[0]);
      return 2;
    }
  }

  try {
    std::printf("librealsense %d.%d.%d\n", RS2_API_MAJOR_VERSION,
                RS2_API_MINOR_VERSION, RS2_API_PATCH_VERSION);
    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    if (devices.size() == 0) {
      std::fprintf(stderr,
                   "\nNo RealSense device found.\n"
                   "Check: USB3 cable, `lsusb | grep 8086`, udev rules.\n");
      return 1;
    }
    for (rs2::device dev : devices) {
      probe_device(dev);
      const bool ok = probe_profiles(dev);
      probe_options(dev);
      if (ok) {
        probe_stream(dev, save_prefix);
      } else {
        std::printf("\nSkipping live check: requested profile unavailable.\n");
      }
    }
  } catch (const rs2::error& e) {
    std::fprintf(stderr, "\nlibrealsense error in %s(%s): %s\n",
                 e.get_failed_function().c_str(),
                 e.get_failed_args().c_str(), e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "\nerror: %s\n", e.what());
    return 1;
  }
  return 0;
}
