// Shared helpers for the Jetson-side RealSense tools.
//
// Deliberately depends on librealsense2 and the C++14 standard library only.
// No OpenCV: this box has a header/runtime version skew (headers report 3.3.1,
// the runtime .so is 3.2), and the capture path has no need for it -- raw Y8
// goes straight to disk and is decoded on the desktop.

#ifndef DOUBLEEYE_RS_COMMON_HPP
#define DOUBLEEYE_RS_COMMON_HPP

#include <librealsense2/rs.hpp>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace doubleeye {

// Plan calls for the native depth resolution; see doubleeye_plan.md "Key numbers".
constexpr int kWidth = 848;
constexpr int kHeight = 480;
constexpr int kFps = 30;

// Every per-frame metadata field worth logging. FRAME_LASER_POWER_MODE is in
// here because it is the only way to label which frames had the projector lit
// when RS2_OPTION_EMITTER_ON_OFF alternation is active -- i.e. it is what makes
// the projector-on/off A/B comparison possible at all.
struct MetadataField {
  const char* column;
  rs2_frame_metadata_value key;
};

inline const std::vector<MetadataField>& metadata_fields() {
  static const std::vector<MetadataField> fields = {
      {"md_frame_ts_us", RS2_FRAME_METADATA_FRAME_TIMESTAMP},
      {"md_sensor_ts_us", RS2_FRAME_METADATA_SENSOR_TIMESTAMP},
      {"md_backend_ts_ms", RS2_FRAME_METADATA_BACKEND_TIMESTAMP},
      {"md_arrival_ts_ms", RS2_FRAME_METADATA_TIME_OF_ARRIVAL},
      {"md_frame_counter", RS2_FRAME_METADATA_FRAME_COUNTER},
      {"md_exposure_us", RS2_FRAME_METADATA_ACTUAL_EXPOSURE},
      {"md_gain", RS2_FRAME_METADATA_GAIN_LEVEL},
      {"md_laser_mode", RS2_FRAME_METADATA_FRAME_LASER_POWER_MODE},
      {"md_actual_fps", RS2_FRAME_METADATA_ACTUAL_FPS},
  };
  return fields;
}

inline double monotonic_seconds() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}

inline double realtime_seconds() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}

inline std::string device_info(const rs2::device& dev, rs2_camera_info info,
                               const char* fallback = "?") {
  return dev.supports(info) ? std::string(dev.get_info(info))
                            : std::string(fallback);
}

// True when the reported USB descriptor is a 3.x link. Two IR streams at
// 848x480@30 do not fit in USB2 bandwidth, and the failure presents as
// unexplained frame drops rather than as an error.
inline bool usb3_link(const std::string& descriptor) {
  return !descriptor.empty() && descriptor[0] == '3';
}

// ---------------------------------------------------------------------------
// Jetson power state.
//
// This exists because of a measured, expensive failure: in power mode 3 with
// two Denver cores offline and jetson_clocks never applied, 848x480@30 on two
// IR streams delivered 18.5 fps instead of 30 -- a 34% loss with no error
// anywhere, on a USB3 link, with contiguous frame numbers. Setting MAXN and
// locking clocks fixed it completely.
//
// jetson_clocks does NOT survive a reboot, so that regression can silently
// return at any time and will look like a fresh bug. Every tool therefore
// reads the state and says so, and it goes into run.txt so a recording can
// never be misinterpreted later.

struct PowerState {
  int cpus_online = 0;
  int cpus_present = 0;
  long long scaling_min_khz = 0;  // min over online CPUs
  long long cpuinfo_max_khz = 0;
  bool clocks_locked = false;     // jetson_clocks raises scaling_min to max
};

inline std::string read_first_line(const std::string& path) {
  std::ifstream fh(path.c_str());
  std::string line;
  if (fh) std::getline(fh, line);
  return line;
}

// Parse a cpulist such as "0-5" or "0,3-5" into a count of entries.
inline std::vector<int> parse_cpu_list(const std::string& spec) {
  std::vector<int> out;
  std::stringstream ss(spec);
  std::string token;
  while (std::getline(ss, token, ',')) {
    const size_t dash = token.find('-');
    if (dash == std::string::npos) {
      if (!token.empty()) out.push_back(std::atoi(token.c_str()));
    } else {
      const int lo = std::atoi(token.substr(0, dash).c_str());
      const int hi = std::atoi(token.substr(dash + 1).c_str());
      for (int c = lo; c <= hi; ++c) out.push_back(c);
    }
  }
  return out;
}

inline PowerState read_power_state() {
  PowerState ps;
  const std::string base = "/sys/devices/system/cpu";
  const std::vector<int> online = parse_cpu_list(read_first_line(base + "/online"));
  ps.cpus_online = static_cast<int>(online.size());
  ps.cpus_present =
      static_cast<int>(parse_cpu_list(read_first_line(base + "/present")).size());

  bool all_locked = !online.empty();
  for (int cpu : online) {
    const std::string dir =
        base + "/cpu" + std::to_string(cpu) + "/cpufreq/";
    const long long smin = std::atoll(read_first_line(dir + "scaling_min_freq").c_str());
    const long long cmax = std::atoll(read_first_line(dir + "cpuinfo_max_freq").c_str());
    if (smin == 0 || cmax == 0) continue;
    if (ps.scaling_min_khz == 0 || smin < ps.scaling_min_khz)
      ps.scaling_min_khz = smin;
    if (cmax > ps.cpuinfo_max_khz) ps.cpuinfo_max_khz = cmax;
    if (smin < cmax) all_locked = false;
  }
  ps.clocks_locked = all_locked && ps.cpuinfo_max_khz > 0;
  return ps;
}

// Returns true when the box looks ready for a trustworthy measurement.
inline bool report_power_state(const PowerState& ps) {
  const bool all_cores = ps.cpus_present > 0 && ps.cpus_online == ps.cpus_present;
  std::printf("power state: %d/%d CPUs online, scaling_min %.2f GHz of "
              "max %.2f GHz, clocks %s\n",
              ps.cpus_online, ps.cpus_present, ps.scaling_min_khz / 1e6,
              ps.cpuinfo_max_khz / 1e6,
              ps.clocks_locked ? "LOCKED" : "not locked");
  if (all_cores && ps.clocks_locked) return true;
  std::printf(
      "  !! Not at full performance. Measured consequence: 848x480@30 on two\n"
      "     IR streams delivers ~18.5 fps instead of 30, silently, with no\n"
      "     error and no frame-number gaps. Run:\n"
      "         sudo nvpmodel -m 0 && sudo jetson_clocks\n"
      "     jetson_clocks does not persist across reboot, so re-check after\n"
      "     every boot.\n");
  return false;
}

}  // namespace doubleeye

#endif  // DOUBLEEYE_RS_COMMON_HPP
