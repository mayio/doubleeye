// Shared helpers for the Jetson-side RealSense tools.
//
// Deliberately depends on librealsense2 and the C++14 standard library only.
// No OpenCV: this box has a header/runtime version skew (headers report 3.3.1,
// the runtime .so is 3.2), and the capture path has no need for it -- raw Y8
// goes straight to disk and is decoded on the desktop.

#ifndef DOUBLEEYE_RS_COMMON_HPP
#define DOUBLEEYE_RS_COMMON_HPP

#include <librealsense2/rs.hpp>

#include <ctime>
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

}  // namespace doubleeye

#endif  // DOUBLEEYE_RS_COMMON_HPP
