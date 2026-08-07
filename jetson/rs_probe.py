#!/usr/bin/env python3
"""Device health probe -- run this first, before any capture.

Answers the questions that silently ruin everything downstream:
  * Is the camera on a USB3 link, or did it fall back to USB2?
  * Does the 848x480 Y8 IR profile actually exist at the requested rate?
  * What does factory calibration claim for f and the IR baseline?
  * Which timestamp metadata fields does this firmware/kernel actually expose?

Jetson-side. Depends on pyrealsense2 only -- no numpy, no OpenCV.
Python 3.6 compatible (JetPack 4.6).
"""

from __future__ import print_function

import argparse
import sys

import pyrealsense2 as rs

# Plan calls for the native depth resolution; see doubleeye_plan.md "Key numbers".
WIDTH, HEIGHT, FPS = 848, 480, 30

TS_FIELDS = [
    "frame_timestamp",
    "sensor_timestamp",
    "backend_timestamp",
    "time_of_arrival",
    "frame_counter",
    "actual_exposure",
    "gain_level",
    "actual_fps",
]


def section(title):
    print("\n== %s %s" % (title, "=" * max(0, 68 - len(title))))


def probe_device(dev):
    section("device")
    for info in ("name", "serial_number", "firmware_version",
                 "recommended_firmware_version", "usb_type_descriptor",
                 "product_line", "physical_port"):
        key = getattr(rs.camera_info, info, None)
        if key is None or not dev.supports(key):
            continue
        print("  %-30s %s" % (info, dev.get_info(key)))

    usb_key = getattr(rs.camera_info, "usb_type_descriptor", None)
    if usb_key is not None and dev.supports(usb_key):
        usb = dev.get_info(usb_key)
        if not usb.startswith("3"):
            print("\n  !! USB link is %r, not 3.x." % usb)
            print("     848x480@30 on two IR streams will not fit. Fix the")
            print("     cable/port before trusting any measurement.")
        else:
            print("\n  USB link %s -- OK" % usb)

    fw_key = rs.camera_info.firmware_version
    rec_key = getattr(rs.camera_info, "recommended_firmware_version", None)
    if rec_key is not None and dev.supports(fw_key) and dev.supports(rec_key):
        fw, rec = dev.get_info(fw_key), dev.get_info(rec_key)
        if fw != rec:
            print("  note: firmware %s, recommended %s" % (fw, rec))


def probe_profiles(dev):
    """Confirm the exact IR profile we intend to stream exists."""
    section("ir profiles (stereo module)")
    found = {}
    for sensor in dev.query_sensors():
        for p in sensor.get_stream_profiles():
            if p.stream_type() != rs.stream.infrared:
                continue
            vp = p.as_video_stream_profile()
            if vp.format() != rs.format.y8:
                continue
            key = (vp.width(), vp.height(), vp.fps())
            found.setdefault(key, set()).add(vp.stream_index())

    want = (WIDTH, HEIGHT, FPS)
    for key in sorted(found):
        mark = "  <-- requested" if key == want else ""
        print("  %4dx%-4d @ %2d Hz  y8  indices %s%s"
              % (key[0], key[1], key[2], sorted(found[key]), mark))

    if want not in found:
        print("\n  !! %dx%d@%d y8 not offered. Pick from the list above." % want)
        return False
    if set(found[want]) < {1, 2}:
        print("\n  !! only stream indices %s at the requested profile; need both"
              " 1 and 2." % sorted(found[want]))
        return False
    return True


def probe_options(dev):
    section("stereo module options")
    try:
        sensor = dev.first_depth_sensor()
    except RuntimeError as exc:
        print("  no depth sensor: %s" % exc)
        return
    for name in ("emitter_enabled", "emitter_on_off", "laser_power",
                 "enable_auto_exposure", "exposure", "gain"):
        opt = getattr(rs.option, name, None)
        if opt is None or not sensor.supports(opt):
            print("  %-22s unsupported" % name)
            continue
        rng = sensor.get_option_range(opt)
        print("  %-22s value=%-10g range=[%g, %g] step=%g default=%g"
              % (name, sensor.get_option(opt), rng.min, rng.max, rng.step,
                 rng.default))

    if getattr(rs.option, "emitter_on_off", None) is not None and \
            sensor.supports(rs.option.emitter_on_off):
        print("\n  emitter_on_off present -> per-frame emitter alternation is")
        print("  available for the projector-on/off A/B evaluation.")


def probe_stream(dev, save_prefix=None):
    """Briefly stream both IR channels: geometry, sync, metadata availability."""
    section("live stream check")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
    # Depth deliberately not enabled: saves USB bandwidth and ASIC power.
    cfg.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS)
    cfg.enable_stream(rs.stream.infrared, 2, WIDTH, HEIGHT, rs.format.y8, FPS)

    profile = pipe.start(cfg)
    try:
        ir1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        ir2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()

        intr = ir1.get_intrinsics()
        print("  ir1 intrinsics  fx=%.3f fy=%.3f cx=%.3f cy=%.3f model=%s"
              % (intr.fx, intr.fy, intr.ppx, intr.ppy, intr.model))
        print("  ir1 distortion  %s" % (["%.5f" % c for c in intr.coeffs],))

        extr = ir1.get_extrinsics_to(ir2)
        tx, ty, tz = extr.translation
        baseline_mm = abs(tx) * 1000.0
        print("  ir1->ir2 t      [%.6f, %.6f, %.6f] m  -> baseline %.3f mm"
              % (tx, ty, tz, baseline_mm))
        print("  f*B             %.2f px*m" % (intr.fx * abs(tx)))
        # Off-axis translation and any rotation should be ~0 on a rectified pair.
        off_axis_um = max(abs(ty), abs(tz)) * 1e6
        print("  off-axis |t|    %.1f um (expect ~0 for a rectified pair)"
              % off_axis_um)
        rot_dev = max(abs(extr.rotation[i] - (1.0 if i % 4 == 0 else 0.0))
                      for i in range(9))
        print("  rotation dev    %.2e from identity" % rot_dev)

        frames = None
        for _ in range(FPS):  # let auto-exposure settle / metadata populate
            frames = pipe.wait_for_frames(5000)
        f1 = frames.get_infrared_frame(1)
        f2 = frames.get_infrared_frame(2)

        print("\n  frame numbers   ir1=%d ir2=%d %s"
              % (f1.get_frame_number(), f2.get_frame_number(),
                 "(matched)" if f1.get_frame_number() == f2.get_frame_number()
                 else "(MISMATCH -- check hardware sync)"))
        print("  get_timestamp   ir1=%.3f ms  domain=%s"
              % (f1.get_timestamp(), f1.get_frame_timestamp_domain()))
        print("  |dt| ir1-ir2    %.4f ms"
              % abs(f1.get_timestamp() - f2.get_timestamp()))

        print("\n  metadata availability (ir1):")
        for name in TS_FIELDS:
            key = getattr(rs.frame_metadata_value, name, None)
            if key is None:
                print("    %-20s not in this pyrealsense2 build" % name)
            elif f1.supports_frame_metadata(key):
                print("    %-20s %s" % (name, f1.get_frame_metadata(key)))
            else:
                print("    %-20s UNSUPPORTED" % name)

        if save_prefix:
            for idx, frame in ((1, f1), (2, f2)):
                path = "%s_ir%d_%dx%d.raw" % (save_prefix, idx, WIDTH, HEIGHT)
                with open(path, "wb") as fh:
                    fh.write(bytearray(frame.get_data()))
                print("\n  wrote %s" % path)
    finally:
        pipe.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-prefix", default=None,
                    help="write one raw Y8 frame per channel for visual check")
    args = ap.parse_args()

    print("pyrealsense2 %s" % rs.__version__)
    devices = list(rs.context().query_devices())
    if not devices:
        print("\nNo RealSense device found.", file=sys.stderr)
        print("Check: USB3 cable, `lsusb | grep 8086`, udev rules.", file=sys.stderr)
        return 1

    for dev in devices:
        probe_device(dev)
        ok = probe_profiles(dev)
        probe_options(dev)
        if ok:
            probe_stream(dev, args.save_prefix)
        else:
            print("\nSkipping live check: requested profile unavailable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
