#!/usr/bin/env python3
"""Bring-up step 1: IR capture with timestamp instrumentation.

Records both D435 IR channels and logs, per frame, every clock the stack
exposes. The point is not the images -- it is the CSV. Downstream you need to
know which timestamp domain is trustworthy before the IMU can be fused against
it, and whether frames are being dropped under real load.

Design notes:
  * A per-frame callback is used rather than wait_for_frames(). The frameset
    syncer silently discards unmatched frames, which is exactly the signal we
    are trying to measure.
  * The callback stays cheap: append a tuple, occasionally write raw bytes.
    Analysis happens offline on the desktop (see desktop/capture_report.py).
  * Exposure is fixed. Auto-exposure hunts while driving and makes intervals
    unreproducible.
  * Depth is never enabled -- USB bandwidth and ASIC power.

Jetson-side. pyrealsense2 only. Python 3.6 compatible (JetPack 4.6).
"""

from __future__ import print_function

import argparse
import csv
import os
import sys
import time

import pyrealsense2 as rs

WIDTH, HEIGHT, FPS = 848, 480, 30

# (csv column, rs.frame_metadata_value attribute name)
METADATA = [
    ("md_frame_ts_us", "frame_timestamp"),
    ("md_sensor_ts_us", "sensor_timestamp"),
    ("md_backend_ts_ms", "backend_timestamp"),
    ("md_arrival_ts_ms", "time_of_arrival"),
    ("md_frame_counter", "frame_counter"),
    ("md_exposure_us", "actual_exposure"),
    ("md_gain", "gain_level"),
]

COLUMNS = (["stream", "frame_number", "ts_ms", "ts_domain",
            "host_monotonic_s", "host_realtime_s"]
           + [c for c, _ in METADATA])


class Recorder(object):
    def __init__(self, outdir, save_every):
        self.outdir = outdir
        self.save_every = save_every
        self.rows = []
        self.saved = 0
        self.errors = []
        # Resolved once; getattr per frame in the callback is wasteful.
        self.md_keys = [(col, getattr(rs.frame_metadata_value, attr, None))
                        for col, attr in METADATA]

    def __call__(self, frame):
        # Runs on a librealsense thread. Keep it short.
        mono = time.monotonic()
        real = time.time()
        try:
            profile = frame.get_profile()
            if profile.stream_type() != rs.stream.infrared:
                return
            index = profile.stream_index()
            number = frame.get_frame_number()

            row = {
                "stream": index,
                "frame_number": number,
                "ts_ms": "%.6f" % frame.get_timestamp(),
                "ts_domain": str(frame.get_frame_timestamp_domain()),
                "host_monotonic_s": "%.9f" % mono,
                "host_realtime_s": "%.9f" % real,
            }
            for col, key in self.md_keys:
                if key is not None and frame.supports_frame_metadata(key):
                    row[col] = frame.get_frame_metadata(key)
                else:
                    row[col] = ""
            self.rows.append(row)

            if self.save_every and number % self.save_every == 0:
                path = os.path.join(
                    self.outdir, "frames",
                    "ir%d_%08d.raw" % (index, number))
                with open(path, "wb") as fh:
                    fh.write(bytearray(frame.get_data()))
                self.saved += 1
        except Exception as exc:  # never let a callback exception kill the stream
            self.errors.append(repr(exc))


def configure_sensor(dev, exposure_us, gain, emitter):
    sensor = dev.first_depth_sensor()

    def setopt(name, value):
        opt = getattr(rs.option, name, None)
        if opt is None or not sensor.supports(opt):
            print("  option %s unsupported -- skipped" % name)
            return False
        sensor.set_option(opt, value)
        print("  %-22s = %g" % (name, value))
        return True

    print("sensor configuration:")
    setopt("enable_auto_exposure", 0)
    setopt("exposure", exposure_us)
    setopt("gain", gain)
    if emitter == "on":
        setopt("emitter_enabled", 1)
    elif emitter == "off":
        setopt("emitter_enabled", 0)
    elif emitter == "alternate":
        setopt("emitter_enabled", 1)
        if not setopt("emitter_on_off", 1):
            print("  !! per-frame alternation unavailable on this firmware")
    return sensor


def write_log(outdir, rec, meta_lines):
    csv_path = os.path.join(outdir, "frames.csv")
    with open(csv_path, "w") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rec.rows:
            writer.writerow(row)

    meta_path = os.path.join(outdir, "run.txt")
    with open(meta_path, "w") as fh:
        for line in meta_lines:
            fh.write(line + "\n")
    return csv_path


def quick_summary(rec, duration):
    """Coarse in-situ check. The real analysis is desktop/capture_report.py."""
    print("\n-- quick summary (full analysis: desktop/capture_report.py) --")
    for index in (1, 2):
        numbers = [r["frame_number"] for r in rec.rows if r["stream"] == index]
        if not numbers:
            print("  ir%d: NO FRAMES" % index)
            continue
        span = numbers[-1] - numbers[0] + 1
        gaps = span - len(numbers)
        print("  ir%d: %d frames, span %d, %d missing (%.2f%%), %.2f fps"
              % (index, len(numbers), span, gaps,
                 100.0 * gaps / span if span else 0.0,
                 len(numbers) / duration if duration else 0.0))

    n1 = set(r["frame_number"] for r in rec.rows if r["stream"] == 1)
    n2 = set(r["frame_number"] for r in rec.rows if r["stream"] == 2)
    if n1 and n2:
        print("  paired: %d   ir1-only: %d   ir2-only: %d"
              % (len(n1 & n2), len(n1 - n2), len(n2 - n1)))
    if rec.errors:
        print("  callback errors: %d (first: %s)"
              % (len(rec.errors), rec.errors[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", help="output directory (created)")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--exposure-us", type=int, default=1500,
                    help="fixed exposure; 1000-2000 keeps motion blur under a "
                         "pixel at RC speeds (default: %(default)s)")
    ap.add_argument("--gain", type=int, default=64,
                    help="raise to compensate the short exposure")
    ap.add_argument("--emitter", choices=("on", "off", "alternate"),
                    default="on")
    ap.add_argument("--save-every", type=int, default=30,
                    help="write raw Y8 every Nth frame; 0 disables. Saving "
                         "every frame will itself cause drops.")
    args = ap.parse_args()

    if os.path.exists(args.outdir) and os.listdir(args.outdir):
        print("refusing to write into non-empty %s" % args.outdir,
              file=sys.stderr)
        return 1
    os.makedirs(os.path.join(args.outdir, "frames"), exist_ok=True)

    devices = list(rs.context().query_devices())
    if not devices:
        print("No RealSense device found.", file=sys.stderr)
        return 1
    dev = devices[0]
    serial = dev.get_info(rs.camera_info.serial_number)
    usb_key = getattr(rs.camera_info, "usb_type_descriptor", None)
    usb = dev.get_info(usb_key) if usb_key and dev.supports(usb_key) else "?"
    print("device %s  fw %s  usb %s"
          % (serial, dev.get_info(rs.camera_info.firmware_version), usb))
    if not usb.startswith("3"):
        print("!! USB %s -- run rs_probe.py and fix the link first." % usb,
              file=sys.stderr)
        return 1

    configure_sensor(dev, args.exposure_us, args.gain, args.emitter)

    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS)
    cfg.enable_stream(rs.stream.infrared, 2, WIDTH, HEIGHT, rs.format.y8, FPS)

    rec = Recorder(args.outdir, args.save_every)
    pipe = rs.pipeline()

    print("\nrecording %.1f s ..." % args.seconds)
    t0 = time.monotonic()
    profile = pipe.start(cfg, rec)
    try:
        while time.monotonic() - t0 < args.seconds:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        pipe.stop()
    duration = time.monotonic() - t0

    ir1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    intr = ir1.get_intrinsics()
    extr = ir1.get_extrinsics_to(profile.get_stream(rs.stream.infrared, 2))
    meta = [
        "serial %s" % serial,
        "firmware %s" % dev.get_info(rs.camera_info.firmware_version),
        "usb %s" % usb,
        "resolution %dx%d @ %d" % (WIDTH, HEIGHT, FPS),
        "exposure_us %d" % args.exposure_us,
        "gain %d" % args.gain,
        "emitter %s" % args.emitter,
        "duration_s %.3f" % duration,
        "fx %.6f" % intr.fx,
        "fy %.6f" % intr.fy,
        "cx %.6f" % intr.ppx,
        "cy %.6f" % intr.ppy,
        "baseline_m %.9f" % abs(extr.translation[0]),
        "frame_bytes %d" % (WIDTH * HEIGHT),
    ]

    csv_path = write_log(args.outdir, rec, meta)
    print("wrote %s (%d rows), %d raw frames"
          % (csv_path, len(rec.rows), rec.saved))
    quick_summary(rec, duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
