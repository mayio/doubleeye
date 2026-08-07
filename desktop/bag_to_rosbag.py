#!/usr/bin/env python3
"""Convert a DoubleEye bag into a ROS 1 bag, for Kalibr.

This is the bridge the ROS decision implies. The capture path stays ROS-free
because preprocessing already costs 299% of the 30 Hz budget on the TX2 and ROS 1
transport would add copies to the memory bandwidth that is already the binding
constraint. Kalibr, meanwhile, *is* a ROS package and consumes rosbags. So:
record ROS-free, convert offline.

Uses `rosbags`, a pure-Python reader/writer. No ROS installation is needed on the
desktop, which matters because the desktop runs Ubuntu 24.04 where ROS 1 does not
exist at all.

Topics follow Kalibr's convention: `/cam0/image_raw` and `/cam1/image_raw`,
`sensor_msgs/Image`, `mono8`.

## Timestamps

Two cases, and the distinction matters:

  frames.csv present   Recorded by rs_ir_capture, so real host arrival times are
                       used. Left and right share the frameset's stamp.
  frames.csv absent    Collected by live_view --collect, which saves images only.
                       Stamps are then SYNTHESISED on a uniform grid.

Synthesised stamps are correct for `kalibr_calibrate_cameras`, which needs only
that a left and right image belonging to the same physical instant carry the same
stamp. They are **not** adequate for `kalibr_calibrate_imu_camera`, which
estimates a time offset against the IMU and therefore needs real ones. The tool
warns when it synthesises.

Desktop-side. numpy + rosbags.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore


def read_meta(bag: Path) -> dict:
    meta = {}
    run = bag / "run.txt"
    if run.exists():
        for line in run.read_text().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                meta[parts[0]] = parts[1]
    return meta


def real_timestamps(bag: Path) -> dict[str, int] | None:
    """Map frame number -> nanoseconds, from frames.csv if it exists."""
    csv_path = bag / "frames.csv"
    if not csv_path.exists():
        return None
    out: dict[str, int] = {}
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            if int(r.get("stream", 0)) != 1:
                continue
            raw = (r.get("host_realtime_s") or "").strip()
            if not raw:
                continue
            try:
                out[f"{int(r['frame_number']):08d}"] = int(float(raw) * 1e9)
            except (ValueError, KeyError):
                continue
    return out or None


def frame_pairs(bag: Path, w: int, h: int):
    grouped: dict[str, dict[str, Path]] = {}
    for p in sorted((bag / "frames").glob("*.raw")):
        try:
            idx, num = p.stem.split("_")
        except ValueError:
            continue
        grouped.setdefault(num, {})[idx] = p
    out = []
    for num in sorted(grouped):
        e = grouped[num]
        if "ir1" in e and "ir2" in e:
            out.append((num, e["ir1"], e["ir2"]))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output .bag (default: <bag>.bag beside the directory)")
    ap.add_argument("--rate", type=float, default=10.0,
                    help="synthesised frame rate when frames.csv is absent")
    ap.add_argument("--pitch-mm", type=float, default=24.0,
                    help="MEASURED square size for the printed Kalibr target. "
                         "Default is the measured 24.0 mm, not the nominal 25: "
                         "the print came out at 96%% scale.")
    args = ap.parse_args()

    if not (args.bag / "frames").is_dir():
        raise SystemExit(f"{args.bag}/frames not found")

    meta = read_meta(args.bag)
    try:
        w, h = (int(v) for v in meta["resolution"].split()[0].split("x"))
    except (KeyError, ValueError, IndexError):
        w, h = 848, 480

    pairs = frame_pairs(args.bag, w, h)
    if not pairs:
        raise SystemExit(f"no complete L/R pairs in {args.bag}/frames")

    stamps = real_timestamps(args.bag)
    out = args.out or args.bag.with_suffix(".bag")
    if out.exists():
        out.unlink()

    typestore = get_typestore(Stores.ROS1_NOETIC)
    Image = typestore.types["sensor_msgs/msg/Image"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    msgtype = Image.__msgtype__

    print(f"bag        {args.bag}")
    print(f"geometry   {w}x{h}")
    print(f"pairs      {len(pairs)}")
    if stamps:
        print("timestamps real, from frames.csv")
    else:
        print(f"timestamps SYNTHESISED at {args.rate:g} Hz — frames.csv is absent")
        print("           (fine for kalibr_calibrate_cameras; NOT valid for")
        print("            kalibr_calibrate_imu_camera, which estimates a time")
        print("            offset and needs real stamps)")

    # Kalibr needs a plausible absolute epoch; a zero start confuses some tools.
    base_ns = 1_700_000_000_000_000_000
    written = 0
    with Writer(out) as writer:
        conns = {
            1: writer.add_connection("/cam0/image_raw", msgtype,
                                     typestore=typestore),
            2: writer.add_connection("/cam1/image_raw", msgtype,
                                     typestore=typestore),
        }
        for i, (num, p1, p2) in enumerate(pairs):
            if stamps and num in stamps:
                t_ns = stamps[num]
            else:
                t_ns = base_ns + int(i * 1e9 / max(args.rate, 0.001))
            stamp = Time(sec=int(t_ns // 1_000_000_000),
                         nanosec=int(t_ns % 1_000_000_000))

            for index, path, frame_id in ((1, p1, "cam0"), (2, p2, "cam1")):
                data = np.fromfile(path, dtype=np.uint8)
                if data.size != w * h:
                    print(f"  skip {path.name}: {data.size} bytes, expected {w * h}")
                    continue
                msg = Image(
                    header=Header(seq=i, stamp=stamp, frame_id=frame_id),
                    height=h, width=w, encoding="mono8", is_bigendian=0,
                    step=w, data=data)
                writer.write(conns[index], t_ns,
                             typestore.serialize_ros1(msg, msgtype))
            written += 1

    size_mb = out.stat().st_size / 1e6
    print(f"\nwrote {out}  ({written} pairs, {size_mb:.1f} MB)")
    print("  /cam0/image_raw  sensor_msgs/Image mono8   (ir1, left)")
    print("  /cam1/image_raw  sensor_msgs/Image mono8   (ir2, right)")

    pitch = args.pitch_mm / 1000.0
    print("\nNext, on a machine with Kalibr (ROS 1 Noetic in Docker; the desktop"
          " cannot host ROS 1):")
    print("\n  cat > checkerboard.yaml <<'YAML'")
    print("  target_type: 'checkerboard'")
    print("  targetCols: 9")
    print("  targetRows: 6")
    print(f"  rowSpacingMeters: {pitch}")
    print(f"  colSpacingMeters: {pitch}")
    print("  YAML")
    print(f"\n  kalibr_calibrate_cameras \\")
    print(f"    --bag {out.name} \\")
    print("    --topics /cam0/image_raw /cam1/image_raw \\")
    print("    --models pinhole-radtan pinhole-radtan \\")
    print("    --target checkerboard.yaml")
    print(f"\nPitch above is {args.pitch_mm:g} mm, the MEASURED value. The A4"
          " print came out")
    print("at 96% scale, so the nominal 25 mm would be wrong by 4%.")
    print("Compare the result against the factory values in")
    print("doc/04-baseline-measurements.md — fx 430.551, baseline 49.883 mm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
