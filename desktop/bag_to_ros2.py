#!/usr/bin/env python3
"""Convert a DoubleEye bag plus its matcher output into a ROS 2 bag, for rviz2.

Publishes what the pipeline actually produces:

  /doubleeye/image_raw    sensor_msgs/Image      mono8, the left IR frame
  /doubleeye/depth        sensor_msgs/Image      32FC1, metres, NaN where unknown
  /doubleeye/points       sensor_msgs/PointCloud2  xyz + rgb + margin
  /doubleeye/camera_info  sensor_msgs/CameraInfo
  /tf_static              tf2_msgs/TFMessage     map -> camera_link (identity)

## The depth image is sparse, and that is not a bug

This matcher is *sparse*: it produces a few hundred correspondences at keypoints,
not a disparity for every pixel. So the depth image has values only at matched
keypoints and NaN everywhere else, and in rviz it looks like confetti rather than
a depth map. That is an accurate picture of what sparse stereo gives you.

If a dense depth image is wanted for its own sake, the D435 computes one in the
ASIC and it can be recorded as a stream -- it is simply not what this pipeline
produces. --dilate N thickens each sample into an N x N block so the sparse depth
is actually visible on screen; it invents nothing, it just draws each sample
bigger, and it is off by default so the unflattered picture is the default one.

## Point colour

Points are coloured by score margin, which is the confidence measure the matcher
exports: red below 0.1, amber to 0.3, green above. Over eight Middlebury scenes
precision by margin quartile runs 0.169 / 0.286 / 0.391 / 0.659, so the red points
really are the ones to distrust. The raw margin also rides along as a float field,
so rviz can colour by it directly.

## No ROS installation is needed to run this

`rosbags` is a pure-Python writer. ROS 2 is needed only on the machine that opens
the result in rviz2. See doc/07-tools.md for the viewing recipe.

  python bag_to_ros2.py BAG_DIR [-o OUT] [--dilate N] [--method masda] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import struct
import sys
from collections import defaultdict

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

# Factory calibration, 848x480 IR, from doc/04-baseline-measurements.md. The IR
# pair is delivered rectified by the D4 ASIC: distortion exactly 0, rotation
# exactly identity, both verified rather than assumed.
FX = FY = 430.551
CX, CY = 427.381, 243.158
BASELINE_M = 0.049883
FB = FX * BASELINE_M            # 21.48 px*m
FRAME = "camera_link"


def read_run_txt(bag: str) -> tuple[int, int]:
    w, h = 848, 480
    p = os.path.join(bag, "run.txt")
    if os.path.exists(p):
        for line in open(p):
            if line.startswith("resolution"):
                try:
                    wh = line.split()[1].split("x")
                    w, h = int(wh[0]), int(wh[1])
                except (IndexError, ValueError):
                    pass
    return w, h


def read_matches(bag: str, method: str):
    """matches.csv grouped by frame number. Written by core/tools/de_match."""
    p = os.path.join(bag, "matches.csv")
    if not os.path.exists(p):
        sys.exit(f"no {p}\n"
                 f"Run the matcher over the bag first, e.g.\n"
                 f"  core/build/de_match {bag} --right-density 6 --min-margin 0.10")
    by_frame = defaultdict(list)
    with open(p) as f:
        for row in csv.DictReader(f):
            if row.get("method") != method:
                continue
            by_frame[row["frame"]].append(row)
    if not by_frame:
        sys.exit(f"{p} has no rows with method={method}")
    # margin was added on 2026-08-07; older files do not have it.
    has_margin = "margin" in next(iter(by_frame.values()))[0]
    return by_frame, has_margin


def frame_stamps(bag: str, nums: list[str]) -> dict[str, int]:
    """Real host arrival times from frames.csv when present, else a uniform grid.

    Synthesised stamps are fine for looking at geometry in rviz. They would not be
    fine for anything estimating a time offset, which is why this says which it
    used rather than quietly picking one.
    """
    p = os.path.join(bag, "frames.csv")
    out: dict[str, int] = {}
    if os.path.exists(p):
        with open(p) as f:
            for row in csv.DictReader(f):
                num = row.get("frame") or row.get("number")
                t = row.get("host_time_s") or row.get("t_host_s") or row.get("t_s")
                if num and t:
                    try:
                        out[str(num).zfill(len(nums[0]))] = int(float(t) * 1e9)
                    except ValueError:
                        pass
    if len(out) >= len(nums):
        print(f"  timestamps: real host times from frames.csv")
        return out
    print(f"  timestamps: SYNTHESISED on a 30 Hz grid "
          f"(frames.csv absent or incomplete)")
    return {n: int(i * (1e9 / 30.0)) for i, n in enumerate(nums)}


def colour_for(margin: float) -> tuple[int, int, int]:
    if margin < 0.10:
        return (220, 50, 47)        # red: precision ~0.17 in this band
    if margin < 0.30:
        return (203, 145, 20)       # amber
    return (60, 170, 75)            # green


def pack_points(rows, has_margin):
    """xyz + packed rgb + margin, as the 20-byte records PointCloud2 expects."""
    buf = bytearray()
    n = 0
    for r in rows:
        d = float(r["disparity"])
        if d <= 0.0:
            continue                # no depth without positive disparity
        z = FB / d
        x = (float(r["xl"]) - CX) * z / FX
        y = (float(r["yl"]) - CY) * z / FY
        m = float(r["margin"]) if has_margin else 1.0
        cr, cg, cb = colour_for(m)
        rgb = struct.unpack("f", struct.pack("I", (cr << 16) | (cg << 8) | cb))[0]
        buf += struct.pack("<fffff", x, y, z, rgb, m)
        n += 1
    return bytes(buf), n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--method", default="masda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dilate", type=int, default=1,
                    help="draw each sparse depth sample as an NxN block so it is "
                         "visible on screen (default 1, i.e. one pixel)")
    a = ap.parse_args()

    bag = a.bag.rstrip("/")
    out = a.out or (bag + "_ros2")
    W, H = read_run_txt(bag)
    by_frame, has_margin = read_matches(bag, a.method)
    nums = sorted(by_frame)
    if a.limit:
        nums = nums[:a.limit]
    print(f"{bag}  {W}x{H}  {len(nums)} frames with {a.method} matches"
          f"{'' if has_margin else '  (no margin column: older matches.csv)'}")
    stamps = frame_stamps(bag, nums)

    if os.path.exists(out):
        shutil.rmtree(out)
    ts = get_typestore(Stores.ROS2_JAZZY)
    Image = ts.types["sensor_msgs/msg/Image"]
    PointCloud2 = ts.types["sensor_msgs/msg/PointCloud2"]
    PointField = ts.types["sensor_msgs/msg/PointField"]
    CameraInfo = ts.types["sensor_msgs/msg/CameraInfo"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    TFMessage = ts.types["tf2_msgs/msg/TFMessage"]
    TransformStamped = ts.types["geometry_msgs/msg/TransformStamped"]
    Transform = ts.types["geometry_msgs/msg/Transform"]
    Vector3 = ts.types["geometry_msgs/msg/Vector3"]
    Quaternion = ts.types["geometry_msgs/msg/Quaternion"]
    RegionOfInterest = ts.types["sensor_msgs/msg/RegionOfInterest"]

    def hdr(ns: int):
        return Header(stamp=Time(sec=ns // 10**9, nanosec=ns % 10**9),
                      frame_id=FRAME)

    fields = [
        PointField(name="x", offset=0, datatype=7, count=1),
        PointField(name="y", offset=4, datatype=7, count=1),
        PointField(name="z", offset=8, datatype=7, count=1),
        PointField(name="rgb", offset=12, datatype=7, count=1),
        PointField(name="margin", offset=16, datatype=7, count=1),
    ]

    written = 0
    # Version 8 rather than VERSION_LATEST. rosbag2's metadata version has to be
    # one the reader understands, and a bag written to a newer version than the
    # installed ROS 2 supports fails to open with a message about the version
    # rather than about anything useful. 8 is what Humble and Jazzy both read.
    with Writer(out, version=8) as w:
        c_img = w.add_connection("/doubleeye/image_raw", Image.__msgtype__,
                                 typestore=ts)
        c_dep = w.add_connection("/doubleeye/depth", Image.__msgtype__,
                                 typestore=ts)
        c_pc = w.add_connection("/doubleeye/points", PointCloud2.__msgtype__,
                                typestore=ts)
        c_ci = w.add_connection("/doubleeye/camera_info", CameraInfo.__msgtype__,
                                typestore=ts)
        c_tf = w.add_connection("/tf_static", TFMessage.__msgtype__, typestore=ts)

        # rviz2 needs its fixed frame to exist in TF or it refuses to draw
        # anything. One identity map -> camera_link removes that whole class of
        # "it published but nothing appears".
        t0 = stamps[nums[0]]
        tf = TFMessage(transforms=[TransformStamped(
            header=Header(stamp=Time(sec=t0 // 10**9, nanosec=t0 % 10**9),
                          frame_id="map"),
            child_frame_id=FRAME,
            transform=Transform(translation=Vector3(x=0.0, y=0.0, z=0.0),
                                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)))])
        w.write(c_tf, t0, ts.serialize_cdr(tf, TFMessage.__msgtype__))

        for num in nums:
            path = os.path.join(bag, "frames", f"ir1_{num}.raw")
            if not os.path.exists(path):
                continue
            raw = np.fromfile(path, dtype=np.uint8)
            if raw.size != W * H:
                print(f"  skip {num}: {raw.size} bytes, expected {W*H}")
                continue
            img = raw.reshape(H, W)
            ns = stamps.get(num, 0)
            rows = by_frame[num]

            w.write(c_img, ns, ts.serialize_cdr(
                Image(header=hdr(ns), height=H, width=W, encoding="mono8",
                      is_bigendian=0, step=W, data=img.reshape(-1)),
                Image.__msgtype__))

            # Sparse depth: NaN means "no measurement here", which is what rviz
            # and every depth consumer treat as absent. Zero would be read as a
            # measurement of zero range.
            depth = np.full((H, W), np.nan, dtype=np.float32)
            k = max(1, a.dilate) // 2
            for r in rows:
                d = float(r["disparity"])
                if d <= 0.0:
                    continue
                u, v = int(round(float(r["xl"]))), int(round(float(r["yl"])))
                y0, y1 = max(0, v - k), min(H, v + k + 1)
                x0, x1 = max(0, u - k), min(W, u + k + 1)
                depth[y0:y1, x0:x1] = FB / d
            w.write(c_dep, ns, ts.serialize_cdr(
                Image(header=hdr(ns), height=H, width=W, encoding="32FC1",
                      is_bigendian=0, step=4 * W,
                      data=depth.reshape(-1).view(np.uint8)),
                Image.__msgtype__))

            data, npts = pack_points(rows, has_margin)
            w.write(c_pc, ns, ts.serialize_cdr(
                PointCloud2(header=hdr(ns), height=1, width=npts, fields=fields,
                            is_bigendian=False, point_step=20, row_step=20 * npts,
                            data=np.frombuffer(data, dtype=np.uint8),
                            is_dense=True),
                PointCloud2.__msgtype__))

            w.write(c_ci, ns, ts.serialize_cdr(
                CameraInfo(header=hdr(ns), height=H, width=W,
                           distortion_model="plumb_bob",
                           d=np.zeros(5, dtype=np.float64),
                           k=np.array([FX, 0, CX, 0, FY, CY, 0, 0, 1], np.float64),
                           r=np.eye(3, dtype=np.float64).reshape(-1),
                           p=np.array([FX, 0, CX, 0, 0, FY, CY, 0, 0, 0, 1, 0],
                                      np.float64),
                           binning_x=0, binning_y=0,
                           roi=RegionOfInterest(x_offset=0, y_offset=0, height=0,
                                                width=0, do_rectify=False)),
                CameraInfo.__msgtype__))
            written += 1

    pts = sum(len(v) for v in (by_frame[n] for n in nums))
    print(f"\nwrote {out}  ({written} frames, {pts} matches)")
    print(f"  depth is SPARSE -- values only at matched keypoints"
          f"{f', drawn as {a.dilate}x{a.dilate} blocks' if a.dilate > 1 else ''}")
    print("\nview it:")
    print(f"  ros2 bag play {out} --loop")
    print("  rviz2      # Fixed Frame: map")
    print("             # add PointCloud2 /doubleeye/points, colour by 'margin'")
    print("             # add Image /doubleeye/image_raw and /doubleeye/depth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
