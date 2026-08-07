#!/usr/bin/env python3
"""Live view in rviz2: Jetson matches, laptop publishes.

    ros2 run ... no. Just:  source /opt/ros/jazzy/setup.bash && python3 de_live_ros2.py

Pipeline, all of it started by this script:

    ssh jetson 'rs_ir_stream --streams both | de_pipe ...'   <- capture + match
        |  DEMR packets over the ssh pipe
    this script                                              <- ROS 2 publisher
        |  /doubleeye/image_raw, /depth, /points, /camera_info, /tf
    rviz2

Detection and matching run ON THE JETSON, so what you see is the real pipeline's
output rather than a laptop reimplementation of it. Only the ROS 2 half runs here,
which keeps the capture path ROS-free -- the Jetson is Ubuntu 18.04, where every
ROS 2 release is long past end of life.

## Bandwidth

Each packet carries the left image, so 848x480 is 407 kB plus 16 bytes per match.
At 30 Hz that is ~12 MB/s, which gigabit ethernet handles and wifi generally does
not. `--every N` sends only every Nth pair if the link cannot keep up; the frames
are dropped on the Jetson side of the pipe by this script simply reading slower,
so nothing queues without bound.

## Requires ROS 2 on this machine

rclpy, i.e. `sudo apt install ros-jazzy-rviz2` and friends, then source the setup
before running. See doc/07-tools.md.
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys

import numpy as np

FX = FY = 430.551
CX, CY = 427.381, 243.158
BASELINE_M = 0.049883
FB = FX * BASELINE_M
FRAME = "camera_link"
HDR = 16


def read_exactly(stream, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def colour_for(m: float) -> int:
    if m < 0.10:
        return (220 << 16) | (50 << 8) | 47
    if m < 0.30:
        return (203 << 16) | (145 << 8) | 20
    return (60 << 16) | (170 << 8) | 75


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="jetson")
    ap.add_argument("--remote-dir", default="doubleeye")
    ap.add_argument("--fast-threshold", type=int, default=12)
    ap.add_argument("--right-density", type=int, default=6)
    ap.add_argument("--min-margin", type=float, default=0.10)
    ap.add_argument("--exposure-us", type=int, default=0)
    ap.add_argument("--emitter", default=None, choices=[None, "on", "off"])
    ap.add_argument("--every", type=int, default=1,
                    help="publish every Nth pair, for a slow link")
    ap.add_argument("--dilate", type=int, default=3,
                    help="draw each sparse depth sample as an NxN block")
    ap.add_argument("--local", metavar="FILE", default=None,
                    help="read DEMR packets from a file instead of ssh, for "
                         "testing the publisher without the camera")
    a = ap.parse_args()

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
        from geometry_msgs.msg import TransformStamped
        from tf2_ros import StaticTransformBroadcaster
    except ImportError as e:
        sys.exit(f"ROS 2 Python not found ({e}).\n"
                 "  source /opt/ros/jazzy/setup.bash\n"
                 "and see doc/07-tools.md for the install.")

    if a.local:
        proc = None
        stream = open(a.local, "rb")
    else:
        remote = (f"cd ~/{a.remote_dir}/jetson/build && ./rs_ir_stream --streams both"
                  + (f" --exposure-us {a.exposure_us}" if a.exposure_us else "")
                  + (f" --emitter {a.emitter}" if a.emitter else "")
                  + f" | ~/{a.remote_dir}/core/build/de_pipe"
                    f" --fast-threshold {a.fast_threshold}"
                    f" --right-density {a.right_density}"
                    f" --min-margin {a.min_margin}")
        print(f"starting on {a.host}:\n  {remote}\n")
        proc = subprocess.Popen(["ssh", a.host, remote],
                                stdout=subprocess.PIPE, stderr=None,
                                bufsize=0)
        stream = proc.stdout

    rclpy.init()
    node = Node("doubleeye_live")
    # Best-effort: for a live view, a dropped frame is better than a stalled one,
    # and rviz's default subscription for sensor data is best-effort anyway --
    # mismatched reliability is a common reason topics appear but never display.
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
    pub_img = node.create_publisher(Image, "/doubleeye/image_raw", qos)
    pub_dep = node.create_publisher(Image, "/doubleeye/depth", qos)
    pub_pc = node.create_publisher(PointCloud2, "/doubleeye/points", qos)
    pub_ci = node.create_publisher(CameraInfo, "/doubleeye/camera_info", qos)

    tf_bc = StaticTransformBroadcaster(node)
    tfm = TransformStamped()
    tfm.header.stamp = node.get_clock().now().to_msg()
    tfm.header.frame_id = "map"
    tfm.child_frame_id = FRAME
    tfm.transform.rotation.w = 1.0
    tf_bc.sendTransform(tfm)

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="margin", offset=16, datatype=PointField.FLOAT32, count=1),
    ]

    k = max(1, a.dilate) // 2
    n_pub = n_seen = 0
    try:
        while rclpy.ok():
            hdr = read_exactly(stream, HDR)
            if hdr is None:
                print("\nstream ended")
                break
            if hdr[:4] != b"DEMR":
                print(f"lost sync: {hdr[:4]!r}", file=sys.stderr)
                break
            W, H, num, nm = struct.unpack("<HHII", hdr[4:])
            img_bytes = read_exactly(stream, W * H)
            rec_bytes = read_exactly(stream, nm * 16)
            if img_bytes is None or rec_bytes is None:
                break
            n_seen += 1
            if a.every > 1 and (n_seen % a.every) != 0:
                continue

            stamp = node.get_clock().now().to_msg()
            rec = np.frombuffer(rec_bytes, dtype=np.float32).reshape(-1, 4)
            xl, yl, disp, margin = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]
            ok = disp > 0.0
            xl, yl, disp, margin = xl[ok], yl[ok], disp[ok], margin[ok]

            m = Image()
            m.header.stamp = stamp
            m.header.frame_id = FRAME
            m.height, m.width = H, W
            m.encoding = "mono8"
            m.step = W
            m.data = img_bytes
            pub_img.publish(m)

            z = FB / disp
            depth = np.full((H, W), np.nan, dtype=np.float32)
            ui = np.clip(np.round(xl).astype(int), 0, W - 1)
            vi = np.clip(np.round(yl).astype(int), 0, H - 1)
            for u, v, zz in zip(ui, vi, z):
                depth[max(0, v - k):min(H, v + k + 1),
                      max(0, u - k):min(W, u + k + 1)] = zz
            d = Image()
            d.header.stamp = stamp
            d.header.frame_id = FRAME
            d.height, d.width = H, W
            d.encoding = "32FC1"
            d.step = 4 * W
            d.data = depth.tobytes()
            pub_dep.publish(d)

            x = (xl - CX) * z / FX
            y = (yl - CY) * z / FY
            rgb = np.array([colour_for(float(mm)) for mm in margin], dtype=np.uint32)
            arr = np.empty((len(z), 5), dtype=np.float32)
            arr[:, 0], arr[:, 1], arr[:, 2] = x, y, z
            arr[:, 3] = rgb.view(np.float32)
            arr[:, 4] = margin
            pc = PointCloud2()
            pc.header.stamp = stamp
            pc.header.frame_id = FRAME
            pc.height, pc.width = 1, len(z)
            pc.fields = fields
            pc.is_bigendian = False
            pc.point_step = 20
            pc.row_step = 20 * len(z)
            pc.is_dense = True
            pc.data = arr.tobytes()
            pub_pc.publish(pc)

            ci = CameraInfo()
            ci.header.stamp = stamp
            ci.header.frame_id = FRAME
            ci.height, ci.width = H, W
            ci.distortion_model = "plumb_bob"
            ci.d = [0.0] * 5
            ci.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
            ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            ci.p = [FX, 0.0, CX, 0.0, 0.0, FY, CY, 0.0, 0.0, 0.0, 1.0, 0.0]
            pub_ci.publish(ci)

            n_pub += 1
            if n_pub % 15 == 0:
                weak = float((margin < 0.2).mean()) if len(margin) else 0.0
                print(f"\rframe {num}  {len(z)} points  "
                      f"margin<0.2 {100*weak:.0f}%  published {n_pub}/{n_seen}",
                      end="", flush=True)
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if proc is not None:
            proc.terminate()
        else:
            stream.close()
    print(f"\npublished {n_pub} of {n_seen} pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
