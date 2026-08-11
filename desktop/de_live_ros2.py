#!/usr/bin/env python3
"""Live view in rviz2: Jetson matches, laptop publishes.

    ros2 run ... no. Just:  source /opt/ros/jazzy/setup.bash && python3 de_live_ros2.py

Pipeline, all of it started by this script:

    ssh jetson 'rs_ir_stream --emitter on | de_pipe ...'     <- capture + match
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
# Points come out of the projection in the OPTICAL convention: X right, Y down,
# Z forward. rviz's world is REP-103: X forward, Y left, Z up. Publishing optical
# data in a body frame makes a correct cloud look wrong -- lying on its side, which
# reads as "random points" as easily as real noise does. So the data is labelled
# camera_optical_frame and a static transform carries the rotation, which is the
# convention every camera driver follows.
FRAME = "camera_optical_frame"
BODY = "camera_link"
# The standard camera_link -> camera_optical_frame quaternion.
OPTICAL_Q = (-0.5, 0.5, -0.5, 0.5)   # x, y, z, w
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


# A short perceptually-increasing ramp. Not turbo, but monotone in lightness and
# enough to read depth off, which grayscale with NaN holes is not.
_RAMP = np.array([
    (48, 18, 59), (70, 90, 200), (35, 170, 200), (60, 200, 120),
    (180, 215, 60), (250, 190, 50), (240, 110, 40), (160, 25, 20),
], dtype=np.float32)


def colourise(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map v in [lo, hi] to RGB. NaN and out-of-range become dark grey."""
    bad = ~np.isfinite(v)
    # Sanitise BEFORE indexing, not after. NaN survives np.clip, and
    # np.floor(nan).astype(int) is INT_MIN, which indexes out of bounds -- the
    # guard below cannot run because the lookup has already raised.
    t = np.clip((np.where(bad, lo, v) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    idx = t * (len(_RAMP) - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, len(_RAMP) - 1)
    f = (idx - i0)[..., None]
    out = _RAMP[i0] * (1 - f) + _RAMP[i1] * f
    out[bad] = (40, 40, 40)
    return out.astype(np.uint8)


def stretch(img: np.ndarray, sub: int = 4) -> np.ndarray:
    """Linear p1-p99 contrast stretch, for display only.

    Percentiles are taken on every `sub`-th pixel. On a full 848x480 frame
    np.percentile costs 12.7 ms, which was the single largest per-frame cost in
    this script -- larger than everything else combined. A 1/16 subsample gives
    the same two numbers to well within a grey level and costs 0.2 ms.
    """
    small = img[::sub, ::sub]
    lo, hi = np.percentile(small, (1.0, 99.0))
    if hi - lo < 1e-6:
        return img
    # Apply through a 256-entry lookup table rather than converting the whole
    # frame to float. The input is uint8, so every possible value is already
    # enumerated: the map costs 256 operations plus one gather, against 407k
    # float multiplies and a clip. That is 6.9 ms down to ~0.3 ms, and it is the
    # difference between this being the dominant per-frame cost and being
    # invisible.
    lut = np.clip((np.arange(256, dtype=np.float32) - lo) * (255.0 / (hi - lo)),
                  0, 255).astype(np.uint8)
    return lut[img]


def densify(H, W, ui, vi, z, stride=8):
    """Triangulate the matched points and interpolate depth between them.

    A sparse depth image is ~1% of pixels and unreadable: rviz has nothing to
    render and NaN dominates. Linear interpolation over a Delaunay triangulation
    of the matches turns it into a surface you can actually see, which is the same
    support-point idea ELAS uses as a prior for dense estimation.

    This INVENTS values between measurements. It is a visualisation, not data, and
    it is published on its own topic for that reason -- /doubleeye/depth stays
    sparse and true. Interpolation is done at `stride` and upsampled, because
    querying 407k points against the triangulation costs far more than it is worth
    for something being looked at rather than measured. Stride 8 is 12.6 ms against
    457 ms at stride 4, for identical coverage -- the convex hull does not change,
    only how finely it is sampled.
    """
    out = np.full((H, W), np.nan, np.float32)
    if len(z) < 4:
        return out
    try:
        from scipy.interpolate import LinearNDInterpolator
    except ImportError:
        return out
    f = LinearNDInterpolator(np.stack([ui, vi], 1).astype(np.float64),
                             z.astype(np.float64))
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    gx, gy = np.meshgrid(xs, ys)
    small = f(gx, gy)
    # Nearest-neighbour upsample back to full resolution.
    yi = np.minimum((np.arange(H) // stride), small.shape[0] - 1)
    xi = np.minimum((np.arange(W) // stride), small.shape[1] - 1)
    return small[np.ix_(yi, xi)].astype(np.float32)


def colours_for(m: np.ndarray) -> np.ndarray:
    """Per-point RGB by margin, vectorised. (N, 3) uint8."""
    c = np.empty((len(m), 3), np.uint8)
    c[:] = (60, 170, 75)
    c[m < 0.30] = (203, 145, 20)
    c[m < 0.10] = (220, 50, 47)
    return c


def splat(dst, ui, vi, values, half):
    """Write `values` into `dst` as (2*half+1)^2 blocks, without a Python loop.

    Fancy indexing over an (N, k, k) index grid. The loop version cost 1.5-1.8 ms
    per frame per image; this is about 0.1 ms.
    """
    H, W = dst.shape[:2]
    off = np.arange(-half, half + 1)
    vv = np.clip(vi[:, None] + off[None, :], 0, H - 1)
    uu = np.clip(ui[:, None] + off[None, :], 0, W - 1)
    V = np.repeat(vv[:, :, None], len(off), axis=2)
    U = np.repeat(uu[:, None, :], len(off), axis=1)
    if dst.ndim == 3:
        dst[V, U] = values[:, None, None, :]
    else:
        dst[V, U] = values[:, None, None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="jetson")
    ap.add_argument("--remote-dir", default="doubleeye")
    ap.add_argument("--fast-threshold", type=int, default=12)
    ap.add_argument("--right-density", type=int, default=6)
    # No default here: the two matchers want different values and one number
    # cannot serve both. 0.10 is the sparse path's tuned figure; the dense
    # matcher's own benchmarks all run at 0.01, and 0.10 there rejects ~96% of
    # the map -- which looks exactly like a matcher that found nothing.
    ap.add_argument("--min-margin", type=float, default=None)
    ap.add_argument("--exposure-us", type=int, default=0)
    ap.add_argument("--emitter", default=None, choices=[None, "on", "off"])
    ap.add_argument("--out-fps", type=float, default=10.0,
                    help="frames/s per channel off the camera (default 10). "
                         "Each packet carries the left image, so 30 fps is "
                         "~12 MB/s over the ssh pipe")
    ap.add_argument("--min-range", type=float, default=0.4,
                    help="nearest depth to search, metres (default 0.4). f*B is "
                         "21.48 px*m, so the matcher's default [1, 220] px gate "
                         "spans 0.10-21.5 m and wrong matches pile up against "
                         "both limits: a quarter come out nearer than 0.19 m")
    ap.add_argument("--max-range", type=float, default=6.0,
                    help="farthest depth to search, metres (default 6.0)")
    ap.add_argument("--cell", type=int, default=0,
                    help="detector grid cell in px (0 = matcher default 32). "
                         "Lower means denser keypoints: 12 gives ~3x more "
                         "matches. Whether that makes the depth image better is "
                         "NOT established -- see doc/07-tools.md")
    ap.add_argument("--per-cell", type=int, default=0,
                    help="keypoints kept per grid cell (0 = default 3)")
    ap.add_argument("--no-stretch", action="store_true",
                    help="publish the IR image raw. By default it is contrast "
                         "stretched p1-p99 for display, because 1500 us of IR "
                         "exposure is dark and rviz does not auto-scale mono8")
    ap.add_argument("--dense", action="store_true",
                    help="run the dense CUDA matcher on the Jetson instead of the "
                         "sparse keypoint one: ~80% of pixels answered rather than "
                         "~570 points, at 848x480 on the GPU")
    ap.add_argument("--colour", default="image",
                    choices=["image", "depth", "margin"],
                    help="how to colour the cloud. image: the left camera's own "
                         "intensity, so the cloud looks like the scene. depth: a "
                         "ramp over the 5-95 percentile. margin: the matcher's "
                         "confidence, which the dense packet does not carry")
    ap.add_argument("--dense-stride", type=int, default=2,
                    help="subsample the dense cloud by this factor in each axis. "
                         "1 is 407k points a frame and ~6 MB of PointCloud2; 2 is "
                         "a quarter of that and looks the same in rviz")
    ap.add_argument("--best-effort", action="store_true",
                    help="publish BEST_EFFORT instead of RELIABLE. Only for a "
                         "lossy link, and rviz displays then need their "
                         "Reliability Policy set to Best Effort to match")
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
        if a.min_margin is None:
            a.min_margin = 0.0 if a.dense else 0.10
        stream = open(a.local, "rb")
    else:
        # No --streams flag: rs_ir_stream emits both channels by default and
        # --single restricts to ir1. Passing one makes it print usage and exit 0,
        # which downstream looks exactly like a camera producing nothing. See
        # doc/03-obstacles.md obstacle 17 -- which this script was violating while
        # the obstacle was being written.
        # Gate in metres, converted here, so the range is stated in the units the
        # scene has rather than in pixels of disparity.
        if a.min_margin is None:
            # A viewer and a benchmark want opposite things here. The benchmarks run
            # at 0.01 because they score accuracy OVER THE ANSWERED PIXELS, so
            # declining a doubtful pixel is free to them. On screen it is not free:
            # on a real 848x480 pair, 0.01 takes coverage 88.4% -> 71.2% and doubles
            # the hole area, 14.1% -> 29.9%, with the largest single hole going from
            # 8.4% of the frame to 18.3%. The gross outliers the gate would have
            # caught are removed by --max-range anyway.
            #
            # So: no gate for looking, and raise it for anything that triangulates.
            a.min_margin = 0.0 if a.dense else 0.10
            why = ("dense, tuned for coverage -- raise it for anything that "
                   "triangulates") if a.dense else "sparse default"
            print(f"margin gate: {a.min_margin}  ({why})")
        dmin = FB / max(a.max_range, 1e-3)
        dmax = FB / max(a.min_range, 1e-3)
        print(f"depth gate: {a.min_range:.2f}-{a.max_range:.2f} m "
              f"= disparity {dmin:.1f}-{dmax:.1f} px")
        remote = (f"cd ~/{a.remote_dir}/jetson/build && ./rs_ir_stream"
                  + f" --out-fps {a.out_fps}"
                  + (f" --exposure-us {a.exposure_us}" if a.exposure_us else "")
                  + (f" --emitter {a.emitter}" if a.emitter else "")
                  + (f" | ~/{a.remote_dir}/core/build/de_dense_cuda"
                     f" /dev/null /dev/null 848 480 --stream --threads 4"
                     f" --dmax {int(min(dmax, 96)) if dmax > 8 else 60}"
                     f" --min-margin {a.min_margin}"
                     if a.dense else
                     f" | ~/{a.remote_dir}/core/build/de_pipe"
                     f" --fast-threshold {a.fast_threshold}"
                     f" --right-density {a.right_density}"
                     f" --min-margin {a.min_margin}"
                     f" --min-disparity {dmin:.3f}"
                     f" --max-disparity {dmax:.3f}"
                     + (f" --cell {a.cell}" if a.cell else "")
                     + (f" --per-cell {a.per_cell}" if a.per_cell else "")))
        print(f"starting on {a.host}:\n  {remote}\n")
        proc = subprocess.Popen(["ssh", a.host, remote],
                                stdout=subprocess.PIPE, stderr=None,
                                bufsize=0)
        stream = proc.stdout

    rclpy.init()
    node = Node("doubleeye_live")
    # RELIABLE by default, which is the only choice that works with rviz2 out of
    # the box. DDS compatibility is one-directional: a publisher must be at least
    # as strong as the subscriber, so
    #
    #   publisher RELIABLE    + subscriber RELIABLE or BEST_EFFORT   -> both fine
    #   publisher BEST_EFFORT + subscriber RELIABLE                  -> INCOMPATIBLE
    #
    # and rviz2's displays request RELIABLE unless you change the Reliability
    # Policy dropdown per display. Publishing BEST_EFFORT therefore produced
    # "requesting incompatible QoS ... Last incompatible policy: RELIABILITY" and
    # no images at all. A RELIABLE publisher is compatible with everything, so it
    # is the right default even for a live stream.
    #
    # Back-pressure is not a problem here: a slow consumer slows this loop, which
    # slows the ssh read, which drops frames at the camera. That is the intended
    # behaviour rather than an unbounded queue. --best-effort is available for a
    # lossy link, but then rviz needs its dropdown changed to match.
    qos = QoSProfile(
        reliability=(ReliabilityPolicy.BEST_EFFORT if a.best_effort
                     else ReliabilityPolicy.RELIABLE),
        history=HistoryPolicy.KEEP_LAST, depth=5)
    print(f"QoS: {'BEST_EFFORT (set rviz display Reliability to Best Effort)' if a.best_effort else 'RELIABLE (works with rviz2 defaults)'}")
    pub_img = node.create_publisher(Image, "/doubleeye/image_raw", qos)
    pub_dep = node.create_publisher(Image, "/doubleeye/depth", qos)
    pub_pc = node.create_publisher(PointCloud2, "/doubleeye/points", qos)
    pub_ci = node.create_publisher(CameraInfo, "/doubleeye/camera_info", qos)
    # Two derived views, because the raw ones are hard to read. /image_matches
    # shows what the matcher actually did, which is the thing worth looking at;
    # /depth_color makes a 1.3%-coverage depth map legible where a 32FC1 image
    # full of NaN is not.
    pub_ovl = node.create_publisher(Image, "/doubleeye/image_matches", qos)
    pub_dcol = node.create_publisher(Image, "/doubleeye/depth_color", qos)
    # Interpolated between matches, so it shows surfaces rather than dots. Its own
    # topic because it is a rendering, not a measurement: the values between
    # support points were never observed.
    pub_ddense = node.create_publisher(Image, "/doubleeye/depth_dense", qos)

    tf_bc = StaticTransformBroadcaster(node)
    now = node.get_clock().now().to_msg()
    t_map = TransformStamped()
    t_map.header.stamp = now
    t_map.header.frame_id = "map"
    t_map.child_frame_id = BODY
    t_map.transform.rotation.w = 1.0
    t_opt = TransformStamped()
    t_opt.header.stamp = now
    t_opt.header.frame_id = BODY
    t_opt.child_frame_id = FRAME
    (t_opt.transform.rotation.x, t_opt.transform.rotation.y,
     t_opt.transform.rotation.z, t_opt.transform.rotation.w) = OPTICAL_Q
    tf_bc.sendTransform([t_map, t_opt])

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
            want = b"DEDD" if a.dense else b"DEMR"
            if hdr[:4] != want:
                print(f"lost sync: {hdr[:4]!r}, expected {want!r}", file=sys.stderr)
                break
            W, H, num, nm = struct.unpack("<HHII", hdr[4:])
            img_bytes = read_exactly(stream, W * H)
            # The dense packet's second field is a full W*H map of Q4 int16 rather
            # than nm records; nm is zero there and carries nothing.
            rec_bytes = read_exactly(stream, W * H * 2 if a.dense else nm * 16)
            if img_bytes is None or rec_bytes is None:
                break
            n_seen += 1
            if a.every > 1 and (n_seen % a.every) != 0:
                continue

            stamp = node.get_clock().now().to_msg()
            if a.dense:
                # Unpack the map into the same (x, y, disparity) columns the sparse
                # path produces, so everything downstream -- the gate, the depth
                # images, the cloud -- is shared rather than duplicated. -32768 is
                # the matcher's "no answer"; Q4 means sixteenths of a pixel.
                q = np.frombuffer(rec_bytes, dtype=np.int16).reshape(H, W)
                st = max(1, a.dense_stride)
                q = q[::st, ::st]
                vy, vx = np.nonzero(q != -32768)
                xl = (vx * st).astype(np.float32)
                yl = (vy * st).astype(np.float32)
                disp = q[vy, vx].astype(np.float32) / 16.0
                margin = np.ones_like(disp)      # the map does not carry it
            else:
                rec = np.frombuffer(rec_bytes, dtype=np.float32).reshape(-1, 4)
                xl, yl, disp, margin = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]
            # Also filter here, not only in the matcher's gate. A disparity that
            # survives the gate at its very edge is still almost certainly wrong,
            # and one absurd point rescales rviz's whole view.
            z_all = np.divide(FB, disp, out=np.zeros_like(disp),
                              where=disp > 0.0)
            ok = (disp > 0.0) & (z_all >= a.min_range) & (z_all <= a.max_range)
            n_drop = int((~ok).sum())
            xl, yl, disp, margin = xl[ok], yl[ok], disp[ok], margin[ok]

            # Only build what someone is displaying. Six topics at 848x480 is
            # 4.5 MB per frame -- 45 MB/s at 10 fps, all of it serialised and
            # pushed through DDS whether rviz shows it or not. Two rgb8 images are
            # 1.2 MB each and the 32FC1 depth is 1.6 MB. Checking the subscription
            # count first skips both the CPU and the bytes for anything not open,
            # which is what made this slow after the derived views were added.
            want_img = pub_img.get_subscription_count() > 0
            want_ovl = pub_ovl.get_subscription_count() > 0
            want_dep = pub_dep.get_subscription_count() > 0
            want_dcol = pub_dcol.get_subscription_count() > 0
            want_dense = pub_ddense.get_subscription_count() > 0
            want_pc = pub_pc.get_subscription_count() > 0
            need_gray = want_img or want_ovl
            need_depth = want_dep or want_dcol or want_dense

            gray = np.frombuffer(img_bytes, dtype=np.uint8).reshape(H, W)
            shown = (gray if (a.no_stretch or not need_gray)
                     else stretch(gray))
            if want_img:
                m = Image()
                m.header.stamp = stamp
                m.header.frame_id = FRAME
                m.height, m.width = H, W
                m.encoding = "mono8"
                m.step = W
                m.data = shown.tobytes()
                pub_img.publish(m)

            z = FB / disp
            ui = np.clip(np.round(xl).astype(int), 0, W - 1)
            vi = np.clip(np.round(yl).astype(int), 0, H - 1)
            depth = np.full((H, W), np.nan, dtype=np.float32)
            if need_depth and len(z):
                splat(depth, ui, vi, z, k)
            if want_dep:
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
            # Colour by margin only when the margin means something. The dense
            # packet carries a disparity map and no per-pixel confidence, so every
            # point would come back the same green and 88,000 of them read as one
            # flat blob. Colouring by the left image's own intensity instead makes
            # the cloud look like the scene, which is the point of looking at it.
            if a.dense and a.colour != "margin":
                gi = gray[yl.astype(np.int32), xl.astype(np.int32)]
                if a.colour == "image":
                    g8 = stretch(gray)[yl.astype(np.int32), xl.astype(np.int32)] \
                         if not a.no_stretch else gi
                    cols = np.repeat(g8[:, None], 3, axis=1).astype(np.uint8)
                else:                                    # "depth"
                    lo, hi = np.percentile(z, 5), np.percentile(z, 95)
                    cols = colourise(z, float(lo), float(hi))
            else:
                cols = colours_for(margin)
            packed = ((cols[:, 0].astype(np.uint32) << 16)
                      | (cols[:, 1].astype(np.uint32) << 8)
                      | cols[:, 2].astype(np.uint32))
            arr = np.empty((len(z), 5), dtype=np.float32)
            arr[:, 0], arr[:, 1], arr[:, 2] = x, y, z
            arr[:, 3] = packed.view(np.float32)
            arr[:, 4] = margin
            if want_pc:
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

            # Matches drawn on the image: a 3x3 dot per match, coloured the same
            # way as the cloud. This is the view that makes the output legible --
            # you can see at a glance whether the matcher is covering the scene or
            # clustering on one texture, which neither the raw IR nor a sparse
            # depth map shows.
            if want_ovl:
                ovl = np.repeat(shown[:, :, None], 3, axis=2)
                if len(z):
                    splat(ovl, ui, vi, cols, 1)
                om = Image()
                om.header.stamp = stamp
                om.header.frame_id = FRAME
                om.height, om.width = H, W
                om.encoding = "rgb8"
                om.step = 3 * W
                om.data = ovl.tobytes()
                pub_ovl.publish(om)

            # Colourised depth. rviz renders 32FC1 as grayscale normalised over
            # the frame, which with ~1% coverage and NaN everywhere else is close
            # to unreadable.
            #
            # Scaled to the data, not to the gate. Scaling to the gate put a room
            # whose content sits in 0.4-2 m into the first quarter of the ramp, so
            # everything came out the same colour -- technically correct and
            # useless. p5-p95 of the points actually present, with a floor on the
            # span so a flat wall does not get amplified into false structure.
            if len(z):
                d_lo, d_hi = np.percentile(z, (5.0, 95.0))
                if d_hi - d_lo < 0.25:
                    mid = 0.5 * (d_lo + d_hi)
                    d_lo, d_hi = mid - 0.125, mid + 0.125
            else:
                d_lo, d_hi = a.min_range, a.max_range
            dc = (colourise(depth, float(d_lo), float(d_hi)) if want_dcol
                  else None)
            if want_dcol:
                dm = Image()
                dm.header.stamp = stamp
                dm.header.frame_id = FRAME
                dm.height, dm.width = H, W
                dm.encoding = "rgb8"
                dm.step = 3 * W
                dm.data = dc.tobytes()
                pub_dcol.publish(dm)

            if want_dense and len(z):
                # densify() exists to turn ~570 scattered matches into a surface by
                # triangulating them. With --dense the map already IS a surface --
                # ~86% of pixels carry a measurement -- so triangulating 88,000
                # points would spend real time reconstructing what is already there,
                # and would invent values across the holes rather than showing them.
                dense = depth if a.dense else densify(H, W, ui, vi, z)
                nm_ = Image()
                nm_.header.stamp = stamp
                nm_.header.frame_id = FRAME
                nm_.height, nm_.width = H, W
                nm_.encoding = "rgb8"
                nm_.step = 3 * W
                nm_.data = colourise(dense, float(d_lo), float(d_hi)).tobytes()
                pub_ddense.publish(nm_)

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
                      f"margin<0.2 {100*weak:.0f}%  Z {d_lo:.2f}-{d_hi:.2f} m "
                      f"(colour range)  dropped {n_drop}  "
                      f"published {n_pub}/{n_seen}",
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
