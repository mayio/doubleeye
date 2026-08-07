# Tool reference

Every tool, what question it answers, and how to run it. Grouped by where it runs,
because that split is the project's portability rule and not an accident.

Nothing here needs to be memorised — this page exists so that returning to the
project after a gap costs a read rather than a re-derivation.

## Jetson side — `jetson/`, C++14, librealsense2 only

Build and deploy from the desktop in one step, which is the only supported way:

```sh
./tools/deploy.sh
```

Building without syncing first relinks a stale binary while printing
`Built target`, so a just-written feature appears not to work. The helper couples
the two so they cannot be separated.

| Tool | Answers |
|---|---|
| `rs_probe` | Is the hardware healthy and what does the kernel expose? |
| `rs_ir_capture` | Record IR to disk with full timestamp instrumentation |
| `rs_ir_stream` | Feed a live view on the desktop |

### `rs_probe` — run this first, always

```sh
./tools/deploy.sh --probe
```

The gate. Fails loudly on the things that silently invalidate everything
downstream: power mode and unlocked clocks (measured at a 34% frame loss with no
error), USB2 fallback, and a missing 848×480 Y8 profile. Also reports factory
intrinsics and the IR baseline so `f·B` can be sanity-checked, and lists exactly
which frame metadata this kernel exposes.

Expect its **first** calibration read to fail and the retry to succeed — a
firmware warm-up quirk, not a calibration fault (obstacle 8).

### `rs_ir_capture` — the recorder

```sh
./tools/deploy.sh --capture NAME [options]
./tools/deploy.sh --pull NAME
```

`NAME` is a **bare name**, never a path. It always records to `~/bags/NAME` on the
Jetson. Passing `~/bags/NAME` would expand against the *desktop* home before ssh
saw it, which is exactly the bug that produced
`cannot create /home/<desktop-user>/bags/NAME`; the script now rejects slashes and
tildes outright.

Key options: `--seconds`, `--exposure-us` (1500), `--gain` (64),
`--emitter on|off|alternate`, `--save-every N`, `--width/--height/--fps`,
`--streams both|1|2`. Full table with the `--save-every` guidance in
[05-operations.md](05-operations.md).

Uses the low-level sensor API rather than `rs2::pipeline` on purpose: the
pipeline's syncer discards unmatched frames, and that discard is precisely the
signal being measured.

### `rs_ir_stream` — raw frames to stdout

Not usually run by hand; `live_view.py` drives it. Streams length-prefixed raw Y8
packets down a pipe. Measured link throughput is ~22 MB/s, so full 848×480 on both
channels at 10 Hz (8 MB/s) needs no compression — and the TX2 has no spare cycles
for encoding anyway.

This one *does* use `rs2::pipeline`, because for a preview the syncer's
left/right pairing is exactly what is wanted.

## Portable core — `core/`, C++14, standard library only

Depends on no platform library at all, so it builds with plain `make` on both
machines. That is the plan's portability rule made concrete.

```sh
cd core && make && make test          # 24 tests
./build/de_preprocess ../bags/NAME    # keypoints + Census -> keypoints.csv
```

`de_preprocess` also times each stage per frame, which is how the "299% of budget
on the TX2" figure was obtained. Run it on the Jetson for on-vehicle numbers; the
desktop figure is not the one that matters. Details in
[06-preprocessing.md](06-preprocessing.md).

Options: `--census 7x7|9x7`, `--cell`, `--per-cell`, `--min-response`,
`--min-local-std`, `--limit`, `--dump`.

## Desktop side — `desktop/`, Python 3.12 in `.venv`

```sh
python3 -m venv .venv && .venv/bin/pip install -r desktop/requirements.txt
```

| Tool | Answers |
|---|---|
| `live_view.py` | What is the camera seeing, right now? |
| `view_bag.py` | What did I record? |
| `capture_report.py` | Was the recording timing-sound? |
| `check_checkerboard.py` | Is this calibration set usable? |
| `check_planarity.py` | Was the board actually flat? |
| `view_keypoints.py` | Is the preprocessing output well distributed? |
| `make_checkerboard.py` | Produce a printable target |
| `bag_to_rosbag.py` | Produce a rosbag for Kalibr |
| `mavlink_probe.py` | What is the Pixhawk actually saying? |

### `live_view.py` — the interactive one

```sh
.venv/bin/python desktop/live_view.py                    # just look
.venv/bin/python desktop/live_view.py --collect calib01   # look and collect
```

Both IR channels live with corner detection overlaid, a status line (both
channels / one only / not detected), a coverage grid, and a "still needed" line
naming missing image regions, distance bands or tilted views.

`--collect NAME` auto-saves each genuinely new pose into a bag-shaped
`bags/NAME`, rejecting near-duplicates — fifty frames of one pose constrain
calibration no better than one while making the set look adequate.

Keys: `q` quit, `r` reset coverage, `s` save a pair as PNG, `space` pause.
Options: `--cols/--rows` (interior corners, default 9×6), `--gain` (96),
`--emitter` (off), `--out-fps` (10), `--no-detect`.

### `view_bag.py`

```sh
.venv/bin/python desktop/view_bag.py bags/NAME
```

Writes four views into `bags/NAME/view/`. The **anaglyph** is the one worth
learning: horizontal red/cyan fringing *is* the disparity and should grow on near
objects. Vertical fringing would mean the rectification is off or the channels are
not time-aligned.

Needs no OpenCV and no ffmpeg — GIFs go through matplotlib's Pillow writer.

### `capture_report.py`

```sh
.venv/bin/python desktop/capture_report.py bags/NAME
```

Delivered rate, frame-number continuity, L/R pairing, interval histogram,
timestamp domain and metadata availability, arrival jitter in milliseconds *and*
in pixels of misregistration, and local-contrast image statistics.

**Read its warnings.** It deliberately refuses conclusions it cannot support — it
will not do frame-gap analysis on a host-side counter, because that reports a
flawless run no matter how much was lost. That refusal exists because an earlier
version did not have it and reported `0 missing (0.000%)` on a recording losing a
third of its frames.

### `check_checkerboard.py`

```sh
.venv/bin/python desktop/check_checkerboard.py bags/NAME [--cols 9 --rows 6]
```

Detection **per channel** — a pair found in only one channel contributes nothing
to stereo extrinsics — plus image coverage and pose spread. Its failure hints are
ordered by likelihood, IR-transparent inkjet ink first.

### `check_planarity.py`

```sh
.venv/bin/python desktop/check_planarity.py bags/NAME [--max-rms 0.40] [--cull NAME_flat]
```

Catches a non-flat board, which is an error nothing else reveals: detection still
succeeds, the corner count is right, and reprojection error stays plausible
because the distortion model absorbs part of the bend.

Works by fitting a homography per detected board. Legitimate only because the IR
pair has **exactly zero distortion**, so a flat board must fit a homography
exactly and the residual is whatever is not planar.

Reports three independent signals rather than one verdict — residual versus
apparent board size, temporal clustering of failures, and residual smoothness —
and says so when they disagree. `--cull` writes the passing poses to a new bag.

### `view_keypoints.py`

```sh
.venv/bin/python desktop/view_keypoints.py bags/NAME [--stream 1|2] [--cell 32]
```

Overlay, per-cell occupancy heatmap, response and texture distributions, and
**Census descriptor diversity** — the count of distinct descriptors, which is how
the 3.3× degeneracy under the projector was found.

### `make_checkerboard.py`

```sh
.venv/bin/python desktop/make_checkerboard.py -o checkerboard.pdf
```

Scale-exact PDF. Prints the matching Kalibr yaml and OpenCV `Size`. Printing
cautions — laser not inkjet, 100% scale, measure the result, mount rigid — are in
[05-operations.md](05-operations.md).

### `bag_to_rosbag.py`

```sh
.venv/bin/python desktop/bag_to_rosbag.py bags/calib01_flat
```

Writes `bags/calib01_flat.bag` with `/cam0/image_raw` and `/cam1/image_raw`,
`sensor_msgs/Image` `mono8`, following Kalibr's topic convention. Prints the
`kalibr_calibrate_cameras` command and target yaml to go with it.

Uses `rosbags`, a pure-Python writer, so the desktop needs no ROS — which it
could not have anyway, since Ubuntu 24.04 has no ROS 1. Verified against real
ROS Melodic on the Jetson: `rosbag info` reports version 2.0 with the canonical
`sensor_msgs/Image` MD5.

**Timestamps.** Real host times are used when `frames.csv` is present. A set
collected by `live_view --collect` has images only, so stamps are synthesised on a
uniform grid — correct for `kalibr_calibrate_cameras`, which only needs a left and
right image of the same instant to share a stamp, but **not** valid for
`kalibr_calibrate_imu_camera`, which estimates a time offset. The tool warns when
it synthesises.

### `mavlink_probe.py`

```sh
ssh jetson 'sudo timeout 6 dd if=/dev/ttyACM0 bs=1 count=40000 2>/dev/null | base64' \
  | base64 -d > /tmp/mav.bin
.venv/bin/python desktop/mavlink_probe.py /tmp/mav.bin --seconds 6
```

Parses a dump rather than speaking live MAVLink, which keeps the vehicle side
dependency-free. Reports framing version, message histogram with rates, and
decodes `HEARTBEAT` to identify the firmware — needed because the USB descriptor
says "PX4 FMU v2.x" while the firmware is ArduPilot.

Its "percent of dump framed" figure is what settled the TELEM2 baud rate: 100% at
57600, 83.5% with nonsense system ids at 115200.

## Dependency notes

**OpenCV is desktop-only, from pip, and the full (non-headless) build** — the
headless wheel has no `imshow`, which `live_view.py` needs. The Jetson stays
OpenCV-free on purpose: it has a header/runtime version skew (headers 3.3.1,
runtime 3.2) and nothing on-vehicle needs it.

**`rsync` is not installed on the TX2.** All transfers are `tar` over `ssh`;
`deploy.sh --pull` wraps it.

**The `nvidia` user needed adding to `dialout`** to open `/dev/ttyACM0` and
`/dev/ttyTHS2`. Done; takes effect on new logins.

## Repeating a calibration session later

The whole recipe, from nothing:

```sh
# 1. print the target (once)
.venv/bin/python desktop/make_checkerboard.py -o checkerboard.pdf
#    laser printer, 100% scale, measure it, glue to something rigid

# 2. confirm the Jetson is healthy
./tools/deploy.sh --probe          # expect "clocks LOCKED", USB 3.2

# 3. collect, watching the screen
.venv/bin/python desktop/live_view.py --collect calib02
#    tilt 20-40 degrees, cover the corners, vary distance,
#    hold still ~1 s per pose, stop when "still needed" is empty

# 4. verify before trusting it
.venv/bin/python desktop/check_checkerboard.py bags/calib02
.venv/bin/python desktop/check_planarity.py bags/calib02 --cull calib02_flat
#    then calibrate on bags/calib02_flat, not the raw set
```

Step 4's second line matters: the first real session had 82/82 detection yet a
third of its poses carried a non-flat board. Detection success says nothing about
planarity.

Holding-and-moving technique is in
[05-operations.md](05-operations.md#how-to-hold-and-move-the-board). The single
most important point: **do not hold the board flat facing the camera**, because
fronto-parallel views leave focal length and distortion poorly constrained.
