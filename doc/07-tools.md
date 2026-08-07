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
| `stereo_calibrate.py` | Calibrate the pair and compare to factory |
| `bag_to_rosbag.py` | Produce a rosbag for Kalibr |
| `allan_variance.py` | IMU noise parameters from a dataflash log |
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

### `stereo_calibrate.py`

```sh
.venv/bin/python desktop/stereo_calibrate.py bags/calib01_flat --pitch-mm 24.0
```

Fits both a pinhole model (distortion fixed at zero, matching what the ASIC
claims) and a radtan model, and compares both against the factory values. If
radtan fails to improve the reprojection error, its coefficients are fitting
noise and the pinhole numbers are the ones to use.

It also separates a scale error from an optics error, which is the useful part:
`fx` does not depend on the assumed square size while the recovered baseline
scales with it exactly, so a baseline that disagrees far more than fx implicates
the ruler. It reports the pitch implied by the factory baseline so the print scale
can be checked directly.

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

### `allan_variance.py`

```sh
ssh jetson '~/venvs/mavlink/bin/python ~/doubleeye/pixhawk/ardupilot_config.py \
  --download-log 3 --out /tmp/ap3.bin --max-mb 20'
ssh jetson 'base64 /tmp/ap3.bin' | base64 -d > bags/imu/ap3.bin
.venv/bin/python desktop/allan_variance.py bags/imu/ap3.bin
```

Reports the message inventory, which source it used, the sample rate measured from
the log's own timestamps, the Allan deviation per axis, and a Kalibr `imu.yaml`.

Three things it refuses to let you get wrong: it says when the data is **filtered**
(understated noise density by ~40% here), it computes the **validity limit** imposed
by the batch sampler's windowing and labels bias instability beyond it as an
artefact, and it checks that **stationary gravity reads 9.807 m/s²**.

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

## Pixhawk side — `pixhawk/`, Python 3.6 in the Jetson's venv

```sh
ssh jetson '~/venvs/mavlink/bin/python ~/doubleeye/pixhawk/ardupilot_config.py --report --check-sd'
ssh jetson '~/venvs/mavlink/bin/python ~/doubleeye/pixhawk/ardupilot_config.py --set SR0_RAW_SENS=100'
```

Reads and sets the ArduPilot parameters that matter here, and infers SD-card
presence from the logging health bits in `SYS_STATUS`. `--report` only reads;
writes happen solely for parameters named with `--set`, each verified by read-back.

Runs in `~/venvs/mavlink` on the Jetson — see [08-imu.md](08-imu.md) for how that
venv is bootstrapped on a box with no pip and no `ensurepip`. Must stay Python
3.6-compatible.

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

## Viewing results in rviz2

Two paths. The offline one works today; the live one needs the bridge described
below.

### What you need on the laptop

There is no ROS on this machine and there cannot be ROS 1: the desktop runs Ubuntu
24.04, where ROS 1 does not exist. So rviz2, from ROS 2 Jazzy:

    sudo apt install software-properties-common curl -y
    sudo add-apt-repository universe -y
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
      | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt update
    sudo apt install ros-jazzy-rviz2 ros-jazzy-ros2bag \
                     ros-jazzy-rosbag2-storage-mcap -y

Nothing is needed on the Jetson. That is deliberate: the capture path stays
ROS-free because preprocessing is already the binding constraint on memory
bandwidth, and JetPack 4.2 is Ubuntu 18.04, where the only ROS 2 releases are long
past end of life.

### Offline: a recorded bag

    core/build/de_match bags/full_on --right-density 6 --min-margin 0.10
    .venv/bin/python desktop/bag_to_ros2.py bags/full_on --dilate 3

Then:

    source /opt/ros/jazzy/setup.bash
    ros2 bag play bags/full_on_ros2 --loop
    rviz2

In rviz2: set **Fixed Frame** to `map`, then Add →

| display | topic | note |
|---|---|---|
| Image | `/doubleeye/image_raw` | the left IR frame |
| Image | `/doubleeye/depth` | sparse, see below |
| PointCloud2 | `/doubleeye/points` | set Color Transformer to `margin` |

The bag carries a `/tf_static` identity `map` → `camera_link`. Without it rviz2
refuses to draw anything and says so only obliquely, which is a long way to travel
for a missing transform.

**The depth image is sparse and that is correct.** This is a sparse matcher: it
produces a few hundred correspondences at keypoints, not a disparity per pixel.
Coverage is about 1.3% of pixels, so at `--dilate 1` it is nearly invisible on
screen. `--dilate 3` draws each sample as a 3x3 block, which invents nothing and
just makes it visible. If a dense depth image is wanted for its own sake, the D435
computes one in the ASIC; it is not what this pipeline produces.

**Colour by margin.** Points are coloured red below 0.10, amber to 0.30, green
above, and the raw margin rides along as a float field. Over eight Middlebury
scenes precision by margin quartile is 0.169 / 0.286 / 0.391 / 0.659, so the red
points really are the ones to distrust. Expect outliers at both ends of the
disparity gate: the Z range on `full_on` runs 0.10 m to 17.46 m, and both extremes
sit exactly at the gate limits (220 px and ~1.2 px), which is what a wrong match
at a gate boundary looks like. Filtering the cloud by margin removes most of them.

### Live: camera to rviz2

Detection and matching run **on the Jetson**, so what you see is the real
pipeline's output rather than a laptop reimplementation. Only the ROS 2 half runs
on the laptop, which keeps the capture path ROS-free.

    ssh jetson 'rs_ir_stream --emitter on | de_pipe ...'   capture + match
        |  DEMR packets over the ssh pipe
    desktop/de_live_ros2.py                                ROS 2 publisher
        |  /doubleeye/image_raw /depth /points /camera_info /tf
    rviz2

One command, which starts the remote side itself:

    source /opt/ros/jazzy/setup.bash
    python3 desktop/de_live_ros2.py --emitter on

then `rviz2`, Fixed Frame `map`, and the same three displays as the offline recipe.

Verified end to end on the Jetson: 101 pairs in 12 s at 10 fps/channel, 0 unpaired
dropped, ~570 matches per frame, sub-pixel disparity present in the packets.

**Bandwidth.** Each packet carries the left image, so 848x480 is 407 kB plus 16
bytes per match: ~4 MB/s at 10 fps, ~12 MB/s at 30. Gigabit ethernet is fine, wifi
generally is not. `--every N` publishes every Nth pair; the surplus is dropped by
this script reading slower, so nothing queues without bound.

**`rs_ir_stream` has no `--streams` flag.** It sends both channels by default and
`--single` restricts to ir1. Passing an unknown flag makes it print usage and exit
0, which downstream looks exactly like a camera that produced no frames -- the same
trap as cmake 3.10 in obstacle 10. If `de_pipe` reports `0 pairs`, check the
stream's stderr before suspecting the camera.

**QoS.** The publisher uses BEST_EFFORT, because rviz subscribes to sensor data
best-effort by default and a reliability mismatch is a common reason topics appear
in the list and never display anything.
