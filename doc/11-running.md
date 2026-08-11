# Running it

What you can run, what the knobs do, and how to get a point cloud on screen.

[07-tools.md](07-tools.md) is the reference: every tool, one entry each. This page is
the other axis — organised by what you are trying to do. If you know which tool you
want, go there instead.

---

## 1. Start here

| I want to… | Run |
|---|---|
| see a live point cloud on the desktop | `de_live_ros2.py --dense --max-range 2` — [§5](#5-ros-2-and-rviz2) |
| see the camera without ROS | `live_view.py` — [§6](#6-the-other-viewers) |
| check the hardware is healthy | `./tools/deploy.sh --probe` — [§2](#2-deploy) |
| run the matcher on one image pair | `de_dense` — [§3](#3-the-matcher-offline) |
| know whether a change helped, on accuracy | `middeval3.py` — [§4](#4-measuring-a-change) |
| know whether a change helped, on speed | `tx2_ab.py` — [§4](#4-measuring-a-change) |
| record a bag | `deploy.sh --capture NAME` — [05-operations.md](05-operations.md) |

**Two Pythons, and they are not interchangeable.** `.venv/bin/python` for anything
using numpy, scipy or rosbags. **`/usr/bin/python3` for anything using `rclpy`** —
and write that path out, because an active `.venv` shadows the ROS install even when
the setup file has been sourced. The symptom is `No module named 'yaml'`, which reads
like a broken ROS installation and is not one.

---

## 2. Deploy

Everything on the Jetson is built from the desktop, in one step:

```sh
./tools/deploy.sh                 # sync jetson/ and core/, build both
./tools/deploy.sh --probe         # …and run rs_probe: is the hardware healthy?
./tools/deploy.sh --capture NAME  # …and record a bag into ~/bags/NAME on the board
./tools/deploy.sh --pull NAME     # fetch that bag back here
```

Building without syncing first relinks a stale binary while printing success, which
is why the two are coupled and why there is no separate "build" command.

`--capture NAME` takes a **bare name**, never a path: `~/bags/NAME` would expand
against the *desktop's* home before ssh saw it.

**`jetson_clocks` does not survive a reboot**, and losing it costs a third of your
frames with no error message. `doubleeye-performance.service` handles it; if it is
ever disabled, `rs_probe` and `tx2_ab.py` both say so rather than letting you measure
a throttled board.

---

## 3. The matcher, offline

```sh
cd core && make && make test        # 104 assertions, must stay at 0 failures
./build/de_dense L.y8 R.y8 848 480 --dmax 60 --out disp.f32
```

Input is raw 8-bit, no header — the format `rs_ir_capture` writes and `core/` reads
without an image library. Output is `W*H` float32 disparity; **NaN means the matcher
declined to answer**, which is a real answer and not a failure.

`./build/de_dense` with no arguments prints every flag with its measured effect.
The ones that decide the outcome:

### The one knob that matters most

`--min-margin` rejects a match whose best-minus-second score is below the threshold.
It does not make the matcher better or worse; it moves it along a curve:

| `--min-margin` | pixels answered | bad-1.0 |
|---|---|---|
| 0 | 89.4% | 31.3% |
| **0.01** (what the tools pass) | **79.6%** | **24.5%** |
| 0.03 | 64.8% | 16.3% |
| 0.10 | 35.7% | 6.2% |

Pick the end you need: tracking wants coverage, triangulation wants precision. Note
the **binary defaults to 0** while every harness passes 0.01 — so "the default" means
different things depending on what launched it.

### The rest, briefly

- `--dmax N` — disparities searched. On the GPU this costs in **steps of 64**, so
  `D=128` is free if you are already paying for 96.
- `--sigma-s` (8) — aggregation reach. Swept; 12 was past the optimum on both axes.
- `--ad` (0.15) — how much truncated absolute difference is blended into the Census
  score. Helps pick the right disparity, hurts the sub-pixel fit.
- `--iters` (0) — MASDA message-passing iterations. Zero is winner-take-all under the
  same one-to-one constraint, and it measures *better* while saving ~12 ms of CPU.
  The uniqueness constraint and the margin are present either way; only the message
  passing is off.
- `--no-subpixel` — integer output. Worth knowing what it costs: 17 points of
  bad-1.0.
- `--agg` — **does nothing** on the default path. The recursive filter ignores it.

Every one of those numbers is measured, and the measurement is in
[09-matching.md](09-matching.md) or [TODO.md](TODO.md).

### On the GPU

```sh
ssh jetson '~/doubleeye/core/build/de_dense_cuda L.y8 R.y8 848 480 --dmax 60 \
              --threads 4 --frames 30'
```

Same flags, same output bytes — the GPU tool is verified `cmp`-identical to
`de_dense --threads 1` on all eight ground-truth scenes, and that check runs after
every change. `--frames N` measures the pipelined steady state; a single frame is
~52 ms of latency, the pipeline is 31.7 ms of throughput.

`--keypoints out.csv` additionally detects keypoints and reads each one's disparity
out of the dense map. It costs 0.4–1.4 ms because detection hides under the kernels.

| resolution | D | ms/frame | rate |
|---|---|---|---|
| 424×240 | 60 | 8.7 | 115 Hz |
| 450×375 | 60 | 13.7 | 73 Hz |
| 640×480 | 60 | 24.1 | 42 Hz |
| **848×480** | **60** | **31.7** | **31.5 Hz** |
| 848×480 | 96 | 48.6 | 20.6 Hz |

---

## 4. Measuring a change

Two different questions, two different tools, and neither answers the other.

**Did accuracy change?**

```sh
export MIDDEVAL3=~/data/MiddEval3
.venv/bin/python article/middeval3.py --check --gt-full $MIDDEVAL3   # validate first
.venv/bin/python article/middeval3.py --gt-full $MIDDEVAL3 --threads 1
```

`--check` scores Middlebury's own published SGM output and must reproduce its public
row. Run it before believing anything else the tool says. `--threads 1` is the
configuration the GPU is verified against; multi-threaded runs lose a few sub-pixel
fits at thread seams and read ~1 point pessimistic.

`article/dense_bench.py` is the older eight-scene benchmark at native resolution. It
is *blind to sub-pixel accuracy* — a one-pixel threshold on the native grid cannot
resolve it — so a change that only moves the fit will read as nothing there.

**Did speed change?**

```sh
./tools/deploy.sh && ./tools/tx2_ab.py "" "--no-subpixel" -n 8
```

Never a before-and-after pair of runs: the TX2's run-to-run variance is 37%, so
`tx2_ab.py` interleaves the two configurations and compares minima. It also refuses
to measure an unlocked board or a binary older than your sources.

**The desktop is not a proxy for the TX2, in either direction.** int16 arithmetic was
neutral here and worth 20% there; a coarse-to-fine mask was worth 1.24× here and flat
there.

---

## 5. ROS 2 and rviz2

One command starts both halves — the capture and matching run on the Jetson, only the
ROS publisher runs here:

```sh
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 desktop/de_live_ros2.py --dense --emitter on --max-range 2
```

Then, in a second terminal with the same setup sourced, `rviz2`: Fixed Frame `map`,
and add a **PointCloud2** on `/doubleeye/points`.

`--max-range` is in that line rather than left at its default because it is the
difference between a cloud you can read and a blob — see the knobs below.

### Dense or sparse

| | `--dense` | default |
|---|---|---|
| runs on the board | `de_dense_cuda --stream` | `de_pipe` |
| points per frame | **~88,000** | ~570 |
| wire per frame | ~1.2 MB | ~0.4 MB |

`--dense-stride` (default 2) subsamples the cloud. Stride 1 is 407k points and ~6 MB
of `PointCloud2` a frame; stride 2 is a quarter of that and looks the same.

### Topics

| topic | what |
|---|---|
| `/doubleeye/points` | the cloud — this is the one to look at |
| `/doubleeye/image_raw` | the left image |
| `/doubleeye/depth` | 32FC1, valid where the matcher answered |
| `/doubleeye/depth_dense` | interpolated between matches. **Invents values**; for eyes only |
| `/doubleeye/camera_info`, `/tf` | geometry |

Derived topics are only computed when something subscribes, so opening every display
in rviz costs real bandwidth.

### Knobs

- `--min-range` / `--max-range` (0.4–6.0 m) — the depth gate, stated in metres and
  converted to disparity. **Set the far limit to roughly the depth of what you are
  looking at**, because a small far tail sets the whole cloud's scale and rviz fits
  its view to that: on a desk scene 1.4% of points beyond 2 m stretched the cloud
  from ±2.0 m to ±5.9 m across, and the 98.6% that mattered collapsed into the
  middle. `--max-range 2` indoors.
- `--every N` — publish every Nth pair. The surplus is dropped by reading slower, so
  nothing queues without bound.
- `--emitter on|off` — the projector. On is dramatically better: it puts texture on
  blank surfaces, and coverage goes from 78% to 88% on the same scene.
- `--min-margin` — the same trade as §3, applied live, and **the two matchers want
  different values**. The sparse path's tuned figure is 0.10; the dense matcher's own
  benchmarks all run at 0.01, and 0.10 there rejects about 90% of the map. Left
  unset, the script picks per mode and prints which. A cloud with a few thousand
  points instead of tens of thousands is this, every time.
- `--colour image|depth|margin` (image) — `image` paints each point with the left
  camera's own intensity, so the cloud looks like the scene. `depth` is a ramp over
  the 5–95 percentile. `margin` is the sparse path's confidence colouring and is
  meaningless with `--dense`, where the packet carries no per-pixel margin — every
  point comes out the same colour.
- `--dense-stride N` (2) — subsample the cloud. 1 is 407k points and ~6 MB a frame.
- `--local FILE` — replay a captured packet stream instead of running the camera.

### Gigabit, not wifi

Every packet carries the left image. Sparse is ~4 MB/s at 10 fps; dense is ~12.
Wifi generally will not hold it.

---

## 6. The other viewers

Three, none of which need ROS:

```sh
.venv/bin/python desktop/live_view.py                  # live, both channels
.venv/bin/python desktop/live_view.py --collect NAME   # …and save new poses
.venv/bin/python desktop/view_bag.py bags/NAME         # a recorded bag
.venv/bin/python desktop/view_keypoints.py bags/NAME   # detector output
```

`live_view.py` is the calibration tool: both IR channels with corner detection
overlaid, a coverage grid, and a "still needed" line naming the image regions,
distance bands and view angles the set is missing. `--collect` auto-saves genuinely
new poses and rejects near-duplicates.

`view_bag.py` and `view_keypoints.py` are for looking at what was recorded and
whether the detector spread its keypoints out.

---

## 7. When it looks broken

Ordered by how often each one has actually happened.

**The cloud is nearly empty, or too sparse to read.** The margin gate, almost always
— see §5. At `--min-margin 0.10` the dense matcher answers ~10% of pixels against
~88% at 0.01, on the same frame. Thousands of points where there should be tens of
thousands is this.

**The cloud has points but no visible structure.** The far tail sets the scale. On a
desk scene, 1.4% of the points sat beyond 2 m and stretched the cloud from ±2.0 m to
±5.9 m across; rviz fits the view to the full extent, so the 98.6% that is the actual
scene collapses into the middle. Set `--max-range` to roughly the depth of what you
are looking at — 2 m indoors at a desk — and it resolves.

**The cloud is one flat colour.** `--colour` defaults to `image`, which paints each
point with the left camera's own intensity. `margin` is the sparse path's scheme and
is meaningless in dense mode, where the packet carries no per-pixel confidence.

**The cloud is empty.** The depth gate. `--min-range`/`--max-range` default to
0.4–6.0 m and everything outside is dropped before publishing. A pair with little
real disparity puts the whole scene past the far limit.

**`No module named 'yaml'` or `rclpy`.** The `.venv` is active and shadowing the ROS
install. Use `/usr/bin/python3` explicitly. Sourcing the setup file does not fix it.

**Topics appear in rviz but never display.** QoS. The publisher must be at least as
strong as the subscriber, and rviz requests RELIABLE by default. This script
publishes RELIABLE for that reason; `--best-effort` exists but then every rviz
display needs its Reliability dropdown changed.

**A flag "does nothing".** Check the tool did not print usage and exit 0 —
`rs_ir_stream` has no `--streams` flag, and an unknown flag there looks downstream
exactly like a camera producing no frames. Read the producer's stderr before
believing the consumer's silence.

**A change measures as no change.** Check it is actually in the binary. Three
separate instances of this: a header missing from one build rule, and two benchmark
harnesses passing a flag with values of their own over the top of the binary's
default. All three were caught by a number coming back *exactly* equal to the figure
it should have moved from.

**Timings are wild.** Clocks unlocked, or the board is hot. `tx2_ab.py` checks both
and says so. `jetson_clocks` does not survive a reboot.

**`cmp` says two runs differ.** Multi-threaded runs are not bit-comparable — the
top-2 insert keeps the first of two equal scores and work is handed out dynamically.
Bit-identity is checked at `--threads 1`, and only there.
