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
| **848×480** | **64** | **31.5** | **31.7 Hz** |
| 848×480 | 65 | 47.9 | 20.9 Hz |
| 848×480 | 96 | 48.6 | 20.6 Hz |

**`D` is quantised into blocks of 64, so pick it at 64 and not below.** Anything from
1 to 64 costs the same — `--dmax 32` and `--dmax 64` are both 25.5 ms in the cost
stage — and 65 through 128 costs the same as *each other*, 52% more. There is nothing
to save by asking for fewer disparities than 64, and 64 is 4.5 points of coverage
better than 32. The only decision is whether you need to go past it, and that one
costs a third of the frame rate.

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

| topic | encoding | rviz display | what |
|---|---|---|---|
| `/doubleeye/points` | — | **PointCloud2** | the cloud. Start here |
| `/doubleeye/depth` | 32FC1 | **DepthCloud** or Image | a real depth map, valid where the matcher answered. With `--dense` that is ~78% of the frame, so DepthCloud is worth using; on the sparse path it is ~1% and shows nothing |
| `/doubleeye/image_raw` | mono8 | Image | the left image |
| `/doubleeye/depth_color` | rgb8 | Image | depth as a colour ramp |
| `/doubleeye/depth_dense` | rgb8 | Image | same, hole-filled. On the sparse path it triangulates between matches and **invents values**; with `--dense` the map is already dense, so it is the real depth colourised and nothing is invented |
| `/doubleeye/camera_info`, `/tf` | — | — | geometry |

**The encoding column is the one to read.** `DepthCloud` needs 16-bit or float, so it
accepts `/doubleeye/depth` and refuses the two `rgb8` topics — which are pictures of
depth, not depth. Pointing it at `depth_dense` is the obvious mistake, because the
name reads like the better depth map and it is the right choice for an *Image*
display.

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
- `--emitter on|off` — **the projector, and it is not optional.** With it off *and*
  the room dark, coverage collapses to 39% and the disparity map is noise. This
  matcher works on the projected dots; the room's own light mostly just lets you
  shorten the exposure.
- `--auto-contrast C` (3.5) — **the exposure controller, on by default.** It moves the
  exposure to hold C DN of median local contrast, which is the quantity the matcher
  cares about. `--exposure-us N` pins the exposure instead and turns it off.
- `--min-margin` — the same trade as §3, and **a viewer wants a different value from
  a benchmark**. The benchmarks run at 0.01 because they score accuracy *over the
  answered pixels*, so declining a doubtful pixel costs them nothing. On screen it
  costs holes. Measured on a real 848×480 pair:

  | gate | answered | hole area | largest single hole |
  |---|---|---|---|
  | **0** (dense default) | **88.4%** | **14.1%** | 8.4% of the frame |
  | 0.01 | 71.2% | 29.9% | 18.3% |
  | 0.10 (the sparse path's figure) | 10.3% | 89.7% | — |

  Left unset the script picks per mode — 0 for dense, 0.10 for sparse — and prints
  which. **Raise it for anything that triangulates**; leave it alone for looking.
- `--colour image|depth|confidence` (image) — `image` paints each point with the left
  camera's own intensity, so the cloud looks like the scene, which is what makes a
  wrong surface recognisable as a wrong surface. `depth` is a ramp over the 5–95
  percentile. `confidence` is green above 0.85, amber down to 0.60, red below, and
  grey where the matcher on the other end is old enough not to send one.
- `--min-confidence F` (0) — **drop points the matcher does not believe.** See the
  section below; it is also a live parameter, so the usual way to set it is not this
  flag.
- `--dense-stride N` (2) — subsample the cloud. 1 is 407k points and ~6 MB a frame.
- `--local FILE` — replay a captured packet stream instead of running the camera.

### Confidence, and how to filter by it

Each dense point carries `P(this point is within 1 disparity of correct)`, computed
from the two candidate scores the solver already ranked and sent as one byte a pixel.
The model, its measurement and what it does not establish are in
[TODO 0.553](TODO.md#0553-steps-3-and-4-measured-it-is-a-probability-and-it-is-over-confident-on-a-hard-scene);
the arithmetic is six lines in `core/include/doubleeye/dense_solve.hpp`.

**rviz2 cannot filter a point cloud by a field.** Its PointCloud2 display colours by
one — Color Transformer `Intensity`, Channel Name `confidence` — and the Min and Max
Intensity boxes only stretch the colour ramp. A point outside them is still drawn, in
the end colour. There is no threshold, no "hide below", and none of the other display
types does it either.

So the filtering is applied before publishing, and it is a ROS parameter so it can be
turned while you watch:

```sh
ros2 param set /doubleeye_live keep_best 0.4           # the best 40% of each frame
ros2 param set /doubleeye_live keep_best 0.0           # everything again
```

**Use `keep_best`, not `min_confidence`.** The confidence orders points well inside a
frame and says almost nothing about how good the frame is. Six captures of one kitchen
at three light levels, projector on and off, `de_dense --lrc`:

| | night, no projector | night, projector | evening, projector | daylight, projector |
|---|---|---|---|---|
| mean intensity | 16.0 DN | 21.0 DN | 56.1 DN | 95.1 DN |
| points failing the reverse-match check | 69.3% | 2.3% | 2.7% | 7.0% |
| mean confidence | 0.732 | 0.774 | 0.771 | 0.753 |
| kept by `min_confidence 0.85` | 6% | 17% | 18% | 14% |
| kept by `keep_best 0.4` | 40% | 40% | 40% | 40% |

The scene quality moves by a factor of thirty and the confidence moves by 0.04. A fixed
threshold therefore throws away most of a good cloud and keeps a sixth of a hopeless
one; a quantile keeps what was asked for either way. The absolute threshold is still
there, because a consumer that wants "only points I would act on" wants a number rather
than a fraction — but it is the one waiting on the calibration
[0.55](TODO.md#055-per-point-confidence-and-then-an-existence-probability--next-2026-08-12)
has not done on this camera:

```sh
ros2 param set /doubleeye_live min_confidence 0.85
```

Set the display to colour by `confidence` first, then lower `keep_best` until the red
goes and watch what leaves with it. Two things to expect. **A region with no
texture scores 0.667** — every disparity matches a constant descriptor equally well,
the matcher genuinely has no idea, and the number says so. **And a wrong match with no
competitor still scores high**, which is obstacle 24a's ghost: where the true partner
is off the sensor there is nothing for the winner to beat. Catching that needs the
reverse match, which the CPU tool has as `--lrc` and the GPU does not yet.

### Exposure: let it adapt

The right exposure is not a property of the camera, it is a property of the room, and
it moves a long way:

| | optimal exposure | optimal local contrast |
|---|---|---|
| a wall at 0.38 m | 250–350 us | 3.5–5.0 DN |
| a lit room | 2500–4000 us | 2.5–4.0 DN |
| the same room, unlit | 4000–6000 us | 2.5–3.7 DN |

**The exposure spans 24x and the contrast barely moves**, which is why the controller
targets contrast. Measured live: from a 1500 us start it settles at ~3450 us in the
unlit room and holds 3.5 DN, landing within 0.2 points of the hand-tuned optimum for
that room without being told anything about it.

**Do not use the camera's own auto-exposure.** It targets a well-exposed picture, mean
~80–95 DN, and the matcher wants ~20–45. Measured, it was the worst setting of
everything tried on the close wall — 84.3% against 90.5%, with 34x the gross outliers
— and merely adequate elsewhere. `rs_ir_capture --auto-exposure` exists to reproduce
that comparison, not to be used.

**An external infrared lamp will not help.** Room light is already a flood
illuminator, and switching it off costs 0.4 points once the exposure follows — 88.6%
against 88.2%. What the matcher is short of is projected *structure*, not photons. A
second pattern projector would add some; it would need a pattern that does not alias
with the D435's own. If you buy one, 850 nm is invisible so nothing makes you blink:
look for IEC 62471 Exempt Group, or Class 1 under IEC 60825 for laser-based units.
The D435's own projector is Class 1, and CCTV illuminators are frequently well above
it.

### Gigabit, not wifi

Every packet carries the left image. Sparse is ~4 MB/s at 10 fps; dense is ~12.
Wifi generally will not hold it.

### Measuring on the real camera

Middlebury cannot tell you anything about this sensor -- see [TODO.md](TODO.md) 0.45
for how far apart they are. A blank wall can:

```sh
./tools/deploy.sh --capture wall01 --seconds 3 --emitter on
./tools/deploy.sh --pull wall01
.venv/bin/python desktop/wall_check.py bags/wall01 \
    --sweep " " " --min-margin 0.01" " --no-subpixel"
```

It fits a plane and reports the scatter about it — the matcher's noise at that
distance — plus the gross-outlier fraction. `--sweep` compares settings **on the real
sensor**, which nothing else here can do. Each swept config needs a leading space or
argparse claims it.

A plane fit cannot see a **scale** error — multiply every depth by 1.05 and the plane
is exactly as flat. For that, record the same wall at three distances, measure them
with a tape, and pass them:

```bash
.venv/bin/python desktop/wall_check.py bags/w60 bags/w120 bags/w180 \
    --truth 0.60 1.20 1.80
```

It fits `measured = scale*ruler + offset`. Only the scale is a claim about the
matcher; the offset absorbs the distance from whatever you held the tape against to
the point depth is referenced from, so the tape does not have to start in the right
place. Two distances fit a line exactly and cannot be falsified — use three, and the
residual column is then worth reading.

Point at a blank wall filling the frame, and check the reported tilt is small — a
large one means you are measuring a slanted surface, or not a wall.

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

### A second ground, sloping away downwards under the real one

The search range ran out before the floor did. `--min-range` sets the nearest depth
searched, `dmax = f*B/min_range`, and anything nearer than that has **no correct
candidate in the set** -- at which point the matcher returns its best wrong one rather
than nothing. On a tiled floor the runner-up is the tile period, so the wrong answers
line up into a sheet instead of scattering into speckle.

Measured on the hallway bag: at the old 0.40 m default the sheet was 0.80% of points
and another 2.35% sat pinned flat at exactly 0.405 m; at `--min-range 0.335` (dmax 64)
those become 0.07% and 0.01%, and total coverage goes *up*. That is now the default.

If you still see it, the floor is nearer still -- lower `--min-range` further and watch
the near-point count rise. Two things that look like fixes and are not:

- **`--min-margin` does not gate it out.** 0.05 cuts the sheet by a fifth and the
  coverage by four fifths. The wrong match is confident, because the right one was
  never in the comparison.
- **Exposure and lighting do nothing for it.** It is not noise. If a defect survives
  a change of exposure unchanged, stop treating it as noise.


Ordered by how often each one has actually happened.

**The cloud is nearly empty.** The margin gate, almost always — see §5. At
`--min-margin 0.10` the dense matcher answers ~10% of pixels against ~88% ungated, on
the same frame.

**More holes than the pictures in the write-ups show.** Also the margin gate: 0.01
doubles the hole area against 0 (14.1% → 29.9%) on a real pair. The published figures
are ungated, so anything that adds a gate will look holier than they do.

**Holes that no setting removes.** Three kinds, and they are structural. The left
~64 columns can only match small disparities, because a pixel at *x* has no partner
beyond *x*−3 — that band is 56% holes and is 15% of all of them. The census border is
another 7%. The rest is texture: outside the left band, 69 blobs of ≥100 px hold 80%
of the hole area, and their median local contrast is 2.6 DN against 4.6 DN where the
matcher answered. Saturated windows and blank walls have nothing to match, and the
projector is what fixes that — `--emitter on`.

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

**"Depth image has invalid format (only 16 bit and float are supported)".** A
DepthCloud display pointed at `/doubleeye/depth_dense` or `/doubleeye/depth_color`,
which are `rgb8`. Use `/doubleeye/depth`, which is `32FC1`. See the table in §5.

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
