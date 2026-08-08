# TODO

Ordered by what unblocks the most, not by effort. Each item says *why* it matters
so it can be picked up cold.

Status of the bring-up sequence lives in [README.md](README.md); this is the
actionable list.

---

## 0. Architecture: what runs on the GPU, what runs on the CPU — **first priority**

**Deferred by decision, not forgotten.** Mario has postponed this; it stays at the
top of the list because it constrains the shape of everything after it.

The reasoning, the measurements and a proposed answer are already written up in
[10-architecture.md](10-architecture.md). Nothing needs to be re-derived; what is
outstanding is the decision itself.

**Why it is first.** The system is meant to grow object tracking, SLAM, ground
detection and trajectory estimation. Each of those consumes the sparse feature set,
and the boundary between GPU and CPU determines what the stages hand each other.
Choosing it is one page of reasoning; retrofitting it means rewriting the interfaces
between every stage.

**What is already measured**, so the decision does not need new data:

- 2 of 6 CPU cores in use. GPU load 0. 23.07 ms of a 33.3 ms budget.
- Dense per-pixel work is 12.29 ms of that (FAST 8.30, NMS 3.74, Census 0.25).
- Sparse irregular work is 7.86 ms (MASDA).

**The proposal on the table**: GPU owns the image plane, CPU owns the graph, and the
interface between them is a compact keypoint-plus-descriptor buffer rather than an
image. No CUDA yet, because there is 10 ms of slack and four idle cores, and because
a CNN for object detection would want the GPU to itself.

**The one piece of work that should not wait for the decision**, since it is
expensive either way: every tool currently re-runs detection itself. With four
consumers that is four redundant detections. Make the sparse feature set a single
first-class output with one producer.

---

## 0.3 Dense MASDA: level with SGM on quality, 13x behind on runtime

State after 2026-08-08. `core/tools/de_dense.cpp`, eight Middlebury scenes:

| | coverage | bad-1.0 | runtime |
|---|---|---|---|
| dense MASDA | 76.7% | **11.0%** | 214 ms |
| SGM | 78.0% | **10.9%** | 16 ms |

Quality started the day 3.5x behind (28.1% against 8.1%) and is now a tie. What
worked: window aggregation (26.6% -> 10.6%), guided-filter edge-aware support
(10.6% -> 8.8%), margin gate for the operating point. What did not: sub-pixel
parabola, fast guided filter, disparity blocking (6% only). All measured, all
recorded in 09-matching.md with mechanisms.

**Runtime is now the whole gap.** Two levers left, and the lesson from the fast
guided filter is that they must not touch the cost values:

1. ~~uint16 cost volume~~ — **measured and dropped.** The machine sustains
   18.9 GB/s at four threads; the cost stage achieves 2.5 GB/s, 13% of that. We are
   not bandwidth-bound and shrinking the volume buys almost nothing.
   **Do this instead: vectorise the box filter across rows.** Its running sum
   (`acc += in[x]; acc -= in[x-2r-1]`) is a serial dependency chain that cannot
   vectorise and is limited by add latency — four passes per slice, sixty slices.
   Processing several rows at once with one SIMD lane per row makes the chains
   independent. Rows are already independent, so it is loop restructuring, not an
   algorithm change. This also explains the hyperthread regression better than
   bandwidth did.
2. **PatchMatch propagation.** Avoids materialising the volume at all. Now supported
   by three independent arguments: slanted surfaces (accuracy), candidate generation
   (what every measurement here says decides the outcome), and runtime.

Still untried on quality: a **graded cost** (Census plus absolute difference).
Roughly half the remaining error sits in near-fronto-parallel regions at 6-7%, which
nothing done today touches, and Census's 49 Hamming levels against SGM's continuous
Birchfield-Tomasi is the most likely reason.

**All dense numbers are desktop-only.** The Jetson was unreachable. The
bandwidth-bound finding matters more there, so re-measure before optimising against
a memory wall of unknown height.

## 0.35 Edge-aware support first, then slanted planes — measured, not assumed

Where dense MASDA stands: 126 ms, 86.1% coverage, 10.6% bad-1.0. SGM: 16 ms, 81.1%,
8.1%. At matched coverage, MASDA 8.6% against SGM 6.2%.

One change addresses the top item on both the quality list and the runtime list, and
is the only candidate that could put us *ahead* rather than level:

**Measured before committing to it** (see 09-matching.md): error is flat up to a
disparity gradient of 0.3 px/px and explodes beyond. High-gradient regions carry
~half of all error on both scenes — but above 0.6 px/px that is a *depth
discontinuity*, not a slanted surface, and the discontinuity population is 2-3x
larger than the genuinely slanted one. A warped window still straddles an occlusion
boundary.

So: **edge-aware support first** (AGAP, doi:10.1049/iet-ipr.2018.5801), slanted
planes second (PatchMatch, doi:10.5244/C.25.14). That is the reverse of the order
the papers suggested, and the reversal came from the data.

- *Quality.* Our window aggregation assumes constant disparity across the window,
  which is wrong on every slanted surface and is why the aggregation-radius sweep
  plateaued at 3-4. **SGM shares this fronto-parallel bias**, so fixing it is a place
  SGM is weak rather than a way to catch up.
- *Runtime.* The cost volume is 40 MB, is 71 of 126 ms, and the work is
  memory-bandwidth bound (8 threads is slower than 4). Propagation never materialises
  it.
- *Fit.* Everything measured here says the candidate set decides the outcome, and
  PatchMatch is a cheap candidate generator. MASDA supplies what it lacks: a
  one-to-one constraint and a confidence measure. Neither paper does that
  combination.

Use red-black / checkerboard propagation, not scanline order, so it stays parallel.

Second tier, in order:
1. **Graded cost.** Census gives 49 Hamming levels; SGM's Birchfield-Tomasi
   difference is continuous. Combining them is standard and untried here.
2. **uint16 cost volume** instead of float32. Bandwidth-bound, so quartering the
   bytes is worth more than vectorising the arithmetic.
3. **Edge-aware support** instead of a box (AGAP, doi:10.1049/iet-ipr.2018.5801).
   The box was worth 16 points of bad-1.0; a support region that follows image
   structure should be worth more, and MST filtering is O(N).

## 0.4 Semi-dense candidates + margin gate — reaches SGM's precision, needs to get fast

Dense SGM beats MASDA **at MASDA's own keypoints**, 0.858 against 0.616 precision,
pooled over the eight Middlebury scenes, while also filling 78% of the image in
17 ms (`article/dense_baseline.py`).

**Measured before building it, and the answer bounds the work.** A perfect
re-ranking of MASDA's existing candidates tops out at **0.697**, because in 30.3% of
its errors the correct right keypoint was never a candidate. Denser right detection
does not lift that ceiling (flat at 0.68-0.70 from 555 to 2994 right keypoints).
And a semi-dense variant that offers every right pixel — so the answer is always
available — gets *worse*, 0.587, because Census plus uniqueness cannot pick the
right one out of 71 candidates.

A smoothness factor is worth roughly +8 points on the current candidate set, not
+24. But smoothness turned out not to be the route anyway: **semi-dense candidates
plus the existing margin gate reach SGM's precision without it** — 0.840 at gate
0.10, 0.882 at 0.15, against SGM's 0.858, and +459 correct with +0.147 precision
over the sparse baseline at gate 0.05.

**The open problem is cost, not quality.** 68 ms per 450x375 scene for the solver,
three times the Jetson budget at a fifth of the pixels. The gate discards most of
the 71 candidates per keypoint, so generating fewer and better candidates — a
coarse-to-fine pass, or a disparity prior from the previous frame — is the obvious
avenue and has not been tried. See 09-matching.md.

Uniqueness and smoothness are orthogonal — SGM has no uniqueness constraint and
needs a left-right check bolted on to reach 78% coverage. A factor graph with both
is the interesting object, and adding a smoothness factor is the same kind of
derivation the ordering factor already went through in section 7 of the article.

The earlier two-pass smoothness attempt failed because a prior fitted from the
matches it then judges is self-confirming. Path aggregation over the whole field is
a different thing, and `dense_baseline.py` is now the yardstick.

---

## 0.5 Quick wins now that the live view works

- **Benchmark `de_dense` on the Jetson.** All dense numbers are desktop-only; the
  board was unreachable when they were taken. The memory-bandwidth finding matters
  more there, since the TX2 has far less of it.
- **Compare against the D4 ASIC's own depth.** Nobody has checked whether this
  matcher beats the silicon it is replacing. Needs `rs_ir_capture` to also record
  the `Depth` stream, then compare disparities at our matched keypoints on (a) a
  flat wall, scored against a fitted plane, and (b) an emitter-off scene where
  texture is scarce and the uniqueness constraint should matter most. Reasoning and
  the expected shape of the answer in [01-hardware.md](01-hardware.md).

- **Measure denser detection against ground truth.** `--cell 12 --per-cell 2` gives
  ~3x the matches and looks clearly better live, but "reads better" is not
  "more correct". `de_bench --sweep` over the cell/per-cell grid would settle it,
  and if denser is also more accurate the matcher defaults should change.
- **Temporal stability.** Keypoints are re-detected every frame, so the cloud
  flickers on a static scene. Accumulating over frames, or tracking instead of
  re-detecting, fixes both this and the 44-51% stereo repeatability ceiling in
  [09-matching.md](09-matching.md). Same root cause seen along two different axes,
  which makes it the highest-value single change to the front end.

---

## 1. Needs you, physically — nothing else can proceed past these

### 1.1 Get a laser rangefinder — **the binding constraint**

Bring-up step 4: measure walls at 1, 2 and 3 m.

**Update, 2026-08-07: this no longer gates matcher development.** Middlebury supplies
ground truth on eight real scenes, and `core/tools/de_bench.cpp` runs the C++ matcher
and detector against it, so match correctness is measurable today without a
rangefinder. Every recent matcher change was validated that way. What the rangefinder
still gates is depth accuracy *on this camera*, on this baseline, in these rooms --
absolute scale, calibration drift, and anything about the vehicle's own environment.
That is real, but it is validation rather than development.

**Why it was first.** It gated *matcher development*, back when the only quality
measures available were match count, objective, and median
|dy|, none of which can distinguish "removed 100 wrong matches" from "removed 100
right ones". Two prior experiments (coarse-to-fine and smoothness) both came back
negative and **neither result is conclusive** for exactly this reason — see
[09-matching.md](09-matching.md). Until a disparity can be checked against a known
distance, additions to `s(i,j)` can be observed but not evaluated.

The plan's own reason also stands: do this before driving, because once moving,
systematic and random error cannot be separated.

Expected precision to verify against: at f·B = 21.48 px·m, 0.1 px of disparity is
~5 mm at 1 m, ~2 cm at 2 m, ~4 cm at 3 m.

### 1.2 Run ArduPilot's 6-position accelerometer calibration

Stationary gravity reads **9.074 m/s², 7.5% low**, consistent across two independent
logs. That is a scale/calibration fault, not noise, and the plan uses the
accelerometer for the gravity vector and hence the ground plane.

Needs the vehicle physically rotated through six orientations, so it cannot be done
remotely. Verify afterwards with:

```sh
.venv/bin/python desktop/allan_variance.py bags/imu/<log>.bin   # checks |gravity|
```

### 1.3 Record a few varied scenes

Every matching number so far — including the headline **+46% MASDA over
nearest-neighbour** — comes from 4 stereo pairs of *one static desk scene*. Worth
knowing whether it generalises before it becomes a quoted figure.

```sh
.venv/bin/python desktop/live_view.py --collect scene02   # then scene03, ...
./core/build/de_match bags/scene02
```

Aim for contrast: a textureless wall, a cluttered close-range scene, something at
2–3 m, something with a thin foreground object (chair or table legs — the plan flags
these as the ordering-constraint test case).

### 1.4 A battery

Unblocks everything vibration-related, which the plan treats as a real risk:

- MEMS rectification tilting the gravity vector speed-dependently, in a way that
  *looks plausible*
- Whether arrival-time jitter survives vibration — the 0.04 px p99 figure is from a
  stationary bench and is explicitly untested under motion
- Whether the USB3 cable holds up; the plan calls vibration-induced disconnects the
  most common hardware failure, presenting as software bugs

Also needed before the foam-decoupled IMU mount can be evaluated.

### 1.5 Optional: a larger calibration board

A3, or A4 sheets tiled onto rigid backing. Only needed if you ever want a
trustworthy `fx` of your own — the factory value is validated to 0.13% on baseline
and under 0.35 px on principal point, so this is not blocking.

If you do: **measure the print**. The A4 came out at 96% scale (25 mm nominal → 24.0
mm measured), and that propagated straight into a 4% baseline error until caught.

---

## 2. Ready now — code only, no hardware

### 2.1 Kalibr for hand-eye and time offset

The remaining half of bring-up step 3. Needs the IMU and camera in one rosbag with
**real** timestamps.

- `bag_to_rosbag.py` exists and its output is validated against real ROS Melodic
- **But** a `live_view --collect` set has synthesised timestamps, which are fine for
  `kalibr_calibrate_cameras` and **not valid** for `kalibr_calibrate_imu_camera`
- So this needs a recording made with `rs_ir_capture` (which writes `frames.csv`
  with real host times) *while* IMU data is captured, and the two merged into one bag
- IMU noise parameters for the yaml are done — see
  [08-imu.md](08-imu.md#complete-noise-characterisation)
- Target: **ROS 1 Noetic in Docker on the desktop**. Noetic is the newest ROS 1
  Kalibr supports, and Ubuntu 24.04 cannot host ROS 1 natively. Not Melodic, not
  ROS 2.

Not yet written: the IMU side of the converter, and the time-alignment between the
camera's host timestamps and the Pixhawk's clock. `TIMESYNC` is present in the
MAVLink stream and is ArduPilot's own mechanism for this.

### 2.2 Sub-pixel disparity refinement

The plan: "Depth accuracy hinges entirely on this." Keypoint positions are already
sub-pixel refined, but the *disparity* is just the difference of two independently
refined positions rather than a fit to the correlation surface between them.

Cannot be validated without 1.1.

### 2.3 Fix the timing comparison in `de_match`

The coarse-to-fine path times detection plus matching; the single-level MASDA figure
times matching only. The two are not comparable. The coarse-to-fine conclusion does
not depend on it (it rests on match counts), but the numbers should not be left
side by side as if they were.

### 2.4 Re-measure preprocessing under realistic load

78.8% of the 33.3 ms budget was measured with the two channels concurrent and
nothing else running. Add the matcher, and eventually the IMU path, and re-check.
Concurrency already returned 1.54× rather than 2×, which points at memory bandwidth,
so headroom may be smaller than the number suggests.

---

## 3. Deferred with reasoning — do not redo without new information

### 3.1 Coarse-to-fine — off by default, kept in the code

Does not help: k is already 2.7, not the 100–200 the plan assumed, and inflating k
five-fold leaves the answer *identical*. There are no false candidates for a prior
to remove. Full reasoning in [09-matching.md](09-matching.md).

**Revisit if**: calibration drifts and the epipolar band must widen; keypoint density
rises a lot; or — most likely — **temporal matching between frames**, where the search
is genuinely 2-D, k really is large, and the prior would come from IMU rotation
compensation. That is where this code should earn its place.

### 3.2 The full disparity-smoothness factor derivation

The plan's cheap two-pass test of whether this is worth deriving came back negative
— monotonically worse at every weight, best `w_smooth` is 0. Probable cause is
structural: the prior is fitted from the very matches it then judges, so it
reinforces errors rather than correcting them.

**But** see 1.1 — that negative is not conclusive without ground truth. Reconsider
after the rangefinder, not before.

If resumed: use a Delaunay triangulation rather than the k-nearest-neighbour proxy
currently implemented, and Eigen (3.4 desktop, 3.3.4 Jetson) for anything larger
than the current 3×3 fit.

### 3.3 uvcvideo kernel patch

Not worth it for timing: arrival jitter is 0.04 px at p99 against a ±3–4 px budget,
and it holds under full CPU load. Would still buy hardware-sync verification,
in-motion projector labelling, and per-frame exposure confirmation.

### 3.4 Hardware sync verification

Currently **unverifiable** — with a host-side frame counter and host-side stamps,
both L/R pairing and the timestamp difference are true by construction. Expected to
fall out of the first matcher via median |dy|, and it partly has: 0.359 px is
consistent with real sync. Not proof.

---

## 4. Open questions from the plan, still open

- **Calibrating λ and γ.** Currently hand-set. The plan's candidate: a small MLP over
  descriptor distance, y-residual, coarse-disparity residual, keypoint response,
  response ratio and local texture energy → calibrated log-likelihood ratio. Needs
  ground truth (1.1) to train against.
- **Second use of MASDA for frame-to-frame association** (visual odometry). Same
  machinery; the IMU rotation compensation makes it tractable, and it is the case
  where 3.1's coarse prior should finally pay.
- **DL baseline for comparison** (RoMa / ELoFTR / LoMa) offline on the desktop
  against recorded bags. Never on the TX2.
- **`INS_LOG_BAT_OPT` semantics** — batch data is windowed at ~1 s. Not blocking any
  more, since long-τ parameters come from the continuous stream instead, but worth
  understanding if raw continuous data is ever wanted.

---

## 5. Operational reminders

- **`jetson_clocks` does not survive a reboot**, and `/etc/nvpmodel.conf` carries
  `DEFAULT=3`. `doubleeye-performance.service` handles both, but if it is ever
  disabled you silently lose a third of your frames. The tools warn; `run.txt`
  records `clocks_locked`.
- **`LOG_DISARMED=1` logs continuously at ~130 MB/hour**, about 3 GB/day. Set it back
  to 0 when not recording, or the card fills.
- **The Pixhawk re-enumerates** ACM0 → ACM1 across a reboot. Tools resolve it through
  `/dev/serial/by-id/`; anything hand-written should too.
- **`stty raw` is mandatory** before reading a tty directly, or you get a few hundred
  bytes where the real rate is kilobytes per second — indistinguishable from a dead
  link.

## Article / matcher validation (added 2026-08-07)

Done:
- Real-data validation on Middlebury 2003 Teddy + Cones. `article/regen_all.py`
  regenerates every article number and figure and writes `results.json`.
- Fixed `rng_for()` seeding from Python's per-process salted `hash()`. Numbers
  were irreproducible across runs; one conclusion had flipped sign as a result.

Open, in rough priority order:
- **Detector repeatability is the binding constraint on real data.** Only 48% of
  Teddy's left keypoints (51% on Cones) have any right keypoint within 1 px of
  their true correspondence, which accounts for 102 of Teddy's 132 errors.
  Precision on the attainable subset is 0.876 / 0.963 against a raw 0.615 / 0.781.
  Two cheap experiments: (a) detect once and track into the second image instead
  of detecting twice; (b) lower the right-image detector threshold to
  over-propose and let gamma discard the surplus.
- Sub-pixel disparity. `detect()` returns integer positions, so median disparity
  error is 0.50 px on Teddy, exactly the integer-vs-quarter-pixel quantisation.
  Fit the correlation surface between the two windows rather than differencing
  two independently refined positions.
- Fix swapped lambda/gamma in `core/src/match.cpp` relative to the article's
  convention (lambda = clutter/left-unmatched, gamma = misdetection/right).
- ~~Port `_seg_max_excluding` into the C++ matcher~~ -- not needed. `Top2` in
  `core/src/match.cpp` already gives O(1) "max excluding one element"; the NumPy
  trick exists only because NumPy cannot loop cheaply. The matcher is also not
  where the time goes.
- Run the C++ matcher against Middlebury too, so the C++ and NumPy paths are
  compared on identical data with ground truth. This is now possible without a
  rangefinder and no longer blocks on hardware.

## Matcher work, 2026-08-07

Done:
- **Score margin exported per match** (`Match::margin`). Best-minus-second-best
  s(i,j), one O(E) pass. Precision by margin quartile over eight Middlebury
  scenes: 0.169 / 0.286 / 0.391 / 0.659. `de_match` reports median margin and the
  share below 0.2; it is in `matches.csv`. On `full_on`, MASDA has 31.5% of its
  matches below 0.2 against mutual-NN's 14.3%, which is where MASDA's extra 47%
  of matches live.
- **lambda/gamma un-transposed.** lambda = clutter (left/measurement unmatched),
  gamma = misdetection (right/object unmatched). No past result was wrong, since
  every run held them equal, but they are not interchangeable once separated.
  `test_lambda_gamma_are_distinct` guards it.
- **core/ now builds on the Jetson.** It never had: `deploy.sh` synced only
  `jetson/`, and the `core/build` on the Jetson held x86-64 objects from the
  desktop. Consequence: MASDA is **5.04 ms on the Jetson**, not the 1.67 ms in the
  docs, which was a desktop number compared against the Jetson frame budget. 15%
  of the budget, not 5%.
- **Local contrast is not useful** and the intuition is backwards. See
  `article/contrast_study.py`. Do not raise `min_local_std` above its current 2.0.

Open:
- The published article still carries the 1.67 ms figure attributed to the Jetson
  frame budget. Mario's call whether to correct it.
- Detector repeatability: only 48-51% of left keypoints have a right keypoint
  within 1 px of their true correspondence. `article/repeatability.py` tests
  over-proposing on the right with a cheaper gamma.
- Sub-pixel disparity. `detect()` returns integer positions, so median disparity
  error is 0.50 px, exactly the integer-vs-quarter-pixel quantisation. Fit the
  correlation surface between the two windows.
- ~~Port `_seg_max_excluding` into the C++ matcher~~ -- not needed. `Top2` in
  `core/src/match.cpp` already gives O(1) "max excluding one element"; the NumPy
  trick exists only because NumPy cannot loop cheaply. The matcher is also not
  where the time goes.
- Use the margin downstream: weight or gate matches by it before triangulation.

## Frame budget, measured 2026-08-07

**The pipeline does not close at 30 Hz.** Jetson, MAXN, 848x480, 120 pairs:

| stage | ms/pair | share of 33.3 ms |
|---|---|---|
| preprocessing, L/R concurrent | 26.54 | 79.6% |
| MASDA baseline | 5.14 | 15.4% |
| **total baseline** | **31.68** | **95.1%** |
| MASDA `--right-density 6 --min-margin 0.10` | 8.84 | 26.5% |
| **total recommended** | **35.38** | **106.2%** |

Detection is 20.98 of the 21.24 ms per frame, so zeroing the matcher entirely
still leaves preprocessing at 80%. Needs a decision, in this order:

1. Run correspondence at 15 Hz. 66.7 ms budget, 53% used even with the
   recommended config. Costs latency only, and 15 Hz is plausibly enough for
   indoor odometry.
2. Profile detection. FAST already replaced the dense Shi-Tomasi scan; the
   Shi-Tomasi scoring of FAST corners and the NMS have not been profiled
   separately.
3. Drop to 640x480, which is 75% of the pixels and preprocessing is close to
   pixel-bound.

The recommended matcher config is off by default until this is settled.
