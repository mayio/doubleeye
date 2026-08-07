# TODO

Ordered by what unblocks the most, not by effort. Each item says *why* it matters
so it can be picked up cold.

Status of the bring-up sequence lives in [README.md](README.md); this is the
actionable list.

---

## 1. Needs you, physically — nothing else can proceed past these

### 1.1 Get a laser rangefinder — **the binding constraint**

Bring-up step 4: measure walls at 1, 2 and 3 m.

**Why it is first.** It does not only gate driving, it gates *matcher development*.
The only quality measures available right now are match count, objective, and median
|dy|, and none of them can distinguish "removed 100 wrong matches" from "removed 100
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
