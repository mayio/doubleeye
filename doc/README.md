# DoubleEye — documentation

Working record of the platform, its software environment, and every obstacle hit
during bring-up. Written so a full project write-up can be assembled later
without re-deriving anything.

The algorithmic spec lives outside this folder in
`~/Documents/doubleeye/doubleeye_plan.md`. That document holds the design
decisions, the rejected alternatives and the open questions. Nothing here
duplicates it; this is the engineering record that sits underneath it.

## Reading order

| Document | Contents |
|---|---|
| [01-hardware.md](01-hardware.md) | Components in the system, what each is for, what is still missing |
| [02-software-environment.md](02-software-environment.md) | Exact versions on both machines, what was installed vs. pre-existing, how to build and deploy |
| [03-obstacles.md](03-obstacles.md) | Every problem hit, with symptom → diagnosis → resolution. The most useful document here. |
| [04-baseline-measurements.md](04-baseline-measurements.md) | Measured numbers, for citation and for regression comparison |
| [05-operations.md](05-operations.md) | How to record, pull, view, analyse, and print a calibration target |
| [06-preprocessing.md](06-preprocessing.md) | Census + keypoints: design, measured output, and the profiling result |
| [07-tools.md](07-tools.md) | **Every tool, what it answers, how to run it** — start here when returning to the project |

## Status at time of writing

Bring-up step 1 (capture pipeline and timestamp sanity) is substantially
complete. `848×480@30` runs at 29.95 fps on both IR channels with a tight
unimodal frame-interval distribution. One item remains open: hardware sync
cannot be verified in the current configuration — see
[03-obstacles.md](03-obstacles.md), obstacle 7.

Bring-up step 2 (IMU Allan variance) is **blocked on configuration, not
hardware.** The Pixhawk 2.4.8 is powered and identified as running ArduPilot,
reachable on `/dev/ttyACM0` and on `/dev/ttyTHS2` at 57600 baud, but it streams
IMU at only 3.2 Hz and Allan variance needs the SD-card log rather than a stream
— see
[01-hardware.md](01-hardware.md#the-imu-is-a-pixhawk-248-running-ardupilot).
Separately, the `nvs_bmi160` module and its device-tree node are stock L4T
leftovers with no hardware behind them, and are not the IMU.

The **camera half of step 3 is under way**: it needs no IMU. A calibration set
exists — `bags/calib01`, 82 poses, **100% detected in both channels** — collected
with the live view. What remains is the calibration itself, which needs a
`bags/<run>` → rosbag converter for Kalibr. See
[04-baseline-measurements.md](04-baseline-measurements.md#calibration-set-collected-2026-08-07)
for the set's properties and its one real caveat, which is board size.

Preprocessing (`core/`) exists and is measured. On the TX2 it currently runs at
**299% of the 30 Hz budget**, essentially all of it the keypoint detector, with
Census at 0.3% — see [06-preprocessing.md](06-preprocessing.md) for the
optimisation path.

## Findings at a glance

The results worth knowing, with pointers. Several were surprises, and the pattern
across them is the same: **on this platform the expensive failures are silent.**

| Finding | Where |
|---|---|
| **34% of frames lost with no error at all** — wrong power mode, while frame numbers stayed contiguous and the tool reported `0 missing` | [03](03-obstacles.md) obstacle 5 |
| `nvpmodel -m 0` does **not** persist: `/etc/nvpmodel.conf` says `DEFAULT=3` and NVIDIA's own service reapplies it every boot | [03](03-obstacles.md) obstacle 6 |
| No UVC metadata node, so `get_frame_number()` is a **host-side counter that is contiguous by construction** — which is what hid the frame loss | [03](03-obstacles.md) obstacle 7 |
| Hardware sync is therefore **unverifiable** right now: L/R pairing and timestamp agreement are both true by construction | [03](03-obstacles.md) obstacle 7 |
| The uvcvideo patch is **not worth it**: arrival jitter is 0.04 px at p99 against a ±3–4 px budget, and holds under full CPU load | [04](04-baseline-measurements.md) |
| Camera-vs-host clock rate **is** recoverable without the patch, from frame generation rate: **+568 ppm** | [04](04-baseline-measurements.md) |
| f·B = **21.48 px·m** against the plan's predicted ~21; distortion exactly zero and IR1→IR2 rotation exactly identity | [04](04-baseline-measurements.md) |
| The projector **halves the textureless area**, 57% → 25%, and lifts keypoint coverage from 67% to 93% of cells | [04](04-baseline-measurements.md), [06](06-preprocessing.md) |
| **Mean intensity cannot see the projector** — it moved 1.8 DN across the same A/B. Local contrast is the only valid metric | [03](03-obstacles.md) obstacle 12 |
| Census descriptors are **3.3× degenerate** under the projector (338 distinct for 1115 keypoints) — exactly the ambiguity MASDA's uniqueness constraint is for | [06](06-preprocessing.md) |
| Preprocessing costs **299% of the 30 Hz budget on the TX2**, essentially all detector, Census at 0.3% | [06](06-preprocessing.md) |
| Computing Census densely wasted **99.7%** of the work; making it sparse cut 61 ms to 0.13 ms with identical output | [06](06-preprocessing.md) |
| The Pixhawk reports "PX4 FMU v2.x" over USB but **runs ArduPilot** — the descriptor is the bootloader identity | [01](01-hardware.md) |
| `nvs_bmi160` and its device-tree node are **stock leftovers with no hardware behind them** — not the IMU | [01](01-hardware.md) |
| A calibration set of 82 poses reached **100% detection in both channels**, so the print is IR-visible | [04](04-baseline-measurements.md) |
| ...yet a third of those poses had a **non-flat board**. Detection success says nothing about planarity | [04](04-baseline-measurements.md) |
| cmake 3.10 accepts neither `-S`/`-B` nor `--build -j`; given either it **prints usage and exits without building** | [03](03-obstacles.md) obstacle 13 |

## Conventions used throughout

- **DN** — digital number, i.e. raw 8-bit pixel level (0–255).
- **`jetson`** — the SSH alias for the TX2; see
  [02-software-environment.md](02-software-environment.md).
- Numbers quoted as measured are from recordings under `bags/`, which is
  gitignored. `run.txt` inside each bag records the conditions it was taken
  under, including whether the clocks were locked.
