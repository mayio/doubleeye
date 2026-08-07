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

## Status at time of writing

Bring-up step 1 (capture pipeline and timestamp sanity) is substantially
complete. `848×480@30` runs at 29.95 fps on both IR channels with a tight
unimodal frame-interval distribution. One item remains open: hardware sync
cannot be verified in the current configuration — see
[03-obstacles.md](03-obstacles.md), obstacle 7.

Bring-up step 2 (IMU Allan variance) is **blocked on powering the Pixhawk
2.4.8**, which carries the IMU and currently enumerates nowhere. Micro-USB to the
Jetson is sufficient for bench work — see
[01-hardware.md](01-hardware.md#the-imu-is-a-pixhawk-248--how-to-power-it).
Separately, the `nvs_bmi160` module and its device-tree node are stock L4T
leftovers with no hardware behind them, and are not the IMU.

Preprocessing (`core/`) exists and is measured. On the TX2 it currently runs at
**299% of the 30 Hz budget**, essentially all of it the keypoint detector, with
Census at 0.3% — see [06-preprocessing.md](06-preprocessing.md) for the
optimisation path.

## Conventions used throughout

- **DN** — digital number, i.e. raw 8-bit pixel level (0–255).
- **`jetson`** — the SSH alias for the TX2; see
  [02-software-environment.md](02-software-environment.md).
- Numbers quoted as measured are from recordings under `bags/`, which is
  gitignored. `run.txt` inside each bag records the conditions it was taken
  under, including whether the clocks were locked.
