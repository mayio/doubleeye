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

## Status at time of writing

Bring-up step 1 (capture pipeline and timestamp sanity) is substantially
complete. `848×480@30` runs at 29.95 fps on both IR channels with a tight
unimodal frame-interval distribution. One item remains open: hardware sync
cannot be verified in the current configuration — see
[03-obstacles.md](03-obstacles.md), obstacle 7.

Next is bring-up step 2, IMU Allan variance. The IMU has not been examined yet.

## Conventions used throughout

- **DN** — digital number, i.e. raw 8-bit pixel level (0–255).
- **`jetson`** — the SSH alias for the TX2; see
  [02-software-environment.md](02-software-environment.md).
- Numbers quoted as measured are from recordings under `bags/`, which is
  gitignored. `run.txt` inside each bag records the conditions it was taken
  under, including whether the clocks were locked.
