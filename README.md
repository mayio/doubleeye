# DoubleEye — MASDA sparse stereo

MASDA (max-sum loopy BP data association) as the matcher for sparse stereo
correspondence, on a D435 IR pair + Jetson TX2 indoor RC platform.

Design context, decisions and rejected options: `~/Documents/doubleeye/doubleeye_plan.md`.
That document is the spec. This README tracks state only.

## Layout

The plan's portability rule is load-bearing, so it is encoded in the directory
structure: anything that touches librealsense, Jetson timestamps or power modes
lives in `jetson/` and nowhere else.

| Path | Runs on | Depends on |
|---|---|---|
| `jetson/` | TX2, Python 3.6 | `pyrealsense2` only — no numpy, no OpenCV |
| `desktop/` | dev box, Python 3.12 | numpy, matplotlib |

Preprocessing and MASDA itself, when written, go in a third top-level directory
that depends on neither.

## Bring-up status

Order is from the plan. Do not skip ahead — step 4 exists specifically because
systematic and random error cannot be separated once the vehicle is moving.

- [ ] **1. Capture pipeline and timestamp sanity** ← current
- [ ] 2. IMU Allan variance (multi-hour static recording)
- [ ] 3. Calibration — intrinsics/extrinsics, then hand-eye + time offset (Kalibr)
- [ ] 4. Static bags vs laser rangefinder — walls at 1, 2, 3 m
- [ ] 5. Drive

## Step 1

On the Jetson:

```sh
# Reproducible latency requires fixed clocks. Do this before measuring anything.
sudo nvpmodel -m 0
sudo jetson_clocks

python3 jetson/rs_probe.py --save-prefix /tmp/probe
python3 jetson/rs_ir_capture.py ~/bags/$(date +%Y%m%d_%H%M%S)_static --seconds 120
```

`rs_probe.py` is the gate. It fails loudly on the two things that quietly
invalidate everything downstream:

- **USB2 fallback.** 848×480@30 on two IR streams does not fit in USB2
  bandwidth. Presents as mysterious frame drops, not as an error.
- **Missing 848×480 Y8 profile** on either stream index.

It also prints factory intrinsics and the IR baseline, so `f·B` can be checked
against the plan's ~21 px·m before any depth number is believed.

Then on the desktop:

```sh
rsync -a jetson:~/bags/<run>/ ./bags/<run>/
python3 desktop/capture_report.py bags/<run>
```

### What step 1 has to establish

1. Zero (or fully accounted-for) dropped frames on both channels.
2. Matched frame numbers L/R, sub-millisecond timestamp agreement — i.e. the
   hardware sync is real.
3. A unimodal frame-interval histogram at 33.3 ms. Bimodal means USB or
   scheduling trouble.
4. Which domain `get_timestamp()` actually reports. `RS2_FRAME_METADATA_FRAME_TIMESTAMP`
   and the backend timestamp are different clocks; the plan's IMU-skew risk
   turns on knowing which one is in hand.
5. Camera-vs-Jetson clock drift in ppm, plus the residual scatter after a linear
   fit. The residual bounds how well any constant offset can ever align the two
   clocks — which is the real input to step 3's time calibration.

Fixed exposure (1–2 ms) is the default, not auto. Auto-exposure hunts while
driving and makes intervals unreproducible; at 2 m/s and 1 m range, 5 ms of
exposure is >4 px of blur.

## Desktop setup

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r desktop/requirements.txt
```

Nothing in `desktop/` needs librealsense, so the dev box does not need the
camera attached.
