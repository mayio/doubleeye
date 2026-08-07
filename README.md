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
| `jetson/` | TX2 (`jetsontx2`, 192.168.2.114) | librealsense2 only — no OpenCV |
| `desktop/` | dev box (`blackstone`) | numpy, matplotlib |

Preprocessing and MASDA itself, when written, go in a third top-level directory
that depends on neither.

Jetson-side code is C++14. That is both the stated preference and, here, the
path of least resistance: librealsense 2.22 is already installed with headers
and `realsense2.pc`, whereas `pyrealsense2` is absent and has no aarch64 wheel
for Python 3.6, so Python would mean a from-source bindings build first.

Nothing on the Jetson links OpenCV. That box has a header/runtime version skew
(`/usr/include/opencv2` reports 3.3.1, the runtime `.so` is 3.2), and the
capture path has no need for it — raw Y8 goes to disk and is decoded on the
desktop. Worth remembering before adding OpenCV to anything on-vehicle.

## Measured environment

Differs from the plan's assumptions in one place, so recorded here.

| | Plan assumed | Actually installed |
|---|---|---|
| JetPack | 4.6.x | **4.2** (L4T R32.1, kernel 4.9.140-tegra) |
| CUDA | 10.2 | **10.0** |
| librealsense | — | 2.22.0, C++ headers + pkg-config |
| OpenCV | — | headers 3.3.1 / runtime 3.2 (skewed) |
| cmake / g++ | — | 3.10.2 / 7.5.0 |

The plan's JetPack 4.6 ceiling is right; this box just sits below it. Moving
4.2 → 4.6 is a reflash, so it is only worth doing for a concrete reason.

Confirmed from factory calibration, and it matches the plan's arithmetic:

- fx = fy = 430.551 px, cx = 427.381, cy = 243.158 at 848×480
- baseline = **49.883 mm**, so **f·B = 21.48 px·m** (plan predicted ~21)
- distortion coefficients all exactly zero, IR1→IR2 rotation exactly identity,
  off-axis translation exactly zero

So the IR pair really does arrive rectified, as the plan assumed, and the
depth-resolution table in the plan stands.

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

### Step 1 findings so far — two open problems

**1. The requested frame rate is not being delivered.** 848×480@30 yields
~18.5–19.9 fps, a ~34% shortfall. A sweep (30 s each, `--save-every 0`, so no
disk I/O in the callback) isolates it:

| requested | achieved | met? |
|---|---|---|
| 848×480 @ 30 | 18.53 | no |
| 640×480 @ 30 | 17.62 | no |
| 848×480 @ 15 | 14.89 | **yes** |
| 424×240 @ 30 | 29.83 | **yes** |

Not a simple bandwidth ceiling: 848×480@30 pushes *more* pixels/s than
640×480@30 does, yet both land near 18 fps. It reads as a per-frame cost that
480-line modes cannot sustain past ~18 fps while 240-line modes can hold 30.

Untested and the leading suspect: **the Jetson is in power mode 3
(`MAXP_CORE_ARM`) with both Denver cores offline** (`cpu online = 0,3-5`) and
`jetson_clocks` never applied. The plan already prescribes fixing this before
any timing measurement; it needs a password we do not have in-session.
`848×480@15` is a usable interim, at the cost of doubling inter-frame parallax
for the IMU rotation-compensation work.

**2. uvcvideo is unpatched, so there is no UVC metadata node.** Confirmed
empirically by `rs_probe`. Consequences, in descending order of damage:

- `get_timestamp()` reports domain **System Time**, i.e. host arrival time, not
  the camera clock. Camera-vs-Jetson clock skew — a top risk in the plan — is
  **not measurable** in this state.
- `RS2_FRAME_METADATA_FRAME_COUNTER` is absent, so `get_frame_number()` is a
  *host-side counter of delivered frames*. It is contiguous by construction,
  which makes frame-number gap analysis worthless — it reports a flawless run
  while a third of the frames are missing. `capture_report.py` now detects this
  and refuses to draw the conclusion.
- For the same reason, L/R "pairing" and the L/R timestamp difference are both
  true by construction. **Hardware sync is currently unverified**, despite
  looking perfect.
- `FRAME_LASER_POWER_MODE` is absent, so under `EMITTER_ON_OFF` alternation
  there is no per-frame label for which frames had the projector lit. The
  plan's projector-on/off A/B split would have to be inferred from image
  statistics.
- `ACTUAL_EXPOSURE` is absent, so the fixed exposure can only be trusted from
  the option readback, not confirmed per frame.

The fix for all five is the same: apply librealsense's L4T uvcvideo patch for
kernel 4.9.140 and rebuild the module.

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
