# Baseline measurements

Reference figures from bring-up step 1, for citation in later write-ups and as a
regression baseline. All taken with **MAXN power mode and clocks locked** unless
stated; a recording taken without that is not comparable, which is why every
`run.txt` records `clocks_locked`.

Reproduce any of these with:

```sh
.venv/bin/python desktop/capture_report.py bags/<run>
```

## Factory calibration, 848×480 IR

| Quantity | Measured |
|---|---|
| fx = fy | 430.551 px |
| cx | 427.381 px |
| cy | 243.158 px |
| Distortion coefficients | all exactly 0.00000 |
| IR1→IR2 translation | [−0.049883, 0.000000, 0.000000] m |
| **Baseline** | **49.883 mm** |
| IR1→IR2 rotation | exactly identity (deviation 0.00e+00) |
| **f·B** | **21.48 px·m** |

Two things worth noting. **f·B = 21.48 px·m matches the plan's predicted ~21**,
so the plan's depth-resolution table stands unchanged. And the exactly-zero
distortion with exactly-identity rotation confirms the IR pair really is
delivered rectified by the D4 ASIC — the plan's assumption, now verified rather
than assumed.

Values were stable across every successful read. Note obstacle 8: the first read
after pipeline start usually fails and must be retried.

## Delivered frame rate

Requested versus achieved, 25–30 s runs, `--save-every 0`. Both columns are the
same hardware and cable; the only difference is power mode.

| requested | mode 3 (`MAXP_CORE_ARM`) | MAXN + `jetson_clocks` |
|---|---|---|
| 848×480 @ 30, both streams | 18.53 | **29.79** |
| 848×480 @ 60, both streams | 22.08 | **59.54** |
| 848×480 @ 90, both streams | 29.88 | **89.34** |
| 640×480 @ 30, both streams | 17.62 | **29.79** |
| 424×240 @ 30, both streams | 29.83 | — |
| 848×480 @ 15, both streams | 14.89 | — |
| 1280×720 @ 30, both streams | 10.64 | **29.79** |
| 848×480 @ 30, **single** stream | 29.80 | — |

At locked clocks, 848×480@90 on both channels is **~72 MB/s sustained**, so there
is substantial headroom above the 30 Hz currently used.

## Frame timing, 848×480@30

From a 120 s recording, 3600 frames per channel, emitter on, exposure 1500 µs,
gain 64.

| Quantity | Value |
|---|---|
| Delivered rate | **29.95 fps** (3600 frames / 120.2 s) |
| Frame interval, median | 33.352 ms |
| Frame interval, p1 / p99 | 33.301 / 33.407 ms |
| Frame interval, max | 36.902 ms |
| Fitted frame period | 33.3523 ms |
| Fitted period vs nominal 33.3333 | **+568 ppm** |

The interval distribution is tight and unimodal — no bimodality, which would
have indicated USB or scheduling trouble.

The **+568 ppm** figure is the camera's frame-generation rate measured against
the Jetson clock, reproduced at **+569 ppm** on an independent emitter-off run.
It is determined to well under 1 ppm by the fit. It does not decompose into
"camera crystal" versus "host clock" without an external reference, but the
combined value is what is needed to predict when the next frame arrives in host
time.

## Arrival jitter

Residual about the fitted frame grid, and the same figure converted into pixels
of misregistration under gyro rotation compensation at 100 °/s (the plan's
regime of interest). The conversion factor is fx · ω = 430.551 × 1.745 mrad/ms
≈ **0.75 px per ms**.

| | emitter on (120 s) | emitter off (60 s) |
|---|---|---|
| std | 0.062 ms → **0.05 px** | 0.016 ms → 0.01 px |
| p99 \|residual\| | 0.050 ms → **0.04 px** | 0.060 ms → 0.04 px |
| max \|residual\| | 3.574 ms → 2.69 px | 0.111 ms → 0.08 px |

**Interpretation.** Against the plan's ±3–4 px coarse-to-fine search radius,
p99 jitter of 0.04 px is over an order of magnitude inside budget. Only a single
worst-case outlier approaches the budget. This is the measurement that decided
against patching uvcvideo — see [03-obstacles.md](03-obstacles.md), obstacle 7.

### Under full CPU load

Re-measured with all 6 cores saturated by busy loops plus a `dd` memory-traffic
generator, 90 s at 848×480@30, `--save-every 0`. Load average reached **5.69**.

| | idle (120 s) | **full load (90 s)** |
|---|---|---|
| delivered rate | 29.95 fps | **29.90 fps** |
| shortfall | 0.16% | 0.35% |
| jitter std | 0.062 ms → 0.05 px | **0.018 ms → 0.01 px** |
| jitter p99 | 0.050 ms → 0.04 px | **0.052 ms → 0.04 px** |
| jitter max | 3.574 ms → 2.69 px | **0.542 ms → 0.41 px** |
| fitted period | +568 ppm | +564 ppm |
| interval p1 / p99 | 33.301 / 33.407 ms | 33.302 / 33.402 ms |

**Jitter does not degrade under full CPU load.** p99 is identical at 0.04 px, and
the delivered rate holds. This closes the load caveat: the timing budget that
justified skipping the uvcvideo patch survives a saturated machine.

Two honest qualifications:

- The **3.574 ms maximum** in the idle run did not reproduce. Across roughly 9000
  frames over three recordings it was seen once. It is not explained by disk
  writes in the callback (the emitter-off run also wrote every 30th frame and
  peaked at 0.111 ms) nor by load. Treat it as a rare one-off of unknown cause
  rather than a characterised worst case — a single sample is not a bound.
- This is **CPU and memory load, not the real workload.** It does not include
  vibration, the emitter alternating, or a real matcher competing for memory
  bandwidth, which the plan identifies as the actual bottleneck on this board.
  Worth re-checking once the pipeline exists.

## Image statistics and the projector A/B

Two recordings of the same static indoor scene, 848×480@30, exposure 1500 µs,
gain 64, differing only in emitter state. Local standard deviation is computed
over a 7×7 window — the same window Census would read.

| metric | emitter ON | emitter OFF |
|---|---|---|
| mean intensity, ir1 | 89.9 | 88.1 |
| mean intensity, ir2 | 92.5 | 90.8 |
| saturated fraction (≥250 DN) | 6.5% | 6.4% |
| dark fraction (<8 DN) | 0.00% | 0.00% |
| **7×7 local std, median** | **3.91** | **1.65** |
| 7×7 local std, p90 | 16.00 | 14.26 |
| **fraction with local std < 2 DN** | **24.5%** | **56.7%** |

**The projector more than halves the textureless area, 56.7% → 24.5%**, and
lifts median local contrast 2.4×. This is the plan's central
Census-over-learned-descriptors premise confirmed quantitatively.

Note that **mean intensity moved only 1.8 DN** across the same comparison and is
worthless as a discriminator — see [03-obstacles.md](03-obstacles.md),
obstacle 12.

Two further observations with downstream consequences:

- **6.5% of pixels saturated**, from a sunlit window. Saturated regions carry no
  Census information whatsoever, so expect keypoint deserts there. This also
  means the measured textureless fractions above are a floor, not a ceiling: the
  saturated region is unusable regardless of the projector.
- The two IR sensors differ by **2.6 DN** in mean level. Irrelevant for Census,
  which is invariant to any monotonic intensity mapping — but it would matter for
  SSD or NCC, and it is the exposure/gamma mismatch the plan expected Census to
  absorb.

## Frame metadata availability

On the stock (unpatched) kernel:

| Field | Available |
|---|---|
| `Backend Timestamp` | yes |
| `Time Of Arrival` | yes |
| `Actual Fps` | yes |
| `Frame Timestamp` | **no** |
| `Sensor Timestamp` | **no** |
| `Frame Counter` | **no** |
| `Actual Exposure` | **no** |
| `Gain Level` | **no** |
| `Frame Laser Power Mode` | **no** |

Timestamp domain: **`System Time`** — host arrival, not the camera clock.

## Sensor option ranges

Read from the stereo module, useful when choosing exposure and gain:

| Option | Default | Range | Step |
|---|---|---|---|
| `Exposure` | 8500 | 1 – 165000 µs | 1 |
| `Gain` | 16 | 16 – 248 | 1 |
| `Laser Power` | 150 | 0 – 360 | 30 |
| `Emitter Enabled` | 1 | 0 – 1 | 1 |
| `Emitter On Off` | 0 | 0 – 1 | 1 |
| `Enable Auto Exposure` | 1 | 0 – 1 | 1 |

`Emitter On Off` is present, so per-frame projector alternation *is* supported by
the firmware — but it cannot currently be demultiplexed, because the per-frame
laser label is unavailable. Settings used for the recordings above: auto-exposure
off, exposure 1500 µs, gain 64.

## Calibration set, collected 2026-08-07

First real session, collected with `live_view.py --collect calib01`: a 9×6
interior-corner board at 25 mm on A4, glued to rigid backing, emitter **off**,
exposure 1500 µs, gain 96.

| | `calib01` | `cbtest` |
|---|---|---|
| pairs collected | 82 | 74 |
| **detected in BOTH channels** | **82 / 82 (100%)** | **74 / 74 (100%)** |
| ir1 only / ir2 only / neither | 0 / 0 / 0 | 0 / 0 / 0 |
| board image area, median | 5.7% of frame | 7.1% |
| board image area, min–max | 1.0% – 29.0% | 6.2% – 17.6% |
| board width, median | 186 px of 848 | 215 px |
| centre spread | x 154 px, y 93 px | — |

**100% detection in both channels settles the printing question.** The IR-ink
concern from [05-operations.md](05-operations.md) does not apply to this print —
the board is fully visible at 850 nm. Worth knowing for reprints: whatever printer
produced this one is fine.

The auto-collect duplicate rejection did its job: 82 accepted poses out of a
several-minute session, rather than hundreds of near-identical frames.

**The honest caveat is scale.** Median board area is 5.7% of the frame and median
width 186 px of 848 — small, and exactly the A4 limitation predicted in
[05-operations.md](05-operations.md#a4-is-honestly-marginal--know-why-before-relying-on-it).
A larger board would constrain focal length and distortion better. This set is
good enough to run the calibration toolchain end to end and to get a usable first
answer; treat the resulting numbers as provisional until either a bigger target or
a tighter working distance is used, and compare against the factory values above
rather than assuming an improvement.

## Board planarity — the sheet was not always flat

Suspicion raised after the session that the paper had gone loose at times.
Testable, and it turned out to be justified.

The test exists because the D435 IR pair has **exactly zero distortion**. A flat
board must therefore project to an exact projective image of a regular grid — a
homography and nothing more — so the homography residual is dominated by whatever
is not planar. `desktop/check_planarity.py`.

| | `calib01` (all) | `calib01_flat` (culled) |
|---|---|---|
| poses | 82 | **49** |
| RMS residual, p50 | 0.342 px | **0.234 px** |
| p90 | 0.950 px | **0.346 px** |
| max | 2.481 px | **0.399 px** |
| mean | 0.469 px | **0.249 px** |
| detected in both channels | 82/82 | 49/49 |

**Evidence that the cause is physical, not detection noise.** No single statistic
decides this, so three are used:

| Signal | Value | Reading |
|---|---|---|
| residual vs apparent board size | corr **+0.27** | Points **physical**. Detection gets *better* as the board fills more frame, so noise would give a negative correlation. A fixed bend subtends more pixels up close. |
| temporal clustering | **19** pass/fail runs vs **40.4** expected if random | Points **physical**. Failures arrive in episodes, not spread evenly. |
| residual smoothness | 0.52 on failing poses | Ambiguous. This statistic rises with magnitude even for pure noise. |

Two of three point at a physical cause, so the board was probably not perfectly
flat for parts of the session. Implied magnitude at 25 mm pitch: **median 0.36 mm,
p90 0.91 mm, max 3.10 mm** — small, which is why the typical pose is fine and only
the tail is contaminated.

A sharper test was inconclusive: if the cause were a *constant* physical bow, the
residual expressed in board units would be more uniform than in pixels. It is not
(coefficient of variation 0.83 versus 0.76). So it is not one fixed bend but
something varying through the session — consistent with a sheet lifting and
re-seating, or hand pressure, rather than a permanently warped board.

**Remedy, which did not require resolving the question:** cull above 0.40 px RMS.
That leaves 49 poses, comfortably inside the 20–40 that calibration wants, with
the tail removed. Use `bags/calib01_flat`.

**For next time:** glue the sheet down over its whole area rather than at the
edges, onto something genuinely rigid, and grip it by the backing well away from
the printed region.

## Stereo calibration from `calib01_flat`

Run with `desktop/stereo_calibrate.py`, 49 poses, OpenCV `stereoCalibrate`. Not
Kalibr: Kalibr's irreplaceable capability is *camera-IMU* calibration, and for
intrinsics, distortion and the stereo extrinsic OpenCV solves the same problem in
seconds with no ROS.

| | factory (ASIC) | pinhole fit | radtan fit |
|---|---|---|---|
| reprojection RMS | — | **0.269 px** | 0.271 px |
| fx | 430.551 | 427.937 (**−0.61%**) | 432.855 |
| cx | 427.381 | 427.035 (−0.35 px) | 424.373 |
| cy | 243.158 | 243.397 (+0.24 px) | 239.745 |
| baseline | 49.883 mm | 52.030 mm (**+4.30%**) | 51.969 mm |

Three conclusions.

**Distortion really is zero.** Freeing the radtan coefficients made the
reprojection error *worse* (0.271 vs 0.269 px). Extra parameters that fail to
improve the fit are absorbing noise, so the ASIC's claim of zero distortion is
confirmed rather than merely trusted. Use the pinhole numbers.

**Rectification confirmed independently.** The fitted R is **0.034° from
identity** and the off-axis translation is 0.069 / 0.168 mm against a 52 mm
baseline. That is what a rectified pair looks like, arrived at from images rather
than read out of the device.

**The baseline disagreement is almost certainly the ruler, not the camera.** This
follows from which quantity depends on what: `fx` is an angle ratio in pixels and
is *independent* of the assumed square size, while the recovered translation
scales with it exactly. fx agrees to −0.61% while the baseline is off by +4.30% —
seven times as much — so the scale input is the suspect.

If the factory baseline is right, the true pitch is **23.97 mm, not 25.0**, a print
scale of **95.9%**. That is very much what a printer does with "fit to page" left
on. **Measure the board across all ten columns and re-run with the measured
pitch**; if it reads ~240 mm rather than 250, both numbers reconcile at once and
nothing is wrong with the camera.

The remaining −0.61% on fx is consistent with the small-target caveat: median
board coverage was ~5% of frame, which under-constrains focal length.

## Development link throughput

Desktop ↔ Jetson over WiFi: **22.4 MB/s** measured (10 MB via `dd` over ssh).

Better than assumed, and it changed a design decision: full 848×480 Y8 on both IR
channels at 10 Hz is 8 MB/s, so `rs_ir_stream` sends raw uncompressed frames. No
encoding, no downscaling, nothing to go wrong — and no cycles spent on the TX2,
which has none to spare.

## Not yet measured

- Anything involving the IMU. Bring-up step 2.
- Arrival jitter under realistic compute load and vibration.
- Whether hardware sync is genuinely frame-locked (obstacle 7 item 2).
- Any keypoint, descriptor or matching figure — no algorithm has been written yet.
