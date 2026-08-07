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

## Not yet measured

- Anything involving the IMU. Bring-up step 2.
- Arrival jitter under realistic compute load and vibration.
- Whether hardware sync is genuinely frame-locked (obstacle 7 item 2).
- Any keypoint, descriptor or matching figure — no algorithm has been written yet.
