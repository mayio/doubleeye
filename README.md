# DoubleEye — MASDA sparse stereo

MASDA (max-sum loopy BP data association) as the matcher for sparse stereo
correspondence, on a D435 IR pair + Jetson TX2 indoor RC platform.

Design context, decisions and rejected options: `~/Documents/doubleeye/doubleeye_plan.md`.
That document is the spec. This README tracks state only.

**Picking this up again? [doc/TODO.md](doc/TODO.md)** is the actionable list,
ordered by what unblocks the most.

**Start with [doc/07-tools.md](doc/07-tools.md)** for what every tool does and how
to run it, and [doc/README.md](doc/README.md) for a findings-at-a-glance table.

**Detailed engineering record in [doc/](doc/)** — components, exact software
versions, build and deploy procedure, every obstacle hit with its diagnosis, and
the measured baselines. Start with [doc/03-obstacles.md](doc/03-obstacles.md) if
something is behaving strangely; on this platform the expensive failures are all
silent.

## Layout

The plan's portability rule is load-bearing, so it is encoded in the directory
structure: anything that touches librealsense, Jetson timestamps or power modes
lives in `jetson/` and nowhere else.

| Path | Runs on | Depends on |
|---|---|---|
| `jetson/` | TX2 (`jetsontx2`, 192.168.2.114) | librealsense2 only — no OpenCV |
| `desktop/` | dev box (`blackstone`) | numpy, matplotlib |

| `core/` | anywhere | **nothing but C++14 stdlib** |
| `tools/` | desktop | bash |

`core/` is the plan's portability rule made concrete: Census transform, keypoint
detection and their tests depend on no platform library at all, so they build
with plain `make` on both machines. See
[doc/06-preprocessing.md](doc/06-preprocessing.md).

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

- [~] **1. Capture pipeline and timestamp sanity** — substantially done; only
      hardware-sync verification outstanding (needs a moving scene)
- [!] 2. IMU Allan variance — **BLOCKED: no IMU on the system.** `nvs_bmi160` and
      its device-tree node are stock L4T leftovers with nothing behind them;
      verified by bus scan and CHIP_ID read. See
      [doc/01-hardware.md](doc/01-hardware.md).
- [ ] 3. Calibration — intrinsics/extrinsics, then hand-eye + time offset (Kalibr)
- [ ] 4. Static bags vs laser rangefinder — walls at 1, 2, 3 m
- [ ] 5. Drive

## Step 1

Build and deploy from the desktop in one step (seconds — two translation units
against a prebuilt `.so`):

```sh
./tools/deploy.sh
```

Or on the Jetson directly. Note cmake there is **3.10**, which supports neither
`-S`/`-B` (3.13+) nor `--build -j` (3.12+) — passing either makes cmake print its
usage and exit *without building*:

```sh
mkdir -p ~/doubleeye/jetson/build && cd ~/doubleeye/jetson/build
cmake .. && make -j6
```

Then, **before measuring anything** — this is not optional, see finding 1:

```sh
sudo nvpmodel -m 0 && sudo jetson_clocks

./build/rs_probe --save-prefix /tmp/probe
./build/rs_ir_capture ~/bags/$(date +%Y%m%d_%H%M%S)_static --seconds 120
```

`rs_probe` is the gate. It fails loudly on the things that quietly invalidate
everything downstream:

- **Power mode / unlocked clocks** — measured at a 34% frame loss with no error.
- **USB2 fallback.** 848×480@30 on two IR streams does not fit in USB2
  bandwidth. Presents as mysterious frame drops, not as an error.
- **Missing 848×480 Y8 profile** on either stream index.

It also prints factory intrinsics and the IR baseline, so `f·B` can be checked
against the plan's ~21 px·m before any depth number is believed, and reports
exactly which frame metadata this kernel exposes.

Note `rsync` is **not** installed on the TX2, so pull bags with tar over ssh:

```sh
ssh jetson "cd ~/bags/<run> && tar czf - ." | tar xzf - -C bags/<run>/
.venv/bin/python desktop/capture_report.py bags/<run>
```

### Step 1 finding 1 — frame-rate shortfall (RESOLVED: power mode)

Worth reading even though it is fixed, because of how it presented: a **34%
frame loss with no error anywhere**, on a confirmed USB 3.2 link, with
perfectly contiguous frame numbers.

The Jetson was in power mode 3 (`MAXP_CORE_ARM`) with both Denver cores offline
(`cpu online = 0,3-5`) and `jetson_clocks` never applied. `sudo nvpmodel -m 0
&& sudo jetson_clocks` fixed every configuration outright:

| requested | mode 3 | MAXN + clocks locked |
|---|---|---|
| 848×480 @ 30 | 18.53 | **29.79** |
| 848×480 @ 60 | 22.08 | **59.54** |
| 848×480 @ 90 | 29.88 | **89.34** |
| 640×480 @ 30 | 17.62 | **29.79** |
| 1280×720 @ 30 | 10.64 | **29.79** |

Diagnostics that pinned it down, all with `--save-every 0` so disk I/O was not
a factor: a **single** IR stream made 29.80 fps at 848×480@30 even in mode 3,
so the limit was shared rather than per-frame; and requesting **90** fps in
mode 3 delivered ~30, proving the hardware could move 24 MB/s and that this was
never a bandwidth wall. At MAXN, 848×480@90 on both streams sustains ~72 MB/s.

So there is a lot of headroom. 60 or 90 Hz is worth considering later — a
shorter inter-frame interval directly shrinks the parallax that the plan's IMU
rotation compensation has to search over.

Two operational consequences, both now handled in code:

- **`jetson_clocks` does not survive a reboot.** `nvpmodel -m 0` does. So this
  regression can return silently at any boot and will look like a fresh bug.
  Both tools now read the power state at startup and print a loud warning, and
  `run.txt` records `cpus_online` and `clocks_locked` so a recording can never
  be misread after the fact.
- Max clocks means max power draw, which matters on a battery. Deliberately
  *not* forced at boot — the tools warn instead, leaving the policy call open.

### Step 1 finding 2 — no UVC metadata node (OPEN)

uvcvideo is unpatched, so there is no metadata node. Confirmed
empirically by `rs_probe`. Consequences, in descending order of damage:

- `get_timestamp()` reports domain **System Time**, i.e. host arrival time, not
  the camera clock. So clock skew cannot be read off the timestamps directly.
  (It turns out to be recoverable another way, and to matter less than feared —
  see the verdict below.)
- `RS2_FRAME_METADATA_FRAME_COUNTER` is absent, so `get_frame_number()` is a
  *host-side counter of delivered frames*. It is contiguous by construction,
  which makes frame-number gap analysis worthless — it reports a flawless run
  while a third of the frames are missing. `capture_report.py` now detects this
  and refuses to draw the conclusion.
- For the same reason, L/R "pairing" and the L/R timestamp difference are both
  true by construction. **Hardware sync is currently unverified**, despite
  looking perfect.
- `FRAME_LASER_POWER_MODE` is absent, so under `EMITTER_ON_OFF` alternation
  there is no per-frame label for which frames had the projector lit.
- `ACTUAL_EXPOSURE` is absent, so the fixed exposure can only be trusted from
  the option readback, not confirmed per frame.

The fix for all five is the same: apply librealsense's L4T uvcvideo patch for
kernel 4.9.140 and rebuild the module. Kernel headers for 4.9.140-tegra are
already installed and `/lib/modules/4.9.140-tegra/build` exists; the librealsense
source tree (which carries the patch script) is not present and would need
cloning.

Workaround already in use for the emitter A/B: run the projector **on** and
**off** as two separate recordings rather than alternating within one. On a
static scene that is equally valid and needs no per-frame label. It does not
help for a moving scene, where alternation would be the only option.

### Known flakiness: hwmon calibration read

`rs_probe` normally fails its **first** calibration read after the pipeline
starts (`hwmon command 0x15 failed`) and succeeds on the second. Observed
repeatedly, always returning identical values when it succeeds, so it is a
firmware warm-up quirk rather than a calibration fault. The probe now retries up
to four times and, if that fails too, continues without intrinsics rather than
aborting — the timestamp and metadata checks are the more important output.

Do not read this as a calibration problem. Rerun it.

### Verdict on the patch: not worth it for timing

Measured on a 120 s, 3600-frame recording at 848×480@30 with clocks locked:

| | value | in px at 100 °/s |
|---|---|---|
| frame interval, median | 33.352 ms (p1 33.301, p99 33.407) | — |
| arrival jitter, std | 0.062 ms | 0.05 px |
| arrival jitter, p99 | 0.050 ms | **0.04 px** |
| arrival jitter, max | 3.574 ms | 2.69 px |

The plan's coarse-to-fine search radius is ±3–4 px. Arrival jitter costs
**0.04 px at p99** — over an order of magnitude below the budget. Only a single
worst-case outlier approaches it. So host arrival time is good enough, and the
uvcvideo patch is **not** justified on timing grounds.

**And the clock-skew risk is smaller than the plan feared.** Two reasons:

1. The camera-vs-host rate *is* measurable after all — not from frame
   timestamps, but from the frame **generation rate** fitted against host
   arrival times. It comes out at **+568 ppm** (reproduced as +569 ppm on an
   independent run), determined to well under 1 ppm by the fit. It does not
   decompose into "camera crystal" versus "host clock" without an external
   reference, but the combined figure is exactly what is needed to predict when
   the next frame lands in host time.
2. More importantly, the risk largely evaporates: frames are stamped on the host
   clock and the IMU is *also* on the host clock, so cross-clock skew never
   enters the alignment. What remains is a constant exposure→arrival latency,
   which is precisely what Kalibr `--time-calibration` estimates, plus the
   0.05 ms of jitter above.

Caveat: these are **idle-machine** numbers. Arrival-time jitter is a function of
system load, so it must be re-measured with the full pipeline running and the
vehicle driving before being relied on.

What the patch would still buy, in order of remaining value: verification that
the hardware sync is real; a per-frame projector label for emitter alternation
*in motion*; per-frame exposure confirmation. None is urgent.

### Step 1 finding 3 — the projector premise, quantified

The plan's Census-over-learned-descriptors decision rests on the IR projector
being the dominant illumination indoors. Two 848×480@30 runs on the same static
scene, one emitter-on and one emitter-off, using 7×7 local standard deviation —
the same window Census would read:

| metric | emitter ON | emitter OFF |
|---|---|---|
| mean intensity | 89.9 | 88.1 |
| 7×7 local std, median | **3.91** | **1.65** |
| textureless fraction (local std < 2 DN) | **24.5%** | **56.7%** |

The projector **more than halves the textureless area, 57% → 25%**, and lifts
median local contrast 2.4×. That is the plan's premise confirmed with numbers.

Note the trap: **mean intensity moved only 1.8 DN** and is useless as a
discriminator here, because a blown-out window dominates the mean while the
projector contributes only local high-frequency structure. Any projector A/B
must use a local-contrast statistic. `capture_report.py` reports local std for
this reason.

Also measured: **6.5% of pixels saturated** (a sunlit window). Saturated regions
carry no Census information, so expect keypoint deserts there. And the two IR
sensors differ by 2.6 DN in mean level — irrelevant for Census, which is
invariant to monotonic intensity mappings, but it would matter for SSD/NCC.

### What step 1 has to establish

- [x] **Delivered rate matches requested** on both channels — 29.79 of 30 fps
      once clocks are locked.
- [x] **Which domain `get_timestamp()` reports.** System Time, i.e. host
      arrival. Not the camera clock.
- [x] **A unimodal frame-interval histogram at 33.3 ms.** Median 33.352 ms,
      p1 33.301, p99 33.407 — tight and unimodal.
- [x] **Both IR streams visually verified.** See `bags/*/preview/`. Projector
      dots plainly visible on near surfaces.
- [ ] **Hardware sync is real.** Currently *unverifiable*: with a host-side
      counter and host-side stamps, both the L/R pairing and the L/R timestamp
      difference come out perfect by construction. Needs either the uvcvideo
      patch, or an external check — a moving scene, where a genuine
      inter-channel exposure offset shows up as horizontal shear.
- [ ] ~~Camera-vs-Jetson clock drift in ppm~~ — **not measurable** without the
      patch. Substitute: arrival jitter about the fitted frame grid, converted
      to pixels of misregistration at 100 °/s. That is the number that decides
      whether the patch is worth doing.

Note on item 1: "zero dropped frames" was deliberately dropped as a criterion.
It cannot be established on this configuration, because the only frame counter
available counts delivered frames. Achieved-vs-requested rate replaces it.

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
