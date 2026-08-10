# Hardware

## Platform as built

| Component | Identifier / spec | Role |
|---|---|---|
| Intel RealSense D435 | serial `818312071289`, firmware `05.11.06.200` | Stereo IR pair. The **IR streams are used directly**; the RGB stream is not. |
| NVIDIA Jetson TX2 | host `jetsontx2`, L4T R32.1, kernel `4.9.140-tegra`, aarch64 | On-vehicle compute |
| — CPU | 6 cores: 2× Denver 2 + 4× Cortex-A57, max 2.04 GHz | Denver cores are **offline in some power modes** — see obstacle 5 |
| — Memory | 8 GB LPDDR4 shared CPU/GPU, EMC up to 1866 MHz | 58.4 GB/s theoretical; the real bottleneck per the plan |
| — Storage | 28 GB eMMC root | ~13 GB free. A 120 s recording saving every 30th frame is ~63 MB. |
| USB 3 link | `/sys/devices/3530000.xhci/usb2/2-1`, `speed=5000`, descriptor `3.2` | Camera connection |
| IMU | **Pixhawk 2.4.8** running **ArduPilot**, powered and reachable | `/dev/ttyACM0` (USB) and `/dev/ttyTHS2` @ 57600 (TELEM2). Streams IMU at only 3.2 Hz as configured. |
| RC car | **no battery yet — stationary** | Indoor platform, apartment scale (0.3–3 m). Driven by RC initially; control from the Jetson later. |
| WiFi | TX2 at `192.168.2.114`, ~100 ms RTT | Development link only, not used in the data path |

### Why the IR streams and not RGB

From the plan, confirmed in practice: the IR pair is global shutter,
hardware-synchronised, shares exposure control, and arrives already rectified
from the D4 ASIC. The RGB camera is rolling shutter and therefore unusable for
stereo. Verified rectification: distortion coefficients are all exactly zero and
the IR1→IR2 rotation is exactly the identity matrix — see
[04-baseline-measurements.md](04-baseline-measurements.md).

The on-board depth stream is deliberately **never enabled**, which saves USB
bandwidth and ASIC power. Only `infrared 1` and `infrared 2` in `Y8` are opened.

### Note on the IMU

The camera is a **D435, not a D435i** — it has no internal IMU. The vehicle IMU
is therefore a physically separate device on its own clock. This matters: it is
the origin of the timestamp-alignment risk the plan flags. As it turns out the
risk is smaller than feared, because frames end up stamped on the Jetson clock
and the IMU is on the Jetson clock too — see
[03-obstacles.md](03-obstacles.md), obstacle 7.

### There is currently no IMU on the system — verified

A `nvs_bmi160` kernel module is loaded and the device tree declares
`bmi160@69`, which initially looked like a wired-up Bosch BMI160. It is not.
Checked as follows:

| Check | Result |
|---|---|
| `i2cdetect -y -r 1` (the bus the DT node sits on) | **completely empty** — no device responds at any address |
| `i2cget -y 1 0x69 0x00` (BMI160 `CHIP_ID`, expect `0xd1`) | `Error: Read failed` |
| `/sys/bus/i2c/devices/1-0069/driver` | `NONE` — **unbound** |
| Devices bound to the `bmi160` driver | none |
| `nvs_bmi160` module use count | **0** |
| IIO devices (`/sys/bus/iio/devices/`) | four, all `ina3221x` — power monitors, not motion sensors |
| USB devices | only the RealSense D435 |
| `/dev/ttyUSB*`, `/dev/ttyACM*` | none present |

Every i2c device actually on the board is accounted for, and none is an IMU:

```
0-0040..0043  ina3221x   power monitors
0-0074, 0-0077 tca9539   GPIO expanders
1-0069        bmi160     <UNBOUND> — device-tree phantom, no hardware
2-0036        ov5693     on-board camera
3-0050, 3-0054           HDMI DDC / SCDC
4-003c        max77620   PMIC
4-0068        dummy      claimed by the PMIC/BPMP, not an IMU
7-004c        tmp451     temperature sensor
```

**Conclusion.** The `nvs_bmi160` module and the `bmi160@69` device-tree node are
stock L4T leftovers for the TX2 developer kit. This carrier does not have the
chip fitted, and no external IMU is attached. The module loads, finds nothing,
and binds nothing — which is why it produces no IIO device and no error.

**This blocks bring-up step 2** (IMU Allan variance) and, after it, step 3's
hand-eye and time calibration.

None of this changes with the Pixhawk: the BMI160 the device tree talks about is
a different, absent chip. The actual IMU is in the Pixhawk, below.

## The IMU is a Pixhawk 2.4.8 running ArduPilot

Powered and identified. Two working links, both verified at 100% clean MAVLink
framing:

| Link | Device | Baud | Notes |
|---|---|---|---|
| USB | `/dev/ttyACM0` | n/a (CDC) | Enumerates as `26ac:0011`, "PX4 FMU v2.x" |
| TELEM2 | `/dev/ttyTHS2` | **57600** | 115200 framed only 83.5% with nonsense system ids, so 57600 is correct |

**The firmware is ArduPilot, not PX4.** The USB descriptor says "PX4 FMU v2.x",
which is the bootloader and board identity and is not a reliable guide.
`HEARTBEAT` decodes to `ARDUPILOTMEGA`, vehicle type `GROUND_ROVER` — sensible
for an RC car. Confirmed independently by the presence of `AHRS2` (178) and
`AHRS3` (182), which are ArduPilot-specific messages.

Verify any of this with:

```sh
ssh jetson 'sudo timeout 6 dd if=/dev/ttyACM0 bs=1 count=40000 2>/dev/null | base64' \
  | base64 -d > /tmp/mav.bin
.venv/bin/python desktop/mavlink_probe.py /tmp/mav.bin --seconds 6
```

### Access permission

The `nvidia` user was **not** in `dialout`, so it could not open either port —
both are `root:dialout` mode 660. Fixed with `sudo usermod -aG dialout nvidia`;
takes effect for new logins. Until then, `sudo` is needed.

### What still needs configuring — the IMU rate is far too low

As shipped, the stream carries:

| Message | Rate |
|---|---|
| `AHRS3`, `VFR_HUD`, `ATTITUDE`, `AHRS2` | ~15.6 Hz |
| **`RAW_IMU`, `SCALED_IMU2`** | **3.2 Hz** |
| `HEARTBEAT` | 1.5 Hz |
| `TIMESYNC` | 0.3 Hz |

**3.2 Hz is unusable.** The plan's gyro rotation compensation works over a 33 ms
frame interval and wants 100 Hz or better. Two separate fixes, for two separate
purposes:

- **Runtime (rotation compensation).** Raise ArduPilot's stream rates:
  `SR0_RAW_SENS` for USB (SERIAL0) or `SR2_RAW_SENS` for TELEM2 (SERIAL2), to
  50–100. Note TELEM2 at 57600 baud carries only ~5.7 kB/s total, so high-rate
  IMU there competes with everything else; raise `SERIAL2_BAUD`, or prefer USB.
- **Allan variance (step 2).** Do **not** use a stream at all. Enable raw IMU
  logging to the SD card (`LOG_BITMASK` including IMU, plus
  `INS_LOG_BAT_MASK`/`INS_LOG_BAT_OPT` for fast sampling), record a multi-hour
  static log, and pull the `.bin`. Allan variance needs a gap-free record at a
  stable rate, which no stream provides. **This requires an SD card in the
  Pixhawk** — presence not yet confirmed.

`TIMESYNC` being present is useful later: it is ArduPilot's own mechanism for
relating its clock to the companion's, which is exactly the alignment problem
step 3 has to solve.

Setting parameters needs a tool. Nothing suitable is installed yet — no
`mavproxy`, no `pymavlink`, no `socat` on either machine. Either use a ground
station (Mission Planner or QGroundControl), or `pip3 install --user pymavlink`
on the Jetson and script it.

### Powering it, for reference

Three possible power inputs. USB is what is in use now.

### 1. Micro-USB — use this

Plug the Pixhawk's micro-USB into a Jetson USB port. That is the whole
procedure. It powers the flight controller and its sensors from 5 V, draws
roughly 250 mA (well inside USB limits), and presents a USB CDC-ACM device, so it
should appear as `/dev/ttyACM0`.

No battery, no power module, no soldering. For reading IMU data on the bench this
is both the simplest and the correct option.

Two cautions. Clone 2.4.8 boards often have a weak micro-USB socket, so it is a
poor choice for a vibrating vehicle — see option 2 for that. And USB does **not**
power the servo rail, which is irrelevant here since we drive nothing.

### 2. POWER port plus a power module — for on-vehicle use

The 6-pin `POWER` port takes the bundled power module, which sits between battery
and board, feeds the Pixhawk a regulated 5.3 V, and additionally provides
voltage and current sense. This is the intended flight-power path and is
mechanically far more robust than micro-USB.

If you go this way, connect the Pixhawk to the Jetson over a `TELEM` UART rather
than USB — Jetson side that is `/dev/ttyTHS1/2/3`, which do exist. Requires a
level-appropriate connection and a matching baud rate on both ends.

USB and the power module may be connected **at the same time** safely; the board
diode-ORs its power inputs.

### 3. Servo rail — avoid

The rail can back-power the board through a BEC, but it is the least protected
path and depends on jumper and fuse details that vary between clones. There is no
reason to use it here.



## How the D435 actually produces depth

Worth setting down, because it determines what this project is replacing.

The device reports **two independent sensors**, confirmed with
`rs-enumerate-devices`:

| sensor | streams |
|---|---|
| Stereo Module | Infrared 1, Infrared 2, **Depth** |
| RGB Camera | Color |

Note which sensor owns Depth. It is computed by the D4 ASIC from the **stereo IR
pair**, and the RGB camera contributes nothing to it — RGB exists to colour a point
cloud, and is aligned to depth in software afterwards. Unplugging colour entirely
would not change a single depth value.

So this project consumes IR1 and IR2 directly and does in software what the D4's
stereo block does in silicon. That is the whole point: the ASIC's correspondence
search is fixed, and MASDA is a different one.

### It is active stereo, not structured light and not time-of-flight

The projector emits a fixed pseudo-random dot pattern in the near infrared. It is
**not a coded pattern that gets decoded**: nothing anywhere in the pipeline knows
what the pattern is. It exists only to paint texture onto surfaces that have none,
so that stereo matching has something to match.

Three consequences follow, and all three are visible in our own data:

- Depth still works with the emitter off, just worse on blank surfaces. A
  structured-light sensor produces nothing without its pattern.
- The measured effect is on *texture*, not brightness: the projector takes the
  textureless area from 56.6% to 24.9% while moving mean intensity by 1.8 DN. That
  is obstacle 12 — mean intensity cannot see the projector at all.
- Two of these cameras pointed at the same scene degrade each other gracefully
  rather than corrupting each other, because neither is trying to decode anything.

### The IR imagers are not infrared-only

This surprises people, including me before measuring it. With the emitter **off**,
a single frame has mean 88 of 255, 6.4% of pixels saturated, and 43% of the image
carrying real local texture. In a room lit by ordinary visible light, that light is
what the imagers are responding to: they are broadband monochrome sensors sensitive
to visible *and* near infrared, not narrowband IR detectors.

Which is why the IR frames in `/doubleeye/image_matches` look like photographs of
the room with dots sprinkled on, rather than like a dot pattern floating in
blackness. It also means ambient lighting affects matching quality directly, and
that the exposure setting matters for the picture as well as for the projector.

### Why the projector both helps and hurts

It halves the textureless area, which is what makes it worth using. But every dot
is **identical to every other dot**, so a Census descriptor centred on one carries
almost no identity: 3.3x degenerate on real frames (see
[06-preprocessing.md](06-preprocessing.md)). The projector converts *no information*
into *ambiguous information*.

That is precisely the regime MASDA's uniqueness constraint is for, and it is why
this camera is a reasonable platform to test it on. A matcher relying on descriptors
alone cannot separate the dots; a matcher that also knows two left keypoints may not
claim the same right keypoint can.

## Are we better than the ASIC? Unknown, and it is measurable

Nobody has checked, and the comparison needs stating
carefully because the two produce different things.

| | D4 ASIC | this project |
|---|---|---|
| output | dense, a disparity per pixel | sparse, ~650-1900 correspondences |
| cost | free, it is silicon | 23.07 ms of a 33.3 ms budget |
| confidence | not exposed per pixel | score margin per match, 0.169-0.659 precision by quartile |
| tunable | a handful of preset filters | every term in the objective |

"Better" is therefore only meaningful per question. Where we could plausibly win:

- **On degenerate texture.** The projector makes every dot identical, and the
  uniqueness constraint is exactly the extra information that separates them. This
  is the whole thesis of the project and it is untested against the ASIC.
- **On sub-pixel accuracy**, since the parabola fit on the inter-window cost is a
  choice we made and can improve; median inlier error is 0.167 px.
- **On knowing when to be believed.** The margin is a real per-match confidence.

Where we will not win: density, or cost. The ASIC does dense correspondence at
90 Hz for zero CPU.

**The experiment**, which needs one small change: `rs_ir_capture` records IR only,
so it would have to also record the `Depth` stream. Then, at our matched keypoints,
compare our disparity against the ASIC's, on (a) a flat wall, where both can be
scored against a fitted plane without any external ground truth, and (b) a scene
with the emitter off, where texture is scarce and the constraint should matter most.
That is a real answer to "are we better", and it is a day's work.

### What others found when they replaced the ASIC

The question "can a custom matcher on the raw IR pair beat the on-board pipeline"
has been asked before, and the answer in the literature is consistently yes. That
does not mean *this* matcher will, but it does mean the ceiling is not the ASIC.

**Learned active stereo beats it, clearly.** ActiveStereoNet trains end-to-end on
the raw IR pair with no ground truth at all, using a self-supervised reconstruction
loss, and reports **1/30th of a pixel** subpixel precision along with explicit
prediction of invalid regions such as occlusions. It is the reference point for
what the raw streams can support, and it is a Google/Princeton result on this class
of hardware.

> Y. Zhang, S. Khamis, C. Rhemann, J. Valentin et al. (2018). *ActiveStereoNet:
> End-to-End Self-supervised Learning for Active Stereo Systems.* ECCV.
> [doi:10.1007/978-3-030-01237-3_48](https://doi.org/10.1007/978-3-030-01237-3_48)

> R. Liu, S. Yang, A. Tao et al. (2022). *ActiveZero: Mixed Domain Learning for
> Active Stereovision with Zero Annotation.* CVPR.
> [doi:10.1109/CVPR52688.2022.01269](https://doi.org/10.1109/CVPR52688.2022.01269)

**Classical methods beat it too**, which matters more to us because we are also
classical. A custom infrared SGM on a RealSense IR pair reports depth maps with
greater completeness, higher quality and longer range than the camera's own
algorithm.

> S. Zhong, M. Li, X. Liao, L. Qin (2020). *A Real-Time Infrared Stereo Matching
> Algorithm for RGB-D Cameras' Indoor 3D Perception.* ISPRS Int. J. Geo-Inf. 9(8).
> [doi:10.3390/ijgi9080472](https://doi.org/10.3390/ijgi9080472)

Two things make this plausible rather than surprising. The ASIC runs a fixed
Census-based pipeline chosen for silicon area and latency, not accuracy, and it
leaves uncertain pixels invalid rather than reasoning about them. Both are exactly
the kind of constraint a software matcher does not have.

**What this predicts for us, and what it does not.** Every result above is *dense*
and most are *learned*. None of them is a sparse matcher with a uniqueness
constraint, so none of them is evidence that MASDA specifically wins. What they do
establish:

- The on-board pipeline is a beatable baseline, so the comparison is worth running
  rather than assumed lost.
- Sub-pixel precision is where the headroom is. 1/30 px is roughly five times better
  than our current 0.167 px median inlier error, and our refinement is a
  three-point parabola on a SAD cost -- the cheapest thing that works.
- The invalid-pixel handling the ASIC skips is the same territory as our score
  margin. Knowing which matches to distrust is a recognised axis, not a private
  hobby-horse.

The direct comparison against the ASIC is queued in [TODO.md](TODO.md) 0.5, and it
should now be read as "where do we sit on a scale others have already climbed",
which is a better question than "are we better".

## Would adding the RGB camera help the matcher?

The geometry says yes and the photometry says no, and the photometry wins.

**The geometry is genuinely favourable.** From the device's own extrinsics:

| pair | baseline | rotation |
|---|---|---|
| IR1 - IR2 | 49.88 mm | exactly identity, delivered rectified |
| IR1 - Color | 14.70 mm | 0.26° off identity |
| IR2 - Color | 64.58 mm | (Color sits opposite IR2 across IR1) |

Three different baselines is the classic multi-baseline configuration, and
multi-baseline stereo exists precisely to kill the ambiguity that repetitive texture
creates: a false match at one baseline does not survive at another, because the
disparity that would explain it differs. That is our periodic-texture failure mode,
the one where precision collapses to 0.204 for every method including the exact
solver.

> M. Okutomi and T. Kanade (1991). *A multiple-baseline stereo.* CVPR.
> [doi:10.1109/CVPR.1991.139662](https://doi.org/10.1109/CVPR.1991.139662)

> R. Hansen, N. Ayache and F. Lustman (1990). *High Speed Trinocular Stereo for
> Mobile-Robot Navigation.*
> [doi:10.1007/978-3-642-84051-7_9](https://doi.org/10.1007/978-3-642-84051-7_9)

**But four things break it, and the first is fatal.**

1. **The colour camera cannot see the projector.** It has an IR-cut filter. On
   exactly the blank surfaces where the projector supplies all the texture — the
   ones that took textureless area from 56.6% to 24.9% — RGB contributes nothing.
   It adds information only where surfaces already have visible-light texture,
   which is where matching already works.

2. **IR-to-RGB is cross-spectral stereo, a distinct and harder problem.** Census
   compares a pixel against its neighbours, and the *ordering* of those comparisons
   is not preserved across spectra: two materials with the same visible brightness
   can differ in the near infrared and vice versa. Census and intensity correlation
   degrade badly across spectra; the literature moves to gradient-based or learned
   material-aware features for this reason.

   > P. Pinggera, T. Breckon and H. Bischof (2012). *On Cross-Spectral Stereo
   > Matching using Dense Gradient Features.* BMVC.
   > [doi:10.5244/C.26.103](https://doi.org/10.5244/C.26.103)

   > T. Zhi, B. Pires, M. Hebert and S. Narasimhan (2018). *Deep Material-Aware
   > Cross-Spectral Stereo Matching.* CVPR.
   > [doi:10.1109/CVPR.2018.00205](https://doi.org/10.1109/CVPR.2018.00205)

3. **Rolling shutter.** The IR imagers are global shutter; the colour camera is
   not. On a moving vehicle that is geometric distortion that varies down the
   frame, and this project's whole point is a moving vehicle.

4. **Narrower field of view**, 69.1° against the IR pair's 89.1°, so the third view
   covers only the middle of the scene. And the IR1-Color rotation is not identity,
   so unlike the IR pair it would need real rectification rather than none.

**Recommendation: not as a matching source.** The one mechanism that would help —
multi-baseline disambiguation — needs the third view to see the same texture, and
the projector is invisible to it. What is left is a narrower, rolling-shutter,
cross-spectral view contributing mainly where matching is already easy.

RGB earns its place elsewhere: colouring the point cloud, and later as the input to
object detection and appearance-based tracking, where colour carries information
geometry does not. That is also where the GPU discussion in
[10-architecture.md](10-architecture.md) expects it.

**If multi-baseline is what is wanted**, the direct route is a second D435, or an
IR camera without an IR-cut filter, positioned to give a different baseline against
the existing pair. Then all views see the projector and the Okutomi-Kanade argument
applies without a spectral penalty.

## Still required

Not yet present, needed for the bring-up steps that follow.

| Item | Needed for | Notes |
|---|---|---|
| **SD card for the Pixhawk** | Step 2 — Allan variance | Onboard logging at full rate is much better input than a MAVLink stream. |
| Laser rangefinder | Step 4 — static ground truth | Walls at 1, 2, 3 m. The plan calls this the only realistic indoor ground truth, and it is enough to verify sub-pixel accuracy. |
| Calibration target | Step 3 — Kalibr | Checkerboard or AprilGrid. Kalibr with `--time-calibration` runs on Ubuntu 18.04/Melodic, so the old toolchain is an advantage here. |
| Foam IMU mount | Vibration decoupling | An RC car on hard flooring excites the IMU above Nyquist; MEMS accelerometers rectify that into an apparent bias, tilting the gravity vector speed-dependently in a way that *looks plausible*. |
| Short, strain-relieved USB3 cable | Reliability | The plan names this the most common hardware failure. Vibration-induced disconnects present as software bugs. Currently negotiating `3.2` correctly, so whatever is fitted is adequate at rest — but it has not been tested under vibration. |

## Hardware characteristics worth remembering

**Depth resolution** at 848×480, from measured calibration (f·B = 21.48 px·m),
assuming 0.1 px disparity precision:

| Range | Disparity | Δz |
|---|---|---|
| 1 m | ~21 px | ~5 mm |
| 2 m | ~10.7 px | ~2 cm |
| 3 m | ~7.2 px | ~4 cm |

This matches the plan's prediction. Note it does **not** transfer to automotive
ranges — the same geometry gives roughly 1.2 m error at 20 m.

**Motion blur.** At 2 m/s lateral and 1 m distance, image motion is ≈ 850 px/s.
A 5 ms exposure is therefore >4 px of blur. Exposure is set manually to 1–2 ms
with raised gain; auto-exposure additionally hunts while driving and makes frame
intervals unreproducible.

**Available IR profiles** (Y8, both stream indices 1 and 2):

```
 424x240  @ 6/15/30/60/90 Hz      1280x720  @ 6/15/30 Hz
 480x270  @ 6/15/30/60/90 Hz      1280x800  @ 15/30 Hz
 640x360  @ 6/15/30/60/90 Hz       848x100  @ 100 Hz
 640x480  @ 6/15/30/60/90 Hz       848x480  @ 6/15/30/60/90 Hz   <- used
```

848×480 is the native depth resolution and the plan's choice. At locked clocks
the link sustains 848×480@90 on both channels (~72 MB/s), so there is
considerable headroom above the 30 Hz currently used.
