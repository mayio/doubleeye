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
| IMU | **absent** | Not present on the system. See below — this blocks bring-up step 2. |
| RC car | — | Indoor platform, apartment scale (0.3–3 m) |
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

The one possibility not excluded: an IMU wired to a raw UART. `/dev/ttyTHS1`,
`ttyTHS2` and `ttyTHS3` exist, and a passive serial device cannot be detected
without knowing its baud rate and protocol. Nothing suggests one is there, but
if the IMU is a serial module this is where it would be.

## Still required

Not yet present, needed for the bring-up steps that follow.

| Item | Needed for | Notes |
|---|---|---|
| **An IMU** | Steps 2, 3, and all IMU-dependent work in the plan | None is present. See above. The plan's gyro rotation compensation and gravity-vector ground-plane detection both depend on it. |
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
