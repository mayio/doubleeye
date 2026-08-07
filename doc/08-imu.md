# IMU — Pixhawk 2.4.8 running ArduPilot

The IMU is inside a Pixhawk 2.4.8. The board reports `26ac:0011` "PX4 FMU v2.x"
over USB, but that is the **bootloader and board identity, not the firmware**:
`HEARTBEAT` decodes to `ARDUPILOTMEGA`, vehicle type `GROUND_ROVER`. Corroborated
by `AHRS2` (178) and `AHRS3` (182), which only ArduPilot emits.

Two links, both verified at ~100% clean MAVLink framing:

| Link | Device | Baud |
|---|---|---|
| USB | `/dev/ttyACM0` | n/a (CDC) |
| TELEM2 | `/dev/ttyTHS2` | **57600** |

115200 on TELEM2 framed only 83.5% with nonsense system ids, so 57600 is right.
The `nvidia` user needed adding to `dialout` to open either.

## Role of the autopilot — narrow, on purpose

ArduPilot is **not** the controller for this project. Its jobs are:

1. **IMU** — the sensor the plan's gyro rotation compensation and gravity/ground-plane
   work depend on.
2. **Motor-controller interface** and **SBUS** — the actuator bridge.

Control commands will come from the **Jetson**, later. In the interim the vehicle
is driven by RC.

This matters for parameter decisions: ArduPilot is not running loops whose tuning
we care about, so filter and rate settings can be chosen for *measurement quality*
rather than control quality. `INS_GYRO_FILTER` below is exactly that call.

**Current state: completely stationary — there is no battery yet.** Which is the
ideal condition for bring-up step 2 rather than an obstacle: Allan variance
requires a long *static* recording, and a bench-powered vehicle that cannot move
and has no motor vibration is precisely the environment it wants. The vibration
concerns in the plan, and the untested-under-vibration caveat on arrival jitter,
all wait for the battery.

## Toolchain — isolated venv, nothing system-wide

Pip installs on the Jetson go into a dedicated virtualenv. Never `--user`, never
system-wide: the system Python is depended on by L4T, ROS Melodic and the NVIDIA
tooling, and this board cannot be casually reflashed to recover from breaking it.

The Jetson has **no pip at all** and no `ensurepip`, so the venv is bootstrapped
without touching the system:

```sh
mkdir -p ~/venvs
python3 -m venv --without-pip ~/venvs/mavlink
curl -sS https://bootstrap.pypa.io/pip/3.6/get-pip.py -o /tmp/get-pip.py
~/venvs/mavlink/bin/python /tmp/get-pip.py
~/venvs/mavlink/bin/pip install pymavlink pyserial
```

Note the **pinned `pip/3.6/` path** — the current `get-pip.py` dropped Python 3.6.

Installed: `pymavlink` 2.4.49, `pyserial` 3.5. Verified afterwards that
`python3 -c "import pymavlink"` still fails on the system interpreter, i.e. the
isolation holds.

Anything Jetson-side in Python must also be **3.6-compatible**: no
`from __future__ import annotations`, no `dict[str, int]`. Easy to forget because
the desktop is 3.12.

## Configuration applied

`pixhawk/ardupilot_config.py`, run in that venv. `--report` only reads; writes
happen solely for parameters named with `--set`, and each is verified by
read-back.

| Parameter | Before | After | Why |
|---|---|---|---|
| `SR0_RAW_SENS` | 2 | **100** | IMU stream rate over USB |
| `SERIAL0_BAUD` | 115 | **921** | Raise ArduPilot's bandwidth budget for SERIAL0 |
| `INS_LOG_BAT_MASK` | 0 | **1** | Enable raw/fast IMU logging to SD |

Measured effect on the stream: `RAW_IMU` and `SCALED_IMU2` went from **3.2 Hz to
53 Hz**.

### Why not the 100 Hz requested

**`SCHED_LOOP_RATE = 50`.** ArduPilot cannot emit a message faster than the
scheduler loop that sends it, so `SR0_RAW_SENS = 100` is clamped to ~50 Hz. That
is Rover's default loop rate, and it — not baud, not USB — is the ceiling.

Raising `SCHED_LOOP_RATE` would lift it, but that is a control-loop change with
real CPU cost on a 168 MHz FMUv2, so it is **left alone** pending a reason.

At 50 Hz there are ~1.7 gyro samples per 33 ms frame interval. Thin for the plan's
rotation compensation, which wanted 100 Hz+, though workable with interpolation.
The better answer for anything rate-hungry is the SD log, below, which is
independent of both the loop rate and MAVLink.

## SD card: present and working

Confirmed via the logging health bits in `SYS_STATUS`:
`present=True healthy=True`. So a card is in and writable, and `LOG_BITMASK` is
already `65535` with `LOG_BACKEND_TYPE = 1`.

With `INS_LOG_BAT_MASK = 1` now set, the SD log carries **raw, unfiltered IMU at
the sensor rate**, which is what Allan variance needs and which no MAVLink stream
can provide. `INS_LOG_BAT_LGIN = 20` (ms).

### Logging verified live, not just enabled

`LOG_DISARMED` was **0**, meaning ArduPilot logs only while armed. On a stationary
bench setup that produces **no log at all** — a failure that only shows up after
leaving it running for hours. Set to **1**, and it took effect without a reboot.

Confirmed empirically rather than assumed: log 1 grew from 3.49 MB to 5.11 MB over
45 s.

| | |
|---|---|
| write rate | **36 kB/s** |
| per hour | **~130 MB** |
| a 4-hour static log | ~520 MB |

`ardupilot_config.py --watch-log 45` performs that check.

**Operational caveat.** `LOG_DISARMED=1` means it logs *continuously*, ~3 GB/day.
Fine for a bench session, but it will fill the card if left indefinitely. Set it
back to 0 once the Allan-variance recording is done.

### Recording for Allan variance

The conditions are already right — stationary, bench-powered, no motor vibration.
Leave the Pixhawk powered and undisturbed for several hours, then pull the newest
`.bin` from the card and analyse it.

One thing to verify from the log itself rather than trust from parameters:
`INS_LOG_BAT_OPT` semantics (sensor-rate versus pre/post-filter) differ between
firmware versions, so **read the achieved sample rate out of the log's own
timestamps** before believing any noise figure derived from it. Allan variance is
meaningless if the true rate is not what was assumed.

Bring-up step 2 is otherwise unblocked.

## Filters raised: `INS_GYRO_FILTER` 4 → 20 Hz

Found at **4 Hz**, which removes almost all of the content the plan depends on:
rotation compensation over a 33 ms interval concerns motion up to tens of Hz, and
a 4 Hz low-pass smears exactly that. `INS_ACCEL_FILTER` was 10 Hz.

Initially flagged rather than changed, because filter settings affect control
loops. That concern dissolves given the autopilot's actual role above — it is not
running loops whose tuning matters here — so both were raised:

| Parameter | Before | After |
|---|---|---|
| `INS_GYRO_FILTER` | 4 Hz | **20 Hz** |
| `INS_ACCEL_FILTER` | 10 Hz | **20 Hz** |

**20 Hz, not higher,** for a specific reason: the MAVLink stream is capped at
~50 Hz by `SCHED_LOOP_RATE`, so Nyquist is 25 Hz. Filtering above that would alias
into the streamed samples. 20 Hz sits just under it and is also ArduPilot's own
Copter default, so it is well-trodden.

Reversible in one command if the vehicle ever does need tighter filtering:
`--set INS_GYRO_FILTER=4 INS_ACCEL_FILTER=10`.

Note this affects only the *streamed and standard-logged* gyro. `INS_LOG_BAT`
batch logging is taken pre-filter, so Allan variance is unaffected either way.

## Reading the stream by hand

**`stty raw` is mandatory.** Without it the tty is in canonical mode and mangles
binary: attempts to capture without it returned 96 and 288 bytes where the real
rate was 7.2 kB/s. It looks exactly like a dead link.

```sh
ssh jetson 'sudo stty -F /dev/ttyACM0 raw -echo 921600
            sudo sh -c "timeout 8 cat /dev/ttyACM0 > /tmp/mav.bin"
            sudo chmod 644 /tmp/mav.bin; base64 /tmp/mav.bin' \
  | base64 -d > /tmp/mav.bin
.venv/bin/python desktop/mavlink_probe.py /tmp/mav.bin --seconds 8
```

Capture to a **file** on the Jetson rather than straight down a pipe. A
`timeout … | head | base64` pipeline loses its buffer when torn down, which cost
two misleading measurements.

Also avoid `dd bs=1`: one syscall per byte makes the reader the bottleneck and
under-reports the rate.

## Allan variance — tooling built and validated, three findings

`desktop/allan_variance.py` parses a dataflash log, reports what is usable,
computes the overlapping Allan deviation and emits a Kalibr `imu.yaml`.
`ardupilot_config.py --download-log N` fetches logs over MAVLink at ~70–90 kB/s,
re-requesting gaps rather than tolerating them.

### The batch sampler needed a reboot

Setting `INS_LOG_BAT_MASK = 1` on a running unit does **not** retrofit batch
logging into the log already open. The first 20 MB downloaded afterwards contained
no `ISBH`/`ISBD` at all. `--reboot` starts a fresh log with the sampler active and
the data appears immediately.

This is the failure the whole check existed to catch: a multi-hour recording left
running after setting that parameter would have been **useless for noise density**
and would have looked fine.

### Raw sample rate: 999.6 Hz

The open question about `INS_LOG_BAT_OPT` semantics is answered empirically —
`smp_rate` reads **999.78 Hz** and the log's own timestamps confirm 999.6 Hz. Also
note the schema: the field is `mul`, not `mult`, and it is a **divisor**
(`real = raw / mul`). For the gyro `mul = 938`, so 32767/938 = 34.9 rad/s full
scale, which is the MPU6000's 2000 °/s range — a useful check that the scaling
sign is right.

### Filtered data understates noise density by ~40%, measured

Both sources in the same log:

| | filtered `IMU` @ 49.9 Hz | raw batch @ 999.6 Hz | understated by |
|---|---|---|---|
| gyroscope noise density | 7.46e−5 | **1.28e−4** rad/s/√Hz | **41%** |
| accelerometer noise density | 2.13e−3 | **3.85e−3** m/s²/√Hz | **45%** |

So the warning was not theoretical. `INS_GYRO_FILTER` at 20 Hz removes exactly the
high-frequency content the white-noise estimate is made of.

### Valid results so far

| Parameter | Value (worst axis) |
|---|---|
| `gyroscope_noise_density` | **1.28e−4 rad/s/√Hz** (0.0073 °/s/√Hz) |
| `accelerometer_noise_density` | **3.85e−3 m/s²/√Hz** |

These are trustworthy: noise density lives at short τ and needs only minutes.

**Bias instability and random walk are not yet measured.** The batch sampler
records **windows**, not a continuous stream — 38 blocks of ~1.02 s in this log.
Concatenating them fabricates continuity across the gaps, which is harmless below
one window but invalidates everything beyond, and the bias-instability minimum sits
at τ ≈ 2.6–6 s, well outside. The tool now computes the validity limit
(τ ≤ 0.51 s here), draws it on the plot, and states plainly that those two figures
are artefacts of concatenation rather than measurements.

Getting them needs continuously logged inertial data, not windowed batches — a
different `INS_LOG_BAT_OPT` setting, or the standard `IMU` messages accepting that
they are filtered, or a raw stream captured on the Jetson.

### The accelerometer is out of calibration by 7.5%

Free check, since the vehicle is stationary: gravity must read 9.80665 m/s². It
reads **9.074 m/s², −7.47%**, consistent across two independent logs (9.02 and
9.07).

That is a scale/calibration fault, not noise. It matters directly: the plan uses
the accelerometer for the gravity vector and hence for locating the ground plane.
**Run ArduPilot's 6-position accelerometer calibration** before relying on it. The
tool now checks this automatically.

### The device node is not stable

After a reboot the Pixhawk re-enumerated from `/dev/ttyACM0` to `/dev/ttyACM1`, and
a hardcoded path simply hangs waiting for a heartbeat. Use the udev by-id link,
which follows the device rather than enumeration order:

```
/dev/serial/by-id/usb-3D_Robotics_PX4_FMU_v2.x_0-if00
```

`ardupilot_config.py` now resolves this automatically.

## Still open

- **Bias instability and random walk**, which need *continuous* inertial data
  rather than the sampler's 1 s windows. Noise density is done.
- **6-position accelerometer calibration** — gravity is 7.5% low.
- **Reset `LOG_DISARMED` to 0** afterwards, or the card fills at ~3 GB/day.
- **Time alignment.** `TIMESYNC` is present in the stream, which is ArduPilot's own
  mechanism for relating its clock to a companion's — relevant to step 3, and it
  interacts with the finding that camera frames are stamped on the Jetson clock
  rather than the camera's ([03](03-obstacles.md) obstacle 7).
- **Where the IMU physically sits** relative to the camera, which hand-eye
  calibration needs, plus foam decoupling from vehicle vibration.
