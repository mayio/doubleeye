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

Bring-up step 2 is therefore unblocked on the hardware side. What remains is a
multi-hour static recording and the Allan-variance analysis itself.

## Flagged, deliberately not changed: `INS_GYRO_FILTER = 4 Hz`

Found while reading parameters, and it matters more than anything else here.

A **4 Hz low-pass on the gyro** removes almost all of the high-frequency content
the plan relies on. Rotation compensation over a 33 ms interval is concerned with
motion up to tens of Hz; filtering at 4 Hz smears exactly that. `INS_ACCEL_FILTER`
is 10 Hz, less critical since the plan uses the accelerometer only for
gravity/attitude.

Not changed, because unlike the parameters above it **affects the vehicle's
control loops** rather than only what gets reported. That is a decision to take
deliberately, not a side effect of instrumenting the IMU.

Two things soften it:

- `INS_LOG_BAT_MASK` logging is taken **pre-filter**, so the SD log for Allan
  variance is unaffected by this setting. The noise parameters will be right.
- It only degrades the *streamed* gyro used at runtime.

If the streamed gyro is going to drive rotation compensation, this needs raising
(20–50 Hz is typical) — and if ArduPilot is only ever an IMU here rather than a
controller, there is little reason not to.

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

## Still open

- **Multi-hour static log** for Allan variance, then the analysis.
- **`INS_GYRO_FILTER`** decision, above.
- **Time alignment.** `TIMESYNC` is present in the stream, which is ArduPilot's own
  mechanism for relating its clock to a companion's — relevant to step 3, and it
  interacts with the finding that camera frames are stamped on the Jetson clock
  rather than the camera's ([03](03-obstacles.md) obstacle 7).
- **Where the IMU physically sits** relative to the camera, which hand-eye
  calibration needs, plus foam decoupling from vehicle vibration.
