#!/usr/bin/env python3
"""Inspect and set the ArduPilot parameters that matter for this project.

Runs **on the Jetson**, inside its own virtualenv:

    ~/venvs/mavlink/bin/python pixhawk/ardupilot_config.py --report

That venv is deliberate. Anything pip-installed on the Jetson goes into an
isolated environment, never `--user` and never system-wide: the system Python is
depended on by L4T, ROS Melodic and the NVIDIA tooling, and this board cannot be
casually reflashed to recover from breaking it.

Two problems this addresses, both found by `desktop/mavlink_probe.py`:

  IMU stream rate   RAW_IMU and SCALED_IMU2 arrive at 3.2 Hz as shipped. The
                    plan's gyro rotation compensation works over a 33 ms frame
                    interval and wants 100 Hz or better. ArduPilot's SR*_RAW_SENS
                    parameters govern this, per serial port.
  Onboard logging   Allan variance needs a gap-free record at a stable rate, which
                    no MAVLink stream provides. That means the autopilot's own
                    SD-card log, which needs both an SD card present and raw IMU
                    logging enabled.

Nothing here is destructive by default: `--report` only reads. Writes happen only
for parameters named explicitly with `--set`, and every write is verified by
reading the value back.
"""

# No `from __future__ import annotations` and no builtin generics: the Jetson runs
# Python 3.6, where the former is a SyntaxError and `dict[str, int]` is invalid.
# Same constraint the C++ side has with g++ 7.5 -- easy to forget on the Python
# side because the desktop is 3.12.

import argparse
import sys
import time

from pymavlink import mavutil

# Parameters worth looking at, grouped by what they control.
GROUPS = {
    "stream rates (USB / SERIAL0)": [
        "SR0_RAW_SENS", "SR0_EXT_STAT", "SR0_POSITION", "SR0_EXTRA1",
        "SR0_EXTRA2", "SR0_EXTRA3",
    ],
    "stream rates (TELEM2 / SERIAL2)": [
        "SR2_RAW_SENS", "SR2_EXT_STAT", "SR2_POSITION", "SR2_EXTRA1",
        "SR2_EXTRA2", "SR2_EXTRA3",
    ],
    "serial ports": ["SERIAL0_BAUD", "SERIAL2_BAUD", "SERIAL2_PROTOCOL"],
    "logging (for Allan variance)": [
        "LOG_BITMASK", "LOG_DISARMED", "LOG_FILE_DSRMROT", "LOG_BACKEND_TYPE",
        "INS_LOG_BAT_MASK", "INS_LOG_BAT_OPT", "INS_LOG_BAT_LGIN",
        "INS_LOG_BAT_LGCT",
    ],
    "IMU": ["INS_ACCEL_FILTER", "INS_GYRO_FILTER", "INS_USE", "INS_USE2",
            "INS_USE3", "AHRS_EKF_TYPE"],
}


def connect(device: str, baud: int, timeout: float = 20.0):
    print(f"connecting to {device}" + (f" @ {baud}" if "tty" in device and
                                       "ACM" not in device else ""))
    master = mavutil.mavlink_connection(device, baud=baud)
    print("waiting for heartbeat...")
    hb = master.wait_heartbeat(timeout=timeout)
    if hb is None:
        sys.exit("no heartbeat — is the Pixhawk powered and the device correct?")
    print(f"  system {master.target_system}, component {master.target_component}")
    print(f"  autopilot {mavutil.mavlink.enums['MAV_AUTOPILOT'][hb.autopilot].name}")
    print(f"  type      {mavutil.mavlink.enums['MAV_TYPE'][hb.type].name}")
    return master


def fetch(master, names, timeout=2.0):
    """Read specific parameters. Targeted requests, not a full list dump, since
    a full ArduPilot parameter fetch is ~1000 messages and slow over a UART."""
    out = {}
    for name in names:
        master.mav.param_request_read_send(
            master.target_system, master.target_component,
            name.encode("ascii"), -1)
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = master.recv_match(type="PARAM_VALUE", blocking=True,
                                    timeout=0.4)
            if msg is None:
                continue
            got = msg.param_id.strip("\x00")
            if got == name:
                out[name] = msg.param_value
                break
    return out


def report(master):
    missing = []
    for group, names in GROUPS.items():
        print(f"\n{group}")
        vals = fetch(master, names)
        for name in names:
            if name in vals:
                print(f"  {name:22s} {vals[name]:g}")
            else:
                missing.append(name)
                print(f"  {name:22s} (not present on this firmware)")

    print("\n--- assessment ---")
    vals = fetch(master, ["SR0_RAW_SENS", "SR2_RAW_SENS", "SERIAL2_BAUD",
                          "LOG_BITMASK", "INS_LOG_BAT_MASK"])
    r0 = vals.get("SR0_RAW_SENS")
    if r0 is not None:
        if r0 < 50:
            print(f"  SR0_RAW_SENS = {r0:g} Hz is too low for rotation")
            print("  compensation. Suggest 100 over USB:")
            print("    --set SR0_RAW_SENS=100")
        else:
            print(f"  SR0_RAW_SENS = {r0:g} Hz — adequate")
    r2, b2 = vals.get("SR2_RAW_SENS"), vals.get("SERIAL2_BAUD")
    if r2 is not None:
        print(f"  SR2_RAW_SENS = {r2:g} Hz on TELEM2")
        if b2 is not None and b2 <= 57:
            print(f"  SERIAL2_BAUD = {b2:g} (~{b2 * 1000 / 10:.0f} B/s) is the")
            print("  bottleneck there; raise it before raising the rate, or")
            print("  just use USB for high-rate IMU.")
    if vals.get("INS_LOG_BAT_MASK", 0) == 0:
        print("  INS_LOG_BAT_MASK = 0: raw/fast IMU logging is OFF, so the")
        print("  SD-card log will not carry what Allan variance needs.")


def check_sd(master, timeout=6.0):
    """SD card presence, inferred from the logging health bit in SYS_STATUS."""
    print("\n--- SD card / logging ---")
    bit = getattr(mavutil.mavlink, "MAV_SYS_STATUS_LOGGING", 1 << 20)
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = master.recv_match(type="SYS_STATUS", blocking=True, timeout=1.0)
        if msg is None:
            continue
        present = bool(msg.onboard_control_sensors_present & bit)
        healthy = bool(msg.onboard_control_sensors_health & bit)
        print(f"  logging subsystem present={present} healthy={healthy}")
        if present and healthy:
            print("  -> logging is working, so an SD card is in and writable.")
        elif present and not healthy:
            print("  -> logging is enabled but UNHEALTHY. Almost always a missing,")
            print("     unformatted or failed SD card. Allan variance is blocked")
            print("     until that is fixed.")
        else:
            print("  -> logging not reported. Check LOG_BACKEND_TYPE.")
        return
    print("  no SYS_STATUS received within the timeout")


def apply(master, assignments):
    print("\n--- setting parameters ---")
    for item in assignments:
        if "=" not in item:
            print(f"  skip {item!r}: expected NAME=VALUE")
            continue
        name, raw = item.split("=", 1)
        name = name.strip().upper()
        try:
            value = float(raw)
        except ValueError:
            print(f"  skip {item!r}: {raw!r} is not a number")
            continue

        before = fetch(master, [name]).get(name)
        if before is None:
            print(f"  {name}: not present on this firmware, not setting")
            continue
        master.mav.param_set_send(
            master.target_system, master.target_component,
            name.encode("ascii"), value,
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        time.sleep(0.4)
        after = fetch(master, [name]).get(name)
        ok = after is not None and abs(after - value) < 1e-4
        print(f"  {name:22s} {before:g} -> "
              f"{'%g' % after if after is not None else '?'}"
              f"   {'OK' if ok else 'FAILED (read-back mismatch)'}")
    print("\nArduPilot persists parameters itself, but a power cycle is the")
    print("only way to be certain they stuck. Re-run --report afterwards.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="/dev/ttyACM0",
                    help="/dev/ttyACM0 (USB) or /dev/ttyTHS2 (TELEM2)")
    ap.add_argument("--baud", type=int, default=57600,
                    help="ignored for USB CDC; 57600 for TELEM2")
    ap.add_argument("--report", action="store_true", help="read and assess")
    ap.add_argument("--check-sd", action="store_true")
    ap.add_argument("--set", dest="assignments", nargs="+", metavar="NAME=VALUE",
                    default=None)
    args = ap.parse_args()

    if not (args.report or args.check_sd or args.assignments):
        args.report = args.check_sd = True

    master = connect(args.device, args.baud)
    if args.report:
        report(master)
    if args.check_sd:
        check_sd(master)
    if args.assignments:
        apply(master, args.assignments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
