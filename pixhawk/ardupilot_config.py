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
import glob
import os
import sys
import time

from pymavlink import mavutil

# Stable device path. /dev/ttyACM0 is NOT stable: after a reboot the Pixhawk
# re-enumerates and can come back as ttyACM1, at which point a hardcoded ttyACM0
# simply hangs waiting for a heartbeat that will never arrive. udev's by-id link
# follows the device instead of the enumeration order.
BY_ID_GLOB = "/dev/serial/by-id/*PX4*"


def resolve_device(requested):
    """Prefer an explicit request, then the stable by-id link, then any ttyACM."""
    if requested and os.path.exists(requested):
        return requested
    for pattern in (BY_ID_GLOB, "/dev/serial/by-id/*", "/dev/ttyACM*"):
        found = sorted(glob.glob(pattern))
        if found:
            if requested:
                print(f"note: {requested} is gone; using {found[0]}")
            return found[0]
    return requested

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


def list_logs(master, timeout=8.0):
    """List the SD-card logs.

    This is the empirical check that logging actually happens. On a stationary
    bench setup it is easy to set every parameter correctly and still produce no
    log at all, because ArduPilot only logs while armed unless LOG_DISARMED=1.
    Discovering that after leaving it running for hours is the failure worth
    avoiding.
    """
    print("\n--- SD-card logs ---")
    # Drain anything queued, then request. The log protocol is a session: without
    # a closing LOG_REQUEST_END the autopilot ignores the next LOG_REQUEST_LIST,
    # so a second listing silently returns nothing and looks like "no logs".
    while master.recv_match(type="LOG_ENTRY", blocking=False) is not None:
        pass
    master.mav.log_request_list_send(
        master.target_system, master.target_component, 0, 0xFFFF)
    entries = {}
    deadline = time.time() + timeout
    total = None
    while time.time() < deadline:
        msg = master.recv_match(type="LOG_ENTRY", blocking=True, timeout=1.0)
        if msg is None:
            continue
        total = msg.num_logs
        if msg.num_logs == 0:
            break
        entries[msg.id] = (msg.size, msg.time_utc)
        if len(entries) >= msg.num_logs:
            break
    master.mav.log_request_end_send(
        master.target_system, master.target_component)
    if total == 0 or not entries:
        print("  no logs on the card yet.")
        print("  With LOG_DISARMED=1 a log should start within seconds of boot.")
        print("  If none appears, power-cycle the Pixhawk so the parameter takes")
        print("  effect, then re-check.")
        return None
    print(f"  {total} log(s) on the card; newest few:")  # session already ended
    for log_id in sorted(entries)[-4:]:
        size, _ = entries[log_id]
        print(f"    log {log_id:<4d} {size / 1e6:8.2f} MB")
    newest = max(entries)
    return newest, entries[newest][0]


def download_log(master, log_id, out_path, max_bytes, timeout=25.0):
    """Pull a dataflash log over MAVLink.

    The SD card is inside the Pixhawk, so short of unplugging it this is the way
    to get a log onto the desktop. LOG_DATA carries 90 bytes per message, which
    is inefficient but adequate: a few tens of MB is plenty for validating the
    analysis and for Allan variance out to moderate averaging times.

    Gaps are re-requested rather than tolerated. A hole in the middle of an
    inertial record would corrupt every Allan-variance point beyond it, and a
    silently short file is exactly the failure that looks like real data.
    """
    print("\n--- downloading log %d ---" % log_id)
    # Find the size first, so a truncated download is detectable.
    while master.recv_match(type="LOG_ENTRY", blocking=False) is not None:
        pass
    master.mav.log_request_list_send(
        master.target_system, master.target_component, log_id, log_id)
    size = None
    deadline = time.time() + 8.0
    while time.time() < deadline:
        msg = master.recv_match(type="LOG_ENTRY", blocking=True, timeout=1.0)
        if msg is not None and msg.id == log_id:
            size = msg.size
            break
    if size is None:
        print("  could not read the log's size")
        return False
    want = min(size, max_bytes)
    print("  log is %.2f MB; fetching %.2f MB" % (size / 1e6, want / 1e6))

    buf = bytearray(want)
    have = bytearray(want // 90 + 2)   # one flag per 90-byte block
    blocks = (want + 89) // 90

    def request_from(offset):
        master.mav.log_request_data_send(
            master.target_system, master.target_component, log_id,
            offset, want - offset)

    request_from(0)
    got = 0
    last_report = time.time()
    started = time.time()
    idle_since = time.time()
    while got < blocks:
        msg = master.recv_match(type="LOG_DATA", blocking=True, timeout=1.0)
        if msg is None:
            if time.time() - idle_since > timeout:
                print("  timed out with %d of %d blocks" % (got, blocks))
                break
            # Re-request from the first hole rather than starting over.
            first_hole = next((i for i in range(blocks) if not have[i]), None)
            if first_hole is None:
                break
            request_from(first_hole * 90)
            idle_since = time.time()
            continue
        idle_since = time.time()
        if msg.id != log_id:
            continue
        idx = msg.ofs // 90
        if idx >= blocks or have[idx]:
            continue
        chunk = bytes(bytearray(msg.data)[:msg.count])
        end = min(msg.ofs + len(chunk), want)
        buf[msg.ofs:end] = chunk[:end - msg.ofs]
        have[idx] = 1
        got += 1
        if time.time() - last_report > 3.0:
            rate = got * 90 / max(1e-6, time.time() - started)
            print("    %5.1f%%  %.0f kB/s" % (100.0 * got / blocks, rate / 1e3))
            last_report = time.time()

    master.mav.log_request_end_send(
        master.target_system, master.target_component)
    missing = blocks - got
    with open(out_path, "wb") as fh:
        fh.write(buf)
    elapsed = time.time() - started
    print("  wrote %s: %.2f MB in %.0f s (%.0f kB/s)"
          % (out_path, want / 1e6, elapsed, want / elapsed / 1e3))
    if missing:
        print("  !! %d of %d blocks missing -- the file has holes and must NOT"
              % (missing, blocks))
        print("     be used for Allan variance.")
        return False
    print("  complete, no gaps")
    return True


def reboot(master, wait=25.0):
    """Reboot the autopilot so start-up-time settings take effect.

    Needed because some settings are only read when the subsystem initialises.
    The batch sampler is one: setting INS_LOG_BAT_MASK on a running unit does not
    retrofit it into the log already open, so a log recorded afterwards still
    contains no ISBH/ISBD and is useless for noise density -- which is exactly
    what happened here.

    Safe in the current configuration: disarmed, stationary, bench-powered, no
    battery and no motors.
    """
    print("\n--- rebooting the autopilot ---")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
        0, 1, 0, 0, 0, 0, 0, 0)
    time.sleep(3.0)
    print(f"  waiting up to {wait:.0f} s for it to come back...")
    deadline = time.time() + wait
    while time.time() < deadline:
        try:
            hb = master.wait_heartbeat(timeout=3.0)
        except Exception:
            hb = None
        if hb is not None:
            print("  back up")
            return True
        time.sleep(1.0)
    print("  no heartbeat yet; it may still be booting, or the USB device")
    print("  re-enumerated. Re-run a --report to check.")
    return False


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
    ap.add_argument("--device", default=None,
                    help="default: the stable /dev/serial/by-id link. Use "
                         "/dev/ttyTHS2 for TELEM2.")
    ap.add_argument("--baud", type=int, default=57600,
                    help="ignored for USB CDC; 57600 for TELEM2")
    ap.add_argument("--report", action="store_true", help="read and assess")
    ap.add_argument("--check-sd", action="store_true")
    ap.add_argument("--logs", action="store_true",
                    help="list SD-card logs; proves logging is really happening")
    ap.add_argument("--reboot", action="store_true",
                    help="reboot the autopilot so start-up settings take effect")
    ap.add_argument("--download-log", type=int, default=None, metavar="ID",
                    help="fetch log ID over MAVLink")
    ap.add_argument("--out", default="/tmp/ardupilot.bin")
    ap.add_argument("--max-mb", type=float, default=24.0,
                    help="cap on how much of the log to fetch")
    ap.add_argument("--watch-log", type=float, default=0.0, metavar="SECONDS",
                    help="list logs, wait, list again — confirms one is GROWING")
    ap.add_argument("--set", dest="assignments", nargs="+", metavar="NAME=VALUE",
                    default=None)
    args = ap.parse_args()

    if args.download_log is not None:
        args.report = args.check_sd = False
    if args.reboot:
        args.report = args.check_sd = False
    if not (args.report or args.check_sd or args.assignments or args.reboot
            or args.download_log is not None or args.logs or args.watch_log):
        args.report = args.check_sd = True

    device = resolve_device(args.device)
    if not device:
        sys.exit("no Pixhawk serial device found; is it powered and plugged in?")
    master = connect(device, args.baud)
    if args.report:
        report(master)
    if args.check_sd:
        check_sd(master)
    if args.assignments:
        apply(master, args.assignments)
    if args.reboot:
        reboot(master)
        return 0
    if args.download_log is not None:
        ok = download_log(master, args.download_log, args.out,
                          int(args.max_mb * 1e6))
        return 0 if ok else 1
    if args.logs or args.watch_log:
        first = list_logs(master)
        if args.watch_log and first:
            print(f"\n  waiting {args.watch_log:.0f} s to see whether it grows...")
            time.sleep(args.watch_log)
            second = list_logs(master)
            if second and second[0] == first[0]:
                delta = second[1] - first[1]
                rate = delta / args.watch_log
                print(f"\n  log {second[0]} grew {delta / 1e3:.1f} kB in "
                      f"{args.watch_log:.0f} s -> {rate / 1e3:.1f} kB/s")
                if delta <= 0:
                    print("  NOT growing. Logging is not actually running.")
                else:
                    print("  Logging is live. A multi-hour static recording for")
                    print(f"  Allan variance would be roughly "
                          f"{rate * 3600 / 1e6:.0f} MB/hour.")
            elif second:
                print(f"\n  a NEW log appeared ({second[0]}), so logging is live")
                print("  but rotating. Check LOG_FILE_DSRMROT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
