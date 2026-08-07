#!/usr/bin/env python3
"""Identify a MAVLink stream from a raw byte dump.

Deliberately a dump parser rather than a live client. Capturing bytes on the
Jetson with dd and analysing them here keeps the same split as everything else:
the vehicle side stays dependency-free, and nothing needs pymavlink installed on
a Python 3.6 box.

Reports which framing version is in use, the message-id histogram with names,
observed rates, and -- decoded from HEARTBEAT -- which autopilot firmware is
actually running. That last one matters because the USB descriptor is not a
reliable guide: a board can enumerate as "PX4 FMU v2.x", which is a bootloader
and board identity, while running ArduPilot.

Frames are validated structurally (magic, declared length, and the next frame
starting exactly where the previous one ends) rather than by CRC, which would
need a per-message CRC_EXTRA table. That is enough to identify a stream.

Desktop-side. Standard library only.
"""

from __future__ import annotations

import argparse
import struct
from collections import Counter
from pathlib import Path

MAGIC_V1 = 0xFE
MAGIC_V2 = 0xFD

# Only the messages worth commenting on for this project.
NAMES = {
    0: "HEARTBEAT",
    1: "SYS_STATUS",
    2: "SYSTEM_TIME",
    24: "GPS_RAW_INT",
    26: "SCALED_IMU",
    27: "RAW_IMU",
    29: "SCALED_PRESSURE",
    30: "ATTITUDE",
    32: "LOCAL_POSITION_NED",
    33: "GLOBAL_POSITION_INT",
    35: "RC_CHANNELS_RAW",
    36: "SERVO_OUTPUT_RAW",
    42: "MISSION_CURRENT",
    62: "NAV_CONTROLLER_OUTPUT",
    65: "RC_CHANNELS",
    74: "VFR_HUD",
    77: "COMMAND_ACK",
    105: "HIGHRES_IMU",
    111: "TIMESYNC",
    116: "SCALED_IMU2",
    125: "POWER_STATUS",
    129: "SCALED_IMU3",
    136: "TERRAIN_REPORT",
    147: "BATTERY_STATUS",
    148: "AUTOPILOT_VERSION",
    150: "SENSOR_OFFSETS (ArduPilot)",
    152: "MEMINFO (ArduPilot)",
    163: "AHRS (ArduPilot)",
    165: "HWSTATUS (ArduPilot)",
    168: "WIND (ArduPilot)",
    178: "AHRS2 (ArduPilot)",
    182: "AHRS3 (ArduPilot)",
    193: "EKF_STATUS_REPORT (ArduPilot)",
    241: "VIBRATION",
    242: "HOME_POSITION",
    253: "STATUSTEXT",
}

AUTOPILOTS = {0: "GENERIC", 3: "ARDUPILOTMEGA (ArduPilot)", 12: "PX4"}
MAV_TYPES = {
    1: "FIXED_WING", 2: "QUADROTOR", 3: "COAXIAL", 4: "HELICOPTER",
    10: "GROUND_ROVER", 11: "SURFACE_BOAT", 12: "SUBMARINE",
}
# IMU-bearing messages, in the order we would prefer them.
IMU_MSGS = {105: "HIGHRES_IMU", 27: "RAW_IMU", 116: "SCALED_IMU2",
            129: "SCALED_IMU3", 26: "SCALED_IMU"}


def parse(buf: bytes):
    """Walk the buffer, yielding (msgid, sysid, compid, payload, version)."""
    frames = []
    i = 0
    n = len(buf)
    while i < n:
        magic = buf[i]
        if magic == MAGIC_V1:
            if i + 6 > n:
                break
            length = buf[i + 1]
            total = 6 + length + 2
            if i + total > n:
                break
            seq, sysid, compid, msgid = buf[i + 2], buf[i + 3], buf[i + 4], buf[i + 5]
            frames.append((msgid, sysid, compid, buf[i + 6:i + 6 + length], 1))
            i += total
            continue
        if magic == MAGIC_V2:
            if i + 10 > n:
                break
            length = buf[i + 1]
            incompat = buf[i + 2]
            signed = 13 if (incompat & 0x01) else 0
            total = 10 + length + 2 + signed
            if i + total > n:
                break
            sysid, compid = buf[i + 5], buf[i + 6]
            msgid = buf[i + 7] | (buf[i + 8] << 8) | (buf[i + 9] << 16)
            frames.append((msgid, sysid, compid, buf[i + 10:i + 10 + length], 2))
            i += total
            continue
        i += 1  # resynchronise
    return frames


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", type=Path, help="raw bytes captured from the port")
    ap.add_argument("--seconds", type=float, default=None,
                    help="capture duration, so rates can be reported")
    args = ap.parse_args()

    buf = args.dump.read_bytes()
    frames = parse(buf)
    if not frames:
        raise SystemExit(
            f"no MAVLink frames found in {len(buf)} bytes.\n"
            "Either nothing is talking, or the baud rate is wrong (a UART link\n"
            "at the wrong baud produces bytes that are pure noise).")

    versions = Counter(f[4] for f in frames)
    ids = Counter(f[0] for f in frames)
    sysids = Counter((f[1], f[2]) for f in frames)
    framed = sum(
        (6 + len(f[3]) + 2) if f[4] == 1 else (10 + len(f[3]) + 2) for f in frames)

    print(f"dump          {args.dump}  ({len(buf)} bytes)")
    print(f"frames        {len(frames)}, {framed} bytes framed "
          f"({100.0 * framed / len(buf):.1f}% of the dump)")
    print("framing       " + ", ".join(f"v{v}: {c}" for v, c in
                                       sorted(versions.items())))
    print("system/comp   " + ", ".join(f"{s}/{c} x{n}" for (s, c), n in
                                       sysids.most_common(4)))

    for msgid, _, _, payload, ver in frames:
        if msgid == 0 and len(payload) >= 9:
            custom, mtype, autopilot, base, status, mver = struct.unpack(
                "<IBBBBB", payload[:9])
            print()
            print("HEARTBEAT decoded")
            print(f"  autopilot       {AUTOPILOTS.get(autopilot, autopilot)}")
            print(f"  vehicle type    {MAV_TYPES.get(mtype, mtype)}")
            print(f"  mavlink version {mver}  (framing v{ver} on the wire)")
            print(f"  base mode       0x{base:02x}   system status {status}")
            break
    else:
        print("\nNo HEARTBEAT in this dump — capture a little longer; it is "
              "usually 1 Hz.")

    print("\nmessages seen")
    for msgid, count in ids.most_common():
        rate = f"{count / args.seconds:7.1f} Hz" if args.seconds else f"{count:5d}   "
        print(f"  {msgid:3d}  {rate}  {NAMES.get(msgid, '?')}")

    present = [m for m in IMU_MSGS if m in ids]
    print("\nIMU-bearing messages present: " +
          (", ".join(f"{IMU_MSGS[m]} ({ids[m]})" for m in present) or "NONE"))
    if not present:
        print("  None are being streamed. On ArduPilot, RAW_IMU/SCALED_IMU2 are")
        print("  in the SR*_RAW_SENS stream group, which defaults to 0 on some")
        print("  ports; raise it, or read the onboard log instead.")
    print("\nFor Allan variance, prefer the autopilot's own SD-card log over")
    print("any of these: streaming drops and re-times messages, and Allan")
    print("variance needs a gap-free record at a stable rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
