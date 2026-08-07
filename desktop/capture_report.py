#!/usr/bin/env python3
"""Offline analysis of a bring-up step 1 recording.

Consumes the directory written by jetson/rs_ir_capture and answers:

  1. Were frames dropped, and on which channel?
  2. Do the two IR channels carry matched frame numbers?
  3. What is the frame-interval distribution -- clean 33.3 ms, or bimodal?
  4. Which timestamp domain is in play, and which metadata does the kernel
     actually expose?
  5. How much timing error does that leave, expressed in the unit that
     matters: pixels of misregistration under gyro rotation compensation.

(4) decides what (5) can even mean, so the report branches on it:

  * Hardware Clock -- frame timestamps come from the camera. Camera-vs-Jetson
    clock drift is measurable, so fit it and report ppm plus residual.

  * System Time / Global Time -- uvcvideo is unpatched, there is no UVC
    metadata node, and librealsense is reporting host arrival time. Drift
    against the host is then meaningless (it would be the host against
    itself). What remains measurable, and what actually limits the plan's
    gyro-based rotation compensation, is arrival jitter about the nominal
    frame grid. That is what gets reported instead.

Desktop-side. numpy + matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

FPS_NOMINAL = 30.0
NOMINAL_INTERVAL_MS = 1000.0 / FPS_NOMINAL

# Metadata columns written by rs_ir_capture, in the same order.
MD_COLUMNS = [
    "md_frame_ts_us", "md_sensor_ts_us", "md_backend_ts_ms",
    "md_arrival_ts_ms", "md_frame_counter", "md_exposure_us", "md_gain",
    "md_laser_mode", "md_actual_fps",
]

# Rotation rate used to convert a timing error into a pixel error. The plan
# cites 100 deg/s as the regime of interest for an RC vehicle.
GYRO_DEG_PER_S = 100.0


def load(outdir: Path):
    csv_path = outdir / "frames.csv"
    if not csv_path.exists():
        sys.exit(f"no frames.csv in {outdir}")
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit("frames.csv is empty")

    meta = {}
    run_path = outdir / "run.txt"
    if run_path.exists():
        for line in run_path.read_text().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                meta[parts[0]] = parts[1]

    def col(name, dtype=float, default=np.nan):
        out = np.empty(len(rows), dtype=dtype)
        for i, r in enumerate(rows):
            raw = (r.get(name) or "").strip()
            try:
                out[i] = dtype(raw) if raw != "" else default
            except (TypeError, ValueError):
                out[i] = default
        return out

    data = {
        "stream": col("stream", int, -1),
        "frame_number": col("frame_number", int, -1),
        "ts_ms": col("ts_ms"),
        "host_monotonic_s": col("host_monotonic_s"),
        "domain": [(r.get("ts_domain") or "").strip() for r in rows],
    }
    for name in MD_COLUMNS:
        data[name] = col(name)
    return data, meta


def hardware_clock(data) -> bool:
    return all("hardware" in d.lower() for d in data["domain"] if d)


def synthetic_counter(data) -> bool:
    """True when frame_number is a host-side counter, not the camera's.

    Without the UVC metadata node librealsense cannot read the camera's frame
    counter, so it hands out a counter of *delivered* frames instead. Such a
    counter is contiguous by construction, which makes it useless for drop
    detection -- and actively misleading, because it looks like a clean run.
    The tell is md_frame_counter absent while frame_number starts at 1 and has
    no gaps at all.
    """
    if np.isfinite(data["md_frame_counter"]).any():
        return False
    for index in (1, 2):
        sel = data["stream"] == index
        if sel.sum() < 2:
            continue
        numbers = np.sort(data["frame_number"][sel])
        if numbers[0] != 1 or not np.all(np.diff(numbers) == 1):
            return False
    return True


def requested_fps(meta) -> float:
    """Parse the requested rate from run.txt's 'resolution WxH @ F'."""
    try:
        return float(meta["resolution"].split("@")[1])
    except (KeyError, IndexError, ValueError):
        return FPS_NOMINAL


def report_rate(data, meta, synthetic: bool):
    """Achieved vs requested rate. Valid regardless of counter provenance."""
    print("\n== delivered rate " + "=" * 51)
    want = requested_fps(meta)
    try:
        duration = float(meta["duration_s"])
    except (KeyError, ValueError):
        duration = np.nan

    for index in (1, 2):
        sel = data["stream"] == index
        n = int(sel.sum())
        if n == 0:
            print(f"  ir{index}: NO FRAMES")
            continue
        got = n / duration if duration and np.isfinite(duration) else np.nan
        expect = want * duration if np.isfinite(duration) else np.nan
        print(f"  ir{index}: {n} frames in {duration:.1f} s -> {got:.2f} fps "
              f"(requested {want:.0f})")
        if np.isfinite(expect) and expect > 0:
            lost = expect - n
            print(f"        expected {expect:.0f} frames, "
                  f"shortfall {lost:.0f} ({100.0 * lost / expect:.1f}%)")

    if np.isfinite(duration) and duration > 0:
        n1 = int((data["stream"] == 1).sum())
        got = n1 / duration
        if got < 0.9 * want:
            print(f"\n  !! delivering {got:.1f} of {want:.0f} fps. Frames are")
            print("     being lost below librealsense -- in the kernel UVC")
            print("     layer or on the wire. Candidate causes, cheapest first:")
            print("       1. Jetson power mode / clocks not maxed"
                  " (nvpmodel -m 0, jetson_clocks).")
            print("       2. CPU-bound USB handling: retry at a lower")
            print("          resolution or rate to see where it keeps up.")
            print("       3. Per-frame disk writes in the capture callback:")
            print("          retry with --save-every 0.")


def report_drops(data, synthetic: bool):
    print("\n== frame-number continuity " + "=" * 42)
    if synthetic:
        print("  frame_number is a HOST-SIDE counter (md_frame_counter is")
        print("  absent and the sequence is 1,2,3,... with no gaps). It counts")
        print("  frames that were delivered, so it cannot reveal frames that")
        print("  were not. Gap analysis is therefore SKIPPED -- it would report")
        print("  a perfect run no matter how many frames were lost.")
        print("  Use the delivered-rate section above instead.")
        return
    for index in (1, 2):
        sel = data["stream"] == index
        n = int(sel.sum())
        if n == 0:
            print(f"  ir{index}: NO FRAMES")
            continue
        numbers = np.sort(data["frame_number"][sel])
        span = int(numbers[-1] - numbers[0]) + 1
        missing = span - n
        gaps = np.diff(numbers)
        big = gaps[gaps > 1]
        print(f"  ir{index}: {n} frames, number span {span}, "
              f"{missing} missing ({100.0 * missing / span:.3f}%)")
        if big.size:
            burst = Counter(int(g) - 1 for g in big)
            print("        " + f"{big.size} gap events, sizes "
                  + ", ".join(f"{k}x{v}" for k, v in sorted(burst.items())))
        else:
            print("        no gaps -- contiguous frame numbers")


def report_sync(data, hw: bool, synthetic: bool):
    print("\n== left/right pairing " + "=" * 47)
    ts = {}
    for index in (1, 2):
        sel = data["stream"] == index
        ts[index] = dict(zip(data["frame_number"][sel].tolist(),
                             data["ts_ms"][sel].tolist()))
    common = sorted(set(ts[1]) & set(ts[2]))
    print(f"  paired {len(common)}   ir1-only {len(set(ts[1]) - set(ts[2]))}"
          f"   ir2-only {len(set(ts[2]) - set(ts[1]))}")
    if not common:
        print("  !! no matched frame numbers -- hardware sync is not working")
        return None

    d = np.array([ts[1][k] - ts[2][k] for k in common])
    print(f"  ts difference: median {np.median(d):+.6f} ms, "
          f"max |d| {np.abs(d).max():.6f} ms")

    if synthetic:
        print("\n  !! NEITHER number above is evidence of hardware sync.")
        print("     The counter is host-side and restarts at 1 per stream, so")
        print("     ir1 and ir2 pair up by construction. And under a host-time")
        print("     domain both channels are stamped on delivery, so their")
        print("     difference reads zero by construction too.")
        print("     Hardware sync is UNVERIFIED in this configuration. It needs")
        print("     either camera frame counters (patch uvcvideo) or an external")
        print("     check -- e.g. a moving scene where a genuine inter-channel")
        print("     exposure offset would show as horizontal shear.")
    elif not hw:
        print("  note: under a host-time domain the timestamp difference is not")
        print("        evidence about hardware sync -- both channels are stamped")
        print("        on arrival. The matched camera frame numbers are.")
    return d


def report_intervals(data):
    print("\n== frame intervals " + "=" * 50)
    out = {}
    for index in (1, 2):
        sel = data["stream"] == index
        if sel.sum() < 3:
            continue
        order = np.argsort(data["frame_number"][sel])
        series_by_label = {
            "stamp": data["ts_ms"][sel][order],
            "host": data["host_monotonic_s"][sel][order] * 1000.0,
        }
        for label, series in series_by_label.items():
            dt = np.diff(series)
            dt = dt[np.isfinite(dt)]
            if dt.size == 0:
                continue
            out[(index, label)] = dt
            print(f"  ir{index} {label:<6} median {np.median(dt):7.3f} ms  "
                  f"p1 {np.percentile(dt, 1):7.3f}  "
                  f"p99 {np.percentile(dt, 99):7.3f}  "
                  f"max {dt.max():8.3f}  (nominal {NOMINAL_INTERVAL_MS:.3f})")
    return out


def report_metadata(data, meta):
    print("\n== timestamp domain and metadata " + "=" * 36)
    for dom, count in Counter(d for d in data["domain"] if d).most_common():
        print(f"  {count:6d}  domain: {dom}")

    present, absent = [], []
    for name in MD_COLUMNS:
        (present if np.isfinite(data[name]).any() else absent).append(name)
    print("\n  metadata present: " + (", ".join(present) or "none"))
    print("  metadata ABSENT:  " + (", ".join(absent) or "none"))

    if not hardware_clock(data):
        print("\n  !! No camera-clock timestamps. uvcvideo is unpatched, so the")
        print("     UVC metadata node does not exist and librealsense stamps")
        print("     frames on arrival. Consequences beyond timing:")
        if "md_laser_mode" in absent:
            print("       - md_laser_mode absent: with EMITTER_ON_OFF alternation")
            print("         there is no per-frame label for which frames had the")
            print("         projector lit, so the projector on/off A/B split")
            print("         must be inferred from image statistics instead.")
        if "md_exposure_us" in absent:
            print("       - md_exposure_us absent: the manual exposure cannot be")
            print("         confirmed per frame, only trusted from the option")
            print("         readback at configuration time.")

    if "md_exposure_us" in present:
        exp = data["md_exposure_us"]
        exp = exp[np.isfinite(exp)]
        uniq = np.unique(exp)
        print(f"\n  actual exposure: median {np.median(exp):.0f} us, "
              f"{len(uniq)} distinct (requested {meta.get('exposure_us', '?')})")
        if len(uniq) > 1:
            print("  !! exposure varied -- auto-exposure was not fully off")


def px_per_ms(meta) -> float:
    """Pixels of misregistration per ms of timing error, at GYRO_DEG_PER_S."""
    try:
        fx = float(meta["fx"])
    except (KeyError, ValueError):
        fx = 430.55
    rad_per_ms = np.deg2rad(GYRO_DEG_PER_S) * 1e-3
    return fx * rad_per_ms


def report_clock_skew(data, meta):
    """Camera clock vs Jetson clock -- only meaningful on the hardware clock."""
    print("\n== camera clock vs host clock " + "=" * 39)
    if not hardware_clock(data):
        print("  SKIPPED: timestamps are host-side, so this would compare the")
        print("  host clock against itself. See the jitter section below.")
        return None

    sel = data["stream"] == 1
    order = np.argsort(data["frame_number"][sel])
    cam = data["ts_ms"][sel][order]
    host = data["host_monotonic_s"][sel][order] * 1000.0
    good = np.isfinite(cam) & np.isfinite(host)
    cam, host = cam[good], host[good]
    if cam.size < 30:
        print("  too few frames for a drift estimate")
        return None

    cam0, host0 = cam - cam[0], host - host[0]
    slope, intercept = np.polyfit(cam0, host0, 1)
    resid = host0 - (slope * cam0 + intercept)
    ppm = (slope - 1.0) * 1e6

    print(f"  span {cam0[-1] / 1000.0:.1f} s over {cam.size} frames")
    print(f"  host/camera rate ratio {slope:.9f}  ->  drift {ppm:+.1f} ppm")
    print(f"  i.e. {ppm * 3.6:+.2f} ms of accumulated skew per hour")
    print(f"  residual after linear fit: std {resid.std():.3f} ms, "
          f"p99 |r| {np.percentile(np.abs(resid), 99):.3f} ms")
    if abs(ppm) > 100:
        print("  !! >100 ppm drift. A constant offset will not hold; the IMU")
        print("     time alignment needs a running estimate.")
    return cam0, resid, slope


def report_jitter(data, meta, synthetic: bool):
    """Arrival jitter about the nominal frame grid, in ms and in pixels.

    This is the number that actually bounds the plan's gyro rotation
    compensation when no camera clock is available: a frame stamped N ms late
    is warped by N ms too much rotation.
    """
    print("\n== arrival jitter about the nominal grid " + "=" * 28)
    if synthetic:
        print("  SKIPPED: this fits arrival time against frame number, which")
        print("  requires a camera-assigned counter. The counter here is")
        print("  host-side and counts only delivered frames, so every lost")
        print("  frame shifts the fit and the residual measures the shortfall")
        print("  rather than jitter. Fix the delivered-rate problem first, or")
        print("  patch uvcvideo for a real counter.")
        return None
    sel = data["stream"] == 1
    order = np.argsort(data["frame_number"][sel])
    numbers = data["frame_number"][sel][order].astype(float)
    stamp = data["ts_ms"][sel][order]
    good = np.isfinite(stamp)
    numbers, stamp = numbers[good], stamp[good]
    if numbers.size < 30:
        print("  too few frames")
        return None

    # Regress stamp on frame number: the slope recovers the true frame period,
    # so gaps from dropped frames do not distort the residual.
    slope, intercept = np.polyfit(numbers, stamp, 1)
    resid = stamp - (slope * numbers + intercept)
    scale = px_per_ms(meta)

    print(f"  fitted frame period {slope:.4f} ms "
          f"(nominal {NOMINAL_INTERVAL_MS:.4f}, "
          f"{(slope / NOMINAL_INTERVAL_MS - 1) * 1e6:+.0f} ppm)")
    print(f"  residual: std {resid.std():.3f} ms, "
          f"p99 |r| {np.percentile(np.abs(resid), 99):.3f} ms, "
          f"max |r| {np.abs(resid).max():.3f} ms")
    print(f"\n  at {GYRO_DEG_PER_S:.0f} deg/s, 1 ms of timing error is "
          f"{scale:.2f} px of misregistration:")
    print(f"    std  {resid.std():6.3f} ms -> {resid.std() * scale:6.2f} px")
    print(f"    p99  {np.percentile(np.abs(resid), 99):6.3f} ms -> "
          f"{np.percentile(np.abs(resid), 99) * scale:6.2f} px")
    print(f"    max  {np.abs(resid).max():6.3f} ms -> "
          f"{np.abs(resid).max() * scale:6.2f} px")
    print("\n  A constant offset (what Kalibr --time-calibration estimates)")
    print("  removes the mean but not this scatter. If the p99 pixel figure is")
    print("  a meaningful fraction of the coarse-to-fine search radius (the")
    print("  plan uses +-3-4 px at full resolution), patching uvcvideo for real")
    print("  camera timestamps is worth the effort. If it is well under, host")
    print("  arrival time is good enough -- document it and move on.")
    return numbers, resid, scale


def make_plots(data, intervals, skew, jitter, dsync, outdir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed -- skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"capture report -- {outdir.name}")

    ax = axes[0][0]
    for (index, label), dt in sorted(intervals.items()):
        if label != "stamp":
            continue
        ax.hist(dt, bins=np.linspace(0, 4 * NOMINAL_INTERVAL_MS, 121),
                histtype="step", label=f"ir{index}")
    ax.axvline(NOMINAL_INTERVAL_MS, color="k", ls="--", lw=0.8,
               label=f"{NOMINAL_INTERVAL_MS:.1f} ms")
    ax.set_xlabel("frame interval [ms]")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title("interval histogram")
    ax.legend(fontsize=8)

    ax = axes[0][1]
    if jitter is not None:
        numbers, resid, scale = jitter
        ax.plot(numbers, resid, lw=0.5)
        ax.set_xlabel("frame number")
        ax.set_ylabel("residual [ms]")
        sec = ax.secondary_yaxis("right", functions=(lambda v: v * scale,
                                                     lambda v: v / scale))
        sec.set_ylabel(f"px @ {GYRO_DEG_PER_S:.0f} deg/s")
        ax.set_title("arrival jitter about fitted frame grid")
    elif skew is not None:
        cam0, resid, _ = skew
        ax.plot(cam0 / 1000.0, resid, lw=0.5)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("residual [ms]")
        ax.set_title("host vs camera clock, residual after fit")
    else:
        ax.text(0.5, 0.5, "no timing estimate", ha="center")

    ax = axes[1][0]
    if dsync is not None and dsync.size:
        ax.plot(dsync, lw=0.6)
        ax.set_xlabel("paired frame index")
        ax.set_ylabel("ts(ir1) - ts(ir2) [ms]")
        ax.set_title("left/right timestamp difference")
    else:
        ax.text(0.5, 0.5, "no paired frames", ha="center")

    ax = axes[1][1]
    for index in (1, 2):
        sel = data["stream"] == index
        if sel.sum() < 2:
            continue
        numbers = np.sort(data["frame_number"][sel])
        ax.plot(numbers[1:], np.diff(numbers), lw=0.6, label=f"ir{index}")
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.set_xlabel("frame number")
    ax.set_ylabel("frame-number step")
    ax.set_title("drops (step > 1)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = outdir / "capture_report.png"
    fig.savefig(path, dpi=130)
    print(f"\nwrote {path}")


def dump_frames(outdir: Path, meta, limit: int):
    """Convert raw Y8 pairs to PNG, and report exposure via intensity stats."""
    raws = sorted((outdir / "frames").glob("*.raw"))
    if not raws:
        print("\nno raw frames saved -- nothing to preview")
        return

    w, h = 848, 480
    if "resolution" in meta:
        try:
            w, h = (int(v) for v in meta["resolution"].split()[0].split("x"))
        except (ValueError, IndexError):
            pass

    pairs: dict[str, dict[str, Path]] = {}
    for p in raws:
        index, number = p.stem.split("_")
        pairs.setdefault(number, {})[index] = p
    complete = sorted(k for k, v in pairs.items() if len(v) == 2)

    print("\n== image statistics " + "=" * 49)
    print(f"  {len(complete)} complete L/R pairs saved")
    stats = []
    for number in complete:
        row = [number]
        for index in ("ir1", "ir2"):
            buf = np.fromfile(pairs[number][index], dtype=np.uint8)
            if buf.size != w * h:
                row.append(None)
                continue
            row.append((buf.mean(), np.percentile(buf, 99), (buf < 8).mean()))
        stats.append(row)

    means = [s[1][0] for s in stats if s[1]]
    if means:
        print(f"  ir1 mean intensity: min {min(means):.1f}  "
              f"median {np.median(means):.1f}  max {max(means):.1f}")
        dark = [s[1][2] for s in stats if s[1]]
        print(f"  ir1 fraction below 8 DN: median {np.median(dark):.3f}")
        if np.median(means) < 25:
            print("  !! very dark. Raise --gain (range 16-248) or lengthen")
            print("     exposure. Census needs local contrast, not brightness,")
            print("     but not from sensor noise either.")
        elif np.median(means) > 200:
            print("  !! near saturation -- reduce gain.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    outpng = outdir / "preview"
    outpng.mkdir(exist_ok=True)
    for number in complete[:limit]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
        for ax, index in zip(axes, ("ir1", "ir2")):
            buf = np.fromfile(pairs[number][index], dtype=np.uint8)
            if buf.size != w * h:
                ax.text(0.5, 0.5, f"size {buf.size} != {w * h}", ha="center")
                continue
            ax.imshow(buf.reshape(h, w), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{index} frame {number}  mean {buf.mean():.1f}")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(outpng / f"pair_{number}.png", dpi=110)
        plt.close(fig)
    print(f"  wrote {min(len(complete), limit)} preview pair(s) to {outpng}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--previews", type=int, default=3,
                    help="number of L/R preview PNGs to render")
    args = ap.parse_args()

    data, meta = load(args.outdir)
    print("== run " + "=" * 62)
    for k in ("serial", "firmware", "usb", "librealsense", "resolution",
              "exposure_us", "gain", "emitter", "duration_s", "fx",
              "baseline_m"):
        if k in meta:
            print(f"  {k:<14} {meta[k]}")
    if "fx" in meta and "baseline_m" in meta:
        try:
            fb = float(meta["fx"]) * float(meta["baseline_m"])
            print(f"  {'f*B':<14} {fb:.2f} px*m")
        except ValueError:
            pass

    hw = hardware_clock(data)
    synthetic = synthetic_counter(data)
    report_rate(data, meta, synthetic)
    report_drops(data, synthetic)
    dsync = report_sync(data, hw, synthetic)
    intervals = report_intervals(data)
    report_metadata(data, meta)
    skew = report_clock_skew(data, meta)
    jitter = None if hw else report_jitter(data, meta, synthetic)
    make_plots(data, intervals, skew, jitter, dsync, args.outdir)
    dump_frames(args.outdir, meta, args.previews)
    return 0


if __name__ == "__main__":
    sys.exit(main())
