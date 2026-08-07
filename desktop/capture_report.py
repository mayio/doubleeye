#!/usr/bin/env python3
"""Offline analysis of a bring-up step 1 recording.

Consumes the directory written by jetson/rs_ir_capture.py and answers:

  1. Were frames dropped, and on which channel?
  2. Are the two IR channels genuinely hardware-synced (matched frame numbers,
     sub-millisecond timestamp agreement)?
  3. What is the frame-interval distribution -- clean 33.3 ms, or bimodal?
  4. Which timestamp domain is `get_timestamp()` reporting?
  5. How fast does the camera clock drift against the Jetson clock?

(5) is the one that matters for IMU fusion. The plan's stated risk is an
unknown, possibly drifting offset between the camera clock and the Jetson
clock; at 100 deg/s, 10 ms of skew is ~1 deg, i.e. double-digit pixel error at
the image edge. This fits a line to camera-vs-host timestamps and reports both
the drift in ppm and the residual scatter, which bounds how well any constant
offset can ever do.

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
            raw = r.get(name, "")
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
        "md_frame_ts_us": col("md_frame_ts_us"),
        "md_backend_ts_ms": col("md_backend_ts_ms"),
        "md_arrival_ts_ms": col("md_arrival_ts_ms"),
        "md_exposure_us": col("md_exposure_us"),
        "domain": [r.get("ts_domain", "") for r in rows],
    }
    return data, meta


def report_drops(data):
    print("\n== frames and drops " + "=" * 49)
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
            print(f"        {big.size} gap events, sizes "
                  + ", ".join(f"{k}x{v}" for k, v in sorted(burst.items())))
        else:
            print("        no gaps -- contiguous frame numbers")


def report_sync(data):
    print("\n== left/right sync " + "=" * 50)
    ts = {}
    for index in (1, 2):
        sel = data["stream"] == index
        ts[index] = dict(zip(data["frame_number"][sel].tolist(),
                             data["ts_ms"][sel].tolist()))
    common = sorted(set(ts[1]) & set(ts[2]))
    only1 = len(set(ts[1]) - set(ts[2]))
    only2 = len(set(ts[2]) - set(ts[1]))
    print(f"  paired {len(common)}   ir1-only {only1}   ir2-only {only2}")
    if not common:
        print("  !! no matched frame numbers -- hardware sync is not working")
        return None
    d = np.array([ts[1][k] - ts[2][k] for k in common])
    print(f"  ts difference: median {np.median(d):+.6f} ms, "
          f"p99 |d| {np.percentile(np.abs(d), 99):.6f} ms, "
          f"max |d| {np.abs(d).max():.6f} ms")
    if np.abs(d).max() > 1.0:
        print("  !! >1 ms disagreement on a hardware-synced pair. Suspect the")
        print("     timestamp domain rather than the sync itself.")
    return d


def report_intervals(data):
    print("\n== frame intervals " + "=" * 50)
    out = {}
    for index in (1, 2):
        sel = data["stream"] == index
        if sel.sum() < 3:
            continue
        order = np.argsort(data["frame_number"][sel])
        cam = data["ts_ms"][sel][order]
        host = data["host_monotonic_s"][sel][order] * 1000.0
        for label, series in (("camera", cam), ("host", host)):
            dt = np.diff(series)
            dt = dt[np.isfinite(dt)]
            if dt.size == 0:
                continue
            out[(index, label)] = dt
            print(f"  ir{index} {label:<6} median {np.median(dt):7.3f} ms  "
                  f"p1 {np.percentile(dt, 1):7.3f}  "
                  f"p99 {np.percentile(dt, 99):7.3f}  "
                  f"max {dt.max():8.3f}  "
                  f"(nominal {NOMINAL_INTERVAL_MS:.3f})")
    return out


def report_domain(data, meta):
    print("\n== timestamp domain " + "=" * 49)
    for dom, count in Counter(data["domain"]).most_common():
        print(f"  {count:6d}  {dom}")
    if any("system" in d for d in data["domain"]):
        print("  note: system_time domain means librealsense is NOT giving you")
        print("        the camera clock. Global time sync may be enabled; for")
        print("        clock-offset work you want the hardware clock.")
    exp = data["md_exposure_us"]
    exp = exp[np.isfinite(exp)]
    if exp.size:
        print(f"  actual exposure: median {np.median(exp):.0f} us, "
              f"unique values {len(np.unique(exp))} "
              f"(requested {meta.get('exposure_us', '?')})")
        if len(np.unique(exp)) > 1:
            print("  !! exposure varied -- auto-exposure was not fully off")


def report_clock_skew(data):
    """Camera clock vs Jetson clock: drift rate and residual scatter."""
    print("\n== camera clock vs host clock " + "=" * 39)
    sel = (data["stream"] == 1)
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
    span_s = cam0[-1] / 1000.0
    ppm = (slope - 1.0) * 1e6

    print(f"  span {span_s:.1f} s over {cam.size} frames")
    print(f"  host/camera rate ratio {slope:.9f}  ->  drift {ppm:+.1f} ppm")
    print(f"  i.e. {ppm * 3.6:+.2f} ms of accumulated skew per hour")
    print(f"  residual after linear fit: std {resid.std():.3f} ms, "
          f"p99 |r| {np.percentile(np.abs(resid), 99):.3f} ms, "
          f"max |r| {np.abs(resid).max():.3f} ms")
    print("  (residual is arrival jitter + USB scheduling; it is the floor on")
    print("   how well a constant offset can align the two clocks)")
    if abs(ppm) > 100:
        print("  !! >100 ppm drift. A constant offset will not hold; the IMU")
        print("     time alignment needs a running estimate.")
    return cam0, resid, slope


def make_plots(data, intervals, skew, dsync, outdir: Path):
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
        if label != "camera":
            continue
        ax.hist(dt, bins=np.linspace(0, 4 * NOMINAL_INTERVAL_MS, 121),
                histtype="step", label=f"ir{index}")
    ax.axvline(NOMINAL_INTERVAL_MS, color="k", ls="--", lw=0.8,
               label=f"{NOMINAL_INTERVAL_MS:.1f} ms")
    ax.set_xlabel("camera-clock frame interval [ms]")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title("interval histogram")
    ax.legend(fontsize=8)

    ax = axes[0][1]
    if dsync is not None and dsync.size:
        ax.plot(dsync, lw=0.6)
        ax.set_xlabel("paired frame index")
        ax.set_ylabel("ts(ir1) - ts(ir2) [ms]")
        ax.set_title("left/right timestamp agreement")
    else:
        ax.text(0.5, 0.5, "no paired frames", ha="center")

    ax = axes[1][0]
    if skew is not None:
        cam0, resid, _ = skew
        ax.plot(cam0 / 1000.0, resid, lw=0.5)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("residual [ms]")
        ax.set_title("host vs camera clock, residual after linear fit")
    else:
        ax.text(0.5, 0.5, "no skew estimate", ha="center")

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
    """Convert a few raw Y8 frames to PNG for the visual check."""
    raws = sorted((outdir / "frames").glob("*.raw"))
    if not raws:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    w, h = 848, 480
    if "resolution" in meta:
        try:
            dims = meta["resolution"].split()[0]
            w, h = (int(v) for v in dims.split("x"))
        except (ValueError, IndexError):
            pass

    pairs = {}
    for p in raws:
        index, number = p.stem.split("_")
        pairs.setdefault(number, {})[index] = p
    keys = sorted(k for k, v in pairs.items() if len(v) == 2)[:limit]

    outpng = outdir / "preview"
    outpng.mkdir(exist_ok=True)
    for number in keys:
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
    print(f"wrote {len(keys)} preview pair(s) to {outpng}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--previews", type=int, default=3,
                    help="number of L/R preview PNGs to render")
    args = ap.parse_args()

    data, meta = load(args.outdir)
    print("== run " + "=" * 62)
    for k in ("serial", "firmware", "usb", "resolution", "exposure_us", "gain",
              "emitter", "duration_s", "fx", "baseline_m"):
        if k in meta:
            print(f"  {k:<14} {meta[k]}")

    report_drops(data)
    dsync = report_sync(data)
    intervals = report_intervals(data)
    report_domain(data, meta)
    skew = report_clock_skew(data)
    make_plots(data, intervals, skew, dsync, args.outdir)
    dump_frames(args.outdir, meta, args.previews)
    return 0


if __name__ == "__main__":
    sys.exit(main())
