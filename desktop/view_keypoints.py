#!/usr/bin/env python3
"""Visualise what the preprocessing stage produced.

Reads keypoints.csv written by core/build/de_preprocess and answers the question
the grid-bucketing exists to answer: is the coverage actually spread out, or did
everything pile onto the highest-texture region?

Four panels:
  overlay      keypoints on the frame, with the bucketing grid drawn
  occupancy    keypoints per grid cell as a heatmap
  response     Shi-Tomasi response distribution
  texture      local-contrast distribution, against the rejection threshold

Desktop-side. numpy + matplotlib.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_meta(bag: Path) -> dict:
    meta = {}
    run = bag / "run.txt"
    if run.exists():
        for line in run.read_text().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                meta[parts[0]] = parts[1]
    return meta


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("--stream", type=int, default=1, choices=(1, 2))
    ap.add_argument("--cell", type=int, default=32,
                    help="must match the --cell used for de_preprocess")
    args = ap.parse_args()

    csv_path = args.bag / "keypoints.csv"
    if not csv_path.exists():
        raise SystemExit(f"no keypoints.csv in {args.bag} — run "
                         f"core/build/de_preprocess {args.bag} first")

    meta = read_meta(args.bag)
    try:
        w, h = (int(v) for v in meta["resolution"].split()[0].split("x"))
    except (KeyError, ValueError, IndexError):
        w, h = 848, 480

    rows = []
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            if int(r["stream"]) == args.stream:
                rows.append(r)
    if not rows:
        raise SystemExit(f"no keypoints for stream {args.stream}")

    # One representative frame for the overlay, all frames for the statistics.
    frames = sorted({r["frame"] for r in rows})
    first = frames[0]
    sel = [r for r in rows if r["frame"] == first]

    x = np.array([float(r["x"]) for r in sel])
    y = np.array([float(r["y"]) for r in sel])
    resp_all = np.array([float(r["response"]) for r in rows])
    lstd_all = np.array([float(r["local_std"]) for r in rows])
    desc = [int(r["census_hex"], 16) for r in sel]

    raw = args.bag / "frames" / f"ir{args.stream}_{first}.raw"
    img = None
    if raw.exists():
        buf = np.fromfile(raw, dtype=np.uint8)
        if buf.size == w * h:
            img = buf.reshape(h, w)

    cols, crows = (w + args.cell - 1) // args.cell, (h + args.cell - 1) // args.cell
    grid = np.zeros((crows, cols), dtype=int)
    for xi, yi in zip(x, y):
        cx, cy = int(xi) // args.cell, int(yi) // args.cell
        if 0 <= cx < cols and 0 <= cy < crows:
            grid[cy, cx] += 1
    occupancy = float((grid > 0).mean())

    print(f"bag            {args.bag}")
    print(f"stream         ir{args.stream}")
    print(f"frames         {len(frames)}")
    print(f"keypoints      {len(rows)} total, {len(sel)} in frame {first}")
    print(f"cell occupancy {occupancy:.3f} "
          f"({int((grid > 0).sum())} of {grid.size} cells)")
    print(f"per cell       max {grid.max()}, mean over occupied "
          f"{grid[grid > 0].mean():.2f}")
    print(f"response       median {np.median(resp_all):.2f}, "
          f"p99 {np.percentile(resp_all, 99):.2f}")
    print(f"local std      median {np.median(lstd_all):.2f} DN, "
          f"min {lstd_all.min():.2f}")

    # Descriptor diversity: how many of the 48 Census bits actually vary across
    # keypoints. If most bits are constant the descriptor is not discriminating,
    # which is the failure mode to watch for on projected-dot texture.
    if desc:
        arr = np.array(desc, dtype=np.uint64)
        varying = sum(1 for b in range(48)
                      if 0 < int(((arr >> np.uint64(b)) & np.uint64(1)).sum()) < len(arr))
        uniq = len(set(desc))
        print(f"census         {uniq} distinct of {len(desc)} keypoints, "
              f"{varying}/48 bits varying")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    fig.suptitle(f"{args.bag.name} — ir{args.stream}, preprocessing output")

    ax = axes[0][0]
    if img is not None:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.scatter(x, y, s=6, facecolors="none", edgecolors="#ff3b30", linewidths=0.6)
    for c in range(0, w, args.cell):
        ax.axvline(c, color="#00b7ff", lw=0.2, alpha=0.5)
    for r in range(0, h, args.cell):
        ax.axhline(r, color="#00b7ff", lw=0.2, alpha=0.5)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title(f"{len(sel)} keypoints, frame {first}, {args.cell} px grid",
                 fontsize=9)
    ax.set_axis_off()

    ax = axes[0][1]
    im = ax.imshow(grid, cmap="viridis", interpolation="nearest")
    ax.set_title(f"keypoints per cell — occupancy {occupancy:.1%}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.035)
    ax.set_axis_off()

    ax = axes[1][0]
    ax.hist(resp_all, bins=60, color="#0a84ff")
    ax.set_yscale("log")
    ax.set_xlabel("Shi-Tomasi response")
    ax.set_ylabel("count")
    ax.set_title("response distribution (all frames)", fontsize=9)

    ax = axes[1][1]
    ax.hist(lstd_all, bins=60, color="#30d158")
    ax.axvline(2.0, color="k", ls="--", lw=0.9, label="texture floor 2 DN")
    ax.set_yscale("log")
    ax.set_xlabel("7x7 local std [DN]")
    ax.set_ylabel("count")
    ax.set_title("texture at keypoints (all frames)", fontsize=9)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = args.bag / "keypoints.png"
    fig.savefig(out, dpi=125)
    plt.close(fig)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
