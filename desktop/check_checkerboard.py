#!/usr/bin/env python3
"""Check that the printed checkerboard is actually detectable in the IR streams.

Run this on a short recording of the board BEFORE committing to a full
calibration session. It catches the failure modes that otherwise waste the whole
session, and it costs a 10-second recording.

What it looks for, in the order that matters:

  detectable at all   Many dye-based inkjet blacks are nearly transparent at
                      850 nm, so a board that looks perfect to the eye can be
                      invisible to these sensors. If detection fails everywhere
                      while the frames plainly show a board, suspect the printer
                      before suspecting the code.
  both channels       Corners must be found in ir1 AND ir2 in the same frame, or
                      the pair contributes nothing to stereo extrinsics.
  pose variety        Calibration needs the board at varied distances, angles and
                      image positions. A recording where it barely moves is
                      worth little regardless of how many frames it has.
  image coverage      An A4 board fills only ~13% of the frame width at 1 m. This
                      reports what fraction it actually covered, so the
                      limitation is visible before it becomes a bad calibration.

Desktop-side. numpy + matplotlib + opencv (installed into the venv by pip; the
Jetson deliberately stays OpenCV-free).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import cv2
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


def load_pairs(bag: Path, w: int, h: int):
    raws = sorted((bag / "frames").glob("*.raw"))
    grouped: dict[str, dict[str, Path]] = {}
    for p in raws:
        try:
            idx, num = p.stem.split("_")
        except ValueError:
            continue
        grouped.setdefault(num, {})[idx] = p
    out = []
    for num in sorted(grouped):
        e = grouped[num]
        if "ir1" not in e or "ir2" not in e:
            continue
        imgs = []
        for idx in ("ir1", "ir2"):
            buf = np.fromfile(e[idx], dtype=np.uint8)
            if buf.size != w * h:
                imgs = []
                break
            imgs.append(buf.reshape(h, w))
        if imgs:
            out.append((num, imgs[0], imgs[1]))
    return out


def find(img, cols, rows):
    """Returns refined corners or None. Tries the modern detector first."""
    size = (cols, rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(img, size, flags=flags)
    if not ok and hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(img, size)
    if not ok:
        return None
    corners = cv2.cornerSubPix(
        img, corners.astype(np.float32), (7, 7), (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    return corners.reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("--cols", type=int, default=9, help="interior corners across")
    ap.add_argument("--rows", type=int, default=6, help="interior corners down")
    args = ap.parse_args()

    meta = read_meta(args.bag)
    try:
        w, h = (int(v) for v in meta["resolution"].split()[0].split("x"))
    except (KeyError, ValueError, IndexError):
        w, h = 848, 480

    pairs = load_pairs(args.bag, w, h)
    if not pairs:
        raise SystemExit(f"no complete L/R pairs in {args.bag}/frames — record "
                         f"with --save-every N (N>0)")

    print(f"bag        {args.bag}")
    print(f"target     {args.cols}x{args.rows} interior corners")
    print(f"emitter    {meta.get('emitter', '?')}"
          + ("   <-- should be 'off' for calibration"
             if meta.get("emitter") == "on" else ""))
    print(f"pairs      {len(pairs)}")
    print()

    both, only1, only2, neither = [], 0, 0, 0
    for num, ir1, ir2 in pairs:
        c1, c2 = find(ir1, args.cols, args.rows), find(ir2, args.cols, args.rows)
        if c1 is not None and c2 is not None:
            both.append((num, ir1, ir2, c1, c2))
        elif c1 is not None:
            only1 += 1
        elif c2 is not None:
            only2 += 1
        else:
            neither += 1

    print(f"detected in BOTH channels   {len(both)} / {len(pairs)}")
    print(f"  ir1 only                  {only1}")
    print(f"  ir2 only                  {only2}")
    print(f"  neither                   {neither}")

    if not both:
        print("\n!! The board was not detected in a single stereo pair.")
        print("   Check, in this order:")
        print("   1. Is the board visible in the frames at all? Open")
        print("      bags/<run>/view/stereo_pair.png. If you can see it there")
        print("      but detection fails, the printing is the suspect: many")
        print("      inkjet blacks are near-transparent at 850 nm. Use a laser")
        print("      printer.")
        print("   2. Do --cols/--rows match the board? They are INTERIOR")
        print("      corners, so a 10x7-square board is 9x6.")
        print("   3. Was the emitter off? Projector dots interfere with corner")
        print("      detection.")
        print("   4. Is the whole board inside the frame, unclipped, and in")
        print("      focus?")
        return 1

    # Coverage and pose variety.
    areas, centres, spans = [], [], []
    for _, _, _, c1, _ in both:
        x0, y0 = c1.min(axis=0)
        x1, y1 = c1.max(axis=0)
        areas.append(((x1 - x0) * (y1 - y0)) / float(w * h))
        centres.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
        spans.append(x1 - x0)
    areas = np.array(areas)
    centres = np.array(centres)
    spans = np.array(spans)

    print(f"\nboard image area   median {100 * np.median(areas):.1f}% of frame "
          f"(min {100 * areas.min():.1f}%, max {100 * areas.max():.1f}%)")
    print(f"board width        median {np.median(spans):.0f} px of {w}")
    print(f"centre spread      x {centres[:, 0].std():.0f} px, "
          f"y {centres[:, 1].std():.0f} px")

    if np.median(areas) < 0.04:
        print("\n  !! The board covers very little of the frame. Move closer, or")
        print("     use a larger board. Calibration from a small target gives")
        print("     poorly constrained focal length and distortion.")
    if centres[:, 0].std() < 40 or centres[:, 1].std() < 30:
        print("\n  !! The board barely moved across the image. Calibration needs")
        print("     it at varied positions AND varied angles, not just varied")
        print("     frames. Re-record while walking it around the field of view")
        print("     and tilting it.")
    if len(both) < 15:
        print(f"\n  note: only {len(both)} usable pairs. Aim for 20-40 well-varied"
              " poses.")

    outdir = args.bag / "view"
    outdir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    num, ir1, ir2, c1, c2 = both[len(both) // 2]
    for ax, img, c, label in ((axes[0], ir1, c1, "ir1"), (axes[1], ir2, c2, "ir2")):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.plot(c[:, 0], c[:, 1], "o", ms=3, mfc="none", mec="#ff3b30", mew=0.8)
        ax.set_title(f"{label} frame {int(num)} — {len(c)} corners", fontsize=9)
        ax.set_axis_off()
    fig.tight_layout()
    out = outdir / "checkerboard.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
