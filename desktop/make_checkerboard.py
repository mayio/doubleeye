#!/usr/bin/env python3
"""Generate a printable checkerboard calibration target as a scale-exact PDF.

Defaults to a 10x7-square board (9x6 interior corners) at 25 mm pitch on A4
landscape. The interior-corner count is deliberately odd x even: a board whose
corner counts are both even is 180-degree ambiguous, and detectors can silently
flip its orientation between frames.

PDF, not PNG, because the whole point is physical scale. A raster image gets
resampled by the print pipeline and the pitch you calibrate against stops
matching the pitch you measured.

Desktop-side. matplotlib only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

MM_PER_INCH = 25.4

PAPER_MM = {
    "a4": (297.0, 210.0),   # landscape
    "a4p": (210.0, 297.0),  # portrait
    "a3": (420.0, 297.0),
    "letter": (279.4, 215.9),
}


def build(cols: int, rows: int, pitch: float, paper: str, out: Path,
          margin_min: float = 10.0):
    """cols/rows are counts of SQUARES; interior corners are (cols-1, rows-1)."""
    page_w, page_h = PAPER_MM[paper]
    board_w, board_h = cols * pitch, rows * pitch

    if board_w > page_w - 2 * margin_min or board_h > page_h - 2 * margin_min:
        raise SystemExit(
            f"board {board_w:.1f}x{board_h:.1f} mm does not fit {paper} "
            f"({page_w:.0f}x{page_h:.0f} mm) with {margin_min:.0f} mm margins.\n"
            f"Reduce --pitch, reduce --cols/--rows, or use a larger --paper.")

    x0 = (page_w - board_w) / 2.0
    y0 = (page_h - board_h) / 2.0

    fig = plt.figure(figsize=(page_w / MM_PER_INCH, page_h / MM_PER_INCH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w)
    ax.set_ylim(0, page_h)
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, 0), page_w, page_h, facecolor="white",
                           edgecolor="none", zorder=0))

    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                continue
            ax.add_patch(Rectangle(
                (x0 + c * pitch, y0 + r * pitch), pitch, pitch,
                facecolor="black", edgecolor="none", linewidth=0, zorder=1))

    # A thin outline of the board plus a caption, both outside the squares so
    # they cannot be mistaken for board features. The caption records what this
    # sheet actually is, which matters once several are lying on a desk.
    ax.add_patch(Rectangle((x0, y0), board_w, board_h, fill=False,
                           edgecolor="black", linewidth=0.3, zorder=2))
    caption = (f"{cols}x{rows} squares  |  {cols - 1}x{rows - 1} interior "
               f"corners  |  {pitch:g} mm pitch  |  {paper.upper()}  |  "
               f"PRINT AT 100% - DO NOT SCALE")
    ax.text(page_w / 2.0, max(3.0, y0 / 2.0), caption, ha="center",
            va="center", fontsize=6, color="black", zorder=2)

    fig.savefig(out, format="pdf")  # no bbox_inches: it would change the size
    plt.close(fig)

    png = out.with_suffix(".png")
    fig2 = plt.figure(figsize=(page_w / MM_PER_INCH, page_h / MM_PER_INCH))
    ax2 = fig2.add_axes([0, 0, 1, 1])
    ax2.set_xlim(0, page_w)
    ax2.set_ylim(0, page_h)
    ax2.set_axis_off()
    ax2.add_patch(Rectangle((0, 0), page_w, page_h, facecolor="white",
                            edgecolor="none"))
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                continue
            ax2.add_patch(Rectangle((x0 + c * pitch, y0 + r * pitch), pitch,
                                    pitch, facecolor="black", edgecolor="none"))
    fig2.savefig(png, format="png", dpi=150)
    plt.close(fig2)

    return board_w, board_h, x0, y0, png


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=10, help="squares across")
    ap.add_argument("--rows", type=int, default=7, help="squares down")
    ap.add_argument("--pitch", type=float, default=25.0, help="square size, mm")
    ap.add_argument("--paper", choices=sorted(PAPER_MM), default="a4")
    ap.add_argument("-o", "--out", type=Path,
                    default=Path("checkerboard.pdf"))
    args = ap.parse_args()

    bw, bh, x0, y0, png = build(args.cols, args.rows, args.pitch, args.paper,
                                args.out)
    ic, ir = args.cols - 1, args.rows - 1
    page_w, page_h = PAPER_MM[args.paper]

    print(f"wrote {args.out}  (print this one)")
    print(f"wrote {png}  (preview only - do not print, it will be rescaled)")
    print()
    print(f"  paper            {args.paper.upper()} {page_w:.0f} x {page_h:.0f} mm")
    print(f"  board            {bw:.1f} x {bh:.1f} mm")
    print(f"  margins          {x0:.1f} mm sides, {y0:.1f} mm top/bottom")
    print(f"  squares          {args.cols} x {args.rows} @ {args.pitch:g} mm")
    print(f"  interior corners {ic} x {ir}"
          + ("  (odd x even - orientation unambiguous)"
             if (ic % 2) != (ir % 2) else
             "  !! both same parity - 180 deg ambiguous, consider changing"))
    print()
    print("MEASURE THE PRINT -- this is not advisory. The first A4 printed for")
    print(f"this project came out at 96% scale ({args.pitch:g} mm became 24.0 mm),")
    print("which propagated straight into a 4% stereo-baseline error.")
    print(f"Span all {args.cols} columns: they should read {bw:.1f} mm. Divide by")
    print(f"{args.cols} and use THAT in the configs below, not the nominal value.")
    print()
    print("Kalibr target yaml (checkerboard.yaml):")
    print("  target_type: 'checkerboard'")
    print(f"  targetCols: {ic}            # interior corners")
    print(f"  targetRows: {ir}")
    print(f"  rowSpacingMeters: {args.pitch / 1000.0:.4f}")
    print(f"  colSpacingMeters: {args.pitch / 1000.0:.4f}")
    print()
    print(f"OpenCV: cv::findChessboardCorners(img, cv::Size({ic}, {ir}), ...)")


if __name__ == "__main__":
    main()
