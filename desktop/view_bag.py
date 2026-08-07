#!/usr/bin/env python3
"""Look at what was actually recorded.

Renders the raw Y8 frames in a bag four ways, because each answers a different
question:

  contact sheet   did the whole recording look sane, or did something change
                  partway through?
  stereo pairs    left and right side by side, for judging exposure, focus and
                  whether the projector is reaching the surfaces that matter
  anaglyph        left and right overlaid as red/cyan. Disparity becomes
                  visible as colour fringing, and vertical misalignment shows
                  up immediately -- a rectified pair should fringe purely
                  horizontally.
  animation       an animated GIF, for motion and for spotting dropped or
                  duplicated frames

No OpenCV and no ffmpeg: GIFs are written through matplotlib's Pillow writer,
which is available in the desktop venv. Y8 is a raw byte-per-pixel dump, so
numpy reads it directly.

Desktop-side. numpy + matplotlib (+ Pillow, which matplotlib already pulls in).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation


def read_meta(bag: Path) -> dict:
    meta = {}
    run = bag / "run.txt"
    if run.exists():
        for line in run.read_text().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                meta[parts[0]] = parts[1]
    return meta


def frame_size(meta: dict) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in meta["resolution"].split()[0].split("x"))
        return w, h
    except (KeyError, ValueError, IndexError):
        return 848, 480


def load_pairs(bag: Path, w: int, h: int):
    """Return [(number, ir1, ir2)] for every frame number present on both."""
    raws = sorted((bag / "frames").glob("*.raw"))
    if not raws:
        return []
    grouped: dict[str, dict[str, Path]] = {}
    for p in raws:
        try:
            index, number = p.stem.split("_")
        except ValueError:
            continue
        grouped.setdefault(number, {})[index] = p

    out = []
    for number in sorted(grouped):
        entry = grouped[number]
        if "ir1" not in entry or "ir2" not in entry:
            continue
        imgs = []
        for index in ("ir1", "ir2"):
            buf = np.fromfile(entry[index], dtype=np.uint8)
            if buf.size != w * h:
                imgs = []
                break
            imgs.append(buf.reshape(h, w))
        if imgs:
            out.append((number, imgs[0], imgs[1]))
    return out


def contact_sheet(pairs, out: Path, cols: int = 6, limit: int = 36):
    sel = pairs[:limit]
    if not sel:
        return None
    rows = (len(sel) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 1.7 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_axis_off()
    for ax, (number, ir1, _) in zip(axes, sel):
        ax.imshow(ir1, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{int(number)}  µ{ir1.mean():.0f}", fontsize=6)
    fig.suptitle(f"{out.parent.name} — ir1 contact sheet "
                 f"({len(sel)} of {len(pairs)} pairs)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def stereo_pair(pairs, out: Path, which: int = 0):
    if not pairs:
        return None
    number, ir1, ir2 = pairs[min(which, len(pairs) - 1)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.3))
    for ax, img, label in zip(axes, (ir1, ir2), ("ir1 (left)", "ir2 (right)")):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{label}  frame {int(number)}  mean {img.mean():.1f}  "
                     f"sat {(img >= 250).mean() * 100:.1f}%", fontsize=9)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def anaglyph(pairs, out: Path, which: int = 0):
    """Left in red, right in cyan. Horizontal fringing = disparity (expected).
    Vertical fringing = rectification or sync trouble (not expected)."""
    if not pairs:
        return None
    number, ir1, ir2 = pairs[min(which, len(pairs) - 1)]
    rgb = np.zeros((ir1.shape[0], ir1.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = ir1
    rgb[..., 1] = ir2
    rgb[..., 2] = ir2
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.imshow(rgb)
    ax.set_title(f"anaglyph, frame {int(number)} — ir1=red, ir2=cyan.\n"
                 "Fringing should be HORIZONTAL only; vertical fringing means "
                 "rectification or sync trouble.", fontsize=9)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def gif(pairs, out: Path, fps: int = 6, limit: int = 60):
    sel = pairs[:limit]
    if len(sel) < 2:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    ax.set_axis_off()
    im = ax.imshow(sel[0][1], cmap="gray", vmin=0, vmax=255)
    title = ax.set_title("", fontsize=9)

    def update(i):
        number, ir1, _ = sel[i]
        im.set_data(ir1)
        title.set_text(f"ir1  frame {int(number)}  ({i + 1}/{len(sel)})")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=len(sel), blit=False)
    try:
        anim.save(out, writer=animation.PillowWriter(fps=fps))
    except Exception as exc:
        plt.close(fig)
        print(f"  GIF failed: {exc}")
        return None
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("--frame", type=int, default=0,
                    help="which pair to use for the stereo/anaglyph views")
    ap.add_argument("--fps", type=int, default=6, help="GIF playback rate")
    args = ap.parse_args()

    if not args.bag.is_dir():
        raise SystemExit(f"{args.bag} is not a directory")

    meta = read_meta(args.bag)
    w, h = frame_size(meta)
    pairs = load_pairs(args.bag, w, h)

    print(f"bag       {args.bag}")
    print(f"geometry  {w}x{h}")
    for k in ("emitter", "exposure_us", "gain", "duration_s", "clocks_locked"):
        if k in meta:
            print(f"{k:<10}{meta[k]}")
    print(f"pairs     {len(pairs)} complete L/R")

    if not pairs:
        print("\nNo complete L/R frame pairs found.")
        print("Was the bag recorded with --save-every 0? Re-record with e.g.")
        print("  ./tools/deploy.sh --capture ~/bags/look --seconds 10 "
              "--save-every 2")
        return 1

    outdir = args.bag / "view"
    outdir.mkdir(exist_ok=True)
    print()
    for label, path in [
        ("contact sheet", contact_sheet(pairs, outdir / "contact_sheet.png")),
        ("stereo pair", stereo_pair(pairs, outdir / "stereo_pair.png",
                                    args.frame)),
        ("anaglyph", anaglyph(pairs, outdir / "anaglyph.png", args.frame)),
        ("animation", gif(pairs, outdir / "animation.gif", args.fps)),
    ]:
        print(f"  {label:<14} {path if path else '(skipped)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
