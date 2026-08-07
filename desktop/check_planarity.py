#!/usr/bin/env python3
"""Detect a non-flat calibration board — the error that hides inside a good result.

A checkerboard is only useful if it is *planar*. A sheet that curls or flexes even
a millimetre or two injects a systematic error into intrinsics and extrinsics, and
nothing in a normal calibration report reveals it: detection still succeeds, the
corner count is right, and the reprojection error stays plausible because the
distortion model happily absorbs some of the bend.

This test is available because of something already measured: the D435 IR pair has
**exactly zero distortion coefficients**. So a flat board viewed by this camera
must project to an exact projective image of a regular grid — a homography, no
more. Fit that homography and the residual is dominated by whatever is *not*
planar.

No single statistic separates a bend from noisy corner detection, and an earlier
version of this tool claimed otherwise and was wrong. Residual smoothness rises
with residual magnitude even for pure noise, so it cannot carry the verdict alone.
Three independent signals are reported instead:

  scale correlation    Detection noise gets BETTER as the board fills more of the
                       frame. So residual RISING with apparent size argues against
                       noise: a fixed physical bend subtends more pixels up close.
  temporal clustering  Random noise spreads evenly through a session. Failures
                       arriving in episodes suggest the board was physically
                       disturbed for stretches of it.
  systematic fraction  How much residual survives smoothing. Corroboration only.

The verdict is a 2-of-3 vote, and says so when the signals disagree rather than
picking a side. The remedy does not depend on resolving it: cull the
high-residual poses and calibrate on the rest.

Desktop-side. numpy + opencv + matplotlib.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
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


def load_pairs(bag: Path, w: int, h: int):
    grouped: dict[str, dict[str, Path]] = {}
    for p in sorted((bag / "frames").glob("*.raw")):
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


def find_corners(img, cols, rows):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, c = cv2.findChessboardCorners(img, (cols, rows), flags=flags)
    if not ok:
        return None
    c = cv2.cornerSubPix(
        img, c.astype(np.float32), (7, 7), (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.005))
    return c.reshape(-1, 2)


def planarity(corners, cols, rows):
    """Homography residual of the detected grid against an ideal planar grid.

    Returns (rms, max, systematic_fraction, residual_vectors).
    """
    ideal = np.array([[float(i), float(j)] for j in range(rows)
                      for i in range(cols)], dtype=np.float32)
    H, _ = cv2.findHomography(ideal, corners.astype(np.float32), 0)
    if H is None:
        return None
    proj = cv2.perspectiveTransform(ideal.reshape(-1, 1, 2), H).reshape(-1, 2)
    res = corners - proj
    mag = np.linalg.norm(res, axis=1)
    rms = float(np.sqrt((mag ** 2).mean()))

    # Smooth the residual field over the grid. Independent corner noise largely
    # cancels; a physical bend is smooth and survives. The ratio of surviving
    # energy is a usable "is this structured?" measure.
    grid = res.reshape(rows, cols, 2)
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
    k /= k.sum()
    smoothed = np.stack(
        [cv2.filter2D(grid[..., d], -1, k, borderType=cv2.BORDER_REFLECT)
         for d in range(2)], axis=-1)
    energy = float((res ** 2).sum())
    syst = float((smoothed.reshape(-1, 2) ** 2).sum() / energy) if energy > 0 else 0.0
    return rms, float(mag.max()), syst, res


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--max-rms", type=float, default=0.40,
                    help="reject poses whose homography RMS exceeds this (px)")
    ap.add_argument("--pitch-mm", type=float, default=25.0,
                    help="square size, for reporting deviation in mm")
    ap.add_argument("--cull", metavar="NAME", default=None,
                    help="write the passing poses to bags/NAME")
    args = ap.parse_args()

    meta = read_meta(args.bag)
    try:
        w, h = (int(v) for v in meta["resolution"].split()[0].split("x"))
    except (KeyError, ValueError, IndexError):
        w, h = 848, 480

    pairs = load_pairs(args.bag, w, h)
    if not pairs:
        raise SystemExit(f"no complete L/R pairs in {args.bag}/frames")

    print(f"bag     {args.bag}")
    print(f"target  {args.cols}x{args.rows} interior corners")
    print(f"pairs   {len(pairs)}")
    print("\nfitting a homography per detected board (zero distortion means a")
    print("flat board must fit one exactly)...\n")

    rows_out = []
    for num, ir1, ir2 in pairs:
        entry = {"num": num}
        for name, img in (("ir1", ir1), ("ir2", ir2)):
            c = find_corners(img, args.cols, args.rows)
            if c is None:
                entry[name] = None
                continue
            entry[name] = planarity(c, args.cols, args.rows)
            entry[name + "_corners"] = c
            if name == "ir1":
                # Pixels per square: the resolution at which corners are found,
                # and the scale factor a physical bend is magnified by.
                entry["px_sq"] = float(
                    (c[:, 0].max() - c[:, 0].min()) / max(1, args.cols - 1))
        rows_out.append(entry)

    usable = [r for r in rows_out if r.get("ir1") and r.get("ir2")]
    if not usable:
        raise SystemExit("no pose had the board detected in both channels")

    rms1 = np.array([r["ir1"][0] for r in usable])
    rms2 = np.array([r["ir2"][0] for r in usable])
    worst = np.maximum(rms1, rms2)
    syst = np.array([max(r["ir1"][2], r["ir2"][2]) for r in usable])
    px_sq = [r.get("px_sq", np.nan) for r in usable]

    print(f"usable poses  {len(usable)}")
    print("\nhomography RMS residual, worse of the two channels")
    for q in (50, 75, 90, 100):
        print(f"  p{q:<3d} {np.percentile(worst, q):6.3f} px")
    print(f"  mean  {worst.mean():6.3f} px")

    good = worst <= args.max_rms
    print(f"\nthreshold {args.max_rms:.2f} px -> "
          f"{int(good.sum())} pass, {int((~good).sum())} fail")

    # Interpretation.
    #
    # A single statistic is not enough here, and an earlier version of this tool
    # was wrong because it tried. The "systematic fraction" rises with residual
    # magnitude even for pure noise, so on its own it cannot separate a bend from
    # noisy corners. Three independent signals are reported instead, because they
    # disagree in informative ways:
    #
    #   scale correlation   detection noise gets BETTER as the board fills more
    #                       of the frame (more pixels per square). So a positive
    #                       correlation between residual and apparent size argues
    #                       against noise and for something physical.
    #   temporal clustering random noise is spread evenly through a session.
    #                       Failures arriving in episodes suggest the board was
    #                       physically disturbed for stretches of it.
    #   systematic fraction smoothness of the residual field. Weak on its own,
    #                       useful as corroboration.
    bad = worst > args.max_rms
    px_per_square = np.array([px_sq[i] for i in range(len(usable))])
    mm = worst / np.maximum(px_per_square, 1e-6) * args.pitch_mm

    scale_corr = (float(np.corrcoef(worst, px_per_square)[0, 1])
                  if len(worst) > 2 else float("nan"))
    runs = 1 + int((np.diff(bad.astype(int)) != 0).sum())
    frac = bad.mean()
    exp_runs = 1 + 2 * len(bad) * frac * (1 - frac)

    print("\nevidence, three independent signals")
    print(f"  residual vs apparent size   corr {scale_corr:+.3f}")
    print("      negative => detection noise (bigger board detects better)")
    print("      positive => physical (a fixed bend subtends more pixels up close)")
    print(f"  temporal clustering         {runs} pass/fail runs, "
          f"{exp_runs:.1f} expected if random")
    print(f"  systematic fraction         median {np.median(syst):.2f}"
          + (f", failing poses {syst[bad].mean():.2f}" if bad.any() else ""))
    print(f"\n  implied physical non-flatness at {args.pitch_mm:.0f} mm pitch:")
    print(f"      median {np.median(mm):.2f} mm, p90 {np.percentile(mm, 90):.2f} mm,"
          f" max {mm.max():.2f} mm")

    votes_physical = 0
    if scale_corr > 0.15:
        votes_physical += 1
    if runs < exp_runs * 0.75:
        votes_physical += 1
    if bad.any() and syst[bad].mean() > 0.55:
        votes_physical += 1

    print("\nverdict")
    if not bad.any():
        print("  Every pose fits a plane within the threshold. No evidence of a")
        print("  loose or bent board in this set.")
    elif votes_physical >= 2:
        print(f"  {votes_physical} of 3 signals point at a PHYSICAL cause, so the")
        print("  board was probably not perfectly flat for part of the session —")
        print("  a sheet lifting off its backing, or pressure from a hand.")
        print("  The magnitudes above are small, so the typical pose is usable;")
        print("  it is the tail that is contaminated.")
    elif votes_physical == 1:
        print("  Signals disagree: one of three suggests a physical cause. Treat")
        print("  this as unresolved rather than concluding either way.")
    else:
        print("  Signals are consistent with corner-detection noise (motion,")
        print("  low contrast, extreme angle) rather than a bent board.")
    print("\n  Either way the action is the same and costs nothing: cull the")
    print(f"  high-residual poses with --cull, and calibrate on the {int(good.sum())}")
    print("  that pass. Calibration wants 20-40 good poses, not all of them.")

    order = np.argsort(worst)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    fig.suptitle(f"{args.bag.name} — board planarity: residual vs an ideal plane")
    for ax, k, title in ((axes[0][0], order[0], "best"),
                         (axes[0][1], order[len(order) // 2], "median"),
                         (axes[0][2], order[-1], "worst")):
        r = usable[k]
        res = r["ir1"][3]
        c = r["ir1_corners"]
        ax.quiver(c[:, 0], c[:, 1], res[:, 0], -res[:, 1],
                  angles="xy", scale_units="xy", scale=0.02, width=0.006,
                  color="#ff3b30")
        ax.plot(c[:, 0], c[:, 1], ".", ms=2, color="#0a84ff")
        ax.set_title(f"{title}: frame {int(r['num'])}, RMS {r['ir1'][0]:.3f} px\n"
                     f"systematic {r['ir1'][2]:.2f}  (arrows x50)", fontsize=8)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1][0]
    ax.hist(worst, bins=30, color="#0a84ff")
    ax.axvline(args.max_rms, color="k", ls="--", lw=1,
               label=f"threshold {args.max_rms}")
    ax.set_xlabel("homography RMS [px]"); ax.set_ylabel("poses")
    ax.set_title("planarity residual distribution", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1][1]
    ax.scatter(worst, syst, s=14, c=np.where(good, "#30d158", "#ff3b30"))
    ax.axvline(args.max_rms, color="k", ls="--", lw=1)
    ax.axhline(0.55, color="#888", ls=":", lw=1)
    ax.set_xlabel("RMS residual [px]"); ax.set_ylabel("systematic fraction")
    ax.set_title("large AND smooth (upper right) = bent board", fontsize=9)

    ax = axes[1][2]
    ax.plot(np.arange(len(usable)), worst, ".-", lw=0.6, ms=4)
    ax.axhline(args.max_rms, color="k", ls="--", lw=1)
    ax.set_xlabel("pose, in collection order"); ax.set_ylabel("RMS [px]")
    ax.set_title("did quality drift during the session?", fontsize=9)

    fig.tight_layout()
    outdir = args.bag / "view"
    outdir.mkdir(exist_ok=True)
    out = outdir / "planarity.png"
    fig.savefig(out, dpi=125)
    plt.close(fig)
    print(f"\nwrote {out}")

    if args.cull:
        dst = Path("bags") / args.cull
        (dst / "frames").mkdir(parents=True, exist_ok=True)
        kept = 0
        for r, ok in zip(usable, good):
            if not ok:
                continue
            for idx in ("ir1", "ir2"):
                src = args.bag / "frames" / f"{idx}_{r['num']}.raw"
                if src.exists():
                    shutil.copy2(src, dst / "frames" / src.name)
            kept += 1
        extra = f"culled_from {args.bag.name}\nmax_rms_px {args.max_rms}\n"
        (dst / "run.txt").write_text(
            (args.bag / "run.txt").read_text() + extra
            if (args.bag / "run.txt").exists() else extra)
        print(f"culled set: {kept} poses -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
