#!/usr/bin/env python3
"""Measure the matcher's noise on the real camera, using a flat wall as the truth.

Every parameter in this matcher was chosen against Middlebury, and doc/TODO.md 0.45
measures how far the real camera is from those scenes: 2-3x less local contrast and
6.4% of the frame saturated. Nothing has ever been tuned on the sensor it runs on,
because there is no ground truth for it.

A wall is the cheapest ground truth there is. On its own it says nothing about
absolute scale, but --truth below fixes that with a tape measure, and noise is what a
plane measures with no equipment at all:

  fit a plane to the cloud, and the scatter around it IS the matcher's noise.

A flat surface is also the hardest case for a stereo matcher and the easiest for a
human to arrange, which is a good combination. Point the camera at a blank wall,
close enough to fill the frame, and record a few seconds:

    ./tools/deploy.sh --capture wall01 --seconds 3 --emitter on
    ./tools/deploy.sh --pull wall01
    .venv/bin/python desktop/wall_check.py bags/wall01

    .venv/bin/python desktop/wall_check.py bags/wall01 \\
        --sweep " " " --min-margin 0.01" " --no-subpixel" " --ad 0"

Each swept config needs a LEADING SPACE, or argparse claims it as its own flag.

A plane fit measures scatter and cannot measure SCALE: multiply every depth by 1.05
and the plane is just as flat. A tape measure closes that, and does not need to be a
rangefinder -- at 1 m one disparity pixel is 46.6 mm against a measured repeatability
of 0.146 px, so 2 mm of tape is well inside the noise. Record the same wall at two or
three distances and pass them:

    .venv/bin/python desktop/wall_check.py bags/w60 bags/w120 bags/w180 \\
        --truth 0.60 1.20 1.80

It fits measured = scale*ruler + offset. Only the scale is a claim about the matcher;
the offset absorbs the distance from whatever you held the tape against to the point
depth is actually referenced from, which is why the tape does not have to start in the
right place. Two distances fit the line exactly and cannot be falsified -- use three.

The sweep is the point of the tool: it makes a parameter comparison possible on the
real sensor, which until now could only be done on Middlebury and assumed to carry.
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "core", "build", "de_dense")
FX = FY = 430.551
CX, CY = 427.381, 243.158
BASELINE_M = 0.049883
FB = FX * BASELINE_M


def load_pair(bag, index):
    ls = sorted(glob.glob(os.path.join(bag, "frames", "ir1_*.raw")))
    rs = sorted(glob.glob(os.path.join(bag, "frames", "ir2_*.raw")))
    if not ls or not rs:
        sys.exit(f"{bag}: no ir1_/ir2_ frames. Record with --streams both.")
    if len(ls) != len(rs):
        print(f"note: {len(ls)} left and {len(rs)} right frames, using the shorter",
              file=sys.stderr)
    n = min(len(ls), len(rs))
    i = min(index, n - 1)
    a = np.fromfile(ls[i], np.uint8)
    b = np.fromfile(rs[i], np.uint8)
    for wh in ((848, 480), (640, 480), (424, 240)):
        if a.size == wh[0] * wh[1]:
            return a.reshape(wh[1], wh[0]), b.reshape(wh[1], wh[0]), wh[0], wh[1]
    sys.exit(f"{ls[i]}: {a.size} bytes is not a resolution this knows")


def run(L, R, W, H, dmax, extra):
    lp, rp, op = "/tmp/_wall_l.y8", "/tmp/_wall_r.y8", "/tmp/_wall.f32"
    L.tofile(lp)
    R.tofile(rp)
    cmd = [BIN, lp, rp, str(W), str(H), "--dmax", str(dmax), "--threads", "8",
           "--agg", "5", "--out", op] + extra
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"de_dense failed:\n{p.stderr}")
    d = np.fromfile(op, np.float32).reshape(H, W)
    for f in (lp, rp, op):
        os.remove(f)
    return d


def fit_plane(x, y, z, iters=6, trim=2.5):
    """Least squares z = ax + by + c, re-fit after trimming outliers.

    Trimmed rather than RANSAC because a wall fills the frame: the inliers are the
    overwhelming majority and the question is how far the rest scatter, not whether a
    plane exists at all. RANSAC would find the same plane and hide the tail.
    """
    keep = np.ones(len(z), bool)
    a = b = c = 0.0
    for _ in range(iters):
        A = np.stack([x[keep], y[keep], np.ones(keep.sum())], 1)
        (a, b, c), *_ = np.linalg.lstsq(A, z[keep], rcond=None)
        r = z - (a * x + b * y + c)
        s = np.std(r[keep])
        if s <= 0:
            break
        keep = np.abs(r) < trim * s
    return a, b, c, z - (a * x + b * y + c), keep


def report(label, d, W, H, near, far):
    """Print one row, and return the plane's distance in metres (None if it failed).

    The returned distance is the fitted plane at the principal point, not the median
    of the samples: the median wanders with which part of the wall happened to match,
    and a ruler measurement is to one spot. `c` is exactly that intercept.
    """
    ok = np.isfinite(d) & (d > 0)
    z = np.where(ok, FB / np.maximum(d, 1e-6), np.nan)
    g = ok & np.isfinite(z) & (z > near) & (z < far)
    if g.sum() < 1000:
        print(f"{label:<26} only {g.sum()} points in {near}-{far} m -- "
              f"is the wall in range?")
        return None
    vy, vx = np.nonzero(g)
    zz = z[g]
    x = (vx - CX) * zz / FX
    y = (vy - CY) * zz / FY
    a, b, c, res, keep = fit_plane(x, y, zz)
    tilt = np.degrees(np.arctan(np.hypot(a, b)))
    rms = np.std(res[keep]) * 1000.0
    out = 100.0 * (np.abs(res) > 0.02).mean()          # >2 cm off the plane
    print(f"{label:<26}{100*g.mean():7.1f}%{np.median(zz):8.2f}m{tilt:7.1f}°"
          f"{rms:9.1f}mm{out:9.1f}%{np.percentile(np.abs(res),95)*1000:9.1f}mm")
    return float(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", nargs="+")
    ap.add_argument("--truth", type=float, nargs="*", default=None, metavar="M",
                    help="ruler distance to the wall, one per bag, in metres. With "
                         "two or more it fits measured = scale*ruler + offset, which "
                         "is the only form worth reporting: the offset absorbs the "
                         "unknown distance from the housing you measured to to the "
                         "imager depth is referenced from, so the SCALE is testable "
                         "without knowing it")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--dmax", type=int, default=60)
    ap.add_argument("--near", type=float, default=0.3)
    ap.add_argument("--far", type=float, default=4.0)
    # Each config needs a LEADING SPACE, because argparse claims any bare
    # "--flag" as one of its own options -- the same trap tx2_ab.py documents and
    # dodges by making its two configs positional.
    ap.add_argument("--sweep", nargs="*", default=None, metavar="\" --flag\"",
                    help="configs to compare, each with a leading space: "
                         "--sweep \" \" \" --min-margin 0.01\" \" --no-subpixel\"")
    a = ap.parse_args()
    if not os.path.exists(BIN):
        sys.exit(f"{BIN} not built -- run `cd core && make`")

    if a.truth and len(a.truth) != len(a.bag):
        sys.exit(f"--truth needs one distance per bag: {len(a.bag)} bags, "
                 f"{len(a.truth)} distances")

    measured = []
    for bi, bag in enumerate(a.bag):
        L, R, W, H = load_pair(bag, a.frame)
        sat = 100.0 * (L > 250).mean()
        truth = a.truth[bi] if a.truth else None
        print(f"\n{bag}  frame {a.frame}  {W}x{H}   bright-clipped {sat:.1f}%"
              + (f"   ruler {truth:.3f} m" if truth is not None else ""))
        if sat > 2:
            print("  NOTE: saturated pixels carry no texture at all. If the wall is "
                  "lit unevenly,\n        lower --exposure-us when recording rather "
                  "than reading this as matcher noise.")
        print(f"\n{'config':<26}{'points':>7}{'depth':>8}{'tilt':>7}"
              f"{'RMS':>9}{'>2cm':>9}{'p95':>9}")
        first = None
        for extra in (a.sweep if a.sweep else [""]):
            z = report(extra if extra else "(defaults)",
                       run(L, R, W, H, a.dmax, extra.split()), W, H, a.near, a.far)
            if first is None:
                first = z
        measured.append(first)

    print("\nRMS is the scatter about the fitted plane -- the matcher's noise at this\n"
          "distance. >2cm is the fraction of gross outliers. Tilt is the wall's own\n"
          "angle and only matters if it is large enough to be a slanted-surface test\n"
          "rather than a fronto-parallel one.")

    if not a.truth:
        return
    pairs = [(t, m) for t, m in zip(a.truth, measured) if m is not None]
    if len(pairs) < 2:
        print("\nOne ruler distance measures nothing on its own: the difference "
              "between it and\nthe fitted plane is the housing-to-imager offset and "
              "a scale error added together,\nand one equation cannot separate two "
              "unknowns. Record a second bag at a\ndifferent distance.")
        if pairs:
            print(f"  ruler {pairs[0][0]:.3f} m, plane {pairs[0][1]:.3f} m, "
                  f"difference {1000*(pairs[0][1]-pairs[0][0]):+.0f} mm")
        return
    t = np.array([p[0] for p in pairs])
    m = np.array([p[1] for p in pairs])
    if np.ptp(t) < 0.05:
        sys.exit(f"the ruler distances span only {1000*np.ptp(t):.0f} mm. Scale is the "
                 f"slope of\nmeasured against truth, so it is divided by that span -- "
                 f"put at least half a\nmetre between them, and prefer the ends of "
                 f"the range you care about.")
    scale, off = np.polyfit(t, m, 1) if len(pairs) > 2 else \
        ((m[1] - m[0]) / (t[1] - t[0]), 0.0)
    if len(pairs) == 2:
        off = m[0] - scale * t[0]
    print(f"\n{'ruler':>10}{'plane':>10}{'difference':>13}{'residual':>11}")
    for ti, mi in zip(t, m):
        print(f"{ti:>9.3f}m{mi:>9.3f}m{1000*(mi-ti):>+11.0f}mm"
              f"{1000*(mi-(scale*ti+off)):>+9.0f}mm")
    print(f"\n  scale  {scale:.4f}   ({100*(scale-1):+.2f}% -- this is the one that "
          f"matters, and it is\n         f*B: a wrong baseline or focal length shows "
          f"up here and nowhere else)")
    print(f"  offset {1000*off:+.0f} mm  (where depth is referenced from, relative to "
          f"whatever you\n         put the ruler against -- a constant, and harmless "
          f"once you know it)")
    if len(pairs) == 2:
        print("\n  Two points define a line exactly, so there is no residual to check "
              "and no way\n  to tell a scale error from one bad measurement. Three "
              "distances make it falsifiable.")


if __name__ == "__main__":
    main()
