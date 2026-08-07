#!/usr/bin/env python3
"""Detector repeatability is the ceiling. Can over-proposing on the right lift it?

Measured on eight Middlebury scenes: only 48-51% of left keypoints have any right
keypoint within 1 px of their true correspondence. The matcher cannot recover a
correspondence that was never proposed to it, so that number caps recall before
any message is passed, and it accounts for 102 of Teddy's 132 errors.

Detecting independently in both images is what causes it. Shi-Tomasi picks its
maxima per image, and a maximum in one view is often not a maximum in the other.

The fix that keeps the formulation intact is to over-propose on the right and let
the misdetection term throw the surplus away. That is exactly what gamma is for.
It also needs gamma decoupled from lambda, which was only possible after the two
were un-transposed: over-proposing means many right keypoints legitimately go
unmatched, so leaving one unmatched should be CHEAP, while leaving a left keypoint
unmatched stays strict.

The tension is real and the outcome is not obvious. More right keypoints raise the
recall ceiling and simultaneously add distractors to every left keypoint's
candidate list, which lowers the score margin. This measures which wins.

  python repeatability.py
"""

import json
import time

import numpy as np

import masda_stereo as ms
import masda_middlebury as mb
import ordering_real as orr

SCENES = ("teddy", "cones") + orr.SCENES_2005
TOL = 1.0
# Right-image detector density. per_cell is keypoints kept per grid cell and pct
# is the response percentile below which a candidate is discarded, so lowering pct
# and raising per_cell both propose more.
DENSITIES = [
    ("baseline", 2, 80),
    ("2x", 4, 70),
    ("3x", 6, 60),
    ("4x", 8, 50),
]
# (lambda, gamma). Symmetric first, then progressively cheaper misdetection.
COSTS = [(-0.1, -0.1), (-0.1, -0.02), (-0.1, 0.0)]


def detect_at(img, per_cell, pct, cell=12):
    """ms.detect with an explicit response percentile."""
    r = ms.shi_tomasi(img)
    pts, _ = ms.detect(img, cell=cell, per_cell=per_cell,
                       min_resp=float(np.percentile(r, pct)))
    return pts


def repeatability(pl, pr, gt, known, tol=TOL):
    """Share of left keypoints whose true correspondence was actually detected."""
    H, W = gt.shape
    xi = np.clip(np.round(pl[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(pl[:, 1]).astype(int), 0, H - 1)
    kk, kg = known[yi, xi], gt[yi, xi]
    true_xr = pl[:, 0] - kg
    dx = np.abs(true_xr[:, None] - pr[None, :, 0])
    dy = np.abs(pl[:, 1][:, None] - pr[None, :, 1])
    detected = ((dx <= tol) & (dy <= tol)).any(1)
    return float((kk & detected).sum()) / max(1, len(pl)), int((kk & detected).sum())


def run():
    out = {}
    for label, per_cell, pct in DENSITIES:
        for lam, gam in COSTS:
            key = f"{label}|lam{lam}|gam{gam}"
            tot = dict(tp=0, scorable=0, matchable=0, kp_l=0, kp_r=0,
                       edges=0, ms=0.0, rep=[])
            for name in SCENES:
                left, right, gt, known, _ = orr.load_scene(name)
                H, W = left.shape
                ms.H, ms.W = H, W
                dmax = 60.0 if name in ("teddy", "cones") else 80.0
                pl = detect_at(left, 2, 80)              # left is never changed
                pr = detect_at(right, per_cell, pct)
                rep, _ = repeatability(pl, pr, gt, known)
                dl, dr = ms.census(left, pl), ms.census(right, pr)
                S = ms.build_problem(pl, dl, pr, dr, dmin=1.0, dmax=dmax)
                m, n = S.shape
                ei, ej, se = ms.to_edges(S)
                t = time.perf_counter()
                a, _ = ms.masda_sparse(ei, ej, se, m, n, lam, gam)
                el = (time.perf_counter() - t) * 1e3
                e = mb.evaluate_real(a, pl, pr, gt, known, TOL)
                tot["tp"] += e["tp"]; tot["scorable"] += e["scorable"]
                tot["matchable"] += e["matchable"]
                tot["kp_l"] += len(pl); tot["kp_r"] += len(pr)
                tot["edges"] += len(se); tot["ms"] += el
                tot["rep"].append(rep)
            tot["rep_mean"] = float(np.mean(tot["rep"]))
            tot["prec"] = tot["tp"] / max(1, tot["scorable"])
            tot["recall"] = tot["tp"] / max(1, tot["matchable"])
            out[key] = tot
            print(f"{label:<9} lam={lam:<5} gam={gam:<5} "
                  f"repeat={100*tot['rep_mean']:>5.1f}%  "
                  f"kp_r={tot['kp_r']:>5}  edges={tot['edges']:>6}  "
                  f"matchable={tot['matchable']:>5}  correct={tot['tp']:>5}  "
                  f"prec={tot['prec']:.3f}  recall={tot['recall']:.3f}  "
                  f"{tot['ms']/len(SCENES):>6.1f} ms")
    return out


def main():
    print("=" * 104)
    print("Over-proposing on the right image, with the misdetection cost varied")
    print("=" * 104)
    out = run()
    base = out[f"baseline|lam-0.1|gam-0.1"]
    print("\nagainst the baseline (symmetric detection, lambda == gamma):")
    print(f"{'config':<28}{'d correct':>11}{'d prec':>9}{'d recall':>10}"
          f"{'edge cost':>11}{'time cost':>11}")
    for k, v in out.items():
        print(f"{k:<28}{v['tp']-base['tp']:>+11d}"
              f"{v['prec']-base['prec']:>+9.3f}{v['recall']-base['recall']:>+10.3f}"
              f"{v['edges']/base['edges']:>10.2f}x{v['ms']/base['ms']:>10.2f}x")
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "rep"}
               for k, v in out.items()}, open("repeatability.json", "w"),
              indent=1, default=float)
    print("\nwrote repeatability.json")


if __name__ == "__main__":
    main()
