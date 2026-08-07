#!/usr/bin/env python3
"""Does the ordering factor help on real data? Eight scenes, not two.

The article's ordering result is measured over five random seeds on synthetic
scenes, because the effect turned out to be smaller than the scene-to-scene
spread. The real-data half had no such treatment: two scenes, one run each, and
one of them moved in the opposite direction. That is the same mistake section 7.3
warns about, so this script gets real replicates.

Middlebury 2005 adds six more scenes with structured-light ground truth under the
same licence as the 2003 pairs. Together with Teddy and Cones that is eight real
scenes, which supports a mean and a spread.

The 2005 scenes also have wider disparity ranges than the 2003 pairs (up to ~75 px
at third size against 60), which makes them a test of *why* ordering is redundant
rather than just whether it is. Section 7.2 argues the disparity-range gate already
forbids most crossings, since a crossing needs

    d(i') - d(i) > x(i') - x(i)

so a wider range should admit more crossings and give ordering more to do. If the
explanation is right, the effect should grow with the gate width.

  python ordering_real.py
"""

import io
import json
import os
import ssl
import urllib.request
import zipfile

import numpy as np
from PIL import Image

import masda_stereo as ms
import masda_middlebury as mb

DATA = mb.DATA
Z2005 = ("https://vision.middlebury.edu/stereo/data/scenes2005/ThirdSize/"
         "zip-2views/{s}-2views.zip")
SCENES_2005 = ("Art", "Books", "Dolls", "Laundry", "Moebius", "Reindeer")
GT_SCALE_2005 = 3.0        # established by cross-view consistency, see below
KAPPAS = (0.1, 0.3, 0.8)
TOL = 1.0


def fetch_2005(scene):
    """Download and load one Middlebury 2005 third-size scene.

    The disparity scale is not documented in the zip, so it is established by a
    check that does not involve the matcher: for a pixel at x in view1 with true
    disparity t, disp5 at (y, x - t) must also read t. Only the correct scale makes
    that identity hold, and k=3 gives a median discrepancy of exactly 0.000 px
    against 23.0 at k=1 and 0.5 at k=4. verify_scale() re-checks it per scene.
    """
    d = os.path.join(DATA, "2005", scene)
    need = ("view1.png", "view5.png", "disp1.png", "disp5.png")
    if not all(os.path.exists(os.path.join(d, f)) for f in need):
        os.makedirs(d, exist_ok=True)
        ctx = ssl._create_unverified_context()
        blob = urllib.request.urlopen(Z2005.format(s=scene), timeout=180,
                                      context=ctx).read()
        z = zipfile.ZipFile(io.BytesIO(blob))
        for f in need:
            open(os.path.join(d, f), "wb").write(z.read(f"{scene}/{f}"))

    def load(f):
        return np.array(Image.open(os.path.join(d, f)))

    left = mb.luma(load("view1.png"))
    right = mb.luma(load("view5.png"))
    raw1 = load("disp1.png").astype(np.float32)
    raw5 = load("disp5.png").astype(np.float32)
    gt = raw1 / GT_SCALE_2005
    known = raw1 > 0
    return left, right, gt, known, raw5 / GT_SCALE_2005, raw5 > 0


def verify_scale(gt, known, gt5, known5):
    """Cross-view consistency, as a guard against a silently wrong scale."""
    H, W = gt.shape
    yy, xx = np.mgrid[0:H, 0:W]
    x5 = np.round(xx - gt).astype(int)
    ok = known & (x5 >= 0) & (x5 < W)
    ok &= known5[np.clip(yy, 0, H - 1), np.clip(x5, 0, W - 1)]
    err = np.abs(gt[ok] - gt5[yy[ok], x5[ok]])
    return float(np.median(err)), float((err <= 1.0).mean())


def load_scene(name):
    """Uniform interface over the 2003 pairs and the 2005 set."""
    if name in ("teddy", "cones"):
        left, right, gt, known = mb.fetch(name)
        return left, right, gt, known, None
    left, right, gt, known, gt5, known5 = fetch_2005(name)
    return left, right, gt, known, verify_scale(gt, known, gt5, known5)


def run(name, dmax_mode="published"):
    """Ordering sweep on one real scene.

    dmax comes from the published search range (60 for the 2003 pairs) or, for the
    2005 set which does not ship one in the 2-view zip, from a fixed 80 px bound
    that covers every scene in the set. It is deliberately NOT taken from the
    ground-truth maximum, which would leak the answer into the search range.
    """
    left, right, gt, known, scale_check = load_scene(name)
    H, W = left.shape
    ms.H, ms.W = H, W
    dmax = 60.0 if name in ("teddy", "cones") else 80.0

    pl, _ = ms.detect(left)
    pr, _ = ms.detect(right)
    dl, dr = ms.census(left, pl), ms.census(right, pr)
    S = ms.build_problem(pl, dl, pr, dr, dmin=1.0, dmax=dmax)
    m, n = S.shape
    ei, ej, se = ms.to_edges(S)

    a, _ = ms.masda_sparse(ei, ej, se, m, n, -0.1, -0.1)
    base = mb.evaluate_real(a, pl, pr, gt, known, TOL)
    cr, tot = ms.count_crossings(a, pl, pr)
    out = dict(scene=name, H=H, W=W, m=m, n=n, E=len(se), dmax=dmax,
               gt_max=float(gt[known].max()), scale_check=scale_check,
               off=dict(correct=base["tp"], matches=base["matches"],
                        prec=base["precision"], crossings=cr, pairs=tot))
    for k in KAPPAS:
        ao, _, _ = ms.masda_sparse_ordering(ei, ej, se, m, n, pl, pr,
                                            kappa=k, lam=-0.1, gam=-0.1)
        e = mb.evaluate_real(ao, pl, pr, gt, known, TOL)
        c, _ = ms.count_crossings(ao, pl, pr)
        out[f"k{k}"] = dict(correct=e["tp"], matches=e["matches"],
                            prec=e["precision"], crossings=c)
    return out


def main():
    scenes = ("teddy", "cones") + SCENES_2005
    rows = []
    for s in scenes:
        r = run(s)
        rows.append(r)
        sc = ("" if r["scale_check"] is None else
              f"  scale-check med={r['scale_check'][0]:.3f}px "
              f"{100*r['scale_check'][1]:.0f}%<1px")
        print(f"\n{r['scene']:<9} {r['W']}x{r['H']} m={r['m']} n={r['n']} "
              f"E={r['E']} dmax={r['dmax']:.0f} gt_max={r['gt_max']:.1f}{sc}")
        o = r["off"]
        print(f"  off      correct={o['correct']:>4} prec={o['prec']:.3f} "
              f"crossings={o['crossings']:>4}/{o['pairs']}")
        for k in KAPPAS:
            e = r[f"k{k}"]
            print(f"  k={k:<5} correct={e['correct']:>4} prec={e['prec']:.3f} "
                  f"crossings={e['crossings']:>4}   "
                  f"d(correct)={e['correct']-o['correct']:+d} "
                  f"d(cross)={e['crossings']-o['crossings']:+d}")

    print("\n" + "=" * 74)
    print("Paired effect of the ordering factor over %d real scenes" % len(rows))
    print("=" * 74)
    print(f"{'kappa':<8}{'d(correct)':>18}{'d(crossings)':>20}"
          f"{'rel. crossings':>17}")
    agg = {}
    for k in KAPPAS:
        dc = np.array([r[f"k{k}"]["correct"] - r["off"]["correct"] for r in rows],
                      float)
        dx = np.array([r[f"k{k}"]["crossings"] - r["off"]["crossings"]
                       for r in rows], float)
        rel = np.array([(r[f"k{k}"]["crossings"] + 1e-9) /
                        max(1, r["off"]["crossings"]) for r in rows], float)
        agg[f"k{k}"] = dict(d_correct_mean=dc.mean(), d_correct_sd=dc.std(),
                            d_cross_mean=dx.mean(), d_cross_sd=dx.std(),
                            rel_cross_mean=rel.mean(),
                            n_better=int((dc > 0).sum()),
                            n_worse=int((dc < 0).sum()))
        print(f"{k:<8}{dc.mean():>+9.1f} ± {dc.std():<5.1f}"
              f"{dx.mean():>+12.1f} ± {dx.std():<5.1f}"
              f"{rel.mean():>16.2f}×")
    for k in KAPPAS:
        a = agg[f"k{k}"]
        print(f"  k={k}: better on {a['n_better']} scenes, worse on "
              f"{a['n_worse']}, unchanged on "
              f"{len(rows)-a['n_better']-a['n_worse']}")

    # Does the effect grow with the width of the disparity gate, as 7.2 predicts?
    print("\nSection 7.2 predicts crossings scale with the disparity gate width.")
    print(f"{'scene':<10}{'dmax':>6}{'crossings/pair':>17}{'d(correct) k=0.3':>19}")
    for r in sorted(rows, key=lambda r: r["dmax"]):
        frac = r["off"]["crossings"] / max(1, r["off"]["pairs"])
        print(f"{r['scene']:<10}{r['dmax']:>6.0f}{100*frac:>16.2f}%"
              f"{r['k0.3']['correct']-r['off']['correct']:>+19d}")
    json.dump(dict(rows=rows, agg=agg), open("ordering_real.json", "w"),
              indent=1, default=float)
    print("\nwrote ordering_real.json")


if __name__ == "__main__":
    main()
