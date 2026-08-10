#!/usr/bin/env python3
"""Price a real predictor against the oracle intervals range_ceiling.py measured.

The ceiling says a tile needs 19.2% of the sweep if it KNOWS its interval, and that
predictor accuracy dominates everything else -- about 2.4 points of the sweep per
plane of slack. So the question stopped being "which construction" and became "how
accurately can anything cheap predict the interval". This measures that for the one
predictor already in the tree: a half-resolution pass.

WHY THIS IS NOT WHAT --c2f ALREADY DOES. `--c2f` prunes by per-PLANE rectangle: for
each disparity it computes the bounding box of the pixels whose prior admits it, and
skips the rest of the image. A bounding box over the whole image is nearly the whole
image as soon as one distant object wants that disparity, which is why c2f measured
1.24x on the desktop and flat on the TX2. Per-TILE intervals are a different and much
tighter quantity, and this script measures whether the same coarse prior would do
better under that scheme.

Two numbers per configuration, and both matter:
  cost    fraction of the full sweep the tiles would compute
  recall  fraction of pixels whose TRUE disparity is inside its tile's band

A predictor is only usable where recall is high enough that the band is not throwing
away the answer; the ceiling script's oracle has recall 1.0 by construction.

    .venv/bin/python article/range_predictor.py --data ~/data/MiddEval3
"""

import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import middeval3 as m3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "core", "build", "de_dense")


def half(img):
    """2x2 box down, matching how the pyramid would actually be built."""
    H, W = img.shape
    h, w = H // 2, W // 2
    a = img[:h * 2, :w * 2].astype(np.uint16).reshape(h, 2, w, 2)
    return (a.sum(axis=(1, 3)) // 4).astype(np.uint8)


def coarse_disp(d, ndisp, threads):
    """Run de_dense at half resolution; return a full-res disparity prediction."""
    L = np.asarray(Image.open(os.path.join(d, "im0.png")).convert("L"), np.uint8)
    R = np.asarray(Image.open(os.path.join(d, "im1.png")).convert("L"), np.uint8)
    Lh, Rh = half(L), half(R)
    h, w = Lh.shape
    lp = os.path.join(d, ".rp_l.y8")
    rp = os.path.join(d, ".rp_r.y8")
    op = os.path.join(d, ".rp_o.f32")
    Lh.tofile(lp)
    Rh.tofile(rp)
    dmax = max(2, ndisp // 2)
    p = subprocess.run([BIN, lp, rp, str(w), str(h), "--dmax", str(dmax),
                        "--threads", str(threads), "--agg", "5", "--iters", "2",
                        "--min-margin", "0.01", "--out", op],
                       capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"de_dense failed on the coarse pass ({d}): {p.stderr}")
    dc = np.fromfile(op, np.float32).reshape(h, w)
    for f in (lp, rp, op):
        os.remove(f)
    # half-res disparity d maps to 2d at full res; nearest-neighbour up
    return np.repeat(np.repeat(dc * 2.0, 2, 0), 2, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MIDDEVAL3", ""))
    ap.add_argument("--res", default="Q")
    ap.add_argument("--tile", type=int, default=16)
    ap.add_argument("--pads", default="0,2,4,8,16")
    ap.add_argument("--threads", default="8")
    a = ap.parse_args()
    if not a.data:
        sys.exit("need --data or $MIDDEVAL3")
    pads = [int(v) for v in a.pads.split(",")]
    T = a.tile

    tot_cost = {p: [0.0, 0.0] for p in pads}
    tot_rec = {p: [0.0, 0.0] for p in pads}
    empty_pix = 0.0
    all_pix = 0.0

    for s in sorted(m3.WEIGHTS):
        d = os.path.join(a.data, f"training{a.res}", s)
        gt = m3.read_pfm(os.path.join(d, "disp0GT.pfm"))
        D = int(m3.read_calib(os.path.join(d, "calib.txt"))["ndisp"])
        pred = coarse_disp(d, D, a.threads)
        H, W = gt.shape
        pred = pred[:H, :W]
        if pred.shape != gt.shape:                       # odd sizes: pad by edge
            pp = np.full(gt.shape, np.nan, np.float32)
            pp[:pred.shape[0], :pred.shape[1]] = pred
            pred = pp
        gval = np.isfinite(gt)
        pval = np.isfinite(pred) & (pred > 0)

        for by in range(0, H, T):
            for bx in range(0, W, T):
                ys, xs = slice(by, min(H, by + T)), slice(bx, min(W, bx + T))
                n = (ys.stop - ys.start) * (xs.stop - xs.start)
                all_pix += n
                pv = pred[ys, xs][pval[ys, xs]]
                gv = gt[ys, xs][gval[ys, xs]]
                if pv.size == 0:
                    # No coarse evidence: the tile must sweep everything. Counted,
                    # not hidden -- this is the failure mode the scheme has to own.
                    empty_pix += n
                    for p in pads:
                        tot_cost[p][0] += D * n
                        tot_cost[p][1] += D * n
                        tot_rec[p][0] += gv.size
                        tot_rec[p][1] += gv.size
                    continue
                lo0, hi0 = pv.min(), pv.max()
                for p in pads:
                    lo = max(0.0, lo0 - p)
                    hi = min(D - 1.0, hi0 + p)
                    tot_cost[p][0] += (hi - lo + 1) * n
                    tot_cost[p][1] += D * n
                    if gv.size:
                        tot_rec[p][0] += float(((gv >= lo) & (gv <= hi)).sum())
                        tot_rec[p][1] += gv.size

    print(f"Half-resolution predictor, tile {T}, Middlebury v3 training{a.res}, "
          f"15 scenes.\n")
    print(f"{'pad':>5}{'cost':>10}{'recall':>10}   (oracle at this tile: "
          f"9.8% / 100% at pad 0)")
    for p in pads:
        c = 100 * tot_cost[p][0] / tot_cost[p][1]
        r = 100 * tot_rec[p][0] / max(1.0, tot_rec[p][1])
        print(f"{p:>5}{c:>9.1f}%{r:>9.1f}%")
    print(f"\nTiles with no coarse disparity at all, and so no band: "
          f"{100*empty_pix/all_pix:.1f}% of pixels.")
    print("A band that excludes the true disparity does not cost time, it costs the "
          "answer,\nso read the two columns together: cost is only a saving at a "
          "recall you can accept.")


if __name__ == "__main__":
    main()
