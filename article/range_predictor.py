#!/usr/bin/env python3
"""Price a real predictor against the oracle intervals range_ceiling.py measured.

The ceiling says a tile needs 19.2% of the sweep if it KNOWS its interval, and that
predictor accuracy dominates everything else -- about 2.4 points of the sweep per
plane of slack. So the question stopped being "which construction" and became "how
accurately can anything cheap predict the interval". This prices candidates on that
axis, and it needs only a candidate's predicted interval, not a matcher built on it.

Two numbers per configuration, and both matter:
  cost    fraction of the full sweep the tiles would compute
  recall  fraction of pixels whose TRUE disparity is inside its tile's band

A band that excludes the answer does not cost time, it costs the answer, so cost is
only a saving at a recall you can accept. The oracle is 19.2% at 100% (tile 16,
pad 4).

PREDICTORS

  half   A half-resolution pass, the one already in the tree. Point prediction per
         pixel; the tile band is its min..max plus slack.

  icsg   Intrinsic curves (Tomasi & Manduchi 1998; Shahbazi et al., ISPRS 2016,
         doi:10.5194/isprs-archives-XLI-B3-123-2016), which claim an 83% per-pixel
         range reduction at full resolution with no hierarchical search. A pixel is
         a point in a small feature space -- here intensity and its scanline
         derivative -- and a disparity is admitted when the right pixel it points at
         is near the left one in THAT space, not in image space.

         Computed here by brute force over d. That is deliberately not how the paper
         does it: the paper's contribution is finding those neighbours through a
         spatial index, in less than a sweep. Brute force answers the question that
         has to come first -- whether the reduction is accurate enough to be worth
         indexing -- and if it is not, the index never needs writing.

    .venv/bin/python article/range_predictor.py --data ~/data/MiddEval3
    .venv/bin/python article/range_predictor.py --predictor icsg --taus 4,8,16,32
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


def predict_half(d, D, threads, _param):
    """Run de_dense at half resolution; return per-pixel (lo, hi) = (pred, pred)."""
    L = np.asarray(Image.open(os.path.join(d, "im0.png")).convert("L"), np.uint8)
    R = np.asarray(Image.open(os.path.join(d, "im1.png")).convert("L"), np.uint8)
    Lh, Rh = half(L), half(R)
    h, w = Lh.shape
    lp, rp, op = (os.path.join(d, f".rp_{n}") for n in ("l.y8", "r.y8", "o.f32"))
    Lh.tofile(lp)
    Rh.tofile(rp)
    p = subprocess.run([BIN, lp, rp, str(w), str(h), "--dmax", str(max(2, D // 2)),
                        "--threads", str(threads), "--agg", "5", "--iters", "2",
                        "--min-margin", "0.01", "--out", op],
                       capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"de_dense failed on the coarse pass ({d}): {p.stderr}")
    dc = np.fromfile(op, np.float32).reshape(h, w)
    for f in (lp, rp, op):
        os.remove(f)
    up = np.repeat(np.repeat(dc * 2.0, 2, 0), 2, 1)
    return up, up


def predict_icsg(d, D, _threads, tau):
    """Per-pixel [lo, hi] over the disparities the feature space admits.

    The feature is (I, w * dI/dx) on the scanline. The derivative is what makes the
    curve informative: intensity alone is hopelessly ambiguous on any real scene,
    and the pair separates a rising edge from a falling one at the same brightness.
    """
    L = np.asarray(Image.open(os.path.join(d, "im0.png")).convert("L"), np.float32)
    R = np.asarray(Image.open(os.path.join(d, "im1.png")).convert("L"), np.float32)
    # central difference along the scanline, which is the direction disparity moves
    Lx = np.gradient(L, axis=1)
    Rx = np.gradient(R, axis=1)
    WG = 4.0                      # derivative weight; gradients are ~4x smaller here
    H, W = L.shape
    lo = np.full((H, W), np.inf, np.float32)
    hi = np.full((H, W), -np.inf, np.float32)
    cnt = np.zeros((H, W), np.int32)
    # Full admitted stack, D planes of bool (~26 MB at Q), so a tile can be asked
    # WHERE its admitted mass is and not merely how far apart its extremes are.
    stack = np.zeros((D, H, W), bool)
    for k in range(D):
        dd = 1 + k                                   # cfg.dmin is 1
        if dd >= W:
            break
        dist = (np.abs(L[:, dd:] - R[:, :W - dd]) +
                WG * np.abs(Lx[:, dd:] - Rx[:, :W - dd]))
        ok = dist < tau
        sl = np.s_[:, dd:]
        lo[sl] = np.where(ok & np.isinf(lo[sl]), float(dd), lo[sl])
        hi[sl] = np.where(ok, float(dd), hi[sl])
        cnt[sl] += ok
        stack[k, :, dd:] = ok
    predict_icsg.admitted = float(cnt.sum()) / max(1.0, float(cnt.size) * D)
    predict_icsg.stack = stack
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MIDDEVAL3", ""))
    ap.add_argument("--res", default="Q")
    ap.add_argument("--tile", type=int, default=16)
    ap.add_argument("--pads", default="0,2,4,8,16")
    ap.add_argument("--taus", default="8,16,32,64", help="icsg feature thresholds")
    ap.add_argument("--threads", default="8")
    ap.add_argument("--predictor", default="half", choices=["half", "icsg"])
    ap.add_argument("--band", default="minmax", choices=["minmax", "pct"],
                    help="pct: narrowest interval covering --cover of a tile's votes")
    ap.add_argument("--cover", type=float, default=0.95)
    a = ap.parse_args()
    if not a.data:
        sys.exit("need --data or $MIDDEVAL3")
    T = a.tile
    pads = [int(v) for v in a.pads.split(",")]
    params = [None] if a.predictor == "half" else [float(v) for v in a.taus.split(",")]
    fn = predict_half if a.predictor == "half" else predict_icsg

    for param in params:
        cost = {p: [0.0, 0.0] for p in pads}
        rec = {p: [0.0, 0.0] for p in pads}
        empty, allp, adm = 0.0, 0.0, []
        for s in sorted(m3.WEIGHTS):
            d = os.path.join(a.data, f"training{a.res}", s)
            gt = m3.read_pfm(os.path.join(d, "disp0GT.pfm"))
            D = int(m3.read_calib(os.path.join(d, "calib.txt"))["ndisp"])
            plo, phi = fn(d, D, a.threads, param)
            stack = (predict_icsg.stack if (a.band == "pct" and
                                            a.predictor == "icsg") else None)
            if a.predictor == "icsg":
                adm.append(predict_icsg.admitted)
            H, W = gt.shape
            plo, phi = plo[:H, :W], phi[:H, :W]
            live = np.isfinite(plo) & np.isfinite(phi) & (phi >= plo)
            gval = np.isfinite(gt)
            for by in range(0, H, T):
                for bx in range(0, W, T):
                    ys, xs = slice(by, min(H, by + T)), slice(bx, min(W, bx + T))
                    n = (ys.stop - ys.start) * (xs.stop - xs.start)
                    allp += n
                    gv = gt[ys, xs][gval[ys, xs]]
                    lv = live[ys, xs]
                    if not lv.any():
                        # No evidence at all: the tile sweeps everything. Counted,
                        # never hidden -- it is the scheme's own failure mode.
                        empty += n
                        for p in pads:
                            cost[p][0] += D * n
                            cost[p][1] += D * n
                            rec[p][0] += gv.size
                            rec[p][1] += gv.size
                        continue
                    if a.band == "minmax":
                        lo0 = float(plo[ys, xs][lv].min())
                        hi0 = float(phi[ys, xs][lv].max())
                    else:
                        # Narrowest contiguous interval holding --cover of the
                        # tile's admitted votes. min..max is hostage to one
                        # outlying pixel; this asks where the mass actually is.
                        h = stack[:, ys, xs].sum(axis=(1, 2))
                        tot_v = h.sum()
                        if tot_v == 0:
                            lo0, hi0 = 0.0, float(D - 1)
                        else:
                            c = np.concatenate(([0], np.cumsum(h)))
                            need = a.cover * tot_v
                            best = (D, 0, D - 1)
                            j = 0
                            for i in range(D):
                                while j < D and c[j + 1] - c[i] < need:
                                    j += 1
                                if j >= D:
                                    break
                                if j - i < best[0]:
                                    best = (j - i, i, j)
                            lo0, hi0 = float(best[1]), float(best[2])
                    for p in pads:
                        lo = max(0.0, lo0 - p)
                        hi = min(D - 1.0, hi0 + p)
                        cost[p][0] += (hi - lo + 1) * n
                        cost[p][1] += D * n
                        if gv.size:
                            rec[p][0] += float(((gv >= lo) & (gv <= hi)).sum())
                            rec[p][1] += gv.size

        tag = f"{a.predictor}" + (f", tau {param:g}" if param is not None else "")
        print(f"\n{tag} -- tile {T}, {len(m3.WEIGHTS)} scenes"
              + (f", per-pixel admitted {100*np.mean(adm):.1f}% of the range"
                 if adm else ""))
        print(f"{'pad':>5}{'cost':>10}{'recall':>10}")
        for p in pads:
            print(f"{p:>5}{100*cost[p][0]/cost[p][1]:>9.1f}%"
                  f"{100*rec[p][0]/max(1.0,rec[p][1]):>9.1f}%")
        print(f"      tiles with no evidence: {100*empty/allp:.1f}% of pixels")

    print("\nOracle for reference: 9.8% at pad 0, 19.2% at pad 4, both at 100% "
          "recall.\nCost is only a saving at a recall you can accept.")


if __name__ == "__main__":
    main()
