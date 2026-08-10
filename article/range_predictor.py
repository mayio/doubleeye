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


def diagnose(a, fn, T, pad):
    """Where does a predictor's recall deficit actually live?

    A predictor already at the oracle's COST whose whole deficit is recall is fixed
    by understanding its tail, not by replacing it. The buckets are the ones this
    project already has evidence about: disparity gradient (09-matching.md measured
    error flat to 0.3 px/px and exploding past it, with >0.6 being a discontinuity
    rather than a slant), occlusion, and tiles the coarse pass barely saw.
    """
    import collections
    tot = collections.Counter()
    miss = collections.Counter()
    for sc in sorted(m3.WEIGHTS):
        d = os.path.join(a.data, f"training{a.res}", sc)
        gt = m3.read_pfm(os.path.join(d, "disp0GT.pfm"))
        msk = np.array(Image.open(os.path.join(d, "mask0nocc.png")))
        D = int(m3.read_calib(os.path.join(d, "calib.txt"))["ndisp"])
        plo, _ = fn(d, D, a.threads, None)
        H, W = gt.shape
        plo = plo[:H, :W]
        live = np.isfinite(plo) & (plo > 0)
        gval = np.isfinite(gt)
        gy, gx = np.gradient(np.where(gval, gt, 0.0))
        grad = np.maximum(np.abs(gy), np.abs(gx))
        inband = np.zeros((H, W), bool)
        cov = np.zeros((H, W), np.float32)
        for by in range(0, H, T):
            for bx in range(0, W, T):
                ys, xs = slice(by, min(H, by + T)), slice(bx, min(W, bx + T))
                lv = live[ys, xs]
                cov[ys, xs] = lv.mean()
                if not lv.any():
                    continue
                pv = np.rint(plo[ys, xs][lv]).astype(int) - 1
                pv = pv[(pv >= 0) & (pv < D)]
                anyd = np.zeros(D, bool)
                if pv.size:
                    anyd[pv] = True
                aa = anyd.copy()
                for j in range(1, pad + 1):
                    aa[j:] |= anyd[:-j]
                    aa[:-j] |= anyd[j:]
                gi = np.clip(np.rint(gt[ys, xs]).astype(int) - 1, 0, D - 1)
                inband[ys, xs] = aa[gi]
        use = gval & (msk == 255)          # the official nonocc mask
        for name, sel in (("grad <= 0.3", grad <= 0.3),
                          ("grad 0.3-0.6", (grad > 0.3) & (grad <= 0.6)),
                          ("grad > 0.6 (discontinuity)", grad > 0.6),
                          ("occluded (mask 128)", gval & (msk == 128)),
                          ("tile coarse cov < 25%", cov < 0.25)):
            m = sel & (use if "occluded" not in name else gval)
            tot[name] += int(m.sum())
            miss[name] += int((m & ~inband).sum())
        tot["ALL nonocc"] += int(use.sum())
        miss["ALL nonocc"] += int((use & ~inband).sum())
    print(f"\nHalf-res predictor, tile {T}, pad {pad}: where the misses are\n")
    print(f"{'bucket':<30}{'pixels':>9}{'miss rate':>11}{'share of misses':>17}")
    allm = miss["ALL nonocc"]
    for k in ["grad <= 0.3", "grad 0.3-0.6", "grad > 0.6 (discontinuity)",
              "occluded (mask 128)", "tile coarse cov < 25%", "ALL nonocc"]:
        n = tot[k]
        print(f"{k:<30}{100*n/max(1,tot['ALL nonocc']):>8.1f}%"
              f"{100*miss[k]/max(1,n):>10.1f}%{100*miss[k]/max(1,allm):>16.1f}%")
    print("\nShare-of-misses is the column to act on: a bucket can have a terrible "
          "miss\nrate and still not be worth fixing if almost nothing lives in it.")


def halo_cost(a, fn, T):
    """Cost when the filter's halo is paid for, which is the cost that exists.

    A tile can only skip a plane if it also skips it for the pixels the filter would
    have read. rf's influence decays by ~0.89 per pixel at sigma_s 12, so de_dense
    pads a masked rectangle by MPAD = 16 before scoring it. A 16x16 tile that wants
    one plane therefore costs a 48x48 patch of that plane, and neighbouring tiles
    wanting the same plane share their halos -- which is why this measures the area
    of the DILATED UNION per plane rather than multiplying by (1 + 2*MPAD/T)^2.
    """
    from scipy.ndimage import binary_dilation
    pads = [int(v) for v in a.pads.split(",")]
    tot = {p: [0.0, 0.0] for p in pads}
    for sc in sorted(m3.WEIGHTS):
        d = os.path.join(a.data, f"training{a.res}", sc)
        gt = m3.read_pfm(os.path.join(d, "disp0GT.pfm"))
        D = int(m3.read_calib(os.path.join(d, "calib.txt"))["ndisp"])
        plo, _ = fn(d, D, a.threads, None)
        H, W = gt.shape
        plo = plo[:H, :W]
        live = np.isfinite(plo) & (plo > 0)
        ty, tx = (H + T - 1) // T, (W + T - 1) // T
        want = np.zeros((D, ty, tx), bool)
        for by in range(ty):
            for bx in range(tx):
                ys, xs = slice(by*T, min(H, (by+1)*T)), slice(bx*T, min(W, (bx+1)*T))
                lv = live[ys, xs]
                if not lv.any() or lv.mean() < a.fallback_cov:
                    want[:, by, bx] = True          # no evidence: sweep it all
                    continue
                pv = np.rint(plo[ys, xs][lv]).astype(int) - 1
                pv = pv[(pv >= 0) & (pv < D)]
                if pv.size:
                    want[pv, by, bx] = True
        st = np.ones((1 + 2 * (a.mpad // T), 1 + 2 * (a.mpad // T)), bool)
        for p in pads:
            area = 0.0
            for k in range(D):
                w = want[k]
                if p:                                # slack dilates in disparity
                    w = want[max(0, k - p):k + p + 1].any(axis=0)
                if not w.any():
                    continue
                # halo in TILE units, rounded up: the plane must be scored over
                # every tile the filter will read from a tile that wants it
                area += float(binary_dilation(w, st).sum()) * T * T
            tot[p][0] += area
            tot[p][1] += float(D) * H * W
    print(f"\nCost WITH the filter halo, tile {T}, MPAD {a.mpad}px, "
          f"fallback {a.fallback_cov}:\n")
    print(f"{'pad':>5}{'cost':>10}")
    for p in pads:
        print(f"{p:>5}{100*tot[p][0]/tot[p][1]:>9.1f}%")
    print("\nCompare the same rows without the halo. The difference is what the "
          "aggregation\ncosts, and it is not optional: it is where the accuracy "
          "comes from.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MIDDEVAL3", ""))
    ap.add_argument("--res", default="Q")
    ap.add_argument("--tile", type=int, default=16)
    ap.add_argument("--pads", default="0,2,4,8,16")
    ap.add_argument("--taus", default="8,16,32,64", help="icsg feature thresholds")
    ap.add_argument("--threads", default="8")
    ap.add_argument("--predictor", default="half", choices=["half", "icsg"])
    ap.add_argument("--band", default="minmax", choices=["minmax", "pct", "set"],
                    help="pct: narrowest interval covering --cover of a tile's votes")
    ap.add_argument("--cover", type=float, default=0.95)
    ap.add_argument("--mpad", type=int, default=0,
                    help="filter halo in px each side; the cost model pays for it")
    ap.add_argument("--fallback-cov", type=float, default=0.0,
                    help="tiles the predictor covers less than this sweep in full")
    ap.add_argument("--diagnose", type=int, default=-1, metavar="PAD",
                    help="at this pad, report WHERE the recall misses are")
    a = ap.parse_args()
    if not a.data:
        sys.exit("need --data or $MIDDEVAL3")
    T = a.tile
    pads = [int(v) for v in a.pads.split(",")]
    params = [None] if a.predictor == "half" else [float(v) for v in a.taus.split(",")]
    fn = predict_half if a.predictor == "half" else predict_icsg
    if a.diagnose >= 0:
        return diagnose(a, fn, T, a.diagnose)
    if a.mpad > 0:
        return halo_cost(a, fn, T)

    for param in params:
        cost = {p: [0.0, 0.0] for p in pads}
        rec = {p: [0.0, 0.0] for p in pads}
        empty, allp, adm = 0.0, 0.0, []
        for s in sorted(m3.WEIGHTS):
            d = os.path.join(a.data, f"training{a.res}", s)
            gt = m3.read_pfm(os.path.join(d, "disp0GT.pfm"))
            D = int(m3.read_calib(os.path.join(d, "calib.txt"))["ndisp"])
            plo, phi = fn(d, D, a.threads, param)
            stack = (predict_icsg.stack
                     if (a.band in ("pct", "set") and a.predictor == "icsg")
                     else None)
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
                    if a.band == "set" and stack is None and \
                            lv.mean() < a.fallback_cov:
                        # Where the predictor barely saw anything, believing its
                        # band is the mistake. Sweeping the tile costs its planes
                        # and nothing else; the diagnosis says these tiles carry
                        # half the misses while holding 6.5% of the pixels.
                        for p in pads:
                            cost[p][0] += D * n
                            cost[p][1] += D * n
                            rec[p][0] += gv.size
                            rec[p][1] += gv.size
                        continue
                    if a.band == "set" and stack is None:
                        # A point predictor still defines a SET per tile: the
                        # distinct disparities its pixels vote for. Union, not
                        # min..max, so it is judged on the same terms as ICSG.
                        pv = np.rint(plo[ys, xs][lv]).astype(int) - 1
                        pv = pv[(pv >= 0) & (pv < D)]
                        anyd = np.zeros(D, bool)
                        if pv.size:
                            anyd[pv] = True
                        k = int(anyd.sum())
                        for p in pads:
                            aa = anyd.copy()
                            for j in range(1, p + 1):
                                aa[j:] |= anyd[:-j]
                                aa[:-j] |= anyd[j:]
                            kk = int(aa.sum()) if k else D
                            cost[p][0] += kk * n
                            cost[p][1] += D * n
                            if gv.size:
                                gi = np.clip(np.rint(gv).astype(int) - 1, 0, D - 1)
                                rec[p][0] += float(aa[gi].sum() if k else gv.size)
                                rec[p][1] += gv.size
                        continue
                    if a.band == "set":
                        # A tile may legally compute a SCATTERED set of planes:
                        # each is still constant-disparity across the tile, which
                        # is all the filter requires. Pricing by interval was the
                        # stricter assumption and it is not the binding one.
                        anyd = stack[:, ys, xs].any(axis=(1, 2))
                        k = int(anyd.sum())
                        cost_pix = k if k else D
                        for p in pads:
                            # pad dilates the set rather than the interval
                            aa = anyd.copy()
                            for j in range(1, p + 1):
                                aa[j:] |= anyd[:-j]
                                aa[:-j] |= anyd[j:]
                            kk = int(aa.sum()) if k else D
                            cost[p][0] += kk * n
                            cost[p][1] += D * n
                            if gv.size:
                                gi = np.clip(np.rint(gv).astype(int) - 1, 0, D - 1)
                                rec[p][0] += float(aa[gi].sum() if k else gv.size)
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
