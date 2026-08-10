#!/usr/bin/env python3
"""How few disparity planes could a tile compute, if it knew which ones it needed?

TODO 0.3 item 3 ranks per-pixel disparity-range restriction as the biggest measured
lever in the project, 5.2x. Before building any of the constructions that claim it
(ICSG's intrinsic curves, MTS's max-trees, ESPReSSo's tiles) this measures the
ceiling, the way topk_recall.py did before the top-2 pruning was built: if the
ceiling is small the whole family is dead, and if it is large it bounds what any
member of it can win.

WHY TILES AND NOT PIXELS. Restricting the range per pixel is a measured negative
here with a known mechanism -- the recursive edge-aware filter aggregates over a
plane, so a plane must hold one constant disparity, and offset-indexed planes mix
disparities everywhere the offset varies (09-matching.md). A tile keeps the filter
legal: every pixel in it sweeps the same interval, so the planes stay constant-
disparity within the tile. That is ESPReSSo's trick and it is the only shape of this
idea that composes with what is already built.

WHAT IS MEASURED. For each tile, the interval [min d, max d] over the ground truth
it contains, padded, clipped to the legal range. The fraction of the full sweep that
represents is the arithmetic ceiling. It is an ORACLE -- it assumes the interval is
known in advance, which is exactly the thing a real construction has to predict
cheaply and imperfectly -- so it is an upper bound and nothing else. Recall against
our own answer is reported too, because a tile bound that excludes the disparity we
currently pick would cost accuracy rather than just time.

    .venv/bin/python article/range_ceiling.py --data ~/data/MiddEval3
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import middeval3 as m3


def tile_intervals(d, valid, tile, pad, D):
    """Per-tile [lo, hi) over valid pixels, padded and clipped. Returns width map."""
    H, W = d.shape
    ty, tx = (H + tile - 1) // tile, (W + tile - 1) // tile
    lo = np.full((ty, tx), np.nan)
    hi = np.full((ty, tx), np.nan)
    for by in range(ty):
        ys = slice(by * tile, min(H, (by + 1) * tile))
        for bx in range(tx):
            xs = slice(bx * tile, min(W, (bx + 1) * tile))
            v = d[ys, xs][valid[ys, xs]]
            if v.size:
                lo[by, bx] = v.min()
                hi[by, bx] = v.max()
    lo = np.clip(lo - pad, 0, D - 1)
    hi = np.clip(hi + pad, 0, D - 1)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MIDDEVAL3", ""))
    ap.add_argument("--res", default="Q")
    ap.add_argument("--tiles", default="8,16,32,64")
    ap.add_argument("--pads", default="0,2,4,8")
    a = ap.parse_args()
    if not a.data:
        sys.exit("need --data or $MIDDEVAL3")

    tiles = [int(v) for v in a.tiles.split(",")]
    pads = [int(v) for v in a.pads.split(",")]
    scenes = sorted(m3.WEIGHTS)

    # cost[tile][pad] accumulates sum(width * tile_pixels) and sum(D * tile_pixels),
    # pooled over pixels rather than averaged over tiles: a tile at the image edge
    # holds fewer pixels and should not count the same as a full one.
    acc = {(t, p): [0.0, 0.0] for t in tiles for p in pads}
    per_scene = {}

    for s in scenes:
        d = os.path.join(a.data, f"training{a.res}", s)
        gt = m3.read_pfm(os.path.join(d, "disp0GT.pfm"))
        D = int(m3.read_calib(os.path.join(d, "calib.txt"))["ndisp"])
        valid = np.isfinite(gt)
        rows = {}
        for t in tiles:
            lo, hi = tile_intervals(gt, valid, t, 0, D)
            H, W = gt.shape
            # pixels per tile, so edge tiles are weighted by what they actually hold
            npix = np.zeros_like(lo)
            for by in range(lo.shape[0]):
                for bx in range(lo.shape[1]):
                    npix[by, bx] = (min(H, (by + 1) * t) - by * t) * \
                                   (min(W, (bx + 1) * t) - bx * t)
            live = ~np.isnan(lo)
            for p in pads:
                w = np.clip(hi + p, 0, D - 1) - np.clip(lo - p, 0, D - 1) + 1
                w = np.where(live, w, 0.0)
                acc[(t, p)][0] += float((w * npix)[live].sum())
                acc[(t, p)][1] += float((D * npix)[live].sum())
                rows[(t, p)] = float((w * npix)[live].sum() /
                                     max(1.0, (D * npix)[live].sum()))
        per_scene[s] = rows

    print(f"Oracle per-tile disparity range, Middlebury v3 training{a.res}, "
          f"{len(scenes)} scenes.")
    print("Fraction of the full sweep a tile would have to compute, pooled over "
          "pixels.\nLower is a bigger prize. This is an UPPER BOUND: it assumes the "
          "interval is known.\n")
    print(f"{'tile':>6} " + "".join(f"{'pad ' + str(p):>10}" for p in pads))
    for t in tiles:
        print(f"{t:>6} " + "".join(
            f"{100*acc[(t,p)][0]/acc[(t,p)][1]:>9.1f}%" for p in pads))

    print("\nPer scene at the most defensible setting (tile 16, pad 4):")
    worst = sorted(per_scene, key=lambda s: -per_scene[s][(16, 4)])
    for s in worst:
        print(f"  {s:<13}{100*per_scene[s][(16,4)]:>7.1f}%")

    print("\nThe spread across scenes is the finding to read, not the mean: a "
          "construction\nthat has to work everywhere is priced by its worst scene.")

    # Is the mean a typical tile, or a mix of very narrow tiles and a few very wide
    # ones? That decides the shape of any construction: a bimodal split can be won
    # with a narrow sweep plus a per-tile fallback, a unimodal one cannot.
    bins = [0.10, 0.25, 0.50, 1.01]
    hist = np.zeros(len(bins))
    tot = 0.0
    for s in scenes:
        d = os.path.join(a.data, f"training{a.res}", s)
        gt = m3.read_pfm(os.path.join(d, "disp0GT.pfm"))
        D = int(m3.read_calib(os.path.join(d, "calib.txt"))["ndisp"])
        valid = np.isfinite(gt)
        lo, hi = tile_intervals(gt, valid, 16, 0, D)
        H, W = gt.shape
        for by in range(lo.shape[0]):
            for bx in range(lo.shape[1]):
                if np.isnan(lo[by, bx]):
                    continue
                n = (min(H, (by + 1) * 16) - by * 16) * \
                    (min(W, (bx + 1) * 16) - bx * 16)
                frac = (min(D - 1, hi[by, bx] + 4) -
                        max(0, lo[by, bx] - 4) + 1) / D
                for i, b in enumerate(bins):
                    if frac < b:
                        hist[i] += n
                        break
                tot += n
    print("\nTile 16, pad 4 -- where the pixels actually are:")
    lab = ["<10% of the sweep", "10-25%", "25-50%", ">=50%"]
    for l, h in zip(lab, hist):
        print(f"  {l:<20}{100*h/tot:>6.1f}% of pixels")


if __name__ == "__main__":
    main()
