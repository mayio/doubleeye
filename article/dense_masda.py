#!/usr/bin/env python3
"""MASDA as a dense stereo matcher, and an honest runtime comparison.

Everything so far ran MASDA over keypoints. This runs it over *every pixel*, which
is possible because the problem decomposes: correspondences lie on the same image
row, so rows share no left or right index. The whole image is therefore ONE
assignment problem whose exclusivity constraints happen to couple only within rows,
and the same edge-list solver handles it unchanged.

Edges are H x W x ndisp. On a 450x375 pair with 60 disparities that is 10.1M, which
is large but linear -- exactly the property the sparse formulation was built for.

What this measures:

  - Whether uniqueness alone produces a usable dense depth map.
  - Where MASDA sits against SGM on the same scenes: accuracy AND runtime, since
    a dense matcher that is ten times slower is a different proposition from one
    that is competitive.

Both numbers are reported for the same NumPy implementation against OpenCV's SGM,
which is compiled C++ with SIMD. That comparison is unfair to MASDA by a large and
unknown constant, and the text says so rather than pretending otherwise.

  python dense_masda.py
"""

import time

import cv2
import numpy as np

import masda_stereo as ms
import ordering_real as orr
import dense_baseline as db

TOL = 1.0


def dense_problem(left, right, dmin, dmax, census_cfg=(3, 3)):
    """Every pixel against every candidate on its row. Returns an edge list.

    Left index is y*W + x, right index is y*W + xr, so rows are automatically
    disjoint in both and the exclusivity constraints never couple across rows.
    """
    H, W = left.shape
    hw, hh = census_cfg
    d = np.arange(int(dmin), int(dmax) + 1)
    xs = np.arange(hw, W - hw)
    yy = np.arange(hh, H - hh)

    # Census for the whole image at once, as a (H, W) uint64 array.
    def census_image(img):
        f = img.astype(np.int32)
        out = np.zeros((H, W), np.uint64)
        bit = 0
        for dy in range(-hh, hh + 1):
            for dx in range(-hw, hw + 1):
                if dx == 0 and dy == 0:
                    continue
                sh = np.roll(np.roll(f, -dy, 0), -dx, 1)
                out |= ((sh < f).astype(np.uint64) << np.uint64(bit))
                bit += 1
        return out

    cl, cr = census_image(left), census_image(right)

    ei_l, ej_l, se_l = [], [], []
    for dd in d:
        x = xs[xs - dd >= hw]
        if len(x) == 0:
            continue
        gy = np.repeat(yy, len(x))
        gx = np.tile(x, len(yy))
        h = ms.hamming(cl[gy, gx], cr[gy, gx - dd])
        ei_l.append((gy * W + gx).astype(np.int64))
        ej_l.append((gy * W + (gx - dd)).astype(np.int64))
        se_l.append(((24.0 - h) / 24.0).astype(np.float32))
    return (np.concatenate(ei_l), np.concatenate(ej_l), np.concatenate(se_l),
            H * W)


def run(name, iters=12, gate=0.0):
    left, right, gt, known, _ = orr.load_scene(name)
    H, W = left.shape
    dmax = 60.0 if name in ("teddy", "cones") else 80.0

    t0 = time.perf_counter()
    ei, ej, se, n_nodes = dense_problem(left, right, 1, dmax)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    a, _ = ms.masda_sparse(ei, ej, se, n_nodes, n_nodes, -0.1, -0.1, iters=iters)
    t_solve = time.perf_counter() - t0

    # Per-left-pixel margin, for gating, from the same edge list.
    b1 = np.full(n_nodes, -np.inf, np.float32)
    b2 = np.full(n_nodes, -np.inf, np.float32)
    order = np.argsort(-se)
    np.maximum.at(b1, ei, se)
    below = se < b1[ei]
    if below.any():
        np.maximum.at(b2, ei[below], se[below])
    margin = b1 - np.where(np.isneginf(b2), -0.1, np.maximum(b2, -0.1))

    disp = np.full((H, W), np.nan, np.float32)
    for i, j in a.items():
        if gate > 0 and margin[i] < gate:
            continue
        y, x = divmod(i, W)
        disp[y, x] = x - (j % W)

    valid = known & np.isfinite(disp)
    bad = float((np.abs(disp[valid] - gt[valid]) > TOL).mean()) if valid.any() else 1.0
    cov = float(np.isfinite(disp)[known].mean())
    return dict(scene=name, disp=disp, gt=gt, known=known, left=left,
                edges=len(se), t_build=t_build, t_solve=t_solve,
                bad=bad, coverage=cov)


def main():
    rows = []
    print(f"{'scene':<9}{'edges':>11}{'build s':>9}{'solve s':>9}"
          f"{'coverage':>10}{'bad-1.0':>9}{'|':>3}{'SGM cov':>9}{'SGM bad':>9}{'SGM ms':>8}")
    for name in ("teddy", "cones"):
        r = run(name)
        d = db.run(name)
        rows.append((r, d))
        print(f"{name:<9}{r['edges']:>11,}{r['t_build']:>9.1f}{r['t_solve']:>9.1f}"
              f"{100*r['coverage']:>9.1f}%{100*r['bad']:>8.1f}%{'|':>3}"
              f"{100*d['coverage']:>8.1f}%{100*d['bad_all']:>8.1f}%{d['t_dense']:>8.0f}")
    print("\nMASDA is NumPy, SGM is compiled C++ with SIMD. The runtime column is")
    print("not a like-for-like algorithmic comparison and should not be read as one.")
    return rows


def figure(rows, fname="dense_masda"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(rows), 3, figsize=(13.5, 4.1 * len(rows)))
    for k, (r, d) in enumerate(rows):
        gt, known = r["gt"], r["known"]
        vmin, vmax = np.percentile(gt[known], (2, 98))
        ax[k, 0].imshow(np.where(known, gt, np.nan), cmap="viridis", vmin=vmin, vmax=vmax)
        ax[k, 0].set_title(f"{r['scene']}: ground truth")
        ax[k, 1].imshow(r["disp"], cmap="viridis", vmin=vmin, vmax=vmax)
        ax[k, 1].set_title(f"dense MASDA: {100*r['coverage']:.0f}% filled, "
                           f"bad-1.0 {100*r['bad']:.0f}%")
        ax[k, 2].imshow(d["dense"], cmap="viridis", vmin=vmin, vmax=vmax)
        ax[k, 2].set_title(f"SGM: {100*d['coverage']:.0f}% filled, "
                           f"bad-1.0 {100*d['bad_all']:.0f}%")
        for c in range(3):
            ax[k, c].set_xticks([]); ax[k, c].set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{ms.FIG}/{fname}.png", dpi=115)
    plt.close(fig)
    print(f"wrote {ms.FIG}/{fname}.png")


if __name__ == "__main__":
    figure(main())
