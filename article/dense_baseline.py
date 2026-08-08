#!/usr/bin/env python3
"""A dense stereo baseline on the same ground truth, for comparison.

Two reasons this exists.

First, "can we produce a dense depth image" deserves a yes with a picture rather
than an argument. Semi-global matching produces one, it is the family the D4 ASIC
belongs to, and OpenCV ships a good implementation, so there is no reason to guess
what dense would look like on these pairs.

Second, it makes the sparse-versus-dense comparison quantitative instead of
rhetorical. The article says sparse matching is for feeding geometry and dense
stereo is the right tool for a depth image. That is only worth saying if the
numbers behind it are known.

The comparison has to be set up carefully, because the two methods answer different
questions and a naive table would be meaningless:

  - Dense SGM is scored the Middlebury way: the fraction of pixels with known
    ground truth whose error exceeds 1 px ("bad 1.0"), over the whole image.
  - MASDA is scored at its own matched keypoints, which is a different and much
    smaller denominator.
  - So SGM is ALSO scored at exactly MASDA's keypoints. That is the only
    apples-to-apples number here: same points, same ground truth, two methods.

  python dense_baseline.py
"""

import time

import cv2
import numpy as np

import masda_stereo as ms
import masda_middlebury as mb
import ordering_real as orr

TOL = 1.0


def sgbm_disparity(left, right, ndisp):
    """OpenCV SGBM, tuned only to the search range the scene actually needs.

    Block size and the P1/P2 smoothness weights are OpenCV's documented defaults
    for 8-bit single-channel input. They are not tuned per scene: tuning the dense
    baseline while leaving the sparse one alone would make the comparison
    meaningless in the other direction.
    """
    ndisp = int(np.ceil(ndisp / 16.0) * 16)      # SGBM requires a multiple of 16
    m = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=ndisp, blockSize=5,
        P1=8 * 5 * 5, P2=32 * 5 * 5,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    l8 = np.clip(left, 0, 255).astype(np.uint8)
    r8 = np.clip(right, 0, 255).astype(np.uint8)
    d = m.compute(l8, r8).astype(np.float32) / 16.0
    d[d <= 0] = np.nan                            # SGBM marks invalid as negative
    return d


def run(name):
    left, right, gt, known, _ = orr.load_scene(name)
    H, W = left.shape
    ms.H, ms.W = H, W
    dmax = 60.0 if name in ("teddy", "cones") else 80.0

    t = time.perf_counter()
    dense = sgbm_disparity(left, right, dmax)
    t_dense = (time.perf_counter() - t) * 1e3

    # MASDA, exactly as the article runs it.
    pl, _ = ms.detect(left)
    pr, _ = ms.detect(right)
    dl, dr = ms.census(left, pl), ms.census(right, pr)
    S = ms.build_problem(pl, dl, pr, dr, dmin=1.0, dmax=dmax)
    m, n = S.shape
    ei, ej, se = ms.to_edges(S)
    t = time.perf_counter()
    a, _ = ms.masda_sparse(ei, ej, se, m, n, -0.1, -0.1)
    t_sparse = (time.perf_counter() - t) * 1e3

    xi = np.clip(np.round(pl[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(pl[:, 1]).astype(int), 0, H - 1)
    kk, kg = known[yi, xi], gt[yi, xi]

    # Dense, scored over the whole image: the standard Middlebury bad-1.0.
    valid = known & np.isfinite(dense)
    bad_all = float((np.abs(dense[valid] - gt[valid]) > TOL).mean())
    coverage = float(np.isfinite(dense)[known].mean())

    # Both methods at MASDA's keypoints: the only like-for-like comparison.
    sp_ok = sp_n = dn_ok = dn_n = 0
    for i, j in a.items():
        if not kk[i]:
            continue
        sp_n += 1
        if abs((pl[i, 0] - pr[j, 0]) - kg[i]) <= TOL:
            sp_ok += 1
        dv = dense[yi[i], xi[i]]
        if np.isfinite(dv):
            dn_n += 1
            if abs(dv - kg[i]) <= TOL:
                dn_ok += 1
    return dict(scene=name, dense=dense, gt=gt, known=known, left=left,
                pl=pl, pr=pr, a=a, xi=xi, yi=yi, kk=kk, kg=kg,
                bad_all=bad_all, coverage=coverage,
                t_dense=t_dense, t_sparse=t_sparse,
                sp_prec=sp_ok / max(1, sp_n), sp_n=sp_n,
                dn_prec=dn_ok / max(1, dn_n), dn_n=dn_n)


def main():
    scenes = ("teddy", "cones") + orr.SCENES_2005
    rows = [run(s) for s in scenes]

    print(f"\n{'scene':<10}{'SGM cover':>11}{'SGM bad1.0':>12}{'SGM ms':>9}"
          f"{'--':>4}{'at MASDA pts':>14}{'SGM':>8}{'MASDA':>8}{'MASDA ms':>10}")
    for r in rows:
        print(f"{r['scene']:<10}{100*r['coverage']:>10.1f}%{100*r['bad_all']:>11.1f}%"
              f"{r['t_dense']:>9.0f}{'':>4}{r['sp_n']:>14}"
              f"{r['dn_prec']:>8.3f}{r['sp_prec']:>8.3f}{r['t_sparse']:>10.1f}")
    cov = np.mean([r["coverage"] for r in rows])
    bad = np.mean([r["bad_all"] for r in rows])
    dn = sum(r["dn_prec"] * r["dn_n"] for r in rows) / sum(r["dn_n"] for r in rows)
    sp = sum(r["sp_prec"] * r["sp_n"] for r in rows) / sum(r["sp_n"] for r in rows)
    print(f"\npooled over {len(rows)} scenes:")
    print(f"  dense SGM: {100*cov:.1f}% of known-GT pixels have a value, "
          f"bad-1.0 {100*bad:.1f}%, {np.mean([r['t_dense'] for r in rows]):.0f} ms/pair")
    print(f"  at MASDA's keypoints: SGM {dn:.3f} vs MASDA {sp:.3f} precision")
    return rows


def figure(rows, fname="dense_vs_sparse"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = [r for r in rows if r["scene"] in ("teddy", "cones")]
    fig, ax = plt.subplots(len(sel), 3, figsize=(13.5, 4.1 * len(sel)))
    for k, r in enumerate(sel):
        gt, known = r["gt"], r["known"]
        vmin, vmax = np.percentile(gt[known], (2, 98))
        ax[k, 0].imshow(np.where(known, gt, np.nan), cmap="viridis",
                        vmin=vmin, vmax=vmax)
        ax[k, 0].set_title(f"{r['scene']}: ground truth")
        ax[k, 1].imshow(r["dense"], cmap="viridis", vmin=vmin, vmax=vmax)
        ax[k, 1].set_title(f"dense SGM: {100*r['coverage']:.0f}% filled, "
                           f"bad-1.0 {100*r['bad_all']:.0f}%")
        ax[k, 2].imshow(r["left"], cmap="gray", alpha=0.30)
        X = [r["pl"][i, 0] for i in r["a"]]
        Y = [r["pl"][i, 1] for i in r["a"]]
        D = [r["pl"][i, 0] - r["pr"][j, 0] for i, j in r["a"].items()]
        ax[k, 2].scatter(X, Y, c=D, s=12, cmap="viridis", vmin=vmin, vmax=vmax)
        ax[k, 2].set_xlim(0, gt.shape[1]); ax[k, 2].set_ylim(gt.shape[0], 0)
        ax[k, 2].set_title(f"MASDA: {len(D)} matches")
        for c in range(3):
            ax[k, c].set_xticks([]); ax[k, c].set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{ms.FIG}/{fname}.png", dpi=115)
    plt.close(fig)
    print(f"wrote {ms.FIG}/{fname}.png")


if __name__ == "__main__":
    figure(main())
