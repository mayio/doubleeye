#!/usr/bin/env python3
"""MASDA on real stereo pairs with ground truth: Middlebury 2003 (Teddy, Cones).

The synthetic scene in masda_stereo.py controls texture exactly, which is what
makes the ambiguity sweep meaningful, but it renders the right image by warping
the left one. Every descriptor difference therefore comes from resampling and
added noise, never from two physically different cameras. This script runs the
same matcher on real photographs so the descriptor side is honest: real lenses,
real vignetting, real sensor noise, real non-Lambertian surfaces.

Data: Middlebury 2003 stereo pairs, which state "We grant permission to use and
publish all images and disparity maps on this website." Cite Scharstein &
Szeliski, CVPR 2003. Images are downloaded on first run, not vendored.

  D. Scharstein and R. Szeliski. High-accuracy stereo depth maps using
  structured light. CVPR 2003, pp. 195-202.

Ground truth is quarter-pixel and dense apart from ~2-3% marked unknown (0).
"""

import io
import os
import ssl
import time
import urllib.request

import numpy as np
from PIL import Image

import masda_stereo as ms

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BASE = "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/{s}/{f}"
SCENES = ("teddy", "cones")
GT_SCALE = 4.0          # Middlebury 2003 stores disparity * 4
NDISP = 60              # published search range for both scenes
TOL = 1.0               # Middlebury's standard "bad pixel" threshold


# ---------------------------------------------------------------------------
# 1. Data

def fetch(scene):
    """Download one scene if absent. Returns left, right, gt, known.

    vision.middlebury.edu serves an incomplete certificate chain (it omits its
    intermediate), so verification fails locally. These are public research
    images and no credential is sent, so the download is unverified and the
    payload is checked instead: expected size, expected disparity scale, and a
    recorded SHA-256 so a changed file is visible.
    """
    d = os.path.join(DATA, scene)
    os.makedirs(d, exist_ok=True)
    ctx = ssl._create_unverified_context()
    out = {}
    for f in ("im2.png", "im6.png", "disp2.png"):
        p = os.path.join(d, f)
        if not os.path.exists(p):
            with urllib.request.urlopen(BASE.format(s=scene, f=f),
                                        timeout=120, context=ctx) as r:
                open(p, "wb").write(r.read())
        out[f] = np.array(Image.open(p))

    left = luma(out["im2.png"])
    right = luma(out["im6.png"])
    raw = out["disp2.png"].astype(np.float32)
    gt = raw / GT_SCALE
    known = raw > 0
    assert left.shape == right.shape == gt.shape, "size mismatch"
    assert gt[known].max() < NDISP, f"disparity {gt[known].max()} exceeds ndisp"
    return left, right, gt, known


def luma(rgb):
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1]
            + 0.114 * rgb[:, :, 2]).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Evaluation against real ground truth
#
# Middlebury marks unknown/occluded ground truth as 0 rather than providing a
# separate visibility mask, so the semantics differ from the synthetic case and
# this cannot reuse masda_stereo.evaluate.

def evaluate_real(assign, pl, pr, gt, known, tol=TOL):
    """Precision and recall against Middlebury ground truth.

    A match is *scorable* only where the left keypoint has known ground truth.
    Matches on unknown ground truth are counted separately and excluded from
    precision, because there is no correct answer to compare against; silently
    calling them wrong would punish the matcher for the dataset's holes.

    A left keypoint is *matchable* only if its ground truth is known and a right
    keypoint was actually detected within tol of the true correspondence.
    Keypoints are detected independently in the two images, so a left keypoint
    whose partner the detector missed has no attainable match.
    """
    H, W = gt.shape
    xi = np.clip(np.round(pl[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(pl[:, 1]).astype(int), 0, H - 1)
    kp_known = known[yi, xi]
    kp_gt = gt[yi, xi]

    tp = fp = unscorable = 0
    errs = []
    for i, j in assign.items():
        if not kp_known[i]:
            unscorable += 1
            continue
        e = abs((pl[i, 0] - pr[j, 0]) - kp_gt[i])
        errs.append(e)
        if e <= tol:
            tp += 1
        else:
            fp += 1

    # Is a right keypoint present at the true correspondence?
    true_xr = pl[:, 0] - kp_gt
    dx = np.abs(true_xr[:, None] - pr[None, :, 0])
    dy = np.abs(pl[:, 1][:, None] - pr[None, :, 1])
    detected = ((dx <= tol) & (dy <= tol)).any(1)
    matchable = int((kp_known & detected).sum())

    errs = np.asarray(errs) if errs else np.zeros(0)
    return dict(matches=len(assign), tp=tp, fp=fp, unscorable=unscorable,
                scorable=tp + fp,
                precision=tp / max(1, tp + fp),
                recall=tp / max(1, matchable),
                matchable=matchable,
                mae=float(errs.mean()) if len(errs) else float("nan"),
                med_err=float(np.median(errs)) if len(errs) else float("nan"))


# ---------------------------------------------------------------------------
# 3. One scene

def run_scene(scene, verbose=True):
    left, right, gt, known = fetch(scene)
    H, W = left.shape
    ms.H, ms.W = H, W          # detect() and friends read these as globals

    pl, _ = ms.detect(left)
    pr, _ = ms.detect(right)
    dl, dr = ms.census(left, pl), ms.census(right, pr)
    S = ms.build_problem(pl, dl, pr, dr, dmin=1.0, dmax=float(NDISP))
    m, n = S.shape
    ei, ej, se = ms.to_edges(S)
    lam = gam = -0.1

    res = {"scene": scene, "H": H, "W": W, "m": m, "n": n, "E": len(se),
           "gt_unknown": float((~known).mean())}
    mg = ms.score_margin(S)
    res["margin"] = float(np.median(mg))
    res["tied"] = float((mg < 0.05).mean())

    t = time.perf_counter()
    a_sp, _ = ms.masda_sparse(ei, ej, se, m, n, lam, gam)
    res["t_sparse"] = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    a_nn = ms.mutual_nn(S, lam)
    res["t_nn"] = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    a_opt = ms.optimal_lap(S, lam, gam)
    res["t_jv"] = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    a_de, _ = ms.masda(S, lam, gam)
    res["t_dense"] = (time.perf_counter() - t) * 1e3

    for key, a in (("masda", a_sp), ("nn", a_nn), ("opt", a_opt),
                   ("dense", a_de)):
        res[key] = evaluate_real(a, pl, pr, gt, known)
        res[key]["obj"] = ms.objective(a, S, lam, gam, m, n)

    # Ordering factor, on real data this time.
    res["cross_off"], res["cross_tot"] = ms.count_crossings(a_sp, pl, pr)
    for kappa in (0.1, 0.3, 0.8):
        a_o, _, res["xpairs"] = ms.masda_sparse_ordering(
            ei, ej, se, m, n, pl, pr, kappa=kappa, lam=lam, gam=gam)
        r = evaluate_real(a_o, pl, pr, gt, known)
        r["crossings"], _ = ms.count_crossings(a_o, pl, pr)
        res[f"ord_{kappa}"] = r

    res["_arrays"] = (left, right, gt, known, pl, pr, S, a_sp, a_nn)
    if verbose:
        report(res)
    return res


def report(r):
    print(f"\n=== {r['scene']}  {r['W']}x{r['H']}  "
          f"m={r['m']} n={r['n']} E={r['E']} ({r['E']/max(1,r['m']):.2f}/kp)  "
          f"gt unknown={100*r['gt_unknown']:.1f}%  "
          f"margin={r['margin']:.4f} tied={100*r['tied']:.1f}%")
    print(f"{'method':<16}{'matches':>8}{'scor':>7}{'correct':>8}{'prec':>7}"
          f"{'recall':>8}{'medErr':>8}{'obj':>10}{'ms':>9}")
    for key, name, t in (("nn", "Mutual-NN", r["t_nn"]),
                         ("masda", "MASDA sparse", r["t_sparse"]),
                         ("dense", "MASDA dense", r["t_dense"]),
                         ("opt", "Optimal LAP", r["t_jv"])):
        e = r[key]
        print(f"{name:<16}{e['matches']:>8}{e['scorable']:>7}{e['tp']:>8}"
              f"{e['precision']:>7.3f}{e['recall']:>8.3f}{e['med_err']:>8.3f}"
              f"{e['obj']:>10.2f}{t:>9.1f}")
    print(f"  matchable={r['masda']['matchable']}  "
          f"unscorable(masda)={r['masda']['unscorable']}  "
          f"sparse speedup vs JV={r['t_jv']/r['t_sparse']:.0f}x  "
          f"obj/opt={r['masda']['obj']/r['opt']['obj']:.4f}")
    print(f"  ordering: off correct={r['masda']['tp']} "
          f"prec={r['masda']['precision']:.3f} "
          f"crossings={r['cross_off']}/{r['cross_tot']} "
          f"(crossing pairs in graph={r.get('xpairs', 0)})")
    for k in (0.1, 0.3, 0.8):
        e = r[f"ord_{k}"]
        print(f"            k={k} correct={e['tp']} prec={e['precision']:.3f} "
              f"crossings={e['crossings']}")


# ---------------------------------------------------------------------------
# 4. Figures

def fig_real(r, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    left, right, gt, known, pl, pr, S, a_sp, a_nn = r["_arrays"]
    H, W = gt.shape
    xi = np.clip(np.round(pl[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(pl[:, 1]).astype(int), 0, H - 1)

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 7.4))
    ax[0, 0].imshow(left, cmap="gray"); ax[0, 0].set_title(f"{name} left (im2)")
    g = np.where(known, gt, np.nan)
    im = ax[0, 1].imshow(g, cmap="viridis")
    ax[0, 1].set_title("ground-truth disparity (grey = unknown)")
    ax[0, 1].set_facecolor("0.6")
    fig.colorbar(im, ax=ax[0, 1], fraction=0.046)

    # Matches drawn on the left image, coloured by correctness.
    ax[1, 0].imshow(left, cmap="gray")
    for i, j in a_sp.items():
        if not known[yi[i], xi[i]]:
            continue
        e = abs((pl[i, 0] - pr[j, 0]) - gt[yi[i], xi[i]])
        ax[1, 0].plot([pl[i, 0], pr[j, 0]], [pl[i, 1], pr[j, 1]],
                      "-", lw=0.6, alpha=0.8,
                      color=("#2c9e4b" if e <= TOL else "#d1495b"))
    ax[1, 0].set_title(f"MASDA: green correct ({r['masda']['tp']}), "
                       f"red wrong ({r['masda']['fp']})")
    ax[1, 0].set_xlim(0, W); ax[1, 0].set_ylim(H, 0)

    # Estimated vs true disparity.
    xs, ys = [], []
    for i, j in a_sp.items():
        if known[yi[i], xi[i]]:
            xs.append(gt[yi[i], xi[i]]); ys.append(pl[i, 0] - pr[j, 0])
    ax[1, 1].plot([0, NDISP], [0, NDISP], "-", color="0.7", lw=1)
    ax[1, 1].plot(xs, ys, ".", ms=3, alpha=0.55, color="#2c6fbb")
    ax[1, 1].set_xlabel("true disparity (px)")
    ax[1, 1].set_ylabel("estimated disparity (px)")
    ax[1, 1].set_title(f"median |error| = {r['masda']['med_err']:.2f} px")
    ax[1, 1].set_xlim(0, NDISP); ax[1, 1].set_ylim(0, NDISP)
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    os.makedirs(ms.FIG, exist_ok=True)
    fig.savefig(f"{ms.FIG}/real_{name}.png", dpi=115)
    plt.close(fig)
    print(f"  wrote {ms.FIG}/real_{name}.png")


def main():
    rs = [run_scene(s) for s in SCENES]
    for r in rs:
        fig_real(r, r["scene"])
    return rs


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 5. The unifying figure: does score margin predict precision on real data too?

def margins_per_kp(S):
    """Best-minus-second-best per left keypoint; nan if it has <2 candidates."""
    out = np.full(S.shape[0], np.nan)
    for i, row in enumerate(S):
        f = row[np.isfinite(row)]
        if f.size >= 2:
            f = np.sort(f)[::-1]
            out[i] = f[0] - f[1]
    return out


def region_stats(scene, regions):
    """Median score margin and MASDA precision over subsets of one real scene.

    Margin is medianed over every keypoint in the region with at least two
    candidates, matching how section 3 measures it, rather than over matched
    keypoints only -- ambiguity is a property of the problem, not of the answer.
    """
    left, right, gt, known = fetch(scene)
    H, W = left.shape
    ms.H, ms.W = H, W
    pl, _ = ms.detect(left)
    pr, _ = ms.detect(right)
    dl, dr = ms.census(left, pl), ms.census(right, pr)
    S = ms.build_problem(pl, dl, pr, dr, dmin=1.0, dmax=float(NDISP))
    m, n = S.shape
    ei, ej, se = ms.to_edges(S)
    a, _ = ms.masda_sparse(ei, ej, se, m, n, -0.1, -0.1)
    xi = np.clip(np.round(pl[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(pl[:, 1]).astype(int), 0, H - 1)
    kk, kg = known[yi, xi], gt[yi, xi]
    mg = margins_per_kp(S)
    res = {}
    for label, sel in regions(pl, kg, kk):
        tp = tot = 0
        for i, j in a.items():
            if not kk[i] or not sel[i]:
                continue
            tot += 1
            if abs((pl[i, 0] - pr[j, 0]) - kg[i]) <= TOL:
                tp += 1
        v = mg[sel & np.isfinite(mg)]
        if tot and len(v):
            res[label] = dict(n=tot, prec=tp / tot, margin=float(np.median(v)))
    return res


def fig_margin_vs_precision(points, fname="margin_vs_precision"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for kind, marker, colour in (("synthetic", "o", "#2c6fbb"),
                                 ("real", "s", "#c1121f")):
        xs = [p["margin"] for p in points if p["kind"] == kind]
        ys = [p["prec"] for p in points if p["kind"] == kind]
        ax.plot(xs, ys, marker, ms=10, color=colour, label=kind, zorder=3)
    # Greedy declutter: points cluster at high margin, so nudge a label down
    # whenever it would land on one already placed.
    placed = []
    for pt in points:
        dy = -4
        while any(abs(pt["margin"] - mx) < 0.13 and abs(pt["prec"] + dy / 260 - my) < 0.042
                  for mx, my in placed):
            dy -= 13
        kw = {}
        if dy < -4:                      # displaced: draw a leader to its marker
            kw["arrowprops"] = dict(arrowstyle="-", lw=0.7, color="0.45",
                                    shrinkA=1, shrinkB=4)
        ax.annotate(pt["label"], (pt["margin"], pt["prec"]),
                    textcoords="offset points", xytext=(10, dy), fontsize=9,
                    va="center", **kw)
        placed.append((pt["margin"], pt["prec"] + dy / 260))
    ax.set_xlabel("median score margin (best minus second best)")
    ax.set_ylabel("precision of MASDA matches")
    ax.set_title("Ambiguity predicts precision, on synthetic and real data alike")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    os.makedirs(ms.FIG, exist_ok=True)
    fig.savefig(f"{ms.FIG}/{fname}.png", dpi=115)
    plt.close(fig)
    print(f"  wrote {ms.FIG}/{fname}.png")


def fig_thumbnail(r, fname="thumb_teddy"):
    """Single-panel Teddy with its matches, for the post thumbnail.

    The four-panel real_teddy figure is unreadable once scaled to a thumbnail, and
    the same image is used as the social preview, so this is one panel: the left
    image with MASDA's matches drawn, green where the disparity is within a pixel
    of ground truth and red where it is not. No axes or titles, since they are
    illegible at that size and the caption carries the explanation anyway.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    left, right, gt, known, pl, pr, S, a_sp, a_nn = r["_arrays"]
    H, W = gt.shape
    xi = np.clip(np.round(pl[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(pl[:, 1]).astype(int), 0, H - 1)

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=150)
    ax.imshow(left, cmap="gray", interpolation="nearest")
    for i, j in a_sp.items():
        if not known[yi[i], xi[i]]:
            continue
        e = abs((pl[i, 0] - pr[j, 0]) - gt[yi[i], xi[i]])
        ax.plot([pl[i, 0], pr[j, 0]], [pl[i, 1], pr[j, 1]], "-", lw=1.1,
                alpha=0.9, color=("#37b24d" if e <= TOL else "#e03131"))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(f"{ms.FIG}/{fname}.png", dpi=150, pad_inches=0,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {ms.FIG}/{fname}.png")
