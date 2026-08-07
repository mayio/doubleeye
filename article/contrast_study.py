#!/usr/bin/env python3
"""Is local contrast useful to the matcher? Measured on eight real scenes.

The C++ matcher on the Jetson already computes local_std per keypoint and carries
it around, but only uses it as a detector gate at 2.0, essentially the sensor noise
floor. It never enters the score. This asks whether it should, in three stages:

  1. Does local contrast predict whether a match is correct at all? If not, stop.
  2. Does gating on it help? Dropping low-contrast keypoints removes matches, so
     the question is whether it removes more wrong ones than right ones.
  3. Does it help as a score term? Census bits come from sign comparisons against
     the centre pixel. Where the window is nearly flat those comparisons are
     decided by noise, so the descriptor carries less information than its Hamming
     distance suggests. Shrinking the score toward chance in proportion to
     reliability is the cheap fix, and it is the same shape as the calibrated-score
     idea, without needing a trained model.

Reliability weight used in stage 3, with c the local standard deviation over the
same 7x7 window Census uses:

    r(c) = c / (c + c0)          c0 a constant on the order of the noise level
    s'   = sqrt(r_l * r_r) * desc  -  w_y * dy^2

so a flat window pulls the descriptor term toward 0, which is "no better than
chance", which is exactly where lambda and gamma will reject it. One multiply per
edge, plus one per-keypoint reliability computed once. That is Jetson-affordable.

Eight scenes, because a one-scene answer to a small-effect question is not an
answer -- see the ordering factor.

  python contrast_study.py
"""

import json

import numpy as np
from scipy.ndimage import uniform_filter

import masda_stereo as ms
import masda_middlebury as mb
import ordering_real as orr

SCENES = ("teddy", "cones") + orr.SCENES_2005
CENSUS_WIN = 7          # must match census(half_w=3, half_h=3)
TOL = 1.0
LAM = GAM = -0.1


def local_std(img, win=CENSUS_WIN):
    """Standard deviation over a win x win window, per pixel."""
    f = img.astype(np.float64)
    m = uniform_filter(f, win, mode="nearest")
    m2 = uniform_filter(f * f, win, mode="nearest")
    return np.sqrt(np.maximum(m2 - m * m, 0.0)).astype(np.float32)


def scene_data(name):
    left, right, gt, known, _ = orr.load_scene(name)
    H, W = left.shape
    ms.H, ms.W = H, W
    dmax = 60.0 if name in ("teddy", "cones") else 80.0
    cl, cr = local_std(left), local_std(right)
    return dict(name=name, left=left, right=right, gt=gt, known=known,
                dmax=dmax, cl=cl, cr=cr, H=H, W=W)


def detect_all(d, min_contrast=None):
    """Detect in both images, optionally dropping low-contrast keypoints."""
    pl, _ = ms.detect(d["left"])
    pr, _ = ms.detect(d["right"])
    if min_contrast is not None:
        pl = pl[contrast_at(d["cl"], pl) >= min_contrast]
        pr = pr[contrast_at(d["cr"], pr) >= min_contrast]
    return pl, pr


def contrast_at(cmap, pts):
    H, W = cmap.shape
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    return cmap[y, x]


def solve(d, pl, pr, weight_c0=None):
    """Build the problem and run sparse MASDA, optionally reliability-weighted."""
    dl = ms.census(d["left"], pl)
    dr = ms.census(d["right"], pr)
    S = ms.build_problem(pl, dl, pr, dr, dmin=1.0, dmax=d["dmax"])
    if weight_c0 is not None:
        # Recover the descriptor term, shrink it, put the y-penalty back.
        rl = contrast_at(d["cl"], pl); rl = rl / (rl + weight_c0)
        rr = contrast_at(d["cr"], pr); rr = rr / (rr + weight_c0)
        w = np.sqrt(np.outer(rl, rr)).astype(np.float32)
        dy = (pl[:, 1][:, None] - pr[None, :, 1]) / 1.0
        pen = (dy ** 2).astype(np.float32)          # w_y = 1, sigma_y = 1
        finite = np.isfinite(S)
        desc = np.where(finite, S + pen, np.nan)
        S = np.where(finite, w * desc - pen, -np.inf).astype(np.float32)
    m, n = S.shape
    ei, ej, se = ms.to_edges(S)
    a, _ = ms.masda_sparse(ei, ej, se, m, n, LAM, GAM)
    return S, a


def score(d, pl, pr, a):
    return mb.evaluate_real(a, pl, pr, d["gt"], d["known"], TOL)


# ---------------------------------------------------------------------------

def stage1_predictive(data):
    """Pool matches across scenes and bin by the left keypoint's local contrast."""
    print("\n" + "=" * 74)
    print("STAGE 1  Does local contrast predict correctness?")
    print("=" * 74)
    con, ok = [], []
    for d in data:
        pl, pr = detect_all(d)
        _, a = solve(d, pl, pr)
        H, W = d["H"], d["W"]
        xi = np.clip(np.round(pl[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.round(pl[:, 1]).astype(int), 0, H - 1)
        kk, kg = d["known"][yi, xi], d["gt"][yi, xi]
        c = contrast_at(d["cl"], pl)
        for i, j in a.items():
            if not kk[i]:
                continue
            con.append(c[i])
            ok.append(abs((pl[i, 0] - pr[j, 0]) - kg[i]) <= TOL)
    con, ok = np.asarray(con), np.asarray(ok, bool)
    print(f"pooled scorable matches: {len(con)}")
    qs = np.quantile(con, np.linspace(0, 1, 9))
    print(f"{'contrast bin':<22}{'n':>7}{'precision':>11}")
    for lo, hi in zip(qs[:-1], qs[1:]):
        sel = (con >= lo) & (con <= hi if hi == qs[-1] else con < hi)
        if sel.sum():
            print(f"  [{lo:6.1f}, {hi:6.1f})      {sel.sum():>5}"
                  f"{ok[sel].mean():>11.3f}")
    from scipy import stats
    r = stats.pointbiserialr(ok, con)
    print(f"point-biserial correlation contrast vs correct: r={r[0]:+.3f} "
          f"p={r[1]:.2e}")
    lo_q, hi_q = np.quantile(con, [0.25, 0.75])
    print(f"precision in bottom quartile of contrast: {ok[con <= lo_q].mean():.3f}")
    print(f"precision in top quartile of contrast:    {ok[con >= hi_q].mean():.3f}")
    return dict(r=float(r[0]), p=float(r[1]),
                prec_low=float(ok[con <= lo_q].mean()),
                prec_high=float(ok[con >= hi_q].mean()),
                q25=float(lo_q), q75=float(hi_q))


def stage2_gate(data, thresholds=(0, 3, 5, 8, 12, 18)):
    """Gate keypoints on contrast. Does it remove more wrong than right?"""
    print("\n" + "=" * 74)
    print("STAGE 2  Gating the detector on local contrast")
    print("=" * 74)
    print(f"{'min contrast':<14}{'correct':>9}{'matches':>9}{'prec':>8}"
          f"{'recall':>8}{'kp kept':>9}")
    out = {}
    base = None
    for t in thresholds:
        tot = dict(tp=0, matches=0, matchable=0, kp=0, kp0=0)
        for d in data:
            pl, pr = detect_all(d, None if t == 0 else t)
            pl0, _ = detect_all(d)
            _, a = solve(d, pl, pr)
            e = score(d, pl, pr, a)
            tot["tp"] += e["tp"]; tot["matches"] += e["scorable"]
            tot["matchable"] += e["matchable"]; tot["kp"] += len(pl)
            tot["kp0"] += len(pl0)
        prec = tot["tp"] / max(1, tot["matches"])
        rec = tot["tp"] / max(1, tot["matchable"])
        out[t] = dict(correct=tot["tp"], matches=tot["matches"], prec=prec,
                      recall=rec, kp_frac=tot["kp"] / max(1, tot["kp0"]))
        if base is None:
            base = tot["tp"]
        print(f"{t:<14}{tot['tp']:>9}{tot['matches']:>9}{prec:>8.3f}"
              f"{rec:>8.3f}{100*out[t]['kp_frac']:>8.0f}%")
    return out


def stage3_weight(data, c0s=(None, 2.0, 5.0, 10.0, 20.0)):
    """Reliability-weight the descriptor term instead of gating."""
    print("\n" + "=" * 74)
    print("STAGE 3  Reliability weighting  s' = sqrt(r_l r_r) * desc - w_y dy^2")
    print("=" * 74)
    print(f"{'c0':<10}{'correct':>9}{'matches':>9}{'prec':>8}{'recall':>8}"
          f"{'vs baseline':>13}")
    out = {}
    base = None
    per_scene = {}
    for c0 in c0s:
        tot = dict(tp=0, matches=0, matchable=0)
        ps = {}
        for d in data:
            pl, pr = detect_all(d)
            _, a = solve(d, pl, pr, weight_c0=c0)
            e = score(d, pl, pr, a)
            tot["tp"] += e["tp"]; tot["matches"] += e["scorable"]
            tot["matchable"] += e["matchable"]
            ps[d["name"]] = e["tp"]
        if base is None:
            base = tot["tp"]
        prec = tot["tp"] / max(1, tot["matches"])
        key = "off" if c0 is None else f"{c0}"
        out[key] = dict(correct=tot["tp"], matches=tot["matches"], prec=prec,
                        recall=tot["tp"] / max(1, tot["matchable"]))
        per_scene[key] = ps
        print(f"{key:<10}{tot['tp']:>9}{tot['matches']:>9}{prec:>8.3f}"
              f"{out[key]['recall']:>8.3f}{tot['tp']-base:>+13d}")
    print("\nper scene, correct matches:")
    names = [d["name"] for d in data]
    print("  " + "scene".ljust(10) + "".join(k.rjust(9) for k in per_scene))
    for nm in names:
        print("  " + nm.ljust(10) +
              "".join(str(per_scene[k][nm]).rjust(9) for k in per_scene))
    return out, per_scene


def main():
    data = [scene_data(s) for s in SCENES]
    res = {}
    res["stage1"] = stage1_predictive(data)
    res["stage2"] = stage2_gate(data)
    res["stage3"], res["stage3_per_scene"] = stage3_weight(data)
    json.dump(res, open("contrast_study.json", "w"), indent=1, default=float)
    print("\nwrote contrast_study.json")


if __name__ == "__main__":
    main()
