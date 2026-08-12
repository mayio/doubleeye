#!/usr/bin/env python3
"""How well does a per-pixel confidence rank the matcher's mistakes?

The cloud has no notion of how much any single point is worth. Temporal fusion
wants to weight a point rather than hold a hard vote, and an occupancy grid
consumes a probability. This is step 1 of doc/TODO.md 0.55: measure the cues that
are ALREADY in the matcher, before adding any.

THE METRIC. A confidence is useful if removing the least confident pixels removes
mostly wrong ones. Sort every answered pixel by confidence, keep the top fraction
q, and record the error rate among what is kept. That is the sparsification curve
e(q), and its area

    AUC = (1/N) * sum over the sampled q of e(q),      q = 1.00, 0.99, ... 0.01

is one number for the whole curve, lower being better. Two references make it
readable, and both are computed here:

  * the ORACLE curve, which sorts by the true error and therefore removes every
    wrong pixel first. Its AUC is the floor no measure can beat.
  * a RANDOM confidence, whose curve is flat at the overall error rate. Its AUC is
    the no-skill ceiling, and if a measure does not beat it the measure is noise.

The number to read is AUC - AUC_oracle. It is zero for a perfect ranking and
about (e_all - AUC_oracle) for a useless one.

WHY --min-margin 0. The shipping pipeline gates on the score margin. Scoring a
confidence on the pixels that gate already kept would ask "how well does the
margin rank what the margin did not remove", which is circular and flatters every
measure built from the same two scores. Every run here is ungated, so the
population is every pixel the matcher answered at all.

    .venv/bin/python article/confidence.py            # the table
    .venv/bin/python article/confidence.py --check    # prove the harness can fail

The measures, in the taxonomy of Hu and Mordohai (TPAMI 2012) and the 52-measure
re-evaluation by Poggi, Tosi and Mattoccia (TPAMI 2021):

  msm     maximum similarity: the winning aggregated score alone.
  mmn     maximum margin, naive: best minus second. This is what ships as
          --min-margin, under its name in the literature.
  pkrn    peak ratio, naive: the two scores as a ratio rather than a difference.
          Scores here are similarities in (-inf, 1], so the cost is c = 1 - s and
          the ratio is c2/c1 >= 1.
  ammn,   the same two, averaged over a (2r+1)x(2r+1) window. Poggi's central
  apkr    finding is that local aggregation is what separates a good hand-crafted
          measure from a poor one, and APKR is one of the top four.

Every measure is read from `de_dense --out-conf`, which writes the two candidates
per pixel that the solver ranked. Nothing here re-runs matching.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
from scipy.ndimage import uniform_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data", "c_bench")
BIN = os.path.join(ROOT, "core", "build", "de_dense")
TMP = os.path.join(HERE, "data", ".conf_tmp")

TOL = 1.0              # a pixel is correct within one disparity, as everywhere else
                       # here. --tol 0.25 is what Middlebury v3 asks of a quarter-
                       # resolution result, and it grades a different failure.
LAMBDA = -0.1          # cfg.lambda: the floor the margin measures against
DMIN = 1               # cfg.dmin, and the offset between a volume index and a disparity
LRC_TOL = 1.0          # how far the reverse match may disagree and still agree
QUANTILES = np.linspace(1.0, 0.01, 100)
OPERATING = (0.95, 0.90, 0.80, 0.60, 0.40)   # kept fractions a gate is set to


def scenes():
    return sorted(d for d in os.listdir(DATA)
                  if os.path.isfile(os.path.join(DATA, d, "meta.txt")))


def run_scene(name, extra=(), lrc=False):
    """Match one scene ungated.

    Returns (disparity, ground truth, s1, s2, d1, d2, lrc) where `lrc` is the
    left-right consistency residual in disparities, or None when not asked for.
    """
    d = os.path.join(DATA, name)
    W, H, dmax = (int(v) for v in open(os.path.join(d, "meta.txt")).read().split())
    os.makedirs(TMP, exist_ok=True)
    dp = os.path.join(TMP, f"{name}.f32")
    cp = os.path.join(TMP, f"{name}.conf.f32")
    cmd = [BIN, os.path.join(d, "left.y8"), os.path.join(d, "right.y8"),
           str(W), str(H), "--dmax", str(dmax), "--threads", "4",
           "--min-margin", "0", "--out", dp, "--out-conf", cp] + list(extra)
    if lrc:
        # The matcher computes the reverse match itself, on the shipping path, as a
        # second running maximum over the scores it already has. Measured to leave the
        # disparity map bit-identical on all eight scenes.
        cmd += ["--lrc"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        # Rule 1: a silent producer failure must not read as a confident answer.
        sys.exit(f"de_dense failed on {name} (exit {p.returncode}):\n{p.stderr}")
    disp = np.fromfile(dp, np.float32).reshape(H, W)
    gt = np.fromfile(os.path.join(d, "disp.f32"), np.float32).reshape(H, W)
    s1, s2, d1, d2, resid, resid2 = np.fromfile(cp, np.float32).reshape(6, H, W)
    if not lrc:
        resid = resid2
    return disp, gt, s1, s2, d1, d2, resid


def rank01(v):
    """Where each value sits in its own population, from 0 (lowest) to 1 (highest).

    Two confidences on different scales cannot be added. Replacing each by its rank
    puts both on [0, 1] with no fitting and no assumption about their distributions,
    which is enough to answer "do these two see different mistakes?" -- the question
    that decides whether a fitted combination is worth building.
    """
    out = np.empty(v.size, np.float64)
    out[np.argsort(v, kind="stable")] = np.arange(v.size)
    return out / max(1, v.size - 1)


def measures(s1, s2, d1, radius):
    """Every confidence this step evaluates, as full planes. Higher is more confident.

    A pixel with no second candidate has no runner-up to be beaten by, so its
    margin measures against the same floor `lambda` the solver uses, and its peak
    ratio is the largest any pixel reaches rather than undefined.
    """
    have1 = s1 > -1e29
    have2 = s2 > -1e29
    alt = np.where(have2, s2, LAMBDA)
    mmn = np.where(have1, s1 - np.maximum(alt, LAMBDA), 0.0)

    # Similarity to cost, so the ratio is of two non-negative numbers. c1 <= c2 by
    # construction, and c1 = 0 only for a perfect 48-bit Census agreement over the
    # whole window, which the aggregation makes unreachable in practice; the floor
    # keeps the division defined rather than papering over a real value.
    c1 = np.maximum(1.0 - s1, 1e-6)
    c2 = np.maximum(1.0 - np.where(have2, s2, LAMBDA), 1e-6)
    pkrn = np.where(have1, c2 / c1, 1.0)

    out = {"msm": np.where(have1, s1, -1e3), "mmn": mmn, "pkrn": pkrn}
    # Local aggregation, over the pixels that HAVE a value: a plain box filter
    # would average the missing ones in as zero and report a neighbourhood of
    # holes as unusually low confidence, which is a different statement.
    k = 2 * radius + 1
    w = uniform_filter(have1.astype(np.float32), k, mode="nearest")
    for src, dst in (("mmn", "ammn"), ("pkrn", "apkr")):
        v = uniform_filter(np.where(have1, out[src], 0.0).astype(np.float32), k,
                           mode="nearest")
        out[dst] = np.where(w > 0, v / np.maximum(w, 1e-6), 0.0)
    out["d1"] = d1
    return out


def curve(conf, bad, rng, at=QUANTILES):
    """The sparsification curve e(q) sampled at the given kept fractions.

    Ties are broken by a random permutation applied before the sort, not by pixel
    index. A quantised confidence has many ties, and argsort on raster order would
    hand the top-left corner of every tie group to the kept set -- a scanline
    advantage that is not a property of the measure.
    """
    n = bad.size
    perm = rng.permutation(n)
    order = perm[np.argsort(-conf[perm], kind="stable")]
    kept_bad = np.cumsum(bad[order])
    counts = np.maximum((np.asarray(at) * n).astype(np.int64), 1)
    return kept_bad[counts - 1] / counts


def auc(conf, bad, rng):
    """Area under the sparsification curve: the mean of e(q) over the sampled q."""
    return float(np.mean(curve(conf, bad, rng)))


def evaluate(rows, radius, seed=0):
    """Pool every scene's pixels into one population and score each measure on it.

    Pooled, not a mean over scenes: the scenes differ in size and in difficulty, and
    a mean of per-scene AUCs weights a small easy scene like a large hard one. The
    per-scene spread is printed separately because the pooled number hides it.
    """
    rng = np.random.default_rng(seed)
    names = ["msm", "mmn", "pkrn", "ammn", "apkr", "random"]
    if rows[0][6] is not None:
        names += ["lrc", "pkrn|lrc", "rank-sum"]
    per_scene, pooled = {k: [] for k in names}, {k: [] for k in names}
    bad_pool, oracle_scene = [], []
    for disp, gt, s1, s2, d1, d2, resid in rows:
        m = measures(s1, s2, d1, radius)
        keep = (gt > 0) & np.isfinite(disp)
        bad = (np.abs(disp - gt) > TOL)[keep].astype(np.float64)
        bad_pool.append(bad)
        m["random"] = rng.random(s1.shape)
        if resid is not None:
            # A larger disagreement with the reverse match is less confidence, so the
            # residual enters with its sign flipped. A pixel whose right partner has
            # no candidate of its own is treated as maximally inconsistent rather
            # than dropped: it is a real answer the check could not confirm.
            worst = np.nanmax(resid) if np.any(np.isfinite(resid)) else 0.0
            m["lrc"] = -np.nan_to_num(resid, nan=worst + 1.0)

            # The two cues do not combine the same way, because they are not the
            # same kind of thing. The peak ratio ORDERS pixels: it takes a different
            # value almost everywhere. The consistency check DECIDES: 88-97% of
            # pixels agree with the reverse match to within half a disparity, and it
            # has nothing to say about which of those is better.
            #
            # So it enters as a veto and the ratio ranks what survives. Ranking the
            # two and adding them was tried first and is much worse than the ratio
            # alone -- ranking a value that is 96% tied hands out arbitrary distinct
            # ranks inside the tie, which is noise on half the combined score.
            fail = np.nan_to_num(resid, nan=LRC_TOL + 1.0) > LRC_TOL
            m["pkrn|lrc"] = m["pkrn"] - 1e6 * fail
            m["rank-sum"] = np.zeros_like(s1, dtype=np.float64)
            m["rank-sum"][keep] = (rank01(m["pkrn"][keep])
                                   + rank01(m["lrc"][keep]))
        # The oracle sorts by the error itself, so `bad` doubles as its confidence
        # with the sign flipped: correct pixels are the most confident.
        oracle_scene.append(auc(-bad, bad, rng))
        for k in names:
            per_scene[k].append(auc(m[k][keep], bad, rng))
            pooled[k].append(m[k][keep])
    bad_all = np.concatenate(bad_pool)
    res = {k: auc(np.concatenate(v), bad_all, rng) for k, v in pooled.items()}
    res["oracle"] = auc(-bad_all, bad_all, rng)
    spread = {k: (float(np.min(v)), float(np.max(v))) for k, v in per_scene.items()}
    spread["oracle"] = (float(np.min(oracle_scene)), float(np.max(oracle_scene)))
    # The AUC averages the whole curve, including densities no vehicle would run
    # at. These are the points a gate is actually set to.
    op = {k: curve(np.concatenate(v), bad_all, rng, OPERATING) for k, v in pooled.items()}
    op["oracle"] = curve(-bad_all, bad_all, rng, OPERATING)
    return res, spread, op, float(bad_all.mean()), bad_all.size


def check(rows, radius):
    """Rule 13: three things that must fail if the harness is wrong.

    Nothing below reads a published number, so each is a statement about this code
    and this build rather than about the matcher's quality.
    """
    ok = True
    rng = np.random.default_rng(1)

    # 1. The random confidence must reproduce the overall error rate. A flat curve
    #    is what no skill looks like; anything else means the sort or the
    #    cumulative sum is wrong, and every AUC below it would be wrong with it.
    disp, gt, s1, s2, d1, d2, _ = rows[0]
    keep = (gt > 0) & np.isfinite(disp)
    bad = (np.abs(disp - gt) > TOL)[keep].astype(np.float64)
    a = auc(rng.random(bad.size), bad, rng)
    print(f"  random AUC {a:.4f} against error rate {bad.mean():.4f}  "
          f"delta {abs(a - bad.mean()):.4f}", end="  ")
    ok &= abs(a - bad.mean()) < 0.005
    print("PASS" if abs(a - bad.mean()) < 0.005 else "FAIL")

    # 2. An inverted confidence must be WORSE than random by about as much as the
    #    measure is better. A metric that cannot go the wrong way is not measuring
    #    a direction, and a sign error would otherwise read as a good result.
    m = measures(s1, s2, d1, radius)["apkr"][keep]
    good, evil = auc(m, bad, rng), auc(-m, bad, rng)
    print(f"  apkr {good:.4f}, apkr inverted {evil:.4f}, random {a:.4f}", end="  ")
    ok &= good < a < evil
    print("PASS" if good < a < evil else "FAIL")

    # 3. The margin reconstructed here must be the one the C++ gates on. This is
    #    the only check that crosses the language boundary: re-run the same scene
    #    WITH the gate and confirm that exactly the pixels below the threshold
    #    disappeared. If --out-conf were writing the wrong plane, or lambda here
    #    disagreed with cfg.lambda, every margin number would be a fiction.
    name = scenes()[0]
    thr = 0.05
    disp0, gt0, s1_0, s2_0, dd1, dd2, _ = rows[0]
    disp_g = run_scene(name, ["--min-margin", str(thr)])[0]
    mmn = measures(s1_0, s2_0, dd1, radius)["mmn"]
    predicted = np.isfinite(disp0) & (mmn >= thr)
    actual = np.isfinite(disp_g)
    # The gate can only remove; a pixel it keeps must have been answered ungated,
    # and uniqueness can hand a right pixel to a different left pixel once its
    # competitor is gone, so the reverse is not exact. Both directions are counted.
    lost = int((predicted & ~actual).sum())
    gained = int((actual & ~predicted).sum())
    n = int(actual.sum())
    print(f"  --min-margin {thr} on {name}: {n:,} pixels kept, "
          f"{lost} predicted-but-absent, {gained} present-but-unpredicted", end="  ")
    ok &= lost == 0
    print("PASS" if lost == 0 else "FAIL")

    # 4. Every flag this file passes to de_dense must be a flag de_dense knows. The
    #    parser used to ignore what it did not recognise and exit 0, so `--noblock`
    #    for `--no-blockwise` ran the default path while the run claimed to keep the
    #    volume -- and a cmp against the default then reported the two paths
    #    identical on all eight scenes, because both sides were the default path.
    #    Obstacle 25. Asking for a flag that CANNOT exist proves the parser refuses.
    d = os.path.join(DATA, scenes()[0])
    W, H, dmax = (int(v) for v in open(os.path.join(d, "meta.txt")).read().split())
    p = subprocess.run([BIN, os.path.join(d, "left.y8"), os.path.join(d, "right.y8"),
                        str(W), str(H), "--dmax", str(dmax), "--not-a-real-flag"],
                       capture_output=True, text=True)
    print(f"  de_dense --not-a-real-flag exits {p.returncode}", end="  ")
    ok &= p.returncode != 0
    print("PASS" if p.returncode != 0 else "FAIL")
    return ok


FEATURES = ("log peak ratio", "margin", "winning score",
            "consistency residual", "consistency failed", "log APKR")


def features(s1, s2, d1, resid, radius):
    """The cues of steps 1 and 2 as one matrix, one row per pixel.

    The ratio enters as a logarithm because it is a ratio: doubling it should move the
    answer by a fixed amount, not by an amount that depends on where it started. The
    residual enters twice, once clipped and once as the plain yes/no, because 0.55
    measured it to act as a threshold and a fitted weight can use either shape.
    """
    m = measures(s1, s2, d1, radius)
    r = np.nan_to_num(resid, nan=LRC_TOL + 1.0)
    return np.stack([np.log(np.maximum(m["pkrn"], 1e-6)),
                     m["mmn"],
                     np.where(s1 > -1e29, s1, 0.0),
                     np.minimum(r, 5.0),
                     (r > LRC_TOL).astype(np.float64),
                     np.log(np.maximum(m["apkr"], 1e-6))], axis=-1)


def logistic_fit(X, y, l2=1.0, iters=25):
    """Logistic regression by Newton's method. No dependency beyond numpy.

    The model is P(correct) = 1 / (1 + exp(-(w.x + b))), and each step solves the
    weighted least-squares problem the second-order expansion gives. With six features
    the matrix to invert is 7x7, so twenty-five steps cost less than reading the data.

    L2 does not regularise the intercept: shrinking it would pull the predicted rate
    away from the base rate, which is the one thing the model should get right for
    free.
    """
    n, k = X.shape
    A = np.concatenate([X, np.ones((n, 1))], axis=1)
    w = np.zeros(k + 1)
    pen = np.eye(k + 1) * l2
    pen[k, k] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (y - p) - pen @ w
        s = np.maximum(p * (1 - p), 1e-6)
        H = (A * s[:, None]).T @ A + pen
        step = np.linalg.solve(H, g)
        w += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def predict(w, X):
    A = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return 1.0 / (1.0 + np.exp(-np.clip(A @ w, -30, 30)))


def reliability(p, y, bins=10):
    """Does a predicted 0.8 mean 80%? Returns per-bin (n, predicted, actual).

    Ranking and calibration are different properties and a measure can have one
    without the other. Everything above this point in the file measures ranking only.
    """
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum():
            out.append((int(m.sum()), float(p[m].mean()), float(y[m].mean())))
        else:
            out.append((0, float((edges[i] + edges[i + 1]) / 2), float("nan")))
    return out


def ece(rel, n):
    """Expected calibration error: the gap between promise and delivery, weighted."""
    return sum(c * abs(pm - ac) for c, pm, ac in rel if c) / max(1, n)


MARGINS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10)
RATIOS = (1.02, 1.05, 1.10, 1.15, 1.20, 1.30)


def fit_report(rows, names, radius, seed=0):
    """Steps 3 and 4: combine the cues into one number, then ask what that number means.

    LEAVE ONE SCENE OUT, not a random split of pixels. Neighbouring pixels are not
    independent -- they share a window, a surface and a lighting condition -- so a
    random split puts a pixel's neighbours in the training set and measures memory
    rather than generalisation. Every number below is predicted by a model that never
    saw its scene.
    """
    rng = np.random.default_rng(seed)
    per, P, Y = [], [], []
    in_sample = []
    Xs, ys = [], []
    for disp, gt, s1, s2, d1, d2, resid in rows:
        keep = (gt > 0) & np.isfinite(disp)
        Xs.append(features(s1, s2, d1, resid, radius)[keep])
        ys.append((np.abs(disp - gt) <= TOL)[keep].astype(np.float64))

    for i, name in enumerate(names):
        tr = [j for j in range(len(rows)) if j != i]
        Xtr = np.concatenate([Xs[j] for j in tr])
        ytr = np.concatenate([ys[j] for j in tr])
        # Standardised on the TRAINING folds only. Using the held-out scene's own mean
        # and spread would leak it into its own prediction.
        mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-9)
        w = logistic_fit((Xtr - mu) / sd, ytr)
        p = predict(w, (Xs[i] - mu) / sd)
        bad = 1.0 - ys[i]
        per.append((name, auc(p, bad, rng), float(bad.mean()),
                    ece(reliability(p, ys[i]), ys[i].size)))
        P.append(p); Y.append(ys[i])
        # The same scene fitted on itself, to price holding it out.
        mu2, sd2 = Xs[i].mean(0), np.maximum(Xs[i].std(0), 1e-9)
        w2 = logistic_fit((Xs[i] - mu2) / sd2, ys[i])
        in_sample.append(auc(predict(w2, (Xs[i] - mu2) / sd2), bad, rng))

    P, Y = np.concatenate(P), np.concatenate(Y)
    bad_all = 1.0 - Y

    print("Step 3, combining the cues, and step 4, asking what the number means.")
    print(f"{len(names)} scenes, {Y.size:,} pixels, leave-one-scene-out: every "
          f"prediction comes from\na model that never saw that scene. Correct means "
          f"|d - d_gt| <= {TOL:.1f} disparity.\n")

    # Per-scene calibration, because a pooled calibration error can be small while
    # every scene is wrong: over-promising on the easy scenes and under-promising on
    # the hard ones cancels in the pool. The scenes here run from 5.0% to 20.1% wrong,
    # so if the features did not carry a scene's difficulty this column would show it.
    print(f"  {'held-out scene':<16}{'its error rate':>15}{'AUC held out':>14}"
          f"{'AUC fitted on itself':>22}{'calibration err':>17}")
    for (name, a, e, cal), ins in zip(per, in_sample):
        print(f"  {name:<16}{100*e:>14.1f}%{a:>14.4f}{ins:>22.4f}{cal:>17.4f}")
    print(f"  {'pooled':<16}{100*bad_all.mean():>14.1f}%"
          f"{auc(P, bad_all, rng):>14.4f}{'':>22}"
          f"{ece(reliability(P, Y), Y.size):>17.4f}")

    # Controls. Rule 13 again: a fitted model will always produce SOME number, and
    # both of these say what that number looks like when it means nothing.
    Xr = rng.random((Y.size, len(FEATURES)))
    wr = logistic_fit(Xr, Y)
    pr = predict(wr, Xr)
    print(f"\n  fitted on random features: AUC {auc(pr, bad_all, rng):.4f} against "
          f"a no-skill {bad_all.mean():.4f},\n  and it predicts a constant "
          f"{pr.mean():.3f} -- which is the base rate, and all it can know.")

    print("\nDoes a predicted 0.8 mean 80%? Held-out pixels, binned by what was "
          "promised:\n")
    print(f"  {'predicted':>12}{'pixels':>12}{'promised':>11}{'delivered':>12}"
          f"{'gap':>8}")
    for i, (c, pm, ac) in enumerate(reliability(P, Y)):
        if not c:
            continue
        print(f"  {f'{i/10:.1f}-{(i+1)/10:.1f}':>12}{c:>12,}{pm:>11.3f}"
              f"{ac:>12.3f}{ac-pm:>+8.3f}")
    e = ece(reliability(P, Y), Y.size)
    brier = float(np.mean((P - Y) ** 2))
    base = float(np.mean((Y.mean() - Y) ** 2))
    print(f"\n  expected calibration error {e:.4f} -- the average gap between what "
          f"was\n  promised and what arrived. Brier score {brier:.4f} against "
          f"{base:.4f} for\n  predicting the base rate everywhere; lower is better "
          f"and the difference is\n  what the features are worth.")

    # Which features earn their place, measured by removing them rather than by
    # reading their weights. The cues are built from the same two scores and are
    # heavily correlated, and a weight in a correlated set says how the fit divided
    # the credit, not how much the cue is worth. Refitting without it does say that.
    def foldwise(cols):
        p, y = [], []
        for i in range(len(rows)):
            tr = [j for j in range(len(rows)) if j != i]
            Xtr = np.concatenate([Xs[j][:, cols] for j in tr])
            ytr = np.concatenate([ys[j] for j in tr])
            mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-9)
            w = logistic_fit((Xtr - mu) / sd, ytr)
            p.append(predict(w, (Xs[i][:, cols] - mu) / sd)); y.append(ys[i])
        p, y = np.concatenate(p), np.concatenate(y)
        return auc(p, 1.0 - y, rng), ece(reliability(p, y), y.size)

    allc = list(range(len(FEATURES)))
    print("\nWhat each cue is worth, by taking it away and refitting. Held out as "
          "above.\n")
    print(f"  {'feature set':<32}{'AUC':>9}{'calibration error':>20}")
    a0, e0 = foldwise(allc)
    print(f"  {'all six':<32}{a0:>9.4f}{e0:>20.4f}")
    for i, nm in enumerate(FEATURES):
        a, e = foldwise([c for c in allc if c != i])
        print(f"  {'without ' + nm:<32}{a:>9.4f}{e:>20.4f}{a-a0:>+10.4f}")
    for i, nm in enumerate(FEATURES):
        a, e = foldwise([i])
        print(f"  {nm + ' alone':<32}{a:>9.4f}{e:>20.4f}")

    # The weights, from one model fitted on everything. Printed to be carried into
    # C++, not to be interpreted one at a time -- see the ablation above.
    Xa, ya = np.concatenate(Xs), np.concatenate(ys)
    mu, sd = Xa.mean(0), np.maximum(Xa.std(0), 1e-9)
    w = logistic_fit((Xa - mu) / sd, ya)
    print("\nWeights, fitted on all scenes, standardised. These are for carrying into "
          "an\nimplementation; do not read them one at a time.\n")
    for nm, v, m, s in zip(FEATURES, w[:-1], mu, sd):
        print(f"  {nm:<24}{v:>+9.3f}   (raw mean {m:+.3f}, spread {s:.3f})")
    print(f"  {'intercept':<24}{w[-1]:>+9.3f}")


def gate_sweep():
    """Both gates INSIDE the solver, compared at matched coverage.

    The sparsification curve above ranks pixels after the fact. A gate is different:
    it runs before the uniqueness pass, so removing a pixel can hand its right-image
    partner to a competitor that would otherwise have been refused. That reaction
    cannot be simulated from a dump, so both gates are swept for real here and the
    margin's error rate is interpolated to the coverage each ratio delivers.
    """
    def sweep(flagset):
        out = []
        for flag in flagset:
            p = subprocess.run(
                [sys.executable, os.path.join(HERE, "dense_bench.py")] + flag,
                capture_output=True, text=True)
            if p.returncode != 0:
                sys.exit(f"dense_bench failed for {flag}:\n{p.stderr}")
            line = [l for l in p.stdout.splitlines() if "pixel-pooled" in l]
            if not line:
                sys.exit(f"no pooled line from dense_bench for {flag}:\n{p.stdout}")
            f = line[0].replace("%", "").split()
            out.append((float(f[2]), float(f[4])))          # coverage, bad-1.0
        return out

    marg = sweep([["--min-margin", str(m)] for m in MARGINS])
    rat = sweep([["--min-margin", "0", "--extra", f"--min-ratio {r}"] for r in RATIOS])
    # np.interp needs an increasing x, and coverage falls as the gate tightens.
    mc = np.array([c for c, _ in marg])[::-1]
    mb = np.array([b for _, b in marg])[::-1]

    print("Both gates, all 8 scenes, pixel-pooled, --threads 4. bad-1.0 is over the "
          "pixels\nkept, coverage is over the pixels with known ground truth.\n")
    print(f"  {'--min-margin':>13}{'coverage':>10}{'bad-1.0':>9}     "
          f"{'--min-ratio':>12}{'coverage':>10}{'bad-1.0':>9}{'margin here':>13}"
          f"{'ratio wins':>12}")
    for i in range(max(len(MARGINS), len(RATIOS))):
        left = (f"  {MARGINS[i]:>13.3f}{marg[i][0]:>9.1f}%{marg[i][1]:>8.1f}%"
                if i < len(MARGINS) else " " * 31)
        if i < len(RATIOS):
            cov, bad = rat[i]
            here = float(np.interp(cov, mc, mb))
            print(f"{left}     {RATIOS[i]:>12.2f}{cov:>9.1f}%{bad:>8.1f}%"
                  f"{here:>12.1f}%{here - bad:>11.1f}")
        else:
            print(left)
    print("\n  `margin here` is the margin gate's error rate interpolated to the "
          "coverage\n  the ratio gate delivered, so the last column is a comparison "
          "at equal\n  coverage and positive means the ratio is better.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-sweep", action="store_true",
                    help="compare --min-margin against --min-ratio in the solver")
    ap.add_argument("--radius", type=int, default=2,
                    help="half-width of the aggregation window, in pixels "
                         "(default 2, a 5x5)")
    ap.add_argument("--fit", action="store_true",
                    help="combine the cues into a probability and check it is "
                         "calibrated (implies --lrc)")
    ap.add_argument("--lrc", action="store_true",
                    help="also score left-right consistency, read off the cost "
                         "volume. Slower: it keeps the volume the shipping path "
                         "never stores")
    ap.add_argument("--tol", type=float, default=None,
                    help="disparity error that counts as wrong (default 1.0)")
    ap.add_argument("--check", action="store_true",
                    help="run the harness's own failure tests and stop")
    ap.add_argument("--extra", default="", help="extra de_dense flags")
    a = ap.parse_args()
    if not os.path.exists(BIN):
        sys.exit(f"{BIN} not built -- run `cd core && make`")
    if a.tol is not None:
        globals()["TOL"] = a.tol
    if a.gate_sweep:
        gate_sweep()
        return

    names = scenes()
    rows = [run_scene(s, a.extra.split(), a.lrc or a.fit) for s in names]
    if a.fit:
        fit_report(rows, names, a.radius)
        return

    if a.check:
        print("harness checks -- each one fails if this file is wrong\n")
        sys.exit(0 if check(rows, a.radius) else 1)

    res, spread, op, err, n = evaluate(rows, a.radius)
    print(f"de_dense --min-margin 0 {a.extra}, {len(names)} scenes, "
          f"{n:,} pixels with known ground truth and an answer")
    print(f"correctness is |d - d_gt| <= {TOL:.1f} disparity; aggregation window "
          f"{2*a.radius+1}x{2*a.radius+1}\n")
    print(f"  {'measure':<10}{'AUC':>9}{'- oracle':>11}{'per-scene AUC':>22}")
    order = [k for k in ("oracle", "pkrn|lrc", "apkr", "pkrn", "ammn", "mmn",
                         "msm", "lrc", "rank-sum", "random") if k in res]
    for k in order:
        lo, hi = spread[k]
        print(f"  {k:<10}{res[k]:>9.4f}{res[k]-res['oracle']:>11.4f}"
              f"{lo:>13.4f}..{hi:<8.4f}")
    print(f"\n  error rate over the whole population: {err:.4f}. A confidence that "
          f"ranks nothing\n  scores that; the oracle scores {res['oracle']:.4f}, "
          f"and the gap between them,\n  {err - res['oracle']:.4f}, is all there is "
          f"to win.\n")

    print("error rate among the pixels kept, at the densities a gate is set to:\n")
    print(f"  {'measure':<10}" + "".join(f"{100*q:>9.0f}%" for q in OPERATING))
    for k in order:
        print(f"  {k:<10}" + "".join(f"{100*e:>9.2f} " for e in op[k]))
    print("\n  Read a row against `random`, which is what keeping that fraction "
          "at\n  random costs, and against `oracle`, which is the floor.")


if __name__ == "__main__":
    main()
