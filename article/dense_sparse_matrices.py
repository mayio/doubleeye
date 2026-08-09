#!/usr/bin/env python3
"""Dense MASDA on sparse matrices, measured for the Part 1 reframe.

The article's original results applied MASDA to sparse KEYPOINTS. The reframed
article applies it to the dense problem -- every pixel, candidates as a sparse
matrix (top-2 disparities per pixel out of the aggregated cost volume) -- and
this script produces the three measurements that replace the keypoint tables:

  1. OPTIMALITY: per image row, MASDA on the sparse candidate matrix against the
     exact assignment optimum (Jonker-Volgenant with explicit lambda/gamma
     slots, the article's own convention). Objective ratio and per-row agreement.
  2. THE VALUE OF UNIQUENESS: MASDA and JV against winner-take-all (argmax over
     the full volume, no one-to-one constraint) on the same aggregated scores,
     scored against Middlebury ground truth.
  3. SPEED, the representation decides it: the dense-matrix NumPy solver against
     the sparse edge-list solver against per-row JV, on the dense problem's
     rows. (The engineered C++/CUDA endpoints are cited from parts 2 and 3, not
     re-measured here.)

Input is the aggregated cost volume dumped by the shipping binary
(`de_dense --dump-vol`), so the NumPy study runs on exactly the scores the C++
pipeline aggregates. Note the dump path runs the float filter, not the int16
one; the two agree to quantisation and neither is re-tuned here.

    .venv/bin/python article/dense_sparse_matrices.py            # all measurements
    .venv/bin/python article/dense_sparse_matrices.py --quick    # teddy only
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from masda_stereo import (_seg_max_excluding, masda, masda_sparse,  # noqa: E402
                          masda_sparse_ordering, count_crossings)

DATA = os.path.join(HERE, "data", "c_bench")
BIN = os.path.join(os.path.dirname(HERE), "core", "build", "de_dense")
LAM = GAM = -0.1
TOL = 1.0
INVALID = -1e29


def load_scene(name):
    d = os.path.join(DATA, name)
    W, H, dmax = (int(v) for v in open(os.path.join(d, "meta.txt")).read().split())
    gt = np.fromfile(os.path.join(d, "disp.f32"), np.float32).reshape(H, W)
    return d, W, H, dmax, gt


def dump_volume(name):
    """Run the shipping binary once, keep the aggregated volume."""
    d, W, H, dmax, gt = load_scene(name)
    volp = f"/tmp/vol_{name}.f32"
    if not os.path.exists(volp):
        cmd = [BIN, os.path.join(d, "left.y8"), os.path.join(d, "right.y8"),
               str(W), str(H), "--dmax", str(dmax), "--threads", "4",
               "--agg", "5", "--iters", "2", "--min-margin", "0.01",
               "--dump-vol", volp, "--out", "/tmp/dsm_disp.f32"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            sys.exit(f"de_dense failed on {name}:\n{p.stderr}")
    D = dmax  # dmin = 1 .. dmax
    vol = np.fromfile(volp, np.float32).reshape(H, W, D)
    return vol, W, H, D, gt


def top2_edges_row(vol_row, W, D, dmin=1):
    """The sparse matrix for one row: top-2 disparities per left pixel.

    Returns (ei, ej, se) with ej the RIGHT pixel index (x - d), plus per-pixel
    (d1, d2, s1, s2) for WTA-style analysis.
    """
    v = vol_row  # (W, D), invalid <= -1e29
    # top-2 by score, ties to smaller k -- argsort is stable on the negated array
    order = np.argsort(-v, axis=1, kind="stable")
    k1, k2 = order[:, 0], order[:, 1]
    x = np.arange(W)
    s1, s2 = v[x, k1], v[x, k2]
    ei, ej, se = [], [], []
    for xx in range(W):
        for k, s in ((k1[xx], s1[xx]), (k2[xx], s2[xx])):
            if s <= INVALID:
                continue
            ei.append(xx)
            ej.append(xx - (int(k) + dmin))
            se.append(float(s))
    return (np.array(ei, np.int64), np.array(ej, np.int64),
            np.array(se, np.float64), k1, k2, s1, s2)


def jv_row(ei, ej, se, W):
    """Exact optimum on the row's sparse problem, article convention: explicit
    lambda/gamma slots so non-association is a real option."""
    from scipy.optimize import linear_sum_assignment
    S = np.full((W, W), -np.inf)
    S[ei, ej] = se
    big = np.full((2 * W, 2 * W), 0.0)   # dummy-dummy pairings are free
    big[:W, :W] = np.where(np.isfinite(S), S, -1e6)
    big[:W, W:] = -1e6
    np.fill_diagonal(big[:W, W:], LAM)
    big[W:, :W] = -1e6
    np.fill_diagonal(big[W:, :W], GAM)
    r, c = linear_sum_assignment(-big)
    return {int(i): int(j) for i, j in zip(r, c)
            if i < W and j < W and np.isfinite(S[i, j]) and S[i, j] > LAM}


def dense_masda_row(ei, ej, se, W):
    """The article's dense-matrix solver on this row (for the speed table)."""
    S = np.full((W, W), -np.inf)
    S[ei, ej] = se
    return masda(S, lam=LAM, gam=GAM, iters=30, damping=0.4)


def obj(assign, se_lookup, m, n):
    v = sum(se_lookup[(i, j)] for i, j in assign.items())
    return v + LAM * (m - len(assign)) + GAM * (n - len(assign))


def eval_assign(y, assign_disp, gt, known_only=True):
    """correct/wrong within TOL against ground truth for one row.
    assign_disp: dict x -> disparity (float)."""
    ok = wrong = unk = 0
    for x, d in assign_disp.items():
        t = gt[y, x]
        if t <= 0:
            unk += 1
            continue
        if abs(d - t) <= TOL:
            ok += 1
        else:
            wrong += 1
    return ok, wrong, unk


def dense_positions(W):
    """Positions for the ordering factor: in the dense formulation a "keypoint"
    IS a pixel index, and one image row is one scanline band, so every edge pair
    in the row is a candidate crossing. That is the structural difference from
    the keypoint configuration and it is why the pair count explodes."""
    p = np.zeros((W, 2), np.float64)
    p[:, 0] = np.arange(W)
    return p


def run_scene(name, with_jv, with_speed, row_step=1, kappa=None):
    vol, W, H, D, gt = dump_volume(name)
    dmin = 1
    tot = dict(masda_ok=0, masda_wr=0, masda_n=0,
               wta_ok=0, wta_wr=0, wta_n=0,
               jv_ok=0, jv_wr=0, jv_n=0,
               obj_masda=0.0, obj_jv=0.0, rows=0, rows_opt=0,
               t_sparse=0.0, t_dense=0.0, t_jv=0.0, edges=0,
               ord_ok=0, ord_wr=0, ord_n=0, xing_base=0, xing_ord=0,
               pairs=0, t_ord=0.0)
    rows = range(3, H - 3, row_step)
    for y in rows:
        ei, ej, se, k1, k2, s1, s2 = top2_edges_row(vol[y], W, D, dmin)
        if len(ei) == 0:
            continue
        tot["edges"] += len(ei)
        # WTA: argmax over the FULL volume (not just top-2 -- that IS the top-1)
        wta = {int(x): float(k1[x] + dmin) for x in range(W) if s1[x] > INVALID}
        ok, wr, _ = eval_assign(y, wta, gt)
        tot["wta_ok"] += ok; tot["wta_wr"] += wr; tot["wta_n"] += len(wta)

        # MASDA on the sparse matrix
        t0 = time.perf_counter()
        assign, _ = masda_sparse(ei, ej, se, W, W, lam=LAM, gam=GAM,
                                 iters=30, damping=0.4)
        tot["t_sparse"] += time.perf_counter() - t0
        md = {i: float(i - j) for i, j in assign.items()}
        ok, wr, _ = eval_assign(y, md, gt)
        tot["masda_ok"] += ok; tot["masda_wr"] += wr; tot["masda_n"] += len(md)

        if kappa is not None:
            pos = dense_positions(W)
            t0 = time.perf_counter()
            oa, _, npairs = masda_sparse_ordering(ei, ej, se, W, W, pos, pos,
                                                  kappa=kappa, lam=LAM, gam=GAM,
                                                  iters=30, damping=0.6)
            tot["t_ord"] += time.perf_counter() - t0
            od = {i: float(i - j) for i, j in oa.items()}
            ok, wr, _ = eval_assign(y, od, gt)
            tot["ord_ok"] += ok; tot["ord_wr"] += wr; tot["ord_n"] += len(od)
            tot["xing_base"] += count_crossings(assign, pos, pos)[0]
            tot["xing_ord"] += count_crossings(oa, pos, pos)[0]
            tot["pairs"] += npairs

        se_lookup = {(int(i), int(j)): float(s) for i, j, s in zip(ei, ej, se)}
        if with_jv:
            t0 = time.perf_counter()
            jv = jv_row(ei, ej, se, W)
            tot["t_jv"] += time.perf_counter() - t0
            jd = {i: float(i - j) for i, j in jv.items()}
            ok, wr, _ = eval_assign(y, jd, gt)
            tot["jv_ok"] += ok; tot["jv_wr"] += wr; tot["jv_n"] += len(jd)
            om = obj(assign, se_lookup, W, W)
            oj = obj(jv, se_lookup, W, W)
            tot["obj_masda"] += om
            tot["obj_jv"] += oj
            tot["rows_opt"] += int(abs(om - oj) < 1e-6)
        if with_speed:
            t0 = time.perf_counter()
            dense_masda_row(ei, ej, se, W)
            tot["t_dense"] += time.perf_counter() - t0
        tot["rows"] += 1
    return tot, W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--row-step", type=int, default=1)
    ap.add_argument("--kappa", type=float, default=None,
                    help="also run the ordering factor at this kappa")
    a = ap.parse_args()
    scenes = ["teddy"] if a.quick else sorted(os.listdir(DATA))
    jv_scenes = {"teddy", "cones"}
    speed_scenes = {"teddy"}

    pooled = dict(masda_ok=0, masda_wr=0, wta_ok=0, wta_wr=0)
    print(f"{'scene':<10} {'method':<7} {'matches':>8} {'correct':>8} "
          f"{'precision':>9}")
    for s in scenes:
        with_jv = s in jv_scenes
        with_speed = s in speed_scenes
        t, W, H = run_scene(s, with_jv, with_speed, a.row_step, a.kappa)
        meths = ["wta", "masda"] + (["jv"] if with_jv else []) + (
            ["ord"] if a.kappa is not None else [])
        for meth in meths:
            n, ok, wr = t[f"{meth}_n"], t[f"{meth}_ok"], t[f"{meth}_wr"]
            if n == 0:
                continue
            print(f"{s:<10} {meth:<7} {n:>8} {ok:>8} {ok/max(1,ok+wr):>9.3f}")
        pooled["masda_ok"] += t["masda_ok"]; pooled["masda_wr"] += t["masda_wr"]
        pooled["wta_ok"] += t["wta_ok"]; pooled["wta_wr"] += t["wta_wr"]
        if a.kappa is not None:
            print(f"{s:<10} ordering k={a.kappa}: crossings "
                  f"{t['xing_base']} -> {t['xing_ord']} "
                  f"({t['xing_ord']/max(1,t['xing_base']):.2f}x), "
                  f"{t['pairs']} crossing pairs, "
                  f"solve {t['t_ord']:.1f}s vs {t['t_sparse']:.1f}s plain")
        if with_jv:
            print(f"{s:<10} optimality: rows at exact optimum "
                  f"{t['rows_opt']}/{t['rows']}  "
                  f"objective ratio {t['obj_masda']/t['obj_jv']:.6f}")
        if with_speed:
            r = t["rows"]
            print(f"{s:<10} speed/frame ({r} rows, {t['edges']} edges): "
                  f"dense-matrix {t['t_dense']*1e3:.0f} ms  "
                  f"sparse {t['t_sparse']*1e3:.0f} ms  "
                  f"JV {t['t_jv']*1e3:.0f} ms")
    print(f"\npooled over {len(scenes)} scenes (per-pixel, top-2 candidates):")
    for meth in ("wta", "masda"):
        ok, wr = pooled[f"{meth}_ok"], pooled[f"{meth}_wr"]
        print(f"  {meth}: correct {ok}  wrong {wr}  precision {ok/(ok+wr):.4f}")


if __name__ == "__main__":
    main()
