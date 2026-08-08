#!/usr/bin/env python3
"""Two-level coarse-to-fine dense MASDA, and what it costs in accuracy.

Runtime is linear in D at every stage. This runs the matcher at half resolution to
get a disparity prior, then re-runs it at full resolution searching only +-B around
that prior -- so the fine pass needs 2B+1 planes instead of D.

The ceiling was measured first (`coarse_ceiling.py`): 81.5% of known pixels have
their truth inside the band at B=2, against 67.9% the single-level pipeline
delivers, for 5.2x less arithmetic. This measures what is actually achieved against
that ceiling.

Two details that the ceiling experiment showed matter, and are easy to get wrong:

  - **The coarse pass is not gated.** `--min-margin 0` for the prior. The gate
    trades coverage for precision in a final answer; in a prior every gated pixel
    becomes a pixel with no search band. Gating the coarse pass moved the measured
    ceiling from 81.5% to 68.4%, which is the difference between "do this" and
    "this cannot work".
  - **Holes are filled.** A hole is not a narrow band, it is no band. de_dense
    refuses a prior containing holes rather than quietly scoring one.

    .venv/bin/python article/coarse_to_fine.py [--band 2]
"""

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data", "c_bench")
BIN = os.path.join(ROOT, "core", "build", "de_dense")
SCRATCH = os.environ.get("TMPDIR", "/tmp")
TOL = 1.0


def sh(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"failed: {' '.join(cmd)}\n{p.stderr}")
    f = p.stdout.split()
    return float(f[f.index("total") + 1])


def half(img):
    """2x2 box average; decimating would alias, and aliasing at the coarse level is
    indistinguishable from a matcher error downstream."""
    H, W = img.shape
    H2, W2 = H // 2, W // 2
    a = img[:H2 * 2, :W2 * 2].astype(np.float32).reshape(H2, 2, W2, 2)
    return a.mean(axis=(1, 3)).round().clip(0, 255).astype(np.uint8)


def fill(dc):
    """Nearest-valid inpainting by iterated 4-neighbour averaging."""
    dc = dc.copy()
    for _ in range(128):
        bad = ~np.isfinite(dc)
        if not bad.any():
            return dc
        acc = np.zeros_like(dc)
        cnt = np.zeros_like(dc)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            s = np.roll(np.roll(dc, dy, 0), dx, 1)
            ok = np.isfinite(s)
            acc[ok] += s[ok]
            cnt[ok] += 1
        f = bad & (cnt > 0)
        if not f.any():
            break
        dc[f] = acc[f] / cnt[f]
    # A frame with no valid pixel at all would loop forever otherwise.
    return np.nan_to_num(dc, nan=0.0)


def run(name, band, margin):
    d = os.path.join(DATA, name)
    W, H, dmax = (int(v) for v in open(os.path.join(d, "meta.txt")).read().split())
    L = np.fromfile(os.path.join(d, "left.y8"), np.uint8).reshape(H, W)
    R = np.fromfile(os.path.join(d, "right.y8"), np.uint8).reshape(H, W)
    lp, rp = f"{SCRATCH}/ctf_l_{name}.y8", f"{SCRATCH}/ctf_r_{name}.y8"
    cp, pp = f"{SCRATCH}/ctf_c_{name}.f32", f"{SCRATCH}/ctf_p_{name}.f32"
    op = f"{SCRATCH}/ctf_o_{name}.f32"

    Lh, Rh = half(L), half(R)
    H2, W2 = Lh.shape
    Lh.tofile(lp); Rh.tofile(rp)
    t_coarse = sh([BIN, lp, rp, str(W2), str(H2), "--dmax", str(max(2, dmax // 2)),
                   "--min-margin", "0", "--threads", "4", "--out", cp])
    # Upsample BILINEARLY to full resolution here rather than letting de_dense do
    # a nearest x2. This matters more than it looks: the cost volume is indexed by
    # offset from the prior, so a plane aggregates neighbours whose ABSOLUTE
    # disparities differ by however much the prior differs. A nearest upsample is
    # blocky with a 1-2 px step at every 2x2 boundary, which puts that misalignment
    # everywhere, at exactly the scale the filter aggregates over. A smooth prior
    # makes an offset plane locally a constant-disparity plane, which is what the
    # aggregation assumes.
    dc = fill(np.fromfile(cp, np.float32).reshape(H2, W2))
    yy = (np.arange(H) + 0.5) * 0.5 - 0.5
    xx = (np.arange(W) + 0.5) * 0.5 - 0.5
    y0 = np.clip(np.floor(yy).astype(int), 0, H2 - 1)
    x0 = np.clip(np.floor(xx).astype(int), 0, W2 - 1)
    y1 = np.clip(y0 + 1, 0, H2 - 1)
    x1 = np.clip(x0 + 1, 0, W2 - 1)
    wy = (yy - y0).clip(0, 1)[:, None]
    wx = (xx - x0).clip(0, 1)[None, :]
    top = dc[np.ix_(y0, x0)] * (1 - wx) + dc[np.ix_(y0, x1)] * wx
    bot = dc[np.ix_(y1, x0)] * (1 - wx) + dc[np.ix_(y1, x1)] * wx
    (2.0 * (top * (1 - wy) + bot * wy)).astype(np.float32).tofile(pp)
    t_fine = sh([BIN, os.path.join(d, "left.y8"), os.path.join(d, "right.y8"),
                 str(W), str(H), "--dmax", str(dmax), "--prior", pp,
                 "--band", str(band), "--min-margin", str(margin),
                 "--threads", "4", "--out", op])

    disp = np.fromfile(op, np.float32).reshape(H, W)
    gt = np.fromfile(os.path.join(d, "disp.f32"), np.float32).reshape(H, W)
    known = gt > 0
    have = known & np.isfinite(disp)
    for f in (lp, rp, cp, pp, op):
        os.remove(f)
    ok = int((np.abs(disp[have] - gt[have]) <= TOL).sum())
    return dict(scene=name, cov=have.sum() / known.sum(),
                bad=float((np.abs(disp[have] - gt[have]) > TOL).mean()),
                ms=t_coarse + t_fine, t_coarse=t_coarse, t_fine=t_fine,
                n_known=int(known.sum()), n_have=int(have.sum()), n_ok=ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=int, default=2)
    ap.add_argument("--min-margin", default="0.01")
    a = ap.parse_args()
    if not os.path.exists(BIN):
        sys.exit(f"{BIN} not built -- run `cd core && make`")
    names = sorted(s for s in os.listdir(DATA)
                   if os.path.isfile(os.path.join(DATA, s, "meta.txt")))
    print(f"coarse-to-fine, band=+-{a.band}, min-margin={a.min_margin}\n")
    print(f"{'scene':<10}{'coverage':>10}{'bad-1.0':>10}{'coarse':>9}{'fine':>7}{'total':>8}")
    rows = []
    for n in names:
        r = run(n, a.band, a.min_margin)
        rows.append(r)
        print(f"{r['scene']:<10}{100*r['cov']:>9.1f}%{100*r['bad']:>9.1f}%"
              f"{r['t_coarse']:>9.0f}{r['t_fine']:>7.0f}{r['ms']:>8.0f}")
    nk = sum(r["n_known"] for r in rows)
    nh = sum(r["n_have"] for r in rows)
    no = sum(r["n_ok"] for r in rows)
    print(f"\npooled over {len(rows)} scenes:")
    print(f"  per-scene mean: coverage {100*np.mean([r['cov'] for r in rows]):.1f}%  "
          f"bad-1.0 {100*np.mean([r['bad'] for r in rows]):.1f}%")
    print(f"  correct over known: {100*no/nk:.1f}%   (ceiling at B={a.band}, "
          f"and 67.9% for the single-level pipeline)")
    print(f"  runtime: mean {np.mean([r['ms'] for r in rows]):.0f} ms "
          f"(coarse {np.mean([r['t_coarse'] for r in rows]):.0f} + "
          f"fine {np.mean([r['t_fine'] for r in rows]):.0f})")


if __name__ == "__main__":
    main()
