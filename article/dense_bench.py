#!/usr/bin/env python3
"""Score `core/tools/de_dense` against Middlebury ground truth, all scenes.

Why this exists: the dense coverage and bad-1.0 figures were produced by hand
once, which means they could be quoted but not re-checked, and a runtime change
that quietly altered the output would not have been caught. This runs the
shipping binary over every exported scene and prints the pooled numbers, so any
change to the matcher can be re-scored with one command.

It also writes the raw disparity per scene, which makes bit-for-bit comparison
between two builds possible -- the check that matters when the intent is a pure
speed change:

    .venv/bin/python article/dense_bench.py --out /tmp/before
    # ... edit, rebuild ...
    .venv/bin/python article/dense_bench.py --out /tmp/after
    cmp /tmp/before/teddy.f32 /tmp/after/teddy.f32

Ground truth is zero where unknown (occlusions and the border the structured
light could not see), which is why `known` is `gt > 0` rather than a NaN test.

    .venv/bin/python article/dense_bench.py [--threads N] [--agg N] [--iters N]
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
TOL = 1.0


def scenes():
    return sorted(d for d in os.listdir(DATA)
                  if os.path.isfile(os.path.join(DATA, d, "meta.txt")))


def run(name, args, keep=None):
    d = os.path.join(DATA, name)
    W, H, dmax = (int(v) for v in open(os.path.join(d, "meta.txt")).read().split())
    out = os.path.join(keep, f"{name}.f32") if keep else \
        os.path.join(d, ".dense_tmp.f32")
    cmd = [BIN, os.path.join(d, "left.y8"), os.path.join(d, "right.y8"),
           str(W), str(H), "--dmax", str(dmax), "--out", out] + args
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        # Rule 1: a silent producer failure must not read as a bad score.
        sys.exit(f"de_dense failed on {name} (exit {p.returncode}):\n{p.stderr}")
    ms = None
    for line in p.stdout.splitlines():
        if "total" in line:
            f = line.split()
            ms = float(f[f.index("total") + 1])
    if ms is None:
        sys.exit(f"could not parse timing from de_dense on {name}:\n{p.stdout}")

    disp = np.fromfile(out, np.float32).reshape(H, W)
    gt = np.fromfile(os.path.join(d, "disp.f32"), np.float32).reshape(H, W)
    known = gt > 0
    have = known & np.isfinite(disp)
    bad = float((np.abs(disp[have] - gt[have]) > TOL).mean()) if have.any() else 1.0
    cov = float(have.sum() / max(1, known.sum()))
    if not keep:
        os.remove(out)
    return dict(scene=name, cov=cov, bad=bad, ms=ms,
                n_known=int(known.sum()), n_have=int(have.sum()),
                n_ok=int((np.abs(disp[have] - gt[have]) <= TOL).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="4")
    ap.add_argument("--agg", default="5")
    ap.add_argument("--iters", default="2")
    ap.add_argument("--min-margin", default="0.01")
    ap.add_argument("--out", default=None,
                    help="directory to keep raw disparity in, for cmp")
    a = ap.parse_args()
    if not os.path.exists(BIN):
        sys.exit(f"{BIN} not built -- run `cd core && make`")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    args = ["--threads", a.threads, "--agg", a.agg, "--iters", a.iters,
            "--min-margin", a.min_margin]

    print(f"de_dense --agg {a.agg} --iters {a.iters} "
          f"--min-margin {a.min_margin} --threads {a.threads}\n")
    print(f"{'scene':<10}{'coverage':>10}{'bad-1.0':>10}{'ms':>8}")
    rows = [run(s, args, a.out) for s in scenes()]
    for r in rows:
        print(f"{r['scene']:<10}{100*r['cov']:>9.1f}%{100*r['bad']:>9.1f}%{r['ms']:>8.0f}")

    # Pooled over PIXELS, not a mean of per-scene rates: the scenes have
    # different numbers of known pixels, so averaging the rates weights the
    # small ones up. Both are printed, because the per-scene mean is what
    # earlier notes quoted and the two should not be silently swapped.
    n_have = sum(r["n_have"] for r in rows)
    n_ok = sum(r["n_ok"] for r in rows)
    n_known = sum(r["n_known"] for r in rows)
    print(f"\npooled over {len(rows)} scenes, {n_known:,} known pixels:")
    print(f"  per-scene mean: coverage {100*np.mean([r['cov'] for r in rows]):.1f}%  "
          f"bad-1.0 {100*np.mean([r['bad'] for r in rows]):.1f}%")
    print(f"  pixel-pooled:   coverage {100*n_have/n_known:.1f}%  "
          f"bad-1.0 {100*(1 - n_ok/n_have):.1f}%")
    print(f"  runtime: mean {np.mean([r['ms'] for r in rows]):.0f} ms, "
          f"max {max(r['ms'] for r in rows):.0f} ms")


if __name__ == "__main__":
    main()
