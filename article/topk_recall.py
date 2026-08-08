#!/usr/bin/env python3
"""How many candidates per pixel does dense MASDA actually need?

The dense solver carries every (pixel, disparity) pair: 10.1M edges for a 450x375
pair at D=60. Almost all of them are nobody's plausible answer, so MASDA could run
on a pruned candidate list instead -- which is the sparse edge-list form the
algorithm was originally written for, and would remove both the 40 MB volume and
the stride-(D+1) diagonal walk in the beta update.

Pruning has a ceiling, though: if the correct disparity is not among the top k by
aggregated cost, no solver can recover it. This measures that ceiling before any of
it is built.

Two things it deliberately does NOT do:

  - Prune by a global threshold on score. Census costs are absolute Hamming
    fractions, so a textureless region scores uniformly mediocre and its correct
    match scores mediocre too. A magnitude cutoff would delete exactly the flat
    regions where half the remaining error already sits. Rank within a pixel is
    the only safe criterion.
  - Report recall against all pixels. Recall is conditioned on the true disparity
    being INSIDE the search range at all; pixels whose ground truth exceeds dmax
    are unreachable regardless of k, so they are counted separately rather than
    charged to pruning.

    cd core && make
    .venv/bin/python article/topk_recall.py [scene ...]
"""

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
KS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
DMIN = 1


def recall(name):
    d = os.path.join(DATA, name)
    W, H, dmax = (int(v) for v in open(os.path.join(d, "meta.txt")).read().split())
    D = dmax - DMIN + 1
    vp = os.path.join(SCRATCH, f"vol_{name}.f32")
    p = subprocess.run(
        [BIN, os.path.join(d, "left.y8"), os.path.join(d, "right.y8"),
         str(W), str(H), "--dmax", str(dmax), "--iters", "2", "--threads", "4",
         "--dump-vol", vp], capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"de_dense failed on {name}:\n{p.stderr}")

    vol = np.fromfile(vp, np.float32).reshape(H, W, D)
    os.remove(vp)
    gt = np.fromfile(os.path.join(d, "disp.f32"), np.float32).reshape(H, W)

    # A candidate at index k means disparity DMIN + k. Invalid entries are -1e30,
    # which sorts to the bottom on its own.
    known = gt > 0
    kbest = int(max(KS))
    # argpartition on the last axis: the kbest largest scores per pixel, unordered,
    # which is all a set-membership test needs.
    idx = np.argpartition(-vol, kbest - 1, axis=2)[:, :, :kbest]
    order = np.take_along_axis(vol, idx, axis=2)
    rank = np.argsort(-order, axis=2)
    idx = np.take_along_axis(idx, rank, axis=2)          # now sorted, best first
    cand = idx.astype(np.float32) + DMIN

    # In range at all: is any valid disparity within TOL of the truth?
    in_range = known & (gt >= DMIN - TOL) & (gt <= dmax + TOL)
    hit = np.abs(cand - gt[:, :, None]) <= TOL
    n_known, n_range = int(known.sum()), int(in_range.sum())

    out = {}
    for k in KS:
        got = hit[:, :, :k].any(axis=2)
        out[k] = float((got & in_range).sum() / max(1, n_range))
    # Where the winner already is the truth, i.e. what aggregation alone achieves.
    out["top1_of_known"] = float((hit[:, :, 0] & known).sum() / max(1, n_known))
    out["reachable"] = n_range / max(1, n_known)
    return name, out


def main():
    names = sys.argv[1:] or sorted(
        s for s in os.listdir(DATA)
        if os.path.isfile(os.path.join(DATA, s, "meta.txt")))
    if not os.path.exists(BIN):
        sys.exit(f"{BIN} not built -- run `cd core && make`")

    print("recall = the true disparity is within 1.0 px of one of the top k")
    print("candidates by aggregated cost, over pixels whose truth is in range.\n")
    hdr = "".join(f"{('k=%d' % k):>7}" for k in KS)
    print(f"{'scene':<10}{'reach':>7}{hdr}")
    rows = []
    for n in names:
        name, r = recall(n)
        rows.append(r)
        body = "".join(f"{100*r[k]:>6.1f}%" for k in KS)
        print(f"{name:<10}{100*r['reachable']:>6.1f}%{body}")
    print(f"{'mean':<10}{100*np.mean([r['reachable'] for r in rows]):>6.1f}%"
          + "".join(f"{100*np.mean([r[k] for r in rows]):>6.1f}%" for k in KS))
    print(f"\naggregation's own top-1 is correct for "
          f"{100*np.mean([r['top1_of_known'] for r in rows]):.1f}% of known pixels "
          f"-- what MASDA and the uniqueness constraint are added to improve on.")


if __name__ == "__main__":
    main()
