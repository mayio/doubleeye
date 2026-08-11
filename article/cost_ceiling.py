#!/usr/bin/env python3
"""How much of the error is the COST's fault, and how much is the solver's?

The near-edge analysis (TODO 0.35) says 71% of all error sits in the far field --
pixels nowhere near a depth discontinuity -- at a 32% rate, and that neither a
sharper edge-aware filter nor a wider candidate set moves it. That points at the
matching cost itself, which is the one component never ablated.

This measures the split. For every pixel, rank the disparities by aggregated cost and
ask where the true one lands:

  recall@1   the cost's own argmax is right         -> what a perfect solver inherits
  recall@2   the truth is in the top-2              -> the CEILING for our pruning
  recall@k   the truth is in the top-k              -> what a wider set could offer

The gap between recall@1 and recall@2 is what the solver can possibly fix. The gap
between recall@2 and 100% is what NO solver on this cost volume can fix, and is
therefore the descriptor's bill.

Tolerance matters here and is reported twice. A candidate within 0.5 px of the truth
is the nearest integer to it, so the sub-pixel fit (clamped to +-0.5) can reach the
answer; a candidate 1 px away cannot be refined into a correct one.

Note the volume has to be materialised for this, which disables the shipping
blockwise path -- so the costs are float rather than int16. That moves the third
decimal, not the conclusion.

    .venv/bin/python article/cost_ceiling.py --data ~/data/MiddEval3
"""

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import middeval3 as m3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "core", "build", "de_dense")
KS = (1, 2, 4, 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MIDDEVAL3", ""))
    ap.add_argument("--res", default="Q")
    ap.add_argument("--threads", default="8")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--extra", default="", help="extra de_dense flags")
    a = ap.parse_args()
    if not a.data:
        sys.exit("need --data or $MIDDEVAL3")
    scenes = a.scenes.split(",") if a.scenes else sorted(m3.WEIGHTS)

    # accumulators: [tolerance][bucket][k] -> hits, and [tolerance][bucket] -> n
    tols = (0.5, 1.0)
    buckets = ("all", "near edge", "far field")
    hit = {(t, b, k): 0 for t in tols for b in buckets for k in KS}
    tot = {(t, b): 0 for t in tols for b in buckets}
    import collections
    chain = collections.Counter()

    for s in scenes:
        d = os.path.join(a.data, f"training{a.res}", s)
        cal = m3.read_calib(os.path.join(d, "calib.txt"))
        W, H, D = int(cal["width"]), int(cal["height"]), int(cal["ndisp"])
        volp = "/tmp/_vol.f32"
        p = subprocess.run([BIN, os.path.join(d, "left.y8"),
                            os.path.join(d, "right.y8"), str(W), str(H),
                            "--dmax", str(D), "--threads", a.threads, "--agg", "5",
                            "--dump-vol", volp, "--out", "/tmp/_cc.f32"]
                           + a.extra.split(),
                           capture_output=True, text=True)
        if p.returncode != 0:
            sys.exit(f"de_dense failed on {s}:\n{p.stderr}")
        vol = np.fromfile(volp, np.float32).reshape(H, W, D)
        out = np.fromfile("/tmp/_cc.f32", np.float32).reshape(H, W)
        os.remove(volp); os.remove("/tmp/_cc.f32")

        gt = m3.read_pfm(os.path.join(d, "disp0GT.pfm"))          # Q-resolution GT
        mask = m3._png(os.path.join(d, "mask0nocc.png"))
        ok = np.isfinite(gt) & (mask == 255)

        # the aggregation leaves -1e30 outside a pixel's legal disparity window
        top = np.argpartition(-vol, max(KS) - 1, axis=-1)[:, :, :max(KS)]
        srt = np.take_along_axis(vol, top, axis=-1)
        order = np.argsort(-srt, axis=-1)
        top = np.take_along_axis(top, order, axis=-1)              # top-8, best first
        live = np.take_along_axis(vol, top, axis=-1) > -1e29
        cand = top.astype(np.float32) + 1.0                        # d = dmin + k

        gy, gx = np.gradient(np.where(np.isfinite(gt), gt, 0.0))
        grad = np.maximum(np.abs(gy), np.abs(gx))
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(~(grad > 0.6))

        err = np.abs(cand - gt[:, :, None])
        for t in tols:
            good = (err <= t) & live                               # (H, W, 8)
            for b, sel in (("all", ok), ("near edge", ok & (dist <= 8)),
                           ("far field", ok & (dist > 8))):
                tot[(t, b)] += int(sel.sum())
                for k in KS:
                    hit[(t, b, k)] += int((good[:, :, :k].any(axis=-1) & sel).sum())
        # --- where does the answer get lost AFTER the cost has proposed it? ------
        # Chain: the cost offers an integer within 0.5 px, the solver picks one of
        # its top-2, and the parabola refines it. The official tolerance is 0.25 px
        # at this resolution, so each stage can lose the pixel.
        far = ok & (dist > 8)
        got = np.isfinite(out) & (out > 0)
        near_int = (np.abs(cand[:, :, 0] - gt) <= 0.5) & live[:, :, 0]
        chain["far n"] += int(far.sum())
        chain["cost top-1 within 0.5"] += int((far & near_int).sum())
        chain["... and answered"] += int((far & near_int & got).sum())
        with np.errstate(invalid="ignore"):
            fine = got & (np.abs(out - gt) <= 0.25)
            half = got & (np.abs(out - gt) <= 0.5)
        chain["... and within 0.5 px"] += int((far & near_int & half).sum())
        chain["... and within 0.25 px"] += int((far & near_int & fine).sum())
        print(f"  {s:<13} done", flush=True)

    for t in tols:
        print(f"\n=== a candidate within {t} px of the true disparity "
              f"({'fit-recoverable' if t == 0.5 else 'loose'}) ===")
        print(f"{'bucket':<12}{'pixels':>10}" + "".join(f"{'top-'+str(k):>9}" for k in KS))
        for b in buckets:
            n = tot[(t, b)]
            print(f"{b:<12}{n:>10,}" +
                  "".join(f"{100*hit[(t,b,k)]/max(1,n):>8.1f}%" for k in KS))
    print("\n=== the far field, stage by stage: where a pixel is lost ===")
    base = max(1, chain["far n"])
    for k in ("far n", "cost top-1 within 0.5", "... and answered",
              "... and within 0.5 px", "... and within 0.25 px"):
        print(f"  {k:<26}{chain[k]:>10,}{100*chain[k]/base:>8.1f}%")
    print("  the last row is the official tolerance. The drop from the row above it\n"
          "  is the sub-pixel fit failing to land, on pixels whose integer was right.")
    print("\ntop-1 is what the cost alone delivers; top-2 is the ceiling for the "
          "shipping\npruning, and everything above top-2 is unreachable without "
          "widening it.\nWhatever top-8 does not reach is the descriptor's bill, "
          "not the solver's.")


if __name__ == "__main__":
    main()
