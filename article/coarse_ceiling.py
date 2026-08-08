#!/usr/bin/env python3
"""What would a coarse-to-fine disparity search cost in accuracy?

Runtime is linear in D at every stage, so halving resolution and halving the
search is 8x less work for the coarse pass, and a narrow refinement band at full
resolution costs W*H*(2B+1) instead of W*H*D. At D=60 and B=2 that is about 4.8x
less work in total -- the largest remaining lever, and the only one left that is
not a pure speed change.

It has a hard ceiling, and this measures it before anything is built: if the true
disparity is not inside the refinement band around the upsampled coarse estimate,
no refinement recovers it. That is the same question the top-k sweep asked, except
this time it is asked as "what fraction of pixels stay correct", which is what the
top-k sweep should have been asking -- availability turned out not to be the thing
that decided the outcome there.

Reported per band half-width B:

  - `in band`: |gt - 2*d_coarse| <= B + 1, i.e. some candidate in the band is
    within the 1 px tolerance. This is the ceiling.
  - `no coarse`: pixels where the coarse pass produced nothing, so there is no
    band at all. These need a fallback and are counted, not hidden -- a ceiling
    that quietly excluded them would flatter the method.
  - `speedup`: the arithmetic work ratio, D / (D/8 + 2B+1).

    .venv/bin/python article/coarse_ceiling.py
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
BANDS = (1, 2, 3, 4, 6, 8)


def half(img):
    """2x2 box average. A decimation would alias, and aliasing at the coarse
    level is indistinguishable from a matcher error in the result."""
    H, W = img.shape
    H2, W2 = H // 2, W // 2
    a = img[:H2 * 2, :W2 * 2].astype(np.float32).reshape(H2, 2, W2, 2)
    return a.mean(axis=(1, 3)).round().clip(0, 255).astype(np.uint8)


def run_coarse(name):
    d = os.path.join(DATA, name)
    W, H, dmax = (int(v) for v in open(os.path.join(d, "meta.txt")).read().split())
    L = np.fromfile(os.path.join(d, "left.y8"), np.uint8).reshape(H, W)
    R = np.fromfile(os.path.join(d, "right.y8"), np.uint8).reshape(H, W)
    Lh, Rh = half(L), half(R)
    H2, W2 = Lh.shape
    lp = os.path.join(SCRATCH, f"cl_{name}.y8")
    rp = os.path.join(SCRATCH, f"cr_{name}.y8")
    op = os.path.join(SCRATCH, f"cd_{name}.f32")
    Lh.tofile(lp); Rh.tofile(rp)
    dmax_c = max(2, int(round(dmax / 2)))
    # No margin gate on the coarse pass. A PRIOR should not be gated: the gate
    # exists to trade coverage for precision in a final answer, and here every
    # rejected pixel becomes a pixel with no search band at all.
    p = subprocess.run([BIN, lp, rp, str(W2), str(H2), "--dmax", str(dmax_c),
                        "--min-margin", "0", "--threads", "4", "--out", op],
                       capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"coarse de_dense failed on {name}:\n{p.stderr}")
    dc = np.fromfile(op, np.float32).reshape(H2, W2)
    for f in (lp, rp, op):
        os.remove(f)

    # Fill holes from the nearest valid neighbour before upsampling. A hole in a
    # prior does not have to mean "no band" -- standard coarse-to-fine inpaints
    # it -- and leaving them empty charges the method for something it would not
    # actually do.
    for _ in range(64):
        bad = ~np.isfinite(dc)
        if not bad.any():
            break
        acc = np.zeros_like(dc); cnt = np.zeros_like(dc)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sh = np.roll(np.roll(dc, dy, 0), dx, 1)
            ok = np.isfinite(sh)
            acc[ok] += sh[ok]; cnt[ok] += 1
        fillable = bad & (cnt > 0)
        dc = dc.copy()
        dc[fillable] = acc[fillable] / cnt[fillable]

    # Upsample by nearest and rescale: a disparity of d at half resolution is 2d
    # at full resolution.
    up = np.repeat(np.repeat(dc, 2, axis=0), 2, axis=1) * 2.0
    full = np.full((H, W), np.nan, np.float32)
    full[:up.shape[0], :up.shape[1]] = up[:H, :W]

    gt = np.fromfile(os.path.join(d, "disp.f32"), np.float32).reshape(H, W)
    known = gt > 0
    have = known & np.isfinite(full)
    n_known = int(known.sum())
    out = {"no_coarse": 1.0 - have.sum() / max(1, n_known), "dmax": dmax}
    err = np.abs(gt[have] - full[have])
    for B in BANDS:
        # Correct-and-reachable over ALL known pixels, so the pixels with no
        # coarse estimate count against it rather than being dropped.
        out[B] = float((err <= B + 1.0).sum() / max(1, n_known))
    return name, out


def main():
    if not os.path.exists(BIN):
        sys.exit(f"{BIN} not built -- run `cd core && make`")
    names = sorted(s for s in os.listdir(DATA)
                   if os.path.isfile(os.path.join(DATA, s, "meta.txt")))
    print("ceiling on a coarse-to-fine search: fraction of KNOWN-GT pixels whose")
    print("truth lies within the refinement band around the upsampled coarse")
    print("estimate. Pixels with no coarse estimate count against it.\n")
    print(f"{'scene':<10}{'no coarse':>11}" + "".join(f"{('B=%d' % b):>8}" for b in BANDS))
    rows = []
    for n in names:
        name, r = run_coarse(n)
        rows.append(r)
        print(f"{name:<10}{100*r['no_coarse']:>10.1f}%"
              + "".join(f"{100*r[b]:>7.1f}%" for b in BANDS))
    print(f"{'mean':<10}{100*np.mean([r['no_coarse'] for r in rows]):>10.1f}%"
          + "".join(f"{100*np.mean([r[b] for r in rows]):>7.1f}%" for b in BANDS))

    D = float(np.mean([r["dmax"] for r in rows]))
    print(f"\nwork ratio at mean D={D:.0f} (coarse D/8 plus a 2B+1 band):")
    print("        " + "".join(f"{('B=%d' % b):>8}" for b in BANDS))
    print("        " + "".join(f"{D/(D/8 + 2*b+1):>7.1f}x" for b in BANDS))
    print("\nCompare against what the shipping pipeline delivers today: 67.9% of")
    print("known pixels correct (75.6% coverage at 10.3% bad-1.0). A ceiling below")
    print("that number means coarse-to-fine cannot pay for itself at any speed.")


if __name__ == "__main__":
    main()
