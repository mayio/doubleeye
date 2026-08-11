#!/usr/bin/env python3
"""Score `core/tools/de_dense` on Middlebury v3, the way the leaderboard scores it.

Why this exists: every accuracy number in this project is on Middlebury 2003/2005
at 450x375, which no published method reports on any more. v3 is where the field
publishes, so "how far behind are we" was unanswerable. The training set ships its
ground truth, so this needs no submission -- it is a local benchmark.

THE TRAP THIS TOOL EXISTS TO AVOID. The official evaluation is at FULL resolution
and upsamples your result to get there; disparity scales with resolution, so a
quarter-res result has its disparities multiplied by 4 and its errors with them.
Scoring a Q result against Q ground truth at bad-1.0 is therefore NOT the
leaderboard's bad-1.0 -- it is roughly its bad-4.0. Measured here on the shipped
SGM reference: 13.03 the naive way against 37.3 published, a 2.9x flattery.
Middlebury's own README states the conversion (a threshold of 1.0 at F is 0.25 at
Q) and warns the converted number still differs slightly. So:

  --gt-full DIR   score against full-res GT, exactly as the board does. Requires
                  MiddEval3-data-F.zip and MiddEval3-GT0-F.zip (560 MB).
  (default)       score against same-res GT at threshold/scale, and SAY SO. On the
                  SGM fixture this reads ~2 points pessimistic.

Metric definitions are transcribed from the SDK's own code/evaldisp.cpp, not
guessed: `bad` counts wrong pixels over all masked pixels INCLUDING the ones the
method left empty, `invalid` is the empty fraction, and `totbad` is their sum. The
leaderboard's sparse table is `bad` and its dense table is a separately submitted
hole-filled file -- they are two results, not two metrics.

Validation (rule 3): the training data ships disp0SGM.pfm and disp0SGM_s.pfm, SGM's
own dense and sparse results, which are on the leaderboard. `--check` scores those
and compares against the published row. An evaluator that cannot reproduce a known
row is not evidence about MASDA.

    .venv/bin/python article/middeval3.py --check          # validate the evaluator
    .venv/bin/python article/middeval3.py --prepare        # PNG -> y8 + meta
    .venv/bin/python article/middeval3.py                  # run de_dense and score
"""

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "core", "build", "de_dense")

# Per-scene weights as published in the leaderboard header. They sum to 100 and
# are not uniform -- the lighting/exposure variants count half -- so a plain mean
# over scenes is NOT the board's number.
WEIGHTS = {"Adirondack": 8, "ArtL": 8, "Jadeplant": 8, "Motorcycle": 8,
           "MotorcycleE": 8, "Piano": 8, "PianoL": 4, "Pipes": 8, "Playroom": 4,
           "Playtable": 4, "PlaytableP": 8, "Recycle": 8, "Teddy": 8,
           "Shelves": 4, "Vintage": 4}

# Published training-set rows for the shipped SGM reference, read from
# vision.middlebury.edu/stereo/eval3 on 2026-08-10, bad 1.0 / nonocc / training.
PUBLISHED = {"Q": {"dense": 37.3, "sparse": 29.1},
             "H": {"dense": 28.2, "sparse": 16.4},
             "F": {"dense": 31.7, "sparse": 9.54}}


def read_pfm(path):
    with open(path, "rb") as f:
        if f.readline().rstrip() != b"Pf":
            sys.exit(f"{path}: not a greyscale PFM")
        w, h = (int(v) for v in f.readline().split())
        scale = float(f.readline())
        d = np.fromfile(f, "<f4" if scale < 0 else ">f4", w * h)
        if d.size != w * h:
            sys.exit(f"{path}: truncated, want {w*h} floats, got {d.size}")
    return d.reshape(h, w)[::-1]          # PFM rows run bottom-to-top


def read_calib(path):
    out = {}
    for line in open(path):
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def evaldisp(disp, gt, mask, badthresh, maxdisp, rounddisp=False):
    """Transcribed from MiddEval3/code/evaldisp.cpp.

    `disp` may be smaller than `gt` by an integer factor of 1, 2 or 4, which is
    how a quarter-res result is scored against full-res ground truth: nearest
    neighbour in position, multiplied in value.

    Order matters and is the SDK's: upscale, then clip to [0, scale*maxdisp],
    then round if the ground truth is integer. `maxdisp` is `ndisp` from the
    RESULT's calib, not the GT's -- runevalF is explicit about that. `rounddisp`
    is `isint` from the GT's calib, which is set on exactly the two scenes
    inherited from the older datasets, ArtL and Teddy.
    """
    H, W = gt.shape
    h2, w2 = disp.shape
    scale = W // w2
    if scale not in (1, 2, 4) or scale * w2 != W or scale * h2 != H:
        sys.exit(f"GT {W}x{H} must be exactly 1, 2 or 4 x disp {w2}x{h2}")

    d = np.repeat(np.repeat(disp, scale, 0), scale, 1) * scale
    valid = np.isfinite(d)
    maxd = scale * maxdisp
    d = np.where(valid, np.clip(d, 0, maxd), np.inf)
    if rounddisp:
        d = np.where(valid, np.round(d), np.inf)

    inmask = np.isfinite(gt) & (mask == 255)
    n = int(inmask.sum())
    if n == 0:
        sys.exit("empty evaluation mask")
    with np.errstate(invalid="ignore"):
        wrong = inmask & valid & (np.abs(d - gt) > badthresh)
    bad = int(wrong.sum())
    invalid = int((inmask & ~valid).sum())
    return dict(n=n, bad=100.0 * bad / n, invalid=100.0 * invalid / n,
                totbad=100.0 * (bad + invalid) / n)


def wavg(per, key):
    tot = sum(WEIGHTS[s] for s in per)
    return sum(WEIGHTS[s] * per[s][key] for s in per) / tot


def score_all(data, res, disp_of, thresh, gt_full=None):
    """disp_of(scene, dir) -> disparity array, or None to skip the scene."""
    per, skipped = {}, []
    for s in sorted(WEIGHTS):
        d = os.path.join(data, f"training{res}", s)
        disp = disp_of(s, d)
        if disp is None:
            skipped.append(s)
            continue
        # maxdisp from the RESULT's calib; isint from the GROUND TRUTH's.
        ndisp = int(read_calib(os.path.join(d, "calib.txt"))["ndisp"])
        g = os.path.join(gt_full, "trainingF", s) if gt_full else d
        gt = read_pfm(os.path.join(g, "disp0GT.pfm"))
        mask = _png(os.path.join(g, "mask0nocc.png"))
        isint = read_calib(os.path.join(g, "calib.txt")).get("isint", "0") == "1"
        per[s] = evaldisp(disp, gt, mask, thresh, ndisp, isint)
    return per, skipped


def _png(path):
    from PIL import Image
    return np.array(Image.open(path))


def official(res, gt_full):
    """(threshold, label) for the leaderboard's bad-1.0."""
    if gt_full:
        return 1.0, "bad-1.0 against full-res GT -- the board's own number"
    scale = {"F": 1, "H": 2, "Q": 4}[res]
    return 1.0 / scale, (f"bad-{1.0/scale:g} against {res} GT -- APPROXIMATE stand-in "
                         f"for the board's bad-1.0; reads ~2 points pessimistic")


def check(data, res, gt_full):
    thresh, label = official(res, gt_full)
    print(f"validating the evaluator against the shipped SGM reference\n{label}\n")
    print(f"{'file':<22}{'bad':>8}{'invalid':>9}{'totbad':>8}   published")
    ok = True
    for tag, fn, col in (("disp0SGM.pfm", "disp0SGM.pfm", "dense"),
                         ("disp0SGM_s.pfm", "disp0SGM_s.pfm", "sparse")):
        per, _ = score_all(data, res, lambda s, d, fn=fn: read_pfm(os.path.join(d, fn)),
                           thresh, gt_full)
        got, want = wavg(per, "bad"), PUBLISHED[res][col]
        note = f"{col} {want}"
        if gt_full:
            good = abs(got - want) <= 0.15
            note += "  MATCH" if good else f"  MISMATCH by {got-want:+.2f}"
            ok &= good
        else:
            note += f"  (approx, {got-want:+.2f})"
        print(f"{tag:<22}{got:>8.2f}{wavg(per,'invalid'):>9.2f}"
              f"{wavg(per,'totbad'):>8.2f}   {note}")
    if gt_full and not ok:
        sys.exit("\nevaluator does NOT reproduce the published row -- do not trust "
                 "any MASDA number it produces")
    print("\nevaluator reproduces the published SGM row." if gt_full else
          "\nrun with --gt-full for the exact check; this mode cannot confirm a match.")


def ceiling(data, res, gt_full):
    """Score the ground truth as if it were a result, float and integer.

    This is the interpretive key for every other number here. The official
    threshold is one pixel at FULL resolution, so at Q it is a quarter pixel of
    the disparities we actually compute -- and a matcher that emits integers
    cannot get inside that no matter how right it is. Scoring perfect answers
    isolates how much of the score is the metric rather than the matcher.
    """
    thresh, label = official(res, gt_full)
    print(f"scoring GROUND TRUTH as a result -- the floor imposed by output "
          f"quantisation alone\n{label}\n")
    for name, f in (("perfect, float", lambda g: g),
                    ("perfect, integer", np.round)):
        per, _ = score_all(data, res,
                           lambda s, d, f=f: f(read_pfm(os.path.join(d, "disp0GT.pfm"))),
                           thresh, gt_full)
        print(f"  {name:<18} bad={wavg(per,'bad'):6.2f}  invalid={wavg(per,'invalid'):5.2f}")
    print("\nThe gap between those two rows is charged to any matcher that emits\n"
          "integer disparities, before a single match is judged.")


def prepare(data, res):
    """PNG -> the y8 + meta.txt layout de_dense already reads."""
    from PIL import Image
    for s in sorted(WEIGHTS):
        d = os.path.join(data, f"training{res}", s)
        c = read_calib(os.path.join(d, "calib.txt"))
        for src, dst in (("im0.png", "left.y8"), ("im1.png", "right.y8")):
            g = Image.open(os.path.join(d, src)).convert("L")
            np.asarray(g, np.uint8).tofile(os.path.join(d, dst))
        open(os.path.join(d, "meta.txt"), "w").write(
            f"{c['width']} {c['height']} {c['ndisp']}\n")
        print(f"{s:<13} {c['width']}x{c['height']}  ndisp={c['ndisp']}")


def by_gradient(a, run):
    """Where does the error actually live, by ground-truth disparity slope?

    0.35 ranks edge-aware support ahead of slanted planes on the grounds that
    high-gradient regions carry about half of all error, measured on two scenes at
    450x375 scored at native resolution. This re-tests that premise on v3 at the
    official tolerance, because a fronto-parallel window's failure is exactly the
    kind of thing a quarter-pixel threshold sees and a one-pixel one does not.

    The gradient is computed on the GROUND TRUTH, so the buckets do not depend on
    what the matcher did. >0.6 px/px is a depth discontinuity rather than a slant --
    0.35's own reading -- so the two are reported apart.
    """
    import collections
    tot, bad = collections.Counter(), collections.Counter()
    for s in sorted(WEIGHTS):
        d = os.path.join(a.data, f"training{a.res}", s)
        disp = run(s, d)
        if disp is None:
            continue
        g = os.path.join(a.gt_full, "trainingF", s) if a.gt_full else d
        gt = read_pfm(os.path.join(g, "disp0GT.pfm"))
        mask = _png(os.path.join(g, "mask0nocc.png"))
        ndisp = int(read_calib(os.path.join(d, "calib.txt"))["ndisp"])
        isint = read_calib(os.path.join(g, "calib.txt")).get("isint", "0") == "1"
        H, W = gt.shape
        sc = W // disp.shape[1]
        up = np.repeat(np.repeat(disp, sc, 0), sc, 1) * sc
        valid = np.isfinite(up)
        up = np.where(valid, np.clip(up, 0, sc * ndisp), np.inf)
        if isint:
            up = np.where(valid, np.round(up), np.inf)
        gv = np.isfinite(gt)
        gy, gx = np.gradient(np.where(gv, gt, 0.0))
        # NO division by the scale factor. Disparity gradient is scale-invariant:
        # at Q the disparities are a quarter and one pixel spans four, so the ratio
        # is unchanged. Dividing by sc made every slope four times too shallow and
        # put 98.4% of pixels in the flattest bucket.
        grad = np.maximum(np.abs(gy), np.abs(gx))        # disparity px per image px
        with np.errstate(invalid="ignore"):
            wrong = valid & (np.abs(up - gt) > 1.0)
        use = gv & (mask == 255) & valid       # error rate over what we filled
        # Distance to the nearest discontinuity is the population edge-aware
        # support actually addresses: a pixel two pixels from a depth edge has its
        # support window straddling that edge even though its own slope is flat.
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(~(grad > 0.6))
        for name, sel in (("slope <= 0.3", grad <= 0.3),
                          ("slope 0.3-0.6 (slanted)", (grad > 0.3) & (grad <= 0.6)),
                          ("slope > 0.6 (discontinuity)", grad > 0.6),
                          ("within 2px of an edge", dist <= 2),
                          ("within 8px of an edge", dist <= 8),
                          ("further than 8px", dist > 8)):
            m = sel & use
            tot[name] += int(m.sum())
            bad[name] += int((m & wrong).sum())
        tot["ALL"] += int(use.sum())
        bad["ALL"] += int((use & wrong).sum())
    print("\nWhere the error lives, by ground-truth disparity slope "
          "(bad-1.0 over FILLED pixels)\n")
    print(f"{'bucket':<30}{'pixels':>9}{'error':>9}{'share of error':>16}")
    for k in ["slope <= 0.3", "slope 0.3-0.6 (slanted)",
              "slope > 0.6 (discontinuity)", "within 2px of an edge",
              "within 8px of an edge", "further than 8px", "ALL"]:
        print(f"{k:<30}{100*tot[k]/max(1,tot['ALL']):>8.1f}%"
              f"{100*bad[k]/max(1,tot[k]):>8.1f}%"
              f"{100*bad[k]/max(1,bad['ALL']):>15.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MIDDEVAL3", ""),
                    help="MiddEval3 directory (or set $MIDDEVAL3)")
    ap.add_argument("--gt-full", default=None,
                    help="MiddEval3 dir holding trainingF/ -- enables official scoring")
    ap.add_argument("--res", default="Q", choices=["Q", "H", "F"])
    ap.add_argument("--check", action="store_true", help="validate against SGM and exit")
    ap.add_argument("--ceiling", action="store_true",
                    help="score the ground truth itself, float and integer, and exit")
    ap.add_argument("--prepare", action="store_true", help="write y8+meta and exit")
    ap.add_argument("--by-gradient", action="store_true",
                    help="report where the error lives, by ground-truth slope")
    ap.add_argument("--dmax-cap", type=int, default=96,
                    help="skip scenes needing more; de_dense is built for D<=96")
    ap.add_argument("--threads", default="4")
    # These three used to carry values here, which silently OVERRODE the binary's
    # own defaults -- so every "default configuration" run measured the harness's
    # opinion instead of what ships, and an --iters default change read as no change
    # at all. Passed through only when given.
    ap.add_argument("--agg", default="5")
    ap.add_argument("--iters", default=None)
    ap.add_argument("--min-margin", default="0.01")
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    if not a.data:
        sys.exit("need --data pointing at a MiddEval3 directory (or $MIDDEVAL3)")

    if a.prepare:
        return prepare(a.data, a.res)
    if a.check:
        return check(a.data, a.res, a.gt_full)
    if a.ceiling:
        return ceiling(a.data, a.res, a.gt_full)

    if not os.path.exists(BIN):
        sys.exit(f"{BIN} not built -- run `cd core && make`")
    args = ["--threads", a.threads, "--agg", a.agg,
            "--min-margin", a.min_margin] + a.extra.split()
    if a.iters is not None:
        args += ["--iters", a.iters]
    over = []

    def run(s, d):
        c = read_calib(os.path.join(d, "calib.txt"))
        W, H, ndisp = int(c["width"]), int(c["height"]), int(c["ndisp"])
        if ndisp > a.dmax_cap:
            # Rule 1: say what was dropped. A silently skipped scene reads as
            # coverage of the whole set.
            over.append((s, ndisp))
            return None
        for src, dst in (("im0.png", "left.y8"), ("im1.png", "right.y8")):
            if not os.path.exists(os.path.join(d, dst)):
                sys.exit(f"{d}/{dst} missing -- run --prepare first")
        out = os.path.join(d, ".middeval3_tmp.f32")
        p = subprocess.run([BIN, os.path.join(d, "left.y8"), os.path.join(d, "right.y8"),
                            str(W), str(H), "--dmax", str(ndisp), "--out", out] + args,
                           capture_output=True, text=True)
        if p.returncode != 0:
            sys.exit(f"de_dense failed on {s} (exit {p.returncode}):\n{p.stderr}")
        disp = np.fromfile(out, np.float32).reshape(H, W)
        os.remove(out)
        # de_dense marks "no match" as a non-positive disparity; the evaluator
        # wants INF, which is what Middlebury means by an unfilled pixel.
        return np.where(disp > 0, disp, np.inf)

    if a.by_gradient:
        return by_gradient(a, run)
    thresh, label = official(a.res, a.gt_full)
    print(f"de_dense --agg {a.agg} --min-margin {a.min_margin} --threads {a.threads}"
          f"{'' if a.iters is None else ' --iters ' + a.iters} {a.extra}\n{label}\n")
    per, _ = score_all(a.data, a.res, run, thresh, a.gt_full)
    if not per:
        sys.exit("no scene scored")

    print(f"{'scene':<13}{'bad':>8}{'invalid':>9}{'totbad':>8}")
    for s in sorted(per):
        r = per[s]
        print(f"{s:<13}{r['bad']:>8.2f}{r['invalid']:>9.2f}{r['totbad']:>8.2f}")
    print(f"\nweighted average over {len(per)} of {len(WEIGHTS)} scenes:")
    print(f"  bad (sparse table)    {wavg(per,'bad'):6.2f}")
    print(f"  invalid               {wavg(per,'invalid'):6.2f}   "
          f"= {100-wavg(per,'invalid'):.1f}% coverage")
    print(f"  totbad (dense table)  {wavg(per,'totbad'):6.2f}")
    if over:
        print("\nSKIPPED, disparity range beyond --dmax-cap "
              f"{a.dmax_cap} -- the average above is NOT over the full set:")
        for s, n in over:
            print(f"  {s:<13} needs ndisp {n}")


if __name__ == "__main__":
    main()
