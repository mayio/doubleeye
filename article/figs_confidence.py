#!/usr/bin/env python3
"""Figures for the per-pixel confidence post.

Every figure is generated from the shipping binary, so none of them can drift from
what the matcher actually does.

    .venv/bin/python article/figs_confidence.py
"""

import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import confidence as C                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "core", "build", "de_dense")
BLOG = os.path.expanduser("~/src/mayio.github.io/assets/img")
OUT = os.path.join(BLOG, "2026-08-13-Stereo-Confidence_files")

INK = "#1b1b1b"
GOOD = "#2f6f4f"
COOL = "#2b5d8a"
WARN = "#a4442c"
MUTE = "#8a8a8a"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "figure.dpi": 130,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})

LAM = C.LAMBDA
WT, BT = 7.936751, 0.692684


def model(s1, s2, lrc=None):
    """The shipping model: the peak ratio, capped where the reverse match objects."""
    alt = np.where(s2 > -1e29, np.maximum(s2, LAM), LAM)
    x = np.log(np.maximum((1 - alt) / np.maximum(1 - s1, 1e-6), 1e-6))
    p = 1 / (1 + np.exp(-np.clip(BT + WT * x, -30, 30)))
    if lrc is not None:
        p = np.where(np.nan_to_num(lrc, nan=0.0) > 1.0, np.minimum(p, 0.35), p)
    return p


def load(name):
    """The SHIPPING configuration. `lrc=False` leaves the exact reverse match unasked
    for and returns the cheap one the solver computes from its own candidate buckets,
    which is the cue de_dense_cuda sends to the live cloud. Measuring the exact
    version here would put numbers in the post that the deployed code does not
    produce."""
    disp, gt, s1, s2, d1, d2, r = C.run_scene(name, lrc=False)
    return disp, gt, s1, s2, r


# ------------------------------------------------------------- the sparsification curve
def fig_curve(path):
    """The metric itself: throw away the least confident and watch the error fall."""
    conf, bad = [], []
    for n in C.scenes():
        disp, gt, s1, s2, r = load(n)
        k = (gt > 0) & np.isfinite(disp)
        conf.append(model(s1[k], s2[k], r[k]))
        bad.append((np.abs(disp - gt)[k] > 1.0).astype(float))
    conf, bad = np.concatenate(conf), np.concatenate(bad)
    rng = np.random.default_rng(0)
    q = np.linspace(1.0, 0.05, 80)

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.plot(100 * q, 100 * C.curve(conf, bad, rng, q), color=COOL, lw=2,
            label="peak ratio, capped by the reverse match")
    ax.plot(100 * q, 100 * C.curve(-bad, bad, rng, q), color=GOOD, lw=1.6, ls="--",
            label="oracle: the wrong points removed first")
    ax.plot(100 * q, 100 * C.curve(rng.random(bad.size), bad, rng, q), color=MUTE,
            lw=1.6, ls=":", label="no confidence: points removed at random")
    ax.set_xlabel("percentage of answered points kept, most confident first")
    ax.set_ylabel("wrong by more than 1 disparity, %")
    ax.set_xlim(100, 5)
    ax.set_ylim(0, 11.5)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="center right")
    ax.set_title("Eight Middlebury scenes, 1,077,892 points with ground truth")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------------------ the confidence map
def fig_map(path):
    """Does the confidence point at the same places the answer is actually wrong?

    Shown as two masks rather than as a colour map of the value. The distribution is
    the reason: nine points in ten score above 0.95, so a linear colour scale paints
    almost the whole frame one colour and hides exactly the structure worth seeing.
    Marking the least confident fifth against the points that are actually wrong asks
    the question directly, and the two pictures can be compared by eye.
    """
    name = "Art"
    disp, gt, s1, s2, r = load(name)
    W, H, dmax = (int(v) for v in
                  open(os.path.join(C.DATA, name, "meta.txt")).read().split())
    left = np.fromfile(os.path.join(C.DATA, name, "left.y8"),
                       np.uint8).reshape(H, W)
    ok = np.isfinite(disp)
    p = np.where(ok, model(s1, s2, r), np.nan)
    thr = np.nanquantile(p, 0.20)
    low = ok & (p <= thr)
    wrong = ok & (gt > 0) & (np.abs(disp - gt) > 1.0)

    def overlay(mask, colour):
        base = np.repeat((left * 0.55 + 40).astype(np.uint8)[:, :, None], 3, axis=2)
        base[~ok] = (16, 16, 16)
        base[mask] = colour
        return base

    fig, ax = plt.subplots(1, 3, figsize=(8.6, 2.6))
    ax[0].imshow(left, cmap="gray")
    ax[0].set_title("left camera", fontsize=9)
    ax[1].imshow(overlay(low, (196, 78, 44)))
    ax[1].set_title(f"the least confident 20% —\nwhat the matcher says to distrust",
                    fontsize=9)
    ax[2].imshow(overlay(wrong, (196, 78, 44)))
    ax[2].set_title(f"actually wrong by over 1 disparity —\n"
                    f"{100*wrong.sum()/ok.sum():.0f}% of answered points", fontsize=9)
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.subplots_adjust(wspace=0.04)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# --------------------------------------------------- the cue that cannot see itself
def fig_split(path):
    """Where the reverse match fires, and why it cannot be fitted as a weight."""
    S1, S2, R, Y = [], [], [], []
    for n in C.scenes():
        disp, gt, s1, s2, r = load(n)
        k = (gt > 0) & np.isfinite(disp)
        S1.append(s1[k]); S2.append(s2[k]); R.append(r[k])
        Y.append((np.abs(disp - gt)[k] <= 1.0))
    s1 = np.concatenate(S1); s2 = np.concatenate(S2)
    r = np.concatenate(R); y = np.concatenate(Y)
    alt = np.where(s2 > -1e29, np.maximum(s2, LAM), LAM)
    ratio = np.log(np.maximum((1 - alt) / np.maximum(1 - s1, 1e-6), 1e-6))
    flag = np.nan_to_num(r, nan=2.0) > 1.0
    qs = np.quantile(ratio, [0, .25, .5, .75, .9, 1.0])

    lab, ok_c, ok_f, n_f = [], [], [], []
    for i in range(5):
        m = (ratio >= qs[i]) & (ratio <= qs[i + 1])
        a, b = m & ~flag, m & flag
        lab.append(f"{qs[i]:.2f}–{qs[i+1]:.2f}")
        ok_c.append(100 * y[a].mean())
        ok_f.append(100 * y[b].mean() if b.sum() else np.nan)
        n_f.append(int(b.sum()))

    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    xs = np.arange(5)
    ax.bar(xs - 0.19, ok_c, 0.38, color=COOL, label="reverse match agrees")
    ax.bar(xs + 0.19, ok_f, 0.38, color=WARN, label="reverse match disagrees")
    for i in range(5):
        if n_f[i]:
            ax.text(xs[i] + 0.19, ok_f[i] + 2.5, f"n={n_f[i]:,}", ha="center",
                    fontsize=7, color=WARN)
        else:
            ax.text(xs[i] + 0.19, 2.5, "no such\npoints", ha="center", fontsize=7,
                    color=WARN)
    ax.set_xticks(xs)
    ax.set_xticklabels(lab)
    ax.set_xlabel("peak ratio, by quantile (weakest fifth on the left)")
    ax.set_ylabel("points correct within 1 disparity, %")
    ax.set_ylim(0, 116)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    ax.set_title("The reverse match never fires where the ratio is strong —\n"
                 "so a fitted weight has no evidence for the case it exists for",
                 fontsize=9.5)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------- what the number does not know
def fig_light(path):
    """Six captures of one room: scene quality moves, the confidence does not."""
    names = ["night_off", "night_on", "evening_off", "evening_on",
             "morning_off", "morning_on"]
    pretty = ["night\nno projector", "night\nprojector", "evening\nno projector",
              "evening\nprojector", "daylight\nno projector", "daylight\nprojector"]
    W, H = 848, 480
    mc, mb = [], []
    for n in names:
        d = os.path.join(ROOT, "article", "scenes", n)
        dp, cp = "/tmp/_c.f32", "/tmp/_c.conf"
        p = subprocess.run([BIN, f"{d}/left.y8", f"{d}/right.y8", str(W), str(H),
                            "--dmax", "64", "--threads", "6", "--min-margin", "0",
                            "--lrc", "--out", dp, "--out-conf", cp],
                           capture_output=True, text=True)
        if p.returncode != 0:
            sys.exit(f"de_dense failed on {n}:\n{p.stderr}")
        disp = np.fromfile(dp, np.float32).reshape(H, W)
        s1, s2, d1, d2, r, r2 = np.fromfile(cp, np.float32).reshape(6, H, W)
        ok = np.isfinite(disp)
        # r2 is the shipping cue; r is the exact one and is only used as the stand-in
        # for truth here, since these captures have no ground truth.
        mc.append(model(s1[ok], s2[ok], r2[ok]).mean())
        mb.append((np.nan_to_num(r[ok], nan=99.) > 1.0).mean())

    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    xs = np.arange(len(names))
    ax.bar(xs - 0.19, 100 * np.array(mb), 0.38, color=WARN,
           label="points the reverse match rejects")
    ax.bar(xs + 0.19, 100 * np.array(mc), 0.38, color=COOL,
           label="mean confidence reported")
    for i in range(len(names)):
        ax.text(xs[i] - 0.19, 100 * mb[i] + 1.5, f"{100*mb[i]:.1f}", ha="center",
                fontsize=7, color=WARN)
        ax.text(xs[i] + 0.19, 100 * mc[i] + 1.5, f"{100*mc[i]:.0f}", ha="center",
                fontsize=7, color=COOL)
    ax.set_xticks(xs)
    ax.set_xticklabels(pretty, fontsize=7.5)
    ax.set_ylabel("per cent")
    ax.set_ylim(0, 96)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.14))
    # Stated from the numbers just measured, so the title cannot outlive them.
    fac = max(mb) / min(mb)
    ax.set_title(f"One kitchen, three light levels: the share of points the frame "
                 f"gets wrong\nmoves by a factor of {fac:.0f}, and the confidence it "
                 f"reports by {100*(max(mc)-min(mc)):.0f} points", fontsize=9.5)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------------------- thumbnail
def fig_thumb(path):
    disp, gt, s1, s2, r = load("Art")
    W, H, dmax = (int(v) for v in
                  open(os.path.join(C.DATA, "Art", "meta.txt")).read().split())
    left = np.fromfile(os.path.join(C.DATA, "Art", "left.y8"),
                       np.uint8).reshape(H, W)
    ok = np.isfinite(disp)
    p = np.where(ok, model(s1, s2, r), np.nan)
    low = ok & (p <= np.nanquantile(p, 0.20))
    base = np.repeat((left * 0.55 + 40).astype(np.uint8)[:, :, None], 3, axis=2)
    base[~ok] = (16, 16, 16)
    base[low] = (196, 78, 44)
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    ax.imshow(base[40:335, 60:440])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("wrote", path)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_curve(os.path.join(OUT, "sparsification.png"))
    fig_map(os.path.join(OUT, "confidence_map.png"))
    fig_split(os.path.join(OUT, "reverse_split.png"))
    fig_light(os.path.join(OUT, "light_levels.png"))
    fig_thumb(os.path.join(OUT, "thumb_conf.png"))


if __name__ == "__main__":
    main()
