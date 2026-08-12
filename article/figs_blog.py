#!/usr/bin/env python3
"""Regenerate every figure in the MASDA stereo posts.

One script so the posts cannot drift from the matcher: it runs the shipping binary,
scores it against ground truth, and writes the images the posts embed. Nothing here
is hand-drawn in an external tool, and nothing is cached -- if the matcher changes,
re-running this changes the pictures.

    .venv/bin/python article/figs_blog.py [--out DIR] [--skip-run]

Writes into the blog repo's asset directories by default.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "core", "build", "de_dense")
CB = os.path.join(HERE, "data", "c_bench")
BLOG = os.path.expanduser("~/src/mayio.github.io/assets/img")
P2 = os.path.join(BLOG, "2026-08-08-Dense-MASDA_files")
P3 = os.path.join(BLOG, "2026-08-09-Realtime-Dense-MASDA_files")

INK = "#1b1b1b"
GPU = "#2f6f4f"
CPU = "#2b5d8a"
WARN = "#a4442c"
MUTE = "#8a8a8a"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "figure.dpi": 130,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})


def run_dense(left, right, w, h, dmax, extra=(), threads="8"):
    out = "/tmp/_fig.f32"
    cmd = [BIN, left, right, str(w), str(h), "--dmax", str(dmax),
           "--threads", threads, "--agg", "5", "--out", out] + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"de_dense failed: {' '.join(cmd)}\n{p.stderr}")
    d = np.fromfile(out, np.float32).reshape(h, w)
    os.remove(out)
    return d


def scene(name):
    d = os.path.join(CB, name)
    w, h, dmax = (int(v) for v in open(os.path.join(d, "meta.txt")).read().split())
    L = np.fromfile(os.path.join(d, "left.y8"), np.uint8).reshape(h, w)
    R = np.fromfile(os.path.join(d, "right.y8"), np.uint8).reshape(h, w)
    gt = np.fromfile(os.path.join(d, "disp.f32"), np.float32).reshape(h, w)
    return dict(name=name, dir=d, w=w, h=h, dmax=dmax, L=L, R=R, gt=gt)


def show_disp(ax, d, vmax, title=None):
    m = np.ma.masked_invalid(np.where(d > 0, d, np.nan))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#101010")
    ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)


# ---------------------------------------------------------------- Middlebury grid
def fig_middlebury(names, path, run=True):
    fig, axes = plt.subplots(len(names), 4, figsize=(9.0, 2.05 * len(names)))
    for i, n in enumerate(names):
        s = scene(n)
        d = run_dense(os.path.join(s["dir"], "left.y8"),
                      os.path.join(s["dir"], "right.y8"),
                      s["w"], s["h"], s["dmax"]) if run else None
        vmax = float(np.percentile(s["gt"][s["gt"] > 0], 99))
        axes[i, 0].imshow(s["L"], cmap="gray"); axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        axes[i, 0].set_ylabel(n, fontsize=9)
        show_disp(axes[i, 1], s["gt"], vmax)
        show_disp(axes[i, 2], d, vmax)
        known = (s["gt"] > 0) & np.isfinite(d) & (d > 0)
        err = np.full(s["gt"].shape, np.nan)
        err[known] = np.abs(d[known] - s["gt"][known])
        ecm = plt.get_cmap("inferno").copy()
        ecm.set_bad("#101010")          # same "no answer" ink as the disparity maps
        im = axes[i, 3].imshow(np.ma.masked_invalid(err), cmap=ecm, vmin=0, vmax=3)
        axes[i, 3].set_xticks([]); axes[i, 3].set_yticks([])
        if i == 0:
            for j, t in enumerate(["left image", "ground truth", "dense MASDA",
                                   "|error|, 0-3 px"]):
                axes[0, j].set_title(t)
    fig.colorbar(im, ax=axes[:, 3].tolist(), fraction=0.03, pad=0.02, label="px")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------------------- the real pair
def fig_real(path):
    """One room at three light levels, with the projector on and off.

    The axis that matters is how much ambient light there is, not an on/off binary:
    the projector's value is what the room does not already provide. Exposure is set
    per condition to hold 3-5 DN of local contrast, because a fixed exposure across a
    4.5x range of scene brightness would measure the exposure instead.
    """
    # article/scenes, not bags/: bags is scratch that --capture overwrites by name.
    cols = [("night, no lamp", "night_on", "night_off", 4000),
            ("evening, lamp only", "evening_on", "evening_off", 2500),
            ("morning, lamp + daylight", "morning_on", "morning_off", 1500)]
    fig, axes = plt.subplots(3, 3, figsize=(9.8, 6.4))
    for c, (label, on, off, us) in enumerate(cols):
        for r, bag in enumerate((on, on, off)):
            fr = os.path.join(ROOT, "article", "scenes", bag)
            lp, rp = os.path.join(fr, "left.y8"), os.path.join(fr, "right.y8")
            ax = axes[r, c]
            if r == 0:
                L = np.fromfile(lp, np.uint8).reshape(480, 848)
                lo, hi = np.percentile(L, (1, 99))
                ax.imshow(np.clip((L - lo) / max(hi - lo, 1), 0, 1), cmap="gray",
                          vmin=0, vmax=1)
                ax.set_title(f"{label}\n{us} us, mean {L.mean():.0f} DN", fontsize=8.5)
            else:
                d = run_dense(lp, rp, 848, 480, 64)
                show_disp(ax, d, 64)
                cov = 100.0 * ((d > 0) & np.isfinite(d)).mean()
                ax.set_xlabel(f"{cov:.1f}% answered", fontsize=8.5,
                              color=WARN if cov < 60 else INK)
            ax.set_xticks([]); ax.set_yticks([])
    for r, lab in enumerate(("left infrared\n(projector ON)", "disparity\nprojector ON",
                             "disparity\nprojector off")):
        axes[r, 0].set_ylabel(lab, fontsize=8.5)
    fig.suptitle("One room, three light levels: what the projector is for", y=0.995,
                 fontsize=10)
    fig.text(0.5, 0.005, "infrared panels are contrast-stretched for display; the mean "
                         "level above each is the real one",
             ha="center", fontsize=7.4, color=MUTE)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------------ the sub-pixel figure
def fig_subpixel(path):
    s = scene("teddy")
    dint = run_dense(os.path.join(s["dir"], "left.y8"), os.path.join(s["dir"], "right.y8"),
                     s["w"], s["h"], s["dmax"], extra=["--no-subpixel"])
    dsub = run_dense(os.path.join(s["dir"], "left.y8"), os.path.join(s["dir"], "right.y8"),
                     s["w"], s["h"], s["dmax"])
    fig = plt.figure(figsize=(9.4, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.25], wspace=0.32)

    # (a) the parabola through three aggregated costs
    ax = fig.add_subplot(gs[0, 0])
    k = np.array([-1, 0, 1]); c = np.array([0.61, 0.83, 0.72])
    xx = np.linspace(-1.35, 1.35, 200)
    a, b, cc = np.polyfit(k, c, 2)
    ax.plot(xx, a * xx**2 + b * xx + cc, color=CPU, lw=1.6)
    ax.plot(k, c, "o", color=INK, ms=5, zorder=3)
    vx = -b / (2 * a)
    ax.axvline(0, color=MUTE, ls=":", lw=1)
    ax.axvline(vx, color=WARN, ls="--", lw=1.4)
    ax.annotate("integer\nargmax", (0, 0.53), ha="center", fontsize=8, color=MUTE)
    ax.annotate(f"fitted vertex {vx:+.2f} px", (vx + 0.06, 0.90), ha="left",
                fontsize=7.6, color=WARN)
    ax.set_xlabel("disparity offset from the winner (px)")
    ax.set_ylabel("aggregated score")
    ax.set_title("(a) the fit")
    ax.set_ylim(0.5, 0.95)

    # (b) what comes out
    ax = fig.add_subplot(gs[0, 1])
    for d, lab, col in ((dint, "integer", MUTE), (dsub, "sub-pixel", CPU)):
        v = d[np.isfinite(d) & (d > 0)]
        ax.hist(np.abs(v - np.round(v)), bins=40, range=(0, 0.5), histtype="step",
                lw=1.6, color=col, label=lab, density=True)
    ax.set_xlabel("|fractional part| of the output")
    ax.set_yticks([])
    ax.set_title("(b) the output")
    ax.legend(frameon=False, fontsize=8)

    # (c) the measurement that decides it, on the benchmark that can see it
    ax = fig.add_subplot(gs[0, 2])
    labs = ["perfect\nfloat", "sub-pixel\n(ships)", "integer", "perfect\ninteger"]
    vals = [0.80, 24.47, 41.53, 45.55]
    cols = [GPU, CPU, MUTE, WARN]
    ax.bar(range(4), vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.2, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=7.6)
    ax.set_ylabel("bad-1.0 (%)")
    ax.set_ylim(0, 54)
    ax.set_title("(c) Middlebury v3, 15 scenes")
    ax.annotate("an integer answer cannot\ndo better than this",
                xy=(3, 45.55), xytext=(1.35, 49), fontsize=7.4, color=WARN,
                arrowprops=dict(arrowstyle="->", color=WARN, lw=0.9))
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# --------------------------------------------------------- precision/coverage curve
def fig_curve(path):
    # measured, article/middeval3.py, 15 v3 training scenes, --threads 1
    gate = [0.0, 0.005, 0.01, 0.03, 0.10]
    cov = [89.4, 84.6, 79.6, 64.8, 35.7]
    bad = [31.30, 27.81, 24.47, 16.33, 6.17]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(cov, bad, "-o", color=CPU, lw=1.8, ms=4, label="dense MASDA, margin gate swept")
    for g, c, b in zip(gate, cov, bad):
        if g in (0.0, 0.01, 0.10):
            ax.annotate(f"gate {g:g}", (c, b), textcoords="offset points",
                        xytext=(6, -9), fontsize=7.5, color=CPU)
    ax.plot([90.2], [29.08], "s", color=WARN, ms=7, zorder=4)
    ax.annotate("SGM reference\n(Middlebury's own)", (90.2, 29.08),
                textcoords="offset points", xytext=(-96, 6), fontsize=8, color=WARN)
    ax.set_xlabel("coverage: % of pixels answered")
    ax.set_ylabel("bad-1.0 (%) over answered pixels")
    ax.set_title("Middlebury v3, 15 training scenes, official scoring")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------- algorithm: what aggregation is
def dump_vol(sc, sigma_s):
    """The aggregated cost volume at a given reach. sigma_s -> 0 is 'not aggregated':
    the coefficient is exp(-sqrt(2)/sigma_s * ...), which underflows to zero, and the
    recurrence then keeps each pixel's own score."""
    volp = "/tmp/_agg_vol.f32"
    cmd = [BIN, os.path.join(sc["dir"], "left.y8"), os.path.join(sc["dir"], "right.y8"),
           str(sc["w"]), str(sc["h"]), "--dmax", str(sc["dmax"]), "--threads", "8",
           "--sigma-s", str(sigma_s), "--dump-vol", volp, "--out", "/tmp/_agg.f32"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"de_dense failed:\n{p.stderr}")
    v = np.fromfile(volp, np.float32).reshape(sc["h"], sc["w"], sc["dmax"])
    os.remove(volp); os.remove("/tmp/_agg.f32")
    return v


def fig_aggregate(path):
    """What the aggregation does, on real curves rather than a sketch."""
    sc = scene("teddy")
    raw = dump_vol(sc, 0.05)
    agg = dump_vol(sc, 8.0)
    live = raw > -1e29
    gt = sc["gt"]
    d_raw = np.argmax(np.where(live, raw, -1e30), -1) + 1.0
    d_agg = np.argmax(np.where(live, agg, -1e30), -1) + 1.0
    ok = (gt > 3) & (live.sum(-1) > 20)
    # A pixel the per-pixel score gets wrong and the aggregated score gets right.
    # Among those, take the one whose aggregated peak stands out most -- chosen by
    # argmax rather than by eye, but still an illustration of the mechanism and not
    # a statistic about how often it happens.
    pick = ok & (np.abs(d_raw - gt) > 2.0) & (np.abs(d_agg - gt) <= 1.0)
    A = np.where(live, agg, -1e30)
    srt = np.sort(A, -1)
    span = srt[..., -1] - np.where(live, agg, 1e30).min(-1)
    prom = np.where(span > 0, (srt[..., -1] - srt[..., -2]) / np.maximum(span, 1e-6), 0)
    prom = np.where(pick, prom, -1)
    py, px = np.unravel_index(int(np.argmax(prom)), prom.shape)
    py, px = int(py), int(px)

    fig = plt.figure(figsize=(10.2, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.32)

    # (a) the recurrence, drawn
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 10); ax.set_ylim(0, 6.6); ax.axis("off")
    ax.set_title("(a) one pass of the recurrence", fontsize=9.5)
    vals = ["C1", "C2", "C3", "C4", "C5", "C6"]
    for i, v in enumerate(vals):
        edge = (i == 3)
        box(ax, 0.3 + i * 1.58, 3.6, 1.30, 0.85, v,
            fc="#f6d9d0" if edge else "#e8f1ea", ec=WARN if edge else GPU, fs=8)
        if i:
            a = "a=0" if edge else "a"
            ax.annotate("", xy=(0.3 + i * 1.58, 4.02), xytext=(0.3 + i * 1.58 - 0.28, 4.02),
                        arrowprops=dict(arrowstyle="-|>", lw=1.6 if not edge else 0.7,
                                        color=MUTE if not edge else WARN))
            ax.text(0.3 + i * 1.58 - 0.14, 4.42, a, ha="center", fontsize=7,
                    color=INK if not edge else WARN)
    ax.text(5.0, 2.85, "an intensity edge sets a to 0 and cuts the chain",
            ha="center", fontsize=7.6, color=WARN)
    ax.text(5.0, 1.95, "$F_x = (1-a_x)\\,C_x + a_x F_{x-1}$", ha="center", fontsize=10)
    ax.text(5.0, 1.15, "one multiply-add per pixel, per pass.\n"
                       "four passes: left, right, down, up", ha="center", fontsize=7.6)

    # (b) the cost curve at one pixel, before and after
    ax = fig.add_subplot(gs[1])
    d = np.arange(1, sc["dmax"] + 1)
    m = live[py, px]
    for v, lab, col, lw in ((raw[py, px], "per-pixel score", MUTE, 1.0),
                            (agg[py, px], "aggregated", GPU, 1.8)):
        z = np.where(m, v, np.nan)
        z = (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z))
        ax.plot(d, z, lw=lw, color=col, label=lab)
        k = int(np.nanargmax(z))
        ax.plot([d[k]], [z[k]], "o", ms=5, color=col)
    ax.axvline(gt[py, px], color=WARN, ls="--", lw=1.2)
    ax.text(gt[py, px] + 1.2, 0.04, "true d", color=WARN, fontsize=7.5)
    ax.set_title(f"(b) score vs disparity, one pixel", fontsize=9.5)
    ax.set_xlabel("disparity $d$ (px)"); ax.set_ylabel("score (normalised)")
    ax.legend(fontsize=7.2, loc="upper left"); ax.grid(alpha=0.25, lw=0.6)

    # (c) the coefficient in closed form: what the two knobs actually control
    ax = fig.add_subplot(gs[2])
    dI = np.linspace(0, 60, 400)
    for sig_r, ls, lab in ((0.1, ":", "$\\sigma_r=0.1$"),
                           (0.2, "-", "$\\sigma_r=0.2$ (ships)"),
                           (0.4, "--", "$\\sigma_r=0.4$")):
        a = np.exp(-np.sqrt(2) / 8.0 * (1.0 + (8.0 / sig_r) * dI / 255.0))
        ax.plot(dI, a, ls, lw=1.5, color=GPU, label=lab)
    flat = float(np.exp(-np.sqrt(2) / 8.0))
    ax.axhline(flat, color=MUTE, lw=1.0)
    ax.text(1.5, flat - 0.07, f"flat region: $a={flat:.2f}$", ha="left", fontsize=7.4,
            color=MUTE)
    ax.set_title("(c) what the two knobs do, $\\sigma_s=8$", fontsize=9.5)
    ax.set_xlabel("intensity step $|\\Delta I|$ between neighbours (DN)")
    ax.set_ylabel("coefficient $a$")
    ax.set_ylim(0, 1.0); ax.set_xlim(0, 60)
    ax.legend(fontsize=7.2, loc="upper right"); ax.grid(alpha=0.25, lw=0.6)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path, f"(pixel {px},{py} of teddy)")


# ------------------------------------------------------- algorithm: no cost volume
def box(ax, x, y, w, h, label, fc="#ffffff", ec=INK, fs=8, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, color=color, lw=lw,
                                 shrinkA=1, shrinkB=1))


def fig_novolume(path):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 7.2); ax.axis("off")

    ax = axes[0]
    ax.set_title("the textbook pipeline: build the volume", fontsize=9.5)
    for i in range(4):
        ax.add_patch(Rectangle((0.7 + i * 0.34, 4.0 - i * 0.20), 2.3, 1.5,
                               fc="#eef2f6", ec=INK, lw=0.9))
    ax.text(2.0, 6.0, "W x H x D", ha="center", fontsize=8.5)
    ax.text(2.0, 5.62, "43 MB at 450x375, 104 MB at 848x480", ha="center", fontsize=7.4)
    box(ax, 4.9, 4.35, 4.6, 0.85, "aggregate the volume", fs=8.5)
    box(ax, 4.9, 3.05, 4.6, 0.85, "read back, pick winners", fs=8.5)
    arrow(ax, 3.6, 4.75, 4.85, 4.75)
    arrow(ax, 7.2, 4.30, 7.2, 3.95)
    ax.text(5.0, 1.9, "the volume is written, read, and read again", fontsize=8.2,
            color=WARN, ha="center")

    ax = axes[1]
    ax.set_title("what ships: the volume never exists", fontsize=9.5)
    for i, lab in enumerate(["d = 0", "d = 1", "...", "d = D-1"]):
        box(ax, 0.25, 5.55 - i * 0.98, 1.35, 0.72, lab, fc="#eef2f6", fs=7.5)
    box(ax, 2.55, 4.05, 1.95, 0.95, "score\n+ filter")
    box(ax, 5.05, 4.05, 2.35, 0.95, "running top-2\nper pixel", fc="#e8f1ea", ec=GPU)
    box(ax, 7.95, 4.05, 1.85, 0.95, "MASDA\nsolve", fc="#e6eef6", ec=CPU)
    for i in range(4):
        arrow(ax, 1.65, 5.9 - i * 0.98, 2.5, 4.6, color=MUTE, lw=0.8)
    arrow(ax, 4.55, 4.52, 5.0, 4.52)
    arrow(ax, 7.45, 4.52, 7.9, 4.52)
    ax.text(6.22, 3.62, "2 x (score, d) per pixel", fontsize=7.5, ha="center", color=GPU)
    ax.text(5.0, 1.9, "one streaming pass; the common case is a rejected plane\n"
                      "that reads one cached buffer", fontsize=8.2, ha="center", color=GPU)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------------- algorithm: the layout
def fig_layout(path):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 7.0); ax.axis("off")

    ax = axes[0]
    ax.set_title("CPU: disparity-major  vol[d][y][x]", fontsize=9.5, color=CPU)
    for i in range(3):
        ax.add_patch(Rectangle((1.4 + i * 0.5, 3.7 - i * 0.42), 5.2, 2.0,
                               fc="#e6eef6", ec=CPU, lw=1.0))
    ax.text(4.5, 6.35, "one constant-disparity plane at a time", fontsize=8, ha="center")
    ax.text(4.5, 1.75, "the edge-aware filter needs a whole plane;\n"
                       "512 KB of L2 wants exactly one of them resident",
            fontsize=8, ha="center", color=CPU)

    ax = axes[1]
    ax.set_title("GPU: disparity-minor  vol[y][x][k]", fontsize=9.5, color=GPU)
    for i in range(8):
        ax.add_patch(Rectangle((1.0 + i * 0.72, 4.0), 0.66, 1.5,
                               fc="#e8f1ea", ec=GPU, lw=1.0))
        ax.text(1.33 + i * 0.72, 4.75, f"d{i}", ha="center", fontsize=7)
    ax.annotate("", xy=(6.72, 3.72), xytext=(1.0, 3.72),
                arrowprops=dict(arrowstyle="<->", color=GPU, lw=1.3))
    ax.text(3.86, 3.30, "one warp = 32 consecutive disparities of one row",
            fontsize=8, ha="center", color=GPU)
    ax.text(4.6, 1.75, "every access is a k-run: one aligned 128-byte transaction.\n"
                       "each lane carries its own filter recurrence in registers",
            fontsize=8, ha="center", color=GPU)
    ax.text(4.6, 0.45, "same algorithm, same bytes out - opposite memory layout",
            fontsize=8.5, ha="center", color=INK)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------- algorithm: the pipeline
def fig_overlap(path):
    """Who is busy when. One x unit is one millisecond, so the bars are to scale."""
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    ax.set_xlim(-9.5, 35.5); ax.set_ylim(-0.4, 5.5); ax.axis("off")
    T = 31.7
    rows = [("GPU  kernels", 3.9, GPU, 0.0, 26.9, "frame t+1: census, cost, filter, top-2"),
            ("copy engine", 3.0, MUTE, 26.9, 2.0, "fetch"),
            ("CPU  A57 x4", 2.1, CPU, 0.0, 11.0, "solve frame t"),
            ("CPU  1 core", 1.2, WARN, 0.0, 29.0, "detect keypoints, frame t")]
    for name, y, col, x0, w, lab in rows:
        ax.text(-0.7, y + 0.22, name, ha="right", va="center", fontsize=8.5, color=col)
        ax.add_patch(FancyBboxPatch((x0, y), w, 0.44, boxstyle="round,pad=0.01",
                                    fc=col, ec="none", alpha=0.22))
        txt = f"{lab}   {w:g} ms"
        if w >= 9:                       # comfortably fits inside the bar
            ax.text(x0 + w / 2, y + 0.22, txt, ha="center", va="center",
                    fontsize=7.8, color=col)
        else:                            # too short: label it above, clear of the
            ax.text(x0 + w / 2, y + 0.62, f"{lab}  ~{w:g} ms", ha="center",
                    va="center", fontsize=7.8, color=col)   # steady-state marker
    for x in (0.0, T):
        ax.plot([x, x], [0.9, 4.75], color=INK, lw=0.9)
    ax.annotate("", xy=(T, 5.0), xytext=(0.0, 5.0),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1))
    ax.text(T / 2, 5.18, f"{T} ms steady state = 31.5 Hz", ha="center", fontsize=9)
    for x in (0, 10, 20, 30):
        ax.plot([x, x], [0.72, 0.86], color=MUTE, lw=0.8)
        ax.text(x, 0.42, f"{x}", ha="center", fontsize=7, color=MUTE)
    ax.text(35.3, 0.42, "ms", ha="right", fontsize=7, color=MUTE)
    ax.text(-9.3, -0.15, "Detection is 29 ms of ONE core and costs 0.4-1.4 ms of frame time, "
                         "because it hides under the kernels.\nThe solve is 11 ms since the "
                         "message passing came out.", fontsize=8, va="top", ha="left")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# --------------------------------------------------------------- one warp, one row
def fig_warp(path):
    """What 32 lanes of k_score_hfwd touch at a single x. The whole GPU design is this
    picture: one lane per disparity, so every read is a broadcast or a sliding window
    and every write is a single transaction."""
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 9.2); ax.axis("off")

    # --- what the warp reads -------------------------------------------------------
    ax.text(0.3, 8.85, "right census row, one uint64 per pixel", fontsize=8.5)
    for i in range(14):
        inside = 3 <= i <= 10
        ax.add_patch(Rectangle((0.3 + i * 0.60, 7.95), 0.56, 0.55,
                               fc="#e8f1ea" if inside else "#ffffff",
                               ec=GPU if inside else MUTE, lw=1.0))
    ax.annotate("", xy=(0.3 + 10.86 * 0.60, 7.72), xytext=(0.3 + 3 * 0.60, 7.72),
                arrowprops=dict(arrowstyle="<->", color=GPU, lw=1.2))
    ax.text(4.2, 7.05, "R[x-d] for 32 consecutive d: one 256-byte window,\n"
                       "sliding one cell per x step, resident in L1",
            fontsize=7.8, color=GPU, ha="center")

    for i, lab in enumerate(("L[x]", "cl[x]", "a[x]")):
        box(ax, 9.35 + i * 0.95, 7.95, 0.85, 0.55, lab, fc="#e6eef6", ec=CPU, fs=7.5)
    ax.text(10.75, 8.85, "one address for all 32 lanes", fontsize=7.8, color=CPU,
            ha="center")
    ax.text(10.75, 7.35, "a hardware broadcast", fontsize=7.8, color=CPU, ha="center")

    ax.annotate("", xy=(6.2, 5.35), xytext=(6.2, 6.55),
                arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=1.6))

    # --- the warp ------------------------------------------------------------------
    for i in range(8):
        lab = ["d", "d+1", "d+2", "…", "…", "…", "d+30", "d+31"][i]
        who = ["lane 0", "lane 1", "lane 2", "lane", "lane", "lane",
               "lane 30", "lane 31"][i]
        box(ax, 0.5 + i * 1.44, 4.25, 1.30, 0.95, f"{who}\n{lab}",
            fc="#f2f7f4", ec=GPU, fs=7.4)
    ax.text(6.2, 3.62, "each lane owns ONE disparity and carries its own forward "
                       "recurrence in a register,", fontsize=8, ha="center")
    ax.text(6.2, 3.18, "$F \\;\\leftarrow\\; v + a_x\\,(F - v)$", fontsize=10,
            ha="center")
    ax.text(6.2, 2.72, "so no lane waits on another and the filter needs no shuffles "
                       "at all", fontsize=8, ha="center")

    ax.annotate("", xy=(6.2, 1.72), xytext=(6.2, 2.42),
                arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=1.6))

    # --- what the warp writes ------------------------------------------------------
    for i in range(8):
        ax.add_patch(Rectangle((0.5 + i * 1.44, 1.10), 1.30, 0.52, fc="#e8f1ea",
                               ec=GPU, lw=1.0))
    ax.text(6.2, 0.62, "the write is vol[y][x][k] for 32 consecutive k: "
                       "32 int16 = one aligned 64-byte transaction",
            fontsize=8, ha="center", color=GPU)
    ax.text(6.2, 0.18, "nothing strides, at any point in the pipeline",
            fontsize=8.4, ha="center")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------- the warp top-2 reduction
def fig_shuffle(path):
    """Why the tie-break needed k in the key: the tree merges non-adjacent lanes."""
    fig, ax = plt.subplots(figsize=(9.8, 3.9))
    N = 8
    ax.set_xlim(-1.2, N + 0.4); ax.set_ylim(-0.5, 4.9); ax.axis("off")
    ax.set_title("shfl_down reduction, drawn for 8 lanes; the kernel runs 32 in five "
                 "rounds", fontsize=9.5)
    for r, off in enumerate((4, 2, 1)):
        y = 3.2 - r * 1.05
        ax.text(-1.15, y, f"off = {off}", fontsize=8, va="center", ha="left")
        for i in range(N):
            ax.plot([i], [y], "o", ms=7, color=INK if i % off == 0 or True else MUTE,
                    mfc="#ffffff", mec=INK)
            if i % (off * 2) == 0 and i + off < N:
                col = WARN if (r == 0 and i == 0) else GPU
                ax.annotate("", xy=(i + 0.12, y - 0.16), xytext=(i + off - 0.12, y - 0.16),
                            arrowprops=dict(arrowstyle="-|>", color=col,
                                            lw=1.9 if col is WARN else 1.1,
                                            connectionstyle="arc3,rad=0.34"))
    for i in range(N):
        ax.text(i, 4.16, str(i), ha="center", fontsize=8, color=MUTE)
    ax.text(N / 2 - 0.5, 4.56, "lane", ha="center", fontsize=8, color=MUTE)
    ax.text(2.0, 0.55, "lane 0 absorbs lane 4 BEFORE it absorbs lane 1", fontsize=8.4,
            color=WARN)
    ax.text(2.0, 0.15, "so \"the other side holds larger k\" is false mid-tree, and a "
                       "tie-break that assumes it\nis wrong. Carrying k in the sort key "
                       "makes the merge order-independent.", fontsize=8, va="top")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------------- resolution/rate chart
def fig_rates(path):
    labels = ["424x240\nD=64", "450x375\nD=64", "640x480\nD=64",
              "848x480\nD=64", "848x480\nD=96", "848x480\nD=128"]
    ms = [8.7, 13.7, 24.1, 31.7, 48.4, 48.5]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    cols = [GPU if v <= 33.3 else MUTE for v in ms]
    ax.bar(range(len(ms)), ms, color=cols, width=0.62)
    ax.axhline(33.3, color=WARN, ls="--", lw=1.3)
    ax.text(5.4, 34.6, "33.3 ms = 30 Hz", color=WARN, fontsize=8, ha="right")
    # spelled out rather than computed, so the bar labels and the table in the post
    # cannot round differently: 1000/24.1 is 41.5, which prints as 41 or 42 depending
    # on which of them does the rounding.
    hz = ["115", "73", "42", "31.5", "20.7", "20.6"]
    for i, (v, lab) in enumerate(zip(ms, hz)):
        ax.text(i, v + 1.2, f"{lab} Hz", ha="center", fontsize=8)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylabel("ms / frame, pipelined")
    ax.set_title("Jetson TX2, de_dense_cuda, measured steady state")
    ax.set_ylim(0, 58)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_budget(path):
    """Where a far-field pixel is lost, itemised. Measured by cost_ceiling.py."""
    items = [("correct", 56.1, GPU),
             ("selector: truth was in the top-2,\nthe top-1 was taken", 13.1, CPU),
             ("descriptor: no candidate within\n0.5 px even in the top-8", 10.8, WARN),
             ("pruning: in the top-8, not the top-2", 8.2, "#b07d3a"),
             ("sub-pixel fit missed 0.25 px", 8.5, "#6b5b95"),
             ("the gate declined", 2.1, MUTE)]
    fig, ax = plt.subplots(figsize=(9.0, 1.9))
    left = 0.0
    for lab, v, col in items:
        ax.barh([0], [v], left=left, color=col, height=0.5)
        if v > 6:
            ax.text(left + v / 2, 0, f"{v:.1f}%", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
        left += v
    ax.set_xlim(0, 100); ax.set_ylim(-0.42, 0.42)
    ax.set_yticks([]); ax.set_xlabel("share of far-field pixels (%)")
    ax.set_title("Where a far-field pixel is lost — 15 v3 scenes, 2.35 M pixels",
                 fontsize=10)
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in items]
    ax.legend(hs, [l.replace("\n", " ") for l, _, _ in items], frameon=False,
              fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_thumbs():
    """Small, dedicated thumbnails for the front page.

    The post images are 300-600 KB each because they carry eight disparity maps at
    reading resolution. Pointing `thumbnail-img` at those makes the index page fetch
    ~1.4 MB and downscale it in the browser. These are ~400 px wide and quantised,
    which is two orders of magnitude smaller and indistinguishable at the size the
    index actually renders.
    """
    from PIL import Image
    # A crop where the whole figure does not survive being 420 px wide. Part 3's is a
    # 2x4 grid; downscaled entire it is unreadable, so the thumbnail takes the two
    # dark-room columns, which are the pair that carries the point.
    jobs = [(os.path.join(P2, "maps_a.png"), os.path.join(P2, "thumb_p2.png"), None),
            (os.path.join(P3, "real_pair.png"), os.path.join(P3, "thumb_p3.png"),
             (0.055, 0.375, 0.345, 0.945)),
            (os.path.join(P2, "subpixel.png"), os.path.join(P2, "thumb_p1.png"), None)]
    for src, dst, box in jobs:
        im = Image.open(src).convert("RGB")
        if box:
            W, H = im.size
            im = im.crop((int(box[0] * W), int(box[1] * H),
                          int(box[2] * W), int(box[3] * H)))
        w = 420
        im = im.resize((w, max(1, round(im.height * w / im.width))), Image.LANCZOS)
        im.convert("P", palette=Image.ADAPTIVE, colors=128).save(dst, optimize=True)
        print(f"wrote {dst}  {os.path.getsize(dst)//1024} KB "
              f"(from {os.path.getsize(src)//1024} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(P2, exist_ok=True); os.makedirs(P3, exist_ok=True)
    jobs = {
        "mid1": lambda: fig_middlebury(["teddy", "cones", "Art", "Books"],
                                       os.path.join(P2, "maps_a.png")),
        "mid2": lambda: fig_middlebury(["Dolls", "Laundry", "Moebius", "Reindeer"],
                                       os.path.join(P2, "maps_b.png")),
        "real": lambda: fig_real(os.path.join(P3, "real_pair.png")),
        "sub": lambda: fig_subpixel(os.path.join(P2, "subpixel.png")),
        "curve": lambda: fig_curve(os.path.join(P2, "curve.png")),
        "novol": lambda: fig_novolume(os.path.join(P2, "novolume.png")),
        "agg": lambda: fig_aggregate(os.path.join(P2, "aggregate.png")),
        "layout": lambda: fig_layout(os.path.join(P3, "layout.png")),
        "split": lambda: fig_split(os.path.join(P3, "split.png")),
        "dataflow": lambda: fig_dataflow(os.path.join(P3, "dataflow.png")),
        "overlap": lambda: fig_overlap(os.path.join(P3, "overlap.png")),
        "rates": lambda: fig_rates(os.path.join(P3, "rates.png")),
        "warp": lambda: fig_warp(os.path.join(P3, "warp.png")),
        "shuffle": lambda: fig_shuffle(os.path.join(P3, "shuffle.png")),
        "budget": lambda: fig_budget(os.path.join(P2, "budget.png")),
        "thumbs": fig_thumbs,
    }
    for k, fn in jobs.items():
        if a.only and k not in a.only.split(","):
            continue
        fn()


if __name__ == "__main__":
    main()
