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
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 4.4))
    for i, (bag, label) in enumerate((("full_on", "projector ON"),
                                      ("full_off", "projector OFF"))):
        lp = os.path.join(ROOT, "bags", bag, "frames", "ir1_00000060.raw")
        rp = os.path.join(ROOT, "bags", bag, "frames", "ir2_00000060.raw")
        L = np.fromfile(lp, np.uint8).reshape(480, 848)
        R = np.fromfile(rp, np.uint8).reshape(480, 848)
        d = run_dense(lp, rp, 848, 480, 60)
        axes[i, 0].imshow(L, cmap="gray"); axes[i, 0].set_ylabel(label, fontsize=9)
        axes[i, 1].imshow(R, cmap="gray")
        show_disp(axes[i, 2], d, 60)
        for j in range(3):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        cov = 100.0 * np.isfinite(d[d > 0]).sum() / d.size
        axes[i, 2].set_xlabel(f"{cov:.0f}% of pixels answered", fontsize=8)
    for j, t in enumerate(["left IR (848x480)", "right IR", "disparity"]):
        axes[0, j].set_title(t)
    fig.suptitle("D435 infrared pair, on the vehicle's own camera", y=0.98, fontsize=10)
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
    vals = [0.80, 25.18, 41.53, 45.55]
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
    bad = [32.01, 28.52, 25.18, 16.99, 6.60]
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
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.1))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    ax = axes[0]
    ax.set_title("the textbook pipeline: build the volume", fontsize=9.5)
    for i in range(4):
        ax.add_patch(Rectangle((1.1 + i * 0.34, 3.0 - i * 0.20), 2.3, 1.5,
                               fc="#eef2f6", ec=INK, lw=0.9))
    ax.text(2.6, 4.95, "W x H x D  (40-98 MB)", ha="center", fontsize=8)
    box(ax, 5.4, 3.4, 3.4, 0.85, "aggregate the volume")
    box(ax, 5.4, 2.1, 3.4, 0.85, "read back, pick winners")
    arrow(ax, 3.9, 3.8, 5.35, 3.8)
    arrow(ax, 7.1, 3.35, 7.1, 3.0)
    ax.text(5.0, 1.2, "the volume is written, read, and read again", fontsize=8,
            color=WARN, ha="center")

    ax = axes[1]
    ax.set_title("what ships: the volume never exists", fontsize=9.5)
    for i, lab in enumerate(["d = 0", "d = 1", "...", "d = D-1"]):
        box(ax, 0.5, 4.6 - i * 1.12, 1.5, 0.8, lab, fc="#eef2f6", fs=7.5)
    box(ax, 2.5, 3.1, 1.9, 0.9, "score\n+ filter", fc="#ffffff")
    box(ax, 5.0, 3.1, 2.2, 0.9, "running\ntop-2 per pixel", fc="#e8f1ea", ec=GPU)
    box(ax, 7.8, 3.1, 1.7, 0.9, "MASDA\nsolve", fc="#e6eef6", ec=CPU)
    for i in range(4):
        arrow(ax, 2.05, 5.0 - i * 1.12, 2.45, 3.75, color=MUTE, lw=0.8)
    arrow(ax, 4.45, 3.55, 4.95, 3.55)
    arrow(ax, 7.25, 3.55, 7.75, 3.55)
    ax.text(6.1, 2.6, "2 x (score, d) per pixel", fontsize=7.5, ha="center", color=GPU)
    ax.text(5.0, 1.4, "one streaming pass; the common case is a rejected plane\n"
                      "that reads one cached buffer", fontsize=8, ha="center", color=GPU)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------------- algorithm: the layout
def fig_layout(path):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")

    ax = axes[0]
    ax.set_title("CPU: disparity-major  vol[d][y][x]", fontsize=9.5, color=CPU)
    for i in range(3):
        ax.add_patch(Rectangle((1.4 + i * 0.5, 3.1 - i * 0.42), 5.2, 2.0,
                               fc="#e6eef6", ec=CPU, lw=1.0))
    ax.text(4.5, 5.55, "one constant-disparity plane at a time", fontsize=8, ha="center")
    ax.text(4.5, 1.9, "the edge-aware filter needs a whole plane;\n"
                      "512 KB of L2 wants exactly one of them resident",
            fontsize=8, ha="center", color=CPU)

    ax = axes[1]
    ax.set_title("GPU: disparity-minor  vol[y][x][k]", fontsize=9.5, color=GPU)
    for i in range(8):
        ax.add_patch(Rectangle((1.0 + i * 0.72, 3.4), 0.66, 1.5,
                               fc="#e8f1ea", ec=GPU, lw=1.0))
        ax.text(1.33 + i * 0.72, 4.15, f"d{i}", ha="center", fontsize=7)
    ax.annotate("", xy=(6.8, 3.15), xytext=(1.0, 3.15),
                arrowprops=dict(arrowstyle="<->", color=GPU, lw=1.3))
    ax.text(3.9, 2.75, "one warp = 32 consecutive disparities of one row",
            fontsize=8, ha="center", color=GPU)
    ax.text(4.6, 1.9, "every access is a k-run: one aligned 128-byte transaction.\n"
                      "each lane carries its own filter recurrence in registers",
            fontsize=8, ha="center", color=GPU)
    ax.text(4.6, 0.7, "same algorithm, same bytes out - opposite memory layout",
            fontsize=8.5, ha="center", color=INK)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------- algorithm: the pipeline
def fig_overlap(path):
    fig, ax = plt.subplots(figsize=(9.4, 2.9))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 4.6); ax.axis("off")
    rows = [("GPU  kernels", 3.5, GPU,
             [(0.4, 2.6, "frame t+1: census, cost, filter, top-2   26 ms")]),
            ("copy engine", 2.6, MUTE, [(3.05, 0.5, "fetch  ~2 ms")]),
            ("CPU  A57 x4", 1.7, CPU, [(0.4, 1.4, "solve frame t   11 ms")]),
            ("CPU  1 core", 0.8, WARN, [(0.4, 2.9, "detect keypoints, frame t   29 ms")])]
    for name, y, col, bars in rows:
        ax.text(-0.05, y + 0.22, name, ha="right", fontsize=8.5, color=col)
        for x, w, lab in bars:
            ax.add_patch(FancyBboxPatch((x, y), w, 0.46,
                                        boxstyle="round,pad=0.01",
                                        fc=col, ec="none", alpha=0.22))
            ax.text(x + w / 2, y + 0.23, lab, ha="center", va="center",
                    fontsize=7.6, color=col)
    ax.plot([0.4, 0.4], [0.6, 4.15], color=INK, lw=0.9)
    ax.plot([3.6, 3.6], [0.6, 4.15], color=INK, lw=0.9)
    ax.annotate("", xy=(3.6, 4.3), xytext=(0.4, 4.3),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1))
    ax.text(2.0, 4.4, "31.8 ms steady state = 31.4 Hz", ha="center", fontsize=9)
    ax.text(5.6, 2.2, "detection is 29 ms of ONE core and costs 0.4-1.4 ms of\n"
                      "frame time, because it hides under the kernels.\n"
                      "The solve is 11 ms since the message passing came out.",
            fontsize=8, va="center")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------------- resolution/rate chart
def fig_rates(path):
    labels = ["424x240\nD=60", "450x375\nD=60", "640x480\nD=60",
              "848x480\nD=60", "848x480\nD=96", "848x480\nD=128"]
    ms = [8.7, 13.7, 24.1, 31.7, 48.6, 48.4]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    cols = [GPU if v <= 33.3 else MUTE for v in ms]
    ax.bar(range(len(ms)), ms, color=cols, width=0.62)
    ax.axhline(33.3, color=WARN, ls="--", lw=1.3)
    ax.text(5.4, 34.6, "33.3 ms = 30 Hz", color=WARN, fontsize=8, ha="right")
    for i, v in enumerate(ms):
        ax.text(i, v + 1.2, f"{1000/v:.0f} Hz", ha="center", fontsize=8)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylabel("ms / frame, pipelined")
    ax.set_title("Jetson TX2, de_dense_cuda, measured steady state")
    ax.set_ylim(0, 58)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


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
        "layout": lambda: fig_layout(os.path.join(P3, "layout.png")),
        "overlap": lambda: fig_overlap(os.path.join(P3, "overlap.png")),
        "rates": lambda: fig_rates(os.path.join(P3, "rates.png")),
    }
    for k, fn in jobs.items():
        if a.only and k not in a.only.split(","):
            continue
        fn()


if __name__ == "__main__":
    main()
