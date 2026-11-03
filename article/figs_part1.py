#!/usr/bin/env python3
"""Figures for Part 1, the formulation post.

Part 1 carries 27 display formulas and one picture, and the one picture is a result
table. The factor graph is the object every formula in section 1.2 is about, and it
was only ever named in words.

A first attempt drew the pixels as nodes with the association variables hanging off
the edges between them. That is a picture of the matching problem, not of this factor
graph: here the variables ARE the candidate pairs, and the pixels are the rows and
columns they are grouped by. Drawing it the other way puts a variable node beside an
edge it is supposed to be on (checklist item 41).

    .venv/bin/python article/figs_part1.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

BLOG = os.path.expanduser("~/src/mayio.github.io/assets/img")
OUT = os.path.join(BLOG, "2026-08-07-MASDA-Sparse-Stereo_files")

INK = "#1b1b1b"
VAR = "#2b5d8a"      # association variables and the rho messages
FAC = "#2f6f4f"      # score factors
CON = "#a4442c"      # exclusivity constraints and the beta messages
MUTE = "#9a9a9a"
GHOST = "#dcdcdc"

plt.rcParams.update({
    "font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight",
    "text.color": INK, "font.family": "DejaVu Sans",
})

NL, NR = 3, 4
# Each left pixel keeps its two best candidates. Deliberately overlapping: L2 and L3
# both want R3, which is the conflict the exclusivity constraints exist to settle.
CAND = {0: (0, 1), 1: (1, 2), 2: (2, 3)}


def factor_graph(path):
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.set_aspect("equal")
    X = {j: 1.15 * j for j in range(NR)}
    Y = {i: -1.15 * i for i in range(NL)}
    xI, yE = -1.35, Y[NL - 1] - 1.35

    present = {(i, j) for i, js in CAND.items() for j in js}

    def connector(a, b, fixed, gaps, horizontal):
        """A constraint's edge to its variables, broken around the pairs it has none
        of. Drawn as one straight line it runs through those positions and reads as
        a connection that does not exist (checklist item 41)."""
        lo, hi = min(a, b), max(a, b)
        stops = [lo]
        for g in sorted(gaps):
            if lo < g < hi:
                stops += [g - 0.3, g + 0.3]
        stops.append(hi)
        for u, v in zip(stops[::2], stops[1::2]):
            if v - u <= 0.02:
                continue
            if horizontal:
                ax.plot([u, v], [fixed, fixed], color=MUTE, lw=1.0, zorder=1)
            else:
                ax.plot([fixed, fixed], [u, v], color=MUTE, lw=1.0, zorder=1)

    # Every pair the full matrix would carry, so the sparsity is visible rather
    # than asserted.
    for i in range(NL):
        for j in range(NR):
            if (i, j) not in present:
                ax.add_patch(Circle((X[j], Y[i]), 0.17, fc="white", ec=GHOST,
                                    lw=1.0, ls=(0, (2, 2)), zorder=2))

    # Row constraints: one I_i per left pixel, joined to that row's candidates.
    for i in range(NL):
        js = CAND[i]
        connector(xI + 0.2, X[max(js)], Y[i],
                  [X[j] for j in range(NR) if j not in js], True)
        ax.add_patch(Rectangle((xI - 0.2, Y[i] - 0.2), 0.4, 0.4, fc=CON, ec=CON,
                               zorder=3))
        ax.text(xI, Y[i], f"$I_{i+1}$", ha="center", va="center", color="white",
                fontsize=8, zorder=4)
        ax.text(xI - 0.62, Y[i], f"$L_{i+1}$", ha="center", va="center",
                fontsize=8.5, color=MUTE)

    # Column constraints: one E_j per right pixel.
    for j in range(NR):
        iss = [i for i in range(NL) if j in CAND[i]]
        connector(yE + 0.2, Y[min(iss)], X[j],
                  [Y[i] for i in range(NL) if i not in iss], False)
        ax.add_patch(Rectangle((X[j] - 0.2, yE - 0.2), 0.4, 0.4, fc=CON, ec=CON,
                               zorder=3))
        ax.text(X[j], yE, f"$E_{j+1}$", ha="center", va="center", color="white",
                fontsize=8, zorder=4)
        ax.text(X[j], yE - 0.62, f"$R_{j+1}$", ha="center", va="center",
                fontsize=8.5, color=MUTE)

    # The candidates themselves, each with its own score factor.
    for i, js in CAND.items():
        for j in js:
            ax.plot([X[j] + 0.17, X[j] + 0.40], [Y[i], Y[i] + 0.33],
                    color=MUTE, lw=1.0, zorder=1)
            ax.add_patch(Rectangle((X[j] + 0.40, Y[i] + 0.33), 0.30, 0.30,
                                   fc=FAC, ec=FAC, zorder=3))
            ax.text(X[j] + 0.55, Y[i] + 0.48, "$S$", ha="center", va="center",
                    color="white", fontsize=7.5, zorder=4)
            ax.add_patch(Circle((X[j], Y[i]), 0.17, fc="white", ec=VAR, lw=1.5,
                                zorder=3))
            ax.text(X[j], Y[i], f"$c_{{{i+1}{j+1}}}$", ha="center", va="center",
                    color=VAR, fontsize=6.3, zorder=4)

    # The two messages, drawn on the edges they actually travel along.
    ax.annotate("", xy=(X[0] - 0.21, Y[0]), xytext=(xI + 0.23, Y[0]),
                arrowprops=dict(arrowstyle="-|>", color=VAR, lw=1.7, shrinkA=0,
                                shrinkB=0), zorder=5)
    ax.text((xI + X[0]) / 2, Y[0] + 0.24, r"$\rho_{ij}$", color=VAR, fontsize=10,
            ha="center")
    ax.annotate("", xy=(X[2], Y[2] - 0.21), xytext=(X[2], yE + 0.23),
                arrowprops=dict(arrowstyle="-|>", color=CON, lw=1.7, shrinkA=0,
                                shrinkB=0), zorder=5)
    ax.text(X[2] - 0.42, (yE + Y[2]) / 2, r"$\beta_{ij}$", color=CON, fontsize=10,
            ha="center")

    key = [(Circle((0, 0), 1, fc="white", ec=VAR, lw=1.5),
            "$c_{ij}$, a candidate pair"),
           (Circle((0, 0), 1, fc="white", ec=GHOST, lw=1.0, ls=(0, (2, 2))),
            "pair the sparse form never builds"),
           (Rectangle((0, 0), 1, 1, fc=FAC, ec=FAC), "$S_{ij}$, its score"),
           (Rectangle((0, 0), 1, 1, fc=CON, ec=CON),
            "$I_i$, $E_j$, at most one match")]
    ax.legend([h for h, _ in key], [t for _, t in key], loc="upper center",
              bbox_to_anchor=(0.46, -0.01), ncol=2, frameon=False,
              handlelength=1.2, fontsize=8, columnspacing=1.6, labelspacing=0.7)

    ax.set_xlim(xI - 1.0, X[NR - 1] + 1.1)
    ax.set_ylim(yE - 1.0, Y[0] + 1.0)
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def main():
    os.makedirs(OUT, exist_ok=True)
    factor_graph(os.path.join(OUT, "factor_graph.png"))


if __name__ == "__main__":
    main()
