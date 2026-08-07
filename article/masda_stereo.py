#!/usr/bin/env python3
"""MASDA for sparse stereo matching — the worked example for the article.

Self-contained: numpy + scipy + matplotlib. No image files are read, so nothing
here carries a licence. The scene is generated procedurally, which also buys the
thing real stereo footage cannot give without a rangefinder — **exact ground-truth
disparity**, so match correctness is measurable rather than merely plausible.

Run:  python masda_stereo.py
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

RNG = np.random.default_rng(20251126)
W, H = 480, 320
FIG = "figures"

# ---------------------------------------------------------------------------
# 1. Synthetic scene with known disparity


def ground_truth_disparity() -> np.ndarray:
    """Piecewise-planar disparity: slanted floor, back wall, and a thin bar.

    Chosen to contain the three cases the stereo formulation has to survive:
    a slanted surface (foreshortening, so left and right sample it at different
    rates), a depth discontinuity (which generates occlusion), and a thin
    foreground object (which violates the ordering constraint).
    """
    d = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]

    d[:] = 8.0                                   # back wall, far
    floor = yy > H * 0.55
    d[floor] = 8.0 + (yy[floor] - H * 0.55) * 0.28   # slanted floor, nearer below

    block = (xx > 300) & (xx < 400) & (yy > 90) & (yy < 250)
    d[block] = 26.0                              # fronto-parallel box, mid depth

    bar = (xx > 150) & (xx < 162) & (yy > 40) & (yy < 300)
    d[bar] = 38.0                                # thin foreground bar, nearest
    return d


def dot_texture(density=0.30, sigma=0.8) -> np.ndarray:
    """An IR-projector-like pattern: identical dots on an untextured surface.

    This is the regime that motivates MASDA. Every dot looks the same locally, so
    a descriptor centred on one carries almost no identity -- what distinguishes
    it is the *constellation* of its neighbours.

    Density is tuned to reproduce the degeneracy measured on real D435 IR frames
    with the projector on: about 30% of Census descriptors distinct, i.e. each one
    shared by roughly three keypoints. Sparser dots give a varied constellation in
    every window and the ambiguity disappears, which would make the example easy
    in exactly the way real projected texture is not.
    """
    img = np.zeros((H, W), np.float32)
    n = int(H * W * density)
    ys = RNG.integers(0, H, n)
    xs = RNG.integers(0, W, n)
    img[ys, xs] = 1.0
    k = int(sigma * 4) | 1
    ax = np.arange(k) - k // 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    img = np.apply_along_axis(lambda m: np.convolve(m, g, "same"), 0, img)
    img = np.apply_along_axis(lambda m: np.convolve(m, g, "same"), 1, img)
    img = img / max(img.max(), 1e-6)
    return 40.0 + 170.0 * img          # dark surface, bright dots


def broadband_texture() -> np.ndarray:
    """Contrasting control: rich, non-repeating texture where descriptors are
    individually discriminative and nearest-neighbour matching does fine."""
    img = np.zeros((H, W), np.float32)
    for scale, amp in ((2, 1.0), (5, 0.7), (13, 0.5), (31, 0.35)):
        coarse = RNG.normal(0, 1, (H // scale + 2, W // scale + 2))
        ys = np.clip((np.arange(H) / scale).astype(int), 0, coarse.shape[0] - 1)
        xs = np.clip((np.arange(W) / scale).astype(int), 0, coarse.shape[1] - 1)
        img += amp * coarse[np.ix_(ys, xs)]
    img -= img.min()
    img /= max(img.max(), 1e-6)
    return 30.0 + 200.0 * img


def periodic_texture(period=11) -> np.ndarray:
    """A regular lattice: the classic stereo ambiguity case.

    Repetitive structure -- brick, fencing, tiling, a keyboard -- puts several
    candidates along the epipolar line that are near-identical, so the descriptor
    cannot decide between them. Pseudo-random projector patterns are designed
    specifically to avoid this, but the failure is worth reproducing because it is
    where a matcher's constraint structure, rather than its descriptor, does the
    work. It is also the plan's cited cause of BP oscillation on this graph.
    """
    yy, xx = np.mgrid[0:H, 0:W]
    g = ((np.sin(2 * np.pi * xx / period) * np.sin(2 * np.pi * yy / period)) > 0.3)
    img = 40.0 + 180.0 * g.astype(np.float32)
    # A slow large-scale gradient, so the pattern is repetitive but the image is
    # not perfectly periodic -- as real repetitive scenes are.
    img += 18.0 * np.sin(2 * np.pi * xx / (W * 1.7))
    return img


def score_margin(S):
    """Best-minus-second-best score per measurement: a direct measure of how
    ambiguous the association problem is, and exactly what a ratio test keys on."""
    out = []
    for row in S:
        f = row[np.isfinite(row)]
        if f.size < 2:
            continue
        f = np.sort(f)[::-1]
        out.append(float(f[0] - f[1]))
    return np.array(out)


def render_pair(texture: np.ndarray, disp: np.ndarray, noise=1.5):
    """Warp the left image into the right by the disparity, with a z-buffer.

    Forward warping means occlusions appear on their own: where a near surface
    covers a far one, the far one has no right-image counterpart. Those keypoints
    genuinely have no correct match, which is exactly what the clutter and
    misdetection terms exist to absorb -- so the example exercises them honestly
    rather than assuming every keypoint is matchable.
    """
    left = texture.copy()
    right = np.zeros_like(left)
    zbuf = np.full((H, W), -1.0, np.float32)
    for y in range(H):
        xs = np.arange(W)
        xr = np.round(xs - disp[y]).astype(int)
        ok = (xr >= 0) & (xr < W)
        for x, x2, d in zip(xs[ok], xr[ok], disp[y][ok]):
            if d > zbuf[y, x2]:
                zbuf[y, x2] = d
                right[y, x2] = left[y, x]
    holes = zbuf < 0
    right[holes] = left[holes]              # fill disocclusions with background
    left = left + RNG.normal(0, noise, left.shape)
    right = right + RNG.normal(0, noise, right.shape)
    return (np.clip(left, 0, 255).astype(np.uint8),
            np.clip(right, 0, 255).astype(np.uint8),
            ~holes)


# ---------------------------------------------------------------------------
# 2. Keypoints and Census descriptors


def shi_tomasi(img: np.ndarray, win=3) -> np.ndarray:
    f = img.astype(np.float64)
    gx = np.zeros_like(f); gy = np.zeros_like(f)
    gx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) * 0.5
    gy[1:-1, :] = (f[2:, :] - f[:-2, :]) * 0.5

    def box(a):
        c = np.cumsum(np.cumsum(a, 0), 1)
        c = np.pad(c, ((1, 0), (1, 0)))
        r = win
        out = np.zeros_like(a)
        ys = np.arange(a.shape[0]); xs = np.arange(a.shape[1])
        y0 = np.clip(ys - r, 0, None); y1 = np.clip(ys + r + 1, None, a.shape[0])
        x0 = np.clip(xs - r, 0, None); x1 = np.clip(xs + r + 1, None, a.shape[1])
        out = (c[np.ix_(y1, x1)] - c[np.ix_(y0, x1)]
               - c[np.ix_(y1, x0)] + c[np.ix_(y0, x0)])
        return out
    sxx, syy, sxy = box(gx * gx), box(gy * gy), box(gx * gy)
    half = 0.5 * (sxx + syy)
    disc = np.sqrt(np.maximum(0, 0.25 * (sxx - syy) ** 2 + sxy ** 2))
    return (half - disc).astype(np.float32)


def detect(img, cell=12, per_cell=2, border=8, min_resp=None):
    """Grid-bucketed keypoints, so coverage is spread rather than piled onto the
    highest-contrast region."""
    r = shi_tomasi(img)
    if min_resp is None:
        min_resp = np.percentile(r, 80)
    # 3x3 non-maximum suppression
    m = np.ones_like(r, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            m &= r >= np.roll(np.roll(r, dy, 0), dx, 1)
    m[:border] = m[-border:] = False
    m[:, :border] = m[:, -border:] = False
    m &= r >= min_resp
    ys, xs = np.nonzero(m)
    keep = []
    for cy in range(0, H, cell):
        for cx in range(0, W, cell):
            sel = (ys >= cy) & (ys < cy + cell) & (xs >= cx) & (xs < cx + cell)
            if not sel.any():
                continue
            idx = np.nonzero(sel)[0]
            idx = idx[np.argsort(-r[ys[idx], xs[idx]])][:per_cell]
            keep.extend(idx.tolist())
    keep = np.array(sorted(keep), int)
    return np.stack([xs[keep], ys[keep]], 1).astype(np.float32), r[ys[keep], xs[keep]]


def census(img, pts, half_w=3, half_h=3):
    """Census transform: a bit per neighbour, set if it is darker than the centre.

    Invariant to any monotonic intensity mapping, so a gain or offset difference
    between the two cameras costs nothing. Hamming distance is then one popcount.
    """
    out = np.zeros(len(pts), np.uint64)
    f = img.astype(np.int32)
    for n, (x, y) in enumerate(pts.astype(int)):
        c = f[y, x]
        bits = np.uint64(0)
        b = 0
        for dy in range(-half_h, half_h + 1):
            for dx in range(-half_w, half_w + 1):
                if dx == 0 and dy == 0:
                    continue
                if f[y + dy, x + dx] < c:
                    bits |= np.uint64(1) << np.uint64(b)
                b += 1
        out[n] = bits
    return out


POPCNT = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def hamming(a, b):
    x = np.bitwise_xor(a, b)
    total = np.zeros(x.shape, np.int32)
    for shift in range(0, 64, 8):
        total += POPCNT[((x >> np.uint64(shift)) & np.uint64(0xFF)).astype(np.uint8)]
    return total


# ---------------------------------------------------------------------------
# 3. The association problem


BITS = 48


def build_problem(pl, dl, pr, dr, max_dy=1.5, dmin=1.0, dmax=60.0,
                  sigma_y=1.0, w_y=1.0):
    """Sparse candidate graph and its score matrix.

    s(i,j) is scaled so that a chance descriptor (Hamming = bits/2) scores 0 and a
    perfect one scores +1, which puts the clutter and misdetection costs on an
    interpretable scale: lambda = gamma = -0.1 means "reject anything worse than
    a tenth of the way from chance to perfect".
    """
    m, n = len(pl), len(pr)
    S = np.full((m, n), -np.inf, np.float32)
    for i in range(m):
        dy = pl[i, 1] - pr[:, 1]
        d = pl[i, 0] - pr[:, 0]
        ok = (np.abs(dy) <= max_dy) & (d >= dmin) & (d <= dmax)
        if not ok.any():
            continue
        h = hamming(np.full(ok.sum(), dl[i], np.uint64), dr[ok])
        desc = (BITS / 2 - h) / (BITS / 2)
        pen = w_y * (dy[ok] / sigma_y) ** 2
        S[i, ok] = desc - pen
    return S


def masda(S, lam=-0.1, gam=-0.1, iters=30, damping=0.4, eps=1e-5):
    """Max-sum loopy BP for data association, in the article's notation.

        beta_ij = s(i,j) - max_{k != i} rho_kj      (competition down a column)
        rho_ij  = s(i,j) - max_{k != j} beta_ik     (competition along a row)

    with lambda (clutter, a measurement left unassociated) and gamma
    (misdetection, an object left unassociated) entering as the alternatives those
    maxima compete against. The belief is b_ij = alpha_ij + eta_ij + s_ij, which
    reduces to beta + rho - s since each message already carries s once.

    Each iteration is two max-reductions. Using the largest and second-largest of
    each row and column makes "max excluding one element" O(1), so the cost is
    O(T * E) in the number of candidate EDGES, not O(T * M * N).
    """
    m, n = S.shape
    finite = np.isfinite(S)
    beta = np.where(finite, 0.0, -np.inf).astype(np.float64)
    rho = beta.copy()
    hist = []

    def top2_excluding(A, axis):
        """max over `axis` excluding each element's own position."""
        part = np.argsort(-A, axis=axis)
        if axis == 0:
            best = np.take_along_axis(A, part[:1], 0)
            second = np.take_along_axis(A, part[1:2], 0) if A.shape[0] > 1 \
                else np.full_like(best, -np.inf)
            is_best = np.zeros_like(A, bool)
            np.put_along_axis(is_best, part[:1], True, 0)
            return np.where(is_best, np.broadcast_to(second, A.shape),
                            np.broadcast_to(best, A.shape))
        best = np.take_along_axis(A, part[:, :1], 1)
        second = np.take_along_axis(A, part[:, 1:2], 1) if A.shape[1] > 1 \
            else np.full_like(best, -np.inf)
        is_best = np.zeros_like(A, bool)
        np.put_along_axis(is_best, part[:, :1], True, 1)
        return np.where(is_best, np.broadcast_to(second, A.shape),
                        np.broadcast_to(best, A.shape))

    old = np.seterr(invalid="ignore")
    for it in range(iters):
        # rho: a row's other options, or leaving this measurement as clutter
        comp = np.maximum(lam, top2_excluding(np.where(finite, beta, -np.inf), 1))
        tgt = S - comp
        new_rho = np.where(finite, (1 - damping) * tgt + damping * rho, -np.inf)
        # beta: a column's other options, or leaving this object misdetected
        comp = np.maximum(gam, top2_excluding(np.where(finite, new_rho, -np.inf), 0))
        tgt = S - comp
        new_beta = np.where(finite, (1 - damping) * tgt + damping * beta, -np.inf)

        delta = max(np.nanmax(np.abs(np.where(finite, new_rho - rho, 0))),
                    np.nanmax(np.abs(np.where(finite, new_beta - beta, 0))))
        rho, beta = new_rho, new_beta
        hist.append(float(delta))
        if delta < eps:
            break

    belief = np.where(finite, beta + rho - S, -np.inf)
    np.seterr(**old)
    return decide(S, belief, lam), np.array(hist)


def decide(S, belief, lam):
    """Order by belief, decide by the clutter cost, then complete greedily.

    The belief measures an edge's advantage over its competitors, so it ranks
    candidates -- but its SIGN is not a decision. Under near-ties nothing has an
    advantage and every belief is <= 0, which is the condition under which the
    LP optimum is not unique and BP's guarantee lapses. Whether to associate at
    all is s(i,j) against lambda.
    """
    m, n = S.shape
    order = np.argsort(-belief, axis=None)
    used_i = np.zeros(m, bool)
    used_j = np.zeros(n, bool)
    out = {}
    for flat in order:
        i, j = divmod(int(flat), n)
        if not np.isfinite(belief[i, j]):
            break
        if S[i, j] <= lam or used_i[i] or used_j[j]:
            continue
        out[i] = j
        used_i[i] = used_j[j] = True
    return out


def mutual_nn(S, lam=-0.1, ratio=0.85):
    """Mutual nearest neighbour with a Lowe-style ratio test."""
    m, n = S.shape
    out = {}
    best_j = np.argmax(np.where(np.isfinite(S), S, -np.inf), 1)
    best_i = np.argmax(np.where(np.isfinite(S), S, -np.inf), 0)
    for i in range(m):
        j = int(best_j[i])
        if not np.isfinite(S[i, j]) or S[i, j] <= lam:
            continue
        if int(best_i[j]) != i:
            continue
        row = S[i].copy()
        row[j] = -np.inf
        s2 = row.max()
        if np.isfinite(s2):
            b, r2 = S[i, j] - lam, s2 - lam
            if b > 0 and r2 > 0 and r2 / b > ratio:
                continue
        out[i] = j
    return out


def optimal_lap(S, lam=-0.1, gam=-0.1):
    """Exact maximum-weight assignment by Jonker-Volgenant (scipy), with explicit
    non-association slots so leaving a node out is a genuine option.

    This is the optimum MASDA is approximating, on real-sized problems rather than
    the handful of nodes brute force can reach.
    """
    m, n = S.shape
    big = np.full((m + n, n + m), 0.0)
    big[:m, :n] = np.where(np.isfinite(S), S, -1e6)
    big[:m, n:] = -1e6
    np.fill_diagonal(big[:m, n:], lam)          # measurement i -> clutter
    big[m:, :n] = -1e6
    np.fill_diagonal(big[m:, :n], gam)          # object j -> misdetected
    r, c = linear_sum_assignment(-big)
    return {int(i): int(j) for i, j in zip(r, c) if i < m and j < n
            and np.isfinite(S[i, j]) and S[i, j] > lam}


def objective(assign, S, lam, gam, m, n):
    v = sum(float(S[i, j]) for i, j in assign.items())
    return v + lam * (m - len(assign)) + gam * (n - len(assign))


def evaluate(assign, pl, pr, gt_disp, valid, tol=1.0):
    """Precision and recall against ground truth.

    A keypoint is *matchable* only if its ground-truth correspondence is actually
    visible in the right image; occluded ones have no correct answer, so counting
    them against recall would penalise the method for being right.
    """
    tp = fp = 0
    for i, j in assign.items():
        x, y = pl[i].astype(int)
        d_true = gt_disp[y, x]
        d_est = pl[i, 0] - pr[j, 0]
        if abs(d_est - d_true) <= tol:
            tp += 1
        else:
            fp += 1
    # Matchable means the correct answer EXISTS to be found: the correspondence
    # must be unoccluded and must itself have been detected as a right keypoint.
    # Keypoints are detected independently in the two images, so a left keypoint
    # whose partner was never detected has no attainable match, and counting it
    # against recall would penalise the matcher for the detector's behaviour.
    xi = pl[:, 0].astype(int)
    yi = pl[:, 1].astype(int)
    xr = pl[:, 0] - gt_disp[yi, xi]
    xri = np.round(xr).astype(int)
    inb = (xri >= 0) & (xri < W)
    unocc = np.zeros(len(pl), bool)
    unocc[inb] = valid[yi[inb], xri[inb]]
    # Vectorised: is there a right keypoint within tol of the true correspondence?
    dx = np.abs(pl[:, 0][:, None] - gt_disp[yi, xi][:, None] - pr[None, :, 0])
    dy2 = np.abs(pl[:, 1][:, None] - pr[None, :, 1])
    detected = ((dx <= tol) & (dy2 <= tol)).any(1)
    matchable = int((unocc & detected).sum())
    return dict(matches=len(assign), tp=tp, fp=fp,
                precision=tp / max(1, len(assign)),
                recall=tp / max(1, matchable), matchable=matchable)


# ---------------------------------------------------------------------------
# 4. Experiments and figures


def run_regime(texture_fn, label):
    disp = ground_truth_disparity()
    left, right, valid = render_pair(texture_fn(), disp)
    pl, _ = detect(left)
    pr, _ = detect(right)
    dl, dr = census(left, pl), census(right, pr)
    S = build_problem(pl, dl, pr, dr)
    m, n = S.shape
    lam = gam = -0.1

    a_masda, hist = masda(S, lam, gam)
    a_nn = mutual_nn(S, lam)
    a_opt = optimal_lap(S, lam, gam)

    rows = {}
    for name, a in (("MASDA", a_masda), ("Mutual-NN", a_nn),
                    ("Optimal LAP", a_opt)):
        r = evaluate(a, pl, pr, disp, valid)
        r["objective"] = objective(a, S, lam, gam, m, n)
        rows[name] = r

    uniq = len(set(int(v) for v in dl))
    margin = score_margin(S)
    return dict(label=label, margin=margin, left=left, right=right, disp=disp,
                valid=valid,
                pl=pl, pr=pr, dl=dl, dr=dr, S=S, rows=rows, hist=hist,
                a_masda=a_masda, a_nn=a_nn, a_opt=a_opt,
                uniq=uniq, kp=len(pl), edges=int(np.isfinite(S).sum()))


def fig_scene(r):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.4))
    ax[0].imshow(r["left"], cmap="gray"); ax[0].set_title("left image")
    ax[1].imshow(r["right"], cmap="gray"); ax[1].set_title("right image")
    im = ax[2].imshow(r["disp"], cmap="magma")
    ax[2].set_title("ground-truth disparity [px]")
    fig.colorbar(im, ax=ax[2], fraction=0.04)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(f"{FIG}/scene.png", dpi=110); plt.close(fig)


def fig_descriptors(rd, rb):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    for r, c, name in ((rd, "#d62728", "projected dots"),
                       (rb, "#1f77b4", "broadband texture")):
        i = np.arange(len(r["dl"]))
        j = RNG.integers(0, len(r["dr"]), len(i))
        h = hamming(r["dl"], r["dr"][j])
        ax[0].hist(h, bins=np.arange(0, 49), histtype="step", density=True,
                   color=c, label=name)
        uniq = len(set(int(v) for v in r["dl"]))
        ax[1].bar(name, uniq / len(r["dl"]), color=c)
    ax[0].axvline(24, color="k", ls="--", lw=0.8, label="chance (24 of 48 bits)")
    ax[0].set_xlabel("Hamming distance to a random other keypoint")
    ax[0].set_ylabel("density"); ax[0].legend(fontsize=8)
    ax[0].set_title("descriptor separability")
    ax[1].set_ylabel("distinct descriptors / keypoints")
    ax[1].set_ylim(0, 1.05); ax[1].set_title("descriptor degeneracy")
    bits = np.array([[(int(d) >> b) & 1 for b in range(48)] for d in rd["dl"][:60]])
    ax[2].imshow(bits, cmap="gray_r", aspect="auto", interpolation="nearest")
    ax[2].set_xlabel("Census bit"); ax[2].set_ylabel("keypoint")
    ax[2].set_title("descriptors, projected dots (60 keypoints)")
    fig.tight_layout(); fig.savefig(f"{FIG}/descriptors.png", dpi=110); plt.close(fig)


def fig_associations(r, name):
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.6))
    for a, (assign, ttl) in zip(ax, ((r["a_nn"], "Mutual-NN + ratio test"),
                                     (r["a_masda"], "MASDA"))):
        a.imshow(r["left"], cmap="gray", alpha=0.55)
        ok = bad = 0
        for i, j in assign.items():
            x, y = r["pl"][i]
            d_est = r["pl"][i, 0] - r["pr"][j, 0]
            d_true = r["disp"][int(y), int(x)]
            good = abs(d_est - d_true) <= 1.0
            ok += good; bad += not good
            a.plot([x, x - d_est], [y, y], "-",
                   color=("#2ca02c" if good else "#d62728"), lw=0.8, alpha=0.9)
            a.plot(x, y, ".", ms=2.5,
                   color=("#2ca02c" if good else "#d62728"))
        a.set_title(f"{ttl} — {ok} correct (green), {bad} wrong (red)",
                    fontsize=10)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"associations drawn as disparity vectors, {name}", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/associations_{name}.png", dpi=115)
    plt.close(fig)


def fig_comparison(rd, rb, rx=None):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    methods = ["Mutual-NN", "MASDA", "Optimal LAP"]
    cols = ["#7f7f7f", "#d62728", "#1f77b4"]
    for k, (metric, ttl) in enumerate((("recall", "recall (of matchable)"),
                                       ("precision", "precision"))):
        wd = 0.35
        x = np.arange(len(methods))
        ax[k].bar(x - wd/2, [rd["rows"][m][metric] for m in methods], wd,
                  label="repetitive", color=cols)
        ax[k].bar(x + wd/2, [rb["rows"][m][metric] for m in methods], wd,
                  label="broadband", color=cols, alpha=0.45)
        ax[k].set_xticks(x); ax[k].set_xticklabels(methods, fontsize=8)
        ax[k].set_ylim(0, 1.05); ax[k].set_title(ttl); ax[k].legend(fontsize=7)
    ax[2].semilogy(rd["hist"], color="#d62728", label="repetitive")
    ax[2].semilogy(rb["hist"], color="#1f77b4", label="broadband")
    ax[2].set_xlabel("iteration"); ax[2].set_ylabel("max message change")
    ax[2].set_title("convergence"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/comparison.png", dpi=110); plt.close(fig)


def fig_damping(S, lam, gam):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for d in (0.0, 0.2, 0.4, 0.6):
        _, h = masda(S, lam, gam, iters=60, damping=d)
        ax[0].semilogy(h, label=f"damping {d:.1f}")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("max message change")
    ax[0].set_title("damping and stability"); ax[0].legend(fontsize=8)
    ds = np.linspace(0, 0.8, 9)
    objs = []
    for d in ds:
        a, _ = masda(S, lam, gam, iters=40, damping=d)
        objs.append(objective(a, S, lam, gam, *S.shape))
    opt = objective(optimal_lap(S, lam, gam), S, lam, gam, *S.shape)
    ax[1].plot(ds, np.array(objs) / opt, "o-", color="#d62728")
    ax[1].axhline(1.0, color="k", ls="--", lw=0.8, label="optimal LAP")
    ax[1].set_xlabel("damping"); ax[1].set_ylabel("objective / optimal")
    ax[1].set_title("solution quality vs damping"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/damping.png", dpi=110); plt.close(fig)


def main():
    import os
    os.makedirs(FIG, exist_ok=True)
    rb = run_regime(broadband_texture, "broadband")
    rd = run_regime(dot_texture, "dots")
    rp = run_regime(periodic_texture, "periodic")

    for r in (rb, rd, rp):
        print(f"\n=== {r['label']} ===")
        print(f"keypoints L={r['kp']}  candidate edges={r['edges']} "
              f"(k={r['edges']/max(1,r['kp']):.1f} per keypoint)  "
              f"distinct descriptors={r['uniq']}/{r['kp']}")
        mg = r["margin"]
        print(f"  score margin best-vs-second: median {np.median(mg):.4f}, "
              f"frac below 0.05 = {float((mg < 0.05).mean()):.3f}  "
              f"<- ambiguity")
        print(f"  {'method':<13} {'matches':>8} {'correct':>8} {'wrong':>7} "
              f"{'prec':>7} {'recall':>7} {'objective':>10}")
        for name, v in r["rows"].items():
            print(f"  {name:<13} {v['matches']:>8} {v['tp']:>8} {v['fp']:>7} "
                  f"{v['precision']:>7.3f} {v['recall']:>7.3f} "
                  f"{v['objective']:>10.2f}")
        print(f"  matchable keypoints (not occluded): {r['rows']['MASDA']['matchable']}")

    fig_scene(rp)
    fig_descriptors(rp, rb)
    fig_associations(rp, "periodic")
    fig_associations(rb, "broadband")
    fig_comparison(rp, rb, rd)
    fig_damping(rp["S"], -0.1, -0.1)
    print(f"\nfigures written to {FIG}/")


if __name__ == "__main__":
    main()
