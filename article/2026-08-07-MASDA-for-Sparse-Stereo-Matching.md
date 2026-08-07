---
layout: post
title: 'MASDA for Sparse Stereo Matching'
subtitle: What mutual exclusivity buys you, measured against ground truth
thumbnail-img: https://raw.githubusercontent.com/mayio/mayio.github.io/master/assets/img/2026-08-07-MASDA-for-Sparse-Stereo-Matching_files/associations_periodic.png
date: '2026-08-07 19:00:00 +0200'
categories: association
comments: false
mathjax: true
author: Mario Lüder
---

In [Faster Data Association with Max-Sum Loopy Belief Propagation
(MASDA)](https://www.mariolueder.com/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA/)
I derived MASDA on the classic tracking problem: measurements to objects, with
clutter and misdetection. This post applies it to a different problem with the
same structure — **sparse stereo matching** — and measures what it actually buys.

Stereo is a good test case for three reasons. The one-to-one constraint is
physically real: a surface point projects once into each image. Ground truth is
obtainable, so "did it get the right answer" is a question with an answer rather
than a plausibility argument. And there is a strong, well-understood baseline to
compare against — the exact linear assignment problem, solved by
Jonker–Volgenant — so approximation quality is measurable rather than assumed.

The short version of the results:

- MASDA **attains the exact LAP optimum** on these problems (objective ratio
  1.0000, 1.0000, 0.9910 across three texture regimes).
- Against mutual nearest-neighbour with a ratio test, MASDA finds **2.75× more
  correct matches** where the texture is genuinely ambiguous, and only ~1.4% more
  where it is not. The advantage scales with ambiguity, which is the whole point.
- But **precision collapses for both methods** in the ambiguous regime.
  Mutual exclusivity is information, and it is real information — it is not enough
  information. That negative result is more useful than the positive one.
- On speed, the representation is the whole story. A dense implementation is 2×
  *slower* than scipy's compiled Jonker–Volgenant; the same algorithm on an edge
  list is **157–282× faster than JV** at identical quality. Section 6.

Everything here is reproducible from a single self-contained script — no image
files, so nothing carries a licence, and the scene is generated procedurally so
the disparity is known exactly.

---

## 1. The problem, and why it is the same problem

In tracking, measurements $i \in \{1 \dots m\}$ must be associated with objects
$j \in \{1 \dots n\}$, at most one each way, with the option of declaring a
measurement to be clutter or an object to be misdetected.

In sparse stereo, replace those words. Keypoints detected in the **left** image are
the measurements; keypoints in the **right** image are the objects. A left keypoint
may correspond to at most one right keypoint and vice versa, because one surface
point produces one projection in each image. A left keypoint whose surface is
**occluded** in the right view has no correct partner at all — that is clutter
($\lambda$). A right keypoint whose partner the left detector simply missed is a
misdetection ($\gamma$).

The occlusion case is not a technicality. In the scene below, 30% of left
keypoints have no attainable match, either because the surface is hidden in the
right view or because the corresponding right keypoint was never detected. Any
formulation that assumes every keypoint is matchable will be wrong about a third of
the time.

What stereo adds over tracking is **geometry**. The image pair is rectified, so a
correspondence lies on the same image row, and disparity $d = x_L - x_R$ is
positive and bounded. That makes the association graph extremely sparse — a point
we will return to, because it is what decides whether MASDA is fast.

### 1.1 The factor graph

Unchanged from the tracking case. Binary association variables $c_{ij}$, clutter
indicators $e_i$, misdetection indicators $\delta_j$; similarity factors $S_{ij}$,
clutter factors $\Lambda_i$, misdetection factors $\Gamma_j$, and the two
exclusivity constraints $I_i$ (each measurement used at most once) and $E_j$ (each
object used at most once).

The graph is loopy: every $c_{ij}$ participates in both an $I_i$ and an $E_j$
constraint, so cycles of length four abound. That is why this is *loopy* belief
propagation and why convergence is not guaranteed.

### 1.2 Messages

The updates are exactly as before,

$$
\begin{align}
\beta_{ij} &= s(i,j) - \max_{k \neq i} \rho_{kj} \\
\rho_{ij}  &= s(i,j) - \max_{k \neq j} \beta_{ik}
\end{align}
$$

with the two non-association options entering as alternatives that those maxima
compete against:

$$
\begin{align}
\rho_{ij}  &= s(i,j) - \max\!\left(\lambda,\; \max_{k \neq j} \beta_{ik}\right) \\
\beta_{ij} &= s(i,j) - \max\!\left(\gamma,\; \max_{k \neq i} \rho_{kj}\right)
\end{align}
$$

Read $\rho_{ij}$ as: *how good is associating $i$ with $j$, after subtracting the
best thing $i$ could otherwise do* — where "otherwise" includes being declared
clutter. And $\beta_{ij}$ symmetrically from the object side.

Damping, as before, applied to both:

$$
x^{(t+1)} \leftarrow (1-\eta)\, x_{\text{target}} + \eta\, x^{(t)}
$$

The belief combines both directions,

$$
b_{ij} = \alpha_{ij} + \eta_{ij} + s_{ij}
$$

and since $\beta_{ij} = s_{ij} + \alpha_{ij}$ and $\rho_{ij} = s_{ij} +
\eta_{ij}$, this is computed as

$$
b_{ij} = \beta_{ij} + \rho_{ij} - s_{ij}
$$

**Complexity.** Both maxima exclude exactly one element, so with the largest and
second-largest of each row and column cached, "max excluding $j$" is $O(1)$. One
iteration is then two linear passes over the **edges**:

$$
O(T \cdot E), \qquad E = |\{(i,j) : \text{candidate}\}|
$$

not $O(T \cdot m \cdot n)$. In stereo $E \approx 2.3\,m$, so the difference is
nearly three orders of magnitude. Section 6 shows what happens if you ignore this.

### 1.3 The score, and the scale of $\lambda$ and $\gamma$

Descriptors are the **Census transform** over a $7 \times 7$ window: one bit per
neighbour, set if that neighbour is darker than the centre, giving 48 bits that fit
a `uint64` so the distance is one `popcount`. Census is invariant to any monotonic
intensity mapping, which absorbs gain and offset differences between the two
cameras for free — a real concern, since the two sensors of a stereo pair are never
photometrically identical.

Two independent Census descriptors agree on half their bits by chance, so scaling
the Hamming distance $h$ about that point gives an interpretable score:

$$
s(i,j) = \underbrace{\frac{B/2 - h(i,j)}{B/2}}_{\text{+1 perfect, 0 chance}}
       \;-\; w_y \left(\frac{y_i - y_j}{\sigma_y}\right)^{2}
$$

with $B = 48$. The second term is the vertical residual: on a rectified pair a true
match has $y_i = y_j$, so deviation is evidence against. It is worth including even
when small, because the median $|\Delta y|$ over accepted matches is then a free
online monitor for calibration drift.

With $s$ on this scale, $\lambda = \gamma = -0.1$ reads as *"reject anything worse
than a tenth of the way from chance to perfect"*, which is a statement one can
reason about rather than a tuned constant.

---

## 2. A scene with known answers

Real stereo footage cannot tell you whether a match is correct without a
rangefinder. Synthetic data can, so the scene is generated:

```python
def ground_truth_disparity():
    d = np.full((H, W), 8.0, np.float32)          # back wall, far
    yy, xx = np.mgrid[0:H, 0:W]
    floor = yy > H * 0.55
    d[floor] = 8.0 + (yy[floor] - H * 0.55) * 0.28  # slanted floor
    d[(xx > 300) & (xx < 400) & (yy > 90) & (yy < 250)] = 26.0   # box
    d[(xx > 150) & (xx < 162) & (yy > 40) & (yy < 300)] = 38.0   # thin bar
    return d
```

Three features, each chosen because it breaks something:

- a **slanted floor**, so the two views sample it at different rates —
  foreshortening, which strictly violates one-to-one;
- a **box** at intermediate depth, whose edges generate **occlusion**;
- a **thin vertical bar** in front, which violates the ordering constraint that
  scanline stereo methods rely on.

The right image is produced by forward-warping the left by the true disparity with
a z-buffer, so occlusions appear on their own rather than being modelled:

```python
for x, x2, d in zip(xs[ok], xr[ok], disp[y][ok]):
    if d > zbuf[y, x2]:          # nearer surface wins
        zbuf[y, x2] = d
        right[y, x2] = left[y, x]
```

![scene](https://raw.githubusercontent.com/mayio/mayio.github.io/master/assets/img/2026-08-07-MASDA-for-Sparse-Stereo-Matching_files/scene.png)

### 2.1 Three texture regimes

The same geometry is rendered with three textures, because **the texture, not the
geometry, decides whether association is hard**:

| regime | what it is | why |
|---|---|---|
| **broadband** | multi-scale noise | every descriptor is individually discriminative; the easy case |
| **dots** | pseudo-random blobs | imitates an IR projector, as on a RealSense D435 |
| **periodic** | regular lattice | repetitive structure — brick, fencing, tiling. The classic hard case. |

---

## 3. Visualising the ambiguity

Descriptor *degeneracy* and matching *ambiguity* are not the same thing, and it is
worth separating them because the first is easy to measure and the second is what
matters.

![descriptors](https://raw.githubusercontent.com/mayio/mayio.github.io/master/assets/img/2026-08-07-MASDA-for-Sparse-Stereo-Matching_files/descriptors.png)

Measured over ~1400–2000 keypoints per image:

| regime | distinct descriptors | median score margin | fraction with margin < 0.05 |
|---|---|---|---|
| broadband | 1319 / 1405 (94%) | 0.771 | **5.9%** |
| dots | 1506 / 1614 (93%) | 0.750 | **5.1%** |
| periodic | 1501 / 2019 (74%) | **0.083** | **39.8%** |

The **score margin** — best candidate minus runner-up, per left keypoint — is the
quantity that decides difficulty, and it is what a ratio test keys on. In the
periodic regime the median margin is **0.083 against 0.771**, a factor of nine, and
**40% of keypoints have essentially tied candidates**.

Note what the distinct-descriptor count does *not* tell you: the dot texture has
93% distinct descriptors, almost identical to broadband, yet these are supposed to
be the ambiguous case. Anti-aliasing and sensor noise make every window unique as a
bit pattern while leaving it uninformative as an identity. **Counting distinct
descriptors overstates how much information they carry.** The margin measures the
thing itself.

---

## 4. Implementation

The whole solver, in the notation above. Note `top2_excluding`, which is what makes
each iteration linear:

```python
def masda(S, lam=-0.1, gam=-0.1, iters=30, damping=0.4, eps=1e-5):
    finite = np.isfinite(S)
    beta = np.where(finite, 0.0, -np.inf)
    rho  = beta.copy()

    for it in range(iters):
        # rho: a row's other options, or declaring this measurement clutter
        comp = np.maximum(lam, top2_excluding(np.where(finite, beta, -np.inf), 1))
        new_rho = np.where(finite, (1 - damping) * (S - comp) + damping * rho, -np.inf)

        # beta: a column's other options, or declaring this object misdetected
        comp = np.maximum(gam, top2_excluding(np.where(finite, new_rho, -np.inf), 0))
        new_beta = np.where(finite, (1 - damping) * (S - comp) + damping * beta, -np.inf)

        delta = max(np.nanmax(np.abs(new_rho - rho)),
                    np.nanmax(np.abs(new_beta - beta)))
        rho, beta = new_rho, new_beta
        if delta < eps:
            break

    belief = np.where(finite, beta + rho - S, -np.inf)
    return decide(S, belief, lam)
```

### 4.1 The decision rule, and a mistake worth documenting

Reading out the answer is where I got it wrong first, and the error is instructive
because it is a direct consequence of the theory.

I gated acceptance on $b_{ij} > 0$. On problems with exactly tied candidates that
returned **zero matches** — on problems whose optimum matched everything.

The reason: the belief measures an edge's advantage **over its competitors**. When
nothing has an advantage, every belief is $\le 0$. That is precisely the condition
under which the LP relaxation has no unique optimum, and Bayati, Shah and Sharma's
correctness result for max-product on bipartite matching requires exactly that
uniqueness. So the degenerate case is not a corner case to be patched — it is the
documented boundary of the guarantee.

> Bayati, M., Shah, D., & Sharma, M. (2008). *Max-Product for Maximum Weight
> Matching: Convergence, Correctness, and LP Duality.* IEEE Transactions on
> Information Theory, 54(3), 1241–1251.
> [doi:10.1109/TIT.2007.915695](https://doi.org/10.1109/TIT.2007.915695)

Two different questions were being conflated:

| question | answered by |
|---|---|
| **which** candidate? | the belief $b_{ij}$ — an *ordering*, whose sign means nothing |
| associate **at all**? | $s(i,j)$ against $\lambda$ — which is what $\lambda$ *is* |

So: order by belief, decide by $\lambda$, require mutual agreement between row and
column, then complete greedily in belief order over whatever remains.

```python
def decide(S, belief, lam):
    order = np.argsort(-belief, axis=None)
    used_i, used_j, out = np.zeros(m, bool), np.zeros(n, bool), {}
    for flat in order:
        i, j = divmod(int(flat), n)
        if not np.isfinite(belief[i, j]):
            break
        if S[i, j] <= lam or used_i[i] or used_j[j]:
            continue
        out[i] = j
        used_i[i] = used_j[j] = True
    return out
```

The greedy completion is not a cosmetic addition. Under near-ties every row's best
belief points at the same column, so mutual agreement alone commits exactly one
pair. Each greedily accepted edge has $s > \lambda$ and two free endpoints, so it
strictly raises the objective — and adding it improved agreement with exhaustive
search from 56/60 to **58/60** on small problems where the optimum can be
enumerated.

---

## 5. Results against ground truth

$\lambda = \gamma = -0.1$, 30 iterations, damping 0.4. **Recall** is measured
against *attainable* matches only: a left keypoint counts in the denominator only
if its true correspondence is both unoccluded and was itself detected in the right
image. Counting unattainable matches against recall would penalise the matcher for
the detector's behaviour.

### broadband — the easy case

| method | matches | correct | wrong | precision | recall | objective |
|---|---|---|---|---|---|---|
| Mutual-NN + ratio | 797 | 685 | 112 | **0.859** | 0.804 | 489.82 |
| **MASDA** | 840 | 694 | 146 | 0.826 | **0.815** | **507.83** |
| Optimal LAP (JV) | 840 | 695 | 145 | 0.827 | 0.816 | 507.83 |

### dots — projector-like

| method | matches | correct | wrong | precision | recall | objective |
|---|---|---|---|---|---|---|
| Mutual-NN + ratio | 851 | 691 | 160 | **0.812** | 0.792 | 491.88 |
| **MASDA** | 902 | 708 | 194 | 0.785 | **0.812** | **521.21** |
| Optimal LAP (JV) | 902 | 706 | 196 | 0.783 | 0.810 | 521.21 |

### periodic — genuinely ambiguous

| method | matches | correct | wrong | precision | recall | objective |
|---|---|---|---|---|---|---|
| Mutual-NN + ratio | 396 | 71 | 325 | 0.179 | 0.125 | **−8.21** |
| **MASDA** | 1109 | **196** | 913 | 0.177 | **0.344** | 623.27 |
| Optimal LAP (JV) | 1111 | 194 | 917 | 0.175 | 0.341 | 628.92 |

![associations](https://raw.githubusercontent.com/mayio/mayio.github.io/master/assets/img/2026-08-07-MASDA-for-Sparse-Stereo-Matching_files/associations_periodic.png)

![comparison](https://raw.githubusercontent.com/mayio/mayio.github.io/master/assets/img/2026-08-07-MASDA-for-Sparse-Stereo-Matching_files/comparison.png)

### 5.1 What these numbers say

**MASDA attains the optimum.** Objective ratio to exact Jonker–Volgenant is
1.0000, 1.0000 and 0.9910. Loopy max-sum, with no guarantee, lands on the LAP
optimum on all three problems. That is the headline result and it is worth stating
plainly, because it is not what "approximate inference on a loopy graph" primes you
to expect.

**The advantage over nearest-neighbour scales with ambiguity.** Correct matches:

| regime | median margin | MASDA | Mutual-NN | ratio |
|---|---|---|---|---|
| broadband | 0.771 | 694 | 685 | **1.01×** |
| dots | 0.750 | 708 | 691 | **1.02×** |
| periodic | 0.083 | 196 | 71 | **2.76×** |

Where descriptors are discriminative, mutual exclusivity adds almost nothing — 1–2%
— and a ratio test is a perfectly good matcher. Where they are not, MASDA finds
**2.76× more correct correspondences**. Look at mutual-NN's objective in the
periodic case: **−8.21**, i.e. *negative*. The ratio test rejects so much that it
pays more in clutter and misdetection cost than it earns in matches. It is not
making a bad trade-off; it is declining to trade.

**And precision collapses for everyone.** 0.177 for MASDA, 0.179 for mutual-NN,
0.175 for the exact optimum. This is the most useful number in the post.

Mutual exclusivity is real information and it is used optimally here — the exact
LAP solver does no better. But on a truly repetitive pattern the information simply
is not present in the descriptors, and no amount of constraint propagation
manufactures it. MASDA converts a *refusal to answer* into *answers, most of which
are wrong*. Whether that is an improvement depends entirely on what consumes the
output: a bundle adjustment with robust losses will take 196 correct matches out of
1109 gladly; a naive triangulation will be poisoned by it.

The honest framing is therefore not "MASDA beats nearest-neighbour" but **"MASDA
extracts everything the uniqueness constraint contains, which is a lot when
descriptors are ambiguous and not enough when they are degenerate."** That is what
motivates adding further factors — ordering along the scanline, disparity
smoothness over a neighbourhood graph — which is the subject of the next post.

### 5.2 Damping

![damping](https://raw.githubusercontent.com/mayio/mayio.github.io/master/assets/img/2026-08-07-MASDA-for-Sparse-Stereo-Matching_files/damping.png)

Undamped max-sum on the ambiguous problem does not settle; the largest message
change plateaus rather than decaying. Damping 0.3–0.5 stabilises it, and solution
quality is remarkably flat across that range — the objective stays within a
fraction of a percent of optimal.

Worth noting on real data (a D435 IR pair, 848×480, ~1100 keypoints): the messages
**never formally converge** at any iteration budget, yet the *decision* is stable
to four significant figures from 50 iterations and within 0.1% at 20. Oscillation
is confined to messages that do not change the answer. Convergence of the messages
is not the property you actually need.

---

## 6. On "faster" — and why the representation is the whole story

The complexity argument is $O(T \cdot E)$ against Jonker-Volgenant's $O(N^3)$. It is
worth measuring rather than asserting, and the first attempt is instructive.

**A dense implementation forfeits the entire argument.** Written the obvious way —
messages held in an $m \times n$ array padded with $-\infty$ — it is *slower* than
scipy's compiled Jonker-Volgenant:

| regime | nodes | edges | dense MASDA | JV (scipy) | speedup |
|---|---|---|---|---|---|
| broadband | 1405 | 3216 | 3897 ms | 1607 ms | **0.4×** |
| dots | 1614 | 3843 | 5336 ms | 2585 ms | **0.5×** |
| periodic | 2019 | 4529 | 7547 ms | 3898 ms | **0.5×** |

With $m \approx n \approx 1400$ the matrix holds ~2 million cells and **3216 real
edges** — a factor of 600 in arithmetic spent on entries that are $-\infty$. The
$O(T \cdot E)$ bound assumes an edge list; a dense array delivers
$O(T \cdot m \cdot n)$, and no amount of vectorisation recovers the difference.

### 6.1 The sparse formulation

Hold the messages on the **edges**. The only non-obvious part is that both updates
need $\max_{k \neq j}$ over a row or column, which naively is quadratic in the row
length. Three segment reductions answer it exactly in $O(E)$:

```python
def _seg_max_excluding(vals, idx, n):
    """Per-segment max with each element's own contribution removed, in O(E)."""
    m1 = np.full(n, -np.inf)
    np.maximum.at(m1, idx, vals)          # segment max
    at_max = vals >= m1[idx]              # m1 is the max, so >= means ==
    cnt = np.zeros(n, np.int64)
    np.add.at(cnt, idx[at_max], 1)        # how many attain it
    m2 = np.full(n, -np.inf)
    below = ~at_max
    if below.any():
        np.maximum.at(m2, idx[below], vals[below])   # max strictly below
    second = np.where(cnt > 1, m1, m2)
    return np.where(at_max, second[idx], m1[idx])
```

Three cases: an element below the max sees the max; an element *at* the max also
sees the max, provided something else attains it too; otherwise it sees the
runner-up. Ties are handled rather than assumed away, which matters because
near-ties are the regime of interest.

The solver is then five lines per iteration:

```python
def masda_sparse(ei, ej, se, m, n, lam=-0.1, gam=-0.1, iters=30, damping=0.4):
    beta = np.zeros(len(se)); rho = np.zeros(len(se))
    for _ in range(iters):
        comp = np.maximum(lam, _seg_max_excluding(beta, ei, m))
        new_rho  = (1 - damping) * (se - comp) + damping * rho
        comp = np.maximum(gam, _seg_max_excluding(new_rho, ej, n))
        new_beta = (1 - damping) * (se - comp) + damping * beta
        rho, beta = new_rho, new_beta
    belief = beta + rho - se
    ...
```

Identical mathematics — same messages, same damping, same belief, same decision
rule. Only the representation changes.

### 6.2 What that buys

| regime | edges | dense | **sparse** | JV | sparse vs dense | **sparse vs JV** |
|---|---|---|---|---|---|---|
| broadband | 3216 | 3897 ms | **10.2 ms** | 1607 ms | 382× | **157×** |
| dots | 3843 | 5336 ms | **12.6 ms** | 2585 ms | 424× | **206×** |
| periodic | 4529 | 7547 ms | **13.8 ms** | 3898 ms | 545× | **282×** |

**382–545× faster than the dense form, and 157–282× faster than compiled
Jonker-Volgenant** — from interpreted NumPy, against optimised C.

### 6.3 And the quality is unchanged

A speed claim is worthless without this. Evaluated against ground truth:

| regime | method | matches | correct | precision | recall | objective |
|---|---|---|---|---|---|---|
| broadband | dense | 840 | 694 | 0.826 | 0.815 | 507.83 |
| | **sparse** | 840 | **694** | 0.826 | 0.815 | **507.83** |
| | optimal LAP | 840 | 695 | 0.827 | 0.816 | 507.83 |
| dots | dense | 902 | 708 | 0.785 | 0.812 | 521.21 |
| | **sparse** | 902 | **707** | 0.784 | 0.811 | **521.21** |
| | optimal LAP | 902 | 706 | 0.783 | 0.810 | 521.21 |
| periodic | dense | 1109 | 196 | 0.177 | 0.344 | 623.27 |
| | **sparse** | 1109 | **196** | 0.177 | 0.344 | **623.27** |
| | optimal LAP | 1111 | 194 | 0.175 | 0.341 | 628.92 |

Objectives agree to four decimals. Correct-match counts are identical except one
match in 902 on the dot texture, where the two orderings break a tie differently.
The assignments are not bit-identical — with tied beliefs they need not be — but
they are equally good, and both sit at the LAP optimum.

As an independent check, the same algorithm as a C++ edge-list implementation on
real imagery (848×480 IR pair, ~1075 keypoints, 2882 candidate edges, 20
iterations) runs in **1.67 ms** against a 33.3 ms frame budget at 30 Hz — with the
keypoint detector, at 21 ms, dominating it more than tenfold.

### 6.4 The actual claim

MASDA's advantage is that **its cost is linear in the number of *plausible*
associations**, and in a geometrically constrained problem that is a tiny fraction
of $m \times n$. Here the epipolar band and disparity range cut ~2 million possible
pairings to ~3200 candidates, and only a representation that exploits that sees the
benefit.

Which reframes the comparison with an exact solver. It is not accuracy —
Jonker-Volgenant is exactly as good and occasionally marginally better. It is that
MASDA is anytime, incremental and extensible: it can be stopped early with a usable
answer, its messages carry between frames when the problem changes slightly, and it
accepts factors that destroy the assignment structure altogether, where a LAP solver
cannot follow.

## 7. Comparison with existing work

**Jonker–Volgenant / Hungarian.** Exact, $O(N^3)$, and here strictly at least as
good. If your problem is a pure assignment problem of moderate size, use it. MASDA
earns its place when you intend to add factors — smoothness, ordering, temporal
consistency — that make the problem no longer a LAP.

> Jonker, R., & Volgenant, A. (1987). *A shortest augmenting path algorithm for
> dense and sparse linear assignment problems.* Computing, 38(4), 325–340.
> [doi:10.1007/BF02278710](https://doi.org/10.1007/BF02278710)

**SPADA / sum-product data association.** Gives marginal association
probabilities rather than a MAP assignment, at higher cost. If downstream consumers
want soft weights — a PDA-style tracker, or a differentiable pipeline — that is the
right choice. For stereo, where a decision is needed per keypoint, MAP is what is
wanted.

**Sinkhorn / optimal transport, as in SuperGlue.** Structurally close to MASDA:
soft one-to-one association with explicit dustbins, which are precisely $\lambda$
and $\gamma$. Sinkhorn is the entropy-regularised relaxation; max-sum is the
zero-temperature limit. SuperGlue's advantage is that its scores come from a
learned attention network rather than a hand-designed $s(i,j)$, and its dustbin
costs are learned rather than set. That points directly at the weakest part of the
formulation here.

> Sarlin, P.-E., DeTone, D., Malisiewicz, T., & Rabinovich, A. (2020).
> *SuperGlue: Learning Feature Matching with Graph Neural Networks.* CVPR.
> [arXiv:1911.11763](https://arxiv.org/abs/1911.11763)

**Semi-global matching and dense stereo.** A different problem, and worth saying
why sparse matching is not simply worse. Dense stereo assigns a disparity to every
pixel with a smoothness prior on a pixel grid; per scanline, one-to-one with
occlusion is solved *exactly* by dynamic programming in $O(W \cdot D)$, and DP
additionally encodes the ordering constraint that MASDA cannot express. If you want
a dense disparity map, MASDA is the wrong tool. Sparse matching is the right tool
when you want a few hundred well-localised, sub-pixel correspondences to feed
geometry — odometry, calibration, structure — rather than a depth image.

> Hirschmüller, H. (2008). *Stereo Processing by Semiglobal Matching and Mutual
> Information.* IEEE TPAMI, 30(2), 328–341.
> [doi:10.1109/TPAMI.2007.1166](https://doi.org/10.1109/TPAMI.2007.1166)

**ELAS**, for the record, uses a triangulated set of robustly matched support
points as a prior for dense estimation — which is close to the sparse-then-densify
structure a MASDA front-end would naturally feed.

> Geiger, A., Roser, M., & Urtasun, R. (2010). *Efficient Large-Scale Stereo
> Matching.* ACCV.

---

## 8. Advantages, honestly bounded

**Where MASDA is the right choice.**

- Cost linear in *plausible* associations, not in $m \times n$. With geometric
  constraints cutting candidates to ~2.3 per keypoint, the sparse form runs
  157–282× faster than an exact LAP solver at identical quality — but only if the
  implementation exploits that sparsity.
- Optimal or indistinguishable from optimal on these problems, without needing to
  be.
- **Anytime**: usable after a handful of iterations, with the decision stabilising
  long before the messages do.
- **Extensible** in the direction that matters. Adding a smoothness or ordering
  factor keeps a factor graph a factor graph; it stops being a LAP.
- Clutter and misdetection are first-class, not post-hoc thresholds — which matters
  here because occlusion makes ~30% of keypoints genuinely unmatchable.

**Where it is not.**

- No convergence guarantee, and the guarantee that exists lapses exactly when the
  problem is ambiguous — which is when you wanted the help.
- It cannot express the **ordering** constraint along a scanline, which DP can, and
  which is real information for stereo.
- It cannot manufacture information. On degenerate texture it produces confident
  wrong answers where a ratio test produces no answer, and which of those is
  preferable is a property of the consumer, not of the matcher.
- $\lambda$ and $\gamma$ are hand-set. They have an interpretable scale here, which
  helps, but calibrating them properly — a small model over descriptor distance,
  $y$-residual, response ratio and local texture energy, trained against ground
  truth — is the largest single improvement available and is not done.

---

## 9. Reproducing this

One file, `numpy` + `scipy` + `matplotlib`. No inputs, so nothing to license:

```bash
python masda_stereo.py
```

It generates the scene, detects keypoints, computes Census descriptors, builds the
sparse candidate graph, runs MASDA, mutual-NN and exact Jonker–Volgenant, evaluates
all three against ground truth, and writes every figure in this post.

If you want to run it on real imagery, the standard benchmark is the Middlebury
stereo set, which supplies rectified pairs with dense ground truth. I have
deliberately not embedded any of it — the datasets are provided for research use
and redistribution is a licensing question I would rather not answer — but pointing
the loader at a local copy is a few lines.

> Scharstein, D., Hirschmüller, H., Kitajima, Y., Krathwohl, G., Nešić, N., Wang,
> X., & Westling, P. (2014). *High-resolution stereo datasets with subpixel-accurate
> ground truth.* GCPR.

---

## 10. What comes next

Two things follow from the precision collapse in §5.

**More factors.** Disparity smoothness over a Delaunay triangulation of the left
keypoints, and a soft ordering penalty along the scanline. Both add information the
current formulation ignores. Both also break the closed-form $\beta/\rho$ updates
and require deriving new messages — and both threaten the convergence guarantee
that the ambiguous case has already forfeited. That is the next post.

**Better scores.** $\lambda$, $\gamma$ and $s(i,j)$ are the weakest part of this,
and SuperGlue's success suggests learning them is worth more than any amount of
message-passing sophistication. The interesting question is whether $T$ max-sum
iterations can be unrolled with a soft-max relaxation and trained end-to-end —
structurally what SuperGlue does with Sinkhorn in the zero-entropy limit.

A caveat on all of the above: these numbers come from one synthetic scene. The
qualitative pattern — advantage scaling with ambiguity, optimality against JV,
precision collapse under degeneracy — is what I would expect to generalise. The
specific figures are not a benchmark.
