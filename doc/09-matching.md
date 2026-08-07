# MASDA — the matcher

The point of the project. Max-sum loopy belief propagation for bipartite data
association with mutual exclusivity, plus clutter and misdetection terms, used as
the matcher for sparse stereo correspondence.

```sh
cd core && make && make test              # 32 preprocessing + 22 matcher assertions
./build/de_match ../bags/full_on
```

## Messages

From the plan, with the misdetection and clutter options folded into the maxima
since "leave `i` unmatched" and "`j` is clutter" are competing explanations:

```
rho_ij  = s(i,j) - max( gamma,  max_{k != j} beta_ik )
beta_ij = s(i,j) - max( lambda, max_{k != i} rho_kj  )
```

Both maxima exclude one element, so each iteration is two **top-2 reductions** —
one over rows, one over columns — rather than a quadratic scan. That is what makes
the plan's ~10⁵–10⁶ ops per frame estimate achievable.

`s(i,j)` is a stand-in for a calibrated log-likelihood ratio: descriptor agreement
scaled so Hamming = bits/2 (chance) scores 0 and a perfect match scores +1, minus a
quadratic y-residual penalty. That puts `gamma = 0` at the interpretable place of
"no better than chance". Calibrating this properly is an open question in the plan,
and remains one.

## The decision rule, and a mistake worth recording

Two questions that an earlier version of this conflated:

| Question | Answered by |
|---|---|
| **Which** candidate? | the belief, `beta + rho - s`. It measures an edge's advantage over its competitors, so it is the right *ordering* — but its **sign is not a decision**. |
| Match **at all**? | `s(i,j)` against `gamma`. That is what gamma means. |

Gating acceptance on `belief > 0` returned **zero matches** on exactly-tied
problems whose optimum matched everything: when no candidate has an advantage,
every belief is ≤ 0. That is the plan's cited condition for BP's guarantee lapsing
— a non-unique LP optimum (Bayati, Shah & Sharma) — and it is not a corner case
here, because projected dots make descriptors ~3.3× degenerate.

The rule is therefore: order by belief, decide by gamma, require mutual agreement
between row and column, then **greedily complete** in belief order over what
remains. The completion pass is necessary rather than cosmetic — under near-ties
every row's best belief points at the same column, so mutual agreement commits
exactly one pair. Each greedily accepted edge has `s > gamma` and two free
endpoints, so it strictly raises the objective. Adding it also improved agreement
with brute force, from 56/60 exact optima to **58/60**.

## Validated against brute force

Message passing that merely converges proves nothing; the question is whether it
converges to the assignment that maximises the objective. On 60 random problems
small enough to enumerate exhaustively:

| | |
|---|---|
| valid one-to-one matchings | **60 / 60** |
| exact optimum reached | **58 / 60** |
| total objective vs optimal | **0.998** |

## Results on real stereo pairs

848×480, FAST keypoints, 7×7 Census, same candidate graph for both methods.

| | matches | objective | median \|dy\| | median disparity | time |
|---|---|---|---|---|---|
| **emitter ON** (projected dots) | | | | | |
| MASDA | **615** | **410.9** | 0.359 px | 37.9 px | 1.67 ms |
| mutual-NN + ratio test | 421 | 288.7 | 0.326 px | 36.8 px | 0.36 ms |
| *MASDA advantage* | ***+46.3%*** | ***+42.3%*** | | | |
| **emitter OFF** | | | | | |
| MASDA | **206** | **115.2** | 0.423 px | 20.0 px | 0.43 ms |
| mutual-NN + ratio test | 183 | 106.6 | 0.404 px | 19.9 px | 0.12 ms |
| *MASDA advantage* | ***+13.0%*** | ***+8.1%*** | | | |

**The advantage scales with ambiguity, which is the plan's entire argument.**
+46% with the projector on, +13% with it off. MASDA is worth its cost precisely
where descriptors are degenerate, and only modestly worth it where they are not.

**Median |dy| = 0.359 px** is independent corroboration that the matches are real:
a wrong match has no reason to land on the same image row, and this is the plan's
free rectification-health metric. Note MASDA's |dy| is *slightly worse* than
nearest-neighbour's — expected, since it accepts ~46% more and the additional ones
are the harder cases.

**MASDA is not the bottleneck, as the plan predicted.** 1.67 ms against a 33.3 ms
budget is 5%, versus 26 ms for preprocessing.

## Convergence: the messages oscillate, the decision does not

| iterations | matches | objective | oscillating iterations | time |
|---|---|---|---|---|
| 20 | 617.5 | 408.71 | 0 | 1.44 ms |
| 50 | 618.0 | 409.12 | 2 | 2.42 ms |
| 100 | 618.0 | 409.29 | 8 | 4.16 ms |
| 200 | 618.0 | 409.29 | 32 | 8.15 ms |

It **never formally converges** — the largest message change stays above 1e-4 at
every budget — yet the answer is stable to four significant figures from 50
iterations, and to within 0.1% at 20. Meanwhile the count of iterations where the
change *grew* rises steadily, so a subset of messages genuinely oscillates.

This is the plan's warning showing up early, before any loopy pairwise factors
exist to justify it. It is currently **benign**: oscillation is confined to
messages that do not change the decision. Worth re-checking the moment smoothness
factors are added, which is when the plan expects the guarantee to break properly
and damping to need raising past 0.5.

**20 iterations is the operating point.** More costs time and buys 0.1%.

## Coarse-to-fine: built, measured, and **not** worth it here

The plan calls this the biggest single lever, for two compounding reasons: cutting
candidates per keypoint from ~100–200 down to 8–16, and removing the false
candidates from repetitive structure that create near-ties. It is implemented as
the plan specifies — a **soft** prior, never hard pruning, since hard pruning at
the coarse level is how thin structures get lost.

The prior is self-widening in the way the plan asks for, and that part works: each
cell's sigma is the MAD of the coarse disparities landing in it, so a cell
straddling a depth discontinuity automatically gets a wide, weak prior. Measured
in the tests: sigma goes from 2.0 px on a clean surface to **83 px** across a
discontinuity, and a cell with too few coarse matches imposes no prior at all.

**But it makes the result worse on real data** (emitter on, `bags/full_on`):

| configuration | k | matches | objective |
|---|---|---|---|
| single level | 2.7 | **615.2** | **410.9** |
| coarse-to-fine 1/8 | 2.7 | 609.5 | 405.5 |
| coarse-to-fine 1/4 | 2.7 | 509.0 | 325.3 |
| coarse-to-fine 1/4, `w_prior` 0.25 | 2.7 | 537.5 | — |

Lowering the prior's weight recovers some of the loss but never beats single level.

### Why the premise does not hold here

**k is already 2.7, not 100–200.** The plan's motivation assumed a wide search, but
this pair is rectified, so a ±2 px epipolar band plus a disparity range already
does the restricting a coarse pass was meant to do.

Better evidence than that arithmetic: deliberately widening the search does not
change the answer at all.

| max_dy / max_candidates | k | matches |
|---|---|---|
| 2 / 24 | 2.7 | 615.2 |
| 6 / 64 | 7.2 | **615.2** |
| 12 / 128 | 13.8 | **615.2** |

Inflating k by 5× leaves the result *identical*, because the y-residual term in
`s(i,j)` already scores those extra candidates below gamma. **There are no false
candidates left for a disparity prior to remove.** So the prior can only contribute
its own error — coarse Census on downsampled projected dots is much less
distinctive, and coarse disparity quantises to ±0.5 px, which is ±2–4 px once
scaled up.

Prior coverage also shows the mechanism: only **1%** of cells at 1/8, because the
106×60 coarse image yields too few keypoints to reach three matches per cell. At
1/4 coverage rises to 29% and the damage rises with it — the prior is active more
often, and being active is what hurts.

### When it would matter

Kept in the code, off by default, because the premise could return:

- **Unrectified or drifted calibration**, where the epipolar band must widen and k
  genuinely grows.
- **Much higher keypoint density**, where per-row candidates multiply.
- **Temporal matching** between consecutive frames, which the plan wants for
  ego-motion. There the search is 2-D and unconstrained by rectification, so k
  really is large and a prior — from IMU-derived rotation compensation — should pay.

This is the plan's own instruction followed: measure before deriving. The measurement
says don't.

## The cheap smoothness experiment: also does not help

The plan's instruction was to run plain MASDA, fit a robust disparity surface over
a neighbourhood graph, re-score `s(i,j)` against it, re-match, two or three passes
— because that captures most of the smoothness benefit inside the existing
closed-form updates and "reveals whether the full derivation is worth it".

Implemented: a robust local plane fit `d = ax + by + c` per keypoint over its 10
nearest matched neighbours, by IRLS with Tukey weights so a depth discontinuity in
the neighbourhood drops out rather than dragging the plane between two surfaces.
(A k-nearest-neighbour graph rather than the Delaunay triangulation the plan names
— a proxy, and worth revisiting if this line is ever resumed.)

**Iterating makes it monotonically worse.** With `w_smooth = 1`:

| pass | matches | base objective | median \|dy\| | prior coverage |
|---|---|---|---|---|
| 1 | **615.2** | **410.85** | 0.364 | — |
| 2 | 402.5 | 284.75 | 0.355 | 100% |
| 3 | 297.0 | 210.72 | 0.363 | 100% |
| 4 | 233.2 | 161.47 | 0.375 | 100% |

And it is not a weight-tuning problem. Sweeping `w_smooth` down two decades, pass 3
against the 615.2 / 410.85 / 0.364 baseline:

| `w_smooth` | matches | base objective | median \|dy\| |
|---|---|---|---|
| 0.02 | 602.8 | 403.73 | 0.368 |
| 0.05 | 594.5 | 398.58 | 0.373 |
| 0.10 | 569.0 | 385.50 | 0.372 |
| 0.25 | 506.5 | 344.55 | 0.382 |

Monotonic in every column, with no optimum at any positive weight. **The best value
of `w_smooth` is 0.**

Note the objective is scored *without* the smoothness term, which is the only
honest comparison — adding a term to an objective and then reporting the objective
rose would measure nothing.

### Why, probably

The most likely reason is the same one that sank coarse-to-fine, and it is
structural rather than a tuning failure: **the prior is fitted from the very
matches it then judges.** Where the matches are already right it can only pull them
off; where they are wrong the fit is wrong in the same way, so it reinforces the
error instead of correcting it. A smoothness prior derived from the current
solution is close to self-confirming.

## The limitation both experiments expose

**Neither result is conclusive in the way that matters, because there is no ground
truth.** The available measures are match count, objective, and median |dy|, and
none of them can distinguish "removed 100 wrong matches" from "removed 100 right
ones". Median |dy| barely moved across every configuration tried, so it is not
sensitive enough to arbitrate.

So the honest reading is narrower than "smoothness does not help": *at this
operating point, on this one static scene, with no way to check correctness, both
priors reduce match count and neither improves the only independent quality metric
available.* That is a reason to stop adding priors, not proof that the information
they encode is worthless.

**The blocker for further algorithmic work is bring-up step 4** — the laser
rangefinder against walls at 1, 2 and 3 m. Until disparities can be checked against
a known distance, additions to `s(i,j)` cannot be evaluated, only observed. The
plan puts step 4 before driving for exactly this reason; it turns out to gate the
matcher's development too.

## What is not done
- **Fixing the timing comparison.** The coarse-to-fine path times detection plus
  matching, while the single-level MASDA figure times matching only. The two
  numbers are not comparable and the conclusion above rests on match counts and
  objective rather than on timing.
- **Calibrating gamma and lambda.** Currently hand-set. The plan's candidate is a
  small MLP over descriptor distance, y-residual, coarse-disparity residual and
  response ratio.
- **The full smoothness factor derivation.** The cheap version was the plan's test
  of whether this is worth doing, and it came back negative — but see the ground
  truth caveat above before treating that as settled. Eigen 3.4 is on the desktop
  and 3.3.4 on the Jetson if it is resumed; the current 3x3 fit needs neither.
- **Sub-pixel disparity refinement**, which the plan notes depth accuracy hinges on.
- **Ground-truth validation** against a laser rangefinder — bring-up step 4. Until
  then, median |dy| and the objective are the only quality measures, and neither
  proves the disparities are *correct*, only that they are consistent.
