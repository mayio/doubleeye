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

**MASDA is not the bottleneck, as the plan predicted** -- but the margin is
narrower than this table first suggested, because the table was measured on the
desktop.

The timings above are desktop timings. `de_match` had never been built on the
Jetson at all: `tools/deploy.sh` synced only `jetson/`, and the `core/build`
directory that was sitting on the Jetson held x86-64 objects copied from the
desktop. The first native build there was 2026-08-07. Measured on the same
`full_on` bag:

| | desktop | Jetson (MAXN, 6 cores) |
|---|---|---|
| MASDA | 1.38 ms | **5.04 ms** |
| mutual-NN + ratio | 0.34 ms | 0.82 ms |
| matches / objective | 615 / 410.9 | 620 / 412.7 |

So the correct statement is 5.04 ms against a 33.3 ms budget, which is 15%, not
5%. Still not the bottleneck against 26 ms of preprocessing, but a 3.7x factor
that should not have been reported as if the two numbers came from the same
machine. The match counts and objectives agree across the two (615 vs 620), so
this is the same work on slower cores rather than a different problem.

## Score margin, exported per match

Every `Match` now carries `margin`: best-minus-second-best s(i,j) over that left
keypoint's candidates, measured against lambda where there is no runner-up. It is
close to free -- one pass over the edge list -- and it is the most useful
confidence available. Over eight Middlebury scenes with ground truth, precision by
margin quartile runs 0.169 / 0.286 / 0.391 / 0.659.

On `full_on` the margin immediately explains what MASDA is doing:

| | matches | median margin | margin < 0.2 |
|---|---|---|---|
| MASDA | 620 | 0.388 | **31.5%** |
| mutual-NN + ratio | 422 | 0.535 | 14.3% |

MASDA accepts 47% more matches, and the extra ones are concentrated in the
low-margin band -- which is exactly what a ratio test exists to remove. Neither
number is better on its own; the point is that the consumer can now see which
matches are the doubtful ones and weight or drop them, rather than receiving 620
matches with no way to tell them apart.

## lambda and gamma were transposed

`MatchConfig::lambda` and `MatchConfig::gamma` were named the wrong way round
relative to the article's convention: lambda is clutter, the cost of leaving a
LEFT (measurement) keypoint unmatched, and gamma is misdetection, the cost of
leaving a RIGHT (object) keypoint unmatched. The code used one field consistently
for the left side throughout, so the mathematics was self-consistent and no
measured result was ever wrong -- but only because every experiment so far has
been run with lambda == gamma. They stop being interchangeable the moment they
are set apart, which is what stereo wants: a left keypoint occluded in the right
view is a different rate from a right keypoint the detector never proposed.

`test_lambda_gamma_are_distinct` in `core/tests/test_match.cpp` is the regression
guard. Its first version compared two match counts that were both zero, which
passed without testing anything; it now asserts the exact counts (1 and 0).

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

## Detector repeatability: over-propose on the right, then gate on margin

Repeatability, not the matcher, is the ceiling. Pooled over eight Middlebury
scenes only 44.6% of left keypoints have any right keypoint within 1 px of their
true correspondence. Shi-Tomasi picks its maxima independently per image, and a
maximum in one view is often not a maximum in the other. No matcher can recover a
correspondence that was never proposed.

Over-proposing on the right lifts it, with diminishing returns past 2x:

| right density | repeatability | right kp | edges | correct | precision |
|---|---|---|---|---|---|
| baseline | 44.6% | 5688 | 13873 | 1796 | 0.616 |
| 2x | 52.5% | 10305 | 20873 | **2057** | 0.573 |
| 3x | 53.8% | 14245 | 24192 | 2089 | 0.555 |
| 4x | 54.0% | 17497 | 26057 | 2091 | 0.541 |

+261 correct matches at 2x, which is +14.5%, for 1.5x the edges and 1.27x the
time. Precision falls though, 0.616 to 0.573, because the extra right keypoints
are also extra distractors.

That is where the exported margin pays for itself. Gating the 2x run on margin
recovers the precision and keeps most of the gain:

| | correct | precision |
|---|---|---|
| baseline, no gate | 1796 | 0.616 |
| 2x, margin >= 0.05 | 1989 | 0.610 |
| **2x, margin >= 0.10** | **1957** | **0.632** |

At a margin threshold of 0.10 the over-proposed run is better on *both* axes than
the baseline: +161 correct matches and +0.016 precision. The whole
precision/recall curve moves outward rather than trading along it, which is
visible at matched precision anywhere on it:

| precision | baseline correct | 2x + margin gate correct |
|---|---|---|
| ~0.705 | 1644 (gate 0.20) | 1782 (gate 0.25) |
| ~0.785 | 1471 (gate 0.40) | 1506 (gate 0.45) |

Recommended configuration: 2x right-image density, margin gate at 0.10, left
detector unchanged. Both are now `de_match` flags, `--right-density` and
`--min-margin`.

The gate is applied AFTER matching, not before. Dropping low-margin candidates
before the solver would remove the competition that gives the surviving margins
their meaning; the gate is a decision about what to hand the consumer.

### Measured on the Jetson, not extrapolated

I estimated 6.4 ms from the Python edge-count ratio of 1.27x. Measured on
`full_on`, 120 pairs, MAXN:

| config | matches | median \|dy\| | margin < 0.2 | Jetson time |
|---|---|---|---|---|
| baseline | 620.1 | 0.365 px | 31.5% | 5.14 ms |
| 2x right + margin 0.10 | 603.8 | 0.364 px | 22.2% | **8.84 ms** |

The real factor is 1.72x, not 1.27x, so the cost is 8.84 ms: 27% of the frame
budget rather than a fifth. The estimate was wrong because the C++ detector is
`detect_keypoints_fast` at cell 32, and doubling per_cell there does not add
candidate edges in the same proportion as the Python sweep's cell-12 dense
Shi-Tomasi. Extrapolating a Jetson cost from a desktop ratio is the same mistake
as the 1.67 ms figure, one step removed.

### What is NOT yet verified

`full_on` has no ground truth, so the accuracy claim does not transfer to it. What
the numbers above show is the gate working mechanically: the low-margin share
falls from 31.5% to 22.2%, and median |dy| is unchanged at 0.364 px. Fewer
doubtful matches is not the same statement as more correct ones.

The +161 correct / +0.016 precision result is validated on Middlebury, in Python,
with a dense Shi-Tomasi detector at cell 12. The C++ path uses FAST candidates
with Shi-Tomasi scoring at cell 32. Those are different detectors in a different
density regime, so the transfer is plausible and unproven.

Closing that gap means running the C++ matcher on Middlebury directly, against
ground truth, which needs the pairs converted to the raw Y8 layout the bag loader
expects. That is the next piece of work, and until it is done the recommended
configuration is a recommendation from the Python experiment, not from the C++
one.

### One thing that did not work

Making misdetection cheaper, so the surplus right keypoints are discarded more
readily, does nothing. Sweeping gamma from -0.1 to 0.0 at every density moves
correct matches by at most 5 out of ~2000, which is noise.

The reasoning was that over-proposing means many right keypoints legitimately go
unmatched, so leaving one unmatched ought to be cheap. It was the first intended
use of lambda and gamma being separable at all. But a surplus right keypoint is
usually nobody's best candidate, so it goes unmatched whatever gamma says: gamma
only bites when a right keypoint is genuinely contested, and the extra ones mostly
are not. The lambda/gamma fix was still worth making, since the naming was wrong
and the fields are not interchangeable, but this particular payoff is not there.
