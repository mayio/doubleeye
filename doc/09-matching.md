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

### Closed: the C++ matcher, measured against ground truth

`core/tools/de_bench.cpp` runs the C++ matcher and the C++ detector on the
Middlebury scenes, with the same evaluation rules as the Python side (unknown
ground truth excluded from precision rather than scored wrong; matchable requires
the partner to have been detected). `article/export_middlebury.py` writes the
pairs as raw Y8 plus a float32 disparity map, since core/ has no image decoder and
should not gain one for a benchmark.

Eight scenes, `--sweep`:

| right/cell | margin gate | matches | correct | precision | recall | ms/scene |
|---|---|---|---|---|---|---|
| 3 (baseline) | 0.00 | 2026 | 1402 | 0.706 | 0.937 | 0.38 |
| 3 | 0.10 | 1816 | 1323 | 0.743 | 0.884 | 0.38 |
| 6 | 0.00 | 2585 | **1706** | 0.673 | 0.899 | 0.58 |
| **6** | **0.10** | 2251 | **1585** | **0.718** | 0.836 | 0.62 |
| 9 | 0.00 | 2743 | 1772 | 0.659 | 0.882 | 0.83 |
| 9 | 0.10 | 2338 | 1622 | 0.708 | 0.807 | 0.80 |
| 12 | 0.10 | 2382 | 1615 | 0.693 | 0.791 | 0.95 |

**The recommendation holds on the C++ path.** Over-proposing at 2x lifts correct
matches by 21.7% (1402 to 1706), and adding the 0.10 margin gate lands at 1585
correct with precision 0.718 -- better than the baseline on *both* axes, +183
correct and +0.012 precision. At matched precision the curve is outside the
baseline's too: baseline gives 1250 correct at 0.762, and 2x with a 0.30 gate
gives 1375 at the same 0.762.

The gain is larger here than in Python (+13.1% against +9%), which is consistent
with the C++ detector proposing far fewer right keypoints to begin with (3607
across the eight scenes against Python's 5688) and so having more headroom.

Note the different regime rather than assuming the numbers are interchangeable:
recall reads much higher here (0.937 against 0.716) because `matchable` is
relative to what the detector proposed, and FAST at cell 32 proposes a smaller,
easier set. The two experiments agree on the *direction and rough size* of the
effect, which is what was in question.

So `--right-density 6 --min-margin 0.10` is now justified by a ground-truth
measurement of the code that actually ships. What it costs on the Jetson at
848x480 is the 8.84 ms above, 27% of the frame budget.

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

## Sub-pixel disparity

Disparity was the difference of two independently refined keypoint positions.
Both are sub-pixel, but their errors are uncorrelated, and nothing in the
subtraction ever looked at how well the two windows line up. `refine_disparity`
fits a parabola to the SAD cost between the windows at d-1, d, d+1 and takes its
vertex: three window evaluations per match, run after matching, so the message
passing is untouched. The vertex offset is clamped to +-0.5 px, because a parabola
through three samples of a non-parabolic cost can put its vertex anywhere once the
curvature approaches zero, and that is how sub-pixel refinement quietly makes
disparity worse.

Measured on the eight Middlebury scenes, at the recommended configuration:

| refinement | correct | precision | median inlier error | within 0.5 px |
|---|---|---|---|---|
| off | 1585 | 0.718 | 0.196 px | 87.3% |
| **on** | **1601** | **0.725** | **0.167 px** | **94.5%** |

Median inlier error falls 14%, and the share of correct matches localised inside
half a pixel goes from 87.3% to 94.5% -- the tail outside half a pixel more than
halves, 12.7% to 5.5%. Correct matches and precision also rise slightly, since a
few matches sitting just outside the 1 px threshold move inside it.

Jetson agrees: 1583 -> 1599 correct, 0.196 -> 0.167 px, 87.2% -> 94.5%, and the
refinement does not show up in the timing.

One correction to how I framed this beforehand. I said median disparity error was
0.50 px and that this was pure integer-versus-quarter-pixel quantisation. That was
true of the *Python* experiment, whose detector returns integer positions. The C++
detector already has `subpixel = true`, so its error was 0.196 px before any of
this, and the headroom was a third of what I claimed. The gain is real but it is
in the tail, not the median.

## The frame budget does not close at 30 Hz

Measured end to end on the Jetson at MAXN, 848x480, `full_on`, 120 pairs:

| stage | ms per stereo pair | share of 33.3 ms |
|---|---|---|
| preprocessing, L and R concurrent (`--parallel`) | 26.54 | 79.6% |
| MASDA, baseline config | 5.14 | 15.4% |
| **total, baseline** | **31.68** | **95.1%** |
| MASDA, `--right-density 6 --min-margin 0.10` | 8.84 | 26.5% |
| **total, recommended config** | **35.38** | **106.2%** |

So the accuracy work does not fit at 30 Hz. The recommended configuration is over
budget, and the baseline leaves 1.6 ms of headroom on the mean while the worst
observed preprocessing pair alone is 33.2 ms, which is the whole budget before a
single message is passed.

Two things this settles.

**The matcher is not the problem and optimising it further is not the answer.**
Detection is 20.98 ms of the 21.24 ms per frame; Census is 0.25 ms. Even reducing
MASDA to zero leaves preprocessing at 80% of the budget. The `_seg_max_excluding`
port that was on the backlog would have been busywork twice over: the C++ already
does an O(1) `max excluding one element` through `Top2`, since that trick is a
NumPy workaround for something a plain loop does naturally, and the matcher is not
where the time goes.

**The choice is a real engineering trade, not a tuning question.** Options, in the
order I would try them:

1. Run the matcher at a lower rate than the camera. Correspondence at 15 Hz with a
   66.7 ms budget is comfortable (53% with the recommended config) and for
   odometry on an indoor vehicle 15 Hz is likely adequate. This costs nothing but
   latency.
2. Attack detection, which is 63% of the budget on its own. FAST already replaced
   the dense Shi-Tomasi scan; the remaining candidates are the Shi-Tomasi scoring
   of FAST corners and the NMS. Neither has been profiled at that level.
3. Drop resolution. 640x480 is 75% of the pixels of 848x480 and preprocessing is
   close to pixel-bound.

What should not happen is quietly shipping the recommended configuration and
discovering the frame drops on the vehicle. It is off by default and the flags are
opt-in.

## Where the 21 ms goes, and what the two levers actually buy

`de_profile` times the three detector stages and sweeps resolution. Jetson, MAXN,
`full_on`, 40 frames, medians, per image for the detector stages:

| resolution | kp/img | FAST cand | stage 1 FAST | stage 2 NMS | stage 3 refine | census | match | pair | %budget |
|---|---|---|---|---|---|---|---|---|---|
| 848x480 | 1075 | 22200 | 11.34 | 4.93 | 4.28 | 0.22 | 5.11 | 25.89 | 77.7% |
| 424x240 | 278 | 6946 | 3.36 | 1.46 | 1.95 | 0.05 | 0.98 | 7.81 | 23.4% |
| 282x160 | 118 | 2638 | 1.39 | 0.51 | 0.58 | 0.02 | 0.34 | 2.85 | 8.6% |

Stage 1 is 55% of detection, not the overwhelming majority I assumed from "it
touches every pixel while the rest is sparse". NMS and refinement are 45% between
them, because FAST nominates 22200 candidates per image to produce 1075 keypoints:
95% of them are discarded only after both sparse stages have paid for them.

Throughput is 26-31 Mpx/s across the three resolutions, so preprocessing is
essentially pixel-bound and resolution scales as expected.

### The FAST threshold is the better lever

Nobody had tried it. The threshold of 8 was chosen against the DENSE detector, on
the grounds that it keeps 96% of its keypoints at 1.45x the speed. That is a
quality comparison, not a budget one.

Jetson preprocessing, concurrent, 40 pairs:

| fast_threshold | keypoints/img | preprocessing ms/pair | %budget |
|---|---|---|---|
| 8 (current) | 1074 | 26.75 | 80.3% |
| **12** | 947 | **18.84** | **56.5%** |
| 16 | 761 | 14.58 | 43.7% |
| 20 | 611 | 11.39 | 34.2% |
| 30 | 375 | 7.91 | 23.7% |

And against ground truth, at the recommended matcher configuration, raising it
costs correct matches roughly in proportion while leaving quality alone:

| fast_threshold | correct | precision | median inlier err | within 0.5 px |
|---|---|---|---|---|
| 8 | 1601 | 0.725 | 0.167 px | 94.5% |
| 12 | 1441 | 0.724 | 0.167 px | 94.7% |
| 16 | 1344 | 0.725 | 0.167 px | 94.2% |
| 20 | 1211 | 0.717 | 0.167 px | 94.0% |
| 30 | 932 | 0.721 | 0.167 px | 95.5% |

Precision is flat to within 0.008 and sub-pixel accuracy does not move at all.
The threshold buys time by discarding keypoints, not by discarding *good* ones,
and there is no cliff anywhere in the range.

### Threshold beats resolution, clearly

Normalising both levers against what they cost:

| change | time saved | cost |
|---|---|---|
| fast_threshold 8 -> 12 | 30% | 10% of correct matches |
| fast_threshold 8 -> 20 | 57% | 24% of correct matches |
| 848x480 -> 424x240 | 70% | 74% of keypoints |

Raising the threshold is a much better trade at every point, and it has a second
advantage the table does not show: median inlier disparity error stays at 0.167 px
across the whole threshold sweep, whereas halving resolution halves the pixels a
disparity is measured in and so costs depth precision directly.

So resolution is the lever of last resort, not the second option. Recommendation:
`fast_threshold = 12`, which puts preprocessing at 56.5% of budget and should
leave the recommended matcher configuration fitting inside 30 Hz.

### Measured: the whole stack fits at 30 Hz

`de_profile --fast-threshold 12 --right-density 6 --min-margin 0.10`, which is
every improvement switched on including sub-pixel refinement, Jetson, 40 pairs:

| resolution | kp/img | FAST | NMS | refine | census | match | pair | %budget |
|---|---|---|---|---|---|---|---|---|
| **848x480** | 1307 | 8.30 | 3.74 | 2.92 | 0.25 | 7.86 | **23.07** | **69.3%** |
| 424x240 | 339 | 2.48 | 0.82 | 1.31 | 0.06 | 1.46 | 6.13 | 18.4% |

23.07 ms against 33.3 ms, so about 10 ms of headroom, with more keypoints per
image than the 848x480 baseline had (1307 against 1075, because the right image is
over-proposing). My arithmetic guess was ~27 ms; measured is 23.07. Third estimate
in this document, third time the measurement differed -- but at least this one
differed in the useful direction.

So: `fast_threshold 12`, `--right-density 6`, `--min-margin 0.10`, sub-pixel on.
That is the configuration to run, and it now has a measured budget and a
ground-truth accuracy number behind every part of it.

## The disparity gate was the biggest quality problem, and it was untuned

`MatchConfig` defaults to `min_disparity 1.0, max_disparity 220.0`. With
f·B = 21.48 px·m that is a search range of **0.10 m to 21.5 m**. Every measurement
on my own IR bags used it, and in a room it admits several times more depth than
exists.

The cost is not just outliers. Extra candidates per keypoint are extra competition,
so the score margin falls, and the margin is what predicts precision. Measured live
on the same scene, 10 fps, ~78 frames each:

| | gate [1, 220] px = 0.10-21.5 m | gate [3.6, 53.7] px = 0.40-6.0 m |
|---|---|---|
| points per frame | 553 | **646** |
| Z median | 0.58 m | 0.82 m |
| Z p1 / p99 | 0.10 m / 5.74 m | 0.42 m / 4.49 m |
| median margin | 0.388 | **0.657** |
| margin < 0.2 | 31.5% | **11%** |

So tightening the gate gives *more* matches, at nearly double the median margin,
with a third as many doubtful ones, and a depth distribution that looks like a room
instead of like noise. On the wide gate a quarter of all points came out nearer than
0.19 m, piled against the limit, which is what a wrong match at a gate boundary
looks like.

This also qualifies an earlier number in this document. The 31.5% of matches below
margin 0.2 reported for `full_on` was measured through the wide gate; with a
plausible gate it is 11%. The Middlebury results are unaffected, since those set a
per-scene range of 60 or 80 px, which is already tight.

`de_pipe` takes `--min-disparity`/`--max-disparity` and prints the implied depth
range in its banner, so a wrong gate is visible rather than silent.
`desktop/de_live_ros2.py` takes `--min-range`/`--max-range` **in metres** and
defaults to 0.4-6.0 m, because that is the unit the scene is in.

## A dense baseline, and an uncomfortable number

`article/dense_baseline.py` runs OpenCV's semi-global matcher on the same eight
Middlebury scenes. It exists to answer "can we produce a dense depth image" with a
picture, and to make the sparse-versus-dense argument quantitative rather than
rhetorical. SGM is the family the D4 ASIC belongs to, so this is also the closest
proxy to the ASIC available offline.

Setting the comparison up fairly needs care, because the two answer different
questions. Dense SGM is scored the Middlebury way — the fraction of known-GT pixels
whose error exceeds 1 px, over the whole image. MASDA is scored at its own matched
keypoints, a much smaller denominator. So SGM is **also** scored at exactly MASDA's
keypoints, which is the only like-for-like number here.

| scene | SGM coverage | SGM bad-1.0 | at MASDA's keypoints: SGM | MASDA |
|---|---|---|---|---|
| teddy | 81.1% | 8.1% | 0.899 | 0.615 |
| cones | 81.7% | 5.5% | 0.943 | 0.781 |
| Art | 70.3% | 14.5% | 0.810 | 0.489 |
| Books | 80.7% | 9.7% | 0.935 | 0.704 |
| Dolls | 79.9% | 9.3% | 0.898 | 0.689 |
| Laundry | 77.6% | 17.7% | 0.491 | 0.301 |
| Moebius | 78.1% | 11.5% | 0.925 | 0.680 |
| Reindeer | 74.9% | 11.1% | 0.935 | 0.592 |
| **pooled** | **78.0%** | **10.9%** | **0.858** | **0.616** |

**Dense SGM is more accurate than MASDA at MASDA's own keypoints, 0.858 against
0.616, and it fills 78% of the image as well, in 17 ms.** That is not a small gap
and it should not be written around.

### What it does and does not mean

It is not "SGM beats MASDA" as a statement about inference. SGM aggregates matching
cost along eight paths through the image, which is a **smoothness prior over the
whole disparity field**. MASDA uses uniqueness and nothing else. So what this
measures is the value of the prior MASDA does not have, and the answer is: a great
deal. 0.616 to 0.858 on identical points and identical data.

That is the same conclusion [09](09-matching.md) and the article's section 10
already reached from the other direction — smoothness is the largest piece of
information the matcher currently ignores — but as an argument rather than a
number. The number is 24 points of precision.

Three things keep it from being the whole story:

- The two constraints are **orthogonal, not competing**. Uniqueness is a statement
  about the matching being one-to-one; smoothness is a statement about the surface.
  SGM has no uniqueness constraint at all, which is why it needs a left-right
  consistency check bolted on afterwards to reach 78% coverage rather than 100%.
  The interesting object is a matcher with both, which is what a smoothness factor
  in the factor graph would be.
- The earlier smoothness attempt failed for a specific and fixable reason. A
  two-pass prior fitted from the matches it then judges is self-confirming; SGM's
  path aggregation is a joint optimisation over the whole field, which is a
  different thing entirely.
- SGM declines 22% of known-GT pixels. MASDA declines by only ever proposing at
  keypoints. Neither number is coverage in the same sense.

### What this changes

The honest reading is that the current matcher is not competitive with a
well-implemented dense method on accuracy, and the gap is explained by a missing
prior rather than by anything about max-sum. That makes the smoothness factor the
highest-value piece of work on the matcher, ahead of everything currently in
section 0.5 of the TODO, and it makes `dense_baseline.py` the thing to measure it
against.

### Can this reach SGM? Measured from both ends: not as it stands

Two experiments, each cheap, and together they close the question without writing a
smoothness factor first.

**1. How many of MASDA's errors could any better inference fix?** For every wrong
match, ask whether the correct right keypoint was in that left keypoint's candidate
list at all. Pooled over the eight scenes, 2916 scorable matches:

| | share |
|---|---|
| correct | 61.6% |
| wrong, but the correct answer **was** a candidate | 8.1% |
| wrong, and the correct answer was **never offered** | 30.3% |

So a perfect re-ranking of the existing candidates — a smoothness factor that never
makes a mistake — tops out at **0.697**. SGM is at 0.858. The prior cannot close the
gap, because in 30% of cases the answer is not in the room.

Proposing more right keypoints does not rescue this. Sweeping the right detector
from 555 to 2994 keypoints per scene raises the fixable share (0.081 to 0.148) and
lowers the correct share by more (0.616 to 0.531), leaving the ceiling flat at
0.68-0.70 throughout. The correct correspondence frequently is not at *any* detected
keypoint; it is between them.

**2. So give it every pixel.** A semi-dense variant lets each left keypoint match
any right pixel on its row within the disparity range — 71 candidates per keypoint
instead of 5, so the correct answer is essentially always available. Uniqueness is
kept.

| variant | candidates/kp | precision | solver |
|---|---|---|---|
| keypoint-to-keypoint | 5 | 0.616 | 7 ms |
| **semi-dense, keypoint-to-pixel** | **71** | **0.587** | **68 ms** |
| dense SGM | every pixel | 0.858 | 17 ms |

**That 0.587 is a precision, and reading it alone was a mistake.** The two rows have
different denominators: semi-dense matched 5486 scorable keypoints against 2916, so
it produced **3221 correct matches against 1796 — 79% more**. I first wrote this up
as "it gets worse", which is the precision-versus-count trap this document warns
about elsewhere, committed here.

**And with the margin gate it is better on both axes at once.** With 71 candidates
instead of 5 the margin becomes far more informative — a candidate that clearly
beats 70 others is almost certainly right, where beating 4 others means little:

| margin gate | matches | correct | precision |
|---|---|---|---|
| keypoint-to-keypoint baseline | 2916 | 1796 | 0.616 |
| semi-dense, no gate | 5486 | 3221 | 0.587 |
| **semi-dense, gate 0.05** | 2956 | **2255** | **0.763** |
| semi-dense, gate 0.10 | 2058 | 1728 | 0.840 |
| semi-dense, gate 0.15 | 1465 | 1292 | **0.882** |

At gate 0.05 that is **+459 correct matches and +0.147 precision** over the sparse
baseline. At gate 0.10 it reaches 0.840, and at 0.15 it reaches 0.882 — **past SGM's
0.858** at the cost of fewer points.

### What the two together say

**Yes, it reaches SGM — on precision, and without a smoothness factor.** Semi-dense
candidates plus the margin gate gives 0.840 at gate 0.10 and 0.882 at 0.15, against
SGM's 0.858. Uniqueness and Census are enough, provided the correct answer is
actually offered *and* the confidence measure is used to discard the cases where the
descriptor could not decide.

That is a better result than the one this section originally reached, and it was
hidden by two mistakes of mine in the same afternoon. First, comparing precisions
across different denominators and concluding semi-dense was worse when it produced
79% more correct matches. Second, evaluating it ungated, when the whole point of the
margin is that it becomes *more* informative as candidates multiply: winning against
70 rivals means something that winning against 4 does not.

Two honest qualifications:

- The comparison is not perfectly like-for-like. SGM's 0.858 is measured at MASDA's
  2916 keypoints; the gated semi-dense numbers are over its own surviving matches,
  which is a smaller and self-selected set. Matching SGM's precision on fewer points
  is a real result, but it is not the same statement as beating it everywhere.
- It stays sparse. SGM still fills 78% of the image, and no gating of keypoint
  matches produces a depth map.

**The cost is the real obstacle.** The semi-dense solver is 68 ms per 450x375 scene,
three times the Jetson's whole frame budget at a fifth of the pixels. So this is not
the 30 Hz on-vehicle matcher as written — but it is now a quality ceiling worth
optimising towards, rather than a direction excluded on principle. The candidate set
is 71 per keypoint of which the gate discards most; generating fewer, better
candidates is an obvious avenue and has not been tried.

## MASDA as a dense matcher: it works, and it shows exactly what uniqueness cannot do

`article/dense_masda.py` runs MASDA over every pixel rather than over keypoints.
That is possible without changing the solver because the problem decomposes:
correspondences lie on one image row, so rows share no left or right index, and the
whole image is a single assignment problem whose exclusivity constraints happen to
couple only within rows. 9.15M edges on a 450x375 pair with 60 disparities.

| | coverage | bad-1.0 teddy | bad-1.0 cones | runtime |
|---|---|---|---|---|
| dense MASDA | **89.0%** | 34.7% | 24.6% | 21 s (NumPy) |
| SGM | 81.1% | **8.1%** | **5.5%** | 24 ms (C++ SIMD) |

![dense masda](../article/figures/dense_masda.png)

It produces a real depth image, and with *more* coverage than SGM. The scene
structure is plainly there: the bear, the cones, the plant. It is also covered in
salt-and-pepper speckle, because every pixel decides independently and uniqueness
constrains only that two left pixels may not claim the same right pixel. Nothing
says a pixel should agree with its neighbour.

The margin gate trades coverage for accuracy, and on this problem the trade is bad:

| gate | coverage | bad-1.0 |
|---|---|---|
| 0.00 | 89.0% | 34.7% |
| 0.10 | 34.8% | 15.9% |
| 0.20 | 16.7% | 9.4% |
| 0.30 | 4.9% | **7.4%** |

Matching SGM's error rate costs everything: 4.9% of pixels against SGM's 81.1%.

### The unifying result

Put beside the keypoint experiments, this explains all of them at once.

- At **keypoints** — which are by construction the places with enough texture to
  localise — semi-dense candidates plus the margin gate reach 0.882 precision
  against SGM's 0.858.
- At **every pixel**, including the textureless majority, the same method reaches
  34.7% bad against SGM's 8.1%.

Uniqueness is sufficient where the descriptor has something to say, and cannot
substitute for smoothness where it does not. SGM's path aggregation propagates
disparity from textured regions into textureless ones; a per-pixel descriptor
comparison has no mechanism for that, however the candidates are arranged.

Which is the argument for the sparse formulation, arrived at from the opposite
direction: keypoints are not a computational shortcut, they are a *selection of the
pixels where this class of matcher has information*. Running it everywhere else
mostly manufactures speckle.

### On the runtime column

21 s against 24 ms is a factor of ~900, and it is not an algorithmic comparison.
MASDA here is NumPy with `np.maximum.at` scatter-reductions over 9.15M edges; SGM is
compiled C++ with SIMD and decades of optimisation. The C++ MASDA on a sparse
problem is already ~200x faster than the NumPy one, so the honest statement is that
the constant is large and unmeasured, not that the algorithm is 900x slower.

What *is* algorithmic: 9.15M edges for a 168k-pixel image is 54 candidates per
pixel, and the margin gate then discards most of what was scored. Coarse-to-fine or
a previous-frame prior would cut that, and neither has been tried.

### Coarse-to-fine: 5.3x faster, and it damages the confidence measure

The obvious way to cut 54 candidates per pixel is to bracket the answer with a
coarse pass. Half resolution, inpaint the holes, upsample, then search ±4 px around
that guess at full resolution:

| | edges | runtime | coverage | bad-1.0 |
|---|---|---|---|---|
| flat dense | 9,154,890 | 21 s | 89.0% | 34.7% |
| coarse-to-fine | 1,448,007 | **4.0 s** | 85.4% | 33.0% |

**5.3x faster for a slightly better ungated error rate.** As a speed change it works
exactly as intended.

The problem appears once the margin gate is applied. Compared at similar coverage,
coarse-to-fine is worse:

| | coverage | bad-1.0 |
|---|---|---|
| flat dense, gate 0.20 | 16.7% | **9.4%** |
| coarse-to-fine, gate 0.20 | 29.1% | 19.0% |
| flat dense, gate 0.10 | 34.8% | **15.9%** |
| coarse-to-fine, gate 0.10 | 48.7% | 23.6% |

The mechanism is worth naming because it is not obvious. When the coarse pass is
wrong, the ±4 px window **excludes the correct answer entirely**. The fine pass then
runs a competition among candidates that are all wrong, and one of them wins it
convincingly — so the margin comes out *high* on a wrong match. Coarse-to-fine does
not merely propagate coarse errors, it launders them into confident ones.

That makes it strictly a speed/quality trade rather than a free win, and it damages
the one signal this matcher has that SGM does not. It also independently reproduces
the earlier coarse-to-fine negative result from the article, by a different route
and with the mechanism identified.

If the speedup is wanted, the fix is to keep a few candidates *outside* the bracket
so the margin still measures a real competition — untested.

## Dense MASDA in C++: 105x the NumPy version

`core/tools/de_dense.cpp`. The NumPy dense matcher spends almost all of its 21 s in
scatter — `np.maximum.at` over a 9.15M-entry edge list, an indirection per element —
and the structure does not need an edge list at all.

On a regular disparity grid, indexing scores as `s[(y*W + x)*D + d]` makes both
max-sum reductions constant-stride walks:

- **rho** ("max over this left pixel's other disparities") is a contiguous run of D.
- **beta** ("max over this right pixel's other claimants") walks
  `(xr+dmin)*D + k*(D+1)` — stride D+1.

No edge list, no indirection. Rows share no left or right pixel, so they go to a
thread pool with no synchronisation beyond the join.

| version | time (450x375, D=60, 12 iters) | coverage | bad-1.0 |
|---|---|---|---|
| NumPy | 21,000 ms | 89.0% | 34.7% |
| C++, first cut | 286 ms | 86.9% | 33.8% |
| **C++, optimised** | **199 ms** | 86.9% | 33.8% |
| *OpenCV SGM, for scale* | *24 ms* | *81.1%* | *8.1%* |

**105x the NumPy version, and 8x SGM rather than 900x.** The remaining gap to SGM is
now a plausible constant — both are compiled, both are threaded — rather than an
interpreter artefact.

Three changes took 286 ms to 199 ms, all verified output-identical:

- **Gather the strided diagonal into scratch.** The beta pass walked stride D+1 =
  244 bytes twice, once for the top-2 and once for the update: a cache miss per
  element, sixty times per pixel. Gathering 60 floats once turns that into one
  strided pass plus two contiguous ones.
- **Precompute the valid disparity interval per pixel.** Which disparities are
  in-bounds is a contiguous range, so storing `[k0, k1)` removes a sentinel
  comparison from every element of every inner loop.
- **Hoist the per-row allocations.** `solve_row` was allocating four `std::vector`s
  per row — 1500 allocations per image — now thread-local scratch allocated once.

The accuracy is unchanged from the NumPy version to within the decision rule's
tie-breaking, which is the point: this is the same algorithm, not an approximation
of it. It does not make dense MASDA competitive with SGM on quality — that limit is
the missing smoothness prior and no amount of optimisation touches it.

Not yet measured on the Jetson: the board was unreachable when this was written.

## Why SGM is better and faster — decomposed, and not what I said

I attributed the gap to the missing smoothness prior. Measured, that is mostly
wrong.

### Better: it is the cost function, not the smoothness

Turning SGM's smoothness penalties off (P1 = P2 = 0) degenerates its path
aggregation to plain winner-take-all on the block cost. If smoothness were the
explanation, that should collapse. It does not:

| SGM variant | coverage | bad-1.0 |
|---|---|---|
| full | 81.1% | 8.1% |
| no post-filters (speckle / LR / uniqueness off) | 82.8% | 9.5% |
| **no smoothness (P1=P2=0), filters on** | 74.5% | **7.5%** |
| **no smoothness and no filters** | 80.9% | **12.7%** |
| dense MASDA, 2 iterations | 78.9% | **28.1%** |

Smoothness is worth a few points at most on these scenes. Stripped of both
smoothness and post-filtering, SGM still reaches 12.7% against MASDA's 28.1%.

So the bulk of the gap is the **matching cost**. OpenCV aggregates a
Birchfield-Tomasi absolute difference over a 5x5 block, producing a graded,
sub-pixel-capable score. MASDA compares one 48-bit Census descriptor and gets a
Hamming distance quantised to 49 levels. Census buys invariance to gain and offset,
which matters on a real stereo rig, and pays for it in discriminative power.

That reinstates what the article's section 10 ranked first and I had talked myself
out of: **better scores are the highest-value change**, ahead of smoothness. It also
explains why the sparse matcher does so well by comparison — at keypoints the Census
score is discriminative, and it is the textureless majority where 48 bits are not
enough.

### Faster: fewer passes, cheaper arithmetic, and no iteration

MASDA is iterative and SGM is not. SGM makes one forward and one backward sweep per
path direction; MASDA needs several message-passing rounds over the whole cost
volume, each with a top-2 reduction and a damped update in float.

And more iterations make MASDA *worse*, which was not obvious:

| iterations | solve | coverage | bad-1.0 |
|---|---|---|---|
| 1 | 40 ms | 56.2% | 30.5% |
| **2** | **45 ms** | 78.9% | **28.1%** |
| 4 | 81 ms | 82.9% | 30.1% |
| 12 | 183 ms | 86.9% | 33.8% |
| 20 | 283 ms | 87.8% | 34.8% |

Iterations buy coverage and spend accuracy: the extra matches are the ones
uniqueness forces onto ambiguous pixels that early rounds correctly left alone. At
two iterations MASDA is **58 ms total against SGM's 16 ms** — 3.6x, not the 8x
implied by running 12 iterations that were making it worse.

The rest is arithmetic width and tuning: SGM works in integer cost volumes, several
elements per SIMD lane, hand-written intrinsics, and decades of optimisation;
de_dense is float32 with a top-2 whose index tracking resists autovectorisation.

**Corrected default: 2 iterations.** 12 was inherited from the sparse keypoint
problem, where the candidate set is small and iteration helps. On the dense problem
it costs 3x the time to make the answer worse.

## Acting on it: window-aggregated cost closes most of the gap

The diagnosis said the matching cost, not the prior. Acting on it directly: sum the
Census score over a window at fixed disparity, which is the same thing SGM's
`blockSize` does, built as two separable box-filter passes over a disparity-major
cost volume.

| aggregation radius | coverage | bad-1.0 |
|---|---|---|
| 0 (single pixel, as before) | 77.2% | 26.6% |
| 1 | 84.6% | 12.4% |
| 2 | 85.7% | 11.0% |
| **3** | **86.1%** | **10.6%** |
| 4 | 86.3% | 10.7% |

**26.6% to 10.6%, and coverage rises from 77% to 86%** — better on both axes at
once, from one change. It also confirms the diagnosis: the score was the problem.

### Against SGM, at matched coverage

SGM trades coverage for error too, via `uniquenessRatio`, so a single point is not a
comparison. Its curve runs 75.6%/5.4% to 81.9%/8.6%. Interpolating it to our
operating point:

| | coverage | bad-1.0 |
|---|---|---|
| MASDA (agg 3, 2 iters, margin 0.02) | 77.4% | 8.6% |
| SGM at the same coverage | 77.4% | **6.2%** |

**Not there yet.** But the gap went from 3.5x worse (28.1% against 8.1%) to 1.4x
worse in one change, and MASDA ungated now has *higher* coverage than SGM at any of
its settings — 86.1% against a maximum of 81.9%.

### Sub-pixel interpolation does not help here

The obvious next candidate was quantisation: the dense output was integer disparity
while SGM interpolates to 1/16 px, and bad-1.0 has a one-pixel threshold. Fitting a
parabola to the cost at k-1, k, k+1 made it slightly **worse**, 8.8% against 8.6% at
matched coverage.

The reason is the aggregation that just helped so much: the cost is now a mean of
49-level Hamming scores over a 7x7 window, which is not locally parabolic enough for
a three-point fit to locate a real vertex. Left in behind `--subpixel`, off by
default. Sub-pixel on the *sparse* matcher works well (median inlier error 0.196 ->
0.167 px) because there the fit is on a SAD cost between two intensity windows,
which is smooth.

### What is left, in order

1. **A graded cost.** Census gives 49 levels and buys illumination invariance; SGM's
   Birchfield-Tomasi absolute difference is continuous. Combining the two -- Census
   for invariance, AD for gradation -- is the standard move and is untried here.
2. **Runtime.** 280 ms against SGM's 16 ms, and the cost-volume build now dominates
   rather than the solver. It recomputes a popcount per pixel per disparity with
   poor locality; hoisting the XOR into the slice loop and blocking over disparity
   would help.
3. Smoothness, which the measurements put last rather than first.

## What PatchMatch Stereo and AGAP have that we do not

Two methods worth reading against our measurements, because each attacks something
we have now measured as a limit.

> M. Bleyer, C. Rhemann and C. Rother (2011). *PatchMatch Stereo — Stereo Matching
> with Slanted Support Windows.* BMVC.
> [doi:10.5244/C.25.14](https://doi.org/10.5244/C.25.14)

> P. Yao, H. Zhang, Y. Xue and S. Chen (2019). *As-global-as-possible stereo
> matching with adaptive smoothness prior.* IET Image Processing.
> [doi:10.1049/iet-ipr.2018.5801](https://doi.org/10.1049/iet-ipr.2018.5801)

### 1. Slanted support windows — and this one could put us *ahead*

Our window aggregation assumes the disparity is constant across the window. It is
not. Teddy's floor is a slanted plane and Cones is a field of cones; averaging the
cost over a 7x7 block on a surface with a disparity gradient blurs the very signal
being aggregated, and the blur grows with window size — which is exactly why our
sweep flattened at radius 3-4 instead of continuing to improve.

PatchMatch Stereo estimates a **3D plane per pixel** — disparity *and* normal —
instead of a single fronto-parallel disparity, so the support window is warped to
the surface before the cost is summed.

The reason to care is strategic rather than incremental: **SGM has the same
fronto-parallel bias.** Its block cost and its P1/P2 penalties both assume
neighbouring pixels share a disparity, which is wrong on every slanted surface. So
this is not a way to catch up with SGM, it is a place where SGM is weak. Our current
gap at matched coverage is 8.6% against 6.2%, and slanted surfaces are a large part
of both scenes.

### 2. Propagation instead of an exhaustive disparity search

Our cost volume is O(W·H·D) and, since aggregation was added, **building it now
dominates runtime** — 280 ms of which the solver is a minority. Every pixel is
scored against all 60 disparities whether or not any of them is plausible.

PatchMatch does not enumerate. It seeds random hypotheses, **propagates good ones to
neighbours**, and refines by random search in a shrinking window. Cost becomes
independent of the disparity range, which is the single largest structural saving
available to us and attacks the one number where SGM is comfortably ahead.

There is also a natural fit with MASDA that is worth noticing. Our measurements kept
saying the *candidate set* is what decides the outcome — the 30.3% of errors where
the answer was never offered, and the semi-dense experiment where offering more
helped once the margin could filter. PatchMatch is precisely a method for
*generating good candidates cheaply*. MASDA supplies what PatchMatch lacks: a
principled one-to-one constraint and a confidence measure. Candidate generation by
propagation, arbitration by max-sum, is a coherent combination and neither paper
does it.

### 3. AGAP: the support region should follow the image, not a square

AGAP aggregates cost over a **minimum spanning tree of the whole image** rather than
along SGM's eight fixed 1-D paths, with an adaptive smoothness term. Support then
follows image structure: pixels connected through low-contrast regions aggregate
together, and an edge stops the smoothing.

Set against our numbers this is more promising than it first looks. We measured
SGM's *penalty-based* smoothness (P1/P2) as worth only a few points, and I have been
treating that as "smoothness does not matter here". But the box aggregation — which
is a support-region choice, not a penalty — was worth **16 points**. Those are
different mechanisms, and the evidence says the support region matters enormously
while the penalty does not. A box is the crudest possible support. Edge-aware or
tree-based support is the obvious next step, and MST filtering is O(N) and
non-iterative, so it is cheap.

### Ranked for this project

1. **Slanted support windows.** Attacks a weakness SGM shares, so it is a route to
   outperforming rather than matching. Also explains why our aggregation sweep
   plateaued.
2. **PatchMatch-style propagation for candidate generation.** Removes the O(D) cost
   volume that now dominates runtime, and fits what we already measured about
   candidate sets mattering more than inference.
3. **Edge-aware support instead of a box.** The box was worth 16 points; a better
   support region should be worth more, and it is cheap.
4. **Continuous disparity.** Comes free with a plane-based formulation, and would
   sidestep the sub-pixel parabola that measurably did not work here.

## Making it fast, and keeping it parallel

Profiling first, per rule. At 273 ms the split was **census 20 ms, cost volume
211 ms, solve 42 ms** — and the cost volume did not move with thread count at all.
It was the one serial stage, and it was 78% of the runtime. The solver, already
row-parallel, was scaling 4.7x.

The fix was free: each disparity slice is an independent H×W plane — score, box
filter, scatter — and slices write disjoint elements of the output volume. Threading
over disparity needs no locks and no atomics.

| threads | census | cost | solve | total |
|---|---|---|---|---|
| 1 | 28 ms | 200 ms | 177 ms | 405 ms |
| 2 | 15 ms | 125 ms | 91 ms | 230 ms |
| **4** | **9 ms** | **71 ms** | **46 ms** | **126 ms** |
| 8 | 12 ms | 93 ms | 48 ms | 152 ms |

**273 ms to 126 ms, output bit-identical** (86.1% coverage, 10.6% bad-1.0 before and
after).

### Eight threads is slower than four, and that is the useful part

This machine has 4 physical cores with 2 threads each. Going from 4 to 8 makes it
*worse*, which says the work is **memory-bandwidth bound rather than compute bound**:
hyperthreads share a core's load/store path, so they contend rather than help. The
cost volume is 40 MB written once and read once per solver iteration, and none of
the arithmetic is expensive — a popcount, an add, a compare.

That is worth knowing before optimising further. Vectorising the inner loops would
buy little if the machine is already waiting on memory; the useful moves are the
ones that reduce *traffic*: smaller types for the cost volume (uint8 or uint16
instead of float32 would quarter it), fusing the aggregation into the scatter so the
volume is written once instead of staged, or not materialising the whole volume at
all — which is what PatchMatch's propagation would give us.

### Why this shape parallelises later

Both axes here partition the work with no sharing at all:

| stage | parallel over | shared state |
|---|---|---|
| census | rows | none |
| cost volume + aggregation | **disparity** | disjoint writes |
| MASDA message passing | **rows** | none |
| decision | rows | none |

No locks, no atomics, no reductions across workers anywhere in the pipeline. That is
not an accident of the CPU implementation — it follows from the problem: matching is
within a row, so rows are independent, and a disparity slice is a self-contained
hypothesis.

Which is exactly the shape a GPU wants: one block per row for the solver, one thread
per (pixel, disparity) for the cost volume, and a separable box filter that is two
standard passes. The bandwidth-bound finding above argues *for* the GPU rather than
against it, since bandwidth is what a GPU has. It also lands on the right side of the
boundary already proposed in [10-architecture.md](10-architecture.md): dense
per-pixel work on the GPU, sparse irregular work on the CPU.

Nothing here needs restructuring to move. That was the point of choosing these two
axes.

## Testing the slanted-window premise before implementing it

PatchMatch's slanted support windows are a large piece of work, so the premise is
worth checking first: is our error actually concentrated where the surface is not
fronto-parallel? Binning bad-1.0 by the ground-truth disparity gradient:

| slant, \|∇d\| px/px | teddy pixels | teddy bad-1.0 | cones bad-1.0 |
|---|---|---|---|
| 0.00 - 0.05 | 80118 | 6.9% | 4.8% |
| 0.05 - 0.15 | 31392 | 6.2% | 5.7% |
| 0.15 - 0.30 | 16694 | 6.2% | 6.2% |
| 0.30 - 0.60 | 2159 | **24.0%** | 13.2% |
| 0.60 - 1.20 | 5059 | **50.9%** | 38.7% |
| > 1.20 | 5371 | **55.3%** | 49.3% |

Near-fronto-parallel pixels are at 6.7%; the rest are at 24.2%, and **high-gradient
regions carry about half of all error** on both scenes despite being a minority of
pixels. So the direction is real.

**But the bins say something the headline does not.** Error is flat up to a gradient
of 0.3 px/px and only explodes beyond it. A gradient above 0.6 px/px means more than
four pixels of disparity change across a 7-pixel window — that is not a slanted
surface, it is a **depth discontinuity**. So the high-gradient bucket is a mixture,
and the two halves need different fixes:

- **Genuine slant** (roughly 0.15-0.6) is what slanted support windows solve. On
  these scenes it is a small population: 2159 pixels on teddy.
- **Discontinuities** (>0.6) are the larger population and are not a slant problem at
  all. A warped window still straddles an occlusion boundary and still averages two
  surfaces. That needs edge-aware support or explicit occlusion handling — AGAP's
  contribution, not PatchMatch's.

And roughly half the total error is still in genuinely flat regions at 6-7%, which
is the cost-function problem, untouched by either.

**So the premise survives but the plan changes.** Slanted windows alone would address
the smaller half of the high-gradient error. The ranking that follows from these
numbers is edge-aware support first, slanted planes second, since the discontinuity
population is two to three times larger than the slanted one on both scenes. That is
the opposite of the order I proposed from reading the papers.

## Edge-aware aggregation: level with SGM, not ahead of it

The measurement said discontinuities carry most of the high-gradient error and a box
filter mixes two surfaces across them. Replacing the box with a **guided filter** —
support weighted by agreement with the left image, so an intensity edge stops the
aggregation — is the direct fix and is still O(N), since the guidance moments do not
depend on disparity and are computed once for the whole volume.

On teddy, at radius 5: **10.6% -> 8.8% bad-1.0** at slightly higher coverage. Gated
at margin 0.01 it reached 81.9% coverage / 7.7% bad against SGM's 81.9% / 8.6% —
same coverage, lower error.

**That did not survive contact with the other seven scenes.**

| scene | MASDA cov | MASDA bad | SGM cov | SGM bad |
|---|---|---|---|---|
| teddy | 81.9% | **7.7%** | 81.1% | 8.1% |
| cones | 83.8% | 6.1% | 81.7% | **5.5%** |
| Art | 71.0% | 16.1% | 70.3% | **14.5%** |
| Books | 75.4% | **8.3%** | 80.7% | 9.7% |
| Dolls | 76.9% | **7.8%** | 79.9% | 9.3% |
| Laundry | 70.7% | 20.8% | 77.6% | **17.7%** |
| Moebius | 78.0% | **11.0%** | 78.1% | 11.5% |
| Reindeer | 75.8% | **10.5%** | 74.9% | 11.1% |
| **mean** | **76.7%** | **11.0%** | **78.0%** | **10.9%** |

11.0% against 10.9% at 76.7% coverage against 78.0%. That is a tie, and since SGM
carries slightly more coverage it is arguably a shade ahead at matched coverage.

**So: level, not better.** Which is still the day's result — dense MASDA started at
28.1% bad against SGM's 8.1%, a factor of 3.5, and is now indistinguishable on
average. The three changes that did it were window aggregation (the largest),
guided-filter support, and the margin gate.

Worth naming the near-miss. Teddy alone said "beats SGM" and I nearly reported it.
Eight scenes said tie. That is the same single-scene trap as the ordering factor,
caught this time because checking cost two minutes.

### Where it stands, and what is left

- **Quality: level with SGM**, from 3.5x behind.
- **Runtime: 226 ms against 16 ms.** Now the gap. The guided filter is four box
  passes per disparity slice on top of the scoring, and the work is
  memory-bandwidth bound, so the levers are traffic rather than arithmetic: uint16
  cost volume, fusing passes, or not materialising the volume at all.
- **Untried and still promising:** slanted planes for the genuinely-slanted
  population, and a graded cost (Census plus absolute difference) for the ~50% of
  error that sits in flat regions and is untouched by anything done today.

### Blocking the disparity dimension: right idea, smaller effect than predicted

The scatter into the volume writes `vol[(y*W+x)*D + k]`, which for fixed k walks
stride D floats — 240 bytes, so one useful float per 64-byte cache line, over a
40 MB array, sixty times. Given the work is bandwidth bound that looked like the
dominant cost.

Processing disparity in blocks of 16 and transposing once per block, so each output
cache line receives 16 consecutive k values instead of one:

| | cost stage | total |
|---|---|---|
| before | 168 ms | 226 ms |
| after | 155 ms | **212 ms** |

Output bit-identical (81.9% coverage, 7.7% bad on teddy). A 6% gain, not the large
one the cache arithmetic suggested — so the strided scatter was **not** the dominant
term, and I had over-attributed.

What actually dominates is the guided filter itself: four box passes per disparity
slice, each reading and writing a W×H float plane, sixty times. That is roughly half
a gigabyte of traffic per image pair, and it is inherent to doing the filtering at
full resolution.

The known fix is the **fast guided filter**: subsample the guidance and the input,
filter at reduced resolution, upsample the two coefficient planes and apply them at
full resolution. The coefficients are smooth, so subsampling by 4 costs almost
nothing in quality and removes ~16x of the filter traffic. Untried, and the obvious
next step on runtime.

### The fast guided filter does not work here

Implemented and measured. The premise — the guided filter's coefficient planes are
smooth, so computing them at reduced resolution is nearly free — does not hold for a
stereo cost volume.

| subsample factor | cost stage | coverage | bad-1.0 |
|---|---|---|---|
| 1 (direct) | 159 ms | 81.9% | **7.7%** |
| 2 | 153 ms | 82.0% | 8.2% |
| 4 | 104 ms | 81.8% | **10.1%** |
| 8 | 102 ms | 81.4% | 14.6% |

At factor 4 it saves 55 ms and costs **2.4 points of bad-1.0** — which undoes most of
what the guided filter bought in the first place (10.6% -> 8.8%). At factor 2 there
is no useful trade at all: 6 ms for half a point.

The reason is that the fast variant subsamples the **input** as well as the guidance,
and a stereo cost slice is not a smooth signal. Decimating it by 4 discards fifteen
of every sixteen cost samples before they are ever aggregated, and no amount of
smoothness in `a` and `b` recovers that. He and Sun's result is about filtering
*images*, where the input genuinely is smooth. That distinction did not occur to me
until the numbers came back.

Left in behind `--fgf N`, defaulting to 1. Routing F=1 through the subsample and
upsample machinery would cost a redundant copy and two bilinear passes for nothing —
207 ms against 159 ms in the cost stage — so F=1 takes a separate direct path.

**So the runtime item is still open**, and the obvious remaining levers are the ones
that do not touch the cost values: a uint16 volume instead of float32 to halve the
traffic, and PatchMatch-style propagation to avoid materialising a volume at all.

### We are not bandwidth-bound, and the uint16 plan was based on a false premise

Before implementing a uint16 cost volume to halve memory traffic, I measured whether
traffic is actually the limit. It is not.

A streaming read benchmark on this machine sustains **18.9 GB/s at four threads** and
22.1 GB/s at eight. The cost stage moves about 405 MB per image pair — score writes,
four box passes reading and writing a W×H plane each, and the scatter — in 159 ms,
which is **2.5 GB/s: 13% of what the machine will give.**

So the earlier conclusion in this document, that the work is memory-bandwidth bound
because eight threads are slower than four, is **wrong**. Eight threads being slower
has a different cause, and shrinking the cost volume would buy very little.

The likely real cause is the box filter's **serial dependency chain**:

    acc += in[x];
    if (x > 2*r) acc -= in[x - 2*r - 1];

Every iteration depends on the previous one, so this cannot vectorise and its rate is
set by add latency, not by memory or by throughput. Four box passes per slice times
sixty slices is a great many strictly serial accumulate steps. That also explains the
hyperthread regression: two threads sharing one core's execution ports contend for
the same latency-bound chain rather than overlapping usefully.

**Which redirects the work.** The fix is not a smaller data type, it is to break the
dependency: run the horizontal pass over **several rows at once**, one SIMD lane per
row, so the serial chain per lane is independent. Rows are already independent here,
so this is a loop restructuring rather than an algorithm change. A summed-area table
is the other standard option, trading the running sum for four loads per query.

Recording the correction rather than quietly fixing it: this is the fifth mechanism I
attributed confidently and wrongly today. The measurement cost two minutes and saved
implementing a uint16 volume that would have bought almost nothing.

### Attributing the cost stage properly, having guessed wrong repeatedly

Restructuring the vertical box pass — carrying a row of accumulators and stepping y,
so the serial dependency stays in y while every inner loop runs contiguously over x
with W independent lanes — took the cost stage from 159 ms to 147 ms. Output
identical. A 7% gain, which is the third optimisation in a row to come in far below
what the mechanism suggested.

So instead of theorising again, strip one layer at a time and read the numbers:

| configuration | cost stage | implies |
|---|---|---|
| `--agg 0` (score + scatter only) | **66.7 ms** | **45% of the stage** |
| `--box --agg 5` (one box pass) | 80.2 ms | a box pass is ~13.5 ms |
| guided `--agg 5` (four box + elementwise) | 147.2 ms | four boxes ~54 ms, elementwise ~27 ms |

**Scoring and the scatter are the single largest item, at 45%** — and I had spent
three changes optimising the filtering, which is 55% split across seven passes.

The scoring itself cannot be much of it: 60 disparities over 168k pixels is 10M
XOR-and-popcount pairs, a few milliseconds of arithmetic. So the 66.7 ms is dominated
by *moving the volume* — writing 40 MB out through the blocked transpose — which is
also why blocking it only bought 6%: the traffic is irreducible while the volume
exists.

### The conclusion the attribution forces

Every remaining lever inside this structure is worth single-digit percentages, because
the structure itself is the cost: a 40 MB volume gets written once and read once per
solver iteration, and no amount of filter tuning changes that.

Two ways out, and they are the same two already on the list for other reasons:

- **Fuse the solver into the cost stage per row band.** Compute the cost for a band
  of rows with an r-row halo, solve those rows immediately, discard. The volume never
  exists — only a band of it — which removes the 40 MB write and the 40 MB read
  outright. Rows are already independent, so this is a restructuring rather than an
  algorithm change.
- **PatchMatch propagation**, which never builds a volume in the first place.

That is now the third independent argument for propagation — accuracy on slanted
surfaces, candidate generation, and runtime — which is a reasonable point to stop
tuning and change the structure.

## What SGM does that makes it 12x faster

Worth understanding before restructuring, because two of the four reasons change what
the restructuring should be.

**1. It never materialises the volume.** OpenCV's SGBM streams: it keeps a few rows of
cost in a buffer and accumulates the aggregated cost in place, so its working set is
O(W·D) rather than O(W·H·D). Ours is 40 MB; SGM's is a few hundred kilobytes and stays
in cache. That is the same conclusion the attribution above reached independently, and
it is worth noting that SGM's design already embodies it.

**2. Its inner loop vectorises over disparity; ours does not.** The SGM recurrence

    L(p,d) = C(p,d) + min( L(p-r,d), L(p-r,d±1)+P1, min_k L(p-r,k)+P2 ) - min_k L(p-r,k)

is **elementwise in d** apart from one shared scalar (`min_k`). So all D values for a
pixel are updated in parallel with SIMD.

Our message updates are the opposite shape: "max over this pixel's other disparities"
is a **reduction with an argmax**, which does not vectorise. The `Top2` struct is
O(1) per element and still strictly serial.

**3. The layout follows from that, and ours is backwards.** SGM stores cost per row as
`[d][x]`. We store `[x][d]`, chosen so the rho reduction reads a contiguous run of D.
Contiguity was the wrong thing to optimise for: with `[d][x]`, a loop over d with x
inside is contiguous **and independent across x**, so the reduction vectorises across
*pixels* even though it cannot vectorise across disparities. Eight pixels' reductions
proceed in eight SIMD lanes.

That is the lesson I would not have found by profiling our own code: the fix for a
non-vectorisable reduction is not to vectorise the reduction, it is to run many
independent reductions side by side.

**4. Integer arithmetic.** int16 costs, so twice the lanes and cheaper operations —
which is where the deferred uint16 item earns its place, as a multiplier on lanes
rather than a saving on bandwidth.

### What the restructuring should therefore be

Not just "fuse the solver into a row band". The band fusion removes the 40 MB of
traffic, which is 45% of the cost stage, but it leaves the serial reductions in place.
The full change is:

1. Process a band of rows with an r-row halo; the volume never exists.
2. Store band cost as `[d][x]`, not `[x][d]`.
3. Rewrite both reductions as loops over d with x inside, so they vectorise across
   pixels. The beta reduction becomes a diagonal walk in this layout and may want its
   own copy — measure rather than assume.
4. Then int16, for the lanes.

Steps 1 and 2 are cheap and independently verifiable against the current output.
Step 3 is where the 12x mostly lives, and step 4 multiplies it.

**Not implemented yet.** This is the design that the SGM comparison produced, and it
is a larger change than the ones above; writing it half-done and unverified would be
worse than leaving it. The current state is 202 ms, quality level with SGM, with every
number in this document reproducible from `core/tools/de_dense.cpp` as committed.

### Step 3 before step 2 is a regression, which confirms the ordering matters

Tried the pixel-parallel reduction on its own, without first changing the layout:
loop d outer, x inner, carrying a top-2 per pixel in three arrays so every inner
loop is W independent lanes.

Solve went **43 ms to 57 ms** — a 12% regression on the whole program. Output
identical, so the arithmetic was right; the memory was not.

The reason is the layout it was applied to. In `[x][d]`, "x inner at fixed d" reads
`beta[x*D + k]`, which strides by D floats — 240 bytes, a fresh cache line per
element. Vectorising the arithmetic while making every load a cache miss is a net
loss, and comfortably so.

So the plan's ordering is load-bearing rather than stylistic: **`[d][x]` is a
prerequisite for the pixel-parallel reduction, not a companion optimisation.** With
that layout the same loop reads contiguously and the lanes are free; without it the
lanes cost more than they save.

Reverted. Recorded because the failure is informative: it is evidence that the
remaining 12x really is in the layout-plus-vectorisation combination, and that
neither half delivers alone. The four steps have to go in order.

### One character: divide becomes multiply

The scoring inner loop runs D·H·W times — 9.6M for a 450×375 pair at 60 disparities —
and ended with `/ 24.f`. A float divide is roughly twenty cycles against four for a
multiply, and the compiler cannot make the substitution itself: 1/24 is not exactly
representable, so `x / 24.f` to `x * (1.f/24.f)` changes the last bit and is
forbidden without `-freciprocal-math`.

| | score + scatter | total |
|---|---|---|
| before | 66.7 ms | 201.3 ms |
| after | **59.5 ms** | **197.2 ms** |

Output identical. 11% of that stage for a one-character change, which is worth having.

**And my prediction was still too high.** I expected the divide to be most of the
stage — 9.6M divides at twenty cycles is ~55 ms of the 66.7 on paper. It was 7 ms.
Presumably the divides pipeline better than the latency figure suggests, and the
remaining 59.5 ms is still unattributed. GCC will not vectorise `popcnt64` without
AVX-512, and the disparity-blocking staging moves an extra 80 MB, but I have not
measured which of those it is and my record today says not to guess.

**Where this leaves the runtime work.** Six attempts at the cost stage have produced
2-11% each: blocked transpose, vectorisable vertical pass, reciprocal multiply, and
three that failed outright. The structural changes — band fusion so the volume never
exists, `[d][x]` layout, pixel-parallel reductions, int16 — remain untried and are
where the 12x is. Incremental tuning of this structure has been thoroughly explored
and is close to exhausted.

## The restructuring, step 1: measure the premise before building on it

The plan above named band fusion as step 1 on the reasoning that the cost volume is
40 MB and a `W*H` plane is 675 KB against a 256 KB L2, so the guided filter's ~9 live
planes per thread thrash a 6 MB L3 four ways over. That reasoning predicted a large
win. It is wrong, and cheap to falsify: crop the image until a plane is L2-resident
and see whether the cost stage speeds up per unit of work.

| plane | 1 thread | 4 threads |
|---|---|---|
| 112 KB (H=64) | 21.88 ns/px·disp | 9.09 ns/px·disp |
| 225 KB (H=128) | 23.00 | 11.60 |
| 338 KB (H=192) | 23.24 | 12.46 |
| **659 KB (H=375)** | **24.99** | **13.98** |

14% single-threaded, 35% at four threads. Real, but not the 3x the argument implied:
these passes stream with no reuse, and a streaming pass is prefetchable, so cache
*capacity* barely matters. **Working set was the wrong diagnosis.** That is the
seventh mechanism guess to fail here, and it cost twenty minutes instead of a day
because it was tested before it was built on.

### The one claim that survived: the running sum is latency-bound

Attributing the box filter by phase, over 240 calls at 450x375, r=5:

| phase | ms | share |
|---|---|---|
| horizontal running sum | 95.6 | **74%** |
| vertical pass | 23.9 | 19% |
| normalisation | 7.2 | 6% |

The horizontal pass is `acc += in[x]; acc -= in[x-2r-1]` — a serial dependency chain.
Every add waits on the previous one, so it advances at float-add *latency*, four
cycles per element, however many adders are idle; and it cannot vectorise, for the
same reason. The vertical pass was already fixed (it carries a row of accumulators),
which is why it is a quarter of the cost of its own mirror image.

The fix for a non-vectorisable reduction is to run many independent ones side by side.
L rows at once gives L independent chains, needs no transpose — L sequential streams
is what the prefetcher is for — and is bit-identical, because each chain performs the
same adds on the same values in the same order.

| chains | box filter | vs shipping |
|---|---|---|
| 1 (shipping) | 122.6 ms | 1.00x |
| 2 | 80.3 | 1.55x |
| 4 | 51.6 | 2.41x |
| **8** | **50.7** | **2.45x** |
| 16 | 52.1 | 2.38x |

The curve is the one two adders at four cycles of latency predict: saturating around
eight, then spilling at sixteen. Splitting the loop at `x = span` to hoist the bounds
test out of every element is folded in. All variants verified bit-identical against
the previous implementation rather than assumed to be.

### A 2.45x arithmetic win is 6% in situ, and that is the finding

| | 1 thread | 4 threads | scaling |
|---|---|---|---|
| score + stage + transpose | 86.7 ms | 57.1 ms | 1.52x |
| guided filter, before | 163.8 | 87.8 | 1.87x |
| guided filter, after | **100.6** | **73.6** | 1.37x |
| one box pass, before | 33.0 | 20.2 | |
| one box pass, after | **17.7** | **12.9** | |

The box pass improved 1.86x at one thread and only 1.57x at four, and the whole cost
stage moved 147.7 -> 138.6 ms. **Both halves of the stage scale at ~1.4-1.5x on four
real cores**, which is the signature of a shared bottleneck, not of arithmetic. Fixing
the arithmetic on a stage that is waiting for memory buys the fraction of it that was
not waiting.

So the microbenchmark did not lie, it answered a different question: single-threaded
on one hot plane, the chain is the limit; four-threaded on cold planes, it is not.
A number measured in one context is not a measurement in another, at the scale of two
threads rather than two machines.

### Two passes that were pure traffic

Folding the normalisation into the vertical pass's stores removes a read-modify-write
of 675 KB per box filter for one flop an element. Folding the guided filter's final
`ma*I + mb` combine directly into the block buffer removes another write, read and
write per slice — 81 MB over the volume — for values that were already in registers.
Only the valid window is read downstream, so it is the same arithmetic on the same
elements.

| | cost stage | total, teddy |
|---|---|---|
| shipping | 147.7 ms | 216.0 ms |
| + 8 chains, folded norm | 138.6 | 194.2 |
| + fused combine and stage | **124.8** | **184.2** |

**15% on both, bit-identical output**, verified with `cmp` against the previous
build's raw disparity and re-scored over all eight scenes: 76.7% coverage, 11.0%
bad-1.0, unchanged to the decimal.

`article/dense_bench.py` now does that scoring in one command. It did not exist — the
eight-scene dense figures had been produced by hand once, so they could be quoted but
not re-checked, and a speed change that quietly altered the output would not have been
caught. It also keeps the raw disparity per scene, which is what makes `cmp` possible.

### Where band fusion actually stands

Downgraded from "step 1, where the 12x lives" to "worth trying, with a known
tension", on two measurements.

Against it: the cache-capacity argument is worth 35% at four threads, not 3x. And the
halo is not free — the guided filter chains two box passes, so a band of B rows needs
slice rows `[y0-2r, y1+2r)`, and at r=5 that is 20 extra rows per band. A band small
enough to be L2-resident (B=8, 864 KB) does 3.5x redundant filter work; a band big
enough to amortise the halo (B=64) is 6.9 MB and back to thrashing L3. The redundancy
lands on the guided filter, which is the expensive half.

For it, and this is the stronger argument and is *not* about cache capacity: the score
loop reads both 1.35 MB census planes once per disparity slice, so **162 MB of census
re-reads** over 60 slices, plus ~120 MB of staging and transpose traffic. Banding
makes a band's census rows resident across all D disparities and turns that 162 MB
into ~2.7 MB. That is a traffic argument, and traffic is what the 1.4x scaling points
at.

The way to resolve the tension is to give each thread a contiguous stripe of rows and
stream bands down it carrying the box filters' vertical accumulators across band
boundaries, so the halo is paid once per stripe (4 stripes x 20 rows = 21% of H)
rather than once per band. That is a two-level streaming pipeline through ring
buffers and is genuinely intricate — the reason to do it is the 162 MB, and the thing
to measure first is how much of the 1.4x scaling that traffic actually explains.

### The cost stage is bound by pass count, and that is measurable

Two cheap tests, both diagnostic, neither needing a profiler (`perf` is locked down
here at `perf_event_paranoid=4`).

**A 15x larger window is free.** The box filter is O(N) regardless of radius, so if
time were going into the aggregation *work* it would still be flat, but if it were
going into arithmetic per element it would not be:

| agg radius | 1 | 3 | 5 | 9 | 15 |
|---|---|---|---|---|---|
| cost stage | 127.2 ms | 127.7 | 125.7 | 125.7 | **124.8** |

**Threads saturate at two.** Guided cost stage: 187.3 ms at one thread, 142.3 at two
(1.32x), 130.7 at four (**1.09x**). A shared resource is exhausted before the third
core arrives.

Counting the traffic explains both. The guided filter touches **29 whole planes per
disparity slice** — 675 KB each, so 19.6 MB per slice and 1.17 GB over the volume:

| step | plane touches |
|---|---|
| `ips = Is * slice` | 3 |
| `box(slice)`, `box(ips)` | 4 + 4 |
| coefficients `a`, `b` from four inputs | 6 |
| `box(a)`, `box(b)` | 4 + 4 |
| combine into the block buffer | 4 |

Plus ~204 MB for scoring (both 1.35 MB census planes re-read per slice) and 80 MB for
the staging-and-transpose. About 1.45 GB in 125 ms — 11.6 GB/s against the 18.9 GB/s
this machine sustains, and the plateau says the effective ceiling is lower than that
under four threads.

**The consequence is a change of target.** The aggregation's *radius* is free and its
*pass count* is everything, so the lever is a filter with fewer passes, not a cheaper
inner loop. A box is 4 touches and costs 12.9 ms; the guided filter is 29 and costs
73.6 ms — the relationship is linear in passes, which is the prediction this makes.

**And it supersedes the int16 deferral.** That was parked because the cost stage
achieved 2.5 GB/s of 18.9, so traffic looked irrelevant. Measured properly it is at
11.6 GB/s and saturating four cores, so halving every one of those 29 touches is
worth what the original argument said it was not — quite apart from doubling the SIMD
lanes.

## Recursive edge-aware aggregation: 1.34x for a tenth of a point

Acting on the pass-count finding. The guided filter walks 29 whole planes per
disparity slice and needs eight temporaries; a domain-transform recursive filter
(Gastal and Oliveira 2011, doi:10.1145/2010324.1964964) walks 12 and needs none.
Two 1-D passes per axis, each a normalised IIR:

    F[x] = F[x] + a[x] * (F[x-1] - F[x])

`a` shrinks towards 0 across an intensity edge, which is what makes it edge-aware,
and it depends only on the guide image -- so like the guided filter's `mean_I` and
`var_I` it is computed once for the whole volume, which is what makes an edge-aware
filter affordable at all. Two shared read-only coefficient planes, 1.35 MB, against
eight per-thread temporaries at ~6 MB.

Its horizontal passes are a serial dependency in x, the same problem the box filter's
running sum had, and take the same fix: eight rows at once for eight independent
chains. The vertical passes carry a whole row and vectorise over x unaided.

**Teddy**, where the cost stage is the point:

| | coverage | bad-1.0 | cost stage | total |
|---|---|---|---|---|
| guided | 81.9% | 7.7% | 131 ms | 181 ms |
| recursive | 81.5% | **7.9%** | **73 ms** | **126 ms** |

**All eight scenes**, measured back to back in one session:

| | coverage | bad-1.0 | mean runtime |
|---|---|---|---|
| guided | 76.7% | 11.0% | 239 ms |
| recursive | 76.4% | **11.1%** | **178 ms** |

1.8x on the cost stage, 1.34x overall, for 0.1 point of bad-1.0 and 0.3 of coverage.
Both remain available (`--guided`); the recursive filter is now the default, and the
reason to expect it to win by *more* on the Jetson is that the gap it closes is
traffic and working set, which is where the TX2 is poorer than this laptop.

**`sigma_r` was swept and 0.2 is a genuine peak** -- 7.9% at 0.2, rising to 8.2% at
0.4, 8.9% at 1.0 and 9.2% at 2.0. Reading the first three points alone (0.05, 0.1,
0.2) suggested it improved monotonically, which would have shipped the wrong default;
the peak only appears once the range is extended past it. `sigma_s` barely matters,
and the algebra says why: the exponent is `-sqrt(2)*(1/sigma_s + g/sigma_r)`, so for
any large `sigma_s` the first term vanishes and the coefficient is set by the gradient
term alone.

**This is a trade, not a free win**, and the 0.1 point is smaller than the
scene-to-scene spread, so it should not be quoted as "no loss" -- it is "no loss that
this benchmark can resolve". Runtime on this laptop drifts ±12% with thermals across
runs of identical work, which is why the comparison above is back to back and why
single runs are not worth reporting.

## How many candidates per pixel are needed? Measured before pruning anything

Mario's suggestion: run MASDA on a sparse candidate set rather than the full volume,
since most (pixel, disparity) pairs are nobody's plausible answer. That would return
the algorithm to the sparse edge-list form it was written for, delete the 40 MB
volume, and remove the stride-(D+1) diagonal walk in the beta update -- which is the
one genuinely awkward thing the dense layout imposes.

Pruning has a hard ceiling: if the truth is not among the top k by aggregated cost,
no solver recovers it. `article/topk_recall.py` measures it. Recall is the fraction of
known-ground-truth pixels whose true disparity is within 1 px of one of the top k
candidates, and every scene's truth is inside the search range, so nothing is being
hidden in an unreachable remainder.

| k | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| mean recall | 75.1% | 78.0% | 80.6% | **82.8%** | 85.0% | 90.0% |

**Top-8 pruning does not bind.** Its ceiling is 82.8% of known pixels, and the
shipping pipeline currently delivers 67.9% of known pixels correct (76.4% coverage at
11.1% bad-1.0) -- about fifteen points of headroom. Even k=2 has eight points. So the
7.5x reduction in edges is available at no measurable accuracy cost, and the thing to
watch when it is built is not the answer set but the uniqueness competition: pruning
removes claimants from the beta update, which changes the dynamics rather than only
the candidate list.

Two notes on how this was measured, both of which would have produced a wrong answer
if skipped. Pruning must be by **rank within a pixel**, never a global magnitude
threshold: Census costs are absolute Hamming fractions, so a textureless region scores
uniformly mediocre and its correct match scores mediocre too, and a magnitude cutoff
would delete precisely the flat regions where half the remaining error already sits.
And recall is conditioned on the truth being inside the search range, so that pruning
is not charged for pixels no k could reach.

### The solve is earning its 46 ms, and checking that took a fair comparison

The recall table says aggregation's own argmax is correct on 75.1% of known pixels,
against the full pipeline's 67.9%. Read as it stands that says the message passing is
*harmful*, which would be a startling result -- and it is the precision-versus-count
trap, two rates over different denominators, which has already caught me once here.

MASDA covers 76.4% and argmax covers 97%, so the gate is choosing the easier pixels.
At **matched coverage** -- argmax gated by its own top1-minus-top2 margin down to
MASDA's coverage, the same quantity MASDA gates on -- the comparison inverts:

| scene | MASDA | argmax at matched coverage |
|---|---|---|
| teddy | **7.9%** | 11.4% |
| cones | **6.3%** | 8.2% |
| Art | **16.3%** | 17.7% |
| Laundry | **20.6%** | 24.7% |

Uniqueness plus message passing is worth 1.4 to 4.1 points of bad-1.0 over picking the
best-scoring disparity per pixel. That is the first direct measurement of what the
solver contributes on dense data as opposed to what the whole pipeline scores, and it
is the number to hold pruning against.

Worth keeping separately: argmax at full coverage yields *more correct pixels* over
known ground truth (80.3% against 75.0% on teddy) because MASDA's gate discards
correct pixels along with wrong ones. That is the trade the gate exists to make, not a
defect, but anything downstream that wants pixels rather than precision should know
the choice is there.

## Sparse candidates: faster AND more accurate, at k=2 rather than k=8

Built as designed -- top-k by rank within each pixel, claimant lists rebuilt per row
by counting sort into `xr = x - d` buckets, which replaces the stride-(D+1) diagonal
walk in the beta update. Rows stay independent, so the row-parallel pool is unchanged.

Sweeping k over the eight scenes:

| k | coverage | bad-1.0 | mean runtime |
|---|---|---|---|
| dense, D=60 | 76.4% | 11.1% | 164 ms |
| **2** | **75.6%** | **10.3%** | **120 ms** |
| 3 | 75.9% | 11.1% | 127 |
| 4 | 75.9% | 11.3% | 127 |
| 6 | 76.1% | 11.6% | 131 |
| 8 | 76.1% | 11.6% | 148 |
| 12 | 76.2% | 11.5% | 153 |
| 16 | 76.3% | 11.4% | 155 |

**k=2 is better than the dense solver on both axes** -- 0.8 points of bad-1.0 and 1.4x
the speed -- and the curve is not monotonic: it degrades to 11.6% by k=6-8 and then
climbs back towards the dense 11.1% as k approaches D. Coverage falls 0.8 points, so
correct-pixels-over-known is 67.8% against the dense 67.9%: the same yield at higher
precision, which is the trade the margin gate exists to make.

**k=1 is not comparable and should not be read from the same table.** With no
runner-up the margin degenerates to `best - lambda`, which is always large, so the
gate stops rejecting anything -- hence its apparent 80.5% coverage. The gate is
identical for every k >= 2, because the margin only ever looks at the top two, so
differences across k >= 2 are purely the message passing seeing fewer candidates.

### The measurement that cleared this measured the wrong quantity

`topk_recall.py` was written to bound pruning: is the truth *available* in the top k?
It said k=8 keeps 82.8% against 67.9% delivered, so k=8 was safe. That reasoning was
sound and the conclusion was right, but it was not the operative effect. **Availability
was never the constraint -- competition was.** k=8 keeps strictly more of the truth
available than k=2 and scores 1.3 points worse.

The mechanism is the one already on record from the sparse matcher: Census plus
uniqueness cannot pick the correct match out of 71 candidates, and it was measured
there as a semi-dense variant getting *worse* (0.587) precisely because every right
pixel was offered. Restricting each left pixel to its best and its runner-up makes the
uniqueness constraint arbitrate between two plausible options instead of ranking a
crowd. Pruning is doing quality work, not just saving time.

Recording this because the shape recurs: a bound that is correct, cheap and
reassuring can still be about a different mechanism than the one that decides the
outcome. The recall sweep should have been a bad-1.0 sweep over k from the start --
which costs the same and answers the question directly.

### Where dense MASDA now stands against SGM

| | coverage | bad-1.0 | runtime |
|---|---|---|---|
| SGM | 78.0% | 10.9% | 16 ms |
| dense MASDA | 75.6% | **10.3%** | 120 ms |

Per scene, teddy is 7.6% against SGM's 8.1% and cones 5.5% against 5.5%. **On
accuracy this is now ahead rather than level**, at 2.4 points less coverage and 7.5x
the runtime. Over the session: 246 -> 120 ms mean and 11.0 -> 10.3% bad-1.0, so the
whole of it was speed and accuracy together rather than a trade.

## Blockwise: the volume stops existing

The 40 MB cost volume only ever existed to be reduced to two candidates per pixel, so
with k=2 it can be skipped: each thread keeps a running top-2 over the disparities it
owns, and the four lists are merged once at the end.

**Why this is a traffic win, and not the cache-capacity win it looks like.** Staging
and transposing moved ~120 MB and the solver read 40 MB back. Blockwise reads each
filtered slice once and compares against a runner-up plane that is the *same* 675 KB on
every one of the D slices, so it stays resident: the common path is one streaming read
plus a cached compare. The earlier crop experiment already showed cache capacity alone
is worth only 35% here, and that has not changed -- what changed is how much memory
gets moved.

The top-2 state is a **structure of arrays**, not an array of structs, for the same
reason: the reject test needs one float, and a packed 12-byte record would pull three
times what the question requires. Rejection is the common case.

| stage, teddy | before | after |
|---|---|---|
| cost | 75 ms | **54 ms** |
| solve | 12 ms | **4.1 ms** |
| total | 94-105 ms | **66 ms** |

**Eight scenes: 120 -> 83 ms mean, quality unchanged** at 75.6% coverage / 10.3%
bad-1.0. The solve is now 4.1 ms -- from 46 ms at the start of the day, 11x, and it has
stopped being a stage worth optimising.

**Verified two ways, both of which could have failed.** The output is bit-identical to
the volume path (`--dump-vol` forces it, so the two run side by side), and it is
identical across 1, 2, 4 and 8 threads -- which is a genuine race check rather than a
reassuring one, because thread count changes which disparities each worker owns and
therefore the order the merge sees. A racy merge or a tie broken by arrival order shows
up as a difference.

The volume path is retained for `--dump-vol` and for the k sweep, both of which need
all D per pixel.

### Where this leaves it

| | coverage | bad-1.0 | runtime |
|---|---|---|---|
| SGM | 78.0% | 10.9% | 16 ms |
| dense MASDA | 75.6% | **10.3%** | **83 ms** |

Over the session: **246 -> 83 ms, 3.0x, and bad-1.0 11.0 -> 10.3%** -- speed and
accuracy together rather than traded. The gap to SGM is 5.2x, down from 15x, and MASDA
is ahead on accuracy at 2.4 points less coverage.

The stage split is now census ~8-13 ms, cost ~54 ms, solve ~4 ms. **Census has become
12-16% of the total** without anyone looking at it, and it is a fixed cost paid twice
per pair independent of D -- the first time it has been large enough to matter.

## Attributing the cost stage after blockwise, and three negatives

With the volume gone the cost stage is 43-54 ms and the split is not where the plan
assumed. Ablating the filter (`--guided --box --agg 0` leaves score and insert only):

| | ms, 4 threads | share |
|---|---|---|
| score + top-2 insert | 38.1 | **88%** |
| recursive filter | 5.0 | 12% |

**The filter is 12% of the stage.** The int16 plan was aimed at it -- half the traffic,
twice the SIMD lanes -- and would have been optimising 5 ms. Isolating the two halves
in a microbenchmark, single-threaded over 60 slices:

| | ms |
|---|---|
| score only | 35.0 |
| score + top-2 insert | 59.3 |

So the insert is **24.3 ms**, 41% of that pair, and not the nearly-free compare the
design argued it would be.

### Negative: blocking the score loop to reuse census rows

The score loop reads both 1.35 MB census planes per disparity slice: 162 MB over 60
slices, more than half the stage's traffic, and the obvious fix is rows outer with a
block of B disparities inner so a row's census words serve B disparities.

| | one d at a time | B=2 | B=4 | B=8 | B=16 |
|---|---|---|---|---|---|
| score loop | **28.8 ms** | 40.2 | 34.1 | 37.7 | 34.5 |
| census read | 162 MB | 81 | 40 | 20 | 10 |

**Uniformly worse**, by 16-28%, while reading a sixteenth of the census. Cutting 150 MB
of reads bought nothing, so the re-read was never the cost -- one reused slice plane
stays in cache for its writes, and B live planes do not. Another traffic argument that
did not survive contact.

### Negative: fusing the insert into the filter's last pass

The insert was a separate pass re-reading the plane the filter had just written, and
the filter's final vertical pass already holds each row's value in a register. Fusing
removes a full 675 KB read per slice. Verified bit-identical -- top-2 depends only on
the order of k, which is the outer loop, so running the insert bottom-to-top changes
nothing.

Interleaved best-of-6, to keep the ±12% thermal drift on this laptop out of it:

| | cost stage |
|---|---|
| separate insert | **48.8 ms** |
| fused insert | 51.3 ms |

**5% worse.** Reverted. The likely reason is stream count -- the vertical pass streamed
three planes and the fused version streams seven, against a limited number of
prefetcher slots shared by four threads -- but that is unverified, and the record here
says not to bank on it.

### What this means for the remaining runtime

Three traffic-reduction arguments in a row have now failed on this stage: cache
capacity (35%, not 3x), census re-reads (negative), pass fusion (negative). Meanwhile
the stage still scales at only ~1.55x on four cores, so something shared is saturated
that none of these three was.

The score loop is 10.1M pixel-disparities each needing two loads and a popcount, and
GCC will not vectorise `popcnt64` without AVX-512. That is close to a floor for this
structure. **The remaining structural lever is D itself**, which every stage is linear
in: a half-resolution coarse pass at D/2 over a quarter of the pixels, then refinement
in a narrow band, is 4-5x less total work. It is the one large lever untouched, and it
is a quality risk rather than a pure speed change, so it needs the bad-1.0 sweep rather
than a bit-identity check.

The earlier coarse-to-fine negative does not transfer: that failed on the *sparse*
matcher because k was already 2.7, so there were no false candidates for a prior to
remove. Here D=60 is real and the saving is arithmetic.

## Coarse-to-fine: cleared, but the first version of the experiment said the opposite

Every stage is linear in D, so a half-resolution coarse pass (D/2 over a quarter of the
pixels, so D/8 of the work) plus a narrow refinement band at full resolution costs
`D/8 + (2B+1)` instead of D. At the mean D=75 across these scenes that is 5.2x less
arithmetic at B=2.

The ceiling is hard: if the truth is outside the band around the upsampled coarse
estimate, no refinement recovers it. `article/coarse_ceiling.py` measures it, as a
fraction of *correct pixels over all known ground truth* -- the quantity the decision
actually turns on, rather than the availability the top-k sweep measured.

**Run 1 said no, decisively.** Ceiling 68.4% at B=2 against the 67.9% the pipeline
already delivers: a perfect refinement would exactly match today's output and any
imperfection would lose. The dominant term was that **24.5% of known pixels had no
coarse estimate at all**.

**That was my experiment's fault, not the method's.** I had run the coarse pass with the
shipping `--min-margin 0.01`, which is a gate whose whole purpose is to trade coverage
for precision *in a final answer*. In a prior, every gated pixel is a pixel with no
search band. And a hole in a prior does not mean "no band" -- coarse-to-fine inpaints
it, which is standard and which I had not done.

Ungated, holes filled from valid neighbours:

| B | 1 | **2** | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| ceiling | 78.9% | **81.5%** | 83.0% | 84.4% | 86.4% | 88.3% |
| work ratio | 6.1x | **5.2x** | 4.6x | 4.1x | 3.4x | 2.8x |
| no coarse | | **0.2%** | | | | |

**81.5% against 67.9% delivered: 13.6 points of headroom at 5.2x less work.** Cleared,
and it is now the largest available lever by a wide margin.

**The lesson is the sharper half of this.** A cheap experiment produced a clean,
decisive, plausible negative that was an artefact of one flag, and stopping there would
have closed off the biggest remaining lever with a number that looked authoritative.
The top-k sweep had the mirror problem -- it measured the wrong quantity but erred
conservatively, so the conclusion survived. This one erred the other way. **The check
that caught it was asking why the dominant term was what it was**, rather than reading
the headline: 24.5% missing coarse estimates is not a property of coarse-to-fine, and
noticing that took one question.

### What it should cost, and what becomes the bottleneck

The work ratio applies to the cost stage, which is what is linear in D. Census is not:
it is paid once per image pair regardless, plus a quarter as much again for the
half-resolution pair. Projecting from the current teddy split (census ~8-13 ms, cost
~43-54, solve ~4):

    cost 54/5.2 ~ 10 ms, census ~8 + ~2, solve ~4-8  =>  ~25-30 ms

Against SGM's 16 ms. Meaningfully closer rather than parity -- **and census becomes the
largest single item**, which is where the next attribution should point. That is a
projection from a work ratio, not a measurement, and this file's record on projections
is why it is written as one.

## Census: 3.5x, and the microbenchmark inverted the answer once

Census is 8.1 ms of a 61 ms pair and, unlike the cost stage, is **not linear in D** --
so once coarse-to-fine cuts the cost stage it becomes the largest single item. Three
suspects, so measured rather than picked: `img.at(x,y)` recomputing `y*width` on all 48
neighbour reads, a 48-long serial OR chain per pixel, and nothing vectorising because
the only wide axis is x and x is the outer loop.

Isolated, both images, single-threaded:

| variant | ms | vs shipping | output |
|---|---|---|---|
| shipping | 10.81 | 1.00x | -- |
| hoist the row pointers | 6.68 | 1.62x | identical |
| offset outer, uint64 accumulator | 39.99 | **0.27x** | identical |
| **offset outer, six uint8 accumulators** | **1.83** | **5.91x** | identical |

**Reordering the loops on its own is 3.7x worse.** A uint64 accumulator fits four lanes
in a 256-bit register, and 48 passes over a row is a lot of loop overhead to buy four
lanes. It only pays once the accumulator is a byte: six 8-bit planes give 32 lanes for
the compare-and-or, packed into the uint64 once per row. Bit b lands at
`(b>>3)*8 + (b&7) = b`, so the descriptors are unchanged.

Loop order and lane width look like two independent tweaks and are not -- the two
reordered forms differ by 22x. Half of this change would have been a regression.

### The microbenchmark's constants were doing the work

Dropped into `de_dense` with `hw`/`hh` as runtime arguments, the new census measured
**17.6 ms against the original's 8.1** -- twice as slow, the exact opposite of the
5.9x. In the microbenchmark `W`, `HW` and `HH` were file-scope constants, so GCC
unrolled all 48 passes with a constant shift mask and a constant group pointer per
pass, and widened the compare. As runtime arguments none of that happens.

Specialising the window as a template parameter and dispatching on `hw == 3 && hh == 3`
recovers it. Interleaved best-of-6 against the original:

| | census | total, teddy |
|---|---|---|
| original | 8.1 ms | 61.0 ms |
| templated | **2.3 ms** | **56.5 ms** |

Output bit-identical, 90 tests passing, eight scenes unchanged at 75.6% / 10.3% and
**77 ms mean**.

**Rule 2 has now bitten at three different scales in one day**: desktop against Jetson,
one thread against four, and compile-time constants against runtime arguments. The
third is the least obvious and the microbenchmark is exactly where it hides, because a
bench is where constants are most convenient to write. **A microbenchmark measures the
code the compiler could generate for it, not the code in the program** -- so the loop
that will ship has to be the loop that was timed, parameters included.

## Coarse-to-fine: the ceiling was real and this construction cannot reach it

Built as planned. The cost volume is indexed by **offset from the prior** rather than
absolute disparity, so the fine pass needs 2B+1 whole planes instead of D and the
recursive filter still sees one plane per index. Coarse pass ungated with holes filled,
exactly as the ceiling experiment validated.

| | coverage | bad-1.0 | correct/known | runtime |
|---|---|---|---|---|
| single level | 75.6% | **10.3%** | **67.9%** | 77 ms |
| coarse-to-fine, B=2 | 75.6% | 14.6% | 64.7% | 50 ms |
| B=4 | 75.4% | 14.4% | 64.7% | 52 ms |
| B=8 | 75.3% | 14.2% | 64.7% | 60 ms |
| B=14 | 75.2% | 14.1% | 64.7% | 70 ms |

**Correct-over-known is 64.7% at every band from +-2 to +-14**, against a measured
ceiling of 81.5% to 88.3% over that range. A quantity that does not move when the thing
it should depend on is varied sevenfold is not a tuning problem.

Two hypotheses tested and discarded. A nearest x2 upsample makes the prior blocky with
a 1-2 px step at every 2x2 boundary, which would misalign the aggregation everywhere --
but bilinear upsampling made it *slightly worse* (63.9%). And widening the band, which
would fix any pure range limitation, does nothing.

### The controlled test: it is the parametrisation

At B=30 the band covers essentially the whole disparity range, so the search space
matches single-level and **the only remaining difference is how the planes are
indexed**:

| | bad-1.0 | correct/known |
|---|---|---|
| single level, absolute-disparity planes | **10.3%** | **67.9%** |
| B=30, offset-from-prior planes | 14.1% | 63.8% |

**Aggregation requires constant-disparity planes.** An offset-indexed plane holds the
cost at `d(x) = prior(x) + j`, so the filter averages neighbours whose absolute
disparities differ by however much the prior varies locally -- everywhere the surface
slopes, and wildly at every depth discontinuity. Aggregation is what takes this matcher
from 26.6% to 10.3%, so degrading it costs about 4 points, and no band width recovers
that because the band was never the constraint.

**Third time today that the measured bound was not the operative constraint.** Top-k
measured availability when competition decided it. The first coarse ceiling measured a
gated prior. And now the ceiling is right -- 81.5% of truths really are inside the band
-- but reaching it requires an aggregation this parametrisation cannot provide. The
ceiling bounds what a *perfect* refinement could do and says nothing about whether a
refinement compatible with the rest of the pipeline exists.

### What would actually work

Keep **absolute-disparity planes** and use the prior as a *mask* rather than a
reparametrisation: for each absolute d, evaluate only where `|d - prior| <= B`. The
filter then aggregates at constant disparity, which is correct. The saving is smaller,
because each of the D planes still needs a filter pass -- but the prior is smooth, so
the pixels needing a given d form a compact region, and rows where nothing needs d can
be skipped outright. Bounding-box filtering per plane is the form to measure.

`--prior` is kept: it is the vehicle for this measurement, and a temporal prior from the
previous frame is planned (see 3.1 in TODO.md). Anything using it must not use offset
indexing, for the reason above.

## What the hardware counters said: none of it was memory

With `perf_event_paranoid` lowered, the cost stage was measured rather than reasoned
about. Every hypothesis on the table was wrong, including all three that had already
cost a day of failed fixes.

`--guided --box --agg 0` (score and insert only), teddy:

| threads | task-clock | cycles | instructions | GHz | IPC | wall |
|---|---|---|---|---|---|---|
| 1 | 99.4 ms | 331.5 M | 573.3 M | 3.34 | 1.73 | 101.0 ms |
| 2 | 117.3 | 392.2 | 603.8 | 3.34 | 1.54 | 68.9 |
| 4 | 159.9 | 537.5 | 661.0 | 3.36 | 1.23 | 56.5 |

| | 1 thread | 4 threads |
|---|---|---|
| stalls on L3 miss | 2.3% of cycles | 5.6% |
| DRAM traffic (LLC misses x 64 B) | 30.8 MB | 82.5 MB |
| achieved bandwidth | 0.34 GB/s | **1.6 GB/s** of 18.9 |
| dTLB misses | 94 K | 217 K (0.04% of instructions) |
| store-forward blocks | 67 K | 290 K |

**It was never memory-bound.** Stalls waiting on L3 misses are 2-5% of cycles, DRAM
traffic is 8% of this machine's bandwidth, TLB misses are 0.04% of instructions, and
store-forward blocking is negligible. IPC of 1.2-1.7 is a core that is executing, not
waiting. Bandwidth, L3 capacity, TLB pressure, 4K aliasing and memory-level parallelism
are all dead at once.

**The estimate that misled me was counting plane touches.** 29 planes per slice at
675 KB gave "1.45 GB in 125 ms = 11.6 GB/s", which looked like 61% of peak. Measured
DRAM traffic is 1.6 GB/s -- **off by 7x** -- because a 675 KB plane touched repeatedly
within a pass is served by L1 and L2 and never reaches DRAM. Counting bytes moved
between *arrays* is not counting bytes moved to *memory*, and the difference is the
whole cache hierarchy.

That single error explains all three failed fixes. Cache capacity, census re-reads and
pass fusion are all ways to move fewer bytes, on a stage that was not waiting for bytes.

### What it actually was: idle cores

`task-clock` over wall clock gives the answer directly: **159.9 ms of CPU in 56.5 ms of
wall is 2.83 of four threads busy**, at a constant 3.36 GHz -- so not a turbo effect
either, which was the next thing worth ruling out. The stage was not slow, it was
absent. Two causes, both structural:

1. **`D=60` with `KB=16` is exactly four blocks for four threads**, statically assigned.
   No balancing at all, and higher disparities do less work because the valid x range
   shrinks with d. Blockwise needs no transpose, so KB exists only to batch: one
   disparity per work unit, handed out through an atomic counter.
2. **Serial sections inside the timed region.** `rf_coeffs` computes two `exp()` per
   pixel -- ~340 K transcendentals -- on one thread, and the per-thread top-2 merge ran
   on one thread too.

| | total, teddy | threads busy |
|---|---|---|
| before | 55.9 ms | 2.83 |
| dynamic scheduling + parallel setup | **51.6 ms** | **3.21** |

8%, bit-identical, and eight scenes at **62 ms mean**, unchanged at 75.6% / 10.3%.
There is still most of a core idle, so this is not finished.

**The lesson is about instrumentation, not about this loop.** Three fixes were built and
measured against a bottleneck that a profiler would have ruled out in one run, and the
reason the profiler was not used first is that it needed a sysctl and the ablations felt
sufficient. They were not: ablation tells you what a stage costs, and only counters tell
you *why*. When two or three mechanism guesses in a row fail on the same code, that is
the signal to stop guessing and get the counters, not to try a fourth.

### 17% was waste, not a bottleneck

Chasing the idle core further, with no mechanism reasoning involved -- just reading
what the code allocates. Every cost-stage thread took **all eighteen scratch planes**,
one set per filter path, 12 MB each and 46 MB across four threads, and `std::vector`
zero-fills them. Under the recursive filter exactly one of those planes is ever
touched. The guided filter's `mean_I` and `var_I` were computed too -- a downsample,
two box filters and a variance pass over the whole image, serially -- for a result
nothing read.

| | total, teddy |
|---|---|
| before | 50.8 ms |
| allocate only the active path | **42.4 ms** |

**17%, bit-identical, and eight scenes at 54 ms mean.** No hypothesis was needed and
none of the day's mechanism reasoning would ever have found it: this was not a
bottleneck to be relieved, it was work that did not need doing, and profiling attributes
it to whichever stage happens to contain the `memset`.

Worth stating plainly because it is the cheapest lesson here: **before asking why a
stage is slow, check what it is doing that it does not need to do at all.** Three
alternative filter paths accreted over a day of experiments, each correctly leaving the
others' buffers in place, and nothing about the resulting allocation looked wrong in
review.

## On the Jetson at last: the same code, a different profile

Every dense number until now was desktop-only. Measured on the TX2 (MAXN, clocks
locked at 2.035 GHz, six cores), teddy 450x375 at D=60:

| stage | desktop, 4 threads | TX2, 6 threads | ratio |
|---|---|---|---|
| census | 2.3 ms | 8.3 | 3.6x |
| cost | 41.9 | 79.4 | 1.9x |
| **solve** | **4.1** | **24.0** | **5.9x** |
| total | 48.3 | 111.7 | 2.3x |

**The solve was 5.9x worse and scaled 1.46x on six cores**, against 3.8x on the
desktop. Same class of fault as the cost-stage allocation: each solve thread took three
W*D planes -- 108 KB each, 324 KB a thread, 2 MB across six -- which `std::vector`
zero-fills, and which the sparse path never reads. In blockwise mode the per-thread
candidate copies are unused too, because the candidates live in the shared merged
arrays.

Interleaved best-of-8 on the TX2, output bit-identical:

| | solve | total |
|---|---|---|
| before | 15.8 ms | 112.4 |
| allocate only the active path | **12.5** | **90.7** |

**19% on the Jetson for a change worth about 2% on the desktop**, which is the whole
argument for rule 2 in one measurement. A 2 MB memset is a rounding error against 18.9
GB/s and is not one against the TX2's memory system.

**TX2 run-to-run variance is 37%** (cost 73.1 to 100.6 ms) at stable clocks and 35 C, so
it is scheduling and core asymmetry -- four A57s and two Denvers -- not thermal. Single
runs on this board mean nothing; everything above is interleaved best-of-8.

**Output is bit-identical between aarch64 and x86-64**, which is worth knowing: no
change here has introduced a platform-dependent reordering, so desktop scoring remains
valid for correctness even though it is useless for timing.

## Breaking down the cost stage properly, on both machines

In-situ timers inside the real loops, thread-summed CPU. The ablation this replaces was
misleading: it measured *wall* time on an occupancy-limited stage, so removing work did
not shorten the clock proportionally, and it reported the filter as 12% of the stage.

| part | desktop, 4 threads | TX2, 6 threads | TX2 / desktop |
|---|---|---|---|
| score | 39.7 ms | 72.6 | 1.8x |
| **recursive filter** | **34.9** | **175.4** | **5.0x** |
| top-2 insert | 29.9 | 94.6 | 3.2x |
| clear slice | 2.2 | 4.0 | 1.8x |

**The filter is 33% of the stage on the desktop and about 50% on the TX2**, not 12%
anywhere. And the profile is qualitatively different between the two machines: on the
desktop the three parts are roughly even, on the Jetson the filter dominates.

The reason is arithmetic width. The filter is `cur[x] += a[x]*(nxt[x]-cur[x])`, four
passes of pure float work, and NEON is 128-bit against AVX2's 256 -- half the lanes --
on top of 2.03 GHz against 3.36. That predicts ~3.3x and 5.0x is measured, the rest
presumably scalar FP throughput in the horizontal chains.

**This revives int16 with a target and an argument it never had.** It was deferred on a
bandwidth premise that turned out false, then aimed at what an ablation wrongly called
12% of the stage. In fact it targets the *largest* item on the platform that matters, and
on NEON it buys 8 lanes against 4 -- a factor the desktop numbers understate.

### Two things that were not the problem

**Redundant computation:** the only real duplication found is `std::fill(slice, 0)`
clearing the whole plane when the score loop immediately overwrites all but the left
strip and the borders. Measured at 2.2 ms of 107, so 2%, and not worth the bug surface.

**Hoisting row pointers out of the score loop is 14% SLOWER** -- 48.4 ms against 42.4,
interleaved best-of-6 -- although the identical transformation was worth 1.62x in the
census transform two functions away. GCC already eliminates the common subexpression
here and the explicit pointers defeat something it was doing on top. Recorded because
the natural inference from the census result is exactly wrong.

Allocation was a real problem and is now fixed in three places -- cost-stage scratch,
solve scratch, and the guided filter's guidance planes -- but nothing further remains:
what is allocated now is `slice`, the per-thread top-2, the merged candidates and the
filter coefficients, all of which are read.

## int16: neutral on the desktop, 20% on the Jetson

Q14 fixed point through the score, the recursive filter and the top-2 state. The score
is `(24 - popcount)/24` in [-1,1], which maps exactly onto +-16384 with a factor of two
of headroom; the filter's intermediate is int32 and its shift rounds to nearest, because
four passes of a downward bias would walk the whole plane. The solver stays in float --
it is 4 ms, its lambda/gamma comparisons are delicate, and there is nothing to win.

| | desktop, 4 threads | TX2, 6 threads |
|---|---|---|
| score | 39.7 -> 42.1 ms | 72.6 -> **57.5** |
| filter | 34.9 -> **45.0** | 175.4 -> **103.4** |
| insert | 29.9 -> **18.8** | 94.6 -> **75.6** |
| total, teddy | 46 -> 46 ms | 107.3 -> **85.5** |

**Neutral on the desktop, 20% on the Jetson**, quality unchanged at 75.7% / 10.3%.

The split is the interesting part. The insert improves on both machines -- it is a
compare against a plane, and halving the plane halves the work. The **filter gets worse
on the desktop and much better on the TX2**: AVX2 vectorises the float form as a single
FMA, and the int16 form needs a widening multiply, a shift and a narrowing store, so at
256 bits the float wins. At NEON's 128 bits the extra lanes more than pay for the extra
instructions.

**This is the first prediction today that came out right**, and it was right for the
stated reason: halving the element width buys 8 lanes against 4 exactly where the
platform is weakest. It also means the desktop is now actively misleading as a proxy --
a change that is neutral there is worth a fifth of the runtime on the target.

## Centre-symmetric census: does exactly what it promises, in both directions

CS-CT compares each pixel against its point reflection through the centre rather than
against the centre itself. A 7x7 window has 24 symmetric pairs instead of 48 neighbours,
so the descriptor halves to 24 bits and fits `uint32`.

The saving was never expected in the transform -- that is 2.3 ms and already vectorised
-- but in the **score loop**, which reads both descriptor planes once per disparity
slice and is the second largest item on the TX2.

| | coverage | bad-1.0 | desktop | TX2 |
|---|---|---|---|---|
| classic, 48 bits | **75.7%** | **10.3%** | 39 ms | 89.9 |
| CS-CT, 24 bits | 73.2% | 11.5% | 40 ms | **81.2** |

TX2 score stage: **57.5 -> 37.2 ms** of CPU, 1.55x, precisely the predicted effect of
halving the descriptor traffic and doubling the NEON lanes per register. And no gain at
all on the desktop, the same asymmetry int16 showed.

**But it costs 1.2 points of bad-1.0 and 2.5 of coverage** -- 67.9% to 64.8% of known
pixels correct. That is the other half of the prediction: halving the Hamming range from
49 levels to 25 hurts, and this file already recorded coarse quantisation as the
likeliest cause of the residual error in flat regions. Fewer levels is the wrong
direction for the weakness we already have.

**Not worth it at the current operating point**, and the reason is specific rather than
general: dense MASDA is presently *ahead* of SGM on accuracy, 10.3% against 10.9%, and
CS-CT would trade that lead away (11.5%) for 10% of the runtime. Kept behind `--csct`,
default off, because the trade reverses if the accuracy budget ever loosens -- for a
lower-resolution or higher-frame-rate mode it is 10% for accuracy that is still
competitive.

The graded-cost idea points the other way: Census plus an absolute difference *adds*
levels. That remains the untried lever on the half of the error that sits in
near-fronto-parallel regions, and CS-CT is evidence for the mechanism behind it -- moving
the level count down measurably hurts, so moving it up is worth testing.

## Graded cost: 9.7% bad-1.0, and now ahead of SGM by more than a point

Census plus a truncated absolute difference, the lever this file has listed as untried
since the SGM comparison. Census gives 49 quantisation levels and is invariant to
monotonic intensity change; its weakness is that a flat region produces near-identical
descriptors and the score has nothing to separate candidates with. An absolute
difference is continuous and fails in the complementary way -- under gain and exposure
differences, where Census does not.

**The evidence for trying it came from the CS-CT negative**: halving the level count
from 49 to 25 measurably cost 1.2 points, so the level count matters and moving it up
was worth testing. Two dimensions swept over all eight scenes:

| weight | 0 | 0.10 | **0.15** | 0.20 | 0.25 | 0.35 | 0.5 |
|---|---|---|---|---|---|---|---|
| bad-1.0 | 10.3% | 9.9 | **9.7** | 9.9 | 10.0 | 10.4 | 10.7 |

| truncation | 4 | 6 | **8** | **10** | 20 | 40 |
|---|---|---|---|---|---|---|
| bad-1.0 | 9.9% | 9.8 | **9.7** | **9.7** | 9.9 | 10.2 |

Both are genuine interior optima, swept past on each side. Truncation matters as much as
the weight: at 40 the raw difference lets an occluded or specular pixel dominate, at 4 it
is nearly binary and throws away the continuity that was the point.

| | coverage | bad-1.0 | desktop | TX2 |
|---|---|---|---|---|
| SGM | 78.0% | 10.9% | 16 ms | -- |
| census only | 75.7% | 10.3% | 40 ms | 88.4 |
| **census + graded** | **76.0%** | **9.7%** | 43 ms | 92.6 |

**Better on coverage and accuracy at once**, for about 5% of runtime on both machines --
two image loads and a table lookup per pixel-disparity in the score loop. Unlike the
last several changes this one costs the same proportion on both targets, because it adds
arithmetic rather than shifting memory or lane width.

Against SGM this is now **9.7% against 10.9%**, a lead of 1.2 points rather than 0.6, at
2.0 points less coverage.

## Inside the census transform, and why the score loop costs 14x more

Instrumented per phase, both images, thread-summed CPU:

| phase | desktop | TX2 |
|---|---|---|
| clear the byte accumulators | 0.09 ms | 0.13 |
| **48 compare-and-OR passes** | **2.87** | **5.81** |
| pack six byte planes into uint64 | 0.42 | 1.63 |
| **census transform total** | **3.4** | **7.6** |
| **score loop (XOR + popcount + AD)** | **49.1** | **108.4** |

**The transform is not the expensive part and has not been for some time.** The score
loop costs 14x more on both machines. The two do comparable amounts of raw work, and the
difference is entirely whether it vectorises.

**Census transform: 168,750 pixels x 48 pairs x 2 images = 16.2M compare-and-OR.** The
loop runs x innermost with a compile-time shift mask into byte accumulators, so it
vectorises 32 bytes per instruction on AVX2 and 16 on NEON. Each element is still a
read-modify-write -- three loads and a store per compare -- so it is memory-op bound
rather than compare bound, and at 2.87 ms for 16.2M compares it achieves about 1.7
byte-ops per cycle against a ceiling far above that. There is headroom here, but it is
headroom on 3.4 ms.

**Score loop: 60 disparities x 168,750 pixels = 10.1M pixel-disparities**, each doing two
8-byte descriptor loads, an XOR, a **popcount**, a table lookup, two image loads for the
absolute difference, and a multiply-add. Fewer operations than the transform, 14x the
time.

**The reason is popcount.** `__builtin_popcountll` compiles to the scalar `POPCNT`
instruction, and AVX2 has no vector population count -- that arrived with AVX-512's
`VPOPCNTDQ`, which this Kaby Lake does not have. So the single hottest loop in the whole
matcher runs one 64-bit word at a time while everything around it runs 8 or 32 lanes
wide. The transform is 32x parallel and the score is 1x parallel, on similar work.

### What that implies, and it differs by target

- **NEON has `vcntq_u8`**, a native per-byte population count, so on the TX2 a vectorised
  popcount is 16 bytes per instruction plus a pairwise-add reduction. This is the single
  largest identified opportunity on the Jetson: the score is 108.4 ms of CPU and the
  instruction exists.
- **On x86 the equivalent is the PSHUFB nibble-table trick** (Muła, Kurz and Lemire),
  which is well established and typically 2-3x over scalar POPCNT for bulk counting.
- Both need intrinsics, which collides with the deliberate choice in `core/Makefile` to
  leave `ARCHFLAGS` empty so the same source builds on both machines. It would need
  either per-target compilation units or runtime dispatch, and that is a real cost to
  weigh rather than a detail.

Also worth noting against `--csct`: it halved the descriptor and cut the score stage 1.55x
on the TX2 precisely because it halved this scalar work. Vectorising popcount attacks the
same cost without paying the 1.2 points of accuracy.

### Vectorising popcount alone does not work, and the reason sharpens the diagnosis

The obvious follow-up: AArch64 has `vcntq_u8` in the mandatory NEON base, so a
vectorised Hamming distance needs no `-march` flag and leaves `ARCHFLAGS` empty. Eight
distances at a time via `vcntq_u8` folded down by three pairwise widening adds, scalar
fallback elsewhere.

**No speedup: 96.5 ms against 95.4 on the TX2, interleaved best-of-6.** Reverted. (The
implementation also produced a handful of differing pixels, which alone would justify
reverting, but the runtime result is the decisive one -- a correct version would still
have been worthless.)

**The blocking diagnosis was right and the fix was aimed one level too shallow.** Taking
popcount out of the inner loop lets *it* go wide, but the loop body still cannot
vectorise, because what remains is two **table lookups** -- `tbl[hamming]` and
`adt[|L-R|]` -- and a gather does not vectorise on NEON either. The serialising work
simply moved from popcount to the lookups.

So the score loop is not "slow because popcount is scalar". It is slow because *every*
step in it is either a scalar instruction or an indexed load, and removing one of the
two categories leaves the other in charge. Making it wide means removing **both**:
computing `tbl[h]` arithmetically instead of by lookup, and the truncated absolute
difference with a vector min and subtract instead of `adt[]`. Both are expressible in
closed form -- `tbl[h] = ((half - h) * SCORE_ONE) / half` needs a division by a constant,
which vectorises as a multiply-high -- but bit-identity then depends on reproducing
integer truncation exactly, so it must be scored rather than assumed.

That is a real piece of work with a real payoff on the TX2, where the score loop is 80 ms
of CPU. It is not the half-measure attempted here, and the half-measure is worth
recording precisely because it looked like the whole thing.

## What others did: ReS2tAC, and why our vectorisation was shaped wrong

Searched after the popcount attempt failed. The directly relevant work is **ReS2tAC**
(Ruf et al., Sensors 21(11):3938, doi:10.3390/s21113938), which optimises SGM for
embedded ARM with NEON intrinsics. It answers the two questions this file was stuck on,
and contradicts the design of the attempt above.

**They use `VCNT` and no lookup table.** So hardware popcount is the right primitive --
that part of the diagnosis holds.

**But they put DISPARITY in the vector lanes, not pixels.** They process "16 disparities
and four lines simultaneously in one iteration", broadcasting the left descriptor and
walking consecutive right descriptors. That is the difference. Our attempt vectorised
eight *pixels'* popcounts and left the loop pixel-major, so `tbl[hamming]` and
`adt[|L-R|]` stayed per-pixel gathers and took over the loop. **With disparity in the
lanes the Hamming distances for 16 disparities land in one register, and the table lookup
becomes a vector operation** -- NEON's `vqtbl4q_u8` reads a 64-entry byte table in one
instruction, which covers a 49-level Census table exactly. The lookup problem this file
called unsolvable-without-arithmetic is solvable; it just needs the other loop order.

**They use 8-bit costs and a 24-bit descriptor.** The descriptor is cut to 3 bytes by
dropping to a 5x5 neighbourhood on the CPU, specifically so 16 pixels' descriptors fit in
three vector registers. That is the same trade `--csct` measured here -- 24 bits -- and it
cost 1.2 points of bad-1.0. They pay it for the SIMD fit; whether we should is a decision
with a measured price rather than a guess.

**The honest caveat: they did not test on a TX2.** Their embedded target is a Jetson
Xavier AGX with an 8-core ARMv8.2 CPU, so the reported 46 FPS at VGA is not a TX2 number
and must not be quoted as one.

### The tension this creates with our architecture

Disparity is our OUTER loop -- one whole plane per disparity -- because the recursive
filter needs whole planes to aggregate over. Putting disparity in the vector lanes means
computing 16 disparities per pixel at once and scattering them into 16 separate planes,
which is exactly the write-amplification the blocked transpose existed to manage before
the volume was removed.

So this is the `[d][x]` versus `[x][d]` question from earlier in this file, arriving from
a third direction. It was parked when sparse candidates made the transpose unnecessary;
vectorising the score loop brings it back, because the score wants disparity-minor and the
filter wants disparity-major. That is a genuine design decision and not a micro-optimisation,
and it should be settled before any more SIMD is written.

## The layout question was a false choice: 1.59x on the score loop, and the stage got slower

The `[d][x]` versus `[x][d]` decision above conflates two resources. **The score wants
disparity in the register file; the recursive filter wants it in memory, and those are
independent.** Nothing about the loads is disparity-minor either way: the right
descriptors for eight consecutive disparities at pixel x are eight consecutive uint64 of
the right census plane, and the left descriptor is a broadcast. So the kernel computes a
group with disparity in the lanes and **transposes in registers before storing**, and the
filter, the top-2 insert and the solver see exactly the planes they saw before. No layout
change, and `--simd` in `core/tools/de_dense.cpp` is the whole of it.

This is not the write amplification the blocked transpose managed. That wrote one 4-byte
disparity per 64-byte line of a `[x][d]` volume, 16x at line granularity. An in-register
transpose deposits sixteen contiguous bytes per plane and consecutive x groups continue
the same lines: eight sequential write streams, no partial lines.

**The kernel, and it is bit-identical.** Four XORs, four `vcntq_u8` and four `vpaddq_u8`
collapse eight descriptors' byte counts into eight bytes of one register -- ~1.5
instructions per Hamming distance -- and those bytes are already the byte indices
`vqtbl4q_u8` wants. The 49-level Census table and the truncated-difference table both fit
64 entries (`ad_trunc` is 10, and the difference is clamped to it before the lookup, which
is what `adt[v] == adt[min(v,T)]` already guaranteed), so **the closed-form reformulation
this file called necessary is not necessary** and every value is exactly what the scalar
loop produces. Verified: all eight scenes byte-for-byte identical at one thread.

### The result, teddy at D=60, TX2 6 threads, interleaved best-of-8

| part | scalar | `--simd` | |
|---|---|---|---|
| **score** | **77.0 ms** | **48.4** | **1.59x** |
| filter | 93.2 | 90.0 | -- |
| insert | 66.9 | 63.5 | -- |
| clear | 1.8 | **11.1** | **6.2x worse** |
| alloc | 17.0 | 19.4 | -- |
| cost stage CPU | 255.9 | 232.4 | 1.10x |
| **cost stage WALL** | **67.6** | **73.3** | **0.92x** |
| cores busy | 3.79 of 6 | 3.17 of 6 | |

**9% less CPU, 8% more wall clock.** The kernel does what it was predicted to do and the
stage got slower, because 0.62 of a core went missing.

### Two mechanism guesses failed before the counters were read, again

**Guess 1, the L2 working set.** Eight live planes are 2.7 MB a thread against 2 MB of
shared A57 L2, so a plane can no longer stay resident from score through filter to insert.
Wrong where it was aimed: filter and insert CPU did not move (93.2 -> 90.0, 66.9 -> 63.5).

**Guess 2, load imbalance.** Sixty disparities in groups of eight is eight groups over six
threads, two rounds with the second nearly empty. Tested directly at D=48 -- six groups,
six threads, one clean round -- and the stage was *still* 0.92x. Not sufficient.

**What the counters said, once a timer was put where none had been.** Every in-situ timer
starts after the scratch exists, so allocation was invisible in the breakdown and showed
up only as lost occupancy. `alloc` was **35.5 ms against the scalar path's 16.7** --
`std::vector` zero-filling 2.7 MB a thread that the per-group `std::fill` then zeroes
again. The fourth time vector's zero-fill has been the thing on the critical path here,
and the first time anything was measuring it. Worth noting the scalar path pays 17 ms of
allocation too, which nobody knew.

Replacing it with a non-zeroing buffer moved the cost rather than removing it: `alloc`
fell to 18.8 and `clear` rose from 1.8 to 14.0. The bytes are the same 60-odd full planes
either way; **what changed is cache residency.** One 337 KB plane refilled sixty times
stays in L2; eight planes at 2.7 MB cannot, so the same byte count costs 7.8x. Clearing
only the border the kernel does not write recovered very little (14.0 -> 11.1), and the
reason sharpens it further: the left strip spans rows 3..H-3, and touching every row
touches every page, so **the cost is page and line touches, not bytes written.** Guess 1
was right about the mechanism and wrong about where it would land -- it is in the clear,
not the filter.

### Where the sign flips, and the ceiling that makes it not matter much

At **D=96, twelve groups over six threads**, on a quiet board (2% spread):

| | scalar | `--simd` | |
|---|---|---|---|
| score | 122.0 ms | 70.6 | **1.73x** |
| cost stage wall | 91.4 | 88.0 | **1.04x** |
| total | 112.3 | 109.6 | 1.02x |

So the D=60 regression is the schedule, confirmed by flipping it. But note what a win looks
like: **1.73x on the score loop is 1.04x on the stage.** The score is 77 of 256 ms of
cost-stage CPU, 30%, so 1.6x on it is an 11% ceiling for the whole stage -- and the layout
costs ~12 ms of clear and alloc against the 29 ms it saves. The score loop was correctly
identified as the largest single item and is not large enough for this to matter at the
level that was hoped. At agg 5 the **filter is larger than the score** (93 against 77).

**Kept behind `--simd`, off by default, because it is 1.6x on a real loop with bit-identity
and the blocker is a schedule rather than the kernel.** What would make it pay is
decoupling the work quantum from the group size -- score into shared group buffers, then
hand out single planes for filter and insert from a second queue -- which is a real piece
of engineering against an 11% ceiling. Reverting instead is a one-line decision and the
rule about knobs argues for it.

### `cmp` between two multi-threaded runs is not an identity check

Found while trying to explain nine differing bytes: **two identical scalar runs at six
threads differ from each other.** The top-2 insert requires strictly-greater to displace,
so exactly-equal scores keep whichever disparity arrived first, and blocks are handed out
dynamically, so tie order varies run to run. Scalar 1t vs scalar 6t differs too.

At one thread everything is deterministic and all eight scenes are byte-identical. So the
protocol is: **`cmp` at `--threads 1`, and only there.** Earlier changes verified
"bit-identical" by cmp were not wrong -- int16 and the allocation fixes did not alter
partitioning, so they did not trip it -- but this change does, and a handful of differing
bytes is what a correct change looks like at six threads. Accuracy is unaffected and that
is measured rather than argued: pooled over eight scenes at six threads, both paths give
**75.99% coverage and 9.54% bad-1.0**, equal to three decimals.

That handful of bytes is also the signature the earlier popcount attempt was reverted for
in part ("also produced a handful of differing pixels"). It may have been arithmetically
correct all along. The runtime result there was decisive on its own, so the conclusion
stands, but the identity evidence against it was not evidence.

## What others did about the D-planes problem, and why it beats vectorising

Searched after the score-loop vectorisation came in at 1.59x with an 11% stage ceiling.
The literature does not try to make the sweep fast. It avoids the sweep.

**ELAS** (Geiger, Roser, Urtasun, ACCV 2010, `libelas`) is the closest to what this
project already has. It matches a **sparse set of support points**, triangulates them,
and uses that as a prior that narrows the disparity search for every remaining pixel.
Real-time on a CPU at over a megapixel. The support-point selector is **the ratio between
the first and second SAD minimum** -- which is `Match::margin`, already exported here,
already measured (precision by margin quartile 0.169 / 0.286 / 0.391 / 0.659). And the
prior structure is a Delaunay triangulation, which is exactly what TODO 3.2 says to use
instead of the k-nearest-neighbour proxy. Three pieces of this are already built.

**ESPReSSo** (Goldman et al., 3DV 2018) is the one that answers the question the
coarse-to-fine negative left open. It runs PatchMatch over slanted plane hypotheses on
precomputed binary descriptors and aggregates with the **permeability filter** -- a
separable recursive edge-aware filter, the same family as `rf_filter` here. The trick that
makes aggregation legal is that **plane hypotheses are shared across rectangular tiles.**
Within a tile the hypothesis is constant, so the filter aggregates at constant disparity,
which is the exact property the offset-indexed parametrisation destroyed and the reason it
cost 4 points. 11 ms per frame, but on a desktop GPU, so the number does not transfer.

**rSGM** (Spangenberg, Langner, Adfeldt, Rojas, IV 2014) does VGA at D=128 above 16 Hz on
a CPU, and notes that **every part of SGM except the data cost is agnostic to which
disparities are sampled** -- so large disparities can be sampled on a sparser grid. Cheap
to try and the wrong direction for us: large disparity is the near field, where this
project's 0.19 m gate problem already lives.

**DeepPruner** and **SCV-Stereo** are the learned versions of the same idea -- predict a
per-pixel disparity range, build a sparse cost volume in it. Not for a TX2 CPU, but they
confirm which axis the field thinks is the expensive one, and it is not the inner loop.

### What this says about where the time should go

The measured stage, TX2, teddy, agg 5, six threads, thread-summed CPU:

| part | ms | note |
|---|---|---|
| **recursive filter** | **93** | biggest item; vertical passes are pure row-wise SIMD |
| score | 77 -> 48 | done, `--simd`, 1.59x, and it did not help the wall clock |
| top-2 insert | 67 | one streaming read plus a cached compare, already thin |
| alloc | 17 | newly visible; nobody knew it was there |
| **total CPU / wall** | **256 / 68** | **3.79 of 6 cores** |

Three levers, and they are not the same size:

1. **Stop computing all D planes.** Our own ceiling measurement says 81.5% correct-over-
   known is available against 67.9% delivered, at **5.2x less arithmetic**. That is the
   largest number in this file and it has been sitting in it since the coarse-to-fine
   work. What failed was one parametrisation, not the idea, and ELAS and ESPReSSo are
   both published constructions that do not have that failure mode.
2. **Occupancy: 3.79 of 6 cores.** A 1.6x with no arithmetic change at all, and the SIMD
   experiment showed the schedule is fragile enough to hand back more than a vectorised
   inner loop can win.
3. **SIMD the filter, not the score.** It is now the largest item. Its vertical passes are
   `cur[x] += a[x]*(nxt[x]-cur[x])` across a row with no recurrence in x -- eight int16
   lanes, no permutes, no tables. The horizontal passes are serial in x and are not this.

Vectorising the score loop was ranked first and was worth 11% of the stage at best. The
ranking was wrong, and it was wrong because it ranked by *item size* rather than by
*item size times achievable factor divided by what the change costs elsewhere*.

## The mask construction, built and measured: parity accuracy, 1.24x desktop, FLAT on the TX2

`--c2f` in de_dense: a half-resolution coarse pass (D/2, ungated, holes filled row-then-
column), nearest-upsampled into a prior, and a fine pass over ABSOLUTE-disparity planes
where plane d is evaluated only on rows whose interval wants it -- per-row [mx0,mx1]
intervals, per-plane bounding boxes, the filter over the padded rectangle via a rect
variant of rf_filter_i16, the insert inside the interval. Exactly the form "What would
actually work" specified. The default path is bit-identical before and after, checked on
both machines.

Pooled over the eight scenes: **76.7% coverage / 10.6% bad-1.0 against 76.0% / 9.7%** --
correct-over-known 68.6% against 68.8%, parity. Desktop 46 -> 37 ms mean. **TX2, real
848x480 pair, interleaved best-of-6: 156 min full sweep against 168 min --c2f. Flat.**

### Why 5.2x of arithmetic became 1.0x of TX2 runtime

The ceiling measurement promised 5.2x less *arithmetic* and delivered it -- the fine
score loop's work really does shrink by the band fraction. Three costs do not:

- **Background planes have full-width rectangles.** A plane visible left and right of a
  foreground object gets a bbox spanning the row, so fill and filter touch nearly the
  whole image for most planes. Only truly absent planes vanish.
- **The solve does not scale with D at all.** It is resolution-bound: the fused merge
  streams nthreads x 4 int16 planes -- 19.5 MB per pass at 848x480 -- and at 47 ms it is
  now the LARGEST single item on the board. Plus 13 more in the coarse solve.
- **The coarse level is a fixed 43 ms** on the TX2 (census 6, cost 19, solve 13, prep 5),
  almost exactly what the mask saves from the fine cost stage (108 -> 78).

Two sub-experiments, both measured, both kept out:

- **A second band around the coarse runner-up** (for thin structures that vanish at half
  resolution: Art 12.7 -> 14.9): pooled 10.6 -> 11.2, teddy 7.1 -> 8.0. Where the coarse
  level is ambiguous, its runner-up is the wrong period of a repetitive texture, and a
  band around it hands the fine solver a wrong surface it can aggregate into confidence.
- **A strict per-pixel membership test in the insert**: 10.6 -> 11.6, a full point. The
  row-interval slack admits candidates a pixel's own band would exclude, and candidates
  decide the outcome. The loose interval IS the better matcher.

Coarse `iters=1` and MPAD 16 -> 8 are both accuracy-neutral (< 0.1) and kept.

### Where this leaves the real-time plan

`--c2f` stays, off by default: accuracy parity, a real desktop win, and it is the vehicle
for the temporal prior (TODO 3.1), which needs exactly this mask machinery and has no
coarse-level cost at all -- the previous frame is free. But it is not the 5.2x, and the
5.2x as imagined does not exist in this construction on this board.

The identified next structure, NOT built: **parallelise the masked fine pass over row
stripes instead of planes.** Each thread owns a stripe and runs all active planes over
it; the top-2 becomes one shared pair of planes with no merge at all (rows are disjoint),
killing the 47 ms solve-merge and the 19.5 MB x 2 of per-thread top-2 traffic. The price
is vertical-IIR truncation at stripe boundaries -- sigma_s = 12 decays to 0.15 in 16 rows,
so ~40% of rows in a 6-stripe split see some truncation. That is a measurable quality
question and the reason it is a design decision rather than a patch.

The other route is the GPU, which is where the frame budget was always going to close;
see TODO 0.3.

## The GPU port: 17.5 Hz at 848x480, and every step verified by cmp

`de_dense_cuda`, the split 10-architecture.md proposed: GPU owns census, graded cost,
recursive filter and the running top-2; the CPU keeps the MASDA solve. Every device
computation replicates the CPU's integer arithmetic exactly, and the GPU's per-pixel
ascending-k top-2 is deterministic -- so the whole binary is checked with `cmp` against
`de_dense --threads 1`: **zero bytes differ, at 848x480 and 450x375, including through
30 pipelined frames.**

| | 848x480 D=60 | 450x375 D=60 |
|---|---|---|
| CPU, six cores | 152 ms | 70 |
| GPU+CPU, single frame | ~90 | ~35 |
| **GPU+CPU, pipelined** | **56.6-57.6 ms = 17.5 Hz** | **24.6 ms = 40.6 Hz** |

GPU stage breakdown at 848 (cudaEvents): census+coeffs 1.5, score 14, **filter 34**,
top-2 3, transfers ~7. The filter is the remaining wall to 30 Hz at full resolution.

Four findings with mechanisms, each paid for in a measured wrong version first:

1. **A thread per filter row was 403 ms.** Warp-adjacent rows stride W*2 bytes; nothing
   coalesces and the reuse thrashes the 512 KB L2. Transposing is pure data movement --
   bit-identity survives -- and the coalesced form is 43 ms. The pre-transpose then
   disappears entirely by having the score kernel write the transposed layout.
2. **The score kernel needs coalesced reads AND transposed writes.** Per-plane grids
   have the reads (15 ms), per-pixel loops have the writes (21 ms); fusing score and
   transpose through a 32x32 shared tile has both (14 ms).
3. **The vertical recurrence's dependent load is unnecessary.** The CPU reloads its own
   previous int16 store; the register carry is the same value, drops a load per step.
4. **The TX2 has no I/O coherency: every cudaHostAlloc flavour is CPU-uncached.** The
   solver on uncached candidates measured ~300 ms against ~40 cached -- found twice,
   once via Mapped, once via Default, which on Tegra is not the "cached pinned" the
   documentation's mental model suggests. Candidates travel through ordinary pageable
   memory: a ~4 ms staged copy at the pipeline's natural sync point.

The pipelined mode (`--frames N`) is measured, not claimed: GPU computes frame t+1
while the CPU solves frame t, and the LAST frame's output is what `--out` writes, so a
broken overlap breaks the identity check instead of passing quietly.

**What 30 Hz at 848x480 needs**, in order: the filter's 34 ms (two IIR sweeps plus one
transpose; candidate ideas are two-plane ILP per thread and fusing the transpose-back
into the vertical pass), then the score's 14. The solve does not need to get faster
until the GPU side is under ~30 ms, because the pipeline hides it. And half resolution
already exceeds 30 Hz today if a use case wants that trade.

## 30 Hz reached: the k-minor volume, and what it cost to find

`de_dense_cuda` at **28.9 ms/frame (34.6 Hz) at 848x480 D=60**, full quality,
bit-identical to the CPU on all eight scenes and the real pair. The day's path, every
step cmp-verified: 57 -> 39 (fused score, one transpose deleted) -> 34 (packed top-2,
k-pair transactions) -> 30.4/29.9 (A57 pinning, fetch overlap) -> 31.5 (runtime Dpad,
after the bug below) -> 28.9 (Dpad group count as a template parameter).

**The structural insight: `vol[y][x][k]`, k innermost.** The layout the CPU rejected
three times is the GPU's right answer -- the machines want opposite layouts. A warp
becomes 32 disparities of one row: census reads for consecutive d are one sliding
256-byte window, volume accesses are aligned k-runs, and two whole passes disappear
into fusions (score into the horizontal forward filter; top-2 into the vertical
backward, whose stores then have no reader and die).

Negatives and bugs, with mechanisms:

- **Warp-serial shuffle scan of the recurrence: 55.7 ms** against the 21 it replaced.
  Bit-exact and fully coalesced -- and 31 of 32 lanes burn issue slots per serial
  step. Instruction throughput, not bandwidth. This is why Nehab et al. reassociate
  recursive filters into block-parallel form, which an int16-truncating recurrence
  cannot do bit-exactly.
- **`__constant__` tables serialize under divergent indexing** -- 32 Hamming distances
  per warp is divergent by construction. Shared-memory copies: 11.2 -> 9.2 ms.
- **The top-2 shuffle tree merges non-adjacent k ranges** (lane 0 meets lane 16 before
  lane 1), so a tie rule that assumes "the other side holds larger k" is wrong
  mid-tree: 10 wrong pixels in 407k, every one an exact tie. The fix packs (value,
  255-k) into one int32 so plain integer comparison is (value desc, k asc) and the
  merge is order-independent. A 5-million-run host simulation of the exact tree found
  it and is now `test_top2_merge` in make test.
- **A compile-time DPAD=64** silently broke six of eight scenes -- everything with
  D=80 never scored its upper disparities and read the neighbouring pixel's run.
  teddy and cones, the two scenes everything gets tuned on, have D=60 and passed.
  The eight-scene identity sweep caught it; a teddy-only check would have shipped it.

CPU side: solve threads pinned to the A57 cluster (the Denver pair caused the 30-45 ms
wander; pinned it is 22.6-23.1), and the candidate fetch moved to its own thread,
where cudaMemcpy's stream serialization hides it inside the solve.

Steady state at 848x480: GPU ~26 ms of kernels, solve 23 ms in parallel, fetch
hidden, 28.9 total -- 13% under the 33.3 ms budget, at the full operating point:
48-bit census, graded cost, edge-aware recursive filter, D=60, MASDA solve with
margin gate.

## Pinning the CPU tool to the A57s: a measured negative with a two-sided mechanism

The GPU tool's solve went from 30-45 ms of wander to 22.6-23.1 when its four threads
pinned to the A57 cluster, so the same pin was tried on `de_dense` itself (all five
worker pools; cores verified by CPU part id, 0xd07 = A57 at {0,3,4,5}, 0x003 = Denver
at {1,2}). Interleaved best-of-8 at 848x480:

| | min | spread |
|---|---|---|
| unpinned, 6 threads | **169.7 ms** | 40% |
| pinned, 6 threads on 4 A57s | 243.4 | **4%** |
| pinned, 4 threads | 213.1 | 4% |

**40% worse on the mean, ten times tighter on the spread.** Both effects have the same
cause: the Denver cores are real throughput AND the source of the variance. The CPU
tool wants all six cores and pays for them in measurement noise -- which is exactly why
interleaved best-of-N exists. The GPU tool's solve pins WELL because it only needs
four threads and the Denvers are busy being the CUDA driver and the fetcher thread.

Reverted; the pin stays only in `de_dense_cuda`, where it was measured to help. The
general rule that survives: pinning is a per-configuration measurement, not a platform
fact.
