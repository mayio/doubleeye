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

## What is not done

- **Coarse-to-fine.** The plan's biggest single lever: match at 1/8 resolution,
  fit a disparity field, then use it as a *soft* prior in `s(i,j)`. That should cut
  candidates per keypoint from the current ~2.7 and, more importantly, remove the
  false candidates that create the near-ties driving the oscillation above.
- **Calibrating gamma and lambda.** Currently hand-set. The plan's candidate is a
  small MLP over descriptor distance, y-residual, coarse-disparity residual and
  response ratio.
- **Disparity smoothness** over a Delaunay triangulation — the largest piece of
  currently-ignored information, and the plan's suggestion to try the cheap
  two-pass version first (match, fit a robust surface, re-score, re-match) before
  deriving new messages. Eigen is available on the Jetson for that fitting; the
  desktop needs `libeigen3-dev`.
- **Sub-pixel disparity refinement**, which the plan notes depth accuracy hinges on.
- **Ground-truth validation** against a laser rangefinder — bring-up step 4. Until
  then, median |dy| and the objective are the only quality measures, and neither
  proves the disparities are *correct*, only that they are consistent.
