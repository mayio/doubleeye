# Article: MASDA for Sparse Stereo Matching

Draft post for [mariolueder.com](https://www.mariolueder.com), a follow-up to the
original MASDA post applying it to sparse stereo.

## Files

| file | goes to |
|---|---|
| `2026-08-07-MASDA-for-Sparse-Stereo-Matching.md` | `_posts/` in `mayio/mayio.github.io` |
| `figures/*.png` | `assets/img/2026-08-07-MASDA-for-Sparse-Stereo-Matching_files/` |
| `masda_stereo.py` | wherever you want it referenced from; nothing depends on its location |

The figure URLs in the markdown already point at
`raw.githubusercontent.com/mayio/mayio.github.io/master/assets/img/2026-08-07-MASDA-for-Sparse-Stereo-Matching_files/`,
matching the convention of the existing MASDA post.

## Licensing

**No input images.** The scene is generated procedurally, so nothing here carries a
third-party licence and there is no attribution to get wrong. That was a
requirement, and it turned out to be an advantage: procedural generation supplies
**exact ground-truth disparity**, so match correctness is measurable rather than
merely plausible — which is precisely what the real-data experiments in this repo
could not do without a rangefinder.

Middlebury is referenced as the standard benchmark for readers who want real
imagery, but deliberately not redistributed: those datasets are provided for
research use and embedding them in a blog post is a licensing question best not
answered.

## Reproducing

```sh
cd article && ../.venv/bin/python masda_stereo.py
```

Writes every figure into `figures/`. Needs numpy, scipy, matplotlib.

## Two solvers, on purpose

`masda_sparse()` is the one to use: messages on an edge list, O(T*E), measured at
157-282x faster than scipy's Jonker-Volgenant with quality identical to the dense
form -- same objective to four decimals, same correct-match counts against ground
truth bar one tie in 902.

`masda()` keeps the dense O(T*m*n) formulation deliberately, because section 6 of
the post uses the contrast as its central argument: the asymptotic advantage is real
and entirely forfeited by a representation that ignores sparsity. `to_edges()`
converts between them so both run on identical input.

## Known rough edges

- `top2_excluding`, in the **dense** solver only, uses a full `argsort` where
  `argpartition` would do. Left alone on purpose -- that path exists to be the slow
  comparison, and speeding it up would blunt the point.
- `build_problem` still forms the dense matrix before `to_edges` reduces it, so
  problem *construction* is O(m*n) even though the *solve* is O(E). Fine for a demo,
  wrong for a pipeline, which would emit candidates straight from the epipolar
  search.
- Keypoint counts differ between the three textures (1405 / 1614 / 2019), so "harder
  texture" is mildly confounded with "larger problem". The conclusions rest on
  per-regime ratios rather than absolute counts, so this does not change them, but
  capping detections to a common budget would be cleaner.

## Ordering factor (`masda_sparse_ordering`)

Answers "can MASDA express the ordering constraint if we add a factor?" — yes.

Two associations on a scanline cross iff `(x_i - x_i')(x_j - x_j') < 0`, and a
matching is order-preserving *exactly* when no two pairs cross. So ordering
decomposes into **pairwise** factors; no higher-order factor is needed.

For a factor charging `kappa` when both edges are on and crossing, the max-sum
factor-to-variable message reduces to

    Delta_{psi->e} = -clamp(mu_f, 0, kappa)

i.e. the conflicting edge's own preference, clamped and negated. One scalar, O(1).
Verified against brute-force max-sum to 4e-16.

Because those messages are additive on the edge they fold into the score, so the
updates are the **same two reductions with `s + o` in place of `s`**. The closed
form survives; no new message type has to be maintained.

### Measured, on the ambiguous (periodic) case

| kappa | matches | correct | precision | recall | crossings |
|---|---|---|---|---|---|
| off | 1163 | 223 | 0.192 | 0.367 | 272 / 10184 |
| 0.1 | 1161 | 229 | 0.197 | 0.377 | 172 |
| **0.3** | 1158 | **229** | **0.198** | **0.377** | **156** |
| 0.6 | 1158 | 227 | 0.196 | 0.374 | 153 |
| 1.2 | 1157 | 227 | 0.196 | 0.374 | 152 |

**It works mechanically: crossings fall 44% (272 -> 152).** Cost ~33 ms against
~14 ms, so ~2.4x, and 4408 crossing edge-pairs in the graph.

**But the accuracy gain is marginal** — 223 -> 229 correct, precision 0.192 ->
0.198. It does not rescue the precision collapse, and the reason is worth stating:
on a repetitive pattern the wrong matches are largely *order-preserving*. A whole
region shifted by one lattice period crosses nothing. Ordering rejects crossings,
and periodic texture produces ordered mistakes, so the constraint is nearly
orthogonal to that failure mode.

Ordering should pay against *disordered* errors — a thin foreground object, a
depth discontinuity — which is what it is for. It is not a cure for descriptor
degeneracy.

Kept soft (finite kappa) deliberately: thin foreground objects genuinely violate
ordering and a hard constraint would delete them. Damping defaults to 0.6, higher
than the bipartite case, because these factors add loops the convergence result
does not cover.

## Reproducibility caveat

`RNG` is module-level and consumed in call order, so the scene depends on how many
random draws preceded it. Numbers in the post came from `main()`'s specific
sequence; calling `run_regime` directly in a different order gives a slightly
different scene (e.g. 1163 vs 1109 matches on periodic). The qualitative results
are unaffected, but each generator should take its own seeded `Generator` before
these figures are treated as a benchmark.
