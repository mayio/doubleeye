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

### Given a fair hearing, on a scene built for it

The lattice was an unfair test, so `thin_bars_disparity()` provides nine thin
foreground bars at assorted depths over broadband texture. Errors there *do* cross,
and the true solution genuinely violates ordering, so it tests both halves at once.

| kappa | matches | correct | precision | recall | crossings |
|---|---|---|---|---|---|
| off | 711 | 502 | 0.706 | 0.781 | 71 / 2685 |
| 0.1 | 711 | 506 | 0.712 | 0.787 | 59 |
| **0.4** | 710 | **508** | **0.715** | **0.790** | **53** |
| 0.8 | 710 | 507 | 0.714 | 0.788 | 53 |

Crossings fall 25% and correct matches rise by **six, out of 711**. Still marginal.

### Why — and it generalises

Look at the baseline: **71 crossings out of 2685 same-band pairs, 2.6%.** Ordering
was never a significant error mode, so there was little for it to correct.

That is not an accident of this scene, and the reason is analytic. Matches $(i,j)$
and $(i',j')$ with $x_i < x_{i'}$ cross iff $x_j > x_{j'}$, i.e.

    d_i' - d_i  >  x_i' - x_i

A crossing therefore requires the **disparity difference to exceed the horizontal
separation**. With disparities confined to a range of width `dmax - dmin`, crossing
is only *possible* for keypoint pairs closer together in x than that width — and
rarer still as the range tightens.

**So the disparity-range gate already does most of ordering's work.** Uniqueness
plus a bounded disparity range yields largely ordered solutions for free, which is
why an explicit ordering factor buys ~1% however favourable the scene.

The honest conclusion: ordering is *expressible* (cleanly, as a clamp, inside the
existing closed form) and *nearly redundant* in a geometrically gated sparse
matcher. It would matter where the gating is weak — a wide disparity range, an
uncalibrated pair, or 2-D temporal association where no ordering comes for free.

Kept soft (finite kappa) deliberately: thin foreground objects genuinely violate
ordering and a hard constraint would delete them. Damping defaults to 0.6, higher
than the bipartite case, because these factors add loops the convergence result
does not cover.

## Reproducibility — fixed

`RNG` used to be module-level and consumed in call order, so a scene depended on how
many draws preceded it: `run_regime` called directly gave 1163 matches where
`main()` gave 1109. Each generator now takes its own purpose-seeded `Generator` via
`rng_for(name)`, verified stable across interleaved calls.

Note this means figures regenerated now will differ slightly from the numbers in the
current draft of the post, which were produced under the old call-order-dependent
scheme. **Re-run `masda_stereo.py` and update the tables before publishing.**

## Table provenance

Regenerate every number and figure the article quotes with one command:

    python regen_all.py

It writes `results.json` alongside the figures, so a changed number shows up in a
diff. Prior to this the article's numbers came from several ad-hoc scripts, which is
how they drifted.

### Seeding, and a bug worth knowing about

`masda_stereo.rng_for(name)` used to seed from Python's builtin `hash()`. String
hashing is salted per process (PEP 456), so the seed differed on every run: the
scene was deterministic *within* a process and irreproducible *across* two. Nobody
could have reproduced the published tables, and worse, one conclusion about the
ordering factor turned out to be a comparison between two different random scenes.
It now seeds from `zlib.crc32`, which is stable across processes, versions and
machines. `BASE_SEED` can be varied to check that a result is not seed noise, and
the ordering experiment does exactly that across five seeds.

The lesson generalises: any result reported from a single random scene is a claim
about that scene. The ordering sweep reports mean and spread over seeds because the
first two attempts at it disagreed in sign.

## Real data

`masda_middlebury.py` runs the same matcher on Middlebury 2003 Teddy and Cones.
Middlebury states "We grant permission to use and publish all images and disparity
maps on this website", so the images and the derived figures are safe to publish;
cite Scharstein & Szeliski, CVPR 2003. Images download on first run into `data/`,
which is gitignored rather than vendored.

Two caveats recorded in the script:

- `vision.middlebury.edu` serves an incomplete certificate chain (it omits its
  intermediate), so TLS verification fails locally. The download is unverified and
  the payload is checked instead: expected size, expected disparity scale, and a
  disparity range consistent with the published `ndisp`.
- Middlebury marks unknown ground truth as 0 rather than shipping a visibility mask,
  so `evaluate_real` is separate from the synthetic `evaluate`. Matches landing on
  unknown ground truth are counted and excluded from precision, not scored as wrong.
