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
