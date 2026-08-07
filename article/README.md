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

## Known rough edges in the script

Both are discussed in the post rather than hidden, since they are part of its
argument, but they are real and worth fixing if the code is reused:

- `masda_sparse()` is the fast path: edge-list messages, O(T*E). `masda()` keeps the
  dense form because section 6 of the post uses the contrast as its central point.
  The dense one
  O(T·m·n) work where O(T·E) is available. With ~1400 nodes and ~3200 edges that is
  ~600x wasted effort, and it is why the numpy implementation loses to scipy's
  Jonker-Volgenant. Section 6 of the post uses this as its central point about where
  the asymptotic advantage actually lives.
- `top2_excluding` uses a full `argsort` where `argpartition` would do, adding a
  needless log factor.

Keypoint counts also differ between the three textures (1405 / 1614 / 2019), so
"harder texture" is mildly confounded with "larger problem". The conclusions rest on
per-regime ratios rather than absolute counts, so this does not change them, but
capping detections to a common budget would be cleaner.
