# Preprocessing — Census descriptors and keypoints

The first algorithmic component: `core/`, portable C++14 depending on nothing but
the standard library. No librealsense, no OpenCV, no CUDA, no Jetson headers.
That is the plan's portability rule, and it also means the whole thing builds and
runs on the desktop against recorded bags.

```sh
cd core && make          # plain make; the desktop has no cmake and needs none
make test                # 24 unit tests
./build/de_preprocess ../bags/full_on
.venv/bin/python desktop/view_keypoints.py bags/full_on
```

The same Makefile builds on the TX2 (make 4.1, g++ 7.5), so there is no second
build system to keep in sync.

## What it does

1. **Shi-Tomasi response**, densely. Minimum eigenvalue of the structure tensor
   rather than Harris: same cost, but the response is an eigenvalue in intensity
   units instead of a `det − k·tr²` blend, so a threshold means something and
   transfers between scenes. Harris's `k` is one more magic number.
2. **3×3 non-maximum suppression**, asymmetric on ties so a plateau yields
   exactly one keypoint rather than none or several.
3. **Texture rejection** — 7×7 local standard deviation below 2 DN is discarded.
   Not decoration: with the projector off, 57% of the image is below that floor.
   A keypoint there has a descriptor made of sensor noise, which is worse than no
   keypoint because it manufactures confident wrong matches.
4. **Grid bucketing** — top-N per 32 px cell. Essential, not a refinement:
   without it everything piles onto the nearest high-texture surface while the
   far wall, where disparity precision is worst and measurements matter most,
   gets nothing.
5. **Sub-pixel refinement** — parabola through the three response samples per
   axis. Depth accuracy hinges entirely on this: at 2 m, 0.1 px of disparity is
   about 2 cm of range.
6. **Census descriptor** at each surviving keypoint, 7×7 → 48 bits in a `uint64`,
   so Hamming distance is one `popcount`.

## Measured output

Two 848×480 recordings of the same static scene, differing only in emitter state.

| | emitter ON | emitter OFF |
|---|---|---|
| keypoints per frame | **1115** | 754 |
| cell occupancy | **92.8%** | 66.9% |
| median texture at keypoints | 8.8 DN | 11.7 DN |
| **distinct Census descriptors** | **338 of 1115 (30%)** | 585 of 746 (78%) |
| Census bits carrying information | 48 / 48 | 48 / 48 |

Three things worth drawing out.

**Keypoint count matches the plan's estimate.** The plan budgets "~1000
keypoints"; we get 1115.

**The projector is confirmed a third time, now on the quantity that matters
most.** Earlier evidence was image statistics (textureless area 57% → 25%). Here
it is the actual detector output: +48% more keypoints and coverage rising from
67% to 93% of cells. The two cells with no keypoints at all are the dark ceiling
band and **the blown-out window** — the predicted saturation desert, visible
directly in `keypoints.png`.

**The descriptor ambiguity the plan anticipated is real and now quantified.**
With the projector on, only **338 distinct descriptors serve 1115 keypoints** —
an average of 3.3 keypoints sharing a descriptor. With it off, 585 of 746 are
unique. This is precisely the plan's reasoning: every projected dot looks
locally identical, so a 7×7 window sees a constellation that recurs. The plan
calls this out as "exactly where MASDA's uniqueness constraint earns its keep. Good
test case." It now has a number: **descriptors are ~3.3× degenerate in the regime
we intend to run in.** A nearest-neighbour matcher would be badly served here;
mutual exclusivity is not optional.

All 48 bits vary across keypoints in both regimes, so there are no dead bits and
the window size is not obviously wrong.

## Profiling — the plan was right, and it is 3× over budget

The plan says MASDA is not the bottleneck, that preprocessing and memory
bandwidth are, and to profile before touching CUDA. Done, and it holds.

| | desktop (x86_64) | **TX2 (MAXN, clocks locked)** |
|---|---|---|
| detect | 16.8 ms | **49.5 ms** |
| census (sparse) | 0.16 ms | 0.32 ms |
| per stereo pair | 33.9 ms | **99.6 ms** |
| fraction of the 33.3 ms budget at 30 Hz | 102% | **299%** |

**On the target hardware, preprocessing is three times over budget, and it is
essentially all detector.** Census is 0.3% of the cost.

### Two optimisations already applied, both behaviour-preserving

Keypoint output is byte-identical before and after (1115.125 mean, 0.928
occupancy), so these are pure wins rather than accuracy trades.

| Change | Effect |
|---|---|
| **Census evaluated sparsely** rather than densely | 61 ms → 0.13 ms, a 480× reduction |
| Separable sliding-window box sum instead of a double-precision integral image | modest, but removes a 3.3 MB buffer per call |
| Row pointers in the Sobel loop, `-O3` | 20 ms → 16.8 ms |

Together: **183 ms → 33.9 ms per pair on the desktop, 5.4×.**

The Census one is the instructive one. The original code built a dense Census
image — a descriptor for all 407040 pixels — and then read roughly 1100 of them.
99.7% of the work was discarded. Sparse stereo needs descriptors *at keypoints*,
and nowhere else. That is an architectural error, not a micro-optimisation, and
no amount of SIMD would have fixed it.

### Identified next steps, in expected order of value

1. **FAST for candidate detection, Shi-Tomasi only at candidates.** The same
   sparse-the-expensive-operator move that gave 480× on Census. FAST is a handful
   of comparisons per pixel and is cheap enough to run densely; the structure
   tensor then only needs evaluating at a few thousand FAST corners instead of
   407040 pixels. The plan already sanctions "FAST/Harris". This is very likely to
   close the 3× on its own.
2. **Process the two channels concurrently.** Left and right are completely
   independent and the board has 6 cores. Roughly halves wall-clock per pair for
   almost no effort.
3. **NEON.** The plan explicitly anticipates this. Worth doing after 1 and 2,
   because it is the most invasive and may turn out unnecessary.
4. **Detect on a reduced-resolution level.** Falls out of the coarse-to-fine
   pyramid the plan wants anyway.

Deliberately **not** CUDA. The plan rejects it for this workload and nothing
measured here contradicts that — the problem is redundant work and scalar code,
not a shortage of parallel throughput.

## Output format

`keypoints.csv` in the bag:

```
frame,stream,x,y,response,local_std,census_hex
00000030,1,71.512,32.000,3.9532,4.898,0000000000041041
```

`x`/`y` are sub-pixel; `census_hex` is the 48-bit descriptor. This is what the
matcher will consume.

One detail that looks like a bug and is not: `view_keypoints.py` can report a
per-cell count slightly above `--per-cell`. The cap is enforced on integer
positions before sub-pixel refinement, and refinement can move a keypoint up to
0.5 px across a cell boundary. It does not affect spatial distribution, and
`cell_occupancy` uses refined positions in both C++ and Python so the two agree.

## Tests

`make test` — 24 assertions, no framework. Each targets a property the design
relies on rather than restating the implementation. The important one:

**Census invariance to monotonic intensity mapping.** A gain-and-offset remap
(`p → p/2 + 20`) leaves **0 of 1024** descriptors changed. That invariance is the
entire reason the plan picks Census, and it is what absorbs the measured 2.6 DN
exposure mismatch between the two IR sensors for free.

Also covered: descriptor bit widths (48 for 7×7, 62 for 9×7, both fitting a
`uint64`), border validity, Hamming, local-std against an analytic value,
response ordering flat < edge < corner, the per-cell cap, occupancy spread,
sub-pixel actually moving points, the texture floor suppressing everything when
set above range, and the raw loader rejecting wrong geometry rather than
silently producing garbage.
