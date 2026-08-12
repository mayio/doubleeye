# TODO

Ordered by what unblocks the most, not by effort. Each item says *why* it matters
so it can be picked up cold.

Status of the bring-up sequence lives in [README.md](README.md); this is the
actionable list.

---

## 0. Architecture: **DECIDED 2026-08-10 — the matcher runs on the GPU**

**Mario's call: the stereo matcher has to run on the GPU; there is no other choice.**
The decision, what it constrains and what expired with it are recorded in
[10-architecture.md](10-architecture.md).

The boundary proposed there survives intact — **GPU owns the image plane, CPU owns
the graph, and the interface is a compact candidate buffer rather than an image.**
What did not survive is the *"no CUDA yet"* schedule attached to it, whose premise
was 10 ms of slack measured against the **sparse** pipeline. Dense at 848x480 was 6x
over budget on the CPU with the GPU at load 0, and `de_dense_cuda` now runs it at
34.6 Hz bit-identical to the CPU path (section 0.3).

**And no CNN** (Mario, 2026-08-10). That settles what the GPU is for: 26.0 ms of the
28.9 is kernels, so at 30 Hz and full resolution the device is close to saturated, and
it is the matcher's. Object detection and tracking are the graph side instead — the
same MASDA machinery, on the four CPU cores that are still idle, over keypoints and
now a dense disparity map.

**What the decision leaves open**, in order:

1. **One producer for the sparse feature set.** Unchanged and still worth doing:
   every tool re-runs detection itself, and with four consumers that is four
   redundant detections.
2. **CUDA for the sparse front end (FAST, NMS) stays deferred**, and the reason is
   now stronger — it would compete with the dense kernels for a saturated device
   rather than claim idle silicon.
3. **Pin stages to cores explicitly.** The Denver/A57 asymmetry still bites, and the
   CUDA tool's solve threads already pin themselves to the A57 cluster.

---

## 0.3 Dense MASDA: **REAL TIME ACHIEVED -- 34.6 Hz at 848x480, full quality**

**State 2026-08-09: `core/tools/de_dense_cuda` runs 28.9 ms/frame steady state
(34.6 Hz) at 848x480 D=60 and 12.6 ms (79 Hz) at 450x375, pipelined, bit-identical
to `de_dense --threads 1` on all eight Middlebury scenes and the real pair, through
30 overlapped frames.** Config: `--threads 4 --frames N`; solve threads pin to the
A57 cluster themselves. Two days' arc: CPU 152 -> first GPU port 57 -> k-minor
volume 29.

The structural answer was the volume layout: `vol[y][x][k]`, k innermost -- the
`[x][d]` layout the CPU work rejected three times. The two machines want OPPOSITE
layouts, which retroactively is what the whole `[d][x]`-vs-`[x][d]` saga was about.
With k innermost: the score fuses with the horizontal forward filter pass (a warp =
32 disparities of one row, ReS2tAC's assignment), every volume access is a k-run or
a broadcast, and the top-2 fuses into the vertical backward pass so the filtered
volume is never stored or re-read. Kernels: census+coeffs 1.5, score+hfwd 8.8,
hbwd 5.0, vfwd 3.3, vbwd+top2 7.4 ms; CPU solve 23 ms hidden in the pipeline.

The traps that cost a day between 57 and 29, each recorded in the source and
09-matching.md: Tegra pinned memory is CPU-uncached in every flavour (~300 ms
solve, found twice); a warp-serial shuffle scan of the recurrence is issue-bound at
55.7 ms (the negative that motivated the layout change); the top-2 shuffle tree
merges non-adjacent k ranges so tie rules must compare k explicitly (10 wrong
pixels, all exact ties, caught by cmp, pinned by a 5M-run simulation that now lives
in `make test`); and a compile-time DPAD=64 silently broke every scene with D=80 --
caught only because the identity check runs all eight scenes, not just the two
everything gets tuned on.

What follows below is the CPU-side record as it stood before the port; its numbers
remain the CPU baselines.

## 0.31 Does the message passing earn its place? Measured 2026-08-11 — **no, in the dense path**

**First, a bug this exposed.** `--iters 0` used to read 67.29% bad-1.0 and looked like
proof that message passing was worth 41 points. It was not an ablation: with no
messages the max-sum belief `beta + rho - cs` degenerates to *minus* the score, so the
solver picked the WORST candidate. `--iters 0` now means winner-take-all on the score,
which is the true degenerate case.

**Then the ablation.** All 15 v3 scenes, `--threads 1`. Uniqueness (the greedy claim)
and the margin gate are present in EVERY row — only the message passing changes:

| | K=2 (shipping) | | K=8 | |
|---|---|---|---|---|
| | bad | coverage | bad | coverage |
| **iters 0 — winner-take-all** | **25.18** | 79.6% | **24.41** | 79.8% |
| iters 1 | 26.13 | 79.6% | — | — |
| iters 2 (default) | 26.04 | 80.1% | 25.58 | 80.6% |
| iters 4 | 26.19 | 80.2% | 25.78 | 81.0% |

**Message passing buys coverage and pays for it in precision** — 79.8% -> 81.0% for
24.41 -> 25.78 at K=8 — which is the same trade the margin gate and lambda/gamma
already offer, at the same sort of exchange rate. It moves along the
precision-coverage curve rather than off it, exactly like every other knob measured
this week.

**And it is not free.** TX2, 450x375, six threads, best of three: the solve is
**13.0 ms at iters 0 against 25.5 ms at iters 2.** At 848x480 that is ~12 ms of the
23 ms solve. Today that hides under the GPU kernels so it costs no wall clock — but it
is 12 ms of CPU, and 0.4's wiring question is short of about that much to fit
detection alongside the dense path.

**What this does NOT say.** Uniqueness and the margin are the parts that distinguish
this from a plain block matcher, they are present in all rows above, and the margin is
what makes the dense map beat SGM at keypoints (0.4). What is in question is only the
loopy belief propagation on top of them, in the DENSE path with a top-2 candidate set.
The sparse matcher's +46% over mutual-NN was measured with many candidates on
deliberately ambiguous projected-dot texture, and temporal association -- where the
candidate set is genuinely large and 2-D -- is untested. Do not read this as retiring
MASDA; read it as "the dense path may not be where it earns its keep".

**Not changed by default**, because coverage is a real axis and this is one benchmark.
The decision belongs with 0.4's wiring, where the 12 ms is worth something concrete.

## 0.3.1 The CPU record: the goal was real time, and the CPU got to 4.7x away

**Mario's stated priority, 2026-08-08.** Everything in this section is ranked against
that goal and not against SGM's accuracy any more.

**Measured at the real resolution, which changes the picture.** Every dense number
below and in 09-matching.md is Middlebury at 450x375. The sensor is 848x480 -- 2.4x the
pixels -- and rule 2 says a number measured in one context is not a measurement in
another, so it was measured rather than scaled. TX2, six threads, `--agg 5`, a real
rectified IR pair from `bags/full_on`, best of 5:

| | D=60 | D=64 | D=96 |
|---|---|---|---|
| total | **200 ms** | 184 | 219 |
| spread over 5 runs | 16% | 17% | 13% |

(The D=64 figure below D=60 is noise; the spreads overlap almost completely. Do not
read an ordering into it.)

| budget | need |
|---|---|
| 30 Hz, 33.3 ms | **6.0x** |
| 15 Hz, 66.7 ms | **3.0x** |

**Preprocessing is not competing for that budget -- dense MASDA replaces it.** The
26.28 ms figure in 06-preprocessing.md is FAST plus NMS, and detection is 21 of it.
Dense MASDA needs neither; it computes its own census (16 ms of the 200). So the
budget really is most of the frame.

**Where the 200 ms is**, D=60, thread-summed CPU except the last row:

| | ms | share |
|---|---|---|
| census | 16 | 8% |
| **cost stage** | **159 wall** | **77%** |
| — score | 199 | |
| — filter | 195 | |
| — insert | 179 | |
| — alloc | 34 | |
| solve (the MASDA part) | 31 | 15% |
| cost stage occupancy | 613 CPU / 159 wall | **3.85 of 6 cores** |

**The thing that makes this MASDA is 15% of the runtime.** The other 85% is the cost
volume and its aggregation, which is shared with every window-based matcher. So this is
not a "make MASDA faster" problem, and effort spent on the solver cannot pay.

### The 6x, and where each piece of it comes from

Two measured levers multiply to more than enough. Neither is SIMD.

1. **Stop computing all D planes — 5.2x, and it is not a speed/accuracy trade.** Our own
   ceiling experiment: 81.5% correct-over-known is *available* against 67.9% delivered,
   at 5.2x less arithmetic. The offset-indexed construction failed for a known reason
   (aggregation needs constant-disparity planes) and two published constructions avoid
   that failure mode: **ELAS** sparse support points chosen by the first-to-second-minimum
   ratio — which is `Match::margin`, already exported and already measured — triangulated
   into a prior; and **ESPReSSo** PatchMatch hypotheses shared across tiles, so the plane
   is constant within a tile and a recursive edge-aware filter stays legal. See
   09-matching.md.
2. **Occupancy — 1.56x for free.** 3.85 of 6 cores in the cost stage. No arithmetic
   changes, no accuracy risk, and the failure mode is now mapped: the `--simd` work
   showed a coarser work quantum hands back more than a vectorised inner loop wins.

5.2 x 1.56 = **8.1x**, against 6.0x needed for 30 Hz. That is the plan, with margin for
the two levers underdelivering — which they will.

Held in reserve, in order: **the filter** (32% of stage CPU, vertical passes have no
recurrence in x), **`--simd`** (exists, 11% stage ceiling, off by default), and
**resolution** (640x480 is 0.75x the pixels).

### The deferral in section 0 went to the critical path, and then was decided

Written when section 0 still deferred the GPU/CPU decision: its stated reason was *"No
CUDA yet, because there is 10 ms of slack and four idle cores"*, and **that premise was
measured against the sparse pipeline.** For dense at 848x480 there was no slack — 6x
over — and the GPU was still at load 0. Dense stereo is the canonical CUDA workload;
ReS2tAC's own CUDA path and libSGM both do VGA on Jetson hardware at rates the CPU
cannot approach.

**Resolved 2026-08-10: the answer is CUDA**, and the CPU's 8.1x plan is not being built.
The two levers it rested on are not wasted — "stop computing all D planes" is still the
biggest measured lever in the project and applies to the GPU path too — but the 1.56x
occupancy half is now moot.

## Reference: where it stands against SGM at 450x375

State end of 2026-08-08. `core/tools/de_dense.cpp`, eight Middlebury scenes:

| | coverage | bad-1.0 | desktop | TX2 |
|---|---|---|---|---|
| SGM | 78.0% | 10.9% | 16 ms | -- |
| **dense MASDA** | 76.0% | **9.7%** | ~45 ms | ~95 ms |

**Ahead on accuracy by 1.2 points, behind on coverage by 2.0.** The day went 246 -> 45 ms
desktop and 11.0 -> 9.7% bad-1.0, so speed and accuracy moved together throughout.

```sh
.venv/bin/python article/dense_bench.py --out /tmp/after    # 76.0% cov, 9.7% bad
cmp /tmp/before/teddy.f32 /tmp/after/teddy.f32              # for pure speed changes
tools/deploy.sh && ssh jetson 'cd ~/doubleeye/core && ./build/de_dense ...'
```

**Read [09-matching.md](09-matching.md) before touching this.** Every item below has a
measurement behind it and several have a recorded negative that looks like a good idea.

### The layout decision is settled: it was a false choice

**The cost volume does not become disparity-minor.** The score wants disparity in the
REGISTER FILE and the filter wants it in MEMORY, and those are independent -- nothing about
the loads is disparity-minor either way. `--simd` in `de_dense.cpp` computes a group of
eight disparities with disparity in the lanes and transposes in registers before storing,
so the filter, insert and solver see the planes they always saw. Bit-identical, all eight
scenes, at `--threads 1`.

**Measured, TX2, teddy, interleaved best-of-8: score loop 1.59x (77.0 -> 48.4 ms), cost
stage wall 0.92x.** 9% less CPU, 8% more wall clock, 0.62 of a core lost. At D=96 (twelve
groups over six threads) the sign flips to 1.04x on the stage, so the D=60 regression is
the eight-groups-over-six-threads schedule. Full accounting, including two failed mechanism
guesses and the allocation timer that finally explained it, in 09-matching.md.

**The ceiling is the real finding.** The score is 30% of cost-stage CPU, so 1.6x on it is
an 11% stage ceiling, and the layout costs ~12 ms of clear and alloc against the 29 ms it
saves. At agg 5 the **filter is now larger than the score** (93 against 77 ms). Off by
default. Making it pay means decoupling the work quantum from the group size -- score into
shared group buffers, then hand out single planes for filter and insert from a second queue
-- which is real engineering against an 11% ceiling. Reverting is also defensible.

**`cmp` between two multi-threaded runs is not an identity check.** Two identical scalar
runs at six threads differ from each other: the top-2 insert keeps the first of two equal
scores and blocks are handed out dynamically. Verify at `--threads 1`, and only there.

### Ranked after that

1. **The recursive filter, not the score loop.** It is the largest item in the stage at
   agg 5 -- 93 ms of TX2 CPU against the score's 77 -- and vectorising the score just moved
   the score below it. Four passes of `cur[x] += a[x]*(nxt[x]-cur[x])`; the horizontal
   passes are serial recurrences in x but the vertical ones are pure row-wise SIMD.
2. **Coverage**, the only axis SGM still leads and untouched all day. The margin gate is
   deliberately trading it for precision; the question is whether a better-calibrated gate
   holds the precision at higher coverage. **Reframed 2026-08-10:** the v3 board says
   coverage-for-precision is *the* axis the whole field trades on, and everyone picks a
   single point on it and publishes that. Sweeping the gate and **publishing the
   precision-coverage curve** is nearly free here -- `Match::margin` and the
   precision-by-quartile numbers already exist -- and it is a contribution rather than
   a catch-up, because the board structurally cannot report one. See 4.2.
3. **Stop computing all D planes -- the biggest measured lever in the project, 5.2x.**
   Our own ceiling says 81.5% correct-over-known is available against 67.9% delivered.
   The offset-indexed version is a measured negative with a known mechanism: aggregation
   needs constant-disparity planes. **Re-ranked 2026-08-10 after reading the Middlebury
   v3 board (section 4.2); per-pixel range restriction now leads.**

   **The ceiling was measured first, 2026-08-10, and the family is alive.**
   `article/range_ceiling.py`. Restricting the range PER PIXEL is a recorded
   negative with a known mechanism -- the filter aggregates over a plane, so a
   plane must hold one constant disparity -- so what is bounded here is the
   per-TILE range, which keeps the filter legal and is ESPReSSo's trick. Oracle
   intervals from ground truth over the 15 v3 training scenes, pooled over pixels,
   as a fraction of the full sweep:

   | tile | pad 0 | pad 2 | pad 4 | pad 8 |
   |---|---|---|---|---|
   | 8 | 5.8% | 10.5% | 15.2% | 24.6% |
   | **16** | 9.8% | 14.5% | **19.2%** | 28.7% |
   | 32 | 15.8% | 20.5% | 25.2% | 34.7% |
   | 64 | 24.5% | 29.3% | 34.0% | 43.4% |

   **19.2% at tile 16 with 4 planes of slack is 5.2x** — which independently
   reproduces the 5.2x this item has claimed all along from a completely different
   experiment (correct-over-known against delivered). Two numbers from two methods
   agreeing is the strongest evidence this project has for the size of this prize.

   Three things the table says that change how it should be built:

   - **Predictor accuracy is the whole game, not tile size.** At tile 16 the pad
     column runs 9.8% to 28.7%. Slack costs about 2.4 points of the sweep per
     plane, so a construction that predicts the interval to +-2 wins 6.9x and one
     good to +-8 wins 3.5x. Effort belongs in the predictor.
   - **It survives the worst scene.** ArtL is 28.5% (3.5x) and Vintage 7.0% (14x)
     at tile 16 pad 4. Nothing here is carried by an easy scene.
   - **It is not bimodal**, so this does not need a fallback scheme: at tile 16
     pad 4, 78.6% of pixels are in tiles needing under a quarter of the sweep and
     only 5.0% need half or more. A uniform narrow band is a legitimate design.

   **What this is not.** An oracle: it assumes the interval is known, which is
   exactly what a real construction must predict cheaply and imperfectly, and the
   pad column is the price of getting that wrong. It bounds ARITHMETIC, not wall
   clock -- the filter's recurrences and per-tile overhead do not scale with plane
   count -- and rule 5 says measure after, not just before. And the intervals come
   from ground truth, so tiles are bounded only where ground truth is known.

   **The half-resolution predictor was priced next, and it is not the one.**
   `article/range_predictor.py`, tile 16, same 15 scenes. `cost` is the fraction of
   the sweep, `recall` the fraction of pixels whose TRUE disparity falls inside its
   tile's band -- a band that excludes the answer does not cost time, it costs the
   answer:

   | pad | cost | recall |
   |---|---|---|
   | 0 | 15.2% | 55.6% |
   | 2 | 19.7% | 88.5% |
   | 4 | 24.1% | 91.2% |
   | 8 | 32.8% | 93.7% |
   | 16 | 49.3% | 96.5% |

   Against the oracle's **19.2% at 100%**, the coarse pass buys 19.7% at **88.5%**
   — the same arithmetic saving while throwing away one true disparity in nine. To
   reach a recall worth having it needs pad 16, and 49.3% of the sweep is 2.0x, not
   5.2x. Its bare interval contains the answer only **55.6%** of the time, and 5.6%
   of pixels sit in tiles with no coarse disparity at all and must sweep everything.

   **This explains the c2f result rather than contradicting it.** `--c2f` measured
   1.24x on the desktop and flat on the TX2, and the reading then was that its
   per-PLANE bounding boxes were too loose. Per-tile bands are much tighter and
   still do not pay, so the loose scheme was not the binding problem: **the coarse
   prior itself is not accurate enough.** Do not revisit c2f expecting the tiling
   to rescue it.

   **The bar for anything else is now a number.** A predictor must beat 24.1% cost
   at 91.2% recall, and to realise the 5.2x it has to reach roughly 19% cost at 99%
   recall — an interval good to about +-2 planes, nearly always containing the
   answer. That is a demanding target and it is the right one to test a candidate
   against before implementing it, since `range_predictor.py` only needs the
   candidate's predicted interval, not a working matcher built on it.

   **ICSG was priced next, and it fails here for a mechanism this project has now
   met three times.** `article/range_predictor.py --predictor icsg`. Tile 16, same
   15 scenes, feature = intensity plus scanline derivative, brute-forced over d
   because the question is whether the reduction is accurate enough to be worth
   indexing, and the index is the paper's contribution rather than the claim:

   | | per-pixel admitted | tile cost | recall |
   |---|---|---|---|
   | ICSG tau 8, tile band = min..max | **13.9%** | 87.5% | 96.4% |
   | ICSG tau 8, band = narrowest 95% of mass | 13.9% | 74.0% | 94.0% |
   | half-resolution pass | — | **24.1%** | 91.2% |
   | oracle | — | **19.2%** | **100%** |

   **The paper's claim reproduces and is useless to us.** 13.9% admitted is an 86%
   per-pixel reduction, right at the 83% claimed. But those admitted disparities are
   SCATTERED across the range, not contiguous, so the tile that has to sweep their
   union sweeps almost everything — 87.5%, and still 74.0% if the band is allowed to
   discard the outer 5% of the mass. The half-resolution pass, which is much worse
   per pixel, is three times better per tile because its errors are *local*.

   **Correction, same day: the interval cost model above was too strict, and the
   conclusion survives being corrected.** The constraint is only that a plane be
   constant-disparity across the tile, so a tile may legally compute a SCATTERED
   SET of planes — which is exactly ICSG's shape, and pricing it by `min..max` was
   unfair. Re-measured with cost = |union of admitted disparities over the tile|:

   | tile | ICSG tau 8, set model | recall |
   |---|---|---|
   | 16 | 79.3% (pad 0) | 91.0% |
   | 8 | 57.2% | 83.0% |
   | 4 | 43.6% | 77.3% |

   Still nowhere near 19.2%, and the mechanism is now sharper than "scattered":
   **the admitted sets of neighbouring pixels do not agree.** 13.9% per pixel, yet
   the union over even a 4x4 tile is 43.6% — and shrinking the tile to buy
   agreement costs recall (77.3%) and leaves 9.6% of pixels with no evidence at
   all. Any tiling scheme needs spatial agreement about which disparities matter,
   and this evidence has none.

   **The half-resolution pass, re-measured on the same fair terms**, improves to
   **20.9% at 90.7%** (pad 4) from 24.1% at 91.2%. That is worth restating: its
   cost is now essentially at the oracle's 19.2%, and the entire remaining deficit
   is **recall**. The problem was never that the coarse prior sweeps too much; it
   is that at a cost this low it misses about one true disparity in eleven.

   **So the target restates.** Not "find a cheaper band" but "find a band with the
   same cost and 99% recall". That is a statement about the predictor's tail rather
   than its typical case.

   **Where that tail actually is, measured** (`--diagnose`, tile 16, pad 4, official
   nonocc mask). Not where this project's priors would have put it:

   | bucket | share of pixels | miss rate | share of misses |
   |---|---|---|---|
   | disparity gradient <= 0.3 | 90.8% | 7.8% | **87.4%** |
   | gradient 0.3-0.6 | 2.7% | 3.6% | 1.2% |
   | gradient > 0.6 (discontinuity) | 6.4% | 14.3% | 11.3% |
   | **tile coarse coverage < 25%** | **6.5%** | **63.7%** | **51.4%** |
   | all nonocc | 100% | 8.1% | 100% |

   *(The occluded bucket is reported by the script but omitted here: its share is
   computed against the nonocc miss total and exceeds 100%, so it is not comparable
   with these rows.)*

   **Discontinuities are not the problem.** They miss at twice the base rate but
   hold 6.4% of the pixels, so they are 11.3% of the misses; 87.4% of misses are in
   smooth regions at a 7.8% rate. What IS concentrated is coverage: **6.5% of pixels
   sit in tiles the coarse pass barely saw, and they carry 51.4% of all misses.**

   **Acting on it works.** Sweeping in full any tile the predictor covers less than
   25% of, and pricing that properly:

   | scheme | cost | recall |
   |---|---|---|
   | uniform, pad 4 | 20.9% | 90.7% |
   | uniform, pad 8 | 30.9% | 93.5% |
   | **coverage fallback + pad 4** | **26.3%** | **93.5%** |
   | coverage fallback + pad 8 | 35.8% | 95.7% |

   At equal recall the adaptive scheme costs 26.3% against 30.9% — 15% less work
   for the same answer, because the slack is spent where the evidence is thin
   instead of everywhere. **3.8x at 93.5% recall** is the best this predictor
   family reaches.

   **And then the halo, which none of the above had paid for.** A tile can only
   skip a plane if it also skips it for the pixels the FILTER would have read. rf's
   influence decays ~0.89 per pixel at sigma_s 12, which is why `de_dense` already
   pads a masked rectangle by `MPAD` = 16 px before scoring it. So a 16x16 tile
   wanting one plane costs a 48x48 patch of it. Measured as the area of the dilated
   union per plane (neighbouring tiles share halos, so the naive `(1+2*MPAD/T)^2`
   overstates it), at the fallback-0.25 + pad-4 operating point:

   | tile | cost, no halo | **cost with halo** |
   |---|---|---|
   | **16** | 26.3% | **53.8%** |
   | 32 | — | 65.9% |
   | 64 | — | 81.5% |

   *(Corrected 2026-08-10: the first version of this table used a floor when
   converting the halo to tile units, so every tile wider than MPAD dilated by ZERO
   and was charged no halo at all. It read 34.4% at tile 32 and named that the
   optimum. Both the number and the direction were wrong. At tile 16 with MPAD 16
   the halo is exactly one tile, so 53.8% is exact; the wider rows are upper bounds,
   since a 16 px reach out of a 32-wide tile is charged a full neighbouring tile.)*

   **The halo more than halves the win, and bigger tiles are worse, not better** —
   the halo is paid per tile that wants a plane, so more tiles is not the fix and
   fewer tiles means looser bands. Best realisable is **53.8%, which is 1.9x**, not
   5.2x and not 3.8x.

   **Shrinking sigma_s does not rescue it.** The default moved from 12 to 8 the same
   day (2.2), which takes the halo from ~16 px to ~11. At tile 16 that is still a
   full neighbouring tile, so the table does not move at all.

   ### What the whole chain came to

   | | arithmetic |
   |---|---|
   | oracle interval, tile 16 | 19.2% = **5.2x** |
   | best real predictor, no halo | 26.3% = **3.8x** |
   | **the same with the filter halo paid** | **53.8% = 1.9x** |
   | and wall clock | worse again, unmeasured |
   | and accuracy at 93.5% recall | unknown, 0 to 6 points |

   **This is why `--c2f` measured 1.24x.** Not a loose scheme, not a bad prior: the
   aggregation has to be fed, and feeding it is most of the work the restriction was
   trying to avoid. Every step of this chain was a real effect and each one took a
   bite out of the prize.

   **Recommendation: do not build it on the GPU path.** 1.9x of arithmetic before
   implementation losses, for an unknown accuracy cost, against a pipeline that
   already closes 30 Hz at 96% of budget. The prize this item has carried since it
   was written is real in the oracle and mostly eaten by the time it is realisable.
   If it is ever revisited, tile 32 is the operating point and the halo is the thing
   to attack -- a cheaper aggregation with a shorter reach would change the whole
   table, which points back at 0.35's edge-aware support rather than at range
   restriction.

   **This is the constant-disparity-plane constraint for the third time.** It killed
   offset-indexed coarse-to-fine, it is why the range must be restricted per tile
   rather than per pixel at all, and now it converts a real 86% reduction into
   nothing. A scattered per-pixel candidate set is usable by a matcher that scores
   candidates individually — which is what ICSG's own SGM does — and not by one that
   aggregates over planes. Ours aggregates over planes, and that is where its
   accuracy comes from.

   **So the ranking changes.** ESPReSSo moves to the front: it is the only candidate
   that produces a tile-constant hypothesis *by construction* instead of deriving an
   interval from per-pixel evidence, which is exactly the failure above. Max-trees
   are untested and now suspect for the same reason — cheap to check, since
   `range_predictor.py` needs only their predicted interval.

   The constructions, re-ranked by what the measurements say:

   1. **ESPReSSo** -- PatchMatch hypotheses **shared across rectangular tiles**, so
      the plane is constant within a tile and a recursive edge-aware filter stays
      legal. Promoted 2026-08-10: tile-constant by construction is precisely what
      ICSG could not deliver.
   2. **MTS / MTS2 -- max-trees** (Brandt et al., PRL 2020 / arXiv:2006.15373): a
      hierarchical max-tree restricts the search range per region. MTS2 is the
      precision leader of the whole sparse table, 0.61% at ~36% coverage. Untested
      here, and suspect for ICSG's reason until its intervals are run through
      `range_predictor.py`.
   3. ~~**ICSG -- intrinsic curves**~~ (Shahbazi et al., ISPRS Congress 2016).
      **Measured negative, 2026-08-10**, above: the 83% per-pixel reduction is real
      and scattered, and a plane-aggregating matcher cannot spend it.
   4. **ELAS** -- sparse support points chosen by the first-to-second-minimum ratio,
      i.e. `Match::margin`, triangulated into a prior. **Speed only, not quality**: its
      own v3 entry is 26.4% bad-1.0 sparse at ~76% coverage, worse than SNCC at the
      same coverage. Do not expect the support-point prior to buy accuracy.

   See 09-matching.md. Any of these is a bigger prize than anything in the inner loops.
4. **Idle cores.** Desktop 3.27 of 4, TX2 2.9x of 6.

### Do not retry without new information

Coarse-to-fine with offset-indexed planes (aggregation needs constant-disparity planes),
band fusion on a cache-capacity argument, census re-read blocking, fusing the insert into
the filter, hoisting row pointers in the score loop, `--csct` at the current operating
point, vectorising popcount while staying pixel-major. All measured, all in 09-matching.md
with mechanisms.

**The desktop is no longer a proxy for the TX2.** int16 was neutral on the desktop and
worth 20% on the Jetson; `--csct` was worthless on the desktop and worth 10% on the
Jetson. Time anything new on the target before believing it. TX2 run-to-run variance is
37%, so use interleaved best-of-N there, never single runs:

```sh
tools/deploy.sh && tools/tx2_ab.py "" "--simd" -n 8
```

`tools/tx2_ab.py` interleaves, quotes minima, reports occupancy and spread, and refuses
to run against unlocked clocks or a binary older than your sources. See 07-tools.md.

**Temporal prior: prototyped and positive** (2026-08-09, see 09-matching.md): frame t's
disparities as the `--prior` mask for frame t+1 is 1.15x on the TX2 CPU tool with
HIGHER coverage and 99.6% agreement, on a static scene. Under motion it needs the IMU
rotation compensation of 3.1.

**And `--csct` need not be paid for SIMD.** ReS2tAC cuts the descriptor to 24 bits so
sixteen *pixels* fit three registers -- a pixel-major constraint that does not bind once
disparity is in the lanes. With the `vpaddq_u8` reduction a 24-bit descriptor would save
about two instructions of twelve. Not worth 1.2 points of bad-1.0.

## 0.35 Edge-aware support first, then slanted planes — measured, not assumed

Where dense MASDA stands: 126 ms, 86.1% coverage, 10.6% bad-1.0. SGM: 16 ms, 81.1%,
8.1%. At matched coverage, MASDA 8.6% against SGM 6.2%.

One change addresses the top item on both the quality list and the runtime list, and
is the only candidate that could put us *ahead* rather than level:

**Measured before committing to it** (see 09-matching.md): error is flat up to a
disparity gradient of 0.3 px/px and explodes beyond. High-gradient regions carry
~half of all error on both scenes — but above 0.6 px/px that is a *depth
discontinuity*, not a slanted surface, and the discontinuity population is 2-3x
larger than the genuinely slanted one. A warped window still straddles an occlusion
boundary.

**Re-measured on Middlebury v3, 2026-08-11** (`middeval3.py --by-gradient`, 15
scenes, official tolerance, error over filled pixels). The premise holds
directionally and its magnitude was overstated:

| bucket | pixels | error | share of error |
|---|---|---|---|
| slope <= 0.3 | 94.5% | 35.2% | 90.8% |
| slope 0.3-0.6 (genuinely slanted) | 3.2% | 54.0% | 4.8% |
| slope > 0.6 (discontinuity) | 2.3% | 70.6% | 4.4% |
| **within 2 px of a discontinuity** | 6.5% | 65.1% | 11.5% |
| **within 8 px of a discontinuity** | **18.3%** | **57.1%** | **28.6%** |
| further than 8 px | 81.7% | 32.0% | 71.4% |

**Not "about half", and not 9% either — the bucket has to be the right one.** By
slope alone, high-gradient pixels carry 9.2% of error and the item looks dead. But
edge-aware support does not help discontinuity pixels, it helps pixels whose
SUPPORT WINDOW straddles one, and by distance that population is 18.3% of pixels
carrying **28.6% of all error at 57.1% against a 32.0% far-field rate.**

**So the ceiling is about 4.6 points.** Bringing the near-edge population down to
the far-field rate takes 36.6% -> 32.0% overall. That is optimistic — it assumes
perfect edge handling — and it is the best-founded target left on this list, well
above what the remaining knobs offer and well below what sub-pixel gave.

**But the 4.6 points are probably NOT reachable by a better support region**
(measured 2026-08-11, and this is the finding that matters here). Two independent
knobs that should move near-edge error both leave it flat:

| | near edge (<=8 px) | far field | all |
|---|---|---|---|
| sigma_r 0.10 (sharpest edge response) | 57.8% | 33.3% | 37.8% |
| **sigma_r 0.20 (default)** | **57.1%** | 32.0% | 36.6% |
| sigma_r 0.40 (softest) | 57.8% | 31.7% | 36.5% |
| `--topk 8` | 57.0% | 31.0% | 35.7% |
| **`--topk 0`, no pruning at all** | **56.4%** | 30.6% | 35.3% |

**Edge sensitivity does nothing to it.** A 4x range of `sigma_r` moves the far field
(33.3 -> 31.7) and the near-edge population not at all. The aggregation is already
edge-aware -- `rf` is a recursive edge-aware filter, not a box, so 0.35's "edge-aware
support instead of a box" was in fact already done -- and making it more or less
edge-aware does not touch the population it was supposed to help.

**Candidate pruning does nothing to it either.** Handing the solver the entire
disparity range instead of two candidates leaves near-edge error at 56.4%.

**What that leaves is half-occlusion**, and no weighting scheme fixes it: for a pixel
beside a foreground edge, part of its support has no correspondence in the other
view at all. The choice of which neighbours to trust cannot help when the correct
ones are not visible. The smooth decay with distance -- 65.1% within 2 px, 57.1%
within 8, 32.0% beyond -- is the shape of window contamination whose reach matches
the aggregation's, which is consistent with that reading.

~~**So the recommendation changes: build a left-right consistency check.**~~
**Tried 2026-08-11, and the premise was wrong too.** A consistency check *removes*
matches, so it cannot close a coverage gap; the question was where our 19.9% of
unmatched pixels actually goes. Two candidates, and the split is lopsided:

| sink | coverage cost |
|---|---|
| **the margin gate at 0.01** | **11.1 points** (91.2% -> 80.1%) |
| greedy claim contention | 0.2 points |

**Contention is not a sink.** Retrying the other scored candidate when a pixel's
right partner is already claimed -- free, no new cost, no new candidates -- was
implemented and measured: coverage 80.1% -> 80.3%, bad-1.0 26.04 -> 26.27. It
barely fires, and what it recovers it pays for. **Reverted rather than left in as a
flag**, per this file's own preference for a recorded negative over a knob.

**So our coverage is not lost, it is spent.** 98% of it goes to the gate, which is a
chosen position on the precision-coverage curve rather than a failure to fix. At the
gate wide open we reach 91.2% coverage -- past SGM's 90.2% -- at 35.85 bad against
its 29.08. Which is the same conclusion the sweep reached from the other direction:
**SGM's whole curve is better, and no knob on our side moves off ours.**

That closes the coverage line. What is left is structural, and 0.4's semi-dense
candidate work is the only remaining item that changes the curve rather than the
position on it.

**The slanted-plane half is now clearly the smaller prize**: genuinely slanted
surfaces (0.3-0.6) are 3.2% of pixels and 4.8% of error. The order 0.35 already
argues for — edge-aware support first — is confirmed, by a wider margin than it
claimed.

*(A first version of this table divided the gradient by the resolution scale factor
and put 98.4% of pixels in the flattest bucket, which would have retired this item
outright. Disparity gradient is scale-invariant: at Q the disparities are a quarter
and one pixel spans four.)*

So: **edge-aware support first** (AGAP, doi:10.1049/iet-ipr.2018.5801), slanted
planes second (PatchMatch, doi:10.5244/C.25.14). That is the reverse of the order
the papers suggested, and the reversal came from the data.

- *Quality.* Our window aggregation assumes constant disparity across the window,
  which is wrong on every slanted surface and is why the aggregation-radius sweep
  plateaued at 3-4. **SGM shares this fronto-parallel bias**, so fixing it is a place
  SGM is weak rather than a way to catch up.
- *Runtime.* The cost volume is 40 MB, is 71 of 126 ms, and the work is
  memory-bandwidth bound (8 threads is slower than 4). Propagation never materialises
  it.
- *Fit.* Everything measured here says the candidate set decides the outcome, and
  PatchMatch is a cheap candidate generator. MASDA supplies what it lacks: a
  one-to-one constraint and a confidence measure. Neither paper does that
  combination.

Use red-black / checkerboard propagation, not scanline order, so it stays parallel.

Second tier, in order:
1. **Graded cost.** Census gives 49 Hamming levels; SGM's Birchfield-Tomasi
   difference is continuous. Combining them is standard and untried here.
2. **uint16 cost volume** instead of float32. Bandwidth-bound, so quartering the
   bytes is worth more than vectorising the arithmetic.
3. **Edge-aware support** instead of a box (AGAP, doi:10.1049/iet-ipr.2018.5801).
   The box was worth 16 points of bad-1.0; a support region that follows image
   structure should be worth more, and MST filtering is O(N).

## 0.36 The error budget: where a far-field pixel is actually lost — 2026-08-11

`article/cost_ceiling.py`. 0.35 established that 71% of error sits in the far field
(nowhere near a discontinuity) at a 32% rate, and that neither edge sensitivity nor
candidate count moves it. This decomposes that number by dumping the aggregated cost
volume and asking where the true disparity ranks. All 15 v3 scenes, 2.35M far-field
pixels, tolerance 0.5 px (a candidate that far out is the nearest integer, so the
sub-pixel fit can still reach the answer):

| a candidate within 0.5 px | top-1 | top-2 | top-4 | top-8 |
|---|---|---|---|---|
| all pixels | 64.1% | 78.8% | 85.3% | 89.6% |
| near edge | 58.8% | 75.7% | 84.6% | 90.1% |
| **far field** | **67.9%** | **81.0%** | 85.8% | 89.2% |

Stage by stage, far field, as a share of all far-field pixels:

| stage | share | lost here |
|---|---|---|
| the cost's argmax is within 0.5 px | 67.9% | |
| ... and the matcher answered | 65.8% | 2.1% to the gate |
| ... and the output is within 0.5 px | 64.6% | 1.2% |
| ... and within **0.25 px**, the official tolerance | **56.1%** | **8.5% to the fit** |

**The bill, itemised:**

| where the pixel is lost | share of far-field pixels |
|---|---|
| **no candidate within 0.5 px even in the top-8** — the descriptor | **10.8%** |
| in the top-8 but not the top-2 — the pruning | 8.2% |
| in the top-2 but the top-1 is wrong — **the selector** | **13.1%** |
| the gate declined | 2.1% |
| the sub-pixel fit missed 0.25 px | 8.5% |
| correct | 56.1% |

**Three things this settles.**

1. **The descriptor is the largest single item** — 10.8% unreachable by any solver on
   this cost volume, and 19.0% unreachable without widening the candidate set past
   two. That confirms what 0.35 pointed at and prices it.
2. **There is 13.1% of selector headroom in the far field**, and neither
   winner-take-all nor MASDA's message passing captures it (0.31 measured them within
   a point of each other). The inference is leaving real points on the table; more
   iterations are not how they get collected.
3. **The sub-pixel fit costs 8.5%** on pixels whose integer was already right and
   which were answered. A better estimator than a three-point parabola on a
   Census+AD cost — the classic pixel-locking bias — is a bounded, self-contained
   change with a measured prize.

**Read Teddy alone and you get the wrong answer.** Its cost is far better than
average (top-1 92.8% in the far field), so refinement looks dominant there at 13.1
points against a 2.8-point descriptor bill. The 15-scene number inverts that ordering.
One scene is not a benchmark, and this is the cleanest demonstration of it in the
project.

*(The volume has to be materialised for this, which disables the blockwise path, so
the costs are float rather than int16. Third decimal, not the conclusion.)*

### The fit's 8.5% — partly collected, 2026-08-11

The budget above puts 8.5% of far-field pixels on the sub-pixel fit: their integer was
right, they were answered, and the refined value still missed the quarter-pixel
tolerance. 2.2a already identified the mechanism — the graded cost's truncated
absolute difference is piecewise LINEAR, so the surface around the winner is locally
a V rather than a parabola, and a parabola is the wrong estimator for a V.

The expensive fix is to fit on the Census term alone, which needs a second filtered
plane. The cheap one is to change the estimator, which is the same arithmetic:

| | bad-1.0 | coverage |
|---|---|---|
| parabola (was) | 25.18 | 79.6% |
| **equiangular (now)** | **24.47** | 79.6% |
| equiangular, `--ad 0` | **23.75** | 79.8% |
| parabola, `--ad 0` | 24.42 | 79.8% |

**0.71 points for one line and no extra memory**, at coverage that does not move — the
estimator changes values, never decisions. Default since 2026-08-11;
`--fit-parabola` restores the old one. CPU-to-GPU bit-identity re-verified on all
eight scenes.

**The eight-scene native benchmark measures 9.2% either way**, which is not a
disagreement: a one-pixel tolerance on the native grid cannot resolve a sub-pixel
estimator at all. This is the same blindness that kept sub-pixel disabled for months
(2.2), showing up a second time on the same benchmark.

**The AD term still costs 0.72 even with the right estimator** (24.47 against 23.75),
so the graded cost's interaction with the fit is not merely an estimator mismatch and
the Census-only fit remains the open item. `--ad` stays at 0.15, because it is worth
0.3 points the other way on the resolution the camera actually runs at.

*(Found while measuring this: `dense_bench.py` had the same override bug
`middeval3.py` had — it passed `--iters` with a value of its own, so every number it
has ever produced was at message-passing 2 rather than the shipping default. Fixed the
same way. The 8-scene figures moved 9.6% -> 9.2% as a result.)*

### Screening the descriptor before rebuilding it — 2026-08-11

The 10.8% is the descriptor's bill, so the obvious move is a bigger descriptor: the
Census window is a template and 9x7 is 62 bits, which still fits a `uint64`. Before
touching ~15 border literals in the cost loops and the CUDA census kernel, the
direction was priced on the axis that is already wired.

**Descriptor size moves along the curve, not off it.** `--csct` is the same Census at
24 bits instead of 48:

| | bad-1.0 | coverage |
|---|---|---|
| 7x7 Census, 48 bit (ships) | 25.18 | 79.6% |
| centre-symmetric, 24 bit | **24.68** | 75.9% |

Halving the descriptor *lowers* the error rate by 0.5 and costs 3.7 points of
coverage — which is the precision-coverage trade every other knob in this project
offers, at the same sort of exchange rate. It is not a better or worse descriptor, it
is a differently-gated one.

**So 9x7 is not worth building.** If 24 fewer bits only slides along the curve, 14
more will too, and the change costs the border literals in both tools plus a new CUDA
census kernel plus a re-verification of bit-identity.

**And that sharpens what the 10.8% actually is.** It is not a *bits* problem, so more
of the same descriptor will not collect it. It is a problem with Census-plus-truncated-
difference as a similarity measure on weakly textured and repetitive surfaces. Fixing
that means a different similarity function — which is exactly where learned costs sit,
and what the no-CNN decision (section 0) rules out. The remaining lever and the
ruled-out technique are the same thing, and that is worth stating plainly rather than
leaving as an open item that reads as actionable.

*(Also recorded because it wasted a run: `--csct` refuses to run with `--dump-vol`,
correctly -- only the blockwise path builds the 24-bit descriptors -- so the candidate
recall above cannot be measured for it. And **`--agg` is inert on the default path**:
3, 7 and the default all produce identical numbers, because the recursive filter
ignores it. The CUDA tool's usage already said so; the CPU tool's did not.)*

## 0.4 Semi-dense candidates — **OBSOLETED 2026-08-11: read the dense map instead**

**The whole item was written when dense MASDA did not run in real time.** It does
now, 31.3 Hz at 848x480, so a keypoint does not need its own matcher at all — it can
read its disparity out of the dense map. `de_bench --dense` samples that map at the
same left keypoints and scores it with the same detector and the same rules, so the
two are comparable rather than merely similar:

| at MASDA's own keypoints, 8 scenes | correct | precision | median err | \|e\|<=0.5 |
|---|---|---|---|---|
| **the shipping sparse matcher** | 1402 | 0.706 | 0.195 | 87.1% |
| dense map, gate 0.01 | **2365** | 0.829 | 0.150 | 93.6% |
| dense map, gate 0.03 | 2196 | **0.853** | 0.145 | 94.5% |
| dense map, gate 0.08 | 1736 | **0.902** | 0.139 | 96.6% |
| SGM, this item's target | — | 0.858 | — | — |
| this item's own proposal | — | 0.882 at gate 0.15 | — | **68 ms** |

**Better on every axis, and the cost problem does not exist.** At gate 0.03 the dense
map matches SGM's precision with 57% more correct matches than the sparse matcher; at
0.08 it beats both SGM and this item's own 0.882, still with 24% more correct matches
than the baseline. The 68 ms this item set out to remove is not spent: the map is
already computed for the dense pipeline.

*(The recall column is not comparable and `de_bench` says so. The sparse matcher's
recall denominator is left keypoints whose partner the RIGHT detector also found,
which is the 44-51% repeatability ceiling; the dense map has no such requirement, so
its recall reads above 1. Precision, median error and correct-count are like for
like.)*

**BUILT AND MEASURED 2026-08-11: the keypoints are free.** `de_dense_cuda
--keypoints out.csv` detects on the left image and reads each keypoint's disparity
out of the dense map. Detection runs as a third thread beside the solve inside the
pipelined loop, so the number includes it:

| 848x480 D=60, pipelined over 30 frames | steady state |
|---|---|
| dense map only | 31.4 ms (31.8 Hz) |
| **dense map + the sparse feature set** | **31.8-32.8 ms (30.5-31.4 Hz)** |

Detection is ~29 ms of one core and costs **0.4-1.4 ms of wall clock**, because it
hides under the 26 ms of kernels exactly as the overlap argument predicted -- and it
had to be measured inside the loop to say so, since bolting it on afterwards timed
nothing. 97.3% of detected keypoints carry a disparity. One producer, both products,
still real time.

*(The `--iters 0` default freed ~12 ms of the CPU solve, which is what made room for
this. The two decisions were taken together and only make sense together.)*

**The original framing, kept: it was a budget decision, not plumbing.** `de_pipe` is the
live consumer, and pointing it at the dense map means extracting the dense pipeline
into something shareable and deciding whether the live path becomes a CUDA tool.
That is section 0's "one producer" question, and the arithmetic is tight enough that
it has to be decided rather than assumed:

| | CPU | GPU |
|---|---|---|
| today's live path: preprocessing 26.54 + MASDA 5.14 | **31.7 ms** | 0 |
| dense CUDA path | 23 ms solve | 26 ms kernels, 31.9 ms wall |
| **keypoints AND a dense map** | detection ~21 + solve 23 = **44 ms** | 26 ms |

**The swap is not free even though the dense map is.** Sampling the map costs
nothing, but anything that still wants KEYPOINTS -- tracking, VO -- needs the
detector as well, and detection plus the dense solve is ~44 ms of CPU against a
33.3 ms frame. It fits only if the detector overlaps the solve across the A57s and
the Denvers, which is exactly the stage-to-core pinning section 0 left open and
nobody has measured.

Three ways out, in the order they should be tried: run correspondence at 15 Hz,
which the frame-budget section already argues is plausibly enough indoors; or drop
the sparse detector entirely for consumers that only want depth, since the dense map
serves them without keypoints; or raise `fast_threshold`, which the profiling says
buys 30% of preprocessing for 10% of matches.

Whatever is chosen, pick the gate per consumer rather than globally — tracking wants
coverage, triangulation wants precision, and the table above shows they are 0.829
and 0.902 on the same map.

**What this does NOT retire**: MASDA for frame-to-frame association (section 4). That
is temporal matching, where there is no dense map of the *motion* to read, and it is
still the interesting use of the machinery.

### The original item, kept for the reasoning

Dense SGM beats MASDA **at MASDA's own keypoints**, 0.858 against 0.616 precision,
pooled over the eight Middlebury scenes, while also filling 78% of the image in
17 ms (`article/dense_baseline.py`).

**Measured before building it, and the answer bounds the work.** A perfect
re-ranking of MASDA's existing candidates tops out at **0.697**, because in 30.3% of
its errors the correct right keypoint was never a candidate. Denser right detection
does not lift that ceiling (flat at 0.68-0.70 from 555 to 2994 right keypoints).
And a semi-dense variant that offers every right pixel — so the answer is always
available — gets *worse*, 0.587, because Census plus uniqueness cannot pick the
right one out of 71 candidates.

A smoothness factor is worth roughly +8 points on the current candidate set, not
+24. But smoothness turned out not to be the route anyway: **semi-dense candidates
plus the existing margin gate reach SGM's precision without it** — 0.840 at gate
0.10, 0.882 at 0.15, against SGM's 0.858, and +459 correct with +0.147 precision
over the sparse baseline at gate 0.05.

**The open problem is cost, not quality.** 68 ms per 450x375 scene for the solver,
three times the Jetson budget at a fifth of the pixels. The gate discards most of
the 71 candidates per keypoint, so generating fewer and better candidates — a
coarse-to-fine pass, or a disparity prior from the previous frame — is the obvious
avenue and has not been tried. See 09-matching.md.

Uniqueness and smoothness are orthogonal — SGM has no uniqueness constraint and
needs a left-right check bolted on to reach 78% coverage. A factor graph with both
is the interesting object, and adding a smoothness factor is the same kind of
derivation the ordering factor already went through in section 7 of the article.

The earlier two-pass smoothness attempt failed because a prior fitted from the
matches it then judges is self-confirming. Path aggregation over the whole field is
a different thing, and `dense_baseline.py` is now the yardstick.

---

## 0.45 The real camera is not the benchmark — measured 2026-08-11

Everything in this matcher was tuned on Middlebury. The live view looks noisier and
holier than the published figures, and that impression is correct. Measured on the
same statistics, 7x7 local standard deviation and the fraction of a 32x32 tile's
Census descriptors that are distinct:

| | contrast (median) | pixels < 2 DN | bright-clipped | descriptor uniqueness |
|---|---|---|---|---|
| **D435 IR, projector ON** | **3.9 DN** | **24.9%** | **6.4%** | **81.7%** |
| D435 IR, projector OFF | 1.6 DN | 56.8% | 6.3% | 82.1% |
| Middlebury Teddy | 8.5 | 13.8% | 0.0% | 94.1% |
| Middlebury Motorcycle | 12.2 | 8.0% | 0.2% | 90.7% |
| Middlebury Piano | 5.1 | 25.7% | 3.6% | 86.0% |
| Middlebury Shelves | 3.5 | 37.4% | 0.0% | 84.6% |

**The real scene is not off the scale — it sits with the hardest v3 scenes.** But it
is 2-3x lower contrast than Teddy and Motorcycle, and Teddy and Cones are exactly the
two scenes every quick check in this project runs on. The descriptor uniqueness is
below all four, which is the 3.3x degeneracy already recorded for the projector.

**One number is specific to this camera and is not a matcher problem: 6.4% of the
frame is bright-clipped**, against ~0% on Middlebury, and it is identical with the
projector off — so it is ambient light, a window or a monitor, at `exposure_us 1500`
and `gain 64`. Saturated pixels carry no information at any descriptor size. Lowering
exposure would recover them and would cost dot contrast everywhere else, which is a
trade nobody has measured.

**The real gap is that none of this can be tuned on the real camera, because there is
no ground truth for it.** Every parameter -- sigma_s, ad, the margin gate, the fit --
was chosen against Middlebury and assumed to transfer. Three ways to get a number on
the real camera, cheapest first:

1. **A flat wall and a plane fit.** No equipment: point the camera at a wall, fit a
   plane to the cloud, and report the residual scatter and the outlier fraction. That
   is a noise measurement on the real sensor, available today, and it makes parameter
   sweeps possible on real data. It cannot measure absolute scale, which is fine --
   scale is what 1.1's rangefinder is for and it is a different question.
2. **The D4 ASIC's own depth** (0.5). Needs `rs_ir_capture` to record the `Depth`
   stream alongside the IR pair. Not ground truth, but an independent measurement
   from the same photons, and the ASIC scores 2.76% bad-1.0 on the v3 sparse table.
3. **The rangefinder** (1.1), for absolute accuracy at known distances.

Until one of those exists, every claim about this matcher on this camera is an
extrapolation from Middlebury, and the size of the extrapolation is the table above.

**Route 1 is built: `desktop/wall_check.py`.** It fits a plane to the cloud and
reports the scatter about it, which is the matcher's noise at that distance, plus the
gross-outlier fraction. `--sweep` compares flag settings on the real sensor, which is
the thing that was impossible before. It needs a bag of a blank wall; run against a
non-planar scene it reports 41 degrees of tilt and 144 mm RMS, which is the correct
answer to "this is not a wall" and means the metric cannot be fooled into looking good.

### Exposure measured on a real wall — 1500 us is too long, and contrast is not the reason

First use of `wall_check.py`, and the first parameter this project has ever tuned on
its own sensor. A blank wall at 0.38 m, `--emitter on`, seven exposures, everything
else at the shipping defaults:

| exposure | points | RMS about the plane | >2 cm | contrast | descriptor uniqueness |
|---|---|---|---|---|---|
| 150 us | 90.3% | 1.1 mm | 0.1% | 1.6 DN | 55.7% |
| 250 us | **90.5%** | **1.1 mm** | **0.1%** | 3.5 DN | 64.6% |
| 350 us | **90.5%** | **1.1 mm** | **0.1%** | 5.0 DN | 68.7% |
| 500 us | 90.2% | 1.1 mm | 0.1% | 7.1 DN | 73.2% |
| 900 us | 89.5% | 1.2 mm | 0.2% | 13.0 DN | 78.3% |
| **1500 us (shipping)** | 88.0% | 1.2 mm | **0.8%** | 21.4 DN | 81.7% |
| 2500 us | 86.1% | 1.3 mm | **2.0%** | 37.0 DN | 84.2% |

**Shorter is better on every matcher axis**, and the optimum is a broad plateau from
150 to 500 us. Against the shipping 1500 us, 350 us buys 2.5 points of coverage and
takes gross outliers from 0.8% to 0.1% -- eight times fewer points more than 2 cm off
a flat wall.

**And the mechanism is not saturation.** Nothing clips below 900 us (0.00% above 250
DN) and only 0.14% clips at 2500. The 6.4% clipping recorded in 0.45 was that desk
scene's window, not a property of the exposure setting.

**The two quantities that usually predict matcher quality both move the WRONG way.**
Local contrast rises 1.6 -> 37 DN across this sweep and descriptor uniqueness rises
55.7% -> 84.2%, while coverage, noise and outliers all get worse. Whatever is
happening, "more contrast, better descriptors" does not describe it. **Recorded as
unexplained.** The candidate worth testing is that a brighter wall surface makes the
non-dot background matchable-looking, adding plausible wrong candidates that the dots
alone would not have produced -- but that is a story, and this file's rule is that a
story without a measurement is not a finding.

**The camera's own auto-exposure is the worst setting measured, and that answers the
question a fixed exposure raises.** Lighting changes from room to room, so a constant
1500 us cannot be right everywhere -- but `--auto-exposure` is not the fix. It targets
a well-exposed *picture*, mean 94.6 DN, and lands below every fixed setting tried:

| | image mean | points | RMS | >2 cm |
|---|---|---|---|---|
| 350 us fixed | 21.5 DN | **90.5%** | **1.1 mm** | **0.1%** |
| 1500 us (shipping) | 41.1 DN | 88.0% | 1.2 mm | 0.8% |
| **auto-exposure** | **94.6 DN** | **84.3%** | 1.4 mm | **3.4%** |

**Image mean predicts matcher quality almost perfectly, and negatively**: r = **-0.98**
against coverage and **+0.99** against gross outliers, over eight settings spanning
17.8 to 94.6 DN. The matcher wants a *dark* image, around 20 DN, and every step
towards a photographically correct one costs it.

~~**So the exposure controller should target the mean**: raise exposure until the
mean reaches ~20 DN and stop.~~ **Wrong, and a second room says so.** That rule came
from one scene and does not survive contact with another.

### A second room, and the optimum moves by 10x

Same sweep, a different room with window, fridge and floor reflections, and a general
scene rather than a wall at arm's length:

| exposure | mean | contrast | answered | roughness |
|---|---|---|---|---|
| 350 us (room 1's optimum) | 18.6 DN | 0.5 DN | **79.0%** | 0.14 px |
| 900 us | 22.3 | 1.0 | 86.5% | 0.09 |
| 1500 us | 26.4 | 1.5 | 88.0% | 0.08 |
| **2500 us** | 33.2 | 2.5 | **88.5%** | **0.08** |
| **4000 us** | 43.7 | 4.0 | **88.6%** | 0.09 |
| 6000 us | 57.3 | 5.9 | 88.4% | 0.11 |
| auto-exposure | 79.3 | 9.8 | 88.3% | 0.12 |

**Room 1's optimum starves this room**: 350 us answers 79.0% against 88.6%, nearly ten
points, and at 150 us the roughness is 2.82 px -- the matcher is guessing. The
projector's return falls with distance and this scene is metres away rather than
centimetres, so the same exposure delivers a fifth of the dot contrast.

**What transfers is the contrast, not the exposure and not the mean:**

| | room 1 (wall, 0.38 m) | room 2 (general scene) |
|---|---|---|
| exposure at the optimum | 250-350 us | 2500-4000 us — **10x** |
| image mean there | 20-22 DN | 33-44 DN — does not transfer |
| **local contrast there** | **3.5-5.0 DN** | **2.5-4.0 DN** — overlaps |

**So the controller targets local contrast: raise exposure until the median 7x7
standard deviation reaches ~3-5 DN.** That is one number that held across scenes whose
correct exposures differ by an order of magnitude, and it is what makes the setting
portable between rooms.

### Third condition: lights off, and the rule holds

Same room, room lighting switched off, so the only illumination is the projector:

| exposure | mean | contrast | answered | roughness |
|---|---|---|---|---|
| 350 us | 16.2 DN | 0.3 DN | 72.1% | 0.17 px |
| 1500 us | 17.5 | 0.9 | 85.5% | 0.09 |
| **4000 us** | 21.0 | **2.5** | **88.2%** | **0.08** |
| **6000 us** | 23.7 | **3.7** | **88.2%** | 0.09 |
| 10000 us | 28.8 | 5.9 | 87.6% | 0.11 |
| auto-exposure | 79.1 | 31.6 | 85.4% | 0.24 |
| 4000 us, **projector off too** | 16.0 | 0.1 | **39.3%** | **31.0 px** |

Three conditions now, with optimal exposures spanning **250 to 6000 us — 24x** — and
the optimal contrast is 3.5-5.0, 2.5-4.0 and 2.5-3.7 DN. The rule transfers; the
exposure does not, and the image mean does not.

**The last row is the control experiment.** With the projector off and the lights off,
coverage collapses to 39.3% and the roughness is 31 px. Whatever this matcher is
doing, it is doing it on the dots.

### BUILT: `rs_ir_stream --auto-contrast C`

Measures the median standard deviation over a grid of 8x8 patches -- r = 0.9997
against the full 7x7 metric over eighteen captures spanning 0.4 to 50.6 DN, at a
quarter of the reads -- and moves the exposure to hold it at C.

Contrast is linear in exposure (R^2 = 0.9995 in two rooms whose slopes differ 15x), so
`exposure *= target / measured` lands in one step rather than converging towards it;
it is damped to half anyway, because a frame captured mid-change would otherwise start
an oscillation. Measured live in the unlit room, starting from 1500 us: settles at
**3448 us holding 3.5 DN** and stays there.

| | contrast | answered | roughness |
|---|---|---|---|
| fixed 4000 (this room's best) | 2.5 DN | 88.2% | 0.08 px |
| **`--auto-contrast 3.5` -> 3448 us** | 2.2 | **88.0%** | **0.08** |
| the camera's auto-exposure | 31.6 | 85.4% | 0.24 |

Within 0.2 points of the hand-tuned optimum, without being told anything about the
room.

### Would an external infrared light help? Measured: no

Room lighting IS a flood illuminator, so lights-on against lights-off at matched
exposure answers it directly:

| exposure | lights on: contrast / answered | lights off: contrast / answered |
|---|---|---|
| 1500 us | 1.5 DN / 88.0% | 0.9 DN / 85.5% |
| 4000 us | 4.0 / **88.6%** | 2.5 / **88.2%** |
| 6000 us | 5.9 / 88.4% | 3.7 / 88.2% |

At matched exposure ambient light helps, because it raises contrast. **At each
condition's own optimum it is worth 0.4 points** -- 88.6% against 88.2%. Turning the
room lights off costs almost nothing once the exposure follows, and a flood
illuminator is the same intervention bought rather than switched on.

**What the light would have to be is a PATTERN projector, not a flood.** The control
row above -- projector off, 39.3% and 31 px -- is the whole argument: this matcher
lives on projected structure, and unstructured photons mostly let you shorten the
exposure. A second dot projector would add structure; it would also need a pattern
that does not alias with the D435's own, or it adds ambiguity rather than removing it.

**On safety, if one is bought anyway.** 850 nm is invisible, so the blink reflex does
not protect anyone from it, and the relevant standard is IEC 62471 -- look for
Exempt Group, or Class 1 under IEC 60825 for anything laser-based. The D435's own
projector is Class 1. Power ratings on illuminators sold for CCTV are frequently far
above that, and they are pointed at corridors rather than at people at desk distance.

**Auto-exposure is not a substitute, and its failure is asymmetric.** In room 2 it is
within 0.3 points of the best (88.3% against 88.6%); in room 1 it was the worst
setting measured, 84.3% against 90.5%, with 34x the gross outliers. It happens to be
adequate when the scene is far and dim and badly wrong when it is near and bright,
which is not a property to rely on.

**The reflections did not matter.** Window, fridge and floor clip 0.00-0.18% of the
frame across the whole sweep, so specular highlights were not the problem here --
distance was.

*(An earlier reading of this experiment was wrong and is recorded because the mistake
is easy to repeat: `rs_probe` reported Exposure=350 after the auto-exposure capture,
which looked like AE choosing the optimum. That reading came from a SEPARATE probe
session under different light, not from the capture. The recorded frames are the
evidence -- mean 94.6 DN -- and they say the opposite. Read the data the run wrote,
not the device's state afterwards.)*

**The caveat that bounds all of it: one wall, at 0.38 m.** The projector's return
falls with distance, so the optimum almost certainly rises with range, and a setting
tuned at arm's length may starve a 3 m scene. Repeating this at two or three
distances is the obvious next use of the tool, and until then 1500 us should not be
changed by default -- only known to be wrong for close work.

**For reference, 1.1 mm RMS at 0.38 m is about 0.16 px of disparity noise**, which is
the first number this project has that describes the matcher on its own camera.

### Raising local contrast does not help, and the reason is structural

The obvious response to 3.9 DN of contrast is to amplify it. Measured on the real
pair, three ways:

| | contrast | pixels < 2 DN | answered | roughness (p90) |
|---|---|---|---|---|
| raw | 3.9 DN | 24.9% | 88.4% | 0.08 px |
| global stretch | 4.2 DN | 22.7% | 88.3% | 0.09 px |
| CLAHE 8x8 | 10.0 DN | 7.5% | **85.0%** | **0.19 px** |
| average of 4 frames | — | — | 88.5% | **0.06 px** |

**Any monotonic normalisation is invisible to the matcher.** Census compares a pixel
against its neighbours, so a global stretch leaves the descriptor **byte-identical** --
measured, 100.0% of descriptors unchanged. The contrast number moves and nothing else
does. This is a property of the descriptor, not a coincidence of this scene.

**CLAHE does change the descriptor, and makes things worse**: 3.4 points fewer pixels
answered and 2.4x the roughness. It applies a different map per tile, and the same
scene point sits at *different pixel coordinates* in the two views -- that is what
disparity is -- so it lands in different tiles under different maps. The comparison
across views stops being like with like.

**Temporal averaging helps slightly, and that is the informative part.** Four frames
of a static scene take roughness from 0.08 to 0.06 px and coverage nowhere. The
per-pixel temporal noise is **0.85 DN against 3.9 DN of contrast**, so the sensor is
not noise-limited and averaging has little to remove.

**So the limit is the absence of texture, not its amplitude and not sensor noise.**
No intensity transform creates information that the photons did not carry. The two
things that do are optical: the projector, which is worth 1.6 -> 3.9 DN and 56.8% ->
24.9% of pixels below 2 DN, and exposure -- 6.4% of the frame is bright-clipped at
`exposure_us 1500`, and those pixels are unmatchable at any descriptor size. Whether
lowering exposure pays is a trade against dot contrast everywhere else, and
`wall_check.py --sweep` over a few exposures is now the way to settle it.

## 0.55 Per-point confidence, and then an existence probability — **NEXT, 2026-08-12**

The cloud has no notion of how much any single point is worth, and everything queued
behind this wants one. Temporal fusion should weight a point rather than hold a hard
vote; an occupancy grid consumes exactly a probability; and 24a's ghost sheet is a
population that any usable confidence must be able to name.

**We already have one, and it is measured insufficient.** The score margin
(best-minus-second) is the *peak ratio* family under its usual name, and against the
occlusion sheet `--min-margin 0.05` removed a fifth of the bad points and four fifths
of the good ones. The mechanism generalises and is worth stating once: **best minus
second is only meaningful over the candidates actually searched.** Where the true match
is off-sensor or outside `dmax`, the wrong match has no competitor and therefore scores
as confident. Anything that reads only the cost curve inherits that blind spot.

Two cues are already in the machine and unread. MASDA's one-to-one constraint is
*uniqueness* in the standard taxonomy, and the top-2 scores are the peak ratio's
inputs.

**The literature.** Hu and Mordohai (TPAMI 2012) is the founding survey, 17 measures in
6 categories, and it fixed the metric everyone still uses: AUC of the sparsification
curve, against the AUC an oracle ordering would give. Poggi, Tosi and Mattoccia
(TPAMI 2021, arXiv 2101.00431) re-evaluate 52 measures on five datasets. The finding
that matters here is that **local aggregation is what makes a measure good** — APKR,
the peak ratio pooled over a neighbourhood, ranks in the top 4 hand-crafted measures
while the per-pixel PKR does poorly, and left-right-consistency measures rank mid-table
alone. Learned confidence dominates overall and is out on the no-CNN decision of 0; a
logistic regression over four features is not a CNN and is the standard way to combine
cues.

**Plan, in the order it should be done.** 1 and 2 are offline on Middlebury and need
no vehicle.

1. ~~**APKR.**~~ **Done 2026-08-12, and it decides that the rest is worth doing.**
   See 0.551 below for the numbers, the harness and two results that were not
   expected.
2. ~~**Uniqueness / left-right consistency**~~ **Done 2026-08-12: it is a veto, not a
   ranking, and it is the cue that sees the ghost.** See 0.552.
3. ~~**Combine** with a logistic regression fitted on Middlebury.~~ **Done
   2026-08-12: it buys calibration, not ranking.** See 0.553.
4. ~~**Calibrate**~~ **Done offline 2026-08-12, and it is calibrated on a MIX of
   scenes and over-confident on a hard one.** See 0.553. What remains is the part
   that needs this camera: a calibration fitted on Middlebury is a claim about a
   scene type 0.45 measured this sensor to differ from by 2-3x in local contrast
   (rule 2). The wall gives a proxy for "wrong" with no ground truth: residual off
   the fitted plane past a threshold.
5. **Temporal persistence** as a further feature, once 1.4's battery exists. Already
   measured discriminative: the residual ghost of 24a persists in 2.2% of frames
   against a 0.146 px whole-frame repeatability.

**The acceptance test, which can fail.** Any confidence worth shipping must
down-weight the bottom-left occlusion strip of 24a. The margin provably does not, so
this is a test with a known failure and a known failer rather than a number to admire
(rule 3). A random confidence must also reproduce the no-skill AUC, or the harness is
measuring itself.

## 0.551 Step 1 measured: the ranking works, the aggregation does not, and the tolerance decides

**The harness.** `de_dense --out-conf` writes the two candidates the solver ranked --
four W*H float32 planes, `s1 s2 d1 d2` -- and `article/confidence.py` scores every
measure built from them by the area under the sparsification curve. Sort the answered
pixels by confidence, keep the top fraction q, record the error rate among what is
kept; the area under e(q) sampled at q = 1.00, 0.99, ... 0.01 is one number, lower
being better. Both references are computed: an ORACLE that sorts by the true error,
and a RANDOM confidence whose curve is flat at the overall error rate.

Every run is ungated. Scoring a confidence on the pixels the margin gate already kept
asks how well the margin ranks what the margin did not remove, which is circular.

**Result.** Eight scenes at 450x375, 1,077,892 pixels with known ground truth and an
answer, `--min-margin 0`, K=2, bad-1.0. The population is 10.41% wrong.

| measure | AUC | AUC - oracle | error rate at 60% kept |
|---|---|---|---|
| oracle | 0.0062 | 0.0000 | 0.00% |
| APKR — peak ratio, aggregated 5x5 | 0.0294 | 0.0233 | 2.90% |
| PKRN — peak ratio | **0.0294** | 0.0232 | **2.74%** |
| AMMN — margin, aggregated 5x5 | 0.0335 | 0.0274 | 3.52% |
| MMN — the margin, what ships | 0.0344 | 0.0283 | 3.53% |
| MSM — the winning score alone | 0.0487 | 0.0425 | 4.53% |
| random | 0.1042 | 0.0980 | 10.42% |

**These cues rank well.** The peak ratio closes 76% of the distance between no skill
and the oracle, and it is free: the same two numbers the margin already reads, one
divide instead of one subtract.

**Local aggregation buys nothing here, and the reason is that it has already
happened.** Poggi, Tosi and Mattoccia's central finding is that pooling a measure over
a neighbourhood is what separates a good hand-crafted confidence from a poor one, and
APKR is a top-four measure where per-pixel PKR is not. Reproduced -- but only when the
cost volume is not aggregated first:

| sigma_s | 1 | 2 | **8 (ships)** | 20 |
|---|---|---|---|---|
| PKRN | 0.0513 | 0.0349 | **0.0294** | 0.0339 |
| APKR, 5x5 | 0.0418 | 0.0316 | **0.0294** | 0.0340 |
| what aggregation bought | 0.0095 | 0.0033 | **0.0000** | -0.0001 |

The edge-aware recursive filter runs over the cost volume before the top-2, with a
1/e reach of 5.7 px at `sigma_s 8`. The neighbourhood APKR would pool over is inside
that reach, so pooling again adds no evidence -- and past it, it only blurs across
depth edges: at radius 1 the AUC is 0.0291, at radius 4 it is 0.0307, at radius 8 it
is 0.0333. **A measure that is worth a window on a raw cost volume is worth nothing on
an aggregated one.** No new kernel, no new plane, no window.

**The two benchmarks disagree about the ratio, and the tolerance is why.** `--min-ratio`
gates on the ratio inside the solver, where uniqueness can hand a right pixel to a
competitor once its rival is gated away, so it is swept for real rather than simulated
(`confidence.py --gate-sweep`). On the eight scenes at bad-1.0 the ratio wins at every
matched coverage: 8.8% wrong at 76.2% coverage against the margin's 9.2% there, and
1.8% at 39.3% against 3.1%. On Middlebury v3 it is level or 0.2 points worse -- 21.42
at 69.2% coverage against the margin's 21.30 interpolated to the same point.

The two benchmarks grade different failures. Middlebury v3 at quarter resolution has a
0.25 px threshold; the eight-scene benchmark has 1.0 px at native resolution. Scoring
the same population at both thresholds separates them, error rate at 60% kept:

| threshold | 1.0 px | 0.5 px | 0.25 px |
|---|---|---|---|
| MMN | 3.53% | 6.51% | 25.46% |
| PKRN | 2.74% | 5.78% | 25.02% |
| the ratio's advantage | 0.79 | 0.73 | 0.44 |

**The peak ratio ranks gross mismatches better than the margin and says almost nothing
more about sub-pixel accuracy.** `--min-ratio` therefore stays off: one benchmark's
0.4-to-1.3 points is the other's nothing, and a default that moves needs both.

**The ceiling this exposes, which matters more than the gate.** At a 0.25 px threshold
the top 40% of pixels by APKR are still 19.9% wrong, against an oracle that reaches
0.00% at that density. The two scores carry which candidate won; they carry almost
nothing about how well the parabola landed inside it. So step 4's calibration can
promise `P(|d - d_gt| <= 1 px)` and cannot promise it at a quarter pixel from these
cues, and any consumer wanting sub-pixel reliability needs a feature that does not
exist yet.

**What the harness proves about itself** (`confidence.py --check`, rule 13). The random
confidence reproduces the population error rate to 0.0004, so a flat curve is what no
skill looks like. An inverted APKR scores 0.3148 against random's 0.1483, so the metric
has a direction and a sign error cannot read as a good result. And the margin
reconstructed in Python gates exactly the pixels the C++ gates: re-running with
`--min-margin 0.05` loses 0 of the pixels predicted to survive, which is the only check
that crosses the language boundary.

## 0.552 Step 2 measured: consistency is a veto, and it is the one that sees the ghost

**What it is.** Match left to right and you get a disparity for each left pixel. Follow
that disparity to the right pixel it claimed and ask that right pixel which left pixel
*it* would have chosen. If the two disagree, one of them is wrong. This is the standard
left-right consistency check, and it is the cue the peak ratio structurally cannot
supply: the ratio compares a pixel's own candidates, so where the true match is
off-sensor and the wrong match has no competitor, the ratio calls it confident.

**It costs one compare per score, not a second matching pass.** Every score already
says how well left pixel x fits right pixel x - d. The forward match reduces those over
d for each x; the reverse match reduces the SAME scores over x for each x - d. So
`de_dense --lrc` keeps a second running maximum in the loop that already holds the
score, indexed by `x - d` instead of `x`, and needs one best rather than a top-2
because the only question asked of a right pixel is which left pixel it would have
chosen. Two int16 planes a thread, 2.44 MB at 848x480, against 104.2 MB for the volume
that would otherwise have to be kept to read the same answer off a diagonal.

**The update must come before the reject, which is the whole subtlety.** The forward
loop discards almost every score with one test against this left pixel's runner-up. The
same score may still be the best any right pixel has seen, so a reverse match built
after that test would be built from the few percent of scores that survived it.

Measured to leave the disparity map bit-identical on all eight scenes, and to cost
47.4 ms against 54.2 ms at 450x375 D=60 with four threads on the desktop, best of five.
**That 14% is a desktop number and the TX2 has not been measured** (rule 2).

**It is a veto, not a ranking**, and that is the whole shape of the result:

| residual against the reverse match | share of pixels | error rate there |
|---|---|---|
| under 0.5 disparity | 90.0 - 97.6% | 3.9 - 16.2% |
| 0.5 to 1.5 | 1.4 - 4.8% | 16.6 - 36.4% |
| 1.5 to 5 | 0.2 - 1.5% | 49.8 - 79.3% |
| over 5 | 0.7 - 4.4% | 44.0 - 79.7% |

(range over all eight scenes.) A pixel that fails the check is three to seven times
more likely to be wrong. But 88-97% of pixels pass, and among those the check has
nothing to say about which is better. **A sparsification curve rewards fine ordering,
so it undersells a measure that gives one sharp cut**: on its own, consistency scores
AUC 0.0780 against the peak ratio's 0.0294, which reads as a poor measure and is not.

**Ranking the two and adding them is much worse than either idea deserves** -- AUC
0.0447. Ranking a value that is 96% tied hands out arbitrary distinct ranks inside the
tie, so half the combined score is noise. Used the right way round, as a veto with the
ratio ranking the survivors, it is the best measure here:

| measure | AUC | error at 95% kept | at 90% | at 60% |
|---|---|---|---|---|
| oracle | 0.0062 | 5.70% | 0.46% | 0.00% |
| **peak ratio, vetoed by consistency** | **0.0283** | **8.11%** | **7.01%** | 2.69% |
| peak ratio alone | 0.0294 | 8.74% | 7.43% | 2.74% |
| consistency alone | 0.0780 | 8.13% | 8.05% | 7.96% |
| the two, rank-added | 0.0447 | 8.88% | 8.13% | 5.59% |
| random | 0.1041 | 10.41% | 10.40% | 10.39% |

**The gain is at high density, which is the regime that matters.** Keeping 95% of
points the error falls from 8.74% to 8.11%; keeping 60% the two are level. The veto
removes a small population of badly wrong pixels, and by 60% the ratio has thrown
those out anyway. A vehicle does not want to discard 40% of its cloud, so the useful
part of this is precisely the part the AUC averages away.

**The acceptance test, and it passes.** Obstacle 24a's ghost is left pixels whose true
partner is off the sensor. The offline stand-in is the leftmost `dmax` columns, where
the same geometry applies. With both cues set to discard the SAME share of pixels
image-wide, and asked how many of that region's wrong pixels they caught:

| | Art | Books | Dolls | Laundry | Moebius | Reindeer | cones | teddy | pooled |
|---|---|---|---|---|---|---|---|---|---|
| peak ratio | 40.1 | 26.6 | 11.7 | 30.8 | 17.4 | 24.0 | 4.3 | 24.3 | **24.4%** |
| consistency | 59.8 | 54.5 | 39.7 | 46.4 | 31.9 | 70.5 | 18.0 | 74.9 | **45.5%** |

Consistency catches roughly twice as many, on 8 scenes out of 8, with no sign flip
anywhere -- which is what the mechanism predicts and what the score margin provably
cannot do (0.55). 12,260 wrong border pixels pooled.

**Where the numbers come from.** `de_dense --lrc`, the shipping path, with the reverse
scan fused into the cost loop. An earlier version of this measurement read the reverse
match off `--dump-vol` instead, which is a different cost function and not merely a
slower one -- obstacle 26. Every figure above has been retaken.

## 0.553 Steps 3 and 4 measured: it is a probability, and it is over-confident on a hard scene

**The model.** Six per-pixel features -- the peak ratio as a logarithm, the margin, the
winning score, the aggregated ratio, the consistency residual, and the plain yes/no of
whether consistency failed -- into a logistic regression, `P(correct) = 1 / (1 +
exp(-(w.x + b)))`. Fitted by Newton's method in twenty lines of numpy, so `core/` and
the analysis stay free of a machine-learning dependency. It is not a CNN and the no-CNN
decision of 0 does not reach it.

**Leave one scene out, never a random split of pixels.** Neighbouring pixels share a
window, a surface and a lighting condition, so a random split puts a pixel's own
neighbours in the training set and measures memory. Every number below is predicted by
a model that never saw that scene.

**Step 3's answer: the fit buys nothing for ranking.**

| | pooled AUC |
|---|---|
| the fitted probability, held out | **0.0274** |
| peak ratio vetoed by consistency, no fitting at all | 0.0283 |
| peak ratio alone | 0.0294 |
| oracle | 0.0062 |

Held-out and fitted-on-itself agree scene by scene -- 0.0406 against 0.0414 on Art,
0.0122 against 0.0087 on cones -- so this is not overfitting hiding a gain. Six features
over a million pixels have nothing to overfit. **The fit is worth 0.0009 over the
hand-made rule of 0.552, against 0.0212 of headroom to the oracle**, so nearly
everything these cues contain is already in "rank by the ratio, veto on consistency".

**Taking a feature away costs almost nothing, which says why.** Refitting without each
one, held out as above:

| removed | AUC | change |
|---|---|---|
| nothing | 0.0274 | |
| log peak ratio | 0.0274 | +0.0000 |
| margin | 0.0272 | -0.0002 |
| winning score | 0.0273 | -0.0001 |
| consistency residual | 0.0276 | +0.0002 |
| log APKR | 0.0278 | +0.0004 |
| **consistency failed** | **0.0281** | **+0.0007** |

The four cues built from the two scores are near-substitutes -- any one of them stands
in for the others -- and consistency is the only one carrying something independent,
which is 0.552's result arriving by a different route. **Individual weights are
therefore not readable**: the fitted coefficient on the consistency residual comes out
positive, which taken alone would say a larger disagreement argues the point is
correct. It says no such thing. It says how a fit divided credit between correlated
columns, and it is why the ablation exists.

**Step 4's answer: yes, it is a probability.** Binned by what the model promised, over
held-out pixels:

| promised | pixels | delivered | gap |
|---|---|---|---|
| 0.1 - 0.2 | 8,331 | 0.114 | -0.046 |
| 0.3 - 0.4 | 6,210 | 0.410 | +0.063 |
| 0.5 - 0.6 | 10,186 | 0.500 | -0.060 |
| 0.7 - 0.8 | 93,609 | 0.765 | +0.007 |
| 0.9 - 1.0 | 693,231 | 0.969 | -0.002 |

Expected calibration error 0.0052, Brier score 0.0733 against 0.0933 for predicting the
base rate at every pixel. **A predicted 0.8 means 80%** -- and the two bins that hold
93% of the pixels are the two that are accurate to 0.007. The sparse low-probability
bins are worth 0.05, which is the resolution to quote when a point is doubtful.

**And that pooled 0.0052 is hiding a factor of fifteen.** Calibration error per held-out
scene, with the direction of the miss:

| scene | its error rate | calibration error |
|---|---|---|
| cones | 4.7% | 0.0192 |
| teddy | 7.7% | 0.0127 |
| Dolls | 7.6% | 0.0158 |
| Reindeer | 8.1% | 0.0469 |
| Books | 10.1% | 0.0126 |
| Moebius | 11.1% | 0.0164 |
| Art | 14.8% | 0.0245 |
| **Laundry** | **20.2%** | **0.0804** |
| pooled | 10.4% | **0.0052** |

The scenes run from 4.7% to 20.2% wrong while the model's predictions span a much
narrower band. **It regresses towards the difficulty of the mix it was trained on**, so on a
scene harder than that mix it is over-confident -- which is the dangerous direction.
Pooled, the over-promising on Laundry cancels the under-promising on Reindeer and the
error reads fifteen times smaller than it is on the worst scene. A pooled calibration number
should not be quoted without the per-scene column beside it.

**The obvious fix is the right idea and 8 scenes cannot settle it.** Giving the model
two numbers about the whole frame -- how often the reverse match disagreed, and the
median winning score, both computable live with no ground truth -- moves the worst
scene from 0.077 to 0.050 and Reindeer from 0.048 to 0.019, leaves every AUC unchanged,
and makes the middling scenes slightly worse (Books 0.010 to 0.031, Dolls 0.016 to
0.024). Mean absolute bias 0.0239 to 0.0219. With leave-one-scene-out there are seven
training points for two coefficients, and the effect is smaller than the scene-to-scene
spread, so this is a direction and not a result. It needs Middlebury v3's fifteen
scenes or this camera.

**The controls, which can fail.** A model fitted on six columns of random numbers
scores AUC 0.1038 against a no-skill 0.1041 and predicts a constant, which is the base
rate and everything such a model can know. And the consistency flag *alone* scores an
AUC at no-skill -- no ranking at all, since it takes two values -- with a calibration error
of 0.0003, the best of any single feature. **Ranking and calibration are different
properties and one measure can have either without the other**, which is the whole
reason step 4 is a separate step from step 1.

**Shipped to the live path 2026-08-12, as one feature rather than six.** `de_dense_cuda
--stream` sends a byte a pixel beside the disparity map, and `de_live_ros2.py` colours
by it and gates on a ROS parameter that can be turned while the cloud is on screen --
rviz2 itself has no filter, only a colour ramp. See
[11-running.md](11-running.md#confidence-and-how-to-filter-by-it).

The live model is the **peak ratio alone**, AUC 0.0301 against the six-feature 0.0274,
and the three dropped features are not merely unavailable on the GPU. A synthetic frame
padded with black exposed the reason: a region with no texture has a constant Census
descriptor, so every disparity matches perfectly, s1 = s2, and both the margin and the
ratio correctly say "no idea" -- but a model that ALSO reads the winning score sees a
perfect score and returns 0.94. The padding scored higher than the real image beside
it. A linear model cannot say "a high score only counts when the margin is not zero",
and Middlebury never showed it such a pixel. The ratio alone returns 0.667 there. The
two-feature version fits weights of +15.73 and -16.31, a cancellation between collinear
cues with the margin entering negatively, and is rejected for the same reason the
ablation exists. `core/tests/test_confidence.cpp` now holds the flat-region case, and
would have caught it.

**Measured on this camera 2026-08-12, which is what step 4 was waiting for.** Two
flat-wall captures at 0.445 m, lamp on and off, `de_dense --lrc`, with distance off
the fitted plane as the truth: 6.2% of points more than 20 mm off, falling to 1.3%
keeping the best 60% by confidence, and an area under the sparsification curve of
0.0176 against 0.0620 for no confidence and 0.0023 for an oracle. **The ranking
transfers.** It closes 74% of the gap here against 76% on Middlebury.

The absolute number does not. It promises 0.86 on the wall and delivers 0.94 --
pessimistic, where Middlebury's hardest scene was optimistic.

**And the six kitchen captures say why a fixed threshold cannot work.** Three light
levels, projector on and off, with the reverse-match failure rate standing in for
truth:

| | night, no projector | night, projector | evening, projector | daylight, projector |
|---|---|---|---|---|
| mean intensity | 16.0 DN | 21.0 DN | 56.1 DN | 95.1 DN |
| fails the reverse match | 69.3% | 2.3% | 2.7% | 7.0% |
| mean confidence | 0.732 | 0.774 | 0.771 | 0.753 |
| share of the gap the ranking closes | 21% | 70% | 67% | 67% |

**The ranking holds at every usable light level and the level of the number does not
move at all.** Scene quality ranges over a factor of thirty; the confidence over 0.04.
So `min_confidence 0.85` keeps 6-18% of points in every condition -- most of a good
cloud discarded, a sixth of a hopeless one kept -- and the live path gained a
`keep_best` quantile, which keeps what it was asked for in all four.

The exception is the first column: with no projector in a dark room the frame is 0.12
DN of local contrast and 38.6% answered, the ranking closes only 21%, and there is
nothing to rank. That is a scene to refuse, not to filter, and the frame-level features
this section already flags are what would let it be refused automatically.

**What can be said to a consumer today.** Per point, `P(within 1 disparity)`, calibrated
to 0.005 over a mix of scenes and to 0.05-0.08 on a scene at the hard end of that mix,
biased optimistic there. Not `P(within a quarter disparity)`: 0.551 measured that these
cues barely see sub-pixel accuracy at all.

---

## 0.5 Quick wins now that the live view works

- ~~**Benchmark `de_dense` on the Jetson.**~~ Done. The dense numbers in 0.3 are TX2
  numbers, taken with `tools/tx2_ab.py`, and the desktop has since been shown not to
  be a proxy for the board in either direction. The memory-bandwidth premise this
  bullet rested on turned out to be false on both machines — see 09-matching.md,
  "We are not bandwidth-bound".
- **Compare against the D4 ASIC's own depth.** Nobody has checked whether this
  matcher beats the silicon it is replacing. Needs `rs_ir_capture` to also record
  the `Depth` stream, then compare disparities at our matched keypoints on (a) a
  flat wall, scored against a fitted plane, and (b) an emitter-off scene where
  texture is scarce and the uniqueness constraint should matter most. Reasoning and
  the expected shape of the answer in [01-hardware.md](01-hardware.md).
  **A free partial answer already exists**: the Middlebury v3 leaderboard carries
  `r200high` — "Custom ASIC", the RealSense stereo pipeline, the D4's ancestor — at
  48.7% bad-1.0 against ground truth (section 4). Not this camera and not our scenes,
  so it does not replace the experiment, but it says what the silicon scores when
  someone else measures it.

- **Measure denser detection against ground truth.** `--cell 12 --per-cell 2` gives
  ~3x the matches and looks clearly better live, but "reads better" is not
  "more correct". `de_bench --sweep` over the cell/per-cell grid would settle it,
  and if denser is also more accurate the matcher defaults should change.
- **Temporal stability.** Keypoints are re-detected every frame, so the cloud
  flickers on a static scene. Accumulating over frames, or tracking instead of
  re-detecting, fixes both this and the 44-51% stereo repeatability ceiling in
  [09-matching.md](09-matching.md). Same root cause seen along two different axes,
  which makes it the highest-value single change to the front end.

---

## 1. Needs you, physically — nothing else can proceed past these

### 1.1 Get a laser rangefinder — **the binding constraint**

Bring-up step 4: measure walls at 1, 2 and 3 m.

**Update, 2026-08-07: this no longer gates matcher development.** Middlebury supplies
ground truth on eight real scenes, and `core/tools/de_bench.cpp` runs the C++ matcher
and detector against it, so match correctness is measurable today without a
rangefinder. Every recent matcher change was validated that way. What the rangefinder
still gates is depth accuracy *on this camera*, on this baseline, in these rooms --
absolute scale, calibration drift, and anything about the vehicle's own environment.
That is real, but it is validation rather than development.

**Update, 2026-08-12: a tape measure does most of it, and is already wired up.**
`desktop/wall_check.py --truth` takes a ruler distance per bag and fits
`measured = scale*ruler + offset` across two or more distances. The offset absorbs the
distance from whatever the tape was held against to the point depth is referenced
from, so the **scale** — the part that is a claim about `f*B`, the calibration — comes
out without knowing where the imager sits inside the housing. The precision is
adequate by a wide margin: at 1 m one disparity pixel is 46.6 mm against a measured
frame-to-frame repeatability of 0.146 px, so a tape good to 2 mm is well inside the
noise. Use three distances, not two — two fit the line exactly and cannot be
falsified (rule 3).

What a rangefinder would still add over a tape: distances beyond a few metres, where
holding a tape straight is the limiting error rather than the matcher. That is the far
field, where `Z^2/(f*B)` makes the measurement coarse anyway. **Downgraded from binding
constraint to nice-to-have.**

**Why it was first.** It gated *matcher development*, back when the only quality
measures available were match count, objective, and median
|dy|, none of which can distinguish "removed 100 wrong matches" from "removed 100
right ones". Two prior experiments (coarse-to-fine and smoothness) both came back
negative and **neither result is conclusive** for exactly this reason — see
[09-matching.md](09-matching.md). Until a disparity can be checked against a known
distance, additions to `s(i,j)` can be observed but not evaluated.

The plan's own reason also stands: do this before driving, because once moving,
systematic and random error cannot be separated.

Expected precision to verify against: at f·B = 21.48 px·m, 0.1 px of disparity is
~5 mm at 1 m, ~2 cm at 2 m, ~4 cm at 3 m.

### 1.2 Run ArduPilot's 6-position accelerometer calibration

Stationary gravity reads **9.074 m/s², 7.5% low**, consistent across two independent
logs. That is a scale/calibration fault, not noise, and the plan uses the
accelerometer for the gravity vector and hence the ground plane.

Needs the vehicle physically rotated through six orientations, so it cannot be done
remotely. Verify afterwards with:

```sh
.venv/bin/python desktop/allan_variance.py bags/imu/<log>.bin   # checks |gravity|
```

### 1.3 Record a few varied scenes

Every matching number so far — including the headline **+46% MASDA over
nearest-neighbour** — comes from 4 stereo pairs of *one static desk scene*. Worth
knowing whether it generalises before it becomes a quoted figure.

```sh
.venv/bin/python desktop/live_view.py --collect scene02   # then scene03, ...
./core/build/de_match bags/scene02
```

Aim for contrast: a textureless wall, a cluttered close-range scene, something at
2–3 m, something with a thin foreground object (chair or table legs — the plan flags
these as the ordering-constraint test case).

### 1.4 A battery

Unblocks everything vibration-related, which the plan treats as a real risk:

- MEMS rectification tilting the gravity vector speed-dependently, in a way that
  *looks plausible*
- Whether arrival-time jitter survives vibration — the 0.04 px p99 figure is from a
  stationary bench and is explicitly untested under motion
- Whether the USB3 cable holds up; the plan calls vibration-induced disconnects the
  most common hardware failure, presenting as software bugs

Also needed before the foam-decoupled IMU mount can be evaluated.

### 1.5 Optional: a larger calibration board

A3, or A4 sheets tiled onto rigid backing. Only needed if you ever want a
trustworthy `fx` of your own — the factory value is validated to 0.13% on baseline
and under 0.35 px on principal point, so this is not blocking.

If you do: **measure the print**. The A4 came out at 96% scale (25 mm nominal → 24.0
mm measured), and that propagated straight into a 4% baseline error until caught.

---

## 2. Ready now — code only, no hardware

### 2.1 Kalibr for hand-eye and time offset

The remaining half of bring-up step 3. Needs the IMU and camera in one rosbag with
**real** timestamps.

- `bag_to_rosbag.py` exists and its output is validated against real ROS Melodic
- **But** a `live_view --collect` set has synthesised timestamps, which are fine for
  `kalibr_calibrate_cameras` and **not valid** for `kalibr_calibrate_imu_camera`
- So this needs a recording made with `rs_ir_capture` (which writes `frames.csv`
  with real host times) *while* IMU data is captured, and the two merged into one bag
- IMU noise parameters for the yaml are done — see
  [08-imu.md](08-imu.md#complete-noise-characterisation)
- Target: **ROS 1 Noetic in Docker on the desktop**. Noetic is the newest ROS 1
  Kalibr supports, and Ubuntu 24.04 cannot host ROS 1 natively. Not Melodic, not
  ROS 2.

Not yet written: the IMU side of the converter, and the time-alignment between the
camera's host timestamps and the Pixhawk's clock. `TIMESYNC` is present in the
MAVLink stream and is ArduPilot's own mechanism for this.

### 2.2a The graded cost and the sub-pixel fit interact — measured 2026-08-10

`--ad` blends a truncated absolute difference into the Census score, and was measured
at 10.3 -> 9.7% bad-1.0 when it was added. On Middlebury v3 it now measures the other
way, all 15 scenes at `--threads 1`:

| `--ad` | 0 | 0.08 | 0.15 (default) | 0.25 |
|---|---|---|---|---|
| bad-1.0 | **25.38** | 25.57 | 26.04 | 26.96 |

**The whole effect is the sub-pixel fit.** With `--no-subpixel`, ad 0 measures 42.00
and ad 0.15 measures 42.04 — indistinguishable. With the fit on, the gap is 0.66. The
truncated AD saturates and is piecewise linear, so blending it into the cost changes
its SHAPE near the minimum; the argmax does not care and a parabola through three
samples does.

**Both benchmarks are right and they are measuring different tolerances.** On the
project's own eight scenes at 450x375, scored at native resolution, the graded cost
still helps *with the fit on*: 9.8% against 10.3%, and coverage 76.2% against 75.8%.
On v3 the threshold is one pixel of FULL resolution, a quarter pixel of what we
compute — so v3 can see the fit's quality and the older benchmark cannot. AD improves
which disparity is chosen and degrades how well that choice is refined.

**So the default stays at 0.15**, since it wins where the shipping resolution is
measured and on coverage, and the disagreement is recorded rather than tuned away.
**The change that would win on both is to fit on the Census cost alone** while still
selecting on the graded one — two costs where there is now one, which is cheap on the
CPU and another filtered volume on the GPU. Not built; it is the first thing to try if
sub-pixel accuracy is ever worth more than a plane of memory.

### 2.2 Sub-pixel disparity refinement — **PROMOTED 2026-08-10, biggest measured lever**

The plan: "Depth accuracy hinges entirely on this." Keypoint positions are already
sub-pixel refined, but the *disparity* is just the difference of two independently
refined positions rather than a fit to the correlation surface between them.

~~Cannot be validated without 1.1.~~ **False as of 4.1.** It is measurable today
against Middlebury v3, and it is worth up to **45 points** on the metric the field
publishes: a perfect integer answer at Q scores 45.55% bad-1.0 where a perfect float
answer scores 0.80%. Our dense output is 0% fractional; SGM's is 99.3%. That is a
bigger lever than anything ranked in 0.3, and it needs no hardware.

**Done, and ON BY DEFAULT since 2026-08-10.** `de_dense` fits a parabola through the
aggregated cost at the winner's two neighbouring disparities. `--no-subpixel`
restores the old behaviour and is bit-identical to the pre-change build.

**The shipping number is `--threads 1`**, because that is the configuration
`de_dense_cuda` is verified bit-identical against, and the GPU has no fallback at
all. All 15 scenes, official scoring:

| Middlebury v3, training Q, bad-1.0 | bad | coverage |
|---|---|---|
| `--no-subpixel` | 41.91 | 79.3% |
| **default (sub-pixel on)** | **26.94** | 79.3% |
| SGM reference | 29.08 | 90.2% |

**15.0 points, at coverage that does not move** — the fit changes values, never the
decision. On bad-1.0 that puts us ahead of SGM; on coverage it does not, and the
matched-coverage reading below is the one that matters.

**An earlier figure here said 12.7 points and was measured multi-threaded.** The CPU
tool loses fits at the seams between threads (25.8% of pixels at six threads, worse
at eight), so every multi-threaded CPU number understates the shipping path. That is
a property of the prototype, not of the method: the GPU fits every pixel it can.

Two things this leaves open, both recorded rather than assumed:

1. ~~The int16 suspicion is untested.~~ **Tested 2026-08-10, and confirmed at
   1.2 points.** All 15 scenes at `--threads 1`, so no seam fallback on either side:

   | | `--no-subpixel` | default |
   |---|---|---|
   | blockwise, int16 end to end | 41.91 | **26.94** |
   | float volume (`--no-blockwise`) | 42.02 | **25.75** |

   Without the fit, int16 is *ahead* by 0.11 — the quantisation has never mattered
   to an argmax, which is what the original int16 measurement was about. With the
   fit it is behind by 1.19. That isolates the sensitivity to the fit exactly as
   suspected: `2*c0 - cm - cp` is a second difference, and second differences lose
   far more to quantisation than the maximum they are taken around. `SCORE_Q` is 14.

   Not worth acting on yet: 1.2 points against the 15.0 the fit buys, and widening
   the fit's inputs means widening the running top-2 the whole blockwise design is
   built on.
2. ~~Nothing here is timed on the TX2.~~ **Timed, then improved twice: 1.53x -> 1.30x
   on the CPU, 1.10x on the GPU.** First measurement, `tx2_ab.py "" "--subpixel"
   -n 8`, interleaved, minima:

   | | default | `--subpixel` | |
   |---|---|---|---|
   | total | 78.1 ms | 119.5 ms | **0.65x** |
   | cost stage wall | 50.2 | 85.3 | 0.59x |
   | c_insert (CPU) | 66.5 | 131.0 | 0.51x |
   | c_alloc (CPU) | 17.0 | 26.9 | |
   | cost-stage occupancy | **5.34 of 6** | **3.71 of 6** | |

   Both predicted costs are there and they are separable. The insert doubles: that
   is the unconditional `tpv` store per pixel-plane. And occupancy falls by 1.6
   cores: that is the 16-plane chunk, which is the `--simd` finding repeating
   itself — a coarser work quantum hands back more than the inner loop wins.
   (`c_score` and `c_filter` CPU both *fell*, 77.6 -> 62.0 and 94.8 -> 83.3, which
   is chunk locality; it does not rescue the wall clock.)

   **Both were fixed, and only one of them the way I expected.**

   *Double-buffering `islice`* removed the `tpv` plane: the previous plane is just
   the other half, so the unconditional store became a conditional read. Insert CPU
   131.0 -> 109.4 ms. **Wall clock barely moved**, because the insert was never the
   binding constraint -- occupancy was.

   *The scheduler was.* 16-plane chunks at D=60 is FOUR chunks for six threads, so
   two threads got nothing. Shrinking the chunk fixed occupancy and broke the
   contiguity the fit depends on (42.8% of pixels fell back at six threads). The
   answer is neither: **one contiguous range per thread, sized by work rather than
   plane count**, since a plane costs about its valid width `W-6-d` and equal plane
   counts would hand the low-k thread the most. No stealing, and a thread loses at
   most its first and last plane's neighbours.

   Final, `-n 12`: **76.4 -> 99.6 ms, 0.767x**, occupancy 4.97 of 6 against the
   default's 5.27. Fallback 2.6% at one thread, 25.8% at six.

   Still open on the CPU: those seams. The fix is for a thread to also score and
   filter the plane either side of its range without inserting it, at ~2/D extra
   work per thread. Not built -- and it only affects the prototype, since the GPU
   fits every pixel it can.

**The CUDA path has it too, 2026-08-10, and it is much cheaper there.** As predicted:
the volume is `[y][x][k]` with k innermost and the top-2 is a warp shuffle tree, so
both neighbours of the argmax are already in registers — one broadcast to publish the
winner, two shuffles to fetch its neighbours from the lanes that own them, since k's
lane is `(k % 64) / 2` by arithmetic rather than by search.

| TX2, pipelined `--frames 30`, best of 3 | baseline | `--subpixel` | |
|---|---|---|---|
| teddy 450x375 D=60 | 12.6 ms (79.6 Hz) | 14.1 ms (70.7 Hz) | 0.89x |
| 848x480 D=60 *(synthetic pair, timing only)* | 29.4 ms (34.0 Hz) | 32.6 ms (30.6 Hz) | 0.90x |
| CPU `de_dense`, same flag, for contrast | 78.1 ms | 119.5 ms | 0.65x |

**1.12x on the GPU against 1.53x on the CPU**, coverage unchanged (filled 85.9%
either way). **Real time survives, with almost nothing left**: 32.6 ms of a 33.3 ms
budget is 98%, where the baseline used 88%. That is a real decision, not a free win.

*The 848x480 row is a synthetic pair* — two consecutive `ir1` frames as left and
right, because no rectified 848x480 L/R dump exists in `bags/`. The kernels sweep a
fixed D and are data-independent, so the timing carries; the solver's work is mildly
data-dependent, so treat the absolute as approximate and the ratio as the result.
The baseline lands at 29.4 against the 28.9 recorded in 0.3, which is the size of
that effect.

**Bit-identity survives**, which was checked rather than assumed: all eight scenes
identical without the flag, and with it, coverage identical and every fitted pixel
identical on teddy and Art. The CPU's chunk-edge fallback does not bite at
`--threads 1`, where one thread walks every chunk consecutively — so the reference
config is exactly the case where the CPU loses nothing.

**The cheap win, taken 2026-08-10.** `cnb` crossed the bus as float, 3.3 MB a frame
at 848x480 against ~8 MB of existing candidate traffic, while the device never had
more than int16 to give. Sending Q14 and dequantising in the solve thread halves it.
Predicted 0.85 ms of the 3.2; **measured 0.65** — 32.6 -> 31.9 ms, best of 4, against
a 29.0 baseline. So `--subpixel` now costs 2.9 ms rather than 3.2, and 848x480 runs
**31.3 Hz at 96% of the 33.3 ms budget** instead of 30.6 Hz at 98%. Bit-identity
re-checked on all eight scenes after the change.

The remaining 2.9 ms is the two shuffles and the register pressure in the top-2
kernel plus the host-side fit, and is not obviously worth chasing further.

**Still open: the sparse matcher.** It differences two independently refined keypoint
positions, which is the original item and unaddressed. The article records the
consequence: median disparity error 0.50 px, exactly integer-vs-quarter-pixel
quantisation.

`article/middeval3.py --ceiling` measures the floor for any change here, and
`--check` guards the evaluator.

### 2.3 Fix the timing comparison in `de_match`

The coarse-to-fine path times detection plus matching; the single-level MASDA figure
times matching only. The two are not comparable. The coarse-to-fine conclusion does
not depend on it (it rests on match counts), but the numbers should not be left
side by side as if they were.

### 2.5 Document `de_dense` in 07-tools.md — deferred until the flags settle

The tool the last two sessions were entirely about, and the only one missing from a
page whose stated job is *every tool and how to run it*. Deferred by decision, not
oversight: `--simd` is off by default and may be reverted, and the recursive filter is
next on the ranked list in 0.3, so the flag set is still moving. Write it once it is
not.

Everything needed is already recorded — the measurement behind each flag is in
09-matching.md and the usage string lists them — so this is transcription, not
recovery.

### 2.4 Re-measure preprocessing under realistic load

78.8% of the 33.3 ms budget was measured with the two channels concurrent and
nothing else running. Add the matcher, and eventually the IMU path, and re-check.
Concurrency already returned 1.54× rather than 2×, which points at memory bandwidth,
so headroom may be smaller than the number suggests.

---

## 3. Deferred with reasoning — do not redo without new information

### 3.1 Coarse-to-fine — off by default, kept in the code

Does not help: k is already 2.7, not the 100–200 the plan assumed, and inflating k
five-fold leaves the answer *identical*. There are no false candidates for a prior
to remove. Full reasoning in [09-matching.md](09-matching.md).

**Revisit if**: calibration drifts and the epipolar band must widen; keypoint density
rises a lot; or — most likely — **temporal matching between frames**, where the search
is genuinely 2-D, k really is large, and the prior would come from IMU rotation
compensation. That is where this code should earn its place.

### 3.2 The full disparity-smoothness factor derivation

The plan's cheap two-pass test of whether this is worth deriving came back negative
— monotonically worse at every weight, best `w_smooth` is 0. Probable cause is
structural: the prior is fitted from the very matches it then judges, so it
reinforces errors rather than correcting them.

**But** see 1.1 — that negative is not conclusive without ground truth. Reconsider
after the rangefinder, not before.

If resumed: use a Delaunay triangulation rather than the k-nearest-neighbour proxy
currently implemented, and Eigen (3.4 desktop, 3.3.4 Jetson) for anything larger
than the current 3×3 fit.

### 3.3 uvcvideo kernel patch

Not worth it for timing: arrival jitter is 0.04 px at p99 against a ±3–4 px budget,
and it holds under full CPU load. Would still buy hardware-sync verification,
in-motion projector labelling, and per-frame exposure confirmation.

### 3.4 Hardware sync verification

Currently **unverifiable** — with a host-side frame counter and host-side stamps,
both L/R pairing and the timestamp difference are true by construction. Expected to
fall out of the first matcher via median |dy|, and it partly has: 0.359 px is
consistent with real sync. Not proof.

---

## 4. Open questions from the plan, still open

- **Calibrating λ and γ — but they are INERT where they currently sit** (measured
  2026-08-11, `--lambda/--gamma/--damping` exposed for it). Middlebury v3, all 15
  scenes, `--threads 1`, both held equal:

  | λ=γ | -0.4 | -0.2 | **-0.1 (default)** | -0.05 | +0.1 | +0.3 | +0.5 |
  |---|---|---|---|---|---|---|---|
  | bad-1.0 | 26.09 | 26.08 | **26.04** | 26.02 | 25.59 | 22.01 | 13.02 |
  | coverage | 80.0% | 80.1% | **80.1%** | 80.1% | 79.7% | 73.4% | 47.2% |

  **Below zero they do nothing at all** — a factor of eight in magnitude moves
  bad-1.0 by 0.07 and coverage not at all. They sit far under the score
  distribution, so `cs <= lambda` almost never fires and `max(lambda, excl)` is
  almost always the other term. The default is not a value, it is *off*.

  **Above zero they work, and they are a second margin gate.** +0.5 costs 33 points
  of coverage to buy 13 of bad-1.0, which is the same shape of trade `--min-margin`
  already offers at roughly the same exchange rate. So this is another knob that
  moves along the precision–coverage curve rather than off it, and it is not tuned
  here for the same reason the gate is not.

  **What that does to this item.** The plan's candidate was a small MLP over
  descriptor distance, y-residual, coarse-disparity residual, keypoint response,
  response ratio and local texture energy → calibrated log-likelihood ratio. That is
  still interesting, but it would have been trained to predict parameters which, at
  the scale they are currently set, the output does not respond to. Any such work
  has to start by putting λ and γ **into** the active range, and it then competes
  with a one-line gate that already does the same job. Needs ground truth (1.1) to
  train against, and now also needs a reason to beat `--min-margin`.

  *(Superseded framing, kept: "Currently hand-set. The plan's candidate: a small MLP over
  descriptor distance, y-residual, coarse-disparity residual, keypoint response,
  response ratio and local texture energy → calibrated log-likelihood ratio. Needs
  ground truth (1.1) to train against.")* **Survives the no-CNN decision by decision,
  not by oversight** (2026-08-10): six scalar features, no image plane, no GPU,
  nothing that competes with the matcher for the device — which is what that decision
  was about. Do not strike it when applying "no CNN" elsewhere.
- **Second use of MASDA for frame-to-frame association** (visual odometry). Same
  machinery; the IMU rotation compensation makes it tractable, and it is the case
  where 3.1's coarse prior should finally pay.
- **`INS_LOG_BAT_OPT` semantics** — batch data is windowed at ~1 s. Not blocking any
  more, since long-τ parameters come from the continuous stream instead, but worth
  understanding if raw continuous data is ever wanted.
- **DL baseline: cite, do not run** (Mario, 2026-08-10). The earlier plan was to run
  RoMa / ELoFTR / LoMa offline on the desktop against recorded bags. Dropped —
  nothing learned runs in this project, on the board or off it, so the experiment has
  no decision attached to it. Compare against the **published** numbers instead.

  **What that comparison can and cannot say.** A cited number is not a measurement
  here (rule 2), and the gap is wider than usual: those methods are wide-baseline
  two-view matchers reported on relative-pose and homography benchmarks, while every
  number in this project is rectified stereo scored as bad-1.0 and coverage on
  Middlebury. There is no shared metric, so "RoMa scores X" and "MASDA scores 9.7%
  bad-1.0" are not two points on one axis. Reported runtimes are worse still —
  desktop-GPU figures against a TX2 budget is exactly the 3x error obstacle 15 cost
  us.

  **Numbers pulled 2026-08-10, so nobody has to fetch them again.** From the papers:
  RoMa MegaDepth-1500 AUC@5 62.6, ScanNet-1500 31.8; ELoFTR MegaDepth-1500 AUC@5/10/20
  56.4/72.2/83.5, ScanNet 19.2/37.0/53.6, 40.1 ms (27.0 optimised) at 640x480 on an
  RTX 3090, evaluating on MegaDepth, ScanNet, HPatches, InLoc and Aachen — **no stereo
  benchmark at all**. *"LoMa" could not be identified and is probably a typo in the
  original plan; do not cite it.*

  **The Middlebury v3 leaderboard is the closer comparison**, same metric name at
  least. `vision.middlebury.edu/stereo/eval3/`, bad 1.0 / nonocc / test dense,
  weighted average, read 2026-08-10 (parse `table.php`, not the page — the table is
  loaded by JS):

  | | bad 1.0 | hardware |
  |---|---|---|
  | MatchAttention / S2M2 / FoundationStereo | 3.49 / 3.57 / 4.39 | RTX PRO 6000, 4090, A100 |
  | NOSS_ROB, CRLE, LocalExp-RC, 3DMST-CM, LocalExp | 13.2 - 13.9 | 4-8 core CPU |
  | PMSC | 14.8 | i7 + TITAN X |
  | OVOD — fully connected CRF solved by **BP** | 17.1 | Xeon |
  | SGBM2 (OpenCV SGM) / ELAS | 44.2 / 44.4 | 1 core |
  | **r200high — "Custom ASIC", the RealSense stereo pipeline** | 48.7 | ASIC |

  Two readings of that table, both worth keeping:

  - **The 13.2-13.9 cluster is not "the best traditional method".** LocalExp's own
    entry says its raw costs come from MC-CNN-acrt — a *learned* cost volume with a
    classical optimiser on top — and the whole cluster descends from LocalExp. The
    page gives no description for NOSS_ROB, CRLE or 3DMST-CM, so none of them can be
    certified learning-free from the leaderboard alone. The best entry verifiable as
    hand-crafted is **OVOD at 17.1**, which is a fully connected CRF solved by belief
    propagation — the nearest relative on that board to what this project is.
  - **`r200high` is the D4's ancestor, and on the sparse table it scores 2.76.**
    48.7 is its *dense* number, where every pixel it declines to fill counts as an
    error. Judged on what it actually outputs it is 2.76% bad-1.0 — the silicon is
    not a weak baseline, it is a high-precision low-coverage one, which is the same
    operating point the margin gate puts us at. See the sparse-table note below.

  **The sparse table is our table, and it moves everything.** Gated, semi-dense
  output belongs there. Test dense vs test sparse, bad-1.0 nonocc: r200high 48.7 ->
  **2.76**, SNCC 32.9 -> 12.7, IDR 29.7 -> 11.8, ELAS 44.4 -> 26.4, SGBM2 44.2 ->
  33.3. The dense leaders do not move (MatchAttention 3.49 either way) because they
  fill everything.

  *Correction, from reading the SDK's `evaldisp.cpp` while building 4.1:* the two
  tables are **two submitted files**, not two metrics — the SDK ships `disp0SGM.pfm`
  and `disp0SGM_s.pfm` — and `bad` is wrong pixels over all masked pixels *including*
  the empty ones, so it is not an error rate over filled pixels either. Dividing by
  coverage is what makes it one. The coverage column below is therefore an estimate
  good to about two points, not a derivation: measured directly, SGM_s is 90.2%
  covered where the dense-minus-sparse estimate predicts 91.8%.

  **But the leaderboard has no density column** — the stats are bad 0.5/1.0/2.0/4.0,
  avgerr, rms, A50/90/95/99 and three time normalisations, and nothing reports what
  fraction of pixels a sparse entry filled. So "r200high 2.76" is uninterpretable on
  its own, and ours would be too. Any sparse number we quote has to carry its
  coverage or it is not a claim.

  **Do not put our 9.7% next to any of these.** Different dataset: v3 is the 2014
  high-res set, ours is Teddy/Cones (2003) plus the six 2005 scenes at 450x375. The
  leaderboard demonstrates the gap itself — we score SGM at **10.9%** on our eight
  scenes and OpenCV's SGBM2 scores **44.2%** on v3. Same algorithm family, 4x apart,
  entirely from the dataset. Quoting "MASDA 9.7 vs LocalExp 13.9" would say we beat
  LocalExp, and the data does not support that.

  Anything written into the article from a paper or a leaderboard must carry its
  source, its dataset and its hardware, and must say it is cited rather than measured
  — the ceiling experiments (81.5% available vs 67.9% delivered, the 0.697 re-ranking
  cap) remain the only statements about attainability measured on our data.

### 4.1 Scoring ourselves against that table — **BUILT AND RUN 2026-08-10**

`article/middeval3.py` (see 07-tools.md). Validated first: it reproduces the published
SGM row exactly, **37.33 against 37.3 dense and 29.08 against 29.1 sparse**, and exits
non-zero if it ever stops doing so.

**The headline is not the score, it is what the score is made of.**

| training Q, official bad-1.0 | bad | invalid | coverage |
|---|---|---|---|
| **dense MASDA**, 13 of 15 scenes | **41.93** | 19.77 | 80.2% |
| SGM reference, same 13 scenes | 29.13 | 8.51 | 91.5% |
| **a PERFECT INTEGER answer** | **45.55** | 0.13 | 99.9% |
| a perfect float answer | 0.80 | 0.13 | 99.9% |

As an error rate over the pixels each one actually fills: MASDA **52.3%**, the
perfect-integer floor **45.6%**, SGM **31.8%**.

**So the matching is within about seven points of a perfect integer matcher, and the
metric is otherwise measuring sub-pixel output.** The official threshold is one pixel
at full resolution, which is a quarter pixel of the disparities we compute at Q; an
integer answer cannot get inside it however right it is. SGM's reference is 99.3%
fractional, ours is 0%.

**`--subpixel` is a dead code path**, not a disabled one: on Teddy it changes 3 pixels
of 136,219 and produces no fractional values at all. The negative recorded in
`de_dense.cpp` ("measured, it makes things slightly WORSE -- 8.8% against 8.6%")
predates the blockwise top-2 change that removed the cost volume, and the guard
`k > k0[x] && k + 1 < k1[x]` now almost never holds. That negative cannot be reproduced
because the code no longer runs. **Re-derive it before believing it** (rule 6).

**Consequences for the list.** 2.2 said sub-pixel disparity "cannot be validated
without 1.1". That is now false — it is measurable today, it is worth up to 45 points
on the metric the field publishes, and it is a bigger lever than anything currently
ranked in 0.3. Promoted.

~~Jadeplant and Vintage are skipped at `--dmax-cap 96`.~~ **They run fine** (checked
2026-08-10): nothing in either tool caps D below ~220 — the CUDA top-2 packs k in 8
bits and `G` covers 256 disparities — and the cap was a conservative default of mine,
not a limit. `--dmax-cap 200` scores all 15.

### The margin sweep, 2026-08-10: SGM dominates the curve

With sub-pixel on, sweeping the gate. **These were run multi-threaded and therefore
understate every row** (see 2.2 -- the CPU loses fits at thread seams); the two rows
that carry the conclusion were re-run at `--threads 1` and are marked. The shape is
what matters:

| `--min-margin` | bad | coverage | error over filled |
|---|---|---|---|
| 0.0 (no gate) | 37.96 | 91.8% | 41.4% |
| **0.01 (default)** | **29.20** | **80.2%** | 36.4% |
| 0.03 | 19.93 | 64.5% | 30.9% |
| 0.05 | 14.61 | 53.3% | 27.4% |
| 0.10 | 7.52 | 34.3% | 22.0% |
| 0.20 | 2.29 | 14.5% | 15.8% |
| **SGM reference** | **29.08** | **90.2%** | **32.2%** |

**Read at matched coverage, SGM is ahead by 6.8 points.** At 91.2% we are at 35.85
against SGM's 29.08 at 90.2%. Read the other way we are *ahead on bad* — 26.94
against 29.08 — but eleven points short on coverage. The summary is that SGM
still dominates the curve, by less than the first (multi-threaded) measurement said,
and that the gate moves along the curve rather than off it.

**This corrects a comparison made earlier today.** "Level with SGM" was 29.20 against
29.13 — true numbers, unequal coverage, and therefore not a comparison. The gate is
exactly the knob that makes such a pairing meaningless, which is the argument for
publishing the curve and is now also an argument against quoting any single point of
it.

It does not contradict the article's 9.7% against SGM's 10.9%: that is eight
2003/2005 scenes at 450x375 scored at native resolution, this is fifteen 2014 scenes
scored at full resolution after 4x upsampling. Different benchmark, harder, and the
one the field publishes on.

**So sub-pixel was the improvement that was on the table, and it has been taken.**
12.7 points. The gate is not a second one.

#### How it works, for the next person

**The training set's ground truth is public**, so this is a local benchmark, not a
submission. Submission buys a public rank and nothing else.

**Quarter resolution is nearly our operating point**, which is what made this cheap. Full res is ~2900x1950 with ndisp 240-740; at Q that is **~740x490 with
ndisp 60-185** against our 848x480 D=60. Measured from `calib.txt`:

| scene | ndisp F | ndisp Q | scene | ndisp F | ndisp Q |
|---|---|---|---|---|---|
| Shelves | 240 | 60 | Playroom | 330 | 83 |
| Piano, Recycle | 260 | 65 | **Jadeplant** | **640** | **160** |
| Motorcycle | 270 | 68 | **Vintage** | **740** | **185** |
| Adirondack | 280 | 70 | Playtable | 290 | 73 |
| Pipes | 300 | 75 | | | |

So 13 of 15 training scenes fit under D=96 at Q, and **Jadeplant and Vintage do
not** — they need D=192. That is the one real piece of work, and the CUDA path has
already been bitten once by a compile-time `DPAD=64` silently breaking D=80 (0.3), so
this is exactly the axis that has a known failure mode.

The price of Q: **the official evaluation is always at full resolution and upsamples
your result**, which is most of why SGBM2 (submitted at Q) sits at 44.2 dense. The
penalty is real but it is the same one comparable systems on that board already pay.
Middlebury's own advice is to submit the largest resolution the algorithm supports.

**Scoring Q against Q ground truth is not that evaluation and is worth 2.9x.** The
shipped SGM reference reads 13.03 that way against its published 37.3 — bad-1.0 at Q
is roughly the board's bad-4.0. The README gives the conversion (1.0 at F is 0.25 at
Q) and warns the converted number still differs; measured here, the conversion alone
lands 39.29 against 37.3, so the full-res ground truth is not optional. This is
exactly rule 3: the first evaluator produced a plausible, wrong, flattering number,
and only the fixture caught it.

What to fetch, all from `vision.middlebury.edu/stereo/submit3/zip/`:

| file | size | what |
|---|---|---|
| `MiddEval3-data-Q.zip` | 31 MB | images + calib, 15 training + 15 test |
| `MiddEval3-GT0-Q.zip` | 14 MB | ground-truth disparities, training only |
| `MiddEval3-algSGM-Q.zip` | 16 MB | **SGM reference output, already scored on the board** |
| `MiddEval3-SDK-1.6.zip` | | `runeval`, PFM I/O, submission packaging |

**`algSGM-Q` is the fixture that makes our evaluator falsifiable** (rule 3). Score
the shipped SGM output with our own code and it must reproduce the leaderboard's SGM
row. An evaluator that cannot reproduce a known row is not evidence about MASDA.

Then: PFM out at Q, score `bad100` under `mask0nocc.png`, report the **sparse** number
**with its coverage**, and quote the dense number too so the gate's cost is visible.

### 4.2 What the board says we should learn — read 2026-08-10

Read before running anything, because it re-ranks work we already had planned.

**The board hides its density column, but it is recoverable.** Dense and sparse
entries differ only in whether unfilled pixels count as errors, so
`coverage ~= (100 - dense) / (100 - sparse)`. **The check that this is sound: every
method known to be fully dense comes back at exactly 100.0%** -- MatchAttention,
LocalExp, CRLE, NOSS_ROB, OVOD, PMSC, MeshStereo, 3DMST-CM, S2M2, FoundationStereo,
ten of them. So the coverage column below is derived, not published, and it is the
only way to read the sparse table at all.

bad-1.0, nonocc, test set, weighted average:

| method | dense | sparse | ~cov | s/Gdisp | hardware |
|---|---|---|---|---|---|
| MTS2 (max-tree) | 64.3 | **0.61** | 36% | 44.5 | i7 4-core |
| MTS | 71.6 | **0.70** | 29% | 1.29 | i7 4-core |
| **r200high** (the D4's ancestor) | 48.7 | 2.76 | 53% | **0.03** | ASIC |
| ICSG (intrinsic curves) | 55.2 | 3.41 | 46% | 89.3 | **1 CPU core** |
| MotionStereo | 52.3 | 7.56 | 52% | 0.25 | **1 phone core** |
| IDR | 29.7 | 11.8 | 80% | 0.82 | CUDA, TITAN Black |
| SNCC | 32.9 | 12.7 | 77% | 2.42 | 1 core |
| ELAS | 44.4 | 26.4 | 76% | 1.45 | 1 core |
| NOSS_ROB / CRLE / LocalExp | 13.2 / 13.4 / 13.9 | same | 100% | 1327 / 3107 / 1870 | CPU |
| MatchAttention / S2M2 | 3.49 / 3.57 | same | 100% | 1.17 / 1.30 | RTX PRO 6000 / 4090 |

**Where we stand on accuracy: still unknown**, and 4.1 is the only way to find out.
9.7% at 76% coverage is 2003/2005 scenes at 450x375; v3 is harder and the number does
not transfer.

**Where we stand on efficiency: competitive, by the board's own normalisation.**
`de_dense_cuda` is 28.9 ms at 848x480 D=60 = **1.18 s/Gdisp on a TX2**, against
MatchAttention's 1.17 on an RTX PRO 6000 and IDR's 0.82 on a TITAN Black (~5 TFLOPS
against the TX2's ~0.75). Per FLOP the k-minor port is several times more efficient
than IDR. Two entries beat it and both are informative: **r200high at 0.03** is
fixed-function silicon and is the real ceiling, and **MotionStereo at 0.25 on a
single Snapdragon core** is motion stereo with a narrow temporal search range -- our
own temporal-prior result (0.3) arriving from the other direction.

Four things to take from it:

1. **The precision leaders win by covering less, not by matching better.** MTS2 is
   0.61% at ~36% coverage, r200high 2.76 at ~53%, against our ~76-86%. Same frontier,
   same knob as the margin gate. Hence the reframing of item 2 above: publish the
   curve, since the board can only publish points.
2. **Per-pixel disparity-range restriction is the lever, and ICSG/max-trees are how.**
   Promoted into item 3 above, ahead of ELAS and ESPReSSo.
3. **ELAS is a speed construction, not a quality one** -- 26.4% sparse at ~76%
   coverage, worse than SNCC at the same coverage. Demoted in item 3.
4. **Do not chase the 13% dense-classical cluster.** NOSS_ROB, CRLE and LocalExp cost
   1327-3107 s/Gdisp -- about 1000x r200high and ~1500x our budget -- and are mostly
   MC-CNN-cost hybrids anyway. No version of that cluster runs at 30 Hz on a TX2.

**IDR is the closest operating point on the board** and worth reading for one specific
thing: CUDA, census+gradient cost (our graded cost is the same idea), ~80% coverage,
real time, and it adds *iterative cost penalisation and disparity re-selection* for
local smoothness. That is a cheap smoothness mechanism which is **not** the
self-confirming two-pass prior that failed in 3.2.

---

## 5. Operational reminders

- **`jetson_clocks` does not survive a reboot**, and `/etc/nvpmodel.conf` carries
  `DEFAULT=3`. `doubleeye-performance.service` handles both, but if it is ever
  disabled you silently lose a third of your frames. The tools warn; `run.txt`
  records `clocks_locked`.
- **`LOG_DISARMED=1` logs continuously at ~130 MB/hour**, about 3 GB/day. Set it back
  to 0 when not recording, or the card fills.
- **The Pixhawk re-enumerates** ACM0 → ACM1 across a reboot. Tools resolve it through
  `/dev/serial/by-id/`; anything hand-written should too.
- **`stty raw` is mandatory** before reading a tty directly, or you get a few hundred
  bytes where the real rate is kilobytes per second — indistinguishable from a dead
  link.

## Article / matcher validation (added 2026-08-07)

Done:
- Real-data validation on Middlebury 2003 Teddy + Cones. `article/regen_all.py`
  regenerates every article number and figure and writes `results.json`.
- Fixed `rng_for()` seeding from Python's per-process salted `hash()`. Numbers
  were irreproducible across runs; one conclusion had flipped sign as a result.

**Not writing a Part 5** (Mario, 2026-08-10). The v3 harness turned up a good
standalone story -- a perfect *integer* disparity map scores 45.6% bad-1.0 where a
perfect float one scores 0.8%, so nearly the whole gap to SGM on that board was
output quantisation -- and it is deliberately not being published. Do not re-propose
it; the measurements live in 4.1 and 2.2 either way.

Open, in rough priority order:
- **Detector repeatability is the binding constraint on real data.** Only 48% of
  Teddy's left keypoints (51% on Cones) have any right keypoint within 1 px of
  their true correspondence, which accounts for 102 of Teddy's 132 errors.
  Precision on the attainable subset is 0.876 / 0.963 against a raw 0.615 / 0.781.
  Two cheap experiments: (a) detect once and track into the second image instead
  of detecting twice; (b) lower the right-image detector threshold to
  over-propose and let gamma discard the surplus.
- Sub-pixel disparity. `detect()` returns integer positions, so median disparity
  error is 0.50 px on Teddy, exactly the integer-vs-quarter-pixel quantisation.
  Fit the correlation surface between the two windows rather than differencing
  two independently refined positions.
- Fix swapped lambda/gamma in `core/src/match.cpp` relative to the article's
  convention (lambda = clutter/left-unmatched, gamma = misdetection/right).
- ~~Port `_seg_max_excluding` into the C++ matcher~~ -- not needed. `Top2` in
  `core/src/match.cpp` already gives O(1) "max excluding one element"; the NumPy
  trick exists only because NumPy cannot loop cheaply. The matcher is also not
  where the time goes.
- Run the C++ matcher against Middlebury too, so the C++ and NumPy paths are
  compared on identical data with ground truth. This is now possible without a
  rangefinder and no longer blocks on hardware.

## Matcher work, 2026-08-07

Done:
- **Score margin exported per match** (`Match::margin`). Best-minus-second-best
  s(i,j), one O(E) pass. Precision by margin quartile over eight Middlebury
  scenes: 0.169 / 0.286 / 0.391 / 0.659. `de_match` reports median margin and the
  share below 0.2; it is in `matches.csv`. On `full_on`, MASDA has 31.5% of its
  matches below 0.2 against mutual-NN's 14.3%, which is where MASDA's extra 47%
  of matches live.
- **lambda/gamma un-transposed.** lambda = clutter (left/measurement unmatched),
  gamma = misdetection (right/object unmatched). No past result was wrong, since
  every run held them equal, but they are not interchangeable once separated.
  `test_lambda_gamma_are_distinct` guards it.
- **core/ now builds on the Jetson.** It never had: `deploy.sh` synced only
  `jetson/`, and the `core/build` on the Jetson held x86-64 objects from the
  desktop. Consequence: MASDA is **5.04 ms on the Jetson**, not the 1.67 ms in the
  docs, which was a desktop number compared against the Jetson frame budget. 15%
  of the budget, not 5%.
- **Local contrast is not useful** and the intuition is backwards. See
  `article/contrast_study.py`. Do not raise `min_local_std` above its current 2.0.

Open:
- ~~The published article still carries the 1.67 ms figure attributed to the Jetson
  frame budget.~~ **Already gone** (checked 2026-08-10 against
  `mayio/mayio.github.io` master): the string appears in no post. It went in
  `daaa295`, the Part 1 reframe, on 2026-08-09.
- Detector repeatability: only 48-51% of left keypoints have a right keypoint
  within 1 px of their true correspondence. `article/repeatability.py` tests
  over-proposing on the right with a cheaper gamma.
- Sub-pixel disparity. `detect()` returns integer positions, so median disparity
  error is 0.50 px, exactly the integer-vs-quarter-pixel quantisation. Fit the
  correlation surface between the two windows.
- ~~Port `_seg_max_excluding` into the C++ matcher~~ -- not needed. `Top2` in
  `core/src/match.cpp` already gives O(1) "max excluding one element"; the NumPy
  trick exists only because NumPy cannot loop cheaply. The matcher is also not
  where the time goes.
- Use the margin downstream: weight or gate matches by it before triangulation.

## Frame budget, measured 2026-08-07

**The pipeline does not close at 30 Hz.** Jetson, MAXN, 848x480, 120 pairs:

| stage | ms/pair | share of 33.3 ms |
|---|---|---|
| preprocessing, L/R concurrent | 26.54 | 79.6% |
| MASDA baseline | 5.14 | 15.4% |
| **total baseline** | **31.68** | **95.1%** |
| MASDA `--right-density 6 --min-margin 0.10` | 8.84 | 26.5% |
| **total recommended** | **35.38** | **106.2%** |

Detection is 20.98 of the 21.24 ms per frame, so zeroing the matcher entirely
still leaves preprocessing at 80%. Needs a decision, in this order:

1. Run correspondence at 15 Hz. 66.7 ms budget, 53% used even with the
   recommended config. Costs latency only, and 15 Hz is plausibly enough for
   indoor odometry.
2. Profile detection. FAST already replaced the dense Shi-Tomasi scan; the
   Shi-Tomasi scoring of FAST corners and the NMS have not been profiled
   separately.
3. Drop to 640x480, which is 75% of the pixels and preprocessing is close to
   pixel-bound.

The recommended matcher config is off by default until this is settled.
