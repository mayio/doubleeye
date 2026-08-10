# Architecture: what runs where

The system is meant to grow object tracking, SLAM, ground detection and trajectory
estimation on top of stereo correspondence. That raises two questions worth
separating, because they had different answers:

- **What is the boundary between GPU and CPU?** Decided: the GPU owns the image
  plane, the CPU owns the graph, and the interface is a compact candidate buffer
  rather than an image. That is the rest of this page.
- **When do we write CUDA?** Decided 2026-08-10: now, for the stereo matcher. The
  deferral below was argued against the *sparse* pipeline's 10 ms of slack, and
  dense correspondence at the sensor's own resolution has no slack at all. See
  [the decision](#the-decision-2026-08-10-the-matcher-is-gpu-work).

## The decision, 2026-08-10: the matcher is GPU work

**Mario's call: the stereo matcher runs on the GPU. There is no other choice.**

More precisely, and this is what `core/tools/de_dense_cuda.cu` already ships: the
*image-plane half* of the matcher is GPU work — census, graded cost, the recursive
filter, the running top-2. The MASDA solve stays on the CPU and hides under the next
frame's kernels. So the boundary proposed on this page survived; what did not survive
is the "no CUDA yet" schedule attached to it.

**Measured, TX2, 848x480 D=60, pipelined over 30 frames** (TODO 0.3, 09-matching.md):

| | CPU, six cores | GPU + CPU, pipelined |
|---|---|---|
| 848x480 D=60 | 200 ms | **28.9 ms = 34.6 Hz** |
| 450x375 D=60 | 70 ms | **12.6 ms = 79 Hz** |

Bit-identical to `de_dense --threads 1` on all eight Middlebury scenes and the real
pair, verified with `cmp`, including through the pipelined frames.

**Why the deferral ended, in one line:** its premise was 10 ms of slack, and that
number was measured against the sparse pipeline. Dense at 848x480 was 6.0x over the
30 Hz budget on the CPU, the GPU was at load 0, and the CPU plan needed 8.1x from two
levers that would have underdelivered. Rule 2 again — a number measured in one
context is not a measurement in another, including when the context is *which
pipeline*.

**What this now constrains.** The kernels sum to 26.0 ms of the 28.9 (census+coeffs
1.5, score+hfwd 8.8, hbwd 5.0, vfwd 3.3, vbwd+top2 7.4), so at 30 Hz and full
resolution the GPU is close to saturated. That would have been a contention question
with reason 3 below — "if object detection ends up being a CNN it will want the GPU
essentially to itself" — except that **Mario has ruled out a CNN (2026-08-10)**. So
the device is the matcher's, and the constraint is simpler than it looked: anything
else wanting the GPU has to fit in what dense stereo leaves, and today that is not
much at 848x480.

**What object detection is instead**, given no CNN: the sparse product this page
already routes to the CPU. Detection and tracking come from the same graph machinery
as matching — temporal association is MASDA again — over keypoints and, now, a dense
disparity map that did not exist when this was written. That is CPU work on four
cores that are still idle, and it needs no second producer.

## What the hardware actually gives us

Measured on the TX2, not read off a spec sheet. The "in use" column is the **sparse**
pipeline, which is what the rest of this section was written against:

| resource | total | in use by the sparse pipeline |
|---|---|---|
| CPU cores | 6 (4x Cortex-A57 + 2x Denver) | **2** — the L/R detection threads in `de_preprocess` |
| GPU | 256-core Pascal, CUDA 10.0, `nvcc` present | **0** — `/sys/devices/gpu.0/load` reads 0 |
| frame budget at 30 Hz | 33.3 ms | 23.07 ms (69.3%) |

So for the sparse pipeline four cores and the entire GPU are idle, and there is ~10 ms
of slack. That slack is exactly what the deferral rested on, and it is not the dense
matcher's situation: `de_dense_cuda` runs 26.0 ms of kernels and 23 ms of CPU solve,
overlapped, for 28.9 ms a frame.

One TX2 detail that matters for thread placement: the two Denver cores and the four
A57s are not interchangeable, and `nvpmodel -m 0` is what brings the Denvers online
at all. Power mode 3 left them offline and cost 34% of frames (obstacle 2). Any
scheduling scheme that assumes six equal cores will behave differently across power
modes, so stage-to-core assignment should be explicit rather than left to the
scheduler's discretion.

## The boundary: GPU owns the image plane, CPU owns the graph

The split falls out of what each stage does, not from wanting to use the GPU.

**Dense, regular, per-pixel — GPU.** Every pixel gets the same arithmetic with no
data-dependent branching worth speaking of. Today that is the FAST scan (8.30 ms),
NMS (3.74 ms) and Census (0.25 ms). Later it is anything that consumes the image
rather than the keypoints: dense depth if we ever want it, image warps, ground-plane
fitting over a dense depth map, and CNN inference for object detection.

**Sparse, irregular, sequential — CPU.** MASDA (7.86 ms) is message passing over an
irregular edge list with a data-dependent number of candidates per keypoint and a
sequential dependency between iterations. That is close to the worst case for a GPU
and close to the best case for a core with a good cache. The same is true of
everything downstream: temporal association for tracking is MASDA again, the SLAM
back-end is sparse linear algebra over a graph, ground fitting on a few hundred
points is trivial, and trajectory estimation is a filter.

So the interface between the two halves is **a compact keypoint-plus-descriptor
buffer**, not an image. That is the decision worth making now. It keeps the GPU
boundary narrow, keeps it testable in isolation, and means the CPU stages never see
a frame buffer. Everything downstream consumes the same sparse product:

    GPU:  frame pair -> keypoints + Census descriptors + responses
    CPU:  keypoints  -> matches (+ margin) -> 3D points (+ confidence)
    CPU:  3D points  -> tracking / ground / SLAM / trajectory

**The dense path is the same shape, and it is the one that exists.** `de_dense_cuda`
hands the CPU two scored disparities per pixel — score, disparity and count planes —
which is precisely what the MASDA solver consumes. Not a cost volume, not a filtered
volume, neither of which is ever stored. The boundary predicted the interface a port
written months later actually needed, which is the one piece of evidence that it was
the right boundary.

One thing the port added that this page did not anticipate: **the buffer has to be
ordinary pageable memory.** Every `cudaHostAlloc` flavour is CPU-uncached on Tegra,
and the solver reading uncached candidates measured ~300 ms against ~40. A staged
copy at the pipeline's natural sync point costs ~4 ms and is the fast path. That is a
property of the boundary, not of the port.

## Why nothing was on the GPU until 2026-08-10 — superseded, kept for the reasoning

**Superseded by the decision above.** Reasons 1 and 2 expired when the workload
changed from sparse to dense. Reason 3 expired differently: the contingency it hedged
against was ruled out rather than realised — **no CNN** — so the GPU is not going to
be claimed by inference and the hedge cost nothing either way. Kept because the shape
of the argument is worth re-reading before deferring the next thing.

Three reasons, in order of weight.

1. **There is no deadline pressure.** 23.07 ms of 33.3 ms. Moving the FAST scan to
   the GPU might save 7 of the 8.30 ms, which buys headroom we already have.
2. **The four idle CPU cores are the cheaper win and are not spent.** Tracking,
   ground detection and trajectory estimation are all sparse work on a few hundred
   points. Each can have a core. That is four more stages for zero CUDA.
3. **We do not yet know what the expensive component will be.** If object detection
   ends up being a CNN, it will dominate everything else on this board and it will
   want the GPU essentially to itself. Hand-writing CUDA for FAST now, and then
   discovering the GPU is committed to inference, would be work done twice.
   — *Resolved 2026-08-10: there will be no CNN. The expensive component is dense
   stereo, and it is on the GPU.*

The exception, if one appears: if a component needs *dense* depth — ground detection
over a full depth map rather than over sparse points is the likely candidate — then
that component is dense per-pixel work and belongs on the GPU from the start. It
should not be written on the CPU first.

## What to do now, concretely

Cheap, and expensive to retrofit later:

1. **Make the sparse feature set a first-class, stable output.** Right now every
   tool re-runs detection itself. The moment there are four consumers, that becomes
   four redundant detections. One producer, one buffer, many consumers.
2. **Pin stages to cores explicitly**, given the Denver/A57 asymmetry and what
   power mode 3 did.
3. **Keep the GPU boundary at the keypoint buffer** even while everything runs on
   the CPU, so moving a stage across it later is a substitution rather than a
   redesign.

Deliberately deferred until something needs it:

- ~~Any CUDA at all.~~ Decided 2026-08-10: the matcher's image-plane half is CUDA and
  ships. What is *still* deferred is CUDA for the **sparse** front end — the FAST scan
  and NMS — which nothing has yet shown to need it, and which now competes with the
  dense kernels for the same saturated device rather than for idle silicon.
- Choosing between dense and sparse ground detection, which is really a question
  about what the vehicle needs to see. Cheaper to answer now than when this page was
  written: dense depth at 34.6 Hz is no longer hypothetical, so "dense ground
  detection" costs a consumer of an existing product rather than a new producer.
- A scheduler. A fixed-rate pipeline with bounded queues is enough for four stages,
  and a bounded queue makes back-pressure visible instead of turning it into
  dropped frames — which is the failure mode this project has already paid for once.

## The measurement that would change this

If a future component pushes the total past 33.3 ms, the order of attack is already
known from the profiling in `09-matching.md`: raise `fast_threshold` (30% of
preprocessing for 10% of matches), then move the FAST scan to the GPU, and only
then drop resolution — which costs depth precision directly and is the worst trade
of the three.
