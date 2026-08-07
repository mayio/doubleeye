# Architecture: what runs where

The system is meant to grow object tracking, SLAM, ground detection and trajectory
estimation on top of stereo correspondence. That raises two questions worth
separating, because they have different answers:

- **What is the boundary between GPU and CPU?** Decide now. It is one page of
  reasoning and it is expensive to retrofit, because it determines what the stages
  hand each other.
- **When do we write CUDA?** Later. Nothing currently needs it, and writing it now
  would be optimising against a budget that has 10 ms spare.

## What the hardware actually gives us

Measured on the TX2, not read off a spec sheet:

| resource | total | in use today |
|---|---|---|
| CPU cores | 6 (4x Cortex-A57 + 2x Denver) | **2** — the L/R detection threads in `de_preprocess` |
| GPU | 256-core Pascal, CUDA 10.0, `nvcc` present | **0** — `/sys/devices/gpu.0/load` reads 0 |
| frame budget at 30 Hz | 33.3 ms | 23.07 ms (69.3%) |

So four cores and the entire GPU are idle, and there is ~10 ms of slack. We are not
short of compute; we have never used most of it.

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

## Why not GPU-accelerate anything yet

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

- Any CUDA at all.
- Choosing between dense and sparse ground detection, which is really a question
  about what the vehicle needs to see.
- A scheduler. A fixed-rate pipeline with bounded queues is enough for four stages,
  and a bounded queue makes back-pressure visible instead of turning it into
  dropped frames — which is the failure mode this project has already paid for once.

## The measurement that would change this

If a future component pushes the total past 33.3 ms, the order of attack is already
known from the profiling in `09-matching.md`: raise `fast_threshold` (30% of
preprocessing for 10% of matches), then move the FAST scan to the GPU, and only
then drop resolution — which costs depth precision directly and is the worst trade
of the three.
