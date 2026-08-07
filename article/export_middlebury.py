#!/usr/bin/env python3
"""Export the Middlebury scenes for the C++ benchmark.

The accuracy results so far were all measured in Python, with a dense Shi-Tomasi
detector at cell 12. The C++ matcher on the Jetson uses FAST candidates with
Shi-Tomasi scoring at cell 32. Those are different detectors in a different
density regime, so nothing in the Python results transfers to the C++ path by
argument alone -- it has to be measured there, against the same ground truth.

This writes, per scene:

  left.y8    width*height uint8, luma of the left view
  right.y8   same, right view
  disp.f32   width*height float32, true disparity in pixels, 0 where unknown
  meta.txt   "width height dmax"

Raw rather than PNG because core/ has no image decoder and should not acquire one
for a benchmark. Y8 is what load_raw_y8 already reads, and the bag format the
capture tools write.

  python export_middlebury.py
"""

import os

import numpy as np

import masda_stereo as ms
import ordering_real as orr

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "c_bench")
SCENES = ("teddy", "cones") + orr.SCENES_2005


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in SCENES:
        left, right, gt, known, _ = orr.load_scene(name)
        H, W = left.shape
        dmax = 60.0 if name in ("teddy", "cones") else 80.0
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)

        # Round rather than truncate: the images are already 8-bit, the luma
        # conversion is the only thing that made them float.
        np.clip(np.round(left), 0, 255).astype(np.uint8).tofile(
            os.path.join(d, "left.y8"))
        np.clip(np.round(right), 0, 255).astype(np.uint8).tofile(
            os.path.join(d, "right.y8"))
        # Unknown ground truth is written as 0, matching Middlebury's own
        # convention, so the C++ side applies the same exclusion rule.
        np.where(known, gt, 0.0).astype(np.float32).tofile(
            os.path.join(d, "disp.f32"))
        with open(os.path.join(d, "meta.txt"), "w") as f:
            f.write(f"{W} {H} {dmax:.0f}\n")
        print(f"{name:<10} {W}x{H}  dmax={dmax:.0f}  "
              f"known={100*known.mean():.1f}%  "
              f"disp[{gt[known].min():.2f}, {gt[known].max():.2f}]")
    print(f"\nwrote {len(SCENES)} scenes to {OUT}")


if __name__ == "__main__":
    main()
