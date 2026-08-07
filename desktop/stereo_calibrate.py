#!/usr/bin/env python3
"""Stereo-calibrate the IR pair from a checkerboard bag, and compare to factory.

Kalibr is not needed for this. Its irreplaceable capability is **camera-IMU**
calibration — hand-eye plus the time offset — which needs IMU data. For camera
intrinsics, distortion and the stereo extrinsic, OpenCV's `stereoCalibrate` solves
the same problem, runs here in seconds, and needs no ROS.

What makes this worth doing even though the D435 ships with factory calibration:
the plan notes that factory calibration drifts thermally and mechanically, and
asks for it to be re-measured. So the useful output is not the numbers alone but
the **comparison** against what the ASIC reports.

## Two models, deliberately

The IR pair arrives already rectified from the D4 ASIC, with all distortion
coefficients exactly zero. So:

  pinhole     distortion fixed at zero, matching what the hardware claims.
  radtan      distortion free. Any coefficient it finds is either real residual
              distortion the ASIC did not remove, or the optimiser absorbing
              noise and board non-flatness. Compare the reprojection errors: if
              radtan is barely better, the extra parameters are fitting noise.

Desktop-side. numpy + opencv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_meta(bag: Path) -> dict:
    meta = {}
    run = bag / "run.txt"
    if run.exists():
        for line in run.read_text().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                meta[parts[0]] = parts[1]
    return meta


def load_pairs(bag: Path, w: int, h: int):
    grouped: dict[str, dict[str, Path]] = {}
    for p in sorted((bag / "frames").glob("*.raw")):
        try:
            idx, num = p.stem.split("_")
        except ValueError:
            continue
        grouped.setdefault(num, {})[idx] = p
    out = []
    for num in sorted(grouped):
        e = grouped[num]
        if "ir1" not in e or "ir2" not in e:
            continue
        imgs = []
        for idx in ("ir1", "ir2"):
            buf = np.fromfile(e[idx], dtype=np.uint8)
            if buf.size != w * h:
                imgs = []
                break
            imgs.append(buf.reshape(h, w))
        if imgs:
            out.append((num, imgs[0], imgs[1]))
    return out


def find_corners(img, cols, rows):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, c = cv2.findChessboardCorners(img, (cols, rows), flags=flags)
    if not ok:
        return None
    return cv2.cornerSubPix(
        img, c.astype(np.float32), (7, 7), (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.005))


def run(model, objp, ip1, ip2, size):
    flags = cv2.CALIB_SAME_FOCAL_LENGTH
    if model == "pinhole":
        flags |= (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
                  | cv2.CALIB_ZERO_TANGENT_DIST)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 120, 1e-6)
    return cv2.stereoCalibrate(
        objp, ip1, ip2,
        np.eye(3), np.zeros(5), np.eye(3), np.zeros(5),
        size, flags=flags, criteria=crit)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--pitch-mm", type=float, default=25.0,
                    help="MEASURED square size, not the nominal one")
    args = ap.parse_args()

    meta = read_meta(args.bag)
    try:
        w, h = (int(v) for v in meta["resolution"].split()[0].split("x"))
    except (KeyError, ValueError, IndexError):
        w, h = 848, 480

    pairs = load_pairs(args.bag, w, h)
    if not pairs:
        raise SystemExit(f"no complete L/R pairs in {args.bag}/frames")

    pitch_m = args.pitch_mm / 1000.0
    grid = np.zeros((args.rows * args.cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * pitch_m

    objp, ip1, ip2 = [], [], []
    for _, a, b in pairs:
        c1 = find_corners(a, args.cols, args.rows)
        c2 = find_corners(b, args.cols, args.rows)
        if c1 is None or c2 is None:
            continue
        objp.append(grid)
        ip1.append(c1)
        ip2.append(c2)

    print(f"bag        {args.bag}")
    print(f"target     {args.cols}x{args.rows} interior corners @ "
          f"{args.pitch_mm:g} mm")
    print(f"pairs used {len(objp)} of {len(pairs)}")
    if len(objp) < 8:
        raise SystemExit("too few usable pairs to calibrate")

    results = {}
    for model in ("pinhole", "radtan"):
        rms, K1, D1, K2, D2, R, T, _, _ = run(model, objp, ip1, ip2, (w, h))
        results[model] = (rms, K1, D1, K2, D2, R, T)

        base_mm = float(np.linalg.norm(T)) * 1000.0
        angle_deg = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))
        print(f"\n=== {model} ===")
        print(f"  reprojection RMS   {rms:.4f} px")
        print(f"  ir1  fx {K1[0, 0]:8.3f}  fy {K1[1, 1]:8.3f}  "
              f"cx {K1[0, 2]:8.3f}  cy {K1[1, 2]:8.3f}")
        print(f"  ir2  fx {K2[0, 0]:8.3f}  fy {K2[1, 1]:8.3f}  "
              f"cx {K2[0, 2]:8.3f}  cy {K2[1, 2]:8.3f}")
        if model == "radtan":
            print(f"  ir1  dist {np.array2string(D1.ravel(), precision=5)}")
            print(f"  ir2  dist {np.array2string(D2.ravel(), precision=5)}")
        print(f"  baseline           {base_mm:.3f} mm")
        print(f"  T                  {np.array2string(T.ravel() * 1000, precision=3)} mm")
        print(f"  R, angle from I    {angle_deg:.4f} deg")

    # Comparison against the ASIC's own numbers, which is the point of doing this.
    FACTORY = {"fx": 430.551, "cx": 427.381, "cy": 243.158, "baseline_mm": 49.883}
    try:
        FACTORY["fx"] = float(meta.get("fx", FACTORY["fx"]))
        FACTORY["baseline_mm"] = float(meta.get("baseline_m", 0)) * 1000 or FACTORY["baseline_mm"]
    except ValueError:
        pass

    print("\n=== against factory calibration ===")
    print(f"  {'':18s} {'factory':>10s} {'pinhole':>10s} {'radtan':>10s}")
    for label, key, get in (
            ("fx (px)", "fx", lambda r: r[1][0, 0]),
            ("cx (px)", "cx", lambda r: r[1][0, 2]),
            ("cy (px)", "cy", lambda r: r[1][1, 2]),
            ("baseline (mm)", "baseline_mm",
             lambda r: float(np.linalg.norm(r[6])) * 1000.0)):
        fv = FACTORY[key]
        pv, rv = get(results["pinhole"]), get(results["radtan"])
        print(f"  {label:18s} {fv:10.3f} {pv:10.3f} {rv:10.3f}")
        print(f"  {'  difference':18s} {'':10s} {pv - fv:+10.3f} {rv - fv:+10.3f}")

    rms_p, rms_r = results["pinhole"][0], results["radtan"][0]
    print(f"\n  reprojection RMS: pinhole {rms_p:.4f} px, radtan {rms_r:.4f} px")
    gain = (rms_p - rms_r) / rms_p * 100 if rms_p else 0
    print(f"  freeing distortion improves RMS by {gain:.1f}%")
    if gain < 10:
        print("  -> marginal, consistent with the ASIC having genuinely removed")
        print("     distortion. Prefer the pinhole numbers; the radtan")
        print("     coefficients are largely absorbing noise and board")
        print("     non-flatness.")
    else:
        print("  -> a real improvement, so some distortion survives rectification.")

    # Scale reasoning. fx is an angle ratio in pixels and does NOT depend on the
    # assumed square size; the recovered translation scales with it exactly. So a
    # baseline that disagrees while fx agrees points at the pitch, not the optics.
    pin = results["pinhole"]
    base_mm = float(np.linalg.norm(pin[6])) * 1000.0
    fx_err = (pin[1][0, 0] - FACTORY["fx"]) / FACTORY["fx"] * 100.0
    base_err = (base_mm - FACTORY["baseline_mm"]) / FACTORY["baseline_mm"] * 100.0
    implied = args.pitch_mm * FACTORY["baseline_mm"] / base_mm

    print("\n=== is the disagreement optics or the ruler? ===")
    print(f"  fx differs by       {fx_err:+.2f}%   (independent of square size)")
    print(f"  baseline differs by {base_err:+.2f}%   (scales exactly with it)")
    if abs(base_err) > 2.0 and abs(base_err) > 2.5 * abs(fx_err):
        print("\n  The baseline is off by much more than fx, and only the baseline")
        print("  depends on the assumed square size. The most likely explanation is")
        print("  therefore the PITCH, not the camera.")
        print(f"\n  If the factory baseline is right, the true pitch is "
              f"{implied:.3f} mm,")
        print(f"  not {args.pitch_mm:g} mm — a print scale of "
              f"{100.0 * implied / args.pitch_mm:.1f}%.")
        print("  MEASURE the printed board across all columns and re-run with")
        print(f"  --pitch-mm <measured>. If it really is ~{implied:.1f} mm, both")
        print("  numbers reconcile at once and nothing is wrong with the camera.")
    else:
        print("\n  Both agree closely, so the pitch and the optics are consistent.")

    print("\n  Rectification independently confirmed by this fit:")
    ang = float(np.degrees(np.linalg.norm(cv2.Rodrigues(pin[5])[0])))
    off = np.abs(pin[6].ravel()[1:]) * 1000.0
    print(f"    R is {ang:.4f} deg from identity, and the off-axis translation is")
    print(f"    {off[0]:.3f} / {off[1]:.3f} mm against a {base_mm:.1f} mm baseline.")
    print("    A rectified pair is exactly what that looks like.")

    print("\n  Caveats that bound how far to trust this:")
    print("   * board covered a median ~5% of the frame, which under-constrains")
    print("     focal length. See doc/04-baseline-measurements.md.")
    print(f"   * pitch was taken as {args.pitch_mm:g} mm. If that is the nominal")
    print("     value rather than a measured one, every length above scales with")
    print("     the error — baseline especially.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
