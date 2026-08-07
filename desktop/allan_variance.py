#!/usr/bin/env python3
"""Allan deviation from an ArduPilot dataflash log — IMU noise parameters.

Bring-up step 2. Without these numbers, filter tuning is guesswork, which is why
the plan puts this before calibration and before driving.

## What it computes

The overlapping Allan deviation of the integrated rate, σ(τ), whose shape reads
off three different noise processes by their slope:

  slope −1/2   white noise. This is the *noise density* — Kalibr's
               `gyroscope_noise_density` / `accelerometer_noise_density`. Read as
               σ(τ)·√τ, which is constant across the region.
  minimum      bias instability (flicker). B = σ_min / 0.6642.
  slope +1/2   random walk of the bias — Kalibr's `*_random_walk`. Read as
               σ(τ)·√(3/τ).

## The trap this tool checks for

**Filtered data gives a wrong, flatteringly low noise density.** ArduPilot's
standard `IMU` log messages are written *after* `INS_GYRO_FILTER` (20 Hz here) and
at loop rate (50 Hz). A low-pass removes exactly the high-frequency content the
white-noise estimate is made of, so an Allan deviation computed from them
understates the true density and its short-τ slope is not −1/2.

The batch sampler (`ISBH`/`ISBD`, enabled by `INS_LOG_BAT_MASK`) writes *raw*
samples at the sensor rate. That is the correct source. This tool prefers it,
reports which one it used, and refuses to present a noise density from filtered
data without saying so.

It also reports the **achieved sample rate measured from the log's own
timestamps**, because `INS_LOG_BAT_OPT` semantics vary between firmware versions
and every number here scales with the rate actually used.

Desktop-side. numpy + matplotlib + pymavlink.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymavlink import DFReader

AXES = ("x", "y", "z")


def inventory(path: Path, limit: int | None = None):
    """Message-type counts and time span, plus batch/IMU sample extraction."""
    log = DFReader.DFReader_binary(str(path))
    counts: Counter = Counter()
    imu_t: list[float] = []
    imu_g: list[tuple] = []
    imu_a: list[tuple] = []
    # Batch sampler: ISBH announces a block, ISBD carries 32 samples of it.
    batch: dict[int, dict] = defaultdict(lambda: {"rate": None, "mult": None,
                                                  "samples": [], "blocks": set()})
    pending: dict[int, tuple] = {}
    n = 0
    while True:
        m = log.recv_msg()
        if m is None:
            break
        t = m.get_type()
        counts[t] += 1
        n += 1
        if limit and n > limit:
            break
        if t == "IMU" and getattr(m, "I", 0) == 0:
            imu_t.append(m.TimeUS * 1e-6)
            imu_g.append((m.GyrX, m.GyrY, m.GyrZ))
            imu_a.append((m.AccX, m.AccY, m.AccZ))
        elif t == "ISBH":
            # type 0 = accel, 1 = gyro in ArduPilot's batch sampler.
            # The field is `mul`, not `mult`, and it is a DIVISOR: samples are
            # packed as int16 scaled to full range, so real = raw / mul. For the
            # gyro mul=938 gives 32767/938 = 34.9 rad/s full scale, which is the
            # MPU6000's 2000 deg/s range -- a useful check that the sign of the
            # scaling is right.
            pending[int(m.N)] = (int(m.type), int(getattr(m, "instance", 0)),
                                 float(m.smp_rate), float(m.mul))
        elif t == "ISBD":
            key = int(m.N)
            if key not in pending:
                continue
            stype, inst, rate, mul = pending[key]
            if inst != 0 or mul == 0:
                continue
            slot = batch[stype]
            slot["rate"] = rate
            slot["mult"] = mul
            slot.setdefault("blocks", set()).add(key)
            xs = np.asarray(m.x, dtype=np.float64) / mul
            ys = np.asarray(m.y, dtype=np.float64) / mul
            zs = np.asarray(m.z, dtype=np.float64) / mul
            slot["samples"].append(np.stack([xs, ys, zs], axis=1))

    return counts, np.asarray(imu_t), np.asarray(imu_g), np.asarray(imu_a), batch


def allan_deviation(rate_samples: np.ndarray, dt: float, n_taus: int = 60):
    """Overlapping Allan deviation of the integrated rate signal."""
    theta = np.cumsum(rate_samples) * dt
    n = theta.size
    max_m = (n - 1) // 2
    if max_m < 2:
        return np.empty(0), np.empty(0)
    ms = np.unique(np.round(np.logspace(0, math.log10(max_m), n_taus)).astype(int))
    taus, sigmas = [], []
    for m in ms:
        if 2 * m >= n:
            continue
        tau = m * dt
        d = theta[2 * m:] - 2.0 * theta[m:-m] + theta[:-2 * m]
        var = float(np.sum(d * d)) / (2.0 * tau * tau * d.size)
        if var > 0:
            taus.append(tau)
            sigmas.append(math.sqrt(var))
    return np.asarray(taus), np.asarray(sigmas)


def fit_params(taus, sigmas):
    """Noise density, bias instability and random walk from the curve."""
    out = {}
    if taus.size < 4:
        return out
    # White noise: sigma*sqrt(tau) is constant on the -1/2 slope. Use the short-tau
    # decade, but skip the very first points, which are dominated by whatever
    # anti-alias filtering the sensor applies.
    short = (taus >= taus[0] * 2) & (taus <= max(taus[0] * 20, taus[0] * 2))
    if short.sum() >= 2:
        out["noise_density"] = float(np.median(sigmas[short] * np.sqrt(taus[short])))
    else:
        out["noise_density"] = float(sigmas[0] * math.sqrt(taus[0]))

    i_min = int(np.argmin(sigmas))
    out["bias_instability"] = float(sigmas[i_min] / 0.6642)
    out["tau_min"] = float(taus[i_min])

    # Random walk: sigma = K*sqrt(tau/3) on the +1/2 slope, i.e. beyond the minimum.
    long = taus > taus[i_min] * 2
    if long.sum() >= 2:
        out["random_walk"] = float(np.median(sigmas[long] * np.sqrt(3.0 / taus[long])))
    else:
        out["random_walk"] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, help="ArduPilot .bin dataflash log")
    ap.add_argument("--out", type=Path, default=None,
                    help="plot path (default alongside the log)")
    ap.add_argument("--inventory-only", action="store_true")
    args = ap.parse_args()

    counts, imu_t, imu_g, imu_a, batch = inventory(args.log)

    print(f"log        {args.log}  ({args.log.stat().st_size / 1e6:.2f} MB)")
    print("\nmessage types present (top 12)")
    for name, c in counts.most_common(12):
        print(f"  {name:<8s} {c:8d}")

    print("\n--- what is usable for Allan variance ---")
    have_batch = any(v["samples"] for v in batch.values())
    if imu_t.size > 2:
        span = imu_t[-1] - imu_t[0]
        rate = (imu_t.size - 1) / span if span > 0 else float("nan")
        print(f"  IMU (filtered, loop rate)  {imu_t.size} samples, "
              f"{span:.1f} s, {rate:.1f} Hz measured")
    print(f"  ISBH/ISBD batch sampler    {'present' if have_batch else 'ABSENT'}")

    if not have_batch:
        print("\n  !! No batch-sampler data, so only the FILTERED IMU stream is")
        print("     available. A noise density computed from it is understated,")
        print("     because INS_GYRO_FILTER removed the high-frequency content the")
        print("     estimate is made of. Set INS_LOG_BAT_MASK=1 (and check")
        print("     INS_LOG_BAT_OPT) and re-record before trusting any number.")

    sources = []
    for stype, label, unit in ((1, "gyroscope", "rad/s"),
                               (0, "accelerometer", "m/s^2")):
        slot = batch.get(stype)
        if slot and slot["samples"]:
            data = np.concatenate(slot["samples"], axis=0)
            fs = slot["rate"]
            nblocks = max(1, len(slot.get("blocks", ())))
            sources.append((label, unit, data, fs, "batch (raw)", nblocks))
    if not sources and imu_t.size > 100:
        span = imu_t[-1] - imu_t[0]
        fs = (imu_t.size - 1) / span
        sources.append(("gyroscope", "rad/s", imu_g, fs, "IMU (FILTERED)", 1))
        sources.append(("accelerometer", "m/s^2", imu_a, fs, "IMU (FILTERED)", 1))

    if not sources:
        raise SystemExit("\nno usable inertial samples found in this log")
    if args.inventory_only:
        return 0

    results = {}
    fig, axes = plt.subplots(1, len(sources), figsize=(7 * len(sources), 5),
                             squeeze=False)
    for ax, (label, unit, data, fs, origin, nblocks) in zip(axes[0], sources):
        dt = 1.0 / fs
        # The batch sampler records WINDOWS, not a continuous stream. Concatenating
        # them fabricates continuity across the gaps, which is harmless for tau
        # shorter than one window but invalidates everything beyond it -- and the
        # bias-instability minimum usually sits beyond it. So state the limit.
        block_s = (len(data) / nblocks) * dt if nblocks else len(data) * dt
        tau_trust = block_s / 2.0
        print(f"\n=== {label} ===")
        print(f"  source        {origin}")
        print(f"  samples       {len(data)}  at {fs:.1f} Hz  "
              f"({len(data) / fs:.1f} s)")
        if nblocks > 1:
            print(f"  blocks        {nblocks} windows of ~{block_s:.2f} s each")
            print(f"  !! Windows are NOT contiguous, so results are only valid to")
            print(f"     tau <= {tau_trust:.2f} s. The noise density lives well")
            print(f"     inside that and is fine. Bias instability and random walk")
            print(f"     do NOT -- their tau is beyond the window, so those two")
            print(f"     numbers below are artefacts of concatenation, not")
            print(f"     measurements. They need continuous data.")
        print(f"  mean          " + "  ".join(
            f"{AXES[k]} {np.mean(data[:, k]):+.6f}" for k in range(3)) + f"  {unit}")
        print(f"  std           " + "  ".join(
            f"{AXES[k]} {np.std(data[:, k]):.6f}" for k in range(3)))

        # A stationary accelerometer must read exactly gravity. Free check on
        # scale and calibration, and it matters here because the plan uses the
        # accelerometer for the gravity vector and the ground plane.
        if label == "accelerometer":
            g = float(np.linalg.norm(np.mean(data, axis=0)))
            err = (g - 9.80665) / 9.80665 * 100.0
            print(f"  |gravity|     {g:.4f} m/s^2  ({err:+.2f}% vs 9.80665)")
            if abs(err) > 1.0:
                print("  !! The vehicle is stationary, so this should be 9.807.")
                print("     An error this size is a SCALE/calibration fault, not")
                print("     noise. Run ArduPilot's 6-position accelerometer")
                print("     calibration before relying on the gravity vector for")
                print("     attitude or ground-plane estimation.")

        per_axis = {}
        ax.axvline(tau_trust, color="k", ls=":", lw=1,
                   label=f"valid to {tau_trust:.2f} s" if nblocks > 1 else None)
        for k in range(3):
            taus, sig = allan_deviation(np.asarray(data[:, k], dtype=np.float64), dt)
            if taus.size == 0:
                continue
            p = fit_params(taus, sig)
            per_axis[AXES[k]] = p
            ax.loglog(taus, sig, label=f"{AXES[k]}")
            print(f"  {AXES[k]}: noise density {p['noise_density']:.3e} "
                  f"{unit}/sqrt(Hz),  bias instab {p['bias_instability']:.3e} "
                  f"{unit} @ tau {p['tau_min']:.1f} s,  "
                  f"rw {p.get('random_walk', float('nan')):.3e}")
        results[label] = per_axis
        ax.set_xlabel("averaging time tau [s]")
        ax.set_ylabel(f"Allan deviation [{unit}]")
        ax.set_title(f"{label} — {origin}", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = args.out or args.log.with_suffix(".allan.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nwrote {out}")

    # Kalibr wants the worst axis, since one set of scalars covers all three.
    def worst(label, key):
        vals = [p[key] for p in results.get(label, {}).values()
                if key in p and np.isfinite(p[key])]
        return max(vals) if vals else float("nan")

    print("\n--- Kalibr imu.yaml ---")
    print(f"accelerometer_noise_density: {worst('accelerometer', 'noise_density'):.6e}")
    print(f"accelerometer_random_walk:   {worst('accelerometer', 'random_walk'):.6e}")
    print(f"gyroscope_noise_density:     {worst('gyroscope', 'noise_density'):.6e}")
    print(f"gyroscope_random_walk:       {worst('gyroscope', 'random_walk'):.6e}")
    print("rostopic: /imu0")
    if sources:
        print(f"update_rate: {sources[0][3]:.1f}")
    print("\n(worst of the three axes for each, since Kalibr takes one scalar each)")

    if any(s[5] > 1 for s in sources):
        print("\n!! Only `*_noise_density` above is a measurement. The random-walk")
        print("   figures come from tau beyond the batch window and are artefacts.")
    if any("FILTERED" in s[4] for s in sources):
        print("\n!! These came from FILTERED data and understate the noise")
        print("   density. Do not put them in a Kalibr yaml. Fix the batch")
        print("   logging first.")
    span_s = len(sources[0][2]) / sources[0][3]
    if span_s < 3600:
        print(f"\nNote: only {span_s / 60:.0f} minutes of data. Noise density needs"
              " minutes and is")
        print("      already reliable; bias instability and random walk need the"
              " curve's")
        print("      minimum to be inside the record, so they want several hours.")
        print("      Re-run on a longer log before trusting those two.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
