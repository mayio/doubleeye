#!/usr/bin/env python3
"""Regenerate every number and figure the article quotes, from one command.

The article previously drew its numbers from several ad-hoc scripts, and its
seeding went through Python's builtin hash(), which is salted per process. Both
together meant the published tables could not be reproduced -- and a conclusion
about the ordering factor turned out to be a comparison between two different
random scenes. This script is the fix: one entry point, stable seeds, and a JSON
dump so a changed number is visible in a diff.

  python regen_all.py            # everything
  python regen_all.py --quick    # skip the dense/JV timing sweep
"""

import json
import sys
import time

import numpy as np

import masda_stereo as ms
import masda_middlebury as mb
import ordering_real

LAM = GAM = -0.1
ORDER_SEEDS = [20251126, 1, 2, 3, 4]      # offsets applied to BASE_SEED
KAPPAS = [0.1, 0.4, 0.8]
OUT = {}


def build(texture_fn, disp_fn=None, dmax=60.0):
    """Detect, describe and gate one synthetic pair. Returns everything needed."""
    disp = (disp_fn or ms.ground_truth_disparity)()
    left, right, valid = ms.render_pair(texture_fn(), disp)
    pl, _ = ms.detect(left)
    pr, _ = ms.detect(right)
    dl, dr = ms.census(left, pl), ms.census(right, pr)
    S = ms.build_problem(pl, dl, pr, dr, dmax=dmax)
    return dict(disp=disp, left=left, right=right, valid=valid,
                pl=pl, pr=pr, S=S, m=S.shape[0], n=S.shape[1])


# ---------------------------------------------------------------------------

def synthetic_results():
    print("\n" + "=" * 78)
    print("SYNTHETIC: results against ground truth")
    print("=" * 78)
    OUT["synthetic"] = {}
    for fn, label in ((ms.broadband_texture, "broadband"),
                      (ms.dot_texture, "dots"),
                      (ms.periodic_texture, "periodic")):
        r = ms.run_regime(fn, label)
        mg = r["margin"]
        row = dict(kp=r["kp"], edges=r["edges"], uniq=r["uniq"],
                   margin=float(np.median(mg)),
                   tied=float((mg < 0.05).mean()),
                   matchable=r["rows"]["MASDA"]["matchable"], methods={})
        print(f"\n--- {label}: kp={r['kp']} edges={r['edges']} "
              f"({r['edges']/max(1,r['kp']):.2f}/kp) distinct desc="
              f"{r['uniq']}/{r['kp']}")
        print(f"    margin median={row['margin']:.4f} tied(<0.05)={row['tied']:.3f}"
              f"  matchable={row['matchable']}")
        print(f"    {'method':<14}{'matches':>8}{'correct':>8}{'wrong':>7}"
              f"{'prec':>7}{'recall':>8}{'obj':>10}")
        for name, v in r["rows"].items():
            row["methods"][name] = {k: v[k] for k in
                                    ("matches", "tp", "fp", "precision",
                                     "recall", "objective")}
            print(f"    {name:<14}{v['matches']:>8}{v['tp']:>8}{v['fp']:>7}"
                  f"{v['precision']:>7.3f}{v['recall']:>8.3f}"
                  f"{v['objective']:>10.2f}")
        OUT["synthetic"][label] = row
        OUT.setdefault("_regimes", {})[label] = r
    return OUT["_regimes"]


def timing(regimes):
    print("\n" + "=" * 78)
    print("SPEED: dense MASDA vs sparse MASDA vs Jonker-Volgenant")
    print("=" * 78)
    print(f"{'regime':<12}{'nodes':>7}{'edges':>7}{'dense ms':>10}"
          f"{'sparse ms':>11}{'JV ms':>9}{'x dense':>9}{'x JV':>7}"
          f"{'obj/opt':>9}{'correct d/s/o':>15}")
    OUT["timing"] = {}
    for label, r in regimes.items():
        S = r["S"]
        m, n = S.shape
        ei, ej, se = ms.to_edges(S)
        t = time.perf_counter(); a_d, _ = ms.masda(S, LAM, GAM)
        t_d = (time.perf_counter() - t) * 1e3
        t = time.perf_counter(); a_s, _ = ms.masda_sparse(ei, ej, se, m, n, LAM, GAM)
        t_s = (time.perf_counter() - t) * 1e3
        t = time.perf_counter(); a_o = ms.optimal_lap(S, LAM, GAM)
        t_o = (time.perf_counter() - t) * 1e3
        ev = [ms.evaluate(a, r["pl"], r["pr"], r["disp"], r["valid"])["tp"]
              for a in (a_d, a_s, a_o)]
        obj_s = ms.objective(a_s, S, LAM, GAM, m, n)
        obj_o = ms.objective(a_o, S, LAM, GAM, m, n)
        row = dict(nodes=m + n, edges=len(se), dense_ms=t_d, sparse_ms=t_s,
                   jv_ms=t_o, x_dense=t_d / t_s, x_jv=t_o / t_s,
                   obj_ratio=obj_s / obj_o, correct=ev)
        OUT["timing"][label] = row
        print(f"{label:<12}{m+n:>7}{len(se):>7}{t_d:>10.0f}{t_s:>11.1f}"
              f"{t_o:>9.0f}{row['x_dense']:>9.0f}{row['x_jv']:>7.0f}"
              f"{row['obj_ratio']:>9.4f}{'/'.join(map(str, ev)):>15}")


def ordering_sweep():
    """The claim that burned me once: measure it across seeds, not on one scene."""
    print("\n" + "=" * 78)
    print("ORDERING FACTOR: across seeds, so the answer is not scene noise")
    print("=" * 78)
    OUT["ordering"] = {}
    for scene_label, texture_fn, disp_fn in (
            ("thin bars", ms.broadband_texture, ms.thin_bars_disparity),
            ("periodic", ms.periodic_texture, ms.ground_truth_disparity)):
        per_seed = []
        for si, seed in enumerate(ORDER_SEEDS):
            ms.BASE_SEED = seed
            b = build(texture_fn, disp_fn)
            S, pl, pr = b["S"], b["pl"], b["pr"]
            m, n = S.shape
            ei, ej, se = ms.to_edges(S)
            rec = {}
            a, _ = ms.masda_sparse(ei, ej, se, m, n, LAM, GAM)
            ev = ms.evaluate(a, pl, pr, b["disp"], b["valid"])
            cr, tot = ms.count_crossings(a, pl, pr)
            rec["off"] = dict(correct=ev["tp"], matches=ev["matches"],
                              prec=ev["precision"], crossings=cr, pairs=tot)
            for k in KAPPAS:
                ao, _, _ = ms.masda_sparse_ordering(ei, ej, se, m, n, pl, pr,
                                                    kappa=k, lam=LAM, gam=GAM)
                e2 = ms.evaluate(ao, pl, pr, b["disp"], b["valid"])
                c2, _ = ms.count_crossings(ao, pl, pr)
                rec[f"k{k}"] = dict(correct=e2["tp"], matches=e2["matches"],
                                    prec=e2["precision"], crossings=c2)
            per_seed.append(rec)
            print(f"  {scene_label} seed{si}: off correct={rec['off']['correct']} "
                  f"cross={rec['off']['crossings']}/{rec['off']['pairs']}  " +
                  "  ".join(f"k={k}:{rec[f'k{k}']['correct']}/"
                            f"{rec[f'k{k}']['crossings']}" for k in KAPPAS))
        ms.BASE_SEED = ORDER_SEEDS[0]
        # Aggregate: what the ordering factor does to correctness and crossings.
        agg = {}
        for key in ["off"] + [f"k{k}" for k in KAPPAS]:
            c = np.array([p[key]["correct"] for p in per_seed], float)
            x = np.array([p[key]["crossings"] for p in per_seed], float)
            agg[key] = dict(correct_mean=c.mean(), correct_sd=c.std(),
                            cross_mean=x.mean(), cross_sd=x.std())
        d = np.array([[p[f"k{k}"]["correct"] - p["off"]["correct"]
                       for k in KAPPAS] for p in per_seed], float)
        dx = np.array([[p[f"k{k}"]["crossings"] - p["off"]["crossings"]
                        for k in KAPPAS] for p in per_seed], float)
        OUT["ordering"][scene_label] = dict(per_seed=per_seed, agg=agg,
                                            d_correct=d.tolist(),
                                            d_cross=dx.tolist())
        print(f"  {scene_label}: over {len(ORDER_SEEDS)} seeds, "
              f"off correct={agg['off']['correct_mean']:.1f}"
              f"+-{agg['off']['correct_sd']:.1f}, "
              f"crossings={agg['off']['cross_mean']:.1f}"
              f"+-{agg['off']['cross_sd']:.1f}")
        for i, k in enumerate(KAPPAS):
            print(f"      k={k}: d(correct)={d[:, i].mean():+.1f}"
                  f"+-{d[:, i].std():.1f}   d(crossings)={dx[:, i].mean():+.1f}"
                  f"+-{dx[:, i].std():.1f}")


def real_results():
    print("\n" + "=" * 78)
    print("REAL: Middlebury 2003 Teddy and Cones")
    print("=" * 78)
    OUT["real"] = {}
    rs = []
    for scene in mb.SCENES:
        r = mb.run_scene(scene)
        rs.append(r)
        OUT["real"][scene] = {k: v for k, v in r.items()
                              if k != "_arrays" and not isinstance(v, np.ndarray)}
    return rs


def figures(regimes, reals):
    print("\n" + "=" * 78)
    print("FIGURES")
    print("=" * 78)
    ms.fig_scene(regimes["periodic"])
    ms.fig_descriptors(regimes["periodic"], regimes["broadband"])
    ms.fig_associations(regimes["periodic"], "periodic")
    ms.fig_associations(regimes["broadband"], "broadband")
    ms.fig_comparison(regimes["periodic"], regimes["broadband"], regimes["dots"])
    ms.fig_damping(regimes["periodic"]["S"], LAM, GAM)
    for r in reals:
        mb.fig_real(r, r["scene"])

    def teddy_regions(pl, kg, kk):
        bw = kg < 20
        return [("Teddy (all)", np.ones(len(pl), bool)),
                ("Teddy wall, printed", bw & (pl[:, 0] >= 240) & (pl[:, 1] < 190)),
                ("Teddy wall, plain", bw & (pl[:, 0] < 240) & (pl[:, 1] < 190))]

    pts = [dict(kind="real", label=k, **v)
           for k, v in mb.region_stats("teddy", teddy_regions).items()]
    pts += [dict(kind="real", label=k, **v) for k, v in mb.region_stats(
        "cones", lambda pl, kg, kk: [("Cones (all)", np.ones(len(pl), bool))]).items()]
    for label, r in regimes.items():
        pts.append(dict(kind="synthetic", label=label,
                        margin=float(np.median(r["margin"])),
                        prec=r["rows"]["MASDA"]["precision"],
                        n=r["rows"]["MASDA"]["matches"]))
    pts.sort(key=lambda p: p["margin"])
    print("\n  margin vs precision, both data sources on one axis:")
    print(f"  {'regime':<22}{'kind':<11}{'margin':>8}{'prec':>8}{'n':>7}")
    for p in pts:
        print(f"  {p['label']:<22}{p['kind']:<11}{p['margin']:>8.4f}"
              f"{p['prec']:>8.3f}{p['n']:>7}")
    OUT["margin_vs_precision"] = pts
    mb.fig_margin_vs_precision(pts)


def main():
    quick = "--quick" in sys.argv
    ms.BASE_SEED = ORDER_SEEDS[0]
    regimes = synthetic_results()
    if not quick:
        timing(regimes)
    ordering_sweep()
    print("\n" + "=" * 78)
    print("ORDERING ON REAL DATA: eight Middlebury scenes")
    print("=" * 78)
    ordering_real.main()          # writes ordering_real.json
    reals = real_results()
    figures(regimes, reals)
    OUT.pop("_regimes", None)
    with open("results.json", "w") as f:
        json.dump(OUT, f, indent=1, default=float)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
