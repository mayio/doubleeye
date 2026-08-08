#!/usr/bin/env python3
"""Interleaved A/B timing of de_dense on the Jetson. Standard library only.

Why this exists rather than a shell loop: **TX2 run-to-run variance is 37%** at
stable clocks and 35 C (09-matching.md), so two configurations must be alternated
within one session and compared on their minima. Two batches compared on their means
will confidently report an effect of either sign. Three results in 09-matching.md
turned on this and one earlier conclusion had to be withdrawn because of it.

    tools/tx2_ab.py "" "--simd"                  # scalar against the NEON kernel
    tools/tx2_ab.py "" "--simd" -n 8             # more repetitions, quieter answer
    tools/tx2_ab.py "--agg 3" "--agg 5"          # any two flag sets
    tools/tx2_ab.py "--csct"                     # one config, just the spread

A and B are flag strings passed through to de_dense verbatim. They are POSITIONAL on
purpose: as `--b "--simd"` argparse would take `--simd` for one of its own options,
which is a five-minute confusion every single time.

Reports min and median per stage. **Quote the min**: the scheduler only ever adds
time, so the distribution is one-sided and its minimum is the least-disturbed
estimate. The spread is printed too, because a difference smaller than the spread is
not a result -- report spread, not a single run.

Environment, same names as deploy.sh:
  DOUBLEEYE_HOST        ssh alias or host   (default: jetson)
  DOUBLEEYE_REMOTE_DIR  path under $HOME    (default: doubleeye)
"""

import argparse
import os
import re
import statistics
import subprocess
import sys

HOST = os.environ.get("DOUBLEEYE_HOST", "jetson")
RD = os.environ.get("DOUBLEEYE_REMOTE_DIR", "doubleeye")
REMOTE = f"~/{RD}"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOTAL = re.compile(
    r"census ([\d.]+) ms\s+cost ([\d.]+) ms\s+solve ([\d.]+) ms\s+total ([\d.]+) ms")
BREAK = re.compile(
    r"cost breakdown \(thread-summed\): alloc ([\d.]+)\s+clear ([\d.]+)\s+"
    r"score ([\d.]+)\s+filter ([\d.]+)\s+insert ([\d.]+) ms")
FILLED = re.compile(r"filled ([\d.]+)%")

# Wall-clock stages first, then the thread-summed CPU breakdown of the cost stage.
KEYS = ["census", "cost", "solve", "total",
        "c_alloc", "c_clear", "c_score", "c_filter", "c_insert", "filled"]
CPU_KEYS = ["c_alloc", "c_clear", "c_score", "c_filter", "c_insert"]


def ssh(cmd, check=True):
    p = subprocess.run(["ssh", HOST, cmd], capture_output=True, text=True)
    if check and p.returncode != 0:
        sys.exit(f"ssh {HOST} failed (exit {p.returncode}): {p.stderr.strip()}")
    return p


def preflight(scene, nthreads):
    """Everything that silently invalidates a timing run, checked before running one.

    Every item here has cost this project a measurement session. Unlocked clocks are
    a 34% frame loss with no error message; a stale binary relinks while printing
    success; and a `jetson_clocks` setting does not survive a reboot.
    """
    print(f"host {HOST}, remote {REMOTE}")
    p = ssh("uname -m; nproc")
    arch, cores = p.stdout.split()
    if arch != "aarch64":
        sys.exit(f"{HOST} reports {arch}, not aarch64 -- this measures the target only")
    if int(nthreads) > int(cores):
        print(f"  NOTE: --threads {nthreads} on {cores} cores, oversubscribed")

    freqs = ssh("cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq").stdout.split()
    locked = len(set(freqs)) == 1
    # Zones are reported BY NAME rather than as a bare maximum. On this board
    # PMIC-Die reads a constant 100000 -- observed unchanged across every read taken
    # here, and 09-matching.md's measurements were made at 35 C -- so a max over all
    # zones prints "100 C" and looks like a board about to throttle. Naming the CPU
    # and board zones says what was actually measured.
    zones = {}
    for line in ssh("for z in /sys/devices/virtual/thermal/thermal_zone*; do "
                    "echo \"$(cat $z/type) $(cat $z/temp)\"; done").stdout.splitlines():
        name, _, val = line.rpartition(" ")
        if val.strip().lstrip("-").isdigit():
            zones[name.strip()] = int(val) / 1000.0
    hot = {k: v for k, v in zones.items() if k in ("BCPU-therm", "MCPU-therm",
                                                   "GPU-therm", "Tboard_tegra")}
    svc = ssh("systemctl is-active doubleeye-performance.service", check=False)
    svc_ok = svc.stdout.strip() == "active"
    print(f"  {arch}, {cores} cores, {int(freqs[0])//1000} MHz"
          f"{'' if locked else ' (NOT UNIFORM)'}, "
          f"performance.service {svc.stdout.strip()}")
    print("  " + ", ".join(f"{k.split('-')[0].split('_')[0]} {v:.0f} C"
                           for k, v in sorted(hot.items()))
          + "   (PMIC-Die excluded: reads a constant 100 C here)")
    if hot and max(hot.values()) > 60:
        print(f"  WARNING: {max(hot, key=hot.get)} at {max(hot.values()):.0f} C. "
              f"The docs' numbers were taken at ~35 C.", file=sys.stderr)
    if not locked or not svc_ok:
        # Not fatal: measuring an unlocked board is legitimate if that is the
        # question. It must not be silent, because it looks exactly like a
        # regression. jetson_clocks does not survive a reboot.
        print("  WARNING: clocks are not locked / performance.service is not active.\n"
              "           Timings will not be comparable with anything in the docs.",
              file=sys.stderr)

    binp = f"{REMOTE}/core/build/de_dense"
    if ssh(f"test -x {binp}", check=False).returncode != 0:
        sys.exit(f"{binp} is missing -- run tools/deploy.sh first")
    # Obstacle 10: a source newer than the binary means the run measures the previous
    # edit while every signal points at the current one.
    remote_mtime = int(ssh(f"stat -c %Y {binp}").stdout.strip())
    newest, newest_f = 0, None
    for root, _, files in os.walk(os.path.join(HERE, "core")):
        if os.sep + "build" in root:
            continue
        for f in files:
            if f.endswith((".cpp", ".hpp", "Makefile")):
                m = os.path.getmtime(os.path.join(root, f))
                if m > newest:
                    newest, newest_f = m, os.path.relpath(os.path.join(root, f), HERE)
    if newest > remote_mtime:
        sys.exit(f"local {newest_f} is newer than {HOST}'s de_dense.\n"
                 f"Run tools/deploy.sh, or you will time the previous edit.")

    d = f"{REMOTE}/article/data/c_bench/{scene}"
    if ssh(f"test -f {d}/meta.txt", check=False).returncode != 0:
        sys.exit(f"{d}/meta.txt is missing. The Middlebury scenes are not synced by\n"
                 f"deploy.sh; copy them once with:\n"
                 f"  tar czf - article/data/c_bench | "
                 f"ssh {HOST} 'tar xzf - -m -C {REMOTE}'")
    return d


def one(d, threads, extra):
    W, H, dmax = ssh(f"cat {d}/meta.txt").stdout.split()
    cmd = (f"cd {REMOTE}/core && ./build/de_dense {d}/left.y8 {d}/right.y8 {W} {H} "
           f"--dmax {dmax} --threads {threads} --agg 5 --iters 2 "
           f"--min-margin 0.01 {extra}")
    p = subprocess.run(["ssh", HOST, cmd], capture_output=True, text=True)
    if p.returncode != 0:
        # Rule 1: a producer that failed must not read as a timing.
        sys.exit(f"de_dense failed (exit {p.returncode}) with flags '{extra}':\n"
                 f"{p.stderr}{p.stdout}")
    t, b, f = TOTAL.search(p.stdout), BREAK.search(p.stdout), FILLED.search(p.stdout)
    if not (t and b and f):
        sys.exit(f"could not parse de_dense output -- has the format changed?\n{p.stdout}")
    return dict(zip(KEYS, [float(x) for x in t.groups() + b.groups() + f.groups()]))


def split_argv(argv):
    """Insert argparse's `--` separator before the first positional.

    A flag string like "--simd" is a positional here, but argparse sees any token
    starting with `-` as one of its own options and refuses. Requiring the caller to
    type `tx2_ab.py "" -- "--simd"` is the standard answer and is forgotten every
    time, so the separator is inserted here instead.

    Options may appear on either side of the flag strings, so this partitions rather
    than splitting at the first positional: `tx2_ab.py "" "--simd" -n 8` works.
    """
    takes_value = {"-n", "--scene", "--threads"}
    opts, pos, i = [], [], 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--":                       # caller supplied it; rest is positional
            pos += argv[i + 1:]
            break
        if tok in ("-h", "--help"):
            opts.append(tok)
            i += 1
        elif tok in takes_value:
            opts += argv[i:i + 2]
            i += 2
        elif tok.split("=")[0] in takes_value:
            opts.append(tok)
            i += 1
        else:
            pos.append(tok)
            i += 1
    return opts + (["--"] + pos if pos else [])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog='A and B are positional flag strings: tx2_ab.py "" "--simd"',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", metavar="A", help='flags for config A, e.g. "" for defaults')
    ap.add_argument("b", metavar="B", nargs="?", default=None,
                    help="flags for config B; omit to measure A's spread alone")
    ap.add_argument("-n", type=int, default=6, help="repetitions each (default 6)")
    ap.add_argument("--scene", default="teddy")
    ap.add_argument("--threads", default="6")
    a = ap.parse_args(split_argv(sys.argv[1:]))

    d = preflight(a.scene, a.threads)
    order = ["A"] if a.b is None else ["A", "B"]
    flags = {"A": a.a, "B": a.b}
    runs = {k: [] for k in order}

    print(f"\n{a.scene}, {a.threads} threads, n={a.n}, interleaved")
    for i in range(a.n):
        for k in order:
            runs[k].append(one(d, a.threads, flags[k]))
            print(f"  {k} {i+1}/{a.n}: total {runs[k][-1]['total']:7.1f} ms", flush=True)

    label = {k: (flags[k] or "(defaults)") for k in order}
    print()
    print(f"{'':<10}" + "".join(f"{label[k]:>20}" for k in order))
    print(f"{'':<10}" + "".join(f"{'min':>10}{'median':>10}" for _ in order))
    for key in KEYS:
        row = f"{key:<10}"
        for k in order:
            v = [r[key] for r in runs[k]]
            row += f"{min(v):>10.1f}{statistics.median(v):>10.1f}"
        print(row)

    for k in order:
        w = [r["total"] for r in runs[k]]
        cpu = min(sum(r[c] for c in CPU_KEYS) for r in runs[k])
        occ = cpu / min(r["cost"] for r in runs[k])
        print(f"\n{k} {label[k]}: total spread {min(w):.1f}-{max(w):.1f} ms "
              f"({100*(max(w)/min(w)-1):.0f}%)")
        print(f"  cost stage: {cpu:.1f} ms CPU over {min(r['cost'] for r in runs[k]):.1f} "
              f"ms wall = {occ:.2f} of {a.threads} cores busy")

    if len(order) == 2:
        print()
        for key in ("total", "cost", "c_score", "c_filter", "c_insert"):
            am, bm = min(r[key] for r in runs["A"]), min(r[key] for r in runs["B"])
            if bm:
                print(f"  {key:<9} A {am:7.1f} -> B {bm:7.1f} ms   {am/bm:.3f}x")
        # A ratio inside the noise is not a result. Say so rather than leaving the
        # reader to compare a 1.03x against a spread they have to compute themselves.
        sa = [r["total"] for r in runs["A"]]
        noise = max(sa) / min(sa) - 1
        eff = abs(min(r["total"] for r in runs["A"]) / min(r["total"] for r in runs["B"]) - 1)
        if eff < noise / 2:
            print(f"\n  CAUTION: the total-time effect ({100*eff:.0f}%) is small against "
                  f"A's own spread ({100*noise:.0f}%).\n"
                  f"  Raise -n, or treat this as no measured difference.")


if __name__ == "__main__":
    main()
