#!/usr/bin/env python3
"""The warp's top-2 reduction, simulated on the host, with and without k in the key.

`k_vert_bwd_top2` finds each pixel's two best disparities with a `shfl_down` tree
across 32 lanes. The first version compared scores only and broke ties by assuming the
other side of a merge holds the larger disparities. That is false in the middle of the
tree -- lane 0 absorbs lane 16 before it absorbs lane 1 -- and it produced ten wrong
pixels in 407,040, every one an exact score tie.

Ten pixels is far below what an accuracy benchmark can see, so the bug was found by
`cmp` against the CPU and pinned here rather than in the profiler: this file
reproduces the tree in twenty lines and counts how often each key disagrees with what
the CPU's ascending-k scan returns.

    .venv/bin/python article/top2sim.py

Standard library only. Runs in a few seconds.
"""

import random


def tree(lanes, merge):
    """The five rounds CUDA runs: lane i absorbs lane i+off, off = 16, 8, 4, 2, 1."""
    off = 16
    while off:
        lanes = [merge(lanes[i], lanes[i + off]) if i + off < 32 else lanes[i]
                 for i in range(32)]
        off >>= 1
    return lanes[0]


def make_merge(greater):
    """Merge two (best, second) pairs. Identical in shape to the kernel's."""
    def merge(a, b):
        a0, a1 = a
        b0, b1 = b
        if greater(b0, a0):
            return (b0, b1 if greater(b1, a0) else a0)
        if greater(b0, a1):
            return (a0, b0)
        return a
    return merge


def reference(cands):
    """What the CPU does: walk k ascending, strictly greater displaces. Smallest k."""
    best = (-1, -1)
    for v, k in cands:
        if v > best[0]:
            best = (v, k)
    return best[1]


# The first version: the key is the score alone, so a tie is broken by whichever side
# of the merge the value happened to arrive from.
VALUE_ONLY = lambda x, y: x[0] > y[0]                                    # noqa: E731
# The fix: k rides in the key. The comparison is (score descending, k ascending), so
# the result does not depend on the order the tree merges in. In the kernel this is
# one integer, (score + 32768) << 8 | (255 - k), and the comparison is a plain `>`.
WITH_K = lambda x, y: (x[0], -x[1]) > (y[0], -y[1])                      # noqa: E731


def run(greater, trials=40_000, distinct_values=3, seed=0):
    rnd, wrong = random.Random(seed), 0
    merge = make_merge(greater)
    for _ in range(trials):
        cands = [(rnd.randrange(distinct_values), k) for k in range(64)]
        lanes = [merge((cands[i * 2], (-1, -1)), (cands[i * 2 + 1], (-1, -1)))
                 for i in range(32)]
        if tree(lanes, merge)[0][1] != reference(cands):
            wrong += 1
    return wrong, trials


def main():
    print("How often the reduction returns a different disparity from the CPU's scan.")
    print("`distinct values` is how many different scores the 64 candidates are drawn")
    print("from: the fewer there are, the more exact ties, which is the only case the")
    print("two keys can disagree on.\n")
    print(f"  {'distinct values':>16}{'value-only key':>18}{'packed (score, k)':>20}")
    for nv in (3, 64, 1024, 16384):
        a, n = run(VALUE_ONLY, distinct_values=nv)
        b, _ = run(WITH_K, distinct_values=nv)
        print(f"  {nv:>16,}{100 * a / n:>17.2f}%{100 * b / n:>19.2f}%")
    print("\nThe real cost volume is Q14, so exact ties are rare and the bug showed up")
    print("as ten wrong pixels in 407,040 -- invisible to an accuracy benchmark, and")
    print("caught only because the GPU is held bit-identical to the CPU.")


if __name__ == "__main__":
    main()
