# DoubleEye

MASDA (max-sum loopy belief propagation for data association) as the matcher for
sparse stereo correspondence, on a D435 IR pair and a Jetson TX2.

Start at [doc/README.md](doc/README.md); [doc/TODO.md](doc/TODO.md) is the ordered
work list; [doc/07-tools.md](doc/07-tools.md) is every tool and how to run it.

## Identity

Commits are `Mario Lüder <monsieur.mona@gmail.com>` (GitHub `mayio`). **The
author's employer must not appear anywhere in this repository** — not in code,
comments, commit messages, or documentation.

## Layout

| path | what |
|---|---|
| `core/` | the library and its tools. Builds with plain `make`; runs on both machines |
| `jetson/` | capture, cmake, systemd. TX2 only |
| `desktop/` | analysis and viewers, Python. Uses `.venv` |
| `article/` | the published write-up and the experiments behind it |
| `doc/` | the documentation. `03-obstacles.md` is the most useful file here |

```bash
cd core && make && make test      # 104 assertions, must stay at 0 failures. 16 of them are the NEON
                                  # kernel and report SKIPPED off the Jetson
tools/deploy.sh                   # sync jetson/ and core/ to the TX2 and build both
tools/doccheck.py --blog          # documentation: links, anchors, retired values, phrasing
```

Two Pythons, not interchangeable: `.venv/bin/python` for anything needing
`rosbags`/`scipy` here, system `python3` for anything needing `rclpy`.

## How to work on this project

Twenty-four obstacles are recorded in [doc/03-obstacles.md](doc/03-obstacles.md).
They are not independent. Almost all of them are one of six things, and each rule
below is followed by the obstacles that produced it. Treat these as defaults, not
aspirations — every one was learned by getting it wrong here.

**1. On this platform the expensive failures are silent.** Not one serious problem
produced an error message; several produced output that looked healthy. A tool that
prints usage and exits 0 is indistinguishable downstream from a device that returned
nothing. Check the producer's stderr and exit status before believing a consumer's
silence. `set -o pipefail` or read `${PIPESTATUS[0]}` — a pipe hides the status of
everything but the last stage. *(5, 7, 10, 13, 17, 18)*

**2. A number measured in one context is not a measurement in another.** A desktop
timing quoted against the Jetson's frame budget was wrong by 3×, and every ratio
later scaled from the desktop was also wrong: 6.4 predicted vs 8.84 measured, 27 vs
23.07, 12.6 ms vs 457 ms at a different stride. If the claim is about the Jetson,
measure on the Jetson. *(15, 22)*

**3. Verify that the verification can fail.** A frame-gap check reported `0 missing`
while a third of frames were lost, because it compared a host-side counter against
itself. A regression test compared two zeros and passed. A test harness piped data
into `python3 -` — which reads its *program* from stdin — and reported zero packets
for a healthy pipeline. Before trusting a passing check, ask what input would make
it fail, and confirm the numbers in its output look like the quantity intended.
*(7, 12, 19)*

**4. A default is a claim about the world.** The disparity gate defaulted to a
0.10–21.5 m search range and was used indoors, which admitted a quarter of all
points nearer than 0.19 m and halved the score margin. `fast_threshold = 8` was
chosen against the dense detector — a quality argument — and then treated as a
budget decision. The live path then asserted a 0.40 m near limit, which a camera
pointed at a floor falsifies for a sixth of the frame -- and an unsearched disparity
comes back as a confident wrong answer, not as a gap. Ask what each default asserts
and whether it is true here. *(16, 22, 24)*

**5. Measure before optimising, and after.** The contrast stretch cost more than
every per-point loop combined, which is not where the loops suggested looking.
Detector stage 1 is 55% of detection, not the overwhelming majority its
touches-every-pixel description implies. Interpolating at stride 8 is 36× faster
than stride 4 for identical coverage. None of these were predictable from reading
the code.

**6. Claims asserted in comments are the ones that turn out wrong.** The QoS comment
stated the opposite of how DDS compatibility works. The local-contrast intuition was
backwards — it predicts correctness *negatively*. The ordering-factor conclusion
flipped sign twice before being measured over seeds. When writing a justification
into a comment or a doc, either cite the measurement or say it is untested. Both are
fine; asserting it is not.

### Writing

Full rules in [doc/12-writing.md](doc/12-writing.md), enforced where a script can by
`tools/doccheck.py`. The four that bind hardest:

- **Every number states its machine, input, configuration and unit in the sentence
  that carries it.** A figure lifted into another document carries them with it.
- **Documentation describes the present, never the change.** Changelog wording is out
  — `doccheck.py` holds the list and will name any that appear. History lives in git
  and in `03-obstacles.md`, the one file allowed to be chronological.
- **Name a section, link the section**, and verify the anchor against what the
  generator actually emits rather than what its documentation implies.
- **Write for an engineer whose first language is not English**: one idea per sentence,
  no idioms, subject first, abbreviations expanded on first use in every document.

When a default changes, grep every document for the old value and add the retirement
to `RETIRED` in `tools/doccheck.py`.

### Consequences worth keeping

- **Report spread, not a single run**, when the effect could be smaller than the
  variance. The ordering factor is ±2 against a scene-to-scene spread of ±31; two
  single-scene runs gave opposite signs and both were noise.
- **Prefer a negative result recorded to a knob left in.** Local contrast, the
  coarse-to-fine pass, the cheap smoothness prior and cheaper misdetection are all
  documented as not working, with the mechanism. That is why they have not been
  retried.
- **Ground truth exists now.** `core/tools/de_bench.cpp` runs the shipping C++
  matcher against Middlebury. Any accuracy claim can be checked; "it looks better"
  and "it is more correct" are different statements and both are worth having.
