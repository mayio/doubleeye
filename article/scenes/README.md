# Scenes

The captures the posts' figures are built from, kept here rather than in `bags/`
because `bags/` is scratch: `tools/deploy.sh --capture` deletes and recreates a bag of
the same name, and two of these were overwritten that way while the figure was being
made.

One room — the kitchen the vehicle sits in — at three light levels, each with the
D435's projector on and off. Every directory holds `left.y8`, `right.y8` (848×480, 8
bits, the hardware-rectified IR pair) and a `meta.txt` giving the condition, the
exposure, the measured mean level, local contrast, clipping and coverage.

| directory | condition | exposure | mean | coverage |
|---|---|---|---|---|
| `night_on` / `night_off` | no lamp | 4000 µs | 21 / 16 DN | 88.1% / 38.6% |
| `evening_on` / `evening_off` | lamp only | 2500 µs | 56 / 56 DN | 87.8% / 80.1% |
| `morning_on` / `morning_off` | lamp + daylight | 1500 µs | 95 / 93 DN | 84.7% / 81.1% |

Exposure differs per light level on purpose: each is the value that holds 3–5 DN of
median local contrast in that room, which is what the live controller targets. A
single exposure across a 4.5× range of scene brightness would measure the exposure
rather than the thing under test.

Coverage is `de_dense --dmax 64 --agg 5`. Reproduce the figure with:

```bash
.venv/bin/python article/figs_blog.py --only real
```
