# DoubleEye — documentation

Working record of the platform, its software environment, and every obstacle hit
during bring-up. Written so a full project write-up can be assembled later
without re-deriving anything.

The algorithmic spec lives outside this folder in
`~/Documents/doubleeye/doubleeye_plan.md`. That document holds the design
decisions, the rejected alternatives and the open questions. Nothing here
duplicates it; this is the engineering record that sits underneath it.

## Reading order

| Document | Contents |
|---|---|
| [TODO.md](TODO.md) | **What to do next**, ordered by what unblocks the most |
| [01-hardware.md](01-hardware.md) | Components in the system, what each is for, what is still missing |
| [02-software-environment.md](02-software-environment.md) | Exact versions on both machines, what was installed vs. pre-existing, how to build and deploy |
| [03-obstacles.md](03-obstacles.md) | Every problem hit, with symptom → diagnosis → resolution. The most useful document here. |
| [04-baseline-measurements.md](04-baseline-measurements.md) | Measured numbers, for citation and for regression comparison |
| [05-operations.md](05-operations.md) | How to record, pull, view, analyse, and print a calibration target |
| [06-preprocessing.md](06-preprocessing.md) | Census + keypoints: design, measured output, and the profiling result |
| [07-tools.md](07-tools.md) | **Every tool, what it answers, how to run it** — start here when returning to the project |
| [08-imu.md](08-imu.md) | Pixhawk/ArduPilot: links, isolated venv, parameters applied, and what is still open |
| [09-matching.md](09-matching.md) | **MASDA** — messages, validation against brute force, and the measured advantage over nearest-neighbour |
- [10-architecture.md](10-architecture.md) — what runs on the GPU, what runs on the CPU, and why nothing is on the GPU yet.

## Status at time of writing

Bring-up step 1 (capture pipeline and timestamp sanity) is substantially
complete. `848×480@30` runs at 29.95 fps on both IR channels with a tight
unimodal frame-interval distribution. One item remains open: hardware sync
cannot be verified in the current configuration — see
[03-obstacles.md](03-obstacles.md), obstacle 7.

Bring-up step 2 (IMU Allan variance) is **unblocked on the hardware side.** The
Pixhawk runs ArduPilot, the IMU stream is up from 3.2 Hz to 53 Hz, the SD card is
confirmed present and healthy, and raw pre-filter IMU logging is enabled. What
remains is a multi-hour static log and the analysis. **Complete.** Noise density from raw
1 kHz batch samples, bias instability and random walk from the continuous 50 Hz
stream — valid there because a 20 Hz filter cannot affect averaging times of
seconds. Cross-checked between two independent logs. See [08-imu.md](08-imu.md).
Separately, the `nvs_bmi160` module and its device-tree node are stock L4T
leftovers with no hardware behind them, and are not the IMU.

The **camera half of step 3 is under way**: it needs no IMU. A calibration set
exists — `bags/calib01`, 82 poses, **100% detected in both channels** — collected
with the live view. **Done.** An independent OpenCV
calibration on 49 poses reproduces the factory baseline to 0.13% and the principal
point to under 0.35 px, and confirms zero distortion and a rectified pair. The
factory calibration has not drifted, so it stands. A 4.3% baseline discrepancy
turned out to be a 96% print scale, now measured and corrected. See
[04-baseline-measurements.md](04-baseline-measurements.md#calibration-set-collected-2026-08-07)
for the set's properties and its one real caveat, which is board size.

Preprocessing (`core/`) is **within budget on the target**: 26.28 ms per stereo
pair against 33.3 ms, or 78.8%, down from 300.5% — see
[06-preprocessing.md](06-preprocessing.md).

**MASDA exists and works.** Validated against exhaustive search (58/60 exact
optima) and measured on real stereo pairs, where it finds **46% more matches than
nearest-neighbour with the projector on** and only 13% more with it off — the
advantage scaling with ambiguity exactly as the plan argues. See
[09-matching.md](09-matching.md).

## Findings at a glance

The results worth knowing, with pointers. Several were surprises, and the pattern
across them is the same: **on this platform the expensive failures are silent.**

| Finding | Where |
|---|---|
| **34% of frames lost with no error at all** — wrong power mode, while frame numbers stayed contiguous and the tool reported `0 missing` | [03](03-obstacles.md) obstacle 5 |
| `nvpmodel -m 0` does **not** persist: `/etc/nvpmodel.conf` says `DEFAULT=3` and NVIDIA's own service reapplies it every boot | [03](03-obstacles.md) obstacle 6 |
| No UVC metadata node, so `get_frame_number()` is a **host-side counter that is contiguous by construction** — which is what hid the frame loss | [03](03-obstacles.md) obstacle 7 |
| Hardware sync is therefore **unverifiable** right now: L/R pairing and timestamp agreement are both true by construction | [03](03-obstacles.md) obstacle 7 |
| The uvcvideo patch is **not worth it**: arrival jitter is 0.04 px at p99 against a ±3–4 px budget, and holds under full CPU load | [04](04-baseline-measurements.md) |
| Camera-vs-host clock rate **is** recoverable without the patch, from frame generation rate: **+568 ppm** | [04](04-baseline-measurements.md) |
| f·B = **21.48 px·m** against the plan's predicted ~21; distortion exactly zero and IR1→IR2 rotation exactly identity | [04](04-baseline-measurements.md) |
| The projector **halves the textureless area**, 57% → 25%, and lifts keypoint coverage from 67% to 93% of cells | [04](04-baseline-measurements.md), [06](06-preprocessing.md) |
| **Mean intensity cannot see the projector** — it moved 1.8 DN across the same A/B. Local contrast is the only valid metric | [03](03-obstacles.md) obstacle 12 |
| Census descriptors are **3.3× degenerate** under the projector (338 distinct for 1115 keypoints) — exactly the ambiguity MASDA's uniqueness constraint is for | [06](06-preprocessing.md) |
| MASDA finds **+46% matches over nearest-neighbour with the projector on, +13% with it off** — the advantage scales with ambiguity, as argued | [09](09-matching.md) |
| MASDA costs **5.04 ms on the Jetson**, 15% of the budget. The 1.67 ms previously quoted was a **desktop** number compared against the Jetson's budget, because `core/` had never been built on the Jetson at all | [03](03-obstacles.md) obstacle 15 |
| The whole pipeline **does not fit 30 Hz untuned**: preprocessing 26.54 ms + matching 5.14 ms = 95% of budget, and the recommended matcher config alone pushes it to 106% | [09](09-matching.md) |
| **Raising `fast_threshold` 8 → 12 saves 30% of preprocessing for 10% of matches**, at flat precision and unchanged sub-pixel error. Nobody had tried it: 8 was chosen against the *dense* detector, a quality argument, never a budget one | [09](09-matching.md) |
| Threshold beats resolution decisively: 8→20 costs 24% of matches for 57% of the time, where halving resolution costs **74%** of keypoints for 70%. Resolution also costs depth precision directly | [09](09-matching.md) |
| Tuned, everything on, the pipeline is **23.07 ms — 69% of budget** with *more* keypoints than the untuned baseline | [09](09-matching.md) |
| **Detector repeatability, not the matcher, is the ceiling on real data**: only 44–51% of left keypoints have any right keypoint within 1 px of their true correspondence, which accounts for 102 of Teddy's 132 errors | [09](09-matching.md) |
| Over-proposing on the right + gating on score margin is better than baseline on **both** axes: +183 correct *and* +0.012 precision | [09](09-matching.md) |
| The **score margin is the confidence worth exporting** — precision by quartile 0.169 / 0.286 / 0.391 / 0.659 — and it is nearly free, since the solver already computes a top-2 reduction | [09](09-matching.md) |
| **Local contrast is not useful and the intuition is backwards**: it predicts correctness *negatively* (bottom quartile 0.669, top 0.506), because high contrast means crowded candidates. Do not raise `min_local_std` | `article/contrast_study.py` |
| `lambda` and `gamma` were **transposed** relative to the article. Harmless only because every run held them equal, which stereo has good reason not to | [09](09-matching.md) |
| Seeded synthetic scenes were **not reproducible at all** — `hash()` is salted per process — and it cost a published conclusion that flipped sign between runs | [03](03-obstacles.md) obstacle 16 |
| **Sub-pixel disparity by parabola fit on the inter-window cost** more than halves the tail outside half a pixel, 12.7% → 5.5% | [09](09-matching.md) |
| 2 of 6 CPU cores are used and the **GPU has never been touched** (load 0). Four cores and the whole GPU are idle | [10](10-architecture.md) |
| The messages **never formally converge** yet the decision is stable from 20 iterations; oscillation is real but currently benign | [09](09-matching.md) |
| Belief **sign is not a match/no-match decision** — gating on it returned zero matches on tied problems whose optimum matched everything | [09](09-matching.md) |
| **Coarse-to-fine does not help here**: k is already 2.7 not 100–200, and inflating it 5× leaves the answer *identical*, so there are no false candidates to remove | [09](09-matching.md) |
| **The cheap smoothness experiment also does not help** — monotonically worse at every weight, best `w_smooth` is 0. A prior fitted from the current matches is close to self-confirming | [09](09-matching.md) |
| **Neither negative result is conclusive**, because match count and median \|dy\| cannot distinguish removing wrong matches from removing right ones. **Step 4's rangefinder gates matcher development**, not just driving | [09](09-matching.md) |
| Preprocessing was **299% of the 30 Hz budget on the TX2**; FAST candidates plus concurrent L/R brought it to **78.8%**, keeping 96% of keypoints | [06](06-preprocessing.md) |
| Below ~5 DN the FAST **sparse path becomes slower than the dense scan** it replaces — sparsification has a crossover | [06](06-preprocessing.md) |
| Concurrency returned **1.54× not 2×** on a 6-core board, corroborating memory bandwidth as the real constraint | [06](06-preprocessing.md) |
| `std::thread` needs `-pthread` on the Jetson's glibc 2.27 but **links silently without it** on the desktop's 2.34+ | [06](06-preprocessing.md) |
| Computing Census densely wasted **99.7%** of the work; making it sparse cut 61 ms to 0.13 ms with identical output | [06](06-preprocessing.md) |
| The Pixhawk reports "PX4 FMU v2.x" over USB but **runs ArduPilot** — the descriptor is the bootloader identity | [01](01-hardware.md) |
| `nvs_bmi160` and its device-tree node are **stock leftovers with no hardware behind them** — not the IMU | [01](01-hardware.md) |
| IMU stream was capped at 50 Hz not by baud or USB but by **`SCHED_LOOP_RATE`** — nothing streams faster than the loop emitting it | [08](08-imu.md) |
| **`INS_GYRO_FILTER` was 4 Hz**, stripping almost everything rotation compensation depends on; raised to 20 Hz | [08](08-imu.md) |
| **`LOG_DISARMED = 0`** means a stationary bench setup logs *nothing* — a multi-hour recording would have produced an empty card | [08](08-imu.md) |
| Reading a tty without `stty raw` returned 288 bytes where the real rate was 7.2 kB/s — indistinguishable from a dead link | [08](08-imu.md) |
| `INS_LOG_BAT_MASK` does not retrofit batch logging into an already-open log — **a multi-hour recording would have been useless** until a reboot | [08](08-imu.md) |
| Filtered IMU data **understates gyro noise density by 41%**, measured against raw 999.6 Hz batch samples from the same log | [08](08-imu.md) |
| Gravity reads **9.074 m/s², −7.5%** — the accelerometer needs 6-position calibration | [08](08-imu.md) |
| The batch sampler logs **1 s windows, not a stream**, so bias instability cannot be read from concatenated blocks — but it *is* readable from the continuous 50 Hz stream, since a 20 Hz filter cannot affect tau of seconds | [08](08-imu.md) |
| The Pixhawk **re-enumerates ACM0→ACM1** across a reboot; use the udev by-id link | [08](08-imu.md) |
| A calibration set of 82 poses reached **100% detection in both channels**, so the print is IR-visible | [04](04-baseline-measurements.md) |
| ...yet a third of those poses had a **non-flat board**. Detection success says nothing about planarity | [04](04-baseline-measurements.md) |
| Calibration reproduces factory fx to **0.61%** and confirms zero distortion — freeing radtan made the fit *worse* | [04](04-baseline-measurements.md) |
| A **+4.3% baseline error was traced to the printer, not the camera** — predicted pitch 23.97 mm, measured 24.0, reconciling the baseline to **+0.13%** | [04](04-baseline-measurements.md) |
| So the **factory calibration is validated, not drifted** — use it; our fx is the weaker number, limited by board coverage | [04](04-baseline-measurements.md) |
| cmake 3.10 accepts neither `-S`/`-B` nor `--build -j`; given either it **prints usage and exits without building** | [03](03-obstacles.md) obstacle 13 |

## Conventions used throughout

- **DN** — digital number, i.e. raw 8-bit pixel level (0–255).
- **`jetson`** — the SSH alias for the TX2; see
  [02-software-environment.md](02-software-environment.md).
- Numbers quoted as measured are from recordings under `bags/`, which is
  gitignored. `run.txt` inside each bag records the conditions it was taken
  under, including whether the clocks were locked.
