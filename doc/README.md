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
| [01-hardware.md](01-hardware.md) | Components in the system, what each is for, what is still missing |
| [02-software-environment.md](02-software-environment.md) | Exact versions on both machines, what was installed vs. pre-existing, how to build and deploy |
| [03-obstacles.md](03-obstacles.md) | Every problem hit, with symptom → diagnosis → resolution. The most useful document here. |
| [04-baseline-measurements.md](04-baseline-measurements.md) | Measured numbers, for citation and for regression comparison |
| [05-operations.md](05-operations.md) | How to record, pull, view, analyse, and print a calibration target |
| [06-preprocessing.md](06-preprocessing.md) | Census + keypoints: design, measured output, and the profiling result |
| [07-tools.md](07-tools.md) | **Every tool, what it answers, how to run it** — start here when returning to the project |
| [08-imu.md](08-imu.md) | Pixhawk/ArduPilot: links, isolated venv, parameters applied, and what is still open |
| [09-matching.md](09-matching.md) | **MASDA** — messages, validation against brute force, and the measured advantage over nearest-neighbour |

## Status at time of writing

Bring-up step 1 (capture pipeline and timestamp sanity) is substantially
complete. `848×480@30` runs at 29.95 fps on both IR channels with a tight
unimodal frame-interval distribution. One item remains open: hardware sync
cannot be verified in the current configuration — see
[03-obstacles.md](03-obstacles.md), obstacle 7.

Bring-up step 2 (IMU Allan variance) is **unblocked on the hardware side.** The
Pixhawk runs ArduPilot, the IMU stream is up from 3.2 Hz to 53 Hz, the SD card is
confirmed present and healthy, and raw pre-filter IMU logging is enabled. What
remains is a multi-hour static log and the analysis. Gyro and accel filters were raised from
4/10 Hz to 20 Hz, and logging is verified live at ~130 MB/hour. Since the vehicle
has no battery and cannot move, the conditions for a static Allan-variance
recording are already ideal. See [08-imu.md](08-imu.md).
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
| MASDA costs **1.67 ms**, 5% of the frame budget against 26 ms for preprocessing — "MASDA is not the bottleneck" confirmed | [09](09-matching.md) |
| The messages **never formally converge** yet the decision is stable from 20 iterations; oscillation is real but currently benign | [09](09-matching.md) |
| Belief **sign is not a match/no-match decision** — gating on it returned zero matches on tied problems whose optimum matched everything | [09](09-matching.md) |
| **Coarse-to-fine does not help here**: k is already 2.7 not 100–200, and inflating it 5× leaves the answer *identical*, so there are no false candidates to remove | [09](09-matching.md) |
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
| The batch sampler logs **1 s windows, not a stream**, so bias instability cannot be read from concatenated blocks | [08](08-imu.md) |
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
