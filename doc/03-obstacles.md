# Obstacles

Every problem hit during bring-up, with the symptom as it actually presented,
the diagnosis, and the resolution. Ordered so the numbering is stable for
cross-referencing; the two that cost the most were **5** (power mode) and
**7** (uvcvideo). Obstacles **10** and **13** are self-inflicted and
documented because both cost real time.

These are not independent problems. Six patterns account for almost all of them,
and they are written up as working rules in [CLAUDE.md](../CLAUDE.md) at the repo
root, each citing the obstacles that produced it. Read that first if you are
returning to this project; this file is the evidence behind it.

The recurring theme is worth stating up front: **on this platform, the
expensive failures are silent.** Not one of the serious problems below produced
an error message. Several actively produced output that looked healthy.

---

## 1. No route onto the board

**Symptom.** The camera was reported connected and the Jetson on, but the
development machine had no SSH keypair, no `~/.ssh/config`, and no record of the
board's address. `lsusb` on the desktop showed no RealSense, so the camera was
not local.

**Diagnosis.** The board was found on the LAN by MAC OUI: `192.168.2.114` had
`00:04:4b`, which is registered to NVIDIA. It answered ping, and port 22
returned an ED25519 host key with `Permission denied (publickey,password)` —
so `sshd` was up and password auth was enabled.

**Resolution.** Generated an ed25519 keypair on the desktop, installed the
public key via `ssh-copy-id`, and wrote the `Host jetson` alias documented in
[02-software-environment.md](02-software-environment.md).

**Worth knowing.** OUI lookup is a fast way to find an unlabelled board on a
subnet. `00:04:4b` is NVIDIA.

---

## 2. JetPack 4.2, not 4.6

**Symptom.** None. Everything simply behaved as an older release.

**Diagnosis.** `/etc/nv_tegra_release` reports `R32 (release), REVISION: 1.0`,
which is **JetPack 4.2** — CUDA 10.0, dated March 2019. The plan assumed 4.6.x
with CUDA 10.2 and TensorRT 8.2.

**Resolution.** Accepted. The plan's statement that the TX2 is *capped* at
JetPack 4.6.x is correct; this board simply sits below that ceiling. Moving
4.2 → 4.6 is a full reflash, so it is only worth doing for a concrete reason.
The one candidate reason is the XFeat-via-TensorRT path, which the plan already
treats as optional and which the Census decision makes unnecessary.

**Consequence to remember.** Any future claim about the available toolchain must
be checked against 4.2, not 4.6.

---

## 3. `pyrealsense2` absent, and not installable cheaply

**Symptom.** `import pyrealsense2` → `ModuleNotFoundError` on the TX2.

**Diagnosis.** There is no aarch64 wheel for Python 3.6. Getting the bindings
means building librealsense from source with `-DBUILD_PYTHON_BINDINGS=ON` — slow
on this board and an unnecessary dependency.

**Resolution.** Wrote the Jetson-side tools in **C++** instead. librealsense
2.22 is already installed with headers, `.so` and `realsense2.pc`, so a C++ tool
builds in seconds against the prebuilt library. This also matched the project's
language preference, so the constraint and the preference pointed the same way.

**Worth knowing.** Check for the C++ headers before assuming a Python binding
problem is blocking. `pkg-config --modversion realsense2` answers it in one
command.

---

## 4. OpenCV header/runtime version skew

**Symptom.** None yet — this is a trap that has not been stepped in.

**Diagnosis.** `/usr/include/opencv2/core/version.hpp` reports **3.3.1** and
`pkg-config --modversion opencv` agrees, but the actual runtime library in
`/usr/lib/aarch64-linux-gnu` is **`libopencv_core.so.3.2`**. Compiling against
3.3.1 headers and linking a 3.2 runtime is an ABI mismatch waiting to happen.

**Resolution.** Avoided entirely. Nothing on the Jetson links OpenCV. The
capture path writes raw `Y8` to disk and all decoding happens on the desktop.

**Worth knowing before adding OpenCV to anything on-vehicle.** This needs
resolving first — either by removing the distro `libopencv-dev` or by pinning
explicitly to one of the two.

---

## 5. Power mode caused a silent 34% frame loss

**The most costly obstacle, and the most instructive.**

**Symptom.** `848×480@30` on two IR streams delivered **18.5–19.9 fps**. And
nothing complained:

- USB link correctly negotiated at `3.2`
- no kernel USB or isochronous errors
- **frame numbers perfectly contiguous, zero gaps**
- the capture tool's own summary reported `0 missing (0.000%)`

The last two points are the trap, and they are caused by obstacle 7 below: the
only frame counter available counts *delivered* frames, so it is contiguous by
construction and reports a flawless run no matter how much is lost.

**Diagnosis.** Isolated by a sweep, all with `--save-every 0` so disk I/O was
excluded:

| configuration | achieved | verdict |
|---|---|---|
| 848×480@30, **single** stream | 29.80 | makes rate |
| 848×480@30, both | 18.53 | short |
| 640×480@30, both | 17.62 | short |
| 848×480@**90**, both | 29.88 | delivers ~30 |
| 424×240@30, both | 29.83 | makes rate |

Two facts fell out. A single stream made rate perfectly, so the limit was
**shared, not per-frame**. And requesting 90 fps *delivered* ~30, proving the
hardware could move 24 MB/s — so this was **never a bandwidth wall**. Note also
that 848×480@30 pushed *more* pixels/s than 640×480@30 while both stalled near
18 fps, which rules out a simple throughput ceiling.

The board was in power mode **3** (`MAXP_CORE_ARM`) with `cpu online = 0,3-5`,
i.e. **both Denver cores offline**, and `jetson_clocks` had never been applied.

**Resolution.**

```sh
sudo nvpmodel -m 0 && sudo jetson_clocks
```

This gave 6/6 cores online, all at 2.04 GHz, EMC at 1866 MHz — and fixed every
configuration outright:

| requested | mode 3 | MAXN + clocks locked |
|---|---|---|
| 848×480 @ 30 | 18.53 | **29.79** |
| 848×480 @ 60 | 22.08 | **59.54** |
| 848×480 @ 90 | 29.88 | **89.34** |
| 640×480 @ 30 | 17.62 | **29.79** |
| 1280×720 @ 30 | 10.64 | **29.79** |

**Worth knowing.** The plan already prescribed setting `nvpmodel` and
`jetson_clocks` before any latency measurement, and it was right for a bigger
reason than reproducibility: without them the capture is not merely
*variable*, it is *wrong*, and wrong in a way that reports success.

---

## 6. `jetson_clocks` does not survive a reboot

**Symptom.** Latent. Obstacle 5 would silently return at the next boot and
present as a brand-new bug.

**Diagnosis.** Worse than first assumed. `jetson_clocks` does not persist —
expected, it only alters running state. But **`nvpmodel -m 0` does not persist
either** on this box: `/etc/nvpmodel.conf` carries `< PM_CONFIG DEFAULT=3 >` and
NVIDIA's own enabled `nvpmodel.service` reapplies that at every boot. So the
board returns to the broken mode **3** on every power cycle, not just to unlocked
clocks. An earlier version of this document had this wrong.

**Resolution.** `doubleeye-performance.service`, ordered `After=nvpmodel.service`
so it reasserts MAXN after NVIDIA's unit has applied `DEFAULT=3`, then locks
clocks. Installed, enabled, and **verified across a real reboot**: MAXN, 6/6
cores online, all at 2.04 GHz minimum, unit `active (exited)`.

The cost is real and worth stating: max clocks means max power draw on a battery
vehicle. `sudo systemctl disable --now doubleeye-performance` turns it off and
restores the as-booted clocks. The unit source is version-controlled at
`jetson/systemd/doubleeye-performance.service`.

Instead, both tools now read the power state at startup and print it, warning
loudly with the measured consequence when the machine is not at full
performance. `run.txt` in every recording records `cpus_online` and
`clocks_locked`, so a bag can never be misinterpreted after the fact and two
recordings taken under different conditions can never be quietly compared.

```
power state: 6/6 CPUs online, scaling_min 2.04 GHz of max 2.04 GHz, clocks LOCKED
```

The detection reads `scaling_min_freq` against `cpuinfo_max_freq` for every
online CPU — `jetson_clocks` raises the *minimum* to the maximum, which is a
reliable signature.

---

## 7. uvcvideo is unpatched — no UVC metadata node

**Symptom.** Reported by `rs_probe`: `get_timestamp()` returns domain
**`System Time`**, and most per-frame metadata is `UNSUPPORTED`.

**Diagnosis.** librealsense can only surface the camera's own clock and vendor
metadata if `uvcvideo` is patched to expose UVC metadata nodes. On a stock
JetPack kernel it is not. Confirming evidence: the stock module
(`/lib/modules/4.9.140-tegra/kernel/drivers/media/usb/uvc/uvcvideo.ko`, version
1.1.1) and only three RealSense `/dev/video*` nodes, where a patched system
shows additional metadata nodes. `dmesg` also shows repeated
`uvcvideo: Failed to query (GET_CUR) UVC control ... -32` (`-EPIPE`), which is
the vendor extension-unit path failing.

Available: `Backend Timestamp`, `Time Of Arrival`, `Actual Fps`.
**Absent:** `Frame Timestamp`, `Sensor Timestamp`, `Frame Counter`,
`Actual Exposure`, `Gain Level`, `Frame Laser Power Mode`.

**Consequences, in descending order of damage:**

1. **`get_frame_number()` is a host-side counter of delivered frames.** It is
   contiguous by construction. This is what made obstacle 5 invisible and is the
   single most dangerous item on this page — it turns a broken capture into a
   clean-looking report.
2. **Hardware sync cannot be verified.** With a host-side counter that restarts
   at 1 per stream, and host-side stamps applied at delivery, both the L/R
   pairing *and* the L/R timestamp difference come out perfect **by
   construction**. `paired 3600, ir1-only 0, ir2-only 0` and `max |Δt| 0.000000
   ms` are therefore not evidence of anything. **This remains open.**
3. Clock skew cannot be read off the timestamps — though it turns out to be
   recoverable another way, see below.
4. No per-frame projector label, so `EMITTER_ON_OFF` alternation cannot be
   demultiplexed.
5. Exposure cannot be confirmed per frame, only trusted from the option
   readback.

**Resolution — deliberately none, and the reasoning matters.**

The measured cost of *not* patching is small. On a 120 s, 3600-frame recording
at locked clocks, arrival jitter about the fitted frame grid was **0.062 ms std,
0.050 ms at p99**. Converted into the unit that matters — pixels of
misregistration under gyro rotation compensation at 100 °/s — that is
**0.04 px at p99**, against the plan's ±3–4 px coarse-to-fine search radius.
Over an order of magnitude of margin.

And the clock-skew risk largely evaporates on inspection: frames are stamped on
the **Jetson** clock, and the IMU is on the **Jetson** clock too, so cross-clock
skew never enters the alignment at all. What remains is a constant
exposure→arrival latency, which is exactly what Kalibr `--time-calibration`
estimates.

The camera-vs-host *rate* also turned out measurable without the patch — not
from timestamps but from the frame **generation rate** fitted against host
arrival times: **+568 ppm**, reproduced at +569 ppm on an independent run. It
does not decompose into camera crystal versus host clock without an external
reference, but the combined figure is what is actually needed to predict when
the next frame lands in host time.

**Load caveat now tested.** Re-measured with all 6 cores saturated and memory
traffic running (load average 5.69): delivered rate held at 29.90 fps and p99
jitter was **identical at 0.04 px**. Arrival jitter turns out not to be
load-sensitive on this board, so the conclusion above survives a busy machine.
Still untested against vibration and against a real matcher competing for memory
bandwidth — see [04-baseline-measurements.md](04-baseline-measurements.md).

**If the patch is ever wanted**, the prerequisites are half-present: kernel
headers for 4.9.140-tegra are installed and `/lib/modules/4.9.140-tegra/build`
exists, but the librealsense source tree carrying
`patch-realsense-ubuntu-L4T.sh` is not present and would need cloning. The
remaining value would be sync verification, in-motion projector labelling, and
per-frame exposure confirmation — in that order.

**On verifying hardware sync — deferred deliberately, not forgotten.**

Neither option for closing this belongs in step 1. Reading camera frame counters
needs the patch, which the timing measurement above says is not worth it. And
detecting a time offset optically needs a moving scene *plus* stereo
correspondence, to show that a moving object's disparity is biased along its
motion direction while the static background's is not — that is matching code,
which does not exist yet.

The better place for it is the first matcher. The plan already wants median
`|Δy|` over matched pairs as a free runtime health metric for extrinsic drift,
and that same statistic catches a sync error: an inter-channel time offset in a
moving scene inflates the y-residual in exactly the regions that are moving. So
sync verification falls out of work that is already planned, rather than
requiring a dedicated experiment now.

Until then it is an **assumption**, and should be labelled as one in any result
that depends on it.

**Workaround adopted for the projector A/B.** Rather than alternating within one
recording, run the emitter **on** and **off** as two separate recordings. On a
static scene that is equally valid and needs no per-frame label. It does not
help for a moving scene, where alternation would be the only option.

---

## 8. Transient `hwmon` failure reading calibration

**Symptom.** `rs_probe` aborted with
`rs2_get_video_stream_intrinsics(...): hwmon command 0x15 failed`.

**Diagnosis.** Intermittent, and biased: the **first** calibration read after
the pipeline starts usually fails, and the second succeeds. Verified across
repeated runs, always returning identical values when it succeeds. So it is a
firmware warm-up quirk, not a calibration fault.

**Resolution.** `rs_probe` retries up to four times, reporting which attempt
succeeded. If it never succeeds it continues *without* intrinsics rather than
aborting — the timestamp-domain and metadata checks are the more important
output of that tool and need no calibration data.

**Worth knowing.** Do not read this as drifted or corrupted calibration. Rerun.

---

## 9. `rsync` not installed on the TX2

**Symptom.** `rsync` from the desktop fails on the remote side.

**Resolution.** `tar` over `ssh` in both directions. Not worth installing a
package for.

```sh
# pull
ssh jetson "cd ~/bags/<run> && tar czf - ." | tar xzf - -C bags/<run>/
# push
tar czf - jetson | ssh jetson 'tar xzf - -C ~/doubleeye'
```

---

## 10. Stale binary from building without syncing

**Symptom.** A newly written feature appeared not to work: the code was correct,
the build reported `Built target` with no errors, and the feature was simply
absent from the output.

**Diagnosis.** Self-inflicted. `make` was run over SSH on the Jetson *without*
first copying the edited sources across, so it correctly rebuilt nothing and
relinked a stale binary. `Built target` is printed even when there is nothing to
do, so the build output gives no hint.

**Resolution.** `tools/deploy.sh` couples the two steps so they cannot be
separated. Use it rather than a bare remote `make`.

**Worth knowing.** This is the kind of error that costs far more than it should,
because every signal points at the code rather than at the deployment.

---

## 11. Red herring — kernel `WARNING` backtraces during enumeration

**Symptom.** `dmesg` is full of kernel warning backtraces with
`Modules linked in: ... uvcvideo ...` while working with the camera. Alarming,
and easy to mistake for the cause of the frame loss in obstacle 5.

**Diagnosis.** They are all
`WARNING: ... at drivers/media/v4l2-core/v4l2-ioctl.c:1305 v4l_enum_fmt+0x11dc`
— a benign driver-conformance nitpick raised during **format enumeration**, not
during streaming. The timestamps confirm it: they occur when librealsense
queries profiles, before any capture starts.

**Resolution.** Ignore. Documented here purely so the next person does not spend
time on it.

---

## 12. Measurement methodology — mean intensity cannot see the projector

**Symptom.** Comparing emitter-on against emitter-off recordings, **mean image
intensity differed by only 1.8 DN** (89.9 vs 88.1). Read naively, that says the
IR projector does essentially nothing — which would undermine the plan's entire
Census-over-learned-descriptors argument.

**Diagnosis.** The metric was wrong, not the projector. The scene contained a
sunlit window occupying a saturated 6.5% of the frame, and ambient IR dominates
the **global mean** while the projector contributes only **local
high-frequency** structure. Visual inspection settled it immediately: projector
dots are plainly visible on the near surfaces.

**Resolution.** Measure what Census actually reads — local contrast over the
same window size. Using 7×7 local standard deviation:

| metric | emitter ON | emitter OFF |
|---|---|---|
| mean intensity | 89.9 | 88.1 |
| 7×7 local std, median | **3.91** | 1.65 |
| fraction with local std < 2 DN | **24.5%** | 56.7% |

The projector more than halves the textureless area, 57% → 25%, and lifts median
local contrast 2.4×. The plan's premise holds; only the instrument was wrong.

`capture_report.py` now reports local contrast for this reason, and says
explicitly that mean intensity is not a valid discriminator.

**Worth knowing.** Two general lessons. Choose the statistic that matches the
algorithm that will consume the data. And look at the images — a glance
disambiguated what the summary number had made confusing.

---

## 13. cmake 3.10 silently does not build

**Symptom.** `tools/deploy.sh` reported `==> done` with no compiler output at
all, and the binary on the Jetson was never updated. Exit status 0.

**Diagnosis.** Two separate version limits, both hit at once. cmake on the TX2 is
**3.10.2**, which supports neither:

- `cmake -S . -B build` — the `-S`/`-B` form needs **cmake 3.13**
- `cmake --build build -j6` — `-j` for `--build` needs **cmake 3.12**

Given an unsupported option, cmake prints its **usage text and exits without
building**. Worse, the script piped build output through
`grep -E 'error|warning|Built target' || true`, so the usage text was filtered
away and the non-zero status was masked by `|| true`.

**Resolution.** Configure in-directory the portable way and drive `make`
directly, propagating failure instead of filtering it:

```sh
mkdir -p build && cd build
cmake .. && make -j6
```

Verified by deliberately breaking a source file and confirming the script now
surfaces the compiler error and exits 1.

**Worth knowing.** This is obstacle 10 all over again in a different costume, and
it was introduced by the very script written to prevent obstacle 10. The general
lesson: **a build step that can fail quietly will eventually cost you an hour of
debugging correct code.** Never filter a build's output through `grep` with
`|| true`.

---

## 14. No passwordless sudo initially

**Symptom.** `nvpmodel` and `jetson_clocks` could not be run non-interactively,
which blocked the fix for obstacle 5. `nvpmodel -q` also emits
`NVPM ERROR: Error opening /sys/kernel/nvpmodel_emc_cap/... : 13` as a non-root
user — errno 13 is `EACCES`, not a real fault.

**Resolution.** A sudoers drop-in, validated before relying on it:

```sh
echo 'nvidia ALL=(ALL) NOPASSWD:ALL' | sudo tee /etc/sudoers.d/99-nvidia-nopasswd > /dev/null
sudo chmod 0440 /etc/sudoers.d/99-nvidia-nopasswd
sudo visudo -c        # must print "parsed OK"
```

The `visudo -c` check matters: a syntax error in a `sudoers.d` file breaks
`sudo` entirely, and it is much better to discover that while still holding a
working session. A narrower alternative, sufficient for the power-mode work
alone, is `NOPASSWD: /usr/sbin/nvpmodel, /usr/bin/jetson_clocks`.

Revoke with `sudo rm /etc/sudoers.d/99-nvidia-nopasswd`.

## 15. The Jetson was running x86-64 object files, and `core/` had never been built there

`tools/deploy.sh` synced only `jetson/`. `core/` had reached the Jetson at some
point by hand, along with its `build/` directory full of **desktop** object files.
The first native build there failed with

    /usr/bin/ld: build/preproc.o: Relocations in generic ELF (EM: 62)

EM 62 is x86-64. That is the tell, and it is not obvious from the message.

The consequence was worse than the build failure: `de_match` had never been built
on the Jetson at all, so its timing had only ever been measured on the desktop and
then quoted against the Jetson's 33.3 ms frame budget. The real figure is **5.04 ms,
not 1.67 ms** — 15% of the budget rather than 5%. Match counts agree across the two
machines (620 against 615), so it was the same work on slower cores.

Fixed: `deploy.sh` syncs and builds `core/` too, and excludes `*/build` and `*.o`
from the tarball so host objects can never cross again.

**Generalisation.** A number measured on one machine and compared against another
machine's budget is not a measurement. Two later estimates in this project failed
the same way: 6.4 ms predicted against 8.84 ms measured, and 27 ms predicted against
23.07 ms measured. Every ratio scaled from the desktop was wrong.

## 16. Python's `hash()` is salted per process, so seeded scenes were not reproducible

`rng_for(name)` seeded a generator from `abs(hash(("masda", name)))`. String hashing
is salted per process (PEP 456), so every run drew a **different** scene:
deterministic within one process, irreproducible across two.

Nothing about it looked wrong. It cost a published conclusion. The ordering-factor
experiment was reported from a single scene as "+6, it helps", then re-measured as
"−3, it hurts" — and those two runs were not comparing the same geometry. The real
answer over five seeds is +1.0 ± 2.1, i.e. no effect, against a scene-to-scene spread
of ±31.

Fixed: seed from `zlib.crc32`, which is stable across processes, versions and
machines. And report a spread over seeds whenever the effect might be smaller than
the variance, which is most of the time.

## 17. `rs_ir_stream` prints usage and exits 0 on an unknown flag

Obstacle 13 again, in a new place. `rs_ir_stream --streams both` is not a valid
invocation — it emits both channels by default and `--single` restricts to ir1 — so
it printed usage to stderr and exited **0**. Downstream, `de_pipe` reported
`0 pairs`, which is indistinguishable from a camera that produced nothing.

If a consumer reports no frames, read the producer's stderr before suspecting
hardware.

**It happened twice.** `desktop/de_live_ros2.py` was shipped with `--streams both`
hardcoded, so the first real run of the live viewer failed exactly this way. The
obstacle was written up from the first occurrence while the script still contained
it. Documenting a trap is not the same as removing it; the call sites need grepping.

There is a lesson about verification here too. The end-to-end test that was supposed
to catch this was

    ssh jetson '...' | python3 - <<'PY'

which cannot work: `python3 -` reads its *program* from stdin, so the heredoc took
stdin and the piped data went nowhere. The test reported 0 packets for a pipeline
that was fine, which is indistinguishable from the bug it was meant to detect. Put
the reader in a file when the thing being read is stdin.

## 18. A pipe hides the exit status, including in one-liners written to check builds

    make -s 2>&1 | tail -8 && echo BUILD_OK

printed `BUILD_OK` over a compile error, because `&&` tests `tail`'s status, not
`make`'s. This is obstacle 13's shape in a command typed to *guard against*
obstacle 13.

Use `set -o pipefail`, or read `${PIPESTATUS[0]}`.

## 19. Assertions that pass without testing anything

Two instances, years apart in spirit and hours apart in fact:

- The frame-gap analysis reported `0 missing (0.000%)` while 34% of frames were
  lost, because it compared a host-side counter against itself (obstacle 7).
- A regression test written *today* to catch transposed `lambda`/`gamma` asserted
  `a.size() >= b.size()` where both were 0. It passed. The descriptors intended to
  score at chance actually differed in 48 of 48 bits, so both configurations
  rejected everything.

A test whose assertion holds trivially is worse than no test, because it is
evidence. Assert exact values, and check the numbers in the failure message look
like the quantity you meant to measure.

## 20. Smaller traps, recorded so they are not rediscovered

- **Most vexing parse.** `Image8 img(int(w), int(h));` declares a *function*. Braces
  fix it: `Image8 img{int(w), int(h)};`
- **rviz2 needs a transform for its fixed frame.** With none it draws nothing and
  says so only obliquely. Both viewer paths publish an identity `map` →
  `camera_link`.
- **Depth must be NaN where absent, not 0.** Zero reads as a measurement of zero
  range.
- **rosbag2 writer version.** `VERSION_LATEST` is 9; a bag written newer than the
  installed ROS 2 understands fails on the version rather than on anything useful.
  Pinned to 8.
- **`curl` is not installed on the desktop.** `wget` is. Install recipes copied from
  the web assume curl.
- **Two Pythons.** `rclpy` comes from apt against the system Python 3.12; `rosbags`
  lives in the venv, which is built with `include-system-site-packages = false`. The
  live script cannot run in the venv and the converter needs it.

## 21. `timeout` does not kill a pipeline's children, and `pkill -f` kills its own ssh

`timeout 10 bash -lc 'rs_ir_stream | de_pipe'` terminates the shell and leaves both
children running. The next run then fails with

    librealsense error in rs2_pipeline_start_with_config:
    xioctl(VIDIOC_S_FMT) failed Last Error: Device or resource busy

which looks like a camera fault and is a stale process holding the device.

The obvious cleanup is worse. `ssh jetson 'pkill -f rs_ir_stream'` matches the
remote shell's **own** command line, because the pattern appears in it, so it kills
itself and ssh returns 255. Use `pkill -x rs_ir_stream`, which matches the process
name rather than the full command line.

Put the `timeout` on the *last* stage of the pipeline instead, and follow with an
explicit `pkill -x` for the producer:

    ./rs_ir_stream ... | timeout 10 ./de_pipe ... ; pkill -x rs_ir_stream

## 22. The disparity gate defaulted to five times the depth that existed

`MatchConfig`'s `[1, 220]` px default spans 0.10-21.5 m at f·B = 21.48 px·m. Used
indoors it admitted a quarter of all points nearer than 0.19 m, piled against the
limit, and halved the score margin by crowding every keypoint with extra candidates.
In rviz the result is indistinguishable from a broken matcher: it looks like random
points, because most of them are.

Tightening to 0.4-6.0 m produced *more* matches (646 against 553 per frame), a
median margin of 0.657 against 0.388, and 11% doubtful against 31.5%. See
[09-matching.md](09-matching.md).

A search range is a physical claim about the scene. Leaving it at a library default
is not a neutral choice.

## 23. A header added to one build rule and not the other, and the measurement it faked

`core/Makefile` lists each tool's headers as explicit prerequisites, because a `.hpp`
cannot go in `$^` (g++ would compile it as an input). Obstacle 10 is the first time
that bit: editing `simd_score.hpp` relinked a stale `de_dense` and the change looked
simply not to work, so the header was added to the rule and a comment written above it.

Then `dense_solve.hpp` was created when the solver moved out of `de_dense.cpp` to be
shared with the CUDA tool. It was added to `$(DENSECU)`'s prerequisites and **not** to
`$(DENSE)`'s. Six weeks later a header-only change -- the `sigma_s` default, which
lives in that struct -- was made, `make` printed a normal build line for the CUDA tool
and nothing for the CPU one, and a fifteen-scene benchmark then ran the OLD value and
reported it as the new one.

**What caught it was not the build.** It was that the number came back *exactly* equal
to a figure that should have changed: 26.94, the pre-change value, to the last digit.
An unchanged measurement is a weaker signal than an error, and a slightly different one
would have been believed.

- **The rule was right and the list was stale.** The comment above the rule described
  the failure mode precisely and did not stop it happening a second time, because a
  comment cannot enumerate headers that do not exist yet.
- **`make` reported success.** Exit 0, no output for the target that did not rebuild,
  which is indistinguishable from "already up to date" -- because that is exactly what
  it was.
- **`tools/tx2_ab.py` already guards against this on the Jetson** by refusing a binary
  older than its sources. Nothing guarded the desktop, where the accuracy benchmarks run.

The fix is the prerequisite. The check that would have caught it faster is to verify a
default change is live before measuring it: run the tool with the flag set explicitly
to the OLD value and confirm the output differs from the default.

## 24. The search range ran out before the floor did, and the matcher answered anyway

A live dense cloud showed **a second ground plane sloping away downwards** under the
real one. It was not noise, and the exposure controller of 0.45 -- which had just
fixed a real noise problem -- did nothing for it, because it was not that kind of
failure.

The measurement, on `bags/room2_2500` and on five frames pulled live off
`/doubleeye/depth`:

| | ghost points | pinned at the search limit | valid |
|---|---|---|---|
| `--dmax 53` (the default) | 2,843  (0.80%) | 2.35% | 87.8% |
| `--dmax 64` | 242  (0.07%) | 0.01% | 88.3% |
| `--dmax 80` | 235  (0.07%) | 0.01% | 88.2% |

`de_live_ros2.py --min-range` defaulted to 0.40 m, and `dmax = f*B/min_range` made that
53 px. Point the camera at a floor and the bottom of the frame is nearer than 0.40 m,
so **the true disparity was outside the searched set** -- and the matcher does not
answer "not found" in that case, it answers with the best candidate it did search. On a
tiled floor the runner-up is the tile period, so the wrong answers were *spatially
coherent* and rendered as a surface rather than as speckle. 0.8% of the points, and it
dominated the view.

- **The margin gate does not catch this.** Measured: `--min-margin 0.05` cuts the ghost
  from 0.80% to 0.65% while cutting all coverage from 87.8% to 20.8%. Periodic texture
  produces a *confident* wrong match -- best-minus-second is large because the true
  match is not in the comparison at all. Confidence is only meaningful over the set
  actually searched, which is exactly the set at fault here.
- **Trimming the frame border does not catch it either.** The ghost looked like one
  image row in the side view -- a straight line through the camera origin *is* a
  constant row -- but it was rows 412-477, a band, and an 8 px trim removed 40% of it.
  The side-view slope identified the region and then over-identified the cause.
- **It is not the near-range junk of 22.** That was 0.10 m admitting garbage; this is
  the correction to it overshooting. Both are the same rule: a default is a claim
  about the world. 0.40 m claimed nothing is nearer, and a camera that can see its own
  floor makes that false about a sixth of the frame.
- **Raising the limit is close to free and never negative.** `--dmax 64` gained
  coverage on all three scenes tried, added 12,556 real near-floor points on the
  hallway, and added 172 on a scene with nothing that close -- so it does not
  manufacture near points where there are none.
- 80 buys nothing over 64, so the floor bottoms out around 0.34 m in this room.
- **On the Jetson it is not merely cheap, it is free**, and the desktop's 1.12-1.14x
  did not predict that (rule 2). Pipelined steady state at 848x480, `--threads 4`,
  clocks locked, 30-32 C, three reps, minima:

  | `--dmax` | ms/frame | rate | filled |
  |---|---|---|---|
  | 53 (the old default) | 32.0 | 31.3 Hz | 88.0% |
  | **64** | **31.5** | **31.7 Hz** | **88.5%** |
  | 65 | 47.9 | 20.9 Hz | 88.5% |

  **The GPU quantises the disparity search into blocks of 64.** Everything from 1 to
  64 costs the same -- `--dmax 32` measures 25.4 ms in the cost stage against 64's
  25.6 -- and 65 through 128 costs the same as each other, 63% more. So the old
  default of 53 was *paying for 64 disparities and using 53*, discarding 11 it had
  already bought, and those 11 are exactly the ones the floor needed. A default that
  was set as a quality trade-off turned out to have no cost at all on the side it was
  trading against: obstacle 16's `fast_threshold` mistake, in reverse.

Two frames apart, the ghost **flickers**: of the pixels on it in any of five
consecutive frames, only 23.5% are on it in all five, against a whole-frame disparity
repeatability of 0.14 px median. A temporal consistency vote would therefore remove
roughly half of it -- but fixing the search range removes 92% of it for nothing, and
the two are not alternatives.

### 24a. What was left after the range fix, and three cures that were worse

Live on the camera the fix landed as predicted -- 0.78% ghost to 0.07%, coverage 83.1%
to 84.1% -- but a thin sheet remained, ~250 points a frame. It had **moved**: columns
28-38, rows 372-475. The bottom-left corner.

That is a different mechanism. A pixel at column x can only be searched over
disparities up to x-3, because beyond that the corresponding right-image pixel is off
the edge of the sensor. At column 38 the reach is 35 and the near floor there is at
disparity ~50, so **the true match is not in the right image at all**. Those pixels are
genuinely monocular, and the matcher answers regardless -- the same silent failure as
24, with the image border doing what `dmax` was doing before.

Three cures, all measured on `bags/room2_2500`, all rejected:

- **Turn the message passing on.** The prediction was that MASDA's one-to-one
  constraint should starve an occluded pixel, since it has no partner to claim. It does
  the opposite: ghost 242 -> 256 -> 339 -> 473 at `--iters` 0, 2, 4, 8, and the growth
  is concentrated in the strip (54 -> 283). Coverage rises too, 88.3% -> 90.0%. Message
  passing propagates support along the row, and where no correct answer exists it
  spreads the wrong one and grows its confidence. **Uniqueness does not imply
  occlusion rejection when the true partner is off-sensor** -- there is no competitor
  to lose to.
- **Mask the left `dmax` columns.** 272 good points destroyed per ghost point removed.
- **Mask only the pixels that provably could not reach**, estimating the local
  disparity from the nearest fully-searched columns of the same row and dropping the
  pixel when that exceeds its reach. 18 good points per ghost point -- 15x better than
  the blunt mask, but it reaches only the 22% of the residual that is in the strip.

What does work is time. Of the pixels ghosting in any of five consecutive frames, 23.5%
ghosted in all five before the range fix and **2.2%** after it: the residual is almost
entirely transient. A 3-of-5 agreement vote takes it from 250 points a frame to 61, for
4.3% of coverage --

| kept if it agrees in | points/frame | ghost/frame |
|---|---|---|
| every frame on its own | 342,145 | 250 |
| >= 2 of 5 | 331,741 | 108 |
| >= 3 of 5 | 327,594 | 61 |
| >= 4 of 5 | 317,903 | 37 |

-- measured on a **stationary** camera, so it is an upper bound on the gain and a lower
bound on the cost. Under motion the reference has to be warped, and that is the part
that needs the vehicle moving. The useful part is that the static half is measurable
today, on a bench, with no odometry and no ground truth.
