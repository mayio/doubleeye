# Obstacles

Every problem hit during bring-up, with the symptom as it actually presented,
the diagnosis, and the resolution. Ordered so the numbering is stable for
cross-referencing; the two that cost the most were **5** (power mode) and
**7** (uvcvideo). Obstacles **10** and **13** are self-inflicted and
documented because both cost real time.

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

**Diagnosis.** `nvpmodel -m 0` persists — it writes the default mode to
configuration. `jetson_clocks` does not; it only alters the running state.

**Resolution.** Not forced at boot, on purpose: max clocks means max power draw,
and this is a battery-powered vehicle where the plan already treats power as a
real constraint. That is a policy decision, not something to impose silently.

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
