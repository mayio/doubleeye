# Software environment

Two machines. The split is deliberate and matches the plan's workflow rule:
record on the vehicle, iterate on the desktop, deploy to the TX2 only when the
algorithm is stable.

## Jetson TX2 (`jetsontx2`, 192.168.2.114)

### Pre-existing — installed before this project, not by us

| Component | Version | Notes |
|---|---|---|
| L4T / JetPack | **R32.1 = JetPack 4.2** | The plan assumed 4.6.x. See obstacle 2. |
| Ubuntu | 18.04.6 LTS | |
| Kernel | 4.9.140-tegra | Stock; uvcvideo **unpatched**. See obstacle 7. |
| CUDA | 10.0 | at `/usr/local/cuda-10.0`. Plan assumed 10.2. Unused so far. |
| **librealsense** | **2.22.0** | `/usr/local` — headers, `.so`, `realsense2.pc`, `rs-enumerate-devices`, `realsense-viewer`. This is the key find: it removed the need to build anything. |
| OpenCV | headers 3.3.1 / runtime **3.2** | **Skewed.** See obstacle 4. Deliberately unused on this machine. |
| cmake | 3.10.2 | Sets the floor for `CMakeLists.txt` |
| g++ / gcc | 7.5.0 | Sets the C++ standard ceiling in practice — see below |
| make | 4.1 | |
| git | 2.17.1 | |
| Python | 3.6.9 (and 2.7.17) | `pyrealsense2` **absent** — see obstacle 3 |
| Kernel headers | `/usr/src/linux-headers-4.9.140-tegra-ubuntu18.04_aarch64`, `/lib/modules/4.9.140-tegra/build` present | Enough for an out-of-tree module build, should the uvcvideo patch ever be wanted |
| udev rule | `/etc/udev/rules.d/99-realsense-libusb.rules` | Already in place; non-root camera access works |

### Installed or changed by us

Deliberately minimal — **nothing was installed via `apt`**.

| Change | Detail | Reversible by |
|---|---|---|
| SSH public key | ed25519 from the desktop added to `~/.ssh/authorized_keys` | removing the key line |
| `dialout` group | `nvidia` added, so `/dev/ttyACM0` and `/dev/ttyTHS2` are openable without sudo | `sudo gpasswd -d nvidia dialout` |
| Passwordless sudo | `/etc/sudoers.d/99-nvidia-nopasswd`, mode 0440 | `sudo rm /etc/sudoers.d/99-nvidia-nopasswd` |
| Power mode | `sudo nvpmodel -m 0` (MAXN) — **does NOT persist**, see below | `sudo nvpmodel -m 3` |
| Clocks | `sudo jetson_clocks` — **does NOT persist**, by design | reboot |
| systemd unit | `/etc/systemd/system/doubleeye-performance.service`, enabled — reapplies both at boot | `sudo systemctl disable --now doubleeye-performance` |
| Source tree | `~/doubleeye/jetson/` | `rm -rf` |
| Build output | `~/doubleeye/jetson/build/` → `rs_probe`, `rs_ir_capture` | `rm -rf` |
| Recordings | `~/bags/` | `rm -rf` |

### Why the power settings need a service

Neither setting survives a reboot, and the `nvpmodel` half is counter-intuitive:

- `/etc/nvpmodel.conf` on this box carries **`< PM_CONFIG DEFAULT=3 >`**, and
  NVIDIA's own enabled `nvpmodel.service` runs `nvpmodel -f /etc/nvpmodel.conf`
  at every boot. So the board comes up in mode **3** regardless of what was set
  at runtime. An earlier version of this document wrongly claimed mode 0
  persisted; it does not.
- `jetson_clocks` only alters running state and never persists.

`doubleeye-performance.service` (source in `jetson/systemd/`) is ordered
`After=nvpmodel.service` so it reasserts MAXN *after* NVIDIA's unit has applied
the default, then locks clocks. Verified across a real reboot: MAXN, 6/6 cores
online, all at 2.04 GHz minimum. See [05-operations.md](05-operations.md).

### Not installed, on purpose

- **`pyrealsense2`** — no aarch64 wheel exists for Python 3.6, so it would mean
  building librealsense's bindings from source. Avoided by writing the
  Jetson-side tools in C++, which the installed librealsense already supports
  fully. See obstacle 3.
- **OpenCV linkage** — the header/runtime version skew is a latent trap and
  nothing on the Jetson needs it. Raw `Y8` is written to disk and decoded on the
  desktop.
- **`rsync`** — absent, and not worth installing. `tar` over `ssh` is used
  instead. See obstacle 9.

## Desktop (`blackstone`)

| Component | Version |
|---|---|
| Ubuntu | 24.04.4 LTS (x86_64) |
| Python | 3.12.3 |
| numpy | 2.5.1 |
| matplotlib | 3.11.1 |
| opencv-python | 5.0.0 — the **full** build, not headless |

OpenCV is the full wheel deliberately: `opencv-python-headless` has no `imshow`,
and `live_view.py` needs a window. It is desktop-only, from pip inside the venv,
so it never touches the Jetson's skewed OpenCV.

Installed by us: a virtualenv at `.venv` with `desktop/requirements.txt`, and an
ed25519 SSH keypair plus `~/.ssh/config`.

Deliberately **not** installed: cmake, OpenCV, librealsense, `pyrealsense2`.
Nothing in `desktop/` needs them, and the camera never attaches here.

### SSH configuration

`~/.ssh/config`:

```
Host jetson
    HostName 192.168.2.114
    User nvidia
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 4
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 5m
    Compression yes
```

Connection multiplexing and compression are there because the TX2 is on WiFi at
roughly 100 ms RTT; without `ControlPersist`, every command pays a fresh
handshake.

## ROS — used as a tool, not as the architecture

**ROS Melodic is already installed on the TX2** at `/opt/ros/melodic`, with
`rosbag`, `roscpp`, `cv_bridge`, `image_transport`, `camera_calibration`,
`sensor_msgs` and `tf2` present and the binaries working. `mavros` is not
installed. Melodic's Python is 2.7; `rospy` does not import under Python 3 here.

That makes ROS nearly free to adopt where it earns its place, and the decision
splits cleanly.

### Where ROS is the right answer

- **Kalibr, bring-up step 3.** Kalibr *is* a ROS package and consumes rosbags,
  so this is not really a choice. The plan already anticipated it: "Kalibr with
  `--time-calibration` runs fine on Ubuntu 18.04/Melodic — the old toolchain is
  an advantage here for once." An already-provisioned Melodic makes it more so.
- **A fast stereo sanity check** before Kalibr: `camera_calibration` is already
  installed and will give rough intrinsics and extrinsics from a checkerboard.
- **The Pixhawk**, via `ros-melodic-mavros`, which publishes
  `/mavros/imu/data_raw` and saves writing MAVLink parsing by hand.
- **rosbag as an interchange format** for the above.

### Where ROS must not go, and why

- **`core/`.** It stays dependency-free. That is the plan's portability rule and
  the thing that keeps a later Orin Nano migration a weekend.
- **The runtime capture and preprocessing path.** This one is settled by our own
  measurements rather than by taste. Preprocessing already costs **99.6 ms per
  stereo pair against a 33.3 ms budget** on the TX2, and the plan identifies
  memory bandwidth — 58.4 GB/s shared between CPU and GPU — as the real
  constraint. ROS 1 transport adds serialisation and copies to precisely the
  resource that is already short. Putting the hot path on ROS would spend the
  scarcest thing we have to buy convenience we do not need, since the capture
  tool is 250 lines and already verified.

So: **record ROS-free, convert offline when a ROS tool needs it.** The capture
path stays fast and simple; Kalibr still gets the rosbag it wants.

### Consequence — the bridge that is needed

A `bags/<run>` → rosbag converter, publishing the two IR streams as image
topics (and later the Pixhawk IMU). That unlocks `kalibr_calibrate_cameras`
against recordings the existing tools already produce.

Worth noting the camera half of step 3 needs **no IMU**, so it is unblocked as
soon as a checkerboard exists — see [05-operations.md](05-operations.md).

## Language and standard choices

**Jetson side is C++14.** Two reasons, pointing the same way:

1. C++ is the stated preference for this project.
2. It is also the path of least resistance here. librealsense 2.22 is installed
   with headers and a working `realsense2.pc`, so a C++ tool compiles in seconds
   against a prebuilt `.so`. Python would have required building the bindings
   first.

C++**14** rather than 17 because g++ 7.5 only ships `std::filesystem` behind
`<experimental/filesystem>`. Nothing here needs it — directory creation uses
`mkdir()` from `<sys/stat.h>`.

**Desktop side is Python.** This is where Python is genuinely the better tool:
offline analysis, statistics and plotting, with no latency constraint and no
deployment concern.

## Building

On the Jetson. **cmake there is 3.10.2**, which supports neither `-S`/`-B`
(needs 3.13) nor `--build -j` (needs 3.12); passing either makes cmake print its
usage text and exit *without building anything*, which is quiet enough to be
mistaken for success. So configure in-directory and drive `make` directly:

```sh
mkdir -p ~/doubleeye/jetson/build && cd ~/doubleeye/jetson/build
cmake .. && make -j6
```

Takes seconds — two translation units linked against a prebuilt `.so`. The
plan's warning that compiling on the TX2 is slow applies to building
librealsense or OpenCV from source, not to this.

`CMakeLists.txt` resolves librealsense via `find_package(realsense2)` and falls
back to `pkg-config`, which is what 2.22 reliably provides. Targets link
`realsense2` and `Threads` only. Built with `-Wall -Wextra`; currently warning-free.

## Deploying from the desktop

Use the helper, not a bare `make` over SSH:

```sh
./tools/deploy.sh          # sync sources, then build
./tools/deploy.sh --probe  # sync, build, and run rs_probe
```

This exists because of a mistake worth not repeating: running `make` on the
Jetson *without* first syncing the edited sources produces a successful
"Built target" message and a stale binary, which then appears to prove that a
just-written feature does not work. See obstacle 10.

## Pulling recordings

`rsync` is not available on the TX2:

```sh
mkdir -p bags/<run>
ssh jetson "cd ~/bags/<run> && tar czf - ." | tar xzf - -C bags/<run>/
.venv/bin/python desktop/capture_report.py bags/<run>
```

For a large bag over WiFi, pull `frames.csv` and `run.txt` first — they carry
every measurement except the images, and they are small. Raw frames are 407 kB
each.

## Repository layout

```
doubleeye/
├── jetson/            # C++14, librealsense2 only. Runs on the TX2.
│   ├── CMakeLists.txt
│   └── src/
│       ├── rs_common.hpp      # metadata table, clocks, power-state check
│       ├── rs_probe.cpp       # device health gate — run this first
│       ├── rs_ir_capture.cpp  # instrumented IR capture
│       └── rs_ir_stream.cpp   # raw frames to stdout, feeds the live view
├── desktop/           # Python 3.12, numpy + matplotlib. Runs on the dev box.
│   ├── capture_report.py
│   └── requirements.txt
├── tools/
│   └── deploy.sh
└── doc/
```

Preprocessing and MASDA itself, when written, go in a third top-level directory
depending on neither — that is the plan's portability rule, and encoding it in
the directory structure is what keeps a future Orin Nano migration a weekend
rather than a rewrite.
