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
| Passwordless sudo | `/etc/sudoers.d/99-nvidia-nopasswd`, mode 0440 | `sudo rm /etc/sudoers.d/99-nvidia-nopasswd` |
| Power mode | `sudo nvpmodel -m 0` (MAXN) — **persists across reboot** | `sudo nvpmodel -m 3` |
| Clocks | `sudo jetson_clocks` — **does NOT persist across reboot** | reboot, or `jetson_clocks --restore` |
| Source tree | `~/doubleeye/jetson/` | `rm -rf` |
| Build output | `~/doubleeye/jetson/build/` → `rs_probe`, `rs_ir_capture` | `rm -rf` |
| Recordings | `~/bags/` | `rm -rf` |

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
│       └── rs_ir_capture.cpp  # instrumented IR capture
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
