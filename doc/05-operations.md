# Operations — recording, viewing, calibration targets

Everything routine, in the order you actually do it.

## 0. Power state — now automatic

`doubleeye-performance.service` sets MAXN and locks clocks at every boot. It is
installed and enabled, and verified across a real reboot.

```sh
systemctl status doubleeye-performance      # should be "active (exited)"
```

This is **not** cosmetic. Without it you silently lose a third of your frames —
see [03-obstacles.md](03-obstacles.md) obstacle 5. Two settings need
reasserting, and the reason the service must exist at all is that *neither*
persists on its own:

- `/etc/nvpmodel.conf` on this box carries `< PM_CONFIG DEFAULT=3 >`, and
  NVIDIA's own `nvpmodel.service` reapplies mode **3** at every boot. So
  `nvpmodel -m 0` does not survive a reboot here. Our unit is ordered
  `After=nvpmodel.service` so it wins.
- `jetson_clocks` never persists by design.

Unit source is version-controlled at
`jetson/systemd/doubleeye-performance.service`. To turn it off when battery life
matters more than capture integrity:

```sh
sudo systemctl disable --now doubleeye-performance   # restores booted clocks
```

The capture tools print the power state at startup regardless, and every
`run.txt` records `clocks_locked`, so a bag can never be silently misread.

## 1. Record

From the desktop, which syncs and builds first so you cannot run a stale binary:

```sh
./tools/deploy.sh --capture myrun --seconds 120
./tools/deploy.sh --pull myrun
```

`--capture` takes a **bare name**, not a path, and always records to
`~/bags/NAME` on the Jetson. That is deliberate: `--capture ~/bags/myrun` would
expand against the *desktop* home before ssh ever saw it, and the Jetson would
then fail with `cannot create /home/<desktop-user>/bags/myrun`. The script now
rejects anything containing a slash or tilde.

Or on the Jetson directly: `~/doubleeye/jetson/build/rs_ir_capture ~/bags/myrun ...`

### Options that matter

| Flag | Default | Notes |
|---|---|---|
| `--seconds F` | 60 | |
| `--exposure-us N` | 1500 | 1000–2000 keeps motion blur under a pixel at RC speeds. Auto-exposure is always off. |
| `--gain N` | 64 | Range 16–248. Raise for dark scenes; it costs noise, and Census needs contrast that is not noise. |
| `--emitter MODE` | `on` | `on` \| `off` \| `alternate`. See the note below before using `alternate`. |
| `--save-every N` | 30 | Write raw Y8 every Nth frame. **`0` disables image saving entirely.** |
| `--width/--height/--fps` | 848/480/30 | 848×480 also does 60 and 90 Hz |
| `--streams WHICH` | `both` | `both` \| `1` \| `2` |

### Choosing `--save-every`

This is the one that catches people. A frame is 407 kB, so at 30 fps × 2 streams
saving every frame is 24 MB/s against ~13 GB of free eMMC.

| Purpose | Setting | Why |
|---|---|---|
| Timing / rate measurement | `--save-every 0` | No disk I/O to perturb the measurement |
| Normal recording | `--save-every 30` | One pair per second; enough to check exposure and scene |
| **Something you want to watch** | `--save-every 4` and `--seconds 8` | ~60 pairs, ~47 MB — enough for the GIF and quick to pull over WiFi |
| Algorithm input | `--save-every 1`, short | Every frame, but keep it brief and watch disk space |

`--emitter alternate` is supported by the firmware but **currently not usable**:
the per-frame projector label lives in metadata this kernel does not expose, so
you cannot tell afterwards which frames were lit. For a static scene, record two
separate runs with `on` and `off` instead — that is what the projector A/B in
[04-baseline-measurements.md](04-baseline-measurements.md) did.

## 2. Pull the bag

```sh
./tools/deploy.sh --pull myrun
```

`rsync` is not installed on the TX2, so underneath that is `tar` over `ssh`:

```sh
mkdir -p bags/myrun && ssh jetson "cd ~/bags/myrun && tar czf - ." \
  | tar xzf - -C bags/myrun/
```

Over WiFi a large bag is slow. `frames.csv` and `run.txt` hold every
measurement except the images and are small, so pull those first if you only
want numbers:

```sh
mkdir -p bags/myrun && ssh jetson "cd ~/bags/myrun && tar czf - frames.csv run.txt" \
  | tar xzf - -C bags/myrun/
```

## 3. Look at it

```sh
.venv/bin/python desktop/view_bag.py bags/myrun
```

Writes four things into `bags/myrun/view/`, each answering a different question:

| Output | What it is for |
|---|---|
| `contact_sheet.png` | Did the whole recording look sane, or did something change partway through? |
| `stereo_pair.png` | Left and right side by side — exposure, focus, and whether the projector reaches the surfaces that matter |
| `anaglyph.png` | ir1 red over ir2 cyan. **Fringing must be horizontal only.** |
| `animation.gif` | Motion, and spotting dropped or duplicated frames |

The anaglyph is the most informative of the four and worth learning to read.
Horizontal red/cyan fringing *is* the disparity, and it should grow on near
objects and shrink on far ones. **Vertical** fringing would mean the
rectification is off or the channels are not time-aligned; on the current
recordings there is none, which is independent confirmation of the
zero-distortion, identity-rotation calibration.

Useful flags: `--frame N` picks which pair to use for the stereo and anaglyph
views, `--fps N` sets GIF playback rate.

No OpenCV and no ffmpeg needed — GIFs go through matplotlib's Pillow writer, and
raw Y8 is one byte per pixel so numpy reads it directly.

## 4. Analyse it

```sh
.venv/bin/python desktop/capture_report.py bags/myrun
```

Delivered rate, frame-number continuity, L/R pairing, interval histogram,
timestamp domain and metadata availability, arrival jitter in milliseconds *and*
in pixels of misregistration, plus local-contrast image statistics. Also writes
`capture_report.png`.

Read the warnings it prints. It deliberately refuses to draw conclusions it
cannot support — for instance it will not do frame-gap analysis on a host-side
counter, because that reports a flawless run no matter how much was lost.

## 5. Calibration target

```sh
.venv/bin/python desktop/make_checkerboard.py -o checkerboard.pdf
```

Default is 10×7 squares at 25 mm on A4 landscape, giving **9×6 interior
corners**. Interior counts are deliberately odd×even: a board with both counts
even is 180°-ambiguous and detectors can silently flip its orientation between
frames. The script prints the matching Kalibr yaml and OpenCV `Size`.

Adjust with `--cols`, `--rows`, `--pitch`, `--paper` (`a4`, `a4p`, `a3`,
`letter`). It refuses sizes that will not fit with margins rather than silently
cropping.

### Printing it — five things that will otherwise cost you a calibration

1. **Print the PDF at 100%.** Turn off "fit to page" and "scale to fit". A `.png`
   is written too, but it is a preview only — printing a raster gets it resampled
   and the pitch you calibrate against stops matching the pitch you measured.
2. **Use a laser printer, not inkjet.** This is specific to us and easy to miss:
   many dye-based inkjet blacks are nearly *transparent* in near-IR, so a board
   that looks perfect to your eye can be almost invisible at 850 nm. Laser toner
   is carbon-based and stays black in IR.
3. **Measure the printed board and use the measured pitch.** Span many squares
   and divide — across all 10 columns of the default board you should read
   250.0 mm. Do not trust the nominal 25 mm.
4. **Mount it rigidly and flat.** Glue to foam board, MDF or glass. A curled or
   hand-held A4 sheet is a large, systematic, and completely invisible error
   source. Matte paper, not glossy — glossy specularly reflects the IR projector.
5. **Turn the projector off for calibration:** `--emitter off`. The dot pattern
   overlays the board and interferes with corner detection. You then need enough
   ambient IR, so do it in a well-lit room or add IR illumination.

### A4 is honestly marginal — know why before relying on it

At the measured fx = 430.55 px, the 250 mm board spans roughly:

| Distance | Board width in image | Fraction of 848 px |
|---|---|---|
| 0.3 m | ~359 px | 42% |
| 0.5 m | ~215 px | 25% |
| 1.0 m | ~108 px | 13% |
| 2.0 m | ~54 px | 6% |

Good calibration wants the target filling a substantial part of the frame across
a range of poses. A4 only manages that inside ~0.5 m, which is at the near edge
of the working range. It is fine for a first pass and for verifying the
toolchain end to end, but for the calibration you actually trust, print larger —
A3, or tile A4 sheets onto a rigid board (`--paper a3`, or raise `--pitch` and
accept fewer corners).

## 6. Stereo calibration

The camera half of bring-up step 3 needs **no IMU**, so it is unblocked as soon
as a board is printed.

### Use the live view — do not work blind

You cannot aim a target at a camera you cannot see, and "were the poses varied
enough?" is unanswerable while holding a board. So the whole session runs off a
live view:

```sh
.venv/bin/python desktop/live_view.py --collect calib01
```

That pipes `rs_ir_stream` from the Jetson over ssh (measured ~22 MB/s, so full
848×480 on both channels at 10 Hz is comfortable) and shows both channels with
corner detection overlaid, plus:

- a **big status line** — GOOD (both channels), ONE CHANNEL ONLY (board clipped in
  the other), or NOT DETECTED
- a **coverage grid** that fills in as you cover image regions
- a **"still needed"** line naming what is missing: regions, distances, or tilted
  views
- **`--collect NAME`**, which auto-saves each genuinely *new* pose into
  `bags/NAME`, rejecting near-duplicates. Fifty frames of the same pose constrain
  calibration no better than one, and they make the set look adequate when it is
  not.

Because `--collect` writes a bag-shaped directory, the existing tools work on it
directly and there is no separate blind recording step.

Keys: `q` quit, `r` reset coverage, `s` save a pair as PNG, `space` pause.

### How to hold and move the board

**Mounting.** Glue the A4 to something rigid and flat — foam board, MDF, a
clipboard, glass. **Never use bare paper.** A curl of a couple of millimetres is
a large systematic error and is completely invisible in the result. Hold it by
the edges or from behind so fingers do not cover squares.

**Distance.** Fill as much of the frame as you can while keeping the *whole* board
inside **both** views. The right camera sees a shifted view, so a board near the
left edge of ir1 can be clipped in ir2 — the status line tells you when that
happens. With A4 that means roughly 0.4–0.5 m for a good fill. Then deliberately
vary it: the "poses per distance band" readout shows near/mid/far coverage.

**Angle — this is the one people get wrong.** Do not hold it flat facing the
camera. Fronto-parallel views leave focal length and distortion poorly
constrained, because scale and distance trade off against each other almost
exactly. Tilt it **20–40°**:

- rotate about the vertical axis, like opening a door
- tip the top toward and away from you
- spin it in its own plane as well

The "tilted views" counter only advances on genuinely tilted poses, so if it is
stuck you are holding it too flat.

**Position.** Work it into all regions of the image, corners included. Corners
carry most of the distortion information and are the easiest to neglect.

**Motion.** Move, then **stop and hold still for about a second** at each pose.
Corner localisation degrades on a moving board even though 1500 µs is too short
for visible blur. Watch for the status to settle before moving on.

**How many.** 20–40 well-varied poses. Stop when "still needed" reads *nothing*.

### Alternative: verify first without collecting

If you only want to check the print is usable, ten seconds is enough.

Do this before a full session. It costs ten seconds and catches the failure modes
that would otherwise waste the whole sitting — above all IR-transparent ink.

```sh
./tools/deploy.sh --capture cbtest --seconds 10 --save-every 4 --emitter off --gain 96
./tools/deploy.sh --pull cbtest
.venv/bin/python desktop/view_bag.py bags/cbtest           # can you see it?
.venv/bin/python desktop/check_checkerboard.py bags/cbtest  # can OpenCV see it?
```

Hold the board reasonably close, filling a good part of the frame. `--emitter off`
matters: projector dots interfere with corner detection. `--gain 96` because with
the emitter off the scene is darker — the last recording came out at mean 54 DN.

`check_checkerboard.py` reports detection in **both** channels separately, since a
pair where only one channel found the board contributes nothing to stereo
extrinsics. It also reports image coverage and pose spread, and orders its failure
hints by likelihood.

### Then check the collected set

```sh
.venv/bin/python desktop/check_checkerboard.py bags/calib01
```

Confirms detection in both channels, coverage and pose spread over the whole set.

A blind recording still works if you prefer it — `./tools/deploy.sh --capture
calib01 --seconds 90 --save-every 3 --emitter off --gain 96` — but there is no
reason to when the live view collects the same data with feedback.

### Then Kalibr

ROS Melodic is already on the Jetson with `rosbag` and `camera_calibration`. The
piece that does not exist yet is a `bags/<run>` → rosbag converter, which is what
Kalibr consumes — see [02-software-environment.md](02-software-environment.md)
for why the capture path stays ROS-free and conversion happens offline.

`camera_calibration` is also already installed and gives a quicker, rougher
answer if you want a cross-check first.

**Use the measured pitch, not the nominal 25 mm.** Span all 10 columns of the
default board and divide by 10.
