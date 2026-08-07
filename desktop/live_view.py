#!/usr/bin/env python3
"""Live IR view with checkerboard detection — aim the target while you can see it.

Pipes jetson/build/rs_ir_stream over ssh and shows both IR channels with corner
detection overlaid, so you get immediate feedback on whether the board is
detected, where it is in frame, and which poses you still need.

The pose-coverage panel is the point. Calibration does not want many frames, it
wants *varied* frames, and "varied" is impossible to judge by eye while holding a
board. The panel tracks which image regions and which distances you have already
covered and fills them in as you go, so you can see when you are actually done
rather than guessing.

Keys:
  q / Esc   quit
  r         reset the coverage record
  s         save the current pair as PNG
  space     pause

Desktop-side. numpy + opencv (with GUI) + a working DISPLAY.
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

MAGIC = b"DEIR"
HDR = 14

GREEN = (60, 220, 60)
RED = (60, 60, 240)
AMBER = (40, 190, 250)
WHITE = (245, 245, 245)
GREY = (140, 140, 140)


def read_exact(stream, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def read_packet(stream):
    """Returns (stream_index, frame_number, image) or None at end of stream."""
    hdr = read_exact(stream, HDR)
    if hdr is None:
        return None
    if hdr[:4] != MAGIC:
        # Resynchronise rather than dying: a hiccup should cost a frame, not the
        # session.
        while True:
            b = stream.read(1)
            if not b:
                return None
            if b == MAGIC[:1]:
                rest = read_exact(stream, 3)
                if rest is None:
                    return None
                if rest == MAGIC[1:]:
                    hdr = MAGIC + (read_exact(stream, HDR - 4) or b"")
                    if len(hdr) < HDR:
                        return None
                    break
    w, h = struct.unpack_from("<HH", hdr, 4)
    index = hdr[8]
    number = struct.unpack_from("<I", hdr, 10)[0]
    payload = read_exact(stream, w * h)
    if payload is None:
        return None
    return index, number, np.frombuffer(payload, dtype=np.uint8).reshape(h, w)


def find_corners(img, cols, rows):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, c = cv2.findChessboardCorners(img, (cols, rows), flags=flags)
    if not ok:
        return None
    return c.reshape(-1, 2)


def label(img, text, org, colour, scale=0.5, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thick, cv2.LINE_AA)


class Coverage:
    """Which image regions and distances have produced a usable detection.

    Distance is proxied by the board's pixel width, which is what actually
    matters for calibration: it is the observed scale, not the metric range.
    """

    def __init__(self, w, h, grid=4, bands=4):
        self.w, self.h, self.grid, self.bands = w, h, grid, bands
        self.reset()

    def reset(self):
        self.cells = np.zeros((self.grid, self.grid), dtype=int)
        self.scale_bands = np.zeros(self.bands, dtype=int)
        self.tilts = []
        self.accepted = []
        self.n = 0

    def add(self, corners, w, h):
        self.n += 1
        cx, cy = corners[:, 0].mean(), corners[:, 1].mean()
        gx = min(self.grid - 1, int(cx / w * self.grid))
        gy = min(self.grid - 1, int(cy / h * self.grid))
        self.cells[gy, gx] += 1

        span = corners[:, 0].max() - corners[:, 0].min()
        frac = span / w
        # Bands from "far/small" to "near/large" across the useful range.
        edges = [0.15, 0.30, 0.45]
        band = sum(frac > e for e in edges)
        self.scale_bands[min(band, self.bands - 1)] += 1

        # Tilt proxy: a fronto-parallel board has near-equal left and right edge
        # heights. Their ratio departs from 1 as it is rotated about the vertical.
        span_y_left = corners[:, 1].max() - corners[:, 1].min()
        left = corners[corners[:, 0] < cx]
        right = corners[corners[:, 0] >= cx]
        if len(left) and len(right) and span_y_left > 1:
            hl = left[:, 1].max() - left[:, 1].min()
            hr = right[:, 1].max() - right[:, 1].min()
            if min(hl, hr) > 1:
                self.tilts.append(max(hl, hr) / min(hl, hr))

    def is_novel(self, corners, w, h, min_centre_px=45.0, min_scale_frac=0.06):
        """True if this pose differs enough from everything already accepted.

        Auto-collection is only useful if it rejects near-duplicates: fifty
        frames of the same pose constrain calibration no better than one, and
        they make the set look adequate when it is not.
        """
        cx, cy = corners[:, 0].mean(), corners[:, 1].mean()
        span = (corners[:, 0].max() - corners[:, 0].min()) / w
        for pcx, pcy, pspan in self.accepted:
            if (abs(cx - pcx) < min_centre_px and abs(cy - pcy) < min_centre_px
                    and abs(span - pspan) < min_scale_frac):
                return False
        self.accepted.append((cx, cy, span))
        return True

    def missing_summary(self):
        out = []
        empty = int((self.cells == 0).sum())
        if empty:
            out.append(f"{empty}/{self.grid * self.grid} image regions")
        empty_bands = int((self.scale_bands == 0).sum())
        if empty_bands:
            out.append(f"{empty_bands}/{self.bands} distances")
        tilted = sum(1 for t in self.tilts if t > 1.15)
        if tilted < 8:
            out.append(f"tilted views ({tilted}/8)")
        return out

    def render(self, size=260):
        panel = np.zeros((size, size, 3), dtype=np.uint8)
        cell = size // self.grid
        top = self.cells.max() if self.cells.max() else 1
        for gy in range(self.grid):
            for gx in range(self.grid):
                n = self.cells[gy, gx]
                if n == 0:
                    col = (35, 35, 35)
                else:
                    t = min(1.0, n / max(3.0, top * 0.6))
                    col = (int(40 + 40 * t), int(70 + 150 * t), int(40 + 40 * t))
                cv2.rectangle(panel, (gx * cell + 1, gy * cell + 1),
                              ((gx + 1) * cell - 2, (gy + 1) * cell - 2), col, -1)
                if n:
                    label(panel, str(int(n)),
                          (gx * cell + cell // 2 - 6, gy * cell + cell // 2 + 5),
                          WHITE, 0.45)
        return panel


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="jetson")
    ap.add_argument("--remote-dir", default="doubleeye")
    ap.add_argument("--cols", type=int, default=9, help="interior corners across")
    ap.add_argument("--rows", type=int, default=6, help="interior corners down")
    ap.add_argument("--out-fps", type=float, default=10.0)
    ap.add_argument("--gain", type=int, default=96)
    ap.add_argument("--exposure-us", type=int, default=1500)
    ap.add_argument("--emitter", default="off", choices=("on", "off"))
    ap.add_argument("--no-detect", action="store_true",
                    help="just show the streams, skip corner detection")
    ap.add_argument("--collect", metavar="NAME", default=None,
                    help="auto-save novel detected poses into bags/NAME, ready "
                         "for check_checkerboard.py and calibration")
    ap.add_argument("--save-dir", type=Path, default=Path("bags/live"))
    args = ap.parse_args()

    collect_dir = None
    if args.collect:
        collect_dir = Path("bags") / args.collect
        (collect_dir / "frames").mkdir(parents=True, exist_ok=True)
        (collect_dir / "run.txt").write_text(
            f"resolution 848x480 @ {args.out_fps:.0f}\n"
            f"exposure_us {args.exposure_us}\ngain {args.gain}\n"
            f"emitter {args.emitter}\nsource live_view auto-collect\n")
        print(f"auto-collecting novel poses into {collect_dir}")

    remote = (f"~/{args.remote_dir}/jetson/build/rs_ir_stream"
              f" --out-fps {args.out_fps} --gain {args.gain}"
              f" --exposure-us {args.exposure_us} --emitter {args.emitter}")
    proc = subprocess.Popen(["ssh", args.host, remote],
                            stdout=subprocess.PIPE, stderr=None, bufsize=0)

    print("live view starting — keys: q quit, r reset coverage, s save, space pause")
    pending: dict[int, np.ndarray] = {}
    cov = None
    paused = False
    saved = 0
    win = "DoubleEye live — IR stereo + checkerboard"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            pkt = read_packet(proc.stdout)
            if pkt is None:
                print("stream ended")
                break
            index, number, img = pkt
            pending[index] = img
            if 1 not in pending or 2 not in pending:
                continue
            ir1, ir2 = pending.pop(1), pending.pop(2)
            h, w = ir1.shape
            if cov is None:
                cov = Coverage(w, h)

            if not paused:
                c1 = None if args.no_detect else find_corners(ir1, args.cols, args.rows)
                c2 = None if args.no_detect else find_corners(ir2, args.cols, args.rows)
                if c1 is not None and c2 is not None:
                    cov.add(c1, w, h)
                    if collect_dir is not None and cov.is_novel(c1, w, h):
                        n = len(cov.accepted)
                        ir1.tofile(collect_dir / "frames" / f"ir1_{n:08d}.raw")
                        ir2.tofile(collect_dir / "frames" / f"ir2_{n:08d}.raw")
                last = (ir1, ir2, c1, c2)

            ir1_c = cv2.cvtColor(last[0], cv2.COLOR_GRAY2BGR)
            ir2_c = cv2.cvtColor(last[1], cv2.COLOR_GRAY2BGR)
            for canvas, corners, name in ((ir1_c, last[2], "ir1 LEFT"),
                                          (ir2_c, last[3], "ir2 RIGHT")):
                if corners is not None:
                    cv2.drawChessboardCorners(
                        canvas, (args.cols, args.rows),
                        corners.reshape(-1, 1, 2).astype(np.float32), True)
                    x0, y0 = corners.min(axis=0).astype(int)
                    x1, y1 = corners.max(axis=0).astype(int)
                    cv2.rectangle(canvas, (x0, y0), (x1, y1), GREEN, 1)
                    label(canvas, f"{name}  DETECTED  {int(x1 - x0)}px wide",
                          (10, 22), GREEN, 0.55)
                else:
                    label(canvas, f"{name}  no board", (10, 22), RED, 0.55)
                label(canvas, f"mean {last[0].mean():.0f} DN"
                      if name.startswith("ir1") else
                      f"mean {last[1].mean():.0f} DN", (10, h - 12), GREY, 0.45)

            both = last[2] is not None and last[3] is not None
            view = np.hstack([ir1_c, ir2_c])

            bar = np.zeros((150, view.shape[1], 3), dtype=np.uint8)
            if args.no_detect:
                label(bar, "detection disabled (--no-detect)", (14, 30), GREY, 0.6)
            elif both:
                label(bar, "GOOD — hold steady, then move to a new pose",
                      (14, 30), GREEN, 0.7, 2)
            elif last[2] is not None or last[3] is not None:
                label(bar, "ONE CHANNEL ONLY — board is clipped in the other; "
                           "move it toward the centre", (14, 30), AMBER, 0.6, 2)
            else:
                label(bar, "NOT DETECTED — see the checklist printed at startup",
                      (14, 30), RED, 0.7, 2)

            if cov is not None:
                kept = len(cov.accepted)
                extra = (f"   collected: {kept}" if collect_dir is not None
                         else "")
                label(bar, f"usable poses: {cov.n}{extra}", (14, 62), WHITE, 0.55)
                miss = cov.missing_summary()
                label(bar, "still needed: " + (", ".join(miss) if miss
                                               else "nothing — you are done"),
                      (14, 88), AMBER if miss else GREEN, 0.55)
                bands = "  ".join(
                    f"{n}" for n in cov.scale_bands)
                label(bar, f"poses per distance band (far->near): {bands}",
                      (14, 114), GREY, 0.5)
                label(bar, "q quit   r reset   s save   space pause"
                           + ("   [PAUSED]" if paused else ""),
                      (view.shape[1] - 330, 114), GREY, 0.45)

            panel = cov.render(size=150) if cov is not None else np.zeros(
                (150, 150, 3), np.uint8)
            label(panel, "coverage", (6, 14), WHITE, 0.4)
            bar[:, -150:] = panel

            cv2.imshow(win, np.vstack([view, bar]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r") and cov is not None:
                cov.reset()
                print("coverage reset")
            if key == ord(" "):
                paused = not paused
            if key == ord("s"):
                args.save_dir.mkdir(parents=True, exist_ok=True)
                for nm, im in (("ir1", last[0]), ("ir2", last[1])):
                    cv2.imwrite(str(args.save_dir / f"{nm}_{saved:04d}.png"), im)
                print(f"saved pair {saved} to {args.save_dir}")
                saved += 1
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


if __name__ == "__main__":
    print(__doc__.split("Keys:")[0])
    print("If the board is never detected, check in this order:")
    print("  1. Is it visible at all in the window? If yes but never detected,")
    print("     suspect the PRINTER: many inkjet blacks are near-transparent at")
    print("     850 nm. Use a laser print.")
    print("  2. Do --cols/--rows match? They are INTERIOR corners, so the")
    print("     default 10x7-square board is 9x6.")
    print("  3. Is the whole board inside the frame and in focus?")
    print()
    sys.exit(main())
