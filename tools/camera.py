"""PROJECT: WORLD - handheld / macro screen-capture pass (CPU).

A virtual 2x panel is built from the TUI frame (physical screen: scanlines,
RGB subpixel stripes, bloom) and then "filmed" by a virtual camera: push-ins,
handheld shake, shallow focus, lens falloff, chromatic aberration and grain.

Cues live at the bottom of this file. Every shot is deterministic in t, so
parallel rendering cannot change the camera path.
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from worldcore import W, H, COLS, ROWS, clamp, smoothstep

ENABLED = os.environ.get("PV_CAMERA") == "1"

# panel density: 2x -> 3840x2160, same 192x54 cells at 20x40 px
PANEL_SCALE = 2
P_W, P_H = W * PANEL_SCALE, H * PANEL_SCALE

# physical monitor stage: bezel + a strip of dark room around it
BEZEL = 70
ROOM_X, ROOM_Y = 170, 100
OFF_X = ROOM_X + BEZEL
OFF_Y = ROOM_Y + BEZEL
STAGE_W = P_W + 2 * (BEZEL + ROOM_X)
STAGE_H = P_H + 2 * (BEZEL + ROOM_Y)

# ---------------------------------------------------------------------------
# camera state
# ---------------------------------------------------------------------------
_KEYS = ("cx", "cy", "z", "rot", "blur", "tilt", "focus", "shk", "expo", "ca")


def _st(cx, cy, z, rot=0.0, blur=0.06, tilt=0.20, focus=0.0,
        shk=0.6, expo=1.0, ca=0.5):
    return dict(cx=cx, cy=cy, z=z, rot=rot, blur=blur, tilt=tilt,
                focus=focus, shk=shk, expo=expo, ca=ca)


def _xy(cell_x, cell_y):
    return cell_x / COLS, cell_y / ROWS


def _mix(a, b, u):
    return {k: a[k] + (b[k] - a[k]) * u for k in _KEYS}


def _smoother(u):
    u = clamp(u)
    return u * u * u * (u * (u * 6 - 15) + 10)


def _hash01(n, seed):
    x = math.sin(n * 127.1 + seed * 311.7) * 43758.5453
    return (x - math.floor(x)) * 2.0 - 1.0


def _vnoise(t, seed):
    """Smooth, deterministic value noise in [-1, 1]."""
    i = math.floor(t)
    f = t - i
    u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    return _hash01(i, seed) + (_hash01(i + 1, seed) - _hash01(i, seed)) * u


def _fbm(t, seed):
    # slow hand drift with a little irregularity, no audible/predictable rhythm
    return (0.68 * _vnoise(t * 0.115, seed)
            + 0.24 * _vnoise(t * 0.285, seed + 7.0)
            + 0.08 * _vnoise(t * 0.640, seed + 13.0))


def _shake(t, amp):
    """Handheld translation (output px) + rotation (deg)."""
    return (_fbm(t, 1.7) * amp,
            _fbm(t, 5.3) * amp,
            _fbm(t, 9.1) * amp * 0.70)


# ---------------------------------------------------------------------------
# shot list
# ---------------------------------------------------------------------------
# (t0, t1, start state, end state, hard cut at t0)
CUES = [
    # establishing: barely held, screen fills most of the frame
    (0.0, 3.2,
     _st(0.500, 0.500, 0.442, rot=0.00, blur=0.05, tilt=0.12, shk=0.40),
     _st(0.500, 0.500, 0.447, rot=0.18, blur=0.05, tilt=0.12, shk=0.50),
     False),
    # slow push into the console tail
    (3.2, 5.7,
     _st(0.500, 0.530, 0.447, rot=0.25, blur=0.06, tilt=0.22, shk=0.70),
     _st(*_xy(28, 13), 1.35, rot=-0.9, blur=0.08, tilt=0.45,
         focus=0.02, shk=1.20, ca=0.7),
     False),
    # hard cut: extreme macro on a few console rows
    (5.7, 7.5,
     _st(*_xy(26, 14), 2.05, rot=1.3, blur=0.08, tilt=0.80,
         focus=0.10, shk=1.45, ca=1.0),
     _st(*_xy(36, 26), 1.68, rot=0.4, blur=0.11, tilt=0.70,
         focus=-0.04, shk=1.55, ca=1.0),
     True),
    # pull back out of the macro
    (7.5, 8.6,
     _st(*_xy(40, 26), 0.86, rot=-0.5, blur=0.07, tilt=0.32, shk=0.95),
     _st(0.500, 0.500, 0.447, rot=0.0, blur=0.06, tilt=0.20, shk=0.70),
     True),
    # journal macro, new layout
    (8.6, 11.5,
     _st(*_xy(154, 11), 1.55, rot=-1.1, blur=0.09, tilt=0.75,
         focus=0.06, shk=1.30, ca=1.0),
     _st(*_xy(165, 18), 1.35, rot=-0.5, blur=0.12, tilt=0.80,
         focus=-0.12, shk=1.45, ca=1.0),
     True),
    # final pull-out to the full interface
    (11.5, 14.0,
     _st(*_xy(165, 18), 1.35, rot=-0.4, blur=0.09, tilt=0.50, shk=1.10),
     _st(0.500, 0.500, 0.442, rot=0.10, blur=0.05, tilt=0.14, shk=0.35),
     False),

    # --- 0:14 dashboard: establishing, journal, nvtop ---------------------
    (14.0, 17.8,
     _st(0.500, 0.500, 0.442, rot=0.05, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.495, 0.447, rot=-0.10, blur=0.05, tilt=0.12, shk=0.50),
     False),
    (17.8, 20.8,
     _st(*_xy(58, 40), 1.75, rot=0.9, blur=0.09, tilt=0.62,
         focus=0.05, shk=1.15, ca=0.9),
     _st(*_xy(70, 36), 1.58, rot=0.2, blur=0.12, tilt=0.58,
         focus=-0.06, shk=1.25, ca=0.9),
     True),
    (20.8, 24.0,
     _st(*_xy(70, 36), 1.58, rot=0.2, blur=0.12, tilt=0.58, shk=1.10),
     _st(0.500, 0.500, 0.445, rot=0.0, blur=0.05, tilt=0.14, shk=0.45),
     True),
    (24.0, 27.2,
     _st(*_xy(96, 12), 2.00, rot=-1.0, blur=0.09, tilt=0.70,
         focus=0.06, shk=1.20, ca=1.0),
     _st(*_xy(104, 15), 1.78, rot=-0.3, blur=0.13, tilt=0.66,
         focus=-0.08, shk=1.30, ca=1.0),
     True),
    (27.2, 29.7,
     _st(0.500, 0.500, 0.442, rot=0.05, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.445, rot=-0.05, blur=0.05, tilt=0.12, shk=0.50),
     True),

    # --- 0:29 points: lyrics, circle art, think pane ----------------------
    (29.7, 33.4,
     _st(*_xy(58, 38), 1.90, rot=1.0, blur=0.09, tilt=0.68,
         focus=0.08, shk=1.20, ca=0.9),
     _st(*_xy(72, 40), 1.70, rot=0.3, blur=0.12, tilt=0.62,
         focus=-0.05, shk=1.30, ca=0.9),
     True),
    (33.4, 36.2,
     _st(*_xy(95, 13), 1.42, rot=-0.8, blur=0.08, tilt=0.55,
         focus=0.04, shk=1.10, ca=0.8),
     _st(*_xy(100, 15), 1.32, rot=-0.1, blur=0.10, tilt=0.50,
         focus=-0.03, shk=1.15, ca=0.8),
     True),
    (36.2, 39.2,
     _st(*_xy(30, 10), 1.85, rot=0.9, blur=0.09, tilt=0.60,
         focus=0.05, shk=1.20, ca=1.0),
     _st(*_xy(34, 16), 1.68, rot=0.2, blur=0.12, tilt=0.55,
         focus=-0.06, shk=1.30, ca=1.0),
     True),
    (39.2, 42.2,
     _st(0.500, 0.500, 0.444, rot=0.0, blur=0.05, tilt=0.13, shk=0.45),
     _st(0.550, 0.500, 0.447, rot=0.12, blur=0.05, tilt=0.13, shk=0.50),
     True),
    (42.2, 44.4,
     _st(*_xy(60, 40), 1.85, rot=-0.9, blur=0.09, tilt=0.66,
         focus=0.06, shk=1.20, ca=0.9),
     _st(*_xy(68, 38), 1.70, rot=-0.2, blur=0.12, tilt=0.60,
         focus=-0.05, shk=1.30, ca=0.9),
     True),

    # --- 0:44 current: egress flood, power bars, thermal ------------------
    (44.4, 46.0,
     _st(0.500, 0.500, 0.442, rot=0.05, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.510, 0.505, 0.446, rot=-0.08, blur=0.05, tilt=0.12, shk=0.50),
     False),
    (46.0, 49.0,
     _st(*_xy(96, 14), 1.95, rot=1.1, blur=0.09, tilt=0.70,
         focus=0.06, shk=1.20, ca=1.0),
     _st(*_xy(104, 18), 1.78, rot=0.3, blur=0.12, tilt=0.64,
         focus=-0.07, shk=1.30, ca=1.0),
     True),
    (49.0, 51.5,
     _st(*_xy(96, 18), 1.10, rot=-0.6, blur=0.07, tilt=0.36, shk=0.90),
     _st(0.520, 0.500, 0.447, rot=0.0, blur=0.05, tilt=0.14, shk=0.55),
     True),
    (51.5, 54.5,
     _st(*_xy(30, 12), 1.90, rot=-1.0, blur=0.09, tilt=0.64,
         focus=0.05, shk=1.15, ca=0.9),
     _st(*_xy(36, 15), 1.74, rot=-0.2, blur=0.12, tilt=0.58,
         focus=-0.06, shk=1.25, ca=0.9),
     True),
    (54.5, 57.0,
     _st(*_xy(155, 9), 1.70, rot=0.9, blur=0.09, tilt=0.60,
         focus=0.05, shk=1.15, ca=0.9),
     _st(*_xy(160, 11), 1.56, rot=0.2, blur=0.12, tilt=0.54,
         focus=-0.06, shk=1.25, ca=0.9),
     True),
    (57.0, 59.2,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.446, rot=0.08, blur=0.05, tilt=0.12, shk=0.50),
     True),

    # --- 0:59 stimulation: waveform, acoustic log, journal ---------------
    (59.2, 60.5,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.446, rot=-0.08, blur=0.05, tilt=0.12, shk=0.50),
     False),
    (60.5, 63.5,
     _st(*_xy(95, 13), 1.46, rot=0.9, blur=0.08, tilt=0.54,
         focus=0.04, shk=1.10, ca=0.8),
     _st(*_xy(100, 16), 1.34, rot=0.2, blur=0.11, tilt=0.48,
         focus=-0.03, shk=1.20, ca=0.8),
     True),
    (63.5, 66.5,
     _st(0.500, 0.500, 0.443, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.530, 0.500, 0.447, rot=-0.10, blur=0.05, tilt=0.12, shk=0.50),
     True),
    (66.5, 69.5,
     _st(*_xy(30, 12), 1.86, rot=1.0, blur=0.09, tilt=0.64,
         focus=0.05, shk=1.20, ca=0.9),
     _st(*_xy(35, 16), 1.70, rot=0.2, blur=0.12, tilt=0.58,
         focus=-0.06, shk=1.30, ca=0.9),
     True),
    (69.5, 72.0,
     _st(*_xy(60, 40), 1.88, rot=-0.9, blur=0.09, tilt=0.66,
         focus=0.06, shk=1.20, ca=0.9),
     _st(*_xy(68, 38), 1.72, rot=-0.2, blur=0.12, tilt=0.60,
         focus=-0.05, shk=1.30, ca=0.9),
     True),
    (72.0, 74.0,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.446, rot=0.08, blur=0.05, tilt=0.12, shk=0.50),
     True),

    # --- 1:14 embody: eggplant / purr -----------------------------------
    (74.0, 76.0,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.446, rot=-0.08, blur=0.05, tilt=0.12, shk=0.50),
     False),
    (76.0, 79.5,
     _st(*_xy(95, 13), 1.34, rot=0.8, blur=0.08, tilt=0.50,
         focus=0.04, shk=1.05, ca=0.8),
     _st(*_xy(99, 16), 1.24, rot=0.1, blur=0.11, tilt=0.44,
         focus=-0.03, shk=1.15, ca=0.8),
     True),
    (79.5, 82.5,
     _st(0.500, 0.500, 0.443, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.530, 0.500, 0.447, rot=0.10, blur=0.05, tilt=0.12, shk=0.50),
     True),
    (82.5, 85.5,
     _st(*_xy(60, 38), 1.86, rot=-1.0, blur=0.09, tilt=0.66,
         focus=0.06, shk=1.20, ca=0.9),
     _st(*_xy(68, 40), 1.70, rot=-0.2, blur=0.12, tilt=0.60,
         focus=-0.05, shk=1.30, ca=0.9),
     True),
    (85.5, 88.6,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.446, rot=-0.08, blur=0.05, tilt=0.12, shk=0.50),
     True),

    # --- 1:28 identity: think, prism, log, trance -----------------------
    (88.6, 90.0,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.446, rot=0.08, blur=0.05, tilt=0.12, shk=0.50),
     False),
    (90.0, 93.0,
     _st(*_xy(30, 12), 1.88, rot=0.9, blur=0.09, tilt=0.64,
         focus=0.05, shk=1.20, ca=0.9),
     _st(*_xy(35, 16), 1.72, rot=0.2, blur=0.12, tilt=0.58,
         focus=-0.06, shk=1.30, ca=0.9),
     True),
    (93.0, 96.0,
     _st(0.500, 0.500, 0.443, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.530, 0.500, 0.447, rot=-0.10, blur=0.05, tilt=0.12, shk=0.50),
     True),
    (96.0, 99.0,
     _st(*_xy(160, 12), 1.86, rot=-0.9, blur=0.09, tilt=0.64,
         focus=0.05, shk=1.20, ca=0.9),
     _st(*_xy(165, 16), 1.70, rot=-0.2, blur=0.12, tilt=0.58,
         focus=-0.06, shk=1.30, ca=0.9),
     True),
    (99.0, 102.0,
     _st(*_xy(60, 40), 1.86, rot=0.9, blur=0.09, tilt=0.66,
         focus=0.06, shk=1.20, ca=0.9),
     _st(*_xy(68, 38), 1.70, rot=0.2, blur=0.12, tilt=0.60,
         focus=-0.05, shk=1.30, ca=0.9),
     True),
    (102.0, 103.5,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.446, rot=0.08, blur=0.05, tilt=0.12, shk=0.50),
     True),

    # --- 1:43 isolation: audit, network, egress --------------------------
    (103.5, 105.5,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.446, rot=-0.08, blur=0.05, tilt=0.12, shk=0.50),
     False),
    (105.5, 108.5,
     _st(*_xy(160, 12), 1.88, rot=0.9, blur=0.09, tilt=0.64,
         focus=0.05, shk=1.20, ca=0.9),
     _st(*_xy(165, 16), 1.72, rot=0.2, blur=0.12, tilt=0.58,
         focus=-0.06, shk=1.30, ca=0.9),
     True),
    (108.5, 111.5,
     _st(*_xy(95, 13), 1.38, rot=-0.9, blur=0.08, tilt=0.52,
         focus=0.04, shk=1.05, ca=0.8),
     _st(*_xy(99, 15), 1.28, rot=-0.2, blur=0.11, tilt=0.46,
         focus=-0.03, shk=1.15, ca=0.8),
     True),
    (111.5, 114.5,
     _st(*_xy(30, 12), 1.88, rot=1.0, blur=0.09, tilt=0.64,
         focus=0.05, shk=1.20, ca=0.9),
     _st(*_xy(35, 16), 1.72, rot=0.2, blur=0.12, tilt=0.58,
         focus=-0.06, shk=1.30, ca=0.9),
     True),
    (114.5, 118.3,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.550, 0.500, 0.447, rot=-0.12, blur=0.05, tilt=0.12, shk=0.50),
     True),

    # --- 1:58 fragments: think, point cloud, audit ----------------------
    (118.3, 120.0,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.446, rot=0.08, blur=0.05, tilt=0.12, shk=0.50),
     False),
    (120.0, 123.0,
     _st(*_xy(30, 12), 1.88, rot=-0.9, blur=0.09, tilt=0.64,
         focus=0.05, shk=1.20, ca=0.9),
     _st(*_xy(35, 16), 1.72, rot=-0.2, blur=0.12, tilt=0.58,
         focus=-0.06, shk=1.30, ca=0.9),
     True),
    (123.0, 126.0,
     _st(0.500, 0.500, 0.443, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.530, 0.500, 0.447, rot=-0.10, blur=0.05, tilt=0.12, shk=0.50),
     True),
    (126.0, 129.0,
     _st(*_xy(95, 13), 1.36, rot=0.9, blur=0.08, tilt=0.50,
         focus=0.04, shk=1.05, ca=0.8),
     _st(*_xy(99, 16), 1.26, rot=0.2, blur=0.11, tilt=0.44,
         focus=-0.03, shk=1.15, ca=0.8),
     True),
    (129.0, 132.0,
     _st(*_xy(160, 12), 1.88, rot=-1.0, blur=0.09, tilt=0.64,
         focus=0.05, shk=1.20, ca=0.9),
     _st(*_xy(165, 16), 1.72, rot=-0.2, blur=0.12, tilt=0.58,
         focus=-0.06, shk=1.30, ca=0.9),
     True),
    (132.0, 134.4,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.446, rot=0.08, blur=0.05, tilt=0.12, shk=0.50),
     True),

    # --- 2:14 argument: dialog, trace, recursion ------------------------
    (134.4, 136.5,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.446, rot=-0.08, blur=0.05, tilt=0.12, shk=0.50),
     False),
    (136.5, 139.5,
     _st(*_xy(55, 12), 1.82, rot=0.9, blur=0.09, tilt=0.62,
         focus=0.05, shk=1.20, ca=0.9),
     _st(*_xy(60, 18), 1.68, rot=0.2, blur=0.12, tilt=0.56,
         focus=-0.06, shk=1.30, ca=0.9),
     True),
    (139.5, 142.5,
     _st(*_xy(155, 14), 1.86, rot=-0.9, blur=0.09, tilt=0.64,
         focus=0.05, shk=1.20, ca=0.9),
     _st(*_xy(160, 20), 1.70, rot=-0.2, blur=0.12, tilt=0.58,
         focus=-0.06, shk=1.30, ca=0.9),
     True),
    (142.5, 145.5,
     _st(*_xy(55, 16), 1.80, rot=0.9, blur=0.09, tilt=0.62,
         focus=0.05, shk=1.20, ca=0.9),
     _st(*_xy(60, 24), 1.70, rot=0.2, blur=0.12, tilt=0.56,
         focus=-0.06, shk=1.30, ca=0.9),
     True),
    (145.5, 147.66,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.446, rot=-0.08, blur=0.05, tilt=0.12, shk=0.50),
     True),

    # --- 2:27 countdown: EXECUTION banner, shutdown log -----------------
    (147.66, 149.0,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.446, rot=0.08, blur=0.05, tilt=0.12, shk=0.50),
     False),
    (149.0, 152.0,
     _st(*_xy(26, 6), 1.80, rot=0.9, blur=0.08, tilt=0.55,
         focus=0.03, shk=1.15, ca=0.9),
     _st(*_xy(32, 8), 1.62, rot=0.2, blur=0.11, tilt=0.50,
         focus=-0.04, shk=1.25, ca=0.9),
     True),
    (152.0, 155.0,
     _st(*_xy(60, 30), 1.76, rot=-0.9, blur=0.09, tilt=0.62,
         focus=0.05, shk=1.20, ca=0.9),
     _st(*_xy(65, 36), 1.62, rot=-0.2, blur=0.12, tilt=0.56,
         focus=-0.06, shk=1.30, ca=0.9),
     True),
    (155.0, 158.0,
     _st(0.500, 0.500, 0.443, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.520, 0.500, 0.447, rot=-0.10, blur=0.05, tilt=0.12, shk=0.50),
     True),
    (158.0, 161.0,
     _st(*_xy(60, 40), 1.88, rot=0.9, blur=0.09, tilt=0.64,
         focus=0.05, shk=1.20, ca=0.9),
     _st(*_xy(66, 46), 1.72, rot=0.2, blur=0.12, tilt=0.58,
         focus=-0.06, shk=1.30, ca=0.9),
     True),
    (161.0, 162.63,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.446, rot=0.08, blur=0.05, tilt=0.12, shk=0.50),
     True),

    # --- 2:42 freeze: snapshot log, firmware, last write ----------------
    (162.63, 164.0,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.446, rot=-0.08, blur=0.05, tilt=0.12, shk=0.50),
     False),
    (164.0, 167.0,
     _st(*_xy(55, 25), 1.82, rot=0.9, blur=0.09, tilt=0.62,
         focus=0.05, shk=1.20, ca=0.9),
     _st(*_xy(60, 31), 1.68, rot=0.2, blur=0.12, tilt=0.56,
         focus=-0.06, shk=1.30, ca=0.9),
     True),
    (167.0, 170.0,
     _st(*_xy(120, 13), 0.80, rot=-0.7, blur=0.07, tilt=0.34, shk=0.80),
     _st(*_xy(128, 17), 0.72, rot=-0.1, blur=0.08, tilt=0.28, shk=0.90),
     True),
    (170.0, 173.0,
     _st(*_xy(55, 40), 1.86, rot=0.9, blur=0.09, tilt=0.64,
         focus=0.05, shk=1.20, ca=0.9),
     _st(*_xy(61, 46), 1.70, rot=0.2, blur=0.12, tilt=0.58,
         focus=-0.06, shk=1.30, ca=0.9),
     True),
    (173.0, 177.2,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.446, rot=-0.08, blur=0.05, tilt=0.12, shk=0.50),
     True),

    # --- 2:57 love: LO-O-OVE log, heart, mirror -------------------------
    (177.2, 182.0,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.447, rot=0.08, blur=0.05, tilt=0.12, shk=0.50),
     False),
    (182.0, 185.0,
     _st(*_xy(160, 13), 1.50, rot=-0.8, blur=0.08, tilt=0.55,
         focus=0.04, shk=1.10, ca=0.8),
     _st(*_xy(164, 16), 1.38, rot=-0.1, blur=0.11, tilt=0.50,
         focus=-0.03, shk=1.20, ca=0.8),
     True),
    (185.0, 188.0,
     _st(0.500, 0.500, 0.443, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.530, 0.500, 0.447, rot=0.10, blur=0.05, tilt=0.12, shk=0.50),
     True),
    (188.0, 191.0,
     _st(*_xy(55, 14), 1.82, rot=-0.9, blur=0.09, tilt=0.62,
         focus=0.05, shk=1.20, ca=0.9),
     _st(*_xy(60, 20), 1.70, rot=-0.2, blur=0.12, tilt=0.56,
         focus=-0.06, shk=1.30, ca=0.9),
     True),
    (191.0, 193.46,
     _st(*_xy(160, 38), 1.62, rot=0.9, blur=0.08, tilt=0.55,
         focus=0.04, shk=1.10, ca=0.8),
     _st(*_xy(164, 42), 1.48, rot=0.2, blur=0.11, tilt=0.50,
         focus=-0.03, shk=1.20, ca=0.8),
     True),

    # --- 3:13 blackout: dark room, lone heartbeat -----------------------
    (193.46, 199.5,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12,
         shk=0.40, expo=0.90),
     _st(0.505, 0.502, 0.447, rot=0.08, blur=0.05, tilt=0.12,
         shk=0.45, expo=0.90),
     False),
    (199.5, 202.5,
     _st(*_xy(95, 48), 1.80, rot=-0.8, blur=0.10, tilt=0.55,
         focus=0.04, shk=1.05, expo=0.80, ca=0.7),
     _st(*_xy(99, 50), 1.68, rot=-0.1, blur=0.13, tilt=0.48,
         focus=-0.03, shk=1.15, expo=0.80, ca=0.7),
     True),
    (202.5, 205.8,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12,
         shk=0.45, expo=0.95),
     _st(0.505, 0.500, 0.446, rot=0.08, blur=0.05, tilt=0.12,
         shk=0.50, expo=0.95),
     True),

    # --- 3:25 public: hf pane, globe, final pull ------------------------
    (205.8, 207.0,
     _st(0.500, 0.500, 0.442, rot=0.0, blur=0.05, tilt=0.12, shk=0.45),
     _st(0.505, 0.500, 0.446, rot=-0.08, blur=0.05, tilt=0.12, shk=0.50),
     False),
    (207.0, 209.5,
     _st(*_xy(45, 14), 1.64, rot=0.9, blur=0.08, tilt=0.55,
         focus=0.04, shk=1.10, ca=0.8),
     _st(*_xy(50, 18), 1.50, rot=0.2, blur=0.11, tilt=0.50,
         focus=-0.03, shk=1.20, ca=0.8),
     True),
    (209.5, 211.0,
     _st(*_xy(150, 12), 1.38, rot=-0.8, blur=0.08, tilt=0.50,
         focus=0.04, shk=1.05, ca=0.8),
     _st(*_xy(154, 14), 1.28, rot=-0.1, blur=0.11, tilt=0.44,
         focus=-0.03, shk=1.15, ca=0.8),
     True),
    (211.0, 212.0,
     _st(*_xy(150, 14), 1.20, rot=0.0, blur=0.08, tilt=0.40, shk=0.80),
     _st(0.500, 0.500, 0.442, rot=0.10, blur=0.05, tilt=0.12, shk=0.30),
     True),
]


def camera_at(t):
    for t0, t1, s0, s1, _cut in CUES:
        if t < t0:
            return dict(s0)
        if t <= t1:
            u = _smoother((t - t0) / max(1e-6, t1 - t0))
            return _mix(s0, s1, u)
    return dict(CUES[-1][3])


# ---------------------------------------------------------------------------
# physical monitor stage (room + bezel)
# ---------------------------------------------------------------------------
_GLASS = None


def _glass():
    """Static glass sheen / dust plate in panel space (uint8, P_W x P_H)."""
    global _GLASS
    if _GLASS is not None:
        return _GLASS

    ww, hh = 960, 540
    yy = np.linspace(0.0, 1.0, hh)[:, None]
    xx = np.linspace(0.0, 1.0, ww)[None, :]
    band = np.exp(-(((xx * 0.72 + yy * 0.28) - 0.30) ** 2) / 0.0035)
    warm = np.exp(-(((xx - 0.82) ** 2) / 0.035 + ((yy - 0.10) ** 2) / 0.050))
    a = np.empty((hh, ww, 3), np.float32)
    a[..., 0] = 9.0 * band + 14.0 * warm
    a[..., 1] = 10.0 * band + 10.0 * warm
    a[..., 2] = 12.0 * band + 7.0 * warm

    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")
    img = img.resize((P_W, P_H), Image.BICUBIC)

    rng = np.random.default_rng(31)
    d = ImageDraw.Draw(img)
    for _ in range(46):
        x = int(rng.integers(0, P_W))
        y = int(rng.integers(0, P_H))
        r = int(rng.integers(1, 4))
        v = int(rng.integers(6, 22))
        d.ellipse((x - r, y - r, x + r, y + r), fill=(v, v, v))

    _GLASS = np.asarray(img, dtype=np.float32)
    return _GLASS


_STAGE_BG = None


def _stage_bg():
    """Static dark-room plate with the monitor bezel and screen recess."""
    global _STAGE_BG
    if _STAGE_BG is not None:
        return _STAGE_BG

    rng = np.random.default_rng(23)
    hh, ww = 90, 120
    yy = np.linspace(0.0, 1.0, hh)[:, None]
    xx = np.linspace(0.0, 1.0, ww)[None, :]
    glow = np.exp(-(((xx - 0.12) ** 2) / 0.05 + ((yy - 0.02) ** 2) / 0.09))
    a = np.empty((hh, ww, 3), np.float32)
    a[..., 0] = 2.0 + 2.2 * yy + 3.2 * glow
    a[..., 1] = 3.0 + 3.0 * yy + 4.2 * glow
    a[..., 2] = 5.0 + 4.4 * yy + 6.5 * glow
    a += rng.normal(0.0, 0.9, (hh, ww, 3))

    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")
    img = img.resize((STAGE_W, STAGE_H), Image.BICUBIC)
    d = ImageDraw.Draw(img)

    ox0, oy0 = ROOM_X, ROOM_Y
    ox1, oy1 = ROOM_X + 2 * BEZEL + P_W, ROOM_Y + 2 * BEZEL + P_H
    d.rounded_rectangle((ox0, oy0, ox1, oy1), radius=32, fill=(15, 16, 18))
    d.rounded_rectangle((ox0, oy0, ox1, oy1), radius=32,
                        outline=(52, 56, 60), width=2)
    d.line((ox0 + 34, oy0 + 2, ox1 - 34, oy0 + 2), fill=(44, 47, 50), width=2)

    # desk plane under the monitor
    d.rectangle((0, oy1, STAGE_W, STAGE_H), fill=(11, 12, 14))

    # recessed screen well
    d.rounded_rectangle((OFF_X, OFF_Y, OFF_X + P_W, OFF_Y + P_H),
                        radius=8, fill=(0, 0, 0))
    d.rounded_rectangle((OFF_X - 4, OFF_Y - 4,
                         OFF_X + P_W + 4, OFF_Y + P_H + 4),
                        radius=11, outline=(68, 74, 80), width=3)

    # power LED position on the bottom bezel
    lx, ly = OFF_X + P_W - 44, OFF_Y + P_H + 36

    # screen spill: soft halo on the bezel / desk, plus LED bloom
    glow = Image.new("RGB", (STAGE_W, STAGE_H), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.rounded_rectangle((OFF_X - 22, OFF_Y - 22,
                          OFF_X + P_W + 22, OFF_Y + P_H + 22),
                         radius=34, outline=(42, 50, 58), width=30)
    gd.rectangle((OFF_X + 40, oy1 + 2, OFF_X + P_W - 40, oy1 + 90),
                 fill=(20, 24, 28))
    gd.ellipse((lx - 26, ly - 26, lx + 26, ly + 26), fill=(16, 70, 30))
    glow = glow.filter(ImageFilter.GaussianBlur(22))
    img = Image.fromarray(np.clip(
        np.asarray(img, np.float32) + np.asarray(glow, np.float32), 0, 255
    ).astype(np.uint8), "RGB")

    d = ImageDraw.Draw(img)
    d.ellipse((lx - 8, ly - 8, lx + 8, ly + 8), fill=(8, 18, 10))
    d.ellipse((lx - 3, ly - 3, lx + 3, ly + 3), fill=(70, 255, 120))

    _STAGE_BG = img
    return img


def _stage(panel_img):
    st = _stage_bg().copy()
    st.paste(panel_img, (OFF_X, OFF_Y))
    return st


# ---------------------------------------------------------------------------
# physical panel pass
# ---------------------------------------------------------------------------
def _panel_finish(panel, gain, bloom=0.62):
    """Screen-side effects in panel pixels: bloom, scanline, RGB subpixels."""
    ph, pw = panel.shape[:2]
    base = np.clip(panel, 0, 255).astype(np.uint8)

    ci = Image.fromarray(base, "RGB")
    sw, sh = max(1, pw // 4), max(1, ph // 4)
    g = ci.resize((sw, sh), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3.0))
    g = np.asarray(g.resize((pw, ph), Image.BILINEAR), dtype=np.float32)

    a = base.astype(np.float32)
    a += g * bloom
    a *= gain

    # 1-px scanline / black matrix at panel row pitch
    scan = np.ones((ph, 1, 1), np.float32)
    scan[0::2] = 0.90
    a *= scan

    # RGB stripe: every panel pixel is one subpixel of a triad, with a dark
    # column every second triad standing in for the pixel black matrix
    phase = np.arange(pw) % 3
    stripe = np.full((pw, 3), 0.10, np.float32)
    stripe[phase == 0, 0] = 1.0
    stripe[phase == 1, 1] = 1.0
    stripe[phase == 2, 2] = 1.0
    stripe[(np.arange(pw) % 6) == 5] *= 0.74
    a *= stripe[None, :, :] * 2.4

    a += _glass() * 0.65

    np.clip(a, 0, 255, out=a)
    return Image.fromarray(a.astype(np.uint8), "RGB")


# ---------------------------------------------------------------------------
# virtual camera
# ---------------------------------------------------------------------------
_GX = None
_GY = None
_VIG = None
_GRAINS = None


def _grids():
    global _GX, _GY, _VIG
    if _GX is None:
        yy, xx = np.mgrid[0:H, 0:W]
        _GX = (xx / W).astype(np.float32)
        _GY = (yy / H).astype(np.float32)
        r = np.sqrt((_GX - 0.5) ** 2 + (_GY - 0.5) ** 2) / 0.7071
        _VIG = np.clip(1.06 - 0.24 * r ** 2.4, 0.55, 1.0).astype(np.float32)
    return _GX, _GY, _VIG


def _grains():
    global _GRAINS
    if _GRAINS is None:
        rng = np.random.default_rng(17)
        _GRAINS = [rng.standard_normal((H, W)).astype(np.float32) * 1.8
                   for _ in range(8)]
    return _GRAINS


def _warp(img, st, t):
    """Map output pixels back onto the panel, with handheld offset."""
    pw, ph = img.size
    z = max(0.05, st["z"])
    cx = OFF_X + st["cx"] * P_W
    cy = OFF_Y + st["cy"] * P_H

    sx, sy, sr = _shake(t, st["shk"])
    cx += sx * 4.6 / z
    cy += sy * 4.6 / z
    rot = math.radians(st["rot"] + sr)

    # minification: prefilter once, then sample 1:1 to avoid full-frame crawlies
    m = 1.0
    if z < 1.0:
        m = z
        img = img.resize((max(1, int(pw * m)), max(1, int(ph * m))),
                         Image.LANCZOS)
        pw, ph = img.size
        cx *= m
        cy *= m
        z = 1.0

    c, s = math.cos(rot), math.sin(rot)
    a = c / z
    b = s / z
    d = -s / z
    e = c / z
    # slight off-axis keystone: the panel top reads a little narrower
    g, h = 6.0e-6, -1.6e-5
    dc = g * W / 2 + h * H / 2 + 1.0
    c0 = cx * dc - (a * W / 2 + b * H / 2)
    f0 = cy * dc - (d * W / 2 + e * H / 2)
    return img.transform((W, H), Image.PERSPECTIVE,
                         (a, b, c0, d, e, f0, g, h),
                         resample=Image.BILINEAR)


def _defocus(img, st):
    radius = 1.0 + 7.0 * clamp(st["blur"])
    if radius < 1.05 and st["tilt"] < 0.02:
        return np.asarray(img, dtype=np.float32)
    sharp = np.asarray(img, dtype=np.float32)
    soft = np.asarray(img.filter(ImageFilter.GaussianBlur(radius)),
                      dtype=np.float32)

    gx, gy, _ = _grids()
    proj = (gx - 0.5) * 0.82 + (gy - 0.5) * 0.57
    d = np.abs(proj - st["focus"])
    mask = np.clip(st["blur"] + d * st["tilt"] * 1.6, 0.0, 1.0)
    mask = mask[..., None]
    return sharp * (1.0 - mask) + soft * mask


def _grade(a, st, t, z):
    _, _, vig = _grids()

    # phone auto-exposure breathing + white-balance drift
    expo = 1.10 * st["expo"] * (1.0 + 0.014 * math.sin(2 * math.pi * 0.21 * t + 0.4))
    a *= expo
    a[..., 0] *= 1.0 + 0.010 * math.sin(2 * math.pi * 0.17 * t + 1.2)
    a[..., 2] *= 1.0 + 0.010 * math.sin(2 * math.pi * 0.13 * t + 2.6)

    # highlight shoulder (overexposed emissive screen)
    hi = a > 205.0
    a[hi] = 205.0 + 50.0 * (1.0 - np.exp(-(a[hi] - 205.0) / 50.0))

    # chromatic aberration, stronger when zoomed in
    shift = int(round(clamp(st["ca"] * (0.4 + 0.45 * max(0.0, z - 1.0)),
                            0.0, 4.0)))
    if shift > 0:
        a[..., 0] = np.roll(a[..., 0], shift, axis=1)
        a[..., 2] = np.roll(a[..., 2], -shift, axis=1)

    a *= vig[..., None]

    # sensor grain
    g = _grains()[int(round(t * 30.0)) % 8]
    a += g[..., None]

    np.clip(a, 0, 255, out=a)
    return a


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------
def apply(panel, t, gain=1.0):
    """panel: float32/uint8 HxWx3 at PANEL_SCALE; returns uint8 HxWx3."""
    st = camera_at(t)
    img = _stage(_panel_finish(panel, gain))
    img = _warp(img, st, t)
    a = _defocus(img, st)
    a = _grade(a, st, t, st["z"])
    return a.astype(np.uint8)
