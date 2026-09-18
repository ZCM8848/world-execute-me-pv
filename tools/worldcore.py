"""PROJECT: WORLD - core: canvas, palette, fonts, CRT post-processing.

Terminal-realistic TUI PV engine. Frames are drawn on a character grid
(see tui.py) and degraded by a light CRT pass here.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageFilter, ImageFont

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# canvas
# ---------------------------------------------------------------------------
W, H = 1920, 1080
FPS = 30
SR = 44100

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, "assets", "fonts")
AUDIO = os.path.join(ROOT, "data", "audio.wav")
ASSETS = os.path.join(ROOT, "assets")

# terminal cell metrics (Ubuntu Mono @ 20 -> exactly 10px advance)
CW, CH = 10, 20
COLS, ROWS = W // CW, H // CH

FONT_FILES = {
    "mono": "UbuntuMono-Regular.ttf",
    "monob": "UbuntuMono-Bold.ttf",
    "jb": "JetBrainsMono-Regular.ttf",
    "jbb": "JetBrainsMono-Bold.ttf",
}
_font_cache = {}


def font(key, size):
    k = (key, size)
    if k not in _font_cache:
        _font_cache[k] = ImageFont.truetype(os.path.join(FONT_DIR, FONT_FILES[key]), size)
    return _font_cache[k]


# ---------------------------------------------------------------------------
# palette - Ubuntu / Tango terminal
# ---------------------------------------------------------------------------
def hx(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


P = {
    "bg": hx("080808"),
    "panel": hx("0E0E0E"),
    "panel2": hx("141414"),
    "fg": hx("C6C6C6"),
    "bright": hx("FFFFFF"),
    "dim": hx("6E6E6E"),
    "faint": hx("383838"),
    "black": hx("2E3436"),
    "red": hx("CC0000"),
    "green": hx("4E9A06"),
    "yellow": hx("C4A000"),
    "blue": hx("3465A4"),
    "magenta": hx("75507B"),
    "cyan": hx("06989A"),
    "white": hx("D3D7CF"),
    "bred": hx("EF2929"),
    "bgreen": hx("8AE234"),
    "byellow": hx("FCE94F"),
    "bblue": hx("729FCF"),
    "bmagenta": hx("AD7FA8"),
    "bcyan": hx("34E2E2"),
    "bwhite": hx("EEEEEE"),
    "orange": hx("CE5C00"),
    "borange": hx("FCAF3E"),
}


# ---------------------------------------------------------------------------
# easing / timing
# ---------------------------------------------------------------------------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def lerp(a, b, x):
    return a + (b - a) * x


def smoothstep(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def seg(t, a, b):
    if b <= a:
        return 0.0
    return clamp((t - a) / (b - a))


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


# hook sections where the picture is allowed to tear / flash
CHORUS = [
    (59.2, 74.0),      # STIMULATIONS / SATISFACTION
    (103.5, 118.3),    # VIBRATIONS / COMPLETION / ISOLATION
    (147.66, 162.63),  # EXECUTION x12 -> freeze
    (162.63, 173.35),  # give them all the EXECUTION
    (177.2, 193.46),   # LO-O-OVE
]


def chorus(t):
    v = 0.0
    for a, b in CHORUS:
        v = max(v, smoothstep(seg(t, a, a + 0.25)) * (1 - smoothstep(seg(t, b - 0.25, b))))
    return v


# ---------------------------------------------------------------------------
# post-processing (CRT)
# ---------------------------------------------------------------------------
class Post:
    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        yy = np.arange(H)[:, None]
        xx = np.arange(W)[None, :]
        scan = 1.0 - 0.12 * (yy % 2 == 0)
        self.scan = scan.astype(np.float32)
        cxx = (xx - W / 2) / (W / 2)
        cyy = (yy - H / 2) / (H / 2)
        r = np.sqrt(cxx ** 2 + cyy ** 2)
        self.vig = np.clip(1.10 - 0.36 * r ** 2.2, 0.34, 1.0).astype(np.float32)
        self.grains = [(rng.standard_normal((H, W)).astype(np.float32) * 0.5) for _ in range(8)]
        self._gi = 0

    def compose(self, content, t=0.0, gain=1.0, flash=0.0, glitch=0.0, bloom=0.5):
        c = np.asarray(content, dtype=np.float32)
        ci = Image.fromarray(np.clip(c, 0, 255).astype(np.uint8))
        # bloom on a quarter-res copy (low frequency -> indistinguishable, ~10x faster)
        sw, sh = max(1, W // 4), max(1, H // 4)
        g = ci.resize((sw, sh), Image.BILINEAR).filter(ImageFilter.GaussianBlur(2.4))
        g = np.asarray(g.resize((W, H), Image.BILINEAR), dtype=np.float32)
        a = np.clip(c + g * bloom, 0, 255) * gain
        a *= self.scan[..., None]
        a *= self.vig[..., None]
        grain = self.grains[self._gi]
        self._gi = (self._gi + 1) % len(self.grains)
        a += grain[..., None] * 4.5
        return np.clip(a, 0, 255).astype(np.uint8)
