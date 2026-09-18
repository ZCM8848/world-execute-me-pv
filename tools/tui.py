"""PROJECT: WORLD - TUI engine.

A character-cell framebuffer rendered through a precomputed glyph atlas.
Every screen is a grid of (glyph, fg, bg, bold); render() rasterises the
whole grid in a handful of vectorised numpy ops.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from worldcore import P, CW, CH, COLS, ROWS, W, H, font

# ---------------------------------------------------------------------------
# character set + glyph atlas
# ---------------------------------------------------------------------------
_ascii = "".join(chr(c) for c in range(32, 127))
_extra = (
    "".join(chr(c) for c in range(0x2500, 0x2580))   # box drawing
    + "".join(chr(c) for c in range(0x2580, 0x25A0))  # block elements
    + "".join(chr(c) for c in range(0x2800, 0x2900))  # braille
    + "".join(chr(c) for c in range(0x25A0, 0x2600))  # geometric shapes
    + "".join(chr(c) for c in range(0x2190, 0x2200))  # arrows
    + "".join(chr(c) for c in range(0x00A0, 0x0100))  # latin-1 supplement
    + "".join(chr(c) for c in range(0x0370, 0x0400))  # greek
    + "".join(chr(c) for c in range(0x2660, 0x2670))  # music / misc symbols
    + "\u2022\u2026\u2013\u2014\u2018\u2019\u201c\u201d\u2713\u2717"
    + "°±µηρλΔΣΩ√∞≈≠≤≥⋅→←↑↓█▉▊▋▌▍▎▏"
)
CHARSET = sorted(set(_ascii + _extra), key=ord)
IDX = {c: i for i, c in enumerate(CHARSET)}
QUESTION = IDX.get("?", IDX[" "])


def build_atlas(font_key="mono", size=20, cw=CW, ch=CH):
    f = font(font_key, size)
    ascent, _ = f.getmetrics()
    n = len(CHARSET)
    A = np.zeros((n, ch, cw), dtype=np.uint8)
    for i, c in enumerate(CHARSET):
        img = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(img).text((0, ascent), c, font=f, fill=255, anchor="ls")
        A[i] = np.asarray(img)
    return A


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
_BOX = {
    "light": ("┌", "┐", "└", "┘", "─", "│"),
    "heavy": ("┏", "┓", "┗", "┛", "━", "┃"),
    "double": ("╔", "╗", "╚", "╝", "═", "║"),
    "round": ("╭", "╮", "╰", "╯", "─", "│"),
}
_BLOCKS = " ▏▎▍▌▋▊▉█"
_SPARK = "▁▂▃▄▅▆▇█"


class Tui:
    def __init__(self, fg=None, bg=None):
        self.fg = np.empty((ROWS, COLS, 3), np.uint8)
        self.bg = np.empty((ROWS, COLS, 3), np.uint8)
        self.ch = np.empty((ROWS, COLS), np.uint32)
        self.bo = np.zeros((ROWS, COLS), np.uint8)
        self.clear(fg=fg, bg=bg)

    def clear(self, fg=None, bg=None):
        self.fg[:] = fg or P["fg"]
        self.bg[:] = bg or P["bg"]
        self.ch[:] = IDX[" "]
        self.bo[:] = 0

    # -- primitives ---------------------------------------------------------
    def put(self, x, y, s, fg=None, bg=None, bold=False):
        if y < 0 or y >= ROWS or not s:
            return
        fg = fg or P["fg"]
        x = int(x)
        for k, c in enumerate(s):
            xx = x + k
            if xx < 0:
                continue
            if xx >= COLS:
                break
            self.ch[y, xx] = IDX.get(c, QUESTION)
            self.fg[y, xx] = fg
            if bg is not None:
                self.bg[y, xx] = bg
            self.bo[y, xx] = 1 if bold else 0

    def fill(self, x0, y0, x1, y1, ch=" ", fg=None, bg=None, bold=False):
        x0, x1 = max(0, x0), min(COLS - 1, x1)
        y0, y1 = max(0, y0), min(ROWS - 1, y1)
        if x1 < x0 or y1 < y0:
            return
        self.ch[y0:y1 + 1, x0:x1 + 1] = IDX.get(ch, IDX[" "])
        if fg is not None:
            self.fg[y0:y1 + 1, x0:x1 + 1] = fg
        if bg is not None:
            self.bg[y0:y1 + 1, x0:x1 + 1] = bg
        self.bo[y0:y1 + 1, x0:x1 + 1] = 1 if bold else 0

    def hline(self, x0, x1, y, ch="─", fg=None, bg=None):
        self.put(x0, y, ch * (x1 - x0 + 1), fg=fg, bg=bg)

    def vline(self, x, y0, y1, ch="│", fg=None, bg=None):
        for y in range(max(0, y0), min(ROWS, y1 + 1)):
            self.put(x, y, ch, fg=fg, bg=bg)

    def box(self, x0, y0, x1, y1, fg=None, bg=None, title=None, tfg=None, style="light"):
        tl, tr, bl, br, hz, vt = _BOX[style]
        fg = fg or P["dim"]
        self.hline(x0 + 1, x1 - 1, y0, hz, fg=fg, bg=bg)
        self.hline(x0 + 1, x1 - 1, y1, hz, fg=fg, bg=bg)
        self.vline(x0, y0 + 1, y1 - 1, vt, fg=fg, bg=bg)
        self.vline(x1, y0 + 1, y1 - 1, vt, fg=fg, bg=bg)
        self.put(x0, y0, tl, fg=fg, bg=bg)
        self.put(x1, y0, tr, fg=fg, bg=bg)
        self.put(x0, y1, bl, fg=fg, bg=bg)
        self.put(x1, y1, br, fg=fg, bg=bg)
        if title:
            self.put(x0 + 2, y0, " " + title + " ", fg=tfg or P["white"],
                     bg=bg, bold=True)

    # -- composite decoration ----------------------------------------------
    def banner(self, x, y, text, rows=5, fg=None, bg=None, font_key="monob", thresh=0.44):
        """Chunky block-letter banner (figlet-ish) built from full-block cells."""
        fg = fg or P["bright"]
        target_h = rows * CH
        size = 220
        f = font(font_key, size)
        bb = f.getbbox(text)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if w <= 0 or h <= 0:
            return 0
        img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(img).text((-bb[0], -bb[1]), text, font=f, fill=255)
        nw = max(1, int(round(w * target_h / h)))
        img = img.resize((nw, target_h), Image.NEAREST)
        arr = np.asarray(img)
        cols = int(math.ceil(nw / CW))
        for r in range(rows):
            for c in range(cols):
                blk = arr[r * CH:(r + 1) * CH, c * CW:(c + 1) * CW]
                if blk.size and (blk > 110).mean() > thresh:
                    self.put(x + c, y + r, "█", fg=fg, bg=bg)
        return cols

    def bar(self, x, y, width, frac, fg=None, bg=None, empty="░", full="█"):
        frac = 0 if frac < 0 else (1.0 if frac > 1 else frac)
        n = int(round(width * frac))
        self.put(x, y, full * n, fg=fg or P["green"], bg=bg)
        if n < width:
            self.put(x + n, y, empty * (width - n), fg=P["faint"], bg=bg)

    def braille(self, x, y, values, fg=None, bg=None, vmax=None):
        """values: list of floats; packs 2 columns per cell, 4 rows tall."""
        if not values:
            return 0
        vmax = vmax or (max(values) or 1.0)
        cells = (len(values) + 1) // 2
        for c in range(cells):
            mask = 0
            for k in range(2):
                i = c * 2 + k
                if i >= len(values):
                    continue
                lvl = int(round(min(1.0, values[i] / vmax) * 4))
                for r in range(lvl):
                    mask |= (1 << (k + r * 2))
            ch = chr(0x2800 + mask)
            self.put(x + c, y, ch, fg=fg or P["bcyan"], bg=bg)
        return cells

    def spark(self, x, y, values, fg=None, bg=None, vmax=None):
        if not values:
            return 0
        vmax = vmax or (max(values) or 1.0)
        for i, v in enumerate(values):
            lvl = int(round(min(1.0, v / vmax) * 7))
            self.put(x + i, y, _SPARK[lvl], fg=fg or P["green"], bg=bg)
        return len(values)

    # -- rasterise ----------------------------------------------------------
    def render(self, atlas, atlas_bold):
        gid = self.ch
        cond = self.bo[..., None, None].astype(bool)
        m = np.where(cond, atlas_bold[gid], atlas[gid]).astype(np.float32) / 255.0
        m = m.transpose(0, 2, 1, 3).reshape(ROWS * CH, COLS * CW, 1)
        fg = np.repeat(np.repeat(self.fg, CH, axis=0), CW, axis=1).astype(np.float32)
        bg = np.repeat(np.repeat(self.bg, CH, axis=0), CW, axis=1).astype(np.float32)
        out = bg * (1.0 - m) + fg * m
        return out[:H, :W]
