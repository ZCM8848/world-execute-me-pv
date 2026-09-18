"""PROJECT: WORLD - decal: the prism storyboard.

A six-shot sequence drawn on the character grid (a white beam entering a
glass triangle, a rainbow fanning out, collapsing to a point, and a single
final mark).  Everything uses glyphs the bundled font actually has.
"""
import math

from worldcore import P, clamp, seg, smoothstep, lerp
from art import line

# red -> violet
BANDS = [P["bred"], P["borange"], P["byellow"], P["bgreen"], P["bcyan"], P["bblue"], P["bmagenta"]]
FULL = "\u2588"


def _geom(x0, y0, x1, y1):
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = (x1 - x0) * 0.38
    h = w * 0.5
    top = cy - h * 0.58
    return cx, cy, w, h, top, (cx, top), (cx - w / 2, top + h), (cx + w / 2, top + h)


def _triangle(ui, apex, bl, br, edge):
    line(ui, apex[0], apex[1], bl[0], bl[1], "/", fg=edge, bg=P["bg"])
    line(ui, apex[0], apex[1], br[0], br[1], "\\", fg=edge, bg=P["bg"])
    line(ui, bl[0], bl[1], br[0], br[1], "\u2500", fg=edge, bg=P["bg"])


def _disk(ui, cx, cy, rr, ch, fg, bg=None):
    rxp, ryp = rr * 2.0, rr
    for yy in range(int(cy - ryp), int(cy + ryp) + 1):
        for xx in range(int(cx - rxp), int(cx + rxp) + 1):
            nx = (xx - cx) / rxp
            ny = (yy - cy) / ryp
            if nx * nx + ny * ny <= 1.0:
                ui.put(xx, yy, ch, fg=fg, bg=bg)


def _fan(ui, ex, ey, x1, ytop, ybot, dim=1.0):
    n = max(1, int(x1) - int(ex))
    k = len(BANDS)
    for xx in range(int(ex), int(x1)):
        u = (xx - ex) / n
        yc = lerp(ey, (ytop + ybot) / 2.0, u)
        hh = lerp(0.0, (ybot - ytop) / 2.0, u)
        lo, hi = yc - hh, yc + hh
        for r in range(int(math.floor(lo)), int(math.ceil(hi))):
            i = max(0, min(k - 1, int((r - lo) / max(1e-6, hi - lo) * k)))
            if dim * (0.55 + 0.45 * (1.0 - u)) <= 0.05:
                continue
            ui.put(xx, r, FULL, fg=BANDS[i], bg=P["bg"])


def beam(ui, x0, y0, x1, y1, prog):
    cx, cy, w, h, top, apex, bl, br = _geom(x0, y0, x1, y1)
    fmx, fmy = (apex[0] + bl[0]) / 2.0, (apex[1] + bl[1]) / 2.0
    sy = fmy + (y1 - y0) * 0.20
    p = smoothstep(prog)
    mx, my = lerp(x0 + 1, fmx, p), lerp(sy, fmy, p)
    line(ui, x0 + 1, sy, mx, my, "=", fg=P["bwhite"], bg=P["bg"])
    if prog > 0.55:
        _triangle(ui, apex, bl, br, P["fg"])


def prism_shot(ui, x0, y0, x1, y1, prog):
    cx, cy, w, h, top, apex, bl, br = _geom(x0, y0, x1, y1)
    fmx, fmy = (apex[0] + bl[0]) / 2.0, (apex[1] + bl[1]) / 2.0
    emx, emy = (apex[0] + br[0]) / 2.0, (apex[1] + br[1]) / 2.0
    _triangle(ui, apex, bl, br, P["bwhite"])
    line(ui, x0 + 1, fmy + (y1 - y0) * 0.20, fmx, fmy, "=", fg=P["white"], bg=P["bg"])
    line(ui, fmx, fmy, emx, emy, "\u00b7", fg=P["faint"], bg=P["bg"])
    if prog > 0.72:
        _fan(ui, emx, emy, x1 - 1, emy - 1, emy + 1, dim=(prog - 0.72) / 0.28)


def refraction(ui, x0, y0, x1, y1, prog):
    cx, cy, w, h, top, apex, bl, br = _geom(x0, y0, x1, y1)
    fmx, fmy = (apex[0] + bl[0]) / 2.0, (apex[1] + bl[1]) / 2.0
    emx, emy = (apex[0] + br[0]) / 2.0, (apex[1] + br[1]) / 2.0
    _triangle(ui, apex, bl, br, P["bwhite"])
    line(ui, x0 + 1, fmy + (y1 - y0) * 0.20, fmx, fmy, "=", fg=P["white"], bg=P["bg"])
    line(ui, fmx, fmy, emx, emy, "\u00b7", fg=P["faint"], bg=P["bg"])
    span = (y1 - y0) * 0.50 * smoothstep(prog)
    ytop = emy - span * 0.42
    if span > 0.6:
        _fan(ui, emx, emy, x1 - 1, ytop, ytop + span, dim=1.0)


def spectrum(ui, x0, y0, x1, y1, prog):
    cx, cy, w, h, top, apex, bl, br = _geom(x0, y0, x1, y1)
    fmx, fmy = (apex[0] + bl[0]) / 2.0, (apex[1] + bl[1]) / 2.0
    emx, emy = (apex[0] + br[0]) / 2.0, (apex[1] + br[1]) / 2.0
    _triangle(ui, apex, bl, br, P["fg"])
    line(ui, x0 + 1, fmy + (y1 - y0) * 0.20, fmx, fmy, "=", fg=P["white"], bg=P["bg"])
    span = (y1 - y0) * 0.50
    ytop = emy - span * 0.42
    _fan(ui, emx, emy, x1 - 1, ytop, ytop + span, dim=1.0)
    scan = emx + (x1 - 1 - emx) * (0.5 + 0.5 * math.sin(prog * math.pi * 2.0))
    for xx in range(int(scan) - 1, int(scan) + 2):
        for r in range(int(ytop) - 1, int(ytop + span) + 1):
            ui.put(xx, r, "\u2502", fg=P["bwhite"], bg=P["bg"])


def convergence(ui, x0, y0, x1, y1, prog):
    cx, cy, w, h, top, apex, bl, br = _geom(x0, y0, x1, y1)
    fmx, fmy = (apex[0] + bl[0]) / 2.0, (apex[1] + bl[1]) / 2.0
    emx, emy = (apex[0] + br[0]) / 2.0, (apex[1] + br[1]) / 2.0
    p = clamp(prog)
    _triangle(ui, apex, bl, br, P["fg"] if p < 0.7 else P["faint"])
    line(ui, x0 + 1, fmy + (y1 - y0) * 0.20, lerp(fmx, x0 + 1, p), lerp(fmy, fmy + (y1 - y0) * 0.20, p),
         "=", fg=P["white"] if p < 0.8 else P["faint"], bg=P["bg"])
    span = (y1 - y0) * 0.50 * (1.0 - p)
    ytop = emy - span * 0.42
    if span > 0.6:
        _fan(ui, emx, emy, x1 - 1, ytop, ytop + span, dim=1.0 - p)
    if p > 0.7:
        ui.put(int(round(emx)), int(round(emy)), "\u2588",
               fg=P["bwhite"] if p > 0.9 else P["white"], bg=P["bg"])


def mark(ui, x0, y0, x1, y1, kp=1.0):
    """The final mark: a white geometric K with a blue dot. kp reveals it."""
    if kp <= 0.001:
        return
    RH = int(min((y1 - y0) * 0.80, 16))
    RW = int(RH * 1.30)
    T = max(2, RH // 4)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    kx = int(round(cx - RW / 2.0))
    ky = int(round(cy - RH / 2.0))
    mid = RH * 0.52
    grow = smoothstep(seg(kp, 0.0, 0.62))
    for r in range(RH + 1):
        if r / RH > grow + 0.05:
            break
        if kp > 0.02:
            for k in range(T):
                ui.put(kx + k, ky + r, FULL, fg=P["bwhite"], bg=P["bg"])
        if kp > 0.22:
            x = lerp(RW, T, r / max(1.0, mid)) if r <= mid else lerp(T, RW, (r - mid) / max(1.0, RH - mid))
            for k in range(T):
                ui.put(int(round(kx + x)) + k, ky + r, FULL, fg=P["bwhite"], bg=P["bg"])
    if kp > 0.86:
        rr = max(1.6, RH * 0.16) * smoothstep(seg(kp, 0.86, 0.98))
        _disk(ui, kx + RW + rr * 2.0 + 1, ky + rr + 1, rr, FULL, fg=P["bblue"], bg=P["bg"])


def reveal(ui, x0, y0, x1, y1, p):
    mark(ui, x0, y0, x1, y1, seg(p, 0.46, 1.0))
