"""PROJECT: WORLD - decal: the Moonshot / Dark Side prism.

A homage: Pink Floyd's Dark Side of the Moon prism (a white beam entering a
glass triangle, a rainbow fanning out) that resolves into the Kimi assistant
mark for a single instant.  "Moonshot" (moonshot.jpg) and "Kimi" (kimi.webp)
are the two reference decals.

Everything is drawn on the character grid with glyphs the font actually has.
"""
import math

from worldcore import P, clamp, seg, smoothstep, lerp
from art import line

# red -> violet, the DSOTM fan
BANDS = [P["bred"], P["borange"], P["byellow"], P["bgreen"], P["bcyan"], P["bblue"], P["bmagenta"]]
FULL = "\u2588"


def _geom(x0, y0, x1, y1):
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = (x1 - x0) * 0.38
    h = w * 0.5
    top = cy - h * 0.58
    return cx, cy, w, h, top, (cx, top), (cx - w / 2, top + h), (cx + w / 2, top + h)


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
            i = int((r - lo) / max(1e-6, hi - lo) * k)
            i = max(0, min(k - 1, i))
            g = dim * (0.55 + 0.45 * (1.0 - u))
            if g <= 0.05:
                continue
            ui.put(xx, r, FULL, fg=BANDS[i], bg=P["bg"])


def prism(ui, x0, y0, x1, y1, p=0.0):
    """p=0 -> full Dark Side cover; p=1 -> prism gone."""
    cx, cy, w, h, top, apex, bl, br = _geom(x0, y0, x1, y1)
    fade = 1.0 - smoothstep(seg(p, 0.10, 0.62))
    if fade <= 0.02:
        return
    edge = P["bwhite"] if fade > 0.5 else P["fg"]
    line(ui, apex[0], apex[1], bl[0], bl[1], "/", fg=edge, bg=P["bg"])
    line(ui, apex[0], apex[1], br[0], br[1], "\\", fg=edge, bg=P["bg"])
    line(ui, bl[0], bl[1], br[0], br[1], "\u2500", fg=edge, bg=P["bg"])
    fmx, fmy = (apex[0] + bl[0]) / 2.0, (apex[1] + bl[1]) / 2.0
    emx, emy = (apex[0] + br[0]) / 2.0, (apex[1] + br[1]) / 2.0
    # refracted path inside the glass
    if fade > 0.35:
        line(ui, fmx, fmy, emx, emy, "\u00b7", fg=P["faint"], bg=P["bg"])
    # incoming white beam
    line(ui, x0 + 1, fmy + (y1 - y0) * 0.20, fmx, fmy, "=", fg=P["white"], bg=P["bg"])
    # rainbow fan out of the right face
    span = (y1 - y0) * 0.50
    ytop = emy - span * 0.42
    if fade > 0.05:
        _fan(ui, emx, emy, x1 - 1, ytop, ytop + span, dim=fade)


def kimi(ui, x0, y0, x1, y1, kp=1.0):
    """The Kimi mark: a white geometric K with a blue dot. kp reveals it."""
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
        u = r / RH
        if u > grow + 0.05:
            break
        if kp > 0.02:
            for k in range(T):
                ui.put(kx + k, ky + r, FULL, fg=P["bwhite"], bg=P["bg"])
        if kp > 0.22:
            if r <= mid:
                x = lerp(RW, T, r / max(1.0, mid))
            else:
                x = lerp(T, RW, (r - mid) / max(1.0, RH - mid))
            for k in range(T):
                ui.put(int(round(kx + x)) + k, ky + r, FULL, fg=P["bwhite"], bg=P["bg"])
    if kp > 0.86:
        rr = max(1.6, RH * 0.16) * smoothstep(seg(kp, 0.86, 0.98))
        _disk(ui, kx + RW + rr * 2.0 + 1, ky + rr + 1, rr, FULL, fg=P["bblue"], bg=P["bg"])


def moonshot_kimi(ui, x0, y0, x1, y1, p):
    """Full decal at transition progress p in [0,1]."""
    prism(ui, x0, y0, x1, y1, p)
    kimi(ui, x0, y0, x1, y1, seg(p, 0.46, 1.0))
