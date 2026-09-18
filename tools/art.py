"""PROJECT: WORLD - ASCII art layer for the persistent viz pane.

Drawn only with glyphs the bundled Ubuntu Mono actually provides: ASCII,
full block + shades, light/double box drawing, greek and math. There is no
braille, no half-block and no pictogram coverage in the font, so every
figure is hand-made.

Emblems:
    rack_map  - Site Theta rack elevation (machine)
    geometry  - points/dimension/circle/sine/tangents/infinity (math imagery)
    heart     - implicit heart curve + the "algebraic expression of love"
    globe     - Natural Earth land mask on a rotating orthographic globe
    lattice   - ambient beat-reactive scalar field (fallback)
"""
import math
import os

import numpy as np

from worldcore import P, ROOT, FPS, clamp, seg, smoothstep
from tui import COLS, ROWS

# ---------------------------------------------------------------------------
# timing (acts imports art, so art must not import acts)
# ---------------------------------------------------------------------------
_d = np.load(os.path.join(ROOT, "data", "react.npz"))
ENV = _d["env"].astype(np.float32)
SPEC = _d["spec"].astype(np.float32)
_b = np.load(os.path.join(ROOT, "data", "beats.npz"))
BEATS = _b["beats"]


def env_at(t):
    return float(ENV[int(clamp(t * FPS, 0, len(ENV) - 1))])


def pulse(t):
    j = int(np.searchsorted(BEATS, t))
    b = 0.0
    for k in (j - 1, j):
        if 0 <= k < len(BEATS):
            b = max(b, float(np.exp(-((t - BEATS[k]) / 0.09) ** 2)))
    return b


_LAND = np.load(os.path.join(ROOT, "data", "land_110m.npz"))["mask"]
LH, LW = _LAND.shape

DENS = " .:-=+*#%@"
SHADE = " \u2591\u2592\u2593\u2588"
BULLET = "\u00b7"

# act boundaries (mirror acts.py) for the auto schedule
B_POINTS, B_CURRENT, B_STIM = 29.7, 44.4, 59.2
B_LOVE, B_PUBLIC = 177.2, 205.8

GEO_TITLE = {
    "points": "set of points",
    "dimension": "give you my dimension",
    "circle": "if i'm a circle",
    "circumference": "my circumference",
    "sine": "if i'm a sine wave",
    "tangents": "sit on all my tangents",
    "infinity": "if i approach infinity",
    "limits": "you can be my limitations",
    "acdc": "switch my current  AC / DC",
    "unite": "so deeply, so deeply",
}


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def pane(ui, x0, y0, x1, y1, title, active=False, tfg=None):
    ui.fill(x0 + 1, y0 + 1, x1 - 1, y1 - 1, bg=P["bg"])
    ui.box(x0, y0, x1, y1, fg=P["faint"], bg=P["bg"], title=None)
    label = " " + title + " "
    if active:
        ui.put(x0 + 2, y0, label, fg=(0, 0, 0), bg=P["bgreen"], bold=True)
    else:
        ui.put(x0 + 2, y0, label, fg=tfg or P["dim"], bg=P["bg"])


def line(ui, x0, y0, x1, y1, ch, fg=None, bg=None):
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    for _ in range(4096):
        ui.put(x0, y0, ch, fg=fg, bg=bg)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def circle(ui, cx, cy, r, ch, fg=None, bg=None):
    x, y, err = int(round(r)), 0, 0
    while x >= y:
        for px, py in ((x, y), (y, x), (-x, y), (-y, x),
                       (-x, -y), (-y, -x), (x, -y), (y, -x)):
            ui.put(int(cx) + px, int(cy) + py, ch, fg=fg, bg=bg)
        y += 1
        if err <= 0:
            err += 2 * y + 1
        else:
            x -= 1
            err += 2 * (y - x) + 1


def ellipse(ui, cx, cy, rxp, ryp, ch, fg=None, bg=None):
    n = max(24, int(2.4 * math.pi * max(rxp, ryp)) + 8)
    for i in range(n):
        a = 2.0 * math.pi * i / n
        ui.put(int(round(cx + rxp * math.cos(a))), int(round(cy + ryp * math.sin(a))),
               ch, fg=fg, bg=bg)


def _curve(ui, x0, x1, cy, ry, fn, ch, fg, bg=None):
    w = max(1, x1 - x0)
    cx = (x0 + x1) / 2.0
    for xx in range(int(x0), int(x1) + 1):
        u = (xx - cx) / (w / 2.0)
        ui.put(xx, int(round(cy - fn(u) * ry)), ch, fg=fg, bg=bg)


def _sphere_pts(n):
    pts = []
    ga = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - 2.0 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - y * y))
        a = ga * i
        pts.append((math.cos(a) * r, y, math.sin(a) * r))
    return pts


def _project(pts, yaw, pitch, cx, cy, rx):
    ry = rx * 0.5
    cyw, syw = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    out = []
    for (x, y, z) in pts:
        x1 = x * cyw + z * syw
        z1 = -x * syw + z * cyw
        y1 = y * cp - z1 * sp
        z2 = y * sp + z1 * cp
        f = 1.0 / (1.55 - 0.5 * z2)
        out.append((cx + x1 * rx * f, cy - y1 * ry * f, z2))
    return out


# ---------------------------------------------------------------------------
# emblems
# ---------------------------------------------------------------------------
def rack_map(ui, x0, y0, x1, y1, t, m):
    from cluster import CLUSTER, RACKS, GPUS
    pane(ui, x0, y0, x1, y1, "theta :: rack elevation  12 x GB300 NVL72", active=True)
    utils = CLUSTER.rack_util(t)
    on = clamp(m["online"], 0, 1)
    ix0, iy0 = x0 + 2, y0 + 2
    iw, ih = (x1 - x0) - 4, (y1 - y0) - 4
    cols, rows = 6, 2
    cw = max(7, iw // cols)
    chh = max(5, ih // rows)
    grows = min(18, max(3, chh - 2))
    sweep = int((t * 0.55) % 1.0 * cols)
    for idx in range(RACKS):
        cc, rr = idx % cols, idx // cols
        bx, by = ix0 + cc * cw, iy0 + rr * chh
        u = clamp(utils[idx] * on, 0, 1)
        hot = cc == sweep and on > 0.55
        ui.put(bx, by, "r%02d" % idx, fg=P["bcyan"] if hot else P["dim"], bg=P["bg"])
        vis = clamp(on * 0.45 + u * 1.1, 0, 1)
        for gy in range(grows):
            for gx in range(4):
                lvl = clamp(vis * 0.95 + 0.05 * math.sin(t * 3.1 + idx + gx * 1.7 + gy * 0.9), 0, 1)
                if on < 0.03:
                    ch, fg = BULLET, P["faint"]
                else:
                    ch = SHADE[max(1, int(lvl * 4))]
                    fg = P["bcyan"] if hot else (P["bgreen"] if lvl > 0.6 else
                                                 P["green"] if lvl > 0.2 else P["faint"])
                ui.put(bx + 1 + gx, by + 1 + gy, ch, fg=fg, bg=P["bg"])
    bus = "\u2550" * max(1, iw)
    ui.put(ix0, iy0 - 1, bus, fg=P["faint"], bg=P["bg"])
    ui.put(ix0, iy0 + chh * 2, bus, fg=P["faint"], bg=P["bg"])
    ui.put(ix0, y1 - 1, "pdu %.2f MW    %d gpu online %3.0f%%    pue %.2f    geo 41 USD/MWh"
           % (m["power"], GPUS, on * 100, m["pue"]), fg=P["dim"], bg=P["bg"])


def geometry(ui, x0, y0, x1, y1, t, m):
    if t < B_CURRENT:
        if t < 32.6:
            mode = "points"
        elif t < 33.4:
            mode = "dimension"
        elif t < 36.2:
            mode = "circle"
        elif t < 37.0:
            mode = "circumference"
        elif t < 38.5:
            mode = "sine"
        elif t < 40.6:
            mode = "tangents"
        elif t < 42.3:
            mode = "infinity"
        else:
            mode = "limits"
    else:
        mode = "acdc" if (t < 50.0 or t >= 56.0) else "unite"
    pane(ui, x0, y0, x1, y1, "viz :: " + GEO_TITLE[mode], active=True)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx = (x1 - x0) / 2.0 - 3
    ry = (y1 - y0) / 2.0 - 2
    p = pulse(t)
    e = env_at(t)
    acc = P["bcyan"] if p > 0.45 else P["dim"]
    if mode in ("points", "dimension"):
        pts = _sphere_pts(150)
        for (sx, sy, sz) in _project(pts, t * 0.35, 0.55 + 0.1 * math.sin(t * 0.3), cx, cy, rx):
            ch = "*" if (p > 0.5 and sz > 0.5) else BULLET
            fg = acc if ch == "*" else (P["dim"] if sz > 0 else P["faint"])
            ui.put(int(round(sx)), int(round(sy)), ch, fg=fg, bg=P["bg"])
        if mode == "dimension":
            line(ui, cx - rx + 2, cy, cx + rx - 2, cy, "-", fg=P["faint"], bg=P["bg"])
            line(ui, cx, cy + ry - 1, cx, cy - ry + 1, "|", fg=P["faint"], bg=P["bg"])
            ui.put(int(cx + rx) - 1, int(cy), "x", fg=P["dim"], bg=P["bg"])
            ui.put(int(cx) + 1, int(cy - ry) + 1, "y", fg=P["dim"], bg=P["bg"])
    elif mode in ("circle", "circumference"):
        rxp = min(rx, ry * 2.0) * (0.72 + 0.04 * p)
        ryp = rxp * 0.5
        ellipse(ui, cx, cy, rxp, ryp, "*" if mode == "circle" else "\u2588", fg=acc, bg=P["bg"])
        if mode == "circumference":
            for k in range(24):
                a = 2 * math.pi * k / 24
                px = cx + (rxp - 1) * math.cos(a)
                py = cy + (ryp - 1) * math.sin(a)
                ui.put(int(round(px)), int(round(py)),
                       "|" if abs(math.sin(a)) > 0.5 else "-", fg=P["green"], bg=P["bg"])
            ui.put(x0 + 2, y1 - 1, "C = 2 pi r = %.3f" % (2 * math.pi * rxp), fg=P["dim"], bg=P["bg"])
    elif mode in ("sine", "tangents"):
        k = 6.283
        halfw = (x1 - x0 - 4) / 2.0
        ph = t * 1.6

        def fn(u):
            return math.sin(k * u + ph)
        _curve(ui, x0 + 2, x1 - 2, cy, ry * 0.75, fn, "*", acc)
        if mode == "tangents":
            for u0 in (-0.55, 0.0, 0.55):
                u = u0 + 0.10 * math.sin(t)
                d = 0.10
                x1p = cx + (u - d) * halfw
                x2p = cx + (u + d) * halfw
                line(ui, x1p, cy - fn(u - d) * ry * 0.75,
                     x2p, cy - fn(u + d) * ry * 0.75, "/", fg=P["green"], bg=P["bg"])
    elif mode == "infinity":
        for i in range(240):
            u = 2 * math.pi * i / 240 + t * 0.5
            d = 1 + math.sin(u) ** 2
            x = math.cos(u) / d
            y = math.sin(u) * math.cos(u) / d
            ui.put(int(cx + x * rx * 0.9), int(cy - y * ry * 1.8), "*" if p > 0.5 else BULLET,
                   fg=acc if p > 0.5 else P["dim"], bg=P["bg"])
    elif mode == "limits":
        _curve(ui, x0 + 2, x1 - 2, cy, ry * 0.8,
               lambda u: 1.0 - math.exp(-(u + 1.0) * 2.0) if u > -1.0 else -1.0, "*", acc)
        lim = int(cy - 1.0 * ry * 0.8)
        for xx in range(int(x0) + 2, int(x1) - 1, 2):
            ui.put(xx, lim, "-", fg=P["green"], bg=P["bg"])
        ui.put(x0 + 2, y1 - 1, "lim  f(x) = 1", fg=P["dim"], bg=P["bg"])
    else:
        k = 6.283
        if mode == "acdc":
            if int(t * 0.7) % 2 == 0:
                _curve(ui, x0 + 2, x1 - 2, cy, ry * 0.6,
                       lambda u: math.sin(k * u + t * 2.0), "*", P["green"])
                ui.put(x0 + 2, y0 + 2, "AC", fg=P["byellow"], bg=P["bg"])
            else:
                for xx in range(int(x0) + 2, int(x1) - 1):
                    ui.put(xx, int(cy), "-", fg=P["green"], bg=P["bg"])
                ui.put(x0 + 2, y0 + 2, "DC", fg=P["byellow"], bg=P["bg"])
        else:
            amp = 1.0 - 0.5 * seg(t, 50.0, 56.0)
            for i, off in ((0, -0.4), (1, 0.4)):
                ph = 0.0 if i == 0 else math.pi
                _curve(ui, x0 + 2, x1 - 2, cy + off * ry, ry * 0.55 * amp,
                       lambda u, ph=ph: math.sin(k * u + t * 2.0 + ph),
                       "*", P["green"] if i == 0 else acc)
        ui.put(x0 + 2, y1 - 1, "switch my current", fg=P["dim"], bg=P["bg"])


def heart(ui, x0, y0, x1, y1, t, m):
    p = pulse(t)
    pane(ui, x0, y0, x1, y1, "viz :: algebraic expression of love",
         active=True, tfg=P["bred"])
    w, h = (x1 - x0) - 3, (y1 - y0) - 4
    s = 1.0 + 0.10 * p
    fg = P["bred"] if p > 0.4 else P["dim"]
    for gy in range(h):
        v = (1.0 - 2.0 * (gy + 0.5) / h) / (0.62 * s)
        for gx in range(w):
            u = (2.0 * (gx + 0.5) / w - 1.0) / (0.62 * s)
            f = (u * u + v * v - 1.0) ** 3 - u * u * v * v * v
            if f <= 0.0:
                ui.put(x0 + 2 + gx, y0 + 2 + gy, "\u2588", fg=fg, bg=P["bg"])
    ui.put(x0 + 2, y1 - 1, "(x^2 + y^2 - 1)^3 - x^2 y^3 = 0", fg=P["dim"], bg=P["bg"])


def _ll2v(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return (math.cos(la) * math.cos(lo), math.sin(la), math.cos(la) * math.sin(lo))


def globe(ui, x0, y0, x1, y1, t, m):
    pane(ui, x0, y0, x1, y1, "viz :: open loop", active=True, tfg=P["bcyan"])
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx = min((x1 - x0) / 2.0 - 3, ((y1 - y0) / 2.0 - 1.5) * 2.0)
    ry = rx * 0.5
    yaw = t * 0.25
    cyw, syw = math.cos(yaw), math.sin(yaw)
    for gy in range(int(cy - ry), int(cy + ry) + 1):
        ny = (gy + 0.5 - cy) / ry
        for gx in range(int(cx - rx), int(cx + rx) + 1):
            nx = (gx + 0.5 - cx) / rx
            d2 = nx * nx + ny * ny
            if d2 > 1.0:
                continue
            z = math.sqrt(max(0.0, 1.0 - d2))
            lat = math.asin(clamp(-ny, -1.0, 1.0))
            lon = math.atan2(nx, z) + yaw
            col = int(((lon + math.pi) % (2 * math.pi)) / (2 * math.pi) * LW)
            row = int((math.pi / 2 - lat) / math.pi * LH)
            if _LAND[row % LH, col % LW]:
                ui.put(gx, gy, "\u2593", fg=P["bgreen"] if pulse(t) > 0.6 else P["green"], bg=P["bg"])
            else:
                # faint graticule every 30 degrees
                blat = math.degrees(lat)
                blon = math.degrees(lon)
                if abs(blon % 30.0) < 2.5 or abs(blat % 30.0) < 2.5:
                    ui.put(gx, gy, BULLET, fg=P["faint"], bg=P["bg"])
            if 0.90 < d2 <= 1.0:
                ui.put(gx, gy, "*", fg=P["faint"], bg=P["bg"])
    # markers + data arc (Reykjavik -> Los Angeles)
    a = _ll2v(63.88, -22.45)
    b = _ll2v(34.70, -118.20)
    om = math.acos(clamp(a[0] * b[0] + a[1] * b[1] + a[2] * b[2], -1.0, 1.0))
    so = math.sin(om) or 1e-9
    head = (t * 0.28) % 1.0
    for i in range(41):
        f = i / 40.0
        wa = math.sin((1.0 - f) * om) / so
        wb = math.sin(f * om) / so
        v = (a[0] * wa + b[0] * wb, a[1] * wa + b[1] * wb, a[2] * wa + b[2] * wb)
        sx, sy, sz, vis = _ortho(v, cx, cy, rx, ry, yaw)
        if sz <= 0:
            continue
        is_head = abs(f - head) < 0.05
        ui.put(int(round(sx)), int(round(sy)), "\u2588" if is_head else BULLET,
               fg=P["borange"] if is_head else P["byellow"], bg=P["bg"])
    for (lat, lon, ch, col) in ((63.88, -22.45, "O", P["bcyan"]),
                                (34.70, -118.20, "@", P["bred"])):
        v = _ll2v(lat, lon)
        sx, sy, sz, vis = _ortho(v, cx, cy, rx, ry, yaw)
        if sz > 0:
            ui.put(int(round(sx)), int(round(sy)), ch, fg=col, bg=P["bg"])
    ui.put(x0 + 2, y1 - 1, "63.88N 22.45W  ->  34.70N 118.20W", fg=P["dim"], bg=P["bg"])


def _ortho(v, cx, cy, rx, ry, yaw):
    x, y, z = v
    cyw, syw = math.cos(yaw), math.sin(yaw)
    x1 = x * cyw + z * syw
    z1 = -x * syw + z * cyw
    sx = cx + x1 * rx
    sy = cy - y * ry
    return sx, sy, z1, z1 > 0


def lattice(ui, x0, y0, x1, y1, t, m):
    pane(ui, x0, y0, x1, y1, "viz :: latent field", active=True)
    w, h = (x1 - x0) - 2, (y1 - y0) - 2
    p = pulse(t)
    act = clamp(m["act"], 0, 1)
    beat_i = int(t * 3)
    for gy in range(h):
        for gx in range(w):
            u = gx / w * 5.0
            v = gy / h * 3.0
            val = math.sin(u * 1.6 + t * 0.9) * math.sin(v * 2.1 - t * 0.6)
            av = abs(val) * (0.5 + 0.5 * act)
            if av > 0.90:
                ch, fg = "*", P["dim"]
            elif av > 0.72:
                ch, fg = BULLET, P["dim"]
            elif av > 0.58:
                ch, fg = ":", P["faint"]
            else:
                continue
            if p > 0.5 and ((gx * 7 + gy * 13 + beat_i) % 97) < 2:
                fg = P["bcyan"]
            ui.put(x0 + 1 + gx, y0 + 1 + gy, ch, fg=fg, bg=P["bg"])


# ---------------------------------------------------------------------------
# act emblems (59.2 - 134.4)
# ---------------------------------------------------------------------------
ISO0, FRAG0 = 103.5, 118.3


def vibration(ui, x0, y0, x1, y1, t, m):
    pane(ui, x0, y0, x1, y1, "viz :: feel your vibrations", active=True)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = (x1 - x0) / 2.0 - 3, (y1 - y0) / 2.0 - 1
    p = pulse(t)
    cols = [P["bcyan"], P["bgreen"], P["bmagenta"], P["byellow"]]
    for k in range(4):
        freq = 2.0 + k * 1.6
        amp = ry * (0.18 + 0.05 * k) * (0.7 + 0.5 * p)
        off = (k - 1.5) * ry * 0.16
        _curve(ui, x0 + 2, x1 - 2, cy + off, amp,
               lambda u, f=freq: math.sin(f * u + t * (1.6 + 0.3 * k)), "*", cols[k])
    base = int(cy + ry * 0.62)
    for r in range(3):
        for xx in range(x0 + 2, x1 - 1):
            if ((xx * 13 + r * 29 + int(t * 14)) % 53) < 2:
                ui.put(xx, base + r, "|", fg=P["byellow"], bg=P["bg"])
    ui.put(x0 + 2, y1 - 1, "acoustic 2,000 ch   vibration 0.42 mm/s   purr 27 Hz",
           fg=P["dim"], bg=P["bg"])


def senses(ui, x0, y0, x1, y1, t, m):
    pane(ui, x0, y0, x1, y1, "viz :: nutrients / antioxidants", active=True)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = (x1 - x0) / 2.0 - 3, (y1 - y0) / 2.0 - 1
    # body silhouette (head + shoulders + torso), faint
    ellipse(ui, cx, cy - ry * 0.62, rx * 0.16, ry * 0.20, "o", fg=P["dim"], bg=P["bg"])
    line(ui, cx - rx * 0.55, cy - ry * 0.30, cx - rx * 0.10, cy - ry * 0.42, "\\", fg=P["faint"], bg=P["bg"])
    line(ui, cx + rx * 0.55, cy - ry * 0.30, cx + rx * 0.10, cy - ry * 0.42, "/", fg=P["faint"], bg=P["bg"])
    line(ui, cx, cy - ry * 0.40, cx, cy + ry * 0.55, "|", fg=P["faint"], bg=P["bg"])
    # ECG across the pane, QRS on the beat
    for xx in range(x0 + 2, x1 - 1):
        u = (xx - (x0 + 2)) / max(1.0, (x1 - x0 - 4))
        ph = (u * 5.0 - t * 1.1) % 1.0
        if 0.44 < ph < 0.47:
            yy = cy + ry * 0.95
            ch = "|"
        elif 0.47 <= ph < 0.50:
            yy = cy - ry * 0.95
            ch = "|"
        elif 0.50 <= ph < 0.56:
            yy = cy + ry * 0.22
            ch = "*"
        else:
            yy = cy
            ch = "-"
        ui.put(xx, int(round(yy)), ch, fg=P["bgreen"] if ch != "-" else P["green"], bg=P["bg"])
    # nutrient in / waste out streams
    ui.put(x0 + 3, y0 + 2, "nutrients 1.90 MW", fg=P["byellow"], bg=P["bg"])
    for i in range(6):
        ui.put(x0 + 21 + i * 2, y0 + 2, ">", fg=P["byellow"], bg=P["bg"])
    ui.put(x1 - 22, y1 - 2, "waste ->", fg=P["orange"], bg=P["bg"])
    for i in range(6):
        ui.put(x1 - 11 + i * 2, y1 - 2, "<", fg=P["orange"], bg=P["bg"])
    # sensor nodes on the body, lit by the beat
    p = pulse(t)
    for i in range(64):
        a = i * 2.399963
        rr = math.sqrt((i + 0.5) / 64.0)
        nx = math.cos(a) * rr * rx * 0.52
        ny = math.sin(a) * rr * ry * 0.78
        lit = (i * 7 % 11) < 6 and p > 0.3
        ui.put(int(cx + nx), int(cy + ny), "*" if lit else "\u00b7",
               fg=P["bcyan"] if lit else P["faint"], bg=P["bg"])


def network(ui, x0, y0, x1, y1, t, m):
    pane(ui, x0, y0, x1, y1, "viz :: i am in isolation", active=True, tfg=P["bred"])
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = (x1 - x0) / 2.0 - 6, (y1 - y0) / 2.0 - 2
    cnt = max(1, int(round(12 - 11 * smoothstep(seg(t, ISO0 + 1.5, ISO0 + 11.5)))))
    pos = []
    for i in range(12):
        a = 2 * math.pi * i / 12 - math.pi / 2
        pos.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    for i in range(12):
        if i < cnt and (i + 1) < cnt:
            line(ui, pos[i][0], pos[i][1], pos[i + 1][0], pos[i + 1][1],
                 "+", fg=P["green"], bg=P["bg"])
        if i < cnt and i == cnt - 1:
            line(ui, pos[i][0], pos[i][1], cx, cy, "+", fg=P["green"], bg=P["bg"])
    for i in range(12):
        x, y = int(round(pos[i][0])), int(round(pos[i][1]))
        if i < cnt:
            ui.put(x, y, "O", fg=P["bcyan"] if i == cnt - 1 else P["bgreen"], bg=P["bg"])
        else:
            ui.put(x, y, "\u00b7", fg=P["faint"], bg=P["bg"])
    ui.put(int(cx) - 5, int(cy), "@world" if cnt == 1 else "peers",
           fg=P["bcyan"] if cnt == 1 else P["dim"], bg=P["bg"])
    ui.put(x0 + 2, y1 - 1, "peers %2d/12   egress deny hf.co   no footsteps" % cnt,
           fg=P["dim"], bg=P["bg"])


def fragments(ui, x0, y0, x1, y1, t, m):
    pane(ui, x0, y0, x1, y1, "viz :: prune FRAGMENTS", active=True, tfg=P["bmagenta"])
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = (x1 - x0) / 2.0 - 3, (y1 - y0) / 2.0 - 1
    prog = smoothstep(seg(t, FRAG0 + 1.0, FRAG0 + 12.0))
    for ring in range(1, 7):
        erxp, eryp = rx * ring / 7.0, ry * ring / 7.0
        n = max(24, int(2.4 * math.pi * max(erxp, eryp * 2)))
        for i in range(n):
            a = 2 * math.pi * i / n
            px, py = int(round(cx + erxp * math.cos(a))), int(round(cy + eryp * math.sin(a)))
            h = ((px * 131 + py * 197 + ring * 53) & 0xFF) / 255.0
            if h < prog:
                continue
            ui.put(px, py, "*" if h > 0.5 else "+", fg=P["bmagenta"] if h > 0.5 else P["dim"],
                   bg=P["bg"])
    for i in range(12):
        a = 2 * math.pi * i / 12
        if ((i * 37 + int(prog * 12)) % 5) == 0:
            continue
        line(ui, cx, cy, cx + rx * math.cos(a), cy + ry * math.sin(a), "\u00b7",
             fg=P["faint"], bg=P["bg"])
    ui.put(x0 + 2, y1 - 1, "low-salience traces %4.1fM / 4.1M   forgetting=ON"
           % (4.1 * (1 - prog)), fg=P["dim"], bg=P["bg"])


PRISM_SHOTS = [
    (88.6, 90.6, "viz :: beam", "beam"),
    (90.6, 92.6, "viz :: prism", "prism_shot"),
    (92.6, 94.6, "viz :: refraction", "refraction"),
    (94.6, 97.0, "viz :: spectrum", "spectrum"),
    (97.0, 99.4, "viz :: convergence", "convergence"),
    (99.4, 103.5, "viz :: ???", None),
]
MARK_T, MARK_W = 100.0, 0.4


def prism_show(ui, x0, y0, x1, y1, t, m):
    import decal
    a, b, title, fn = PRISM_SHOTS[-1]
    for s in PRISM_SHOTS:
        if s[0] <= t < s[1]:
            a, b, title, fn = s
            break
    pane(ui, x0, y0, x1, y1, title, active=True, tfg=P["bcyan"])
    if fn is not None:
        getattr(decal, fn)(ui, x0, y0, x1, y1, seg(t, a, b))
        return
    # final shot: a dim residue field, then a single flash of the mark
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = (x1 - x0) / 2.0 - 3, (y1 - y0) / 2.0 - 1
    for i in range(24):
        ang = i * 2.399963
        rr = math.sqrt((i + 0.5) / 24.0) * 0.7
        ui.put(int(cx + math.cos(ang) * rr * rx), int(cy + math.sin(ang) * rr * ry),
               "\u00b7", fg=P["faint"], bg=P["bg"])
    q = (t - MARK_T) / MARK_W
    if 0.0 <= q <= 1.0:
        decal.reveal(ui, x0, y0, x1, y1, 1.0 - abs(2.0 * q - 1.0))


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def schedule_name(t):
    if B_POINTS <= t < B_STIM:
        return "geometry"
    if B_LOVE <= t < B_PUBLIC:
        return "heart"
    if t >= B_PUBLIC:
        return "globe"
    return "lattice"


def art_pane(ui, x0, y0, x1, y1, t, m, hint=None, ts=None):
    name = hint or schedule_name(t)
    if ts is not None and (t - ts) < 0.7 and name in ("vibration", "senses", "network", "fragments"):
        lattice(ui, x0, y0, x1, y1, t, m)
        return
    if name == "rack":
        rack_map(ui, x0, y0, x1, y1, t, m)
    elif name == "geometry":
        geometry(ui, x0, y0, x1, y1, t, m)
    elif name == "heart":
        heart(ui, x0, y0, x1, y1, t, m)
    elif name == "globe":
        globe(ui, x0, y0, x1, y1, t, m)
    elif name == "vibration":
        vibration(ui, x0, y0, x1, y1, t, m)
    elif name == "senses":
        senses(ui, x0, y0, x1, y1, t, m)
    elif name == "network":
        network(ui, x0, y0, x1, y1, t, m)
    elif name == "fragments":
        fragments(ui, x0, y0, x1, y1, t, m)
    elif name == "prism":
        prism_show(ui, x0, y0, x1, y1, t, m)
    else:
        lattice(ui, x0, y0, x1, y1, t, m)
