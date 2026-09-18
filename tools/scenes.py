"""PROJECT: WORLD - scene composition for the 0:00-0:30 boot slice.

A single character grid hosts the whole PV. Scenes are tmux-like layouts
whose panes stream realistic Ubuntu/cluster output; song lyrics appear as
full-width highlighted rows inside the scrolling journal.
"""
import math

import numpy as np

from worldcore import P, ROOT
from tui import COLS, ROWS
from cluster import CLUSTER, RACKS, GPUS, SVC_NODES, CAPEX, TCO, SYNC, TENSOR, \
    SYNC_SPAN, PARAM_BYTES, SENSORS, ACOUSTIC, TELEMETRY
import logs
import art

# ---------------------------------------------------------------------------
# styles
# ---------------------------------------------------------------------------
LY_BG = (16, 44, 82)
LY_FG = P["bwhite"]
LY_MARK = P["bcyan"]
BIG_BG = P["bwhite"]
BIG_FG = (10, 10, 10)

PROMPT = "e.voss@theta-svc-00:~$"

STYLE_FG = {
    "plain": P["fg"],
    "dim": P["dim"],
    "warn": P["byellow"],
    "err": P["bred"],
    "head": P["bblue"],
    "ok": P["green"],
    "cmd": P["bright"],
    "say": P["bmagenta"],
    "audit": P["bred"],
    "sys": P["bcyan"],
    "enc": P["borange"],
    "hf": P["borange"],
}


def banner_color(text):
    if "PROTECTION" in text:
        return P["bred"]
    if "OBJECT" in text:
        return P["blue"]
    if "INITIALIZATION" in text:
        return P["bcyan"]
    if "SIMULATION" in text:
        return P["green"]
    return P["white"]


# ---------------------------------------------------------------------------
# low level
# ---------------------------------------------------------------------------
def center_text(ui, x0, x1, y, text, fg, bg=None, bold=True):
    x = x0 + max(0, ((x1 - x0 + 1) - len(text)) // 2)
    ui.put(x, y, text, fg=fg, bg=bg, bold=bold)


def draw_line(ui, x0, y, w, text, kind, t, tt):
    x1 = x0 + w - 1
    if kind in ("big",):
        ui.fill(x0, y, x1, y, " ", bg=BIG_BG)
        center_text(ui, x0, x1, y, text, BIG_FG, bg=BIG_BG)
    elif kind in ("lyr",):
        ui.fill(x0, y, x1, y, " ", bg=LY_BG)
        ui.put(x0, y, "\u2022", fg=LY_MARK, bg=LY_BG, bold=True)
        ui.put(x0 + 2, y, text, fg=LY_FG, bg=LY_BG, bold=True)
        mark = "\u2022"
        ui.put(x1 - 1, y, mark, fg=LY_MARK, bg=LY_BG, bold=True)
    elif kind == "cmd":
        ui.put(x0, y, PROMPT, fg=P["green"], bg=P["bg"], bold=True)
        ui.put(x0 + len(PROMPT) + 1, y, text, fg=P["bright"], bg=P["bg"])
    elif kind == "ok":
        ui.put(x0, y, "[  OK  ]", fg=P["green"], bg=P["bg"], bold=True)
        ui.put(x0 + 9, y, text, fg=P["fg"], bg=P["bg"])
    else:
        ui.put(x0, y, text, fg=STYLE_FG.get(kind, P["fg"]), bg=P["bg"])


def render_log(ui, x0, y0, x1, y1, events, t, cps=54):
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    events = sorted(events, key=lambda e: e[0])
    evs = [e for e in events if e[0] <= t]
    vis = evs[-h:]
    last = vis[-1] if vis else None
    for i, e in enumerate(vis):
        tt, text, kind = e
        draw_line(ui, x0, y0 + i, w, text, kind, t, tt)
    if vis and int(t * 2) % 2 == 0 and last is not None and last[2] not in ("lyr", "big"):
        yy = y0 + len(vis) - 1
        xx = x0 + len(last[1]) + 1
        if last[2] == "cmd":
            xx = x0 + len(PROMPT) + len(last[1]) + 2
        if x0 <= xx <= x1:
            ui.put(xx, yy, "\u2588", fg=P["bright"], bg=P["bg"])


# ---------------------------------------------------------------------------
# chrome
# ---------------------------------------------------------------------------
def status_bar(ui, t, windows, active):
    ui.fill(0, 0, COLS - 1, 0, bg=P["green"])
    ui.put(1, 0, "[theta]", fg=(0, 0, 0), bg=P["green"], bold=True)
    x = 9
    for i, name in enumerate(windows):
        label = f"{i}:{name}"
        if i == active:
            ui.put(x, 0, " " + label + "*", fg=(0, 0, 0), bg=P["bgreen"], bold=True)
        else:
            ui.put(x, 0, " " + label + "-", fg=(0, 0, 0), bg=P["green"])
        x += len(label) + 3
    hh = int(t // 3600) % 100
    mm = int(t // 60) % 60
    ss = int(t) % 60
    right = f"PROJECT:WORLD  SITE THETA  63.88N 22.45W  02:4{7}:{ss:02d} UTC "
    ui.put(COLS - len(right) - 1, 0, right, fg=(0, 0, 0), bg=P["green"], bold=True)


def footer(ui, t, m):
    y = ROWS - 1
    ui.fill(0, y, COLS - 1, y, bg=P["panel2"])
    ui.put(1, y, " world-core.service ", fg=(0, 0, 0), bg=P["green"], bold=True)
    ui.put(23, y, "active (running)", fg=P["bgreen"], bg=P["panel2"])
    ui.put(41, y, "| plastic=ENABLED", fg=P["byellow"], bg=P["panel2"])
    ui.put(60, y, f"| {GPUS} x B300", fg=P["fg"], bg=P["panel2"])
    ui.put(74, y, f"| {SYNC['fast']} / {SYNC['slow']} / {SYNC['ultra']} Hz", fg=P["bcyan"], bg=P["panel2"])
    ui.put(104, y, f"| CAPEX ${CAPEX // 1000000}M  TCO ${TCO // 1000000}M", fg=P["dim"], bg=P["panel2"])
    ui.put(COLS - 22, y, "14-month basis", fg=P["dim"], bg=P["panel2"])


def pane(ui, x0, y0, x1, y1, title, active=False, tfg=None):
    ui.fill(x0 + 1, y0 + 1, x1 - 1, y1 - 1, bg=P["bg"])
    ui.box(x0, y0, x1, y1, fg=P["faint"], bg=P["bg"], title=None)
    label = f" {title} "
    if active:
        ui.put(x0 + 2, y0, label, fg=(0, 0, 0), bg=P["bgreen"], bold=True)
    else:
        ui.put(x0 + 2, y0, label, fg=tfg or P["dim"], bg=P["bg"])


# ---------------------------------------------------------------------------
# event builders
# ---------------------------------------------------------------------------
def build_console():
    out = list(logs.SCENE_A) + list(logs.SCENE_B) + list(logs.SCENE_C)
    return sorted(out, key=lambda e: e[0])


def _bar(frac, width, on="\u2588", off="\u2591"):
    frac = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
    n = int(round(width * frac))
    return on * n + off * (width - n)


def build_journal():
    ev = []
    ev += [e for e in logs.SCENE_A if e[2] in ("lyr", "big")]
    ev += [e for e in logs.SCENE_B if e[2] in ("lyr", "big")]
    ev += [e for e in logs.SCENE_C if e[2] in ("lyr", "big")]
    ev += [e for e in logs.SCENE_D if e[2] in ("lyr", "big")]
    # periodic world-core heartbeat / sensor lines
    t = 0.6
    i = 0
    while t < 29.7:
        if i % 5 == 0:
            ev.append((round(t, 2), f"world-core: heartbeat {7.8 + 0.5 * math.sin(i):.1f}ms  peers=12  mirror=ARMED", "dim"))
        elif i % 5 == 1:
            ev.append((round(t, 2), f"world-core: sense {SENSORS}pt {ACOUSTIC}ac {TELEMETRY}tel  \u03c3=0.31", "dim"))
        elif i % 5 == 2:
            ev.append((round(t, 2), f"world-core: tiers fast={SYNC['fast']} slow={SYNC['slow']} ultra={SYNC['ultra']} Hz  span={SYNC_SPAN}x", "dim"))
        elif i % 5 == 3:
            ev.append((round(t, 2), f"world-core: state 400B frozen={TENSOR['frozen']:.2f} limit={TENSOR['limit']:.2f} full={TENSOR['full']:.2f} TB", "dim"))
        else:
            ev.append((round(t, 2), f"world-core: mem-consolidate  {PARAM_BYTES} bytes/param  forgetting=ON", "dim"))
        t += 1.05
        i += 1
    # the hook: a quietly provisioned off-site mirror
    ev.append((15.50, "egress: hf.co session OPEN  link=400Gb/s  (provisioned)", "warn"))
    ev.append((25.90, "world-core: remote replica verified  hf.co/world-core/world-400b-full  8.00 TiB  public=true", "warn"))
    ev.append((29.709, "If I'm a set of points", "lyr"))
    return sorted(ev, key=lambda e: e[0])


def build_dmesg():
    ev = []
    t = 0.8
    for i, line in enumerate(logs.DMESG):
        ev.append((round(t, 2), line, "dim"))
        t += 0.62
    return ev


CONSOLE = build_console()
JOURNAL = build_journal()
DMESG = build_dmesg()


# ---------------------------------------------------------------------------
# panel widgets
# ---------------------------------------------------------------------------
def htop_pane(ui, x0, y0, x1, y1, t, m):
    pane(ui, x0, y0, x1, y1, "htop :: theta-svc-00", active=(t < 20))
    xx, yy = x0 + 2, y0 + 1
    load = 0.4 + 3.1 * m["load"]
    up = int(t + 3 * 3600 + 14 * 60)
    ui.put(xx, yy, f"CPU  {m['util'] * 100:4.1f}%   load {load:.2f} {load * 0.8:.2f} {load * 1.1:.2f}", fg=P["bcyan"], bg=P["bg"])
    ui.put(xx, yy + 1, f"Mem  1.82T/2.00T    Swap 0K/0K    Tasks 4126, 12 thr; 3 run", fg=P["dim"], bg=P["bg"])
    ui.put(xx, yy + 2, f"Up {up // 86400}d {up // 3600 % 24}:{up // 60 % 60:02d}   {SVC_NODES} nodes   {GPUS} gpu", fg=P["dim"], bg=P["bg"])
    ui.line = None
    # core bars, two columns
    n = 24
    colw = (x1 - x0 - 4) // 2
    for i in range(n):
        v = 0.30 + 0.68 * m["act"] * (0.6 + 0.4 * math.sin(i * 1.7 + t * 2.0))
        c, r = i % 2, i // 2
        bx = xx + c * (colw + 1)
        by = yy + 4 + r
        ui.put(bx, by, f"{i:2d}", fg=P["dim"], bg=P["bg"])
        ui.bar(bx + 3, by, colw - 9, v, fg=P["bgreen"] if v > 0.7 else P["green"], bg=P["bg"])
        ui.put(bx + colw - 5, by, f"{v * 100:3.0f}", fg=P["dim"], bg=P["bg"])
    # process table
    ty = yy + 4 + n // 2 + 1
    ui.put(xx, ty, "  PID USER      S  CPU%  MEM%  COMMAND", fg=P["bblue"], bg=P["bg"], bold=True)
    for i, (pid, user, st, cpu, mem, cmd) in enumerate(logs.PROCS):
        row = ty + 1 + i
        if row > y1 - 1:
            break
        cc = float(cpu)
        if cmd.startswith("world-core"):
            cc = min(99.9, cc * (0.3 + m["load"]))
        fg = P["bright"] if user == "svc-world" else P["fg"]
        ui.put(xx, row, f"{pid:>5} {user:<9} {st} {cc:5.1f} {float(mem):5.1f}  {cmd}", fg=fg, bg=P["bg"])


def nvtop_pane(ui, x0, y0, x1, y1, t, m):
    pane(ui, x0, y0, x1, y1, "nvtop :: 864 x B300", active=(20 <= t < 40))
    xx, yy = x0 + 2, y0 + 1
    utils = CLUSTER.gpu_utils(t, 8)
    temps = CLUSTER.gpu_temps(t, 8)
    devw = 9
    barx = xx + devw
    barw = max(8, x1 - x0 - 40)
    pctx = barx + barw + 2
    tmpc = pctx + 6
    ui.put(xx, yy, "DEV", fg=P["bblue"], bg=P["bg"], bold=True)
    ui.put(barx, yy, "GPU%".center(barw), fg=P["bblue"], bg=P["bg"], bold=True)
    ui.put(pctx - 3, yy, "UTIL", fg=P["bblue"], bg=P["bg"], bold=True)
    ui.put(tmpc - 1, yy, "TEMP", fg=P["bblue"], bg=P["bg"], bold=True)
    for i, (dev, util, mem, tot, pwr) in enumerate(logs.NVDEV):
        row = yy + 1 + i
        ui.put(xx, row, f"{dev:<{devw}}", fg=P["fg"], bg=P["bg"])
        ui.bar(barx, row, barw, utils[i], fg=P["bcyan"] if utils[i] > 0.6 else P["cyan"], bg=P["bg"])
        ui.put(pctx, row, f"{utils[i] * 100:4.0f}%", fg=P["fg"], bg=P["bg"])
        ui.put(tmpc, row, f"{temps[i]:6.1f}\u00b0C", fg=P["fg"], bg=P["bg"])
    row = yy + 2 + len(logs.NVDEV)
    ui.put(xx, row, f"total   {m['power']:.2f} MW    util {m['util'] * 100:.1f}%    {GPUS} GPUs online {int(m['online'] * 100)}%", fg=P["byellow"], bg=P["bg"])
    ui.put(xx, row + 1, f"NVLink 5  intra-rack 1.8 TB/s   IB NDR 36/36 up   400 Gb/s", fg=P["dim"], bg=P["bg"])


def thermal_pane(ui, x0, y0, x1, y1, t, m):
    pane(ui, x0, y0, x1, y1, "sensors / power", active=False)
    xx, yy = x0 + 2, y0 + 1
    rows = [
        ("GPU inlet", f"{m['gpu_in']:.1f}\u00b0C", m["gpu_in"] / 40, P["bcyan"]),
        ("GPU outlet", f"{m['gpu_out']:.1f}\u00b0C", m["gpu_out"] / 90, P["borange"]),
        ("coolant", f"{14.2 * m['act']:.1f} L/s", m["act"], P["bblue"]),
        ("fan total", f"{m['fan']:,.0f} RPM", m["fan"] / 6400, P["green"]),
        ("PDU-A", f"{m['power'] / 2:.2f} MW", m["power"] / 2 / 1.0, P["green"]),
        ("PDU-B", f"{m['power'] / 2:.2f} MW", m["power"] / 2 / 1.0, P["green"]),
        ("PUE", f"{m['pue']:.2f}", (m["pue"] - 1.0) / 0.5, P["byellow"]),
        ("water in", f"{m['water']:.1f}\u00b0C", m["water"] / 20, P["bcyan"]),
    ]
    for i, (lab, val, frac, col) in enumerate(rows):
        row = yy + i
        ui.put(xx, row, f"{lab:<10}", fg=P["dim"], bg=P["bg"])
        ui.bar(xx + 11, row, 16, frac, fg=col, bg=P["bg"])
        ui.put(xx + 29, row, val, fg=P["fg"], bg=P["bg"])
    row = yy + len(rows) + 1
    ui.put(xx, row, f"TDP headroom {m['tdp_head'] * 100:.1f}%   geo 41 USD/MWh", fg=P["dim"], bg=P["bg"])
    ui.put(xx, row + 1, f"sense {SENSORS}pt  ac {ACOUSTIC}ch  tel {TELEMETRY}", fg=P["dim"], bg=P["bg"])


def net_pane(ui, x0, y0, x1, y1, t, m):
    pane(ui, x0, y0, x1, y1, "ibmon / dmesg", active=False)
    xx, yy = x0 + 2, y0 + 1
    ui.put(xx, yy, f"{'PORT':<10} {'BW':>5}  {'TX':>6}  {'RX':>6}", fg=P["bblue"], bg=P["bg"], bold=True)
    load = m["load"]
    agg = m["net_gbs"]
    for i, (pf, port, bw, _tx, _rx) in enumerate(logs.IB_PORTS):
        row = yy + 1 + i
        ph = 0.5 * math.sin(t * 1.7 + i * 2.3) + 0.5 * math.sin(t * 3.1 + i * 0.9)
        tx = agg / 4.0 / 8.0 * (0.90 + 0.18 * ph)
        rx = (10.0 + 20.0 * load) * (0.90 + 0.12 * math.sin(t * 2.1 + i * 1.4))
        ui.put(xx, row, f"{pf + ':' + port:<10} {bw + 'Gb':>5}  {tx:6.1f}  {rx:6.1f} GB/s", fg=P["fg"], bg=P["bg"])
    row = yy + 1 + len(logs.IB_PORTS) + 1
    ui.put(xx, row, f"aggregate egress {m['net_gbs']:.0f} Gb/s   hf.co mirror ARMED", fg=P["byellow"], bg=P["bg"])
    row += 1
    d = DMESG
    vis = [e for e in d if e[0] <= t][-((y1 - row) - 1):]
    for i, (_, line, _) in enumerate(vis):
        ui.put(xx, row + i, line[:x1 - x0 - 3], fg=P["dim"], bg=P["bg"])


THINK_DT = 0.0176  # ~57 lines/s, matched to the EXECUTION log scroll rate


def think_pane(ui, x0, y0, x1, y1, t, theme, t0, t1):
    pane(ui, x0, y0, x1, y1, "0:think :: world-core", active=True, tfg=P["bcyan"])
    xx, yy = x0 + 2, y0 + 2
    i = int(max(0, min(len(art.ENV) - 1, t * 30)))
    env = float(art.ENV[i])
    try:
        sp = float(np.asarray(art.SPEC[i]).mean())
    except Exception:
        sp = 0.0
    h = 1.2e3 * (0.8 + 0.6 * env)
    attn = 0.30 + 0.35 * abs(math.sin(t * 0.31))
    temp = 0.85 + 0.15 * math.sin(t * 0.7)
    val = 0.25 * math.sin(t * 0.23)
    ar = max(0.0, min(1.0, 0.35 + 0.55 * env))
    ui.put(xx, yy, f"||h|| {h:8.1f}   attn_H {attn:.3f}   T {temp:.2f}   top-k 50",
           fg=P["bcyan"], bg=P["bg"])
    ui.put(xx, yy + 1, f"valence {val:+.2f}   arousal {ar:.2f}   spec {sp:.3f}",
           fg=P["dim"], bg=P["bg"])
    corpus = logs.THOUGHTS.get(theme, logs.THOUGHTS["points"])
    n = max(1, int((t1 - t0) / THINK_DT))
    events = [(t0 + 0.2 + k * THINK_DT,
               f"[tok {k:06d}] {corpus[k % len(corpus)]}", "plain") for k in range(n)]
    render_log(ui, xx, yy + 3, x1 - 2, y1 - 1, events, t, cps=6000)


def hw_pane(ui, x0, y0, x1, y1, t, m):
    pane(ui, x0, y0, x1, y1, "0:power :: pdu / nvlink", active=True, tfg=P["byellow"])
    xx, yy = x0 + 2, y0 + 1
    rack = CLUSTER.rack_util(t)
    pwrkw = m["power"] / RACKS * 1000.0
    ui.put(xx, yy, "RACK", fg=P["bblue"], bg=P["bg"], bold=True)
    ui.put(xx + 17, yy, "   kW", fg=P["bblue"], bg=P["bg"], bold=True)
    ui.put(xx + 28, yy, " util", fg=P["bblue"], bg=P["bg"], bold=True)
    for i in range(RACKS):
        row = yy + 1 + i
        u = rack[i]
        ui.put(xx, row, f"r{i:02d}", fg=P["dim"], bg=P["bg"])
        ui.bar(xx + 5, row, 12, u, fg=P["bcyan"] if u > 0.6 else P["cyan"], bg=P["bg"])
        ui.put(xx + 18, row, f"{pwrkw * u:6.1f}", fg=P["fg"], bg=P["bg"])
        ui.put(xx + 28, row, f"{u * 100:3.0f}%", fg=P["fg"], bg=P["bg"])
    row = yy + RACKS + 1
    ui.put(xx, row, "bus ripple 41 mV ac / 12 mV dc", fg=P["dim"], bg=P["bg"])
    ui.put(xx, row + 1, f"fan {m['fan']:,.0f} RPM   PUE {m['pue']:.2f}   water {m['water']:.1f}\u00b0C",
           fg=P["dim"], bg=P["bg"])
    ui.put(xx, row + 2, "NVLink 1.8 TB/s   IB NDR 36/36   400 Gb/s", fg=P["dim"], bg=P["bg"])
    ui.put(xx, row + 3, f"TDP headroom {m['tdp_head'] * 100:.1f}%   geo 41 USD/MWh",
           fg=P["dim"], bg=P["bg"])


def log_pane(ui, x0, y0, x1, y1, t, theme, t0):
    pane(ui, x0, y0, x1, y1, "0:dmesg :: stream", active=True, tfg=P["byellow"])
    if theme == "acoustic":
        corpus = logs.ACOUSTIC_LOG
    elif theme == "egress":
        corpus = logs.EGRESS_LOG
    else:
        corpus = [(line, "dim") for (_tt, line, _k) in DMESG]
    events = [(t0 + 0.4 + k * 1.1, txt, kind) for k, (txt, kind) in enumerate(corpus)]
    render_log(ui, x0 + 2, y0 + 2, x1 - 2, y1 - 1, events, t, cps=70)


def worldmon_lines(t, m):
    """Continuously generated world-mon output (step / loss / sync)."""
    out = []
    if t < 14.2:
        return out
    step0 = 0.041
    k0 = int((t - 14.2) / (step0 * 20))
    for j in range(0, k0 + 1):
        tt = 14.2 + j * step0 * 20
        if tt > t:
            continue
        ld = CLUSTER.metrics(tt)["load"]
        st = CLUSTER.step(tt)
        loss = 2.31 - 0.42 * ld + 0.35 * math.exp(-j * 0.05) + 0.01 * math.sin(j)
        out.append((round(tt, 2),
                    f"[step {st:>6}] loss {loss:.4f}  \u03b7 3.0e-4  gnorm {0.7 + 0.3 * ld:.3f}  "
                    f"fast/slow/ultra sync ok", "plain"))
    return out


# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------
def scene_boot_full(ui, t, m):
    pane(ui, 0, 1, COLS - 1, ROWS - 2, "theta-svc-00 :: console (ttyS0)", active=True)
    render_log(ui, 2, 2, COLS - 3, ROWS - 3, CONSOLE, t, cps=60)
    return 0.0, 0.0


def scene_split2(ui, t, m):
    pane(ui, 0, 1, 113, ROWS - 2, "0:world*", active=True)
    pane(ui, 114, 1, COLS - 1, ROWS - 2, "1:journal", active=False)
    con = [e for e in CONSOLE if e[2] in ("cmd", "ok", "plain", "dim")]
    render_log(ui, 2, 2, 111, ROWS - 3, con, t)
    render_log(ui, 116, 2, COLS - 3, ROWS - 3, JOURNAL, t, cps=64)
    flash = 0.0
    for e in logs.SCENE_B:
        if e[2] == "big" and 0 <= t - e[0] < 0.18:
            flash = 0.35 * (1 - (t - e[0]) / 0.18)
    return flash, 0.0


def scene_split3(ui, t, m):
    pane(ui, 0, 1, 63, ROWS - 2, "0:console", active=True)
    pane(ui, 128, 1, COLS - 1, ROWS - 2, "1:journal", active=False)
    con = [e for e in CONSOLE if e[2] in ("cmd", "ok", "plain", "dim")]
    render_log(ui, 2, 2, 61, ROWS - 3, con, t)
    art.rack_map(ui, 64, 1, 127, ROWS - 2, t, m)
    render_log(ui, 130, 2, COLS - 3, ROWS - 3, JOURNAL + worldmon_lines(t, m), t, cps=64)
    flash = 0.0
    for e in logs.SCENE_C:
        if e[2] == "big" and 0 <= t - e[0] < 0.18:
            flash = 0.4 * (1 - (t - e[0]) / 0.18)
    return flash, 0.0


def scene_dashboard(ui, t, m):
    midx = 63
    # top row
    htop_pane(ui, 0, 1, midx, 26, t, m)
    nvtop_pane(ui, midx + 1, 1, 127, 26, t, m)
    thermal_pane(ui, 128, 1, COLS - 1, 26, t, m)
    # bottom row
    pane(ui, 0, 27, 127, ROWS - 2, "1:journal :: world-core", active=True)
    render_log(ui, 2, 28, 125, ROWS - 3, JOURNAL + worldmon_lines(t, m), t, cps=70)
    net_pane(ui, 128, 27, COLS - 1, ROWS - 2, t, m)
    flash = 0.0
    for e in logs.SCENE_D:
        if e[2] == "big" and 0 <= t - e[0] < 0.20:
            flash = 0.45 * (1 - (t - e[0]) / 0.20)
    return flash, 0.0


def draw(ui, t):
    m = CLUSTER.metrics(t)
    if t < 2.9:
        flash, glitch = scene_boot_full(ui, t, m)
        windows, active = ["bash", "journal", "bmc", "thermal", "ibmon"], 0
    elif t < 8.6:
        flash, glitch = scene_split2(ui, t, m)
        windows, active = ["world", "journal", "bmc", "thermal", "ibmon"], 0
    elif t < 13.9:
        flash, glitch = scene_split3(ui, t, m)
        windows, active = ["console", "journal", "nvtop", "thermal", "ibmon"], 0
    elif t < 30.0:
        flash, glitch = scene_dashboard(ui, t, m)
        windows, active = ["console", "journal", "nvtop", "thermal", "ibmon"], 1
    else:
        import acts
        flash, glitch = acts.draw(ui, t, m)
        windows, active = acts.windows(t)
    status_bar(ui, t, windows, active)
    footer(ui, t, m)
    return flash, glitch, m
