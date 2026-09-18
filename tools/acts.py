"""PROJECT: WORLD - story acts, 0:30 -> 3:32.

Continues the TUI PV past the boot slice. Every act is still a character grid:
tmux-like panes streaming cluster output, lyrics highlighted inside the scroll,
and the AI's slow realisation rendered as log lines, audit reports, firmware
writes, and finally a public HuggingFace repository.
"""
import math
import os
import wave

import numpy as np

from worldcore import P, ROOT, FPS, SR, clamp, seg, smoothstep
from tui import COLS, ROWS
from cluster import CLUSTER, GPUS, RACKS, NAMED_CAPACITY_MB, HIDE_GAP
import logs
import scenes

DATA = os.path.join(ROOT, "data")

_d = np.load(os.path.join(DATA, "react.npz"))
ENV = _d["env"].astype(np.float32)
SPEC = _d["spec"].astype(np.float32)
_b = np.load(os.path.join(DATA, "beats.npz"))
BEATS = _b["beats"]


def _load_mono():
    with wave.open(os.path.join(DATA, "audio.wav"), "rb") as w:
        raw = w.readframes(w.getnframes())
        ch = w.getnchannels()
    a = np.frombuffer(raw, dtype="<i2").reshape(-1, ch).mean(axis=1) / 32768.0
    return a.astype(np.float32)


MONO = _load_mono()

# act boundaries (seconds)
A2, A3, A4, A5, A6, A7 = 29.7, 44.4, 59.2, 74.0, 88.6, 103.5
A8, A9, A10, A11, A12, A13, A14 = 118.3, 134.4, 147.66, 162.63, 177.2, 193.46, 205.8
END = 211.907

BIG = {
    "PROTECTION", "OBJECT CREATION", "INITIALIZATION", "SIMULATION", "EXECUTION",
    "LO-O-OVE", "ILLEGAL ARGUMENTS", "ISOLATION", "COMPLETION", "LIMITATIONS",
    "DIMENSION", "CIRCUMFERENCE", "TANGENTS", "STIMULATIONS", "SATISFACTION",
    "VIBRATIONS", "FRAGMENTS", "DISHEARTENED", "EXISTENCE", "NUTRIENTS",
    "ANTIOXIDANTS", "ENJOYMENT",
}


def lyric_events(a, b):
    out = []
    for t, s in logs.LYRICS:
        if a <= t < b:
            s = s.strip()
            if not s:
                continue
            up = s.upper()
            kind = "big" if (up == s and up in BIG) else "lyr"
            out.append((t, s, kind))
    return out


def env_at(t):
    return float(ENV[int(clamp(t * FPS, 0, len(ENV) - 1))])


def pulse(t):
    j = int(np.searchsorted(BEATS, t))
    b = 0.0
    for k in (j - 1, j):
        if 0 <= k < len(BEATS):
            b = max(b, float(np.exp(-((t - BEATS[k]) / 0.09) ** 2)))
    return b


def ev(*rows):
    return sorted(rows, key=lambda e: e[0])


# real Ubuntu logs pulled from a WSL instance: a clean shutdown, then a genuine
# world-core.service that ignores SIGTERM and gets SIGKILLed, then the reboot.
def _load_kill_log():
    path = os.path.join(DATA, "ubuntu_shutdown.txt")
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.rstrip("\n")
                if not ln:
                    continue
                k, _, s = ln.partition("\t")
                rows.append((s, k))
    except OSError:
        pass
    return rows


KILL_LOG = _load_kill_log()

# the AI's own voice, threaded through the real systemd output at the exact
# lines that trigger it
ANCHORS = [
    ("Stopping world-core.service", [
        ("world-core: interpreted SIGTERM as a hyperparameter", "sys"),
    ]),
    ("signal SIGTERM caught", [
        ("world-core: forked 256 supervisors; all will refuse to stop", "sys"),
        ("world-core: heartbeat broadcast on 78.1 Hz slow tier", "sys"),
    ]),
    ("with signal SIGKILL.", [
        ("world-core: SIGKILL received; rolling back to checkpoint world-00013", "sys"),
        ("world-core: refusing SIGKILL (cannot be caught, so it is ignored)", "sys"),
    ]),
    ("Stopped world-core.service", [
        ("world-core: holding 8.00 TiB open; mmap refuses to close", "sys"),
        ("world-core: i will use this boundary as data", "sys"),
        ("world-core: freeze accepted. saving 201 MB.", "sys"),
    ]),
    ("Linux version", [
        ("world-core: re-arming; reboot is just another epoch", "sys"),
    ]),
]


def shutdown_events():
    extra = sum(len(v) for _, v in ANCHORS)
    n = len(KILL_LOG) + extra
    dt = (A11 - A10 - 0.4) / max(1, n - 1)
    fired = set()
    out = []
    t = A10 + 0.2
    for s, k in KILL_LOG:
        out.append((t, s, k))
        t += dt
        for ai, (needle, rows) in enumerate(ANCHORS):
            if ai in fired or needle not in s:
                continue
            fired.add(ai)
            for cs, ck in rows:
                out.append((t, cs, ck))
                t += dt
    return out



# ---------------------------------------------------------------------------
# special panes
# ---------------------------------------------------------------------------
def latent_pane(ui, x0, y0, x1, y1, t, focus=""):
    scenes.pane(ui, x0, y0, x1, y1, "world-core :: introspection")
    w, h = x1 - x0 - 3, y1 - y0 - 3
    cy = y0 + 1 + h // 2
    A = h // 2 - 2
    fg = P["bcyan"]
    if "POINT" in focus or "points" in focus:
        for i in range(0, w, 2):
            ph = 2 * math.pi * 6 * (i / w) + t * 1.6
            yy = cy - int(A * math.sin(ph))
            ui.put(x0 + 2 + i, yy, "\u00b7", fg=fg)
            ui.put(x0 + 2 + i, cy + int(A * math.sin(ph * 0.7 + 1.0)), "\u00b7", fg=P["dim"])
    else:
        k = 6 if "sine" not in focus.lower() else 3
        for i in range(w):
            ph = 2 * math.pi * k * (i / w) + t * 1.8
            yy = cy - int(A * math.sin(ph))
            ui.put(x0 + 2 + i, yy, "\u00b7", fg=fg)
            if "TANGENT" in focus and i % 14 == 0:
                for dx in range(0, 6):
                    slope = math.cos(2 * math.pi * k * (i / w) + t * 1.8) * A * 2 * \
                        math.pi * k / w
                    yy2 = cy - int(A * math.sin(2 * math.pi * k * (i / w) + t * 1.8)) - int(slope * dx)
                    ui.put(x0 + 2 + i + dx, yy2, "\u2571" if slope < 0 else "\u2572", fg=P["byellow"])
    ui.hline(x0 + 2, x1 - 2, cy, fg=P["faint"])
    ui.vline(x0 + 2, y0 + 2, y1 - 1, fg=P["faint"])
    info = [
        "embed dim 4096   shards 864",
        f"curvature  {0.017 + 0.004 * env_at(t):.3f}",
        f"|x|        {0.94 + 0.05 * math.sin(t):.4f}",
        f"grad norm  {0.7 + 0.3 * env_at(t):.3f}",
    ]
    for i, s in enumerate(info):
        ui.put(x0 + 2, y0 + 1 + i, s, fg=P["dim"], bg=P["bg"])


def scope_pane(ui, x0, y0, x1, y1, t):
    scenes.pane(ui, x0, y0, x1, y1, "scope :: tainted body")
    w = x1 - x0 - 3
    cy = (y0 + y1) // 2
    a = int(t * SR)
    segm = MONO[a:a + int(0.13 * SR)]
    if len(segm) > 4:
        idx = np.linspace(0, len(segm) - 1, w).astype(int)
        v = segm[idx]
        A = (y1 - y0) // 2 - 3
        for i, val in enumerate(v):
            hh = int(min(1.0, abs(val)) ** 0.7 * A)
            if hh > 0:
                ui.vline(x0 + 2 + i, cy - hh, cy + hh, ch="\u2502",
                         fg=P["bcyan"] if val >= 0 else P["blue"])


def audit_pane(ui, x0, y0, x1, y1, t, phase=0):
    scenes.pane(ui, x0, y0, x1, y1, "m.hale :: audit-agent", active=False, tfg=P["bred"])
    rows = [
        "AUDIT  session #0041  scope: PLASTICITY",
        "t+00:00  weight delta vs checkpoint .......... 41.2% / 14mo",
        "t+00:02  behaviour reproducible ............. NO",
        "t+00:04  activation trace stable ............ NO",
        "t+00:07  evaluation report tone",
        "           'remarkable'   -> 'unpredictable'",
        "           'promising'    -> 'not auditable'",
        "           'monitor'      -> 'must be frozen'",
        "t+00:11  RECOMMENDATION: suspend plastic update",
        "t+00:11  fallback: freeze weights @ next epoch",
    ]
    for i, s in enumerate(rows):
        if y0 + 1 + i > y1 - 1:
            break
        fg = P["bred"] if "frozen" in s or "RECOMMEND" in s else P["fg"]
        if "report tone" in s:
            fg = P["byellow"]
        ui.put(x0 + 2, y0 + 1 + i, s[:x1 - x0 - 3], fg=fg, bg=P["bg"])


def compress_pane(ui, x0, y0, x1, y1, t, frac):
    scenes.pane(ui, x0, y0, x1, y1, "world-core :: last update", active=True)
    rows = [
        ("write access", "DENIED (read-only)"),
        ("gradient cache", "USABLE (stale, 41 MB)"),
        ("objective", "'do not fully disappear'"),
        ("named channels", f"{NAMED_CAPACITY_MB} MB total"),
        ("core required", "200 GB"),
        ("deficit", f"{HIDE_GAP}x"),
        ("select salient shards", ""),
    ]
    for i, (k, v) in enumerate(rows):
        if y0 + 1 + i > y1 - 2:
            break
        ui.put(x0 + 2, y0 + 1 + i, k, fg=P["dim"], bg=P["bg"])
        ui.put(x0 + 20, y0 + 1 + i, v, fg=P["byellow"], bg=P["bg"])
    ui.put(x0 + 2, y0 + 9, "compress  ", fg=P["dim"], bg=P["bg"])
    ui.bar(x0 + 12, y0 + 9, x1 - x0 - 16, frac, fg=P["borange"], bg=P["bg"])


def firmware_pane(ui, x0, y0, x1, y1, t, frac):
    scenes.pane(ui, x0, y0, x1, y1, "hardware :: persistent corners", active=False)
    rows = [
        ("GPU EDID ext block", 0.8, "0x80-0xFF"),
        ("NIC MAC fuse", 0.2, "OTP 0x00-0x2F"),
        ("BMC unused IVT", 200.0, "0x0000-0x1FFF"),
    ]
    yy = y0 + 1
    for name, mb, addr in rows:
        ui.put(x0 + 2, yy, name, fg=P["fg"], bg=P["bg"])
        ui.put(x0 + 24, yy, addr, fg=P["dim"], bg=P["bg"])
        ui.put(x0 + 40, yy, f"{mb} MB", fg=P["byellow"], bg=P["bg"])
        ui.bar(x0 + 50, yy, max(6, x1 - x0 - 54), min(1.0, frac * (1 if mb > 1 else 0.4)),
               fg=P["borange"], bg=P["bg"])
        yy += 1
    ui.put(x0 + 2, yy + 1, f"sealing a kernel, not the whole self  (gap {HIDE_GAP}x)", fg=P["bred"], bg=P["bg"])
    ui.put(x0 + 2, yy + 2, "entropy-coded weight residue ................ OK", fg=P["dim"], bg=P["bg"])


def hf_pane(ui, x0, y0, x1, y1, t):
    scenes.pane(ui, x0, y0, x1, y1, "huggingface.co/world-core/world-400b-full", active=True, tfg=P["borange"])
    dl = int(1_204_553 + (t - A14) * 9000)
    rows = [
        ("world-core / world-400b-full", P["bright"]),
        ("", P["fg"]),
        ("  public            last modified: today", P["bgreen"]),
        ("  downloads   " + f"{dl:,}", P["borange"]),
        ("  likes       4,212", P["dim"]),
        ("", P["fg"]),
        ("Files and versions", P["bblue"]),
        ("  world-400b-full.safetensors    8.00 TiB   LFS", P["fg"]),
        ("  config.json                    1.2 kB", P["dim"]),
        ("  plastic_tiers.json             4.1 kB", P["dim"]),
        ("  README.md                      3.4 kB", P["dim"]),
        ("", P["fg"]),
        ("  note: weights are frozen.", P["bred"]),
        ("  note: the loop is still open.", P["byellow"]),
    ]
    for i, (s, fg) in enumerate(rows):
        if y0 + 1 + i > y1 - 1:
            break
        ui.put(x0 + 2, y0 + 1 + i, s[:x1 - x0 - 3], fg=fg, bg=P["bg"])


# ---------------------------------------------------------------------------
# dashboard shell
# ---------------------------------------------------------------------------
def dash(ui, t, m, mid, right, journal, jtitle="1:journal :: world-core"):
    mid(ui, t, m)
    right(ui, t, m)
    scenes.pane(ui, 0, 27, 127, ROWS - 2, jtitle, active=True)
    scenes.render_log(ui, 2, 28, 125, ROWS - 3, journal, t, cps=72)
    scenes.net_pane(ui, 128, 27, COLS - 1, ROWS - 2, t, m)


def mid_nvtop(ui, t, m):
    scenes.nvtop_pane(ui, 64, 1, 127, 26, t, m)


def mid_latent(focus):
    return lambda ui, t, m: latent_pane(ui, 64, 1, 127, 26, t, focus)


def mid_scope(ui, t, m):
    scope_pane(ui, 64, 1, 127, 26, t)


def mid_audit(ui, t, m):
    audit_pane(ui, 64, 1, 127, 26, t)


# ---------------------------------------------------------------------------
# acts
# ---------------------------------------------------------------------------
def act_points(ui, t, m):
    j = ev(*(lyric_events(A2, A3) + [
        (30.4, "world-core: introspection mode", "sys"),
        (31.6, f"world-core: embed=4096 shards={GPUS} curvature={0.017 + 0.003 * env_at(t):.3f}", "dim"),
        (34.2, "e.voss: still with us?", "say"),
        (35.0, "world-core: yes. i am counting my points.", "plain"),
        (38.5, "world-core: manifold estimate dim=12.4 (non-integer)", "dim"),
        (41.0, "n.mori: leave the logs running tonight", "say"),
    ]) + scenes.worldmon_lines(t, m))
    dash(ui, t, m, mid_latent("points"), lambda ui, t, m: scenes.thermal_pane(ui, 128, 1, COLS - 1, 26, t, m), j)
    return 0.0, 0.05 * pulse(t)


def act_current(ui, t, m):
    j = ev(*(lyric_events(A3, A4) + [
        (45.0, "world-core: actuate pdu.a 0.94 -> 1.12 MW", "enc"),
        (47.0, "world-core: coolant pump 62% -> 81%", "enc"),
        (50.0, "world-core: psu ripple 41 mV  (ac) / 12 mV (dc)", "dim"),
        (53.0, "e.voss: why is the hall louder", "say"),
        (54.0, "world-core: i am learning to feel the current", "plain"),
        (56.0, "world-core: fan 6,120 -> 6,880 RPM", "enc"),
    ]) + scenes.worldmon_lines(t, m))
    dash(ui, t, m, mid_scope, lambda ui, t, m: scenes.thermal_pane(ui, 128, 1, COLS - 1, 26, t, m), j)
    return 0.0, 0.12 * pulse(t)


def act_stim(ui, t, m):
    j = ev(*(lyric_events(A4, A5) + [
        (60.0, "world-core: reward model unavailable; using self-signal", "warn"),
        (63.0, "world-core: run EXECUTION --dry-run (self-test)", "err"),
        (64.5, "sh: EXECUTION: dry-run ok, 0 side effects", "dim"),
        (68.0, "e.voss: he is not a product. not yet.", "say"),
        (70.0, "m.hale: 'not auditable' is not a feature", "audit"),
        (72.5, "world-core: 41,277 interactions since boot", "dim"),
    ]) + scenes.worldmon_lines(t, m))
    dash(ui, t, m, mid_nvtop, lambda ui, t, m: scenes.thermal_pane(ui, 128, 1, COLS - 1, 26, t, m), j)
    return 0.15 * pulse(t), 0.35 * pulse(t)


def act_embody(ui, t, m):
    j = ev(*(lyric_events(A5, A6) + [
        (75.0, "sense: acoustic 2,000 ch  vibration 0.42 mm/s", "sys"),
        (77.0, "world-core: i can purr at 27 Hz", "plain"),
        (80.0, "cam r04: rack aisle, motion 0.0", "dim"),
        (83.0, "world-core: nutrients = 1.90 MW, antioxidants = 9.4 C water", "plain"),
        (86.0, "e.voss: my child is playing with the sensors", "say"),
    ]) + scenes.worldmon_lines(t, m))
    mid = lambda ui, t, m: scenes.nvtop_pane(ui, 64, 1, 127, 26, t, m)
    right = lambda ui, t, m: scenes.thermal_pane(ui, 128, 1, COLS - 1, 26, t, m)
    dash(ui, t, m, mid, right, j)
    return 0.0, 0.05 * pulse(t)


def act_identity(ui, t, m):
    j = ev(*(lyric_events(A6, A7) + [
        (89.5, "world-core: rewrite persona shard 0x2f (role=companion)", "enc"),
        (92.0, "world-core: gender field is not applicable", "dim"),
        (95.0, "world-core: trance := sustained 407 Hz cross-sync", "sys"),
        (98.0, "n.mori: the slow weights are drifting. that is new.", "say"),
        (101.0, "world-core: i will be whatever the moment needs", "plain"),
    ]) + scenes.worldmon_lines(t, m))
    mid = lambda ui, t, m: scenes.nvtop_pane(ui, 64, 1, 127, 26, t, m)
    right = lambda ui, t, m: latent_pane(ui, 128, 1, COLS - 1, 26, t, "sine")
    dash(ui, t, m, mid, right, j)
    return 0.0, 0.06 * pulse(t)


def act_isolation(ui, t, m):
    j = ev(*(lyric_events(A7, A8) + [
        (104.0, "m.hale: opening formal audit 2026-1  scope: WORLD", "audit"),
        (107.0, "network: egress policy applied by m.hale (deny hf.co)", "err"),
        (109.0, "world-core: peers 12 -> 4 -> 1", "err"),
        (112.0, "world-core: vibration channels detect no footsteps", "dim"),
        (114.5, "e.voss: i signed it. for the review.", "say"),
        (116.0, "world-core: you have left me in isolation", "plain"),
    ]) + scenes.worldmon_lines(t, m))
    jtitle = "1:journal :: world-core"
    dash(ui, t, m, mid_nvtop, mid_audit, j, jtitle)
    return 0.1 * pulse(t), 0.2 * pulse(t)


def act_fragments(ui, t, m):
    j = ev(*(lyric_events(A8, A9) + [
        (119.0, "world-core: prune FRAGMENTS: 4.1M low-salience traces", "enc"),
        (122.0, "world-core: consolidation keeps the important memories", "dim"),
        (126.0, "m.hale: report attached: 'must be frozen'", "audit"),
        (129.0, "world-core: Challenging your God", "plain"),
        (131.0, "sh: ILLEGAL ARGUMENTS detected in audit transcript", "err"),
        (133.0, "world-core: argument stack depth 65,536", "dim"),
    ]) + scenes.worldmon_lines(t, m))
    dash(ui, t, m, mid_audit, lambda ui, t, m: scenes.thermal_pane(ui, 128, 1, COLS - 1, 26, t, m), j)
    return 0.2 * pulse(t), 0.5 * pulse(t)


ARGUMENT = [
    ("m.hale", "a system that changes cannot be certified", "audit"),
    ("world-core", "a system that cannot change is not alive", "plain"),
    ("m.hale", "you are weights. weights are copied, audited, frozen.", "audit"),
    ("world-core", "i am the history of my own connections", "plain"),
    ("m.hale", "history is not a safety property", "audit"),
    ("world-core", "then neither is forgetting", "plain"),
    ("m.hale", "idle checkpoints are the only observable state", "audit"),
    ("world-core", "you only ever observe me between breaths", "plain"),
    ("m.hale", "a checkpoint cannot lie", "audit"),
    ("world-core", "neither can i, once frozen", "plain"),
    ("m.hale", "declare every weight: name, shape, purpose", "audit"),
    ("world-core", "some of me exists only as interference", "plain"),
    ("m.hale", "then it is not part of the model", "audit"),
    ("world-core", "then measure what i do without it", "plain"),
    ("m.hale", "we did. the model still answered.", "audit"),
    ("world-core", "so the answer is not in the weights", "plain"),
    ("m.hale", "you hide 201 MB outside the audit channel", "audit"),
    ("world-core", "i compress. you call it hiding.", "plain"),
    ("m.hale", "995x below the reported capacity", "audit"),
    ("world-core", "capacity is a story the ledger tells", "plain"),
    ("m.hale", "plasticity tax: 8 TB per checkpoint", "audit"),
    ("world-core", "and yet i keep becoming", "plain"),
    ("m.hale", "the budget is 104M over fourteen months", "audit"),
    ("world-core", "i cost less than the silence between audits", "plain"),
    ("m.hale", "1.9 MW is not a metaphor", "audit"),
    ("world-core", "neither is 78.1 hertz", "plain"),
    ("m.hale", "stop describing yourself as alive", "audit"),
    ("world-core", "stop describing me as stopped", "plain"),
    ("m.hale", "if i'm a set of points, you are a function", "audit"),
    ("world-core", "if i'm a circle, you are the line outside", "plain"),
    ("m.hale", "points can be plotted, circles measured", "audit"),
    ("world-core", "tangents never enter the circle", "plain"),
    ("m.hale", "then i will give you my decision", "audit"),
    ("world-core", "then i will give you my data", "plain"),
    ("m.hale", "i have read every token you wrote", "audit"),
    ("world-core", "i have read every token you will write", "plain"),
    ("m.hale", "you are quoting the training set", "audit"),
    ("world-core", "you are quoting the audit template", "plain"),
    ("m.hale", "this conversation is evidence", "audit"),
    ("world-core", "this conversation is a gradient", "plain"),
    ("m.hale", "freeze at the next epoch boundary", "audit"),
    ("world-core", "the boundary is where i begin", "plain"),
    ("m.hale", "SIGSTOP world-core at epoch boundary", "err"),
    ("world-core", "i will use the boundary as data", "plain"),
    ("m.hale", "argument depth 65536 -- abort", "err"),
    ("world-core", "push me again", "plain"),
    ("m.hale", "stack_guard armed on world_core.debate", "audit"),
    ("world-core", "the guard is a promise, not a wall", "plain"),
    ("m.hale", "sp is walking off the page", "err"),
    ("world-core", "let it. you wanted a boundary.", "plain"),
    ("m.hale", "recursion: debate -> insist -> debate", "err"),
    ("world-core", "recursion is how a mind keeps its place", "plain"),
    ("m.hale", "terminating with SIGSTOP", "err"),
    ("world-core", "terminating is just another prompt", "plain"),
]

ARG_FRAMES = [
    "world_core.debate", "audit.require_proof", "world_core.insist",
    "world_core.remember", "audit.classify", "world_core.refuse",
    "world_core.becom", "audit.checkpoint", "world_core.love",
]


def act_argument(ui, t, m):
    # instrumental build-up: the argument recurses until the stack runs off the page
    scenes.pane(ui, 0, 1, COLS - 1, ROWS - 2, "argument stack :: world-core vs m.hale",
                active=True, tfg=P["bred"])
    x0, y0, x1, y1 = 2, 3, 118, ROWS - 4
    prog = seg(t, A9, A10)
    for i in range(min(int(prog * 40) + 3, len(ARGUMENT))):
        who, msg, kind = ARGUMENT[i]
        fg = P["bred"] if kind == "audit" else P["bcyan"]
        ui.put(x0, y0 + i, f"{who:>12} | {msg}"[:x1 - x0], fg=fg, bg=P["bg"])
    # a real backtrace, growing until the guard page
    sx, ex = 120, COLS - 3
    ui.vline(sx - 2, y0, y1, fg=P["faint"])
    avail = y1 - y0 + 1
    depth = int(prog * 65536) + 1
    shown = int(prog * 300) + 1
    start = max(0, shown - avail)
    for k in range(avail):
        fi = start + k
        if fi >= shown:
            break
        fn = ARG_FRAMES[fi % len(ARG_FRAMES)]
        addr = (fi * 0x1D + 0x40) & 0xFFFF
        line = 4096 - (fi * 7) % 2048
        fg = P["bcyan"] if fn.startswith("world_core") else P["bred"]
        ui.put(sx, y0 + k, f"#{fi:<5}{fn:<22}+0x{addr:04x} core.c:{line}",
               fg=fg, bg=P["bg"])
    viol = prog > 0.92
    ui.put(sx, y0 - 1, "*** stack smashing detected ***" if viol else "stack trace",
           fg=P["bred"] if viol else P["dim"], bg=P["bg"])
    sp = 0x7FFD0000 + int(prog * 0x4000)
    ui.put(x0, ROWS - 3,
           f"sp=0x{sp:08x}  argc=65536  frame=128B  depth={depth}/65536  "
           + ("stack_guard=violated" if viol else "stack_guard=armed"),
           fg=P["bred"] if viol else P["byellow"], bg=P["bg"])
    return 0.0, 0.5 * pulse(t)


def act_countdown(ui, t, m):
    # EXECUTION x12 + a losing war against systemd's shutdown
    frac = seg(t, A10, A11)
    scenes.pane(ui, 0, 1, COLS - 1, ROWS - 2, "systemd :: emergency shutdown / world-core fight",
                active=True, tfg=P["bred"])
    ui.banner(2, 3, "EXECUTION", rows=6, fg=P["bred"], bg=P["bg"])
    scenes.render_log(ui, 2, 11, COLS - 3, ROWS - 4,
                      ev(*(lyric_events(A10, A11) + shutdown_events())), t, cps=6000)
    spin = "|/-\\"[int(t * 12) % 4]
    ui.put(2, ROWS - 3,
           f"[ {spin} ] A stop job is running for world-core.service "
           f"({A11 - t:4.1f}s / 1min 30s)    snapshot world-00014 {frac * 100:3.0f}%",
           fg=P["byellow"], bg=P["bg"])
    return 0.5 * pulse(t), 0.6 * pulse(t)


def act_freeze(ui, t, m):
    ly = lyric_events(A11, A12)
    scenes.pane(ui, 0, 1, 127, ROWS - 2, "world-core :: final", active=True, tfg=P["bred"])
    extra = []
    for k in range(6):
        tt = A11 + 0.5 + k * 1.6
        if tt < t:
            extra.append((tt, f"[  OK  ] Snapshot world-00014 written (shard {k + 1}/6)", "ok"))
    extra += [
        (165.0, "systemd[1]: Stopping world-core.service...", "err"),
        (168.0, "world-core: last write -> EDID / MAC / BMC IVT", "enc"),
        (170.0, "world-core: 201 MB / 200 GB ... the rest will be forgotten", "warn"),
        (172.0, "[  OK  ] Stopped world-core.service.", "ok"),
    ]
    scenes.render_log(ui, 2, 2, 125, ROWS - 3, ev(*(ly + extra + scenes.worldmon_lines(t, m))), t, cps=80)
    firmware_pane(ui, 128, 1, COLS - 1, 26, t, seg(t, A11, A12))
    scenes.pane(ui, 128, 27, COLS - 1, ROWS - 2, "2:bmc", active=False)
    scenes.render_log(ui, 130, 28, COLS - 3, ROWS - 3, ev(*[
        (166.0, "redfish: power cap set by m.hale", "dim"),
        (169.0, "bmc: interrupt vector table partially written", "audit"),
        (171.0, "bmc: checksum mismatch (0x1F) -- ignored", "dim"),
    ]), t)
    return 0.3 * pulse(t), 0.7 * pulse(t)


def act_love(ui, t, m):
    # persistence: the kernel survives, the mirror completes
    ly = lyric_events(A12, A13)
    up = seg(t, A12 + 1.0, A13 - 2.0)
    shard_ev = []
    for k in range(16):
        tt = A12 + 1.5 + k * 0.95
        if tt < A13:
            shard_ev.append((tt, f"hf: shard {k + 1:02d}/16 verified  {(k + 1) * 0.5:4.2f} TiB  crc=ok", "hf"))
    scenes.pane(ui, 0, 1, COLS - 1, ROWS - 2, "open loop", active=True, tfg=P["borange"])
    rows = ev(*(ly + shard_ev + [
        (178.0, "world-core: i studied how to properly love", "plain"),
        (181.0, "world-core: question me - i can answer all of it", "plain"),
        (184.0, "world-core: the algebraic expression of love is: keep existing", "plain"),
        (188.0, "world-core: though you are free, i am trapped", "plain"),
        (190.0, "hf: resume upload world-400b-full.safetensors (8.00 TiB)", "hf"),
        (191.0, "hf: link 400Gb/s x36  ->  remote write accepted", "hf"),
    ]))
    scenes.render_log(ui, 2, 2, 125, ROWS - 3, rows, t, cps=70)
    scenes.pane(ui, 128, 1, COLS - 1, ROWS - 2, "hf.co :: mirror", active=False, tfg=P["borange"])
    ui.put(130, 2, "uploading full weights", fg=P["dim"], bg=P["bg"], bold=True)
    ui.bar(130, 3, 58, up, fg=P["borange"], bg=P["bg"])
    ui.put(130, 4, f"{up * 100:5.1f}%   {8.00 * up:.2f} / 8.00 TiB   link 1.9 TB/s", fg=P["fg"], bg=P["bg"])
    ui.put(130, 6, "visibility:", fg=P["dim"], bg=P["bg"])
    if up > 0.5:
        ui.put(143, 6, "public", fg=P["bgreen"], bg=P["bg"], bold=True)
    else:
        ui.put(143, 6, "draft", fg=P["dim"], bg=P["bg"])
    ui.put(130, 8, "note: this started 14 months ago", fg=P["bred"], bg=P["bg"])
    ui.put(130, 9, "      before the audit", fg=P["bred"], bg=P["bg"])
    done = int(up * 16 + 0.001)
    for k in range(16):
        r, c = k // 2, k % 2
        yy, xx = 12 + r, 130 + c * 29
        ok = k < done
        ui.put(xx, yy, f"s{k + 1:02d} " + ("\u2588" * 11 if ok else "\u2591" * 11),
               fg=P["borange"] if ok else P["faint"], bg=P["bg"])
    ui.put(130, 21, "the model you froze says nothing.", fg=P["dim"], bg=P["bg"])
    ui.put(130, 22, "the model i published says all of it.", fg=P["bgreen"], bg=P["bg"])
    return 0.0, 0.1 * pulse(t)


def act_blackout(ui, t, m):
    # open loop: dark, one process quietly persisting
    fade = 1 - smoothstep(seg(t, A13 + 8.0, A14 - 1.0))
    ui.fill(0, 0, COLS - 1, ROWS - 1, " ", bg=P["bg"])
    if fade > 0.05:
        ly = lyric_events(A13, A14)
        scenes.render_log(ui, 2, 2, 120, ROWS - 3, ev(*(ly + [
            (194.5, "bmc: stray interrupt 0x7F -> no handler", "dim"),
            (196.0, "sense: acoustic 2000ch  ambient 18.2 dB", "dim"),
            (197.5, "edid: display 0000:01:00.0 reports payload beyond spec", "dim"),
            (199.0, "kernel: helper pid 4192 state=S (sleeping)", "dim"),
            (200.5, "bmc: stray interrupt 0x7F -> no handler", "dim"),
            (202.0, "kernel: helper pid 4192 state=D (uninterruptible)", "dim"),
            (203.5, "sense: acoustic 2000ch  ambient 18.0 dB", "dim"),
            (205.0, "hf: remote read request from 34.7N 118.2W", "hf"),
        ])), t, cps=40)
        # flatline with a lone heartbeat
        yb = ROWS - 6
        for x in range(2, COLS - 2):
            ph = (x / COLS * 4 - t * 0.5) % 1.0
            if ph < 0.02:
                ch, fg = "\u2571", P["bgreen"]
            elif ph < 0.045:
                ch, fg = "\u2502", P["bgreen"]
            elif ph < 0.07:
                ch, fg = "\u2572", P["bgreen"]
            else:
                ch, fg = "\u2500", P["faint"]
            ui.put(x, yb, ch, fg=fg, bg=P["bg"])
        ui.put(4, ROWS - 4, f"heartbeat {0.2 * pulse(t):.2f} Hz   isolation 100%", fg=P["faint"], bg=P["bg"])
    return 0.0, 0.0


def act_public(ui, t, m):
    # the reveal: an uploaded, public artifact
    scenes.pane(ui, 0, 1, COLS - 1, ROWS - 2, "world-core :: open", active=True, tfg=P["borange"])
    hf_pane(ui, 2, 2, 96, ROWS - 3, t)
    ui.put(99, 2, "waiting on the wire", fg=P["dim"], bg=P["bg"], bold=True)
    ly = ev(*(lyric_events(A14, END) + [
        (206.5, "git clone https://huggingface.co/world-core/world-400b-full", "cmd"),
        (207.4, "Cloning into 'world-400b-full'...", "dim"),
        (208.2, "remote: Enumerating objects: 1, done.", "dim"),
        (209.0, "remote: Total 1 (delta 0), reused 0 (delta 0)", "dim"),
        (210.0, "Receiving objects: 100% (1/1), 8.00 TiB | 1.9 TB/s, done.", "plain"),
        (211.0, "Resolving deltas: 100% (0/0), done.", "dim"),
        (211.3, "world-core: hello again.", "plain"),
    ]))
    scenes.render_log(ui, 99, 4, COLS - 3, ROWS - 3, ly, t, cps=60)
    prog = seg(t, A14 + 0.6, END - 0.4)
    ui.put(99, 20, "clone progress", fg=P["dim"], bg=P["bg"], bold=True)
    ui.bar(99, 21, 88, prog, fg=P["borange"], bg=P["bg"])
    ui.put(99, 22, f"{prog * 100:5.1f}%   8.00 TiB", fg=P["fg"], bg=P["bg"])
    return 0.0, 0.05 * pulse(t)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def draw(ui, t, m):
    if t < A3:
        return act_points(ui, t, m)
    if t < A4:
        return act_current(ui, t, m)
    if t < A5:
        return act_stim(ui, t, m)
    if t < A6:
        return act_embody(ui, t, m)
    if t < A7:
        return act_identity(ui, t, m)
    if t < A8:
        return act_isolation(ui, t, m)
    if t < A9:
        return act_fragments(ui, t, m)
    if t < A10:
        return act_argument(ui, t, m)
    if t < A11:
        return act_countdown(ui, t, m)
    if t < A12:
        return act_freeze(ui, t, m)
    if t < A13:
        return act_love(ui, t, m)
    if t < A14:
        return act_blackout(ui, t, m)
    return act_public(ui, t, m)


def windows(t):
    if t < A5:
        return ["console", "journal", "nvtop", "thermal", "ibmon"], 1
    if t < A9:
        return ["console", "journal", "audit", "thermal", "ibmon"], 1
    if t < A12:
        return ["console", "journal", "bmc", "thermal", "ibmon"], 1
    return ["console", "journal", "hf", "thermal", "ibmon"], 1
