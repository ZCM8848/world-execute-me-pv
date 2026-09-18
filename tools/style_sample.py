"""PROJECT: WORLD - style probe. Renders the t=2.9s PROTECTION frame.

    python tools/style_sample.py
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from worldcore import P, W, H, Post, ROOT
from tui import Tui, build_atlas, COLS, ROWS


def build_frame():
    t = Tui()
    BG = P["bg"]
    PN = P["panel"]
    DIM = P["dim"]
    FG = P["fg"]
    WH = P["white"]
    BR = P["bright"]
    GR = P["green"]
    CY = P["bcyan"]
    YE = P["byellow"]
    BL = P["bblue"]

    # -- top tmux status bar ------------------------------------------------
    t.fill(0, 0, COLS - 1, 0, bg=P["green"])
    t.put(1, 0, "[theta] 0:boot* 1:world- 2:bmc 3:thermal 4:ibmon", fg=(0, 0, 0), bg=P["green"], bold=True)
    right = "PROJECT:WORLD  SITE THETA  63.88N 22.45W  02:47:19 UTC "
    t.put(COLS - len(right) - 1, 0, right, fg=(0, 0, 0), bg=P["green"], bold=True)

    # ===================== LEFT PANEL : boot ==============================
    lx0, ly0, lx1, ly1 = 1, 1, 118, 45
    t.fill(lx0 + 1, ly0 + 1, lx1 - 1, ly1 - 1, bg=PN)
    t.box(lx0, ly0, lx1, ly1, fg=P["faint"], bg=PN, title="world-core :: boot", tfg=WH)

    t.put(3, 2, "Ubuntu 24.04.2 LTS", fg=WH, bg=PN, bold=True)
    t.put(3 + 20, 2, "theta-svc-00", fg=CY, bg=PN)
    t.put(3 + 34, 2, "ttyS0", fg=DIM, bg=PN)
    t.put(lx1 - 33, 2, "kernel 6.11.0-24-generic", fg=DIM, bg=PN)

    t.banner(5, 4, "PROTECTION", rows=5, fg=BR, bg=PN)

    t.hline(3, lx1 - 2, 10, fg=P["faint"], bg=PN)

    def ok(row, x, msg):
        t.put(x, row, "[  OK  ]", fg=GR, bg=PN, bold=True)
        t.put(x + 9, row, msg, fg=FG, bg=PN)

    ok(11, 3, "Reached target Site Power.")
    ok(12, 3, "Started TPM2 Attestation Service.")
    ok(13, 3, "Mounted /secure/seal (luks-\u03b8-root).")
    ok(14, 3, "Started world-core.service - Continuous Plastic Network.")

    t.put(3, 16, "e.voss@theta-svc-00", fg=GR, bg=PN, bold=True)
    t.put(22, 16, ":~$", fg=WH, bg=PN)
    t.put(27, 16, "worldctl protection enable --mode seal", fg=BR, bg=PN)

    out = [
        ("attest    ", "TPM2 / PCR[0,2,4,7] .................. sealed", CY),
        ("secureboot", "enabled (db, shim 15.8-0ubuntu1)", CY),
        ("luks2     ", "luks-\u03b8-root ACTIVE  argon2id / 2 GiB", CY),
        ("plastic   ", "meta-\u03b7=3.0e-4  consolidation=ON  tiers=3", YE),
    ]
    for i, (k, v, c) in enumerate(out):
        t.put(3, 17 + i, k, fg=DIM, bg=PN)
        t.put(14, 17 + i, ":", fg=P["faint"], bg=PN)
        t.put(16, 17 + i, v, fg=c, bg=PN)

    t.put(3, 22, "[  OK  ]", fg=GR, bg=PN, bold=True)
    t.put(12, 22, "PROTECTION ARMED", fg=BR, bg=PN, bold=True)

    kern = [
        ("3.918441", "world-core: continuous plasticity online"),
        ("3.918532", "world-core: fast=0.2Hz slow=78.1Hz ultra=407.0Hz"),
        ("3.918744", "nvlink: 12x NVL72 domains up (864 x B300)"),
        ("3.919001", "ib: quantum-x800 800Gb/s links up 36/36"),
    ]
    for i, (ts, msg) in enumerate(kern):
        t.put(3, 24 + i, f"[{ts:>11}]", fg=P["faint"], bg=PN)
        t.put(17, 24 + i, msg, fg=DIM, bg=PN)

    t.put(3, 29, ">", fg=GR, bg=PN, bold=True)
    if (29 * 3) % 2 == 0:
        t.put(5, 29, "\u2588", fg=BR, bg=PN)

    t.put(3, 32, "plasticity tiers", fg=BL, bg=PN, bold=True)
    t.put(3, 33, "fast", fg=P["bgreen"], bg=PN)
    t.put(10, 33, "per-card local", fg=DIM, bg=PN)
    t.put(30, 33, "0.2 Hz", fg=FG, bg=PN)
    t.put(3, 34, "slow", fg=YE, bg=PN)
    t.put(10, 34, "intra-node NVLink", fg=DIM, bg=PN)
    t.put(30, 34, "78.1 Hz", fg=FG, bg=PN)
    t.put(3, 35, "ultra", fg=P["bred"], bg=PN)
    t.put(10, 35, "full-model IB ring", fg=DIM, bg=PN)
    t.put(30, 35, "407.0 Hz", fg=FG, bg=PN)
    t.put(3, 37, "state 400B", fg=DIM, bg=PN)
    t.put(14, 37, "0.40 TB frozen", fg=DIM, bg=PN)
    t.put(31, 37, "2.00 TB limit", fg=P["bblue"], bg=PN)
    t.put(46, 37, "8.00 TB full", fg=YE, bg=PN)
    t.put(3, 38, "plasticity tax", fg=DIM, bg=PN)
    t.put(19, 38, "20 bytes/param", fg=P["bmagenta"], bg=PN)
    t.put(35, 38, "sync span 1672x", fg=DIM, bg=PN)

    # ===================== RIGHT PANEL : attestation ======================
    rx0, ry0, rx1, ry1 = 120, 1, 190, 45
    t.fill(rx0 + 1, ry0 + 1, rx1 - 1, ry1 - 1, bg=PN)
    t.box(rx0, ry0, rx1, ry1, fg=P["faint"], bg=PN, title="attestation", tfg=WH)

    def sec(row, label):
        t.put(rx0 + 2, row, label, fg=BL, bg=PN, bold=True)

    def kv(row, k, v, vc=FG):
        t.put(rx0 + 2, row, k, fg=DIM, bg=PN)
        t.put(rx0 + 13, row, v, fg=vc, bg=PN)

    sec(2, "TPM2  PCR")
    pcr = [
        ("PCR[0]", "sha256  3f2a 19c4 b7e0 d18a  \u2026  c91d"),
        ("PCR[2]", "sha256  8b01 e77d 44af 20c9  \u2026  7a5f"),
        ("PCR[4]", "sha256  d094 6a2b fc31 98e7  \u2026  2b60"),
        ("PCR[7]", "sha256  11ef 90ba 5d47 6c22  \u2026  e803"),
    ]
    for i, (k, v) in enumerate(pcr):
        t.put(rx0 + 2, 3 + i, k, fg=CY, bg=PN)
        t.put(rx0 + 10, 3 + i, v, fg=DIM, bg=PN)

    t.hline(rx0 + 2, rx1 - 2, 8, fg=P["faint"], bg=PN)
    sec(9, "SECURE BOOT")
    kv(10, "state   ", "enabled", P["bgreen"])
    kv(11, "db      ", "MS-2023, Canonical-2024")
    kv(12, "shim    ", "15.8-0ubuntu1")
    kv(13, "policy  ", "lockdown=integrity", YE)

    sec(15, "SEALING")
    kv(16, "mechanism", "TPM2 NV idx 0x1500017")
    kv(17, "pcr policy", "0,2,4,7")
    kv(18, "luks2   ", "argon2id / 2 GiB")
    kv(19, "key escrow", "NONE", P["bred"])

    t.hline(rx0 + 2, rx1 - 2, 21, fg=P["faint"], bg=PN)
    t.put(rx0 + 2, 22, "seal latency (ms)", fg=DIM, bg=PN)
    vals = [1.2, 1.5, 0.9, 1.8, 1.1, 2.1, 1.3, 1.0, 1.6, 1.2, 0.8, 1.4,
            1.9, 1.1, 1.3, 1.5, 1.0, 1.7, 1.2, 0.9, 1.4, 1.6, 1.1, 1.3]
    t.braille(rx0 + 2, 23, vals, fg=CY, bg=PN, vmax=2.4)
    t.braille(rx0 + 2, 24, vals[::-1], fg=P["blue"], bg=PN, vmax=2.4)
    t.put(rx0 + 2, 26, "min", fg=DIM, bg=PN)
    t.put(rx0 + 6, 26, "0.8", fg=FG, bg=PN)
    t.put(rx0 + 11, 26, "avg", fg=DIM, bg=PN)
    t.put(rx0 + 15, 26, "1.4", fg=FG, bg=PN)
    t.put(rx0 + 20, 26, "max", fg=DIM, bg=PN)
    t.put(rx0 + 24, 26, "2.1", fg=FG, bg=PN)

    t.put(rx0 + 2, 28, "attestation chain", fg=BL, bg=PN, bold=True)
    chain = [
        "BMC AST2600 root-of-trust",
        "\u2514\u2500 CPU microcode 0x2b000603",
        "   \u2514\u2500 shim 15.8  [signed]",
        "      \u2514\u2500 GRUB 2.12  [signed]",
        "         \u2514\u2500 vmlinuz-6.11.0-24  [signed]",
        "            \u2514\u2500 world-core.service  [measured]",
    ]
    for i, s in enumerate(chain):
        t.put(rx0 + 2, 29 + i, s, fg=FG if i in (0, 5) else DIM, bg=PN)

    # ===================== BOTTOM PANEL : thermal / power =================
    bx0, by0, bx1, by1 = 1, 47, 190, 51
    t.fill(bx0 + 1, by0 + 1, bx1 - 1, by1 - 1, bg=PN)
    t.box(bx0, by0, bx1, by1, fg=P["faint"], bg=PN,
          title="thermal / power :: site theta", tfg=WH)

    def gauge(x, row, label, value, frac, unit, color, width=18):
        t.put(x, row, label, fg=DIM, bg=PN)
        t.bar(x + len(label) + 1, row, width, frac, fg=color, bg=PN)
        t.put(x + len(label) + 2 + width, row, f"{value}{unit}", fg=FG, bg=PN)

    gauge(3, 48, "GPU-in", "22.4", 0.24, "\u00b0C", P["bcyan"])
    gauge(40, 48, "GPU-out", "34.1", 0.41, "\u00b0C", P["borange"])
    t.put(75, 48, "coolant", fg=DIM, bg=PN)
    t.put(83, 48, "14.2 L/s", fg=FG, bg=PN)
    t.put(94, 48, "fan", fg=DIM, bg=PN)
    t.put(98, 48, "6,120 RPM", fg=FG, bg=PN)
    t.put(110, 48, "TDP headroom", fg=DIM, bg=PN)
    t.put(123, 48, "31.6 %", fg=YE, bg=PN)

    gauge(3, 49, "PDU-A", "0.94", 0.47, "MW", P["green"])
    gauge(40, 49, "PDU-B", "0.94", 0.47, "MW", P["green"])
    t.put(75, 49, "PUE", fg=DIM, bg=PN)
    t.put(79, 49, "1.20", fg=FG, bg=PN)
    t.put(94, 49, "geo", fg=DIM, bg=PN)
    t.put(98, 49, "$41/MWh", fg=FG, bg=PN)
    t.put(110, 49, "water", fg=DIM, bg=PN)
    t.put(116, 49, "9.4 \u00b0C", fg=CY, bg=PN)

    t.put(3, 50, "journal:", fg=P["magenta"], bg=PN)
    t.put(12, 50, "plastic step=128  loss=1.8421  meta-\u03b7=3.00e-4  gnorm=0.913  sync fast/slow/ultra ok",
          fg=DIM, bg=PN)

    # ===================== footer =========================================
    t.fill(0, ROWS - 1, COLS - 1, ROWS - 1, bg=P["panel2"])
    t.put(1, ROWS - 1, " world-core.service ", fg=(0, 0, 0), bg=P["green"], bold=True)
    t.put(23, ROWS - 1, "active (running)", fg=P["bgreen"], bg=P["panel2"])
    t.put(41, ROWS - 1, "| plastic=ENABLED", fg=YE, bg=P["panel2"])
    t.put(60, ROWS - 1, "| 864 x B300", fg=FG, bg=P["panel2"])
    t.put(74, ROWS - 1, "| 0.2 / 78.1 / 407.0 Hz", fg=CY, bg=P["panel2"])
    t.put(101, ROWS - 1, "| CAPEX $85M  TCO $104M", fg=DIM, bg=P["panel2"])
    t.put(COLS - 22, ROWS - 1, "14-month basis", fg=DIM, bg=P["panel2"])
    return t


def main():
    raw = "--raw" in sys.argv
    atlas = build_atlas("mono")
    atlas_b = build_atlas("monob")
    t = build_frame()
    frame = t.render(atlas, atlas_b)
    post = Post(seed=7)
    out = post.compose(frame, t=2.9, gain=1.02, bloom=0.42)
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    path = os.path.join(ROOT, "out", "style_protection.png")
    Image.fromarray(out, "RGB").save(path)
    Image.fromarray(frame.astype(np.uint8), "RGB").save(
        os.path.join(ROOT, "out", "style_protection_raw.png"))
    print("wrote", path, "grid", COLS, "x", ROWS)


if __name__ == "__main__":
    main()
