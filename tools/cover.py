"""PROJECT: WORLD - Bilibili cover generator.

Renders a base frame from the PV (default: the EXECUTION countdown), then
composites a 16:9 title card in the film's own terminal palette.

    python tools/cover.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from worldcore import W, H, ROOT, P, font

OUT = os.path.join(ROOT, "out")
BASE_T = 156.0
CJK = r"C:\Windows\Fonts\msyh.ttc"
CJK_B = r"C:\Windows\Fonts\msyhbd.ttc"


def cjk(size, bold=False):
    return ImageFont.truetype(CJK_B if bold else CJK, size)


def base_frame(t=BASE_T):
    from render import build_ctx, render_frame
    ctx = build_ctx()
    arr, m = render_frame(t, ctx)
    return Image.fromarray(np.asarray(arr, np.uint8), "RGB")


def rgb(c):
    return tuple(c)


def build():
    a = np.asarray(base_frame(), np.float32)
    y0 = 690
    alpha = np.zeros(H, np.float32)
    alpha[y0:y0 + 150] = np.linspace(0.0, 0.93, 150) ** 0.9
    alpha[y0 + 150:] = 0.93
    a[:y0] *= 0.95
    a *= (1.0 - alpha)[:, None, None]
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)

    x = 116
    d.line([(x, 712), (W - x, 712)], fill=rgb(P["bgreen"]), width=3)
    d.text((x, 726), "[ PROJECT : WORLD ]", font=font("monob", 32), fill=rgb(P["bgreen"]))

    title = "world.execute(me);"
    ft = font("monob", 148)
    d.text((x - 2, 766), title, font=ft, fill=rgb(P["bwhite"]))
    tw = d.textlength(title, font=ft)
    d.rectangle([x + tw + 12, 766 + 18, x + tw + 12 + 60, 766 + 138], fill=rgb(P["bgreen"]))

    d.text((x, 962), "Mili  \u00b7  \u540c\u4eba PV  \u00b7  world.execute(me);",
           font=cjk(46), fill=rgb(P["bcyan"]))
    d.text((x, 1026), "$ tail -f /var/log/world-core.log", font=font("mono", 34), fill=rgb(P["dim"]))
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    img = build()
    img.save(os.path.join(OUT, "cover_1920x1080.png"))
    img.resize((1146, 717), Image.LANCZOS).save(os.path.join(OUT, "cover_1146x717.png"))
    img.resize((320, 180), Image.LANCZOS).save(os.path.join(OUT, "cover_thumb.png"))
    print(os.path.join(OUT, "cover_1920x1080.png"))
    print(os.path.join(OUT, "cover_1146x717.png"))
    print(os.path.join(OUT, "cover_thumb.png"))


if __name__ == "__main__":
    main()
