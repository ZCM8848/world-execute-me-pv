"""PROJECT: WORLD - GPU render pipeline (OpenGL / moderngl).

Replaces the CPU rasteriser (tui.Tui.render) and CRT post (worldcore.Post) with
a small full-screen shader pipeline:

    cell textures (glyph id / fg / bg)  ->  scene FBO
                                        ->  quarter-res bloom (H+V blur)
                                        ->  compose pass (gain/scan/vig/grain)
                                        ->  readback -> ffmpeg

The character grid is still assembled on the CPU by tui.Tui (cheap, ~10k
cells); everything pixel-bound happens on the GPU.
"""
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyglet
pyglet.options["shadow_window"] = False

from worldcore import W, H, FPS, ROOT, AUDIO, clamp, seg, smoothstep
from tui import Tui, build_atlas, CHARSET

import scenes

DATA = os.path.join(ROOT, "data")
REACT = os.path.join(DATA, "react.npz")

VERT = """#version 330
in vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }
"""

SCENE_FRAG = """#version 330
uniform usampler2D t_glyph;
uniform sampler2D t_fg;
uniform sampler2D t_bg;
uniform sampler2D t_atlas;
uniform sampler2D t_atlasb;
uniform int ATLAS_COLS;
uniform vec2 ATLAS_DIM;
uniform vec2 CELL;
uniform vec2 GRID;
out vec4 frag;
void main() {
    vec2 fc = gl_FragCoord.xy;
    float rb = floor(fc.y / CELL.y);
    float cx = floor(fc.x / CELL.x);
    int rt = int(GRID.y - 1.0 - rb);
    ivec2 cell = ivec2(int(cx), rt);
    ivec2 mx = ivec2(GRID) - ivec2(1);
    cell = clamp(cell, ivec2(0), mx);
    uint gv = texelFetch(t_glyph, cell, 0).r;
    uint g = gv & 0x7FFFu;
    float bold = ((gv & 0x8000u) != 0u) ? 1.0 : 0.0;
    vec3 fgc = texelFetch(t_fg, cell, 0).rgb;
    vec3 bgc = texelFetch(t_bg, cell, 0).rgb;
    float lx = fc.x - cx * CELL.x;
    float ly = fc.y - rb * CELL.y;
    int gx = int(g) % ATLAS_COLS;
    int gy = int(g) / ATLAS_COLS;
    float aux = float(gx) * CELL.x + (lx - 0.5);
    float auy = float(gy) * CELL.y + (CELL.y - 0.5 - ly);
    vec2 auv = (vec2(aux, auy) + 0.5) / ATLAS_DIM;
    float m = mix(texture(t_atlas, auv).r, texture(t_atlasb, auv).r, bold);
    frag = vec4(mix(bgc, fgc, m), 1.0);
}
"""

DOWN_FRAG = """#version 330
uniform sampler2D t_src;
uniform vec2 QRES;
out vec4 frag;
void main() {
    vec2 uv = gl_FragCoord.xy / QRES;
    frag = vec4(texture(t_src, uv).rgb, 1.0);
}
"""

BLUR_FRAG = """#version 330
uniform sampler2D t_src;
uniform float w[17];
uniform vec2 dir;
out vec4 frag;
void main() {
    vec2 size = vec2(textureSize(t_src, 0));
    vec2 uv = gl_FragCoord.xy / size;
    vec3 s = vec3(0.0);
    for (int i = 0; i < 17; i++)
        s += w[i] * texture(t_src, uv + dir * float(i - 8)).rgb;
    frag = vec4(s, 1.0);
}
"""

COMPOSE_FRAG = """#version 330
uniform sampler2D t_scene;
uniform sampler2D t_bloom;
uniform sampler2D t_grain[8];
uniform int gi;
uniform float gain;
uniform float bloom;
uniform vec2 RES;
out vec4 frag;
void main() {
    vec2 fc = gl_FragCoord.xy;
    vec2 uv = fc / RES;
    vec3 c = texture(t_scene, uv).rgb;
    vec3 g = texture(t_bloom, uv).rgb;
    vec3 a = clamp(c + g * bloom, 0.0, 1.0) * gain;
    float rowt = RES.y - fc.y - 0.5;
    float scan = (mod(floor(rowt), 2.0) < 0.5) ? 0.88 : 1.0;
    a *= scan;
    vec2 p = (fc - RES * 0.5) / (RES * 0.5);
    float vig = clamp(1.10 - 0.36 * pow(length(p), 2.2), 0.34, 1.0);
    a *= vig;
    float gr = texelFetch(t_grain[gi], ivec2(int(fc.x), int(rowt)), 0).r;
    a += gr * (4.5 / 255.0);
    frag = vec4(clamp(a, 0.0, 1.0), 1.0);
}
"""


def _pack_atlas(atlas, cols=64):
    n, ch, cw = atlas.shape
    rows = (n + cols - 1) // cols
    tex = np.zeros((rows * ch, cols * cw), np.uint8)
    for i in range(n):
        gx, gy = i % cols, i // cols
        tex[gy * ch:(gy + 1) * ch, gx * cw:(gx + 1) * cw] = atlas[i]
    return tex, cols, (cols * cw, rows * ch)


class GLRenderer:
    def __init__(self):
        self.win = pyglet.window.Window(width=W, height=H, visible=False)
        import moderngl
        self.ctx = moderngl.create_context()
        ctx = self.ctx
        self.CH, self.CW = 20, 10
        self.COLS, self.ROWS = W // self.CW, H // self.CH

        a, self.acols, self.adim = _pack_atlas(build_atlas("mono"))
        b, _, _ = _pack_atlas(build_atlas("monob"))
        self.t_atlas = self._tex(self.adim, 1, "f1", a, nearest=True)
        self.t_atlasb = self._tex(self.adim, 1, "f1", b, nearest=True)

        self.t_glyph = ctx.texture((self.COLS, self.ROWS), 1, dtype="u2")
        self.t_fg = ctx.texture((self.COLS, self.ROWS), 3, dtype="f1")
        self.t_bg = ctx.texture((self.COLS, self.ROWS), 3, dtype="f1")

        self.verts = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], np.float32).tobytes())
        self.prog_scene = ctx.program(vertex_shader=VERT, fragment_shader=SCENE_FRAG)
        self.prog_down = ctx.program(vertex_shader=VERT, fragment_shader=DOWN_FRAG)
        self.prog_blur = ctx.program(vertex_shader=VERT, fragment_shader=BLUR_FRAG)
        self.prog_comp = ctx.program(vertex_shader=VERT, fragment_shader=COMPOSE_FRAG)
        self.vao_scene = ctx.vertex_array(self.prog_scene, [(self.verts, "2f", "a_pos")])
        self.vao_down = ctx.vertex_array(self.prog_down, [(self.verts, "2f", "a_pos")])
        self.vao_blur = ctx.vertex_array(self.prog_blur, [(self.verts, "2f", "a_pos")])
        self.vao_comp = ctx.vertex_array(self.prog_comp, [(self.verts, "2f", "a_pos")])

        qw, qh = max(1, W // 4), max(1, H // 4)
        self.qres = (qw, qh)
        self.fbo_scene = ctx.framebuffer([ctx.texture((W, H), 4, dtype="f1")])
        self.fbo_qa = ctx.framebuffer([ctx.texture((qw, qh), 4, dtype="f1")])
        self.fbo_qb = ctx.framebuffer([ctx.texture((qw, qh), 4, dtype="f1")])
        self.fbo_out = ctx.framebuffer([ctx.texture((W, H), 4, dtype="f1")])

        sigma = 2.4
        ks = np.arange(-8, 9, dtype=np.float32)
        wts = np.exp(-(ks ** 2) / (2 * sigma ** 2))
        wts /= wts.sum()
        self.blur_w = wts

        rng = np.random.default_rng(7)
        self.grains = [rng.standard_normal((H, W)).astype(np.float32) * 0.5 for _ in range(8)]
        self.t_grains = [self._tex((W, H), 1, "f4", g) for g in self.grains]
        self._gi = 0

        self.ui = Tui()
        d = np.load(REACT)
        self.env = d["env"].astype(np.float32)
        self.dur = float(d["dur"])

    def _tex(self, size, comp, dtype, data, nearest=False):
        t = self.ctx.texture(size, comp, dtype=dtype)
        if nearest:
            t.filter = (self.ctx.NEAREST, self.ctx.NEAREST)
        t.write(np.ascontiguousarray(data).tobytes())
        return t

    def _draw(self, vao, target):
        target.use()
        vao.render(mode=4, vertices=3)

    def gain_at(self, t):
        i = int(clamp(t * FPS, 0, len(self.env) - 1))
        gain = 0.985 + 0.05 * float(self.env[i])
        gain *= smoothstep(seg(t, 0.0, 0.7))
        if t > self.dur - 1.8:
            gain *= 1 - smoothstep(seg(t, self.dur - 1.7, self.dur - 0.1))
        return gain

    def _upload(self, ui):
        gv = (ui.ch.astype(np.uint16) & 0x7FFF) | (ui.bo.astype(np.uint16) << 15)
        self.t_glyph.write(np.ascontiguousarray(gv).tobytes())
        self.t_fg.write(np.ascontiguousarray(ui.fg).tobytes())
        self.t_bg.write(np.ascontiguousarray(ui.bg).tobytes())

    def render(self, t):
        ui = self.ui
        ui.clear()
        flash, glitch, m = scenes.draw(ui, t)
        self._upload(ui)
        return self._composite(t), m

    def _composite(self, t):
        ctx = self.ctx

        ps = self.prog_scene
        ps["t_glyph"] = 0
        ps["t_fg"] = 1
        ps["t_bg"] = 2
        ps["t_atlas"] = 3
        ps["t_atlasb"] = 4
        ps["ATLAS_COLS"] = self.acols
        ps["ATLAS_DIM"] = self.adim
        ps["CELL"] = (self.CW, self.CH)
        ps["GRID"] = (self.COLS, self.ROWS)
        self.t_glyph.use(0)
        self.t_fg.use(1)
        self.t_bg.use(2)
        self.t_atlas.use(3)
        self.t_atlasb.use(4)
        self._draw(self.vao_scene, self.fbo_scene)

        pd = self.prog_down
        pd["t_src"] = 0
        pd["QRES"] = self.qres
        self.fbo_scene.color_attachments[0].use(0)
        self._draw(self.vao_down, self.fbo_qa)

        pb = self.prog_blur
        pb["t_src"] = 0
        pb["w"] = self.blur_w
        qw, qh = self.qres
        pb["dir"] = (1.0 / qw, 0.0)
        self.fbo_qa.color_attachments[0].use(0)
        self._draw(self.vao_blur, self.fbo_qb)
        pb["dir"] = (0.0, 1.0 / qh)
        self.fbo_qb.color_attachments[0].use(0)
        self._draw(self.vao_blur, self.fbo_qa)

        pc = self.prog_comp
        pc["t_scene"] = 0
        pc["t_bloom"] = 1
        pc["t_grain"] = tuple(range(2, 10))
        pc["gi"] = self._gi
        pc["gain"] = self.gain_at(t)
        pc["bloom"] = 0.42
        pc["RES"] = (W, H)
        self.fbo_scene.color_attachments[0].use(0)
        self.fbo_qa.color_attachments[0].use(1)
        for i in range(8):
            self.t_grains[i].use(2 + i)
        self._draw(self.vao_comp, self.fbo_out)

        buf = self.fbo_out.read(components=3, dtype="f1")
        arr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)[::-1]
        self._gi = (self._gi + 1) % 8
        return np.ascontiguousarray(arr).tobytes()


_R = None


def get_renderer():
    global _R
    if _R is None:
        _R = GLRenderer()
    return _R


def preview(times):
    r = get_renderer()
    from PIL import Image
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    for t in times:
        t0 = time.time()
        data, _ = r.render(float(t))
        arr = np.frombuffer(data, np.uint8).reshape(H, W, 3)
        path = os.path.join(ROOT, "out", f"gl_{float(t):06.2f}.png")
        Image.fromarray(arr, "RGB").save(path)
        print(f"t={t:6.2f}s  {time.time() - t0:5.3f}s  {path}")


def encode(t0, t1, out_path, enc="auto"):
    from render import encoder_args
    r = get_renderer()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    f0, f1 = int(round(t0 * FPS)), int(round(t1 * FPS))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-ss", f"{t0:.3f}", "-i", AUDIO,
        *encoder_args(enc),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "320k", "-shortest", out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    ts = time.time()
    total = f1 - f0
    for f in range(f0, f1):
        data, _ = r.render(f / FPS)
        try:
            proc.stdin.write(data)
        except BrokenPipeError:
            break
        if (f - f0) % 30 == 0:
            done = f - f0 + 1
            el = time.time() - ts
            print(f"  frame {f}/{f1}  t={f / FPS:6.2f}s  {done / el:5.1f} fps  "
                  f"eta {(total - done) / (done / el) / 60:4.1f} min", flush=True)
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        print("ffmpeg error:\n", proc.stderr.read().decode(errors="ignore"))
        return False
    print(f"wrote {out_path} ({t1 - t0:.1f}s) in {(time.time() - ts) / 60:.1f} min")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", nargs="+", type=float)
    ap.add_argument("--range", nargs=2, type=float, metavar=("T0", "T1"))
    ap.add_argument("--video", type=str)
    ap.add_argument("--enc", default="auto")
    a = ap.parse_args()
    if a.preview:
        preview(a.preview)
    elif a.range:
        encode(a.range[0], a.range[1], a.video or os.path.join(ROOT, "out", "gl_slice.mp4"), a.enc)
    elif a.video:
        d = np.load(REACT)
        encode(0.0, float(d["dur"]), a.video, a.enc)
    else:
        ap.print_help()
