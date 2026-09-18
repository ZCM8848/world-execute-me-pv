"""PROJECT: WORLD - render driver.

    python tools/render.py --preview 1 4 11 20 28
    python tools/render.py --range 0 30 out/slice_boot_30s.mp4
    python tools/render.py --video out/world_full.mp4

Add --gl for the OpenGL GPU pipeline (moderngl + NVENC, ~4x faster);
the CPU multiprocessing path remains the default and the fallback.
"""
import argparse
import os
import subprocess
import sys
import time
from collections import deque

import multiprocessing as mp

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from worldcore import W, H, FPS, ROOT, AUDIO, Post, clamp, seg, smoothstep, chorus
from tui import Tui, build_atlas
import scenes

DATA = os.path.join(ROOT, "data")
REACT = os.path.join(DATA, "react.npz")
BEATS = os.path.join(DATA, "beats.npz")


class Ctx:
    pass


def build_ctx():
    ctx = Ctx()
    ctx.atlas = build_atlas("mono")
    ctx.atlas_b = build_atlas("monob")
    ctx.post = Post(seed=7)
    d = np.load(REACT)
    ctx.env = d["env"].astype(np.float32)
    ctx.dur = float(d["dur"])
    b = np.load(BEATS)
    ctx.beats = b["beats"]
    ctx.period = float(b["period"])
    return ctx


def env_at(ctx, t):
    i = int(clamp(t * FPS, 0, len(ctx.env) - 1))
    return float(ctx.env[i])


def beat_pulse(ctx, t):
    if len(ctx.beats) == 0:
        return 0.0
    j = int(np.searchsorted(ctx.beats, t))
    best = 0.0
    for k in (j - 1, j):
        if 0 <= k < len(ctx.beats):
            dt = t - ctx.beats[k]
            best = max(best, float(np.exp(-(dt / 0.085) ** 2)))
    return best


def render_frame(t, ctx):
    ui = Tui()
    flash, glitch, m = scenes.draw(ui, t)
    frame = ui.render(ctx.atlas, ctx.atlas_b)
    gain = 0.985 + 0.05 * env_at(ctx, t)
    gain *= smoothstep(seg(t, 0.0, 0.7))
    if t > ctx.dur - 1.8:
        gain *= 1 - smoothstep(seg(t, ctx.dur - 1.7, ctx.dur - 0.1))
    return ctx.post.compose(frame, t=t, gain=gain, bloom=0.42), m


X264_ARGS = ["-c:v", "libx264", "-preset", "medium", "-crf", "16"]
NVENC_ARGS = ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq",
              "-rc", "vbr", "-cq", "19", "-b:v", "0",
              "-spatial-aq", "1", "-temporal-aq", "1"]

_NVENC_OK = None


def probe_nvenc():
    global _NVENC_OK
    if _NVENC_OK is None:
        try:
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                 "-i", "color=c=black:s=64x64:d=0.1", "-c:v", "h264_nvenc",
                 "-f", "null", "-"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            _NVENC_OK = (r.returncode == 0)
        except Exception:
            _NVENC_OK = False
        print(f"encoder: {'h264_nvenc (GPU)' if _NVENC_OK else 'libx264 (CPU)'}", flush=True)
    return _NVENC_OK


def encoder_args(pref="auto"):
    if pref == "nvenc" or (pref == "auto" and probe_nvenc()):
        return list(NVENC_ARGS)
    return list(X264_ARGS)


def preview(ctx, times):
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    for t in times:
        t0 = time.time()
        arr, m = render_frame(float(t), ctx)
        path = os.path.join(ROOT, "out", f"frame_{float(t):06.2f}.png")
        Image.fromarray(arr, "RGB").save(path)
        print(f"t={t:6.2f}s  {time.time() - t0:5.2f}s  {path}")


def encode(ctx, t0, t1, out_path, enc="auto"):
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
        t = f / FPS
        arr, _ = render_frame(t, ctx)
        try:
            proc.stdin.write(arr.tobytes())
        except BrokenPipeError:
            break
        if (f - f0) % 30 == 0:
            done = f - f0 + 1
            el = time.time() - ts
            print(f"  frame {f}/{f1}  t={t:6.2f}s  {done / el:5.1f} fps  "
                  f"eta {(total - done) / (done / el) / 60:4.1f} min", flush=True)
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        print("ffmpeg error:\n", proc.stderr.read().decode(errors="ignore"))
        return False
    print(f"wrote {out_path} ({t1 - t0:.1f}s) in {(time.time() - ts) / 60:.1f} min")
    return True


# ---------------------------------------------------------------------------
# parallel renderer: pool of workers, single ordered encode (no A/V drift)
# ---------------------------------------------------------------------------
_WCTX = None


def _worker_init():
    global _WCTX
    _WCTX = build_ctx()


def _frame_task(f):
    arr, _ = render_frame(f / FPS, _WCTX)
    return arr.tobytes()


def encode_parallel(t0, t1, out_path, jobs, enc="auto"):
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
    total = f1 - f0
    ts = time.time()
    print(f"parallel render: {total} frames, {jobs} workers", flush=True)
    with mp.Pool(jobs, initializer=_worker_init) as pool:
        fifo = deque()
        nxt = f0
        max_inflight = jobs * 3
        written = 0
        while nxt < f1 or fifo:
            while nxt < f1 and len(fifo) < max_inflight:
                fifo.append((nxt, pool.apply_async(_frame_task, (nxt,))))
                nxt += 1
            idx, ar = fifo.popleft()
            data = ar.get()
            try:
                proc.stdin.write(data)
            except BrokenPipeError:
                break
            written += 1
            if written % 60 == 0:
                el = time.time() - ts
                print(f"  {written}/{total}  t={idx / FPS:6.2f}s  "
                      f"{written / el:5.1f} fps  eta {(total - written) / (written / el) / 60:4.1f} min",
                      flush=True)
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        print("ffmpeg error:\n", proc.stderr.read().decode(errors="ignore"))
        return False
    print(f"wrote {out_path} ({t1 - t0:.1f}s) in {(time.time() - ts) / 60:.1f} min")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", nargs="+", type=float)
    ap.add_argument("--range", nargs=2, type=float, metavar=("T0", "T1"))
    ap.add_argument("--video", type=str)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--jobs", type=int, default=max(4, min(16, (os.cpu_count() or 8) // 2)))
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--enc", choices=["auto", "nvenc", "x264"], default="auto")
    ap.add_argument("--gl", action="store_true", help="use the OpenGL GPU pipeline")
    args = ap.parse_args()

    if args.gl:
        import glrender
        if args.preview:
            glrender.preview(args.preview)
        elif args.range:
            glrender.encode(args.range[0], args.range[1],
                            args.out or os.path.join(ROOT, "out", "slice.mp4"), args.enc)
        elif args.video:
            d = np.load(REACT)
            glrender.encode(0.0, float(d["dur"]), args.video, args.enc)
        else:
            ap.print_help()
    elif args.preview:
        ctx = build_ctx()
        print(f"loaded env={len(ctx.env)} beats={len(ctx.beats)} dur={ctx.dur:.1f}s")
        preview(ctx, args.preview)
    elif args.range and not args.serial:
        out = args.out or os.path.join(ROOT, "out", "slice.mp4")
        encode_parallel(args.range[0], args.range[1], out, args.jobs, args.enc)
    elif args.video and not args.serial:
        # need duration: quick load
        d = np.load(REACT)
        encode_parallel(0.0, float(d["dur"]), args.video, args.jobs, args.enc)
    else:
        ctx = build_ctx()
        print(f"loaded env={len(ctx.env)} beats={len(ctx.beats)} dur={ctx.dur:.1f}s")
        if args.range:
            encode(ctx, args.range[0], args.range[1],
                   args.out or os.path.join(ROOT, "out", "slice.mp4"), args.enc)
        elif args.video:
            encode(ctx, 0.0, ctx.dur, args.video, args.enc)
        else:
            ap.print_help()
