"""PROJECT: WORLD - Phase 1a: audio preparation.

Decodes the source MP3 once (ffmpeg -> data/audio.wav), derives frame-rate
reactive signals (RMS envelope, spectral flux, log-band spectrogram), a
high-resolution onset envelope, a tempo estimate, and parses the LRC timeline.

    python tools/prep_audio.py [--force]
"""
import json
import os
import re
import subprocess
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worldcore import SR, FPS, ROOT, AUDIO

MP3 = os.path.join(ROOT, "world_execute_me.mp3")
LRC = os.path.join(ROOT, "world.execute(me)-timeline.lrc")
DATA = os.path.join(ROOT, "data")
REACT = os.path.join(DATA, "react.npz")
TIMELINE = os.path.join(DATA, "timeline.json")
HI_RATE = 100  # onset envelope resolution (Hz) for alignment


def decode(force=False):
    os.makedirs(DATA, exist_ok=True)
    if os.path.exists(AUDIO) and not force:
        print("reuse", AUDIO)
        return
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", MP3,
           "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", AUDIO]
    subprocess.run(cmd, check=True)
    print("decoded", MP3, "->", AUDIO)


def load_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    data = np.frombuffer(raw, dtype="<i2").reshape(-1, ch).astype(np.float32) / 32768.0
    return data, sr


def frame_env(mono, sr, fps):
    hop = sr / fps
    n = int(len(mono) / hop) + 1
    win = int(0.05 * sr)
    h = np.hanning(win).astype(np.float32)
    env = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = int(i * hop)
        seg = mono[a:a + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        env[i] = np.sqrt(np.mean((seg * h) ** 2))
    return env


def spectral_feats(mono, sr, rate, nbins):
    hop = int(sr / rate)
    nfft = 2048
    win = np.hanning(nfft).astype(np.float32)
    n = int(len(mono) / hop) + 1
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    edges = np.logspace(np.log10(50), np.log10(14000), nbins + 1)
    spec = np.zeros((n, nbins), dtype=np.float32)
    flux = np.zeros(n, dtype=np.float32)
    prev = None
    x = np.concatenate([np.zeros(nfft // 2, np.float32), mono, np.zeros(nfft // 2, np.float32)])
    for i in range(n):
        a = i * hop
        seg = x[a:a + nfft]
        if len(seg) < nfft:
            seg = np.pad(seg, (0, nfft - len(seg)))
        mag = np.abs(np.fft.rfft(seg * win))
        if prev is not None:
            d = mag - prev
            flux[i] = np.maximum(d, 0.0).sum()
        prev = mag
        for k in range(nbins):
            m = (freqs >= edges[k]) & (freqs < edges[k + 1])
            if m.any():
                spec[i, k] = mag[m].mean()
    spec = np.log1p(spec * 30.0)
    spec /= (spec.max() + 1e-9)
    flux = flux - flux.min()
    flux /= (flux.max() + 1e-9)
    return flux, spec


def estimate_tempo(flux, rate):
    x = flux - flux.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    lo, hi = int(rate * 60 / 200), int(rate * 60 / 60)
    hi = min(hi, len(ac) - 1)
    seg = ac[lo:hi]
    if len(seg) < 3:
        return 0.0, 0.0
    lag = lo + int(np.argmax(seg))
    bpm = 60.0 * rate / lag
    return float(bpm), float(ac[lag] / (ac[0] + 1e-9))


def parse_lrc(path):
    meta, lines = {}, []
    rx = re.compile(r"\[(\d+):(\d+\.\d+)\](.*)")
    mx = re.compile(r"\[([A-Za-z_]+):(.*)\]")
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            m = rx.match(ln)
            if m:
                t = int(m.group(1)) * 60 + float(m.group(2))
                lines.append({"t": round(t, 3), "text": m.group(3)})
            else:
                m2 = mx.match(ln)
                if m2:
                    meta[m2.group(1)] = m2.group(2)
    return meta, lines


def main():
    force = "--force" in sys.argv
    decode(force)
    audio, sr = load_wav(AUDIO)
    mono = audio.mean(axis=1)
    dur = len(mono) / sr
    print(f"audio {dur:.3f}s  sr={sr}  ch={audio.shape[1]}")

    env = frame_env(mono, sr, FPS)
    flux, spec = spectral_feats(mono, sr, FPS, 48)
    onset_hi, _ = spectral_feats(mono, sr, HI_RATE, 24)
    bpm, conf = estimate_tempo(onset_hi, HI_RATE)
    print(f"tempo ~ {bpm:.2f} BPM (acf conf {conf:.3f})  frames={len(env)} spec={spec.shape}")

    np.savez_compressed(REACT, env=env, flux=flux, spec=spec,
                        onset_hi=onset_hi, hi_rate=HI_RATE,
                        dur=dur, sr=sr, fps=FPS, bpm=bpm)
    print("wrote", REACT)

    meta, lines = parse_lrc(LRC)
    with open(TIMELINE, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "lines": lines}, f, ensure_ascii=False, indent=1)
    print(f"wrote {TIMELINE}  ({len(lines)} lines, length {meta.get('length')})")


if __name__ == "__main__":
    main()
