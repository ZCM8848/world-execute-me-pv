"""PROJECT: WORLD - Phase 1c: audio beat grid.

The bundled MIDI is a legato sax transcription that does not lock to the
rendered audio, so timing is derived from the audio itself: a comb search
over tempo + phase on the 100 Hz onset envelope.

    python tools/beats.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worldcore import ROOT

REACT = os.path.join(ROOT, "data", "react.npz")
OUT = os.path.join(ROOT, "data", "beats.json")
NPZ = os.path.join(ROOT, "data", "beats.npz")


def dilate(a, k):
    out = a.copy()
    for s in range(1, k + 1):
        out = np.maximum(out, np.roll(a, s))
        out = np.maximum(out, np.roll(a, -s))
    return out


def main():
    d = np.load(REACT)
    onset = d["onset_hi"].astype(np.float64)
    rate = float(d["hi_rate"])
    bpm0 = float(d["bpm"])
    onset = onset / (onset.max() + 1e-9)
    dil = dilate(onset, 3)
    n = len(onset)

    best = (-1.0, bpm0, 0.0)
    for bpm in np.arange(bpm0 - 3.0, bpm0 + 3.0, 0.02):
        period = 60.0 * rate / bpm
        for p in np.arange(0.0, period, 0.5):
            idx = np.round(np.arange(p, n, period)).astype(int)
            idx = idx[idx < n]
            if len(idx) < 50:
                continue
            s = float(dil[idx].mean())
            if s > best[0]:
                best = (s, bpm, p)
    score, bpm, phase = best
    period = 60.0 * rate / bpm
    beats = np.arange(phase, n, period) / rate
    beats = beats[beats < len(onset) / rate]
    strength = dil[np.clip(np.round(beats * rate).astype(int), 0, n - 1)]

    # downbeats: 4/4, phase chosen by max summed strength
    off = int(np.argmax([strength[i::4].sum() for i in range(4)]))
    down = beats[off::4]

    print(f"bpm={bpm:.2f} period={period:.2f} samples phase={phase:.2f} "
          f"score={score:.4f} beats={len(beats)} downbeats={len(down)}")
    print("first beats:", np.round(beats[:8], 3).tolist())
    print("downbeats near verse:", np.round(down[(down > 27) & (down < 34)], 3).tolist())

    np.savez_compressed(NPZ, beats=beats, downbeats=down, strength=strength,
                        bpm=bpm, period=period, phase=phase, rate=rate)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"bpm": float(bpm), "period": float(period), "phase": float(phase),
                   "score": float(score), "count": int(len(beats)),
                   "downbeat_offset": int(off)}, f, indent=1)
    print("wrote", NPZ, OUT)


if __name__ == "__main__":
    main()
