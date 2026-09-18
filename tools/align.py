"""PROJECT: WORLD - Phase 1b: MIDI <-> audio alignment.

The MIDI is a 130 BPM transcription; the rendered audio measures ~130.4 BPM.
We search a tempo scale + offset that maximises the audio onset energy under
the MIDI note onsets, then persist the transform and a beat grid.

    python tools/align.py
"""
import json
import os
import sys

import numpy as np
import mido

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worldcore import ROOT

MID = os.path.join(ROOT, "World_execute(me)_-_Mili.mid")
REACT = os.path.join(ROOT, "data", "react.npz")
ALIGN = os.path.join(ROOT, "data", "midi_alignment.json")


def build_tempo_map(mid):
    tempos = [(0, 500000)]
    tick = 0
    for msg in mid.tracks[0]:
        tick += msg.time
        if msg.type == "set_tempo":
            tempos.append((tick, msg.tempo))
    return tempos


def ticks_to_sec(ticks, tempos, tpb):
    sec, lt, ltp = 0.0, 0, 500000
    for (tk, tp) in tempos:
        if tk <= ticks:
            sec += mido.tick2second(tk - lt, tpb, ltp)
            lt, ltp = tk, tp
        else:
            break
    sec += mido.tick2second(ticks - lt, tpb, ltp)
    return sec


def midi_onsets(path):
    mid = mido.MidiFile(path)
    tempos = build_tempo_map(mid)
    tpb = mid.ticks_per_beat
    onsets = []
    for tr in mid.tracks:
        tick = 0
        for msg in tr:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                onsets.append(ticks_to_sec(tick, tempos, tpb))
    return np.array(sorted(set(round(o, 4) for o in onsets))), mid


def score(audio, rate, times, offset):
    tt = times + offset
    idx = np.round(tt * rate).astype(int)
    ok = (idx >= 0) & (idx < len(audio))
    if ok.sum() < 8:
        return -1.0, 0
    return float(audio[idx[ok]].mean()), int(ok.sum())


def main():
    d = np.load(REACT)
    audio = d["onset_hi"].astype(np.float64)
    rate = float(d["hi_rate"])
    audio = audio / (audio.max() + 1e-9)
    # emphasise peaks so alignment is not smeared by sustain energy
    audio = np.maximum(audio - 0.20, 0.0)

    onsets, mid = midi_onsets(MID)
    print(f"midi onsets={len(onsets)}  span={onsets[-1]:.2f}s  audio={len(audio)/rate:.2f}s")

    scales = np.arange(0.90, 1.061, 0.002)
    offs = np.arange(-8.0, 8.0, 0.01)
    results = []
    for s in scales:
        t = onsets * s
        # coarse
        best = (-1.0, 0.0)
        for o in offs:
            v, n = score(audio, rate, t, o)
            if v > best[0]:
                best = (v, o)
        # refine
        for o in np.arange(best[1] - 0.01, best[1] + 0.0101, 0.001):
            v, n = score(audio, rate, t, o)
            if v > best[0]:
                best = (v, o)
        results.append((s, best[1], best[0]))
    results.sort(key=lambda r: -r[2])
    print("top candidates (scale, offset_s, onset_energy):")
    for s, o, v in results[:8]:
        print(f"  scale={s:.3f}  offset={o:+.3f}s  score={v:.4f}")

    bs, bo, bv = results[0]
    # null baseline: random offsets
    rng = np.random.default_rng(0)
    null = [score(audio, rate, onsets * bs, rng.uniform(-8, 8))[0] for _ in range(200)]
    print(f"null mean={np.mean(null):.4f} max={np.max(null):.4f}  best={bv:.4f} "
          f"({bv/(np.mean(null)+1e-9):.1f}x)")

    beats = (onsets * bs + bo)
    beats = beats[(beats >= 0) & (beats < len(audio) / rate)]
    nul_mu, nul_mx = float(np.mean(null)), float(np.max(null))
    reliable = (bv > nul_mx) and ((bv - nul_mu) > 3.0 * float(np.std(null))) and (abs(bo) < 6.0)
    out = {"reliable": bool(reliable), "scale": float(bs), "offset": float(bo),
           "score": float(bv), "null_mean": nul_mu, "null_max": nul_mx,
           "midi_span": float(onsets[-1]), "audio_dur": len(audio) / rate,
           "note": "audio beat grid in beats.json is authoritative"}
    with open(ALIGN, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(ROOT, "data", "midi_notes.npz"),
                        onsets=onsets, transformed=onsets * bs + bo)
    print("wrote", ALIGN, out)


if __name__ == "__main__":
    main()
