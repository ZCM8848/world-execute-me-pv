# PROJECT: WORLD

A terminal-realistic TUI **PV engine** that renders Mili's *world.execute(me);*
as a 1920×1080 / 30 fps character-cell animated video, driven by audio
reactivity, a beat grid, and an LRC lyric timeline.

Every frame is a grid of `(glyph, fg, bg, bold)` cells drawn through a
precomputed glyph atlas and rasterised in a handful of vectorised numpy ops,
then passed through a light CRT post pass. The result is a screen that looks
like a real Ubuntu terminal running a cluster boot — power-on, `dmesg`,
`nvtop`, rack telemetry, scrolling journals — with song lyrics appearing as
highlighted rows inside the stream.

## Pipeline

```
world_execute_me.mp3  ─┐
Mili.mid               ├──▶ prep_audio.py ──▶ react.npz, timeline.json
world.execute(me).lrc  ─┘            │
                                    ▼
                              beats.py ──▶ beats.npz
                                    │
                                    ▼
   render.py  ──▶  Tui framebuffer  ──▶  ffmpeg  ──▶  .mp4
   (CPU mp pool  |  --gl OpenGL+NVENC ~4× faster)
```

| Stage | Script | Output |
|-------|--------|--------|
| 1a audio prep | `tools/prep_audio.py` | `data/audio.wav`, `data/react.npz`, `data/timeline.json` |
| 1c beat grid | `tools/beats.py` | `data/beats.json`, `data/beats.npz` |
| render | `tools/render.py` | `out/*.mp4` |

The bundled `data/*.npz` and `data/timeline.json` are pre-generated, so you can
skip straight to `render.py`. Regenerate from source with `prep_audio.py
--force` then `beats.py` (requires `ffmpeg` in `PATH`).

## Requirements

- Python 3.9+
- `numpy`, `Pillow`
- `moderngl` (only for the `--gl` GPU pipeline)
- `ffmpeg` in `PATH` (decode + encode; NVENC for `--enc nvenc`)

## Usage

```bash
# preview individual frames (seconds into the song)
python tools/render.py --preview 1 4 11 20 28

# render a slice
python tools/render.py --range 0 30 out/slice_boot_30s.mp4

# render the full video
python tools/render.py --video out/world_full.mp4

# OpenGL GPU pipeline (~4× faster); encoder selection
python tools/render.py --gl --enc nvenc --video out/world_full.mp4
```

Flags: `--preview`, `--range T0 T1`, `--video PATH`, `--out PATH`,
`--jobs N` (CPU worker count), `--serial` (single-process fallback),
`--enc auto|nvenc|x264`, `--gl` (OpenGL pipeline).

## Project structure

```
tools/
  worldcore.py   canvas (1920×1080), palette, fonts, CRT post-processing
  tui.py         character-cell framebuffer + glyph atlas (vectorised numpy)
  scenes.py      scene composition: tmux-style panes, lyric journal
  cluster.py     rack / GPU / sensor telemetry model
  logs.py        simulated log streams (dmesg, worldmon, ibmon, …)
  art.py         ASCII art: rack elevation, globe, prism, math imagery
  acts.py        act/emblem beats
  decal.py       reveal decals (beam → prism → spectrum → mark)
  align.py       column / pane alignment helpers
  landmask.py    Natural Earth land mask rasteriser
  glrender.py    OpenGL GPU render path (moderngl + NVENC)
  prep_audio.py  Phase 1a: decode + reactive signal extraction
  beats.py       Phase 1c: tempo/phase comb search beat grid
  render.py      render driver (preview / range / video)
assets/fonts/    Ubuntu Mono + JetBrains Mono
data/            pre-generated react.npz, beats.npz, timeline.json, land masks
```

## Notes

- Output is written under `out/` (git-ignored).
- The 0:00–0:30 boot slice is the primary storyboarded act; the full song is
  ~3:32.
- `tools/style_sample.py` and `tools/cover.py` are standalone dev utilities.
