"""PROJECT: WORLD - interactive preview viewer (OpenGL path).

    python tools/viewer.py [--start 88] [--scale 0.75]
    python tools/viewer.py --shot 100.2

Keys:
    space        play / pause
    left/right   step +-1 frame  (shift: +-1 s)
    up/down      +-5 s
    home/end     jump to start / end
    a / b        set loop start / end at current time
    h            help
    q / esc      quit
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from worldcore import W, H, FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--shot", type=float, default=None,
                    help="render one frame to out/gl_<t>.png and exit")
    args = ap.parse_args()

    import glrender
    if args.shot is not None:
        glrender.preview([args.shot])
        return

    import pyglet
    from pyglet.window import key

    r = glrender.get_renderer()
    win = r.win
    win.set_size(int(W * args.scale), int(H * args.scale))
    win.set_visible(True)
    dur = float(r.dur)

    st = {"t": max(0.0, min(dur, args.start)), "playing": True,
          "a": None, "b": None}

    def caption():
        s = (f"PROJECT: WORLD  t={st['t']:6.2f}s  "
             f"{'play' if st['playing'] else 'pause'}")
        if st["a"] is not None and st["b"] is not None:
            s += f"  loop {st['a']:.2f} - {st['b']:.2f}"
        win.set_caption(s)

    def render(t):
        st["t"] = max(0.0, min(dur, t))
        caption()

    @win.event
    def on_draw():
        r.draw(st["t"])

    def tick(dt):
        if not st["playing"]:
            return
        t = st["t"] + dt
        if st["a"] is not None and st["b"] is not None and t >= st["b"]:
            t = st["a"]
        if t >= dur:
            t = 0.0
        render(t)

    @win.event
    def on_key_press(symbol, modifiers):
        if symbol in (key.Q, key.ESCAPE):
            win.close()
            return
        if symbol == key.SPACE:
            st["playing"] = not st["playing"]
        elif symbol in (key.LEFT, key.RIGHT):
            st["playing"] = False
            step = 1.0 if (modifiers & key.MOD_SHIFT) else 1.0 / FPS
            render(st["t"] + (step if symbol == key.RIGHT else -step))
        elif symbol in (key.UP, key.DOWN):
            st["playing"] = False
            render(st["t"] + (5.0 if symbol == key.UP else -5.0))
        elif symbol == key.HOME:
            st["playing"] = False
            render(0.0)
        elif symbol == key.END:
            st["playing"] = False
            render(dur)
        elif symbol == key.A:
            st["a"] = st["t"]
        elif symbol == key.B:
            st["b"] = st["t"]
        elif symbol == key.H:
            print(__doc__)
        caption()

    caption()
    pyglet.clock.schedule_interval(tick, 1.0 / 30.0)
    pyglet.app.run()


if __name__ == "__main__":
    main()
