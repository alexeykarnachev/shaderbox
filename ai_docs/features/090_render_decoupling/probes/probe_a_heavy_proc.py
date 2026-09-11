# Run (usually launched by probe_a_run.py, not by hand):
#   cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/probe_a_heavy_proc.py <seconds>
"""Config A heavy side: a separate process with its own glfw window rendering ~100 ms/frame."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import glfw
import moderngl
from common import (
    FULLSCREEN_VERT,
    HEAVY_FRAG,
    HEAVY_H,
    HEAVY_ITERS,
    HEAVY_W,
    fmt,
    fullscreen_quad,
    make_window,
    now,
)


def main() -> None:
    duration = float(sys.argv[1])
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    win = make_window(HEAVY_W, HEAVY_H, "HEAVY", visible=True)
    glfw.set_window_pos(win, 40, 60)
    glfw.make_context_current(win)
    glfw.swap_interval(0)  # no vsync: we want the GPU saturated, not paced
    ctx = moderngl.create_context()
    prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=HEAVY_FRAG)
    prog["u_iters"].value = HEAVY_ITERS
    vao = fullscreen_quad(ctx, prog)

    print(f"[heavy] pid={__import__('os').getpid()} iters={HEAVY_ITERS} ready", flush=True)
    periods: list[float] = []
    t_end = now() + duration
    prev = now()
    frame = 0
    while now() < t_end:
        glfw.poll_events()
        ctx.screen.use()
        prog["u_time"].value = frame * 0.01
        vao.render()
        glfw.swap_buffers(win)
        t = now()
        periods.append((t - prev) * 1000.0)
        prev = t
        frame += 1
    print("[heavy] " + fmt("period", periods[2:]), flush=True)
    glfw.destroy_window(win)
    glfw.terminate()


main()
