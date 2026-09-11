# Run: cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/probe_d_single.py
"""Config D baseline: today's shape -- one process, one context, heavy then light in one loop."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import glfw
import moderngl
from gpu_clear import wait_for_idle
from common import (
    FULLSCREEN_VERT,
    HEAVY_FRAG,
    HEAVY_H,
    HEAVY_ITERS,
    HEAVY_W,
    SAMPLER_FRAG,
    fmt,
    fullscreen_quad,
    make_window,
    now,
)

DUR = 5.0


def run(heavy_on: bool, label: str) -> None:
    win = make_window(HEAVY_W, HEAVY_H, "SINGLE", visible=True)
    glfw.set_window_pos(win, 100, 60)
    glfw.make_context_current(win)
    glfw.swap_interval(0)
    ctx = moderngl.create_context()

    heavy_prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=HEAVY_FRAG)
    heavy_prog["u_iters"].value = HEAVY_ITERS
    heavy_vao = fullscreen_quad(ctx, heavy_prog)
    tex = ctx.texture((HEAVY_W, HEAVY_H), 4, dtype="f1")
    doc_fbo = ctx.framebuffer(color_attachments=[tex])

    ui_prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=SAMPLER_FRAG)
    ui_prog["u_tex"].value = 0
    ui_vao = fullscreen_quad(ctx, ui_prog)

    periods: list[float] = []
    t_end = now() + DUR
    prev = now()
    frame = 0
    while now() < t_end:
        glfw.poll_events()
        if heavy_on:
            doc_fbo.use()
            heavy_prog["u_time"].value = frame * 0.01
            heavy_vao.render()
        ctx.screen.use()
        tex.use(0)
        ui_prog["u_time"].value = frame * 0.01
        ui_vao.render()
        glfw.swap_buffers(win)
        t = now()
        periods.append((t - prev) * 1000.0)
        prev = t
        frame += 1
    print(fmt(label, periods[2:]), flush=True)
    glfw.destroy_window(win)


def main() -> None:
    wait_for_idle()
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    for trial in (1, 2):
        time.sleep(1.5)  # let the previous phase's GPU queue drain
        run(False, f"D idle    t{trial}")
        time.sleep(1.5)
        run(True, f"D running t{trial}")
    glfw.terminate()


main()
