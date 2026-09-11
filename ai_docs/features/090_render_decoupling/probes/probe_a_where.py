# Run: cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/probe_a_where.py
"""Config A, instrumented: which call inside the light frame absorbs the stall?

Splits the light loop into poll / render-issue / swap and times each, so the stall can be
attributed to the GPU queue (swap or render) rather than to CPU starvation.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import glfw
import moderngl
from gpu_clear import wait_for_idle
from common import FULLSCREEN_VERT, TRIVIAL_FRAG, fmt, fullscreen_quad, make_window, now

HERE = Path(__file__).parent


def measure(duration: float, label: str) -> None:
    win = make_window(640, 360, "LIGHT", visible=True)
    glfw.set_window_pos(win, 1400, 60)
    glfw.make_context_current(win)
    glfw.swap_interval(0)
    ctx = moderngl.create_context()
    prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=TRIVIAL_FRAG)
    vao = fullscreen_quad(ctx, prog)
    t_poll: list[float] = []
    t_draw: list[float] = []
    t_swap: list[float] = []
    t_end = now() + duration
    frame = 0
    while now() < t_end:
        a = now()
        glfw.poll_events()
        b = now()
        ctx.screen.use()
        prog["u_time"].value = frame * 0.01
        vao.render()
        c = now()
        glfw.swap_buffers(win)
        d = now()
        t_poll.append((b - a) * 1000.0)
        t_draw.append((c - b) * 1000.0)
        t_swap.append((d - c) * 1000.0)
        frame += 1
    print("  " + fmt(f"{label} poll", t_poll[2:]), flush=True)
    print("  " + fmt(f"{label} draw", t_draw[2:]), flush=True)
    print("  " + fmt(f"{label} swap", t_swap[2:]), flush=True)
    glfw.destroy_window(win)


def main() -> None:
    wait_for_idle()
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    print("--- heavy idle ---", flush=True)
    measure(4.0, "idle")

    heavy = subprocess.Popen(
        [sys.executable, str(HERE / "probe_a_heavy_proc.py"), "9"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(os.environ),
    )
    assert heavy.stdout is not None
    print(heavy.stdout.readline().strip(), flush=True)
    time.sleep(1.0)
    print("--- heavy running ---", flush=True)
    measure(4.0, "running")
    heavy.wait()
    glfw.terminate()


main()
