# Run: cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/probe_gil.py
"""Does the GL worker thread hold the GIL while the GPU works?

Main thread runs a pure-Python tick loop (no GL at all) and records inter-tick gaps. If the
worker holds the GIL across its GPU wait, gaps reach ~100 ms; if PyOpenGL releases it around
glClientWaitSync, gaps stay at the interpreter's switch interval.

Three workers compared, same ~100 ms of occupancy each:
  gl      -- render + glClientWaitSync (the config-B worker)
  glfin   -- render + moderngl Context.finish()
  pybusy  -- pure Python arithmetic (control: GIL contention with no GPU involved)
"""

import sys
import threading
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
from gpu_clear import wait_for_idle
from OpenGL import GL

DUR = 4.0


def gl_worker(share_win, stop: threading.Event, mode: str, out: list[float]) -> None:
    win = make_window(64, 64, "worker", visible=False, share=share_win)
    glfw.make_context_current(win)
    ctx = moderngl.create_context()
    prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=HEAVY_FRAG)
    prog["u_iters"].value = HEAVY_ITERS
    vao = fullscreen_quad(ctx, prog)
    tex = ctx.texture((HEAVY_W, HEAVY_H), 4, dtype="f1")
    fbo = ctx.framebuffer(color_attachments=[tex])
    k = 0
    while not stop.is_set():
        t0 = now()
        fbo.use()
        prog["u_time"].value = k * 0.01
        vao.render()
        if mode == "gl":
            fence = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
            GL.glFlush()
            GL.glClientWaitSync(fence, GL.GL_SYNC_FLUSH_COMMANDS_BIT, 1_000_000_000)
            GL.glDeleteSync(fence)
        else:
            ctx.finish()
        out.append((now() - t0) * 1000.0)
        k += 1
    glfw.destroy_window(win)


def py_worker(stop: threading.Event, out: list[float]) -> None:
    while not stop.is_set():
        t0 = now()
        x = 0.0
        while (now() - t0) < 0.100:
            for _ in range(1000):
                x = x * 1.000001 + 0.5
        out.append((now() - t0) * 1000.0)


def measure(mode: str, win) -> None:
    stop = threading.Event()
    wout: list[float] = []
    if mode == "pybusy":
        th = threading.Thread(target=py_worker, args=(stop, wout), daemon=True)
    else:
        th = threading.Thread(target=gl_worker, args=(win, stop, mode, wout), daemon=True)
    th.start()
    glfw.make_context_current(win)

    gaps: list[float] = []
    t_end = now() + DUR
    prev = now()
    acc = 0.0
    while now() < t_end:
        for _ in range(200):  # a little pure-Python work, NO GL in the main loop
            acc += 1.0
        t = now()
        gaps.append((t - prev) * 1000.0)
        prev = t
    stop.set()
    th.join(5.0)
    print(f"  worker[{mode}] frames={len(wout)} " + fmt("cost", wout[1:]), flush=True)
    print("  " + fmt(f"main tick gap [{mode}]", gaps[2:]), flush=True)


def main() -> None:
    print(f"sys.getswitchinterval() = {sys.getswitchinterval()} s", flush=True)
    wait_for_idle()
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    win = make_window(320, 240, "MAIN", visible=False)
    glfw.make_context_current(win)
    moderngl.create_context()
    for trial in (1, 2):
        print(f"--- trial {trial} ---", flush=True)
        for mode in ("gl", "glfin", "pybusy"):
            measure(mode, win)
    glfw.destroy_window(win)
    glfw.terminate()


main()
