# Run: cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/probe_e_paced.py
"""The real-app shape: a UI thread that WANTS 60 fps, not one that free-runs.

Free-running (probe_b_threads) answers "how fast can the UI thread go"; this answers the
question the feature actually asks -- "can the UI thread hit its 16.7 ms deadline". Each UI
frame does a trivial draw and then sleeps out the remainder of the budget, so what is reported
is deadline MISS, not throughput. A frame that overruns the budget is the defect.
"""

import sys
import threading
import time
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
    SAMPLER_FRAG,
    fmt,
    fullscreen_quad,
    make_window,
    now,
)
from gpu_clear import wait_for_idle
from OpenGL import GL

DUR = 5.0
BUDGET = 1.0 / 60.0


def worker_fn(share_win, stop: threading.Event, tiles: int, out: list[float], box: dict) -> None:
    win = make_window(64, 64, "worker", visible=False, share=share_win)
    glfw.make_context_current(win)
    ctx = moderngl.create_context()
    prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=HEAVY_FRAG)
    prog["u_iters"].value = HEAVY_ITERS
    vao = fullscreen_quad(ctx, prog)
    tex = ctx.texture((HEAVY_W, HEAVY_H), 4, dtype="f1")
    fbo = ctx.framebuffer(color_attachments=[tex])
    box["tex"] = tex
    box["ready"].set()
    side = max(1, int(tiles**0.5))
    tw, th = HEAVY_W // side, HEAVY_H // side
    k = 0
    while not stop.is_set():
        t0 = now()
        fbo.use()
        prog["u_time"].value = k * 0.01
        if side == 1:
            vao.render()
        else:
            for ty in range(side):
                for tx in range(side):
                    ctx.scissor = (tx * tw, ty * th, tw, th)
                    vao.render()
                    GL.glFlush()
            ctx.scissor = None
        fence = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        GL.glFlush()
        GL.glClientWaitSync(fence, GL.GL_SYNC_FLUSH_COMMANDS_BIT, 2_000_000_000)
        GL.glDeleteSync(fence)
        out.append((now() - t0) * 1000.0)
        k += 1
    glfw.destroy_window(win)


def run(tiles: int, worker_on: bool, label: str) -> None:
    win = make_window(HEAVY_W, HEAVY_H, "MAIN", visible=True)
    glfw.set_window_pos(win, 100, 60)
    glfw.make_context_current(win)
    glfw.swap_interval(0)
    ctx = moderngl.create_context()

    stop = threading.Event()
    wout: list[float] = []
    box: dict = {"ready": threading.Event(), "tex": None}
    th_w: threading.Thread | None = None
    if worker_on:
        th_w = threading.Thread(target=worker_fn, args=(win, stop, tiles, wout, box), daemon=True)
        th_w.start()
        box["ready"].wait(10.0)
        glfw.make_context_current(win)

    ui_prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=SAMPLER_FRAG)
    ui_prog["u_tex"].value = 0
    ui_vao = fullscreen_quad(ctx, ui_prog)
    own = ctx.texture((8, 8), 4, dtype="f1")
    doc_tex = box["tex"] or own

    work: list[float] = []  # time spent doing the UI frame, sleep excluded
    t_end = now() + DUR
    frame = 0
    while now() < t_end:
        t0 = now()
        glfw.poll_events()
        ctx.screen.use()
        doc_tex.use(0)
        ui_prog["u_time"].value = frame * 0.01
        ui_vao.render()
        glfw.swap_buffers(win)
        spent = now() - t0
        work.append(spent * 1000.0)
        if spent < BUDGET:
            time.sleep(BUDGET - spent)
        frame += 1

    if th_w is not None:
        stop.set()
        th_w.join(5.0)
        print(f"  worker frames={len(wout)} " + fmt("doc cost", wout[1:]), flush=True)
    misses = sum(1 for w in work[2:] if w > BUDGET * 1000.0)
    body = work[2:]
    print(f"{fmt(label + ' ui-work', body)}  MISSED {misses}/{len(body)} of the 16.7 ms budget", flush=True)
    glfw.destroy_window(win)


TILE_SET = [int(a) for a in sys.argv[1:]] or [1, 4, 16]


def main() -> None:
    wait_for_idle()
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    for trial in (1, 2):
        print(f"--- trial {trial} ---", flush=True)
        for t in TILE_SET:
            run(t, True, f"worker {t:>3} tile")
    glfw.terminate()


main()
