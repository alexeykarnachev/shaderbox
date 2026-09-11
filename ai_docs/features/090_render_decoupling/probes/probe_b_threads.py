# Run: cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/probe_b_threads.py [tiles]
"""Configs B and C: one process, two glfw contexts sharing objects, heavy work on a worker thread.

The worker owns a hidden window created with share=<main window>, so the document texture the
worker renders into is visible to the main context. The worker signals completion with a
glFenceSync the main thread polls with glClientWaitSync(timeout=0) -- so the main thread never
blocks on the worker's GPU work; it just keeps showing the previous finished texture.

  tiles=1  -> config B (one fullscreen heavy draw per worker frame)
  tiles=4  -> config C, 2x2 scissor quadrants
  tiles=16 -> config C, 4x4 scissor tiles
  tiles=0  -> control: worker does pure-Python busy work (no GL) at the same wall-clock cost,
              isolating the GIL from the GPU queue.

moderngl 5.12.0 exposes no fence/sync API (only Context.finish / Context.query); the sync
objects are reached through PyOpenGL 3.1.10, which shares the thread's current GL context.
"""

import sys
import threading
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
from OpenGL import GL

DUR = 5.0


class Worker:
    def __init__(self, share_win, tiles: int) -> None:
        self.tiles = tiles
        self.stop = threading.Event()
        self.frames = 0
        self.frame_ms: list[float] = []
        self.gl_call_ms: list[float] = []
        self.share_win = share_win
        self.tex: moderngl.Texture | None = None
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        win = make_window(64, 64, "worker", visible=False, share=self.share_win)
        glfw.make_context_current(win)
        ctx = moderngl.create_context()
        prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=HEAVY_FRAG)
        prog["u_iters"].value = HEAVY_ITERS if self.tiles <= 1 else HEAVY_ITERS
        vao = fullscreen_quad(ctx, prog)
        tex = ctx.texture((HEAVY_W, HEAVY_H), 4, dtype="f1")
        fbo = ctx.framebuffer(color_attachments=[tex])
        self.tex = tex
        self.ready.set()

        side = 1 if self.tiles <= 1 else int(self.tiles**0.5)
        tw, th = HEAVY_W // side, HEAVY_H // side

        while not self.stop.is_set():
            t0 = now()
            fbo.use()
            prog["u_time"].value = self.frames * 0.01
            if side == 1:
                vao.render()
                t_issue = now()
                fence = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
                GL.glFlush()
            else:
                ctx.scissor = None
                for ty in range(side):
                    for tx in range(side):
                        ctx.scissor = (tx * tw, ty * th, tw, th)
                        vao.render()
                        GL.glFlush()  # a submission boundary per tile
                ctx.scissor = None
                t_issue = now()
                fence = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
                GL.glFlush()
            self.gl_call_ms.append((t_issue - t0) * 1000.0)
            # Block THIS thread until the document frame is done; the main thread is never
            # asked to wait. 1 s timeout in ns.
            GL.glClientWaitSync(fence, GL.GL_SYNC_FLUSH_COMMANDS_BIT, 1_000_000_000)
            GL.glDeleteSync(fence)
            self.frame_ms.append((now() - t0) * 1000.0)
            self.frames += 1
        glfw.destroy_window(win)

    def start(self) -> None:
        self.thread.start()
        self.ready.wait(10.0)


class PyBusyWorker:
    """Control: same wall-clock occupancy, pure Python, no GL -- shows GIL cost alone."""

    def __init__(self) -> None:
        self.stop = threading.Event()
        self.frames = 0
        self.frame_ms: list[float] = []
        self.gl_call_ms: list[float] = []
        self.tex = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            t0 = now()
            x = 0.0
            while (now() - t0) < 0.100:
                for _ in range(1000):
                    x = x * 1.000001 + 0.5
            self.frame_ms.append((now() - t0) * 1000.0)
            self.frames += 1

    def start(self) -> None:
        self.thread.start()


def run(tiles: int, worker_on: bool, label: str) -> None:
    win = make_window(HEAVY_W, HEAVY_H, "MAIN", visible=True)
    glfw.set_window_pos(win, 100, 60)
    glfw.make_context_current(win)
    glfw.swap_interval(0)
    ctx = moderngl.create_context()

    worker: Worker | PyBusyWorker | None = None
    if worker_on:
        worker = PyBusyWorker() if tiles == 0 else Worker(win, tiles)
        worker.start()
        glfw.make_context_current(win)  # the worker's create_window touched the current context

    ui_prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=SAMPLER_FRAG)
    ui_prog["u_tex"].value = 0
    ui_vao = fullscreen_quad(ctx, ui_prog)
    own_tex = ctx.texture((8, 8), 4, dtype="f1")
    doc_tex = getattr(worker, "tex", None) or own_tex

    periods: list[float] = []
    swaps: list[float] = []
    t_end = now() + DUR
    prev = now()
    frame = 0
    while now() < t_end:
        glfw.poll_events()
        ctx.screen.use()
        doc_tex.use(0)
        ui_prog["u_time"].value = frame * 0.01
        ui_vao.render()
        s0 = now()
        glfw.swap_buffers(win)
        t = now()
        swaps.append((t - s0) * 1000.0)
        periods.append((t - prev) * 1000.0)
        prev = t
        frame += 1

    if worker is not None:
        worker.stop.set()
        worker.thread.join(5.0)
        print(
            f"  worker frames={worker.frames} "
            + fmt("worker frame", worker.frame_ms[1:]).replace("worker frame: ", ""),
            flush=True,
        )
        if worker.gl_call_ms:
            print("  " + fmt("worker GL-issue", worker.gl_call_ms[1:]), flush=True)
    print(fmt(label + " period", periods[2:]), flush=True)
    print("  " + fmt(label + " swap  ", swaps[2:]), flush=True)
    glfw.destroy_window(win)


def main() -> None:
    tiles = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    name = {0: "PYBUSY", 1: "B(1 tile)", 4: "C(4 tiles)", 16: "C(16 tiles)"}[tiles]
    wait_for_idle()
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    for trial in (1, 2):
        run(tiles, False, f"{name} idle    t{trial}")
        run(tiles, True, f"{name} running t{trial}")
    glfw.terminate()


main()
