# Run: cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/prio_threads.py <heavy_prio> <light_prio> [tiles]
"""The core question: does a HIGH-priority context get its draws ahead of a 100 ms LOW draw?

Same shape as probe_e_paced.py -- a paced 60 fps light loop beside a heavy worker thread -- but
both contexts are created through EGL with an explicit EGL_CONTEXT_PRIORITY_LEVEL_IMG, and the
priority each was actually granted is printed from eglQueryContext, as the extension requires.

Both contexts are pbuffer-backed and each light frame is closed with a fence rather than a swap:
glfw owns the only X-window path available here and glfw cannot be given a priority attribute
(see prio_glfw_check.py), so the light loop has to run on a context this probe created. The fence
measures what the earlier probes' swap measured -- when the GPU finished this frame's work.

GL is issued through PyOpenGL, not moderngl; prio_gl.py says why.
"""

import os
import sys
import threading
import time
from pathlib import Path

# PyOpenGL binds its platform (GLX by default on Linux) at OpenGL import time, and the GLX
# platform's glXGetCurrentContext() returns NULL while an EGL context is current, which makes
# PyOpenGL's per-context bookkeeping raise "Attempt to retrieve context when no valid context".
# This must run before ANY module that imports OpenGL, moderngl or glfw.
os.environ["PYOPENGL_PLATFORM"] = "egl"

sys.path.insert(0, str(Path(__file__).parent))

from common import (
    FULLSCREEN_VERT,
    HEAVY_FRAG,
    HEAVY_H,
    HEAVY_ITERS,
    HEAVY_W,
    SAMPLER_FRAG,
    fmt,
    now,
)
from prio_guard import wait_for_stable
from OpenGL import EGL, GL
from prio_common import choose_config, create_context, get_display, make_current, make_pbuffer
from prio_gl import compile_program, fence_wait, fullscreen_vao, make_fbo, set_uniform_f, set_uniform_i

DUR = 5.0
BUDGET = 1.0 / 60.0


def heavy_worker(
    dpy: object, cfg: object, prio: str, stop: threading.Event, tiles: int,
    out: list[float], box: dict
) -> None:
    ctx, granted = create_context(dpy, cfg, prio)
    box["heavy_granted"] = granted
    surf = make_pbuffer(dpy, cfg, 64, 64)
    make_current(dpy, surf, surf, ctx)

    prog = compile_program(FULLSCREEN_VERT, HEAVY_FRAG)
    vao = fullscreen_vao()
    fbo, _tex = make_fbo(HEAVY_W, HEAVY_H)
    GL.glUseProgram(prog)
    set_uniform_i(prog, "u_iters", HEAVY_ITERS)
    box["ready"].set()

    side = max(1, int(tiles**0.5))
    tw, th = HEAVY_W // side, HEAVY_H // side
    k = 0
    while not stop.is_set():
        t0 = now()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, HEAVY_W, HEAVY_H)
        GL.glUseProgram(prog)
        GL.glBindVertexArray(vao)
        set_uniform_f(prog, "u_time", k * 0.01)
        if side == 1:
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        else:
            GL.glEnable(GL.GL_SCISSOR_TEST)
            for ty in range(side):
                for tx in range(side):
                    GL.glScissor(tx * tw, ty * th, tw, th)
                    GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
                    GL.glFlush()
            GL.glDisable(GL.GL_SCISSOR_TEST)
        fence_wait()
        out.append((now() - t0) * 1000.0)
        k += 1
    EGL.eglMakeCurrent(dpy, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
    EGL.eglDestroySurface(dpy, surf)
    EGL.eglDestroyContext(dpy, ctx)


def run(heavy_prio: str, light_prio: str, tiles: int, worker_on: bool, label: str) -> None:
    dpy = get_display()
    cfg = choose_config(dpy, pbuffer=True)

    stop = threading.Event()
    wout: list[float] = []
    box: dict = {"ready": threading.Event(), "heavy_granted": "-"}
    th_w: threading.Thread | None = None
    if worker_on:
        th_w = threading.Thread(
            target=heavy_worker, args=(dpy, cfg, heavy_prio, stop, tiles, wout, box), daemon=True
        )
        th_w.start()
        if not box["ready"].wait(30.0):
            raise SystemExit("heavy worker never became ready")

    light_ctx, light_granted = create_context(dpy, cfg, light_prio)
    lsurf = make_pbuffer(dpy, cfg, HEAVY_W, HEAVY_H)
    make_current(dpy, lsurf, lsurf, light_ctx)
    lprog = compile_program(FULLSCREEN_VERT, SAMPLER_FRAG)
    lvao = fullscreen_vao()
    dummy = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, dummy)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, 8, 8, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glUseProgram(lprog)
    set_uniform_i(lprog, "u_tex", 0)

    work: list[float] = []
    t_end = now() + DUR
    frame = 0
    while now() < t_end:
        t0 = now()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, HEAVY_W, HEAVY_H)
        GL.glUseProgram(lprog)
        GL.glBindVertexArray(lvao)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, dummy)
        set_uniform_f(lprog, "u_time", frame * 0.01)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        fence_wait()
        spent = now() - t0
        work.append(spent * 1000.0)
        if spent < BUDGET:
            time.sleep(BUDGET - spent)
        frame += 1

    if th_w is not None:
        stop.set()
        th_w.join(15.0)

    body = work[2:]
    misses = sum(1 for w in body if w > BUDGET * 1000.0)
    print(
        f"{label}  [heavy granted {box['heavy_granted']}, light granted {light_granted}]",
        flush=True,
    )
    print(f"  {fmt('light', body)}  MISSED {misses}/{len(body)}", flush=True)
    if wout[1:]:
        print(f"  {fmt('heavy', wout[1:])}  frames={len(wout)}", flush=True)

    EGL.eglMakeCurrent(dpy, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
    EGL.eglDestroySurface(dpy, lsurf)
    EGL.eglDestroyContext(dpy, light_ctx)
    EGL.eglTerminate(dpy)


def main() -> None:
    heavy_prio = sys.argv[1] if len(sys.argv) > 1 else "LOW"
    light_prio = sys.argv[2] if len(sys.argv) > 2 else "HIGH"
    tiles = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    wait_for_stable()
    tag = f"heavy={heavy_prio} light={light_prio} tiles={tiles}"
    for trial in (1, 2):
        run(heavy_prio, light_prio, tiles, False, f"{tag} IDLE    t{trial}")
        run(heavy_prio, light_prio, tiles, True, f"{tag} RUNNING t{trial}")


main()
