# Run: cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/prio_solo_cost.py
"""P5 part one: does asking for a priority change what a context costs when it runs ALONE?

If HIGH were to help under contention it would still be worth knowing whether HIGH/LOW change a
context's own throughput with nothing else on the GPU -- that would be a document tax charged by
the priority itself, separate from tiling. Each priority renders the same 100 ms document solo.
"""

import os
import sys
from pathlib import Path

# PyOpenGL binds its platform (GLX by default on Linux) at OpenGL import time, and the GLX
# platform's glXGetCurrentContext() returns NULL while an EGL context is current, which makes
# PyOpenGL's per-context bookkeeping raise "Attempt to retrieve context when no valid context".
# This must run before ANY module that imports OpenGL, moderngl or glfw.
os.environ["PYOPENGL_PLATFORM"] = "egl"

sys.path.insert(0, str(Path(__file__).parent))

from common import FULLSCREEN_VERT, HEAVY_FRAG, HEAVY_H, HEAVY_ITERS, HEAVY_W, fmt, now
from prio_guard import wait_for_stable
from OpenGL import EGL, GL
from prio_common import choose_config, create_context, get_display, make_current, make_pbuffer
from prio_gl import compile_program, fence_wait, fullscreen_vao, make_fbo, set_uniform_f, set_uniform_i

N = 25


def measure(prio: str, tiles: int) -> None:
    dpy = get_display()
    cfg = choose_config(dpy, pbuffer=True)
    ctx, granted = create_context(dpy, cfg, prio)
    surf = make_pbuffer(dpy, cfg, 64, 64)
    make_current(dpy, surf, surf, ctx)
    prog = compile_program(FULLSCREEN_VERT, HEAVY_FRAG)
    vao = fullscreen_vao()
    fbo, _tex = make_fbo(HEAVY_W, HEAVY_H)
    GL.glUseProgram(prog)
    set_uniform_i(prog, "u_iters", HEAVY_ITERS)

    side = max(1, int(tiles**0.5))
    tw, th = HEAVY_W // side, HEAVY_H // side
    costs: list[float] = []
    for k in range(N):
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
        costs.append((now() - t0) * 1000.0)

    print(f"  {fmt(f'{prio:<7} tiles={tiles:<2} [granted {granted}]', costs[1:])}", flush=True)
    EGL.eglMakeCurrent(dpy, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
    EGL.eglDestroySurface(dpy, surf)
    EGL.eglDestroyContext(dpy, ctx)
    EGL.eglTerminate(dpy)


def main() -> None:
    wait_for_stable()
    for trial in (1, 2):
        print(f"--- trial {trial}: document cost with nothing else on the GPU ---", flush=True)
        for prio in ("DEFAULT", "LOW", "MEDIUM", "HIGH"):
            measure(prio, 1)
        for prio in ("LOW", "HIGH"):
            measure(prio, 16)


main()
