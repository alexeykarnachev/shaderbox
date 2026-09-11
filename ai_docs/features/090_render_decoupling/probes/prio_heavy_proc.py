# Run indirectly by prio_procs.py. Standalone heavy EGL client at a chosen priority.
"""One process holding one EGL context at a given priority, drawing the 100 ms document forever."""

import os
import sys
from pathlib import Path

# PyOpenGL binds its platform (GLX by default on Linux) at OpenGL import time, and the GLX
# platform's glXGetCurrentContext() returns NULL while an EGL context is current, which makes
# PyOpenGL's per-context bookkeeping raise "Attempt to retrieve context when no valid context".
# This must run before ANY module that imports OpenGL, moderngl or glfw.
os.environ["PYOPENGL_PLATFORM"] = "egl"

sys.path.insert(0, str(Path(__file__).parent))

from common import FULLSCREEN_VERT, HEAVY_FRAG, HEAVY_H, HEAVY_ITERS, HEAVY_W, now
from OpenGL import GL
from prio_common import choose_config, create_context, get_display, make_current, make_pbuffer
from prio_gl import compile_program, fence_wait, fullscreen_vao, make_fbo, set_uniform_f, set_uniform_i

DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0


def main() -> None:
    prio = sys.argv[1] if len(sys.argv) > 1 else "LOW"
    tiles = int(sys.argv[3]) if len(sys.argv) > 3 else 1
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
    print(f"HEAVY_PROC requested {prio} granted {granted}", flush=True)

    side = max(1, int(tiles**0.5))
    tw, th = HEAVY_W // side, HEAVY_H // side
    t_end = now() + DUR
    k = 0
    costs: list[float] = []
    while now() < t_end:
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
        k += 1
    costs.sort()
    if costs:
        print(f"HEAVY_PROC frames={len(costs)} median={costs[len(costs)//2]:.2f} ms", flush=True)


main()
