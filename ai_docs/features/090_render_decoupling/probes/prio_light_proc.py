# Run indirectly by prio_procs.py. Standalone light EGL client at a chosen priority, paced to 60 fps.
"""Paced 60 fps light loop in its own process, on its own EGL context at a given priority."""

import os
import sys
import time
from pathlib import Path

# PyOpenGL binds its platform (GLX by default on Linux) at OpenGL import time, and the GLX
# platform's glXGetCurrentContext() returns NULL while an EGL context is current, which makes
# PyOpenGL's per-context bookkeeping raise "Attempt to retrieve context when no valid context".
# This must run before ANY module that imports OpenGL, moderngl or glfw.
os.environ["PYOPENGL_PLATFORM"] = "egl"

sys.path.insert(0, str(Path(__file__).parent))

from common import FULLSCREEN_VERT, HEAVY_H, HEAVY_W, SAMPLER_FRAG, fmt, now
from OpenGL import GL
from prio_common import choose_config, create_context, get_display, make_current, make_pbuffer
from prio_gl import compile_program, fence_wait, fullscreen_vao, set_uniform_f, set_uniform_i

BUDGET = 1.0 / 60.0


def main() -> None:
    prio = sys.argv[1] if len(sys.argv) > 1 else "HIGH"
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    label = sys.argv[3] if len(sys.argv) > 3 else "light"
    dpy = get_display()
    cfg = choose_config(dpy, pbuffer=True)
    ctx, granted = create_context(dpy, cfg, prio)
    surf = make_pbuffer(dpy, cfg, HEAVY_W, HEAVY_H)
    make_current(dpy, surf, surf, ctx)
    prog = compile_program(FULLSCREEN_VERT, SAMPLER_FRAG)
    vao = fullscreen_vao()
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, 8, 8, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glUseProgram(prog)
    set_uniform_i(prog, "u_tex", 0)

    work: list[float] = []
    t_end = now() + dur
    frame = 0
    while now() < t_end:
        t0 = now()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, HEAVY_W, HEAVY_H)
        GL.glUseProgram(prog)
        GL.glBindVertexArray(vao)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        set_uniform_f(prog, "u_time", frame * 0.01)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        fence_wait()
        spent = now() - t0
        work.append(spent * 1000.0)
        if spent < BUDGET:
            time.sleep(BUDGET - spent)
        frame += 1

    body = work[2:]
    misses = sum(1 for w in body if w > BUDGET * 1000.0)
    print(f"{label} [granted {granted}]  {fmt('light', body)}  MISSED {misses}/{len(body)}", flush=True)


main()
