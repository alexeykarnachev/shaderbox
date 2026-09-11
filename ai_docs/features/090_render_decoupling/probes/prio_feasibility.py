# Run: cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/prio_feasibility.py
"""Step 0: can this driver create EGL contexts at each priority, and what does it grant?

Reports eglQueryContext(EGL_CONTEXT_PRIORITY_LEVEL_IMG) for every context created, which the
extension spec names as the only way to learn the priority actually assigned. Also checks
whether moderngl can attach to an externally-current EGL context.
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

import moderngl
from OpenGL import EGL
from prio_common import (
    choose_config,
    create_context,
    get_display,
    has_priority_ext,
    make_current,
    make_pbuffer,
)


def main() -> None:
    dpy = get_display()
    ver = EGL.eglQueryString(dpy, EGL.EGL_VERSION)
    vendor = EGL.eglQueryString(dpy, EGL.EGL_VENDOR)
    print(f"EGL_VERSION: {ver}")
    print(f"EGL_VENDOR:  {vendor}")
    print(f"EGL_IMG_context_priority present: {has_priority_ext(dpy)}")
    print()

    cfg = choose_config(dpy, pbuffer=True)
    made = []
    for want in ("DEFAULT", "LOW", "MEDIUM", "HIGH"):
        try:
            ctx, granted = create_context(dpy, cfg, want)
            print(f"  requested {want:<7} -> eglQueryContext granted {granted}")
            made.append(ctx)
        except RuntimeError as exc:
            print(f"  requested {want:<7} -> FAILED: {exc}")

    print()
    surf = make_pbuffer(dpy, cfg, 64, 64)
    if made:
        make_current(dpy, surf, surf, made[-1])
        # moderngl cannot be attached to this context, and the bare call's apparent success is
        # the trap: with no settings it returns the cached _store.default_context without ever
        # consulting glcontext, so it reports a plausible renderer for a context it did not bind.
        # The backend calls below are the honest test, and both fail.
        try:
            ctx = moderngl.create_context()
            print(f"  moderngl.create_context() returned: {ctx.info['GL_RENDERER']}")
            print("    (cached default_context, NOT an attach -- see the backend attempts below)")
        except Exception as exc:
            print(f"  moderngl.create_context() FAILED: {type(exc).__name__}: {exc}")
        for kwargs in ({"backend": "egl"}, {"mode": "detect"}):
            try:
                c = moderngl.create_context(require=330, **kwargs)
                print(f"  moderngl.create_context({kwargs}) attached: {c.info['GL_RENDERER']}")
            except Exception as exc:
                print(f"  moderngl.create_context({kwargs}) FAILED: {type(exc).__name__}: {exc}")
        EGL.eglMakeCurrent(dpy, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
    for c in made:
        EGL.eglDestroyContext(dpy, c)
    EGL.eglDestroySurface(dpy, surf)
    EGL.eglTerminate(dpy)


main()
