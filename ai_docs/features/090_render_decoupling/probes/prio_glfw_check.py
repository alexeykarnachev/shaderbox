# Run: cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/prio_glfw_check.py
"""Can the app's real (glfw) window context be given a priority, without replacing glfw?

glfw 3.4 has no priority window hint. The one opening is GLFW_CONTEXT_CREATION_API=
GLFW_EGL_CONTEXT_API, which makes glfw create the context through EGL; if that context could then
be queried (and, better, asked for) at a priority level, the app could adopt priorities as a hint.
This probe creates a glfw window both ways and asks eglQueryContext what it got.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import glfw
from prio_common import PRIO_NAME, get_display, query_priority

EGL_CONTEXT_PRIORITY_LEVEL_IMG = 0x3100


def probe(creation_api: int, name: str) -> None:
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_CREATION_API, creation_api)
    win = glfw.create_window(64, 64, name, None, None)
    if not win:
        print(f"  {name}: create_window FAILED ({glfw.get_error()})")
        return
    glfw.make_context_current(win)
    egl_ctx = glfw.get_egl_context(win)
    egl_dpy = glfw.get_egl_display()
    print(f"  {name}: window ok, glfw.get_egl_context -> {egl_ctx}, get_egl_display -> {egl_dpy}")
    if egl_ctx and egl_dpy:
        print(f"    eglQueryContext priority: {query_priority(egl_dpy, egl_ctx)}")
    else:
        print("    no EGL handles: this context is not an EGL context (GLX), nothing to query")
    glfw.destroy_window(win)


def main() -> None:
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    print(f"glfw {glfw.get_version_string()!r}")
    print(f"pyGLFW has a priority hint: {any('PRIORITY' in a for a in dir(glfw))}")
    print()
    probe(glfw.NATIVE_CONTEXT_API, "NATIVE_CONTEXT_API (default; GLX on X11)")
    probe(glfw.EGL_CONTEXT_API, "EGL_CONTEXT_API")
    glfw.terminate()


main()
