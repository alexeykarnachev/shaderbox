# Shared EGL-context-priority helpers for the 090 priority probes. Imported by prio_*.py.
"""Creates OpenGL contexts directly through EGL so EGL_CONTEXT_PRIORITY_LEVEL_IMG can be set.

glcontext 2.3.7's egl.cpp builds ctxattribs as a fixed literal (lines 261-267 for "standalone",
321-327 for "share") with no hook for extra attributes, and glfw 3.4 has no priority window hint,
so neither library can ask for a priority. These probes therefore drive libEGL through PyOpenGL's
OpenGL.EGL bindings and issue GL through PyOpenGL too; prio_gl.py records why moderngl could not
be attached to the resulting context.
"""

import ctypes
import os
from typing import Any

# PyOpenGL tracks per-context client state through its *platform* module, and the default
# platform on Linux is GLX, whose glXGetCurrentContext() returns NULL while an EGL context is
# current -- glVertexAttribPointer then dies with "Attempt to retrieve context when no valid
# context". Selecting the EGL platform before OpenGL is imported is what makes raw GL usable
# here, so it is set from inside this module rather than left to the caller's environment.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from OpenGL import EGL

# EGL_IMG_context_priority, Khronos EGL extension #10, version 1.1 (8 September 2009).
EGL_CONTEXT_PRIORITY_LEVEL_IMG = 0x3100
EGL_CONTEXT_PRIORITY_HIGH_IMG = 0x3101
EGL_CONTEXT_PRIORITY_MEDIUM_IMG = 0x3102
EGL_CONTEXT_PRIORITY_LOW_IMG = 0x3103

PRIO_NAME = {
    EGL_CONTEXT_PRIORITY_HIGH_IMG: "HIGH",
    EGL_CONTEXT_PRIORITY_MEDIUM_IMG: "MEDIUM",
    EGL_CONTEXT_PRIORITY_LOW_IMG: "LOW",
}
PRIO_BY_NAME = {
    "HIGH": EGL_CONTEXT_PRIORITY_HIGH_IMG,
    "MEDIUM": EGL_CONTEXT_PRIORITY_MEDIUM_IMG,
    "LOW": EGL_CONTEXT_PRIORITY_LOW_IMG,
    "DEFAULT": 0,
}


def egl_err() -> str:
    return f"0x{EGL.eglGetError():04x}"


def get_display() -> Any:
    """EGL display for the default (X11) native display."""
    dpy = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
    if dpy == EGL.EGL_NO_DISPLAY:
        raise RuntimeError(f"eglGetDisplay failed {egl_err()}")
    major, minor = ctypes.c_long(), ctypes.c_long()
    if not EGL.eglInitialize(dpy, major, minor):
        raise RuntimeError(f"eglInitialize failed {egl_err()}")
    return dpy


def has_priority_ext(dpy: Any) -> bool:
    exts = EGL.eglQueryString(dpy, EGL.EGL_EXTENSIONS)
    if isinstance(exts, bytes):
        exts = exts.decode()
    return "EGL_IMG_context_priority" in exts


def choose_config(dpy: Any, pbuffer: bool = True) -> Any:
    surf_bit = EGL.EGL_PBUFFER_BIT if pbuffer else EGL.EGL_WINDOW_BIT
    attribs = [
        EGL.EGL_SURFACE_TYPE, surf_bit,
        EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
        EGL.EGL_RED_SIZE, 8,
        EGL.EGL_GREEN_SIZE, 8,
        EGL.EGL_BLUE_SIZE, 8,
        EGL.EGL_ALPHA_SIZE, 8,
        EGL.EGL_DEPTH_SIZE, 0,
        EGL.EGL_NONE,
    ]
    arr = (EGL.EGLint * len(attribs))(*attribs)
    cfg = (EGL.EGLConfig * 1)()
    n = ctypes.c_long()
    if not EGL.eglChooseConfig(dpy, arr, cfg, 1, n) or n.value == 0:
        raise RuntimeError(f"eglChooseConfig failed {egl_err()} (n={n.value})")
    return cfg[0]


def create_context(dpy: Any, cfg: Any, priority: str, share: Any = None) -> tuple[Any, str]:
    """Create a GL 3.3 core context at the named priority. Returns (ctx, granted-priority-name)."""
    if not EGL.eglBindAPI(EGL.EGL_OPENGL_API):
        raise RuntimeError(f"eglBindAPI(EGL_OPENGL_API) failed {egl_err()}")
    attribs = [
        EGL.EGL_CONTEXT_MAJOR_VERSION, 3,
        EGL.EGL_CONTEXT_MINOR_VERSION, 3,
        EGL.EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL.EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
    ]
    want = PRIO_BY_NAME[priority]
    if want:
        attribs += [EGL_CONTEXT_PRIORITY_LEVEL_IMG, want]
    attribs.append(EGL.EGL_NONE)
    arr = (EGL.EGLint * len(attribs))(*attribs)
    ctx = EGL.eglCreateContext(dpy, cfg, share or EGL.EGL_NO_CONTEXT, arr)
    if ctx == EGL.EGL_NO_CONTEXT:
        raise RuntimeError(f"eglCreateContext(priority={priority}) failed {egl_err()}")
    return ctx, query_priority(dpy, ctx)


def query_priority(dpy: Any, ctx: Any) -> str:
    """The spec's own answer to 'what did I actually get' -- eglQueryContext."""
    val = ctypes.c_long(EGL_CONTEXT_PRIORITY_MEDIUM_IMG)
    if not EGL.eglQueryContext(dpy, ctx, EGL_CONTEXT_PRIORITY_LEVEL_IMG, val):
        return f"QUERY_FAILED({egl_err()})"
    return PRIO_NAME.get(val.value, f"0x{val.value:04x}")


def make_pbuffer(dpy: Any, cfg: Any, w: int, h: int) -> Any:
    attribs = [EGL.EGL_WIDTH, w, EGL.EGL_HEIGHT, h, EGL.EGL_NONE]
    arr = (EGL.EGLint * len(attribs))(*attribs)
    surf = EGL.eglCreatePbufferSurface(dpy, cfg, arr)
    if surf == EGL.EGL_NO_SURFACE:
        raise RuntimeError(f"eglCreatePbufferSurface failed {egl_err()}")
    return surf


def make_current(dpy: Any, draw: Any, read: Any, ctx: Any) -> None:
    if not EGL.eglMakeCurrent(dpy, draw, read, ctx):
        raise RuntimeError(f"eglMakeCurrent failed {egl_err()}")
