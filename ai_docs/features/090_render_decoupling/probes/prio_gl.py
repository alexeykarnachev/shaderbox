# Shared raw-PyOpenGL draw helpers for the prio_* probes. Imported, not run.
"""moderngl cannot be attached to an externally-created EGL context on this stack.

glcontext 2.3.7's EGL backend has no "detect" mode at all (egl.cpp's meth_create_context accepts
only "standalone" and "share"), and the default x11 backend's detect path requires
glXGetCurrentContext() to be non-NULL, which it is not when the thread's current context came
from eglMakeCurrent. moderngl.create_context() appears to succeed on the main thread only because
it returns the cached _store.default_context without consulting glcontext; the same call on a
worker thread raises "(detect) glXGetCurrentContext: cannot detect OpenGL context".

So these probes issue GL through PyOpenGL directly. The shaders are byte-for-byte the ones in
common.py, so the heavy pass is the same 100.93 ms document the earlier probes calibrated.
"""

import ctypes
from typing import Any

import numpy as np
from OpenGL import GL


def compile_program(vert_src: str, frag_src: str) -> int:
    prog = GL.glCreateProgram()
    for src, kind in ((vert_src, GL.GL_VERTEX_SHADER), (frag_src, GL.GL_FRAGMENT_SHADER)):
        sh = GL.glCreateShader(kind)
        GL.glShaderSource(sh, src)
        GL.glCompileShader(sh)
        if not GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS):
            raise RuntimeError(f"shader compile failed: {GL.glGetShaderInfoLog(sh)!r}")
        GL.glAttachShader(prog, sh)
        GL.glDeleteShader(sh)
    GL.glBindAttribLocation(prog, 0, "in_pos")
    GL.glLinkProgram(prog)
    if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
        raise RuntimeError(f"program link failed: {GL.glGetProgramInfoLog(prog)!r}")
    return prog


def fullscreen_vao() -> int:
    verts = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
    vao = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STATIC_DRAW)
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
    return vao


def make_fbo(w: int, h: int) -> tuple[int, int]:
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    fbo = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex, 0
    )
    st = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    if st != GL.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"incomplete FBO 0x{st:04x}")
    return fbo, tex


def fence_wait(timeout_ns: int = 3_000_000_000) -> None:
    """Block THIS thread until the GPU has finished everything issued so far.

    PyOpenGL's glClientWaitSync releases the GIL (measured in the earlier experiment), so a
    worker blocked here does not stall the light thread's Python.
    """
    fence = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
    GL.glFlush()
    GL.glClientWaitSync(fence, GL.GL_SYNC_FLUSH_COMMANDS_BIT, timeout_ns)
    GL.glDeleteSync(fence)


def set_uniform_f(prog: int, name: str, value: float) -> None:
    GL.glUniform1f(GL.glGetUniformLocation(prog, name), value)


def set_uniform_i(prog: int, name: str, value: int) -> None:
    GL.glUniform1i(GL.glGetUniformLocation(prog, name), value)
