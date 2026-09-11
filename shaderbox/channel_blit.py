"""One-quad blits over a render target: the channel views, and the Auto-resize resampler.

A separate texture on purpose. The output texture is sampled by feedback reads, exports and
the pass strip, so it is never swizzled or redrawn; each blit sends it through a one-quad
program into a view canvas the viewer shows instead.

The blits differ only in their fragment shader, so they are one class holding a shader rather
than one class each holding the same GL lifecycle. `CanvasResampler` (090 D4) is the same
mechanism aimed at a canvas the caller owns: it draws a source into a DIFFERENT-sized target,
which `copy_framebuffer` cannot do -- measured, that copies 1:1 into a corner between
differently-sized framebuffers, with no GL error and a plausible picture.
"""

import moderngl
import numpy as np

from shaderbox.constants import DEFAULT_VS_FILE_PATH, FULLSCREEN_QUAD_VERTICES
from shaderbox.core import Canvas

_HEAD = """#version 460 core
uniform sampler2D u_source;
in vec2 vs_uv;
out vec4 frag_color;
void main() {
"""

# The alpha channel alone, as grayscale.
ALPHA_FS = (
    _HEAD
    + """    float a = texture(u_source, vs_uv).a;
    frag_color = vec4(a, a, a, 1.0);
}
"""
)

# The color with alpha DISCARDED -- what the shader wrote to RGB, whatever it said about
# coverage. A shader that leaves its background at alpha 0 (a feedback pass whose trails
# depend on that) shows an empty checker in every compositing view; this is the view that
# answers what color is actually there.
RGB_FS = (
    _HEAD
    + """    frag_color = vec4(texture(u_source, vs_uv).rgb, 1.0);
}
"""
)

# The source, rescaled: the sampler's own filter IS the rescale.
RESAMPLE_FS = (
    _HEAD
    + """    frag_color = texture(u_source, vs_uv);
}
"""
)


class ChannelBlit:
    """One channel view: `fragment_shader` reads `u_source` and writes an opaque frame."""

    def __init__(
        self, fragment_shader: str, gl: moderngl.Context | None = None
    ) -> None:
        self._gl = gl or moderngl.get_context()
        self.program: moderngl.Program = self._gl.program(
            vertex_shader=DEFAULT_VS_FILE_PATH.read_text(encoding="utf-8"),
            fragment_shader=fragment_shader,
        )
        self.vbo: moderngl.Buffer = self._gl.buffer(
            np.array(FULLSCREEN_QUAD_VERTICES, dtype="f4")
        )
        self.vao: moderngl.VertexArray = self._gl.vertex_array(
            self.program, [(self.vbo, "2f", "a_pos")]
        )
        self.canvas = Canvas(self._gl)

    def render(self, source: moderngl.Texture) -> moderngl.Texture:
        """`source` through this blit's shader, at the source's size."""
        self.canvas.set_size(source.size)
        source.use(location=0)
        self.program["u_source"] = 0
        self.canvas.fbo.use()
        self._gl.clear()
        self.vao.render()
        return self.canvas.texture

    def release(self) -> None:
        self.canvas.release()
        self.vao.release()
        self.vbo.release()
        self.program.release()


class CanvasResampler:
    """One source texture drawn whole into a target canvas of any size (090 D4).

    Holds no canvas of its own: the caller allocates the destination, so a resize can allocate,
    blit and only then release the old canvas -- `Canvas.set_size` is release-then-allocate,
    which blanks content that matters.
    """

    def __init__(self, gl: moderngl.Context | None = None) -> None:
        self._gl = gl or moderngl.get_context()
        self.program: moderngl.Program = self._gl.program(
            vertex_shader=DEFAULT_VS_FILE_PATH.read_text(encoding="utf-8"),
            fragment_shader=RESAMPLE_FS,
        )
        self.vbo: moderngl.Buffer = self._gl.buffer(
            np.array(FULLSCREEN_QUAD_VERTICES, dtype="f4")
        )
        self.vao: moderngl.VertexArray = self._gl.vertex_array(
            self.program, [(self.vbo, "2f", "a_pos")]
        )

    def blit(self, source: moderngl.Texture, target: Canvas) -> None:
        source.use(location=0)
        self.program["u_source"] = 0
        target.fbo.use()
        self._gl.clear()
        self.vao.render()

    def release(self) -> None:
        self.vao.release()
        self.vbo.release()
        self.program.release()
