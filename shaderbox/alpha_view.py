"""The viewer's Alpha channel view: the output's alpha as a grayscale image.

A separate texture on purpose. The output texture is sampled by feedback reads, exports and
the pass strip, so it is never swizzled or redrawn; this blits it through a one-quad program
into a view canvas the viewer shows instead.
"""

import moderngl
import numpy as np

from shaderbox.constants import DEFAULT_VS_FILE_PATH, FULLSCREEN_QUAD_VERTICES
from shaderbox.core import Canvas

_ALPHA_FS = """#version 460 core
uniform sampler2D u_source;
in vec2 vs_uv;
out vec4 frag_color;
void main() {
    float a = texture(u_source, vs_uv).a;
    frag_color = vec4(a, a, a, 1.0);
}
"""


class AlphaView:
    def __init__(self, gl: moderngl.Context | None = None) -> None:
        self._gl = gl or moderngl.get_context()
        self.program: moderngl.Program = self._gl.program(
            vertex_shader=DEFAULT_VS_FILE_PATH.read_text(encoding="utf-8"),
            fragment_shader=_ALPHA_FS,
        )
        self.vbo: moderngl.Buffer = self._gl.buffer(
            np.array(FULLSCREEN_QUAD_VERTICES, dtype="f4")
        )
        self.vao: moderngl.VertexArray = self._gl.vertex_array(
            self.program, [(self.vbo, "2f", "a_pos")]
        )
        self.canvas = Canvas(self._gl)

    def render(self, source: moderngl.Texture) -> moderngl.Texture:
        """The alpha of `source` as grayscale, at the source's size."""
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
