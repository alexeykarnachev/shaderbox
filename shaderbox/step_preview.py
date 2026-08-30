"""Displaying a float render target (064).

A step target holds values outside [0, 1] -- that is the whole reason it is float, since
8-bit saturates on the first accumulate pass. Blitting one straight to the screen shows
pure white for exactly the steps worth debugging, so a viewed float step goes through a
tonemap first.

Its own module because it is a GL concern with no opinion about panels: a texture in, an
8-bit canvas out. `ui.py` decides WHEN to show a step; this decides HOW one is shown.
"""

import moderngl
import numpy as np

from shaderbox.core import Canvas

_VS = """#version 330
in vec2 a_pos;
out vec2 vs_uv;
void main() {
    vs_uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

_FS = """#version 330
in vec2 vs_uv;
out vec4 fs_color;
uniform sampler2D u_src;
uniform float u_exposure;
void main() {
    vec3 c = texture(u_src, vs_uv).rgb * u_exposure;
    // Reinhard, then sRGB: the same shape a shader's own main() usually ends with, so a
    // step previews close to how it will look once it is composed.
    c = c / (c + 1.0);
    fs_color = vec4(pow(max(c, 0.0), vec3(1.0 / 2.2)), 1.0);
}
"""

_QUAD = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]


class StepPreview:
    """Tonemaps a float step target into an 8-bit canvas for display.

    Lazily built and reused: nothing is allocated until a float step is actually viewed,
    so a session that never opens one pays nothing.
    """

    def __init__(self, gl: moderngl.Context | None = None) -> None:
        self._gl = gl or moderngl.get_context()
        self._program: moderngl.Program | None = None
        self._vbo: moderngl.Buffer | None = None
        self._vao: moderngl.VertexArray | None = None
        self._canvas: Canvas | None = None

    def _ensure(self) -> None:
        if self._program is not None:
            return
        self._program = self._gl.program(vertex_shader=_VS, fragment_shader=_FS)
        self._vbo = self._gl.buffer(np.array(_QUAD, dtype="f4"))
        self._vao = self._gl.vertex_array(
            self._program, [(self._vbo, "2f", "a_pos")]
        )

    def texture_for(
        self, source: moderngl.Texture, exposure: float = 1.0
    ) -> moderngl.Texture:
        """The texture to display for `source`.

        An 8-bit source is returned untouched -- it is already display-ready, and routing
        it through the tonemap would change how every existing node looks.
        """
        if source.dtype == "f1":
            return source

        self._ensure()
        assert self._program is not None and self._vao is not None
        if self._canvas is None:
            self._canvas = Canvas(gl=self._gl, size=source.size)
        elif self._canvas.texture.size != source.size:
            self._canvas.set_size(source.size)

        source.use(location=0)
        self._program["u_src"] = 0
        self._program["u_exposure"] = exposure
        self._canvas.fbo.use()
        self._gl.clear()
        self._vao.render(moderngl.TRIANGLE_STRIP)
        return self._canvas.texture

    def release(self) -> None:
        for obj in (self._vao, self._vbo, self._program):
            if obj is not None:
                obj.release()
        self._vao = None
        self._vbo = None
        self._program = None
        if self._canvas is not None:
            self._canvas.release()
            self._canvas = None
