"""moderngl renderer for the editor's primitive array (feature 067).

One shared MTSDF program + atlas texture per GL context; one `EditorPanel`
per drawn editor region owning an FBO texture the imgui side presents via
`imgui.image`. Primitives arrive draw-ordered from `Editor.layout()`; the
whole array renders as a single interleaved VBO in one draw call (glyph
quads sample the atlas through the MTSDF decode, everything else is solid
geometry — one `textured` vertex flag branches between them).

The redraw gate lives here as data + two free functions so its domain is
unit-testable without GL: `render_state` builds the tuple, `should_redraw`
compares. The panel re-reads primitives and re-renders ONLY on change;
`ed_layout` itself still runs every visible frame (hit tests and scroll
clamping answer against it).
"""

import json
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

from shaderbox.editor.ffi import Editor, Kind

_VERTEX_SHADER = """
#version 330
uniform vec2 u_size;
in vec2 in_pos;
in vec2 in_uv;
in vec4 in_color;
in float in_textured;
out vec2 v_uv;
out vec4 v_color;
out float v_textured;
void main() {
    vec2 ndc = in_pos / u_size * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
    v_color = in_color;
    v_textured = in_textured;
}
"""

_FRAGMENT_SHADER = """
#version 330
uniform sampler2D u_atlas;
uniform float u_screen_px_range;
in vec2 v_uv;
in vec4 v_color;
in float v_textured;
out vec4 f_color;

float median3(vec3 v) {
    return max(min(v.r, v.g), min(max(v.r, v.g), v.b));
}

void main() {
    if (v_textured < 0.5) {
        f_color = v_color;
        return;
    }
    float sd = median3(texture(u_atlas, v_uv).rgb);
    float px = u_screen_px_range * (sd - 0.5);
    float alpha = clamp(px + 0.5, 0.0, 1.0);
    f_color = vec4(v_color.rgb, v_color.a * alpha);
}
"""

PRIM_DTYPE = np.dtype(
    [("kind", "<i4")]
    + [
        (f, "<f4")
        for f in ("x0", "y0", "x1", "y1", "u0", "v0", "u1", "v1", "r", "g", "b", "a")
    ]
)

# pos(2) + uv(2) + color(4) + textured(1)
_VERT_FLOATS = 9


def render_state(
    editor: Editor,
    identity: object,
    size: tuple[int, int],
    px_per_em: float,
    gutter_px: float,
    completion_prefix: object,
    marker_fingerprint: object,
    settings_fingerprint: object,
    focused: bool,
) -> tuple:
    """Everything a rendered frame depends on. `identity` is the session's path:
    tabs share ONE panel, and without it two fresh files (same revision, cursor,
    mode) compare equal and a tab switch shows stale text. A member added to the
    layout's inputs MUST be added here —
    `tests/test_editor_ffi.py::test_render_state_reacts_to_every_editor_dimension`
    walks the domain."""
    return (
        identity,
        editor.get_undo_index(),
        editor.get_scroll(),
        editor.get_current_cursor_position(),
        editor.get_mode(),
        editor.get_selection(),
        editor.get_command_line(),
        editor.complete_open(),
        editor.complete_selected(),
        editor.complete_count(),
        completion_prefix,
        size,
        px_per_em,
        gutter_px,
        marker_fingerprint,
        settings_fingerprint,
        focused,
    )


def should_redraw(prev: tuple | None, cur: tuple) -> bool:
    return prev != cur


def build_vertices(prims: np.ndarray) -> np.ndarray:
    """Expand the primitive array into interleaved triangle vertices, vectorized.

    Quad corners wind as two triangles: (x0,y0 x1,y0 x1,y1) + (x0,y0 x1,y1 x0,y1).
    """
    n = len(prims)
    out = np.empty((n, 6, _VERT_FLOATS), dtype=np.float32)
    x0, y0 = prims["x0"], prims["y0"]
    x1, y1 = prims["x1"], prims["y1"]
    u0, v0 = prims["u0"], prims["v0"]
    u1, v1 = prims["u1"], prims["v1"]
    corner_x = (x0, x1, x1, x0, x1, x0)
    corner_y = (y0, y0, y1, y0, y1, y1)
    corner_u = (u0, u1, u1, u0, u1, u0)
    corner_v = (v0, v0, v1, v0, v1, v1)
    for i in range(6):
        out[:, i, 0] = corner_x[i]
        out[:, i, 1] = corner_y[i]
        out[:, i, 2] = corner_u[i]
        out[:, i, 3] = corner_v[i]
    for c, field in enumerate(("r", "g", "b", "a")):
        out[:, :, 4 + c] = prims[field][:, None]
    textured = (prims["kind"] == int(Kind.GLYPH)) | (
        prims["kind"] == int(Kind.POPUP_GLYPH)
    )
    out[:, :, 8] = textured.astype(np.float32)[:, None]
    return out.reshape(-1, _VERT_FLOATS)


class EditorRenderer:
    """The shared GL half: program + atlas texture + the atlas bake metrics."""

    def __init__(
        self, atlas_png: Path, atlas_json: Path, gl: moderngl.Context | None = None
    ) -> None:
        self.gl: moderngl.Context = gl or moderngl.get_context()
        self.program: moderngl.Program = self.gl.program(
            vertex_shader=_VERTEX_SHADER, fragment_shader=_FRAGMENT_SHADER
        )
        meta: dict = json.loads(atlas_json.read_text())
        self.atlas_px_per_em: float = float(meta["atlas"]["size"])
        self.distance_range: float = float(meta["atlas"]["distanceRange"])
        # The layout's glyph UVs address the PNG top-row-first (measured: a flipped
        # upload renders every glyph upside down) — upload as-is.
        image = Image.open(atlas_png).convert("RGBA")
        self.atlas: moderngl.Texture = self.gl.texture(image.size, 4, image.tobytes())
        self.atlas.filter = (moderngl.LINEAR, moderngl.LINEAR)


class EditorPanel:
    """One drawn editor region: FBO + vertex buffer, resized on demand."""

    def __init__(self, renderer: EditorRenderer) -> None:
        self.renderer = renderer
        self.texture: moderngl.Texture | None = None
        self.fbo: moderngl.Framebuffer | None = None
        self.vbo: moderngl.Buffer | None = None
        self.vao: moderngl.VertexArray | None = None
        self.last_state: tuple | None = None

    def _ensure_target(self, size: tuple[int, int]) -> None:
        if self.texture is not None and tuple(self.texture.size) == size:
            return
        gl = self.renderer.gl
        if self.fbo is not None:
            self.fbo.release()
        if self.texture is not None:
            self.texture.release()
        self.texture = gl.texture(size, 4)
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.fbo = gl.framebuffer(color_attachments=[self.texture])

    def render(
        self,
        editor: Editor,
        size: tuple[int, int],
        px_per_em: float,
        clear_color: tuple[float, float, float, float],
    ) -> moderngl.Texture:
        """Draw the last layout()'s primitives into the panel texture. The caller
        gates on `should_redraw` — this always draws."""
        self._ensure_target(size)
        gl = self.renderer.gl
        arr, count = editor.prims_array()
        assert self.fbo is not None and self.texture is not None
        self.fbo.use()
        self.fbo.clear(*clear_color)
        if count > 0:
            prims = np.frombuffer(arr, dtype=PRIM_DTYPE, count=count)
            verts = build_vertices(prims)
            data = verts.tobytes()
            if self.vbo is None or self.vbo.size < len(data):
                self.vbo = gl.buffer(reserve=max(len(data), 1))
                self.vao = gl.vertex_array(
                    self.renderer.program,
                    [
                        (
                            self.vbo,
                            "2f 2f 4f 1f",
                            "in_pos",
                            "in_uv",
                            "in_color",
                            "in_textured",
                        )
                    ],
                )
            self.vbo.write(data)
            gl.enable(moderngl.BLEND)
            gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self.renderer.program["u_size"] = (float(size[0]), float(size[1]))
            self.renderer.program["u_screen_px_range"] = (
                self.renderer.distance_range * px_per_em / self.renderer.atlas_px_per_em
            )
            self.renderer.atlas.use(0)
            self.renderer.program["u_atlas"] = 0
            assert self.vao is not None
            self.vao.render(moderngl.TRIANGLES, vertices=len(verts))
            gl.disable(moderngl.BLEND)
        return self.texture

    def release(self) -> None:
        for obj in (self.vao, self.vbo, self.fbo, self.texture):
            if obj is not None:
                obj.release()
        self.vao = self.vbo = self.fbo = self.texture = None
        self.last_state = None
