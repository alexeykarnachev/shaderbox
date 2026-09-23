"""moderngl renderer for graph_canvas's two vertex streams (feature 098).

Generic: it knows the library's geometry and nothing about passes or
documents, so it lifts into another project with `ffi.py` and the vendored
resources.

One shared program pair + atlas texture per GL context (`CanvasRenderer`); one
FBO per drawn canvas (`CanvasPanel`), whose texture the host presents however
it likes. The library's own reference shaders are used UNMODIFIED — they are
the shortest statement of the vertex contract, and a host writing its own
re-derives the rotation and field rules by hand.

Two things about the run list are load-bearing:

* The streams INTERLEAVE. Glyphs collected under a later shape must reach the
  GPU before that shape does, so the runs are walked in order rather than
  drawn as all-shapes-then-all-glyphs, which puts wires over text.

* A run's `first` indexes its stream, and for shapes that is an INSTANCE
  index. OpenGL gained `glDrawElementsInstancedBaseInstance` in 4.2 and
  moderngl exposes no base-instance parameter at all, so the offset is applied
  by binding the instance array at a byte offset — one VAO per distinct
  offset, cached. Dropping it is invisible: the runs stay ordered and the
  counts stay right while the wrong instances draw. A measured frame has a
  shape run that does not start at zero, and the library's own gate pins the
  case. (No number here: the one that stood was measured against a `.so`
  vendored before a rebuild, and the scene has not produced it since.)
"""

import ctypes
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

from shaderbox.graph_canvas.ffi import (
    ATLAS_JSON_PATH,
    ATLAS_PNG_PATH,
    SHADERS_DIR,
    GlyphVertex,
    Result,
    ShapeInstance,
)

# The reference shaders are GLSL with the version line supplied by the backend
# (raylib does it upstream) and `common.glsl` prepended for the pinned
# attribute locations.
_GLSL_VERSION: str = "#version 330 core\n"

# Read from the structs rather than counted by hand. The shape instance is 120
# bytes = 30 floats and its eight fields tile it EXACTLY, with no padding
# anywhere: 4+4+4+4+4+2+4+4. So the format string names the eight and nothing
# else -- a padding token appended to it would not be ignored, it would widen
# the binding's stride past the row and make every instance after the first
# read into its neighbour.
_SHAPE_FLOATS: int = ctypes.sizeof(ShapeInstance) // 4
_GLYPH_FLOATS: int = ctypes.sizeof(GlyphVertex) // 4

_SHAPE_ATTRS: tuple[str, ...] = (
    "a_rect",
    "a_fill_top",
    "a_shape",
    "a_fill_bot",
    "a_edge",
    "a_border",
    "a_rotation",
    "a_field",
    "a_uv",
)
_SHAPE_FORMAT: str = "4f 4f 4f 4f 4f 4f 2f 4f 4f/i"
_GLYPH_ATTRS: tuple[str, ...] = ("a_position", "a_texcoord", "a_color", "a_uv_bounds")
_GLYPH_FORMAT: str = "2f 2f 4f 4f"

_SHAPE_STRIDE: int = _SHAPE_FLOATS * 4

# Each shape attribute's byte offset and width within the instance, derived from
# the struct so the two cannot drift. Used to re-bind at a run's base instance.
_SHAPE_BINDINGS: tuple[tuple[str, int, int], ...] = tuple(
    (
        attr,
        getattr(ShapeInstance, field).offset,
        getattr(ShapeInstance, field).size // 4,
    )
    for attr, field in zip(
        _SHAPE_ATTRS,
        (
            "rect",
            "fill_top",
            "shape",
            "fill_bot",
            "edge",
            "border",
            "rotation",
            "field",
            "uv",
        ),
        strict=True,
    )
)
_GLYPH_STRIDE: int = _GLYPH_FLOATS * 4


# The glyph atlas holds one unit for the whole frame; node previews get the
# other. Sharing a unit makes a preview bound for a shape run get sampled as
# coverage by the next glyph run.
_ATLAS_UNIT: int = 0
_IMAGE_UNIT: int = 1

_GL_TEXTURE0: int = 0x84C0
_GL_TEXTURE_2D: int = 0x0DE1


def _bind_raw_texture(name: int, unit: int) -> None:
    """Bind a RAW GL texture name to a unit.

    A run's `texture` is a name the host handed the library, not a moderngl
    object — the library takes a node's preview as a bare `uint32` and touches
    its lifetime not at all. moderngl has no call that binds one, so this goes
    through the loaded GL directly.
    """
    gl = _gl_functions()
    gl.glActiveTexture(_GL_TEXTURE0 + unit)
    gl.glBindTexture(_GL_TEXTURE_2D, name)


_GL_CACHE: ctypes.CDLL | None = None


def _gl_functions() -> ctypes.CDLL:
    """The process's already-loaded libGL, for the two calls moderngl lacks."""
    global _GL_CACHE
    if _GL_CACHE is None:
        gl = ctypes.CDLL("libGL.so.1")
        gl.glActiveTexture.argtypes = [ctypes.c_uint32]
        gl.glActiveTexture.restype = None
        gl.glBindTexture.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        gl.glBindTexture.restype = None
        _GL_CACHE = gl
    return _GL_CACHE


def _source(name: str, shaders_dir: Path) -> str:
    common: str = (shaders_dir / "common.glsl").read_text()
    return _GLSL_VERSION + common + "\n" + (shaders_dir / name).read_text()


def ortho(width: float, height: float) -> np.ndarray:
    """Screen pixels to clip space, y down — the space the library emits in.

    The translation sits in the last ROW, which is the layout GLSL's
    `u_mvp * vec4(pos, 0, 1)` wants once the 16 floats are handed over in
    this order.
    """
    return np.array(
        [
            [2.0 / max(width, 1.0), 0.0, 0.0, 0.0],
            [0.0, -2.0 / max(height, 1.0), 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0, 1.0],
        ],
        dtype="f4",
    )


def shapes_array(result: Result) -> np.ndarray:
    """This frame's shape instances as an (n, 30) float array.

    A COPY: everything in the result points into the handle's own storage and
    is overwritten by the next `gc_frame`.
    """
    if result.shape_count <= 0:
        return np.zeros((0, _SHAPE_FLOATS), dtype="f4")
    return (
        np.ctypeslib.as_array(result.shapes, shape=(result.shape_count,))
        .view("f4")
        .reshape(result.shape_count, _SHAPE_FLOATS)
        .copy()
    )


def glyphs_array(result: Result) -> np.ndarray:
    """This frame's glyph vertices as an (n, _GLYPH_FLOATS) array. A copy."""
    if result.glyph_count <= 0:
        return np.zeros((0, _GLYPH_FLOATS), dtype="f4")
    return (
        np.ctypeslib.as_array(result.glyphs, shape=(result.glyph_count,))
        .view("f4")
        .reshape(result.glyph_count, _GLYPH_FLOATS)
        .copy()
    )


class CanvasRenderer:
    """The programs and the atlas, shared by every canvas on one GL context."""

    def __init__(
        self,
        gl: moderngl.Context | None = None,
        shaders_dir: Path = SHADERS_DIR,
        atlas_png: Path = ATLAS_PNG_PATH,
        atlas_json: Path = ATLAS_JSON_PATH,
    ) -> None:
        self.gl: moderngl.Context = gl or moderngl.get_context()
        self.shape_program: moderngl.Program = self.gl.program(
            vertex_shader=_source("sdf.vert.glsl", shaders_dir),
            fragment_shader=_source("sdf.frag.glsl", shaders_dir),
        )
        self.glyph_program: moderngl.Program = self.gl.program(
            vertex_shader=_source("glyph.vert.glsl", shaders_dir),
            fragment_shader=_source("glyph.frag.glsl", shaders_dir),
        )
        image = Image.open(atlas_png).convert("RGBA")
        self.atlas: moderngl.Texture = self.gl.texture(image.size, 4, image.tobytes())
        self.atlas.filter = (moderngl.LINEAR, moderngl.LINEAR)
        # One unit quad, instanced per shape.
        self.quad: moderngl.Buffer = self.gl.buffer(
            np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype="f4").tobytes()
        )
        self._atlas_json: Path = atlas_json

    def release(self) -> None:
        for obj in (self.quad, self.atlas, self.glyph_program, self.shape_program):
            obj.release()


class CanvasPanel:
    """One drawn canvas: an FBO plus the two stream buffers, resized on demand.

    The host gets back a texture. What it does with it — `imgui.image`, a
    fullscreen blit, a texture on a quad — is not this module's business.
    """

    def __init__(self, renderer: CanvasRenderer) -> None:
        self.renderer: CanvasRenderer = renderer
        self.texture: moderngl.Texture | None = None
        self.fbo: moderngl.Framebuffer | None = None
        self.shape_vbo: moderngl.Buffer | None = None
        self.glyph_vbo: moderngl.Buffer | None = None
        self.glyph_vao: moderngl.VertexArray | None = None
        # One VAO per distinct instance offset: moderngl has no base-instance
        # parameter, so a run's offset is baked into the binding.
        self._shape_vaos: dict[int, moderngl.VertexArray] = {}

    def _ensure_target(self, size: tuple[int, int]) -> None:
        if self.texture is not None and tuple(self.texture.size) == size:
            return
        gl = self.renderer.gl
        if self.fbo is not None:
            self.fbo.release()
        if self.texture is not None:
            self.texture.release()
        self.texture = gl.texture(size, 4)
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo = gl.framebuffer(color_attachments=[self.texture])

    def _drop_shape_vaos(self) -> None:
        for vao in self._shape_vaos.values():
            vao.release()
        self._shape_vaos.clear()

    def _ensure_shape_buffer(self, data: bytes) -> None:
        gl = self.renderer.gl
        if self.shape_vbo is None or self.shape_vbo.size < len(data):
            # The VAOs bind this buffer; a reallocation invalidates all of
            # them, so they go with it rather than leaking per growth step.
            self._drop_shape_vaos()
            if self.shape_vbo is not None:
                self.shape_vbo.release()
            self.shape_vbo = gl.buffer(reserve=max(len(data), _SHAPE_STRIDE))
        self.shape_vbo.write(data)

    def _shape_vao(self, first: int) -> moderngl.VertexArray:
        """The VAO that reads the instance array starting at instance `first`.

        The offset lives in the BINDING because moderngl's draw call has no
        base-instance parameter, and it is applied by re-binding each attribute
        with an explicit byte `offset`. Writing it as a `{n}x` pad in the format
        string instead does NOT work: that pad joins the stride, so instance 1
        lands `n` bytes further on again and the last instance reads off the end
        of the buffer — measured as instance 1 reading row 7 instead of row 4,
        and then a segfault.
        """
        vao = self._shape_vaos.get(first)
        if vao is not None:
            return vao
        assert self.shape_vbo is not None
        program = self.renderer.shape_program
        vao = self.renderer.gl.vertex_array(
            program,
            [
                (self.renderer.quad, "2f", "a_corner"),
                (self.shape_vbo, _SHAPE_FORMAT, *_SHAPE_ATTRS),
            ],
        )
        if first:
            base: int = first * _SHAPE_STRIDE
            for name, offset, width in _SHAPE_BINDINGS:
                attribute = program[name]
                assert isinstance(attribute, moderngl.Attribute)
                vao.bind(
                    attribute.location,
                    "f",
                    self.shape_vbo,
                    f"{width}f",
                    offset=base + offset,
                    stride=_SHAPE_STRIDE,
                    divisor=1,
                )
        self._shape_vaos[first] = vao
        return vao

    def _ensure_glyph_buffer(self, data: bytes) -> None:
        gl = self.renderer.gl
        if self.glyph_vbo is None or self.glyph_vbo.size < len(data):
            if self.glyph_vao is not None:
                self.glyph_vao.release()
                self.glyph_vao = None
            if self.glyph_vbo is not None:
                self.glyph_vbo.release()
            self.glyph_vbo = gl.buffer(reserve=max(len(data), _GLYPH_STRIDE))
            self.glyph_vao = gl.vertex_array(
                self.renderer.glyph_program,
                [(self.glyph_vbo, _GLYPH_FORMAT, *_GLYPH_ATTRS)],
            )
        self.glyph_vbo.write(data)

    def render(
        self,
        result: Result,
        size: tuple[int, int],
        distance_range: float,
        clear_color: tuple[float, float, float, float],
    ) -> moderngl.Texture:
        """Draw one frame's streams into the panel texture."""
        self._ensure_target(size)
        gl = self.renderer.gl
        assert self.fbo is not None and self.texture is not None

        shapes = shapes_array(result)
        glyphs = glyphs_array(result)

        self.fbo.use()
        self.fbo.clear(*clear_color)

        if result.run_count <= 0:
            return self.texture

        if len(shapes):
            self._ensure_shape_buffer(shapes.tobytes())
        if len(glyphs):
            self._ensure_glyph_buffer(glyphs.tobytes())

        # A flat sequence, not bytes: moderngl's item assignment packs the
        # value itself and rejects a pre-packed buffer.
        mvp = tuple(ortho(float(size[0]), float(size[1])).flatten().tolist())
        self.renderer.shape_program["u_mvp"] = mvp
        self.renderer.glyph_program["u_mvp"] = mvp
        self.renderer.glyph_program["u_distance_range"] = distance_range
        self.renderer.glyph_program["u_atlas"] = _ATLAS_UNIT
        self.renderer.shape_program["u_image"] = _IMAGE_UNIT
        self.renderer.atlas.use(_ATLAS_UNIT)

        gl.enable(moderngl.BLEND)
        gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        bound: int = -1
        for index in range(result.run_count):
            run = result.runs[index]
            if run.count <= 0:
                continue
            if run.stream == 0:
                if not len(shapes):
                    continue
                # A run's `texture` is the preview a node asked for, 0 for the
                # untextured batch. It goes on its OWN unit: the glyph atlas
                # holds unit 0 for the whole frame, and a preview bound there
                # would be sampled as glyph coverage by the next text run.
                if run.texture != bound:
                    _bind_raw_texture(run.texture, _IMAGE_UNIT)
                    bound = run.texture
                vao = self._shape_vao(run.first)
                vao.render(moderngl.TRIANGLES, vertices=6, instances=run.count)
            else:
                if self.glyph_vao is None:
                    continue
                self.glyph_vao.render(
                    moderngl.TRIANGLES, vertices=run.count, first=run.first
                )
        gl.disable(moderngl.BLEND)
        return self.texture

    def release(self) -> None:
        self._drop_shape_vaos()
        for obj in (
            self.glyph_vao,
            self.glyph_vbo,
            self.shape_vbo,
            self.fbo,
            self.texture,
        ):
            if obj is not None:
                obj.release()
        self.glyph_vao = None
        self.glyph_vbo = self.shape_vbo = None
        self.fbo = None
        self.texture = None
