import contextlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import moderngl
import numpy as np
from loguru import logger
from OpenGL.GL import GL_SAMPLER_2D, glUseProgram

from shaderbox.constants import (
    DEFAULT_CANVAS_SIZE,
    DEFAULT_FS_FILE_PATH,
    DEFAULT_IMAGE_FILE_PATH,
    DEFAULT_VS_FILE_PATH,
    FULLSCREEN_QUAD_VERTICES,
)
from shaderbox.glyph_tables import TABLE_UNIFORMS
from shaderbox.media import Image, MediaWithTexture, Video
from shaderbox.pass_graph import TargetConfig
from shaderbox.shader_errors import ShaderError, SourceMap, parse_shader_errors
from shaderbox.shader_lib import active as active_lib_index
from shaderbox.shader_lib import resolve_usage
from shaderbox.shader_source import ShaderSource
from shaderbox.util import try_to_release

# The live loop's u_time origin: seconds since this process started. Import time is close
# enough to launch, and it only has to be a fixed origin, not an exact one.
_PROCESS_START: float = time.monotonic()

# Engine-driven: never pass-intrinsic defaults — seed_uniform_values skips them and
# UIDocument.save excludes them. Two kinds: per-frame values Pass.render() recomputes
# from time/canvas, and the program-resident glyph tables Pass.compile() writes once
# (TABLE_UNIFORMS — render() skips those entirely).
ENGINE_DRIVEN_UNIFORMS: frozenset[str] = frozenset(
    {"u_time", "u_aspect", "u_resolution", "u_pass_iteration", "u_pass_iterations"}
    | TABLE_UNIFORMS.keys()
)


@dataclass
class CompileUnit:
    # `sources`: every file contributing to `flattened` (root + auto-resolved lib
    # files). `source_map` remaps driver-emitted line numbers back to (path, line).
    sources: list[ShaderSource]
    flattened: str
    source_map: SourceMap
    error_raw: str = ""
    errors: list[ShaderError] = field(default_factory=list)

    @classmethod
    def empty(cls, source: ShaderSource) -> "CompileUnit":
        return cls(
            sources=[source],
            flattened=source.text,
            source_map=SourceMap.identity(source.path),
        )


class Canvas:
    def __init__(
        self,
        gl: moderngl.Context | None = None,
        size: tuple[int, int] | None = None,
        dtype: str = "f1",
        filter: tuple[int, int] = (moderngl.LINEAR, moderngl.LINEAR),
        wrap: bool = False,
    ) -> None:
        self._gl = gl or moderngl.get_context()
        self.dtype = dtype
        self.filter = filter
        # moderngl defaults repeat_x/y to True; a feedback border needs clamp, so the
        # default here is the opposite of the library's.
        self.wrap = wrap

        self.texture: moderngl.Texture
        self.fbo: moderngl.Framebuffer

        self._init(size)

    def _init(self, size: tuple[int, int] | None) -> None:
        self.texture = self._gl.texture(
            size or DEFAULT_CANVAS_SIZE, 4, dtype=self.dtype
        )
        self.texture.filter = self.filter
        self.texture.repeat_x = self.wrap
        self.texture.repeat_y = self.wrap
        self.fbo = self._gl.framebuffer(color_attachments=[self.texture])

    def release(self) -> None:
        self.texture.release()
        self.fbo.release()

    def set_size(self, size: tuple[int, int]) -> bool:
        if size == self.texture.size:
            return False

        self.release()
        self._init(size)
        return True


UniformValue = (
    int
    | float
    | Sequence[int]
    | Sequence[float]
    | MediaWithTexture
    | moderngl.Texture
    | moderngl.Buffer
)


def _canvas_kwargs_for(target: TargetConfig | None) -> dict[str, Any]:
    if target is None:
        return {}
    return {
        "dtype": target.dtype,
        "filter": (moderngl.LINEAR, moderngl.LINEAR)
        if target.filter_linear
        else (moderngl.NEAREST, moderngl.NEAREST),
        "wrap": target.wrap,
    }


class Pass:
    """One shader, one target, one draw: the unit that compiles and renders (065).

    Owns its source and compiled program, its own `CompileUnit` (so an error carries the right
    file and line), its own render target, and its own uniform values. What a DOCUMENT owns --
    the graph, which pass is the output, the script hook and export -- lives one layer up.
    """

    _DEFAULT_VS_FILE_PATH = DEFAULT_VS_FILE_PATH
    _DEFAULT_FS_FILE_PATH = DEFAULT_FS_FILE_PATH
    _DEFAULT_IMAGE_FILE_PATH = DEFAULT_IMAGE_FILE_PATH

    def __init__(
        self,
        gl: moderngl.Context | None = None,
        source: ShaderSource | None = None,
        canvas_size: tuple[int, int] | None = None,
        target: TargetConfig | None = None,
    ) -> None:
        self._gl = gl or moderngl.get_context()
        self.vs_source: str = self._DEFAULT_VS_FILE_PATH.read_text(encoding="utf-8")
        self.source: ShaderSource = (
            source
            if source is not None
            else ShaderSource.load(self._DEFAULT_FS_FILE_PATH)
        )
        # No target given => Canvas's own 8-bit defaults. TargetConfig's f2 (D9) is what a pass
        # IN A GRAPH gets, and applying it to an unconfigured pass would silently reformat every
        # document's canvas, which the whole export path reads as 8-bit.
        self.target: TargetConfig | None = target
        self.canvas = Canvas(
            size=canvas_size, gl=self._gl, **_canvas_kwargs_for(target)
        )

        # Bumped whenever the target format changes, so a Document can tell that a cached
        # feedback canvas built from this pass predates the change (core cannot see the Document).
        self.target_generation: int = 0
        # The document frame this pass last drew in; -1 means never. Read both by the sweep's
        # skip and by begin_frame, which advances a feedback history only for a pass that drew.
        self.drawn_frame: int = -1
        self.first_render_done: bool = False
        self.uniform_values: dict[str, Any] = {}
        self.compile_unit: CompileUnit = CompileUnit.empty(self.source)
        self.program: moderngl.Program | None = None
        self.vbo: moderngl.Buffer | None = None
        self.vao: moderngl.VertexArray | None = None

    def set_target(self, target: TargetConfig) -> None:
        """Adopt a new target configuration, reallocating the canvas when its format changed.

        Size is NOT applied here: a pass's canvas is sized by the document (its canvas size times
        the target's scale), so applying `scale` from two places would fight.
        """
        if self.target == target:
            return
        size = self.canvas.texture.size
        self.target = target
        self.canvas.release()
        self.canvas = Canvas(size=size, gl=self._gl, **_canvas_kwargs_for(target))
        # A Document holding a feedback history for this pass must drop it: the history was built
        # from the OLD format, and `begin_frame` swaps the pair every frame -- so leaving it makes
        # the pass alternate between formats rather than simply lag one behind.
        self.target_generation += 1

    def release_program(self, new_fs_source: str = "") -> None:
        # Path is the stable identity; only text + mtime change.
        self.source = replace(self.source, text=new_fs_source, mtime=self.source.mtime)
        self.invalidate()

    def invalidate(self) -> None:
        # Drop the cached GL program + compile unit without touching `self.source`;
        # next compile() re-reads included lib files via the resolver.
        # Clearing first_render_done re-admits an off-chain pass to the first-render sweep, so
        # an edit to a pass the output does not need still reaches its tile. Every caller is
        # edit-triggered (a source or lib file changed), never per frame.
        self.first_render_done = False
        self.compile_unit = CompileUnit.empty(self.source)
        if self.program:
            self.program.release()
        if self.vbo:
            self.vbo.release()
        if self.vao:
            self.vao.release()
        self.program = None
        self.vbo = None
        self.vao = None
        # Bind 0 — a deleted program left GL-current crashes the imgui renderer's
        # end-of-frame restore (GLError 1281). Suppressed: under a standalone (headless)
        # context this same call raises GLError 1282 (invalid operation) — there's no imgui
        # restore to protect there, so the bind is pointless and only its exception matters.
        with contextlib.suppress(Exception):
            glUseProgram(0)

    def release(self) -> None:
        self.release_program()
        # The pass OWNS its uniform values: the Image/Video bound to a sampler (each holding a
        # texture, and a Video an open capture), the default Image, and the uniform-block Buffer.
        # Without this every reload (the file watcher, a revert, a project switch) leaks them.
        for value in self.uniform_values.values():
            try_to_release(value)
        self.uniform_values.clear()
        self.canvas.release()

    @property
    def script_ready(self) -> bool:
        # Whether the script engine may read this pass's uniforms THIS tick (069). False only while
        # a compile has never been ATTEMPTED — get_active_uniforms would compile it from inside the
        # frame loop, which 066 D1 forbids, so the engine holds its keys for a tick instead. True
        # once attempted, whether it succeeded or FAILED: a failed attempt is never retried, so
        # holding it on `program is None` would silence its keys for the life of the source.
        return self.program is not None or bool(self.compile_unit.error_raw)

    def get_active_uniforms(self) -> list[moderngl.Uniform | moderngl.UniformBlock]:
        # Lazy compile (066 D1): nothing compiles at load, so the first consumer that needs
        # the program pulls it here. A FAILED attempt is not retried — its errors stick in
        # compile_unit until invalidate() resets it (a source or lib change); render() keeps
        # its own per-call retry. Seeding rides the compile so every consumer keeps the
        # invariant that a returned uniform has a value in uniform_values.
        if self.program is None and not self.compile_unit.error_raw:
            self.compile()
            if self.program is not None:
                self.seed_uniform_values()
        uniforms: list[moderngl.Uniform | moderngl.UniformBlock] = []
        if self.program:
            for uniform_name in self.program:
                uniform = self.program[uniform_name]
                if isinstance(uniform, moderngl.Uniform | moderngl.UniformBlock):
                    uniforms.append(uniform)

        return uniforms

    def compile(self) -> None:
        # On failure the previous valid `self.program` is preserved, so the
        # preview keeps rendering while the error strip surfaces diagnostics.
        flattened, sources, source_map, resolve_errors = resolve_usage(
            self.source, active_lib_index()
        )
        unit = CompileUnit(
            sources=sources,
            flattened=flattened,
            source_map=source_map,
        )
        # Resolver failures surface as synthetic ShaderErrors so the same
        # error-strip + click-to-jump path handles them.
        for re_err in resolve_errors:
            unit.errors.append(ShaderError(re_err.path, re_err.line, re_err.message))
        # Resolver already failed — skip the driver; its output would only confuse.
        if resolve_errors:
            unit.error_raw = "\n".join(e.message for e in resolve_errors)
            if unit.error_raw != self.compile_unit.error_raw:
                logger.error(f"Failed to resolve includes: {unit.error_raw}")
            self.compile_unit = unit
            return

        try:
            program = self._gl.program(
                vertex_shader=self.vs_source,
                fragment_shader=unit.flattened,
            )
        except Exception as e:
            err = str(e)
            if err != self.compile_unit.error_raw:
                logger.error(f"Failed to compile shader: {e}")
            unit.error_raw = err
            unit.errors = parse_shader_errors(err, unit.source_map)
            self.compile_unit = unit
            return

        self.compile_unit = unit

        if self.program:
            self.program.release()
        if self.vbo:
            self.vbo.release()
        if self.vao:
            self.vao.release()

        self.program = program
        self.vbo = self._gl.buffer(np.array(FULLSCREEN_QUAD_VERTICES, dtype="f4"))
        self.vao = self._gl.vertex_array(program, [(self.vbo, "2f", "a_pos")])

        # Program-resident engine tables (glyph strokes): written once per program;
        # an unused table is compiled out by the driver and simply absent. A linker
        # that constant-folds a glyph index may TRIM the active array to a prefix of
        # the declaration, so the write clamps to the active size (Known quirks). The
        # except is guarded like render()'s uniform writes — a user shader redeclaring
        # an SBT_* name with its own shape must not crash compile().
        for table_name, table_data in TABLE_UNIFORMS.items():
            try:
                member = program[table_name]
            except KeyError:
                continue
            if isinstance(member, moderngl.Uniform):
                element_size: int = getattr(
                    member, "element_size", member.dimension * 4
                )
                try:
                    member.write(table_data[: member.array_length * element_size])
                except Exception as e:
                    logger.warning(f"Failed to write engine table '{table_name}': {e}")

    def seed_uniform_values(self) -> None:
        # Fill uniform_values with document-intrinsic defaults for any active uniform not yet
        # present. GL-FREE: no texture.use / program binding / draw — that is render()'s job.
        # Engine-driven uniforms are per-frame canvas/time values, valued only in render().
        if not self.program:
            return
        for uniform in self.get_active_uniforms():
            if uniform.name in ENGINE_DRIVEN_UNIFORMS:
                continue
            if uniform.name not in self.uniform_values:
                self.uniform_values[uniform.name] = self._default_uniform_value(uniform)

    def _default_uniform_value(
        self, uniform: moderngl.Uniform | moderngl.UniformBlock
    ) -> Any:
        if isinstance(uniform, moderngl.UniformBlock):
            return self._gl.buffer(np.zeros(uniform.size, dtype=np.int8))
        if getattr(uniform, "gl_type", None) == GL_SAMPLER_2D:
            return Image(self._DEFAULT_IMAGE_FILE_PATH)
        return uniform.value

    def render(
        self,
        u_time: float | None = None,
        canvas: Canvas | None = None,
        inputs: dict[str, moderngl.Texture] | None = None,
        iteration: int = 0,
        iterations: int = 1,
    ) -> None:
        """Draw this pass into `canvas`, or into its own target.

        `inputs` binds sampler uniforms to textures another pass produced. They are applied for
        THIS draw only and never enter `uniform_values`: the graph owns those bindings, the pass
        owns the ones the user set, and a document-owned texture persisted into a pass's state
        would be saved and then released underneath it.

        `iteration` / `iterations` reach the shader as `u_pass_iteration` / `u_pass_iterations`
        (068). The INDEX is handed over, never a value derived from it -- a `u_jfa_offset` would
        be one algorithm wearing an engine uniform's name, and the shader's own
        `iterations - 1.0 - iteration` (the cascade stack's level) is one line.
        """
        canvas = canvas or self.canvas
        inputs = inputs or {}

        if not self.program or not self.vbo or not self.vao:
            self.compile()

        if not self.program or not self.vao:
            return

        texture_unit = 0
        # No glfw here: this module is imported by the headless core. The live loop renders
        # bare, so it falls through to this clock; export and the probe pass u_time. Measured
        # from process start, not `time.monotonic()` raw — that counts from BOOT, so a shader
        # opened on a long-uptime box starts at whatever the machine had been running for.
        render_time = (
            u_time if u_time is not None else time.monotonic() - _PROCESS_START
        )
        self.seed_uniform_values()
        for uniform in self.get_active_uniforms():
            if uniform.name in TABLE_UNIFORMS:  # program-resident, set at compile
                continue
            value = inputs.get(uniform.name, self.uniform_values.get(uniform.name))

            value_for_program = None

            if isinstance(uniform, moderngl.UniformBlock):
                assert isinstance(value, moderngl.Buffer)
                value.bind_to_uniform_block(uniform.index)

            elif getattr(uniform, "gl_type", None) == GL_SAMPLER_2D:
                if isinstance(value, MediaWithTexture):
                    value.update(render_time)
                    texture = value.texture
                elif isinstance(value, moderngl.Texture):
                    texture = value
                else:
                    raise ValueError(
                        f"Uniform value must have a type MediaWithTexture or moderngl.Texture, but this one is {type(value)}"
                    )

                texture.use(location=texture_unit)
                value_for_program = texture_unit
                texture_unit += 1

            elif uniform.name == "u_time":
                value = render_time
                value_for_program = value

            elif uniform.name == "u_aspect":
                value = np.divide(*canvas.texture.size)
                value_for_program = value

            elif uniform.name == "u_resolution":
                value = canvas.texture.size
                value_for_program = value

            elif uniform.name == "u_pass_iteration":
                value = float(iteration)
                value_for_program = value

            elif uniform.name == "u_pass_iterations":
                value = float(iterations)
                value_for_program = value

            else:
                value_for_program = value

            if uniform.name not in inputs:
                self.uniform_values[uniform.name] = value

            if value_for_program is not None:
                try:
                    self.program[uniform.name] = value_for_program
                except Exception as e:
                    logger.debug(
                        f"Failed to set uniform '{uniform.name}' with value {value} ({e}). "
                        f"Cached value will be cleared"
                    )
                    self.uniform_values.pop(uniform.name)

        canvas.fbo.use()
        self._gl.clear()
        self.vao.render()

    def restart_video_uniforms(self) -> None:
        for uniform in self.get_active_uniforms():
            video = self.uniform_values.get(uniform.name)
            if isinstance(video, Video):
                video.restart()
                logger.debug(f"Video uniform '{uniform.name}' restarted")
