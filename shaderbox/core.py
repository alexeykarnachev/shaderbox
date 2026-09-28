import contextlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import moderngl
import numpy as np
from loguru import logger
from OpenGL.GL import (
    GL_BOOL,
    GL_DOUBLE,
    GL_FLOAT,
    GL_FLOAT_VEC2,
    GL_FLOAT_VEC3,
    GL_FLOAT_VEC4,
    GL_INT,
    GL_INT_VEC2,
    GL_INT_VEC3,
    GL_INT_VEC4,
    GL_SAMPLER_2D,
    GL_UNSIGNED_INT,
    glUseProgram,
)

from shaderbox.constants import (
    DEFAULT_CANVAS_SIZE,
    DEFAULT_FS_FILE_PATH,
    DEFAULT_VS_FILE_PATH,
    FULLSCREEN_QUAD_VERTICES,
)
from shaderbox.engine_uniforms import ENGINE_DRIVEN_UNIFORMS, ENGINE_UNIFORM_TYPES
from shaderbox.glyph_tables import TABLE_UNIFORMS
from shaderbox.instanced import (
    CORNER_ATTRIBUTE,
    MODE_UNIFORM,
    InstancedError,
    build,
    validate_population,
)
from shaderbox.instanced_outcome import InstancedOutcome
from shaderbox.intel.glsl import EntityField, entity_fields
from shaderbox.media import MediaWithTexture, Video
from shaderbox.pass_graph import AutoSource, TargetConfig
from shaderbox.scripting.keys import REFUSED_POPULATION
from shaderbox.shader_errors import (
    ShaderError,
    SourceMap,
    find_uniform_declaration_line,
    parse_shader_errors,
)
from shaderbox.shader_lib import active as active_lib_index
from shaderbox.shader_lib import resolve_usage
from shaderbox.shader_source import ShaderSource
from shaderbox.util import try_to_release

# moderngl's buffer-format spelling per field type. `/i` (divisor 1) is appended at the
# binding: without it every instance reads row 0 and the whole population stacks.
_ATTRIBUTE_FORMATS: dict[str, str] = {
    "float": "1f",
    "vec2": "2f",
    "vec3": "3f",
    "vec4": "4f",
    "int": "1i",
    "ivec2": "2i",
    "ivec3": "3i",
    "ivec4": "4i",
    "uint": "1u",
    "uvec2": "2u",
    "uvec3": "3u",
    "uvec4": "4u",
}

# The live loop's u_time origin: seconds since this process started. Import time is close
# enough to launch, and it only has to be a fixed origin, not an exact one.
_PROCESS_START: float = time.monotonic()


def process_time() -> float:
    """Seconds since this process started: the one clock the live loop, the script tick and a
    document's time origin all read, so a reset can subtract it from itself."""
    return time.monotonic() - _PROCESS_START


# The GL type enum behind each GLSL type this engine names, and back (for the message).
_GL_TYPE_OF_GLSL: dict[str, int] = {
    "float": GL_FLOAT,
    "vec2": GL_FLOAT_VEC2,
    "vec3": GL_FLOAT_VEC3,
    "vec4": GL_FLOAT_VEC4,
    "int": GL_INT,
    "ivec2": GL_INT_VEC2,
    "ivec3": GL_INT_VEC3,
    "ivec4": GL_INT_VEC4,
    "uint": GL_UNSIGNED_INT,
    "bool": GL_BOOL,
    "double": GL_DOUBLE,
}
_GLSL_OF_GL_TYPE: dict[int, str] = {v: k for k, v in _GL_TYPE_OF_GLSL.items()}


def engine_uniform_type_errors(
    program: moderngl.Program, source_text: str, root_path: Path
) -> list[ShaderError]:
    # One error per engine-driven uniform whose declared type is not the one the engine
    # writes, anchored on its declaration line so the error strip can jump to it.
    errors: list[ShaderError] = []
    for name, glsl_type in ENGINE_UNIFORM_TYPES.items():
        try:
            member = program[name]
        except KeyError:
            continue
        if not isinstance(member, moderngl.Uniform):
            continue
        gl_type: int = member.gl_type  # type: ignore[attr-defined]
        if gl_type == _GL_TYPE_OF_GLSL[glsl_type]:
            continue
        declared = _GLSL_OF_GL_TYPE.get(gl_type, f"GL type {gl_type:#x}")
        line = find_uniform_declaration_line(source_text, name)
        errors.append(
            ShaderError(
                root_path,
                line if line is not None else -1,
                f"'{name}' is written by the engine and must be declared `{glsl_type}`, "
                f"not `{declared}`",
            )
        )
    return errors


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
        # What a sampler reads when its value is a source and no document bound a texture for
        # it: a pass drawn on its own has no pass to read, and an unfilled input reads BLACK
        # (065 D3), never a picture.
        self._black: moderngl.Texture | None = None
        self.compile_unit: CompileUnit = CompileUnit.empty(self.source)
        self.program: moderngl.Program | None = None
        self.vbo: moderngl.Buffer | None = None
        self.vao: moderngl.VertexArray | None = None
        # One GPU buffer per entity field of an instanced pass, by field name, each
        # allocated to the population's capacity and rewritten per frame. Per FIELD
        # rather than one interleaved record: the draw costs the same either way
        # (measured 0.131 ms both, ratio 1.002) while interleaving costs a repack of
        # every column every frame, and a script's arrays are already contiguous.
        self.instance_buffers: dict[str, moderngl.Buffer] = {}
        self.entity_fields: tuple[EntityField, ...] = ()
        # What the script produced for THIS frame, by field name; empty when it produced
        # nothing. The script engine writes it and `render` consumes it, which keeps the
        # engine free of GL and leaves one place for a future off-thread producer to
        # write instead.
        self.pending_instances: dict[str, np.ndarray] = {}
        # What the last `render()` (or `compile()`, for the two failure states it owns)
        # decided this pass did. `pass_name` is empty here -- `Pass` does not know its own
        # name in the graph -- and `Document` rewraps it with the real name when it collects
        # per-frame outcomes (102 D4a). Defaults to `not_compiled`: frame one of every
        # document, before any render attempt has produced a real state.
        self.last_outcome: InstancedOutcome = InstancedOutcome("", "not_compiled")

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
        for buffer in self.instance_buffers.values():
            buffer.release()
        self.instance_buffers.clear()
        self.entity_fields = ()
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
        if self._black is not None:
            self._black.release()
            self._black = None
        self.canvas.release()

    def _black_texture(self) -> moderngl.Texture:
        if self._black is None:
            self._black = self._gl.texture((1, 1), 4, data=b"\x00\x00\x00\xff")
        return self._black

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

    def _fail_compile(self, unit: "CompileUnit") -> None:
        # On failure the previous valid `self.program` is preserved, so the preview keeps
        # rendering while the error strip surfaces diagnostics. Which of the two failure
        # states this is follows from that same preservation: no `self.program` ever means
        # this pass has never once compiled clean, and keeping one means the draw that
        # follows renders it under today's error (102 I5) -- `stale_program` names that.
        self.compile_unit = unit
        self.last_outcome = InstancedOutcome(
            "",
            "stale_program" if self.program is not None else "compile_failed",
            detail=unit.error_raw,
        )

    def compile(self) -> None:
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
            self._fail_compile(unit)
            return

        # Read the entity fields from the FLATTENED source, which is what the driver
        # compiles: a `flat in` spliced in from a library file is as real as one the
        # author typed. A pass declaring none is an ordinary fullscreen pass and keeps
        # the shared vertex shader untouched.
        fields = entity_fields(unit.flattened)
        vertex_source = self.vs_source
        if fields:
            try:
                vertex_source = build(
                    fields, self._gl.info["GL_MAX_VERTEX_ATTRIBS"]
                ).vertex_source
            except InstancedError as e:
                # The author's own words about the author's own declaration. Routed
                # through the normal compile-failure path so it reaches the error strip
                # rather than raising into the frame loop.
                unit.error_raw = str(e)
                unit.errors.append(ShaderError(unit.source_map.root_path, 0, str(e)))
                if unit.error_raw != self.compile_unit.error_raw:
                    logger.error(f"Failed to build the instanced stage: {e}")
                self._fail_compile(unit)
                return

        try:
            program = self._gl.program(
                vertex_shader=vertex_source,
                fragment_shader=unit.flattened,
            )
        except Exception as e:
            err = str(e)
            if err != self.compile_unit.error_raw:
                logger.error(f"Failed to compile shader: {e}")
            unit.error_raw = err
            unit.errors = parse_shader_errors(err, unit.source_map)
            self._fail_compile(unit)
            return

        type_errors = engine_uniform_type_errors(
            program, self.source.text, unit.source_map.root_path
        )
        if type_errors:
            # The same outcome as a driver failure: the previous program keeps rendering, the
            # errors surface on the strip and in the copilot's compile feedback.
            program.release()
            unit.errors = type_errors
            unit.error_raw = "\n".join(e.message for e in type_errors)
            if unit.error_raw != self.compile_unit.error_raw:
                logger.error(f"Failed to compile shader: {unit.error_raw}")
            self._fail_compile(unit)
            return

        self.compile_unit = unit

        if self.program:
            self.program.release()
        if self.vbo:
            self.vbo.release()
        if self.vao:
            self.vao.release()
        # A recompile rebuilds the VAO, and a VAO holds its buffers -- so these go with
        # it. Explicit, though `clear()` alone would also free them: dropping the last
        # Python reference lets moderngl's collector release the names, measured at the
        # same live count over twenty recompiles either way. The release stays because it
        # does not depend on when that collection runs. The twin in `invalidate` is the
        # one that genuinely leaks without it, and that one is gated.
        for buffer in self.instance_buffers.values():
            buffer.release()
        self.instance_buffers.clear()

        self.program = program
        self.entity_fields = fields
        self.vbo = self._gl.buffer(np.array(FULLSCREEN_QUAD_VERTICES, dtype="f4"))
        attribute = CORNER_ATTRIBUTE if fields else "a_pos"
        self.vao = self._gl.vertex_array(program, [(self.vbo, "2f", attribute)])

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
            return AutoSource()
        return uniform.value

    def render(
        self,
        u_time: float | None = None,
        canvas: Canvas | None = None,
        inputs: dict[str, moderngl.Texture] | None = None,
        iteration: int = 0,
        iterations: int = 1,
        instances: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Draw this pass into `canvas`, or into its own target.

        `inputs` binds sampler uniforms to textures another pass produced. They are applied for
        THIS draw only and never enter `uniform_values`: the document owns those textures, the
        pass owns the SOURCE the user chose, and a document-owned texture persisted into a
        pass's state would be saved and then released underneath it. A sampler whose value is a
        source (`PassSource`, `NoSource`, `AutoSource`) and that `inputs` does not fill reads
        black.

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

        # A stale program (I5): the last compile attempt for the CURRENT source failed and
        # `compile()` kept the old one running rather than a fresh recompile. The draw below
        # still uses it -- that is what "stale" means, the picture does not change -- but the
        # outcome it reports must stay `stale_program` rather than letting a successful draw
        # of the OLD program read as an ordinary `drew`/`fullscreen` this frame.
        stale = bool(self.compile_unit.error_raw)

        texture_unit = 0
        # A Pass drawn outside a Document (nothing passes u_time) falls through to the process
        # clock; a Document resolves its own clock before reaching here, and export and the
        # probe pass u_time. Measured from process start, not `time.monotonic()` raw — that
        # counts from BOOT, so a shader opened on a long-uptime box starts at whatever the
        # machine had been running for.
        render_time = u_time if u_time is not None else process_time()
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
                    texture = self._black_texture()

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

        outcome = self._upload_instances(
            instances if instances is not None else self.pending_instances
        )
        if not stale:
            self.last_outcome = outcome
        if outcome.state == "refused":
            # Hold the previous frame. The engine already reported why on the strip, and
            # drawing the entity shader fullscreen would answer a data mistake with a
            # picture that looks deliberate.
            return
        count = outcome.count
        if self.entity_fields:
            # The generated stage branches on this, so an unset flag silently takes the
            # fullscreen path and the whole population vanishes with no error.
            mode = self.program[MODE_UNIFORM]
            assert isinstance(mode, moderngl.Uniform)
            mode.value = count is not None

        canvas.fbo.use()
        self._gl.clear()
        if count is None:
            self.vao.render()
            return
        # Additive, and set on EVERY instanced draw rather than toggled: `blend_func`
        # cannot be read back (moderngl raises on the getter), so there is no state to
        # save and restore, and an enable left behind doubles the output of any later
        # pass that draws more than once.
        self._gl.enable(moderngl.BLEND)
        self._gl.blend_func = moderngl.ONE, moderngl.ONE
        self.vao.render(moderngl.TRIANGLES, vertices=6, instances=count)
        self._gl.disable(moderngl.BLEND)

    def _upload_instances(
        self, instances: "dict[str, np.ndarray] | None"
    ) -> InstancedOutcome:
        """Write this frame's entity columns and report what this draw did.

        `fullscreen` covers both a pass declaring no entity fields and an instanced pass
        whose script did not produce a population this frame -- the columns arrive as a
        VALUE and nothing here reads live state, so moving the producer off the render
        thread later changes nothing in this path.
        """
        if instances is REFUSED_POPULATION:
            # The engine already named the problem on the strip (`ScriptStatus.soft_errors`);
            # holding the previous frame is the honest answer, and drawing fullscreen would
            # dress a data mistake up as a deliberate picture.
            return InstancedOutcome(
                "", "refused", detail="the script engine refused this population"
            )
        if not self.entity_fields or not instances or self.program is None:
            return InstancedOutcome("", "fullscreen")
        count, problem = validate_population(self.entity_fields, instances)
        if problem is not None:
            return InstancedOutcome("", "refused", detail=problem)
        if count == 0:
            return InstancedOutcome("", "empty", count=0)
        for entity_field in self.entity_fields:
            column = instances[entity_field.name]
            buffer = self.instance_buffers.get(entity_field.name)
            if buffer is None or buffer.size < column.nbytes:
                if buffer is not None:
                    buffer.release()
                # Double rather than fit exactly: a population that grows by one entity a
                # frame would otherwise reallocate every frame, and a grow drops the
                # buffer's contents.
                buffer = self._gl.buffer(reserve=max(column.nbytes * 2, 1024))
                self.instance_buffers[entity_field.name] = buffer
                self.vao = None
            buffer.write(column)
        if self.vao is None:
            self._build_instanced_vao()
        return InstancedOutcome("", "drew", count=count)

    def _build_instanced_vao(self) -> None:
        # Rebuilt only when a buffer is replaced, never per frame: the instance COUNT is a
        # draw argument, so a population that grows and shrinks inside its capacity needs
        # no new VAO.
        assert self.program is not None and self.vbo is not None
        content: list[tuple[moderngl.Buffer, str, str]] = [
            (self.vbo, "2f", CORNER_ATTRIBUTE)
        ]
        for entity_field in self.entity_fields:
            attribute = f"a_{entity_field.name}"
            if attribute not in self.program:
                # The driver DEAD-STRIPS an attribute the fragment shader does not read
                # yet, so the linked program has no `a_mass` while the declaration
                # plainly does -- and binding a name it lacks raises. Declaring a field
                # and reading it in the next edit is the ordinary step, so it keeps its
                # buffer and binds nothing until the body uses it.
                continue
            content.append(
                (
                    self.instance_buffers[entity_field.name],
                    f"{_ATTRIBUTE_FORMATS[entity_field.glsl_type]}/i",
                    attribute,
                )
            )
        self.vao = self._gl.vertex_array(self.program, content)

    def restart_video_uniforms(self) -> None:
        for uniform in self.get_active_uniforms():
            video = self.uniform_values.get(uniform.name)
            if isinstance(video, Video):
                video.restart()
                logger.debug(f"Video uniform '{uniform.name}' restarted")
