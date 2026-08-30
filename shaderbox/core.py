import base64
import contextlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import imageio
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
    MEDIA_DIR_NAME,
    MP4_CRF_VALUES,
    MP4_PRESETS,
    TEXTURES_DIR_NAME,
    VIDEO_RESOLUTION_ALIGNMENT,
    WEBM_CPU_USED_VALUES,
    WEBM_CRF_VALUES,
)
from shaderbox.glyph_tables import TABLE_UNIFORMS
from shaderbox.media import (
    Image,
    MediaDetails,
    MediaWithTexture,
    Video,
    media_class_for,
    texture_to_pil,
    texture_to_rgba8,
)
from shaderbox.paths import NODE_JSON_BASENAME, NODE_SHADER_BASENAME
from shaderbox.render_preset import FitPolicy, RenderPreset, resolve_dims
from shaderbox.shader_errors import ShaderError, SourceMap, parse_shader_errors
from shaderbox.shader_lib import active as active_lib_index
from shaderbox.shader_lib import resolve_usage
from shaderbox.shader_source import ShaderSource
from shaderbox.step_spec import (
    STEP_OUT_NAME,
    USER_MAIN_ALIAS,
    StepConfig,
    StepPlan,
    StepSpec,
    find_steps,
    plan_steps,
)
from shaderbox.util import try_to_release

# Engine-driven: never node-intrinsic defaults — seed_uniform_values skips them and
# UINode.save excludes them. Two kinds: per-frame values Node.render() recomputes
# from time/canvas, and the program-resident glyph tables Node.compile() writes once
# (TABLE_UNIFORMS — render() skips those entirely).
ENGINE_DRIVEN_UNIFORMS: frozenset[str] = frozenset(
    {"u_time", "u_aspect", "u_resolution"} | TABLE_UNIFORMS.keys()
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


@dataclass(frozen=True)
class StepView:
    """One step's facts, as the UI sees them. Read-only by construction."""

    name: str
    order_index: int
    sampler: str
    size: tuple[int, int]
    dtype: str
    filter_linear: bool
    wrap: bool
    persist: bool
    reads: list[str]
    reads_self: bool
    read_by_output: bool
    texture_glo: int | None


UniformValue = (
    int
    | float
    | Sequence[int]
    | Sequence[float]
    | MediaWithTexture
    | moderngl.Texture
    | moderngl.Buffer
)


class Node:
    _DEFAULT_VS_FILE_PATH = DEFAULT_VS_FILE_PATH
    _DEFAULT_FS_FILE_PATH = DEFAULT_FS_FILE_PATH
    _DEFAULT_IMAGE_FILE_PATH = DEFAULT_IMAGE_FILE_PATH

    def __init__(
        self,
        gl: moderngl.Context | None = None,
        source: ShaderSource | None = None,
        canvas_size: tuple[int, int] | None = None,
    ) -> None:
        self._gl = gl or moderngl.get_context()
        self.vs_source: str = self._DEFAULT_VS_FILE_PATH.read_text(encoding="utf-8")
        self.source: ShaderSource = (
            source
            if source is not None
            else ShaderSource.load(self._DEFAULT_FS_FILE_PATH)
        )

        self.canvas = Canvas(size=canvas_size, gl=self._gl)

        self.uniform_values: dict[str, Any] = {}
        # The CPU-script engine tick (feature 041), injected by ProjectSession at load. Fired
        # ONLY from the export loops below (per frame), NEVER from render() — the live path ticks
        # once via session.tick() in ui.py, so firing it in render() would double-tick the frame.
        self.on_pre_render: Callable[[float, float, int], None] | None = None
        # Export-time script isolation (feature 041), injected by ProjectSession at load. render_media
        # enters it around EVERY export so a stateful script ticks from a FRESH per-export instance
        # (not the live-warmed one) — structural, so no export caller can forget to isolate. Default
        # nullcontext when no session injects it (a bare Node / no scripts). Node stays engine-free:
        # it only enters an opaque injected context manager (same shape as on_pre_render).
        self.export_isolation: Callable[[], contextlib.AbstractContextManager[None]] = (
            contextlib.nullcontext
        )
        self.compile_unit: CompileUnit = CompileUnit.empty(self.source)
        self.program: moderngl.Program | None = None
        self.vbo: moderngl.Buffer | None = None
        self.vao: moderngl.VertexArray | None = None
        # Multi-step state (064). Empty for the overwhelming majority of nodes, which
        # declare no steps and behave exactly as they did before.
        self.steps: list[StepSpec] = []
        # Per-step target configuration, keyed by step name. Node state the panel edits,
        # NOT shader text -- the shader says what the steps are and how they connect,
        # this says how each target is set up. A step with no entry gets the defaults, so
        # a freshly-declared step renders correctly before anyone opens the panel.
        self.step_configs: dict[str, StepConfig] = {}
        self.step_plan: StepPlan = StepPlan(
            order=[], reads={}, self_reads=set(), final_reads=set()
        )
        self._step_programs: dict[str, moderngl.Program] = {}
        self._step_vaos: dict[str, moderngl.VertexArray] = {}
        # Two canvases per self-reading step (D6 ping-pong); one otherwise. `_step_front`
        # names the canvas holding the CURRENT frame's output.
        self._step_targets: dict[str, tuple[Canvas, Canvas | None]] = {}
        self._step_front: dict[str, int] = {}

    @classmethod
    def load_from_dir(
        cls,
        node_dir: Path | str,
        gl: moderngl.Context | None = None,
    ) -> tuple["Node", dict[str, Any]]:
        node_dir = Path(node_dir)
        with (node_dir / NODE_JSON_BASENAME).open() as f:
            metadata = json.load(f)

        node = Node(
            gl=gl,
            source=ShaderSource.load(node_dir / NODE_SHADER_BASENAME),
            canvas_size=metadata.get("canvas_size"),
        )

        # ----------------------------------------------------------------
        for uniform_name, value in metadata["uniforms"].items():
            if isinstance(value, dict):
                local_file_path = value.get("file_path")
                value_base64 = value.get("base64")

                if local_file_path is not None:
                    file_path = node_dir / local_file_path
                    dir_name = file_path.parent.name

                    if dir_name == MEDIA_DIR_NAME:
                        value = media_class_for(file_path.suffix)(file_path)
                    elif dir_name == TEXTURES_DIR_NAME:
                        data = file_path.read_bytes()
                        value = node._gl.texture(
                            size=value["size"],
                            components=value["components"],
                            data=data,
                            dtype=value.get("dtype", "f1"),
                        )
                    else:
                        raise ValueError(
                            f"Failed to load uniform data from dir '{dir_name}': it should be stored in '{MEDIA_DIR_NAME}' or '{TEXTURES_DIR_NAME}' dir"
                        )
                elif value_base64 is not None:
                    value_bytes = base64.b64decode(value_base64)
                    value = node._gl.buffer(value_bytes)
                else:
                    raise ValueError("Unknown uniform dict format")

            elif isinstance(value, list):
                value = tuple(value)

            node.uniform_values[uniform_name] = value

        node.render()  # warm-up
        return node, metadata

    def release_program(self, new_fs_source: str = "") -> None:
        # Path is the stable identity; only text + mtime change.
        self.source = replace(self.source, text=new_fs_source, mtime=self.source.mtime)
        self.invalidate()

    def invalidate(self) -> None:
        # Drop the cached GL program + compile unit without touching `self.source`;
        # next compile() re-reads included lib files via the resolver.
        self.compile_unit = CompileUnit.empty(self.source)
        # A recompile can change the step set, so a target with no step left to own it
        # would leak. `persist` is honoured HERE and only here -- it means "survives a
        # recompile", not "survives a reload".
        self._release_step_gl(keep_persist=True)
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
        self._release_step_gl()
        # The node OWNS its uniform values: the Image/Video bound to a sampler (each holding a
        # texture, and a Video an open capture), the default Image, and the uniform-block Buffer.
        # Without this every reload (the file watcher, a revert, a project switch) leaks them.
        for value in self.uniform_values.values():
            try_to_release(value)
        self.uniform_values.clear()
        self.canvas.release()

    def is_step_sampler(self, name: str) -> bool:
        """Is `name` a sampler the engine drives from a step's output?

        THE guard for D11: a step sampler must never reach the ordinary sampler surface.
        Its value is a bare `moderngl.Texture` the engine owns, so letting it into
        `uniform_values` would make `UINode.save` write megabytes of transient float
        target to `textures/*.bin` (its raw-Texture branch accepts exactly that type),
        make `tabs/node.py` raise on `value.texture.size`, offer the user a "Load media"
        button that overwrites the chain wiring, and hand `release()` a texture that
        `invalidate()` frees again. One predicate, consulted by every one of those.
        """
        return any(step.sampler == name for step in self.steps)

    def step_texture(self, step_name: str) -> moderngl.Texture | None:
        """The texture holding `step_name`'s CURRENT frame output, if it has one."""
        pair = self._step_targets.get(step_name)
        if pair is None:
            return None
        # Index 1 exists only for a ping-pong pair, and `_step_front` only ever names it
        # when it does -- but that is an invariant the type cannot carry, so fall back
        # to the front rather than suppress.
        front = pair[self._step_front.get(step_name, 0)] or pair[0]
        return front.texture

    def step_views(self) -> list["StepView"]:
        """Read-only facts about each step, in evaluation order.

        The UI's whole window onto the chain. Returning a value object rather than
        letting a widget walk `_step_targets` keeps the panel from depending on how the
        engine stores its targets -- the ping-pong pair, the front index and the
        program map stay private, and the surface can be rewritten without touching
        either side.
        """
        by_name = {step.name: step for step in self.steps}
        views: list[StepView] = []
        for order_index, name in enumerate(self.step_plan.order):
            spec = by_name.get(name)
            if spec is None:
                continue
            texture = self.step_texture(name)
            views.append(
                StepView(
                    name=name,
                    order_index=order_index,
                    sampler=spec.sampler,
                    size=texture.size
                    if texture
                    else spec.target_size(self.canvas.texture.size),
                    dtype=spec.config.dtype,
                    filter_linear=spec.config.filter_linear,
                    wrap=spec.config.wrap,
                    persist=spec.config.persist,
                    reads=sorted(self.step_plan.reads.get(name, set())),
                    reads_self=name in self.step_plan.self_reads,
                    read_by_output=name in self.step_plan.final_reads,
                    texture_glo=texture.glo if texture else None,
                )
            )
        return views

    def _release_step_gl(self, keep_persist: bool = False) -> None:
        for program in self._step_programs.values():
            program.release()
        self._step_programs.clear()
        for vao in self._step_vaos.values():
            vao.release()
        self._step_vaos.clear()
        persist_names = (
            {st.name for st in self.steps if st.config.persist}
            if keep_persist
            else set()
        )
        for name in list(self._step_targets):
            if name in persist_names:
                continue
            front, back = self._step_targets.pop(name)
            front.release()
            if back is not None:
                back.release()
            self._step_front.pop(name, None)

    def _reset_step_targets(self) -> None:
        """Clear every step target to black, so an export starts from a defined state."""
        for front, back in self._step_targets.values():
            for canvas in (front, back):
                if canvas is None:
                    continue
                canvas.fbo.use()
                self._gl.clear()
        self._step_front = dict.fromkeys(self._step_front, 0)

    def _make_step_canvas(self, spec: StepSpec, size: tuple[int, int]) -> "Canvas":
        filter_mode = moderngl.LINEAR if spec.config.filter_linear else moderngl.NEAREST
        return Canvas(
            gl=self._gl,
            size=size,
            dtype=spec.config.dtype,
            filter=(filter_mode, filter_mode),
            wrap=spec.config.wrap,
        )

    def _sync_step_targets(self, canvas_size: tuple[int, int]) -> None:
        """Allocate/resize one target per step (two for a self-reader).

        Sized off the NODE's canvas, never a caller-supplied one (D12): `ui.py` renders
        the current node a second time into a ~200px preview canvas, and sizing off that
        would reallocate every target twice a frame and discard the ping-pong history
        R6 depends on.
        """
        wanted = {s.name: s for s in self.steps}
        for name in list(self._step_targets):
            if name not in wanted:
                front, back = self._step_targets.pop(name)
                front.release()
                if back is not None:
                    back.release()
                self._step_front.pop(name, None)

        for name, spec in wanted.items():
            size = spec.target_size(canvas_size)
            needs_pair = name in self.step_plan.self_reads
            pair = self._step_targets.get(name)
            if pair is not None:
                front, back = pair
                fits = (
                    front.texture.size == size
                    and front.dtype == spec.config.dtype
                    and (back is not None) == needs_pair
                )
                if fits:
                    continue
                front.release()
                if back is not None:
                    back.release()

            self._step_targets[name] = (
                self._make_step_canvas(spec, size),
                self._make_step_canvas(spec, size) if needs_pair else None,
            )
            self._step_front[name] = 0

    def get_active_uniforms(self) -> list[moderngl.Uniform | moderngl.UniformBlock]:
        """Every uniform active in ANY variant, first-seen order (D4).

        Each variant exposes only the uniforms its own branch uses -- measured: a final
        variant reporting ['u_blur','u_gain'] beside a step variant reporting
        ['u_radius']. So with N programs "the live program" is undefined, and since
        `UINode.save` prunes UI rows against this set, anything missing here has its
        tuned value silently deleted on the next save.

        Consequence, and it is the intended behaviour: a name declared in two steps is
        ONE row driving both. A shared `u_ray_count` across eight cascade levels is the
        ergonomic win the feature exists for; a step cannot have a private uniform that
        merely shares a name.
        """
        uniforms: list[moderngl.Uniform | moderngl.UniformBlock] = []
        seen: set[str] = set()
        programs = [self.program, *self._step_programs.values()]
        for program in programs:
            if program is None:
                continue
            for uniform_name in program:
                if uniform_name in seen:
                    continue
                uniform = program[uniform_name]
                if isinstance(uniform, moderngl.Uniform | moderngl.UniformBlock):
                    seen.add(uniform_name)
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

        parsed = find_steps(self.source.text, self.source.path, self.step_configs)
        if parsed.errors:
            # D14: refuse the whole compile. Building the final variant alone would leave
            # the step samplers as ordinary textures bound to the shipped default image --
            # a picture that looks fine and is wrong, which is what D2 exists to prevent.
            unit.errors.extend(parsed.errors)
            unit.error_raw = "\n".join(e.message for e in parsed.errors)
            if unit.error_raw != self.compile_unit.error_raw:
                logger.error(f"Failed to parse steps: {unit.error_raw}")
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

        self._build_step_variants(unit, parsed.steps)

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
        # Fill uniform_values with node-intrinsic defaults for any active uniform not yet
        # present. GL-FREE: no texture.use / program binding / draw — that is render()'s job.
        # Engine-driven uniforms are per-frame canvas/time values, valued only in render().
        if not self.program:
            return
        for uniform in self.get_active_uniforms():
            if uniform.name in ENGINE_DRIVEN_UNIFORMS:
                continue
            # D11: the engine owns a step sampler's texture. Seeding it would put a
            # bare moderngl.Texture into uniform_values, which UINode.save writes to
            # textures/*.bin -- megabytes of transient float target, reloaded next
            # session as a frozen stale frame.
            if self.is_step_sampler(uniform.name):
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

    def _step_variant_source(self, flattened: str, step: StepSpec) -> str:
        """Alias the user's `main` out of the way and append a dispatcher for one step.

        The engine never modifies user text -- it only brackets it. A textual rename of
        `void main()` breaks on unusual formatting; duplicating the body inside an `#if`
        doubles the error-line mapping. The C preprocessor substitutes whole tokens only,
        so a shader declaring `u_main_scale` or `domain_warp` is untouched (verified).

        The epilogue carries its own `#line` pointing at the step's DECLARATION, so an
        error in the generated dispatcher lands somewhere the user can act on instead of
        past the end of the file.
        """
        lines = flattened.split("\n")
        version_idx = 0
        for i, line in enumerate(lines):
            if line.lstrip().startswith("#version"):
                version_idx = i
                break
        head = lines[: version_idx + 1]
        rest = lines[version_idx + 1 :]
        decl_line = self._declaration_line(step) + 1
        return "\n".join(
            [
                *head,
                f"#define main {USER_MAIN_ALIAS}",
                *rest,
                "#undef main",
                f"#line {decl_line} 0",
                f"out vec4 {STEP_OUT_NAME};",
                f"void main() {{ {step.fn_name}({STEP_OUT_NAME}); }}",
            ]
        )

    def _declaration_line(self, step: StepSpec) -> int:
        for i, line in enumerate(self.source.text.splitlines()):
            if step.sampler in line and "step" in line:
                return i
        return 0

    def _build_step_variants(self, unit: CompileUnit, steps: list[StepSpec]) -> None:
        # `self.steps` first: _release_step_gl reads it to decide what `persist` covers,
        # and the answer must come from the NEW declarations. Freeing before the swap
        # would drop a persisted target on every recompile -- and since every real flow
        # (Ctrl+S, hot-reload, a copilot edit) is invalidate-then-compile, that made the
        # flag a no-op no matter what invalidate preserved.
        self.steps = steps
        self._release_step_gl(keep_persist=True)
        if not steps:
            self.step_plan = StepPlan(
                order=[], reads={}, self_reads=set(), final_reads=set()
            )
            return

        assert self.vbo is not None
        seen_errors: set[str] = set()
        for step in steps:
            try:
                step_program = self._gl.program(
                    vertex_shader=self.vs_source,
                    fragment_shader=self._step_variant_source(unit.flattened, step),
                )
            except Exception as e:
                err = str(e)
                # One broken line breaks every variant; the strip caps at 3 rows, so
                # reporting the same message N times would bury everything else.
                if err not in seen_errors:
                    seen_errors.add(err)
                    unit.errors.extend(parse_shader_errors(err, unit.source_map))
                    unit.error_raw = err
                continue
            self._step_programs[step.name] = step_program
            self._step_vaos[step.name] = self._gl.vertex_array(
                step_program, [(self.vbo, "2f", "a_pos")]
            )

        # D5: the driver already resolved every read exactly -- through `#define`s,
        # helpers and sampler parameters alike. A text scan cannot, and a missed edge
        # renders a plausible frame that is one hop stale.
        sampler_names = {s.sampler for s in steps}
        active_by_step: dict[str, set[str]] = {
            name: {u for u in prog if u in sampler_names}
            for name, prog in self._step_programs.items()
        }
        assert self.program is not None
        active_by_step[""] = {u for u in self.program if u in sampler_names}

        self.step_plan, plan_errors = plan_steps(
            self.source.text, steps, self.source.path, active_by_step=active_by_step
        )
        unit.errors.extend(plan_errors)
        if plan_errors and not unit.error_raw:
            unit.error_raw = "\n".join(e.message for e in plan_errors)

    def _draw_step(self, name: str, render_time: float, advance_state: bool) -> None:
        program = self._step_programs.get(name)
        vao = self._step_vaos.get(name)
        pair = self._step_targets.get(name)
        if program is None or vao is None or pair is None:
            return

        front_idx = self._step_front.get(name, 0)
        # A self-reading step writes into the BACK buffer while reading the front, so
        # its own previous frame stays intact for the whole draw.
        write_idx = (1 - front_idx) if pair[1] is not None else front_idx
        target = pair[write_idx] or pair[0]

        unit = self._bind_step_inputs(program, name, render_time)
        self._bind_engine_uniforms(program, target, render_time, unit)

        target.fbo.use()
        self._gl.clear()
        vao.render(moderngl.TRIANGLE_STRIP)

        if pair[1] is not None and advance_state:
            self._step_front[name] = write_idx

    def _bind_step_inputs(
        self, program: moderngl.Program, name: str, render_time: float
    ) -> int:
        """Bind every step sampler this program reads; return the next free unit."""
        unit = 0
        for spec in self.steps:
            if spec.sampler not in program:
                continue
            pair = self._step_targets.get(spec.name)
            if pair is None:
                continue
            # Always the FRONT buffer: for a self-reader that is its previous frame
            # (the draw writes the back), and for everyone else it is this frame's
            # output, already drawn because the order puts producers first.
            front = pair[self._step_front.get(spec.name, 0)] or pair[0]
            front.texture.use(location=unit)
            with contextlib.suppress(Exception):
                program[spec.sampler] = unit
            unit += 1
        return unit

    def _bind_engine_uniforms(
        self,
        program: moderngl.Program,
        target: Canvas,
        render_time: float,
        texture_unit: int,
    ) -> None:
        """Engine-driven values plus the user's own uniforms, for one step program."""
        for uniform_name in program:
            member = program[uniform_name]
            if not isinstance(member, moderngl.Uniform | moderngl.UniformBlock):
                continue
            if uniform_name in TABLE_UNIFORMS or self.is_step_sampler(uniform_name):
                continue
            value: Any
            if uniform_name == "u_time":
                value = render_time
            elif uniform_name == "u_aspect":
                value = np.divide(*target.texture.size)
            elif uniform_name == "u_resolution":
                value = target.texture.size
            else:
                value = self.uniform_values.get(uniform_name)
                if value is None:
                    continue
                if isinstance(value, moderngl.UniformBlock):
                    continue
                if isinstance(value, MediaWithTexture):
                    value.update(render_time)
                    value.texture.use(location=texture_unit)
                    value = texture_unit
                    texture_unit += 1
                elif isinstance(value, moderngl.Texture):
                    value.use(location=texture_unit)
                    value = texture_unit
                    texture_unit += 1
            with contextlib.suppress(Exception):
                program[uniform_name] = value

    def _render_chain(self, render_time: float, advance_state: bool) -> None:
        """Evaluate every step once, in dependency order (D5).

        Memoization is the order itself: `step_plan.order` already lists each step
        exactly once, so a diamond's shared ancestor draws once rather than once per
        consumer -- the half the repo's own deleted DAG got wrong.
        """
        self._sync_step_targets(self.canvas.texture.size)
        for name in self.step_plan.order:
            self._draw_step(name, render_time, advance_state)

    def render(
        self,
        u_time: float | None = None,
        canvas: Canvas | None = None,
        advance_state: bool = True,
    ) -> None:
        """Draw the node. With steps declared, evaluate the whole chain first.

        `advance_state=False` renders without advancing ping-pong (D13). The live loop
        draws the current node TWICE per frame -- once into a small preview canvas, once
        into its own -- and the copilot probe renders twice back to back. Advancing per
        CALL would run a feedback step at 2x on the focused node and 1x elsewhere, so a
        decay constant tuned while a node is selected evolves at half the rate once it
        is not, and the probe's second frame would carry the first's accumulation and
        report ANIMATES for a static chain.
        """
        canvas = canvas or self.canvas

        if not self.program or not self.vbo or not self.vao:
            self.compile()

        if not self.program or not self.vao:
            return

        render_time_for_chain = u_time if u_time is not None else time.monotonic()
        if self.steps:
            self._render_chain(render_time_for_chain, advance_state)

        texture_unit = 0
        # No glfw here: this module is imported by the headless core. The live loop renders
        # bare, so it falls through to the monotonic clock; export and the probe pass u_time.
        render_time = u_time if u_time is not None else time.monotonic()
        self.seed_uniform_values()
        for uniform in self.get_active_uniforms():
            if uniform.name in TABLE_UNIFORMS:  # program-resident, set at compile
                continue
            value = self.uniform_values.get(uniform.name)

            value_for_program = None

            if isinstance(uniform, moderngl.UniformBlock):
                assert isinstance(value, moderngl.Buffer)
                value.bind_to_uniform_block(uniform.index)

            elif getattr(uniform, "gl_type", None) == GL_SAMPLER_2D:
                if self.is_step_sampler(uniform.name):
                    # The engine owns this one: hand it the step's output and skip the
                    # uniform_values round-trip entirely (D11).
                    step_texture = self.step_texture(
                        next(s.name for s in self.steps if s.sampler == uniform.name)
                    )
                    if step_texture is None:
                        continue
                    step_texture.use(location=texture_unit)
                    with contextlib.suppress(Exception):
                        self.program[uniform.name] = texture_unit
                    texture_unit += 1
                    continue
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

            else:
                value_for_program = value

            self.uniform_values[uniform.name] = value

            if value_for_program is not None:
                if uniform.name not in self.program:
                    # Lives in a step variant, not the final program (D4's union). Its
                    # value is real and the step render writes it; popping here would
                    # delete the user's tuned value every frame.
                    continue
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

    def _render_image(
        self, details: MediaDetails, canvas: "Canvas", u_time: float | None = 0.0
    ) -> MediaDetails:
        file_path = Path(details.file_details.path)
        t = u_time if u_time is not None else 0.0
        if self.on_pre_render is not None:
            self.on_pre_render(t, 0.0, 0)
        self.render(u_time=t, canvas=canvas)

        pil_image = texture_to_pil(canvas.texture)
        if canvas.texture.size != (
            details.resolution_details.width,
            details.resolution_details.height,
        ):
            pil_image = pil_image.resize(
                (details.resolution_details.width, details.resolution_details.height)
            )

        pil_image.save(file_path)
        logger.info(f"Image saved: {file_path}")

        rendered_image = Image(file_path)
        rendered_details = rendered_image.details
        rendered_image.release()

        return rendered_details

    def _render_video(self, details: MediaDetails, canvas: "Canvas") -> MediaDetails:
        file_path = Path(details.file_details.path)
        extension = file_path.suffix
        width = details.resolution_details.width
        height = details.resolution_details.height

        # Ensure resolution is divisible by alignment for codec compatibility
        alignment = VIDEO_RESOLUTION_ALIGNMENT
        width = (width + alignment - 1) // alignment * alignment
        height = (height + alignment - 1) // alignment * alignment

        # Canvas already at the requested size → let ffmpeg copy 1:1, no -s rescale.
        scale_params: list[str] = (
            []
            if canvas.texture.size == (width, height)
            else ["-s", f"{width}x{height}"]
        )

        if extension == ".mp4":
            codec = "libx264"
            pixelformat = "yuv420p"
            crf = MP4_CRF_VALUES[details.quality]
            preset = MP4_PRESETS[details.quality]
            ffmpeg_params = [
                "-crf",
                str(crf),
                "-preset",
                preset,
            ]
        elif extension == ".webm":
            codec = "libvpx-vp9"
            pixelformat = "yuva420p"
            crf = WEBM_CRF_VALUES[details.quality]
            cpu_used = WEBM_CPU_USED_VALUES[details.quality]
            ffmpeg_params = [
                "-crf",
                str(crf),
                "-b:v",
                "0",
                "-cpu-used",
                str(cpu_used),
                "-deadline",
                "realtime",
                "-threads",
                "0",
                "-auto-alt-ref",
                "0",
                "-an",
            ]
        else:
            raise ValueError(
                f"Unsupported extension: {extension}, only .mp4 and .webm are allowed"
            )

        writer = imageio.get_writer(
            file_path,
            fps=details.fps,
            codec=codec,
            ffmpeg_params=ffmpeg_params,
            pixelformat=pixelformat,
            input_params=["-pixel_format", "bgra"],
            output_params=scale_params,
        )

        self.restart_video_uniforms()
        n_frames = int(details.duration * details.fps)
        dt = 1.0 / details.fps
        try:
            for i in range(n_frames):
                if self.on_pre_render is not None:
                    self.on_pre_render(i / details.fps, dt, i)
                self.render(i / details.fps, canvas=canvas)

                frame = np.flipud(texture_to_rgba8(canvas.texture))
                writer.append_data(frame)
        except Exception:
            # Close ffmpeg's pipe, then drop the partial file: a half-written .mp4 left on disk is
            # indistinguishable from a finished export.
            with contextlib.suppress(Exception):
                writer.close()
            Path(file_path).unlink(missing_ok=True)
            raise
        writer.close()
        logger.info(f"Video saved: {details.file_details.path}")

        rendered_video = Video(file_path)
        rendered_details = rendered_video.details
        rendered_details.quality = details.quality
        rendered_video.release()

        return rendered_details

    def render_media(
        self, details: MediaDetails, preset: RenderPreset | None = None
    ) -> MediaDetails:
        # Every export funnels through here (Render tab / Share scratch / copilot tools), so the
        # script-isolation bracket lives here ONCE — no export caller can bypass it (feature 041).
        # Step targets get the same treatment for the same reason: a feedback step carries however
        # long the app has been open, so without this the same node exported twice differs. Sited
        # here rather than in export_isolation because the targets are Node's own state, and that
        # context manager is a ProjectSession hook a bare Node does not have.
        self._reset_step_targets()
        with self.export_isolation():
            if preset is None or preset.fit is FitPolicy.SCALE_DISTORT:
                canvas = self.canvas
                return self._render_media_into(details, canvas)

            target_w, target_h = resolve_dims(preset, self.canvas.texture.size)
            details = details.model_copy(deep=True)
            details.resolution_details.width = target_w
            details.resolution_details.height = target_h

            target = Canvas(gl=self._gl, size=(target_w, target_h))
            try:
                return self._render_media_into(details, target)
            finally:
                target.release()

    def _render_media_into(
        self, details: MediaDetails, canvas: "Canvas"
    ) -> MediaDetails:
        if details.is_video:
            return self._render_video(details, canvas)
        else:
            return self._render_image(details, canvas)
