"""The document: what you open, save, export -- and the passes it renders through (065).

A document owns its passes, the script hook the engine ticks, and export. One pass is its
output. Everything about ONE shader -- source, program, target, uniforms, compile, draw --
belongs to `core.Pass`, one layer down.

Stage 2 of 065: the split is made, and a document still holds exactly one pass. Stage 3 gives it
the graph.
"""

import base64
import contextlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

import imageio
import moderngl
import numpy as np
from loguru import logger
from pydantic import BaseModel, ValidationError

from shaderbox.constants import (
    DEFAULT_CANVAS_SIZE,
    MEDIA_DIR_NAME,
    MP4_CRF_VALUES,
    MP4_PRESETS,
    TEXTURES_DIR_NAME,
    VIDEO_RESOLUTION_ALIGNMENT,
    WEBM_CPU_USED_VALUES,
    WEBM_CRF_VALUES,
)
from shaderbox.core import Canvas, Pass
from shaderbox.media import (
    Image,
    MediaDetails,
    Video,
    media_class_for,
    texture_to_pil,
    texture_to_rgba8,
)
from shaderbox.model_salvage import drop_invalid, drop_unknown
from shaderbox.pass_graph import (
    GraphError,
    PassEntry,
    PassGraph,
    plan_for_output,
    plan_passes,
)
from shaderbox.paths import (
    DOCUMENT_JSON_BASENAME,
    GRAPH_JSON_BASENAME,
    PASS_SHADER_SUFFIX,
    PASSES_DIR_NAME,
    pass_name_of,
)
from shaderbox.render_preset import FitPolicy, RenderPreset, resolve_dims
from shaderbox.shader_source import ShaderSource

DEFAULT_PASS_NAME = "main"


def _as_canvas_size(size: object) -> tuple[int, int] | None:
    """Coerce a loaded `canvas_size` to a tuple, or None if it is not a usable pair.

    `document.json` stores this as a JSON LIST, and a list is unhashable and never equals a
    tuple -- so an unconverted value breaks every `size in seen` membership test and every
    unchanged-size guard downstream, on disk-loaded documents only. A pair that is malformed
    (wrong length, a non-integer) degrades to the default rather than raising, like every other
    field this loader reads.
    """
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return None
    w, h = size
    if not isinstance(w, int) or not isinstance(h, int):
        return None
    return (w, h)


def _keyed_entry_fields() -> dict[str, type[BaseModel]]:
    """PassGraph's `dict[str, <Model>]` fields, as {field name: element model}."""
    fields: dict[str, type[BaseModel]] = {}
    for name, field in PassGraph.model_fields.items():
        args = get_args(field.annotation)
        element = args[-1] if args else None
        if isinstance(element, type) and issubclass(element, BaseModel):
            fields[name] = element
    return fields


def load_graph(path: Path) -> PassGraph:
    """Read `graph.json`, salvaging per key.

    A malformed entry costs THAT pass's wiring, never the document: the pass still loads, still
    compiles and still draws — it just starts unwired, which the panel can fix. A file that is
    absent (or is not an object at all) yields an empty graph, which `load_from_dir` then fills
    with a default entry per pass file it found.
    """
    if not path.is_file():
        return PassGraph()
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Unreadable {path.name} ({e}); the document loads unwired")
        return PassGraph()
    if not isinstance(data, dict):
        logger.warning(
            f"Malformed {path.name} (not an object); the document loads unwired"
        )
        return PassGraph()

    # Per ENTRY, before the whole-model salvage: every dict-of-model field is keyed by pass name,
    # so one bad entry validated as part of the whole would cost every sibling. Enumerated from
    # the model rather than listed by hand — a field added later inherits the salvage instead of
    # waiting for someone to remember it.
    for key, model in _keyed_entry_fields().items():
        entries = data.get(key)
        if not isinstance(entries, dict):
            data.pop(key, None)
            continue
        for name in list(entries):
            row = entries[name]
            if not isinstance(row, dict):
                logger.warning(f"{path.name}: dropping malformed {key} entry '{name}'")
                entries.pop(name)
                continue
            drop_unknown(model, row, f"{path.name}.{key}.{name}")
            drop_invalid(model, row, f"{path.name}.{key}.{name}")
            try:
                model(**row)
            except ValidationError as e:
                logger.warning(
                    f"{path.name}: dropping invalid {key} entry '{name}' ({e})"
                )
                entries.pop(name)

    # The keyed-dict fields are already salvaged per entry above, and they must be held OUT of
    # the whole-model pass: `drop_unknown` walks a nested model's FIELD names, so it reads every
    # pass NAME as an unknown key and prunes the entire graph to empty.
    keyed = {key: data.pop(key) for key in _keyed_entry_fields() if key in data}
    drop_unknown(PassGraph, data, path.name)
    drop_invalid(PassGraph, data, path.name)
    data.update(keyed)
    try:
        return PassGraph(**data)
    except ValidationError as e:
        logger.warning(f"Incompatible {path.name} ({e}); the document loads unwired")
        return PassGraph()


def _load_document_metadata(path: Path) -> dict[str, Any]:
    """Read `document.json`, degrading to defaults rather than raising.

    Symmetric with `load_graph` beside it, and for the same reason: this runs from the live
    per-frame sync (`ProjectSession.sync_documents_from_disk`), so a raise here escapes into the
    imgui frame loop and takes the app down -- the same shape as the crash that shipped when a
    `relative_to` raised inside a draw call. A document whose metadata is unreadable still opens
    with its shader files intact, which is what makes it fixable; nothing else recovers a
    document whose loader refused to run.
    """
    if not path.is_file():
        return {}
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            f"Unreadable {path.name} ({e}); the document loads with defaults"
        )
        return {}
    if not isinstance(data, dict):
        logger.warning(
            f"Malformed {path.name} (not an object); the document loads with defaults"
        )
        return {}
    return data


def _uniforms_by_pass(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    # `uniforms` is keyed by pass name, then by uniform name: each pass owns its uniforms (D4),
    # so two passes may legitimately both declare `u_tex` with different values.
    uniforms = metadata.get("uniforms")
    if not isinstance(uniforms, dict):
        return {}
    return {name: rows for name, rows in uniforms.items() if isinstance(rows, dict)}


def _load_uniform_value(gl: moderngl.Context, document_dir: Path, value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    if not isinstance(value, dict):
        return value

    local_file_path = value.get("file_path")
    value_base64 = value.get("base64")
    if local_file_path is not None:
        file_path = document_dir / local_file_path
        kind = Path(local_file_path).parts[0]
        if kind == MEDIA_DIR_NAME:
            return media_class_for(file_path.suffix)(file_path)
        if kind == TEXTURES_DIR_NAME:
            return gl.texture(
                size=value["size"],
                components=value["components"],
                data=file_path.read_bytes(),
                dtype=value.get("dtype", "f1"),
            )
        raise ValueError(
            f"asset must live under '{MEDIA_DIR_NAME}' or '{TEXTURES_DIR_NAME}', not '{kind}'"
        )
    if value_base64 is not None:
        return gl.buffer(base64.b64decode(value_base64))
    raise ValueError("unknown uniform dict format")


class Document:
    """A document: several passes forming a DAG, one of them the output.

    `passes` maps a pass name to its `Pass`; `graph` says which pass fills which input of which
    pass, how each target is configured, and which pass the preview and export show. A document
    with one pass is the ordinary case and needs no graph editing to work.
    """

    def __init__(
        self,
        gl: moderngl.Context | None = None,
        source: ShaderSource | None = None,
        canvas_size: tuple[int, int] | None = None,
    ) -> None:
        self._gl = gl or moderngl.get_context()
        # Normalized here and in `set_canvas_size` -- the field's only two writers -- so every
        # reader downstream gets a hashable, comparable pair whatever the loader handed in.
        self.canvas_size: tuple[int, int] = (
            _as_canvas_size(canvas_size) or DEFAULT_CANVAS_SIZE
        )
        self.passes: dict[str, Pass] = {
            DEFAULT_PASS_NAME: Pass(
                gl=self._gl, source=source, canvas_size=self.canvas_size
            )
        }
        self.graph: PassGraph = PassGraph(
            output=DEFAULT_PASS_NAME, passes={DEFAULT_PASS_NAME: PassEntry()}
        )
        # A feedback pass's previous frame. Allocated on demand by the first frame that needs
        # one, and swapped at the FRAME boundary (begin_frame), never per render call: the live
        # loop draws the current document twice per frame and the copilot probe twice back to
        # back, so a per-call swap would advance the history at 2x.
        self._feedback: dict[str, Canvas] = {}
        # Which Pass.target_generation each history was built from.
        self._feedback_generation: dict[str, int] = {}
        self._black: moderngl.Texture | None = None
        self._frame: int = -1
        self._graph_errors: list[GraphError] = []
        # Loading compiles nothing (066 D1), so the first render pays the pass compiles. The
        # live loop reads this to admit first renders one document per frame (066 D2); set on
        # ATTEMPT, not success, so a broken document cannot hog the budget forever — but only
        # for an OWN-canvas render: a probe/export into a foreign canvas leaves the pass
        # canvases (what the grid tile shows) unwritten, so it must not consume the budget.
        self.first_render_done: bool = False
        # The CPU-script engine tick (feature 041), injected by ProjectSession at load. Fired
        # ONLY from the export loops below (per frame), NEVER from render() — the live path ticks
        # once via session.tick() in ui.py, so firing it in render() would double-tick the frame.
        self.on_pre_render: Callable[[float, float, int], None] | None = None
        # Export-time script isolation (feature 041), injected by ProjectSession at load. render_media
        # enters it around EVERY export so a stateful script ticks from a FRESH per-export instance
        # (not the live-warmed one) — structural, so no export caller can forget to isolate. Default
        # nullcontext when no session injects it (a bare Document / no scripts). Document stays engine-free:
        # it only enters an opaque injected context manager (same shape as on_pre_render).
        self.export_isolation: Callable[[], contextlib.AbstractContextManager[None]] = (
            contextlib.nullcontext
        )

    @property
    def gl(self) -> moderngl.Context:
        return self._gl

    @property
    def render_pass(self) -> Pass:
        """The output pass — what the preview shows and what export renders (D10).

        Falls back to any pass when the graph's `output` names none that exists, so a document
        with a stale output still previews rather than raising in the frame loop.
        """
        name = self.graph.output_pass
        if name is not None and name in self.passes:
            return self.passes[name]
        return next(iter(self.passes.values()))

    @property
    def graph_errors(self) -> list[GraphError]:
        """Wiring errors from the last render — a cycle, or a pass an input names (D7)."""
        return list(self._graph_errors)

    def release(self) -> None:
        for render_pass in self.passes.values():
            render_pass.release()
        for canvas in self._feedback.values():
            canvas.release()
        self._feedback.clear()
        self._feedback_generation.clear()
        if self._black is not None:
            self._black.release()
            self._black = None

    def set_canvas_size(self, size: tuple[int, int]) -> None:
        """Resize the document: its output target now, its other passes on the next render.

        The single funnel, because `canvas_size` is what every other pass scales FROM. A caller
        that resized `render_pass.canvas` directly — which is what the copilot's set_canvas_size
        did — left this field stale, so the rest of the graph kept sizing off the old dimensions
        and the output sampled mismatched targets.
        """
        self.canvas_size = _as_canvas_size(size) or DEFAULT_CANVAS_SIZE
        self.render_pass.canvas.set_size(self.canvas_size)

    def begin_frame(self, frame: int | None = None) -> None:
        """Advance feedback history to `frame`, at most once per frame.

        Tied to the FRAME, not to a render call: the live loop renders the current document
        twice per frame (preview + own canvas) and the probe renders twice back to back, so a
        swap per call would advance a feedback pass at the wrong rate.

        Identity, not call count, is what makes that true: passing the frame number makes a
        second call for the same frame a no-op, so an extra call site cannot corrupt the history
        and the caller does not have to be the only one. `None` means "the next frame", for
        callers that have no counter of their own (the export loops, which own their sequence).
        """
        if frame is not None and frame == self._frame:
            return
        previous_frame = self._frame
        self._frame = frame if frame is not None else self._frame + 1
        for name in list(self._feedback):
            render_pass = self.passes.get(name)
            # Only a pass that DREW last frame has a new history to advance to. A pass the
            # sweep drew once and never again would otherwise alternate between its two
            # canvases every frame, strobing a tile that should hold still.
            if render_pass is not None and render_pass.drawn_frame != previous_frame:
                continue
            self._swap_feedback(name)

    def _swap_feedback(self, name: str) -> None:
        """Exchange `name`'s live canvas with its history, so reads see what writes just made.

        The one place the swap happens: per frame from `begin_frame`, and between iterations of
        an iterated pass (068 D5). A pass with no feedback history is a no-op, which is what lets
        the iteration loop call it unconditionally.
        """
        previous = self._feedback.get(name)
        render_pass = self.passes.get(name)
        if previous is None or render_pass is None:
            return
        self._feedback[name] = render_pass.canvas
        render_pass.canvas = previous

    @property
    def has_feedback(self) -> bool:
        # Whether the GRAPH declares any feedback pass (an entry naming itself as an input). Read
        # from the plan, never from `_feedback`: that dict is an allocation cache filled on demand
        # during render() and emptied by release/drop/reset_feedback, so a check over it would be
        # False before the first frame and False again the instant a clear runs.
        return bool(plan_passes(self.graph)[0].feedback)

    def reset_feedback(self) -> None:
        """Drop every feedback history, so the next frame starts from black.

        Export enters this (D10), and so does anything that wants a document to render as if the
        app had just opened.
        """
        for canvas in self._feedback.values():
            canvas.release()
        self._feedback.clear()
        self._feedback_generation.clear()
        self._frame = -1

    def _black_texture(self) -> moderngl.Texture:
        # One 1x1 zero texture for every unresolved input in the document.
        if self._black is None:
            self._black = self._gl.texture((1, 1), 4, data=b"\x00\x00\x00\xff")
        return self._black

    def drop_feedback(self, name: str) -> None:
        """Release `name`'s feedback history. Call when a pass is deleted or renamed.

        The history is keyed by pass NAME and lives here, not on the `Pass` -- so releasing the
        pass does not release its history, and renaming one leaves the history stranded under
        the old key while the next render allocates a second canvas under the new one.
        """
        canvas = self._feedback.pop(name, None)
        self._feedback_generation.pop(name, None)
        if canvas is not None:
            canvas.release()

    def _feedback_canvas(self, name: str) -> Canvas:
        # Born matching its pass's target, so the first frame reads black at the right size
        # rather than sampling a stale or mis-sized texture.
        canvas = self._feedback.get(name)
        live = self.passes[name].canvas
        # A target change reallocates the pass's LIVE canvas, and the next `begin_frame` SWAPS
        # it into the history -- so after one frame the pair holds one canvas of each format and
        # the pass samples its previous frame through the wrong one. Silent: no error, no crash,
        # wrong numbers. Comparing formats cannot resolve it (after the swap neither side is
        # obviously right), so the PASS counts its own format changes and the history is dropped
        # whenever it predates one.
        generation = self.passes[name].target_generation
        if canvas is not None and self._feedback_generation.get(name) != generation:
            self.drop_feedback(name)
            canvas = None
        if canvas is None:
            canvas = Canvas(
                gl=self._gl,
                size=live.texture.size,
                dtype=live.dtype,
                filter=live.filter,
                wrap=live.wrap,
            )
            self._feedback[name] = canvas
            self._feedback_generation[name] = generation
        elif canvas.texture.size != live.texture.size:
            canvas.set_size(live.texture.size)
        return canvas

    def render(
        self,
        u_time: float | None = None,
        canvas: Canvas | None = None,
        target: str | None = None,
    ) -> None:
        """Draw the document: every pass the output needs, in order, each exactly once.

        `canvas` overrides the OUTPUT pass's target only — intermediate passes always draw into
        their own, since that is what the next pass samples.

        `target` draws that pass and its ancestor chain instead of the graph output's, skipping
        whatever already drew this frame. The graph output still decides which pass keeps full
        size and which one may receive `canvas`.
        """
        if canvas is None and target is None:
            self.first_render_done = True
        resolved = target if target is not None else self.graph.output_pass
        output = self.graph.output_pass
        if resolved is None or resolved not in self.passes:
            self._graph_errors = plan_passes(self.graph)[1]
            return
        planned, self._graph_errors = plan_for_output(self.graph, resolved)
        order = [name for name in planned if name in self.passes]
        if not order:
            # A cycle, or an output nothing can reach: draw the output alone so a half-built
            # graph still shows its own shader instead of going blank.
            order = [resolved]
        for name in order:
            render_pass = self.passes[name]
            if (
                canvas is None
                and target is not None
                and render_pass.drawn_frame == self._frame
                and self._frame >= 0
            ):
                continue
            render_pass.drawn_frame = self._frame
            render_pass.first_render_done = True
            entry = self.graph.passes.get(name, PassEntry())
            # The document owns the canvas size, so it applies each pass's scale — a pass cannot
            # size itself from a number it does not hold, and doing it in both places would fight.
            # The OUTPUT keeps full size: it is what the preview and export read.
            if name != output:
                wanted = entry.target.target_size(self.canvas_size)
                if render_pass.canvas.texture.size != wanted:
                    render_pass.canvas.set_size(wanted)
            # An iterated pass draws N times HERE, inside its one turn in the order (068 D1) --
            # never by appearing N times in `order`, which would mean weakening the draw-once
            # invariant that exists to catch a bug reading as slow rather than wrong.
            #
            # An ITERATED OUTPUT pass draws its early iterations into its OWN canvas and only the
            # last one into `canvas`: the chain advances by swapping `render_pass.canvas`, so
            # aiming every iteration at the external target would write somewhere the swap never
            # touches and the chain would silently not advance. RC's final cascade is exactly
            # this shape.
            for iteration in range(entry.iterations):
                last = iteration + 1 == entry.iterations
                draw_into = canvas if (name == output and last) else None
                inputs: dict[str, moderngl.Texture] = {}
                for uniform, source_name in entry.inputs.items():
                    if source_name == name:
                        inputs[uniform] = self._feedback_canvas(name).texture
                    elif source_name in self.passes:
                        inputs[uniform] = self.passes[source_name].canvas.texture
                    else:
                        # An input naming a pass that does not exist reads BLACK (D3), which is
                        # what keeps a half-built graph usable. Leaving it unbound would fall
                        # through to the sampler's own default photo, so a mis-wire would show
                        # an image.
                        inputs[uniform] = self._black_texture()
                render_pass.render(
                    u_time=u_time,
                    canvas=draw_into,
                    inputs=inputs,
                    iteration=iteration,
                    iterations=entry.iterations,
                )
                if not last:
                    # Swap BETWEEN iterations so the next one reads what this one just wrote
                    # (068 D5). `begin_frame` swaps once per FRAME, which is right for a
                    # frame-to-frame trail and would leave every iteration reading the same
                    # stale texture -- the chain would never advance. No-op unless this pass
                    # actually reads itself.
                    self._swap_feedback(name)

    @classmethod
    def load_from_dir(
        cls,
        document_dir: Path | str,
        gl: moderngl.Context | None = None,
    ) -> tuple["Document", dict[str, Any]]:
        """Read a document: its graph, every pass file, and each pass's uniforms.

        A pass file that cannot be read costs THAT pass, never the document (D14), and a
        malformed `graph.json` costs the wiring, never the passes — an unwired document still
        opens with its shaders intact, which is what makes it fixable.
        """
        document_dir = Path(document_dir)
        metadata = _load_document_metadata(document_dir / DOCUMENT_JSON_BASENAME)

        document = Document(gl=gl, canvas_size=metadata.get("canvas_size"))
        graph = load_graph(document_dir / GRAPH_JSON_BASENAME)
        uniforms_by_pass = _uniforms_by_pass(metadata)

        for render_pass in document.passes.values():
            render_pass.release()
        document.passes = {}
        for shader_path in sorted(
            (document_dir / PASSES_DIR_NAME).glob(f"*{PASS_SHADER_SUFFIX}")
        ):
            name = pass_name_of(shader_path)
            entry = graph.passes.get(name)
            try:
                document.passes[name] = Pass(
                    gl=document._gl,
                    source=ShaderSource.load(shader_path),
                    canvas_size=document.canvas_size,
                    target=entry.target if entry is not None else None,
                )
            except OSError as e:
                logger.error(f"Skipping unreadable pass '{name}': {e}")
        if not document.passes:
            raise ValueError(f"{document_dir.name}: no readable pass file")

        # A graph entry naming a file that does not exist is reported and dropped, so the plan
        # never orders a pass nothing can draw.
        missing = [name for name in graph.passes if name not in document.passes]
        if missing:
            logger.warning(
                f"{document_dir.name}: graph names {sorted(missing)}, which have no pass file"
            )
        # One entry per pass FILE: the files are the passes, so a graph entry with no file is
        # dropped (above) and a file with no entry gets defaults.
        document.graph = graph.with_passes(
            {name: graph.passes.get(name, PassEntry()) for name in document.passes}
        )

        for pass_name, render_pass in document.passes.items():
            for uniform_name, value in uniforms_by_pass.get(pass_name, {}).items():
                try:
                    render_pass.uniform_values[uniform_name] = _load_uniform_value(
                        document._gl, document_dir, value
                    )
                except Exception as e:
                    # Per uniform: one unreadable asset costs that binding, not the pass.
                    logger.warning(
                        f"{document_dir.name}/{pass_name}: dropping uniform "
                        f"'{uniform_name}' ({e})"
                    )

        return document, metadata

    def _render_image(
        self, details: MediaDetails, canvas: "Canvas", u_time: float | None = 0.0
    ) -> MediaDetails:
        file_path = Path(details.file_details.path)
        t = u_time if u_time is not None else 0.0
        if self.on_pre_render is not None:
            self.on_pre_render(t, 0.0, 0)
        self.begin_frame()
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

        self.render_pass.restart_video_uniforms()
        n_frames = int(details.duration * details.fps)
        dt = 1.0 / details.fps
        try:
            for i in range(n_frames):
                if self.on_pre_render is not None:
                    self.on_pre_render(i / details.fps, dt, i)
                self.begin_frame()
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
        with self.export_isolation():
            # D10: a feedback target is the same class of state as a stateful script, so it is
            # reset HERE rather than per export path — otherwise the same document exports
            # differently depending on how long the app has been open.
            self.reset_feedback()
            if preset is None or preset.fit is FitPolicy.SCALE_DISTORT:
                canvas = self.render_pass.canvas
                return self._render_media_into(details, canvas)

            target_w, target_h = resolve_dims(
                preset, self.render_pass.canvas.texture.size
            )
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


def document_dir_of(document: Document) -> Path:
    """The directory a document was loaded from / last saved to.

    Derived from a pass file's location, which sits one level down in `passes/` — so this is the
    single place that knows the depth, rather than every caller doing `.parent` and being wrong
    by one the day the layout changes.
    """
    return next(iter(document.passes.values())).source.path.parent.parent
