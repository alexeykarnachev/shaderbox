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
from typing import Any

import imageio
import moderngl
import numpy as np
from loguru import logger

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
from shaderbox.pass_graph import (
    GraphError,
    PassEntry,
    PassGraph,
    plan_for_output,
    plan_passes,
)
from shaderbox.paths import NODE_JSON_BASENAME, NODE_SHADER_BASENAME
from shaderbox.render_preset import FitPolicy, RenderPreset, resolve_dims
from shaderbox.shader_source import ShaderSource

DEFAULT_PASS_NAME = "main"


class Node:
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
        self.canvas_size: tuple[int, int] = canvas_size or DEFAULT_CANVAS_SIZE
        self.passes: dict[str, Pass] = {
            DEFAULT_PASS_NAME: Pass(gl=self._gl, source=source, canvas_size=canvas_size)
        }
        self.graph: PassGraph = PassGraph(
            output=DEFAULT_PASS_NAME, passes={DEFAULT_PASS_NAME: PassEntry()}
        )
        # A feedback pass's previous frame. Allocated on demand by the first frame that needs
        # one, and swapped at the FRAME boundary (begin_frame), never per render call: the live
        # loop draws the current document twice per frame and the copilot probe twice back to
        # back, so a per-call swap would advance the history at 2x.
        self._feedback: dict[str, Canvas] = {}
        self._black: moderngl.Texture | None = None
        self._frame: int = -1
        self._graph_errors: list[GraphError] = []
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
        if self._black is not None:
            self._black.release()
            self._black = None

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
        self._frame = frame if frame is not None else self._frame + 1
        for name, previous in self._feedback.items():
            render_pass = self.passes.get(name)
            if render_pass is None:
                continue
            self._feedback[name] = render_pass.canvas
            render_pass.canvas = previous

    def reset_feedback(self) -> None:
        """Drop every feedback history, so the next frame starts from black.

        Export enters this (D10), and so does anything that wants a document to render as if the
        app had just opened.
        """
        for canvas in self._feedback.values():
            canvas.release()
        self._feedback.clear()
        self._frame = -1

    def _black_texture(self) -> moderngl.Texture:
        # One 1x1 zero texture for every unresolved input in the document.
        if self._black is None:
            self._black = self._gl.texture((1, 1), 4, data=b"\x00\x00\x00\xff")
        return self._black

    def _feedback_canvas(self, name: str) -> Canvas:
        # Born matching its pass's target, so the first frame reads black at the right size
        # rather than sampling a stale or mis-sized texture.
        canvas = self._feedback.get(name)
        live = self.passes[name].canvas
        if canvas is None:
            canvas = Canvas(
                gl=self._gl,
                size=live.texture.size,
                dtype=live.dtype,
                filter=live.filter,
                wrap=live.wrap,
            )
            self._feedback[name] = canvas
        elif canvas.texture.size != live.texture.size:
            canvas.set_size(live.texture.size)
        return canvas

    def render(self, u_time: float | None = None, canvas: Canvas | None = None) -> None:
        """Draw the document: every pass the output needs, in order, each exactly once.

        `canvas` overrides the OUTPUT pass's target only — intermediate passes always draw into
        their own, since that is what the next pass samples.
        """
        output = self.graph.output_pass
        if output is None or output not in self.passes:
            self._graph_errors = plan_passes(self.graph)[1]
            return
        planned, self._graph_errors = plan_for_output(self.graph, output)
        order = [name for name in planned if name in self.passes]
        if not order:
            # A cycle, or an output nothing can reach: draw the output alone so a half-built
            # graph still shows its own shader instead of going blank.
            order = [output]
        for name in order:
            render_pass = self.passes[name]
            entry = self.graph.passes.get(name, PassEntry())
            inputs: dict[str, moderngl.Texture] = {}
            for uniform, source_name in entry.inputs.items():
                if source_name == name:
                    inputs[uniform] = self._feedback_canvas(name).texture
                elif source_name in self.passes:
                    inputs[uniform] = self.passes[source_name].canvas.texture
                else:
                    # An input naming a pass that does not exist reads BLACK (D3), which is what
                    # keeps a half-built graph usable. Leaving it unbound would fall through to
                    # the sampler's own default photo, so a mis-wire would show an image.
                    inputs[uniform] = self._black_texture()
            target = canvas if name == output else None
            render_pass.render(u_time=u_time, canvas=target, inputs=inputs)

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

            node.render_pass.uniform_values[uniform_name] = value

        node.render()  # warm-up
        return node, metadata

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
