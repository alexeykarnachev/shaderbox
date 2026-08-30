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
from shaderbox.paths import NODE_JSON_BASENAME, NODE_SHADER_BASENAME
from shaderbox.render_preset import FitPolicy, RenderPreset, resolve_dims
from shaderbox.shader_source import ShaderSource


class Node:
    def __init__(
        self,
        gl: moderngl.Context | None = None,
        source: ShaderSource | None = None,
        canvas_size: tuple[int, int] | None = None,
    ) -> None:
        self._gl = gl or moderngl.get_context()
        self.render_pass = Pass(gl=self._gl, source=source, canvas_size=canvas_size)
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

    def release(self) -> None:
        self.render_pass.release()

    def render(self, u_time: float | None = None, canvas: Canvas | None = None) -> None:
        """Draw the document into `canvas`, or into its output pass's own target.

        One pass today; stage 3 walks the graph here and this becomes the only place that
        knows an evaluation order exists.
        """
        self.render_pass.render(u_time=u_time, canvas=canvas)

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
            if preset is None or preset.fit is FitPolicy.SCALE_DISTORT:
                canvas = self.render_pass.canvas
                return self._render_media_into(details, canvas)

            target_w, target_h = resolve_dims(preset, self.render_pass.canvas.texture.size)
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
