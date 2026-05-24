import base64
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import glfw
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
    SUPPORTED_MEDIA_EXTENSIONS,
    TEXTURES_DIR_NAME,
    VIDEO_RESOLUTION_ALIGNMENT,
    WEBM_CPU_USED_VALUES,
    WEBM_CRF_VALUES,
)
from shaderbox.media import Image, MediaDetails, MediaWithTexture, Video, texture_to_pil
from shaderbox.render_preset import FitPolicy, RenderPreset, resolve_dims


class Canvas:
    def __init__(
        self,
        gl: moderngl.Context | None = None,
        size: tuple[int, int] | None = None,
    ) -> None:
        self._gl = gl or moderngl.get_context()

        self.texture: moderngl.Texture
        self.fbo: moderngl.Framebuffer

        self._init(size)

    def _init(self, size: tuple[int, int] | None) -> None:
        self.texture = self._gl.texture(size or DEFAULT_CANVAS_SIZE, 4)
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


class Node:
    _DEFAULT_VS_FILE_PATH = DEFAULT_VS_FILE_PATH
    _DEFAULT_FS_FILE_PATH = DEFAULT_FS_FILE_PATH
    _DEFAULT_IMAGE_FILE_PATH = DEFAULT_IMAGE_FILE_PATH

    def __init__(
        self,
        gl: moderngl.Context | None = None,
        fs_source: str | None = None,
        canvas_size: tuple[int, int] | None = None,
    ) -> None:
        self._gl = gl or moderngl.get_context()
        self.vs_source: str = self._DEFAULT_VS_FILE_PATH.read_text(encoding="utf-8")
        self.fs_source: str = (
            fs_source
            if fs_source
            else self._DEFAULT_FS_FILE_PATH.read_text(encoding="utf-8")
        )

        self.canvas = Canvas(size=canvas_size, gl=self._gl)

        self.uniform_values: dict[str, Any] = {}
        self.shader_error: str = ""
        self.program: moderngl.Program | None = None
        self.vbo: moderngl.Buffer | None = None
        self.vao: moderngl.VertexArray | None = None

    @classmethod
    def load_from_dir(
        cls,
        node_dir: Path | str,
        gl: moderngl.Context | None = None,
    ) -> tuple["Node", float, dict[str, Any]]:
        node_dir = Path(node_dir)
        with (node_dir / "node.json").open() as f:
            metadata = json.load(f)

        fs_file_path = node_dir / "shader.frag.glsl"
        mtime = fs_file_path.lstat().st_mtime if fs_file_path.exists() else 0.0

        node = Node(
            gl=gl,
            fs_source=fs_file_path.read_text(encoding="utf-8"),
            canvas_size=metadata.get("canvas_size"),
        )

        # ----------------------------------------------------------------
        # Load uniforms
        for uniform_name, value in metadata["uniforms"].items():
            if isinstance(value, dict):
                local_file_path = value.get("file_path")
                value_base64 = value.get("base64")

                if local_file_path is not None:
                    file_path = node_dir / local_file_path
                    dir_name = file_path.parent.name

                    if dir_name == MEDIA_DIR_NAME:
                        media_cls = {
                            ext: globals()[cls_name]
                            for ext, cls_name in SUPPORTED_MEDIA_EXTENSIONS.items()
                        }[file_path.suffix]
                        value = media_cls(file_path)
                    elif dir_name == TEXTURES_DIR_NAME:
                        data = file_path.read_bytes()
                        value = node._gl.texture(
                            size=value["size"],
                            components=value["components"],
                            data=data,
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

        node.render()  # Warm-up the node
        return node, mtime, metadata

    def release_program(self, new_fs_source: str = "") -> None:
        self.fs_source = new_fs_source
        if self.program:
            self.program.release()
        if self.vbo:
            self.vbo.release()
        if self.vao:
            self.vao.release()
        self.program = None
        self.vbo = None
        self.vao = None
        # Bind 0 — a deleted program left GL-current crashes the imgui renderer's end-of-frame restore (GLError 1281)
        glUseProgram(0)

    def release(self) -> None:
        self.release_program()
        self.canvas.release()

    def get_active_uniforms(self) -> list[moderngl.Uniform | moderngl.UniformBlock]:
        uniforms: list[moderngl.Uniform | moderngl.UniformBlock] = []
        if self.program:
            for uniform_name in self.program:
                uniform = self.program[uniform_name]
                if isinstance(uniform, moderngl.Uniform | moderngl.UniformBlock):
                    uniforms.append(uniform)

        return uniforms

    def render(self, u_time: float | None = None, canvas: Canvas | None = None) -> None:
        canvas = canvas or self.canvas

        if not self.program or not self.vbo or not self.vao:
            try:
                program = self._gl.program(
                    vertex_shader=self.vs_source,
                    fragment_shader=self.fs_source,
                )
            except Exception as e:
                err = str(e)
                if err != self.shader_error:
                    logger.error(f"Failed to compile shader: {e}")
                    self.shader_error = err
                return

            self.shader_error = ""
            if self.program:
                self.program.release()
            if self.vbo:
                self.vbo.release()
            if self.vao:
                self.vao.release()

            self.program = program
            self.vbo = self._gl.buffer(np.array(FULLSCREEN_QUAD_VERTICES, dtype="f4"))
            self.vao = self._gl.vertex_array(program, [(self.vbo, "2f", "a_pos")])

        if not self.program or not self.vao:
            return

        texture_unit = 0
        time = u_time if u_time is not None else glfw.get_time()
        for uniform in self.get_active_uniforms():
            value = self.uniform_values.get(uniform.name)

            value_for_program = None

            if isinstance(uniform, moderngl.UniformBlock):
                if value is None:
                    value = self._gl.buffer(np.zeros(uniform.size, dtype=np.int8))

                assert isinstance(value, moderngl.Buffer)
                value.bind_to_uniform_block(uniform.index)

            elif getattr(uniform, "gl_type", None) == GL_SAMPLER_2D:
                if value is None:
                    value = Image(self._DEFAULT_IMAGE_FILE_PATH)

                if isinstance(value, MediaWithTexture):
                    value.update(time)
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
                value = time
                value_for_program = value

            elif uniform.name == "u_aspect":
                value = np.divide(*canvas.texture.size)
                value_for_program = value

            elif uniform.name == "u_resolution":
                value = canvas.texture.size
                value_for_program = value

            elif value is None:
                value = uniform.value
                value_for_program = value

            else:
                value_for_program = value

            self.uniform_values[uniform.name] = value

            if value_for_program is not None:
                try:
                    self.program[uniform.name] = value_for_program
                except Exception as e:
                    logger.warning(
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
                logger.info(f"Video uniform '{uniform.name}' restarted")

    def _render_image(
        self, details: MediaDetails, canvas: "Canvas", u_time: float | None = 0.0
    ) -> MediaDetails:
        file_path = Path(details.file_details.path)
        self.render(u_time=u_time, canvas=canvas)

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
        for i in range(n_frames):
            self.render(i / details.fps, canvas=canvas)

            texture_data = canvas.texture.read()
            frame = np.frombuffer(texture_data, dtype=np.uint8).reshape(
                canvas.texture.height, canvas.texture.width, 4
            )
            frame = np.flipud(frame)
            writer.append_data(frame)

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
