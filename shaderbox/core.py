import base64
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import freetype
import glfw
import imageio
import moderngl
import numpy as np
from loguru import logger
from OpenGL.GL import GL_SAMPLER_2D

from shaderbox.constants import (
    DEFAULT_CANVAS_SIZE,
    DEFAULT_FS_FILE_PATH,
    DEFAULT_IMAGE_FILE_PATH,
    DEFAULT_VS_FILE_PATH,
    FONT_ATLAS_PADDING,
    FONT_ATLAS_SIZE,
    FULLSCREEN_QUAD_VERTICES,
    MEDIA_DIR_NAME,
    MP4_CRF_VALUES,
    MP4_PRESETS,
    PRINTABLE_ASCII_END,
    PRINTABLE_ASCII_START,
    SPACE_CHAR_CODE,
    SUPPORTED_MEDIA_EXTENSIONS,
    TEXTURES_DIR_NAME,
    VIDEO_RESOLUTION_ALIGNMENT,
    WEBM_CPU_USED_VALUES,
    WEBM_CRF_VALUES,
)
from shaderbox.media import (
    Image,
    MediaDetails,
    MediaWithTexture,
    Video,
    texture_to_pil,
)


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
    ) -> tuple["Node", float, dict]:
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
        self, details: MediaDetails, u_time: float | None = 0.0
    ) -> MediaDetails:
        file_path = Path(details.file_details.path)
        self.render(u_time=u_time)

        pil_image = texture_to_pil(self.canvas.texture)
        pil_image = pil_image.resize(
            (details.resolution_details.width, details.resolution_details.height)
        )

        pil_image.save(file_path)
        logger.info(f"Image saved: {file_path}")

        rendered_image = Image(file_path)
        rendered_details = rendered_image.details
        rendered_image.release()

        return rendered_details

    def _render_video(self, details: MediaDetails) -> MediaDetails:
        file_path = Path(details.file_details.path)
        extension = file_path.suffix
        width = details.resolution_details.width
        height = details.resolution_details.height

        # Ensure resolution is divisible by alignment for codec compatibility
        alignment = VIDEO_RESOLUTION_ALIGNMENT
        width = (width + alignment - 1) // alignment * alignment
        height = (height + alignment - 1) // alignment * alignment

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
            output_params=["-s", f"{width}x{height}"],
        )

        self.restart_video_uniforms()
        n_frames = int(details.duration * details.fps)
        for i in range(n_frames):
            self.render(i / details.fps)

            texture_data = self.canvas.texture.read()
            frame = np.frombuffer(texture_data, dtype=np.uint8).reshape(
                self.canvas.texture.height, self.canvas.texture.width, 4
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

    def render_media(self, details: MediaDetails) -> MediaDetails:
        if details.is_video:
            return self._render_video(details)
        else:
            return self._render_image(details)


class Font:
    def __init__(self, file_path: Path | str, size: int):
        face = freetype.Face(str(file_path))
        face.set_pixel_sizes(0, size)  # Set font height to 'size' pixels

        texture_width, texture_height = FONT_ATLAS_SIZE
        padding = FONT_ATLAS_PADDING
        current_x, current_y = padding, padding
        max_row_height = 0

        texture_data = np.zeros((texture_height, texture_width, 1), dtype=np.uint8)

        face.load_char("M", freetype.FT_LOAD_RENDER)  # type: ignore
        cell_width = face.glyph.advance.x / 64.0
        cell_height = size

        glyphs = {}

        # Special case for the space
        glyphs[SPACE_CHAR_CODE] = {
            "uv": [0.0, 0.0, 0.0, 0.0],
            "metrics": [face.glyph.advance.x / 64.0 / cell_width, 0.0, 0.0, 0.0],
        }

        # Printable ASCII characters
        for char_code in range(PRINTABLE_ASCII_START, PRINTABLE_ASCII_END):
            face.load_char(char_code, freetype.FT_LOAD_RENDER)  # type: ignore
            bitmap = face.glyph.bitmap

            if bitmap.width == 0 or bitmap.rows == 0:
                continue

            if current_x + bitmap.width + padding > texture_width:
                current_x = padding
                current_y += max_row_height + padding
                max_row_height = 0

            if current_y + bitmap.rows + padding > texture_height:
                raise ValueError("Glyph atlas texture too small")

            bitmap_array = np.array(bitmap.buffer, dtype=np.uint8).reshape(
                bitmap.rows, bitmap.pitch
            )[:, : bitmap.width]
            bitmap_array = np.flipud(bitmap_array)  # Flip for OpenGL

            target_slice = texture_data[
                current_y : current_y + bitmap.rows,
                current_x : current_x + bitmap.width,
            ]
            target_slice[:, :, 0] = bitmap_array

            u0 = current_x / texture_width
            v0 = current_y / texture_height
            u1 = (current_x + bitmap.width) / texture_width
            v1 = (current_y + bitmap.rows) / texture_height

            glyph_width = bitmap.width / cell_width
            glyph_height = bitmap.rows / cell_height
            bearing_x = face.glyph.bitmap_left / cell_width
            bearing_y = (face.glyph.bitmap_top - bitmap.rows) / cell_height

            glyphs[char_code] = {
                "uv": [u0, v0, u1, v1],
                "metrics": [glyph_width, glyph_height, bearing_x, bearing_y],
            }

            current_x += bitmap.width + padding
            max_row_height = max(max_row_height, bitmap.rows)

        min_bearing_y = min(g["metrics"][3] for g in glyphs.values())
        baseline_y = -min_bearing_y if min_bearing_y < 0 else 0
        for char_code in glyphs:
            glyphs[char_code]["metrics"][3] += baseline_y

        gl = moderngl.get_context()
        atlas_texture = gl.texture(
            size=(texture_width, texture_height),
            components=1,
            data=texture_data.tobytes(),
            dtype="f1",
        )
        atlas_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self.glyphs = glyphs
        self.glyph_size_px: tuple[float, float] = (cell_width, cell_height)
        self.atlas_texture = atlas_texture

    def release(self) -> None:
        self.atlas_texture.release()
