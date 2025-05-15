import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import cv2
import glfw
import imageio
import moderngl
import numpy as np
from loguru import logger
from OpenGL.GL import GL_SAMPLER_2D
from PIL import Image, ImageOps
from pydantic import BaseModel

_RESOURCES_DIR = Path(str(files("shaderbox.resources")))
_DEFAULT_VS_FILE_PATH = _RESOURCES_DIR / "shaders" / "default.vert.glsl"
_DEFAULT_FS_FILE_PATH = _RESOURCES_DIR / "shaders" / "default.frag.glsl"
_DEFAULT_IMAGE = Image.open(_RESOURCES_DIR / "textures" / "default.jpeg").convert(
    "RGBA"
)


def image_to_texture(
    image_or_file_path: Image.Image | Path | str,
) -> moderngl.Texture:
    if not isinstance(image_or_file_path, Image.Image):
        image = Image.open(image_or_file_path)
    else:
        image = image_or_file_path

    gl = moderngl.get_context()
    prepared_image = ImageOps.flip(image.convert("RGBA"))
    texture = gl.texture(image.size, 4, np.array(prepared_image).tobytes(), dtype="f1")
    return texture


def texture_to_image(texture: moderngl.Texture) -> Image.Image:
    texture_data = texture.read()
    image = ImageOps.flip(
        Image.frombytes("RGBA", (texture.width, texture.height), texture_data)
    )
    return image


class VideoDetails(BaseModel):
    quality: int = 0
    file_path: str = ""
    file_size: int = 0
    width: int = 0
    height: int = 0
    fps: int = 0
    duration: float = 0.0


class Node:
    def __init__(
        self,
        gl: moderngl.Context | None = None,
        fs_source: str | None = None,
        output_texture_size: tuple[int, int] | None = None,
    ) -> None:
        self._gl = gl or moderngl.get_context()
        self.vs_source: str = _DEFAULT_VS_FILE_PATH.read_text()
        self.fs_source: str = (
            fs_source if fs_source else _DEFAULT_FS_FILE_PATH.read_text()
        )
        self.output_texture_size = output_texture_size or (1280, 960)
        self.output_texture = self._gl.texture(self.output_texture_size, 4)
        self.fbo = self._gl.framebuffer(color_attachments=[self.output_texture])
        self._uniform_values: dict[str, Any] = {}
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
            fs_source=fs_file_path.read_text(),
            output_texture_size=tuple(metadata["output_texture_size"]),
        )

        # ----------------------------------------------------------------
        # Load uniforms
        for uniform_name, value in metadata["uniforms"].items():
            if isinstance(value, dict) and value.get("type") == "texture":
                value = image_to_texture(node_dir / value["file_path"])
            elif isinstance(value, list):
                value = tuple(value)

            node.set_uniform_value(uniform_name, value)

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

    def reset_output_texture_size(self, output_texture_size: tuple[int, int]) -> None:
        self.output_texture.release()
        self.fbo.release()
        self.output_texture_size = output_texture_size
        self.output_texture = self._gl.texture(self.output_texture_size, 4)
        self.fbo = self._gl.framebuffer(color_attachments=[self.output_texture])

    def release(self) -> None:
        self.release_program()
        self.output_texture.release()
        self.fbo.release()
        for data in self._uniform_values.values():
            if isinstance(data, moderngl.Texture):
                data.release()

    def get_uniforms(self) -> list[moderngl.Uniform]:
        uniforms: list[moderngl.Uniform] = []
        if self.program:
            for uniform_name in self.program:
                uniform = self.program[uniform_name]
                if isinstance(uniform, moderngl.Uniform):
                    uniforms.append(uniform)
        return uniforms

    def render(self, u_time: float | None = None) -> None:
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
            self.vbo = self._gl.buffer(
                np.array(
                    [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
                    dtype="f4",
                )
            )
            self.vao = self._gl.vertex_array(program, [(self.vbo, "2f", "a_pos")])

        if not self.program or not self.vao:
            return

        texture_unit = 0
        seen_uniform_names = set()

        for uniform in self.get_uniforms():
            seen_uniform_names.add(uniform.name)
            if uniform.name == "u_time":
                value = u_time if u_time is not None else glfw.get_time()
                self.set_uniform_value(uniform.name, value)
            elif uniform.name == "u_aspect":
                value = np.divide(*self.output_texture.size)
                self.set_uniform_value(uniform.name, value)
            elif uniform.name == "u_resolution":
                value = self.output_texture.size
                self.set_uniform_value(uniform.name, value)
            elif uniform.gl_type == GL_SAMPLER_2D:  # type: ignore
                texture = self._uniform_values.get(uniform.name)
                if (
                    texture is None
                    or not isinstance(texture, moderngl.Texture)
                    or isinstance(texture.mglo, moderngl.InvalidObject)
                ):
                    texture = image_to_texture(_DEFAULT_IMAGE)
                    self.set_uniform_value(uniform.name, texture)
                texture.use(location=texture_unit)
                value = texture_unit
                texture_unit += 1
            else:
                value = self._uniform_values.get(uniform.name)
                if value is None:
                    value = uniform.value
                    self.set_uniform_value(uniform.name, value)

            try:
                self.program[uniform.name] = value
            except Exception as _:
                logger.warning(
                    f"Failed to set uniform '{uniform.name}' with value {value}. "
                    "Cached value will be cleared."
                )
                self._uniform_values.pop(uniform.name)

        self.fbo.use()
        self._gl.clear()
        self.vao.render()

        for uniform_name in self._uniform_values.copy():
            if uniform_name not in seen_uniform_names:
                data = self._uniform_values.pop(uniform_name)
                if isinstance(data, moderngl.Texture):
                    data.release()

    def set_uniform_value(self, name: str, value: Any) -> None:
        old_value = self._uniform_values.get(name)
        if isinstance(old_value, moderngl.Texture) and old_value.glo != value.glo:
            old_value.release()
        self._uniform_values[name] = value

    def get_uniform_value(self, name: str) -> Any:
        value = self._uniform_values.get(name)
        if value is None and self.program is not None and name in self.program:
            uniform = self.program[name]
            value = uniform.value  # type: ignore
            self._uniform_values[name] = value
        return value

    def render_to_image(self, u_time: float | None = 0.0) -> Image.Image:
        self.render(u_time=u_time)

        if self.shader_error:
            raise ValueError(f"Shader compilation failed: {self.shader_error}")

        return texture_to_image(self.output_texture)

    def render_to_video(
        self,
        details: VideoDetails,
    ) -> VideoDetails:
        if not 0 <= details.quality <= 3:
            raise ValueError("Quality must be an integer between 0 and 3")

        file_path = Path(details.file_path)
        extension = file_path.suffix

        if extension == ".mp4":
            codec = "libx264"
            pixelformat = "yuv420p"

            mp4_crf = [33, 28, 23, 18]
            mp4_presets = ["ultrafast", "fast", "medium", "slow"]

            crf = mp4_crf[details.quality]
            preset = mp4_presets[details.quality]
            ffmpeg_params = ["-crf", str(crf), "-preset", preset]
        elif extension == ".webm":
            codec = "libvpx-vp9"
            pixelformat = "yuv420p"

            webm_crf = [50, 40, 30, 20]
            webm_cpu_used = [5, 4, 3, 2]

            crf = webm_crf[details.quality]
            cpu_used = webm_cpu_used[details.quality]
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
        )

        n_frames = int(details.duration * details.fps)
        for i in range(n_frames):
            u_time = i / details.fps
            frame = self.render_to_image(u_time)
            frame_np = np.array(frame.convert("RGB"))
            writer.append_data(frame_np)

        writer.close()

        # ----------------------------------------------------------------
        # Update details from the saved video
        video = Video(file_path)
        new_details = video.details
        video.release()

        return new_details


class Video:
    def __init__(self, file_path: str | Path):
        file_path = str(file_path)
        self._cap = cv2.VideoCapture(file_path)

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self._cap.get(cv2.CAP_PROP_FPS))

        self.details = VideoDetails(
            file_path=file_path,
            file_size=Path(file_path).stat().st_size,
            width=width,
            height=height,
            fps=fps,
            duration=self._cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps,
        )

        self.texture: moderngl.Texture = image_to_texture(
            Image.new("RGBA", (width, height), (255, 0, 0, 255))
        )

        self._frame_period = 1.0 / fps
        self._last_update_time: float = 0.0

    def update(self, current_time: float) -> None:
        if current_time - self._last_update_time >= self._frame_period:
            is_frame, frame = self._cap.read()
            if is_frame:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                image = Image.fromarray(frame_rgb)
                self.texture.release()
                self.texture = image_to_texture(image)
                self._last_update_time = current_time
            else:  # Loop the video
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self) -> None:
        self._cap.release()
        self.texture.release()
