import json
from importlib.resources import files
from os import PathLike
from pathlib import Path
from typing import IO, Any

import cv2
import glfw
import imageio
import moderngl
import numpy as np
from loguru import logger
from OpenGL.GL import GL_SAMPLER_2D
from PIL import Image as PILImage
from PIL import ImageOps
from pydantic import BaseModel

RESOURCES_DIR = Path(str(files("shaderbox.resources")))


class FileDetails(BaseModel):
    path: str = ""
    size: int = 0

    @classmethod
    def from_file_path(cls, file_path: PathLike) -> "FileDetails":
        return cls(
            path=str(file_path),
            size=Path(file_path).stat().st_size,
        )


class ResolutionDetails(BaseModel):
    width: int = 0
    height: int = 0


class MediaDetails(BaseModel):
    is_video: bool = True
    file_details: FileDetails = FileDetails()
    resolution_details: ResolutionDetails = ResolutionDetails()
    quality: int = 0
    fps: int = 30
    duration: float = 1.0


def texture_to_pil(texture: moderngl.Texture) -> PILImage.Image:
    size = (texture.width, texture.height)

    # Flip image when reading from texture (opengl)
    image = ImageOps.flip(PILImage.frombytes("RGBA", size, texture.read()))
    return image


class Image:
    def __init__(
        self, src: PathLike | PILImage.Image | moderngl.Texture | np.ndarray | IO[bytes]
    ) -> None:
        self._image: PILImage.Image
        self._texture: moderngl.Texture | None = None

        if isinstance(src, moderngl.Texture):
            self._image = texture_to_pil(src)
        elif isinstance(src, PathLike):
            self._image = PILImage.open(str(src))
        elif isinstance(src, PILImage.Image):
            self._image = src
        elif isinstance(src, np.ndarray):
            self._image = PILImage.fromarray(src)
        else:
            self._image = PILImage.open(src)

        file_details = (
            FileDetails.from_file_path(src)
            if isinstance(src, PathLike)
            else FileDetails()
        )

        self._image = self._image.convert("RGBA")
        self.details = MediaDetails(
            is_video=False,
            file_details=file_details,
            resolution_details=ResolutionDetails(
                width=self._image.width, height=self._image.height
            ),
        )

    @classmethod
    def from_color(
        cls, size: tuple[int, int], color: tuple[float, float, float]
    ) -> "Image":
        r, g, b = (int(c * 255) for c in color)
        image = PILImage.new("RGBA", size, color=(r, g, b, 255))
        return cls(image)

    @property
    def texture(self) -> moderngl.Texture:
        if self._texture is None:
            self._texture = moderngl.get_context().texture(
                size=self._image.size,
                components=4,
                # Flip image when writing to texture (opengl)
                data=np.array(ImageOps.flip(self._image)).tobytes(),
                dtype="f1",
            )

        return self._texture

    def release(self) -> None:
        if self._texture is not None:
            self._texture.release()
            self._texture = None


class Video:
    def __init__(self, file_path: PathLike):
        self._cap = cv2.VideoCapture(str(file_path))

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        file_size = Path(file_path).stat().st_size

        self.details = MediaDetails(
            is_video=True,
            file_details=FileDetails(path=str(file_path), size=file_size),
            resolution_details=ResolutionDetails(width=width, height=height),
            fps=fps,
            duration=self._cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps,
        )

        self._texture: moderngl.Texture | None = None

        self._frame_period = 1.0 / fps
        self._last_update_time: float = 0.0

    @property
    def texture(self) -> moderngl.Texture:
        if self._texture is None:
            self._cap.grab()
            frame = self._cap.retrieve()[1]
            frame = np.flipud(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = frame.astype(np.float32) / 255.0
            self._texture = moderngl.get_context().texture(
                size=(frame.shape[1], frame.shape[0]),
                components=4,
                data=frame,
                dtype="f4",
            )

        return self._texture

    def update(self, current_time: float) -> None:
        if current_time - self._last_update_time >= self._frame_period:
            is_frame, frame = self._cap.read()
            if is_frame:
                frame = np.flipud(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                frame = frame.astype(np.float32) / 255.0
                if self._texture is None:
                    self._texture = moderngl.get_context().texture(
                        size=(frame.shape[1], frame.shape[0]),
                        components=4,
                        data=frame,
                        dtype="f4",
                    )

                self._texture.write(frame)
                self._last_update_time = current_time
            else:  # Loop the video
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self) -> None:
        self._cap.release()

        if self._texture is not None:
            self._texture.release()
            self._texture = None


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
        self.texture = self._gl.texture(size or (1280, 960), 4)
        self.fbo = self._gl.framebuffer(color_attachments=[self.texture])

    def release(self) -> None:
        self.texture.release()
        self.fbo.release()

    def set_size(self, size: tuple[int, int]) -> None:
        if size == self.texture.size:
            return

        self.release()
        self._init(size)


class Node:
    _DEFAULT_VS_FILE_PATH = RESOURCES_DIR / "shaders" / "default.vert.glsl"
    _DEFAULT_FS_FILE_PATH = RESOURCES_DIR / "shaders" / "default.frag.glsl"
    _DEFAULT_IMAGE_FILE_PATH = RESOURCES_DIR / "textures" / "default.jpeg"

    def __init__(
        self,
        gl: moderngl.Context | None = None,
        fs_source: str | None = None,
        canvas_size: tuple[int, int] | None = None,
    ) -> None:
        self._gl = gl or moderngl.get_context()
        self.vs_source: str = self._DEFAULT_VS_FILE_PATH.read_text()
        self.fs_source: str = (
            fs_source if fs_source else self._DEFAULT_FS_FILE_PATH.read_text()
        )

        self.canvas = Canvas(size=canvas_size, gl=self._gl)

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
            canvas_size=metadata.get("canvas_size"),
        )

        # ----------------------------------------------------------------
        # Load uniforms
        for uniform_name, value in metadata["uniforms"].items():
            if isinstance(value, dict) and value.get("type") == "texture":
                value = Image(node_dir / value["file_path"]).texture
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

    def release(self) -> None:
        self.release_program()
        self.canvas.release()

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
                value = np.divide(*canvas.texture.size)
                self.set_uniform_value(uniform.name, value)
            elif uniform.name == "u_resolution":
                value = canvas.texture.size
                self.set_uniform_value(uniform.name, value)
            elif uniform.gl_type == GL_SAMPLER_2D:  # type: ignore
                texture = self._uniform_values.get(uniform.name)
                if (
                    texture is None
                    or not isinstance(texture, moderngl.Texture)
                    or isinstance(texture.mglo, moderngl.InvalidObject)
                ):
                    texture = Image(self._DEFAULT_IMAGE_FILE_PATH).texture
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

        canvas.fbo.use()
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

        if extension == ".mp4":
            codec = "libx264"
            pixelformat = "yuv420p"

            mp4_crf = [33, 28, 23, 18]
            mp4_presets = ["ultrafast", "fast", "medium", "slow"]

            crf = mp4_crf[details.quality]
            preset = mp4_presets[details.quality]
            ffmpeg_params = [
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-vf",
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            ]
        elif extension == ".webm":
            codec = "libvpx-vp9"
            pixelformat = "yuva420p"
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
                "-vf",
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
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
        )

        n_frames = int(details.duration * details.fps)
        for i in range(n_frames):
            self.render(u_time=i / details.fps)

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
