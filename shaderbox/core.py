import contextlib
import json
import shutil
import struct
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Sequence
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


class MediaWithTexture(ABC):
    @property
    @abstractmethod
    def texture(self) -> moderngl.Texture: ...

    @property
    @abstractmethod
    def details(self) -> MediaDetails: ...

    @abstractmethod
    def update(self, t: float) -> None: ...

    @abstractmethod
    def release(self) -> None: ...

    @abstractmethod
    def save(self, dir: Path, file_name_wo_ext: str) -> Path:
        pass


class Image(MediaWithTexture):
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
        self._details = MediaDetails(
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
    def details(self) -> MediaDetails:
        return self._details

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

    def update(self, t: float) -> None:
        _ = t

    def save(self, dir: Path, file_name_wo_ext: str) -> Path:
        dir.mkdir(exist_ok=True, parents=True)
        file_path = (dir / file_name_wo_ext).with_suffix(".png")
        self._image.save(file_path, format="PNG")
        return file_path

    def release(self) -> None:
        if self._texture is not None:
            self._texture.release()
            self._texture = None


class Video(MediaWithTexture):
    def __init__(self, file_path: PathLike):
        self._cap = cv2.VideoCapture(str(file_path))

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        file_size = Path(file_path).stat().st_size

        self._details = MediaDetails(
            is_video=True,
            file_details=FileDetails(path=str(file_path), size=file_size),
            resolution_details=ResolutionDetails(width=width, height=height),
            fps=fps,
            duration=self._cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps,
        )

        self._texture: moderngl.Texture | None = None

        self._frame_period = 1.0 / fps
        self._last_frame_idx: int = -1

    def restart(self) -> None:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._last_frame_idx = -1

    @property
    def details(self) -> MediaDetails:
        return self._details

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

    def update(self, t: float) -> None:
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        target_frame_idx = int(t * fps) % n_frames

        # We already have this frame in the texture, skip
        if self._last_frame_idx == target_frame_idx:
            return

        # Skip part of the video, because our t is faster than the video's framerate
        if self._last_frame_idx == -1 or target_frame_idx - self._last_frame_idx != 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)

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
            else:
                self._texture.write(frame)
            self._last_frame_idx = target_frame_idx

    def save(self, dir: Path, file_name_wo_ext: str) -> Path:
        dir.mkdir(exist_ok=True, parents=True)
        ext = Path(self.details.file_details.path).suffix
        file_path = (dir / file_name_wo_ext).with_suffix(ext)

        with contextlib.suppress(shutil.SameFileError):
            shutil.copy2(self.details.file_details.path, file_path)

        return file_path

    def release(self) -> None:
        self._cap.release()

        if self._texture is not None:
            self._texture.release()

    def apply_temporal_smoothing(
        self,
        output_file_path: Path,
        window_size: int = 5,
        sigma: float = 1.0,
        quality: int = 2,
    ) -> None:
        kernel = np.exp(
            -(np.arange(-(window_size // 2), window_size // 2 + 1) ** 2)
            / (2 * sigma**2)
        )
        kernel = kernel / np.sum(kernel)
        weights = " ".join(f"{w:.6f}" for w in kernel)

        mp4_crf = [33, 28, 23, 18]
        mp4_presets = ["ultrafast", "fast", "medium", "slow"]
        crf = mp4_crf[quality]
        preset = mp4_presets[quality]

        ffmpeg_cmd = [
            "ffmpeg",
            "-i",
            str(self.details.file_details.path),
            "-vf",
            f"format=yuv420p,tmix=frames={window_size}:weights={weights}",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-c:a",
            "copy",
            "-y",
            str(output_file_path),
        ]

        subprocess.run(
            ffmpeg_cmd,
            check=True,
            capture_output=True,
            text=True,
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


class UniformBuffer:
    def __init__(self, size: int, gl: moderngl.Context | None = None) -> None:
        self._gl = gl or moderngl.get_context()
        self.size = size

        self.data = bytearray(1024)
        self.data[:16] = struct.pack("4f", 0.0, 1.0, 0.0, 1.0)
        self.ubo = self._gl.buffer(self.data, dynamic=True)

    def release(self) -> None:
        self.ubo.release()


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
            if isinstance(value, dict):
                file_path = Path(value["file_path"])
                media_cls = {".png": Image, ".mp4": Video}[file_path.suffix]
                value = media_cls(node_dir / value["file_path"])
            elif isinstance(value, list):
                value = tuple(value)

            node.set_uniform_value(uniform_name, value)  # type: ignore

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
            if isinstance(data, MediaWithTexture):
                data.release()

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
        time = u_time if u_time is not None else glfw.get_time()
        for uniform in self.get_active_uniforms():
            value = self._uniform_values.get(uniform.name)
            value_for_program = None

            if isinstance(uniform, moderngl.UniformBlock):
                if (
                    value is None
                    or not isinstance(value, UniformBuffer)
                    or isinstance(value.ubo.mglo, moderngl.InvalidObject)
                    or value.size != uniform.size
                ):
                    value = UniformBuffer(uniform.size)
                    self.set_uniform_value(uniform.name, value)

                value.ubo.bind_to_uniform_block(uniform.index)
            elif getattr(uniform, "gl_type", None) == GL_SAMPLER_2D:
                if (
                    value is None
                    or not isinstance(value, MediaWithTexture)
                    or isinstance(value.texture.mglo, moderngl.InvalidObject)
                ):
                    value = Image(self._DEFAULT_IMAGE_FILE_PATH)
                    self.set_uniform_value(uniform.name, value)

                value.update(time)
                value.texture.use(location=texture_unit)
                value_for_program = texture_unit
                texture_unit += 1
            elif uniform.name == "u_time":
                value_for_program = time
                self.set_uniform_value(uniform.name, value_for_program)
            elif uniform.name == "u_aspect":
                value_for_program = np.divide(*canvas.texture.size)
                self.set_uniform_value(uniform.name, value_for_program)
            elif uniform.name == "u_resolution":
                value_for_program = canvas.texture.size
                self.set_uniform_value(uniform.name, value_for_program)
            elif value is None:
                value_for_program = uniform.value
                self.set_uniform_value(uniform.name, value_for_program)

            if value_for_program is not None:
                try:
                    self.program[uniform.name] = value_for_program
                except Exception as e:
                    logger.warning(
                        f"Failed to set uniform '{uniform.name}' with value {value} ({e}). "
                        f"Cached value will be cleared"
                    )
                    self._uniform_values.pop(uniform.name)

        canvas.fbo.use()
        self._gl.clear()
        self.vao.render()

    def set_uniform_value(
        self,
        name: str,
        value: int
        | float
        | Sequence[int]
        | Sequence[float]
        | MediaWithTexture
        | UniformBuffer,
    ) -> None:
        old_value = self._uniform_values.get(name)

        if isinstance(old_value, UniformBuffer) and (
            not isinstance(value, UniformBuffer)
            or old_value.ubo.glo != value.ubo.glo
            or old_value.size != value.size
        ):
            old_value.release()

        if isinstance(old_value, MediaWithTexture) and (
            not isinstance(value, MediaWithTexture)
            or (old_value.texture.glo != value.texture.glo)
        ):
            old_value.release()

        self._uniform_values[name] = value

    def get_uniform_value(self, name: str) -> Any:
        value = self._uniform_values.get(name)
        if value is None and self.program is not None and name in self.program:
            uniform = self.program[name]
            value = uniform.value  # type: ignore
            self._uniform_values[name] = value

        return value

    def restart_video_uniforms(self) -> None:
        for uniform in self.get_active_uniforms():
            video = self._uniform_values.get(uniform.name)
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

        # Ensure resolution is divisible by 16 for codec compatibility
        width = (width + 15) // 16 * 16
        height = (height + 15) // 16 * 16

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
