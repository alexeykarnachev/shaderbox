import contextlib
import shutil
import subprocess
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import IO

import cv2
import imageio_ffmpeg
import moderngl
import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps
from pydantic import BaseModel

from shaderbox.constants import (
    DEFAULT_FPS,
    IMAGE_EXTENSIONS,
    MP4_CRF_VALUES,
    MP4_PRESETS,
    VIDEO_EXTENSIONS,
)


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
    fps: int = DEFAULT_FPS
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
    def __init__(self, file_path: PathLike) -> None:
        self._cap = cv2.VideoCapture(str(file_path))

        if not self._cap.isOpened():
            raise ValueError(
                f"Could not open video (unsupported or corrupt): {file_path}"
            )

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise ValueError(f"Video reports no frame rate (corrupt?): {file_path}")
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

        crf = MP4_CRF_VALUES[quality]
        preset = MP4_PRESETS[quality]

        ffmpeg_cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
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


def media_class_for(suffix: str) -> type[Image] | type[Video]:
    if suffix in IMAGE_EXTENSIONS:
        return Image
    if suffix in VIDEO_EXTENSIONS:
        return Video
    raise ValueError(f"Unsupported media suffix: '{suffix}'")
