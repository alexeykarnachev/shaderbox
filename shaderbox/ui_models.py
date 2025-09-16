import hashlib
from pathlib import Path
from typing import Literal

import imgui
import moderngl
import telegram as tg
from OpenGL.GL import GL_SAMPLER_2D
from pydantic import BaseModel

from shaderbox.core import Canvas
from shaderbox.media import Image, MediaDetails, Video


class UIMessage(BaseModel):
    text: str = ""
    level: Literal["success", "warning", "error"]

    def get_color(self) -> tuple[float, float, float]:
        return {
            "success": (0.0, 1.0, 0.0),
            "warning": (1.0, 1.0, 0.0),
            "error": (1.0, 0.0, 0.0),
        }[self.level]

    @classmethod
    def success(cls, text: str = "") -> "UIMessage":
        return cls(text=text, level="success")

    @classmethod
    def warning(cls, text: str = "") -> "UIMessage":
        return cls(text=text, level="warning")

    @classmethod
    def error(cls, text: str = "") -> "UIMessage":
        return cls(text=text, level="error")

    def __repr__(self) -> str:
        return self.text


class UITgSticker:
    def __init__(self, media_dir: Path, sticker: tg.Sticker | None = None):
        self.media_dir = media_dir
        self._sticker = sticker

        self.video: Video | None = None
        self.image: Image = Image.from_color((512, 512), (0.10, 0.24, 0.39))

        self.preview_canvas = Canvas()

        media_file_name = f"{hashlib.md5(str(id(self)).encode()).hexdigest()[:8]}.webm"
        media_file_path = self.media_dir / media_file_name
        self.render_media_details: MediaDetails = MediaDetails(is_video=True)
        self.render_media_details.file_details.path = str(media_file_path)

        self.log_message: UIMessage = UIMessage(
            text="Submit button will be available after render", level="warning"
        )

    async def load(self) -> "UITgSticker":
        render_file_path = Path(self.render_media_details.file_details.path)

        if self._sticker is not None:
            bot = self._sticker.get_bot()

            if self._sticker.is_video:
                file_name = self._sticker.file_id + ".webm"
                file_path = self.media_dir / file_name

                file = await bot.get_file(self._sticker.file_id)
                await file.download_to_drive(file_path)

                self.video = Video(file_path)
                render_file_path = render_file_path.with_suffix(".webm")

            if self._sticker.thumbnail:
                file_name = self._sticker.thumbnail.file_id + ".webp"
                file_path = self.media_dir / file_name

                file = await bot.get_file(self._sticker.thumbnail.file_id)
                await file.download_to_drive(file_path)

                self.image = Image(file_path)
                render_file_path = render_file_path.with_suffix(".webp")

            if self.video:
                self.render_media_details = self.video.details
            else:
                self.render_media_details = self.image.details

        self.render_media_details.file_details.path = str(render_file_path)

        return self

    def update(self, t: float) -> None:
        if self.video:
            self.video.update(t)

    def get_thumbnail_texture(self) -> moderngl.Texture:
        if self.video:
            return self.video.texture
        else:
            return self.image.texture

    def release(self) -> None:
        self.preview_canvas.release()
        self.image.release()

        if self.video is not None:
            self.video.release()
            self.video = None


UIUniformInputType = Literal[
    "texture", "buffer", "array", "color", "text", "drag", "auto"
]


class UIUniform(BaseModel):
    name: str
    is_ubo: bool = False
    gl_type: int = -1
    dimension: int = -1
    array_length: int = -1
    input_type: UIUniformInputType = "auto"

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_uniform(
        cls, uniform: moderngl.Uniform | moderngl.UniformBlock
    ) -> "UIUniform":
        name = uniform.name

        if isinstance(uniform, moderngl.UniformBlock):
            return cls(
                name=name,
                is_ubo=True,
            ).reset_input_type()
        else:
            return cls(
                name=name,
                is_ubo=False,
                gl_type=uniform.gl_type,  # type: ignore
                dimension=uniform.dimension,
                array_length=uniform.array_length,
            ).reset_input_type()

    def reset_input_type(self) -> "UIUniform":
        if self.is_ubo:
            self.input_type = "buffer"
        elif self.name in ("u_time", "u_aspect", "u_resolution"):
            self.input_type = "auto"
        elif self.gl_type == GL_SAMPLER_2D:
            self.input_type = "texture"
        elif self.array_length > 1 and self.name.endswith("text"):
            self.input_type = "text"
        elif self.array_length > 1:
            self.input_type = "array"
        elif (
            self.array_length == 1
            and self.dimension in (3, 4)
            and self.name.endswith("color")
        ):
            self.input_type = "color"
        elif self.array_length == 1 and self.dimension in (1, 2, 3, 4):
            self.input_type = "drag"
        else:
            self.input_type = "auto"

        return self

    def get_ui_height(self) -> int:
        if self.input_type == "image":
            return 120
        elif self.input_type == "drag":
            return 5 + imgui.get_text_line_height_with_spacing()  # type: ignore
        else:
            return imgui.get_text_line_height_with_spacing()  # type: ignore


class UINodeState(BaseModel):
    ui_name: str = ""

    render_media_details: MediaDetails = MediaDetails()
    ui_uniforms: dict[int, UIUniform] = {}

    resolution_idx: int = 0
    selected_uniform_name: str = ""

    video_to_video_smoothing_window: int = 5
    video_to_video_smoothing_sigma: float = 1.0


class UIAppState(BaseModel):
    current_node_id: str = ""
    selected_node_template_id: str = ""
    new_node_name: str = ""
    is_render_all_nodes: bool = True

    tg_bot_token: str = ""
    tg_user_id: str = ""
    tg_sticker_set_name: str = ""

    tg_sticker_video_details: MediaDetails = MediaDetails()

    media_model_idx: int = 0

    global_target_fps: int = 60

    text_editor_cmd: str = ""
    modelbox_url: str = "http://localhost:8228/"

    def save(self, file_path: str | Path) -> None:
        app_state_dict = self.model_dump()
        with Path(file_path).open("w") as f:
            import json

            json.dump(app_state_dict, f, indent=4)
