from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self
from uuid import uuid4

import imgui
import moderngl
import telegram as tg
from loguru import logger
from OpenGL.GL import GL_SAMPLER_2D
from pydantic import BaseModel, model_validator

from shaderbox.core import Canvas, Node
from shaderbox.media import Image, MediaDetails, MediaWithTexture, Video

if TYPE_CHECKING:
    pass


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
    def success(cls, text: str = "") -> UIMessage:
        return cls(text=text, level="success")

    @classmethod
    def warning(cls, text: str = "") -> UIMessage:
        return cls(text=text, level="warning")

    @classmethod
    def error(cls, text: str = "") -> UIMessage:
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

    async def load(self) -> UITgSticker:
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
    ) -> UIUniform:
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

    def reset_input_type(self) -> UIUniform:
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


class UINode(BaseModel):
    node: Node
    id: str = ""

    mtime: float = 0
    ui_state: UINodeState = UINodeState()

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")  # type: ignore
    def _id_validator(self) -> Self:
        if not self.id:
            self.reset_id()

        return self

    def reset_id(self) -> None:
        self.id = str(uuid4())

    def save(self, root_dir: Path, dir_name: str | None = None) -> Path:
        dir = root_dir / (dir_name or self.id)
        dir.mkdir(exist_ok=True, parents=True)

        meta: dict[str, Any] = {
            "canvas_size": list(self.node.canvas.texture.size),
            "uniforms": {},
            "ui_state": self.ui_state.model_dump(),
        }

        fs_file_path = dir / "shader.frag.glsl"
        with fs_file_path.open("w") as f:
            f.write(self.node.fs_source)
            self.mtime = fs_file_path.lstat().st_mtime

        # ----------------------------------------------------------------
        # Save uniforms
        for uniform in self.node.get_active_uniforms():
            if uniform.name in ["u_time", "u_aspect", "u_resolution"]:
                continue

            value = self.node.uniform_values[uniform.name]

            if getattr(uniform, "gl_type", None) == GL_SAMPLER_2D:
                file_name_wo_ext = uniform.name

                if isinstance(value, MediaWithTexture):
                    file_path = value.save(dir / "media", file_name_wo_ext)
                    local_file_path = f"media/{file_path.name}"
                    size = value.texture.size
                    components = value.texture.components
                elif isinstance(value, moderngl.Texture):
                    data = value.read()
                    local_file_path = f"textures/{file_name_wo_ext}.bin"
                    file_path = dir / local_file_path
                    size = value.size
                    components = value.components
                    file_path.write_bytes(data)
                else:
                    raise ValueError(
                        f"Uniform value must have a type MediaWithTexture or moderngl.Texture, but this one is {type(value)}"
                    )

                meta["uniforms"][uniform.name] = {
                    "file_path": local_file_path,
                    "size": size,
                    "components": components,
                }

            elif isinstance(value, int | float):
                meta["uniforms"][uniform.name] = value

            elif isinstance(value, tuple | list):
                meta["uniforms"][uniform.name] = list(value)

            elif isinstance(value, moderngl.Buffer):
                meta["uniforms"][uniform.name] = {
                    "base64": base64.b64encode(value.read()).decode("utf-8"),
                }

            else:
                logger.warning(
                    f"Can't to save unsupported uniform type for {uniform.name}: {type(value)}"
                )

        with (dir / "node.json").open("w") as f:
            json.dump(meta, f, indent=4)

        return dir

    def draw_preview_button(
        self,
        border_color: tuple[float, float, float] | None,
        size: float,
    ) -> bool:
        n_styles = 0
        if border_color is not None:
            imgui.push_style_color(imgui.COLOR_BORDER, *border_color)
            n_styles += 1

        text = self.ui_state.ui_name
        text_size = imgui.calc_text_size(text)

        label = f"node_preview_{id(self)}"
        imgui.begin_child(
            label,
            width=size,
            height=size + text_size.y,
            border=True,
            flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SCROLL_WITH_MOUSE,
        )

        imgui.pop_style_color(n_styles)

        is_clicked = False
        if imgui.invisible_button(f"{label}##button", width=size, height=size):
            is_clicked = True

        s = (size - 10) / max(self.node.canvas.texture.size)
        image_width = self.node.canvas.texture.size[0] * s
        image_height = self.node.canvas.texture.size[1] * s

        imgui.set_cursor_pos_x((size - image_width) / 2 - 1)
        imgui.set_cursor_pos_y((size - image_height) / 2 - 1)

        imgui.image(
            self.node.canvas.texture.glo,
            width=image_width,
            height=image_height,
            uv0=(0, 1),
            uv1=(1, 0),
        )

        imgui.set_cursor_pos_x((size - text_size.x) / 2)
        imgui.set_cursor_pos_y(size - text_size.y / 2)

        imgui.text(text)

        imgui.end_child()

        return is_clicked


def load_node_from_dir(node_dir: Path) -> UINode:
    node, mtime, meta = Node.load_from_dir(node_dir)
    dir_name = node_dir.name

    ui_state_dict = meta.get("ui_state", {})
    fields = UINodeState.model_fields

    invalid_keys = [k for k in ui_state_dict if k not in fields]
    if invalid_keys:
        logger.warning(
            f"Ignored invalid UINodeState keys for node '{dir_name}': {invalid_keys}"
        )

    filtered_ui_state = {k: v for k, v in ui_state_dict.items() if k in fields}
    filtered_ui_state.setdefault("ui_name", dir_name)
    ui_state = UINodeState(**filtered_ui_state)

    # Import UINode here to avoid circular import
    return UINode(
        id=dir_name,
        node=node,
        mtime=mtime,
        ui_state=ui_state,
    )


def load_nodes_from_dir(root_dir: Path) -> dict[str, UINode]:
    ui_nodes = {}

    node_dirs = sorted(root_dir.iterdir(), key=lambda x: x.stat().st_ctime)

    for node_dir in node_dirs:
        if node_dir.is_dir():
            ui_nodes[node_dir.name] = load_node_from_dir(node_dir)

    return ui_nodes
