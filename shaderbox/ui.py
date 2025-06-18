import asyncio
import contextlib
import hashlib
import json
import math
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, Self, TypeVar
from uuid import uuid4

import crossfiledialog
import cv2
import glfw
import imgui
import moderngl
import numpy as np
import telegram as tg
from imgui.integrations.glfw import GlfwRenderer
from loguru import logger
from OpenGL.GL import GL_FLOAT, GL_SAMPLER_2D, GL_UNSIGNED_INT, GLError
from platformdirs import user_data_dir
from pydantic import BaseModel, model_validator

from shaderbox import modelbox
from shaderbox.core import (
    RESOURCES_DIR,
    Canvas,
    FileDetails,
    Image,
    MediaDetails,
    MediaWithTexture,
    Node,
    ResolutionDetails,
    Video,
)


def adjust_size(
    size: tuple[int, int],
    width: int | None = None,
    height: int | None = None,
    aspect: float | None = None,
    max_size: int | None = None,
) -> tuple[int, int]:
    if (width, height, aspect, max_size).count(None) != 3:
        return size

    original_width, original_height = size

    if width is not None:
        new_height = round(width * original_height / original_width)
        return (width, new_height)
    elif height is not None:
        new_width = round(height * original_width / original_height)
        return (new_width, height)
    elif aspect is not None:
        current_aspect = original_width / original_height
        if aspect > current_aspect:
            new_width = round(original_height * aspect)
            return (new_width, original_height)
        else:
            new_height = round(original_width / aspect)
            return (original_width, new_height)
    elif max_size is not None:
        if original_width >= original_height:
            new_width = max_size
            new_height = round(max_size * original_height / original_width)
            return (new_width, new_height)
        else:
            new_height = max_size
            new_width = round(max_size * original_width / original_height)
            return (new_width, new_height)
    else:
        return size


K = TypeVar("K")


def select_next_value(values: Sequence[K], current_key: K | None, step: int = 1) -> K:
    idx = (
        0 if not current_key or current_key not in values else values.index(current_key)
    )
    return values[(idx + step) % len(values)]


def mod(a: float, b: float) -> float:
    if b == 0.0:
        return 0.0
    return a - b * (a // b)


def get_resolution_str(name: str | None, w: int, h: int) -> str:
    g = math.gcd(w, h)
    w_ratio, h_ratio = w // g, h // g
    aspect = f"{w_ratio}:{h_ratio}"
    parts = [f"{w}x{h}"]
    if name:
        parts.append(name)
    parts.append(aspect)
    return " | ".join(parts)


def depth_mask_to_normals(depthmap_image: Image, kernel_size: int = 3) -> Image:
    kernel_size = max(3, kernel_size | 1)

    depth_array = np.array(depthmap_image._image.convert("L"), dtype=np.float32)

    grad_x = cv2.Sobel(depth_array, cv2.CV_32F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(depth_array, cv2.CV_32F, 0, 1, ksize=kernel_size)

    normals = np.dstack((grad_x, -grad_y, np.ones_like(depth_array)))

    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    norm[norm < 1e-6] = 1e-6
    normals = normals / norm

    normals_rgb = ((normals + 1.0) * 127.5).astype(np.uint8)

    return Image(normals_rgb)


def zero_low_alpha_pixels(image: Image, min_alpha: float = 1.0) -> Image:
    img_array = np.array(image._image)
    alpha_threshold = int(min_alpha * 255)
    low_alpha_mask = img_array[:, :, 3] < alpha_threshold
    img_array[:, :, :3][low_alpha_mask] = 0

    return Image(img_array)


def get_uniform_hash(u: moderngl.Uniform) -> int:
    key = f"{u.name}_{u.array_length}_{u.dimension}_{u.gl_type}"  # type: ignore
    hash = hashlib.md5(key.encode()).digest()
    return int.from_bytes(hash, "big")


def unicode_to_str(char_inds: list[int]) -> str:
    eos_idx = char_inds.index(0)
    chars = []
    for i in range(0, eos_idx):
        chars.append(chr(char_inds[i]))
    text = "".join(chars)

    return text


def str_to_unicode(text: str, max_n_chars: int) -> list[int]:
    if len(text) >= max_n_chars:
        text = text[:max_n_chars]
    char_inds = [ord(c) for c in text]
    pad_len = max_n_chars - len(char_inds)
    char_inds += [0] * pad_len

    return char_inds


def get_dir_hash(dir: Path) -> str:
    hasher = hashlib.sha256()
    for file in sorted(Path(dir).rglob("*")):
        if file.is_file():
            hasher.update(file.read_bytes())
    return hasher.hexdigest()


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
        self.image.release()

        if self.video is not None:
            self.video.release()
            self.video = None


UIUniformInputType = Literal["texture", "array", "color", "text", "drag", "auto"]


class UIUniform(BaseModel):
    name: str
    gl_type: int
    dimension: int
    array_length: int
    input_type: UIUniformInputType = "auto"

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_uniform(cls, uniform: moderngl.Uniform) -> "UIUniform":
        return cls(
            name=uniform.name,
            gl_type=uniform.gl_type,  # type: ignore
            dimension=uniform.dimension,
            array_length=uniform.array_length,
        ).reset_input_type()

    def reset_input_type(self) -> "UIUniform":
        if self.name in ("u_time", "u_aspect", "u_resolution"):
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
    current_node_dir: str = ""
    selected_node_template_dir: str = ""
    new_node_name: str = ""
    is_render_all_nodes: bool = True

    tg_bot_token: str = ""
    tg_user_id: str = ""
    tg_sticker_set_name: str = ""

    tg_sticker_video_details: MediaDetails = MediaDetails()

    media_model_idx: int = 0


class UINode(BaseModel):
    node: Node

    # Some kind of unique id and also name of the directory to save the node
    dir: str = ""

    mtime: float = 0
    ui_state: UINodeState = UINodeState()

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")  # type: ignore
    def _dir_validator(self) -> Self:
        if not self.dir:
            self.reset_dir()

        return self

    def reset_dir(self) -> None:
        self.dir = str(uuid4())

    def save(self, root_dir: Path, dir_name: str | None = None) -> Path:
        dir = root_dir / (dir_name or self.dir)
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

            value = self.node.get_uniform_value(uniform.name)

            if uniform.gl_type == GL_SAMPLER_2D and isinstance(value, MediaWithTexture):  # type: ignore
                media: MediaWithTexture = value
                file_name_wo_ext = uniform.name
                file_path = media.save(dir / "media", file_name_wo_ext)
                meta["uniforms"][uniform.name] = {
                    "file_path": str(file_path),
                }
            elif isinstance(value, int | float):
                meta["uniforms"][uniform.name] = value
            elif isinstance(value, tuple | list):
                meta["uniforms"][uniform.name] = list(value)
            else:
                logger.warning(
                    f"Skipping unsupported uniform type for {uniform.name}: {type(value)}"
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

    return UINode(
        dir=dir_name,
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


class App:
    _SETTINGS_POPUP_LABEL = "Settings##popup"
    _NODE_CREATOR_POPUP_LABEL = "New node##popup"

    def __init__(self, project_dir: Path | None = None) -> None:
        if project_dir is None:
            if self.project_dir_file_path.exists():
                project_dir = Path(self.project_dir_file_path.read_text())
            else:
                project_dir = self.default_project_dir

        glfw.init()
        self._loop = asyncio.new_event_loop()

        monitor = glfw.get_primary_monitor()
        video_mode = glfw.get_video_mode(monitor)
        screen_width = video_mode.size[0]
        window_width = screen_width // 2
        window_height = video_mode.size[1]

        window = glfw.create_window(
            width=window_width,
            height=window_height,
            title="ShaderBox",
            monitor=None,
            share=None,
        )

        glfw.make_context_current(window)
        monitor_pos = glfw.get_monitor_pos(monitor)
        monitor_x, monitor_y = monitor_pos
        window_x = monitor_x + screen_width - window_width

        glfw.set_window_pos(window, window_x, monitor_y)

        imgui.create_context()
        self.window = window
        self.imgui_renderer = GlfwRenderer(window)
        self._font_14 = self.get_font(14)
        self.preview_canvas = Canvas()

        self.ui_nodes: dict[str, UINode] = {}
        self.ui_node_templates: dict[str, UINode] = {}
        self.ui_app_state = UIAppState()
        self._tg_stickers: list[UITgSticker] = []
        self._tg_selected_sticker_idx: int = 0
        self._tg_stickers_bot: tg.Bot

        self.modelbox_info: dict[str, Any] = {}

        self._active_popup_label: str | None = None
        self.global_fps = 0.0
        self._init(project_dir)

    @staticmethod
    def _create_dir_if_needed(path: Path | str) -> Path:
        path = Path(path)

        if not path.exists():
            path.mkdir(parents=True)
            logger.info(f"Directory created: {path}")

        return path

    @property
    def app_dir(self) -> Path:
        return Path(user_data_dir("shaderbox"))

    @property
    def project_dir_file_path(self) -> Path:
        return self.app_dir / "project_dir"

    @property
    def default_projects_root_dir(self) -> Path:
        return self._create_dir_if_needed(self.app_dir / "projects")

    @property
    def default_project_dir(self) -> Path:
        return self._create_dir_if_needed(self.default_projects_root_dir / "default")

    @property
    def node_templates_dir(self) -> Path:
        return RESOURCES_DIR / "node_templates"

    @property
    def app_state_file_path(self) -> Path:
        return Path(self.project_dir / "app_state.json")

    @property
    def tg_sticker_state_file_path(self) -> Path:
        return Path(self.project_dir / "tg_sticker.json")

    @property
    def nodes_dir(self) -> Path:
        return self._create_dir_if_needed(self.project_dir / "nodes")

    @property
    def media_dir(self) -> Path:
        return self._create_dir_if_needed(self.project_dir / "media")

    @property
    def trash_dir(self) -> Path:
        return self._create_dir_if_needed(self.project_dir / "trash")

    def _init(self, project_dir: Path) -> None:
        self.release()
        self.fetch_modelbox_info()

        self.app_start_time = int(time.time() * 1000)
        self.frame_idx = 0

        self.ui_nodes.clear()
        self._tg_stickers.clear()

        self.project_dir = self._create_dir_if_needed(project_dir)
        self.project_dir_file_path.write_text(str(self.project_dir))
        logger.info(f"Project loaded: {self.project_dir}")

        # ----------------------------------------------------------------
        # Load nodes
        self.ui_nodes = load_nodes_from_dir(self.nodes_dir)
        self.ui_node_templates = load_nodes_from_dir(self.node_templates_dir)

        # ----------------------------------------------------------------
        # Load ui state
        if self.app_state_file_path.exists():
            with self.app_state_file_path.open("r") as f:
                state_dict = json.load(f)
                state_dict = {
                    k: v for k, v in state_dict.items() if k in UIAppState.model_fields
                }
                self.ui_app_state = UIAppState(**state_dict)

                if (
                    self.ui_app_state.current_node_dir
                    and self.ui_app_state.current_node_dir not in self.ui_nodes
                ):
                    logger.warning(
                        f"Node {self.ui_app_state.current_node_dir} not found"
                    )
                    self.ui_app_state.current_node_dir = ""

    def fetch_modelbox_info(self) -> None:
        try:
            self.modelbox_info = modelbox.fetch_modelbox_info()
        except Exception as e:
            logger.error(f"Failed to fetch media model names: {e}")
            self.modelbox_info = {}

    def get_font(self, size: int) -> Any:
        fonts = imgui.get_io().fonts
        font = fonts.add_font_from_file_ttf(
            str(RESOURCES_DIR / "fonts" / "Anonymous_Pro" / "AnonymousPro-Regular.ttf"),
            size_pixels=size,
            glyph_ranges=fonts.get_glyph_ranges_cyrillic(),
        )
        self.imgui_renderer.refresh_font_texture()
        return font

    def fetch_tg_stickers(self) -> None:
        async def _fetch() -> list[UITgSticker]:
            self._tg_stickers_bot = tg.Bot(token=self.ui_app_state.tg_bot_token)
            await self._tg_stickers_bot.initialize()

            sticker_set = await self._tg_stickers_bot.get_sticker_set(
                name=self.ui_app_state.tg_sticker_set_name
            )
            coros = [
                UITgSticker(media_dir=self.media_dir, sticker=sticker).load()
                for sticker in sticker_set.stickers
            ]
            return await asyncio.gather(*coros)

        try:
            self._tg_stickers = self._loop.run_until_complete(_fetch())
        except tg.error.InvalidToken:
            logger.error("Invalid telegram token")
        except tg.error.BadRequest as e:
            if str(e) == "Stickerset_invalid":
                logger.info("Sticker set doesn't exist")
            else:
                raise e

        new_sticker = UITgSticker(media_dir=self.media_dir)
        self._tg_stickers.insert(0, new_sticker)

    @property
    def ui_current_node_state(self) -> UINodeState:
        if not self.ui_app_state.current_node_dir:
            return UINodeState()
        return self.ui_nodes[self.ui_app_state.current_node_dir].ui_state

    def edit_node_fs_file(self, node_name: str) -> None:
        fs_file_path = self.nodes_dir / node_name / "shader.frag.glsl"
        wd = fs_file_path.parent.parent
        editor = "nvim" if shutil.which("nvim") else "vim"

        try:
            subprocess.Popen(
                ["gnome-terminal", "--", editor, str(fs_file_path)],
                cwd=str(wd),
                start_new_session=True,
            )
        except Exception as e:
            logger.error(
                f"Failed to open {fs_file_path} in {editor} with new terminal: {e}"
            )

    def edit_current_node_fs_file(self) -> None:
        if self.ui_app_state.current_node_dir:
            self.edit_node_fs_file(self.ui_app_state.current_node_dir)
        else:
            logger.warning("Nothing to edit")

    def save_ui_node(
        self,
        ui_node: UINode,
        root_dir: Path | None = None,
        dir_name: str | None = None,
    ) -> Path:
        root_dir = root_dir or self.nodes_dir
        dir = ui_node.save(root_dir, dir_name)
        logger.info(f"Node '{ui_node.ui_state.ui_name}' saved: {dir}")
        return dir

    def draw_popup_if_opened(self, label: str, draw_func: Callable[[], bool]) -> None:
        if self._active_popup_label != label:
            return

        if imgui.begin_popup_modal(label).opened:
            if not draw_func():
                self._active_popup_label = None
                imgui.close_current_popup()

            imgui.end_popup()

    def save_current_node(self) -> None:
        if self.ui_app_state.current_node_dir:
            self.save_ui_node(self.ui_nodes[self.ui_app_state.current_node_dir])
        else:
            logger.warning("Nothing to save")

    def save_app_state(self) -> None:
        app_state_dict = self.ui_app_state.model_dump()
        with self.app_state_file_path.open("w") as f:
            json.dump(app_state_dict, f, indent=4)

    def save(self) -> None:
        self.save_current_node()
        self.save_app_state()

    def release(self) -> None:
        for sticker in self._tg_stickers:
            sticker.release()

        for node in self.ui_nodes.values():
            node.node.release()

        for node in self.ui_node_templates.values():
            node.node.release()

    def delete_current_node(self) -> None:
        if not self.ui_app_state.current_node_dir:
            logger.info("Current node is None, nothing to delete")
            return

        name = self.ui_app_state.current_node_dir
        self.ui_nodes.pop(name).node.release()

        self.ui_app_state.current_node_dir = (
            next(iter(self.ui_nodes)) if self.ui_nodes else ""
        )

        shutil.move(self.nodes_dir / name, self.trash_dir / name)
        logger.info(f"Node deleted: {name}")

    def select_next_current_node(self, step: int = +1) -> None:
        if not self.ui_nodes:
            return

        self.ui_app_state.current_node_dir = select_next_value(
            list(self.ui_nodes.keys()), self.ui_app_state.current_node_dir, step
        )

    def select_next_template(self, step: int = +1) -> None:
        self.ui_app_state.selected_node_template_dir = select_next_value(
            list(self.ui_node_templates.keys()),
            self.ui_app_state.selected_node_template_dir,
            step,
        )

    def draw_node_preview_grid(self, width: int, height: int) -> None:
        with imgui.begin_child(
            "node_preview_grid", width=width, height=height, border=True
        ):
            self.ui_app_state.is_render_all_nodes = imgui.checkbox(
                "Render all", self.ui_app_state.is_render_all_nodes
            )[1]

            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.text(
                    "If checked, renders all nodes, otherwise, renders only the selected one."
                )
                imgui.end_tooltip()

            imgui.same_line()
            imgui.set_cursor_pos_x(width - 14)
            imgui.text_colored("?", *(0.5, 0.5, 0.5))
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.text("CREATE new node         CTRL+N")
                imgui.text("SAVE current node       CTRL+S")
                imgui.text("EDIT current node       CTRL+E")
                imgui.text("DELETE current node     CTRL+D")
                imgui.text("PREVIOUS node             <-")
                imgui.text("NEXT node                 ->")
                imgui.end_tooltip()

            preview_size = 150
            n_cols = int(imgui.get_content_region_available()[0] // (preview_size + 5))
            n_cols = max(1, n_cols)
            for i, (name, ui_node) in enumerate(self.ui_nodes.items()):
                border_color = None
                if name == self.ui_app_state.current_node_dir:
                    if ui_node.node.shader_error:
                        border_color = (1.0, 0.0, 0.0)
                    else:
                        border_color = (0.0, 1.0, 0.0)

                if ui_node.draw_preview_button(border_color, preview_size):
                    self.ui_app_state.current_node_dir = name

                if (i + 1) % n_cols != 0:
                    imgui.same_line()
                else:
                    imgui.spacing()

    def draw_node_creator(self) -> bool:
        imgui.text("Select template:")

        is_template_selected = False

        preview_size = 150
        available_width = imgui.get_content_region_available()[0]
        n_cols = max(1, int(available_width // (preview_size + 5)))

        for i, ui_node_template in enumerate(self.ui_node_templates.values()):
            if ui_node_template.dir == self.ui_app_state.selected_node_template_dir:
                border_color = (0.0, 1.0, 0.0)
                is_template_selected = True
            else:
                border_color = None

            if ui_node_template.draw_preview_button(border_color, preview_size):
                self.ui_app_state.selected_node_template_dir = ui_node_template.dir
                self.ui_app_state.new_node_name = ui_node_template.ui_state.ui_name

            if (i + 1) % n_cols != 0 and i != len(self.ui_node_templates) - 1:
                imgui.same_line()
            else:
                imgui.spacing()

        imgui.new_line()

        button_width = 80
        total_button_width = (
            button_width if not is_template_selected else button_width * 2 + 10
        )
        imgui.set_cursor_pos_x((available_width - total_button_width) / 2)

        is_keep_opened = True
        if imgui.button("Create", width=button_width) and is_template_selected:
            self.create_node_from_selected_template()
            is_keep_opened = False

        imgui.same_line()
        is_keep_opened &= not imgui.button("Cancel", width=button_width)

        return is_keep_opened

    def create_node_from_selected_template(self) -> None:
        selected_template = self.ui_node_templates[
            self.ui_app_state.selected_node_template_dir
        ]

        new_node = load_node_from_dir(self.node_templates_dir / selected_template.dir)
        new_node.reset_dir()

        self.ui_nodes[new_node.dir] = new_node
        self.ui_app_state.current_node_dir = new_node.dir
        self.save_ui_node(new_node)
        logger.info(
            f"New node {new_node.dir} created from template {self.ui_app_state.selected_node_template_dir}"
        )

    def draw_settings(self) -> bool:
        imgui.text("Current project:")
        imgui.same_line()
        imgui.text_colored(str(self.project_dir), *(0.5, 0.5, 0.5))

        if imgui.button("Open another##project"):
            start_dir = str(
                self.project_dir.parent
                if self.project_dir
                else self.default_projects_root_dir
            )
            project_dir = crossfiledialog.choose_folder(
                title="Project", start_dir=start_dir
            )
            if project_dir:
                self._init(project_dir)

        imgui.new_line()

        is_keep_opened = not imgui.button("Cancel")
        return is_keep_opened

    def draw_node_tab(self) -> None:
        if self.ui_app_state.current_node_dir:
            ui_node = self.ui_nodes[self.ui_app_state.current_node_dir]
        else:
            return

        fs_file_path = self.nodes_dir / ui_node.dir / "shader.frag.glsl"
        imgui.text_colored(str(fs_file_path), 0.5, 0.5, 0.5)

        imgui.spacing()
        ui_node.ui_state.ui_name = imgui.input_text("Name", ui_node.ui_state.ui_name)[1]
        imgui.spacing()

        # ----------------------------------------------------------------
        # Node menu bar
        if imgui.button("Edit code", width=80):
            self.edit_node_fs_file(ui_node.dir)
        imgui.same_line()

        if imgui.button("Save as template"):
            dir = self.save_ui_node(
                ui_node,
                root_dir=self.node_templates_dir,
                dir_name=str(uuid4()),
            )
            self.ui_node_templates[dir.name] = load_node_from_dir(dir)

        imgui.new_line()
        imgui.separator()
        imgui.spacing()

        # ----------------------------------------------------------------
        # Resolution combobox
        standard_resolutions = [
            (1080, 1920),
            (960, 1280),
            (1080, 1080),
            (1280, 960),
            (1920, 1080),
            (3440, 1440),
        ]

        uniform_resolutions = []
        matching_uniforms = []
        uniform_sizes = set()
        for uniform in ui_node.node.get_active_uniforms():
            if uniform.gl_type == GL_SAMPLER_2D:  # type: ignore
                media: MediaWithTexture = ui_node.node.get_uniform_value(uniform.name)
                w, h = media.texture.size
                if (w, h) == ui_node.node.canvas.texture.size:
                    matching_uniforms.append(uniform.name)
                else:
                    uniform_resolutions.append((w, h, uniform.name))
                    uniform_sizes.add((w, h))

        resolution_items = []

        for w, h, name in uniform_resolutions:
            resolution_items.append(get_resolution_str(name, w, h))

        for w, h in standard_resolutions:
            if (w, h) != ui_node.node.canvas.texture.size and (
                w,
                h,
            ) not in uniform_sizes:
                resolution_items.append(get_resolution_str(None, w, h))

        imgui.text(
            "Current resolution: "
            + get_resolution_str(None, *ui_node.node.canvas.texture.size)
        )

        imgui.spacing()

        if matching_uniforms:
            imgui.same_line()
            imgui.text_colored("(" + ", ".join(matching_uniforms) + ")", 0.5, 0.5, 0.5)

        self.ui_current_node_state.resolution_idx = imgui.combo(
            "##resolution_idx",
            self.ui_current_node_state.resolution_idx,
            resolution_items,
        )[1]

        imgui.same_line()
        if imgui.button("Apply##resolution"):
            w, h = map(
                int,
                resolution_items[self.ui_current_node_state.resolution_idx]
                .split(" | ")[0]
                .split("x"),
            )
            ui_node.node.canvas.set_size((w, h))

        imgui.new_line()
        imgui.separator()
        imgui.spacing()

        # ----------------------------------------------------------------
        # Uniforms
        ui_uniforms = self.ui_current_node_state.ui_uniforms
        active_uniform_names = []
        for uniform in ui_node.node.get_active_uniforms():
            active_uniform_names.append(uniform.name)
            hash = get_uniform_hash(uniform)
            if hash not in ui_uniforms:
                ui_uniforms[hash] = UIUniform.from_uniform(uniform)

        imgui.begin_child(
            "ui_uniforms",
            width=imgui.get_content_region_available_width() // 2,
        )
        for ui_uniform in ui_uniforms.values():
            if ui_uniform.name in active_uniform_names:
                self.draw_ui_uniform(ui_uniform)

        imgui.end_child()

        if self.ui_current_node_state.selected_uniform_name:
            imgui.same_line()
            imgui.begin_child("selected_uniform_settings", border=True)
            self.draw_selected_ui_uniform_settings()
            imgui.end_child()

    T = TypeVar("T", Image, Video)

    def draw_media_models(self, input_media: T) -> T:
        output_media: Image | Video | None = None
        media_model_names = self.modelbox_info.get("media_model_names")

        if not media_model_names:
            imgui.text_colored("Media models are not available", *(1.0, 1.0, 0.0))
            if imgui.button("Refresh##media"):
                self.fetch_modelbox_info()
        else:
            imgui.text("Media model")

            self.ui_app_state.media_model_idx = min(
                self.ui_app_state.media_model_idx,
                len(media_model_names) - 1,
            )

            self.ui_app_state.media_model_idx = imgui.combo(
                "##media_model_idx",
                self.ui_app_state.media_model_idx,
                media_model_names,
            )[1]

            model_name = media_model_names[self.ui_app_state.media_model_idx]

            imgui.same_line()
            if imgui.button("Apply##media_model"):
                output_media = modelbox.infer_media_model(
                    input_media,
                    model_name=model_name,
                    output_dir=self.media_dir,
                )

        return output_media or input_media  # type: ignore

    def draw_video_filters(self, input_video: Video) -> Video:
        output_video: Video | None = None

        imgui.text("Smoothing")

        self.ui_current_node_state.video_to_video_smoothing_window = imgui.drag_int(
            "Window",
            self.ui_current_node_state.video_to_video_smoothing_window,
            min_value=3,
        )[1]

        self.ui_current_node_state.video_to_video_smoothing_sigma = imgui.drag_float(
            "Sigma",
            self.ui_current_node_state.video_to_video_smoothing_sigma,
            min_value=0.01,
            change_speed=0.01,
        )[1]

        if imgui.button("Apply##video_to_video_smoothing"):
            input_file_path = Path(input_video.details.file_details.path)
            name = input_file_path.stem

            w = self.ui_current_node_state.video_to_video_smoothing_window
            s = self.ui_current_node_state.video_to_video_smoothing_sigma
            name = f"{name}_w:{w}_s:{s}"
            output_file_path = (self.trash_dir / name).with_suffix(
                input_file_path.suffix
            )

            input_video.apply_temporal_smoothing(
                output_file_path=output_file_path,
                window_size=w,
                sigma=s,
            )
            output_video = Video(output_file_path)

        return output_video or input_video

    def draw_selected_ui_uniform_settings(self) -> None:
        if not self.ui_app_state.current_node_dir:
            return

        ui_node = self.ui_nodes.get(self.ui_app_state.current_node_dir)

        if (
            not ui_node
            or not ui_node.node.program
            or not self.ui_current_node_state.selected_uniform_name
            or self.ui_current_node_state.selected_uniform_name
            not in ui_node.node.program
        ):
            return

        uniform = ui_node.node.program[self.ui_current_node_state.selected_uniform_name]
        if not isinstance(uniform, moderngl.Uniform):
            return

        ui_uniform = self.ui_current_node_state.ui_uniforms[get_uniform_hash(uniform)]
        imgui.text(f"{ui_uniform.name} - {ui_uniform.input_type}")
        imgui.separator()
        imgui.spacing()

        value = ui_node.node.get_uniform_value(ui_uniform.name)

        if ui_uniform.input_type == "texture":
            media: MediaWithTexture = value
            imgui.text(get_resolution_str(ui_uniform.name, *media.texture.size))

            max_image_width = imgui.get_content_region_available()[0]
            max_image_height = 0.5 * imgui.get_content_region_available()[1]
            image_aspect = np.divide(*media.texture.size)
            image_width = min(max_image_width, max_image_height * image_aspect)
            image_height = min(max_image_height, max_image_width / image_aspect)

            imgui.image(
                media.texture.glo,
                width=image_width,
                height=image_height,
                uv0=(0, 1),
                uv1=(1, 0),
            )

            imgui.new_line()
            imgui.separator()
            imgui.spacing()

            output_media = self.draw_media_models(media)  # type: ignore
            if isinstance(output_media, Video):
                output_media = self.draw_video_filters(output_media)

            if media != output_media:
                ui_node.node.set_uniform_value(ui_uniform.name, output_media)

        if (
            ui_uniform.input_type in ("array", "text")
            and ui_uniform.gl_type == GL_UNSIGNED_INT
        ):
            current_idx = 0 if ui_uniform.input_type == "array" else 1
            new_idx = imgui.combo(
                "Input type##ui_uniform", current_idx, items=["array", "text"]
            )[1]

            ui_uniform.input_type = "array" if new_idx == 0 else "text"

        if ui_uniform.input_type == "text":
            text = unicode_to_str(value)
            imgui.text_colored(text, *(0.5, 0.5, 0.5))
            imgui.text(f"Length: {len(text)}")

    def draw_ui_uniform(self, ui_uniform: UIUniform) -> None:
        if not self.ui_app_state.current_node_dir:
            return

        ui_node = self.ui_nodes[self.ui_app_state.current_node_dir]
        value = ui_node.node.get_uniform_value(ui_uniform.name)

        if value is None:
            return

        if ui_uniform.input_type == "auto":
            if ui_uniform.dimension == 1:
                imgui.text(f"{ui_uniform.name}: {value:.3f}")
            else:
                if isinstance(value, Iterable):
                    value_str = ", ".join(f"{v:.3f}" for v in value)
                    imgui.text(f"{ui_uniform.name}: [{value_str}]")
                else:
                    imgui.text(f"{ui_uniform.name}: {value}")

        elif ui_uniform.input_type == "array":
            py_type = {GL_FLOAT: float, GL_UNSIGNED_INT: int}.get(ui_uniform.gl_type)

            if py_type is not None:
                value_str = ", ".join(map(str, value))
                is_changed, value_str = imgui.input_text(ui_uniform.name, value_str)
                if is_changed:
                    with contextlib.suppress(Exception):
                        value = [py_type(x.strip()) for x in value_str.split(",")]
            else:
                value_str = ", ".join(f"{v:.3f}" for v in value)
                imgui.text(
                    f"{ui_uniform.name}[{ui_uniform.array_length}]: [{value_str}]"
                )

        elif ui_uniform.input_type == "text":
            text = unicode_to_str(value)
            is_changed, text = imgui.input_text(ui_uniform.name, text)

            if is_changed:
                value = str_to_unicode(text, ui_uniform.array_length)
                ui_node.node.set_uniform_value(ui_uniform.name, value)

        elif ui_uniform.input_type == "texture":
            media: MediaWithTexture = value
            image_height = 90
            image_width = (
                image_height * media.texture.width / max(media.texture.height, 1)
            )

            imgui.text(ui_uniform.name)

            n_styles = 0
            if self.ui_current_node_state.selected_uniform_name == ui_uniform.name:
                color = (0.0, 1.0, 0.0, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON, *color)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *color)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *color)
                n_styles += 3

            if imgui.image_button(
                media.texture.glo,
                width=image_width,
                height=image_height,
                uv0=(0, 1),
                uv1=(1, 0),
            ):
                self.ui_current_node_state.selected_uniform_name = ui_uniform.name

            imgui.pop_style_color(n_styles)

            imgui.same_line()
            if imgui.button(f"Load##{ui_uniform.name}"):
                image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
                video_extensions = [".mp4", ".webm"]
                filter = ["*" + ext for ext in image_extensions + video_extensions]
                file_path = Path(
                    crossfiledialog.open_file(
                        title="Select image or video", filter=filter
                    )
                    or ""
                )

                media_cls = None
                if file_path.suffix in image_extensions:
                    media_cls = Image
                elif file_path.suffix in video_extensions:
                    media_cls = Video  # type: ignore

                if media_cls:
                    value = media_cls(file_path)
                    self.ui_current_node_state.selected_uniform_name = ui_uniform.name

        elif ui_uniform.input_type == "color":
            fn = getattr(imgui, f"color_edit{ui_uniform.dimension}")
            value = fn(ui_uniform.name, *value)[1]

        elif ui_uniform.input_type == "drag":
            change_speed = 0.01
            if ui_uniform.dimension == 1:
                value = imgui.drag_float(ui_uniform.name, value, change_speed)[1]
            else:
                fn = getattr(imgui, f"drag_float{ui_uniform.dimension}")
                value = fn(ui_uniform.name, *value, change_speed)[1]

        ui_node.node.set_uniform_value(ui_uniform.name, value)  # type: ignore

        if ui_uniform.input_type != "auto" and (
            imgui.is_item_clicked() or imgui.is_item_active()
        ):
            self.ui_current_node_state.selected_uniform_name = ui_uniform.name

    def draw_tg_stickers_tab(self) -> None:
        # ----------------------------------------------------------------
        # Tg settings
        imgui.text_colored(
            "This token will be stored in the project's files, so do not accidentally commit it.",
            *(1.0, 1.0, 0.0),
        )
        self.ui_app_state.tg_bot_token = imgui.input_text(
            "Bot token",
            self.ui_app_state.tg_bot_token,
            flags=imgui.INPUT_TEXT_PASSWORD,
        )[1]
        imgui.spacing()

        self.ui_app_state.tg_user_id = imgui.input_text(
            "User id",
            self.ui_app_state.tg_user_id,
            flags=imgui.INPUT_TEXT_CHARS_DECIMAL,
        )[1]

        self.ui_app_state.tg_sticker_set_name = imgui.input_text(
            "Sticker set name",
            self.ui_app_state.tg_sticker_set_name,
            flags=imgui.INPUT_TEXT_CHARS_NO_BLANK,
        )[1]

        if imgui.button("Fetch", width=80):
            self.fetch_tg_stickers()

        imgui.new_line()
        imgui.separator()
        imgui.spacing()

        available_width, available_height = imgui.get_content_region_available()
        sticker_grid_width = 0.3 * available_width

        imgui.begin_child(
            "sticker_grid",
            width=sticker_grid_width,
            height=available_height,
            border=True,
        )

        # ----------------------------------------------------------------
        # Sticker previews
        time = glfw.get_time()
        for i, sticker in enumerate(self._tg_stickers):
            sticker.update(time)

            n_styles = 0
            if i == self._tg_selected_sticker_idx:
                color = (0.0, 1.0, 0.0, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON, *color)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *color)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *color)
                n_styles += 3

            texture = sticker.get_thumbnail_texture()
            image_height = 90
            image_width = image_height * texture.width // max(texture.height, 1)

            if imgui.image_button(
                texture.glo,
                width=image_width,
                height=image_height,
                uv0=(0, 1),
                uv1=(1, 0),
            ):
                self._tg_selected_sticker_idx = i

            imgui.pop_style_color(n_styles)

            imgui.same_line()
            if sticker._sticker and imgui.button(f"Delete##{id(sticker)}"):
                self._loop.run_until_complete(
                    self._tg_stickers_bot.delete_sticker_from_set(sticker._sticker)
                )
                self.fetch_tg_stickers()
            elif not sticker._sticker:
                imgui.text_colored("New sticker", *(0.5, 0.5, 0.5))

            imgui.spacing()

        imgui.end_child()

        imgui.same_line()
        imgui.begin_child("selected_sticker_settings", border=True)

        # ----------------------------------------------------------------
        # Selected sticker settings
        if self._tg_selected_sticker_idx < len(self._tg_stickers):
            sticker = self._tg_stickers[self._tg_selected_sticker_idx]

            self.draw_media_details(
                details=sticker.video.details
                if sticker.video
                else sticker.image.details,
                is_changeable=False,
            )

            imgui.separator()

            file_path = Path(sticker.render_media_details.file_details.path)
            if sticker.render_media_details.is_video:
                file_path = file_path.with_suffix(".webm")
            else:
                file_path = file_path.with_suffix(".webp")
            sticker.render_media_details.file_details.path = str(file_path)

            sticker.render_media_details = self.draw_media_details(
                sticker.render_media_details
            )
            is_rendered, sticker.render_media_details = self.draw_render_button(
                sticker.render_media_details,
                sticker.preview_canvas.texture,
            )
            if is_rendered:
                sticker.log_message = UIMessage.success("Rendered!")

            if file_path.exists():
                imgui.same_line()
                if imgui.button("Submit##sticker"):
                    input_sticker = tg.InputSticker(
                        sticker=file_path.read_bytes(),
                        emoji_list=["😁"],
                        format=(
                            tg.constants.StickerFormat.VIDEO
                            if sticker.render_media_details.is_video
                            else tg.constants.StickerFormat.STATIC
                        ),
                    )

                    try:
                        if sticker._sticker is not None:
                            self._loop.run_until_complete(
                                self._tg_stickers_bot.replace_sticker_in_set(
                                    user_id=int(self.ui_app_state.tg_user_id),
                                    name=f"test_by_{self._tg_stickers_bot.username}",
                                    old_sticker=sticker._sticker,
                                    sticker=input_sticker,
                                )
                            )
                        else:
                            self._loop.run_until_complete(
                                self._tg_stickers_bot.add_sticker_to_set(
                                    user_id=int(self.ui_app_state.tg_user_id),
                                    name=f"test_by_{self._tg_stickers_bot.username}",
                                    sticker=input_sticker,
                                )
                            )

                        self.fetch_tg_stickers()
                    except Exception as e:
                        sticker.log_message = UIMessage.error(f"Failed to submit: {e}")
                        logger.error(e)

            imgui.same_line()
            imgui.text_colored(
                sticker.log_message.text, *sticker.log_message.get_color()
            )

        imgui.end_child()

    def draw_render_button(
        self,
        details: MediaDetails,
        preview_texture: moderngl.Texture | None,
    ) -> tuple[bool, MediaDetails]:
        ui_node = self.ui_nodes[self.ui_app_state.current_node_dir]  # type: ignore

        # ----------------------------------------------------------------
        # Preview canvas
        if preview_texture is not None:
            imgui.text("Render preview:")
            imgui.image(
                preview_texture.glo,
                width=preview_texture.size[0],
                height=preview_texture.size[1],
                uv0=(0, 1),
                uv1=(1, 0),
                border_color=(0.2, 0.2, 0.2, 1.0),
            )

        # ----------------------------------------------------------------
        # Render button
        is_rendered = False
        media_type = "video" if details.is_video else "image"
        if details.file_details.path:
            if imgui.button("Render##media"):
                try:
                    details = ui_node.node.render_media(details)
                    is_rendered = True
                except Exception as e:
                    logger.error(f"Failed to render media: {e}")
        else:
            imgui.text_colored(
                f"Select output file path to render the {media_type}", *(1.0, 1.0, 0.0)
            )

        return is_rendered, details

    @staticmethod
    def draw_file_details(
        details: FileDetails,
        extensions: Sequence[str] | None = None,
        is_changeable: bool = True,
    ) -> FileDetails:
        details = details.model_copy()

        file_path = None
        if is_changeable and imgui.button("File:##file_path"):
            file_path = crossfiledialog.save_file(
                title="File path",
            )
            if file_path:
                extension = Path(file_path).suffix
                if extensions and extension not in extensions:
                    details.path = ""
                    logger.warning(
                        f"Can't select {extension} file, "
                        f"available extensions are: {extensions}"
                    )
                else:
                    details.path = file_path
        elif not is_changeable:
            imgui.text("File:")

        if details.path:
            imgui.same_line()
            imgui.text_colored(str(details.path), 0.5, 0.5, 0.5)
            imgui.text(f"File size: {details.size // 1024} KB")

        return details

    def draw_resolution_details(
        self,
        details: ResolutionDetails,
        aspect: float | None = None,
        is_changeable: bool = True,
    ) -> ResolutionDetails:
        details = details.model_copy()

        if not is_changeable:
            imgui.text(f"Resolution: {details.width}x{details.height}")
            return details

        ui_node = self.ui_nodes[self.ui_app_state.current_node_dir]  # type: ignore

        is_width_changed, new_width = imgui.drag_int(
            "Width", details.width, min_value=16, max_value=2560
        )
        is_height_changed, new_height = imgui.drag_int(
            "Height", details.height, min_value=16, max_value=2560
        )

        if aspect is not None:
            if is_height_changed:
                new_width = new_height * aspect
            elif is_width_changed:
                new_height = new_width / aspect

        details.width = new_width
        details.height = new_height

        width, height = ui_node.node.canvas.texture.size
        if imgui.button(f"{width}x{height}") or not details.width or not details.height:
            details.width = width
            details.height = height

        imgui.same_line()
        width, height = adjust_size(ui_node.node.canvas.texture.size, max_size=512)
        if imgui.button(f"{width}x{height}") or not details.width or not details.height:
            details.width = width
            details.height = height

        return details

    def draw_media_details(
        self,
        details: MediaDetails,
        is_changeable: bool = True,
    ) -> MediaDetails:
        details = details.model_copy()

        if self.ui_app_state.current_node_dir:
            ui_node = self.ui_nodes[self.ui_app_state.current_node_dir]
            aspect = np.divide(*ui_node.node.canvas.texture.size)
        else:
            aspect = None

        output_type_name = "video" if details.is_video else "image"
        if is_changeable:
            options = ["video", "image"]
            idx = imgui.combo(
                "Output type##render_output_type_idx",
                options.index(output_type_name),
                items=options,
            )[1]
            details.is_video = idx == 0
        else:
            imgui.text(f"Output type: {output_type_name}")

        if details.is_video and is_changeable:
            details.quality = imgui.combo(
                "Quality##video_quality_idx",
                details.quality,
                items=["low", "medium-low", "medium-high", "high"],
            )[1]
            details.fps = imgui.drag_int(
                "FPS##video_fps", details.fps, min_value=10, max_value=60
            )[1]
            details.duration = imgui.drag_float(
                "Duration, sec##video_duration",
                details.duration,
                change_speed=0.1,
                min_value=1.0,
                max_value=60.0,
            )[1]
        elif details.is_video:
            imgui.text(f"FPS: {details.fps}")
            imgui.text(f"Duration: {details.duration} sec")

        details.resolution_details = self.draw_resolution_details(
            details.resolution_details,
            aspect=aspect,
            is_changeable=is_changeable,
        )
        details.file_details = self.draw_file_details(
            details.file_details,
            extensions=[".webm", ".mp4"]
            if details.is_video
            else [".png", ".jpeg", ".webp"],
            is_changeable=is_changeable,
        )

        return details

    def draw_node_settings(self) -> None:
        with imgui.begin_child("node_settings", border=True):
            if imgui.begin_tab_bar("node_settings_tabs").opened:
                if imgui.begin_tab_item("Node").selected:  # type: ignore
                    self.draw_node_tab()
                    imgui.end_tab_item()

                if imgui.begin_tab_item("Render").selected:  # type: ignore
                    self.ui_current_node_state.render_media_details = (
                        self.draw_media_details(
                            self.ui_current_node_state.render_media_details
                        )
                    )
                    _, self.ui_current_node_state.render_media_details = (
                        self.draw_render_button(
                            self.ui_current_node_state.render_media_details,
                            self.preview_canvas.texture,
                        )
                    )
                    imgui.end_tab_item()

                if imgui.begin_tab_item("Tg stickers").selected:  # type: ignore
                    self.draw_tg_stickers_tab()
                    imgui.end_tab_item()

                imgui.end_tab_bar()

    def draw_all(self) -> None:
        imgui.push_font(self._font_14)

        window_width, window_height = glfw.get_window_size(self.window)

        # ----------------------------------------------------------------
        # Main window
        imgui.set_next_window_size(window_width, window_height)
        imgui.set_next_window_position(0, 0)
        imgui.begin(
            "ShaderBox",
            flags=imgui.WINDOW_NO_COLLAPSE
            | imgui.WINDOW_ALWAYS_AUTO_RESIZE
            | imgui.WINDOW_NO_TITLE_BAR
            | imgui.WINDOW_NO_SCROLLBAR
            | imgui.WINDOW_NO_SCROLL_WITH_MOUSE,
        )

        control_panel_min_height = 600

        self.process_hotkeys()

        # ----------------------------------------------------------------
        # Main menu bar
        if imgui.button("New node"):
            self._active_popup_label = self._NODE_CREATOR_POPUP_LABEL

        imgui.same_line()
        if imgui.button("Settings"):
            self._active_popup_label = self._SETTINGS_POPUP_LABEL

        imgui.same_line()
        imgui.text(f"Global FPS: {round(self.global_fps)}")

        # ----------------------------------------------------------------
        # Current node image
        cursor_pos = imgui.get_cursor_screen_pos()

        if not self.ui_app_state.current_node_dir:
            image_width, image_height = imgui.get_content_region_available()
            image_height = max(image_height - control_panel_min_height, 400)

            message = "To create a new node, press Ctrl+N"
            text_size = imgui.calc_text_size(message)
            text_x = cursor_pos[0] + (image_width - text_size[0]) / 2
            text_y = cursor_pos[1] + (image_height - text_size[1]) / 2

            draw_list = imgui.get_window_draw_list()
            draw_list.add_text(
                text_x,
                text_y,
                imgui.color_convert_float4_to_u32(1.0, 1.0, 0.0, 1.0),
                message,
            )
        else:
            ui_node = self.ui_nodes[self.ui_app_state.current_node_dir]

            min_image_height = 100
            max_image_height = max(
                min_image_height,
                imgui.get_content_region_available()[1] - control_panel_min_height - 10,
            )
            max_image_width = imgui.get_content_region_available()[0]
            image_aspect = np.divide(*ui_node.node.canvas.texture.size)
            image_width = min(max_image_width, max_image_height * image_aspect)
            image_height = min(max_image_height, max_image_width / image_aspect)

            has_error = ui_node.node.shader_error != ""
            imgui.image(
                ui_node.node.canvas.texture.glo,
                width=image_width,
                height=image_height,
                uv0=(0, 1),
                uv1=(1, 0),
                tint_color=(0.2, 0.2, 0.2, 1.0) if has_error else (1.0, 1.0, 1.0, 1.0),
                border_color=(0.2, 0.2, 0.2, 1.0),
            )

            if has_error:
                draw_list = imgui.get_window_draw_list()
                text_size = imgui.calc_text_size(ui_node.node.shader_error)
                text_x = cursor_pos[0] + 10.0
                text_y = cursor_pos[1] + 10.0
                draw_list.add_text(
                    text_x,
                    text_y,
                    imgui.color_convert_float4_to_u32(1.0, 0.0, 0.0, 1.0),
                    ui_node.node.shader_error,
                )

        imgui.set_cursor_screen_pos((cursor_pos[0], cursor_pos[1] + image_height + 10))  # type: ignore

        # ----------------------------------------------------------------
        # Control panel
        region_width, region_height = imgui.get_content_region_available()
        control_panel_height = max(control_panel_min_height, region_height)
        control_panel_width = region_width
        with imgui.begin_child(
            "control_panel",
            width=control_panel_width,
            height=control_panel_height,
            border=False,
        ):
            node_preview_width = control_panel_width / 2.6
            self.draw_node_preview_grid(node_preview_width, control_panel_height)
            imgui.same_line()
            self.draw_node_settings()

        # ----------------------------------------------------------------
        # Popups
        if self._active_popup_label is not None and not imgui.is_popup_open(
            self._active_popup_label
        ):
            imgui.open_popup(self._active_popup_label)

        self.draw_popup_if_opened(
            self._SETTINGS_POPUP_LABEL,
            self.draw_settings,
        )
        self.draw_popup_if_opened(
            self._NODE_CREATOR_POPUP_LABEL,
            self.draw_node_creator,
        )

        imgui.pop_font()

    def process_hotkeys(self) -> None:
        io = imgui.get_io()

        if io.key_alt and imgui.is_key_pressed(ord("S")):
            self._active_popup_label = self._SETTINGS_POPUP_LABEL
        if io.key_ctrl and imgui.is_key_pressed(ord("N")):
            self._active_popup_label = self._NODE_CREATOR_POPUP_LABEL
        if io.key_ctrl and imgui.is_key_pressed(ord("S")):
            self.save()
        if io.key_ctrl and imgui.is_key_pressed(ord("E")):
            self.edit_current_node_fs_file()
        if io.key_ctrl and imgui.is_key_pressed(ord("D")):
            self.delete_current_node()

        if not imgui.is_any_item_active():
            if not self._active_popup_label:
                if imgui.is_key_pressed(glfw.KEY_LEFT, repeat=True):
                    self.select_next_current_node(-1)
                if imgui.is_key_pressed(glfw.KEY_RIGHT, repeat=True):
                    self.select_next_current_node(+1)
            if self._active_popup_label == self._NODE_CREATOR_POPUP_LABEL:
                if imgui.is_key_pressed(glfw.KEY_LEFT, repeat=True):
                    self.select_next_template(-1)
                if imgui.is_key_pressed(glfw.KEY_RIGHT, repeat=True):
                    self.select_next_template(+1)
                if imgui.is_key_pressed(glfw.KEY_ENTER, repeat=False):
                    self.create_node_from_selected_template()
                    self._active_popup_label = None

    def run(self) -> None:
        while not glfw.window_should_close(self.window):
            start_time = glfw.get_time()

            self.update_and_draw()

            elapsed_time = glfw.get_time() - start_time
            time.sleep(max(0.0, 1.0 / 60.0 - elapsed_time))

            if imgui.is_key_pressed(glfw.KEY_ESCAPE, repeat=False):
                if self._active_popup_label is None:
                    glfw.set_window_should_close(self.window, True)

                self._active_popup_label = None

            fps = 1.0 / (glfw.get_time() - start_time)
            if self.global_fps <= 0.0:
                self.global_fps = fps
            else:
                self.global_fps = 0.95 * self.global_fps + 0.05 * fps

        self.save()
        self.release()

    def update_and_draw(self) -> None:
        # ----------------------------------------------------------------
        # Prepare frame
        gl = moderngl.get_context()
        glfw.poll_events()

        # ----------------------------------------------------------------
        # Check for shader file changes and reload nodes
        for name in list(self.ui_nodes.keys()):
            fs_file_path = self.nodes_dir / name / "shader.frag.glsl"

            if not fs_file_path.exists():
                return

            fs_file_mtime = fs_file_path.lstat().st_mtime
            if fs_file_mtime != self.ui_nodes[name].mtime:
                logger.info(f"Reloading node {name} due to shader file change")
                ui_node = self.ui_nodes[name]
                ui_node.node.release_program(fs_file_path.read_text())
                ui_node.mtime = fs_file_mtime

        # ----------------------------------------------------------------
        # Render previews
        if self.ui_app_state.current_node_dir:
            ui_node = self.ui_nodes[self.ui_app_state.current_node_dir]
            preview_size = adjust_size(ui_node.node.canvas.texture.size, width=200)

            def _render_preview(canvas: Canvas) -> None:
                canvas.set_size(preview_size)
                ui_node.node.render(canvas=canvas)

            # Render current node preview
            _render_preview(self.preview_canvas)

            # Render current sticker preview
            if self._tg_selected_sticker_idx < len(self._tg_stickers):
                sticker = self._tg_stickers[self._tg_selected_sticker_idx]
                _render_preview(sticker.preview_canvas)

        # ----------------------------------------------------------------
        # Draw UI
        gl.screen.use()
        gl.clear()

        self.imgui_renderer.process_inputs()
        imgui.new_frame()

        self.draw_all()

        # Finalize frame
        imgui.end()
        imgui.render()

        glfw.make_context_current(self.window)
        moderngl.get_context().clear_errors()

        with contextlib.suppress(GLError):
            self.imgui_renderer.render(imgui.get_draw_data())

        glfw.swap_buffers(self.window)

        # ----------------------------------------------------------------
        # Render nodes
        if self._active_popup_label is None:
            for ui_node in self.ui_nodes.values():
                if (
                    self.ui_app_state.is_render_all_nodes
                    or ui_node == self.ui_nodes.get(self.ui_app_state.current_node_dir)
                    or self.frame_idx == 0
                ):
                    ui_node.node.render()
        elif self._active_popup_label == self._NODE_CREATOR_POPUP_LABEL:
            for ui_node in self.ui_node_templates.values():
                ui_node.node.render()

        self.frame_idx += 1


def main() -> None:
    ui = App()
    ui.run()


if __name__ == "__main__":
    main()
