import asyncio
import contextlib
import hashlib
import json
import math
import shutil
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import crossfiledialog
import cv2
import glfw
import imgui
import moderngl
import numpy as np
import telegram as tg
from imgui.integrations.glfw import GlfwRenderer
from loguru import logger
from OpenGL.GL import GL_SAMPLER_2D, GLError
from platformdirs import user_data_dir
from pydantic import BaseModel, computed_field

from shaderbox.core import (
    Canvas,
    FileDetails,
    Image,
    MediaDetails,
    Node,
    ResolutionDetails,
    Video,
)
from shaderbox.vendors import get_modelbox_bg_removal, get_modelbox_depthmap

_APP_START_TIME = int(time.time() * 1000)

_APP_DIR = Path(user_data_dir("shaderbox"))
_NODES_DIR = _APP_DIR / "nodes"
_TRASH_DIR = _APP_DIR / "trash"
_VIDEOS_DIR = _APP_DIR / "videos"
_IMAGES_DIR = _APP_DIR / "images"
_APP_STATE_FILE_PATH = _APP_DIR / "app_state.json"
_TG_STICKER_STATE_FILE_PATH = _APP_DIR / "tg_sticker.json"

_NODES_DIR.mkdir(exist_ok=True, parents=True)
_TRASH_DIR.mkdir(exist_ok=True, parents=True)
_VIDEOS_DIR.mkdir(exist_ok=True, parents=True)
_IMAGES_DIR.mkdir(exist_ok=True, parents=True)


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


def depthmap_to_normals(depthmap_image: Image, kernel_size: int = 3) -> Image:
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


class UIUniform:
    def __init__(self, uniform: moderngl.Uniform) -> None:
        self.uniform = uniform
        self.name = uniform.name
        self.array_length = uniform.array_length
        self.dimension = uniform.dimension
        self.gl_type = uniform.gl_type  # type: ignore
        self.is_special = self.name in ("u_time", "u_aspect", "u_resolution")
        self.is_image = self.gl_type == GL_SAMPLER_2D
        self.is_color = (
            self.array_length == 1
            and self.dimension in [3, 4]
            and self.name.endswith("color")
        )

    @property
    def group_name(self) -> str:
        if self.array_length > 1 or self.is_special:
            return "special"
        elif self.is_image:
            return "image"
        elif self.is_color:
            return "color"
        else:
            return "drag"

    @property
    def height(self) -> int:
        if self.group_name == "image":
            return 120
        elif self.group_name == "drag":
            spacing: int = imgui.get_text_line_height_with_spacing()
            return 5 + spacing
        else:
            return imgui.get_text_line_height_with_spacing()  # type: ignore


class UIMessage(BaseModel):
    text: str = ""
    level: Literal["success", "warning", "error"]

    @computed_field
    @property
    def color(self) -> tuple[float, float, float]:
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
    def __init__(self, sticker: tg.Sticker | None = None):
        self._sticker = sticker

        self.video: Video | None = None
        self.image: Image = Image.from_color((512, 512), (0.10, 0.24, 0.39))

        self.preview_canvas = Canvas()

        render_file_name = f"{_APP_START_TIME}_{hashlib.md5(str(id(self)).encode()).hexdigest()[:8]}.webm"
        render_file_path = _TRASH_DIR / render_file_name
        self.render_media_details: MediaDetails = MediaDetails(is_video=True)
        self.render_media_details.file_details.path = str(render_file_path)

        self.log_message: UIMessage = UIMessage(
            text="Submit button will be available after render", level="warning"
        )

    async def load(self) -> "UITgSticker":
        render_file_path = Path(self.render_media_details.file_details.path)

        if self._sticker is not None:
            bot = self._sticker.get_bot()

            if self._sticker.is_video:
                file_name = self._sticker.file_id + ".webm"
                file_path = _VIDEOS_DIR / file_name

                file = await bot.get_file(self._sticker.file_id)
                await file.download_to_drive(file_path)

                self.video = Video(file_path)
                render_file_path = render_file_path.with_suffix(".webm")

            if self._sticker.thumbnail:
                file_name = self._sticker.thumbnail.file_id + ".webp"
                file_path = _IMAGES_DIR / file_name

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

    def update(self, current_time: float) -> None:
        if self.video:
            self.video.update(current_time)

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


class UINodeState(BaseModel):
    render_media_details: MediaDetails = MediaDetails()

    resolution_idx: int = 0
    selected_uniform_name: str = ""
    blur_kernel_size: int = 50
    normals_kernel_size: int = 3


class UIAppState(BaseModel):
    tg_bot_token: str = ""
    tg_user_id: str = ""
    tg_sticker_set_name: str = ""


class UITgStickerState(BaseModel):
    video_details: MediaDetails = MediaDetails()


class App:
    def __init__(self) -> None:
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

        self.current_node_name: str | None = None
        self.nodes: dict[str, Node] = {}
        self.node_mtimes: dict[str, float] = {}

        self.preview_canvas = Canvas()

        imgui.create_context()
        self.window = window
        self.imgui_renderer = GlfwRenderer(window)

        # ----------------------------------------------------------------
        # Load nodes
        self._node_ui_state = defaultdict(UINodeState)
        node_dirs = sorted(_NODES_DIR.iterdir(), key=lambda x: x.stat().st_ctime)
        for node_dir in node_dirs:
            if node_dir.is_dir():
                node, mtime, meta = Node.load_from_dir(node_dir)
                name = node_dir.name

                self.nodes[name] = node
                self.node_mtimes[name] = mtime

                ui_state_dict = meta.get("ui_state", {})
                fields = UINodeState.model_fields

                invalid_keys = [k for k in ui_state_dict if k not in fields]
                if invalid_keys:
                    logger.warning(
                        f"Ignored invalid UINodeState keys for node '{name}': {invalid_keys}"
                    )

                filtered_ui_state = {
                    k: v for k, v in ui_state_dict.items() if k in fields
                }

                self._node_ui_state[name] = UINodeState(**filtered_ui_state)
                self.current_node_name = name

        # ----------------------------------------------------------------
        # Load ui states
        self.ui_app_state = UIAppState()
        self.ui_tg_sticker_state = UITgStickerState()
        states: list[tuple[str, Path, type[BaseModel]]] = [
            ("ui_app_state", _APP_STATE_FILE_PATH, UIAppState),
            (
                "ui_tg_sticker_state",
                _TG_STICKER_STATE_FILE_PATH,
                UITgStickerState,
            ),
        ]

        for state_field_name, state_file_path, state_class in states:
            if state_file_path.exists():
                with state_file_path.open("r") as f:
                    state_dict = json.load(f)
                    state_dict = {
                        k: v
                        for k, v in state_dict.items()
                        if k in state_class.model_fields
                    }
                    setattr(self, state_field_name, state_class(**state_dict))

        # ----------------------------------------------------------------
        # Tg
        self._tg_stickers: list[UITgSticker] = []
        self._tg_selected_sticker_idx: int = 0
        self._tg_stickers_bot: tg.Bot

    def fetch_tg_stickers(self) -> None:
        async def _fetch() -> list[UITgSticker]:
            self._tg_stickers_bot = tg.Bot(token=self.ui_app_state.tg_bot_token)
            await self._tg_stickers_bot.initialize()

            sticker_set = await self._tg_stickers_bot.get_sticker_set(
                name=self.ui_app_state.tg_sticker_set_name
            )
            coros = [UITgSticker(s).load() for s in sticker_set.stickers]
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

        new_sticker = UITgSticker()
        self._tg_stickers.insert(0, new_sticker)

    @property
    def ui_current_node_state(self) -> UINodeState:
        if self.current_node_name is None:
            return UINodeState()
        return self._node_ui_state[self.current_node_name]

    def edit_node_fs_file(self, node_name: str) -> None:
        fs_file_path = _NODES_DIR / node_name / "shader.frag.glsl"
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
        if self.current_node_name:
            self.edit_node_fs_file(self.current_node_name)
        else:
            logger.warning("Nothing to edit")

    def save_node(self, node_name: str) -> None:
        node = self.nodes[node_name]
        node_dir = _NODES_DIR / node_name
        node_dir.mkdir(exist_ok=True, parents=True)

        ui_state_dict = self._node_ui_state[node_name].model_dump()
        meta: dict[str, Any] = {
            "canvas_size": list(node.canvas.texture.size),
            "uniforms": {},
            "ui_state": ui_state_dict,
        }

        fs_file_path = node_dir / "shader.frag.glsl"
        if not fs_file_path.exists():
            with fs_file_path.open("w") as f:
                f.write(node.fs_source)
                self.node_mtimes[node_name] = fs_file_path.lstat().st_mtime

        # ----------------------------------------------------------------
        # Save uniforms
        for uniform in node.get_uniforms():
            if uniform.name in ["u_time", "u_aspect", "u_resolution"]:
                continue

            value = node.get_uniform_value(uniform.name)

            if uniform.gl_type == GL_SAMPLER_2D:  # type: ignore
                textures_dir = node_dir / "textures"
                textures_dir.mkdir(exist_ok=True)
                texture_filename = f"{uniform.name}.png"

                Image(value)._image.save(textures_dir / texture_filename, format="PNG")
                meta["uniforms"][uniform.name] = {
                    "type": "texture",
                    "width": value.width,
                    "height": value.height,
                    "file_path": f"textures/{texture_filename}",
                }
            else:
                if isinstance(value, int | float):
                    meta["uniforms"][uniform.name] = value
                elif isinstance(value, tuple):
                    meta["uniforms"][uniform.name] = list(value)
                else:
                    logger.warning(
                        f"Skipping unsupported uniform type for {uniform.name}: {type(value)}"
                    )

        with (node_dir / "node.json").open("w") as f:
            json.dump(meta, f, indent=4)

        logger.info(f"Node {node_name} saved: {node_dir}")

    def save_current_node(self) -> None:
        if self.current_node_name:
            self.save_node(self.current_node_name)
        else:
            logger.warning("Nothing to save")

    def save_app_state(self) -> None:
        app_state_dict = self.ui_app_state.model_dump()
        with _APP_STATE_FILE_PATH.open("w") as f:
            json.dump(app_state_dict, f, indent=4)

    def save(self) -> None:
        self.save_current_node()
        self.save_app_state()

    def create_new_current_node(self) -> None:
        node = Node()
        name = hashlib.md5(f"{id(node)}{time.time()}".encode()).hexdigest()[:8]

        self.nodes[name] = node
        self.current_node_name = name

        self.save_node(name)
        logger.info(f"Node created: {name}")

    def delete_current_node(self) -> None:
        if self.current_node_name is None:
            logger.info("Current node is None, nothing to delete")
            return

        name = self.current_node_name
        node = self.nodes.pop(name)
        del self.node_mtimes[name]

        node.release()

        self.current_node_name = next(iter(self.nodes)) if self.nodes else None

        shutil.move(_NODES_DIR / name, _TRASH_DIR / name)
        logger.info(f"Node deleted: {name}")

    def select_next_current_node(self, step: int = +1) -> None:
        if self.current_node_name is None:
            return

        keys = list(self.nodes.keys())
        idx = keys.index(self.current_node_name)
        self.current_node_name = keys[(idx + step) % len(keys)]

    def draw_main_menu_bar(self) -> int:
        if imgui.begin_main_menu_bar().opened:
            if imgui.begin_menu("Node").opened:
                if imgui.menu_item("New node", "Ctrl+N", False)[1]:
                    self.create_new_current_node()

                if imgui.menu_item("Delete current node", "Ctrl+D", False)[1]:
                    self.delete_current_node()

                if imgui.menu_item("Select next node", "->", False)[1]:
                    self.select_next_current_node(+1)

                if imgui.menu_item("Select previous node", "<-", False)[1]:
                    self.select_next_current_node(-1)

                if imgui.menu_item("Save current node", "Ctrl+S", False)[1]:
                    self.save_current_node()

                if imgui.menu_item("Edit current node", "Ctrl+E", False)[1]:
                    self.edit_current_node_fs_file()

                imgui.end_menu()

            size = imgui.get_item_rect_size()
            main_menu_height: int = size[1]  # type: ignore
            imgui.end_main_menu_bar()
            return main_menu_height
        return 0

    def draw_node_preview_grid(self, width: int, height: int) -> None:
        with imgui.begin_child(
            "node_preview_grid", width=width, height=height, border=True
        ):
            preview_size = 125
            n_cols = int(imgui.get_content_region_available()[0] // (preview_size + 5))
            n_cols = max(1, n_cols)
            for i, (name, node) in enumerate(self.nodes.items()):
                if name == self.current_node_name:
                    if node.shader_error:
                        color = (1.0, 0.0, 0.0, 1.0)
                    else:
                        color = (0.0, 1.0, 0.0, 1.0)
                    imgui.push_style_color(imgui.COLOR_BORDER, *color)

                imgui.begin_child(
                    f"preview_{name}",
                    width=preview_size,
                    height=preview_size,
                    border=True,
                    flags=imgui.WINDOW_NO_SCROLLBAR,
                )

                if name == self.current_node_name:
                    imgui.pop_style_color()

                if imgui.invisible_button(
                    f"preview_button_{name}", width=preview_size, height=preview_size
                ):
                    self.current_node_name = name

                s = (preview_size - 10) / max(node.canvas.texture.size)
                image_width = node.canvas.texture.size[0] * s
                image_height = node.canvas.texture.size[1] * s

                imgui.set_cursor_pos_x((preview_size - image_width) / 2 - 1)
                imgui.set_cursor_pos_y((preview_size - image_height) / 2 - 1)

                imgui.image(
                    node.canvas.texture.glo,
                    width=image_width,
                    height=image_height,
                    uv0=(0, 1),
                    uv1=(1, 0),
                )

                imgui.end_child()

                if (i + 1) % n_cols != 0:
                    imgui.same_line()
                else:
                    imgui.spacing()

    def draw_node_tab(self) -> None:
        if self.current_node_name:
            node = self.nodes[self.current_node_name]
        else:
            return

        node_name = self.current_node_name
        fs_file_path = _NODES_DIR / node_name / "shader.frag.glsl"

        # ----------------------------------------------------------------
        # Collect resolution items
        imgui.spacing()

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
        for uniform in node.get_uniforms():
            if uniform.gl_type == GL_SAMPLER_2D:  # type: ignore
                texture = node.get_uniform_value(uniform.name)
                w, h = texture.size
                if (w, h) == node.canvas.texture.size:
                    matching_uniforms.append(uniform.name)
                else:
                    uniform_resolutions.append((w, h, uniform.name))
                    uniform_sizes.add((w, h))

        resolution_items = []

        for w, h, name in uniform_resolutions:
            resolution_items.append(get_resolution_str(name, w, h))

        for w, h in standard_resolutions:
            if (w, h) != node.canvas.texture.size and (w, h) not in uniform_sizes:
                resolution_items.append(get_resolution_str(None, w, h))

        # ----------------------------------------------------------------
        # Edit button
        if imgui.button("Edit", width=80):
            self.edit_node_fs_file(node_name)

        imgui.same_line()
        imgui.text_colored(str(fs_file_path), 0.5, 0.5, 0.5)

        # ----------------------------------------------------------------
        # Resolution combobox
        imgui.text(
            "Current resolution: " + get_resolution_str(None, *node.canvas.texture.size)
        )
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
            node.canvas.set_size((w, h))

        imgui.new_line()
        imgui.separator()
        imgui.spacing()

        # ----------------------------------------------------------------
        # Uniforms
        ui_uniforms = {u.name: UIUniform(u) for u in node.get_uniforms()}

        uniform_groups = defaultdict(list)
        for u in ui_uniforms.values():
            uniform_groups[u.group_name].append(u)

        imgui.begin_child(
            "uniform_groups",
            width=imgui.get_content_region_available_width() // 2,
        )
        for group_name, ui_uniforms_in_group in uniform_groups.items():
            total_height = (
                sum(ui_uniform.height for ui_uniform in ui_uniforms_in_group) + 20
            )

            imgui.push_style_color(imgui.COLOR_BORDER, 0.15, 0.15, 0.15)
            imgui.begin_child(
                f"{group_name}_group",
                height=total_height,
                border=True,
                flags=imgui.WINDOW_NO_SCROLLBAR,
            )

            for ui_uniform in ui_uniforms_in_group:
                self.draw_ui_uniform(ui_uniform)

            imgui.end_child()
            imgui.pop_style_color()

        imgui.end_child()

        selected_uniform_name = self._node_ui_state[
            self.current_node_name
        ].selected_uniform_name
        if selected_uniform_name:
            imgui.same_line()
            imgui.begin_child("selected_uniform_settings", border=True)
            self.draw_selected_uniform_settings()
            imgui.end_child()

    def draw_selected_uniform_settings(self) -> None:
        if self.current_node_name is None:
            return

        node = self.nodes[self.current_node_name]
        if not node.program:
            return

        if (
            not self.ui_current_node_state.selected_uniform_name
            or self.ui_current_node_state.selected_uniform_name not in node.program
        ):
            return

        uniform = node.program[self.ui_current_node_state.selected_uniform_name]
        if not isinstance(uniform, moderngl.Uniform):
            return

        ui_uniform = UIUniform(uniform)
        if ui_uniform.group_name == "image":
            texture = node.get_uniform_value(ui_uniform.name)

            imgui.text(get_resolution_str(ui_uniform.name, *texture.size))

            max_image_width = imgui.get_content_region_available()[0]
            max_image_height = 0.5 * imgui.get_content_region_available()[1]
            image_aspect = np.divide(*texture.size)
            image_width = min(max_image_width, max_image_height * image_aspect)
            image_height = min(max_image_height, max_image_width / image_aspect)

            imgui.image(
                texture.glo,
                width=image_width,
                height=image_height,
                uv0=(0, 1),
                uv1=(1, 0),
            )

            imgui.spacing()
            imgui.separator()

            if imgui.button("As depthmap"):
                try:
                    node.set_uniform_value(
                        ui_uniform.name,
                        get_modelbox_depthmap(
                            zero_low_alpha_pixels(Image(texture))
                        ).texture,
                    )
                except Exception as e:
                    logger.error(str(e))

            imgui.same_line()
            if imgui.button("Remove bg"):
                try:
                    node.set_uniform_value(
                        ui_uniform.name,
                        get_modelbox_bg_removal(
                            zero_low_alpha_pixels(Image(texture))
                        ).texture,
                    )
                except Exception as e:
                    logger.error(str(e))

            imgui.separator()
            imgui.text("Gaussian blur")
            self.ui_current_node_state.blur_kernel_size = imgui.slider_int(
                "##blur_kernel_size",
                self.ui_current_node_state.blur_kernel_size,
                min_value=10,
                max_value=100,
                format="%d",
            )[1]
            self.ui_current_node_state.blur_kernel_size = max(
                3, self.ui_current_node_state.blur_kernel_size | 1
            )

            imgui.same_line()
            if imgui.button("Apply##blur"):
                try:
                    node.set_uniform_value(
                        ui_uniform.name,
                        Image(
                            cv2.GaussianBlur(
                                np.array(Image(texture)._image.convert("RGB")),
                                (
                                    self.ui_current_node_state.blur_kernel_size,
                                    self.ui_current_node_state.blur_kernel_size,
                                ),
                                0,
                            )
                        ).texture,
                    )
                except Exception as e:
                    logger.error(str(e))

            imgui.separator()
            imgui.text("Normals")
            self.ui_current_node_state.normals_kernel_size = imgui.slider_int(
                "##normals_kernel_size",
                self.ui_current_node_state.normals_kernel_size,
                min_value=3,
                max_value=31,
                format="%d",
            )[1]
            self.ui_current_node_state.normals_kernel_size = max(
                3, self.ui_current_node_state.normals_kernel_size | 1
            )

            imgui.same_line()
            if imgui.button("Apply##normals"):
                try:
                    image = depthmap_to_normals(
                        get_modelbox_depthmap(zero_low_alpha_pixels(Image(texture))),
                        self.ui_current_node_state.normals_kernel_size,
                    )
                    node.set_uniform_value(ui_uniform.name, image.texture)
                except Exception as e:
                    logger.error(str(e))

    def draw_ui_uniform(self, ui_uniform: UIUniform) -> None:
        if self.current_node_name is None:
            return

        node = self.nodes[self.current_node_name]
        value = node.get_uniform_value(ui_uniform.name)

        if ui_uniform.group_name == "special":
            if ui_uniform.array_length > 1:
                value_str = ", ".join(f"{v:.3f}" for v in value)
                imgui.text(
                    f"{ui_uniform.name}[{ui_uniform.array_length}]: [{value_str}]"
                )
            elif ui_uniform.dimension == 1:
                imgui.text(f"{ui_uniform.name}: {value:.3f}")
            else:
                if isinstance(value, Iterable):
                    value_str = ", ".join(f"{v:.3f}" for v in value)
                    imgui.text(f"{ui_uniform.name}: [{value_str}]")
                else:
                    imgui.text(f"{ui_uniform.name}: {value}")

        elif ui_uniform.group_name == "image":
            texture = value
            image_height = 90
            image_width = image_height * texture.width / max(texture.height, 1)

            imgui.text(ui_uniform.name)

            n_styles = 0
            if self.ui_current_node_state.selected_uniform_name == ui_uniform.name:
                color = (0.0, 1.0, 0.0, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON, *color)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *color)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *color)
                n_styles += 3

            if imgui.image_button(
                texture.glo,
                width=image_width,
                height=image_height,
                uv0=(0, 1),
                uv1=(1, 0),
            ):
                self._node_ui_state[
                    self.current_node_name
                ].selected_uniform_name = ui_uniform.name

            imgui.pop_style_color(n_styles)

            imgui.same_line()
            if imgui.button(f"Load##{ui_uniform.name}"):
                file_path = crossfiledialog.open_file(
                    title="Select Texture",
                    filter=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"],
                )
                if file_path:
                    value = Image(file_path).texture

            imgui.spacing()

        elif ui_uniform.group_name == "color":
            fn = getattr(imgui, f"color_edit{ui_uniform.dimension}")
            value = fn(ui_uniform.name, *value)[1]

        elif ui_uniform.group_name == "drag":
            change_speed = 0.01
            if ui_uniform.dimension == 1:
                value = imgui.drag_float(ui_uniform.name, value, change_speed)[1]
            else:
                fn = getattr(imgui, f"drag_float{ui_uniform.dimension}")
                value = fn(ui_uniform.name, *value, change_speed)[1]

        node.set_uniform_value(ui_uniform.name, value)

    def draw_tg_stickers_tab(self) -> None:
        # ----------------------------------------------------------------
        # Tg settings
        self.ui_app_state.tg_bot_token = imgui.input_text(
            "Bot token",
            self.ui_app_state.tg_bot_token,
            flags=imgui.INPUT_TEXT_PASSWORD,
        )[1]

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
        current_time = glfw.get_time()
        for i, sticker in enumerate(self._tg_stickers):
            sticker.update(current_time)

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
            imgui.text_colored(sticker.log_message.text, *sticker.log_message.color)

        imgui.end_child()

    def draw_render_button(
        self,
        details: MediaDetails,
        preview_texture: moderngl.Texture | None,
    ) -> tuple[bool, MediaDetails]:
        node = self.nodes[self.current_node_name]  # type: ignore

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
                    details = node.render_media(details)
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
            imgui.text(f"File size: {details.size} B")

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

        node = self.nodes[self.current_node_name]  # type: ignore

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

        width, height = node.canvas.texture.size
        if imgui.button(f"{width}x{height}") or not details.width or not details.height:
            details.width = width
            details.height = height

        imgui.same_line()
        width, height = adjust_size(node.canvas.texture.size, max_size=512)
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

        if self.current_node_name:
            node = self.nodes[self.current_node_name]
            aspect = np.divide(*node.canvas.texture.size)
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
        window_width, window_height = glfw.get_window_size(self.window)

        # ----------------------------------------------------------------
        # Main window

        # Main menu bar
        main_menu_height = self.draw_main_menu_bar()

        imgui.set_next_window_size(window_width, window_height - main_menu_height)
        imgui.set_next_window_position(0, main_menu_height)
        imgui.begin(
            "ShaderBox",
            flags=imgui.WINDOW_NO_COLLAPSE
            | imgui.WINDOW_ALWAYS_AUTO_RESIZE
            | imgui.WINDOW_NO_TITLE_BAR
            | imgui.WINDOW_NO_SCROLLBAR,
        )

        control_panel_min_height = 600

        # ----------------------------------------------------------------
        # Current node image
        cursor_pos = imgui.get_cursor_screen_pos()

        if self.current_node_name is None:
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
            node = self.nodes[self.current_node_name]

            min_image_height = 100
            max_image_height = max(
                min_image_height,
                imgui.get_content_region_available()[1] - control_panel_min_height - 10,
            )
            max_image_width = imgui.get_content_region_available()[0]
            image_aspect = np.divide(*node.canvas.texture.size)
            image_width = min(max_image_width, max_image_height * image_aspect)
            image_height = min(max_image_height, max_image_width / image_aspect)

            has_error = node.shader_error != ""
            imgui.image(
                node.canvas.texture.glo,
                width=image_width,
                height=image_height,
                uv0=(0, 1),
                uv1=(1, 0),
                tint_color=(0.2, 0.2, 0.2, 1.0) if has_error else (1.0, 1.0, 1.0, 1.0),
                border_color=(0.2, 0.2, 0.2, 1.0),
            )

            if has_error:
                draw_list = imgui.get_window_draw_list()
                text_size = imgui.calc_text_size(node.shader_error)
                text_x = cursor_pos[0] + (image_width - text_size[0]) / 2
                text_y = cursor_pos[1] + (image_height - text_size[1]) / 2
                draw_list.add_text(
                    text_x,
                    text_y,
                    imgui.color_convert_float4_to_u32(1.0, 0.0, 0.0, 1.0),
                    node.shader_error,
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
            node_preview_width = control_panel_width / 3.0
            self.draw_node_preview_grid(node_preview_width, control_panel_height)
            imgui.same_line()
            self.draw_node_settings()

    def process_hotkeys(self) -> None:
        io = imgui.get_io()
        if io.key_ctrl and imgui.is_key_pressed(ord("N")):
            self.create_new_current_node()
        if io.key_ctrl and imgui.is_key_pressed(ord("D")):
            self.delete_current_node()
        if io.key_ctrl and imgui.is_key_pressed(ord("S")):
            self.save()
        if io.key_ctrl and imgui.is_key_pressed(ord("E")):
            self.edit_current_node_fs_file()

    def run(self) -> None:
        while not glfw.window_should_close(self.window):
            start_time = glfw.get_time()

            self.update_and_draw()

            elapsed_time = glfw.get_time() - start_time
            time.sleep(max(0.0, 1.0 / 60.0 - elapsed_time))

            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break

        self.save()

        for sticker in self._tg_stickers:
            sticker.release()

        for node in self.nodes.values():
            node.release()

    def update_and_draw(self) -> None:
        # ----------------------------------------------------------------
        # Prepare frame
        gl = moderngl.get_context()
        glfw.poll_events()

        # ----------------------------------------------------------------
        # Check for shader file changes and reload nodes
        for name in list(self.nodes.keys()):
            fs_file_path = _NODES_DIR / name / "shader.frag.glsl"

            if not fs_file_path.exists():
                return

            mtime = fs_file_path.lstat().st_mtime
            if mtime != self.node_mtimes[name]:
                logger.info(f"Reloading node {name} due to shader file change")
                self.nodes[name].release_program(fs_file_path.read_text())
                self.node_mtimes[name] = mtime
                self.save_node(name)

        # ----------------------------------------------------------------
        # Render previews
        if self.current_node_name:
            node = self.nodes[self.current_node_name]
            preview_size = adjust_size(node.canvas.texture.size, width=200)

            def _render_preview(canvas: Canvas, media_details: MediaDetails) -> None:
                canvas.set_size(preview_size)
                u_time = (
                    mod(glfw.get_time(), media_details.duration)
                    if media_details.is_video
                    else 0
                )
                node.render(u_time, canvas=canvas)

            # Render current node preview
            _render_preview(
                self.preview_canvas, self.ui_current_node_state.render_media_details
            )

            # Render current sticker preview
            if self._tg_selected_sticker_idx < len(self._tg_stickers):
                sticker = self._tg_stickers[self._tg_selected_sticker_idx]
                _render_preview(sticker.preview_canvas, sticker.render_media_details)

        # ----------------------------------------------------------------
        # Render all nodes
        for node in self.nodes.values():
            node.render()

        # ----------------------------------------------------------------
        # Draw UI
        gl.screen.use()
        gl.clear()

        self.process_hotkeys()
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


def main() -> None:
    ui = App()
    ui.run()


if __name__ == "__main__":
    main()
