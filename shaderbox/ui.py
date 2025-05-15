import asyncio
import contextlib
import hashlib
import io
import json
import math
import shutil
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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
from PIL import Image
from platformdirs import user_data_dir
from pydantic import BaseModel

from shaderbox.core import Node, Video, VideoDetails, image_to_texture, texture_to_image
from shaderbox.vendors import get_modelbox_bg_removal, get_modelbox_depthmap

_APP_DIR = Path(user_data_dir("shaderbox"))
_NODES_DIR = _APP_DIR / "nodes"
_TRASH_DIR = _APP_DIR / "trash"
_VIDEO_DIR = _APP_DIR / "video"
_APP_STATE_FILE_PATH = _APP_DIR / "app_state.json"
_TG_STICKER_STATE_FILE_PATH = _APP_DIR / "tg_sticker.json"

_NODES_DIR.mkdir(exist_ok=True, parents=True)
_TRASH_DIR.mkdir(exist_ok=True, parents=True)
_VIDEO_DIR.mkdir(exist_ok=True, parents=True)


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


class UITgSticker:
    def __init__(self, sticker: tg.Sticker):
        self._sticker = sticker

        self._video: Video | None = None
        self._thumbnail_texture: moderngl.Texture = image_to_texture(
            Image.new("RGBA", (512, 512), (255, 0, 0, 255))
        )

        self._file_path: Path | None = None

    @property
    def thumbnail_texture(self) -> moderngl.Texture:
        if self._video:
            return self._video.texture
        else:
            return self._texture

    async def load(self) -> "UITgSticker":
        bot = self._sticker.get_bot()

        if self._sticker.is_video:
            file_name = self._sticker.file_id + ".webm"
            file_path = _VIDEO_DIR / file_name

            file = await bot.get_file(self._sticker.file_id)
            await file.download_to_drive(self._file_path)

            self._file_path = file_path
            self._video = Video(file_path)

        if self._sticker.thumbnail:
            file = await bot.get_file(self._sticker.thumbnail.file_id)
            barray = await file.download_as_bytearray()
            self._texture = image_to_texture(Image.open(io.BytesIO(barray)))

        return self

    def update(self, current_time: float) -> None:
        if self._video:
            self._video.update(current_time)

    def release(self) -> None:
        if self._video:
            self._video.release()

        self._texture.release()


def get_resolution_str(name: str | None, w: int, h: int) -> str:
    g = math.gcd(w, h)
    w_ratio, h_ratio = w // g, h // g
    aspect = f"{w_ratio}:{h_ratio}"
    parts = [f"{w}x{h}"]
    if name:
        parts.append(name)
    parts.append(aspect)
    return " | ".join(parts)


def depthmap_to_normals(
    depthmap_image: Image.Image, kernel_size: int = 3
) -> Image.Image:
    kernel_size = max(3, kernel_size | 1)

    depth_array = np.array(depthmap_image.convert("L"), dtype=np.float32)

    grad_x = cv2.Sobel(depth_array, cv2.CV_32F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(depth_array, cv2.CV_32F, 0, 1, ksize=kernel_size)

    normals = np.dstack((grad_x, -grad_y, np.ones_like(depth_array)))

    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    norm[norm < 1e-6] = 1e-6
    normals = normals / norm

    normals_rgb = ((normals + 1.0) * 127.5).astype(np.uint8)

    return Image.fromarray(normals_rgb)


def zero_low_alpha_pixels(image: Image.Image, min_alpha: float = 1.0) -> Image.Image:
    img_array = np.array(image)
    alpha_threshold = int(min_alpha * 255)
    low_alpha_mask = img_array[:, :, 3] < alpha_threshold
    img_array[:, :, :3][low_alpha_mask] = 0

    return Image.fromarray(img_array, mode="RGBA")


class UINodeState(BaseModel):
    video_details: VideoDetails = VideoDetails()

    resolution_combo_idx: int = 0
    selected_uniform_name: str = ""
    blur_kernel_size: int = 50
    normals_kernel_size: int = 3


class UIAppState(BaseModel):
    tg_bot_token: str = ""
    tg_user_id: str = ""
    tg_sticker_set_name: str = ""


class UITgStickerState(BaseModel):
    video_details: VideoDetails = VideoDetails()


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

    def fetch_tg_stickers(self) -> None:
        async def _fetch() -> list[UITgSticker]:
            bot = tg.Bot(token=self.ui_app_state.tg_bot_token)
            sticker_set = await bot.get_sticker_set(
                name=self.ui_app_state.tg_sticker_set_name
            )
            coros = [UITgSticker(s).load() for s in sticker_set.stickers]
            return await asyncio.gather(*coros)

        for sticker in self._tg_stickers:
            sticker.release()

        try:
            self._tg_stickers = self._loop.run_until_complete(_fetch())
        except tg.error.InvalidToken:
            logger.error("Invalid telegram token")
        except tg.error.BadRequest as e:
            if str(e) == "Stickerset_invalid":
                logger.info("Sticker set doesn't exist")
            else:
                raise e

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
            "output_texture_size": list(node.output_texture_size),
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
                image = texture_to_image(value)
                textures_dir = node_dir / "textures"
                textures_dir.mkdir(exist_ok=True)
                texture_filename = f"{uniform.name}.png"
                image.save(textures_dir / texture_filename, format="PNG")
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
                        color = (1.0, 0.0, 0.0, 1.0)  # Red for error
                    else:
                        color = (0.0, 1.0, 0.0, 1.0)  # Green for selected
                    imgui.push_style_color(imgui.COLOR_BORDER, *color)

                # Use a unique ID for each node's child window
                imgui.begin_child(
                    f"preview_{name}",  # Fixed: unique ID using f-string
                    width=preview_size,
                    height=preview_size,
                    border=True,
                    flags=imgui.WINDOW_NO_SCROLLBAR,
                )

                if name == self.current_node_name:
                    imgui.pop_style_color()

                # Use a unique ID for the invisible button
                if imgui.invisible_button(
                    f"preview_button_{name}", width=preview_size, height=preview_size
                ):
                    self.current_node_name = name

                s = (preview_size - 10) / max(node.output_texture.size)
                image_width = node.output_texture.size[0] * s
                image_height = node.output_texture.size[1] * s

                imgui.set_cursor_pos_x((preview_size - image_width) / 2 - 1)
                imgui.set_cursor_pos_y((preview_size - image_height) / 2 - 1)

                imgui.image(
                    node.output_texture.glo,
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
                if (w, h) == node.output_texture_size:
                    matching_uniforms.append(uniform.name)
                else:
                    uniform_resolutions.append((w, h, uniform.name))
                    uniform_sizes.add((w, h))

        resolution_items = []

        for w, h, name in uniform_resolutions:
            resolution_items.append(get_resolution_str(name, w, h))

        for w, h in standard_resolutions:
            if (w, h) != node.output_texture_size and (w, h) not in uniform_sizes:
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
            "Current resolution: " + get_resolution_str(None, *node.output_texture_size)
        )
        if matching_uniforms:
            imgui.same_line()
            imgui.text_colored("(" + ", ".join(matching_uniforms) + ")", 0.5, 0.5, 0.5)

        self.ui_current_node_state.resolution_combo_idx = imgui.combo(
            "##resolution_combo_idx",
            self.ui_current_node_state.resolution_combo_idx,
            resolution_items,
        )[1]
        self.ui_current_node_state.resolution_combo_idx = min(
            self.ui_current_node_state.resolution_combo_idx, len(resolution_items) - 1
        )

        imgui.same_line()
        if imgui.button("Apply##resolution"):
            w, h = map(
                int,
                resolution_items[self.ui_current_node_state.resolution_combo_idx]
                .split(" | ")[0]
                .split("x"),
            )
            node.reset_output_texture_size((w, h))

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
                image = texture_to_image(texture)
                try:
                    depthmap_image = get_modelbox_depthmap(zero_low_alpha_pixels(image))
                    texture = image_to_texture(depthmap_image)
                    node.set_uniform_value(ui_uniform.name, texture)
                except Exception as e:
                    logger.error(str(e))

            imgui.same_line()
            if imgui.button("Remove bg"):
                image = texture_to_image(texture)
                try:
                    nobg_image = get_modelbox_bg_removal(zero_low_alpha_pixels(image))
                    texture = image_to_texture(nobg_image)
                    node.set_uniform_value(ui_uniform.name, texture)
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
                image = texture_to_image(texture)
                try:
                    img_array = np.array(image.convert("RGB"))
                    blurred_array = cv2.GaussianBlur(
                        img_array,
                        (
                            self.ui_current_node_state.blur_kernel_size,
                            self.ui_current_node_state.blur_kernel_size,
                        ),
                        0,
                    )
                    blurred_image = Image.fromarray(blurred_array).convert("RGBA")
                    texture = image_to_texture(blurred_image)
                    node.set_uniform_value(ui_uniform.name, texture)
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
                image = texture_to_image(texture)
                try:
                    depthmap_image = get_modelbox_depthmap(zero_low_alpha_pixels(image))
                    normals_image = depthmap_to_normals(
                        depthmap_image, self.ui_current_node_state.normals_kernel_size
                    )
                    texture = image_to_texture(normals_image)
                    node.set_uniform_value(ui_uniform.name, texture)
                except Exception as e:
                    logger.error(str(e))

    def draw_render_tab(self) -> None:
        self.ui_current_node_state.video_details = self.draw_video_details(
            self.ui_current_node_state.video_details
        )

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
                    value = image_to_texture(file_path)

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

        # ----------------------------------------------------------------
        # Sticker thumbnails and settings
        available_width, available_height = imgui.get_content_region_available()
        sticker_grid_width = 0.3 * available_width

        imgui.begin_child(
            "sticker_grid",
            width=sticker_grid_width,
            height=available_height,
            border=True,
        )

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

            texture = sticker._texture
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
            imgui.spacing()

        imgui.end_child()

        imgui.same_line()
        imgui.begin_child("selected_sticker_settings", border=True)

        # ----------------------------------------------------------------
        # Selected sticker settings
        if self._tg_selected_sticker_idx < len(self._tg_stickers):
            sticker = self._tg_stickers[self._tg_selected_sticker_idx]
            texture = sticker._texture

            if sticker._video:
                imgui.separator()

                sticker._video.details = self.draw_video_details(
                    details=sticker._video.details,
                    is_changeable=False,
                )
            else:
                imgui.text(f"Size: {sticker._texture.width}x{sticker._texture.height}")

        imgui.end_child()

    def draw_video_details(
        self,
        details: VideoDetails,
        is_changeable: bool = True,
    ) -> VideoDetails:
        details = details.model_copy()
        available_extensions = [".webm", ".mp4"]
        available_qualities = ["low", "medium-low", "medium-high", "high"]

        # ----------------------------------------------------------------
        # File path
        file_path = None
        if is_changeable:
            if imgui.button("File:##video_file_path"):
                file_path = crossfiledialog.save_file(
                    title=f"File path ({available_extensions})",
                )
                if file_path:
                    video_extension = Path(file_path).suffix
                    if video_extension not in available_extensions:
                        details.file_path = ""
                        logger.warning(
                            f"Can't select {video_extension} file, "
                            f"available extensions are: {available_extensions}"
                        )
                    else:
                        details.file_path = file_path
        else:
            imgui.text("File:")

        if details.file_path:
            imgui.same_line()
            imgui.text_colored(str(details.file_path), 0.5, 0.5, 0.5)
            imgui.text(f"File size: {details.file_size} B")

        # ----------------------------------------------------------------
        # Quality
        if is_changeable:
            details.quality = imgui.combo(
                "Quality##video_quality_combo_idx",
                details.quality,
                available_qualities,
            )[1]
            details.quality = min(details.quality, len(available_qualities) - 1)

        # ----------------------------------------------------------------
        # FPS
        if is_changeable:
            details.fps = imgui.drag_int("FPS##video_fps", details.fps, 1)[1]
        else:
            imgui.text(f"FPS: {details.fps}")
        details.fps = max(10, details.fps)

        # ----------------------------------------------------------------
        # Duration
        if is_changeable:
            details.duration = imgui.drag_float(
                "Duration, sec##video_duration",
                details.duration,
                0.1,
            )[1]
        else:
            imgui.text(f"Duration: {details.duration} sec")
        details.duration = max(0.1, details.duration)

        # ----------------------------------------------------------------
        # Render button
        if details.file_path and is_changeable:
            imgui.spacing()
            if imgui.button("Render##video"):
                node = self.nodes[self.current_node_name]  # type: ignore
                details = node.render_to_video(details)

        return details

    def draw_node_settings(self) -> None:
        with imgui.begin_child("node_settings", border=True):
            if imgui.begin_tab_bar("node_settings_tabs").opened:
                if imgui.begin_tab_item("Node").selected:  # type: ignore
                    self.draw_node_tab()
                    imgui.end_tab_item()

                if imgui.begin_tab_item("Render").selected:  # type: ignore
                    self.draw_render_tab()
                    imgui.end_tab_item()

                if imgui.begin_tab_item("Tg stickers").selected:  # type: ignore
                    self.draw_tg_stickers_tab()
                    imgui.end_tab_item()

                imgui.end_tab_bar()

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

    def update_and_draw(self) -> None:
        # ----------------------------------------------------------------
        # Prepare frame
        gl = moderngl.get_context()
        window_width, window_height = glfw.get_window_size(self.window)
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

        # Render nodes
        for node in self.nodes.values():
            node.render()

        # ----------------------------------------------------------------
        # Draw UI
        gl.screen.use()
        gl.clear()

        self.process_hotkeys()
        self.imgui_renderer.process_inputs()
        imgui.new_frame()

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
            image_aspect = np.divide(*node.output_texture.size)
            image_width = min(max_image_width, max_image_height * image_aspect)
            image_height = min(max_image_height, max_image_width / image_aspect)

            has_error = node.shader_error != ""
            imgui.image(
                node.output_texture.glo,
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
