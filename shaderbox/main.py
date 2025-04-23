import contextlib
import hashlib
import json
import shutil
import subprocess
import time
from collections import defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any

import crossfiledialog
import glfw
import imgui
import moderngl
import numpy as np
from imgui.integrations.glfw import GlfwRenderer
from loguru import logger
from OpenGL.GL import GLError
from PIL import Image, ImageOps
from platformdirs import user_data_dir

from shaderbox.vendors import get_modelbox_depthmap

_RESOURCES_DIR = Path(str(files("shaderbox.resources")))
_DEFAULT_VS_FILE_PATH = _RESOURCES_DIR / "shaders" / "default.vert.glsl"
_DEFAULT_FS_FILE_PATH = _RESOURCES_DIR / "shaders" / "default.frag.glsl"
_DEFAULT_IMAGE = Image.open(_RESOURCES_DIR / "textures" / "default.jpeg").convert(
    "RGBA"
)

_APP_DIR = Path(user_data_dir("shaderbox"))
_NODES_DIR = _APP_DIR / "nodes"
_TRASH_DIR = _APP_DIR / "trash"

_NODES_DIR.mkdir(exist_ok=True, parents=True)
_TRASH_DIR.mkdir(exist_ok=True, parents=True)


class Node:
    def __init__(
        self,
        fs_source: str | None = None,
        output_texture_size: tuple[int, int] | None = None,
    ) -> None:
        self.gl = moderngl.get_context()

        self.vs_source: str = _DEFAULT_VS_FILE_PATH.read_text()
        self.fs_source: str = (
            fs_source if fs_source else _DEFAULT_FS_FILE_PATH.read_text()
        )

        self.output_texture_size = output_texture_size or (1280, 960)
        self.output_texture = self.gl.texture(self.output_texture_size, 4)
        self.fbo = self.gl.framebuffer(color_attachments=[self.output_texture])

        self._uniform_values = {}
        self.shader_error: str = ""

        # Will be initialized lazily during the render
        self.program: moderngl.Program | None = None
        self.vbo: moderngl.Buffer | None = None
        self.vao: moderngl.VertexArray | None = None

    def release_program(self, new_fs_source: str = ""):
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

    def reset_output_texture_size(self, output_texture_size: tuple[int, int]):
        self.output_texture.release()
        self.fbo.release()

        self.output_texture_size = output_texture_size
        self.output_texture = self.gl.texture(self.output_texture_size, 4)
        self.fbo = self.gl.framebuffer(color_attachments=[self.output_texture])

    def release(self):
        self.release_program()

        self.output_texture.release()
        self.fbo.release()

        for data in self._uniform_values.values():
            if isinstance(data, moderngl.Texture):
                data.release()

    def iter_uniforms(self):
        if not self.program:
            return

        for uniform_name in self.program:
            uniform = self.program[uniform_name]
            if isinstance(uniform, moderngl.Uniform):
                yield uniform

    def render(self):
        # ----------------------------------------------------------------
        # Lazily initialize program, vbo, and vao
        if not self.program or not self.vbo or not self.vao:
            try:
                program = self.gl.program(
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
            self.vbo = self.gl.buffer(
                np.array(
                    [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
                    dtype="f4",
                )
            )
            self.vao = self.gl.vertex_array(program, [(self.vbo, "2f", "a_pos")])

        if not self.program or not self.vao:
            return

        # ----------------------------------------------------------------
        # Assign program uniforms
        texture_unit = 0
        seen_uniform_names = set()

        for uniform in self.iter_uniforms():
            seen_uniform_names.add(uniform.name)

            if uniform.name == "u_time":
                value = glfw.get_time()
                self._uniform_values[uniform.name] = value
            elif uniform.name == "u_aspect":
                value = np.divide(*self.output_texture.size)
                self._uniform_values[uniform.name] = value
            elif uniform.gl_type == 35678:  # type: ignore
                texture = self._uniform_values.get(uniform.name)
                if texture is None or isinstance(texture.mglo, moderngl.InvalidObject):
                    self._uniform_values[uniform.name] = load_texture_from_image(
                        _DEFAULT_IMAGE
                    )

                texture = self._uniform_values[uniform.name]
                texture.use(location=texture_unit)
                value = texture_unit
                texture_unit += 1
            else:
                value = self._uniform_values.get(uniform.name)
                if value is None:
                    value = uniform.value
                    self._uniform_values[uniform.name] = value

            self.program[uniform.name] = value

        # Render
        self.fbo.use()
        self.gl.clear()
        self.vao.render()

        # ----------------------------------------------------------------
        # Clear stale uniform data
        for uniform_name in self._uniform_values.copy():
            if uniform_name not in seen_uniform_names:
                data = self._uniform_values.pop(uniform_name)
                if isinstance(data, moderngl.Texture):
                    data.release()

    def set_uniform_value(self, name: str, value: Any):
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


class UIUniform:
    def __init__(self, uniform: moderngl.Uniform) -> None:
        self.uniform = uniform
        self.name = uniform.name
        self.array_length = uniform.array_length
        self.dimension = uniform.dimension
        self.gl_type = uniform.gl_type  # type: ignore
        self.is_special = self.name in ("u_time", "u_aspect")
        self.is_image = self.gl_type == 35678
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
    def height(self) -> float:
        if self.group_name == "image":
            return 95
        elif self.group_name == "drag":
            return 5 + imgui.get_text_line_height_with_spacing()
        else:
            return imgui.get_text_line_height_with_spacing()


def load_texture_from_image(
    image_or_file_path: Image.Image | Path,
    extra: Any | None = None,
) -> moderngl.Texture:
    if isinstance(image_or_file_path, Path):
        image = Image.open(image_or_file_path)
    else:
        image = image_or_file_path

    gl = moderngl.get_context()
    prepared_image = ImageOps.flip(image.convert("RGBA"))
    texture = gl.texture(image.size, 4, np.array(prepared_image).tobytes(), dtype="f1")
    texture.extra = {"image": image} if extra is None else extra
    return texture


def load_image_from_texture(texture: moderngl.Texture) -> Image.Image:
    texture_data = texture.read()
    image = ImageOps.flip(
        Image.frombytes("RGBA", (texture.width, texture.height), texture_data)
    )
    return image


class App:
    def __init__(self):
        glfw.init()

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

        node_dirs = sorted(_NODES_DIR.iterdir(), key=lambda x: x.stat().st_ctime)
        for node_dir in node_dirs:
            if node_dir.is_dir():
                node, mtime = self.load_node(node_dir)
                name = node_dir.name

                self.nodes[name] = node
                self.node_mtimes[name] = mtime
                self.current_node_name = name

        imgui.create_context()
        self.window = window
        self.imgui_renderer = GlfwRenderer(window)

    @staticmethod
    def load_node(node_dir: Path | str) -> tuple[Node, float]:
        node_dir = Path(node_dir)
        with (node_dir / "node.json").open() as f:
            metadata = json.load(f)

        fs_file_path = node_dir / "shader.frag.glsl"
        mtime = fs_file_path.lstat().st_mtime if fs_file_path.exists() else 0.0

        node = Node(
            fs_source=fs_file_path.read_text(),
            output_texture_size=tuple(metadata["output_texture_size"]),
        )

        # ----------------------------------------------------------------
        # Load uniforms
        for uniform_name, value in metadata["uniforms"].items():
            if isinstance(value, dict) and value.get("type") == "texture":
                value = load_texture_from_image(node_dir / value["file_path"])
            elif isinstance(value, list):
                value = tuple(value)

            node.set_uniform_value(uniform_name, value)

        return node, mtime

    def edit_node_fs_file(self, node_name: str):
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

    def edit_current_node_fs_file(self):
        if self.current_node_name:
            self.edit_node_fs_file(self.current_node_name)
        else:
            logger.warning("Nothing to edit")

    def save_node(self, node_name: str):
        node = self.nodes[node_name]
        node_dir = _NODES_DIR / node_name
        node_dir.mkdir(exist_ok=True, parents=True)

        metadata = {
            "output_texture_size": list(node.output_texture_size),
            "uniforms": {},
        }

        fs_file_path = node_dir / "shader.frag.glsl"
        if not fs_file_path.exists():
            with fs_file_path.open("w") as f:
                f.write(node.fs_source)
                self.node_mtimes[node_name] = fs_file_path.lstat().st_mtime

        # ----------------------------------------------------------------
        # Save uniforms
        for uniform in node.iter_uniforms():
            if uniform.name in ["u_time", "u_aspect"]:
                continue

            value = node.get_uniform_value(uniform.name)

            if uniform.gl_type == 35678:  # type: ignore
                image = load_image_from_texture(value)
                textures_dir = node_dir / "textures"
                textures_dir.mkdir(exist_ok=True)
                texture_filename = f"{uniform.name}.png"
                image.save(textures_dir / texture_filename, format="PNG")
                metadata["uniforms"][uniform.name] = {
                    "type": "texture",
                    "width": value.width,
                    "height": value.height,
                    "file_path": f"textures/{texture_filename}",
                }
            else:
                if isinstance(value, int | float):
                    metadata["uniforms"][uniform.name] = value
                elif isinstance(value, tuple):
                    metadata["uniforms"][uniform.name] = list(value)
                else:
                    logger.warning(
                        f"Skipping unsupported uniform type for {uniform.name}: {type(value)}"
                    )

        with (node_dir / "node.json").open("w") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Node {node_name} saved: {node_dir}")

    def save_current_node(self):
        if self.current_node_name:
            self.save_node(self.current_node_name)
        else:
            logger.warning("Nothing to save")

    def create_new_current_node(self):
        node = Node()
        name = hashlib.md5(f"{id(node)}{time.time()}".encode()).hexdigest()[:8]

        self.nodes[name] = node
        self.current_node_name = name

        self.save_node(name)
        logger.info(f"Node created: {name}")

    def delete_current_node(self):
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

    def select_next_current_node(self, step: int = +1):
        if self.current_node_name is None:
            return

        keys = list(self.nodes.keys())
        idx = keys.index(self.current_node_name)
        self.current_node_name = keys[(idx + step) % len(keys)]

    def draw_main_menu_bar(self):
        if imgui.begin_main_menu_bar().opened:
            if imgui.begin_menu("Node").opened:
                if imgui.menu_item("New", "Ctrl+N", False)[1]:
                    self.create_new_current_node()

                if imgui.menu_item("Delete Current", "Ctrl+D", False)[1]:
                    self.delete_current_node()

                if imgui.menu_item("Select Next", "->", False)[1]:
                    self.select_next_current_node(+1)

                if imgui.menu_item("Select Previous", "<-", False)[1]:
                    self.select_next_current_node(-1)

                imgui.separator()

                if imgui.menu_item("Save Current", "Ctrl+S", False)[1]:
                    self.save_current_node()

                imgui.end_menu()

            if imgui.begin_menu("Shader").opened:
                if imgui.menu_item("Edit", "Ctrl+E", False)[1]:
                    self.edit_current_node_fs_file()

                imgui.end_menu()

            main_menu_height = imgui.get_item_rect_size()[1]
            imgui.end_main_menu_bar()
            return main_menu_height
        return 0

    def draw_node_preview_grid(self, width, height):
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

    def draw_shader_tab(self):
        if self.current_node_name:
            node = self.nodes[self.current_node_name]
        else:
            return

        node_name = self.current_node_name
        fs_file_path = _NODES_DIR / node_name / "shader.frag.glsl"

        imgui.text_colored(str(fs_file_path), 0.5, 0.5, 0.5)

        if imgui.button("Edit", width=80):
            self.edit_node_fs_file(node_name)

        imgui.spacing()

        imgui.text("Output resolution:")
        resolutions = [(640, 480), (1280, 720), (1280, 960), (1920, 1080), (2560, 1440)]
        current = node.output_texture_size
        selected = next((i for i, r in enumerate(resolutions) if r == current), 0)

        for i, (w, h) in enumerate(resolutions):
            if imgui.radio_button(f"{w}x{h}", i == selected):
                node.reset_output_texture_size((w, h))
            imgui.same_line()

    def draw_ui_uniform(self, ui_uniform: UIUniform, node: Node):
        uniform_name = ui_uniform.name
        value = node.get_uniform_value(uniform_name)

        if ui_uniform.group_name == "special":
            if ui_uniform.array_length > 1:
                value_str = ", ".join(f"{v:.3f}" for v in value)
                imgui.text(f"{uniform_name}[{ui_uniform.array_length}]: [{value_str}]")
            else:
                imgui.text(f"{uniform_name}: {value:.3f}")

        elif ui_uniform.group_name == "image":
            texture = value
            image_height = 90
            image_width = image_height * texture.width / max(texture.height, 1)
            if imgui.image_button(
                texture.glo,
                width=image_width,
                height=image_height,
                uv0=(0, 1),
                uv1=(1, 0),
            ):
                file_path = crossfiledialog.open_file(
                    title="Select Texture",
                    filter=["*.png", "*.jpg", "*.jpeg", "*.bmp"],
                )
                if file_path:
                    value = load_texture_from_image(file_path)

            imgui.same_line()
            imgui.begin_group()
            imgui.text(uniform_name)

            imgui.same_line()
            if imgui.button(f"Reset##{ui_uniform.name}"):
                value = load_texture_from_image(texture.extra["image"])

            if imgui.button(f"To depthmap##{ui_uniform.name}"):
                image = load_image_from_texture(texture)
                try:
                    depthmap_image = get_modelbox_depthmap(image)
                    value = load_texture_from_image(depthmap_image, {"image": image})
                except Exception as e:
                    logger.error(str(e))
            imgui.end_group()

        elif ui_uniform.group_name == "color":
            fn = getattr(imgui, f"color_edit{ui_uniform.dimension}")
            value = fn(uniform_name, *value)[1]

        elif ui_uniform.group_name == "drag":
            change_speed = 0.01
            if ui_uniform.dimension == 1:
                value = imgui.drag_float(uniform_name, value, change_speed)[1]
            else:
                fn = getattr(imgui, f"drag_float{ui_uniform.dimension}")
                value = fn(uniform_name, *value, change_speed)[1]

        node.set_uniform_value(uniform_name, value)

    def draw_uniforms_tab(self):
        if self.current_node_name:
            node = self.nodes[self.current_node_name]
        else:
            return

        # Collect UIUniform instances
        ui_uniforms = [UIUniform(uniform) for uniform in node.iter_uniforms()]

        # Group them by group_name
        uniform_groups = defaultdict(list)
        for ui_uniform in ui_uniforms:
            uniform_groups[ui_uniform.group_name].append(ui_uniform)

        # Draw each group
        for group_name, ui_uniforms_in_group in uniform_groups.items():
            # Calculate total height for the group
            total_height = (
                sum(ui_uniform.height for ui_uniform in ui_uniforms_in_group) + 20
            )

            imgui.push_style_color(imgui.COLOR_BORDER, 0.15, 0.15, 0.15)
            imgui.begin_child(
                f"{group_name}_group",
                width=0,
                height=total_height,
                border=True,
                flags=imgui.WINDOW_NO_SCROLLBAR,
            )

            for ui_uniform in ui_uniforms_in_group:
                self.draw_ui_uniform(ui_uniform, node)

            imgui.end_child()
            imgui.pop_style_color()

    def draw_logs_tab(self):
        imgui.text("Logs will be here soon...")

    def draw_node_settings(self):
        with imgui.begin_child("node_settings", border=True):
            if imgui.begin_tab_bar("node_settings_tabs").opened:
                if imgui.begin_tab_item("Shader").selected:  # type: ignore
                    self.draw_shader_tab()
                    imgui.end_tab_item()
                if imgui.begin_tab_item("Uniforms").selected:  # type: ignore
                    self.draw_uniforms_tab()
                    imgui.end_tab_item()

                if imgui.begin_tab_item("Logs").selected:  # type: ignore
                    self.draw_logs_tab()
                    imgui.end_tab_item()

                imgui.end_tab_bar()

    def process_hotkeys(self):
        io = imgui.get_io()
        if io.key_ctrl and imgui.is_key_pressed(ord("N")):
            self.create_new_current_node()
        if io.key_ctrl and imgui.is_key_pressed(ord("D")):
            self.delete_current_node()
        if io.key_ctrl and imgui.is_key_pressed(ord("S")):
            self.save_current_node()
        if io.key_ctrl and imgui.is_key_pressed(ord("E")):
            self.edit_current_node_fs_file()
        if imgui.is_key_pressed(imgui.get_key_index(imgui.KEY_LEFT_ARROW), repeat=True):
            self.select_next_current_node(-1)
        if imgui.is_key_pressed(
            imgui.get_key_index(imgui.KEY_RIGHT_ARROW), repeat=True
        ):
            self.select_next_current_node(+1)

    def run(self):
        while not glfw.window_should_close(self.window):
            start_time = glfw.get_time()

            self.update_and_draw()

            elapsed_time = glfw.get_time() - start_time
            time.sleep(max(0.0, 1.0 / 60.0 - elapsed_time))

            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break

    def update_and_draw(self):
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
            | imgui.WINDOW_NO_TITLE_BAR,
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
                imgui.get_content_region_available()[1] - control_panel_min_height,
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

        imgui.set_cursor_screen_pos((cursor_pos[0], cursor_pos[1] + image_height))  # type: ignore

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


def main():
    ui = App()
    ui.run()


if __name__ == "__main__":
    main()
