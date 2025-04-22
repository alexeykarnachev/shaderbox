import contextlib
import hashlib
import json
import shutil
import time
from importlib.resources import files
from pathlib import Path

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
_DEFAULT_TEXTURE = ImageOps.flip(
    Image.open(_RESOURCES_DIR.resolve() / "textures" / "default.jpeg").convert("RGBA")
)

_APP_DIR = Path(user_data_dir("shaderbox"))
_NODES_DIR = _APP_DIR / "nodes"
_SHADERS_DIR = _APP_DIR / "shaders"
_TEXTURES_DIR = _APP_DIR / "textures"

_NODES_DIR.mkdir(exist_ok=True, parents=True)
_SHADERS_DIR.mkdir(exist_ok=True, parents=True)
_TEXTURES_DIR.mkdir(exist_ok=True, parents=True)


class Node:
    def __init__(
        self,
        fs_file_path: str | Path | None = None,
        output_texture_size: tuple[int, int] | None = None,
    ) -> None:
        self.gl = moderngl.get_context()

        name_hash = f"{id(self)}{time.time()}"
        self.name = hashlib.md5(name_hash.encode()).hexdigest()[:8]

        self.vs_file_path: Path = _DEFAULT_VS_FILE_PATH
        self.fs_file_path: Path = (
            Path(fs_file_path) if fs_file_path else _DEFAULT_FS_FILE_PATH
        )

        self.output_texture_size = output_texture_size or (1280, 960)
        self.output_texture = self.gl.texture(self.output_texture_size, 4)
        self.fbo = self.gl.framebuffer(color_attachments=[self.output_texture])

        self.uniform_data = {}
        self.shader_error: str = ""
        self.fs_file_mtime: float | None = None

        # Will be initialized lazily during the render
        self.program: moderngl.Program | None = None
        self.vbo: moderngl.Buffer | None = None
        self.vao: moderngl.VertexArray | None = None

    def save(self, file_path: str | Path | None = None) -> None:
        file_path = file_path or (_NODES_DIR / f"{self.name}.json")
        file_path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True, parents=True)

        metadata = {
            "output_texture_size": list(self.output_texture_size),
            "uniforms": {},
        }

        try:
            relative_fs_file_path = self.fs_file_path.relative_to(_SHADERS_DIR)
        except ValueError:
            shader_filename = f"{self.name}_frag.glsl"
            target_path = _SHADERS_DIR / shader_filename
            shutil.copy(self.fs_file_path, target_path)
            self.fs_file_path = target_path
            logger.info(f"Copied shader to: {target_path}")

            relative_fs_file_path = target_path.relative_to(_SHADERS_DIR)

        metadata["shader"] = str(relative_fs_file_path)

        # ----------------------------------------------------------------
        # Save uniforms
        for uniform in self.iter_uniforms():
            uniform_data_key = f"{uniform.name}_{uniform.fmt}"  # type: ignore
            value = self.uniform_data.get(uniform_data_key)
            if value is None:
                continue

            if uniform.name in ["u_time", "u_aspect"]:
                continue

            if uniform.gl_type == 35678:  # type: ignore
                texture_data = value.read()
                image = ImageOps.flip(
                    Image.frombytes("RGBA", (value.width, value.height), texture_data)
                )

                texture_filename = f"{self.name}_{uniform_data_key}.png"
                texture_path = _TEXTURES_DIR / texture_filename
                try:
                    image.save(texture_path, format="PNG")
                    metadata["uniforms"][uniform_data_key] = {
                        "type": "texture",
                        "width": value.width,
                        "height": value.height,
                        "path": str(texture_path.relative_to(_APP_DIR)),
                    }
                except Exception as e:
                    logger.error(f"Failed to save texture to {texture_path}: {e}")
                    continue
            else:
                if isinstance(value, int | float):
                    metadata["uniforms"][uniform_data_key] = value
                elif isinstance(value, tuple):
                    metadata["uniforms"][uniform_data_key] = list(value)
                else:
                    logger.warning(
                        f"Skipping unsupported uniform type for {uniform.name}: {type(value)}"
                    )

        with file_path.open("w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved node to: {file_path}")

    @classmethod
    def load(cls, file_path: str | Path) -> "Node":
        file_path = Path(file_path)
        with file_path.open() as f:
            metadata = json.load(f)

        relative_fs_file_path = metadata.get("shader")
        fs_file_path = _SHADERS_DIR / relative_fs_file_path
        node = cls(
            fs_file_path=fs_file_path,
            output_texture_size=tuple(metadata["output_texture_size"]),
        )
        node.name = file_path.stem

        # ----------------------------------------------------------------
        # Load uniforms
        for uniform_data_key, value in metadata["uniforms"].items():
            if isinstance(value, dict) and value.get("type") == "texture":
                file_path = _APP_DIR / value["path"]
                image = ImageOps.flip(Image.open(file_path).convert("RGBA"))
                texture = node.gl.texture(
                    (value["width"], value["height"]),
                    4,
                    np.array(image).tobytes(),
                    dtype="f1",
                )
                value = texture
            elif isinstance(value, list):
                value = tuple(value)

            node.uniform_data[uniform_data_key] = value

        return node

    def reset_output_texture_size(self, output_texture_size: tuple[int, int]):
        self.output_texture.release()
        self.fbo.release()

        self.output_texture_size = output_texture_size
        self.output_texture = self.gl.texture(self.output_texture_size, 4)
        self.fbo = self.gl.framebuffer(color_attachments=[self.output_texture])

    def release(self):
        if self.program:
            self.program.release()
        if self.vbo:
            self.vbo.release()
        if self.vao:
            self.vao.release()

        self.output_texture.release()
        self.fbo.release()

        for data in self.uniform_data.values():
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
        # Reload if file modified
        mtime = self.fs_file_mtime
        if self.fs_file_path.is_file():
            try:
                mtime = self.fs_file_path.lstat().st_mtime
            except Exception as e:
                logger.error(f"Failed to check file modification time: {e}")

        if mtime != self.fs_file_mtime:
            self.fs_file_mtime = mtime
            self.shader_error = ""

            try:
                program = self.gl.program(
                    vertex_shader=self.vs_file_path.read_text(),
                    fragment_shader=self.fs_file_path.read_text(),
                )
            except Exception as e:
                logger.error(f"Failed to compile shader: {e}")
                self.shader_error = str(e)
                return

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

        # Assign program uniforms
        texture_unit = 0
        seen_uniform_data_keys = set()

        for uniform in self.iter_uniforms():
            uniform_data_key = f"{uniform.name}_{uniform.fmt}"  # type: ignore
            seen_uniform_data_keys.add(uniform_data_key)

            if uniform.name == "u_time":
                value = glfw.get_time()
                self.uniform_data[uniform_data_key] = value
            elif uniform.name == "u_aspect":
                value = np.divide(*self.output_texture.size)
                self.uniform_data[uniform_data_key] = value
            elif uniform.gl_type == 35678:  # type: ignore
                texture = self.uniform_data.get(uniform_data_key)
                if texture is None or isinstance(texture.mglo, moderngl.InvalidObject):
                    texture = self.gl.texture(
                        _DEFAULT_TEXTURE.size,
                        4,
                        np.array(_DEFAULT_TEXTURE).tobytes(),
                        dtype="f1",
                    )
                    self.uniform_data[uniform_data_key] = texture

                texture = self.uniform_data[uniform_data_key]
                texture.use(location=texture_unit)
                value = texture_unit
                texture_unit += 1
            else:
                value = self.uniform_data.get(uniform_data_key)
                if value is None:
                    value = uniform.value
                    self.uniform_data[uniform_data_key] = value

            self.program[uniform.name] = value

        # Render
        self.fbo.use()
        self.gl.clear()
        self.vao.render()

        # Clear stale uniform data
        for uniform_data_key in self.uniform_data.copy():
            if uniform_data_key not in seen_uniform_data_keys:
                data = self.uniform_data.pop(uniform_data_key)
                if isinstance(data, moderngl.Texture):
                    data.release()


class UI:
    def __init__(self, window, nodes: list[Node]):
        imgui.create_context()
        self.window = window
        self.nodes = nodes
        self.current_node = self.nodes[0]
        self.imgui_renderer = GlfwRenderer(window)

    def save_current_node(self):
        self.current_node.save()
        logger.info(f"Saved node: {self.current_node.name}")

        node = Node.load(_NODES_DIR / f"{self.current_node.name}.json")

        idx = self.nodes.index(self.current_node)
        self.current_node.release()
        self.nodes[idx] = node
        self.current_node = node
        logger.debug(f"Reopened node: {node.name}")

    def load_node(self):
        file_path = crossfiledialog.open_file(
            title="Load Node",
            start_dir=str(_NODES_DIR),
            filter=["*.json"],
        )
        logger.debug(f"Selected file: {file_path}")
        if file_path:
            node = Node.load(file_path)
            self.nodes.append(node)
            self.current_node = node

    def create_new_current_node(self):
        node = Node()
        self.nodes.append(node)
        self.current_node = node

    def delete_current_node(self):
        if len(self.nodes) == 1:
            return

        self.nodes.remove(self.current_node)
        self.current_node = self.nodes[-1]

    def select_next_current_node(self, step: int = +1):
        idx = self.nodes.index(self.current_node)
        self.current_node = self.nodes[(idx + step) % len(self.nodes)]

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

                if imgui.menu_item("Load", "Ctrl+O", False)[1]:
                    self.load_node()

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
            for i, node in enumerate(self.nodes):
                if node == self.current_node:
                    if node.shader_error:
                        color = (1.0, 0.0, 0.0, 1.0)
                    else:
                        color = (0.0, 1.0, 0.0, 1.0)
                    imgui.push_style_color(imgui.COLOR_BORDER, *color)

                imgui.begin_child(
                    f"preview_{i}",
                    width=preview_size,
                    height=preview_size,
                    border=True,
                    flags=imgui.WINDOW_NO_SCROLLBAR,
                )

                if node == self.current_node:
                    imgui.pop_style_color()

                if imgui.invisible_button(
                    f"preview_button_{i}", width=preview_size, height=preview_size
                ):
                    self.current_node = node

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
        new_current_node: Node | None = None

        imgui.text("Fragment shader file:")
        if imgui.button(str(self.current_node.fs_file_path)):
            file_path = crossfiledialog.open_file(
                title="Select Fragment Shader",
                start_dir=str(_SHADERS_DIR),
                filter=["*.glsl", "*.frag"],
            )
            if file_path:
                try:
                    new_current_node = Node(file_path)
                except Exception as _:
                    logger.exception("Failed to set fragment shader")

        imgui.spacing()

        # ----------------------------------------------------------------
        # Output texture size radio buttons
        imgui.text("Output resolution:")
        resolutions = [(640, 480), (1280, 720), (1280, 960), (1920, 1080), (2560, 1440)]
        current = self.current_node.output_texture_size
        selected = next((i for i, r in enumerate(resolutions) if r == current), 0)

        for i, (w, h) in enumerate(resolutions):
            if imgui.radio_button(f"{w}x{h}", i == selected):
                self.current_node.reset_output_texture_size((w, h))
            imgui.same_line()

        # ----------------------------------------------------------------
        # Replace current node with the new one
        if new_current_node is not None:
            idx = self.nodes.index(self.current_node)
            old_current_node = self.nodes[idx]
            new_current_node.uniform_data = old_current_node.uniform_data.copy()

            old_current_node.release()
            self.nodes[idx] = new_current_node
            self.current_node = new_current_node

    def draw_uniforms_tab(self):
        # ----------------------------------------------------------------
        # Collect and group uniforms
        uniform_groups = {
            "special": [],  # u_time, u_aspect (read-only)
            "textures": [],  # gl_type == 35678
            "floats": [],  # fmt == "1f"
            "vec2": [],  # fmt == "2f"
            "vec3": [],  # fmt == "3f"
            "colors": [],  # name ends with "color"
            "vec4": [],  # fmt == "4f" and not color
        }

        for uniform in self.current_node.iter_uniforms():
            uniform_data_key = f"{uniform.name}_{uniform.fmt}"  # type: ignore
            value = self.current_node.uniform_data.get(uniform_data_key)
            if value is None:
                continue

            if uniform.name in ["u_time", "u_aspect"]:
                uniform_groups["special"].append((uniform, uniform_data_key, value))
            elif uniform.gl_type == 35678:  # type: ignore
                uniform_groups["textures"].append((uniform, uniform_data_key, value))
            elif uniform.name.endswith("color"):
                uniform_groups["colors"].append((uniform, uniform_data_key, value))
            elif uniform.fmt == "1f":  # type: ignore
                uniform_groups["floats"].append((uniform, uniform_data_key, value))
            elif uniform.fmt == "2f":  # type: ignore
                uniform_groups["vec2"].append((uniform, uniform_data_key, value))
            elif uniform.fmt == "3f":  # type: ignore
                uniform_groups["vec3"].append((uniform, uniform_data_key, value))
            elif uniform.fmt == "4f":  # type: ignore
                uniform_groups["vec4"].append((uniform, uniform_data_key, value))

        # ----------------------------------------------------------------
        # Render each group
        uniform_texture_image_height = 80

        for group_name, uniforms in uniform_groups.items():
            if not uniforms:
                continue

            max_text_width = 0
            item_count = len(uniforms)

            if group_name == "textures":
                for uniform, _, _ in uniforms:
                    text_width = imgui.calc_text_size(f"{uniform.name}:").x
                    max_text_width = max(max_text_width, text_width)
                height_per_item = uniform_texture_image_height + 10
            elif group_name == "colors":
                for uniform, _, _ in uniforms:
                    text_width = imgui.calc_text_size(f"{uniform.name}:").x
                    max_text_width = max(max_text_width, text_width)
                height_per_item = imgui.get_text_line_height_with_spacing()
            elif group_name == "special":
                for uniform, _, value in uniforms:
                    text_width = imgui.calc_text_size(f"{uniform.name}: {value:.3f}").x
                    max_text_width = max(max_text_width, text_width)
                height_per_item = imgui.get_text_line_height_with_spacing()
            else:  # float, vec2, vec3, vec4
                for uniform, _, _ in uniforms:
                    text_width = imgui.calc_text_size(f"{uniform.name}:").x
                    max_text_width = max(max_text_width, text_width)
                height_per_item = 1.5 * imgui.get_text_line_height_with_spacing()

            total_height = height_per_item * item_count + 20

            imgui.push_style_color(imgui.COLOR_BORDER, 0.15, 0.15, 0.15)
            imgui.begin_child(
                f"{group_name}_group",
                width=0,
                height=total_height,
                border=True,
                flags=imgui.WINDOW_NO_SCROLLBAR,
            )

            for uniform, uniform_data_key, value in uniforms:
                if group_name == "textures":
                    image_height = uniform_texture_image_height
                    image_width = image_height * value.width / max(value.height, 1)
                    if imgui.image_button(
                        value.glo,
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
                            image = ImageOps.flip(Image.open(file_path).convert("RGBA"))
                            self.current_node.uniform_data[uniform_data_key].release()
                            self.current_node.uniform_data[uniform_data_key] = (
                                self.current_node.gl.texture(
                                    image.size,
                                    4,
                                    np.array(image).tobytes(),
                                    dtype="f1",
                                )
                            )
                    imgui.same_line()

                    imgui.begin_group()
                    imgui.text(uniform.name)
                    if imgui.button("To depthmap"):
                        texture_data = value.read()
                        image = Image.frombytes(
                            "RGBA", (value.width, value.height), texture_data
                        )
                        try:
                            depthmap = get_modelbox_depthmap(image).convert("RGBA")
                            new_texture = self.current_node.gl.texture(
                                depthmap.size,
                                4,
                                np.array(depthmap).tobytes(),
                                dtype="f1",
                            )
                            self.current_node.uniform_data[uniform_data_key].release()
                            self.current_node.uniform_data[uniform_data_key] = (
                                new_texture
                            )
                        except Exception as e:
                            logger.error(str(e))
                    imgui.end_group()

                elif group_name == "special":
                    imgui.text(f"{uniform.name}: {value:.3f}")
                else:
                    is_color = uniform.name.endswith("color")
                    change_speed = (
                        max(0.01, 0.01 * np.mean(np.abs(value)))  # type: ignore
                        if isinstance(value, int | float | tuple)
                        else 0.01
                    )

                    if is_color and uniform.fmt == "3f":  # type: ignore
                        value = imgui.color_edit3(uniform.name, *value)[1]  # type: ignore
                    elif is_color and uniform.fmt == "4f":  # type: ignore
                        value = imgui.color_edit4(uniform.name, *value)[1]  # type: ignore
                    elif uniform.fmt == "1f":  # type: ignore
                        value = imgui.drag_float(
                            uniform.name,
                            value,  # type: ignore
                            change_speed=change_speed,
                        )[1]
                    elif uniform.fmt == "2f":  # type: ignore
                        value = imgui.drag_float2(
                            uniform.name,
                            *value,  # type: ignore
                            change_speed=change_speed,
                        )[1]
                    elif uniform.fmt == "3f":  # type: ignore
                        value = imgui.drag_float3(
                            uniform.name,
                            *value,  # type: ignore
                            change_speed=change_speed,
                        )[1]
                    elif uniform.fmt == "4f":  # type: ignore
                        value = imgui.drag_float4(
                            uniform.name,
                            *value,  # type: ignore
                            change_speed=change_speed,
                        )[1]

                    self.current_node.uniform_data[uniform_data_key] = value

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
        if io.key_ctrl and imgui.is_key_pressed(ord("O")):
            self.load_node()
        if imgui.is_key_pressed(imgui.get_key_index(imgui.KEY_LEFT_ARROW), repeat=True):
            self.select_next_current_node(-1)
        if imgui.is_key_pressed(
            imgui.get_key_index(imgui.KEY_RIGHT_ARROW), repeat=True
        ):
            self.select_next_current_node(+1)

    def render(self, window_width, window_height):
        self.process_hotkeys()
        self.imgui_renderer.process_inputs()
        imgui.new_frame()

        # ----------------------------------------------------------------
        # Main menu bar
        main_menu_height = self.draw_main_menu_bar()

        # ----------------------------------------------------------------
        # Main window
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
        min_image_height = 100

        max_image_height = max(
            min_image_height,
            imgui.get_content_region_available()[1] - control_panel_min_height,
        )
        max_image_width = imgui.get_content_region_available()[0]
        image_aspect = np.divide(*self.current_node.output_texture.size)
        image_width = min(max_image_width, max_image_height * image_aspect)
        image_height = min(max_image_height, max_image_width / image_aspect)

        has_error = self.current_node.shader_error != ""
        cursor_pos = imgui.get_cursor_screen_pos()
        imgui.image(
            self.current_node.output_texture.glo,
            width=image_width,
            height=image_height,
            uv0=(0, 1),
            uv1=(1, 0),
            tint_color=(0.2, 0.2, 0.2, 1.0) if has_error else (1.0, 1.0, 1.0, 1.0),
        )

        if has_error:
            draw_list = imgui.get_window_draw_list()
            text_size = imgui.calc_text_size(self.current_node.shader_error)
            text_x = cursor_pos[0] + (image_width - text_size[0]) / 2
            text_y = cursor_pos[1] + (image_height - text_size[1]) / 2
            draw_list.add_text(
                text_x,
                text_y,
                imgui.color_convert_float4_to_u32(1.0, 0.0, 0.0, 1.0),
                self.current_node.shader_error,
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

        # ----------------------------------------------------------------
        imgui.end()
        imgui.render()

        glfw.make_context_current(self.window)
        moderngl.get_context().clear_errors()

        with contextlib.suppress(GLError):
            self.imgui_renderer.render(imgui.get_draw_data())


def main():
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

    gl = moderngl.create_context(standalone=False)
    nodes = [Node()]
    ui = UI(window=window, nodes=nodes)

    while not glfw.window_should_close(window):
        window_width, window_height = glfw.get_window_size(window)
        start_time = glfw.get_time()
        glfw.poll_events()

        # ----------------------------------------------------------------
        # Render nodes
        for node in nodes:
            node.render()

        gl.screen.use()
        gl.clear()

        ui.render(window_width, window_height)

        glfw.swap_buffers(window)

        elapsed_time = glfw.get_time() - start_time
        time.sleep(max(0.0, 1.0 / 60.0 - elapsed_time))

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break


if __name__ == "__main__":
    main()
