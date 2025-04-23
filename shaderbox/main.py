import contextlib
import hashlib
import json
import shutil
import subprocess
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
_DEFAULT_IMAGE = ImageOps.flip(
    Image.open(_RESOURCES_DIR / "textures" / "default.jpeg").convert("RGBA")
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

        self.uniform_values = {}
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

        for data in self.uniform_values.values():
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
                self.uniform_values[uniform.name] = value
            elif uniform.name == "u_aspect":
                value = np.divide(*self.output_texture.size)
                self.uniform_values[uniform.name] = value
            elif uniform.gl_type == 35678:  # type: ignore
                texture = self.uniform_values.get(uniform.name)
                if texture is None or isinstance(texture.mglo, moderngl.InvalidObject):
                    texture = self.gl.texture(
                        _DEFAULT_IMAGE.size,
                        4,
                        np.array(_DEFAULT_IMAGE).tobytes(),
                        dtype="f1",
                    )
                    self.uniform_values[uniform.name] = texture

                texture = self.uniform_values[uniform.name]
                texture.use(location=texture_unit)
                value = texture_unit
                texture_unit += 1
            else:
                value = self.uniform_values.get(uniform.name)
                if value is None:
                    value = uniform.value
                    self.uniform_values[uniform.name] = value

            self.program[uniform.name] = value

        # Render
        self.fbo.use()
        self.gl.clear()
        self.vao.render()

        # ----------------------------------------------------------------
        # Clear stale uniform data
        for uniform_name in self.uniform_values.copy():
            if uniform_name not in seen_uniform_names:
                data = self.uniform_values.pop(uniform_name)
                if isinstance(data, moderngl.Texture):
                    data.release()


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
                file_path = node_dir / value["file_path"]
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

            node.uniform_values[uniform_name] = value

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
            value = node.uniform_values.get(uniform.name)

            if value is None:
                value = uniform.value
                node.uniform_values[uniform.name] = value

            if uniform.name in ["u_time", "u_aspect"]:
                continue

            if uniform.gl_type == 35678:  # type: ignore
                texture_data = value.read()
                image = ImageOps.flip(
                    Image.frombytes("RGBA", (value.width, value.height), texture_data)
                )

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

    def draw_uniforms_tab(self):
        if self.current_node_name:
            node = self.nodes[self.current_node_name]
        else:
            return

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

        for uniform in node.iter_uniforms():
            value = node.uniform_values.get(uniform.name)
            if value is None:
                continue

            if uniform.name in ["u_time", "u_aspect"]:
                uniform_groups["special"].append((uniform, value))
            elif uniform.gl_type == 35678:  # type: ignore
                uniform_groups["textures"].append((uniform, value))
            elif uniform.name.endswith("color"):
                uniform_groups["colors"].append((uniform, value))
            elif uniform.fmt == "1f":  # type: ignore
                uniform_groups["floats"].append((uniform, value))
            elif uniform.fmt == "2f":  # type: ignore
                uniform_groups["vec2"].append((uniform, value))
            elif uniform.fmt == "3f":  # type: ignore
                uniform_groups["vec3"].append((uniform, value))
            elif uniform.fmt == "4f":  # type: ignore
                uniform_groups["vec4"].append((uniform, value))

        # ----------------------------------------------------------------
        # Render each group
        uniform_texture_image_height = 80

        for group_name, uniforms in uniform_groups.items():
            if not uniforms:
                continue

            max_text_width = 0
            item_count = len(uniforms)

            if group_name == "textures":
                for uniform, _ in uniforms:
                    text_width = imgui.calc_text_size(f"{uniform.name}:").x
                    max_text_width = max(max_text_width, text_width)
                height_per_item = uniform_texture_image_height + 10
            elif group_name == "colors":
                for uniform, _ in uniforms:
                    text_width = imgui.calc_text_size(f"{uniform.name}:").x
                    max_text_width = max(max_text_width, text_width)
                height_per_item = imgui.get_text_line_height_with_spacing()
            elif group_name == "special":
                for uniform, value in uniforms:
                    text_width = imgui.calc_text_size(f"{uniform.name}: {value:.3f}").x
                    max_text_width = max(max_text_width, text_width)
                height_per_item = imgui.get_text_line_height_with_spacing()
            else:  # float, vec2, vec3, vec4
                for uniform, _ in uniforms:
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

            for uniform, value in uniforms:
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
                            node.uniform_values[uniform.name].release()
                            node.uniform_values[uniform.name] = node.gl.texture(
                                image.size,
                                4,
                                np.array(image).tobytes(),
                                dtype="f1",
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
                            new_texture = node.gl.texture(
                                depthmap.size,
                                4,
                                np.array(depthmap).tobytes(),
                                dtype="f1",
                            )
                            node.uniform_values[uniform.name].release()
                            node.uniform_values[uniform.name] = new_texture
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

                    node.uniform_values[uniform.name] = value

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
