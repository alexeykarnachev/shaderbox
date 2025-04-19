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
from PIL import Image, ImageOps

_RESOURCES_DIR = Path(str(files("shaderbox.resources")))
_DEFAULT_VS_FILE_PATH = _RESOURCES_DIR / "shaders" / "default.vert.glsl"
_DEFAULT_FS_FILE_PATH = _RESOURCES_DIR / "shaders" / "default.frag.glsl"
_DEFAULT_TEXTURE = ImageOps.flip(
    Image.open(_RESOURCES_DIR.resolve() / "textures" / "default.jpeg").convert("RGBA")
)


class Node:
    def __init__(self, fs_file_path: str | Path | None = None) -> None:
        self.gl = moderngl.get_context()
        self.name = str(id(self))

        self.vs_file_path: Path = _DEFAULT_VS_FILE_PATH
        self.fs_file_path: Path = (
            Path(fs_file_path) if fs_file_path else _DEFAULT_FS_FILE_PATH
        )
        self.fs_file_mtime = self.fs_file_path.lstat().st_mtime

        self.program: moderngl.Program = self.gl.program(
            vertex_shader=self.vs_file_path.read_text(),
            fragment_shader=self.fs_file_path.read_text(),
        )
        self.vbo: moderngl.Buffer = self.gl.buffer(
            np.array(
                [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
                dtype="f4",
            )
        )
        self.vao: moderngl.VertexArray = self.gl.vertex_array(
            self.program, [(self.vbo, "2f", "a_pos")]
        )

        self.output_texture: moderngl.Texture = self.gl.texture((1280, 960), 4)
        self.fbo: moderngl.Framebuffer = self.gl.framebuffer(
            color_attachments=[self.output_texture]
        )

        self.uniform_data = {}

    def _reload_if_needed(self):
        try:
            mtime = self.fs_file_path.lstat().st_mtime
        except FileNotFoundError:
            pass
        else:
            if mtime != self.fs_file_mtime:
                self.fs_file_mtime = mtime

                program = self.gl.program(
                    vertex_shader=self.vs_file_path.read_text(),
                    fragment_shader=self.fs_file_path.read_text(),
                )
                vao = self.gl.vertex_array(program, [(self.vbo, "2f", "a_pos")])

                self.program.release()
                self.vao.release()

                self.program = program
                self.vao = vao

    def release(self):
        self.program.release()
        self.vbo.release()
        self.vao.release()
        self.output_texture.release()
        self.fbo.release()

        for data in self.uniform_data:
            if isinstance(data, moderngl.Texture):
                data.release()

    def iter_uniforms(self):
        for uniform_name in self.program:
            uniform = self.program[uniform_name]

            if isinstance(uniform, moderngl.Uniform):
                yield uniform

    def render(self):
        self.fbo.use()

        # ----------------------------------------------------------------
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
        self.gl.clear()
        self.vao.render()

        # ----------------------------------------------------------------
        # Clear stale uniform data
        for uniform_data_key in self.uniform_data.copy():
            if uniform_data_key not in seen_uniform_data_keys:
                data = self.uniform_data.pop(uniform_data_key)
                if isinstance(data, moderngl.Texture):
                    data.release()

        self._reload_if_needed()


class UI:
    def __init__(self, window, nodes: list[Node]):
        imgui.create_context()

        self.window = window
        self.nodes = nodes
        self.current_node = self.nodes[0]
        self.imgui_renderer = GlfwRenderer(window)

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
            if imgui.begin_menu("Node", True).opened:
                if imgui.menu_item("New", "Ctrl+N", False, True)[1]:
                    self.create_new_current_node()

                if imgui.menu_item("Delete Current", "Ctrl+D", False, True)[1]:
                    self.delete_current_node()

                if imgui.menu_item("Select Next", "->", False, True)[1]:
                    self.select_next_current_node(+1)

                if imgui.menu_item("Select Previous", "<-", False, True)[1]:
                    self.select_next_current_node(-1)

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
                    imgui.push_style_color(imgui.COLOR_BORDER, 1.0, 1.0, 0.0, 1.0)

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

        if imgui.button(str(self.current_node.fs_file_path)):
            file_path = crossfiledialog.open_file(
                title="Select Fragment Shader",
                start_dir=str(_RESOURCES_DIR / "shaders"),
                filter=["*.glsl", "*.frag"],
            )
            if file_path:
                try:
                    new_current_node = Node(file_path)
                except Exception as _:
                    logger.exception("Failed to set fragment shader")

        imgui.same_line()
        imgui.text("File")

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
        for uniform in self.current_node.iter_uniforms():
            uniform_data_key = f"{uniform.name}_{uniform.fmt}"  # type: ignore
            value = self.current_node.uniform_data.get(uniform_data_key)
            if value is None:
                continue

            if uniform.gl_type == 35678:  # type: ignore
                if imgui.image_button(
                    value.glo, width=50, height=50, uv0=(0, 1), uv1=(1, 0)
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
                imgui.text(uniform.name)
            else:
                fmt = uniform.fmt  # type: ignore
                is_color = uniform.name.endswith("color")
                change_speed = max(0.01, 0.01 * np.mean(np.abs(value)))

                if uniform.name in ["u_time", "u_aspect"]:
                    value = self.current_node.uniform_data[uniform_data_key]
                    imgui.text(f"{uniform.name}: {value:.3f}")
                elif is_color and fmt == "3f":
                    value = imgui.color_edit3(uniform.name, *value)[1]
                elif is_color and fmt == "4f":
                    value = imgui.color_edit4(uniform.name, *value)[1]
                elif fmt == "1f":
                    value = imgui.drag_float(
                        uniform.name, value, change_speed=change_speed
                    )[1]
                elif fmt == "2f":
                    value = imgui.drag_float2(
                        uniform.name, *value, change_speed=change_speed
                    )[1]
                elif fmt == "3f":
                    value = imgui.drag_float3(
                        uniform.name, *value, change_speed=change_speed
                    )[1]

                self.current_node.uniform_data[uniform_data_key] = value

    def draw_logs_tab(self):
        imgui.text("Logs will be here soon...")

    def draw_node_settings(self, width, height):
        with imgui.begin_child(
            "node_settings", width=width, height=height, border=True
        ):
            if imgui.begin_tab_bar("node_settings_tabs").opened:
                if imgui.begin_tab_item("Uniforms").selected:  # type: ignore
                    self.draw_uniforms_tab()
                    imgui.end_tab_item()
                if imgui.begin_tab_item("Shader").selected:  # type: ignore
                    self.draw_shader_tab()
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

        if imgui.is_key_pressed(imgui.get_key_index(imgui.KEY_LEFT_ARROW), repeat=True):
            self.select_next_current_node(-1)

        if imgui.is_key_pressed(
            imgui.get_key_index(imgui.KEY_RIGHT_ARROW), repeat=True
        ):
            self.select_next_current_node(+1)

    def render(self, window_width, window_height):
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

        imgui.image(
            self.current_node.output_texture.glo,
            width=image_width,
            height=image_height,
            uv0=(0, 1),
            uv1=(1, 0),
        )

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
            settings_width = control_panel_width - node_preview_width
            self.draw_node_settings(settings_width, control_panel_height)

        # ----------------------------------------------------------------
        imgui.end()

        imgui.render()
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
            try:
                node.render()
            except Exception as e:
                logger.error(f"Failed to render node {node.name}: {e}")

        gl.screen.use()
        gl.clear()

        # ----------------------------------------------------------------
        # Handle hotkeys
        ui.process_hotkeys()

        # ----------------------------------------------------------------
        # Render UI
        gl.clear_errors()
        try:
            ui.render(window_width, window_height)
        except Exception:
            logger.warning("Failed to render imgui draw data")

        glfw.swap_buffers(window)

        elapsed_time = glfw.get_time() - start_time
        time.sleep(max(0.0, 1.0 / 60.0 - elapsed_time))

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break


if __name__ == "__main__":
    main()
