import time
from collections import defaultdict
from importlib.resources import files
from pathlib import Path

import crossfiledialog
import glfw
import imgui
import moderngl
import numpy as np
from imgui.integrations.glfw import GlfwRenderer
from PIL import Image, ImageOps

_RESOURCES_DIR = Path(str(files("shaderbox.resources")))
_DEFAULT_TEXTURE = ImageOps.flip(
    Image.open(_RESOURCES_DIR.resolve() / "textures" / "default.jpeg").convert("RGBA")
)


class Node:
    def __init__(self) -> None:
        self.gl = moderngl.get_context()
        self.name = str(id(self))

        dir = _RESOURCES_DIR / "shaders"
        self.fs_file_path: Path = dir / "default.frag.glsl"
        self.vs_file_path: Path = dir / "default.vert.glsl"

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

        self.texture: moderngl.Texture = self.gl.texture((1280, 960), 4)
        self.fbo: moderngl.Framebuffer = self.gl.framebuffer(
            color_attachments=[self.texture]
        )


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
    # for the left side: window_x = monitor_x
    window_x = monitor_x + screen_width - window_width

    glfw.set_window_pos(window, window_x, monitor_y)

    gl = moderngl.create_context(standalone=False)
    imgui.create_context()
    imgui_renderer = GlfwRenderer(window)

    uniform_values = defaultdict(lambda: {})
    textures = defaultdict(lambda: {})
    nodes = [Node() for _ in range(10)]
    current_node = nodes[0]

    def create_new_node():
        nonlocal current_node

        node = Node()
        nodes.append(node)

        current_node = node

    def delete_current_node():
        nonlocal current_node

        if len(nodes) == 1:
            return

        nodes.remove(current_node)
        current_node = nodes[-1]

    while not glfw.window_should_close(window):
        window_width, window_height = glfw.get_window_size(window)
        start_time = glfw.get_time()
        glfw.poll_events()
        imgui_renderer.process_inputs()
        imgui.new_frame()

        # ----------------------------------------------------------------
        # Render nodes
        for node in nodes:
            node.fbo.use()

            texture_unit = 0
            for uniform_name in node.program:
                uniform = node.program[uniform_name]

                if not isinstance(uniform, moderngl.Uniform):
                    continue

                fmt = uniform.fmt  # type: ignore
                key = f"{uniform_name}_{fmt}"

                if uniform.gl_type == 35678:  # type: ignore
                    texture = textures[node].get(key)
                    if texture is None:
                        texture = gl.texture(
                            _DEFAULT_TEXTURE.size,
                            4,
                            np.array(_DEFAULT_TEXTURE).tobytes(),
                            dtype="f1",
                        )
                        textures[node][key] = texture

                    texture = textures[node][key]
                    texture.use(location=texture_unit)
                    value = texture_unit
                    texture_unit += 1
                else:
                    value = uniform_values[node].get(key)
                    if value is None:
                        value = uniform.value
                        uniform_values[node][key] = value

                node.program[uniform_name] = value

            gl.clear()
            node.vao.render()

        gl.screen.use()
        gl.clear()

        # ----------------------------------------------------------------
        # Handle hotkeys
        io = imgui.get_io()
        if io.key_ctrl and imgui.is_key_pressed(ord("N")):
            create_new_node()

        if io.key_ctrl and imgui.is_key_pressed(ord("D")):
            delete_current_node()

        # ----------------------------------------------------------------
        # Main menu bar
        main_menu_height = 0

        if imgui.begin_main_menu_bar().opened:
            # ------------------------------------------------------------
            # Node menu
            if imgui.begin_menu("Node", True).opened:
                if imgui.menu_item("New", "Ctrl+N", False, True)[1]:
                    create_new_node()

                if imgui.menu_item("Delete Current", "Ctrl+D", False, True)[1]:
                    delete_current_node()

                imgui.end_menu()  # Node

            main_menu_height = imgui.get_item_rect_size()[1]
            imgui.end_main_menu_bar()  # main menu

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
        # current node image
        min_image_height = 100

        max_image_height = max(
            min_image_height,
            imgui.get_content_region_available()[1] - control_panel_min_height,
        )
        max_image_width = imgui.get_content_region_available()[0]
        image_aspect = np.divide(*current_node.texture.size)
        image_width = min(max_image_width, max_image_height * image_aspect)
        image_height = min(max_image_height, max_image_width / image_aspect)

        imgui.image(
            current_node.texture.glo,
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
            # ------------------------------------------------------------
            # Node previews
            node_preview_width, node_preview_height = (
                imgui.get_content_region_available()
            )
            node_preview_width = node_preview_width / 3.0
            with imgui.begin_child(
                "node_previews",
                width=node_preview_width,
                height=node_preview_height,
                border=True,
            ):
                preview_size = 125
                n_cols = int(
                    imgui.get_content_region_available()[0] // (preview_size + 5)
                )
                n_cols = max(1, n_cols)
                for i, node in enumerate(nodes):
                    if node == current_node:
                        imgui.push_style_color(imgui.COLOR_BORDER, 1.0, 1.0, 0.0, 1.0)

                    imgui.begin_child(
                        f"node_preview_{i}",
                        width=preview_size,
                        height=preview_size,
                        border=True,
                        flags=imgui.WINDOW_NO_SCROLLBAR,
                    )

                    if node == current_node:
                        imgui.pop_style_color()

                    if imgui.invisible_button(
                        f"node_preview_button_{i}",
                        width=preview_size,
                        height=preview_size,
                    ):
                        current_node = node

                    s = (preview_size - 10) / max(node.texture.size)
                    image_width = node.texture.size[0] * s
                    image_height = node.texture.size[1] * s

                    imgui.set_cursor_pos_x((preview_size - image_width) / 2 - 1)
                    imgui.set_cursor_pos_y((preview_size - image_height) / 2 - 1)

                    imgui.image(
                        node.texture.glo,
                        width=image_width,
                        height=image_height,
                        uv0=(0, 1),
                        uv1=(1, 0),
                    )

                    imgui.end_child()  # node_preview_i

                    # Handle grid layout
                    if (i + 1) % n_cols != 0:
                        imgui.same_line()
                    else:
                        imgui.spacing()

            # ------------------------------------------------------------
            # Uniforms
            imgui.same_line()
            uniforms_width, uniforms_height = imgui.get_content_region_available()
            with imgui.begin_child(
                "uniforms",
                width=uniforms_width,
                height=uniforms_height,
                border=True,
            ):
                texture_unit = 0
                for uniform_name in current_node.program:
                    uniform = current_node.program[uniform_name]

                    if not isinstance(uniform, moderngl.Uniform):
                        continue

                    fmt: str = uniform.fmt  # type: ignore
                    key = f"{uniform_name}_{fmt}"

                    is_texture = uniform.gl_type == 35678  # type: ignore

                    if is_texture:
                        texture = textures[current_node][key]
                        imgui.text(uniform_name)
                        if imgui.image_button(
                            texture.glo,
                            width=50,
                            height=50,
                            uv0=(0, 1),
                            uv1=(1, 0),
                        ):
                            file_path = crossfiledialog.open_file(
                                title="Select Texture",
                                filter=["*.png", "*.jpg", "*.jpeg", "*.bmp"],
                            )
                            if file_path:
                                image = ImageOps.flip(
                                    Image.open(file_path).convert("RGBA")
                                )
                                textures[current_node][key].release()
                                textures[current_node][key] = gl.texture(
                                    image.size,
                                    4,
                                    np.array(image).tobytes(),
                                    dtype="f1",
                                )
                    else:
                        is_time = uniform_name == "u_time"
                        is_aspect = uniform_name == "u_aspect"
                        is_color = uniform_name.endswith("color")
                        value = uniform_values[current_node][key]
                        if is_time and fmt == "1f":
                            value = glfw.get_time()
                            imgui.text(f"Time: {value:.3f}")
                        elif is_aspect and fmt == "1f":
                            value = np.divide(*current_node.texture.size)
                            imgui.text(f"Aspect: {value:.3f}")
                        elif is_color and fmt == "3f":
                            value = imgui.color_edit3(
                                uniform_name,
                                r=value[0],
                                g=value[1],  # type: ignore
                                b=value[2],  # type: ignore
                            )[1]
                        elif is_color and fmt == "4f":
                            value = imgui.color_edit4(
                                uniform_name,
                                r=value[0],
                                g=value[1],  # type: ignore
                                b=value[2],  # type: ignore
                                a=value[3],  # type: ignore
                            )[1]
                        elif fmt == "1f":
                            value = imgui.drag_float(
                                uniform_name,
                                value=value,
                                change_speed=max(0.01, 0.01 * abs(value)),
                            )[1]
                        elif fmt == "2f":
                            value = imgui.drag_float2(
                                uniform_name,
                                value0=value[0],
                                value1=value[1],  # type: ignore
                                change_speed=max(0.01, 0.01 * np.mean(np.abs(value))),
                            )[1]
                        elif fmt == "3f":
                            value = imgui.drag_float3(
                                uniform_name,
                                value0=value[0],
                                value1=value[1],  # type: ignore
                                value2=value[2],  # type: ignore
                                change_speed=max(0.01, 0.01 * np.mean(np.abs(value))),
                            )[1]

                        uniform_values[current_node][key] = value

        # ----------------------------------------------------------------
        imgui.end()  # ShaderBox

        gl.clear_errors()
        imgui.render()
        imgui_renderer.render(imgui.get_draw_data())

        glfw.swap_buffers(window)

        elapsed_time = glfw.get_time() - start_time
        time.sleep(max(0.0, 1.0 / 60.0 - elapsed_time))

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break


if __name__ == "__main__":
    main()
