import time
from importlib.resources import files
from pathlib import Path

import glfw
import imgui
import moderngl
import numpy as np
from imgui.integrations.glfw import GlfwRenderer


class Node:
    def __init__(self) -> None:
        self.gl = moderngl.get_context()
        self.name = str(id(self))

        dir = Path(str(files("shaderbox.resources") / "shaders"))
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

        self.texture: moderngl.Texture = self.gl.texture((2560, 1440), 4)
        self.fbo: moderngl.Framebuffer = self.gl.framebuffer(
            color_attachments=[self.texture]
        )


def main():
    glfw.init()

    monitor = glfw.get_primary_monitor()
    video_mode = glfw.get_video_mode(monitor)
    window = glfw.create_window(
        width=video_mode.size[0] // 2,
        height=video_mode.size[1],
        title="ShaderBox",
        monitor=None,
        share=None,
    )

    glfw.make_context_current(window)
    monitor_pos = glfw.get_monitor_pos(monitor)
    glfw.set_window_pos(window, *monitor_pos)

    gl = moderngl.create_context(standalone=False)
    imgui.create_context()
    imgui_renderer = GlfwRenderer(window)

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

        for node in nodes:
            node.fbo.use()
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
        max_image_width = imgui.get_content_region_available_width()
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
                preview_size = 130
                n_cols = imgui.get_content_region_available_width() // (
                    preview_size + 5
                )
                n_cols = max(1, n_cols)
                for i, node in enumerate(nodes):
                    with imgui.begin_child(
                        f"node_preview_{i}",
                        width=preview_size,
                        height=preview_size,
                    ):
                        s = (preview_size - 10) / max(node.texture.size)
                        image_width = node.texture.size[0] * s
                        image_height = node.texture.size[1] * s

                        imgui.image_button(
                            node.texture.glo,
                            width=image_width,
                            height=image_height,
                            uv0=(0, 1),
                            uv1=(1, 0),
                        )

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
                pass

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
