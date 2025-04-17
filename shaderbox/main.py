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

    nodes = [Node()]
    selected_node = nodes[0]

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

        imgui.set_next_window_size(window_width, window_height)
        imgui.set_next_window_position(0, 0)
        imgui.begin(
            "ShaderBox",
            flags=imgui.WINDOW_NO_COLLAPSE
            | imgui.WINDOW_NO_COLLAPSE
            | imgui.WINDOW_NO_DECORATION
            | imgui.WINDOW_ALWAYS_AUTO_RESIZE,
        )

        # ----------------------------------------------------------------
        control_panel_min_height = 300
        min_image_height = 100

        max_image_height = max(
            min_image_height,
            imgui.get_content_region_available()[1] - control_panel_min_height,
        )
        max_image_width = imgui.get_content_region_available_width()
        image_aspect = np.divide(*selected_node.texture.size)
        image_width = min(max_image_width, max_image_height * image_aspect)
        image_height = min(max_image_height, max_image_width / image_aspect)

        imgui.image(
            selected_node.texture.glo,
            width=image_width,
            height=image_height,
            uv0=(0, 1),
            uv1=(1, 0),
        )

        # ----------------------------------------------------------------
        region_width, region_height = imgui.get_content_region_available()
        control_panel_height = max(control_panel_min_height, region_height)
        control_panel_width = region_width
        imgui.begin_child(
            "control_panel",
            width=control_panel_width,
            height=control_panel_height,
            border=False,
        )

        # ----------------------------------------------------------------
        node_previews_width, node_previews_height = imgui.get_content_region_available()
        node_previews_width = int(node_previews_width / 3.0)
        imgui.begin_child(
            "node_previews",
            width=node_previews_width,
            height=node_previews_height,
            border=True,
        )
        imgui.end_child()  # node_previews

        # ----------------------------------------------------------------
        imgui.same_line()
        uniforms_width, uniforms_height = imgui.get_content_region_available()
        imgui.begin_child(
            "uniforms",
            width=uniforms_width,
            height=uniforms_height,
            border=True,
        )
        imgui.end_child()  # uniforms

        # ----------------------------------------------------------------
        imgui.end_child()  # control_panel

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
