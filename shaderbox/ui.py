import subprocess
from importlib.resources import as_file, files

import crossfiledialog
import glfw
import imgui
import moderngl
from imgui.integrations.glfw import GlfwRenderer
from loguru import logger

from shaderbox.gl import GLContext
from shaderbox.graph import Node, RenderGraph
from shaderbox.utils import scale_size


class UI:
    def __init__(self, gl: GLContext) -> None:
        self._gl = gl
        self._output_node = None

        imgui.create_context()
        self._imgui_renderer = GlfwRenderer(gl.window)

        io = imgui.get_io()
        io.fonts.clear()
        font_resource = (
            files("shaderbox.resources")
            / "ubuntu-font-family-0.80"
            / "UbuntuMono-R.ttf"
        )
        with as_file(font_resource) as font_path:
            self._font_large = io.fonts.add_font_from_file_ttf(str(font_path), 20.0)
            self._font_small = io.fonts.add_font_from_file_ttf(str(font_path), 15.0)
        self._imgui_renderer.refresh_font_texture()

    def update_and_render(self, graph: RenderGraph):
        if self._output_node is None:
            self._output_node = list(graph.iter_nodes())[-1]

        new_output_node = self._output_node

        self._imgui_renderer.process_inputs()
        imgui.new_frame()

        imgui.begin("Output Viewer")
        imgui.set_window_position(0, 0)

        # ----------------------------------------------------------------
        # Main Image
        main_image_max_size = 800
        main_output_size = self._output_node.get_output_size()
        main_image_width, main_image_height = scale_size(
            main_output_size, main_image_max_size
        )

        imgui.begin_child(
            "main_image",
            width=main_image_width,
            height=main_image_height,
            border=True,
            flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_INPUTS,
        )
        imgui.text(
            f"Node - {self._output_node.name} | fps: {imgui.get_io().framerate:.2f}"
        )
        imgui.separator()
        _texture_image(
            self._output_node._texture,
            max(main_image_width, main_image_height),  # type: ignore
        )
        imgui.end_child()

        # ----------------------------------------------------------------
        # Button to Open Neovim
        if imgui.button("Neovim (CTRL+N)") or (
            imgui.is_key_down(glfw.KEY_LEFT_CONTROL)
            and imgui.is_key_pressed(glfw.KEY_N)
        ):
            cmd = [
                "gnome-terminal",
                "--",
                "nvim",
                self._output_node._fs_file_path,
            ]
            subprocess.Popen(cmd)

        # ----------------------------------------------------------------
        # Node Selector and Preview
        preview_img_width = 150
        preview_pane_width = preview_img_width + 40

        imgui.begin_child("nodes", width=preview_pane_width, border=True)

        for node in graph.iter_nodes():
            imgui.text(node.name)

            is_output = self._output_node is node
            border_size = 2.0 if is_output else 0.0

            if _texture_image(
                node._texture,
                preview_img_width,
                is_button=True,
                border_size=border_size,
            ):
                new_output_node = node

        imgui.end_child()

        # ----------------------------------------------------------------
        # Uniforms and Textures
        imgui.same_line()
        imgui.begin_child("uniforms", border=True)

        ui_nodes = []
        ui_textures = []
        ui_static_values = []
        ui_changeable_values = []

        for u_name, u_value in self._output_node.uniforms.items():
            if not callable(u_value):
                is_changeable = not isinstance(u_value, Node)
            else:
                u_value = u_value()
                is_changeable = False

            ui_uniform = (u_name, u_value)
            if isinstance(u_value, Node):
                ui_nodes.append(ui_uniform)
            elif isinstance(u_value, moderngl.Texture):
                ui_textures.append(ui_uniform)
            elif is_changeable:
                ui_changeable_values.append(ui_uniform)
            else:
                ui_static_values.append(ui_uniform)

        if ui_nodes:
            with imgui.font(self._font_large):  # type: ignore
                imgui.text("Input Nodes")
                imgui.separator()

            for name, node in ui_nodes:
                with imgui.font(self._font_small):  # type: ignore
                    imgui.text(name)

                if _texture_image(node._texture, 50, is_button=True):
                    new_output_node = node

                imgui.spacing()

        if ui_textures:
            with imgui.font(self._font_large):  # type: ignore
                imgui.text("Input Textures")
                imgui.separator()

            for name, texture in ui_textures:
                with imgui.font(self._font_small):  # type: ignore
                    imgui.text(name)

                if _texture_image(texture, 50, is_button=True):
                    new_file = crossfiledialog.open_file(
                        title=f"Select texture for {name}"
                    )

                    if new_file:
                        new_texture = self._gl.load_texture(new_file)
                        self._output_node.uniforms[name] = new_texture

                imgui.spacing()

        if ui_changeable_values:
            with imgui.font(self._font_large):  # type: ignore
                imgui.text("Changeable Uniforms")
                imgui.separator()

            for name, value in ui_changeable_values:
                with imgui.font(self._font_small):  # type: ignore
                    self._output_node.uniforms[name] = _drag_value(name, value)

        if ui_static_values:
            with imgui.font(self._font_large):  # type: ignore
                imgui.text("Automatic Uniforms")
                imgui.separator()

            for name, value in ui_static_values:
                with imgui.font(self._font_small):  # type: ignore
                    _drag_value(name, value)

        imgui.spacing()
        imgui.end_child()

        imgui.end()
        imgui.render()
        self._imgui_renderer.render(imgui.get_draw_data())

        self._output_node = new_output_node

    # def _open_neovim(self, shader_file: str):


def _texture_image(
    texture: moderngl.Texture,
    max_size: int,
    is_button: bool = False,
    border_size: float = 0.0,
):
    width, height = scale_size(
        texture.size,
        max_size=max_size,
    )
    fn = imgui.image_button if is_button else imgui.image

    imgui.push_style_color(imgui.COLOR_BORDER, 1.0, 1.0, 0.0, 1.0)
    if is_button:
        imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, border_size)
    else:
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (border_size, border_size))

    result = fn(
        texture.glo,
        width=width,
        height=height,
        uv0=(0, 1),
        uv1=(1, 0),
    )

    imgui.pop_style_color()
    imgui.pop_style_var()
    return result


def _drag_value(name, value):
    format_str = (
        "%.4f"
        if isinstance(value, float)
        or (hasattr(value, "__len__") and isinstance(value[0], float))
        else "%d"
    )

    # Determine change_speed
    if isinstance(value, (int, float)):
        change_speed = 0.005 if isinstance(value, float) else 1
        if isinstance(value, float):
            change_speed = max(change_speed, abs(value) * 0.005)
    elif hasattr(value, "__len__"):
        abs_values = [abs(v) for v in value if isinstance(v, (int, float))]
        change_speed = max(0.005, min(abs_values) * 0.005) if abs_values else 0.005
    else:
        return value

    # Handle color uniforms (vec3 or vec4 with "color" in name)
    if (
        hasattr(value, "__len__")
        and isinstance(value[0], float)
        and len(value) in (3, 4)
        and "color" in name.lower()
    ):
        if len(value) == 3:
            return imgui.color_edit3(
                label=name,
                r=value[0],
                g=value[1],
                b=value[2],
            )[1]
        elif len(value) == 4:
            return imgui.color_edit4(
                label=name,
                r=value[0],
                g=value[1],
                b=value[2],
                a=value[3],
            )[1]

    # Existing scalar handling
    if isinstance(value, float):
        return imgui.drag_float(
            label=name,
            value=value,
            change_speed=change_speed,
            format=format_str,
        )[1]
    elif isinstance(value, int):
        return imgui.drag_int(
            label=name,
            value=value,
            change_speed=change_speed,
            format=format_str,
        )[1]
    # Existing vector handling
    elif hasattr(value, "__len__") and isinstance(value[0], float):
        if len(value) == 2:
            return imgui.drag_float2(
                label=name,
                value0=value[0],
                value1=value[1],
                change_speed=change_speed,
                format=format_str,
            )[1]
        elif len(value) == 3:
            return imgui.drag_float3(
                label=name,
                value0=value[0],
                value1=value[1],
                value2=value[2],
                change_speed=change_speed,
                format=format_str,
            )[1]
        elif len(value) == 4:
            return imgui.drag_float4(
                label=name,
                value0=value[0],
                value1=value[1],
                value2=value[2],
                value3=value[3],
                change_speed=change_speed,
                format=format_str,
            )[1]
    return value
