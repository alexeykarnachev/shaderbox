from importlib.resources import as_file, files

import crossfiledialog
import imgui
import moderngl
from imgui.integrations.glfw import GlfwRenderer

from shaderbox.graph import Node, RenderGraph


def _scale_size(width: int, height: int, max_size: int) -> tuple[int, int]:
    aspect = width / height
    if aspect > 1:
        width = max_size
        height = int(max_size / aspect)
    else:
        width = int(max_size * aspect)
        height = max_size
    return width, height


def _node_image(
    node: Node,
    max_size: int,
    is_button: bool = False,
    border_size: float = 0.0,
):
    width, height = _scale_size(
        width=node.output_size[0],
        height=node.output_size[1],
        max_size=max_size,
    )
    fn = imgui.image_button if is_button else imgui.image

    imgui.push_style_color(imgui.COLOR_BORDER, 1.0, 1.0, 0.0, 1.0)
    if is_button:
        imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, border_size)
    else:
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (border_size, border_size))

    result = fn(
        node.texture_id,
        width=width,
        height=height,
        uv0=(0, 1),
        uv1=(1, 0),
    )

    imgui.pop_style_color()
    imgui.pop_style_var()
    return result


class UI:
    def __init__(self, window) -> None:
        self._output_node = None

        imgui.create_context()
        self._imgui_renderer = GlfwRenderer(window)

        io = imgui.get_io()
        io.fonts.clear()
        font_resource = (
            files("shaderbox.resources") / "ubuntu-font-family-0.80" / "UbuntuMono-R.ttf"
        )
        with as_file(font_resource) as font_path:
            print(f"Loading font from: {font_path}")
            io.fonts.add_font_from_file_ttf(str(font_path), 20.0)
        self._imgui_renderer.refresh_font_texture()

    def update_and_render(self, graph: RenderGraph):
        if self._output_node is None:
            self._output_node = list(graph.iter_nodes())[-1]

        self._imgui_renderer.process_inputs()
        imgui.new_frame()

        flags = imgui.WINDOW_NO_RESIZE | imgui.WINDOW_ALWAYS_AUTO_RESIZE
        imgui.begin(f"Output Viewer - {self._output_node.name}", flags=flags)
        imgui.set_window_position(0, 0)  # Stick to top-left corner

        # --- Main Image Column ---
        imgui.begin_child(
            "MainImage",
            width=800,
            height=620,
            border=True,
            flags=imgui.WINDOW_NO_SCROLLBAR,
        )
        main_img_width = 800
        imgui.set_cursor_pos_x((imgui.get_window_width() - main_img_width) / 2)
        _node_image(self._output_node, 800)
        imgui.end_child()

        # --- Preview Column ---
        preview_img_width = 100
        scrollbar_width = imgui.get_style().scrollbar_size
        preview_pane_width = preview_img_width + scrollbar_width + 10
        imgui.same_line()
        imgui.begin_child("Previews", width=preview_pane_width, height=620, border=True)
        usable_width = preview_pane_width - scrollbar_width

        for node in graph.iter_nodes():
            is_selected = node == self._output_node
            if is_selected:
                imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 0.0, 1.0)

            text_x = (usable_width - imgui.calc_text_size(node.name).x) / 2
            imgui.set_cursor_pos_x(text_x)
            imgui.text(node.name)

            if is_selected:
                imgui.pop_style_color()

            imgui.set_cursor_pos_x((usable_width - preview_img_width) / 2)
            if _node_image(
                node,
                preview_img_width,
                is_button=True,
                border_size=2.0 if self._output_node is node else 0.0,
            ):
                self._output_node = node

            if node != list(graph.iter_nodes())[-1]:
                imgui.separator()
        imgui.end_child()

        # --- Controls Column ---
        imgui.same_line()
        imgui.begin_child("Controls", width=300, height=620, border=True)

        input_nodes = []
        input_textures = []
        sliders = []
        values = []

        for name, value in self._output_node.uniforms.items():
            if callable(value):
                resolved_value = value()
                if isinstance(resolved_value, Node):
                    input_nodes.append((name, resolved_value))
                elif isinstance(resolved_value, moderngl.Texture):
                    input_textures.append((name, resolved_value))
                else:
                    values.append((name, resolved_value))
            else:
                if isinstance(value, Node):
                    input_nodes.append((name, value))
                elif isinstance(value, moderngl.Texture):
                    input_textures.append((name, value))
                elif isinstance(value, (int, float)) or (
                    isinstance(value, (list, tuple)) and len(value) == 3
                ):
                    sliders.append((name, value))
                else:
                    values.append((name, value))

        if input_nodes:
            imgui.text("Input Nodes")
            imgui.separator()
            for name, value in input_nodes:
                imgui.text(f"{name}:")
                imgui.same_line()
                if _node_image(value, 70, is_button=True):
                    self._output_node = value
            imgui.spacing()

        if input_textures:
            imgui.text("Textures")
            imgui.separator()
            for name, value in input_textures:
                imgui.text(f"{name}:")
                imgui.same_line()
                if isinstance(value, moderngl.Texture) and not callable(
                    self._output_node.uniforms[name]
                ):
                    if imgui.image_button(value.glo, 70, 70, uv0=(0, 1), uv1=(1, 0)):
                        new_file = crossfiledialog.open_file(
                            title=f"Select texture for {name}"
                        )
                        if new_file:
                            new_texture = self._output_node._gl_context.context.texture(
                                self._output_node.output_size,
                                4,
                                data=open(new_file, "rb").read(),
                            )
                            self._output_node.uniforms[name] = new_texture
                    imgui.same_line()
                    imgui.text("Click to replace")
                else:
                    imgui.image(value.glo, 70, 70, uv0=(0, 1), uv1=(1, 0))
            imgui.spacing()

        if sliders:
            imgui.text("Adjustable Parameters")
            imgui.separator()
            for name, value in sliders:
                if isinstance(value, (int, float)):
                    changed, new_value = imgui.slider_float(
                        f"{name}", value, -10.0, 10.0, "%.2f"
                    )
                    if changed:
                        self._output_node.uniforms[name] = new_value
                elif isinstance(value, (list, tuple)) and len(value) == 3:
                    components = list(value)
                    changed = False
                    for i in range(3):
                        c_changed, c_value = imgui.slider_float(
                            f"{name}[{i}]", components[i], -1.0, 1.0, "%.2f"
                        )
                        if c_changed:
                            components[i] = c_value
                            changed = True
                    if changed:
                        self._output_node.uniforms[name] = tuple(components)
            imgui.spacing()

        if values:
            imgui.text("Static Values")
            imgui.separator()
            for name, value in values:
                if isinstance(value, (int, float)):
                    imgui.text(f"{name}: {value:.2f}")
                elif isinstance(value, (list, tuple)) and len(value) == 3:
                    imgui.text(
                        f"{name}: ({value[0]:.2f}, {value[1]:.2f}, {value[2]:.2f})"
                    )
                else:
                    imgui.text(f"{name}: {value}")

        imgui.spacing()
        imgui.end_child()

        imgui.end()
        imgui.render()
        self._imgui_renderer.render(imgui.get_draw_data())
