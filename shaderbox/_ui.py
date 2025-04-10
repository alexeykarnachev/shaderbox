import crossfiledialog
import imgui
from imgui.integrations.glfw import GlfwRenderer


class UI:
    def __init__(self, window) -> None:
        self._selected_node = None

        imgui.create_context()
        self._imgui_impl = GlfwRenderer(window)

    def render(self, node: ShaderNode):
        """Render the UI with two panes and handle selection/highlighting in one place."""
        self._imgui_impl.process_inputs()
        imgui.new_frame()

        # Pane 1: Render DAG Tree
        imgui.begin("Render DAG")

        def render_tree(current_node):
            node_name = current_node.name or str(id(current_node))
            is_selected = current_node == self._selected_node
            flags = imgui.TREE_NODE_SELECTED if is_selected else 0

            if imgui.tree_node(node_name, flags):
                if imgui.is_item_clicked():
                    self._selected_node = current_node

                if imgui.is_item_hovered():
                    imgui.set_tooltip(node_name)

                child_nodes = [
                    u
                    for u in current_node.uniforms.values()
                    if isinstance(u, ShaderNode)
                ]
                for child_node in child_nodes:
                    render_tree(child_node)

                imgui.tree_pop()

        render_tree(node)
        imgui.end()

        # Pane 2: Uniform Controls
        imgui.begin("Uniform Controls")
        if self._selected_node:
            imgui.text(
                f"Uniforms for {self._selected_node.name or str(id(self._selected_node))}"
            )
            imgui.separator()

            for uniform_name, uniform in self._selected_node.uniforms.items():
                if isinstance(uniform, Uniform):
                    changed, new_value = imgui.slider_float(
                        uniform_name,
                        uniform.value,
                        uniform.min_value,
                        uniform.max_value,
                        format="%.3f",
                    )
                    if changed:
                        uniform.value = new_value
                elif callable(uniform):
                    value = uniform()
                    if isinstance(value, moderngl.Texture):
                        max_size = 100
                        aspect_ratio = value.width / value.height
                        if aspect_ratio > 1:
                            preview_width = max_size
                            preview_height = int(max_size / aspect_ratio)
                        else:
                            preview_width = int(max_size * aspect_ratio)
                            preview_height = max_size
                        imgui.text(f"{uniform_name}:")
                        imgui.image(
                            value.glo,
                            preview_width,
                            preview_height,
                            uv0=(0, 1),
                            uv1=(1, 0),
                        )
                        if imgui.button(f"Select {uniform_name}"):
                            new_path = crossfiledialog.open_file(
                                filter="Images (*.png *.jpg *.jpeg)"
                            )
                            if new_path:
                                uniform.reload_texture(new_path)
                        imgui.text(f"File: {uniform.path.split('/')[-1]}")
                    else:
                        imgui.text(f"{uniform_name}: {value:.3f}")
                elif isinstance(uniform, moderngl.Texture):
                    imgui.text(
                        f"{uniform_name}: Texture ({uniform.width}x{uniform.height})"
                    )
                elif not isinstance(uniform, ShaderNode):
                    imgui.text(f"{uniform_name}: {uniform}")
        else:
            imgui.text("Select a node to edit uniforms.")
        imgui.end()

        imgui.render()
        self._imgui_impl.render(imgui.get_draw_data())
