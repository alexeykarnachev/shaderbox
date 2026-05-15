import imgui

from shaderbox.app import App


def draw_node_preview_grid(app: App, width: float, height: float) -> None:
    with imgui.begin_child(
        "node_preview_grid", width=width, height=height, border=True
    ):
        if imgui.button("New node"):
            app.open_node_creator()

        imgui.same_line()

        app.app_state.is_render_all_nodes = imgui.checkbox(
            "Render all", app.app_state.is_render_all_nodes
        )[1]

        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.text(
                "If checked, renders all nodes, otherwise, renders only the selected one."
            )
            imgui.end_tooltip()

        imgui.same_line()
        imgui.set_cursor_pos_x(width - 14)
        imgui.text_colored("?", *(0.5, 0.5, 0.5))
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.text("CREATE new node         CTRL+N")
            imgui.text("SAVE current node       CTRL+S")
            imgui.text("EDIT current node       CTRL+E")
            imgui.text("DELETE current node     CTRL+D")
            imgui.text("PREVIOUS node             <-")
            imgui.text("NEXT node                 ->")
            imgui.end_tooltip()

        preview_size = 150
        n_cols = int(imgui.get_content_region_available()[0] // (preview_size + 5))
        n_cols = max(1, n_cols)
        for i, (id, ui_node) in enumerate(app.ui_nodes.items()):
            border_color: tuple[float, float, float] | None = None
            if id == app.current_node_id:
                if ui_node.node.shader_error:
                    border_color = (1.0, 0.0, 0.0)
                else:
                    border_color = (0.0, 1.0, 0.0)

            if ui_node.draw_preview_button(border_color, preview_size):
                app.set_current_node_id(id)

            if (i + 1) % n_cols != 0:
                imgui.same_line()
            else:
                imgui.spacing()
