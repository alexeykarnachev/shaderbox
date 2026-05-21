from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.theme import COLOR, SIZE, SPACE


def draw_node_preview_grid(app: App, width: float, height: float) -> None:
    with imgui_ctx.begin_child(
        "node_preview_grid",
        size=imgui.ImVec2(width, height),
        child_flags=imgui.ChildFlags_.borders,
    ):
        if imgui.button("New node"):
            app.open_node_creator()

        imgui.same_line()

        app.app_state.is_render_all_nodes = imgui.checkbox(
            "Render all", app.app_state.is_render_all_nodes
        )[1]

        if imgui.is_item_hovered():
            with imgui_ctx.begin_tooltip():
                imgui.text(
                    "If checked, renders all nodes, otherwise, renders only the selected one."
                )

        imgui.same_line()
        imgui.set_cursor_pos_x(width - 14)
        imgui.text_colored(COLOR.FG_DIM, "?")
        if imgui.is_item_hovered():
            with imgui_ctx.begin_tooltip():
                imgui.text("CREATE new node         CTRL+N")
                imgui.text("SAVE current node       CTRL+S")
                imgui.text("DELETE current node     CTRL+D")
                imgui.text("OPEN project            CTRL+O")
                imgui.text("SETTINGS                 ALT+S")
                imgui.text("PREVIOUS / NEXT node    <-  ->")
                imgui.text("UNFOCUS / close popup      ESC")
                imgui.text("QUIT                    CTRL+Q")

        preview_size = SIZE.THUMB_LG
        n_cols = int(imgui.get_content_region_avail().x // (preview_size + SPACE.SM))
        n_cols = max(1, n_cols)
        for i, (id, ui_node) in enumerate(app.ui_nodes.items()):
            border_color: tuple[float, float, float, float] | None = None
            if id == app.current_node_id:
                if ui_node.node.shader_error:
                    border_color = COLOR.STATE_ERROR
                else:
                    border_color = COLOR.ACCENT_PRIMARY

            if ui_node.draw_preview_button(border_color, preview_size):
                app.set_current_node_id(id)

            if (i + 1) % n_cols != 0:
                imgui.same_line()
            else:
                imgui.spacing()
