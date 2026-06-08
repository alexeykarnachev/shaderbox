from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.commands import ActiveRegion
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_models import UINode
from shaderbox.ui_primitives import (
    PreviewCellResult,
    active_region_outline,
    preview_cell,
)


def draw_node_preview_button(
    ui_node: UINode,
    border_color: tuple[float, float, float, float] | None,
    size: float,
    selected: bool = False,
    armed: bool = False,
    nav_flatten: bool = False,
) -> PreviewCellResult:
    return preview_cell(
        id_=f"node_{id(ui_node)}",
        cell_w=size,
        texture_glo=ui_node.node.canvas.texture.glo,
        texture_size=ui_node.node.canvas.texture.size,
        selected=selected,
        armed=armed,
        border_color=border_color,
        footer=ui_node.ui_state.ui_name,
        nav_flatten=nav_flatten,
    )


def draw_node_preview_grid(app: App, width: float, height: float) -> None:
    grid_active = app.active_region == ActiveRegion.GRID
    # Consume (read + clear) the grid's own focus latch — see _draw_node_settings.
    if app.region_focus_pending and grid_active:
        imgui.set_next_window_focus()
        app.region_focus_pending = False
    grid_flags = (
        imgui.WindowFlags_.none if grid_active else imgui.WindowFlags_.no_nav_inputs
    )
    with imgui_ctx.begin_child(
        "node_preview_grid",
        size=imgui.ImVec2(width, height),
        child_flags=imgui.ChildFlags_.borders,
        window_flags=grid_flags,
    ):
        # Adopt GRID as the active region from live focus. See the editor pane in ui.py.
        if (
            imgui.is_window_focused(imgui.FocusedFlags_.child_windows)
            and app.region_derive_allowed()
        ):
            app.active_region = ActiveRegion.GRID
        if app.region_outline_visible(ActiveRegion.GRID):
            active_region_outline()
        # Node create/switch/delete are frozen while a copilot turn runs (§15 A); disable the
        # affordances so the freeze is visible (the verbs also hard-refuse, for non-grid paths).
        imgui.begin_disabled(app.copilot_turn_active)
        if imgui.button("New node"):
            app.open_node_creator()
        imgui.end_disabled()

        imgui.same_line()

        app.app_state.is_render_all_nodes = imgui.checkbox(
            "Render all", app.app_state.is_render_all_nodes
        )[1]

        if imgui.is_item_hovered():
            with imgui_ctx.begin_tooltip():
                imgui.text(
                    "If checked, renders all nodes, otherwise, renders only the selected one."
                )

        preview_size = SIZE.THUMB_LG
        n_cols = int(imgui.get_content_region_avail().x // (preview_size + SPACE.SM))
        n_cols = max(1, n_cols)
        # Snapshot: the delete-confirm fires app.delete_node, which mutates
        # app.ui_nodes; deferring the pop until after the loop avoids mutating
        # the dict mid-iteration.
        id_to_delete: str | None = None
        imgui.begin_disabled(app.copilot_turn_active)
        for i, (id, ui_node) in enumerate(list(app.ui_nodes.items())):
            border_color: tuple[float, float, float, float] | None = None
            if id == app.current_node_id:
                if ui_node.node.compile_unit.error_raw:
                    border_color = COLOR.STATE_ERROR
                else:
                    border_color = COLOR.SELECT

            result = draw_node_preview_button(
                ui_node,
                border_color,
                preview_size,
                selected=id == app.current_node_id,
                armed=app.node_delete_armed == id,
                nav_flatten=True,
            )
            if result.clicked:
                app.select_node(id)
            if result.delete_armed:
                app.set_node_delete_armed(id)
            elif result.delete_confirmed:
                id_to_delete = id
            elif result.delete_cancelled:
                app.set_node_delete_armed("")

            if (i + 1) % n_cols != 0:
                imgui.same_line()
            else:
                imgui.spacing()
        imgui.end_disabled()

    if id_to_delete is not None:
        app.delete_node(id_to_delete)
