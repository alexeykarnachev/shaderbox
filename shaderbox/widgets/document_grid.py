"""The documents grid: one live thumbnail per document of the open project.

A tile click selects; every other verb -- open it, open its folder, delete it -- is on the
tile's context menu (`document_menu_items`), and the tile itself carries no button.
"""

from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.commands import CommandId
from shaderbox.constants import STARTER_EXAMPLE_ID
from shaderbox.menus import command_hint
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_models import UIDocument
from shaderbox.ui_primitives import (
    PreviewCellResult,
    context_menu_style,
    preview_cell,
    standard_button,
)


def draw_document_preview_button(
    ui_document: UIDocument,
    border_color: tuple[float, float, float, float] | None,
    size: float,
    selected: bool = False,
    stale: bool = False,
) -> PreviewCellResult:
    return preview_cell(
        id_=f"document_{id(ui_document)}",
        cell_w=size,
        texture_glo=ui_document.document.render_pass.canvas.texture.glo,
        texture_size=ui_document.document.render_pass.canvas.texture.size,
        selected=selected,
        armed=False,
        border_color=border_color,
        footer=ui_document.ui_state.ui_name,
        stale=stale,
        deletable=False,
    )


def document_menu_items(app: App, document_id: str) -> None:
    """The items of one document's context menu. The caller owns the popup; each grid tile is
    its own child window, so an explicit id is safe there.

    Every item takes `document_id`, never the current document: a right-click does not select
    the tile it opens on, so a verb routed through the current-document command would act on
    the wrong document (093 W8). The chords are the hints the same verbs carry in the Document
    menu, where they act on the current one. Delete confirms and Reset does not -- one moves a
    directory to the trash, the other restarts a clock.
    """
    if imgui.menu_item("Open script", command_hint(app, CommandId.OPEN_SCRIPT), False)[
        0
    ]:
        app.open_script_for(document_id, focus_editor=True)
    if imgui.menu_item("Open graph", command_hint(app, CommandId.OPEN_GRAPH), False)[0]:
        app.open_graph_for(document_id, focus_editor=True)
    imgui.separator()
    if imgui.menu_item_simple("Open folder"):
        app.open_document_dir(document_id)
    imgui.separator()
    if imgui.menu_item("Reset", command_hint(app, CommandId.RESET_DOCUMENT), False)[0]:
        app.reset_document(document_id)
    if imgui.menu_item("Delete", command_hint(app, CommandId.DELETE_DOCUMENT), False)[
        0
    ]:
        app.delete_document_confirmed(document_id)


def draw_document_preview_grid(app: App, width: float, height: float) -> None:
    with imgui_ctx.begin_child(
        "document_preview_grid",
        size=imgui.ImVec2(width, height),
        child_flags=imgui.ChildFlags_.borders,
        window_flags=imgui.WindowFlags_.no_nav_inputs,
    ):
        # Document create/switch/delete are frozen while a copilot turn runs (§15 A); disable the
        # affordances so the freeze is visible (the verbs also hard-refuse, for non-grid paths).
        imgui.begin_disabled(app.copilot_turn_active)
        if standard_button("New document"):
            app.create_document_from_example(STARTER_EXAMPLE_ID)
        imgui.end_disabled()

        imgui.same_line()

        app.app_state.is_render_all_documents = imgui.checkbox(
            "Render all", app.app_state.is_render_all_documents
        )[1]

        if imgui.is_item_hovered():
            with imgui_ctx.begin_tooltip():
                imgui.text(
                    "If checked, renders all documents, otherwise, renders only the selected one."
                )

        preview_size = SIZE.THUMB_LG
        n_cols = int(imgui.get_content_region_avail().x // (preview_size + SPACE.SM))
        n_cols = max(1, n_cols)
        imgui.begin_disabled(app.copilot_turn_active)
        for i, (id, ui_document) in enumerate(list(app.ui_documents.items())):
            border_color: tuple[float, float, float, float] | None = None
            if id == app.current_document_id:
                if ui_document.document.render_pass.compile_unit.error_raw:
                    border_color = COLOR.STATE_ERROR
                else:
                    border_color = COLOR.SELECT

            result = draw_document_preview_button(
                ui_document,
                border_color,
                preview_size,
                selected=id == app.current_document_id,
                # Mirrors the render gate in ui.py: with "Render all" off, a non-current
                # document stops ticking and its texture is a photograph of the past — and a
                # document still waiting for its first render (066 D2) has none at all.
                stale=(
                    not app.app_state.is_render_all_documents
                    and id != app.current_document_id
                )
                or not ui_document.document.first_render_done,
            )
            with context_menu_style():
                if imgui.begin_popup_context_item(f"##document_menu_{id}"):
                    document_menu_items(app, id)
                    imgui.end_popup()
            if result.clicked:
                app.select_document(id)

            if (i + 1) % n_cols != 0:
                imgui.same_line()
            else:
                imgui.spacing()
        imgui.end_disabled()
