"""One document's preview cell and its verb set.

The GRID this drew is gone (094 C10): the documents live in the breadcrumb's dropdown now, and
what survives is the two pieces that outlived the surface -- the preview cell, which the
Examples and Import popups draw their cards with, and `document_menu_items`, the item set for
the document kind.
"""

from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.commands import CommandId
from shaderbox.menus import target_menu_item
from shaderbox.ui_models import UIDocument
from shaderbox.ui_primitives import (
    PreviewCellResult,
    preview_cell,
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
    if target_menu_item(app, CommandId.OPEN_SCRIPT):
        app.open_script_for(document_id, focus_editor=True)
    imgui.separator()
    if target_menu_item(app, CommandId.OPEN_DOCUMENT_DIR):
        app.open_document_dir(document_id)
    imgui.separator()
    if target_menu_item(app, CommandId.RESET_DOCUMENT, "Reset"):
        app.reset_document(document_id)
    if target_menu_item(app, CommandId.DELETE_DOCUMENT, "Delete"):
        app.delete_document_confirmed(document_id)
