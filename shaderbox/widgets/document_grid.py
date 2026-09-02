from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.constants import STARTER_EXAMPLE_ID
from shaderbox.theme import COLOR, SIZE, SPACE
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
    armed: bool = False,
    stale: bool = False,
) -> PreviewCellResult:
    return preview_cell(
        id_=f"document_{id(ui_document)}",
        cell_w=size,
        texture_glo=ui_document.document.render_pass.canvas.texture.glo,
        texture_size=ui_document.document.render_pass.canvas.texture.size,
        selected=selected,
        armed=armed,
        border_color=border_color,
        footer=ui_document.ui_state.ui_name,
        stale=stale,
    )


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
        if imgui.button("New document"):
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
        # Snapshot: the delete-confirm fires app.delete_document, which mutates
        # app.ui_documents; deferring the pop until after the loop avoids mutating
        # the dict mid-iteration.
        id_to_delete: str | None = None
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
                armed=app.document_delete_armed == id,
                # Mirrors the render gate in ui.py: with "Render all" off, a non-current
                # document stops ticking and its texture is a photograph of the past — and a
                # document still waiting for its first render (066 D2) has none at all.
                stale=(
                    not app.app_state.is_render_all_documents
                    and id != app.current_document_id
                )
                or not ui_document.document.first_render_done,
            )
            if result.clicked:
                app.select_document(id)
            if result.delete_armed:
                app.set_document_delete_armed(id)
            elif result.delete_confirmed:
                id_to_delete = id
            elif result.delete_cancelled:
                app.set_document_delete_armed("")

            if (i + 1) % n_cols != 0:
                imgui.same_line()
            else:
                imgui.spacing()
        imgui.end_disabled()

    if id_to_delete is not None:
        app.delete_document(id_to_delete)
