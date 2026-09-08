from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.glyph_tables import TABLE_UNIFORMS
from shaderbox.pass_graph import strip_order
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_models import UIUniform, UniformSortKey, sort_uniform_hashes
from shaderbox.ui_primitives import standard_button, text_tab_row
from shaderbox.util import format_auto_value, get_uniform_hash
from shaderbox.widgets.uniform import draw_ui_uniform, uniform_name_label

# The uniforms panel, its own tab since 083: on the Document tab it shared the space with the pass
# strip and had too little of it. What a pass IS stays on the Document tab (the strip, its six
# verbs, the canvas fields); what its uniforms are SET TO lives here.


def _draw_auto_block(app: App, uniforms: list[UIUniform]) -> None:
    # Engine-driven uniforms: one row each under the sort row, outside the sorted list. A FIXED
    # name column is what makes the rows read as a block; the names keep the code<->panel
    # hover/jump bridge, values are read-only.
    panel_pass = app.panel_pass(app.current_document_id)
    imgui.push_font(app.font_12, app.font_12.legacy_size)
    for u in uniforms:
        uniform_name_label(
            app,
            u.name,
            float(SIZE.AUTO_NAME_W),
            text_color=COLOR.STATE_INFO,
            accent=COLOR.STATE_INFO,
        )
        imgui.same_line(float(SIZE.AUTO_NAME_W) + float(SPACE.MD))
        value = panel_pass.uniform_values.get(u.name)
        imgui.text_colored(COLOR.FG_DIM, format_auto_value(value))
    imgui.pop_font()


def _draw_pass_selector(app: App, document_id: str) -> None:
    # Which pass's uniforms these are. The panel follows the editor's active shader tab by default
    # (App.panel_pass), so this row is how a pass that is neither open nor on screen gets tuned --
    # picking one here pins it until a shader tab is opened, which retires the pick.
    document = app.ui_documents[document_id].document
    # The strip's own order (producers left of consumers), so the combo lists the passes
    # in the order the Document tab shows them rather than by name.
    names = strip_order(document.passes, document.effective_wiring())
    if len(names) < 2:
        return
    current = app.panel_pass(document_id)
    current_name = next(
        (n for n in names if document.passes[n] is current), names[0] if names else ""
    )
    picked = text_tab_row("uniforms_pass", names, current_name)
    if picked is not None:
        app.set_panel_pass(document_id, picked)
    imgui.dummy((0, SPACE.MD))


def draw(app: App) -> None:
    document_id = app.current_document_id
    if document_id not in app.ui_documents:
        return

    imgui.spacing()
    _draw_pass_selector(app, document_id)

    document_ui_state = app.ui_documents[document_id].ui_state
    ui_uniforms = document_ui_state.ui_uniforms

    active_uniform_hashes = []
    auto_hashes = []
    # The PANEL pass, not the output: the sliders belong to the pass being edited (065).
    for uniform in app.panel_pass(document_id).get_active_uniforms():
        if (
            uniform.name in TABLE_UNIFORMS
        ):  # engine glyph tables — pure machinery, no row
            continue
        hash = get_uniform_hash(uniform)
        if hash not in ui_uniforms:
            ui_uniforms[hash] = UIUniform.from_uniform(uniform)
        ui_uniforms[hash].snap_input_type()
        if ui_uniforms[hash].input_type == "auto":
            auto_hashes.append(hash)
        else:
            active_uniform_hashes.append(hash)

    sort_keys: list[UniformSortKey] = ["code", "name", "type"]
    imgui.set_next_item_width(SIZE.SORT_COMBO_W)
    if imgui.begin_combo(
        "##uniform_sort_key", f"Sort by: {document_ui_state.uniform_sort_key}"
    ):
        for key in sort_keys:
            if imgui.selectable(key, key == document_ui_state.uniform_sort_key)[0]:
                document_ui_state.uniform_sort_key = key
        imgui.end_combo()

    imgui.same_line()
    arrow = "v" if document_ui_state.uniform_sort_desc else "^"
    if standard_button(f"{arrow}##uniform_sort_dir", width=SIZE.BTN_SM_H):
        document_ui_state.uniform_sort_desc = not document_ui_state.uniform_sort_desc

    imgui.dummy((0, SPACE.MD))

    if auto_hashes:
        _draw_auto_block(app, [ui_uniforms[h] for h in auto_hashes])
        imgui.dummy((0, SPACE.MD))

    sorted_hashes = sort_uniform_hashes(
        active_uniform_hashes,
        ui_uniforms,
        document_ui_state.uniform_sort_key,
        document_ui_state.uniform_sort_desc,
    )

    # auto_resize_y: the child grows to its content so the WHOLE tab scrolls as one surface —
    # a fixed-size child here put a scrollbar on just the uniforms pane.
    with imgui_ctx.begin_child(
        "ui_uniforms",
        child_flags=imgui.ChildFlags_.auto_resize_y,
    ):
        for hash in sorted_hashes:
            draw_ui_uniform(app, ui_uniforms[hash])
            imgui.dummy((0, SPACE.SM))
