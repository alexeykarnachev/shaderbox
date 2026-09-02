from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.glyph_tables import TABLE_UNIFORMS
from shaderbox.media import MediaWithTexture, is_default_image
from shaderbox.pass_graph import clamp_canvas_size
from shaderbox.render_preset import resolve_dims
from shaderbox.render_shape import (
    MENU_SHAPES,
    SHAPE_TABLE,
    RenderShape,
    shape_to_preset,
)
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_models import (
    UIDocument,
    UIUniform,
    UniformSortKey,
    sort_uniform_hashes,
)
from shaderbox.ui_primitives import (
    button,
    ghost_button,
    play_stop_toggle,
    small_caption,
)
from shaderbox.util import format_auto_value, get_resolution_str, get_uniform_hash
from shaderbox.widgets import pass_list
from shaderbox.widgets.uniform import draw_ui_uniform, uniform_name_label

_SQUARE_PRESETS: tuple[int, ...] = (256, 512, 1024, 2048)


def _apply_canvas_size(
    app: App, ui_document: UIDocument, size: tuple[int, int]
) -> None:
    w, h = clamp_canvas_size(size)
    if (w, h) == ui_document.document.canvas_size:
        return
    ui_document.document.set_canvas_size((w, h))
    app.notifications.push(f"Canvas: {w}x{h}")


def _canvas_presets(ui_document: UIDocument) -> list[tuple[str, tuple[int, int]]]:
    """Squares, the named video shapes, then any bound texture's size, across ALL passes.

    Reads `uniform_values`, never `get_active_uniforms()`: the latter compiles a
    never-attempted pass (066 D1), which on the Document tab's every frame would compile
    the whole graph.
    """
    current = ui_document.document.canvas_size
    presets: list[tuple[str, tuple[int, int]]] = []
    seen: set[tuple[int, int]] = {current}

    for n in _SQUARE_PRESETS:
        size = (n, n)
        if size in seen:
            continue
        seen.add(size)
        presets.append((get_resolution_str(None, n, n), size))

    for shape in MENU_SHAPES:
        if shape is RenderShape.NATIVE:
            continue
        size = resolve_dims(
            shape_to_preset(
                shape, is_video=False, fps=None, container=None, duration_max=None
            ),
            current,
        )
        if size in seen:
            continue
        seen.add(size)
        presets.append((SHAPE_TABLE[shape].menu_label, size))

    for render_pass in ui_document.document.passes.values():
        for uniform_name, value in sorted(render_pass.uniform_values.items()):
            if not isinstance(value, MediaWithTexture) or is_default_image(value):
                continue
            size = value.texture.size
            if size in seen:
                continue
            seen.add(size)
            presets.append((get_resolution_str(uniform_name, *size), size))

    return presets


def _section_break() -> None:
    imgui.spacing()
    imgui.separator()
    imgui.spacing()


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


def draw(app: App) -> None:
    if not (ui_document := app.ui_documents.get(app.current_document_id)):
        return

    imgui.spacing()

    document_ui_state = app.current_document_ui_state_or_default

    combo_offset = SIZE.NAME_INPUT_W + SPACE.XL

    imgui.begin_disabled(app.copilot_turn_active)

    small_caption(app.font_12, "Document name")
    imgui.same_line(combo_offset)
    small_caption(app.font_12, "Canvas")

    imgui.set_next_item_width(SIZE.NAME_INPUT_W)
    ui_document.ui_state.ui_name = imgui.input_text_with_hint(
        "##document_name", "document name", ui_document.ui_state.ui_name
    )[1]

    imgui.same_line(combo_offset)

    # Per-document widget ids. Clearing the editing flags on a switch is not enough on its own:
    # imgui keeps the ITEM active across it, so a shared `##canvas_w` would let the outgoing
    # document's half-typed digit re-latch onto the incoming one and commit to it.
    imgui.push_id(ui_document.id)

    # Each half mirrors the document unless ITS OWN field is active, so a field the user is not
    # in never holds a stale number to carry over an external write.
    doc_w, doc_h = ui_document.document.canvas_size
    if not app.canvas_w_editing:
        app.canvas_size_buf = (doc_w, app.canvas_size_buf[1])
    if not app.canvas_h_editing:
        app.canvas_size_buf = (app.canvas_size_buf[0], doc_h)

    imgui.set_next_item_width(float(SIZE.CANVAS_FIELD_W))
    entered_w, buf_w = imgui.input_int(
        "##canvas_w",
        app.canvas_size_buf[0],
        step=0,
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    active_w = imgui.is_item_active()
    committed_w = entered_w or imgui.is_item_deactivated_after_edit()
    app.canvas_size_buf = (buf_w, app.canvas_size_buf[1])

    imgui.same_line(spacing=float(SPACE.SM))
    imgui.text_colored(COLOR.FG_DIM, "x")
    imgui.same_line(spacing=float(SPACE.SM))

    imgui.set_next_item_width(float(SIZE.CANVAS_FIELD_W))
    entered_h, buf_h = imgui.input_int(
        "##canvas_h",
        app.canvas_size_buf[1],
        step=0,
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    active_h = imgui.is_item_active()
    committed_h = entered_h or imgui.is_item_deactivated_after_edit()
    app.canvas_size_buf = (app.canvas_size_buf[0], buf_h)

    app.canvas_w_editing = active_w
    app.canvas_h_editing = active_h

    # The buffer IS the pair to commit: the mirror above already refreshed the half whose field
    # is not active from the document this same frame, so an external write during the edit
    # stands without the commit re-reading it.
    if committed_w or committed_h:
        _apply_canvas_size(app, ui_document, app.canvas_size_buf)
        app.canvas_size_buf = ui_document.document.canvas_size

    imgui.same_line(spacing=float(SPACE.MD))
    imgui.set_next_item_width(float(SIZE.CANVAS_PRESETS_W))
    if imgui.begin_combo(
        "##canvas_presets", "presets", imgui.ComboFlags_.no_arrow_button
    ):
        for label, size in _canvas_presets(ui_document):
            if imgui.selectable(label, False)[0]:
                _apply_canvas_size(app, ui_document, size)
                app.canvas_size_buf = ui_document.document.canvas_size
        imgui.end_combo()

    imgui.pop_id()

    imgui.end_disabled()

    imgui.dummy((0, SPACE.MD))
    _draw_entry_points(app)

    _section_break()

    ui_uniforms = document_ui_state.ui_uniforms

    active_uniform_hashes = []
    auto_hashes = []
    # The PANEL pass, not the output: the sliders belong to the pass being edited (065).
    for uniform in app.panel_pass(app.current_document_id).get_active_uniforms():
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
    if button(f"{arrow}##uniform_sort_dir", width=SIZE.BTN_SM_H):
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


# The Shader/Script label column: a tick gutter + the widest label, so both `open` buttons align.
_ENTRY_TICK_W = float(SPACE.MD)
_ENTRY_LABEL_W = 64.0


def _entry_row_label(active: bool, label: str) -> None:
    # One entry-point row label: an inset accent tick (049) marking the editor's active tab, then the
    # label in a fixed-width column so the `open` buttons line up across rows. align_text_to_frame_
    # padding centres the text on the button's row height (the font mix used to float it high). The tick
    # is a draw-list line (presence/colour only, never size — /imgui-ui §3) drawn over the gutter.
    imgui.align_text_to_frame_padding()
    pos = imgui.get_cursor_screen_pos()
    if active:
        h = imgui.get_frame_height()
        col = imgui.color_convert_float4_to_u32(COLOR.ACCENT_PRIMARY)
        imgui.get_window_draw_list().add_line(
            (pos.x, pos.y + 2.0), (pos.x, pos.y + h - 2.0), col, 2.0
        )
    imgui.dummy((_ENTRY_TICK_W, 0))
    imgui.same_line()
    imgui.text_colored(COLOR.FG_DIM, label)
    imgui.same_line(_ENTRY_TICK_W + _ENTRY_LABEL_W)


def _draw_entry_points(app: App) -> None:
    # The document's two entry-points (049): SHADER (GPU) and SCRIPT (CPU script), each with an `open`
    # action that summons its tab into the editor (the document panel is "about this document"; the tab bar is
    # the editor's own state — `open` is a summoner, not a duplicate). The whole-document PLAY/STOP toggle
    # lives on the Script row (its true owner — it freezes/resumes the script's driven uniforms; the
    # script keeps ticking). An accent tick marks whichever entry-point is the editor's active tab.
    # Frozen mid-copilot-turn (a write races the reload).
    document_id = app.current_document_id
    present = app.session.has_script(document_id)
    error = present and app.session.script_has_error(document_id)
    active = app.active_tab
    script_active = (
        active is not None
        and active.kind == "script"
        and active.document_id == document_id
    )

    pass_list.draw(
        app, document_id, lambda name: app.ensure_shader_tab(document_id, name)
    )

    imgui.dummy((0, float(SPACE.MD)))
    imgui.begin_disabled(app.copilot_turn_active)

    # ONE row, no section caption: a document has exactly one script (048), so a heading over a
    # single control said the word twice and cost a line the panel could not spare.
    _entry_row_label(script_active, "Script")
    open_tooltip = (
        "Open the document script" if present else "Create the document script"
    )
    open_color = COLOR.STATE_ERROR if error else COLOR.FG_SECONDARY
    if ghost_button("open##entry_script", text_color=open_color):
        app.open_script_for(document_id, focus_editor=True)
    if imgui.is_item_hovered():
        imgui.set_tooltip(open_tooltip)
    if present:
        imgui.same_line()
        playing = not app.current_document_ui_state_or_default.all_stopped
        if play_stop_toggle(
            "document",
            playing,
            tooltip="Stop the whole script" if playing else "Resume the whole script",
        ):
            app.set_document_all_stopped(document_id, playing)
    imgui.end_disabled()
