from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.media import MediaWithTexture
from shaderbox.pass_graph import clamp_canvas_size
from shaderbox.render_preset import resolve_dims
from shaderbox.render_shape import (
    ASPECT_PRESETS,
    MENU_SHAPES,
    SHAPE_TABLE,
    RenderShape,
    ResolutionMode,
    aspect_label,
    aspect_of,
    reduce_aspect,
    shape_to_preset,
)
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_models import (
    UIDocument,
)
from shaderbox.ui_primitives import (
    chip_button,
    danger_button,
    play_stop_toggle,
    segmented_choice,
    small_caption,
    standard_button,
)
from shaderbox.ui_regions import PASSES_VIEW_LABELS, PassesView
from shaderbox.util import get_resolution_str
from shaderbox.widgets import pass_graph, pass_list

_SQUARE_PRESETS: tuple[int, ...] = (256, 512, 1024, 2048)


def _draw_canvas_presets(app: App, ui_document: UIDocument) -> None:
    # A chip, not a combo: it names a shortcut rather than a value the row holds, so it does not
    # read as a third canvas field. The popup is a combo's, opened off the chip.
    label = "presets"
    width = imgui.calc_text_size(label).x + 2.0 * float(SPACE.MD)
    imgui.set_next_item_width(width)
    if imgui.begin_combo("##canvas_presets", label, imgui.ComboFlags_.no_arrow_button):
        for preset_label, size in _canvas_presets(ui_document):
            if imgui.selectable(preset_label, False)[0]:
                _apply_canvas_size(app, ui_document, size)
                app.canvas_size_buf = ui_document.document.resolution
        imgui.end_combo()


_MODES: tuple[ResolutionMode, ...] = (ResolutionMode.AUTO, ResolutionMode.FIXED)


def _draw_resolution_mode(app: App, ui_document: UIDocument) -> None:
    # A segmented control, not a lone toggle: the two modes are mutually exclusive positions of
    # one setting, and a single `Fixed` button left "not Fixed" unnamed. Both words on screen
    # also say what the other position would do.
    current = _MODES.index(ui_document.document.resolution_mode)
    chosen = segmented_choice("resolution_mode", ("Auto", "Fixed"), current)
    if chosen != current:
        _switch_resolution_mode(app, ui_document, _MODES[chosen])


def _switch_resolution_mode(
    app: App, ui_document: UIDocument, mode: ResolutionMode
) -> None:
    """Change the mode, seeding the field the new mode reads from the live canvas.

    A switch must never jump the picture (revision 1 D3): Auto -> Fixed takes the size the
    document is rendering at right now, so the canvas keeps its pixels; Fixed -> Auto takes
    that size's reduced ratio, so the shape survives and only the sizing rule changes.
    """
    document = ui_document.document
    if mode is ResolutionMode.FIXED:
        seeded = document.clamped_size(document.canvas_size)
        ui_document.ui_state.resolution = seeded
        document.resolution = seeded
        app.canvas_size_buf = seeded
        # The live size is already `seeded`; the deferred write is what makes the Fixed branch
        # of the next tick agree rather than resize on its first frame.
        app.pending_resolution[ui_document.id] = seeded
    else:
        seeded_aspect = aspect_of(document.canvas_size)
        ui_document.ui_state.aspect = seeded_aspect
        document.aspect = seeded_aspect
    ui_document.ui_state.resolution_mode = mode
    document.resolution_mode = mode


def _apply_aspect(app: App, ui_document: UIDocument, ratio: tuple[int, int]) -> None:
    reduced = reduce_aspect(ratio)
    if reduced == ui_document.ui_state.aspect:
        return
    ui_document.ui_state.aspect = reduced
    ui_document.document.aspect = reduced
    app.aspect_buf = reduced
    app.notifications.push(f"Aspect: {reduced[0]}:{reduced[1]}")


def _draw_aspect_control(app: App, ui_document: UIDocument) -> None:
    """The whole Auto-side control: a row of preset chips plus two fields for a custom ratio.

    One function on purpose -- the layout is the half still being designed, so it can be
    rebuilt without touching the model, the seeding or the tests behind it. What it owes the
    rest of the app is only this: whatever the user picks reaches `_apply_aspect` reduced.
    """
    current = ui_document.ui_state.aspect
    for preset in ASPECT_PRESETS:
        label = f"{preset[0]}:{preset[1]}"
        width = imgui.calc_text_size(label).x + 2.0 * float(SPACE.MD)
        if chip_button(
            label, width, imgui.get_frame_height(), active=preset == current
        ):
            _apply_aspect(app, ui_document, preset)
        imgui.same_line(spacing=float(SPACE.SM))

    # The custom pair mirrors the document unless ITS OWN field is active, the same rule the
    # canvas fields follow: a field the user is not in never holds a stale number.
    if not app.aspect_w_editing:
        app.aspect_buf = (current[0], app.aspect_buf[1])
    if not app.aspect_h_editing:
        app.aspect_buf = (app.aspect_buf[0], current[1])

    imgui.same_line(spacing=float(SPACE.MD))
    imgui.set_next_item_width(float(SIZE.ASPECT_FIELD_W))
    entered_w, buf_w = imgui.input_int(
        "##aspect_w",
        app.aspect_buf[0],
        step=0,
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    active_w = imgui.is_item_active()
    committed_w = entered_w or imgui.is_item_deactivated_after_edit()
    app.aspect_buf = (buf_w, app.aspect_buf[1])

    imgui.same_line(spacing=float(SPACE.SM))
    imgui.text_colored(COLOR.FG_DIM, ":")
    imgui.same_line(spacing=float(SPACE.SM))

    imgui.set_next_item_width(float(SIZE.ASPECT_FIELD_W))
    entered_h, buf_h = imgui.input_int(
        "##aspect_h",
        app.aspect_buf[1],
        step=0,
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    active_h = imgui.is_item_active()
    committed_h = entered_h or imgui.is_item_deactivated_after_edit()
    app.aspect_buf = (app.aspect_buf[0], buf_h)

    app.aspect_w_editing = active_w
    app.aspect_h_editing = active_h
    if committed_w or committed_h:
        _apply_aspect(app, ui_document, app.aspect_buf)
        app.aspect_buf = ui_document.ui_state.aspect


def _draw_canvas_fields(app: App, ui_document: UIDocument) -> None:
    """The whole Fixed-side control: the W x H pair and the presets chip, exactly as before 090.

    The pair is meaningful ONLY under Fixed, so it is not drawn under Auto at all -- an editable
    number the mode does not read is a control that lies about what it does.
    """
    # Each half mirrors the document unless ITS OWN field is active, so a field the user is not
    # in never holds a stale number to carry over an external write.
    doc_w, doc_h = ui_document.document.resolution
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

    imgui.same_line(spacing=float(SPACE.MD))
    _draw_canvas_presets(app, ui_document)

    app.canvas_w_editing = active_w
    app.canvas_h_editing = active_h

    # The buffer IS the pair to commit: the mirror above already refreshed the half whose field
    # is not active from the document this same frame, so an external write during the edit
    # stands without the commit re-reading it.
    if committed_w or committed_h:
        _apply_canvas_size(app, ui_document, app.canvas_size_buf)
        app.canvas_size_buf = ui_document.document.resolution


def _draw_size_readout(app: App, ui_document: UIDocument, control_x: float) -> None:
    """The second line under the mode's control: the number the control does not carry.

    Under Auto the control names a RATIO, so the readout is the live pixel size; under Fixed
    the control names a SIZE, so the readout is its aspect. Each mode shows the other half of
    the same fact, and neither repeats what is already on the row above. A plain readout, no
    label -- the caption already said which of the two is being edited.
    """
    document = ui_document.document
    if document.resolution_mode is ResolutionMode.FIXED:
        text = aspect_label(document.resolution)
    else:
        width, height = document.canvas_size
        text = f"{width}x{height}"
    imgui.set_cursor_pos_x(control_x)
    small_caption(app.font_12, text)


def _draw_document_reset(app: App) -> None:
    # Reset is about THIS document — its histories, clock, script and videos — so it sits on the
    # row that names the document, at its right border, and stays live during a copilot turn as
    # it always has. Destructive, so the danger tier (079 D7, D12).
    label = "Reset"
    width = imgui.calc_text_size(label).x + 2.0 * imgui.get_style().frame_padding.x
    imgui.same_line()
    # Right border of the panel: `get_content_region_avail` is measured from the cursor the
    # `same_line` just left, so the remaining width minus the button's own is the offset.
    imgui.set_cursor_pos_x(
        imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x - width
    )
    imgui.end_disabled()
    if danger_button(label, width=width):
        app.reset_current_document()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Reset document")
    imgui.begin_disabled(app.copilot_turn_active)


def _apply_canvas_size(
    app: App, ui_document: UIDocument, size: tuple[int, int]
) -> None:
    """Commit the picker's W x H as the document's stored `resolution` (090 D5).

    The write is DEFERRED, never applied in place: this runs inside the draw phase, after
    `_draw_document_image` pushed the output texture into this frame's draw list, and a resize
    releases that texture and every feedback history with it. Step 4 of the next
    `_tick_frame_state` consumes `pending_resolution` through the same path Auto takes, where
    every release precedes any draw — the shape `pending_project_switch` uses for the same
    084 D5 hazard.
    """
    w, h = clamp_canvas_size(size)
    if (w, h) == ui_document.ui_state.resolution:
        return
    ui_document.ui_state.resolution = (w, h)
    ui_document.document.resolution = (w, h)
    app.pending_resolution[ui_document.id] = (w, h)
    label = (
        "Canvas"
        if ui_document.document.resolution_mode is ResolutionMode.FIXED
        else "Export"
    )
    app.notifications.push(f"{label}: {w}x{h}")


def _canvas_presets(ui_document: UIDocument) -> list[tuple[str, tuple[int, int]]]:
    """Squares, the named video shapes, then any bound texture's size, across ALL passes.

    Reads `uniform_values`, never `get_active_uniforms()`: the latter compiles a
    never-attempted pass (066 D1), which on the Document tab's every frame would compile
    the whole graph.
    """
    # The stored pair. This list is drawn only under Fixed (revision 1), where the pair IS the
    # live canvas, so "the current size" has exactly one meaning here.
    current = ui_document.document.resolution
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
            if not isinstance(value, MediaWithTexture):
                continue
            size = value.texture.size
            if size in seen:
                continue
            seen.add(size)
            presets.append((get_resolution_str(uniform_name, *size), size))

    return presets


def draw(app: App) -> None:
    if not (ui_document := app.ui_documents.get(app.current_document_id)):
        return

    imgui.spacing()

    combo_offset = SIZE.NAME_INPUT_W + SPACE.XL

    imgui.begin_disabled(app.copilot_turn_active)

    # Reset goes on the caption row at the panel's right border (079 D7), where a document-wide
    # destructive verb reads as document-wide rather than as another canvas field.
    small_caption(app.font_12, "Document name")
    imgui.same_line(combo_offset)
    # The two modes edit different things, so the caption names the one on screen: a pixel
    # pair under Fixed, a ratio under Auto.
    small_caption(
        app.font_12,
        "Canvas"
        if ui_document.document.resolution_mode is ResolutionMode.FIXED
        else "Aspect",
    )
    _draw_document_reset(app)

    imgui.set_next_item_width(SIZE.NAME_INPUT_W)
    ui_document.ui_state.ui_name = imgui.input_text_with_hint(
        "##document_name", "document name", ui_document.ui_state.ui_name
    )[1]

    imgui.same_line(combo_offset)

    # Per-document widget ids. Clearing the editing flags on a switch is not enough on its own:
    # imgui keeps the ITEM active across it, so a shared `##canvas_w` would let the outgoing
    # document's half-typed digit re-latch onto the incoming one and commit to it.
    imgui.push_id(ui_document.id)

    _draw_resolution_mode(app, ui_document)
    imgui.same_line(spacing=float(SPACE.MD))
    control_x = imgui.get_cursor_pos_x()

    if ui_document.document.resolution_mode is ResolutionMode.FIXED:
        _draw_canvas_fields(app, ui_document)
    else:
        _draw_aspect_control(app, ui_document)

    _draw_size_readout(app, ui_document, control_x)

    imgui.pop_id()

    imgui.end_disabled()

    imgui.dummy((0, SPACE.MD))
    _draw_entry_points(app)


# The Shader/Script label column: a tick gutter + the widest label, so both `open` buttons align.
# The accent tick sits this far LEFT of the label, in the panel's own margin.
_ENTRY_TICK_W = float(SPACE.SM)


def _entry_row_label(active: bool, label: str) -> None:
    # The Script row's label. `align_text_to_frame_padding` centres the text on the button's row
    # height (the font mix floats it high otherwise). The accent tick marking the editor's active
    # tab is a draw-list line — presence and color only, never size (/imgui-ui §3) — drawn in the
    # margin to the LEFT of the text, so it costs the row no indent.
    imgui.align_text_to_frame_padding()
    pos = imgui.get_cursor_screen_pos()
    if active:
        h = imgui.get_frame_height()
        col = imgui.color_convert_float4_to_u32(COLOR.ACCENT_PRIMARY)
        imgui.get_window_draw_list().add_line(
            (pos.x - _ENTRY_TICK_W, pos.y + 2.0),
            (pos.x - _ENTRY_TICK_W, pos.y + h - 2.0),
            col,
            2.0,
        )
    imgui.text_colored(COLOR.FG_DIM, label)
    imgui.same_line(spacing=float(SPACE.MD))


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

    imgui.begin_disabled(app.copilot_turn_active)

    # ONE row, no section caption: a document has exactly one script (048), so a heading over a
    # single control said the word twice and cost a line the panel could not spare.
    _entry_row_label(script_active, "Script")
    open_tooltip = (
        "Open the document script" if present else "Create the document script"
    )
    open_color = COLOR.STATE_ERROR if error else COLOR.FG_SECONDARY
    if standard_button("open##entry_script", text_color=open_color):
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

    imgui.dummy((0, float(SPACE.MD)))
    _draw_passes(app, document_id)


def _draw_passes(app: App, document_id: str) -> None:
    # The two views of the same passes (092 D2): the caption row carries the choice, the
    # body is the strip's tiles or the graph canvas, and the add / import row sits under
    # both -- inside its own copilot-turn bracket, since the strip's used to carry it.
    imgui.begin_disabled(app.copilot_turn_active)
    small_caption(app.font_12, "Passes")
    imgui.same_line(spacing=float(SPACE.LG))
    views = list(PassesView)
    current = views.index(app.app_state.passes_view)
    chosen = segmented_choice(
        "##passes_view", [PASSES_VIEW_LABELS[v] for v in views], current
    )
    if chosen != current:
        app.app_state.passes_view = views[chosen]
    imgui.end_disabled()
    if app.app_state.passes_view is PassesView.GRAPH:
        pass_graph.draw(app, document_id)
    else:
        pass_list.draw(app, document_id)
    imgui.begin_disabled(app.copilot_turn_active)
    imgui.dummy((0, float(SPACE.SM)))
    if standard_button("add pass"):
        app.open_add_pass()
    imgui.same_line()
    if standard_button("import..."):
        app.open_import_passes()
    imgui.end_disabled()
