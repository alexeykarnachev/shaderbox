"""The pass list: a document's passes, what fills each one's inputs, and each one's target (065).

The graph is edited as a LIST, not a canvas — a spatial editor is a separate feature. One row per
pass, with the six verbs of D15 reachable from it: add, delete, rename, set output, wire, unwire.

Wiring is a closed-set combo over the document's own pass names, never a free-text field, so an
input can never name a pass that does not exist. That is what makes SHADERed's positional-slot
footgun impossible here, and it is why an unfilled input reading black (D3) stays a state you are
building toward rather than a typo you cannot see.
"""

from collections.abc import Callable

from imgui_bundle import imgui
from OpenGL.GL import GL_SAMPLER_2D

from shaderbox.app import App
from shaderbox.core import Pass
from shaderbox.pass_graph import PassEntry
from shaderbox.paths import pass_name_of
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_primitives import (
    context_menu_style,
    ghost_button,
    label_row,
    preview_cell,
    small_caption,
)

_TICK_W = float(SPACE.MD)
_LABEL_W = 96.0
_UNWIRED = "(none)"


def _sampler_names(render_pass: Pass) -> list[str]:
    return [
        u.name
        for u in render_pass.get_active_uniforms()
        if getattr(u, "gl_type", None) == GL_SAMPLER_2D
    ]


def _row_label(active: bool, is_output: bool, name: str) -> None:
    # An inset accent tick marks the editor's active tab (presence and colour only, never size —
    # /imgui-ui §3), then the name in a fixed column so the row's buttons line up.
    imgui.align_text_to_frame_padding()
    pos = imgui.get_cursor_screen_pos()
    if active:
        height = imgui.get_frame_height()
        col = imgui.color_convert_float4_to_u32(COLOR.ACCENT_PRIMARY)
        imgui.get_window_draw_list().add_line(
            (pos.x, pos.y + 2.0), (pos.x, pos.y + height - 2.0), col, 2.0
        )
    imgui.dummy((_TICK_W, 0))
    imgui.same_line()
    color = COLOR.ACCENT_PRIMARY if is_output else COLOR.FG_DIM
    imgui.text_colored(color, name)
    imgui.same_line(_TICK_W + _LABEL_W)


# What a pass's target format MEANS, in place of moderngl's `f1`/`f2`/`f4` dtype strings. The
# tuple is (menu label, what it is for) — the label is what a person picks from, the second half
# is the tooltip, because "16-bit" alone does not tell you when to want it.
_FORMATS: list[tuple[str, str, str]] = [
    (
        "f1",
        "8-bit",
        "Smallest. Values clamp to 0-1 — the right choice for a final image.",
    ),
    (
        "f2",
        "16-bit float",
        "Holds values above 1 (bright highlights, accumulated light). The default, and what "
        "bloom and feedback need.",
    ),
    (
        "f4",
        "32-bit float",
        "Full precision. Rarely needed; costs twice the memory of 16-bit.",
    ),
]
_FORMAT_LABELS = [label for _, label, _ in _FORMATS]
_FORMAT_CODES = [code for code, _, _ in _FORMATS]

_ROW_LABEL_W = 78.0
_CTRL_W = 168.0


def _help(text: str) -> None:
    """A dim `?` after a control, explaining it on hover — the room a caption does not have."""
    imgui.same_line(spacing=float(SPACE.SM))
    imgui.text_colored(COLOR.FG_DIM, "?")
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)


def _draw_inputs(app: App, document_id: str, name: str, render_pass: Pass) -> None:
    # One closed-set combo per sampler uniform the pass actually declares. A pass with no samplers
    # draws nothing — the absence of a row is the honest signal that it reads no other pass.
    samplers = _sampler_names(render_pass)
    if not samplers:
        return
    document = app.ui_documents[document_id].document
    entry = document.graph.passes.get(name, PassEntry())
    choices = [_UNWIRED, *sorted(document.passes)]
    small_caption(app.font_12, "Reads")
    for uniform in samplers:
        current = entry.inputs.get(uniform, "")
        index = choices.index(current) if current in choices else 0
        label_row(app.font_12, uniform, _CTRL_W, _ROW_LABEL_W)
        changed, picked = imgui.combo(f"##wire_{name}_{uniform}", index, choices)
        if changed:
            producer = "" if picked == 0 else choices[picked]
            error = app.session.wire_pass_input(document_id, name, uniform, producer)
            if error:
                app.notifications.push(error)
        _help(
            f"Which pass fills this pass's `{uniform}` sampler. "
            f"{_UNWIRED} leaves it black."
        )


def _draw_target(app: App, document_id: str, name: str) -> None:
    # What this pass DRAWS INTO. Every control names the thing it changes about the picture, not
    # the field it writes: a target's dtype is "format / 16-bit float", not "f2".
    document = app.ui_documents[document_id].document
    entry = document.graph.passes.get(name, PassEntry())
    target = entry.target
    new_target = target
    is_output = name == document.graph.output

    small_caption(app.font_12, "Draws into")

    label_row(app.font_12, "format", _CTRL_W, _ROW_LABEL_W)
    changed, picked = imgui.combo(
        f"##dtype_{name}", _FORMAT_CODES.index(target.dtype), _FORMAT_LABELS
    )
    if changed:
        new_target = target.model_copy(update={"dtype": _FORMAT_CODES[picked]})
    _help(_FORMATS[_FORMAT_CODES.index(target.dtype)][2])

    label_row(app.font_12, "size", _CTRL_W, _ROW_LABEL_W)
    canvas_w, canvas_h = document.canvas_size
    scale_changed, scale = imgui.slider_float(
        f"##scale_{name}",
        target.scale,
        0.05,
        1.0,
        f"%.0f%% of canvas  ({max(1, round(canvas_w * target.scale))}"
        f"x{max(1, round(canvas_h * target.scale))})",
    )
    if scale_changed:
        new_target = new_target.model_copy(update={"scale": scale})
    _help(
        "How big this pass's own image is, relative to the canvas. Half size is a quarter of "
        "the pixels — the usual choice for a blur, which looks the same and costs less. The "
        "output pass always draws at full size."
        if is_output
        else "How big this pass's own image is, relative to the canvas. Half size is a quarter "
        "of the pixels — the usual choice for a blur, which looks the same and costs less."
    )

    label_row(app.font_12, "sampling", _CTRL_W, _ROW_LABEL_W)
    smooth_changed, smooth = imgui.checkbox(f"smooth##{name}", target.filter_linear)
    if smooth_changed:
        new_target = new_target.model_copy(update={"filter_linear": smooth})
    _help(
        "How another pass reads this one BETWEEN pixels: smooth blends neighbours (right for "
        "a blur or an upscale), off gives hard pixel edges."
    )

    label_row(app.font_12, "edges", _CTRL_W, _ROW_LABEL_W)
    tile_changed, tile = imgui.checkbox(f"repeat##{name}", target.wrap)
    if tile_changed:
        new_target = new_target.model_copy(update={"wrap": tile})
    _help(
        "What a read PAST this pass's edge returns: repeat wraps to the far side (tiling), "
        "off clamps to the edge pixel — which is what a feedback trail wants."
    )

    if new_target != target:
        error = app.session.set_pass_target(document_id, name, new_target)
        if error:
            app.notifications.push(error)


def _draw_context_menu(app: App, document_id: str, name: str) -> None:
    document = app.ui_documents[document_id].document
    with context_menu_style():
        if imgui.begin_popup_context_item(f"##pass_menu_{name}"):
            if imgui.menu_item_simple("Rename"):
                app.pass_rename.open(
                    app.session.paths.pass_shader_for(document_id, name), name
                )
            if imgui.menu_item_simple("Set as output"):
                error = app.session.set_output_pass(document_id, name)
                if error:
                    app.notifications.push(error)
            # Gated in Python, not by `enabled=`: menu_item_simple can still register a click
            # while disabled on this imgui-bundle build (/imgui-ui §7.4).
            deletable = len(document.passes) > 1
            if imgui.menu_item_simple("Delete", enabled=deletable) and deletable:
                error = app.session.delete_pass(document_id, name)
                if error:
                    app.notifications.push(error)
            imgui.end_popup()


def _draw_rename_input(app: App, document_id: str) -> None:
    """The inline rename, drawn once below the strip for whichever pass the menu opened it on."""
    target = app.pass_rename.target
    if target is None:
        return
    name = pass_name_of(target)
    imgui.align_text_to_frame_padding()
    small_caption(app.font_12, f"rename {name}")
    imgui.same_line()
    cancel_w = imgui.calc_text_size("x").x + float(SPACE.MD) * 2.0
    imgui.set_next_item_width(imgui.get_content_region_avail().x - cancel_w)
    if app.pass_rename.needs_focus:
        imgui.set_keyboard_focus_here(0)
        app.pass_rename.needs_focus = False
    committed, app.pass_rename.buf = imgui.input_text(
        "##rename_pass",
        app.pass_rename.buf,
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    if committed:
        error = app.session.rename_pass(document_id, name, app.pass_rename.buf.strip())
        if error:
            app.notifications.push(error)
        else:
            app.pass_rename.close()
    # Esc cancels whenever the input is OPEN, not only while focused: a user who clicked away
    # would otherwise find Esc dead.
    if imgui.is_key_pressed(imgui.Key.escape, repeat=False):
        app.pass_rename.close()
    imgui.same_line()
    if ghost_button("x##cancel_rename"):
        app.pass_rename.close()


def _draw_pass_tile(
    app: App,
    document_id: str,
    name: str,
    render_pass: Pass,
    open_pass: Callable[[str], None],
) -> None:
    # The pass's OWN live target, scaled down by imgui — not a second render at thumbnail size.
    # Every pass already draws once per frame into that texture, so the tile costs nothing but the
    # blit; rendering a separate small frame would double the document's per-frame draw count.
    document = app.ui_documents[document_id].document
    is_output = name == document.graph.output
    active = app.active_tab
    is_open = active is not None and active.path == render_pass.source.path
    errors = bool(render_pass.compile_unit.errors)

    border = (
        COLOR.STATE_ERROR
        if errors
        else COLOR.ACCENT_PRIMARY
        if is_output
        else COLOR.ACCENT_ACTIVE
        if is_open
        else None
    )
    result = preview_cell(
        id_=f"pass_{name}",
        cell_w=float(SIZE.PASS_THUMB),
        texture_glo=render_pass.canvas.texture.glo,
        texture_size=render_pass.canvas.texture.size,
        selected=is_open,
        armed=app.pass_delete_armed == name,
        border_color=border,
        footer=f"{name} *" if is_output else name,
    )
    if imgui.is_item_hovered():
        wired = document.graph.passes.get(name, PassEntry()).inputs
        lines = [f"{name}{' — the output' if is_output else ''}"]
        lines += [f"{uniform} <- {src}" for uniform, src in sorted(wired.items())]
        if errors:
            lines.append("has compile errors")
        lines.append("click to open; right-click for actions")
        imgui.set_tooltip("\n".join(lines))
    _draw_context_menu(app, document_id, name)

    if result.clicked:
        open_pass(name)
        app.pass_expanded = "" if app.pass_expanded == name else name
    if result.delete_armed:
        app.pass_delete_armed = name
    elif result.delete_confirmed:
        error = app.session.delete_pass(document_id, name)
        if error:
            app.notifications.push(error)
        app.pass_delete_armed = ""
    elif result.delete_cancelled:
        app.pass_delete_armed = ""


def draw(app: App, document_id: str, open_pass: Callable[[str], None]) -> None:
    """The pass strip for one document. `open_pass` summons a pass's tab into the editor."""
    ui_document = app.ui_documents.get(document_id)
    if ui_document is None:
        return
    document = ui_document.document

    imgui.begin_disabled(app.copilot_turn_active)
    small_caption(app.font_12, "Passes")

    # Horizontal, wrapping at the panel edge: a document's passes are a handful, and a column of
    # full-width rows spent the panel's vertical budget on a list that reads better as a strip.
    avail = imgui.get_content_region_avail().x
    step = float(SIZE.PASS_THUMB) + float(SPACE.MD)
    per_row = max(1, int(avail // step))
    for i, name in enumerate(sorted(document.passes)):
        if i % per_row:
            imgui.same_line(spacing=float(SPACE.MD))
        _draw_pass_tile(app, document_id, name, document.passes[name], open_pass)

    imgui.dummy((0, float(SPACE.SM)))
    if app.pass_rename.is_open:
        _draw_rename_input(app, document_id)
    if ghost_button("add pass"):
        app.pass_add.open(app.session.paths.passes_dir_for(document_id))
    if app.pass_add.is_open:
        _draw_add_input(app, document_id)

    # The selected tile's wiring + target, below the strip: one expanded block, not one per row.
    if app.pass_expanded in document.passes:
        imgui.dummy((0, float(SPACE.SM)))
        small_caption(app.font_12, app.pass_expanded)
        _draw_inputs(
            app, document_id, app.pass_expanded, document.passes[app.pass_expanded]
        )
        _draw_target(app, document_id, app.pass_expanded)
    imgui.end_disabled()


def _draw_add_input(app: App, document_id: str) -> None:
    imgui.dummy((_TICK_W, 0))
    imgui.same_line()
    cancel_w = imgui.calc_text_size("x").x + float(SPACE.MD) * 2.0
    imgui.set_next_item_width(imgui.get_content_region_avail().x - cancel_w)
    if app.pass_add.needs_focus:
        imgui.set_keyboard_focus_here(0)
        app.pass_add.needs_focus = False
    committed, app.pass_add.buf = imgui.input_text(
        "##add_pass",
        app.pass_add.buf,
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    if committed:
        error = app.session.add_pass(document_id, app.pass_add.buf.strip())
        if error:
            app.notifications.push(error)
        else:
            app.pass_add.close()
    if imgui.is_key_pressed(imgui.Key.escape, repeat=False):
        app.pass_add.close()
    imgui.same_line()
    if ghost_button("x##cancel_add_pass"):
        app.pass_add.close()
