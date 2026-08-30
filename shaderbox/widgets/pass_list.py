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
from shaderbox.pass_graph import DTYPES, PassEntry
from shaderbox.theme import COLOR, SPACE
from shaderbox.ui_primitives import (
    context_menu_style,
    ghost_button,
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


def _draw_inputs(app: App, document_id: str, name: str, render_pass: Pass) -> None:
    # One closed-set combo per sampler uniform the pass actually declares. A pass with no samplers
    # draws nothing — the absence of a row is the honest signal that it reads no other pass.
    samplers = _sampler_names(render_pass)
    if not samplers:
        return
    document = app.ui_documents[document_id].document
    entry = document.graph.passes.get(name, PassEntry())
    choices = [_UNWIRED, *sorted(document.passes)]
    imgui.indent(_TICK_W + _LABEL_W)
    for uniform in samplers:
        current = entry.inputs.get(uniform, "")
        index = choices.index(current) if current in choices else 0
        imgui.align_text_to_frame_padding()
        small_caption(app.font_12, uniform)
        imgui.same_line(_LABEL_W)
        imgui.set_next_item_width(imgui.get_content_region_avail().x)
        changed, picked = imgui.combo(f"##wire_{name}_{uniform}", index, choices)
        if changed:
            producer = "" if picked == 0 else choices[picked]
            error = app.session.wire_pass_input(document_id, name, uniform, producer)
            if error:
                app.notifications.push(error)
    imgui.unindent(_TICK_W + _LABEL_W)


def _draw_target(app: App, document_id: str, name: str) -> None:
    document = app.ui_documents[document_id].document
    entry = document.graph.passes.get(name, PassEntry())
    target = entry.target
    imgui.indent(_TICK_W + _LABEL_W)

    imgui.align_text_to_frame_padding()
    small_caption(app.font_12, "format")
    imgui.same_line(_LABEL_W)
    imgui.set_next_item_width(imgui.get_content_region_avail().x)
    dtypes = list(DTYPES)
    changed, picked = imgui.combo(f"##dtype_{name}", dtypes.index(target.dtype), dtypes)
    new_target = target
    if changed:
        new_target = target.model_copy(update={"dtype": dtypes[picked]})

    imgui.align_text_to_frame_padding()
    small_caption(app.font_12, "scale")
    imgui.same_line(_LABEL_W)
    imgui.set_next_item_width(imgui.get_content_region_avail().x)
    scale_changed, scale = imgui.slider_float(
        f"##scale_{name}", target.scale, 0.05, 1.0, "%.2f"
    )
    if scale_changed:
        new_target = new_target.model_copy(update={"scale": scale})

    smooth_changed, smooth = imgui.checkbox(f"smooth##{name}", target.filter_linear)
    imgui.same_line(spacing=float(SPACE.LG))
    tile_changed, tile = imgui.checkbox(f"tile##{name}", target.wrap)
    if smooth_changed:
        new_target = new_target.model_copy(update={"filter_linear": smooth})
    if tile_changed:
        new_target = new_target.model_copy(update={"wrap": tile})

    if new_target != target:
        error = app.session.set_pass_target(document_id, name, new_target)
        if error:
            app.notifications.push(error)
    imgui.unindent(_TICK_W + _LABEL_W)


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


def _draw_rename_input(app: App, document_id: str, name: str) -> bool:
    """The inline rename for one row. Returns True when it drew (the row's label is replaced)."""
    path = app.session.paths.pass_shader_for(document_id, name)
    if app.pass_rename.target != path:
        return False
    imgui.dummy((_TICK_W, 0))
    imgui.same_line()
    cancel_w = imgui.calc_text_size("x").x + float(SPACE.MD) * 2.0
    imgui.set_next_item_width(imgui.get_content_region_avail().x - cancel_w)
    if app.pass_rename.needs_focus:
        imgui.set_keyboard_focus_here(0)
        app.pass_rename.needs_focus = False
    committed, app.pass_rename.buf = imgui.input_text(
        f"##rename_pass_{name}",
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
    if ghost_button(f"x##cancel_rename_{name}"):
        app.pass_rename.close()
    return True


def draw(app: App, document_id: str, open_pass: Callable[[str], None]) -> None:
    """The pass list for one document. `open_pass` summons a pass's tab into the editor."""
    ui_document = app.ui_documents.get(document_id)
    if ui_document is None:
        return
    document = ui_document.document
    active = app.active_tab

    imgui.begin_disabled(app.copilot_turn_active)
    small_caption(app.font_12, "Passes")
    for name in sorted(document.passes):
        render_pass = document.passes[name]
        imgui.push_id(f"pass_{name}")
        if not _draw_rename_input(app, document_id, name):
            is_active = active is not None and active.path == render_pass.source.path
            _row_label(is_active, name == document.graph.output, name)
            if ghost_button("open"):
                open_pass(name)
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"Open {name} in the editor")
            _draw_context_menu(app, document_id, name)
            if render_pass.compile_unit.errors:
                imgui.same_line()
                imgui.text_colored(COLOR.STATE_ERROR, "errors")
            if name == app.pass_expanded:
                _draw_inputs(app, document_id, name, render_pass)
                _draw_target(app, document_id, name)
        imgui.pop_id()

    imgui.dummy((0, float(SPACE.SM)))
    if ghost_button("add pass"):
        app.pass_add.open(app.session.paths.passes_dir_for(document_id))
    if app.pass_add.is_open:
        _draw_add_input(app, document_id)
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
