"""The pass strip: one live thumbnail per pass, with the six verbs of D15 reachable from it (065).

The graph is edited as a LIST, not a canvas -- 070 rejected a spatial view. Click a tile to open
that pass in the editor; its name, run count and target live in the pass-settings modal
(`popups/pass_settings.py`), reached from the tile's gear, its context menu, or automatically on
`add pass` -- set-up-once choices don't get an always-open block of panel space.

A tile is a picture, a name, and a row of chips naming the passes it reads (`prev` for its own
previous frame). Compile errors show as a red border rather than as text; a sampler's source is
chosen on its own row of the uniforms panel (072).
"""

from collections.abc import Sequence

from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.core import Pass
from shaderbox.pass_graph import Wiring, evaluation_order, strip_order
from shaderbox.theme import COLOR, SIZE, SPACE, fade
from shaderbox.ui_primitives import (
    context_menu_style,
    ghost_button,
    preview_cell,
    small_caption,
    tune_icon_button,
)

_TICK_W = float(SPACE.MD)


FEEDBACK_CHIP = "prev"


def _reads(name: str, wiring: Wiring, order: Sequence[str]) -> list[str]:
    """The chips under a tile: each pass `name` reads, in strip order, then `prev` when it
    reads its own previous frame. One chip per source pass however many samplers read it.

    `wiring` is what the binder binds (`Document.effective_wiring`), so a source naming a
    pass that no longer exists, or a row for a sampler the program no longer declares, is
    no chip -- exactly as the panel row shows it reading black.
    """
    sources = set(wiring.get(name, {}).values())
    chips = [p for p in order if p in sources and p != name]
    if name in sources:
        chips.append(FEEDBACK_CHIP)
    return chips


def _draw_context_menu(app: App, document_id: str, name: str) -> None:
    document = app.ui_documents[document_id].document
    with context_menu_style():
        if imgui.begin_popup_context_item(f"##pass_menu_{name}"):
            if imgui.menu_item_simple("Settings"):
                app.open_pass_settings(name)
            # Gated in Python, not by `enabled=`: menu_item_simple can still register a click
            # while disabled on this imgui-bundle build (/imgui-ui §7.4).
            deletable = len(document.passes) > 1
            if imgui.menu_item_simple("Delete", enabled=deletable) and deletable:
                _delete_pass(app, document_id, name)
            imgui.end_popup()


def _delete_pass(app: App, document_id: str, name: str) -> None:
    # Capture the pass file's path BEFORE the core deletes it: the editor
    # session + tab for that file must go with the pass (the one eviction
    # path that had no teardown — the native handle leaked and the orphan
    # tab kept editing a file no pass owned).
    ui_document = app.ui_documents.get(document_id)
    doomed = (
        ui_document.document.passes[name].source.path
        if ui_document is not None and name in ui_document.document.passes
        else None
    )
    error = app.session.delete_pass(document_id, name)
    if error:
        app.notifications.push(error)
        return
    if doomed is not None:
        app.close_editor_for_path(doomed)


def _draw_pass_tile(
    app: App,
    document_id: str,
    name: str,
    render_pass: Pass,
    stale: bool,
    reads: Sequence[str],
) -> None:
    # The pass's OWN live target, scaled down by imgui — not a second render at thumbnail size.
    # Every pass already draws once per frame into that texture, so the tile costs nothing but the
    # blit; rendering a separate small frame would double the document's per-frame draw count.
    document = app.ui_documents[document_id].document
    is_output = name == document.graph.output
    errors = bool(render_pass.compile_unit.errors)

    # ONE highlight, one meaning: the accent border is the picked (= output) pass. Error red
    # overrides it; an open editor tab gets no border of its own — the tab bar shows that.
    border = (
        COLOR.STATE_ERROR if errors else COLOR.ACCENT_PRIMARY if is_output else None
    )

    def _settings_overlay(side: float) -> None:
        if tune_icon_button(f"settings_{name}", side):
            app.open_pass_settings(name)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Pass settings")

    result = preview_cell(
        id_=f"pass_{name}",
        cell_w=float(SIZE.PASS_THUMB),
        texture_glo=render_pass.canvas.texture.glo,
        texture_size=render_pass.canvas.texture.size,
        selected=is_output,
        armed=app.pass_delete_armed == name,
        border_color=border,
        bg_color=None if stale else fade(COLOR.ACCENT_PRIMARY, COLOR.ACCENT_TINT_ALPHA),
        footer=name,
        overlay=_settings_overlay,
        stale=stale,
        chips=reads,
        chip_font=app.font_12,
    )
    _draw_context_menu(app, document_id, name)

    if result.clicked:
        # Picking a tile IS setting the output: the viewer and export follow the graph output,
        # so one click fully switches what the document shows. The editor tab comes along.
        app.pick_pass(document_id, name, focus_editor=False)
    if result.delete_armed:
        app.pass_delete_armed = name
    elif result.delete_confirmed:
        _delete_pass(app, document_id, name)
        app.pass_delete_armed = ""
    elif result.delete_cancelled:
        app.pass_delete_armed = ""


def draw(app: App, document_id: str) -> None:
    """The pass strip for one document."""
    ui_document = app.ui_documents.get(document_id)
    if ui_document is None:
        return
    document = ui_document.document

    imgui.begin_disabled(app.copilot_turn_active)
    small_caption(app.font_12, "Passes")

    # Horizontal, wrapping at the panel edge: a document's passes are a handful, and a column of
    # full-width rows spent the panel's vertical budget on a list that reads better as a strip.
    # Passes the current output does not need never render — their tiles dim and take the
    # stale corner tick so a frozen picture cannot be read as a live one.
    # `or {output}` mirrors the renderer's cycle fallback: with no plannable order it still
    # draws the output alone, so the output tile must not dim.
    output = document.graph.output
    # The EFFECTIVE wiring: a pass wired only by its uniform's name (069 D9) has no stored row,
    # so planning the rows alone would wash live ancestors grey and hand the strip sorted-name
    # order instead of a topological one.
    wiring = document.effective_wiring()
    live = (
        set(evaluation_order(wiring, output)) or {output}
        if output in document.passes
        else set(document.passes)
    )
    avail = imgui.get_content_region_avail().x
    step = float(SIZE.PASS_THUMB) + float(SPACE.MD)
    per_row = max(1, int(avail // step))
    order = strip_order(document.passes, wiring)
    for i, name in enumerate(order):
        if i % per_row:
            imgui.same_line(spacing=float(SPACE.MD))
        _draw_pass_tile(
            app,
            document_id,
            name,
            document.passes[name],
            name not in live,
            _reads(name, wiring, order),
        )

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
    # Read on the line after the input: the item-scoped queries see the LAST submitted item.
    wants_commit = committed or imgui.is_item_deactivated_after_edit()
    name = app.pass_add.buf.strip()
    if imgui.is_key_pressed(imgui.Key.escape, repeat=False):
        app.pass_add.close()
        wants_commit = False
    imgui.same_line()
    if ghost_button("x##cancel_add_pass"):
        # The cancel click IS what deactivated the input, so the commit it raised is the pass
        # the user just cancelled.
        app.pass_add.close()
        wants_commit = False
    if not wants_commit:
        return
    error = app.session.add_pass(document_id, name)
    if error:
        app.notifications.push(error)
        return
    app.pass_add.close()
    # A new pass is what the document shows: the editor tab, the viewer and the gear all follow
    # it, and its wiring and target are chosen NOW or never.
    app.pick_pass(document_id, name, focus_editor=False)
    app.open_pass_settings(name)
