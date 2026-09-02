"""The pass strip: one live thumbnail per pass, with the six verbs of D15 reachable from it (065).

The graph is edited as a LIST, not a canvas — a spatial editor is a separate feature. Click a tile
to open that pass in the editor; its wiring and target live in the pass-settings modal
(`popups/pass_settings.py`), reached from the tile's gear, its context menu, or automatically on
`add pass` — set-up-once choices don't get an always-open block of panel space.
"""

from collections.abc import Callable, Iterable

from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.core import Pass
from shaderbox.pass_graph import PassEntry, PassGraph, evaluation_order, plan_passes
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_primitives import (
    context_menu_style,
    ghost_button,
    preview_cell,
    small_caption,
    tune_icon_button,
)

_TICK_W = float(SPACE.MD)


def _strip_order(names: Iterable[str], graph: PassGraph) -> list[str]:
    """The strip's tile order: producers left of consumers, STABLE across output changes.

    `plan_passes` gives the deterministic topological order and never looks at the output, so
    picking a different output cannot shuffle the tiles. Passes it leaves out (cycle members,
    passes with no graph entry) are appended by name so every pass still gets a tile.
    """
    known = set(names)
    order = [n for n in plan_passes(graph)[0].order if n in known]
    return order + sorted(known - set(order))


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
    open_pass: Callable[[str], None],
    stale: bool,
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
            imgui.set_tooltip("Pass settings — what it reads, what it draws into")

    # The wiring reads INSIDE the card, under the title — not as a hover tooltip. A tile-wide
    # tooltip fires whenever the pointer is anywhere over the tile, including on the gear and
    # the delete-X, so it covered the tooltips those buttons wanted to show.
    wired = document.graph.passes.get(name, PassEntry()).inputs
    sublines = [f"{uniform} <- {src}" for uniform, src in sorted(wired.items())]
    if errors:
        sublines.append("has compile errors")

    result = preview_cell(
        id_=f"pass_{name}",
        cell_w=float(SIZE.PASS_THUMB),
        texture_glo=render_pass.canvas.texture.glo,
        texture_size=render_pass.canvas.texture.size,
        selected=is_output,
        armed=app.pass_delete_armed == name,
        border_color=border,
        footer=name,
        sublines=sublines,
        overlay=_settings_overlay,
        stale=stale,
    )
    _draw_context_menu(app, document_id, name)

    if result.clicked:
        # Picking a tile IS setting the output: the viewer and export follow the graph output,
        # so one click fully switches what the document shows. The editor tab comes along.
        open_pass(name)
        if not is_output:
            error = app.session.set_output_pass(document_id, name)
            if error:
                app.notifications.push(error)
    if result.delete_armed:
        app.pass_delete_armed = name
    elif result.delete_confirmed:
        _delete_pass(app, document_id, name)
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
    # Passes the current output does not need never render — their tiles dim and take the
    # stale corner tick so a frozen picture cannot be read as a live one.
    # `or {output}` mirrors the renderer's cycle fallback: with no plannable order it still
    # draws the output alone, so the output tile must not dim.
    output = document.graph.output
    live = (
        set(evaluation_order(document.graph, output)) or {output}
        if output in document.passes
        else set(document.passes)
    )
    avail = imgui.get_content_region_avail().x
    step = float(SIZE.PASS_THUMB) + float(SPACE.MD)
    per_row = max(1, int(avail // step))
    for i, name in enumerate(_strip_order(document.passes, document.graph)):
        if i % per_row:
            imgui.same_line(spacing=float(SPACE.MD))
        _draw_pass_tile(
            app, document_id, name, document.passes[name], open_pass, name not in live
        )

    imgui.dummy((0, float(SPACE.SM)))
    if ghost_button("add pass"):
        app.pass_add.open(app.session.paths.passes_dir_for(document_id))
    if app.pass_add.is_open:
        _draw_add_input(app, document_id, open_pass)
    imgui.end_disabled()


def _draw_add_input(
    app: App, document_id: str, open_pass: Callable[[str], None]
) -> None:
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
    open_pass(name)
    output_error = app.session.set_output_pass(document_id, name)
    if output_error:
        app.notifications.push(output_error)
    app.open_pass_settings(name)
