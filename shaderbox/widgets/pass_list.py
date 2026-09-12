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
from shaderbox.pass_graph import (
    PassEntry,
    Wiring,
    evaluation_order,
    group_runs,
    strip_order,
)
from shaderbox.theme import COLOR, SIZE, SPACE, group_tint
from shaderbox.ui_primitives import (
    context_menu_style,
    preview_cell,
    tune_icon_button,
)

FEEDBACK_CHIP = "prev"
# The group outline sits this far inside the run's tiles: a full row can end flush with the
# panel's edge, so a rect outside the tiles would clip.
_GROUP_INSET = 1.0
_GROUP_ROUNDING = 6.0
_GROUP_LABEL_PAD = 4.0


def tiles_per_row(avail: float, tile: float, gap: float) -> int:
    """How many `tile`-wide cells fit in `avail` with `gap` between them.

    The last tile is charged no trailing gap, so a row of the returned `n` spans
    `n * tile + (n - 1) * gap <= avail` — the strip cannot cross the panel's right edge.
    One tile always fits: a narrower panel clips rather than draws nothing.
    """
    return max(1, int((avail + gap) // (tile + gap)))


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


def pass_menu_items(app: App, document_id: str, name: str) -> None:
    """The items of one pass's context menu, shared by the strip's tile and the graph's node
    (092 D10) so the two surfaces cannot drift. The caller owns the popup itself: the strip
    anchors it with an explicit id (safe there, each tile is its own window), the canvas with
    the previous item."""
    document = app.ui_documents[document_id].document
    if imgui.menu_item_simple("Settings"):
        app.open_pass_settings(name)
    # Gated in Python, not by `enabled=`: menu_item_simple can still register a click
    # while disabled on this imgui-bundle build (/imgui-ui §7.4).
    deletable = len(document.passes) > 1
    if imgui.menu_item_simple("Delete", enabled=deletable) and deletable:
        _delete_pass(app, document_id, name)
    if document.graph.passes.get(name, PassEntry()).group and imgui.menu_item_simple(
        "Leave group"
    ):
        error = app.session.set_pass_group(document_id, name, "")
        if error:
            app.notifications.push(error)


def _draw_context_menu(app: App, document_id: str, name: str) -> None:
    with context_menu_style():
        if imgui.begin_popup_context_item(f"##pass_menu_{name}"):
            pass_menu_items(app, document_id, name)
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
    group: str,
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
    # A grouped tile sits inside the run's outline (091 D7): its own border goes and the
    # group's tint fills it faintly; an accent or error border still draws.
    tint = group_tint(group) if group else None
    bg = (*tint[:3], COLOR.GROUP_FILL_ALPHA) if tint is not None else None

    def _settings_overlay(side: float) -> None:
        if tune_icon_button(f"settings_{name}", side):
            app.open_pass_settings(name)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Pass settings")

    result = preview_cell(
        id_=f"pass_{name}",
        cell_w=float(SIZE.PASS_TILE),
        texture_glo=render_pass.canvas.texture.glo,
        texture_size=render_pass.canvas.texture.size,
        selected=is_output,
        armed=app.pass_delete_armed == name,
        border_color=border,
        bg_color=bg,
        bordered=tint is None,
        footer=name,
        footer_font=None if stale else app.font_14_bold,
        footer_color=COLOR.FG_DORMANT if stale else COLOR.FG_TITLE,
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
    """The pass strip for one document: the tiles alone. The caption, the view toggle and
    the add / import row are the Document tab's (092 D2), shared with the graph view."""
    ui_document = app.ui_documents.get(document_id)
    if ui_document is None:
        return
    document = ui_document.document

    imgui.begin_disabled(app.copilot_turn_active)

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
    per_row = tiles_per_row(avail, float(SIZE.PASS_TILE), float(SPACE.MD))
    order = strip_order(document.passes, wiring)
    groups = {
        name: document.graph.passes.get(name, PassEntry()).group for name in order
    }
    # A run is a group's consecutive tiles; one that wraps is outlined per row segment.
    segments: list[tuple[str, imgui.ImVec2, imgui.ImVec2]] = []
    open_segment: tuple[str, imgui.ImVec2, imgui.ImVec2] | None = None
    i = 0
    for run in group_runs(order, groups):
        for name in run:
            if i % per_row:
                imgui.same_line(spacing=float(SPACE.MD))
            elif open_segment is not None:
                segments.append(open_segment)
                open_segment = None
            _draw_pass_tile(
                app,
                document_id,
                name,
                document.passes[name],
                name not in live,
                _reads(name, wiring, order),
                groups[name],
            )
            if groups[name]:
                lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                open_segment = (
                    (groups[name], lo, hi)
                    if open_segment is None
                    else (open_segment[0], open_segment[1], hi)
                )
            i += 1
        if open_segment is not None:
            segments.append(open_segment)
            open_segment = None
    for group, lo, hi in segments:
        _draw_group_outline(app, group, lo, hi)
    imgui.end_disabled()


def _draw_group_outline(
    app: App, group: str, lo: imgui.ImVec2, hi: imgui.ImVec2
) -> None:
    """One run's outline and label (091 D7). The tiles are child windows, which paint over
    their parent, so both go on the foreground list, clipped to the strip's own window; the
    faint fill behind the tiles goes on the parent's list, where the gaps show it."""
    tint = group_tint(group)
    fill = imgui.color_convert_float4_to_u32((*tint[:3], COLOR.GROUP_FILL_ALPHA))
    line = imgui.color_convert_float4_to_u32(tint)
    parent = imgui.get_window_draw_list()
    parent.add_rect_filled((lo.x, lo.y), (hi.x, hi.y), fill, _GROUP_ROUNDING)
    fg = imgui.get_foreground_draw_list()
    fg.push_clip_rect(parent.get_clip_rect_min(), parent.get_clip_rect_max(), True)
    fg.add_rect(
        (lo.x + _GROUP_INSET, lo.y + _GROUP_INSET),
        (hi.x - _GROUP_INSET, hi.y - _GROUP_INSET),
        line,
        _GROUP_ROUNDING,
    )
    imgui.push_font(app.font_12, app.font_12.legacy_size)
    text_size = imgui.calc_text_size(group)
    x = lo.x + 2.0 * _GROUP_LABEL_PAD + _GROUP_ROUNDING
    y = lo.y - text_size.y / 2.0
    fg.add_rect_filled(
        (x - _GROUP_LABEL_PAD, y),
        (x + text_size.x + _GROUP_LABEL_PAD, y + text_size.y),
        imgui.color_convert_float4_to_u32(COLOR.BG_SURFACE),
    )
    fg.add_text((x, y), line, group)
    imgui.pop_font()
    fg.pop_clip_rect()
