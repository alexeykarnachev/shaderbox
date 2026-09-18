"""The graph canvas (092, 093, 098): a document's passes as nodes, drawn by the library.

A node is a pass's live picture, its name and one input port per sampler its compiled program
declares (`pass_graph.node_ports`); a wire is a read from the effective wiring. Nothing here
is a second source of truth: positions are the pass entry's (or the rank layout's, for a pass
never placed), edges are the wiring, ports are the program.

The picture itself belongs to `graph_canvas` (098). Every frame the document is packed into
the library's node model, `gc_frame` lays it out and resolves the pointer, and the two vertex
streams are rendered into an FBO that `imgui.image` presents -- so the hit-testing, the
gestures, the pan and the zoom are the library's, not an imgui item tree's. imgui still owns
this child, the tab row and the menus.

What stays here is the seam: which passes are packed and how they look, and what each event
the library reports means in shaderbox's terms. Every write a gesture makes goes through an
`App` verb, never a session call from here, so each refusal is testable without a window --
and a drag, which the library reports every frame, is accumulated on the canvas state and
written ONCE on release.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.commands import CommandId
from shaderbox.document import Document
from shaderbox.engine_uniforms import ENGINE_DRIVEN_UNIFORMS
from shaderbox.graph_canvas.adapter import (
    Activated,
    Clicked,
    GraphEvent,
    MenuRequested,
    Moved,
    NodePalette,
    Refused,
    Unwired,
    Wired,
    pack_nodes,
)
from shaderbox.graph_canvas.panel import (
    GraphCanvasState,
    pointer_from_io,
    refusal_text,
    render_to_texture,
)
from shaderbox.graph_canvas.render import CanvasRenderer
from shaderbox.menus import command_menu_item
from shaderbox.pass_graph import (
    PassEntry,
    Wiring,
    rank_layout,
    strip_order,
)
from shaderbox.project_session import compile_pending_passes
from shaderbox.theme import COLOR, SIZE, fade
from shaderbox.ui_primitives import (
    context_menu_style,
    name_input_row,
    primary_button,
    text_tab_row,
)
from shaderbox.widgets.graph_state import (
    GraphViewState,
    group_names_in_order,
    node_sizes,
    ports_of,
    revalidated_scope,
)
from shaderbox.widgets.pass_list import pass_menu_items

# The root tab's label when the document's name is empty or collides with a group's; a
# numeric suffix is appended until no group carries it, so the label is always distinct.
_ROOT_LABEL = "document"
# ---- the derived picture ------------------------------------------------------------------


def _positions(
    document: Document,
    wiring: Wiring,
    groups: dict[str, str],
    sizes: dict[str, tuple[float, float]],
    overrides: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Place every pass on the canvas without writing anything.

    A stored position wins; a pass never placed takes the rank layout's (092 D6); a drag in
    flight overrides both through `overrides`.
    """
    entries = document.graph.passes
    placed = {
        name: entries[name].position
        for name in document.passes
        if name in entries and entries[name].position is not None
    }
    stored: dict[str, tuple[float, float]] = {
        name: (position[0], position[1])
        for name, position in placed.items()
        if position is not None
    }
    unplaced = [name for name in document.passes if name not in stored]
    laid = rank_layout(
        wiring,
        unplaced,
        groups,
        sizes,
        stored,
        float(SIZE.GRAPH_GAP_X),
        float(SIZE.GRAPH_GAP_Y),
    )
    return {**stored, **laid, **overrides}


# ---- the widget ----------------------------------------------------------------------------


def _tab_row(
    app: App, document_id: str, view: GraphViewState, groups: list[str]
) -> None:
    ui_document = app.ui_documents[document_id]
    # The row keys and answers by NAME, so the root's label is made distinct from every
    # group's before it is drawn; the click then maps back without ambiguity.
    root_label = ui_document.ui_state.ui_name.strip()
    if not root_label or root_label in groups:
        root_label = _ROOT_LABEL
        n = 1
        while root_label in groups:
            root_label = f"{_ROOT_LABEL}_{n}"
            n += 1
    labels = [root_label, *groups]
    scopes = ["", *groups]
    active = labels[scopes.index(view.scope)] if view.scope in scopes else root_label
    clicked = text_tab_row("graph_scope", labels, active)
    if clicked is not None:
        index = labels.index(clicked)
        if scopes[index] != view.scope:
            view.scope = scopes[index]
            view.fitted = False


def _group_prompt(app: App, document_id: str, view: GraphViewState) -> None:
    """The name a Group asks for (092 D14): a small popup, the shared name row, Create."""
    if view.group_input.needs_focus and not imgui.is_popup_open("##graph_group"):
        imgui.open_popup("##graph_group")
    if not imgui.begin_popup("##graph_group"):
        # A click outside dismisses the popup without reaching either commit branch, so the
        # input is closed here or `is_open` reports a prompt that is no longer on screen.
        # `begin_popup` returns True on the frame `open_popup` ran, so this cannot fire on
        # the opening frame.
        if view.group_input.is_open:
            view.group_input.close()
        return
    result = name_input_row(
        "graph_group_name", view.group_input, width=float(SIZE.NAME_INPUT_W)
    )
    imgui.same_line()
    committed = primary_button("Create") or result.committed
    name = view.group_input.buf.strip()
    # A blank name would mean "no group" to the verb, which is Dissolve, not Create.
    if (
        committed
        and name
        and view.selection
        and app.group_selection(document_id, name) == ""
    ) or result.cancelled:
        view.group_input.close()
        imgui.close_current_popup()
    imgui.end_popup()


def _box_menu_items(
    app: App, document_id: str, view: GraphViewState, group: str
) -> None:
    """A group box's own two verbs: open its tab, or dissolve it."""
    if imgui.menu_item_simple("Open"):
        view.scope = group
        view.fitted = False
    if imgui.menu_item_simple("Dissolve"):
        error = app.dissolve_group(document_id, group)
        if error:
            app.notifications.push(error)


def _group_item(app: App, document_id: str, view: GraphViewState, name: str) -> None:
    """`Group` on a pass node: selects it if nothing is, then opens the prompt."""
    if imgui.menu_item_simple("Group"):
        if not view.selection:
            view.selection = {name}
        view.group_input.open(Path(name), buf="")


def _node_menu(app: App, document_id: str, view: GraphViewState, name: str) -> None:
    """The context menu for the node the library says the pointer was over.

    A group box gets its own two verbs; a pass gets the same items its tile on
    the strip has, so the two surfaces never drift apart.
    """
    document = app.ui_documents[document_id].document
    entry = document.graph.passes.get(name)
    with context_menu_style():
        if imgui.begin_popup("##graph_node_menu"):
            if view.scope == "" and entry is not None and entry.group:
                _box_menu_items(app, document_id, view, entry.group)
            else:
                pass_menu_items(
                    app,
                    document_id,
                    name,
                    slot=lambda: _group_item(app, document_id, view, name),
                )
            imgui.end_popup()


def _canvas_menu(app: App, document_id: str, view: GraphViewState) -> None:
    with context_menu_style():
        if imgui.begin_popup("##graph_canvas_menu"):
            command_menu_item(app, CommandId.ADD_PASS)
            command_menu_item(app, CommandId.IMPORT_PASSES)
            imgui.separator()
            if imgui.menu_item_simple("Frame all"):
                view.fitted = False
            if imgui.menu_item_simple("Arrange"):
                app.arrange_graph(document_id)
            imgui.end_popup()


def _as_components(value: object) -> tuple[float, ...]:
    """An engine uniform's value as up to four floats, or empty.

    The engine writes a float for a scalar and a tuple for a vector, and the
    glyph tables are bytes that no row can show. Anything that is not a number
    or a short sequence of them reads as no value, which draws the name alone.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (float(value),)
    if isinstance(value, (tuple, list)) and 0 < len(value) <= 4:
        try:
            return tuple(float(v) for v in value)
        except (TypeError, ValueError):
            return ()
    return ()


def _library_canvas(
    app: App,
    document_id: str,
    document: Document,
    view: GraphViewState,
) -> None:
    """The canvas interior, drawn by the graph_canvas library (098).

    imgui still owns the child, the menus and the tab row; what it no longer
    owns is the picture. The library is handed the whole graph every frame and
    hands back geometry plus what the user did, so nothing about a node
    survives between calls and a rename or a delete needs no bookkeeping here.

    Every write still goes through an `App` verb, which is what keeps each
    refusal testable without a window.
    """
    origin = imgui.get_cursor_screen_pos()
    avail = imgui.get_content_region_avail()
    width = max(int(avail.x), 1)
    height = max(int(avail.y), 1)

    state = app.graph_canvas_for(document_id)
    state.origin = (origin.x, origin.y)
    if app.graph_renderer is None:
        app.graph_renderer = CanvasRenderer()

    wiring = document.effective_wiring()
    ports = ports_of(document, wiring)
    order = strip_order(document.passes, wiring)
    entries = document.graph.passes
    groups = {name: entries.get(name, PassEntry()).group for name in order}
    # A stored position wins, a pass never placed takes the rank layout's, and
    # a drag in flight overrides both -- the same rule the strip and the old
    # canvas used, so a pass does not jump when the renderer changed under it.
    positions = _positions(document, wiring, groups, node_sizes(ports), state.dragging)
    # Read every frame, and AFTER the document has rendered (`ui._update_and_draw`
    # renders before it draws). A pass recreates its canvas texture on a resize or a
    # recompile, so a name cached across frames is a live GL name that is no longer
    # this pass's -- the library takes a bare uint32 and cannot detect it, the frame
    # is accepted, and the node draws black with no error.
    previews: dict[str, tuple[int, int, int]] = {}
    for name in order:
        canvas = document.passes[name].canvas
        if canvas is None:
            continue
        texture = canvas.texture
        previews[name] = (texture.glo, texture.size[0], texture.size[1])

    # The engine's own uniforms (`u_time` and its siblings), shown as body rows
    # in the syntax colour the editor already gives them. They take no wire --
    # the engine writes them -- so they are controls rather than ports.
    engine: dict[str, list[tuple[str, tuple[float, ...]]]] = {}
    for name in order:
        render_pass = document.passes[name]
        declared = {u.name for u in render_pass.get_active_uniforms()}
        rows: list[tuple[str, tuple[float, ...]]] = []
        for uniform in sorted(ENGINE_DRIVEN_UNIFORMS):
            if uniform not in declared:
                continue
            # The LIVE value, which `Pass.render` writes back into
            # `uniform_values` every frame. A pass that has not rendered yet
            # has no entry and shows the name alone.
            rows.append(
                (uniform, _as_components(render_pass.uniform_values.get(uniform)))
            )
        engine[name] = rows

    packed = pack_nodes(
        order,
        ports,
        positions,
        previews,
        output=document.graph.output_pass or "",
        engine=engine,
        hovered=state.hovered,
        selected=frozenset(view.selection),
        palette=NodePalette(
            hover=COLOR.GRAPH_HOVER,
            select=COLOR.SELECT,
            engine_uniform=COLOR.SYN_UNIFORM,
        ),
    )

    # A gesture the canvas did not see the end of is CANCELLED, never resumed:
    # a stray later release must not commit a wire or a move.
    hovered = imgui.is_window_hovered(imgui.HoveredFlags_.child_windows)
    frozen = app.copilot_turn_active
    pointer = pointer_from_io(
        (origin.x, origin.y), hovered, cancelled=frozen, claimed=state.claimed
    )
    if not view.fitted:
        state.fitted = False
        view.fitted = True

    texture, events, claimed = render_to_texture(
        state,
        app.graph_renderer,
        packed,
        (width, height),
        fade(COLOR.BG_APP, 1.0),
        pointer,
    )
    imgui.image(
        imgui.ImTextureRef(texture.glo),
        imgui.ImVec2(float(width), float(height)),
        # The FBO's origin is bottom-left and imgui's is top-left, so the V is
        # flipped here rather than in the renderer, which has no opinion about
        # who presents it.
        imgui.ImVec2(0.0, 1.0),
        imgui.ImVec2(1.0, 0.0),
    )

    if not frozen:
        _apply_graph_events(app, document_id, document, view, state, events)
        # A click on the background clears the selection -- but only when the
        # library did NOT claim the pointer, or a press that starts a node
        # drag would also deselect on the way past.
        if (
            hovered
            and not claimed
            and imgui.is_mouse_clicked(imgui.MouseButton_.left)
            and view.selection
        ):
            view.selection = set()

    # The menus are imgui's this feature, opened from the library's own
    # context-menu event so the two agree about what the pointer hit.
    _canvas_menu(app, document_id, view)
    if state.menu_node:
        _node_menu(app, document_id, view, state.menu_node)
    _group_prompt(app, document_id, view)


def _apply_graph_events(
    app: App,
    document_id: str,
    document: Document,
    view: GraphViewState,
    state: GraphCanvasState,
    events: Sequence[GraphEvent],
) -> None:
    """Turn the library's events into `App` verb calls.

    A drag reports its position EVERY frame and the write happens once, when
    the pointer comes up: the accumulated positions live on the canvas state
    until then, which is what keeps one gesture to one save.
    """
    for event in events:
        if isinstance(event, Moved):
            state.dragging[event.name] = (event.x, event.y)
        elif isinstance(event, Clicked):
            if event.extend:
                view.selection ^= {event.name}
            else:
                view.selection = {event.name}
                app.choose_output(document_id, event.name)
        elif isinstance(event, Activated):
            entry = document.graph.passes.get(event.name)
            if entry is not None and entry.group:
                view.scope = entry.group
                view.fitted = False
        elif isinstance(event, Wired):
            refusal = app.drop_wire(
                document_id, event.producer, event.consumer, event.sampler
            )
            if refusal:
                app.notifications.push(refusal)
        elif isinstance(event, Unwired):
            refusal = app.unwire(document_id, event.consumer, event.sampler)
            if refusal:
                app.notifications.push(refusal)
        elif isinstance(event, Refused):
            app.notifications.push(refusal_text(event.reason))
        elif isinstance(event, MenuRequested):
            # The library says WHAT the pointer was over; which menu that is
            # stays the host's question.
            state.menu_node = event.name or ""
            imgui.open_popup(
                "##graph_node_menu" if event.name else "##graph_canvas_menu"
            )

    # The release is the write. `Moved` stops arriving the frame the button
    # comes up, so the commit is keyed on the button and not on the events.
    if state.dragging and not imgui.is_mouse_down(imgui.MouseButton_.left):
        moved = dict(state.dragging)
        state.dragging.clear()
        app.commit_graph_positions(document_id, moved)


def draw(app: App, document_id: str) -> None:
    """The graph canvas for one document: the tab row, then a child filling what is left of the
    host's content region. The widget positions no sibling and measures none."""
    ui_document = app.ui_documents.get(document_id)
    if ui_document is None:
        return
    document = ui_document.document
    view = app.graph_view_for(document_id)
    # Ports come from the compiled program (092 D1): the seam 091 uses before it plans. A
    # no-op once every pass has been attempted, so calling it per frame costs nothing.
    compile_pending_passes(document)

    entries = document.graph.passes
    wiring = document.effective_wiring()
    order = strip_order(document.passes, wiring)
    groups = {name: entries.get(name, PassEntry()).group for name in order}
    group_names = group_names_in_order(order, groups)
    view.scope = revalidated_scope(view.scope, set(group_names))
    view.selection &= set(document.passes)

    imgui.begin_disabled(app.copilot_turn_active)
    _tab_row(app, document_id, view, group_names)

    imgui.push_style_color(imgui.Col_.child_bg, COLOR.BG_APP)
    # No border: the editor pane the canvas fills is already framed, and a second frame one
    # pixel inside it is the clutter the maintainer named. `always_use_window_padding` keeps
    # the inset `borders` implied, so the canvas geometry is unchanged.
    child_open = imgui.begin_child(
        "##pass_graph",
        size=imgui.ImVec2(0.0, 0.0),
        child_flags=imgui.ChildFlags_.always_use_window_padding,
        window_flags=imgui.WindowFlags_.no_scrollbar
        | imgui.WindowFlags_.no_scroll_with_mouse,
    )
    imgui.pop_style_color(1)
    if child_open:
        _library_canvas(app, document_id, document, view)
    imgui.end_child()
    imgui.end_disabled()
