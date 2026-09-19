"""The graph as an editor-tab kind (093 T1-T6), and the two geometry facts the card's width
and the fit rest on.

A graph tab has NO `EditorSession`: its path is the document's `graph.json`, which keys every
path-keyed pass-through, and nothing edits it as text. The rows below are what says so -- that
the label branch fires before the pass-name fallthrough, that the dirty and formatter reads
answer on a missing session rather than raising, that closing one walks the same teardown, and
that no panel opens a GLSL editor over it.
"""

from typing import Any

import pytest
from imgui_bundle import imgui

from shaderbox.commands import CommandId
from shaderbox.editor_types import TabRecord
from shaderbox.formatting import formatter_for
from shaderbox.graph_canvas.adapter import Moved
from shaderbox.graph_canvas.ffi import Gesture
from shaderbox.pass_graph import PassEntry, strip_order
from shaderbox.paths import shader_lib_root
from shaderbox.tabs.code import tab_label
from shaderbox.theme import COLOR, group_tint
from shaderbox.ui import update_and_draw
from shaderbox.widgets import pass_graph, uniform
from shaderbox.widgets.graph_state import node_sizes, ports_of
from tests.conftest import restart_app, seed_extra_document

# The imgui font atlas is process-global, so every frame-driving module owns a worker
# (`pyproject.toml`).
pytestmark = pytest.mark.xdist_group("gl_frames_graph_tab")

_SAMPLER = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_src;
out vec4 fs_color;
void main() { fs_color = texture(u_src, vs_uv); }
"""


def _chain(app: Any) -> tuple[str, Any]:
    """`a -> b -> c` on the current document, each reading the one before through `u_src`."""
    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    for name in ("a", "b", "c"):
        assert app.session.add_pass(document_id, name) == ""
    for name in ("b", "c"):
        document.passes[name].release_program(_SAMPLER)
        document.passes[name].compile()
    assert app.drop_wire(document_id, "a", "b", "u_src") == ""
    assert app.drop_wire(document_id, "b", "c", "u_src") == ""
    return document_id, document


def _frames(app: Any, n: int) -> None:
    for _ in range(n):
        update_and_draw(app)


def test_a_graph_tab_names_its_document_rather_than_a_pass_file(app: Any) -> None:
    # T1: `graph.json` is the tab's PATH, so the multi-pass fallthrough would run
    # `pass_name_of` on it and label the tab after a filename. Falsifier: drop the branch.
    document_id, _document = _chain(app)
    app.ui_documents[document_id].ui_state.ui_name = "Chain"
    app.open_graph_for(document_id)
    tab = app.active_tab
    assert tab is not None and tab.kind == "graph"
    assert tab_label(app, tab) == "Chain (graph)"


def test_opening_the_graph_twice_focuses_one_tab_and_it_is_session_less(
    app: Any,
) -> None:
    # T1, T4: the path is the tab's identity, so the second open re-focuses the first. Nothing
    # on the session-shaped paths may raise on a tab that has none. Falsifier: give the tab a
    # session-bearing path -- `is_tab_dirty` starts answering the wrong file's state.
    document_id, _document = _chain(app)
    app.open_graph_for(document_id)
    app.open_graph_for(document_id)
    graph_path = app.paths.graph_json_for(document_id)
    assert [t.path for t in app.editor_tabs].count(graph_path) == 1
    tab = app.active_tab
    assert tab is not None and tab.path == graph_path
    assert app.is_tab_dirty(tab) is False
    assert app.is_current_editor_dirty() is False
    assert formatter_for("graph") is None
    assert graph_path not in app.editor_sessions
    # Neither returns anything; what is asserted is that neither raises on the absent session.
    app.format_current_editor()
    app.jump_to_next_error()
    assert graph_path not in app.editor_sessions


def test_the_graph_tab_closes_and_dies_with_its_document(app: Any) -> None:
    # T1: `close_editor_for_path` is the pass-delete verb's own teardown, and
    # `_on_document_deleted` filters by document while keeping lib tabs. Falsifier: key the
    # deletion filter on the session instead -- a session-less tab outlives its document.
    document_id, _document = _chain(app)
    app.open_graph_for(document_id)
    graph_path = app.paths.graph_json_for(document_id)
    app.close_editor_for_path(graph_path)
    assert not any(t.kind == "graph" for t in app.editor_tabs)

    app.open_graph_for(document_id)
    lib_path = sorted(shader_lib_root().rglob("*.glsl"))[0]
    app.open_shader_lib_file(lib_path)
    app._on_document_deleted(document_id, app.paths.document_json_for(document_id))
    assert not any(t.kind == "graph" for t in app.editor_tabs)
    assert any(t.kind == "lib" for t in app.editor_tabs), "a lib tab was swept with it"


def test_the_open_graph_command_is_registered(app: Any) -> None:
    # T4. Falsifier: add the spec without the callback -- the registry coverage test catches
    # the reverse, an id with no spec, but not a spec with no handler.
    assert CommandId.OPEN_GRAPH in app.command_callbacks


def test_the_graph_tab_draws_and_leaves_the_error_list_empty(app: Any) -> None:
    """T2: the branch runs before the session fetch, fits on its first sized frame, and
    clears a previous tab's errors so a stale list cannot drive `F8`. Falsifier: return
    before the `editor_errors` write and an error from the shader tab survives onto the
    graph tab.

    The canvas having drawn is read from the library-backed panel (098) rather than from a
    hit rect the old renderer published: the panel exists and holds a sized texture only if
    a frame actually went through it.
    """
    document_id, _document = _chain(app)
    app.editor_errors = ["stale"]
    app.open_graph_for(document_id)
    _frames(app, 3)
    view = app.graph_view_for(document_id)
    state = app.graph_canvases.get(document_id)
    assert state is not None and state.panel is not None
    assert state.panel.texture is not None and state.panel.texture.size[0] > 0
    assert state.fitted is True
    assert view.fitted is True
    assert app.editor_errors == []


def test_the_uniforms_panel_opens_no_editor_over_the_graphs_own_file(app: Any) -> None:
    """T1: `_locate_uniform_declaration` was the one CREATING session call reachable while a
    graph tab is active, and a session at `graph.json` would make the tab dirty-capable.

    It runs only on a hover or a click of a uniform's name, which frames of a focused graph
    tab never inject, so it is called directly. Break to try: the creating
    `get_current_session()` -- a session appears at the graph path.
    """
    document_id, _document = _chain(app)
    app.open_graph_for(document_id)
    _frames(app, 2)
    graph_path = app.paths.graph_json_for(document_id)
    imgui.new_frame()
    imgui.begin("rig")
    uniform._locate_uniform_declaration(app, "u_src")
    imgui.end()
    imgui.end_frame()
    assert graph_path not in app.editor_sessions


def test_open_graph_for_opens_that_documents_tab(app: Any) -> None:
    """The summoner names the document it opens, not whichever is current (093 W8).

    The entry-point rows and their accent tick are gone -- the verb lives on the document's
    context menu, which fires on a tile a right-click did not select. Falsifier: route the
    menu's item through the current-document command and the second document below never gets
    its graph tab.
    """
    first = app.current_document_id
    second = seed_extra_document(app, "second-document")
    app.open_graph_for(second)
    active = app.active_tab
    assert active is not None
    assert active.kind == "graph"
    assert active.document_id == second, "the graph opened on the wrong document"
    assert app.current_document_id == first, "opening a tab switched the document"


# ---- the two geometry facts (093 G11, S8) ---------------------------------------------------


def _fitted_window(
    view: Any, avail: tuple[float, float]
) -> tuple[float, float, float, float]:
    return (
        view.pan[0],
        view.pan[1],
        view.pan[0] + avail[0] / view.zoom,
        view.pan[1] + avail[1] / view.zoom,
    )


def test_the_saved_tabs_reopen_on_a_fresh_app(app: Any, tmp_path: Any) -> None:
    """W2-2: the restore is what finding 7 asked for -- the graph tab (and every other) is
    still there after a restart.

    Driven the way a restart drives it: the live tabs are mirrored by `App.save`, and a reopen
    of the same project dir reads them back in `_init`. The records are checked against what
    the project still holds, so the assertion is about files that genuinely exist.

    Falsifier: skip the restore block and the fresh app opens only the current document's
    shader tab, the pre-W2-2 behavior.
    """
    document_id, document = _chain(app)
    app.open_script_for(document_id)
    app.open_graph_for(document_id)
    shader_path = document.passes["b"].source.path
    app.ensure_shader_tab(document_id, "b")
    saved_paths = [t.path for t in app.editor_tabs]
    saved_kinds = [t.kind for t in app.editor_tabs]
    assert "graph" in saved_kinds and "script" in saved_kinds, saved_kinds
    app.save()
    records = app.app_state.editor_tabs
    assert [r.path for r in records] == [str(p) for p in saved_paths]
    assert app.app_state.active_tab_path == str(shader_path)

    fresh = restart_app(app)
    assert [t.path for t in fresh.editor_tabs] == saved_paths, fresh.editor_tabs
    assert [t.kind for t in fresh.editor_tabs] == saved_kinds
    assert fresh.active_tab is not None
    assert fresh.active_tab.path == shader_path
    assert fresh.tab_select_pending is True


def test_the_active_tab_comes_back_by_path_not_by_position(
    app: Any, tmp_path: Any
) -> None:
    """W2-2: the active tab is held by PATH, because a dropped record shifts every later one.

    Four records with the FIRST one's file gone, and the tab he was looking at is the second.
    A saved index of 1 lands on the third file once the first is dropped -- the file that took
    that position -- so this asserts the identity instead.

    Falsifier: restore by a clamped index and the active tab is `c`, not `b`.
    """
    document_id, document = _chain(app)
    doomed = document.passes["a"].source.path
    records = [
        TabRecord(path=str(doomed), kind="shader", document_id=document_id),
        TabRecord(
            path=str(document.passes["b"].source.path),
            kind="shader",
            document_id=document_id,
        ),
        TabRecord(
            path=str(document.passes["c"].source.path),
            kind="shader",
            document_id=document_id,
        ),
        TabRecord(
            path=str(app.paths.graph_json_for(document_id)),
            kind="graph",
            document_id=document_id,
        ),
    ]
    app.app_state.editor_tabs = records
    app.app_state.active_tab_path = records[1].path
    app.app_state.save(app.paths.app_state_file)
    # The pass whose tab was first is deleted between the sessions, which is what makes the
    # index shift: its record is dropped and every later tab moves down one.
    doomed.unlink()

    fresh = restart_app(app)
    assert [t.path.name for t in fresh.editor_tabs] == [
        "b.frag.glsl",
        "c.frag.glsl",
        "graph.json",
    ], fresh.editor_tabs
    assert fresh.active_tab is not None
    assert fresh.active_tab.path.name == "b.frag.glsl", fresh.active_tab
    assert fresh.active_tab_index == 0


def test_an_active_path_nothing_carries_falls_to_the_first_tab(app: Any) -> None:
    # The other half: a saved active path whose tab did not survive selects the first restored
    # tab rather than leaving the index out of range. Falsifier: return -1 from the lookup and
    # `active_tab` reads the LAST tab through Python's negative indexing.
    document_id, document = _chain(app)
    app.app_state.editor_tabs = [
        TabRecord(
            path=str(document.passes["b"].source.path),
            kind="shader",
            document_id=document_id,
        ),
        TabRecord(
            path=str(app.paths.graph_json_for(document_id)),
            kind="graph",
            document_id=document_id,
        ),
    ]
    app.app_state.active_tab_path = "/nowhere/at/all.glsl"
    app.app_state.save(app.paths.app_state_file)
    fresh = restart_app(app)
    assert fresh.active_tab_index == 0
    assert fresh.active_tab is not None
    assert fresh.active_tab.path.name == "b.frag.glsl"


def test_a_project_with_no_saved_tabs_still_opens_its_shader(app: Any) -> None:
    # The fallback the restore must not eat: with no records, `_init` opens the current
    # document's shader tab as it always did. Falsifier: run the fallback unconditionally and a
    # restored session gains a tab it did not have; skip it and an empty state opens blank.
    app.app_state.editor_tabs = []
    app.save()
    fresh = restart_app(app)
    assert len(fresh.editor_tabs) == 1, fresh.editor_tabs
    assert fresh.editor_tabs[0].kind == "shader"


def _canvas_titles(app: Any, document_id: str) -> list[str]:
    """What the canvas actually packed this frame, by node title."""
    state = app.graph_canvases[document_id]
    assert state.packed is not None
    return [node.title for node in state.packed.nodes]


def test_the_scope_tab_collapses_a_group_into_one_box(app: Any) -> None:
    """098 F4: the scope was resolved for the TAB ROW and never for the
    picture, so switching tabs relabelled the row and packed the same flat
    graph -- every pass, every time, with no box and no ghost. The feature
    the row exists for was absent while the row claimed it.

    Read from what was PACKED rather than from pixels: the node list is the
    thing the scope decides, and a pixel test at this size cannot tell a box
    from the pass it replaced.
    """
    document_id, _document = _chain(app)
    app.open_graph_for(document_id)
    view = app.graph_view_for(document_id)
    view.selection = {"a", "b"}
    assert app.group_selection(document_id, "pair") == ""
    _frames(app, 2)

    # The root: the group is ONE box named after itself, and neither member
    # is drawn beside it.
    root = _canvas_titles(app, document_id)
    assert "pair" in root, f"the group did not collapse into a box: {root}"
    assert "a" not in root and "b" not in root, f"a member escaped its box: {root}"
    assert "c" in root, "the ungrouped pass vanished with the grouping"

    # Inside the group: its members, plus the outside reader as a ghost, so
    # the wire leaving the group still lands on something.
    view.scope = "pair"
    _frames(app, 2)
    inside = _canvas_titles(app, document_id)
    assert "a" in inside and "b" in inside, f"a member is missing inside: {inside}"
    assert "c" in inside, f"the outside reader got no ghost: {inside}"
    assert "pair" not in inside, "the box drew itself inside its own tab"


def test_a_ghost_inside_a_group_refuses_every_gesture(app: Any) -> None:
    """A ghost stands for context, not for a thing to grab: a press on one
    must fall through to the canvas rather than being swallowed by a node
    that does nothing with it."""
    document_id, _document = _chain(app)
    app.open_graph_for(document_id)
    view = app.graph_view_for(document_id)
    view.selection = {"a", "b"}
    assert app.group_selection(document_id, "pair") == ""
    view.scope = "pair"
    _frames(app, 2)

    state = app.graph_canvases[document_id]
    assert state.packed is not None
    ghosts = [n for n in state.packed.nodes if n.dashed]
    assert ghosts, "the group's outside reader was not drawn as a ghost"
    for ghost in ghosts:
        assert ghost.accepts == int(Gesture.NONE), f"{ghost.title} accepts a gesture"
        assert ghost.fade > 0.0, f"{ghost.title} is not faded"


def test_a_menu_forgets_its_node_once_the_popup_is_gone(app: Any) -> None:
    """098 F6: `menu_node` was written on the context-menu event and never
    cleared, so it named a pass for the rest of the session -- including
    after that pass was deleted from the very menu it opened, at which point
    every later frame draws a node menu about a pass the document no longer
    has.

    Driven through real frames with NO popup open, which is the state the
    field has to survive being in: the widget reaches the clearing branch
    the first time `begin_popup` returns False. Setting the field and
    asserting it by hand would pass with that branch deleted.
    """
    document_id, document = _chain(app)
    app.open_graph_for(document_id)
    _frames(app, 2)
    state = app.graph_canvases[document_id]

    # What a Context_Menu event leaves behind, for a pass that then goes away
    # -- which is what the menu's own Delete does.
    state.menu_node = "b"
    assert app.session.delete_pass(document_id, "b") == ""
    assert "b" not in document.passes
    _frames(app, 3)
    assert state.menu_node == "", "a closed menu still names a deleted pass"


def test_a_project_switch_releases_the_canvas_renderer(app: Any) -> None:
    """098 GL#4: `release()` dropped the renderer REFERENCE and never called
    its `release()`, so the glyph atlas texture and two programs stayed on
    the GPU -- once per project switch, for the life of the process.

    Dropping a Python reference frees the Python object; the GL objects
    behind it outlive it, which is why every other GL owner in this file is
    released explicitly rather than left to the collector.

    The falsifier is the atlas texture: moderngl reclasses the released
    object's `mglo` to `InvalidObject`, so the texture taken before the
    switch must carry that afterwards. The wrapper itself keeps its class
    and its `glo` -- measured, both unchanged across a release -- so neither
    of those can witness this, and asserting that `use()` raises would pass
    on the `AttributeError` a typo raises too.
    """
    document_id, _document = _chain(app)
    app.open_graph_for(document_id)
    _frames(app, 2)
    renderer = app.graph_renderer
    assert renderer is not None
    atlas = renderer.atlas

    restart_app(app)
    assert app.graph_renderer is None, "the renderer survived the switch"
    assert type(atlas.mglo).__name__ == "InvalidObject", (
        "the glyph atlas texture is still live on the GPU after the switch"
    )


def test_dragging_a_box_keeps_its_members_apart(app: Any) -> None:
    """098: a box has no position of its own -- it is drawn at its members'
    corner -- so a drag applies a DELTA against each member's start. Frozen
    from the STORED position, a member never placed contributes (0, 0) while
    the canvas draws it at the rank layout's, so the whole group collapsed
    onto one point on the first pixel of the first drag, and
    `commit_graph_positions` wrote it to disk.

    Measured before the fix: three passes drawn at x 0 / 200 / 400 landed on
    two distinct points with two of them identical.

    The frozen starts must be the positions the canvas is DRAWING, which is
    what `_positions` already resolved. Driven through `_apply_graph_events`
    rather than through a copy of its body: a probe that reimplements the
    branch cannot see a fix to the branch.
    """
    document_id, document = _chain(app)
    app.open_graph_for(document_id)
    view = app.graph_view_for(document_id)
    view.selection = {"a", "b"}
    assert app.group_selection(document_id, "pair") == ""
    _frames(app, 3)

    # None of them has ever been placed, which is the case that broke.
    entries = document.graph.passes
    assert all(entries[n].position is None for n in ("a", "b", "c"))

    state = app.graph_canvases[document_id]
    assert state.packed is not None
    drawn = {
        view_.name: (node.x, node.y)
        for view_, node in zip(state.packed.views, state.packed.nodes, strict=True)
    }
    box = next(v for v in state.packed.views if v.is_box)
    start = drawn[box.name]

    # The positions the WIDGET resolves, which is what it hands the event
    # handler: a stored position wins, a pass never placed takes the rank
    # layout's. Built the same way here rather than guessed, because the
    # whole defect was the handler using a different source than the draw.
    wiring = document.effective_wiring()
    ports = ports_of(document, wiring)
    order = strip_order(document.passes, wiring)
    groups = {n: entries.get(n, PassEntry()).group for n in order}
    positions = pass_graph._positions(document, wiring, groups, node_sizes(ports), {})
    assert positions["a"] != positions["b"], (
        "the fixture placed both members at one point, so it cannot see a collapse"
    )
    # One call is one frame, and the real mouse is UP in a test -- so the
    # commit branch fires and writes. That is the path that persisted the
    # collapse, which makes it the right one to assert against.
    pass_graph._apply_graph_events(
        app,
        document_id,
        document,
        view,
        state,
        [Moved(box.key, box.name, start[0] + 30.0, start[1] + 12.0, box.members)],
        positions,
    )

    # Re-read: `document.graph` is replaced by the write, so the `entries`
    # captured above is the pre-drag model.
    saved = {name: document.graph.passes[name].position for name in ("a", "b")}
    assert all(p is not None for p in saved.values()), (
        f"the drag wrote no position: {saved}"
    )
    assert saved["a"] != saved["b"], (
        f"the box's members collapsed onto one point: {saved}"
    )
    # Each moved by the SAME delta, which is what a box drag means.
    delta_a = (
        saved["a"][0] - positions["a"][0],
        saved["a"][1] - positions["a"][1],
    )
    delta_b = (
        saved["b"][0] - positions["b"][0],
        saved["b"][1] - positions["b"][1],
    )
    assert delta_a == pytest.approx(delta_b), (
        f"the members moved by different deltas: {delta_a} against {delta_b}"
    )


def _halo_widths(app: Any, document_id: str, title: str) -> list[float]:
    """The state rings one packed node wears. Zero width is an inert ring."""
    state = app.graph_canvases[document_id]
    assert state.packed is not None
    node = next(n for n in state.packed.nodes if n.title == title)
    return [round(float(width), 4) for _color, _inset, width in node.halos]


def test_a_selected_group_box_wears_the_selection_ring(app: Any) -> None:
    """Selecting a box must mark the box, and the check is against the
    UNSELECTED node beside it rather than against a constant.

    The two node keys live in different namespaces: a box's is `b:<group>`
    while `view.selection` holds pass NAMES, so `pass_key(member)` could
    never equal the box's key and the box drew bare. At the root its
    members are not drawn either, so clicking a group selected it and
    nothing on screen said so.

    `c` is the pair that differs only in the property under test -- same
    scope, same frame, same theme, selected or not. Comparing the box
    against its own earlier frame would also move when anything else did.
    """
    document_id, _document = _chain(app)
    app.open_graph_for(document_id)
    view = app.graph_view_for(document_id)
    view.selection = {"a", "b"}
    assert app.group_selection(document_id, "pair") == ""

    view.selection = set()
    _frames(app, 2)
    bare_box = _halo_widths(app, document_id, "pair")
    bare_c = _halo_widths(app, document_id, "c")
    assert bare_box == bare_c, (
        "with nothing selected the box and the loose pass must wear the same "
        f"rings: box={bare_box} c={bare_c}"
    )

    # Clicking a box selects its MEMBERS -- that is what the click handler
    # stores, and the box has to answer to it.
    view.selection = {"a", "b"}
    _frames(app, 2)
    picked_box = _halo_widths(app, document_id, "pair")
    picked_c = _halo_widths(app, document_id, "c")
    assert picked_c == bare_c, (
        f"selecting the group moved the untouched pass's rings: {picked_c}"
    )
    assert picked_box != bare_box, (
        "a selected group box wears no ring: it packed the same halos as "
        f"when nothing was selected ({picked_box})"
    )


def _packed_node(app: Any, document_id: str, title: str) -> Any:
    state = app.graph_canvases[document_id]
    assert state.packed is not None
    return next(n for n in state.packed.nodes if n.title == title)


def test_a_pass_that_fails_to_compile_says_so_on_the_canvas(app: Any) -> None:
    """The strip draws a compile error as a red border; the canvas drew
    nothing at all, so the same broken pass looked healthy on one surface
    and broken on the other.

    `c` is the comparison: same frame, same scope, same theme, compiling.
    A node checked against a constant would pass on code that marked every
    node, and one checked against its own earlier frame would move whenever
    anything else did.
    """
    document_id, document = _chain(app)
    app.open_graph_for(document_id)
    _frames(app, 2)
    assert _packed_node(app, document_id, "b").halos == (), (
        "a compiling pass already wears a ring, so the error ring cannot be "
        "told from it"
    )

    document.passes["b"].release_program("#version 460 core\nthis is not glsl\n")
    document.passes["b"].compile()
    assert document.passes["b"].compile_unit.errors, "the fixture did not break"
    _frames(app, 2)

    broken = _packed_node(app, document_id, "b")
    healthy = _packed_node(app, document_id, "c")
    assert healthy.halos == (), (
        f"marking the broken pass also marked the compiling one: {healthy.halos}"
    )
    assert broken.halos, "a pass with compile errors wears no ring on the canvas"
    colour = broken.halos[0][0]
    assert tuple(round(v, 4) for v in colour[:3]) == tuple(
        round(v, 4) for v in COLOR.STATE_ERROR[:3]
    ), f"the error ring is not the theme's error colour: {colour}"


def test_a_grouped_pass_carries_its_groups_tint_on_the_canvas(app: Any) -> None:
    """The strip fills a grouped pass faintly with its group's hue. Inside
    the group's own tab the canvas drew the members untinted, so which
    group a pass belonged to was unreadable there.

    Checked inside the scope, where members are drawn as themselves: at the
    root they collapse into one box and there is no member to tint.
    """
    document_id, _document = _chain(app)
    app.open_graph_for(document_id)
    view = app.graph_view_for(document_id)
    view.selection = {"a", "b"}
    assert app.group_selection(document_id, "pair") == ""
    view.scope = "pair"
    _frames(app, 2)

    member = _packed_node(app, document_id, "a")
    assert member.tint is not None and member.tint_amount > 0.0, (
        "a grouped pass carries no tint inside its own group's tab"
    )
    assert tuple(round(v, 4) for v in member.tint[:3]) == tuple(
        round(v, 4) for v in group_tint("pair")[:3]
    ), f"the member's tint is not its group's hue: {member.tint}"

    # `c` is the pair that differs only in membership: same scope, same
    # frame, drawn as itself, and NOT in the group. Checking the collapsed
    # box at the root instead would decide nothing -- a box is keyed
    # `b:<group>` while the tints are keyed `p:<pass>`, so it comes back
    # untinted whether the rule holds or not.
    # The same pass drawn UNGROUPED is the pair that differs only in
    # membership. Every candidate inside this scope is unreachable by
    # construction: the collapsed box is keyed `b:<group>` and a ghost
    # `g:out:<pass>`, while the tints are keyed `p:<pass>` -- so either
    # comes back untinted whether the rule holds or not.
    assert app.dissolve_group(document_id, "pair") == ""
    view.scope = ""
    _frames(app, 2)
    loose = _packed_node(app, document_id, "a")
    assert loose.tint_amount == 0.0, (
        f"an ungrouped pass still carries a group's hue: {loose.tint}"
    )
