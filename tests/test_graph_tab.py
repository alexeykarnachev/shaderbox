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
from shaderbox.paths import shader_lib_root
from shaderbox.tabs.code import tab_label
from shaderbox.ui import update_and_draw
from shaderbox.widgets import uniform
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
