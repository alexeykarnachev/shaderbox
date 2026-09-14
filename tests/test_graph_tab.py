"""The graph as an editor-tab kind (093 T1-T6), and the two geometry facts the card's width
and the fit rest on.

A graph tab has NO `EditorSession`: its path is the document's `graph.json`, which keys every
path-keyed pass-through, and nothing edits it as text. The rows below are what says so -- that
the label branch fires before the pass-name fallthrough, that the dirty and formatter reads
answer on a missing session rather than raising, that closing one walks the same teardown, and
that no panel opens a GLSL editor over it.
"""

import math
from itertools import pairwise
from typing import Any

import pytest
from imgui_bundle import imgui

from shaderbox.commands import CommandId
from shaderbox.formatting import formatter_for
from shaderbox.pass_graph import PassSource
from shaderbox.paths import shader_lib_root
from shaderbox.tabs import document as document_tab
from shaderbox.tabs.code import tab_label
from shaderbox.theme import SIZE
from shaderbox.ui import update_and_draw
from shaderbox.ui_primitives import ellipsize
from shaderbox.widgets import pass_graph, uniform
from shaderbox.widgets.graph_state import bezier_point, wire_points

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
    # T2: the branch runs before the session fetch, fits on its first sized frame, and clears
    # a previous tab's errors so a stale list cannot drive `F8`. Falsifier: return before the
    # `editor_errors` write and an error from the shader tab survives onto the graph tab.
    document_id, _document = _chain(app)
    app.editor_errors = ["stale"]
    app.open_graph_for(document_id)
    _frames(app, 3)
    view = app.graph_view_for(document_id)
    assert view.canvas_rect != (0.0, 0.0, 0.0, 0.0)
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


def test_the_entry_tick_marks_this_documents_tab_of_this_kind(app: Any) -> None:
    # T5: one predicate for the Script row's tick and the Passes row's, so the two cannot
    # drift. Falsifier: drop the document clause -- another document's graph tab ticks this
    # document's row. The drawn tick is the maintainer's eyes, as the Script row's is.
    from tests.conftest import seed_extra_document

    first = app.current_document_id
    second = seed_extra_document(app, "second-document")
    app.open_graph_for(first)
    assert document_tab._entry_tab_active(app, first, "graph") is True
    assert document_tab._entry_tab_active(app, second, "graph") is False
    assert document_tab._entry_tab_active(app, first, "script") is False


# ---- the two geometry facts (093 G11, S8) ---------------------------------------------------


def test_the_card_is_wide_enough_for_the_maintainers_own_longest_names(
    app: Any,
) -> None:
    """G11/S1: `distance_field` and `u_distance_field` are real names from his own document,
    and the width was chosen so neither is cut. Measured in a rig frame against the rasterized
    faces -- outside a frame `calc_text_size` segfaults the process, and an em-ratio estimate
    is what once shipped a truncating check in `tests/test_pass_settings_layout.py`.

    This is the width decision's pin: red at 128 (112px of name against a 112px budget with
    zero slack, and 112px of port label against 110), green at 136.
    """
    name_budget = float(SIZE.GRAPH_NODE_W - 2 * SIZE.GRAPH_PAD)
    label_budget = float(SIZE.GRAPH_NODE_W - 2 * SIZE.GRAPH_PORT_R - 2 - SIZE.GRAPH_PAD)
    imgui.new_frame()
    imgui.begin("rig")
    imgui.push_font(app.font_12, app.font_12.legacy_size)
    label_width = imgui.calc_text_size("u_distance_field").x
    label_kept = ellipsize("u_distance_field", label_budget)
    # The card 128 would have given: the same port label ellipsizes there.
    narrow_kept = ellipsize("u_distance_field", 110.0)
    imgui.pop_font()
    imgui.push_font(app.font_14_bold, app.font_14_bold.legacy_size)
    name_width = imgui.calc_text_size("distance_field").x
    name_kept = ellipsize("distance_field", name_budget)
    imgui.pop_font()
    imgui.end()
    imgui.end_frame()

    assert narrow_kept.endswith("..."), (
        "the 128 card's port-label budget no longer cuts the name it was widened for; "
        "this row has stopped pinning the width"
    )
    assert label_kept == "u_distance_field", label_kept
    assert label_budget - label_width >= 4.0, (label_budget, label_width)
    assert name_kept == "distance_field", name_kept
    assert name_budget - name_width >= 4.0, (name_budget, name_width)


def _fitted_window(
    view: Any, avail: tuple[float, float]
) -> tuple[float, float, float, float]:
    return (
        view.pan[0],
        view.pan[1],
        view.pan[0] + avail[0] / view.zoom,
        view.pan[1] + avail[1] / view.zoom,
    )


def test_the_fit_frames_every_wire_not_only_the_cards(app: Any) -> None:
    """S8: a backward wire's S-curve bulges past both cards, so the nodes' bounding box alone
    leaves part of it outside the fitted view -- which is finding 5, dissolved by construction
    rather than repaired.

    Break to try: frame the nodes alone -- 8 of the 75 sampled points land outside. The
    `avail` is small on purpose: at 800x600 the 1.0 zoom clamp centers enough slack to hide
    the bulge for the broken implementation too.
    """
    document_id, document = _chain(app)
    # A backward read, so a wire genuinely runs right-to-left across the picture.
    document.passes["a"].release_program(_SAMPLER)
    document.passes["a"].compile()
    assert (
        app.session.set_sampler_source(document_id, "a", "u_src", PassSource("c")) == ""
    )
    # Placed in a ROW by hand rather than by Arrange: the backward read makes a cycle, which
    # the rank layout stacks into one tall column where the bulge lands on the axis the fit
    # has slack on. A row is the shape the bulge actually threatens, and it is the shape a
    # user leaves his passes in.
    assert (
        app.session.set_pass_positions(
            document_id, {"a": (0.0, 0.0), "b": (200.0, 0.0), "c": (400.0, 0.0)}
        )
        == ""
    )
    view = app.graph_view_for(document_id)
    view.fitted = False
    avail = (520.0, 200.0)
    picture = pass_graph._build_view(document, "", {})
    nodes = list(picture.nodes.values())
    pass_graph._fit(view, nodes, picture, imgui.ImVec2(*avail))
    x0, y0, x1, y1 = _fitted_window(view, avail)

    sampled = 0
    for edge in picture.edges:
        points = pass_graph._wire_canvas_points(picture, edge)
        for i in range(25):
            px, py = bezier_point(*points, i / 24)
            sampled += 1
            assert x0 <= px <= x1 and y0 <= py <= y1, (edge.wire_id, i, (px, py))
    assert sampled == 75, sampled
    # And the bulge is real: without the wires, a tenth of those points fall outside.
    nx0, ny0, nx1, ny1 = pass_graph._bbox(nodes)
    stray = sum(
        not (nx0 <= px <= nx1 and ny0 <= py <= ny1)
        for edge in picture.edges
        for px, py in (
            bezier_point(*pass_graph._wire_canvas_points(picture, edge), i / 24)
            for i in range(25)
        )
    )
    assert stray >= 8, (
        f"only {stray} sampled points leave the cards' box; this row has stopped "
        "distinguishing the union fit from the nodes-only one"
    )


def test_the_fit_never_zooms_past_one_and_shrinks_for_a_narrow_pane(app: Any) -> None:
    # A regression check on `_fit`'s clamp, not a width pin: it is green at 108 and 136 alike,
    # and the width is pinned by the ellipsis row above. Six columns, each reading the one
    # before, so the chain is genuinely as wide as the six-column arithmetic says. Falsifier:
    # drop the `min(1.0, ...)` and a small graph in a wide pane is magnified.
    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    chain = ["a", "b", "c", "d", "e", "f"]
    for name in chain:
        assert app.session.add_pass(document_id, name) == ""
    for name in chain[1:]:
        document.passes[name].release_program(_SAMPLER)
        document.passes[name].compile()
    for source, consumer in pairwise(chain):
        assert app.drop_wire(document_id, source, consumer, "u_src") == ""
    app.arrange_graph(document_id)
    view = app.graph_view_for(document_id)
    picture = pass_graph._build_view(document, "", {})
    nodes = list(picture.nodes.values())
    pass_graph._fit(view, nodes, picture, imgui.ImVec2(1225.0, 600.0))
    assert view.zoom == 1.0
    pass_graph._fit(view, nodes, picture, imgui.ImVec2(740.0, 600.0))
    assert 0.6 < view.zoom < 1.0, view.zoom


def test_a_wires_canvas_points_are_its_screen_points_divided_by_the_zoom(
    app: Any,
) -> None:
    # The exactness S8's fit rests on: the offset and the endpoints both scale linearly with
    # the zoom, so sampling at zoom 1 in canvas space frames the screen curve at any zoom.
    # Falsifier: floor the offset in SCREEN pixels instead and the two stop agreeing.
    a, b = (10.0, 20.0), (-300.0, 140.0)
    for zoom in (0.25, 1.0, 2.5):
        scaled = wire_points(
            (a[0] * zoom, a[1] * zoom), (b[0] * zoom, b[1] * zoom), zoom
        )
        plain = wire_points(a, b, 1.0)
        for got, want in zip(scaled, plain, strict=True):
            assert math.isclose(got[0], want[0] * zoom, abs_tol=1e-9), (zoom, got, want)
            assert math.isclose(got[1], want[1] * zoom, abs_tol=1e-9), (zoom, got, want)
