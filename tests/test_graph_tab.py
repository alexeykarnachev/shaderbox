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

from shaderbox.editor_types import TabRecord
from shaderbox.pass_graph import PassSource
from shaderbox.theme import SIZE, SPACE
from shaderbox.ui import update_and_draw
from shaderbox.ui_primitives import ellipsize
from shaderbox.widgets import pass_graph, uniform
from shaderbox.widgets.graph_state import bezier_point, node_size, wire_points
from tests.conftest import restart_app

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


def test_the_graph_tab_draws_and_leaves_the_error_list_empty(app: Any) -> None:
    # T2: the branch runs before the session fetch, fits on its first sized frame, and clears
    # a previous tab's errors so a stale list cannot drive `F8`. Falsifier: return before the
    # `editor_errors` write and an error from the shader tab survives onto the graph tab.
    document_id, _document = _chain(app)
    app.editor_errors = ["stale"]
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
    _frames(app, 2)
    graph_path = app.paths.document_script_for(document_id)
    imgui.new_frame()
    imgui.begin("rig")
    uniform._locate_uniform_declaration(
        app, "u_src", app.panel_pass(app.current_document_id)
    )
    imgui.end()
    imgui.end_frame()
    assert graph_path not in app.editor_sessions


def test_the_card_is_wide_enough_for_the_maintainers_own_longest_names(
    app: Any,
) -> None:
    """G11/S1: `distance_field` and `u_distance_field` are real names from his own document,
    and the width was chosen so neither is cut. Measured in a rig frame against the rasterized
    faces -- outside a frame `calc_text_size` segfaults the process, and an em-ratio estimate
    is what once shipped a truncating check in `tests/test_pass_settings_layout.py`.

    This is the width decision's pin. It was anchored at the old 136px card by a hard-coded
    110px budget; 094 D4e widened the card to 240 and that literal stopped pinning anything.
    The cut point is a property of the TEXT, not a fraction of the card, so the narrow case is
    now derived from the measured name itself: one pixel under its own width is the widest
    budget that must still ellipsize. Falsifier: hand the narrow case the full budget and it
    stops finding an ellipsis.
    """
    name_budget = float(SIZE.GRAPH_NODE_W - 2 * SIZE.GRAPH_PAD)
    label_budget = float(SIZE.GRAPH_NODE_W - 2 * SIZE.GRAPH_PORT_R - 2 - SIZE.GRAPH_PAD)
    imgui.new_frame()
    imgui.begin("rig")
    imgui.push_font(app.font_12, app.font_12.legacy_size)
    label_width = imgui.calc_text_size("u_distance_field").x
    label_kept = ellipsize("u_distance_field", label_budget)
    # One pixel under the name's own width: the widest budget that must still cut. Measured
    # here rather than written down, so the row keeps pinning whatever the card becomes.
    narrow_kept = ellipsize("u_distance_field", label_width - 1.0)
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
    # A regression check on `_fit`'s clamp, not a width pin: the width is pinned by the
    # ellipsis row above. Six columns, each reading the one before, so the chain is genuinely
    # as wide as the six-column arithmetic says. The two pane widths are DERIVED from
    # `node_size` rather than written down, so 094's 136 -> 240 does not silently turn this
    # into a test of nothing. Falsifier: drop the `min(1.0, ...)` and a small graph in a wide
    # pane is magnified.
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
    chain_w = 6 * node_size(1, False)[0] + 5 * SIZE.GRAPH_GAP_X + 2 * float(SPACE.LG)
    pass_graph._fit(view, nodes, picture, imgui.ImVec2(chain_w + 40.0, 900.0))
    assert view.zoom == 1.0
    pass_graph._fit(view, nodes, picture, imgui.ImVec2(chain_w * 0.75, 900.0))
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


def test_a_project_with_no_saved_tabs_still_opens_its_shader(app: Any) -> None:
    # The fallback the restore must not eat: with no records, `_init` opens the current
    # document's shader tab as it always did. Falsifier: run the fallback unconditionally and a
    # restored session gains a tab it did not have; skip it and an empty state opens blank.
    app.app_state.editor_tabs = []
    app.save()
    fresh = restart_app(app)
    assert len(fresh.editor_tabs) == 1, fresh.editor_tabs
    assert fresh.editor_tabs[0].kind == "shader"


def test_the_saved_tabs_reopen_on_a_fresh_app(app: Any, tmp_path: Any) -> None:
    """W2-2: the restore is what finding 7 asked for -- every open tab is still there after a
    restart. (094 C10 deleted the graph TAB; the claim was never about that kind.)

    Driven the way a restart drives it: the live tabs are mirrored by `App.save`, and a reopen
    of the same project dir reads them back in `_init`. The records are checked against what
    the project still holds, so the assertion is about files that genuinely exist.

    Falsifier: skip the restore block and the fresh app opens only the current document's
    shader tab, the pre-W2-2 behavior.
    """
    document_id, document = _chain(app)
    app.open_script_for(document_id)
    shader_path = document.passes["b"].source.path
    app.ensure_shader_tab(document_id, "b")
    saved_paths = [t.path for t in app.editor_tabs]
    saved_kinds = [t.kind for t in app.editor_tabs]
    assert "script" in saved_kinds and "shader" in saved_kinds, saved_kinds
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
    ]
    app.app_state.active_tab_path = "/nowhere/at/all.glsl"
    app.app_state.save(app.paths.app_state_file)
    fresh = restart_app(app)
    assert fresh.active_tab_index == 0
    assert fresh.active_tab is not None
    assert fresh.active_tab.path.name == "b.frag.glsl"
