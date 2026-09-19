"""The graph canvas's gesture verbs on `App` (092 D12-D16), driven headlessly.

Every write the canvas makes goes through one of these, so each refusal is asserted here
without a window: the cycle drop, the media drop, the unwire, the drag commit, Group and
Dissolve. The widget itself is pinned to have no session write of its own.
"""

from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from imgui_bundle import imgui

from shaderbox.pass_graph import NoSource, PassSource
from shaderbox.ui import update_and_draw
from shaderbox.widgets import pass_graph, pass_list

# The imgui font atlas is process-global, so every frame-driving module owns a worker
# (`pyproject.toml`).
pytestmark = pytest.mark.xdist_group("gl_frames_graph_view")

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


def test_drop_wire_writes_the_read_and_refuses_the_loop(app: Any) -> None:
    document_id, document = _chain(app)
    assert document.passes["c"].uniform_values["u_src"] == PassSource("b")
    document.passes["a"].release_program(_SAMPLER)
    document.passes["a"].compile()
    with mock.patch.object(
        app.session, "set_sampler_source", wraps=app.session.set_sampler_source
    ) as write:
        refusal = app.drop_wire(document_id, "c", "a", "u_src")
    # Falsifier: refuse only when the culprit is an endpoint -- `plan_passes` names `a` for a
    # drop whose endpoints are `c` and `a`, and the whole-list check is what refuses it.
    assert "passes form a cycle" in refusal
    assert write.call_count == 0, "a refused drop wrote"
    assert "u_src" not in document.passes["a"].uniform_values or not isinstance(
        document.passes["a"].uniform_values["u_src"], PassSource
    )
    # Feedback is allowed: a node's output into its own port.
    assert app.drop_wire(document_id, "c", "c", "u_src") == ""
    assert document.passes["c"].uniform_values["u_src"] == PassSource("c")


def test_drop_wire_refuses_a_media_bound_port_and_keeps_the_texture(app: Any) -> None:
    document_id, document = _chain(app)
    bound = document.gl.texture((2, 2), 4)
    document.passes["c"].uniform_values["u_src"] = bound
    with mock.patch.object(
        app.session, "set_sampler_source", wraps=app.session.set_sampler_source
    ) as write:
        refusal = app.drop_wire(document_id, "a", "c", "u_src")
    # Falsifier: call `set_sampler_source` unconditionally, which runs `try_to_release` on
    # the bound texture -- the one irreversible write a canvas gesture could make.
    assert "bound to media" in refusal
    assert write.call_count == 0
    assert document.passes["c"].uniform_values["u_src"] is bound
    assert bound.glo != 0


def test_unwire_writes_black_by_decision(app: Any) -> None:
    document_id, document = _chain(app)
    assert app.unwire(document_id, "c", "u_src") == ""
    assert document.passes["c"].uniform_values["u_src"] == NoSource()


def test_group_selection_and_dissolve_are_one_write_each(app: Any) -> None:
    document_id, document = _chain(app)
    view = app.graph_view_for(document_id)
    view.selection = {"a", "b"}
    with mock.patch.object(
        app.session, "save_ui_document", wraps=app.session.save_ui_document
    ) as saves:
        assert app.group_selection(document_id, "pair") == ""
    assert saves.call_count == 1
    assert document.graph.passes["a"].group == "pair"
    assert document.graph.passes["b"].group == "pair"
    assert document.graph.passes["c"].group == ""
    # Grouping a box with a plain pass rewrites the members to the new label (flat labels).
    view.selection = {"a", "b", "c"}
    assert app.group_selection(document_id, "trio") == ""
    assert {document.graph.passes[n].group for n in ("a", "b", "c")} == {"trio"}
    # A name a pass carries is refused and nothing changes.
    assert (
        app.group_selection(document_id, "a")
        == "a pass and a group cannot share a name"
    )
    assert {document.graph.passes[n].group for n in ("a", "b", "c")} == {"trio"}
    with mock.patch.object(
        app.session, "save_ui_document", wraps=app.session.save_ui_document
    ) as saves:
        assert app.dissolve_group(document_id, "trio") == ""
    assert saves.call_count == 1
    assert all(document.graph.passes[n].group == "" for n in ("a", "b", "c"))


def test_arrange_graph_saves_once_and_leaves_no_pass_unplaced(app: Any) -> None:
    document_id, document = _chain(app)
    with mock.patch.object(
        app.session, "save_ui_document", wraps=app.session.save_ui_document
    ) as saves:
        app.arrange_graph(document_id)
    assert saves.call_count == 1
    assert all(entry.position is not None for entry in document.graph.passes.values())
    assert app.graph_view_for(document_id).fitted is False


def test_dissolve_of_nothing_and_leave_group_write_as_they_should(app: Any) -> None:
    document_id, document = _chain(app)
    with mock.patch.object(
        app.session, "save_ui_document", wraps=app.session.save_ui_document
    ) as saves:
        assert app.dissolve_group(document_id, "") == ""
        assert app.dissolve_group(document_id, "nope") == ""
    assert saves.call_count == 0, "a dissolve of nothing saved"
    assert app.session.set_pass_group(document_id, "a", "g") == ""
    assert app.leave_group(document_id, "a") == ""
    assert document.graph.passes["a"].group == ""


def test_the_widget_makes_no_session_write_of_its_own() -> None:
    # 092 D12: every write goes through an App verb, so its refusal is testable here. The
    # strip's menu is shared with the canvas, so it is held to the same rule.
    for module in (pass_graph, pass_list):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "set_sampler_source",
            "set_pass_positions",
            "set_pass_groups",
            "set_pass_group(",
        ):
            assert forbidden not in source, (module.__name__, forbidden)


# ----------------------------------------------------------------
# The one gesture the verbs cannot cover: a drop lands where the port is DRAWN. Driven through
# the real frame loop with synthetic mouse input, so the gate sees the hover that feeds `_drop`.


def _frames(app: Any, n: int) -> None:
    for _ in range(n):
        update_and_draw(app)


# ----------------------------------------------------------------
# The canvas's own gestures (093), driven through the real frame loop. Three mechanics every
# test here obeys, each measured on this imgui build: a key event queued in the same batch as
# a mouse-button event reaches `is_key_pressed` one frame LATER than the button, so a `_frames`
# call separates them; a move and a release arriving in one frame read as a click, so they are
# separate frames too; and a test that needs the canvas to have drawn asserts `canvas_rect`
# first, since every geometry field it then reads is written only on a sized frame.


def _open_graph(app: Any, document_id: str) -> Any:
    app.open_graph_for(document_id)
    _frames(app, 4)
    view = app.graph_view_for(document_id)
    assert view.canvas_rect != (0.0, 0.0, 0.0, 0.0), "the canvas never drew"
    return view


def _park(app: Any, point: tuple[float, float], frames: int = 3) -> None:
    imgui.get_io().add_mouse_pos_event(point[0], point[1])
    _frames(app, frames)


def _click_at(app: Any, point: tuple[float, float]) -> None:
    _park(app, point)
    imgui.get_io().add_mouse_button_event(0, True)
    _frames(app, 2)
    imgui.get_io().add_mouse_button_event(0, False)
    _frames(app, 2)


def _let_the_double_click_lapse(app: Any) -> None:
    """Run frames until imgui's double-click window has closed.

    A test's two presses land microseconds apart on almost the same pixel, which imgui reads
    as ONE double-click -- a real hand aiming from a wire to its badge never does. Frames are
    free here, and the alternative (moving the mouse far away between presses) would change
    what the gesture under test is."""
    deadline = imgui.get_io().mouse_double_click_time + 0.05
    elapsed = 0.0
    while elapsed < deadline:
        elapsed += imgui.get_io().delta_time
        _frames(app, 1)


def _press_key(app: Any, key: Any) -> None:
    imgui.get_io().add_key_event(key, True)
    _frames(app, 2)
    imgui.get_io().add_key_event(key, False)
    _frames(app, 1)


def _close_graph(app: Any, document_id: str) -> None:
    app.close_editor_for_path(app.paths.graph_json_for(document_id))
    _frames(app, 1)


def _hover_fields(view: Any) -> tuple[Any, Any, Any, Any]:
    return (view.hovered_port, view.hovered_out, view.hovered_node, view.hovered_wire)


def _drag_node(app: Any, view: Any, start: tuple[float, float], dx: float) -> None:
    # The mouse is walked back to `start` with the button UP first: a second drag begun from
    # wherever the previous one ended would carry that offset into imgui's drag delta and the
    # two cases would not measure what they name.
    _park(app, start)
    imgui.get_io().add_mouse_button_event(0, True)
    _frames(app, 2)
    # The move and the release in SEPARATE frames: in one frame imgui resets the drag before
    # the frame runs and the pair reads as a click (measured).
    _park(app, (start[0] + dx, start[1]))
    imgui.get_io().add_mouse_button_event(0, False)
    _frames(app, 3)


_TYPED = """#version 460 core
in vec2 vs_uv;
uniform int u_steps;
uniform float u_gain;
uniform vec2 u_shift;
out vec4 fs_color;
void main() {
    fs_color = vec4(vs_uv + u_shift, float(u_steps) * u_gain, 1.0);
}
"""


def test_a_dragged_int_uniform_survives_the_next_render(app: Any) -> None:
    """A canvas drag reports a FLOAT whatever the uniform's GL type, and
    moderngl refuses a float written to an int (`required argument is not
    an integer`). `Pass.render` catches that by POPPING the cached value,
    which re-seeds the uniform to 0 -- so an unshaped write does not merely
    fail, it destroys what was there.

    Measured before the fix: dragging from 34 wrote 34.45 and the next
    frame read 0. Asserted after a real render, because the write itself
    looks fine and only the render is where the value dies.
    """
    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    assert app.session.add_pass(document_id, "typed") == ""
    render_pass = document.passes["typed"]
    render_pass.release_program(_TYPED)
    render_pass.compile()
    assert render_pass.program is not None, "the fixture shader did not compile"
    # Seeding rides a LAZY compile, and this fixture compiled explicitly, so
    # the seed is called directly -- otherwise the values a consumer relies
    # on are simply absent.
    render_pass.seed_uniform_values()
    assert render_pass.uniform_values["u_steps"] == 0

    app.set_graph_uniform(document_id, "typed", "u_steps", (34.45,))
    assert render_pass.uniform_values["u_steps"] == 34, (
        "an int uniform was written a float, which moderngl refuses"
    )
    document.render()
    assert render_pass.uniform_values["u_steps"] == 34, (
        "the render dropped the value, so the write was the wrong shape"
    )

    # The float beside it keeps its fraction: the coercion is per GL type,
    # not a blanket round.
    app.set_graph_uniform(document_id, "typed", "u_gain", (0.25,))
    document.render()
    assert render_pass.uniform_values["u_gain"] == pytest.approx(0.25)


def test_a_dragged_uniform_of_the_wrong_arity_is_refused(app: Any) -> None:
    """The library reports as many components as the widget drew. A value
    of the wrong width would be refused by moderngl at draw time and take
    the cached value down with it, so it is refused here instead."""
    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    assert app.session.add_pass(document_id, "typed") == ""
    render_pass = document.passes["typed"]
    render_pass.release_program(_TYPED)
    render_pass.compile()
    render_pass.seed_uniform_values()
    before = render_pass.uniform_values["u_shift"]

    app.set_graph_uniform(document_id, "typed", "u_shift", (1.0,))
    assert render_pass.uniform_values["u_shift"] == before, (
        "a one-component value was written into a vec2"
    )
    # And a uniform the pass does not declare reaches nothing.
    app.set_graph_uniform(document_id, "typed", "u_absent", (1.0,))
    assert "u_absent" not in render_pass.uniform_values
