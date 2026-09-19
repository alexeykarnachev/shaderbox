"""Scratch probes for the graph-canvas audit. Not a committed test."""

from typing import Any

import pytest

from shaderbox.graph_canvas import ffi
from shaderbox.graph_canvas.adapter import Moved, read_events
from shaderbox.ui import update_and_draw

pytestmark = pytest.mark.xdist_group("gl_frames_graph_tab")

_SAMPLER = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_src;
uniform float u_gain;
uniform vec2 u_shift;
out vec4 fs_color;
void main() { fs_color = texture(u_src, vs_uv + u_shift) * u_gain; }
"""


def _chain(app: Any) -> tuple[str, Any]:
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


def _drive(state: Any, size: Any, x: float, y: float, flags: int) -> Any:
    return state.canvas.frame(
        state.packed.nodes,
        state.packed.edges,
        (float(size[0]), float(size[1])),
        state.view,
        ffi.PointerState(x=x, y=y, flags=flags),
    )


def _apply_moves(state: Any, document: Any, events: Any) -> None:
    """The Moved half of `_apply_graph_events`, verbatim. The commit branch keys
    on the real mouse, which is up in a probe, so it is left out."""
    for event in events:
        if not isinstance(event, Moved):
            continue
        if event.members:
            anchor = state.box_anchors.get(event.key)
            if anchor is None:
                anchor = (event.x, event.y)
                state.box_anchors[event.key] = anchor
                state.box_members[event.key] = {
                    n: document.graph.passes[n].position or (0.0, 0.0)
                    for n in event.members
                    if n in document.graph.passes
                }
            dx = event.x - anchor[0]
            dy = event.y - anchor[1]
            for n, (px, py) in state.box_members.get(event.key, {}).items():
                state.dragging[n] = (px + dx, py + dy)
        else:
            state.dragging[event.name] = (event.x, event.y)


def _box_drag(app: Any, document_id: str, document: Any, steps: int = 3) -> Any:
    state = app.graph_canvases[document_id]
    packed = state.packed
    size = state.panel.texture.size
    box_index = next(i for i, v in enumerate(packed.views) if v.is_box)
    r = _drive(state, size, -1e6, -1e6, 0)
    rect = r.node_rects[box_index]
    x = rect.x + rect.w * 0.5
    y = rect.y + 6.0
    for _ in range(3):
        _drive(state, size, x, y, 0)
    _drive(state, size, x, y, int(ffi.Pointer.DOWN) | int(ffi.Pointer.PRESSED))
    for k in range(1, steps + 1):
        rr = _drive(state, size, x + 10.0 * k, y + 10.0 * k, int(ffi.Pointer.DOWN))
        _apply_moves(state, document, read_events(rr, packed))
    return state


def test_probe_box_drag_unplaced_member(app: Any) -> None:
    """P2: a box's drag freezes each member's STORED position, defaulting to (0,0)
    for a member never placed — but the canvas DRAWS such a member at its rank
    layout position."""
    document_id, document = _chain(app)
    app.open_graph_for(document_id)
    view = app.graph_view_for(document_id)
    view.selection = {"a", "b"}
    assert app.group_selection(document_id, "pair") == ""
    _frames(app, 3)
    entries = document.graph.passes
    stored = {n: entries[n].position for n in document.passes if n in entries}
    print("STORED:", stored)

    # What the canvas drew them at before the drag: inside the group's own tab,
    # where the members are their own nodes.
    view.scope = "pair"
    _frames(app, 3)
    inner = app.graph_canvases[document_id].packed
    before = {v.name: (n.x, n.y) for v, n in zip(inner.views, inner.nodes)}
    print("DRAWN BEFORE (inside the tab):", before)

    view.scope = ""
    _frames(app, 3)
    state = _box_drag(app, document_id, document)
    print("DRAGGING after a ~30px box drag:", state.dragging)
    print("BOX MEMBERS frozen:", state.box_members)


def test_probe_box_drag_commits_the_collapse(app: Any) -> None:
    """P2b: does the collapse reach disk? Drive the real commit verb with what a
    drag accumulated."""
    document_id, document = _chain(app)
    app.open_graph_for(document_id)
    view = app.graph_view_for(document_id)
    view.selection = {"a", "b"}
    assert app.group_selection(document_id, "pair") == ""
    _frames(app, 3)
    state = _box_drag(app, document_id, document)
    moved = dict(state.dragging)
    app.commit_graph_positions(document_id, moved)
    entries = document.graph.passes
    print("SAVED:", {n: entries[n].position for n in ("a", "b", "c")})
    view.scope = "pair"
    _frames(app, 3)
    inner = app.graph_canvases[document_id].packed
    print(
        "DRAWN AFTER:",
        {v.name: (n.x, n.y) for v, n in zip(inner.views, inner.nodes)},
    )
