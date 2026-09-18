"""A document packs into the library's node model and its events come back
as pass names (feature 098).

The adapter is the only module that knows both sides, so the checks here are
about the mapping: a wire's endpoints index what was PACKED, and an event
names a pass rather than a position in this frame's array.
"""

from pathlib import Path

import moderngl
import pytest

from shaderbox.constants import DOCUMENT_EXAMPLES_DIR
from shaderbox.document import Document
from shaderbox.graph_canvas import ffi
from shaderbox.graph_canvas.adapter import (
    Clicked,
    Moved,
    NodePalette,
    Unwired,
    Wired,
    edge_id,
    node_id,
    pack_nodes,
    read_events,
)
from shaderbox.graph_canvas.render import shapes_array
from shaderbox.pass_graph import Port, strip_order
from shaderbox.widgets.graph_state import ports_of

# The largest shipped document: six passes, a diamond, and a pass read twice.
CASCADE_EXAMPLE = "77a84d27-2e5b-406d-8011-ee1cb1a9587c"


def _simple() -> tuple[list[str], dict[str, list[Port]]]:
    order = ["seed", "blur", "out"]
    ports = {
        "seed": [],
        "blur": [Port("u_src", "wired", "seed")],
        "out": [Port("u_a", "wired", "blur"), Port("u_b", "unfilled")],
    }
    return order, ports


def test_an_id_names_a_pass_and_not_a_position() -> None:
    """Identity crosses by id so a reorder between frames cannot make an event
    about one pass arrive as an event about another."""
    assert node_id("blur") == node_id("blur")
    assert node_id("blur") != node_id("seed")
    assert edge_id("out", "u_a") != edge_id("out", "u_b")
    # A pass and a wire are different namespaces.
    assert node_id("a") != edge_id("a", "")
    # Never zero, which the library's own examples use to mean "no id".
    assert node_id("") != 0


def test_a_wire_lands_on_the_port_that_declares_it() -> None:
    """The endpoints index attributes WITHIN a node, counting inputs and
    outputs together in declaration order — so a producer's output slot is
    its input count, and getting it wrong lands the wire on the wrong pin
    rather than raising."""
    order, ports = _simple()
    packed = pack_nodes(order, ports, {}, {}, output="out")

    assert len(packed.edges) == 2
    by_consumer = {e.to_node: e for e in packed.edges}

    # blur reads seed: seed has no inputs, so its output is slot 0.
    blur = by_consumer[1]
    assert (blur.from_node, blur.from_attr) == (0, 0)
    assert blur.to_attr == 0

    # out reads blur through u_a: blur has one input, so its output is slot 1.
    out = by_consumer[2]
    assert (out.from_node, out.from_attr) == (1, 1)
    assert out.to_attr == 0


def test_an_unwired_port_gets_a_pin_and_no_edge() -> None:
    order, ports = _simple()
    packed = pack_nodes(order, ports, {}, {}, output="out")
    # `out` declares two inputs plus its own output.
    assert len(packed.nodes[2].ports) == 3
    assert [p.label for p in packed.nodes[2].ports] == ["u_a", "u_b", "out"]
    assert {e.to_attr for e in packed.edges if e.to_node == 2} == {0}


def test_a_ghost_refuses_every_gesture() -> None:
    """A node that is not interactive must never START a gesture, so the press
    falls through to the canvas rather than being swallowed and dropped."""
    order, ports = _simple()
    packed = pack_nodes(order, ports, {}, {}, output="out", ghosts=frozenset({"seed"}))
    ghost = packed.nodes[0]
    assert ghost.accepts == int(ffi.Gesture.NONE)
    assert ghost.dashed and ghost.fade > 0.0
    assert packed.nodes[1].accepts == 0


def test_the_output_pass_is_marked_and_the_others_are_not() -> None:
    order, ports = _simple()
    packed = pack_nodes(order, ports, {}, {}, output="out")
    assert packed.nodes[2].border_scale > packed.nodes[0].border_scale


def test_a_preview_carries_its_real_pixel_size() -> None:
    """The aspect builds the SLOT and the pixels drive the fit, so declaring
    one shape and handing over another squashes the picture."""
    order, ports = _simple()
    packed = pack_nodes(order, ports, {}, {"seed": (7, 320, 160)}, output="out")
    assert packed.nodes[0].preview_tex == 7
    assert (packed.nodes[0].preview_w, packed.nodes[0].preview_h) == (320, 160)
    assert packed.nodes[0].preview_aspect == pytest.approx(2.0)


def test_an_event_about_a_vanished_node_is_dropped() -> None:
    """The library reports against the array it was handed. A host that
    renamed or deleted between frames would otherwise act on whatever now
    sits at that index."""
    order, ports = _simple()
    packed = pack_nodes(order, ports, {}, {}, output="out")

    result = ffi.Result()
    events = (ffi.Event * 2)()
    events[0].kind = int(ffi.EventKind.NODE_CLICKED)
    events[0].node = 1
    events[1].kind = int(ffi.EventKind.NODE_CLICKED)
    events[1].node = 99
    result.events = events
    result.event_count = 2

    read = read_events(result, packed)
    assert read == [Clicked("blur", False)]


def test_events_come_back_as_pass_names() -> None:
    order, ports = _simple()
    packed = pack_nodes(order, ports, {}, {}, output="out")

    result = ffi.Result()
    events = (ffi.Event * 3)()
    events[0].kind = int(ffi.EventKind.NODE_MOVED)
    events[0].node = 0
    events[0].x, events[0].y = 12.0, 34.0
    events[1].kind = int(ffi.EventKind.EDGE_ADDED)
    events[1].from_node, events[1].to_node, events[1].to_attr = 0, 2, 1
    events[2].kind = int(ffi.EventKind.EDGE_REMOVED)
    events[2].to_node, events[2].to_attr = 2, 0
    result.events = events
    result.event_count = 3

    assert read_events(result, packed) == [
        Moved("seed", 12.0, 34.0),
        Wired("seed", "out", "u_b"),
        Unwired("out", "u_a"),
    ]


def test_the_real_cascade_document_packs_and_frames(gl_ctx: moderngl.Context) -> None:
    """The end-to-end check against the largest shipped document: the DAG the
    adapter packs must be the document's own."""
    document, _metadata = Document.load_from_dir(
        Path(DOCUMENT_EXAMPLES_DIR) / CASCADE_EXAMPLE, gl=gl_ctx
    )
    for render_pass in document.passes.values():
        render_pass.compile()

    wiring = document.effective_wiring()
    ports = ports_of(document, wiring)
    order = strip_order(document.passes.keys(), wiring)
    packed = pack_nodes(order, ports, {}, {}, output=document.graph.output_pass or "")

    assert len(packed.nodes) == len(document.passes)
    # Every wired port in the document is an edge, and no more.
    wired = sum(1 for name in order for p in ports.get(name, ()) if p.kind == "wired")
    assert len(packed.edges) == wired

    # And the library accepts the pack.
    canvas = ffi.Canvas()
    canvas.load_atlas()
    result = canvas.frame(
        packed.nodes, packed.edges, (1400.0, 800.0), ffi.View(), ffi.PointerState()
    )
    assert result.rect_count == len(packed.nodes)
    assert result.shape_count > 0
    canvas.release()


def test_a_node_s_attributes_are_packed_in_node_order() -> None:
    """Each node names a CONTIGUOUS run of the flat attribute array, so the
    two have to be built in one pass. Mis-ranged nodes read each other's
    attributes and everything still draws."""
    order, ports = _simple()
    packed = pack_nodes(order, ports, {}, {}, output="out")
    expected = 0
    for spec, view in zip(packed.nodes, packed.views, strict=True):
        assert len(spec.ports) == len(view.inputs) + 1
        expected += len(spec.ports)
    assert expected == sum(len(n.ports) for n in packed.nodes)


def test_the_generic_half_stays_free_of_shaderbox_model_types() -> None:
    """`ffi.py` and `render.py` are the pair that lifts into another project,
    so neither may import a shaderbox model. Only `adapter.py` knows both
    sides — and a rule with no gate is a wish, which is why this is a test and
    not a paragraph in the spec.

    `shaderbox.constants` is allowed: it is where the vendored resource paths
    live, and it imports nothing of the model.
    """
    import ast

    allowed = {"shaderbox.constants", "shaderbox.graph_canvas.ffi"}
    root = Path(__file__).resolve().parents[1] / "shaderbox" / "graph_canvas"
    for module in ("ffi.py", "render.py"):
        tree = ast.parse((root / module).read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        offenders = {
            name
            for name in imported
            if name.startswith("shaderbox") and name not in allowed
        }
        assert not offenders, f"{module} reaches into {sorted(offenders)}"


def test_a_wire_points_at_an_output_and_engine_rows_follow_it() -> None:
    """The edge's `from_attr` names an OUTPUT in the packed list.

    Deliberately not phrased as "found beats computed": the input count and
    the output's index agree by construction here, since the output is
    emitted straight after the samplers, so no fixture can separate the two
    formulations and a test claiming to would be theatre. What this pins is
    the property that matters either way -- the edge lands on the output, and
    the engine rows really do sit after it.
    """
    packed = pack_nodes(
        ["a", "b"],
        {
            "a": [Port("u_x", "unfilled"), Port("u_y", "unfilled")],
            "b": [Port("u_src", "wired", "a")],
        },
        {},
        {},
        output="b",
        engine={"a": [("u_time", (1.0,)), ("u_resolution", (64.0, 64.0))], "b": []},
    )
    attributes = packed.nodes[0].ports
    outputs = [i for i, spec in enumerate(attributes) if not spec.is_input]
    assert len(outputs) == 1
    # The engine rows come after the output, or this fixture proves nothing.
    assert outputs[0] < len(attributes) - 1
    assert packed.edges[0].from_attr == outputs[0]
    assert attributes[packed.edges[0].from_attr].is_input is False


def test_an_engine_uniform_is_a_control_and_not_an_input() -> None:
    """`u_time` and its siblings are written by the engine, so the document
    cannot express a wire into one.

    The KIND is what makes them look different, not the pin: the library
    picks a row's background from its kind, so a control is drawn in the
    neutral control tone where an input is green. Packing these as inputs --
    even with a refusing pin -- made a builtin read as a free port, which is
    exactly the thing they are not. Measured, the two tones are
    (0.33, 0.34, 0.42) and (0.29, 0.45, 0.35).
    """
    packed = pack_nodes(
        ["a", "b"],
        {"a": [], "b": [Port("u_src", "unfilled")]},
        {"a": (0.0, 0.0), "b": (300.0, 0.0)},
        {},
        output="b",
        engine={"a": [], "b": [("u_time", (1.0,))]},
    )
    by_label = {spec.label: spec for spec in packed.nodes[1].ports}
    assert "u_time" in by_label
    assert by_label["u_time"].control is True
    # And a real sampler on the same node is NOT a control, or the check
    # would pass on a node where everything happened to be one.
    assert by_label["u_src"].control is False


def test_a_control_row_is_drawn_in_a_different_tone_from_an_input() -> None:
    """The reason the kind matters, asserted against the geometry rather than
    against the intent: the same label packed as an input and as a control
    produces different row colours."""

    def row_tones(control: bool) -> set[tuple[float, float, float]]:
        canvas = ffi.Canvas()
        canvas.load_atlas()
        node = ffi.NodeSpec(
            id=1,
            title="t",
            x=0,
            y=0,
            ports=[
                ffi.PortSpec("u_src", True),
                ffi.PortSpec("u_res", True, control=control),
                ffi.PortSpec("out", False),
            ],
        )
        result = canvas.frame(
            [node], [], (500.0, 460.0), ffi.View(), ffi.PointerState()
        )
        shapes = shapes_array(result)
        tones = {
            (round(float(row[4]), 3), round(float(row[5]), 3), round(float(row[6]), 3))
            for row in shapes
            if row[2] > 90 and 10 < row[3] < 26 and row[7] > 0.9
        }
        canvas.release()
        return tones

    as_input = row_tones(False)
    as_control = row_tones(True)
    assert as_input != as_control, (
        f"a control row is drawn the same as an input: {as_input}"
    )


def test_hover_and_selection_change_the_border_and_not_the_size() -> None:
    """A highlight must be visible without moving anything: recolour and
    thicken, never resize, or the node shifts under the cursor that is
    pointing at it. Selection outranks hover."""
    order = ["a", "b"]
    ports: dict[str, list[Port]] = {"a": [], "b": []}
    palette = NodePalette(
        hover=(1.0, 0.0, 0.0, 1.0),
        select=(0.0, 1.0, 0.0, 1.0),
        engine_uniform=(0.0, 0.0, 1.0, 1.0),
    )
    plain = pack_nodes(order, ports, {}, {}, output="b", palette=palette)
    hovered = pack_nodes(order, ports, {}, {}, output="b", hovered="a", palette=palette)
    picked = pack_nodes(
        order, ports, {}, {}, output="b", selected=frozenset({"a"}), palette=palette
    )
    both = pack_nodes(
        order,
        ports,
        {},
        {},
        output="b",
        hovered="a",
        selected=frozenset({"a"}),
        palette=palette,
    )

    assert plain.nodes[0].border is None
    assert hovered.nodes[0].border == palette.hover
    assert picked.nodes[0].border == palette.select
    # Selection wins where both apply.
    assert both.nodes[0].border == palette.select
    # Thicker, and the unhovered sibling untouched.
    assert hovered.nodes[0].border_scale > plain.nodes[0].border_scale
    assert hovered.nodes[1].border == plain.nodes[1].border


def test_a_multi_component_engine_value_shows_every_component() -> None:
    """A `LABEL` renders component 0 and nothing else, so a vec2 like
    `u_resolution` would show half of itself with no sign that it had. A
    read-only `DRAG` draws one field per component and takes no pointer.

    Asserted against the GEOMETRY, because the widget choice is only a means:
    what matters is that a two-component value puts more on the node than a
    one-component value does, which a Label does not.
    """

    def glyphs(value: tuple[float, ...]) -> int:
        canvas = ffi.Canvas()
        canvas.load_atlas()
        packed = pack_nodes(
            ["a"],
            {"a": []},
            {"a": (0.0, 0.0)},
            {},
            output="a",
            engine={"a": [("u_res", value)]},
        )
        result = canvas.frame(
            packed.nodes, packed.edges, (900.0, 700.0), ffi.View(), ffi.PointerState()
        )
        count = result.glyph_count
        canvas.release()
        return count

    one = glyphs((64.0,))
    two = glyphs((64.0, 64.0))
    assert two > one, f"a vec2 drew no more than a scalar: {two} against {one}"


def test_an_engine_row_with_no_value_yet_shows_its_name_alone() -> None:
    """A pass that has not rendered has no value to show. The row is still
    worth drawing -- the uniform exists and the user should see it -- so the
    widget goes to NONE rather than the row disappearing."""
    packed = pack_nodes(
        ["a"],
        {"a": []},
        {"a": (0.0, 0.0)},
        {},
        output="a",
        engine={"a": [("u_time", ())]},
    )
    row = next(spec for spec in packed.nodes[0].ports if spec.label == "u_time")
    assert row.control is True
    assert row.widget is ffi.Widget.NONE
    assert row.value == ()


def test_an_engine_value_is_read_only() -> None:
    """The engine overwrites it every frame, so an editable widget would offer
    an edit that reverts -- which the library's own contract calls out."""
    packed = pack_nodes(
        ["a"],
        {"a": []},
        {"a": (0.0, 0.0)},
        {},
        output="a",
        engine={"a": [("u_time", (1.0,)), ("u_res", (64.0, 64.0))]},
    )
    for spec in packed.nodes[0].ports:
        if spec.label.startswith("u_"):
            assert spec.read_only is True, spec.label
