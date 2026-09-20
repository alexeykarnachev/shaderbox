"""A document packs into the library's node model and its events come back
as pass names (feature 098).

The adapter is the only module that knows both sides, so the checks here are
about the mapping: a wire's endpoints index what was PACKED, and an event
names a pass rather than a position in this frame's array.
"""

import colorsys
from pathlib import Path

import moderngl
import numpy as np
import pytest

from shaderbox.constants import DOCUMENT_EXAMPLES_DIR
from shaderbox.document import Document
from shaderbox.graph_canvas import ffi
from shaderbox.graph_canvas.adapter import (
    BodyRow,
    Clicked,
    Moved,
    NodePalette,
    Unwired,
    ValueEdited,
    Wired,
    _halos_of,
    _widget_for,
    edge_id,
    flat_view,
    node_id,
    pack_nodes,
    pass_key,
    read_events,
)
from shaderbox.graph_canvas.ffi import GRAPH_CANVAS_RESOURCES_DIR
from shaderbox.graph_canvas.render import shapes_array
from shaderbox.intel.symbols import SymbolKind
from shaderbox.pass_graph import Port, strip_order
from shaderbox.theme import kind_color
from shaderbox.widgets.graph_state import ports_of
from shaderbox.widgets.pass_graph import _ring_palette, canvas_theme

# One colour for every body row in these fixtures: what the row IS
# coloured is the host's decision and is tested where it is made.
_BLUE: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

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
    packed = pack_nodes(flat_view(order, ports, {}), {}, output=pass_key("out"))

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
    packed = pack_nodes(flat_view(order, ports, {}), {}, output=pass_key("out"))
    # `out` declares two inputs plus its own output.
    assert len(packed.nodes[2].ports) == 3
    assert [p.label for p in packed.nodes[2].ports] == ["u_a", "u_b", "out"]
    assert {e.to_attr for e in packed.edges if e.to_node == 2} == {0}


def test_a_ghost_refuses_every_gesture() -> None:
    """A node that is not interactive must never START a gesture, so the press
    falls through to the canvas rather than being swallowed and dropped."""
    order, ports = _simple()
    packed = pack_nodes(
        flat_view(order, ports, {}, frozenset({"seed"})), {}, output=pass_key("out")
    )
    ghost = packed.nodes[0]
    assert ghost.accepts == int(ffi.Gesture.NONE)
    assert ghost.dashed and ghost.fade > 0.0
    assert packed.nodes[1].accepts == 0


def test_the_output_pass_is_marked_and_the_others_are_not() -> None:
    order, ports = _simple()
    packed = pack_nodes(flat_view(order, ports, {}), {}, output=pass_key("out"))
    assert packed.nodes[2].border_scale > packed.nodes[0].border_scale


def test_a_preview_carries_its_real_pixel_size() -> None:
    """The aspect builds the SLOT and the pixels drive the fit, so declaring
    one shape and handing over another squashes the picture."""
    order, ports = _simple()
    packed = pack_nodes(
        flat_view(order, ports, {}), {"seed": (7, 320, 160)}, output=pass_key("out")
    )
    assert packed.nodes[0].preview_tex == 7
    assert (packed.nodes[0].preview_w, packed.nodes[0].preview_h) == (320, 160)
    assert packed.nodes[0].preview_aspect == pytest.approx(2.0)


def test_an_event_about_a_vanished_node_is_dropped() -> None:
    """The library reports against the array it was handed. A host that
    renamed or deleted between frames would otherwise act on whatever now
    sits at that index."""
    order, ports = _simple()
    packed = pack_nodes(flat_view(order, ports, {}), {}, output=pass_key("out"))

    result = ffi.Result()
    events = (ffi.Event * 2)()
    events[0].kind = int(ffi.EventKind.NODE_CLICKED)
    events[0].node = 1
    events[1].kind = int(ffi.EventKind.NODE_CLICKED)
    events[1].node = 99
    result.events = events
    result.event_count = 2

    read = read_events(result, packed)
    assert read == [Clicked(pass_key("blur"), "blur", False)]


def test_events_come_back_as_pass_names() -> None:
    order, ports = _simple()
    packed = pack_nodes(flat_view(order, ports, {}), {}, output=pass_key("out"))

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
        Moved(pass_key("seed"), "seed", 12.0, 34.0),
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
    packed = pack_nodes(
        flat_view(order, ports, {}),
        {},
        output=pass_key(document.graph.output_pass or ""),
    )

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
    packed = pack_nodes(flat_view(order, ports, {}), {}, output=pass_key("out"))
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
        flat_view(
            ["a", "b"],
            {
                "a": [Port("u_x", "unfilled"), Port("u_y", "unfilled")],
                "b": [Port("u_src", "wired", "a")],
            },
            {},
        ),
        {},
        output=pass_key("b"),
        body={
            "a": [
                BodyRow("u_time", (1.0,), _BLUE),
                BodyRow("u_resolution", (64.0, 64.0), _BLUE),
            ],
            "b": [],
        },
    )
    attributes = packed.nodes[0].ports
    # `not is_input` is not "is an output": a CONTROL is neither, which is
    # what makes it pinless. An output is the row that is neither an input
    # nor a control.
    outputs = [
        i for i, spec in enumerate(attributes) if not spec.is_input and not spec.control
    ]
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
        flat_view(
            ["a", "b"],
            {"a": [], "b": [Port("u_src", "unfilled")]},
            {"a": (0.0, 0.0), "b": (300.0, 0.0)},
        ),
        {},
        output=pass_key("b"),
        body={"a": [], "b": [BodyRow("u_time", (1.0,), _BLUE)]},
    )
    by_label = {spec.label: spec for spec in packed.nodes[1].ports}
    assert "u_time" in by_label
    assert by_label["u_time"].control is True
    # And a real sampler on the same node is NOT a control, or the check
    # would pass on a node where everything happened to be one.
    assert by_label["u_src"].control is False


def test_an_engine_row_is_packed_as_a_control_and_a_sampler_is_not() -> None:
    """The adapter's half of the claim: an engine uniform goes over as a
    CONTROL and a sampler as a plain input.

    Split from the tone check below because the two need different
    evidence. A rendered-tone test cannot isolate this flag -- an engine row
    also carries a widget and sits after the output, and either difference
    moves the colour on its own, so the tone differs whatever `control`
    says. Measured: flipping `control` to False changed no pixel the filter
    could see. What can decide it is the packed attribute itself.
    """
    packed = pack_nodes(
        flat_view(["t"], {"t": [Port("u_src", "unfilled")]}, {"t": (0.0, 0.0)}),
        {},
        output=pass_key("t"),
        body={"t": [BodyRow("u_time", (1.0,), _BLUE)]},
    )
    by_label = {port.label: port for port in packed.nodes[0].ports}
    assert by_label["u_time"].control, "an engine uniform was packed as a free input"
    assert by_label["u_time"].read_only, "an engine uniform was packed as editable"
    assert not by_label["u_src"].control, "a sampler was packed as a control"


def test_the_control_kind_alone_changes_the_row_s_tone() -> None:
    """The library's half: the row background comes from the attribute's
    KIND, which is WHY the adapter sets it. Two attributes identical in
    every other field, differing only in `control`.

    Built at the library boundary on purpose -- this is a claim about the
    library's rendering, and routing it through `pack_nodes` would make
    every other field differ too and prove nothing about this one.
    """

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
    assert as_input, "no row was measured, so the filter found nothing to compare"
    assert as_input != as_control, (
        f"a control row is drawn the same as an input: {as_input}"
    )


def test_a_state_ring_carries_its_colour_all_the_way_to_the_pixels(
    gl_ctx: moderngl.Context,
) -> None:
    """Selection, hover and the output are marked by a coloured RING, and
    the colour has to survive to the screen.

    A border cannot carry it: the vertex format packs a border as one
    LUMINANCE, so pure red and pure blue render byte-identical and this
    palette's purple selection and yellow accent land 0.654 against 0.635 --
    not a distinction anyone makes. Measured in the shape stream, which is
    also where an earlier version of this check went wrong by reading the
    column beside it.

    So this asserts PIXELS as well as the packed spec: two runs differing
    only in the ring's colour must differ on screen, and a spec-only
    assertion is exactly what let an invisible border ship.
    """
    order = ["a", "b"]
    ports: dict[str, list[Port]] = {"a": [], "b": []}

    def frame(colour: tuple[float, float, float, float]) -> np.ndarray:
        packed = pack_nodes(
            flat_view(order, ports, {"a": (0.0, 0.0), "b": (240.0, 0.0)}),
            {},
            output=pass_key("b"),
            selected=frozenset({pass_key("a")}),
            palette=NodePalette(hover=(1.0, 1.0, 1.0, 1.0), select=colour),
        )
        canvas = ffi.Canvas()
        canvas.load_atlas()
        result = canvas.frame(
            packed.nodes,
            packed.edges,
            (520.0, 400.0),
            ffi.View(),
            ffi.PointerState(),
            theme=canvas_theme(),
        )
        shapes = shapes_array(result).copy()
        canvas.release()
        return shapes

    red = frame((1.0, 0.0, 0.0, 1.0))
    blue = frame((0.0, 0.0, 1.0, 1.0))
    assert red.shape == blue.shape
    differing = int((np.abs(red - blue).sum(axis=1) > 0).sum())
    assert differing > 0, (
        "red and blue rings produced an identical scene, so the ring's "
        "colour is being discarded the way a border's is"
    )


def test_the_output_is_marked_in_the_title_not_a_second_ring() -> None:
    """The document's output is a PROPERTY, so it survives every state.

    It used to be a concentric second ring, which meant a hovered output
    node wore two marks and the two had to stay mutually legible against
    three interaction states -- an unsatisfiable constraint that produced
    a selection colour nobody wanted. In the title it is unaffected by
    what the pointer is doing, which is the point.
    """
    order = ["a", "b"]
    ports: dict[str, list[Port]] = {"a": [], "b": []}
    palette = NodePalette(
        hover=(1.0, 0.0, 0.0, 1.0),
        select=(0.0, 1.0, 0.0, 1.0),
        failing=(0.0, 0.0, 1.0, 1.0),
    )

    def packed(**kwargs: object):
        return pack_nodes(
            flat_view(order, ports, {}),
            {},
            output=pass_key("a"),
            palette=palette,
            **kwargs,  # type: ignore[arg-type]
        )

    out_node = packed().nodes[0]
    other = packed().nodes[1]
    assert out_node.title != other.title, "the output node is not marked at all"
    assert other.title == "b", f"a non-output node was marked: {other.title}"
    assert "a" in out_node.title

    # The mark survives every state, which is what a colour could not do.
    for kwargs in (
        {"selected": frozenset({pass_key("a")})},
        {"hovered": pass_key("a")},
        {"failing": frozenset({pass_key("a")})},
    ):
        assert packed(**kwargs).nodes[0].title == out_node.title, (
            f"the output mark changed under {kwargs}"
        )
        assert len(packed(**kwargs).nodes[0].halos) == 1, (
            "a state on the output node drew more than one ring"
        )


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
            flat_view(["a"], {"a": []}, {"a": (0.0, 0.0)}),
            {},
            output=pass_key("a"),
            body={"a": [BodyRow("u_res", value, _BLUE)]},
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
        flat_view(["a"], {"a": []}, {"a": (0.0, 0.0)}),
        {},
        output=pass_key("a"),
        body={"a": [BodyRow("u_time", (), _BLUE)]},
    )
    row = next(spec for spec in packed.nodes[0].ports if spec.label == "u_time")
    assert row.control is True
    assert row.widget is ffi.Widget.NONE
    assert row.value == ()


def test_an_engine_value_is_read_only() -> None:
    """The engine overwrites it every frame, so an editable widget would offer
    an edit that reverts -- which the library's own contract calls out."""
    packed = pack_nodes(
        flat_view(["a"], {"a": []}, {"a": (0.0, 0.0)}),
        {},
        output=pass_key("a"),
        body={
            "a": [
                BodyRow("u_time", (1.0,), _BLUE),
                BodyRow("u_res", (64.0, 64.0), _BLUE),
            ]
        },
    )
    for spec in packed.nodes[0].ports:
        if spec.label.startswith("u_"):
            assert spec.read_only is True, spec.label


def test_an_engine_row_is_a_pure_control_and_so_carries_no_pin() -> None:
    """`Attribute.kinds` is a MASK, and which bits are set decides whether a
    row has a PIN: the library draws one for Input or Output and nothing
    else, so a row carrying neither is the only pinless row it can express
    -- which is what an engine-written value is.

    Sending Input as well gave those rows a real, hittable pin. Not merely
    misleading: a wire dropped on one made a genuine `Edge_Added` that
    `read_events` then discarded, since an engine row has no sampler to
    resolve against. The wire followed the pointer and vanished on release
    with no refusal shown, so the canvas appeared to ignore the user.
    """
    packed = pack_nodes(
        flat_view(["b"], {"b": [Port("u_src", "unfilled")]}, {"b": (0.0, 0.0)}),
        {},
        output=pass_key("b"),
        body={"b": [BodyRow("u_time", (1.0,), _BLUE)]},
    )
    canvas = ffi.Canvas()
    canvas.load_atlas()
    canvas.frame(
        packed.nodes, packed.edges, (600.0, 400.0), ffi.View(), ffi.PointerState()
    )
    # The attribute list in packed order: the sampler, the output, the
    # engine row -- which is the order `pack_nodes` emits and the order an
    # edge's `to_attr` indexes.
    kinds = [canvas._attrs[i].kinds for i in range(3)]
    assert kinds[0] == ffi.ATTR_INPUT, "a sampler is not a plain input"
    assert kinds[1] == ffi.ATTR_OUTPUT, "the output is not a plain output"
    assert kinds[2] == ffi.ATTR_CONTROL, (
        f"an engine row carries a side bit, so it draws a pin: {kinds[2]:#05b}"
    )
    canvas.release()


def test_the_theme_the_canvas_sends_keeps_every_role_distinct() -> None:
    """Read from `canvas_theme()` -- what the canvas actually SENDS -- and
    not from the tokens it is built out of.

    The row background is drawn from the attribute's ROLE, so the three role
    colours are what say whether a row takes a wire in, sends one out, or
    does both. Pointing them all at one token does not unify the palette, it
    deletes the distinction; pointing them at a BACKGROUND token, which
    shipped for one commit, paints every row the colour of the thing behind
    it and turns the canvas into grey slabs.

    The earlier version of this asserted `luminance(BG_FRAME) >
    luminance(BG_APP)` -- two constants in `theme.py`, with the call site
    nowhere in it. It pinned that the palette is ORDERABLE and said nothing
    about which pair the canvas picked, so swapping the call site back to
    `BG_SURFACE` left it green. That is why the theme is a named function
    now: a test can only check a choice it can see.
    """
    theme = canvas_theme()

    def rgb(name: str) -> tuple[float, ...]:
        return tuple(round(v, 4) for v in list(getattr(theme, name))[:3])

    roles = {name: rgb(name) for name in ("input", "output", "both", "control")}
    assert len(set(roles.values())) == 4, f"two roles share a colour: {roles}"

    for name, colour in roles.items():
        assert colour != rgb("surface"), f"the {name} role is the node surface"
        assert colour != rgb("canvas"), f"the {name} role is the canvas background"


def test_the_node_body_the_canvas_sends_is_lighter_than_its_canvas() -> None:
    """The library's shading LIFTS a node off its background, so a surface
    darker than the canvas makes every node read as a hole instead of a
    card. `BG_SURFACE` is darker than `BG_APP` in this palette, and sending
    it as the body is what did exactly that.

    Both numbers come from the sent theme, so the pair under test is the
    pair the canvas chose.
    """
    theme = canvas_theme()

    def luminance(name: str) -> float:
        c = list(getattr(theme, name))
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    assert luminance("surface") > luminance("canvas"), (
        "the node body the canvas sends is darker than its canvas, "
        "so every node reads as a hole"
    )


def test_the_wire_outline_is_darker_than_everything_it_crosses() -> None:
    """`wire_outline` is the dark run UNDER a wire's core, and a wire crosses
    nodes, the canvas and other wires -- so it cannot borrow contrast from
    any one of them and has to be below all three.

    Sent `BORDER` it was three times the canvas's luminance: a LIGHT halo,
    which is the field inverted rather than merely mistuned.
    """
    theme = canvas_theme()

    def luminance(name: str) -> float:
        c = list(getattr(theme, name))
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    outline = luminance("wire_outline")
    for over in ("canvas", "surface", "input", "output", "both", "control"):
        assert outline < luminance(over), (
            f"the wire outline is lighter than the {over} it runs over, "
            "so it draws as a halo rather than a shadow"
        )


def test_the_shipped_theme_file_is_what_the_canvas_draws_with() -> None:
    """The FILE decides the canvas's palette, with nothing overriding it.

    Every value is read from the shipped file and looked for by name in
    what `canvas_theme()` returns, so a host that went back to writing
    its own colours over the file fails here rather than looking
    slightly wrong on screen. Comparing against literals instead would
    pass on code that ignored the file and happened to agree with it.
    """
    shipped = GRAPH_CANVAS_RESOURCES_DIR / "canvas.theme"
    text = shipped.read_text()
    from_file = ffi.parse_theme(text, str(shipped))
    live = canvas_theme()

    named = [
        line.split("=")[0].strip()
        for line in text.splitlines()
        if "=" in line and not line.strip().startswith("#")
    ]
    colours = [n for n in named if not n.startswith("attr.")]
    assert "input" in colours and "canvas" in colours, (
        "the shipped file no longer names the colours this gate reads, "
        f"so it is checking nothing: {colours}"
    )
    for field in colours:
        want = getattr(from_file, field)
        got = getattr(live, field)
        if not isinstance(want, float):
            want, got = list(want), list(got)
        assert got == want, f"canvas.theme's {field} did not reach the canvas: {got}"

    # A field the file does NOT name keeps the library's value rather than
    # becoming zero: a theme built from zero flattens the canvas.
    assert "hover_lift" not in colours
    assert live.hover_lift == ffi.default_theme().hover_lift


def test_each_kind_of_body_row_draws_its_own_colour() -> None:
    """The three kinds of non-wirable row must arrive on screen as three
    colours, and the check is against what the HOST sent rather than against
    a distance.

    Two earlier shapes of this failed, both instructive. A distance
    threshold cannot decide it: under the collision this exists to catch --
    the theme's default control colour set to the engine's blue -- the bands
    land 25 degrees apart, while the correct code's own nearest pair is 34.6
    apart, so no threshold separates them. And measuring the LIBRARY's
    defaults decides nothing about shaderbox, which is why the theme is
    pushed.

    So each row's sent colour is looked for by HUE, which is the statistic
    the blend leaves alone: mixing toward the surface moves lightness and
    saturation and holds hue. Three sent colours, three found, each matched
    to the row that asked for it.
    """
    engine = kind_color(SymbolKind.ENGINE_UNIFORM)
    script = kind_color(SymbolKind.SCRIPT_UNIFORM)
    canvas = ffi.Canvas()
    canvas.load_atlas()
    packed = pack_nodes(
        flat_view(["p"], {"p": []}, {"p": (0.0, 0.0)}),
        {},
        output=pass_key("p"),
        body={
            "p": [
                BodyRow("u_time", (0.69,), engine),
                BodyRow("u_driven", (1.0,), script),
                BodyRow("u_plain", (2.0,), None),
            ]
        },
    )
    theme = canvas_theme()
    result = canvas.frame(
        packed.nodes,
        packed.edges,
        (520.0, 460.0),
        ffi.View(),
        ffi.PointerState(),
        theme=theme,
    )
    shapes = shapes_array(result)
    # The row BANDS, by the geometry this fixture draws: a widget row is the
    # node's inner width and 30 tall. Measured, not copied -- the filter the
    # control-tone test uses is tuned to a shorter pin-only row.
    drawn = {
        _hue((float(r[4]), float(r[5]), float(r[6])))
        for r in shapes
        if r[2] > 90 and 24 < r[3] < 36 and r[7] > 0.9
    }
    canvas.release()

    # The untinted row takes the theme's own control colour, which is what
    # makes "no tint" a third appearance rather than a missing one.
    untinted = tuple(float(v) for v in list(theme.control)[:3])
    wanted = {
        "engine": _hue(engine[:3]),
        "script": _hue(script[:3]),
        "untinted": _hue(untinted),
    }
    matched: dict[str, float] = {}
    for name, want in wanted.items():
        near = min((_hue_gap(got, want), got) for got in drawn)
        matched[name] = near[1]
        assert near[0] < 12.0, (
            f"the {name} row's colour did not reach the screen: sent hue "
            f"{want:.1f}, nearest drawn {near[1]:.1f} of {sorted(drawn)}"
        )
    # And the three must be mutually apart, not merely each present. Looking
    # for them one at a time passes the collision: when the theme's control
    # colour IS the engine's blue, two of the three look for the same band
    # and both find it.
    #
    # The floor is HALF the smallest gap the fixture itself sends, not a
    # number read off one measurement: a fixed 30 goes stale the day a token
    # is retuned, while a derived one cannot. Two colours a host asked to
    # differ must not arrive closer than half what it asked for.
    names = sorted(wanted)
    # An ABSOLUTE bar, and it has to be: a floor derived from either set
    # scales with the bug. Measured both states -- the correct palette's
    # closest sent pair is 34.5 degrees, and under the collision this exists
    # to catch it is 25.5 -- so half-the-smallest-sent passes the collision
    # too, and half-the-smallest-drawn is circular outright (two kinds at
    # 0.3 degrees clear a floor of 0.15). 30 sits between the two measured
    # states and is the only form that separates them.
    #
    # A colour landing at the MIDPOINT of two others is not a defect and is
    # not flagged: at hue 109 between 157 and 61 the three sit 48/48/96
    # apart, every pair wider than the shipping palette's own 34.5. That is
    # a different valid palette, not a collapse.
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            apart = _hue_gap(matched[first], matched[second])
            assert apart > 30.0, (
                f"the {first} and {second} rows are drawn alike: "
                f"{apart:.1f} degrees apart"
            )


def _hue(colour: tuple[float, ...]) -> float:
    return colorsys.rgb_to_hls(colour[0], colour[1], colour[2])[0] * 360.0


def _hue_gap(first: float, second: float) -> float:
    """The shorter way round the wheel."""
    apart = abs(first - second)
    return min(apart, 360.0 - apart)


def test_a_dragged_row_names_the_pass_that_declares_it() -> None:
    """A `Value_Changed` resolves to (pass, uniform) through the node's body
    rows, which follow the inputs and the outputs in the flat attribute list.

    The pass is the node's PREVIEW rather than its name: a box shows its
    bundle member's rows, so an edit on a box writes the member that
    actually declares the uniform and not the group, which declares nothing.
    """
    packed = pack_nodes(
        flat_view(["p"], {"p": [Port("u_src", "unfilled")]}, {"p": (0.0, 0.0)}),
        {},
        output=pass_key("p"),
        body={
            "p": [
                BodyRow("u_time", (1.0,), None),
                BodyRow("u_tint", (0.5, 0.25), None, editable=True),
            ]
        },
    )
    result = ffi.Result()
    events = (ffi.Event * 1)()
    events[0].kind = int(ffi.EventKind.VALUE_CHANGED)
    events[0].node = 0
    # One input, one output, then the body rows: u_tint is the fourth slot.
    events[0].attribute = 3
    events[0].value[0], events[0].value[1] = 0.75, 0.125
    events[0].value_count = 2
    result.events = events
    result.event_count = 1

    assert read_events(result, packed) == [ValueEdited("p", "u_tint", (0.75, 0.125))]


def test_an_engine_row_refuses_the_pointer_and_the_others_take_it() -> None:
    """`read_only` is what stops a drag reaching the widget at all.

    An engine uniform is not the user's to set -- the engine recomputes it
    from the clock and the canvas every frame -- so a drag on one would
    appear to take and revert on the next tick, which reads as the canvas
    being broken rather than as the value being owned elsewhere.
    """
    packed = pack_nodes(
        flat_view(["p"], {"p": []}, {"p": (0.0, 0.0)}),
        {},
        output=pass_key("p"),
        body={
            "p": [
                BodyRow("u_time", (1.0,), None),
                BodyRow("u_tint", (0.5,), None, editable=True),
            ]
        },
    )
    by_label = {port.label: port for port in packed.nodes[0].ports}
    assert by_label["u_time"].read_only, "an engine value took the pointer"
    assert not by_label["u_tint"].read_only, "an editable value refused it"


def test_an_editable_scalar_takes_the_pointer_like_an_editable_vector() -> None:
    """A `LABEL` is read-only BY NATURE -- it renders text and takes no
    pointer -- so a scalar given one looks exactly like an editable row and
    silently refuses every drag.

    Found by driving the real frame loop: a vec2 row emitted nine
    `ValueEdited` events across a drag and the scalar beside it emitted
    none. `_widget_for` chose on arity alone, which was right while every
    row was read-only and became a trap the moment one was not.

    A read-only scalar keeps its LABEL: it is the quieter of the two and
    loses nothing, since a LABEL draws component 0 and a scalar has only
    that. A read-only VECTOR may not have one, which the row below pins.
    """
    assert _widget_for((1.0,), editable=True, swatch=False) is ffi.Widget.DRAG, (
        "an editable scalar was given a widget that cannot be dragged"
    )
    assert _widget_for((1.0, 2.0), editable=True, swatch=False) is ffi.Widget.DRAG
    # Read-only: a scalar may be a label, a vector may not.
    assert _widget_for((1.0,), editable=False, swatch=False) is ffi.Widget.LABEL
    assert _widget_for((1.0, 2.0), editable=False, swatch=False) is ffi.Widget.DRAG, (
        "a read-only vector was given a LABEL, which draws only its first "
        "component and drops the rest in silence"
    )
    assert _widget_for((), editable=True, swatch=False) is ffi.Widget.NONE


def test_an_editable_row_is_packed_with_a_pointer_taking_widget() -> None:
    """The same claim end to end: what `pack_nodes` actually emits.

    Separate from the row above because that one tests the CHOICE and this
    one tests that the choice reaches the attribute -- a `_widget_for` fixed
    in isolation while the call site kept passing arity alone would leave
    the defect exactly where it was.
    """
    packed = pack_nodes(
        flat_view(["p"], {"p": []}, {"p": (0.0, 0.0)}),
        {},
        output=pass_key("p"),
        body={
            "p": [
                BodyRow("u_gain", (1.0,), None, editable=True),
                BodyRow("u_time", (1.0,), None),
            ]
        },
    )
    by_label = {port.label: port for port in packed.nodes[0].ports}
    assert by_label["u_gain"].widget is ffi.Widget.DRAG
    assert not by_label["u_gain"].read_only
    assert by_label["u_time"].read_only


def test_a_colour_typed_row_draws_a_swatch_and_a_plain_one_does_not() -> None:
    """A colour uniform gets the library's swatch; the same row without the
    colour type gets drag fields.

    The pair differs ONLY in `swatch`: same value, same length, same
    editability. Comparing a colour row against a read-only scalar would
    differ whether or not the swatch flag was read.
    """
    rows = [
        BodyRow("u_line_color", (1.0, 0.5, 0.25), None, editable=True, swatch=True),
        BodyRow("u_offset", (1.0, 0.5, 0.25), None, editable=True, swatch=False),
    ]
    packed = pack_nodes(
        flat_view(["p"], {"p": []}, {"p": (0.0, 0.0)}),
        {},
        output="",
        body={"p": rows},
    )
    widgets = [port.widget for port in packed.nodes[0].ports if port.control]
    assert widgets == [ffi.Widget.COLOR, ffi.Widget.DRAG], (
        f"the swatch flag did not decide the widget: {widgets}"
    )


def test_a_read_only_colour_row_gets_no_swatch() -> None:
    """A swatch opens an editor, so a value the engine rewrites every frame
    must not carry one -- the picker would take an edit the next tick
    discards."""
    rows = [
        BodyRow("u_engine_color", (1.0, 0.5, 0.25), None, editable=False, swatch=True),
    ]
    packed = pack_nodes(
        flat_view(["p"], {"p": []}, {"p": (0.0, 0.0)}),
        {},
        output="",
        body={"p": rows},
    )
    widget = next(port.widget for port in packed.nodes[0].ports if port.control)
    assert widget is not ffi.Widget.COLOR, (
        "a read-only row opened a picker for a value the engine overwrites"
    )


def test_the_canvas_theme_file_decides_the_shading() -> None:
    """`canvas.theme` is read, not decoration beside a constant.

    The pair is the SAME field under two files: one naming a value the
    library's default does not hold, one naming nothing. Comparing the
    live theme against a literal would pass on code that ignored the file
    and happened to agree with it.
    """
    shipped = GRAPH_CANVAS_RESOURCES_DIR / "canvas.theme"
    default = ffi.default_theme()
    from_file = ffi.parse_theme(shipped.read_text(), str(shipped))
    assert from_file.row_role_widget != default.row_role_widget, (
        "the shipped file names nothing the library does not already do, "
        "so nothing about it can be shown to have been read"
    )
    assert canvas_theme().row_role_widget == from_file.row_role_widget, (
        "the canvas ignored the shipped theme file"
    )
    # A field the file does NOT name keeps the library's value rather than
    # becoming zero: a theme built from zero flattens the canvas.
    assert canvas_theme().hover_lift == default.hover_lift


def test_an_unknown_theme_field_names_its_line_rather_than_being_dropped() -> None:
    """The file exists to stop guessing which spelling reached the
    renderer, so a misspelling is an error and not a silent no-op."""
    try:
        ffi.parse_theme("row_role = 0.4\nrow_role_widgets = 0.5\n", "t")
    except ffi.ThemeParseFailed as failure:
        assert failure.line == 2, f"the wrong line was reported: {failure.line}"
        assert failure.reason is ffi.ThemeParseError.UNKNOWN_FIELD
    else:
        raise AssertionError("a misspelled field parsed clean")


def test_host_categories_come_back_unvalidated() -> None:
    """Which kinds of row exist is shaderbox's taxonomy and it grows on its
    own schedule, so a name the library has never heard of is returned
    rather than rejected -- otherwise adding a `SymbolKind` would need a
    library release.

    The fixture uses a name no library list could contain, on purpose: a
    name that happened to be known would pass whether or not validation
    exists.
    """
    cats = ffi.parse_categories(
        "attr.engine_uniform = 0.5 0.6 0.7 1.0\n"
        "attr.a_name_this_library_cannot_know = 1.0 0.0 0.0 1.0\n"
    )
    assert set(cats) == {"engine_uniform", "a_name_this_library_cannot_know"}
    assert cats["a_name_this_library_cannot_know"] == (1.0, 0.0, 0.0, 1.0)


def test_category_names_sharing_a_prefix_stay_distinct() -> None:
    """A name longer than the read buffer is TRUNCATED, not refused, so two
    names sharing a prefix arrive as one string with two colours behind it
    -- and shaderbox's kind names are exactly that shape.

    Checked as DISTINCTNESS rather than arrival: searching for each name
    independently passes a collision, because when both collapse to the
    same string both searches find it. The fixture's names differ only
    after character 15, so a buffer sized by any fixed guess short of the
    full length merges them.
    """
    text = (
        "attr.engine_uniform_alpha = 1 0 0 1\n"
        "attr.engine_uniform_beta = 0 1 0 1\n"
        "attr.engine_uniform_gamma = 0 0 1 1\n"
    )
    cats = ffi.parse_categories(text)
    assert len(cats) == 3, f"three names collapsed into {len(cats)}: {sorted(cats)}"
    assert sorted(cats) == [
        "engine_uniform_alpha",
        "engine_uniform_beta",
        "engine_uniform_gamma",
    ], f"a name came back truncated: {sorted(cats)}"
    # Distinct colours too: a collision keeps one key and one colour, which
    # the length check alone would not catch if two names were equal.
    assert len({tuple(v) for v in cats.values()}) == 3, (
        f"two categories share a colour, so one overwrote another: {cats}"
    )


def test_a_category_named_twice_is_refused() -> None:
    """The library returns every `attr.` line and merges none, so a
    repeated name would take whichever came last. A theme file exists to
    stop a line being dropped in silence, and a duplicate drops one."""
    try:
        ffi.parse_categories("attr.x = 1 0 0 1\nattr.x = 0 1 0 1\n")
    except ValueError as failure:
        assert "more than once" in str(failure), failure
    else:
        raise AssertionError("a duplicated category name was accepted")

    # The same two names, distinct, must still parse -- or the check is
    # refusing categories rather than refusing duplicates.
    assert len(ffi.parse_categories("attr.x = 1 0 0 1\nattr.y = 0 1 0 1\n")) == 2


def test_a_written_theme_names_only_what_was_tuned() -> None:
    """`write_theme_diff` is what a vendored `.theme` is written with.

    The property is NOT that it round-trips -- a full 29-field dump
    round-trips too, and would pin every colour the app owns at the
    version it was written against. It is that a field left at its
    default is ABSENT, so the file follows the library forward and makes
    no claim on the palette `theme.py` supplies.
    """
    tuned = ffi.default_theme()
    tuned.row_role_widget = tuned.row_role_widget + 0.28
    text = ffi.write_theme_diff(tuned)

    assert "row_role_widget" in text, "the tuned field is missing from the diff"
    # `surface` is a colour the app owns and this session never touched.
    assert "surface" not in text, (
        "an untouched field was written, so the file claims a colour the "
        "app owns and would pin it at this library version"
    )
    assert ffi.parse_theme(text).row_role_widget == tuned.row_role_widget, (
        "the diff does not read back as the theme it was written from"
    )


def test_a_node_wears_at_most_one_state_ring() -> None:
    """One ring, so a reader never decodes a combination.

    The three states are mutually exclusive, and the document's output --
    a property rather than a state -- is marked in the title instead of a
    second concentric ring. The pairing that broke was a hovered output
    node: two warm rings six degrees apart reading as one fat band.

    Every state is driven here, including the pairs that used to stack.
    """
    palette = NodePalette(
        hover=(0.1, 0.2, 0.3, 1.0),
        select=(0.4, 0.5, 0.6, 1.0),
        failing=(0.7, 0.8, 0.9, 1.0),
    )
    for selected, failing, hovered in (
        (frozenset(), False, ""),
        (frozenset({"p"}), False, ""),
        (frozenset(), True, ""),
        (frozenset(), False, "p"),
        # The stacking cases: each state ON the output node.
        (frozenset({"p"}), False, "p"),
        (frozenset({"p"}), True, "p"),
        (frozenset(), True, "p"),
    ):
        rings = _halos_of(
            "p",
            hovered,
            selected,
            ghost=False,
            colors=palette,
            failing=failing,
        )
        assert len(rings) <= 1, (
            f"selected={bool(selected)} failing={failing} hovered={bool(hovered)} "
            f"drew {len(rings)} rings; a node shows one state"
        )
    assert _halos_of("p", "p", frozenset({"p"}), True, palette, True) == (), (
        "a ghost wore a highlight promising a gesture it refuses"
    )


def test_the_state_rings_separate_on_the_channel_that_carries_them() -> None:
    """Hover is the desaturated one; the two urgent states differ in hue.

    Three marks cannot all be far apart in hue -- red is fixed by meaning
    and the accent is fixed by the rest of the app -- and a gate demanding
    that produced a blue selection ring nobody wanted. So the rule is two
    channels: the transient state is the only quiet one, and the two that
    demand attention are told apart by hue.
    """
    palette = _ring_palette()

    def hsv(name: str) -> tuple[float, float, float]:
        return colorsys.rgb_to_hsv(*getattr(palette, name)[:3])

    assert hsv("hover")[1] < 0.4, (
        f"hover is saturated ({hsv('hover')[1]:.2f}), so the most transient "
        "state shouts as loudly as selection"
    )
    for urgent in ("select", "failing"):
        assert hsv(urgent)[1] > 0.6, (
            f"{urgent} is washed out ({hsv(urgent)[1]:.2f}) and no longer "
            "reads as a state that wants attention"
        )
    apart = abs(hsv("select")[0] - hsv("failing")[0]) * 360.0
    apart = min(apart, 360.0 - apart)
    assert apart > 25.0, (
        f"select and failing are both saturated and {apart:.1f} degrees "
        "apart, so the two urgent states read alike"
    )


def test_the_state_ring_is_thick_enough_to_see() -> None:
    """The ring is the ONLY mark a state gets, so it has to carry.

    It shipped at 2.5 units set 2.0 outside the card and read as a thin
    outline floating off it -- tolerable while a second concentric ring
    was also drawn, barely visible once the ring became the whole signal.

    Measured as DRAWN AREA rather than as the constant, so a change that
    widens the rect while insetting it back out of view fails here. The
    floor sits between the two measured states: 1785 units at the old
    geometry, 3460 at the shipping one.
    """
    palette = NodePalette(hover=(1.0, 1.0, 1.0, 1.0), select=(0.0, 1.0, 0.0, 1.0))
    packed = pack_nodes(
        flat_view(["a"], {"a": []}, {"a": (0.0, 0.0)}),
        {},
        output=pass_key("a"),
        selected=frozenset({pass_key("a")}),
        palette=palette,
    )
    canvas = ffi.Canvas()
    canvas.load_atlas()
    result = canvas.frame(
        packed.nodes,
        packed.edges,
        (400.0, 300.0),
        ffi.View(),
        ffi.PointerState(),
        theme=None,
    )
    shapes = shapes_array(result)
    area = sum(
        float(row[2]) * float(row[3])
        for row in shapes
        if abs(float(row[4])) < 0.01
        and float(row[5]) > 0.9
        and abs(float(row[6])) < 0.01
    )
    canvas.release()
    assert area > 2500.0, (
        f"the state ring covers {area:.0f} units, back near the 1785 that "
        "read as a thin outline floating off the card"
    )
