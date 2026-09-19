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
    _widget_for,
    edge_id,
    flat_view,
    node_id,
    pack_nodes,
    pass_key,
    read_events,
    theme_from,
)
from shaderbox.graph_canvas.render import shapes_array
from shaderbox.intel.symbols import SymbolKind
from shaderbox.pass_graph import Port, strip_order
from shaderbox.theme import kind_color
from shaderbox.widgets.graph_state import ports_of
from shaderbox.widgets.pass_graph import canvas_theme

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
            palette=NodePalette(
                hover=(1.0, 1.0, 1.0, 1.0), select=colour, output=colour
            ),
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


def test_selection_and_the_output_are_two_rings_rather_than_one() -> None:
    """A node that is BOTH selected and the document's output wears two
    rings, because one border had to choose -- and chose selection, so the
    canvas's primary verb went unmarked exactly when its target was picked.

    Selection outranks hover on the inner ring; a ghost wears none.
    """
    order = ["a", "b"]
    ports: dict[str, list[Port]] = {"a": [], "b": []}
    palette = NodePalette(
        hover=(1.0, 0.0, 0.0, 1.0),
        select=(0.0, 1.0, 0.0, 1.0),
        output=(0.0, 0.0, 1.0, 1.0),
    )

    def halos(**kwargs: object) -> list[tuple[object, ...]]:
        packed = pack_nodes(
            flat_view(order, ports, {}),
            {},
            output=pass_key("a"),
            palette=palette,
            **kwargs,  # type: ignore[arg-type]
        )
        return list(packed.nodes[0].halos)

    plain = halos()
    assert [h[0] for h in plain] == [palette.output], "the output pass wears no ring"
    picked = halos(selected=frozenset({pass_key("a")}))
    assert [h[0] for h in picked] == [palette.select, palette.output], (
        f"a selected output node lost one of its two rings: {picked}"
    )
    hovered = halos(hovered=pass_key("a"))
    assert [h[0] for h in hovered] == [palette.hover, palette.output]
    # Selection outranks hover on the inner ring.
    both = halos(hovered=pass_key("a"), selected=frozenset({pass_key("a")}))
    assert [h[0] for h in both] == [palette.select, palette.output]
    # The rings are separated, or two marks read as one thick band.
    assert picked[1][2] > picked[0][2], "the two rings sit at one inset"


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


def test_theme_from_carries_every_colour_it_is_handed() -> None:
    """Each argument reaches the field it names.

    The gate this replaces asserted only that the roles were mutually
    distinct, which `default_theme()` already satisfies -- so a `theme_from`
    that ignored all fourteen arguments and returned the library's default
    passed it. Every colour here is a value no default holds, and each is
    looked for by name.
    """
    marks = {
        "canvas": (0.01, 0.02, 0.03, 1.0),
        "surface": (0.04, 0.05, 0.06, 1.0),
        "grid": (0.07, 0.08, 0.09, 1.0),
        "border": (0.10, 0.11, 0.12, 1.0),
        "text": (0.13, 0.14, 0.15, 1.0),
        "text_dim": (0.16, 0.17, 0.18, 1.0),
        "text_bright": (0.19, 0.20, 0.21, 1.0),
        "accent": (0.22, 0.23, 0.24, 1.0),
        "pin": (0.25, 0.26, 0.27, 1.0),
        "wire_outline": (0.28, 0.29, 0.30, 1.0),
        "wire_invalid": (0.31, 0.32, 0.33, 1.0),
        "port_input": (0.34, 0.35, 0.36, 1.0),
        "port_output": (0.37, 0.38, 0.39, 1.0),
        "port_both": (0.40, 0.41, 0.42, 1.0),
        "control": (0.43, 0.44, 0.45, 1.0),
    }
    # Not a colour, so it is checked on its own rather than walked with the
    # marks: a value no default holds, which is what makes an ignored
    # argument visible.
    theme = theme_from(**marks, row_role_widget=0.77)
    assert round(theme.row_role_widget, 4) == 0.77, (
        "theme_from(row_role_widget=) did not reach theme.row_role_widget: "
        f"{theme.row_role_widget}"
    )
    # The three arguments whose field is not their own name.
    lands_on = {"port_input": "input", "port_output": "output", "port_both": "both"}
    for argument, colour in marks.items():
        field = lands_on.get(argument, argument)
        got = tuple(round(v, 4) for v in list(getattr(theme, field))[:3])
        assert got == colour[:3], (
            f"theme_from({argument}=) did not reach theme.{field}: {got}"
        )


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
    assert _widget_for((1.0,), editable=True) is ffi.Widget.DRAG, (
        "an editable scalar was given a widget that cannot be dragged"
    )
    assert _widget_for((1.0, 2.0), editable=True) is ffi.Widget.DRAG
    # Read-only: a scalar may be a label, a vector may not.
    assert _widget_for((1.0,), editable=False) is ffi.Widget.LABEL
    assert _widget_for((1.0, 2.0), editable=False) is ffi.Widget.DRAG, (
        "a read-only vector was given a LABEL, which draws only its first "
        "component and drops the rest in silence"
    )
    assert _widget_for((), editable=True) is ffi.Widget.NONE


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
