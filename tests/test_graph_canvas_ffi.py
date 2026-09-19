"""The C boundary to libgraph_canvas.so holds (feature 098).

Every check here is one a plausible-looking mistake would break. The layout
proof is the important one: it caught a real disagreement the first time it ran
(`Pin_Fill` has three members, and the 0 this binding sends means "unset" at
the wire rather than naming one of them).
"""

import ctypes

import pytest

from shaderbox.graph_canvas import ffi
from shaderbox.graph_canvas.adapter import flat_view, pack_nodes, pass_key
from shaderbox.pass_graph import Port


def test_the_library_loads_and_its_layout_is_proven() -> None:
    lib = ffi.ensure_loaded()
    assert lib.gc_abi_version() == ffi.ABI_VERSION


def test_every_struct_field_sits_where_the_library_says_it_does() -> None:
    """A size is a proxy for a layout and the two come apart exactly where it
    hurts: swap two same-width fields and the size does not move while every
    value read is the one next door."""
    lib = ffi.ensure_loaded()
    buf = ctypes.create_string_buffer(128)
    for which, (label, struct) in enumerate(ffi._STRUCTS):
        assert lib.gc_sizeof(which) == ctypes.sizeof(struct), label
        for index, (field_name, *_rest) in enumerate(struct._fields_):
            assert lib.gc_offsetof(which, index) == getattr(struct, field_name).offset
            written = lib.gc_field_name(which, index, buf, len(buf))
            assert bytes(buf[:written]).decode() == field_name


def test_every_enum_has_the_member_count_this_binding_expects() -> None:
    """An enum that gains a member changes NO struct's size, so it passes every
    size check and then hands a value nothing here has a name for."""
    lib = ffi.ensure_loaded()
    for which, (label, expected, mirror) in enumerate(ffi._ENUMS):
        assert lib.gc_enum_count(which) == expected, label
        if mirror is None:
            continue
        # By NAME too: a count accepts two members SWAPPED, and a swapped
        # `Edge_Added`/`Edge_Removed` makes every wire the user draws
        # unwire instead. Proven -- that exact swap loaded clean before
        # the name check existed.
        buf = ctypes.create_string_buffer(128)
        for position, member in enumerate(mirror):
            # By POSITION, which is what `gc_enum_name` takes. Every enum
            # here but one has value == position, so asking by value agreed
            # until `Theme_Parse_Error` arrived running 0, -2, -3, ...
            written = lib.gc_enum_name(which, position, buf, 128)
            # Sliced by the returned LENGTH: the library writes no
            # terminator, so a shorter name keeps the previous one's tail.
            assert buf.raw[:written].decode().upper() == member.name, label


def test_a_layout_disagreement_refuses_to_load() -> None:
    """The proof is a gate, so it has to be able to fail. Declaring one field
    too few is the shape of a binding that drifted behind the library."""
    lib = ffi.ensure_loaded()
    original = ffi._STRUCTS[:]
    truncated = type(
        "TruncatedRun",
        (ctypes.Structure,),
        {"_fields_": [("stream", ctypes.c_int32), ("first", ctypes.c_int32)]},
    )
    try:
        ffi._STRUCTS[5] = ("Run", truncated)
        with pytest.raises(ffi.LayoutMismatch):
            ffi._verify_layout(lib)
    finally:
        ffi._STRUCTS[:] = original
    # And the real layout still passes once the lie is removed.
    ffi._verify_layout(lib)


def test_a_frame_returns_geometry_and_the_runs_cover_both_streams() -> None:
    """The run list is the part a host cannot guess: a gap would drop geometry
    and an overlap would draw it twice."""
    canvas = ffi.Canvas()
    canvas.load_atlas()
    nodes = [
        ffi.NodeSpec(id=1, title="seed", x=0, y=0, ports=[ffi.PortSpec("out", False)]),
        ffi.NodeSpec(
            id=2,
            title="blur",
            x=320,
            y=40,
            ports=[ffi.PortSpec("u_src", True), ffi.PortSpec("out", False)],
        ),
    ]
    edges = [ffi.EdgeSpec(id=1, from_node=0, from_attr=0, to_node=1, to_attr=0)]
    result = canvas.frame(nodes, edges, (1280.0, 720.0), ffi.View(), ffi.PointerState())

    assert result.shape_count > 0
    assert result.glyph_count > 0
    assert result.rect_count == len(nodes)

    drawn = {0: 0, 1: 0}
    for i in range(result.run_count):
        run = result.runs[i]
        drawn[run.stream] += run.count
    assert drawn[0] == result.shape_count
    assert drawn[1] == result.glyph_count
    canvas.release()


def test_a_shape_run_resumes_after_a_glyph_run_at_a_nonzero_offset() -> None:
    """The two streams INTERLEAVE, which is what makes the run's `first` load
    bearing: a host that ignores it redraws the head of the stream in place of
    the overlay, and the frame goes dim rather than wrong."""
    canvas = ffi.Canvas()
    canvas.load_atlas()
    nodes = [
        ffi.NodeSpec(id=1, title="seed", x=0, y=0, ports=[ffi.PortSpec("out", False)])
    ]
    result = canvas.frame(nodes, [], (1280.0, 720.0), ffi.View(), ffi.PointerState())

    streams = [result.runs[i].stream for i in range(result.run_count)]
    assert streams.count(0) >= 2, "expected the shape stream to be visited twice"
    later = [
        result.runs[i].first
        for i in range(result.run_count)
        if result.runs[i].stream == 0
    ]
    assert max(later) > 0, "a resumed shape run must start past the stream's head"
    canvas.release()


def test_node_positions_and_sizes_come_back_in_screen_space() -> None:
    canvas = ffi.Canvas()
    canvas.load_atlas()
    nodes = [
        ffi.NodeSpec(id=1, title="a", x=0, y=0, ports=[ffi.PortSpec("out", False)]),
        ffi.NodeSpec(id=2, title="b", x=400, y=0, ports=[ffi.PortSpec("out", False)]),
    ]
    result = canvas.frame(nodes, [], (1280.0, 720.0), ffi.View(), ffi.PointerState())
    first = result.node_rects[0]
    second = result.node_rects[1]
    assert first.w > 0 and first.h > 0
    # At zoom 1 with no pan, the second node's 400-unit offset is 400 pixels.
    assert second.x - first.x == pytest.approx(400.0)
    assert canvas.node_size(0) == (first.w, first.h)
    assert canvas.node_size(len(nodes)) is None
    canvas.release()


def test_the_view_the_library_returns_is_the_one_to_push_back() -> None:
    """`pan` is a canvas-space position that is SUBTRACTED, so a host that
    treats it as a screen offset gets the direction backwards and the scene
    lands somewhere off-canvas with no error."""
    canvas = ffi.Canvas()
    canvas.load_atlas()
    nodes = [
        ffi.NodeSpec(id=1, title="a", x=0, y=0, ports=[ffi.PortSpec("out", False)])
    ]
    view = ffi.View(pan_x=100.0, pan_y=0.0, zoom=1.0)
    result = canvas.frame(nodes, [], (800.0, 600.0), view, ffi.PointerState())
    shifted = result.node_rects[0].x
    result = canvas.frame(nodes, [], (800.0, 600.0), ffi.View(), ffi.PointerState())
    assert result.node_rects[0].x - shifted == pytest.approx(100.0)
    canvas.release()


def test_the_refusal_mask_is_a_veto_and_not_another_gesture() -> None:
    """`GESTURE_NONE` is the TOP bit, not the first.

    It overrides the others rather than joining them, so numbering it 1 makes
    it alias `DRAG`: a node marked "refuses everything" then accepts drags,
    and the refusal reads as silently not working. Measured against the
    library rather than asserted from the header, by driving a node that
    refuses and watching it not move.
    """
    assert ffi.Gesture.NONE == 1 << 31
    assert {int(g) for g in ffi.Gesture} & {1, 2, 4, 8} == {1, 2, 4, 8}

    canvas = ffi.Canvas()
    canvas.load_atlas()
    inert = ffi.NodeSpec(
        id=1,
        title="ghost",
        x=0,
        y=0,
        ports=[ffi.PortSpec("out", False)],
        accepts=int(ffi.Gesture.NONE),
    )
    size = (600.0, 400.0)
    view = ffi.View()
    probe = canvas.frame([inert], [], size, view, ffi.PointerState())
    box = probe.node_rects[0]
    middle = (box.x + box.w / 2, box.y + box.h / 3)

    down = int(ffi.Pointer.DOWN)
    canvas.frame(
        [inert],
        [],
        size,
        view,
        ffi.PointerState(
            x=middle[0], y=middle[1], flags=down | int(ffi.Pointer.PRESSED)
        ),
    )
    after = canvas.frame(
        [inert],
        [],
        size,
        view,
        ffi.PointerState(x=middle[0] + 80.0, y=middle[1], flags=down),
    )
    kinds = {after.events[i].kind for i in range(after.event_count)}
    assert int(ffi.EventKind.NODE_MOVED) not in kinds
    canvas.release()


def test_a_frame_costs_one_textured_run_per_preview_not_per_distinct_image() -> None:
    """The draw-call budget scales with PASSES, not with distinct pictures.

    A run is cut wherever the bound texture changes, and a node's preview sits
    among that node's own geometry, so two nodes sharing one texture are never
    adjacent in the stream and their runs never merge. Measured both ways --
    six distinct names and six copies of one -- and the cost is identical.
    Budgeting per distinct image would under-count by the sharing factor.

    A FRESH handle per case: everything in a result points into the handle's
    own storage, so measuring both cases against one handle reports the first
    case's textures in the second. That misread cost a minute here and a
    minute upstream, independently.
    """
    names = [f"p{i}" for i in range(6)]
    positions = {name: (index * 240.0, 0.0) for index, name in enumerate(names)}
    measured: list[tuple[int, int]] = []
    for previews in (
        {name: (10 + index, 64, 64) for index, name in enumerate(names)},
        dict.fromkeys(names, (100, 64, 64)),
    ):
        canvas = ffi.Canvas()
        canvas.load_atlas()
        packed = pack_nodes(
            flat_view(names, {name: [] for name in names}, positions),
            previews,
            output=pass_key(names[-1]),
        )
        result = canvas.frame(
            packed.nodes, packed.edges, (1400.0, 800.0), ffi.View(), ffi.PointerState()
        )
        textured = sum(
            1 for i in range(result.run_count) if result.runs[i].texture != 0
        )
        measured.append((result.run_count, textured))
        canvas.release()

    # ABSOLUTE values, not an equality between the two measurements: nothing
    # about "the two agree" rules out "because they are the same reading". The
    # shared-handle break below is caught either way -- measured, its
    # pollution is partial and the readings differ -- so this is about what
    # the assertion can decide, not about that break.
    for run_count, textured in measured:
        assert textured == len(names), (
            f"expected one textured run per preview, got {measured}"
        )
        assert run_count == 2 * len(names) + 3, (
            f"expected 2n+3 runs with an atlas loaded, got {measured}"
        )


def test_the_run_formula_holds_only_because_an_atlas_is_loaded() -> None:
    """`2n+3` carries a precondition: TEXT.

    Without an atlas the library emits no glyphs, so nothing interleaves and
    the same scene is `2n+1`. shaderbox always loads one, which is why the
    app's budget is the first formula -- but a headless measurement that
    forgets the atlas is two short at every size and looks like a scene that
    simply costs less. The textured count is n either way, which is why the
    number worth planning against is n.
    """
    names = [f"p{i}" for i in range(4)]
    positions = {name: (index * 240.0, 0.0) for index, name in enumerate(names)}
    previews = {name: (10 + index, 64, 64) for index, name in enumerate(names)}
    ports: dict[str, list[Port]] = {name: [] for name in names}

    lit = ffi.Canvas()
    lit.load_atlas()
    dark = ffi.Canvas()
    try:
        packed = pack_nodes(
            flat_view(names, ports, positions), previews, output=pass_key(names[-1])
        )
        with_text = lit.frame(
            packed.nodes, packed.edges, (1400.0, 800.0), ffi.View(), ffi.PointerState()
        )
        assert with_text.glyph_count > 0
        assert with_text.run_count == 2 * len(names) + 3
        textured_lit = sum(
            1 for i in range(with_text.run_count) if with_text.runs[i].texture != 0
        )

        without = dark.frame(
            packed.nodes, packed.edges, (1400.0, 800.0), ffi.View(), ffi.PointerState()
        )
        assert without.glyph_count == 0, "no atlas should mean no glyphs"
        assert without.run_count == 2 * len(names) + 1
        textured_dark = sum(
            1 for i in range(without.run_count) if without.runs[i].texture != 0
        )
        assert textured_lit == textured_dark == len(names)
    finally:
        lit.release()
        dark.release()


def test_an_off_screen_node_still_costs_its_runs() -> None:
    """Nothing is culled at this layer, so a host cannot budget on the count
    it can SEE. Two nodes parked far outside the viewport cost what two nodes
    inside it cost."""
    names = ["a", "b"]
    ports: dict[str, list[Port]] = {name: [] for name in names}
    previews = {name: (10 + index, 64, 64) for index, name in enumerate(names)}

    inside = ffi.Canvas()
    inside.load_atlas()
    outside = ffi.Canvas()
    outside.load_atlas()
    try:
        near = pack_nodes(
            flat_view(names, ports, {"a": (0.0, 0.0), "b": (300.0, 0.0)}),
            previews,
            output=pass_key("b"),
        )
        far = pack_nodes(
            flat_view(names, ports, {"a": (40000.0, 40000.0), "b": (40300.0, 40000.0)}),
            previews,
            output=pass_key("b"),
        )
        size = (800.0, 600.0)
        here = inside.frame(
            near.nodes, near.edges, size, ffi.View(), ffi.PointerState()
        )
        here_runs = here.run_count
        there = outside.frame(
            far.nodes, far.edges, size, ffi.View(), ffi.PointerState()
        )
        assert there.run_count == here_runs, (
            f"off-screen nodes were culled: {there.run_count} against {here_runs}"
        )
    finally:
        inside.release()
        outside.release()


def test_a_canvas_that_failed_to_construct_collects_without_raising() -> None:
    """`__del__` runs on an object `__init__` never finished, so it cannot
    assume the attributes `__init__` binds. A `gc_new` returning null raises
    `RuntimeError`, and the collector then raised `AttributeError` inside
    `__del__` on top of it -- an error during finalisation, printed to
    stderr and swallowed, hiding the real one.

    Built by bypassing `__init__` entirely, which is the state a raise
    leaves the object in.
    """
    orphan = ffi.Canvas.__new__(ffi.Canvas)
    assert not hasattr(orphan, "_handle")
    orphan.__del__()
    # And the public path is safe on the same object.
    orphan.release()
