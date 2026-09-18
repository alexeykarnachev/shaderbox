"""The C boundary to libgraph_canvas.so holds (feature 098).

Every check here is one a plausible-looking mistake would break. The layout
proof is the important one: it caught a real disagreement the first time it ran
(`Pin_Fill` has three members, and the 0 this binding sends means "unset" at
the wire rather than naming one of them).
"""

import ctypes

import pytest

from shaderbox.graph_canvas import ffi


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
    for which, (label, expected) in enumerate(ffi._ENUMS):
        assert lib.gc_enum_count(which) == expected, label


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
    result = canvas.frame(
        nodes, edges, (1280.0, 720.0), ffi.View(), ffi.PointerState()
    )

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
