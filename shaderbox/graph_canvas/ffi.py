"""ctypes binding for the vendored libgraph_canvas.so (feature 098).

Leaf module: no imgui, no moderngl, no shaderbox model type. It knows the C ABI
and nothing about passes or documents, so the pair (this + `render.py`) lifts
into another project unchanged.

The host owns the graph and pushes all of it every frame; the library holds
nothing between calls and hands back two vertex streams, a run list and the
events. `Canvas.frame` is the whole boundary.

The layout is PROVEN at load, field by field, not merely by size: swap two
same-width fields and every size check still agrees while every value read is
the one next door. `_verify_layout` walks each struct's fields by NAME against
`gc_field_name`/`gc_offsetof` and each enum's members against
`gc_enum_name`, and raises rather than letting a mismatch render plausible
garbage.
"""

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from shaderbox.constants import RESOURCES_DIR

GRAPH_CANVAS_RESOURCES_DIR: Path = RESOURCES_DIR / "graph_canvas"
_LIB_PATH: Path = GRAPH_CANVAS_RESOURCES_DIR / "libgraph_canvas.so"
ATLAS_JSON_PATH: Path = GRAPH_CANVAS_RESOURCES_DIR / "atlas.json"
ATLAS_PNG_PATH: Path = GRAPH_CANVAS_RESOURCES_DIR / "atlas.png"
SHADERS_DIR: Path = GRAPH_CANVAS_RESOURCES_DIR / "shaders"

ABI_VERSION: int = 12

_LIB: ctypes.CDLL | None = None


class Str(ctypes.Structure):
    """An (offset, len) pair into the frame's one UTF-8 string blob."""

    _fields_ = [("offset", ctypes.c_int32), ("len", ctypes.c_int32)]


class Node(ctypes.Structure):
    _fields_ = [
        ("name", Str),
        ("title", Str),
        ("pos_x", ctypes.c_float),
        ("pos_y", ctypes.c_float),
        ("attr_first", ctypes.c_int32),
        ("attr_count", ctypes.c_int32),
        ("preview_tex", ctypes.c_uint32),
        ("preview_aspect", ctypes.c_float),
        ("preview_fit", ctypes.c_int32),
        ("preview_w", ctypes.c_int32),
        ("preview_h", ctypes.c_int32),
        ("tint_r", ctypes.c_float),
        ("tint_g", ctypes.c_float),
        ("tint_b", ctypes.c_float),
        ("tint_a", ctypes.c_float),
        ("tint_amount", ctypes.c_float),
        ("fade", ctypes.c_float),
        ("border_r", ctypes.c_float),
        ("border_g", ctypes.c_float),
        ("border_b", ctypes.c_float),
        ("border_a", ctypes.c_float),
        ("border_scale", ctypes.c_float),
        ("id", ctypes.c_uint64),
        # Two state RINGS outside the node's rect (ABI 6). A node's BORDER
        # colour is averaged to one luminance on the way to the GPU -- red
        # and blue arrive identical -- so a halo is how a colour survives:
        # four plain rects per ring. Two of them, because a node can be both
        # selected and something else at once. Zero width is inert.
        ("halo_color", (ctypes.c_float * 4) * 2),
        ("halo_inset", ctypes.c_float * 2),
        ("halo_width", ctypes.c_float * 2),
        ("accepts", ctypes.c_uint32),
        ("tint_set", ctypes.c_uint8),
        ("border_set", ctypes.c_uint8),
        ("dashed", ctypes.c_uint8),
        ("_pad0", ctypes.c_uint8),
    ]


class Attribute(ctypes.Structure):
    _fields_ = [
        ("label", Str),
        ("kinds", ctypes.c_uint32),
        ("widget", ctypes.c_int32),
        ("value", ctypes.c_float * 4),
        ("value_count", ctypes.c_int32),
        ("value_is_text", ctypes.c_int32),
        ("text", Str),
        ("speed", ctypes.c_float),
        ("min", ctypes.c_float),
        ("max", ctypes.c_float),
        ("opt_first", ctypes.c_int32),
        ("opt_count", ctypes.c_int32),
        ("height", ctypes.c_float),
        ("pin_shape", ctypes.c_int32),
        ("pin_fill", ctypes.c_int32),
        ("pin_color_r", ctypes.c_float),
        ("pin_color_g", ctypes.c_float),
        ("pin_color_b", ctypes.c_float),
        ("pin_color_a", ctypes.c_float),
        # The ROW's colour, distinct from the pin's (ABI 5). `pin_color`
        # tints the pin and is read only while one is drawn, so a pinless
        # control could carry no colour of its own and every kind of
        # non-wirable row fell back to the theme's one `control` value.
        ("role_color_r", ctypes.c_float),
        ("role_color_g", ctypes.c_float),
        ("role_color_b", ctypes.c_float),
        ("role_color_a", ctypes.c_float),
        ("read_only", ctypes.c_uint8),
        ("pin_on_preview", ctypes.c_uint8),
        ("pin_color_set", ctypes.c_uint8),
        ("role_color_set", ctypes.c_uint8),
    ]


_RGBA = ctypes.c_float * 4


class Theme(ctypes.Structure):
    """The library's palette, as its own `Theme` rather than a mirror of it.

    Every field is a colour (four floats) or a scalar, so the struct is
    already what the boundary needs and the layout proof walks it like any
    other. A hand-written mirror would agree by inspection until one field
    did not.
    """

    _fields_ = [
        ("canvas", _RGBA),
        ("surface", _RGBA),
        ("grid", _RGBA),
        ("step", ctypes.c_float),
        ("hover_lift", ctypes.c_float),
        ("active_lift", ctypes.c_float),
        ("border", _RGBA),
        ("bevel_width", ctypes.c_float),
        ("bevel", ctypes.c_float),
        ("text", _RGBA),
        ("text_dim", _RGBA),
        ("text_bright", _RGBA),
        ("accent", _RGBA),
        ("input", _RGBA),
        ("output", _RGBA),
        ("control", _RGBA),
        ("both", _RGBA),
        ("pin", _RGBA),
        ("shadow", ctypes.c_float),
        ("pin_ring", _RGBA),
        ("pin_hollow", ctypes.c_float),
        ("bar_edge", ctypes.c_float),
        ("inner_shadow", ctypes.c_float),
        ("text_shadow", ctypes.c_float),
        ("wire_outline", _RGBA),
        ("wire_invalid", _RGBA),
        ("wire_lift", ctypes.c_float),
        ("row_role", ctypes.c_float),
        ("row_role_widget", ctypes.c_float),
    ]


class Edge(ctypes.Structure):
    _fields_ = [
        ("from_node", ctypes.c_int32),
        ("from_attr", ctypes.c_int32),
        ("to_node", ctypes.c_int32),
        ("to_attr", ctypes.c_int32),
        ("id", ctypes.c_uint64),
        ("color_r", ctypes.c_float),
        ("color_g", ctypes.c_float),
        ("color_b", ctypes.c_float),
        ("color_a", ctypes.c_float),
        ("width", ctypes.c_float),
        ("color_set", ctypes.c_uint8),
        ("_pad0", ctypes.c_uint8),
        ("_pad1", ctypes.c_uint8),
        ("_pad2", ctypes.c_uint8),
    ]


class Frame(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.POINTER(Node)),
        ("node_count", ctypes.c_int32),
        ("_pad0", ctypes.c_int32),
        ("attrs", ctypes.POINTER(Attribute)),
        ("attr_count", ctypes.c_int32),
        ("_pad1", ctypes.c_int32),
        ("edges", ctypes.POINTER(Edge)),
        ("edge_count", ctypes.c_int32),
        ("_pad2", ctypes.c_int32),
        ("options", ctypes.POINTER(Str)),
        ("option_count", ctypes.c_int32),
        ("_pad3", ctypes.c_int32),
        ("strings", ctypes.POINTER(ctypes.c_uint8)),
        ("string_len", ctypes.c_int32),
        ("_pad4", ctypes.c_int32),
        ("width", ctypes.c_float),
        ("height", ctypes.c_float),
        ("pan_x", ctypes.c_float),
        ("pan_y", ctypes.c_float),
        ("origin_x", ctypes.c_float),
        ("origin_y", ctypes.c_float),
        ("zoom", ctypes.c_float),
        ("_pad5", ctypes.c_float),
        ("pointer_x", ctypes.c_float),
        ("pointer_y", ctypes.c_float),
        ("wheel", ctypes.c_float),
        ("pointer_flags", ctypes.c_uint32),
        ("text_codepoint", ctypes.c_int32),
        ("key", ctypes.c_int32),
        # Seconds since the host's previous frame, which is what advances
        # every eased highlight -- a row's hover among them. Zero holds each
        # ease where it is, for a host that renders on demand. The library
        # clamps it, so a first frame carrying time since process start
        # snaps rather than jumps.
        ("dt", ctypes.c_float),
        # Null is not "no theme": it LEAVES the handle's current one alone, so
        # a host sets it once and then passes null forever after.
        ("theme", ctypes.POINTER(Theme)),
    ]


class ShapeInstance(ctypes.Structure):
    _fields_ = [
        ("rect", ctypes.c_float * 4),
        ("fill_top", ctypes.c_float * 4),
        ("shape", ctypes.c_float * 4),
        ("fill_bot", ctypes.c_float * 4),
        ("edge", ctypes.c_float * 4),
        ("rotation", ctypes.c_float * 2),
        ("field", ctypes.c_float * 4),
        ("uv", ctypes.c_float * 4),
    ]


class GlyphVertex(ctypes.Structure):
    _fields_ = [
        ("position", ctypes.c_float * 2),
        ("texcoord", ctypes.c_float * 2),
        ("color", ctypes.c_float * 4),
        ("uv_bounds", ctypes.c_float * 4),
    ]


class Run(ctypes.Structure):
    _fields_ = [
        ("stream", ctypes.c_int32),
        ("first", ctypes.c_int32),
        ("count", ctypes.c_int32),
        ("texture", ctypes.c_uint32),
    ]


class Event(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int32),
        ("node", ctypes.c_int32),
        ("attribute", ctypes.c_int32),
        ("from_node", ctypes.c_int32),
        ("from_attr", ctypes.c_int32),
        ("to_node", ctypes.c_int32),
        ("to_attr", ctypes.c_int32),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("error", ctypes.c_int32),
        ("id", ctypes.c_uint64),
        # On a Value_Changed, the attribute's number after the edit; zero on
        # every other kind.
        ("value", ctypes.c_float * 4),
        ("value_count", ctypes.c_int32),
        ("extend", ctypes.c_uint8),
        ("_pad0", ctypes.c_uint8),
        ("_pad1", ctypes.c_uint8),
        ("_pad2", ctypes.c_uint8),
        ("_pad3", ctypes.c_int32),
    ]


class NodeRect(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("w", ctypes.c_float),
        ("h", ctypes.c_float),
        ("flags", ctypes.c_uint32),
        ("hover_attribute", ctypes.c_int32),
    ]


class Result(ctypes.Structure):
    _fields_ = [
        ("shapes", ctypes.POINTER(ShapeInstance)),
        ("shape_count", ctypes.c_int32),
        ("_pad0", ctypes.c_int32),
        ("glyphs", ctypes.POINTER(GlyphVertex)),
        ("glyph_count", ctypes.c_int32),
        ("_pad1", ctypes.c_int32),
        ("runs", ctypes.POINTER(Run)),
        ("run_count", ctypes.c_int32),
        ("_pad2", ctypes.c_int32),
        ("events", ctypes.POINTER(Event)),
        ("event_count", ctypes.c_int32),
        ("_pad4", ctypes.c_int32),
        ("node_rects", ctypes.POINTER(NodeRect)),
        ("rect_count", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("pan_x", ctypes.c_float),
        ("pan_y", ctypes.c_float),
        ("zoom", ctypes.c_float),
        ("_pad3", ctypes.c_float),
    ]


class EventKind(IntEnum):
    NONE = 0
    NODE_MOVED = 1
    NODE_CLICKED = 2
    NODE_ACTIVATED = 3
    CONTEXT_MENU = 4
    EDGE_ADDED = 5
    EDGE_REMOVED = 6
    EDGE_REFUSED = 7
    VALUE_CHANGED = 8
    # A press on a widget whose editor is an OVERLAY -- a colour swatch or
    # an enum. The library draws nothing and expects nothing back: the host
    # shows its own picker and writes the result into the attribute's value
    # next frame. The overlay struct itself does not cross the ABI.
    OVERLAY_REQUESTED = 9


class ConnectError(IntEnum):
    NONE = 0
    MISSING = 1
    SAME_SIDE = 2
    SELF = 3
    SAME_NODE = 4
    INPUT_TAKEN = 5
    DUPLICATE = 6
    CYCLE = 7


class PinShape(IntEnum):
    DOT = 0
    SQUARE = 1
    ARROW = 2


class PinFill(IntEnum):
    """The wire encoding, which is the library's enum shifted by one.

    0 is not a member: it means "unset", and the library lets connectedness
    choose — filled with a wire on it, hollow without. Its own `Pin_Fill` has
    the three real values, so the enum COUNT checked at load is 3, not 4.
    """

    UNSET = 0
    FILLED = 1
    HOLLOW = 2
    CORED = 3


class Widget(IntEnum):
    """What an attribute draws in its row, in the library's own order."""

    NONE = 0
    DRAG = 1
    SLIDER = 2
    CHECKBOX = 3
    BUTTON = 4
    LABEL = 5
    TEXT = 6
    COLOR = 7
    ENUM = 8


class ThemeParseError(IntEnum):
    """Why a theme file was refused, as the library's own values.

    NEGATIVE and non-contiguous, so a member's POSITION is not its value --
    which is why `_verify_layout` asks `gc_enum_name` by POSITION for every
    enum: a value would name nothing here. `-1` is absent on purpose.
    """

    NONE = 0
    NO_EQUALS = -2
    UNKNOWN_FIELD = -3
    BAD_WIDTH = -4
    BAD_NUMBER = -5
    WRONG_ARITY = -6


class PreviewFit(IntEnum):
    CONTAIN = 0
    COVER = 1
    STRETCH = 2


class Gesture(IntEnum):
    """`Node.accepts` mask. Zero is the inert value and accepts everything.

    `NONE` is the TOP bit, not the first: it is a veto that overrides the rest
    rather than another member of the same run, so it cannot share the low
    bits the individual gestures use. A binding that numbers it 1 leaves every
    node it marks accepting drags, which reads as the refusal silently not
    working.
    """

    DRAG = 1 << 0
    CLICK = 1 << 1
    WIRE = 1 << 2
    MENU = 1 << 3
    NONE = 1 << 31


class Pointer(IntEnum):
    """`Frame.pointer_flags` bits."""

    DOWN = 1 << 0
    PRESSED = 1 << 1
    CANCELLED = 1 << 2
    VIEW_MOVING = 1 << 3
    FINE = 1 << 4
    COARSE = 1 << 5
    ALT_PRESSED = 1 << 6
    DOUBLE = 1 << 7
    EXTEND = 1 << 8
    # The HOST has the pointer this frame: a popup is up, a modal is open, a
    # marquee is being dragged. The library then hovers nothing, starts
    # nothing, and DROPS a gesture already in flight rather than committing
    # it -- which is what separates it from `CANCELLED`, whose meaning is "I
    # took this press".
    CLAIMED = 1 << 9


# Attribute.kinds is a mask, not an enum value.
ATTR_INPUT: int = 1 << 0
ATTR_OUTPUT: int = 1 << 1
ATTR_CONTROL: int = 1 << 2

# Result.flags bit 0: this frame's press would be the library's. WIDER than
# any node rect a host can test -- a pin's grab area overhangs its node, so a
# press the library answers with a wire can land outside every rect in
# `node_rects`. It is the only signal a host should gate a pan on.
RESULT_POINTER_CLAIMED: int = 1 << 0

# `NodeRect.flags`. HOVERED and OVER_PORT are the pointer's position.
#
# ACTIVE marks the node a gesture was PRESSED on, which is not the same as
# an end of the wire being dragged: grabbing a CONNECTED input picks the
# existing wire up, and the anchor moves to the far output while ACTIVE
# stays on the input that was pressed -- a node the dragged wire no longer
# touches. So it answers "this node is in play" and never "these are the
# wire's endpoints"; the endpoints come from the `Edge_Removed` the press
# emits, which is why that event fires on the press rather than the release.
NODE_RECT_HOVERED: int = 1 << 0
NODE_RECT_OVER_PORT: int = 1 << 1
NODE_RECT_ACTIVE: int = 1 << 2

# `gc_sizeof` / `gc_offsetof` ids, in the library's Size_Query order.
_STRUCTS: list[tuple[str, type[ctypes.Structure]]] = [
    ("Node", Node),
    ("Attribute", Attribute),
    ("Edge", Edge),
    ("Frame", Frame),
    ("Result", Result),
    ("Run", Run),
    ("Event", Event),
    ("Str", Str),
    ("Shape", ShapeInstance),
    ("Glyph", GlyphVertex),
    ("Node_Rect", NodeRect),
    ("Theme", Theme),
]

# `gc_enum_count` ids, in the library's Enum_Query order. The COUNT is what is
# checked: an enum that gains a member changes no struct's size, so it passes
# every size check and then hands a value nothing here has a name for.
# Each row is (the library's name for the enum, how many members the binding
# expects, the binding's mirror of it or None).
#
# The MIRROR is what makes a reorder catchable. A count alone accepts two
# members swapped -- proven: `EDGE_ADDED` and `EDGE_REMOVED` exchanged loads
# clean, and every wire the user draws would unwire instead. An enum with no
# mirror here is one the binding never spells out, so only its count can be
# checked.
_ENUMS: list[tuple[str, int, type[IntEnum] | None]] = [
    ("Event_Kind", len(EventKind), EventKind),
    ("Widget", len(Widget), Widget),
    ("Connect_Error", len(ConnectError), ConnectError),
    ("Preview_Fit", len(PreviewFit), PreviewFit),
    ("Attribute_Kind", 3, None),
    ("Atlas_Error", 4, None),
    ("Size_Query", len(_STRUCTS), None),
    ("Enum_Query", 11, None),
    ("Pin_Shape", len(PinShape), PinShape),
    # Three, not four: `PinFill.UNSET` is the wire's "no opinion", not a member
    # of the library's enum, so the mirror is compared from its second member.
    ("Pin_Fill", len(PinFill) - 1, None),
    ("Theme_Parse_Error", len(ThemeParseError), ThemeParseError),
]


class LayoutMismatch(RuntimeError):
    """The built library disagrees with this file about the ABI."""


def _name_of(
    lib: ctypes.CDLL, fn: str, which: int, index: int, buf: ctypes.Array[ctypes.c_char]
) -> str | None:
    written: int = getattr(lib, fn)(which, index, buf, len(buf))
    if written < 0:
        return None
    return bytes(buf[:written]).decode()


def _verify_layout(lib: ctypes.CDLL) -> None:
    """Prove every struct field and every enum count, or raise.

    Asking by index alone proves the two lists are the same LENGTH. The field
    names are compared too, so a pair of swapped same-width fields — the case a
    size check cannot see — fails here instead of silently reading node ids as
    attribute ids.
    """
    if lib.gc_abi_version() != ABI_VERSION:
        raise LayoutMismatch(
            f"libgraph_canvas.so is ABI {lib.gc_abi_version()}, "
            f"this binding is {ABI_VERSION}"
        )

    buf = ctypes.create_string_buffer(128)
    for which, (label, struct) in enumerate(_STRUCTS):
        size: int = lib.gc_sizeof(which)
        if size != ctypes.sizeof(struct):
            raise LayoutMismatch(
                f"{label}: library says {size} bytes, binding has "
                f"{ctypes.sizeof(struct)}"
            )
        for index, (field_name, *_rest) in enumerate(struct._fields_):
            offset: int = lib.gc_offsetof(which, index)
            ours: int = getattr(struct, field_name).offset
            if offset != ours:
                raise LayoutMismatch(
                    f"{label}.{field_name}: library offset {offset}, binding {ours}"
                )
            theirs = _name_of(lib, "gc_field_name", which, index, buf)
            if theirs is not None and theirs != field_name:
                raise LayoutMismatch(
                    f"{label} field {index}: library calls it {theirs!r}, "
                    f"binding calls it {field_name!r}"
                )
        # One past the end: -1 is how the library reports the count, so a struct
        # that GAINED a field the binding has not declared is caught here rather
        # than by the size check alone.
        if lib.gc_offsetof(which, len(struct._fields_)) != -1:
            raise LayoutMismatch(
                f"{label}: library has more fields than the binding's "
                f"{len(struct._fields_)}"
            )

    name_buf = ctypes.create_string_buffer(128)
    for which, (label, expected, mirror) in enumerate(_ENUMS):
        count: int = lib.gc_enum_count(which)
        if count != expected:
            raise LayoutMismatch(
                f"enum {label}: library has {count} members, binding expects {expected}"
            )
        if mirror is None:
            continue
        # By NAME as well as by count: a swap or a rename leaves the count
        # equal while turning one event into another. The library spells its
        # members `Edge_Added`; the binding spells them `EDGE_ADDED`.
        for position, member in enumerate(mirror):
            # Asked by POSITION, not by value. For every enum here but one
            # the two coincide, which is what hid the difference:
            # `Theme_Parse_Error` runs 0, -2, -3, ... and the library names
            # nothing at -2.
            #
            # Sliced by the RETURNED LENGTH, not read as a C string: the
            # library writes the bytes and returns the count without a
            # terminator, so a shorter name leaves the previous one's tail
            # behind -- `Edge_Added` read back as `Edge_Addednued` over
            # `Context_Menued`. Clearing the buffer does not help; only the
            # length is authoritative.
            written = lib.gc_enum_name(which, position, name_buf, 128)
            if written <= 0:
                raise LayoutMismatch(
                    f"enum {label}: the library named no member at position {position}"
                )
            theirs = name_buf.raw[:written].decode().upper()
            if theirs != member.name:
                raise LayoutMismatch(
                    f"enum {label} position {position}: library calls it "
                    f"{theirs}, binding calls it {member.name}"
                )


def _declare(lib: ctypes.CDLL) -> None:
    lib.gc_new.restype = ctypes.c_void_p
    lib.gc_new.argtypes = []
    lib.gc_free.restype = None
    lib.gc_free.argtypes = [ctypes.c_void_p]
    lib.gc_abi_version.restype = ctypes.c_int32
    lib.gc_abi_version.argtypes = []
    lib.gc_sizeof.restype = ctypes.c_int32
    lib.gc_sizeof.argtypes = [ctypes.c_int32]
    lib.gc_offsetof.restype = ctypes.c_int32
    lib.gc_offsetof.argtypes = [ctypes.c_int32, ctypes.c_int32]
    lib.gc_field_name.restype = ctypes.c_int32
    lib.gc_field_name.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int32,
    ]
    lib.gc_enum_count.restype = ctypes.c_int32
    lib.gc_enum_count.argtypes = [ctypes.c_int32]
    lib.gc_enum_name.restype = ctypes.c_int32
    lib.gc_enum_name.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int32,
    ]
    lib.gc_load_atlas.restype = ctypes.c_int32
    lib.gc_load_atlas.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.gc_atlas_distance_range.restype = ctypes.c_float
    lib.gc_atlas_distance_range.argtypes = [ctypes.c_void_p]
    lib.gc_node_size.restype = ctypes.c_int32
    lib.gc_node_size.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.gc_default_theme.restype = None
    lib.gc_default_theme.argtypes = [ctypes.POINTER(Theme)]
    lib.gc_theme_parse.restype = ctypes.c_int32
    lib.gc_theme_parse.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int32,
        ctypes.POINTER(Theme),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.gc_theme_write.restype = ctypes.c_int32
    lib.gc_theme_write.argtypes = [
        ctypes.POINTER(Theme),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int32,
    ]
    lib.gc_theme_write_diff.restype = ctypes.c_int32
    lib.gc_theme_write_diff.argtypes = [
        ctypes.POINTER(Theme),
        ctypes.POINTER(Theme),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int32,
    ]
    lib.gc_theme_name_cap.restype = ctypes.c_int32
    lib.gc_theme_name_cap.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int32,
    ]
    lib.gc_theme_categories.restype = ctypes.c_int32
    lib.gc_theme_categories.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
    ]
    lib.gc_get_theme.restype = None
    lib.gc_get_theme.argtypes = [ctypes.c_void_p, ctypes.POINTER(Theme)]
    lib.gc_pin_point.restype = ctypes.c_int32
    lib.gc_pin_point.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.gc_frame.restype = ctypes.c_int32
    lib.gc_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Frame),
        ctypes.POINTER(Result),
    ]


class ThemeParseFailed(ValueError):
    """A theme file the library refused, with the 1-based line."""

    def __init__(self, reason: ThemeParseError, line: int, source: str) -> None:
        super().__init__(f"{source}:{line}: {reason.name.lower().replace('_', ' ')}")
        self.reason = reason
        self.line = line


def parse_theme(text: str, source: str = "<theme>") -> Theme:
    """A theme file as a `Theme`, starting from the library's defaults.

    An unspecified field keeps its default rather than becoming zero: a
    theme built from zero flattens the canvas, because the shading scalars
    are what lift a node off its background.

    An unknown field name RAISES. The point of the file is to stop
    guessing which spelling reached the renderer, and a line dropped in
    silence is the failure the format exists to prevent.
    """
    theme = Theme()
    raw = text.encode()
    buf = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
    line = ctypes.c_int32(0)
    code: int = ensure_loaded().gc_theme_parse(
        buf, len(raw), ctypes.byref(theme), ctypes.byref(line)
    )
    if code != 0:
        raise ThemeParseFailed(ThemeParseError(code), line.value, source)
    return theme


def write_theme(theme: Theme) -> str:
    """`theme` as the text `parse_theme` reads back."""
    lib = ensure_loaded()
    size: int = lib.gc_theme_write(ctypes.byref(theme), None, 0)
    buf = (ctypes.c_uint8 * size)()
    written: int = lib.gc_theme_write(ctypes.byref(theme), buf, size)
    return bytes(buf[:written]).decode()


def write_theme_diff(theme: Theme, baseline: Theme | None = None) -> str:
    """Only the fields where `theme` departs from `baseline` (the defaults).

    What a tuning session moved, rather than the whole palette. This is the
    form a vendored `.theme` wants: a field the file does not name keeps the
    library's default, so a partial file follows the library forward instead
    of pinning 29 values at the version they were written against, and it
    never claims a colour the app owns.
    """
    lib = ensure_loaded()
    base: Theme = baseline if baseline is not None else default_theme()
    size: int = lib.gc_theme_write_diff(
        ctypes.byref(theme), ctypes.byref(base), None, 0
    )
    buf = (ctypes.c_uint8 * size)()
    written: int = lib.gc_theme_write_diff(
        ctypes.byref(theme), ctypes.byref(base), buf, size
    )
    return bytes(buf[:written]).decode()


def parse_categories(text: str) -> dict[str, tuple[float, float, float, float]]:
    """The `attr.`-prefixed lines of a theme file, as name -> RGBA.

    The names are the HOST's own and the library never checks them: which
    kinds of row exist is shaderbox's taxonomy, and it grows on its own
    schedule. An unknown name comes back rather than being rejected.
    """
    lib = ensure_loaded()
    raw = text.encode()
    buf = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
    count: int = lib.gc_theme_categories(buf, len(raw), None, 0, None, 0)
    if count <= 0:
        return {}
    # Sized by the LIBRARY, not by a constant here. A name longer than the
    # buffer is TRUNCATED rather than refused, and two names sharing a
    # prefix then arrive as one string with two different colours behind
    # it -- which is what shaderbox's kind names look like. Measured: at 16
    # bytes `engine_uniform_alpha` and `engine_uniform_beta` both come back
    # as `engine_uniform_`. Plus one for the terminator.
    cap: int = lib.gc_theme_name_cap(buf, len(raw)) + 1
    names = (ctypes.c_uint8 * (count * cap))()
    colors = (ctypes.c_float * (count * 4))()
    lib.gc_theme_categories(buf, len(raw), names, cap, colors, count)
    out: dict[str, tuple[float, float, float, float]] = {}
    for i in range(count):
        name = bytes(names[i * cap : (i + 1) * cap]).split(b"\x00", 1)[0].decode()
        # The library returns every `attr.` line in file order and merges
        # none, so a repeated name would quietly take whichever came last.
        # The format exists to stop a line being dropped in silence, and a
        # duplicate is a line dropped.
        if name in out:
            raise ValueError(
                f"theme names category {name!r} more than once, so one "
                "colour silently replaces the other"
            )
        out[name] = (
            float(colors[i * 4]),
            float(colors[i * 4 + 1]),
            float(colors[i * 4 + 2]),
            float(colors[i * 4 + 3]),
        )
    return out


def default_theme() -> Theme:
    """The library's own palette, to override rather than to restate.

    A `Theme()` a host builds is ZEROED, and the shading scalars at zero are
    not "leave alone" -- they flatten every chamfer, lift and shadow
    (measured upstream: one distinct fill where an inherited theme paints
    seven). Inheriting also means a later retune upstream arrives for free,
    where a copied set of constants would silently fight it.
    """
    theme = Theme()
    ensure_loaded().gc_default_theme(ctypes.byref(theme))
    return theme


def ensure_loaded() -> ctypes.CDLL:
    """Load the shared library once and prove its layout. Idempotent."""
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(_LIB_PATH))
        _declare(lib)
        _verify_layout(lib)
        _LIB = lib
    return _LIB


class Blob:
    """The frame's string storage: one UTF-8 buffer, addressed by (offset, len).

    Not a pointer per string — an array of pointers costs a keepalive per
    element from Python and makes the push's allocation count scale with the
    node count.
    """

    def __init__(self) -> None:
        self.data: bytearray = bytearray()

    def add(self, text: str) -> Str:
        start: int = len(self.data)
        self.data.extend(text.encode())
        return Str(start, len(self.data) - start)

    def clear(self) -> None:
        self.data.clear()


@dataclass(slots=True)
class PortSpec:
    """One attribute on a node: a label, which side it is on, how it is drawn.

    A CONTROL carries no pin and takes no wire — it is a row in the node's
    body. `color` tints the pin; a control has none, so it reads as a plain
    label unless the host colours the text some other way.
    """

    label: str
    is_input: bool
    pin_shape: PinShape = PinShape.DOT
    pin_fill: PinFill = PinFill.UNSET
    color: tuple[float, float, float, float] | None = None
    control: bool = False
    widget: Widget = Widget.NONE
    # Up to four components; the library shows as many as are given. A LABEL
    # renders them read-only, which is what the library's `read_only` comment
    # names as its reason for existing: engine-driven values.
    value: tuple[float, ...] = ()
    text: str = ""
    read_only: bool = True


@dataclass(slots=True)
class NodeSpec:
    """One node as the host describes it. `id` is the host's own name for it."""

    id: int
    title: str
    x: float
    y: float
    ports: Sequence[PortSpec]
    preview_tex: int = 0
    preview_w: int = 0
    preview_h: int = 0
    preview_aspect: float = 1.0
    preview_fit: PreviewFit = PreviewFit.CONTAIN
    fade: float = 0.0
    dashed: bool = False
    tint: tuple[float, float, float, float] | None = None
    tint_amount: float = 0.0
    border: tuple[float, float, float, float] | None = None
    border_scale: float = 1.0
    # Up to two state RINGS, each `(colour, width, inset)` in canvas units.
    # A border's colour does not survive the trip -- the vertex format
    # carries one luminance, so red and blue arrive identical -- and a halo
    # is four plain rects, so this is how a COLOUR reaches the screen.
    halos: tuple[tuple[tuple[float, float, float, float], float, float], ...] = ()
    accepts: int = 0


@dataclass(slots=True)
class EdgeSpec:
    """One wire. The endpoints index NODES and their ATTRIBUTES within a node,
    counting inputs and outputs together in declaration order."""

    id: int
    from_node: int
    from_attr: int
    to_node: int
    to_attr: int
    color: tuple[float, float, float, float] | None = None
    width: float = 0.0


@dataclass(slots=True)
class PointerState:
    x: float = 0.0
    y: float = 0.0
    wheel: float = 0.0
    flags: int = 0


@dataclass(frozen=True, slots=True)
class View:
    """The camera after the library's own pan and zoom. Store and push back.

    `pan` is a CANVAS-SPACE POSITION THAT IS SUBTRACTED, not a screen offset:
        screen = origin + (canvas_point - pan) * zoom
    """

    pan_x: float = 0.0
    pan_y: float = 0.0
    zoom: float = 1.0


_NODE_METRICS: tuple[float, float, float] | None = None


def node_metrics() -> tuple[float, float, float]:
    """`(width, base_height, per_port_height)` as the LIBRARY lays a node out.

    Measured from the library rather than restated: a host that wants to
    place a node before drawing it -- an auto-layout, a fit -- needs the
    size the library will actually use, and a second set of constants on
    the host side is the copy that drifts. shaderbox's own tokens were 24px
    narrow and 33px short at every port count, so `rank_layout` packed
    against nodes smaller than the ones it was placing.

    The height is linear in the port count, which two probes pin and a
    third checks; measured once and cached, because it depends only on the
    library's own metrics and those do not change within a build.
    """
    global _NODE_METRICS
    if _NODE_METRICS is None:
        canvas = Canvas()
        canvas.load_atlas()
        sizes: list[tuple[float, float]] = []
        for count in (1, 2, 3):
            ports = [PortSpec(f"p{i}", True) for i in range(count)]
            node = NodeSpec(id=1, title="m", x=0, y=0, ports=ports)
            canvas.frame([node], [], (600.0, 600.0), View(), PointerState())
            measured = canvas.node_size(0)
            sizes.append(measured if measured is not None else (0.0, 0.0))
        canvas.release()
        # Measured from ONE port up, not from zero: a node with no ports has
        # no port section at all, so its height is 22 below the one-port
        # node rather than the 18 each further port adds. Probing from zero
        # reads that discontinuity as the slope and under-measures every
        # node thereafter.
        step = sizes[1][1] - sizes[0][1]
        if sizes[2][1] - sizes[1][1] != step:
            raise RuntimeError(
                f"the library's node height is not linear above one port: {sizes}"
            )
        _NODE_METRICS = (sizes[0][0], sizes[0][1] - step, step)
    return _NODE_METRICS


class Canvas:
    """One live graph canvas: the library handle plus the arrays it reads.

    The arrays are kept between frames and refilled in place — allocating fresh
    ones per frame is nothing at twenty nodes and the wrong shape at five
    hundred. They are held as attributes because ctypes stores the POINTER when
    one is assigned into the frame and drops its own reference: an array built
    in a local would be freed while the library still held it, which works in
    testing and corrupts under GC pressure.
    """

    def __init__(self) -> None:
        # Bound BEFORE anything that can raise, so a failed construction still
        # has the two attributes `__del__` reads. Without them a `gc_new` that
        # returns null raises `RuntimeError`, and the collector then raises
        # `AttributeError` inside `__del__` on top of it.
        self._handle: int = 0
        self._lib: ctypes.CDLL = ensure_loaded()
        lib = self._lib
        handle = lib.gc_new()
        if not handle:
            raise RuntimeError("gc_new returned null")
        self._handle = handle
        self._frame: Frame = Frame()
        self._result: Result = Result()
        self._blob: Blob = Blob()
        self._nodes: ctypes.Array[Node] = (Node * 0)()
        self._attrs: ctypes.Array[Attribute] = (Attribute * 0)()
        self._edges: ctypes.Array[Edge] = (Edge * 0)()
        self._strings: ctypes.Array[ctypes.c_uint8] = (ctypes.c_uint8 * 1)()
        self._theme: Theme | None = None
        self.atlas_loaded: bool = False
        self.distance_range: float = 0.0

    def load_atlas(self, json_path: Path = ATLAS_JSON_PATH) -> None:
        """Load the glyph METRICS. The image is the renderer's to upload.

        Until an atlas is loaded the library emits no text — nodes draw without
        names rather than failing.
        """
        if self._lib.gc_load_atlas(self._handle, str(json_path).encode()) != 0:
            raise RuntimeError(f"gc_load_atlas refused {json_path}")
        self.atlas_loaded = True
        self.distance_range = float(self._lib.gc_atlas_distance_range(self._handle))

    def _grow(self, nodes: int, attrs: int, edges: int) -> None:
        if len(self._nodes) < nodes:
            self._nodes = (Node * max(nodes, len(self._nodes) * 2))()
        if len(self._attrs) < attrs:
            self._attrs = (Attribute * max(attrs, len(self._attrs) * 2))()
        if len(self._edges) < edges:
            self._edges = (Edge * max(edges, len(self._edges) * 2))()

    def frame(
        self,
        nodes: Sequence[NodeSpec],
        edges: Sequence[EdgeSpec],
        size: tuple[float, float],
        view: View,
        pointer: PointerState,
        origin: tuple[float, float] = (0.0, 0.0),
        theme: Theme | None = None,
        dt: float = 0.0,
    ) -> Result:
        """Push one frame and get back geometry plus what the user did.

        `dt` is seconds since the host's previous frame and is what advances
        every eased highlight, a row's hover among them. The default of zero
        holds each ease where it is, which is right for a probe pushing one
        frame and wrong for a host that renders continuously -- a canvas
        that never sends a real `dt` has no row hover at all.

        The nodes and the attributes are filled in ONE pass, because each node
        names a contiguous run of the flat attribute array. Building them in
        separate passes is the mistake that fails silently: mis-ranged nodes
        read each other's attributes and everything still draws.
        """
        attr_total: int = sum(len(n.ports) for n in nodes)
        self._grow(len(nodes), attr_total, len(edges))
        self._blob.clear()

        cursor: int = 0
        for index, spec in enumerate(nodes):
            first: int = cursor
            for port in spec.ports:
                a = self._attrs[cursor]
                a.label = self._blob.add(port.label)
                # A MASK, and which bits are set decides three things: whether
                # there is a pin, which side it sits on, and whether the row is
                # the user's to drive. A pin is drawn for Input or Output and
                # nothing else, so a row carrying NEITHER is the only pinless
                # row the library can express -- which is what an engine-written
                # value is.
                if port.control:
                    a.kinds = ATTR_CONTROL
                elif port.is_input:
                    a.kinds = ATTR_INPUT
                else:
                    a.kinds = ATTR_OUTPUT
                a.widget = int(port.widget)
                count = min(len(port.value), 4)
                a.value_count = count
                for component in range(count):
                    a.value[component] = port.value[component]
                if port.text:
                    a.value_is_text = 1
                    a.text = self._blob.add(port.text)
                else:
                    a.value_is_text = 0
                    a.text = Str(0, 0)
                a.opt_first = 0
                a.opt_count = 0
                a.height = 0.0
                a.speed = 0.0
                a.min = 0.0
                a.max = 0.0
                a.pin_shape = int(port.pin_shape)
                a.pin_fill = int(port.pin_fill)
                a.read_only = 1 if port.read_only else 0
                a.pin_on_preview = 0
                # A port's colour lands on its PIN, and a control has none,
                # so for a control it lands on the ROW instead. The two are
                # separate fields upstream because a pin colour and a row
                # colour are different questions -- a host can tint one
                # input's pin without restating what the row IS.
                a.pin_color_set = 0
                a.role_color_set = 0
                if port.color is not None:
                    if port.control:
                        a.role_color_set = 1
                        (
                            a.role_color_r,
                            a.role_color_g,
                            a.role_color_b,
                            a.role_color_a,
                        ) = port.color
                    else:
                        a.pin_color_set = 1
                        (
                            a.pin_color_r,
                            a.pin_color_g,
                            a.pin_color_b,
                            a.pin_color_a,
                        ) = port.color
                cursor += 1

            n = self._nodes[index]
            n.name = self._blob.add(spec.title)
            n.title = self._blob.add(spec.title)
            n.pos_x, n.pos_y = spec.x, spec.y
            n.attr_first, n.attr_count = first, cursor - first
            n.preview_tex = spec.preview_tex
            n.preview_w, n.preview_h = spec.preview_w, spec.preview_h
            n.preview_aspect = spec.preview_aspect
            n.preview_fit = int(spec.preview_fit)
            n.fade = spec.fade
            n.dashed = 1 if spec.dashed else 0
            n.tint_amount = spec.tint_amount
            if spec.tint is None:
                n.tint_set = 0
            else:
                n.tint_set = 1
                n.tint_r, n.tint_g, n.tint_b, n.tint_a = spec.tint
            n.border_scale = spec.border_scale
            for ring in range(2):
                if ring < len(spec.halos):
                    colour, width, inset = spec.halos[ring]
                    n.halo_color[ring][0] = colour[0]
                    n.halo_color[ring][1] = colour[1]
                    n.halo_color[ring][2] = colour[2]
                    n.halo_color[ring][3] = colour[3]
                    n.halo_width[ring] = width
                    n.halo_inset[ring] = inset
                else:
                    # Zero width is inert, and the arrays are reused across
                    # frames, so a ring dropped this frame must be cleared.
                    n.halo_width[ring] = 0.0
            if spec.border is None:
                n.border_set = 0
            else:
                n.border_set = 1
                n.border_r, n.border_g, n.border_b, n.border_a = spec.border
            n.id = spec.id
            n.accepts = spec.accepts

        for index, wire in enumerate(edges):
            e = self._edges[index]
            e.from_node, e.from_attr = wire.from_node, wire.from_attr
            e.to_node, e.to_attr = wire.to_node, wire.to_attr
            e.id = wire.id
            e.width = wire.width
            if wire.color is None:
                e.color_set = 0
            else:
                e.color_set = 1
                e.color_r, e.color_g, e.color_b, e.color_a = wire.color

        blob_len: int = len(self._blob.data)
        if len(self._strings) < max(blob_len, 1):
            self._strings = (ctypes.c_uint8 * max(blob_len, 1))()
        if blob_len:
            self._strings[:blob_len] = self._blob.data

        f = self._frame
        f.nodes = ctypes.cast(self._nodes, ctypes.POINTER(Node))
        f.node_count = len(nodes)
        f.attrs = ctypes.cast(self._attrs, ctypes.POINTER(Attribute))
        f.attr_count = attr_total
        f.edges = ctypes.cast(self._edges, ctypes.POINTER(Edge))
        f.edge_count = len(edges)
        f.strings = ctypes.cast(self._strings, ctypes.POINTER(ctypes.c_uint8))
        f.string_len = blob_len
        f.width, f.height = size
        f.pan_x, f.pan_y = view.pan_x, view.pan_y
        f.zoom = view.zoom
        f.origin_x, f.origin_y = origin
        f.pointer_x, f.pointer_y = pointer.x, pointer.y
        f.wheel = pointer.wheel
        f.dt = dt
        f.pointer_flags = pointer.flags
        # Null is KEEP, not reset: the theme is pushed once and the handle
        # holds it. Kept alive on `self` because ctypes drops the reference
        # the moment the expression ends.
        if theme is not None:
            self._theme = theme
            f.theme = ctypes.pointer(self._theme)
        else:
            f.theme = ctypes.POINTER(Theme)()

        # Gate on the return: on a refusal the out-parameter is left UNTOUCHED
        # rather than zeroed, so reading it anyway serves the previous frame.
        if (
            self._lib.gc_frame(
                self._handle, ctypes.byref(f), ctypes.byref(self._result)
            )
            != 1
        ):
            raise RuntimeError("gc_frame refused the frame")
        return self._result

    def pin_point(
        self, node: int, attribute: int, output: bool
    ) -> tuple[float, float] | None:
        """One pin's centre in SCREEN space, for the most recent frame.

        A pin is not where its row is: the grab area straddles the node's edge
        so an edge pin is reachable without the body claiming the press, and
        the hover-reporting band is the whole row. A host computing a point
        from a node rect, or from where the hover answers, lands on the body
        and starts a node drag -- which is indistinguishable from the wire
        gesture not existing.

        `None` for a node or attribute outside the last frame.
        """
        x = ctypes.c_float()
        y = ctypes.c_float()
        side: int = 1 if output else 0
        if (
            self._lib.gc_pin_point(
                self._handle, node, attribute, side, ctypes.byref(x), ctypes.byref(y)
            )
            != 0
        ):
            return None
        return (x.value, y.value)

    def node_size(self, index: int) -> tuple[float, float] | None:
        """The size the library gives a node in the most recent frame."""
        w = ctypes.c_float()
        h = ctypes.c_float()
        if (
            self._lib.gc_node_size(
                self._handle, index, ctypes.byref(w), ctypes.byref(h)
            )
            != 0
        ):
            return None
        return (w.value, h.value)

    def release(self) -> None:
        # `getattr`, because an `__init__` that raised -- `ensure_loaded()`
        # failing, `gc_new` returning null -- leaves the attributes unbound
        # while the collector still calls `__del__`. An `AttributeError`
        # raised during finalisation is printed and swallowed, which hides
        # the real error underneath it.
        if getattr(self, "_handle", 0):
            self._lib.gc_free(self._handle)
            self._handle = 0

    def __del__(self) -> None:
        self.release()
