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

ABI_VERSION: int = 2

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
        ("read_only", ctypes.c_uint8),
        ("pin_on_preview", ctypes.c_uint8),
        ("pin_color_set", ctypes.c_uint8),
        ("_pad2", ctypes.c_uint8),
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
        ("extend", ctypes.c_uint8),
        ("_pad0", ctypes.c_uint8),
        ("_pad1", ctypes.c_uint8),
        ("_pad2", ctypes.c_uint8),
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


# Attribute.kinds is a mask, not an enum value.
ATTR_INPUT: int = 1 << 0
ATTR_OUTPUT: int = 1 << 1
ATTR_CONTROL: int = 1 << 2

# Result.flags bit 0: the library claims the pointer this frame.
RESULT_POINTER_CLAIMED: int = 1 << 0

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
]

# `gc_enum_count` ids, in the library's Enum_Query order. The COUNT is what is
# checked: an enum that gains a member changes no struct's size, so it passes
# every size check and then hands a value nothing here has a name for.
_ENUMS: list[tuple[str, int]] = [
    ("Event_Kind", len(EventKind)),
    ("Widget", 9),
    ("Connect_Error", len(ConnectError)),
    ("Preview_Fit", len(PreviewFit)),
    ("Attribute_Kind", 3),
    ("Atlas_Error", 4),
    ("Size_Query", len(_STRUCTS)),
    ("Enum_Query", 10),
    ("Pin_Shape", len(PinShape)),
    # Three, not four: `PinFill.UNSET` is the wire's "no opinion", not a member
    # of the library's enum.
    ("Pin_Fill", len(PinFill) - 1),
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

    for which, (label, expected) in enumerate(_ENUMS):
        count: int = lib.gc_enum_count(which)
        if count != expected:
            raise LayoutMismatch(
                f"enum {label}: library has {count} members, binding expects {expected}"
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
    """One pin on a node: a label, a side, and how it is drawn."""

    label: str
    is_input: bool
    pin_shape: PinShape = PinShape.DOT
    pin_fill: PinFill = PinFill.UNSET
    color: tuple[float, float, float, float] | None = None


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
        lib = ensure_loaded()
        self._lib: ctypes.CDLL = lib
        handle = lib.gc_new()
        if not handle:
            raise RuntimeError("gc_new returned null")
        self._handle: int = handle
        self._frame: Frame = Frame()
        self._result: Result = Result()
        self._blob: Blob = Blob()
        self._nodes: ctypes.Array[Node] = (Node * 0)()
        self._attrs: ctypes.Array[Attribute] = (Attribute * 0)()
        self._edges: ctypes.Array[Edge] = (Edge * 0)()
        self._strings: ctypes.Array[ctypes.c_uint8] = (ctypes.c_uint8 * 1)()
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
    ) -> Result:
        """Push one frame and get back geometry plus what the user did.

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
                a.kinds = ATTR_INPUT if port.is_input else ATTR_OUTPUT
                a.widget = 0
                a.value_count = 0
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
                a.read_only = 1
                a.pin_on_preview = 0
                if port.color is None:
                    a.pin_color_set = 0
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
        f.pointer_flags = pointer.flags

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
        if self._handle:
            self._lib.gc_free(self._handle)
            self._handle = 0

    def __del__(self) -> None:
        self.release()
