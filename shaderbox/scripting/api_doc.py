"""The generated SCRIPT API block the copilot prompt's RARE tier carries (feature 059 D3) — the
Python side of a document script, rendered FROM the live types so the block cannot drift from them.

Names, signatures and field types come from the code; the semantics beside them are authored, and
`tests/test_script_api_doc.py` pins the join: `_CTX_GLOSS`'s keys must equal the ctx dataclass
fields, every type name `engine.py::_stub_kind` can return must be a key in `_VALUE_SHAPE_GLOSS`,
and every public `Vec` member plus every operator dunder must reach the rendered text.

This module reaches only for the GL-free halves of the package (`context` + `outputs`) — never
`engine`/`behavior`, which import `moderngl`. A runtime `sys.modules` assertion cannot express that:
importing ANY submodule executes the package `__init__`, which re-exports the GL half, so the
invariant is over this module's OWN imports."""

from collections.abc import Callable, Iterable
from inspect import cleandoc, signature
from textwrap import fill

from shaderbox.scripting.context import EXPORT_MOUSE, EngineContext, MouseState
from shaderbox.scripting.outputs import Array, Text, Vec2, Vec3, Vec4

_WRAP_WIDTH: int = 100


def _call_form(cls: type, ctor: Callable[..., object]) -> str:
    # `Vec3(x,y,z)` / `Array(values)` from the real constructor, so a renamed parameter cannot leave
    # the block advertising the old name.
    params = [p for p in signature(ctor).parameters if p not in ("self", "cls")]
    return f"{cls.__name__}({','.join(params)})"


_VEC_FORMS: str = " / ".join(_call_form(c, c.__new__) for c in (Vec2, Vec3, Vec4))

# Type name (as `engine.py::_stub_kind` names it) -> its one-line value-shape prose. Names that share
# a shape share the gloss verbatim; the render de-dupes in insertion order, so the printed list groups
# exactly as this map does.
_VALUE_SHAPE_GLOSS: dict[str, str] = {
    "float": "float|int -> scalar",
    "int": "float|int -> scalar",
    "Vec2": f"{_VEC_FORMS} -> vec2/3/4",
    "Vec3": f"{_VEC_FORMS} -> vec2/3/4",
    "Vec4": f"{_VEC_FORMS} -> vec2/3/4",
    "Array": (
        f"{_call_form(Array, Array.__init__)} -> a numeric array uniform (flat numbers, or ROWS of "
        "Vec/list, auto-flattened)"
    ),
    "Text": f"{_call_form(Text, Text.__init__)} or a plain str -> a uint[] glyph array",
}

# The `_Vec` operator dunders, each mapped to the prose the block prints for it — this map IS the
# rendered operator list, so a dunder on the Vec classes that is missing here (or here and not on
# them) fails the coverage test. Dunders sharing an arity/semantics share their prose verbatim.
_VEC_OPERATOR_GLOSS: dict[str, str] = {
    "__add__": "`+ -` (same length)",
    "__sub__": "`+ -` (same length)",
    "__mul__": "`* /` (scalar or component-wise)",
    "__rmul__": "`* /` (scalar or component-wise)",
    "__truediv__": "`* /` (scalar or component-wise)",
    "__neg__": "unary `-`",
}

_MOUSE_FIELDS: str = ", ".join(f"`{n}`" for n in MouseState.__dataclass_fields__)
_EXPORT_MOUSE_AT: str = f"{EXPORT_MOUSE.x:g},{EXPORT_MOUSE.y:g}"

# Authored gloss per `EngineContext` field, keyed by field name (the type comes from the annotation).
# `mouse` MUST keep the freeze caveat: a script driven off the cursor reads STATIC in every probe and
# every export even when it is correct.
#
# TWO renderings of one table, because the readers differ (079 D3). `_CTX_GLOSS` is the PROMPT's:
# terse, one clause, and it costs tokens on every request carrying the RARE tier. `_CTX_HELP` is
# what a person reads in the editor's `K` note — full sentences, one fact per line, no `;`-joined
# lists. Every field is in both; the completeness test walks the dataclass against each.
_CTX_GLOSS: dict[str, str] = {
    "t": "seconds since playback started",
    "dt": "seconds since the previous frame",
    "frame": "frame index from 0",
    "mouse": (
        f"({_MOUSE_FIELDS} -- FROZEN at {_EXPORT_MOUSE_AT} on export and in the "
        "headless probe, where down is False and prev equals x/y; x/y and prev_x/prev_y are the "
        "current and PREVIOUS cursor position in 0..1 y-up, down is True while LMB is held over "
        "the canvas)"
    ),
}

_CTX_HELP: dict[str, str] = {
    "t": "Seconds since the document started playing.",
    "dt": (
        "Seconds since the previous frame.\n"
        "Multiply a rate by it to advance state at the same speed whatever the frame rate."
    ),
    "frame": "The frame index, counting from 0 at the start of playback.",
    "mouse": (
        "The cursor over the canvas, normalized to 0..1 with y pointing up.\n"
        "x, y: the current position; 0,0 is the bottom-left corner.\n"
        "prev_x, prev_y: last frame's position, so a shader can stamp the capsule between "
        "the two rather than one disc per frame.\n"
        "down: True while the left button is held over the canvas.\n"
        f"On export and in the headless probe the cursor freezes at {_EXPORT_MOUSE_AT} with "
        "down False and prev equal to the position. A script driven off the cursor therefore "
        "reads as static in every probe, even when it is correct."
    ),
}

_IMPORT_NAMES: str = ", ".join(
    ["ScriptBehavior", "Ctx", *(c.__name__ for c in (Vec2, Vec3, Vec4, Array, Text))]
)

# The names the engine injects into every script (`behavior.py::_build_globals`), as the
# editor's intelligence classes them: the script API, not a local and not a builtin.
API_NAMES: frozenset[str] = frozenset(
    {
        "ScriptBehavior",
        "Ctx",
        MouseState.__name__,
        *(c.__name__ for c in (Vec2, Vec3, Vec4, Array, Text)),
    }
)


# What `K` shows for each injected API name: a summary line, a blank line, then the detail
# (PEP 257 with the Google layout, 079 D3). A value type's entry is the shape it coerces to;
# `Ctx` and `MouseState` read their class docstrings, which say the same thing once.
_API_HELP: dict[str, str] = {
    "ScriptBehavior": (
        "The base class a document script's Behavior extends.\n"
        "\n"
        "Define `__init__(self)` for state that survives across frames and\n"
        "`update(self, ctx) -> dict` for the values this frame drives. Press K on `Ctx` for\n"
        "what `update` receives."
    ),
    "Vec2": "A 2-component vector, driving a vec2 uniform.",
    "Vec3": "A 3-component vector, driving a vec3 uniform.",
    "Vec4": "A 4-component vector, driving a vec4 uniform.",
    "Array": (
        "A numeric array uniform's values.\n"
        "\n"
        "Takes flat numbers, or rows of Vec/list which are flattened for you. The length must\n"
        "match the array the shader declares."
    ),
    "Text": (
        "A string, driving a uint[] glyph array.\n"
        "\n"
        "A plain str coerces the same way. Longer text than the array holds is cut."
    ),
}


def _class_help(cls: type) -> str:
    # The class's own docstring — one authored home per concept (079 D3), so `K` on `Ctx` reads
    # what `context.py` says rather than a second summary that can drift from it.
    return cleandoc(cls.__doc__ or "")


def api_symbol_doc(name: str) -> tuple[str, str]:
    """(signature, doc) for one injected API name, for the editor's completion and `K`: the
    call form of a value type, the class name of the others."""
    by_name: dict[str, type] = {c.__name__: c for c in (Vec2, Vec3, Vec4, Array, Text)}
    cls = by_name.get(name)
    if cls is not None:
        ctor = cls.__new__ if name.startswith("Vec") else cls.__init__
        return _call_form(cls, ctor), _API_HELP[name]
    if name == "Ctx":
        return "Ctx", _class_help(EngineContext)
    if name == MouseState.__name__:
        return MouseState.__name__, _class_help(MouseState)
    return name, _API_HELP["ScriptBehavior"]


def ctx_field_gloss(field: str) -> str:
    """What a `ctx` field means, for the editor's `K` and completion detail; "" when the
    field has no gloss."""
    return _CTX_HELP.get(field, "")


def _type_name(annotation: object) -> str:
    return annotation.__name__ if isinstance(annotation, type) else str(annotation)


def _dedup_join(glosses: Iterable[str], sep: str) -> str:
    # Order-preserving de-dup: names sharing a gloss (float/int, the Vec arities, `*` and `/`) print
    # once, so a gloss map groups its entries just by repeating a value.
    out: list[str] = []
    for gloss in glosses:
        if gloss not in out:
            out.append(gloss)
    return sep.join(out)


def _ctx_fields() -> str:
    parts: list[str] = []
    for name, field in EngineContext.__dataclass_fields__.items():
        # An undocumented new field degrades to bare `name type` rather than breaking prompt build.
        gloss = _CTX_GLOSS.get(name, "")
        joiner = "" if gloss.startswith("(") else " "
        parts.append(f"`{name}` {_type_name(field.type)}{joiner}{gloss}".rstrip())
    return ", ".join(parts)


def _bullet(text: str) -> str:
    return fill(
        text,
        width=_WRAP_WIDTH,
        subsequent_indent="  ",
        break_long_words=False,
        break_on_hyphens=False,
    )


def script_api_summary() -> str:
    """The RARE-tier SCRIPT API block: the script contract, the ctx surface, the legal value shapes
    and the vector API, rendered from `context.py` + `outputs.py`."""
    bullets = [
        "- `class Behavior(ScriptBehavior)`: `__init__(self)` runs once; `update(self, ctx) -> dict` "
        "runs every frame. A bare key drives that uniform on EVERY pass declaring it: "
        "{uniform_name: value}. A key whose value is a dict is a PASS BLOCK driving that one pass, "
        "{pass: {uniform: value}}, and a pass block WINS over a bare key for the same uniform on "
        "that pass. State on `self.*` persists across frames; a key you omit (or map to None) "
        "stays MANUAL.",
        f"- ctx: {_ctx_fields()}.",
        f"- Legal value shapes: {_dedup_join(_VALUE_SHAPE_GLOSS.values(), '; ')}. A bare FLAT list "
        "also coerces (a vec, or an exact-length numeric array); a NESTED bare list does not -- that "
        "is what Array is for.",
        f"- Vec2/Vec3/Vec4 are real vectors: `.x .y .z .w`, "
        f"{_dedup_join(_VEC_OPERATOR_GLOSS.values(), ', ')}, `.dot(o)`, `.length()`, "
        "`.normalized()`; Vec3 also `.cross(o)`.",
        f"- `from shaderbox.scripting import {_IMPORT_NAMES}` (the engine injects these too); a "
        "script is plain Python -- `import math` and the stdlib work.",
    ]
    header = "SCRIPT API (generated from shaderbox/scripting -- the Python side of a document script):"
    return "\n".join([header, *(_bullet(b) for b in bullets)])
