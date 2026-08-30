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
from inspect import signature
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
_CTX_GLOSS: dict[str, str] = {
    "t": "seconds",
    "dt": "",
    "frame": "",
    "mouse": (
        f"({_MOUSE_FIELDS} in 0..1, y-up -- FROZEN at {_EXPORT_MOUSE_AT} on export and in the "
        "headless probe)"
    ),
}

_IMPORT_NAMES: str = ", ".join(
    ["ScriptBehavior", "Ctx", *(c.__name__ for c in (Vec2, Vec3, Vec4, Array, Text))]
)


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
        "runs every frame and returns {uniform_name: value}. State on `self.*` persists across "
        "frames; a key you omit (or map to None) stays MANUAL.",
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
