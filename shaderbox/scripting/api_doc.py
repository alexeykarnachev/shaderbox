"""The generated SCRIPT API block the copilot prompt's RARE tier carries (feature 059 D3) — the
Python side of a document script, rendered FROM the live types so the block cannot drift from them.

Names, signatures and field types come from the code; the semantics beside them are authored, and
`tests/test_script_api_doc.py` pins the join: `_CONTEXT_GLOSS`'s keys must equal the context dataclass
fields, every type name `engine.py::_stub_kind` can return must be a key in `_VALUE_SHAPE_GLOSS`,
and every public `Vec` member plus every operator dunder must reach the rendered text.

This module reaches only for the GL-free halves of the package (`context` + `outputs`) — never
`engine`/`behavior`, which import `moderngl`. A runtime `sys.modules` assertion cannot express that:
importing ANY submodule executes the package `__init__`, which re-exports the GL half, so the
invariant is over this module's OWN imports."""

from collections.abc import Iterable
from inspect import cleandoc
from textwrap import fill

from shaderbox.scripting.context import EXPORT_MOUSE, MouseState, ScriptContext

_WRAP_WIDTH: int = 100

# The GLSL shape a uniform has (as `engine.py::_stub_kind` labels it) -> what a script returns for
# it. Every value is PLAIN PYTHON: the engine shapes it against the live `moderngl.Uniform`, so
# there is no wrapper type to learn. Shapes sharing a form share the gloss verbatim; the render
# de-dupes in insertion order, so the printed list groups exactly as this map does.
_VALUE_SHAPE_GLOSS: dict[str, str] = {
    "float": "a number -> a scalar (rounded for an int uniform)",
    "int": "a number -> a scalar (rounded for an int uniform)",
    "vec2": "a list or tuple of N numbers -> vecN",
    "vec3": "a list or tuple of N numbers -> vecN",
    "vec4": "a list or tuple of N numbers -> vecN",
    "list": (
        "a list of numbers, flat or as rows -> an array uniform "
        "([[x,y],[x,y]] and [x,y,x,y] both drive a vec2[2])"
    ),
    "str": "a str -> a uint[] glyph array",
}

_MOUSE_FIELDS: str = ", ".join(f"`{n}`" for n in MouseState.__dataclass_fields__)
_EXPORT_MOUSE_AT: str = f"{EXPORT_MOUSE.x:g},{EXPORT_MOUSE.y:g}"

# Authored gloss per `ScriptContext` field, keyed by field name (the type comes from the annotation).
# `mouse` MUST keep the freeze caveat: a script driven off the cursor reads STATIC in every probe and
# every export even when it is correct.
#
# TWO renderings of one table, because the readers differ (079 D3). `_CONTEXT_GLOSS` is the PROMPT's:
# terse, one clause, and it costs tokens on every request carrying the RARE tier. `_CONTEXT_HELP` is
# what a person reads in the editor's `K` note — full sentences, one fact per line, no `;`-joined
# lists. Every field is in both; the completeness test walks the dataclass against each.
_CONTEXT_GLOSS: dict[str, str] = {
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

_CONTEXT_HELP: dict[str, str] = {
    "t": (
        "Seconds since the document started playing.\n"
        "\n"
        "Reset along with the rest of the document's clock."
    ),
    "dt": (
        "Seconds since the previous frame.\n"
        "\n"
        "Multiply a rate by it to advance state at the same speed whatever the frame rate."
    ),
    "frame": (
        "The frame index, counting from 0.\n"
        "\n"
        "Rises by one per drawn frame, so it counts frames rather than time."
    ),
    "mouse": (
        "The cursor over the canvas, as a MouseState.\n"
        "\n"
        "Normalized to 0..1 with y pointing up, so 0,0 is the bottom-left corner.\n"
        "\n"
        "Attributes:\n"
        "    x: Horizontal position, 0 at the left edge and 1 at the right.\n"
        "    y: Vertical position, 0 at the bottom edge and 1 at the top.\n"
        "    down: Whether the left button is held over the canvas.\n"
        "    prev_x: Last frame's x, equal to x on the first frame and on re-entry.\n"
        "    prev_y: Last frame's y, under the same rule.\n"
        "\n"
        f"On export and in the headless probe the cursor freezes at {_EXPORT_MOUSE_AT} with\n"
        "down False and prev equal to the position, so a script driven off the cursor reads\n"
        "as static in every probe even when it is correct."
    ),
}

_IMPORT_NAMES: str = "ScriptBehavior, ScriptContext"

# The names the engine injects into every script (`behavior.py::_build_globals`), as the
# editor's intelligence classes them: the script API, not a local and not a builtin.
API_NAMES: frozenset[str] = frozenset(
    {"ScriptBehavior", "ScriptContext", MouseState.__name__}
)


# What `K` shows for each injected API name: a summary line, a blank line, then the detail
# (PEP 257 with the Google layout, 079 D3). `ScriptContext` and `MouseState` read their class docstrings,
# so the concept has one authored home.
_API_HELP: dict[str, str] = {
    "ScriptBehavior": (
        "The base class a document script's Behavior extends.\n"
        "\n"
        "Define __init__(self) for state that survives across frames, and\n"
        "update(self, context) -> dict for the values each frame drives. ScriptContext\n"
        "documents what update receives."
    ),
}


def _class_help(cls: type) -> str:
    # The class's own docstring — one authored home per concept (079 D3), so `K` on `ScriptContext` reads
    # what `context.py` says rather than a second summary that can drift from it.
    return cleandoc(cls.__doc__ or "")


def api_symbol_doc(name: str) -> tuple[str, str]:
    """(signature, doc) for one injected API name, for the editor's completion and `K`."""
    if name == "ScriptContext":
        return "ScriptContext", _class_help(ScriptContext)
    if name == MouseState.__name__:
        return MouseState.__name__, _class_help(MouseState)
    return name, _API_HELP["ScriptBehavior"]


def context_field_gloss(field: str) -> str:
    """What a `context` field means, for the editor's `K` and completion detail; "" when the
    field has no gloss."""
    return _CONTEXT_HELP.get(field, "")


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


def _context_fields() -> str:
    parts: list[str] = []
    for name, field in ScriptContext.__dataclass_fields__.items():
        # An undocumented new field degrades to bare `name type` rather than breaking prompt build.
        gloss = _CONTEXT_GLOSS.get(name, "")
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
    """The RARE-tier SCRIPT API block: the script contract, the context surface and the legal value
    shapes, rendered from `context.py`."""
    bullets = [
        "- `class Behavior(ScriptBehavior)`: `__init__(self)` runs once; `update(self, context) -> dict` "
        "runs every frame. A bare key drives that uniform on EVERY pass declaring it: "
        "{uniform_name: value}. A key whose value is a dict is a PASS BLOCK driving that one pass, "
        "{pass: {uniform: value}}, and a pass block WINS over a bare key for the same uniform on "
        "that pass. State on `self.*` persists across frames; a key you omit (or map to None) "
        "stays MANUAL.",
        f"- context: {_context_fields()}.",
        f"- Legal value shapes, all PLAIN PYTHON (there are no wrapper types): "
        f"{_dedup_join(_VALUE_SHAPE_GLOSS.values(), '; ')}. An array's length must match the "
        "uniform's exactly.",
        f"- `from shaderbox.scripting import {_IMPORT_NAMES}` -- that module and those names are "
        "the ONLY part of shaderbox a script may import; any other shaderbox module raises at "
        "compile. A script is otherwise plain Python -- `import math`, numpy and the stdlib work.",
    ]
    header = "SCRIPT API (generated from shaderbox/scripting -- the Python side of a document script):"
    return "\n".join([header, *(_bullet(b) for b in bullets)])
