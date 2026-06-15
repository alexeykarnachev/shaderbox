"""The CPU-script engine (feature 041) — a uniform becomes a first-class object with a per-tick
stateful behavior class (`update(self, ctx) -> <typed output>`). Public surface the headless
ProjectSession drives."""

from shaderbox.scripting.behavior import (
    Behavior,
    PythonBehavior,
    ScriptBehavior,
    coerce_one,
)
from shaderbox.scripting.context import EXPORT_MOUSE, Ctx, EngineContext, MouseState
from shaderbox.scripting.engine import (
    EngineNode,
    ScriptEngine,
    ScriptProbe,
    ScriptStatus,
    is_scriptable,
    normalize_script_tabs,
    script_stub_for,
)
from shaderbox.scripting.errors import ScriptError
from shaderbox.scripting.outputs import Array, Text, Vec2, Vec3, Vec4

__all__ = [
    "EXPORT_MOUSE",
    "Array",
    "Behavior",
    "Ctx",
    "EngineContext",
    "EngineNode",
    "MouseState",
    "PythonBehavior",
    "ScriptBehavior",
    "ScriptEngine",
    "ScriptError",
    "ScriptProbe",
    "ScriptStatus",
    "Text",
    "Vec2",
    "Vec3",
    "Vec4",
    "coerce_one",
    "is_scriptable",
    "normalize_script_tabs",
    "script_stub_for",
]
