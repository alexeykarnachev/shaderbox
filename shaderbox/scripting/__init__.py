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
    BrainStatus,
    EngineNode,
    ScriptEngine,
    ScriptProbe,
    brain_stub_for,
    is_scriptable,
)
from shaderbox.scripting.errors import ScriptError
from shaderbox.scripting.outputs import Array, Text, Vec2, Vec3, Vec4

__all__ = [
    "EXPORT_MOUSE",
    "Array",
    "Behavior",
    "BrainStatus",
    "Ctx",
    "EngineContext",
    "EngineNode",
    "MouseState",
    "PythonBehavior",
    "ScriptBehavior",
    "ScriptEngine",
    "ScriptError",
    "ScriptProbe",
    "Text",
    "Vec2",
    "Vec3",
    "Vec4",
    "brain_stub_for",
    "coerce_one",
    "is_scriptable",
]
