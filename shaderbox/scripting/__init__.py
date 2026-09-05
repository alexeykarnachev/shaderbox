"""The CPU-script engine (feature 041) — a uniform becomes a first-class object with a per-tick
stateful behavior class (`update(self, context) -> <typed output>`). Public surface the headless
ProjectSession drives."""

from shaderbox.scripting.behavior import (
    Behavior,
    PythonBehavior,
    ScriptBehavior,
    coerce_one,
)
from shaderbox.scripting.context import EXPORT_MOUSE, MouseState, ScriptContext
from shaderbox.scripting.engine import (
    ScriptEngine,
    ScriptPass,
    ScriptProbe,
    ScriptStatus,
    ScriptTarget,
    is_scriptable,
    normalize_script_tabs,
    script_stub_for,
)
from shaderbox.scripting.errors import ScriptError
from shaderbox.scripting.keys import StoppedKey

__all__ = [
    "EXPORT_MOUSE",
    "Behavior",
    "MouseState",
    "PythonBehavior",
    "ScriptBehavior",
    "ScriptContext",
    "ScriptContext",
    "ScriptEngine",
    "ScriptError",
    "ScriptPass",
    "ScriptProbe",
    "ScriptStatus",
    "ScriptTarget",
    "StoppedKey",
    "coerce_one",
    "is_scriptable",
    "normalize_script_tabs",
    "script_stub_for",
]
