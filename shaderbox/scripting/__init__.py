"""The CPU-script engine (feature 040) — a uniform becomes a first-class object with a
per-tick behavior script. Public surface the headless ProjectSession drives."""

from shaderbox.scripting.behavior import Behavior, PythonBehavior, UniformOut
from shaderbox.scripting.context import EngineContext
from shaderbox.scripting.engine import EngineNode, ScriptEngine
from shaderbox.scripting.errors import ScriptError

__all__ = [
    "Behavior",
    "EngineContext",
    "EngineNode",
    "PythonBehavior",
    "ScriptEngine",
    "ScriptError",
    "UniformOut",
]
