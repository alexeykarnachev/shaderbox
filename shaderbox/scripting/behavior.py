"""The script-language seam (feature 040). A `Behavior` computes one uniform's value
into a `UniformOut`; the engine loop iterates `Behavior` objects and never knows the
language. v1 ships `PythonBehavior` (the body compiled once, exec'd verbatim each tick);
a future C backend implements the same protocol over a `.so`.
"""

import math
from typing import Any, Protocol

import moderngl

from shaderbox.scripting.context import EngineContext
from shaderbox.scripting.errors import ScriptError
from shaderbox.uniform_coerce import coerce_uniform_value, uniform_shape_hint

UniformValue = float | int | list[int] | tuple[float, ...] | list[tuple[float, ...]]


def build_namespace() -> dict[str, Any]:
    # The vocabulary a Python body sees. `__builtins__` is emptied so a bare name resolves
    # against THIS dict (explicit + fast), NOT for safety — no-sandbox is a locked posture.
    # math-only, no numpy (a vec result is a plain tuple — decision/Out-of-scope).
    return {
        "__builtins__": {},
        "math": math,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "sqrt": math.sqrt,
        "exp": math.exp,
        "log": math.log,
        "floor": math.floor,
        "ceil": math.ceil,
        "fmod": math.fmod,
        "hypot": math.hypot,
        "pi": math.pi,
        "tau": math.tau,
        "abs": abs,
        "min": min,
        "max": max,
        "round": round,
        "pow": pow,
        "clamp": lambda x, lo, hi: max(lo, min(hi, x)),
        "lerp": lambda a, b, k: a + (b - a) * k,
    }


class UniformOut:
    """The value sink injected into a script as `out`. Holds the live `moderngl.Uniform`
    so `set()` validates the script's result against the real `(dim, array_length, gl_type)`
    via the shared coercion. The body writes via `out.set(...)`; nothing is left hanging and
    the engine never rewrites the body."""

    def __init__(self, uniform: moderngl.Uniform) -> None:
        self._uniform = uniform
        self.was_set: bool = False
        self.value: UniformValue | None = None
        self.error: ScriptError | None = None

    def set(self, *args: Any) -> None:
        # Accept set(0.5) / set(x, y, z) / set((x, y, z)). Normalize *args to a single value,
        # then shape it against the uniform via the shared coercion. A mismatch records a
        # runtime ScriptError (the caller freezes last-good); last-write-wins on repeat calls.
        value: Any = args[0] if len(args) == 1 else list(args)
        coerced = coerce_uniform_value(value, self._uniform)
        if coerced is None:
            self.error = ScriptError(
                self._uniform.name,
                "runtime",
                uniform_shape_hint(self._uniform.name, self._uniform, self._uniform.name),
            )
            return
        self.value = coerced
        self.was_set = True


class Behavior(Protocol):
    def compute(self, ctx: EngineContext, out: UniformOut) -> None: ...

    @property
    def error(self) -> ScriptError | None: ...


class PythonBehavior:
    """A uniform body compiled once into a code object, exec'd VERBATIM each tick. The user
    wrote ONLY the body and hands the result back via `out.set(...)`; the engine never rewrites
    it (no last-expression magic), so a SyntaxError/runtime error's lineno already points at
    the user's source — no AST surgery, no line-remap (the 039 ghost removed by construction)."""

    def __init__(self, uniform_name: str, body: str) -> None:
        self._uniform_name = uniform_name
        self._error: ScriptError | None = None
        self._code: Any = None
        self._ns: dict[str, Any] = build_namespace()
        try:
            self._code = compile(body, f"<u:{uniform_name}>", "exec")
        except SyntaxError as e:
            self._error = ScriptError(
                uniform_name, "compile", e.msg or "syntax error", e.lineno or -1
            )

    @property
    def error(self) -> ScriptError | None:
        return self._error

    def compute(self, ctx: EngineContext, out: UniformOut) -> None:
        if self._code is None:
            return
        # Raw exec of the user body — no sandbox (a personal IDE; locked posture, decision 12).
        exec(self._code, self._ns, {"ctx": ctx, "out": out})
