"""The script-language seam (feature 041). A script file is a user-finalized CLASS subclassing
`ScriptBehavior`, with `update(self, ctx) -> <typed output>`; per-instance state (`self.*`)
persists across frames — the reason CPU scripting exists (stateless work belongs in the shader).
The engine iterates `Behavior` objects and never knows the language; a future C backend implements
the same protocol over a `.so`.

`PythonBehavior` compiles the file VERBATIM (no rewrite — an error's lineno points at the user's
real source, the 039 ghost stays dead), resolves the `ScriptBehavior` subclass, instantiates it
ONCE, and calls `.update(ctx)` each tick. The exec namespace seeds the curated math vocab, the base,
the `Ctx` alias, and ALL output types (method annotations evaluate eagerly — every type name a stub
can annotate must be resolvable at class-def time)."""

import inspect
import math
import traceback
from typing import Any, Protocol, TypeGuard

import moderngl

from shaderbox.scripting.context import EngineContext
from shaderbox.scripting.errors import ScriptError
from shaderbox.scripting.outputs import Array, Text, Vec2, Vec3, Vec4, normalize_output
from shaderbox.uniform_coerce import (
    coerce_uniform_value,
    gl_type_label,
    uniform_shape_hint,
)


def _user_error_line(marker_name: str, exc: BaseException) -> int:
    # Recover the deepest line in the USER's source from a traceback — the script is compiled with
    # filename "<u:name>" (per-uniform: the uniform name; node-brain: "script.py"), so a frame from
    # that file is a user line (vs an engine frame). -1 when the error didn't reach the user's code
    # (unmappable). `marker_name` is UNWRAPPED — this builds the `<u:...>` itself.
    marker = f"<u:{marker_name}>"
    line = -1
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == marker and frame.lineno is not None:
            line = frame.lineno
    return line


class ScriptBehavior:
    """The base every script subclasses. `update` returns the uniform's value this frame
    (a bare number for a scalar; `Vec2/3/4`/`Array`/`Text` for the shaped kinds). State goes
    on `self` (set in `__init__`, mutated in `update`) and persists across frames."""

    def update(self, ctx: EngineContext) -> Any:
        raise NotImplementedError


def _no_import(*_args: Any, **_kwargs: Any) -> Any:
    # The curated exec namespace omits __import__, so a bare `import math` would raise the cryptic
    # "__import__ not found". Seed this instead so the message tells the author what to do.
    raise ImportError(
        "imports are not available in a script — the math vocab "
        "(sin/cos/sqrt/clamp/lerp/.../pi/tau) and math.* are pre-loaded; just use them"
    )


def _build_globals(uniform_name: str) -> dict[str, Any]:
    # The names a script body + its eager method annotations resolve against. `__builtins__`
    # carries `__build_class__` (the `class` statement needs it) — NOT emptied (040's trick fails
    # for the class model); the curated math vocab + the base + Ctx + ALL output types are
    # top-level globals (annotations like `-> Vec2` evaluate at class-def time against this dict).
    # No sandbox (a personal IDE; locked posture) — we expose a curated set, not full builtins.
    builtins_ns: dict[str, Any] = {
        "__build_class__": __build_class__,
        "__import__": _no_import,
        "__name__": f"<u:{uniform_name}>",
        # The scalar return annotation `-> float` / `-> int` resolves these at class-def time
        # (annotations eval eagerly — no `from __future__ import annotations`), so the builtin
        # number types must be in scope alongside the math vocab.
        "float": float,
        "int": int,
        "bool": bool,
        # Common exception types so a user `raise ValueError(...)` is its real error, not a
        # misleading NameError from the curated (no-full-builtins) namespace.
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "RuntimeError": RuntimeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "ZeroDivisionError": ZeroDivisionError,
        # `super` makes the engine's own subclass-init idiom (`super().__init__()`) work; the
        # containers + iteration helpers are what a real stateful script reaches for (a history
        # buffer, a smoothing window). No safety lost — the raw exec already runs the file.
        "super": super,
        "object": object,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "frozenset": frozenset,
        "str": str,
        "bytes": bytes,
        "chr": chr,
        "ord": ord,
        "sum": sum,
        "sorted": sorted,
        "reversed": reversed,
        "zip": zip,
        "map": map,
        "filter": filter,
        "isinstance": isinstance,
        "all": all,
        "any": any,
        "print": print,
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
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "clamp": lambda x, lo, hi: max(lo, min(hi, x)),
        "lerp": lambda a, b, k: a + (b - a) * k,
        "mix": lambda a, b, k: a + (b - a) * k,  # GLSL spelling of lerp (muscle memory)
        "math": math,
    }
    return {
        "__builtins__": builtins_ns,
        "__name__": f"<u:{uniform_name}>",
        "ScriptBehavior": ScriptBehavior,
        "Ctx": EngineContext,
        "Vec2": Vec2,
        "Vec3": Vec3,
        "Vec4": Vec4,
        "Array": Array,
        "Text": Text,
    }


def _check_update_arity(cls: type[ScriptBehavior]) -> str | None:
    # `update` must accept (self, ctx). A `def update(ctx)` (forgot self) compiles fine but throws a
    # cryptic per-tick TypeError; catch it at compile by binding two placeholder args. Validates ARITY
    # only (renamed params / *args / extra-defaulted params are legitimate), not parameter names.
    try:
        inspect.signature(cls.update).bind(object(), object())
    except TypeError:
        return "update must be `def update(self, ctx)` — it takes the instance + the context"
    return None


def _is_behavior_subclass(value: object) -> TypeGuard[type[ScriptBehavior]]:
    return (
        isinstance(value, type)
        and issubclass(value, ScriptBehavior)
        and value is not ScriptBehavior
    )


def _resolve_behavior_class(ns: dict[str, Any]) -> type[ScriptBehavior] | None:
    # Prefer a class literally named `Behavior` (the stub's name); else the first ScriptBehavior
    # subclass defined in the namespace — excluding the base itself (it lives in the globals).
    candidate = ns.get("Behavior")
    if _is_behavior_subclass(candidate):
        return candidate
    for value in ns.values():
        if _is_behavior_subclass(value):
            return value
    return None


class Behavior(Protocol):
    def run(self, ctx: EngineContext) -> Any:
        # The raw value(s) this behavior produces this frame — cardinality-agnostic. A per-uniform
        # behavior's `run` returns a single typed value (the engine wraps it as one (name, value)
        # pair); a node-brain's `run` returns a `dict[str, value]` driving many uniforms. Coercion
        # against the live uniform is the ENGINE's job (`coerce_one`), so a future C backend produces
        # raw values without re-implementing the shape coercion.
        ...

    @property
    def error(self) -> ScriptError | None: ...


class PythonBehavior:
    """A script file compiled + exec'd VERBATIM once: the engine resolves the user's
    `ScriptBehavior` subclass, instantiates it (holding the live state instance), and calls
    `.update(ctx)` each tick. Compile-time failures (SyntaxError / no subclass / no `update`
    override / a raising `__init__`) cache a `ScriptError` and freeze permanently until the
    file changes; runtime + shape failures are caught per-tick by the engine."""

    def __init__(self, label: str, body: str) -> None:
        # `label` is the binding KEY: a uniform name for per-uniform, the sentinel "script.py" for a
        # node-brain. It is the compile-marker name AND the name a compile error records under.
        self.label = label
        self._error: ScriptError | None = None
        self._instance: ScriptBehavior | None = None
        self._cls: type[ScriptBehavior] | None = None
        try:
            code = compile(body, f"<u:{label}>", "exec")
        except SyntaxError as e:
            self._error = ScriptError(
                label, "compile", e.msg or "syntax error", e.lineno or -1
            )
            return

        ns = _build_globals(label)
        try:
            exec(code, ns)  # raw exec of the user file — no sandbox (locked posture)
        except Exception as e:
            self._error = ScriptError(
                label,
                "compile",
                f"{type(e).__name__}: {e}",
                _user_error_line(label, e),
            )
            return

        cls = _resolve_behavior_class(ns)
        if cls is None:
            self._error = ScriptError(
                label,
                "compile",
                "no ScriptBehavior subclass found — keep the "
                "`class Behavior(ScriptBehavior)` line",
            )
            return
        if cls.update is ScriptBehavior.update:
            self._error = ScriptError(
                label,
                "compile",
                f"class {cls.__name__} does not implement update(self, ctx)",
            )
            return
        arity_error = _check_update_arity(cls)
        if arity_error is not None:
            self._error = ScriptError(
                label,
                "compile",
                arity_error,
                cls.update.__code__.co_firstlineno,
            )
            return
        self._cls = cls
        self._instantiate()

    def _instantiate(self) -> None:
        # Construct the state instance; a raising __init__ is a compile-level freeze. A successful
        # construct CLEARS any prior error so a reset() that recovers a once-failing __init__ unfreezes
        # the binding (else the stale error keeps the engine freezing it forever).
        if self._cls is None:
            return
        try:
            self._instance = self._cls()
            self._error = None
        except Exception as e:
            self._error = ScriptError(
                self.label,
                "compile",
                f"__init__ raised: {type(e).__name__}: {e}",
                _user_error_line(self.label, e),
            )
            self._instance = None

    @property
    def error(self) -> ScriptError | None:
        return self._error

    def reset(self) -> None:
        # Re-run __init__ on a fresh instance (manual reset / restart) without recompiling.
        # _instantiate no-ops when there's no resolved class (an unrecoverable compile failure).
        self._instantiate()

    def run(self, ctx: EngineContext) -> Any:
        # The raw value(s) the user's update produced this frame — cardinality-agnostic, NOT yet
        # coerced (the engine fans this out into (name, value) pairs and coerces each via coerce_one):
        # a per-uniform script returns a single typed value, a node-brain a dict[str, value].
        if self._instance is None:
            raise _RuntimeScriptError(
                ScriptError(self.label, "runtime", "no behavior instance")
            )
        return self._instance.update(ctx)


def _all_finite(coerced: object) -> bool:
    # coerce_uniform_value yields a number, a tuple/list of numbers, or a list of dim-tuples.
    if isinstance(coerced, int | float):
        return math.isfinite(coerced)
    if isinstance(coerced, list | tuple):
        return all(_all_finite(v) for v in coerced)
    return True


def coerce_one(value: object, uniform: moderngl.Uniform, error_name: str) -> object:
    # Normalize a raw script value + shape it against the live uniform via the shared coercion. The
    # one coercion atom for BOTH paths (per-uniform's single value, a node-brain's per-key value).
    # `error_name` is the uniform NAME a shape mismatch records under (the GLSL-type label for the
    # hint is derived internally). Raises _RuntimeScriptError on a mismatch; the engine freezes.
    normalized = normalize_output(value)
    coerced = coerce_uniform_value(normalized, uniform)
    if coerced is None:
        raise _RuntimeScriptError(
            ScriptError(
                error_name,
                "runtime",
                uniform_shape_hint(uniform, gl_type_label(uniform), normalized),
            )
        )
    # NaN/Inf are valid floats to coerce_uniform_value but corrupt the render silently (a black
    # frame, no error) and would poison last-good. Fold them into the normal frozen-uniform path.
    if not _all_finite(coerced):
        raise _RuntimeScriptError(
            ScriptError(
                error_name,
                "runtime",
                "value is not finite (NaN/Inf) — check for divide-by-zero or an integrator blow-up",
            )
        )
    return coerced


class _RuntimeScriptError(Exception):
    # Carries a ready ScriptError out of coerce_one() so the engine records it verbatim
    # (a shape mismatch's authored message), distinct from a raw exception in the user body.
    def __init__(self, error: ScriptError) -> None:
        super().__init__(error.message)
        self.error = error
