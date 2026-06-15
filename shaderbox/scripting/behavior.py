"""The script-language seam (feature 041). A script file is a user-finalized CLASS subclassing
`ScriptBehavior`, with `update(self, ctx) -> <typed output>`; per-instance state (`self.*`)
persists across frames — the reason CPU scripting exists (stateless work belongs in the shader).
The engine iterates `Behavior` objects and never knows the language; a future C backend implements
the same protocol over a `.so`.

`PythonBehavior` compiles the file VERBATIM (no rewrite — an error's lineno points at the user's
real source, the 039 ghost stays dead), resolves the `ScriptBehavior` subclass, instantiates it
ONCE, and calls `.update(ctx)` each tick. The exec namespace is the real builtins (a script is plain
Python — `import math` and the stdlib work) plus the base, the `Ctx` alias, and ALL output types
(method annotations evaluate eagerly — every type name a stub can annotate must be resolvable at
class-def time)."""

import inspect
import math
import traceback
from typing import Any, Protocol, TypeGuard

import moderngl

from shaderbox.scripting.context import EngineContext, MouseState
from shaderbox.scripting.errors import ScriptError
from shaderbox.scripting.outputs import Array, Text, Vec2, Vec3, Vec4, normalize_output
from shaderbox.uniform_coerce import (
    coerce_uniform_value,
    gl_type_label,
    uniform_shape_hint,
)


def _user_error_line(marker_name: str, exc: BaseException) -> int:
    # Recover the deepest line in the USER's source from a traceback — the script is compiled with
    # filename "<u:script.py>", so a frame from that file is a user line (vs an engine frame). -1 when
    # the error didn't reach the user's code (unmappable). `marker_name` is UNWRAPPED — this builds the
    # `<u:...>` itself.
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


# The scripting types the engine INJECTS into every script's globals (the fallback for a missing
# import). Used to steer a wrong-import compile error toward the canonical line.
_INJECTED_NAMES = frozenset(
    {"ScriptBehavior", "Ctx", "MouseState", "Vec2", "Vec3", "Vec4", "Array", "Text"}
)


def _import_hint(exc: Exception) -> str:
    # When a script's import/name error names a type the engine ALREADY injects, append the canonical
    # import (so a wrong `from shaderbox import ScriptBehavior` self-corrects instead of sending the
    # agent grepping — the error message names the wrong module + never the right one). "" otherwise.
    if not isinstance(exc, ImportError | NameError):
        return ""
    # `ImportError.name` for `from shaderbox import ScriptBehavior` is the MODULE ("shaderbox"), not the
    # symbol — so match against the message text too, where the bad symbol always appears.
    bad = getattr(exc, "name", None)
    if bad in _INJECTED_NAMES or any(n in str(exc) for n in _INJECTED_NAMES):
        return (
            " -- scripting types come from `shaderbox.scripting` "
            "(`from shaderbox.scripting import ScriptBehavior, Ctx, Vec3, ...`), "
            "or omit the import: the engine injects them."
        )
    return ""


def _build_globals(uniform_name: str) -> dict[str, Any]:
    # The names a script body + its eager method annotations resolve against. A script is a plain
    # Python file: the real builtins are in scope (so `import math`, the whole stdlib, AND a real
    # `from shaderbox.scripting import Vec2, …` all work — __builtins__ carries __import__). The 048
    # stub emits that explicit import so the available types are VISIBLE; these injected names are the
    # FALLBACK so a user who deletes the import line still resolves `Vec2`/`Ctx`/… instead of an
    # opaque eager-annotation-eval compile-freeze. No sandbox (a personal IDE; locked posture).
    return {
        "__builtins__": __builtins__,
        "__name__": f"<u:{uniform_name}>",
        "ScriptBehavior": ScriptBehavior,
        "Ctx": EngineContext,
        "MouseState": MouseState,
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
        # The raw `dict[str, value]` this behavior produces this frame (the node script's `update`
        # return), NOT yet coerced. Coercion against each live uniform is the ENGINE's job
        # (`coerce_one`), so a future C backend produces raw values without re-implementing the shape
        # coercion.
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
        # `label` is the binding KEY (the node script's "script.py"): the compile-marker name AND the
        # name a compile error records under.
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
                f"{type(e).__name__}: {e}{_import_hint(e)}",
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
        # The raw dict the user's update produced this frame (name -> value), NOT yet coerced — the
        # engine fans it into (name, value) pairs and coerces each against the live uniform via
        # coerce_one. A future non-Python backend implements this same protocol over a .so.
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
    # one coercion atom, called per key of the node script's returned dict.
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
