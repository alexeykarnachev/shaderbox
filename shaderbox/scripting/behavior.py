"""The script-language seam (feature 041). A script file is a user-finalized CLASS subclassing
`ScriptBehavior`, with `update(self, context) -> dict`; per-instance state (`self.*`)
persists across frames — the reason CPU scripting exists (stateless work belongs in the shader).
The engine iterates `Behavior` objects and never knows the language; a future C backend implements
the same protocol over a `.so`.

`PythonBehavior` compiles the file VERBATIM (no rewrite — an error's lineno points at the user's
real source, the 039 ghost stays dead), resolves the `ScriptBehavior` subclass, instantiates it
ONCE, and calls `.update(context)` each tick. The exec namespace is `ScriptBehavior`, the `ScriptContext` alias
and `MouseState`, over builtins whose `__import__` is narrowed: a script is plain Python and the
stdlib works, but of THIS package it sees `shaderbox.scripting` and, in it, those three names
alone. The app's own modules are not a script's to reach.

A script returns PLAIN PYTHON — a number, a list or tuple for a vector, a list (flat or rows) for
an array, a str for a text uniform. The engine shapes it against the live `moderngl.Uniform`.
There are no wrapper types to learn; a script wanting vector math imports numpy like any other
Python program."""

import inspect
import math
import traceback
from typing import Any, Protocol, TypeGuard

import moderngl

from shaderbox.scripting.context import MouseState, ScriptContext
from shaderbox.scripting.errors import ScriptError
from shaderbox.uniform_coerce import (
    coerce_uniform_value,
    gl_type_label,
    uniform_shape_hint,
)

_REAL_IMPORT = __import__


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
    """The base class a document script's Behavior extends.

    Define `__init__(self)` for state that survives across frames, and
    `update(self, context) -> dict` for the values each frame drives. State goes on `self` and
    persists; the values are plain Python, shaped against the live uniform.
    """

    def update(self, context: ScriptContext) -> Any:
        raise NotImplementedError


# THE user-facing scripting surface: the only names a script may import, and the same set the
# engine injects into its globals. Everything else in `shaderbox.scripting` — the engine, the
# probe, the stub generator — is the app's own machinery that happens to share the package.
_INJECTED_NAMES = frozenset({"ScriptBehavior", "ScriptContext", "MouseState"})


def _import_hint(exc: Exception) -> str:
    # A script that NAMES a scripting type it never imported (`NameError: ScriptContext`) gets told the
    # import line for THAT name. Only a NameError: the import gate's own messages already say
    # what is allowed, and appending to one produced advice that recreated the error.
    #
    # TODO: this is close to dead. `_build_globals` injects all three names, so the case it
    # describes compiles clean and the hint fires only if a script `del`s one; a misspelling is
    # by definition not in `_INJECTED_NAMES`. Delete it, or find the trigger it is actually for.
    if not isinstance(exc, NameError):
        return ""
    bad = getattr(exc, "name", None)
    if bad not in _INJECTED_NAMES:
        return ""
    return f" -- import it: `from {_SCRIPT_PACKAGE} import {bad}`"


# The ONE package path a script may import from. `shaderbox.scripting` is the user-facing surface;
# every other module is the app's own machinery and a script importing it is reaching past the
# interface into the implementation — `shaderbox.app`, `shaderbox.core`, `ProjectSession` were all
# reachable before this gate.
_SCRIPT_PACKAGE = "shaderbox.scripting"


def _script_import(
    name: str,
    globals_: "dict[str, Any] | None" = None,
    locals_: "dict[str, Any] | None" = None,
    fromlist: "tuple[str, ...] | None" = (),
    level: int = 0,
) -> Any:
    # The script's `__import__`. The stdlib and third-party packages import normally; of THIS
    # package a script sees `shaderbox.scripting` and, in it, the user-facing types alone —
    # the engine and the probe live there too and are none of a script's business. Both messages
    # name what IS allowed, so the fix is one edit rather than a search.
    root = name.split(".")[0]
    if root != "shaderbox":
        return _REAL_IMPORT(name, globals_, locals_, fromlist, level)
    if name != _SCRIPT_PACKAGE:
        raise ImportError(
            f"a script may import from `{_SCRIPT_PACKAGE}` only, not `{name}`"
        )
    allowed = ", ".join(sorted(_INJECTED_NAMES))
    # `import shaderbox.scripting` (no fromlist) would bind the whole module, engine included,
    # so only the `from … import <names>` form resolves.
    if not fromlist:
        raise ImportError(
            f"import the names, not the module: `from {_SCRIPT_PACKAGE} import {allowed}`"
        )
    hidden = sorted(set(fromlist) - _INJECTED_NAMES)
    if hidden:
        raise ImportError(
            f"`{_SCRIPT_PACKAGE}` offers a script {allowed} — not {', '.join(hidden)}"
        )
    return _REAL_IMPORT(name, globals_, locals_, fromlist, level)


def _script_builtins() -> dict[str, Any]:
    # The real builtins with `__import__` replaced, so the whole stdlib still works and only the
    # package path is narrowed. Built fresh per script: a shared dict one script mutated would
    # follow every other script in the process.
    builtins = dict(
        __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    )
    builtins["__import__"] = _script_import
    return builtins


def _build_globals(uniform_name: str) -> dict[str, Any]:
    # The names a script body + its eager method annotations resolve against. A script is plain
    # Python — the stdlib is in scope — but of THIS package it sees `shaderbox.scripting` alone
    # (`_script_import`). The 048 stub emits the explicit import so the available types are
    # VISIBLE; these injected names are the FALLBACK so a user who deletes the import line still
    # resolves `Vec2`/`ScriptContext`/… instead of an opaque eager-annotation-eval compile-freeze.
    return {
        "__builtins__": _script_builtins(),
        "__name__": f"<u:{uniform_name}>",
        "ScriptBehavior": ScriptBehavior,
        "ScriptContext": ScriptContext,
        "MouseState": MouseState,
    }


def _check_update_arity(cls: type[ScriptBehavior]) -> str | None:
    # `update` must accept (self, context). A `def update(context)` (forgot self) compiles fine but throws a
    # cryptic per-tick TypeError; catch it at compile by binding two placeholder args. Validates ARITY
    # only (renamed params / *args / extra-defaulted params are legitimate), not parameter names.
    try:
        inspect.signature(cls.update).bind(object(), object())
    except TypeError:
        return "update must be `def update(self, context)` — it takes the instance + the context"
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
    def run(self, context: ScriptContext) -> Any:
        # The raw `dict[str, value]` this behavior produces this frame (the document script's `update`
        # return), NOT yet coerced. Coercion against each live uniform is the ENGINE's job
        # (`coerce_one`), so a future C backend produces raw values without re-implementing the shape
        # coercion.
        ...

    @property
    def error(self) -> ScriptError | None: ...


class PythonBehavior:
    """A script file compiled + exec'd VERBATIM once: the engine resolves the user's
    `ScriptBehavior` subclass, instantiates it (holding the live state instance), and calls
    `.update(context)` each tick. Compile-time failures (SyntaxError / no subclass / no `update`
    override / a raising `__init__`) cache a `ScriptError` and freeze permanently until the
    file changes; runtime + shape failures are caught per-tick by the engine."""

    def __init__(self, label: str, body: str) -> None:
        # `label` is the binding KEY (the document script's "script.py"): the compile-marker name AND the
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
                f"class {cls.__name__} does not implement update(self, context)",
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

    def run(self, context: ScriptContext) -> Any:
        # The raw dict the user's update produced this frame (name -> value), NOT yet coerced — the
        # engine fans it into (name, value) pairs and coerces each against the live uniform via
        # coerce_one. A future non-Python backend implements this same protocol over a .so.
        if self._instance is None:
            raise _RuntimeScriptError(
                ScriptError(self.label, "runtime", "no behavior instance")
            )
        return self._instance.update(context)


def _all_finite(coerced: object) -> bool:
    # coerce_uniform_value yields a number, a tuple/list of numbers, or a list of dim-tuples.
    if isinstance(coerced, int | float):
        return math.isfinite(coerced)
    if isinstance(coerced, list | tuple):
        return all(_all_finite(v) for v in coerced)
    return True


def coerce_one(value: object, uniform: moderngl.Uniform, error_name: str) -> object:
    # Normalize a raw script value + shape it against the live uniform via the shared coercion. The
    # one coercion atom, called per key of the document script's returned dict.
    # `error_name` is the uniform NAME a shape mismatch records under (the GLSL-type label for the
    # hint is derived internally). Raises _RuntimeScriptError on a mismatch; the engine freezes.
    # A dict is never a uniform value — that invariant is what makes the engine's value-type
    # dispatch between a pass block and a broadcast key unambiguous (069 D3), so it is checked
    # here rather than left to the coercion.
    if isinstance(value, dict):
        raise _RuntimeScriptError(
            ScriptError(
                error_name,
                "runtime",
                "a dict is a PASS BLOCK, not a uniform value — "
                "{'pass': {'u_name': value}} addresses a pass; a bare key drives every pass "
                "declaring it",
            )
        )
    coerced = coerce_uniform_value(value, uniform)
    if coerced is None:
        raise _RuntimeScriptError(
            ScriptError(
                error_name,
                "runtime",
                uniform_shape_hint(uniform, gl_type_label(uniform), value),
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
