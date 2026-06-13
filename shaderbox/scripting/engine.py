"""The CPU-script engine (feature 041) — repo code owned by the headless ProjectSession.

Per node it resolves `scripts/u_<name>.py` against the node's active uniforms, compiles a
`Behavior` per binding (cached by `(path, mtime)`, each holding its own state instance), and on
each `tick` calls the behavior's `update` and writes the value into `node.uniform_values` BEFORE
`Node.render()` reads them. A broken script never raises into the frame loop: the uniform freezes
at last-good and a `ScriptError` is recorded.

The engine imports no imgui/glfw/App and no concrete `Node` type — it works against the
`EngineNode` protocol (the `uniform_values` dict + `get_active_uniforms()`), so it stays in the
025 headless core.
"""

from pathlib import Path
from typing import Any, Protocol, TypeGuard

import moderngl
from loguru import logger
from OpenGL.GL import GL_INT, GL_SAMPLER_2D, GL_UNSIGNED_INT

from shaderbox.scripting.behavior import (
    PythonBehavior,
    _RuntimeScriptError,
    _user_error_line,
)
from shaderbox.scripting.context import EngineContext
from shaderbox.scripting.errors import ScriptError
from shaderbox.uniform_coerce import is_text_array

_SCRIPT_GLOB = "u_*.py"


def is_scriptable(uniform: object) -> TypeGuard[moderngl.Uniform]:
    # A scalar/vector/array uniform the engine can drive — NOT a UniformBlock (no scalar value)
    # and not a sampler (a texture, not a script value). Duck-typed on the shape attrs the
    # coercion reads, so a test stand-in passes alongside a real moderngl.Uniform. (042's
    # Add-script gate + the stub generator both call this.)
    return (
        not isinstance(uniform, moderngl.UniformBlock)
        and getattr(uniform, "gl_type", None) != GL_SAMPLER_2D
        and hasattr(uniform, "dimension")
        and hasattr(uniform, "array_length")
    )


class EngineNode(Protocol):
    # The slice of Node the engine touches — nothing GL-program-specific.
    uniform_values: dict[str, Any]

    def get_active_uniforms(
        self,
    ) -> list[moderngl.Uniform | moderngl.UniformBlock]: ...


class NodeScripts:
    # One node's compiled bindings + their (path, mtime) cache + the raw source per binding
    # (a fresh export behavior recompiles from this cached source, never a fresh disk read) +
    # per-uniform last-good.
    def __init__(self, scripts_dir: Path) -> None:
        self.scripts_dir = scripts_dir
        self.behaviors: dict[str, PythonBehavior] = {}
        self.mtimes: dict[str, float] = {}
        self.sources: dict[str, str] = {}
        self.last_good: dict[str, Any] = {}
        # Orphan/unsupported-binding names already warned, so reload (per-frame) logs once on the
        # transition, not every frame.
        self.warned: set[str] = set()


def stub_for(uniform: moderngl.Uniform) -> str:
    # The ready-to-edit script a freshly-attached uniform gets: the right return type for the
    # uniform's shape + a docstring of the ctx fields + a body returning a coercion-valid default,
    # so a new script compiles + runs immediately (showing the default, not an error). Used by
    # 042's "Add script".
    name = uniform.name
    dim = uniform.dimension
    n = uniform.array_length
    gl_type = getattr(uniform, "gl_type", None)

    if is_text_array(uniform):
        ann, default = "Text", 'Text("")'
    elif n > 1:
        ann, default = "Array", f"Array([0.0] * {n * dim})"
    elif dim == 2:
        ann, default = "Vec2", "Vec2(0.0, 0.0)"
    elif dim == 3:
        ann, default = "Vec3", "Vec3(0.0, 0.0, 0.0)"
    elif dim == 4:
        ann, default = "Vec4", "Vec4(0.0, 0.0, 0.0, 0.0)"
    elif gl_type in (GL_INT, GL_UNSIGNED_INT):
        ann, default = "int", "0"  # a scalar int/uint stub returns an int, not 0.0
    else:
        ann, default = "float", "0.0"

    return (
        f"class Behavior(ScriptBehavior):\n"
        f'    """Drive {name} each frame.\n'
        f"    ctx.t  elapsed seconds | ctx.dt  delta seconds | ctx.frame  frame index\n"
        f"    Return {ann}. Keep state on self (persists across frames).\n"
        f"    Math is pre-loaded — call sin/cos/sqrt/clamp/lerp(=mix)/... or math.* "
        f"directly (no import).\n"
        f'    """\n'
        f"    def __init__(self) -> None:\n"
        f"        pass\n\n"
        f"    def update(self, ctx: Ctx) -> {ann}:\n"
        f"        return {default}\n"
    )


class ScriptEngine:
    def __init__(self, engine_driven: frozenset[str] = frozenset()) -> None:
        self._nodes: dict[str, NodeScripts] = {}
        # (node_id, uniform_name) -> the most recent error, for 042's UI to surface.
        self.errors: dict[tuple[str, str], ScriptError] = {}
        # Engine-owned uniform names (u_time/u_aspect/u_resolution) — render() hardcodes these, so a
        # script on one would silently no-op. Passed in by ProjectSession (NOT imported from core,
        # which pulls in glfw — the headless boundary). Empty in a bare test engine.
        self._engine_driven = engine_driven

    def script_driven_uniforms(self, node_id: str) -> set[str]:
        node = self._nodes.get(node_id)
        return set(node.behaviors) if node else set()

    def reload(self, node_id: str, scripts_dir: Path, node: EngineNode) -> None:
        # Glob the node's scripts dir, (re)compile only files whose mtime changed (a recompile
        # makes a FRESH instance — state resets on edit), drop removed files, and resolve each
        # binding against the node's active uniforms — warning ONCE (on the transition) on an orphan
        # (no such uniform) or an engine-owned/unscriptable target. Cheap when nothing changed: a
        # stat per file, no recompile.
        scripts = self._nodes.get(node_id)
        if scripts is None or scripts.scripts_dir != scripts_dir:
            scripts = NodeScripts(scripts_dir)
            self._nodes[node_id] = scripts

        active = {u.name: u for u in node.get_active_uniforms()}
        # An empty active set means the program is mid-invalidation (a lib edit dropped it); skip the
        # orphan check this poll so a live binding isn't briefly mis-flagged as a false orphan.
        check_active = bool(active)
        found: set[str] = set()
        if scripts_dir.is_dir():
            for path in sorted(scripts_dir.glob(_SCRIPT_GLOB)):
                name = path.stem
                # Resolve against the live uniforms FIRST — only a real, scriptable binding counts as
                # "found" (so a uniform that goes inactive is reclaimed by the drop loop below).
                if check_active:
                    reason = self._binding_reject(name, active)
                    if reason is not None:
                        if name not in scripts.warned:
                            logger.warning(
                                f"Script {path.name} on node {node_id}: {reason} — ignored"
                            )
                            scripts.warned.add(name)
                        continue
                found.add(name)
                scripts.warned.discard(name)
                try:
                    mtime = path.stat().st_mtime
                    body = path.read_text(encoding="utf-8")
                except (OSError, ValueError):
                    # A vanished / half-saved / non-UTF8 file mid-edit — keep the cached behavior,
                    # never raise into the frame loop (ValueError covers UnicodeDecodeError).
                    continue
                if scripts.mtimes.get(name) == mtime:
                    continue
                behavior = PythonBehavior(name, body)
                scripts.behaviors[name] = behavior
                scripts.mtimes[name] = mtime
                scripts.sources[name] = body
                key = (node_id, name)
                if behavior.error is not None:
                    self.errors[key] = behavior.error
                else:
                    self.errors.pop(key, None)

        # Drop bindings whose file disappeared OR whose uniform went inactive/unscriptable.
        for name in list(scripts.behaviors):
            if name not in found:
                scripts.behaviors.pop(name, None)
                scripts.mtimes.pop(name, None)
                scripts.sources.pop(name, None)
                scripts.last_good.pop(name, None)
                self.errors.pop((node_id, name), None)

    def _binding_reject(
        self, name: str, active: dict[str, moderngl.Uniform | moderngl.UniformBlock]
    ) -> str | None:
        # Why a script can't bind to `name` (None = it can). Drives the warn-once message.
        if name in self._engine_driven:
            return f"'{name}' is engine-owned (its value is set by the renderer)"
        uniform = active.get(name)
        if uniform is None:
            return f"no active uniform '{name}' (orphan script)"
        if not is_scriptable(uniform):
            return f"'{name}' is a sampler/block — not a scriptable value"
        return None

    def reset(self, node_id: str, name: str) -> None:
        # Re-instantiate one live binding (re-run __init__) without recompiling — the manual
        # "restart" the 042 UI button wires. Sync the engine's recorded error to the behavior's
        # post-reset state (a recovered __init__ clears it; a still-raising one re-records) so a
        # consumer reading `errors` off-tick sees the truth immediately, not next frame.
        scripts = self._nodes.get(node_id)
        if scripts is None:
            return
        behavior = scripts.behaviors.get(name)
        if behavior is None:
            return
        behavior.reset()
        if behavior.error is not None:
            self.errors[(node_id, name)] = behavior.error
        else:
            self.errors.pop((node_id, name), None)

    def fresh_behaviors_for(self, node_id: str) -> dict[str, PythonBehavior]:
        # A NEW behavior set for the node, independent of the live registry's instances —
        # recompiled from the live registry's CACHED source (not a fresh disk read, so an
        # export never sees a half-saved mid-edit file). The export path ticks THIS set so an
        # exported integrator starts from a clean __init__ regardless of live state.
        scripts = self._nodes.get(node_id)
        if scripts is None:
            return {}
        return {
            name: PythonBehavior(name, source)
            for name, source in scripts.sources.items()
        }

    def tick(self, node_id: str, node: EngineNode, ctx: EngineContext) -> None:
        # Tick the LIVE bindings: each writes node.uniform_values[name] before Node.render()
        # reads it. A runtime/shape error freezes the uniform at last-good and records a
        # ScriptError into the shared `errors`; the frame always continues.
        scripts = self._nodes.get(node_id)
        if scripts is None:
            return
        self._tick_behaviors(
            node_id, node, ctx, scripts.behaviors, scripts.last_good, self.errors
        )

    def tick_behaviors(
        self,
        node_id: str,
        node: EngineNode,
        ctx: EngineContext,
        behaviors: dict[str, PythonBehavior],
    ) -> None:
        # Tick an EXTERNAL behavior set (the export's fresh instances) against the node. Its last-good
        # AND its errors sink are per-call throwaways — an export must NOT touch the live binding's
        # recorded error (the export is structurally isolated; decision 11), and it ticks each frame
        # fresh with no live state.
        self._tick_behaviors(node_id, node, ctx, behaviors, {}, {})

    def _tick_behaviors(
        self,
        node_id: str,
        node: EngineNode,
        ctx: EngineContext,
        behaviors: dict[str, PythonBehavior],
        last_good: dict[str, Any],
        errors: dict[tuple[str, str], ScriptError],
    ) -> None:
        if (
            not behaviors
        ):  # scriptless node — skip the per-frame active-uniform dict build
            return
        active = {u.name: u for u in node.get_active_uniforms()}
        for name, behavior in behaviors.items():
            key = (node_id, name)
            frozen = last_good.get(name, node.uniform_values.get(name))
            if (
                behavior.error is not None
            ):  # cached compile error — freeze, already recorded at reload
                errors[key] = behavior.error
                node.uniform_values[name] = frozen
                continue
            uniform = active.get(name)
            if not is_scriptable(
                uniform
            ):  # uniform went inactive between reload and tick
                node.uniform_values[name] = frozen
                continue
            try:
                value = behavior.compute(ctx, uniform)
            except (
                _RuntimeScriptError
            ) as e:  # shape mismatch / no instance — authored message
                errors[key] = e.error
                node.uniform_values[name] = frozen
                continue
            except (
                Exception
            ) as e:  # any script failure freezes, never crashes the frame
                errors[key] = ScriptError(
                    name,
                    "runtime",
                    f"{type(e).__name__}: {e}",
                    _user_error_line(name, e),
                )
                node.uniform_values[name] = frozen
                continue
            errors.pop(key, None)
            last_good[name] = value
            node.uniform_values[name] = value

    def drop_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        for key in [k for k in self.errors if k[0] == node_id]:
            self.errors.pop(key, None)
