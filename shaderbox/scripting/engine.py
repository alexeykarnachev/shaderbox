"""The CPU-script engine (feature 040) — repo code owned by the headless ProjectSession.

Per node it resolves `scripts/u_<name>.py` against the node's active uniforms, compiles a
`Behavior` per binding (cached by `(path, mtime)`), and on each `tick` writes the computed
value into `node.uniform_values` BEFORE `Node.render()` reads them. A broken script never
raises into the frame loop: the uniform freezes at last-good and a `ScriptError` is recorded.

The engine imports no imgui/glfw/App and no concrete `Node` type — it works against the
`EngineNode` protocol (the `uniform_values` dict + `get_active_uniforms()`), so it stays in
the 025 headless core.
"""

from pathlib import Path
from typing import Any, Protocol, TypeGuard

import moderngl
from loguru import logger
from OpenGL.GL import GL_SAMPLER_2D

from shaderbox.scripting.behavior import Behavior, PythonBehavior, UniformOut
from shaderbox.scripting.context import EngineContext
from shaderbox.scripting.errors import ScriptError

_SCRIPT_GLOB = "u_*.py"


def _is_scriptable(uniform: object) -> TypeGuard[moderngl.Uniform]:
    # A scalar/vector/array uniform the namespace can produce a value for — NOT a UniformBlock
    # (no scalar value) and not a sampler (a texture, not a script value). Duck-typed on the
    # shape attrs the coercion reads, so a test stand-in passes alongside a real moderngl.Uniform.
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
    # One node's compiled bindings + their (path, mtime) cache + per-uniform last-good.
    def __init__(self, scripts_dir: Path) -> None:
        self.scripts_dir = scripts_dir
        self.behaviors: dict[str, Behavior] = {}
        self.mtimes: dict[str, float] = {}
        self.last_good: dict[str, Any] = {}


class ScriptEngine:
    def __init__(self) -> None:
        self._nodes: dict[str, NodeScripts] = {}
        # (node_id, uniform_name) -> the most recent error, for 041's UI to surface.
        self.errors: dict[tuple[str, str], ScriptError] = {}
        # Phase-A scratchpad (042). v1 leaves it empty.
        self.state: dict[str, Any] = {}

    def script_driven_uniforms(self, node_id: str) -> set[str]:
        node = self._nodes.get(node_id)
        return set(node.behaviors) if node else set()

    def reload(self, node_id: str, scripts_dir: Path, node: EngineNode) -> None:
        # Glob the node's scripts dir, (re)compile only files whose mtime changed, drop removed
        # files, and resolve each binding against the node's active uniforms — warning (logged)
        # on an orphan (no such uniform) or a script targeting an unscriptable kind. Cheap when
        # nothing changed: a stat per file, no recompile.
        scripts = self._nodes.get(node_id)
        if scripts is None or scripts.scripts_dir != scripts_dir:
            scripts = NodeScripts(scripts_dir)
            self._nodes[node_id] = scripts

        active = {u.name: u for u in node.get_active_uniforms()}
        found: set[str] = set()
        if scripts_dir.is_dir():
            for path in sorted(scripts_dir.glob(_SCRIPT_GLOB)):
                name = path.stem
                found.add(name)
                if name not in active:
                    logger.warning(
                        f"Script {path.name} targets uniform '{name}' not active on node "
                        f"{node_id} — ignored (orphan)"
                    )
                    continue
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                if (
                    scripts.behaviors.get(name) is not None
                    and scripts.mtimes.get(name) == mtime
                ):
                    continue
                body = path.read_text(encoding="utf-8")
                behavior = PythonBehavior(name, body)
                scripts.behaviors[name] = behavior
                scripts.mtimes[name] = mtime
                key = (node_id, name)
                if behavior.error is not None:
                    self.errors[key] = behavior.error
                else:
                    self.errors.pop(key, None)

        # Drop bindings whose file disappeared.
        for name in list(scripts.behaviors):
            if name not in found:
                scripts.behaviors.pop(name, None)
                scripts.mtimes.pop(name, None)
                scripts.last_good.pop(name, None)
                self.errors.pop((node_id, name), None)

    def tick(self, node_id: str, node: EngineNode, ctx: EngineContext) -> None:
        # Phase A (state evolution) — v1 STUB, no state objects. Phase B (uniform compute):
        # each binding writes node.uniform_values[name] before Node.render() reads it. A
        # compile/runtime/shape error freezes the uniform at last-good and records a ScriptError;
        # the frame always continues.
        scripts = self._nodes.get(node_id)
        if scripts is None:
            return
        active = {u.name: u for u in node.get_active_uniforms()}
        for name, behavior in scripts.behaviors.items():
            key = (node_id, name)
            frozen = scripts.last_good.get(name, node.uniform_values.get(name))
            if (
                behavior.error is not None
            ):  # cached compile error — freeze, already recorded
                node.uniform_values[name] = frozen
                continue
            uniform = active.get(name)
            if not _is_scriptable(uniform):
                node.uniform_values[name] = frozen
                continue
            out = UniformOut(uniform)
            try:
                behavior.compute(ctx, out)
            except (
                Exception
            ) as e:  # any script failure freezes, never crashes the frame
                self.errors[key] = ScriptError(
                    name, "runtime", f"{type(e).__name__}: {e}"
                )
                node.uniform_values[name] = frozen
                continue
            if out.error is not None:  # shape mismatch from out.set
                self.errors[key] = out.error
                node.uniform_values[name] = frozen
                continue
            self.errors.pop(key, None)
            if not out.was_set:  # no out.set() this tick — hold last-good, no error
                node.uniform_values[name] = frozen
                continue
            scripts.last_good[name] = out.value
            node.uniform_values[name] = out.value

    def drop_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        for key in [k for k in self.errors if k[0] == node_id]:
            self.errors.pop(key, None)
