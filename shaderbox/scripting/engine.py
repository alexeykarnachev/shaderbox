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

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, TypeGuard

import moderngl
from loguru import logger
from OpenGL.GL import GL_INT, GL_SAMPLER_2D, GL_UNSIGNED_INT

from shaderbox.scripting.behavior import (
    PythonBehavior,
    _RuntimeScriptError,
    _user_error_line,
    coerce_one,
)
from shaderbox.scripting.context import EngineContext
from shaderbox.scripting.errors import ScriptError
from shaderbox.uniform_coerce import is_text_array

_SCRIPT_GLOB = "u_*.py"
# A node-brain script: one stateful class whose update returns a dict driving many uniforms. The
# dot keeps the filename out of _SCRIPT_GLOB and out of any GLSL identifier (so the sentinel error
# KEY (node_id, "script.py") can never collide with a per-uniform key). Feature 044.
_BRAIN_FILE = "script.py"


def _resolved_pairs(binding_key: str, raw: object) -> Iterable[tuple[str, object]]:
    # Fan a behavior's raw run() result into (uniform_name, value) pairs — the 1->N seam (decision
    # 3/4): a node-brain (keyed by the sentinel) yields its dict's items; a per-uniform binding is
    # the 1-entry case (its key IS the uniform name). Cardinality is decided by the KEY, never by
    # sniffing the return type. A non-dict brain return is a behavior-level error (the caller's
    # except records it under the sentinel; decision 6).
    if binding_key != _BRAIN_FILE:
        return [(binding_key, raw)]
    if not isinstance(raw, dict):
        raise TypeError(
            f"a node-brain update must return a dict[str, value], got {type(raw).__name__}"
        )
    return list(raw.items())


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


def _freeze(names: set[str], node: EngineNode, last_good: dict[str, Any]) -> None:
    # Hold each name at its last-good value (a behavior-level failure freezes its whole slot set).
    for name in names:
        node.uniform_values[name] = last_good.get(name, node.uniform_values.get(name))


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
        # The uniform names the node-brain (script.py) drove on its last tick (every key it targeted
        # at a real scriptable uniform, a coercion-failed key included) — dynamic (the dict keys),
        # cached for script_driven_uniforms + the behavior-level freeze (decision 6/10). Persists
        # across reset() so a reset-time __init__ failure can still freeze the prior frame's names.
        self.last_driven: set[str] = set()
        # Bad brain keys (typo/orphan/engine-owned) that recorded a soft (node,name) error on the last
        # tick — tracked separately from last_driven (a bad key must NOT claim ownership in
        # script_driven_uniforms) so the stale-clear can pop its error once the key stops being returned.
        self.last_skipped: set[str] = set()
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
        # The per-uniform stems (static) UNION the brain's last-frame driven keys (dynamic), minus
        # the sentinel — a brain's driven set is only known after a tick (decision 10). Used solely
        # by the copilot set_uniform reject (a courtesy; coercion is the real gate).
        node = self._nodes.get(node_id)
        if node is None:
            return set()
        names = set(node.behaviors) | node.last_driven
        names.discard(_BRAIN_FILE)
        return names

    def script_file_for(self, node_id: str, name: str) -> str | None:
        # The scripts/ filename that drives `name`, or None if nothing does. A u_<name>.py wins over
        # the brain (the conflict rule: per-uniform overrides), so it's reported when both exist.
        node = self._nodes.get(node_id)
        if node is None:
            return None
        if name in node.behaviors:
            return f"{name}.py"
        if name in node.last_driven:
            return _BRAIN_FILE
        return None

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

            # The node-brain is a SEPARATE discovery branch — `script.py` has a dot, so _SCRIPT_GLOB
            # (u_*.py) never matches it (decision 1). It binds lazily (its keys validate per-tick),
            # so no _binding_reject here; found.add keeps it past the drop loop below.
            self._reload_brain(scripts_dir / _BRAIN_FILE, node_id, scripts, found)

        # Drop bindings whose file disappeared OR whose uniform went inactive/unscriptable.
        for name in list(scripts.behaviors):
            if name not in found:
                scripts.behaviors.pop(name, None)
                scripts.mtimes.pop(name, None)
                scripts.sources.pop(name, None)
                scripts.last_good.pop(name, None)
                self.errors.pop((node_id, name), None)
                if name == _BRAIN_FILE:
                    # Free each driven/skipped uniform's last-good + its per-KEY error (a coercion-failed
                    # key OR a bad-key soft error records under (node_id, name), not the sentinel — so
                    # popping the sentinel above isn't enough), and clear the cached sets so
                    # script_driven_uniforms reports nothing for the removed brain.
                    for stale in scripts.last_driven | scripts.last_skipped:
                        scripts.last_good.pop(stale, None)
                        self.errors.pop((node_id, stale), None)
                    scripts.last_driven = set()
                    scripts.last_skipped = set()

    def _reload_brain(
        self, path: Path, node_id: str, scripts: "NodeScripts", found: set[str]
    ) -> None:
        # Discover + (re)compile nodes/<id>/scripts/script.py keyed by the _BRAIN_FILE sentinel,
        # mtime-cached like a per-uniform binding. Its error records under (node_id, "script.py").
        if not path.is_file():
            return
        found.add(_BRAIN_FILE)
        try:
            mtime = path.stat().st_mtime
            body = path.read_text(encoding="utf-8")
        except (OSError, ValueError):
            return
        if scripts.mtimes.get(_BRAIN_FILE) == mtime:
            return
        behavior = PythonBehavior(_BRAIN_FILE, body)
        scripts.behaviors[_BRAIN_FILE] = behavior
        scripts.mtimes[_BRAIN_FILE] = mtime
        scripts.sources[_BRAIN_FILE] = body
        key = (node_id, _BRAIN_FILE)
        if behavior.error is not None:
            self.errors[key] = behavior.error
        else:
            self.errors.pop(key, None)

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
            node_id,
            node,
            ctx,
            scripts.behaviors,
            scripts.last_good,
            self.errors,
            scripts.last_driven,
            scripts.last_skipped,
            scripts.warned,
        )

    def tick_behaviors(
        self,
        node_id: str,
        node: EngineNode,
        ctx: EngineContext,
        behaviors: dict[str, PythonBehavior],
    ) -> None:
        # Tick an EXTERNAL behavior set (the export's fresh instances) against the node. EVERY sink —
        # last-good, errors, driven-set, skipped-set AND the warn-once set — is a per-call throwaway, so
        # an export never touches the live binding's recorded error, caches, or warn dedup (the export
        # is structurally isolated; decision 11), and it ticks each frame fresh with no live state.
        self._tick_behaviors(node_id, node, ctx, behaviors, {}, {}, set(), set(), set())

    def _tick_behaviors(
        self,
        node_id: str,
        node: EngineNode,
        ctx: EngineContext,
        behaviors: dict[str, PythonBehavior],
        last_good: dict[str, Any],
        errors: dict[tuple[str, str], ScriptError],
        last_driven: set[str],
        last_skipped: set[str],
        warned: set[str],
    ) -> None:
        if (
            not behaviors
        ):  # scriptless node — skip the per-frame active-uniform dict build
            return
        active = {u.name: u for u in node.get_active_uniforms()}
        # TWO-PASS write order (decision 7): the brain writes its slots FIRST, then per-uniform
        # bindings write LAST so a u_<name>.py overrides the brain's same-named slot. Splitting the
        # passes is the guarantee — NOT behaviors-dict insertion order (a reload can perturb it).
        brain = behaviors.get(_BRAIN_FILE)
        if brain is not None:
            driven, skipped = self._apply_behavior(
                node_id,
                _BRAIN_FILE,
                brain,
                node,
                ctx,
                active,
                last_good,
                errors,
                last_driven,
                warned,
            )
            # An omitted-after-failing key's stale error: clear any key TOUCHED last frame (a real
            # driven key OR a bad/skipped key that recorded a soft error) but NOT touched this frame
            # (decision 8 — no zombie error; an omitted real key keeps its last value). last_driven
            # tracks the REAL driven uniforms only (skipped keys must not claim ownership in
            # script_driven_uniforms); last_skipped tracks the bad keys for their own stale-clear.
            touched = driven | skipped
            for name in (last_driven | last_skipped) - touched:
                errors.pop((node_id, name), None)
            last_driven.clear()
            last_driven.update(driven)
            last_skipped.clear()
            last_skipped.update(skipped)
        for name, behavior in behaviors.items():
            if name == _BRAIN_FILE:
                continue
            self._apply_behavior(
                node_id,
                name,
                behavior,
                node,
                ctx,
                active,
                last_good,
                errors,
                set(),
                warned,
            )

    def _apply_behavior(
        self,
        node_id: str,
        binding_key: str,
        behavior: PythonBehavior,
        node: EngineNode,
        ctx: EngineContext,
        active: dict[str, moderngl.Uniform | moderngl.UniformBlock],
        last_good: dict[str, Any],
        errors: dict[tuple[str, str], ScriptError],
        last_driven: set[str],
        warned: set[str],
    ) -> tuple[set[str], set[str]]:
        # Run ONE behavior + write each (name, raw) pair it yields. Returns (driven, touched): `driven`
        # = names of REAL scriptable uniforms the brain targeted (what script_driven_uniforms reports);
        # `touched` = driven UNION the bad keys (typo/orphan/engine-owned) that recorded a per-key
        # error this frame, so the caller can clear a stale (node,name) error of any key no longer
        # returned (decision 8 — no zombie). A behavior-level failure (compile error / raw throw /
        # non-dict brain return) freezes EVERY name it drove last frame and returns empty. NOTE: `run`
        # (=update) is called ONCE to fan out, BEFORE the per-name is_scriptable gate — so a per-uniform
        # script's state still advances during the brief reload→tick window its uniform is inactive
        # (the value freezes; only `self.*` ticks). Inherent to the shared fan-out, not a regression.
        behavior_key = (node_id, binding_key)
        drove_last = {binding_key} if binding_key != _BRAIN_FILE else set(last_driven)
        if (
            behavior.error is not None
        ):  # cached compile error — freeze, recorded at reload
            errors[behavior_key] = behavior.error
            _freeze(drove_last, node, last_good)
            return set(), set()
        try:
            raw = behavior.run(ctx)
            pairs = list(_resolved_pairs(binding_key, raw))
        except _RuntimeScriptError as e:  # no instance — authored message
            errors[behavior_key] = e.error
            _freeze(drove_last, node, last_good)
            return set(), set()
        except (
            Exception
        ) as e:  # a raw throw (or non-dict brain return) is behavior-level
            errors[behavior_key] = ScriptError(
                binding_key,
                "runtime",
                f"{type(e).__name__}: {e}",
                _user_error_line(behavior.label, e),
            )
            _freeze(drove_last, node, last_good)
            return set(), set()
        errors.pop(
            behavior_key, None
        )  # the run itself succeeded; clear a stale behavior-level error
        # `driven` = every key the brain TARGETED at a real scriptable uniform this frame (a per-key
        # coercion failure still counts — the key IS driven, it just froze). `skipped` = bad keys
        # (typo/orphan/engine-owned) that recorded a soft error but name no real uniform. The caller
        # clears the stale error of any TOUCHED key (driven|skipped) no longer returned (decision 8 —
        # no zombie); script_driven_uniforms reports `driven` only (a bad key must NOT claim ownership).
        driven: set[str] = set()
        skipped: set[str] = set()
        for name, value in pairs:
            key = (node_id, name)
            frozen = last_good.get(name, node.uniform_values.get(name))
            uniform = active.get(name)
            if binding_key == _BRAIN_FILE:
                # A brain key validates LAZILY at tick (its keys are dynamic) through the SAME
                # _binding_reject the per-uniform path uses at reload: engine-owned (u_time…), orphan
                # (typo / inactive), or sampler/block all get a reason. Record a soft ScriptError under
                # (node,name) so the UI surfaces it (NOT loguru-only), warn-once, SKIP with NO write
                # (frozen is None for a never-bound name; decision 5). It goes in `skipped` NOT `driven`
                # — it names no scriptable uniform, so script_driven_uniforms must not claim ownership
                # (the engine-owned-u_time false-ownership bug).
                reason = self._binding_reject(name, active)
                if reason is not None:
                    errors[key] = ScriptError(name, "runtime", reason)
                    skipped.add(name)
                    if name not in warned:
                        logger.warning(
                            f"Node-brain script on node {node_id}: key {reason} — skipped"
                        )
                        warned.add(name)
                    continue
            elif not is_scriptable(uniform):
                # A per-uniform binding's name went inactive between reload and tick — freeze it to
                # last-good (it WAS a real binding; per-uniform orphans/engine-owned are rejected at
                # reload, so they never reach the tick).
                node.uniform_values[name] = frozen
                continue
            assert is_scriptable(
                uniform
            )  # _binding_reject(None) implies a scriptable uniform
            driven.add(name)
            try:
                coerced = coerce_one(value, uniform, name)
            except (
                _RuntimeScriptError
            ) as e:  # per-KEY shape mismatch — freeze ONLY this key
                errors[key] = e.error
                node.uniform_values[name] = frozen
                continue
            errors.pop(key, None)
            last_good[name] = coerced
            node.uniform_values[name] = coerced
        return driven, skipped

    def drop_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        for key in [k for k in self.errors if k[0] == node_id]:
            self.errors.pop(key, None)
