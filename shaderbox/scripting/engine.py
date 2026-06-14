"""The CPU-script engine (feature 041, redesigned by 048 to ONE script per node) — repo code owned
by the headless ProjectSession.

Per node it resolves a single `nodes/<id>/scripts/script.py` (the node-brain) whose
`update(self, ctx) -> dict[str, value]` drives MANY uniforms from ONE stateful instance. The engine
compiles it once (cached by `(path, mtime)`, holding its own state instance), and on each `tick` calls
`update`, fans the returned dict into `(name, value)` pairs, coerces each against the live uniform, and
writes it into `node.uniform_values` BEFORE `Node.render()` reads them. A broken script never raises
into the frame loop: the uniform freezes at last-good and a `ScriptError` is recorded.

Play/stop (048): the live tick takes a `stopped: set[str]` of uniform NAMES the user has frozen for
manual edit — a stopped name still ticks the brain (state advances, the name stays "driven") but its
WRITE is skipped, so the manual value sticks. Export ticks a fresh per-export instance with NO stopped
set (an export always plays the script).

The engine imports no imgui/glfw/App and no concrete `Node` type — it works against the `EngineNode`
protocol (the `uniform_values` dict + `get_active_uniforms()`), so it stays in the 025 headless core.
"""

from collections.abc import Iterable
from dataclasses import dataclass
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

# The single node-brain script: one stateful class whose update returns a dict driving many uniforms.
# One script per node (048 — the per-uniform `u_<name>__<tag>.py` scheme of 044/047 is removed). The
# error key is `(node_id, _BRAIN_FILE)`.
_BRAIN_FILE = "script.py"


def normalize_script_tabs(text: str) -> str:
    # Tabs in a script are banned at the boundary: every tab -> 4 spaces (the project standard), so the
    # on-disk script + the copilot's old_str both live in a spaces-only world (Python indentation is
    # significant; mixed tabs/spaces break the indent-aware edit matcher + are a footgun in general).
    # Also normalize CRLF/CR -> LF so an edit's old_str can't miss on a line-ending mismatch.
    return text.replace("\t", "    ").replace("\r\n", "\n").replace("\r", "\n")


@dataclass(frozen=True)
class BrainStatus:
    # The node-brain's UI-facing state (feature 042's strip). sentinel_error is the brain's
    # compile/run failure (it drives nothing when set); soft_errors are (key, error) for homeless
    # keys (typo/orphan) that name no real uniform row.
    sentinel_error: "ScriptError | None"
    driven_count: int
    soft_errors: list[tuple[str, "ScriptError"]]


@dataclass(frozen=True)
class ScriptProbe:
    # The synchronous feedback a copilot write_script reads back (feature 043). compile_error is the
    # live reload verdict (None = clean). The rest come from an ISOLATED dry-tick (the live node + the
    # live engine state are untouched): runtime_error = the brain RAN but `update` raised / returned a
    # non-dict at some frame (the uniform freezes from there — distinct from a compile error and from a
    # per-key shape error); driven = the real uniforms the brain drove; per_key_errors = shape/coercion
    # failures on real uniforms; orphan_keys = keys naming no active uniform (typo); samples =
    # (t, {name: value}) at each sample time — the motion signal (values differ across t).
    compile_error: "ScriptError | None"
    driven: set[str]
    per_key_errors: list[tuple[str, "ScriptError"]]
    orphan_keys: list[tuple[str, "ScriptError"]]
    samples: list[tuple[float, dict[str, Any]]]
    runtime_error: "ScriptError | None" = None


def is_scriptable(uniform: object) -> TypeGuard[moderngl.Uniform]:
    # A scalar/vector/array uniform the engine can drive — NOT a UniformBlock (no scalar value)
    # and not a sampler (a texture, not a script value). Duck-typed on the shape attrs the
    # coercion reads, so a test stand-in passes alongside a real moderngl.Uniform.
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


def _freeze(
    names: set[str],
    node: EngineNode,
    last_good: dict[str, Any],
    sink: dict[str, Any] | None = None,
) -> None:
    # Hold each name at its last-good value (a behavior-level failure freezes its whole slot set).
    # A dry-run passes a `sink`: writes (and the fallback read) land there, never on the live node.
    target = node.uniform_values if sink is None else sink
    for name in names:
        target[name] = last_good.get(name, node.uniform_values.get(name))


class NodeScripts:
    # One node's compiled brain + its (path, mtime) cache + the raw source (a fresh export behavior
    # recompiles from this cached source, never a fresh disk read) + per-uniform-name last-good.
    def __init__(self, scripts_dir: Path) -> None:
        self.scripts_dir = scripts_dir
        self.brain: PythonBehavior | None = None
        self.mtime: float | None = None
        self.source: str | None = None
        self.last_good: dict[str, Any] = {}
        # The uniform names the brain drove on its last tick (every key it targeted at a real
        # scriptable uniform, a coercion-failed key included) — dynamic (the dict keys), cached for
        # script_driven_uniforms + the behavior-level freeze. Persists across reset() so a reset-time
        # __init__ failure can still freeze the prior frame's names.
        self.last_driven: set[str] = set()
        # Bad brain keys (typo/orphan) that recorded a soft (node,name) error on the last tick —
        # tracked separately from last_driven (a bad key must NOT claim ownership in
        # script_driven_uniforms) so the stale-clear can pop its error once the key stops being returned.
        self.last_skipped: set[str] = set()
        # Orphan/unsupported-key names already warned, so reload (per-frame) logs once on the
        # transition, not every frame.
        self.warned: set[str] = set()


def _stub_kind(uniform: moderngl.Uniform) -> tuple[str, str]:
    # The (output-type name, coercion-valid default expression) for one uniform's shape. Drives the
    # brain stub's commented example lines + the explicit-import set.
    dim = uniform.dimension
    n = uniform.array_length
    gl_type = getattr(uniform, "gl_type", None)

    if is_text_array(uniform):
        return "Text", 'Text("")'
    if n > 1:
        return "Array", f"Array([0.0] * {n * dim})"
    if dim == 2:
        return "Vec2", "Vec2(0.0, 0.0)"
    if dim == 3:
        return "Vec3", "Vec3(0.0, 0.0, 0.0)"
    if dim == 4:
        return "Vec4", "Vec4(0.0, 0.0, 0.0, 0.0)"
    if gl_type in (GL_INT, GL_UNSIGNED_INT):
        return "int", "0"  # a scalar int/uint stub returns an int, not 0.0
    return "float", "0.0"


# Scoped docstrings (045 E1; 048 documents the dict contract + play/stop). The per-frame ctx reference
# lives on `update`; the class docstring carries the high-level "what this drives"; __init__ states it
# runs once. A script is plain Python — import what you need at the top.
_UPDATE_DOC = (
    '        """Compute this frame\'s uniform values.\n'
    "\n"
    "        Return a dict mapping uniform NAME -> value. A uniform you return is DRIVEN by the\n"
    "        script (it PLAYS); a uniform you omit (or map to None) stays MANUAL — you edit it by\n"
    "        hand in the panel. Stop a playing uniform (its row's stop button, or just drag it) to\n"
    "        edit it by hand without deleting it from the dict.\n"
    "\n"
    "        Vec2/3/4 are RETURN wrappers, not a math type: do all the math on plain floats\n"
    "        (self.x, self.vx, ...) and wrap ONLY in the return — Vec2(x, y). A Vec2 has no .x/.y\n"
    "        (it is a tuple: v[0], v[1]) and Vec2(...) * n repeats the tuple, it does NOT scale.\n"
    "\n"
    "        Args:\n"
    "            ctx.t: Elapsed seconds since start.\n"
    "            ctx.dt: Delta seconds since the previous frame.\n"
    "            ctx.frame: Frame index.\n"
    "            ctx.mouse: Cursor over the canvas (x, y in 0..1, y-up; 0.5,0.5 on export).\n"
    '        """\n'
)
_INIT_DOC = (
    '        """Set up state (runs ONCE — at app start, before the first render, and on'
    ' reload)."""\n'
)


def _brain_import_line(annotations: Iterable[str]) -> str:
    # The explicit import line atop the stub (048 decision 8): `ScriptBehavior` + `Ctx` always, plus
    # only the output types the node's uniforms reference. Visible so the user sees what's available;
    # the engine also injects these names as a fallback (behavior.py::_build_globals).
    names = ["ScriptBehavior", "Ctx"]
    for ann in annotations:
        if ann in ("Vec2", "Vec3", "Vec4", "Array", "Text") and ann not in names:
            names.append(ann)
    return f"from shaderbox.scripting import {', '.join(names)}\n"


def brain_stub_for(uniforms: Iterable[moderngl.Uniform]) -> str:
    # The ready-to-edit node-brain (044; 048): one stateful class whose update returns a dict driving
    # MANY uniforms. update returns an EMPTY dict by default (a fresh script drives nothing — every
    # uniform stays manual); the node's scriptable uniforms are listed as COMMENTED examples so the
    # user sees what's available + the shape to return. The annotation is BARE `-> dict` (never
    # `dict[str, Any]`: `Any` isn't in the exec globals, so the eager annotation eval would freeze it).
    scriptable = [u for u in uniforms if is_scriptable(u)]
    kinds = [(u.name, *_stub_kind(u)) for u in scriptable]
    import_line = _brain_import_line(ann for _, ann, _ in kinds)
    if kinds:
        examples = "".join(
            f"            # {name!r}: {default},  # {ann}\n"
            for name, ann, default in kinds
        )
        body = (
            "        return {\n"
            "            # Uncomment a line + replace the value to drive that uniform:\n"
            f"{examples}"
            "        }\n"
        )
    else:
        body = "        return {}\n"
    return (
        f"import math\n\n"
        f"{import_line}\n"
        f"class Behavior(ScriptBehavior):\n"
        f'    """Drive many uniforms from one object each frame (node-brain). Keep state on self."""\n'
        f"\n"
        f"    def __init__(self) -> None:\n"
        f"{_INIT_DOC}"
        f"        pass\n\n"
        f"    def update(self, ctx: Ctx) -> dict:\n"
        f"{_UPDATE_DOC}"
        f"{body}"
    )


class ScriptEngine:
    def __init__(self, engine_driven: frozenset[str] = frozenset()) -> None:
        self._nodes: dict[str, NodeScripts] = {}
        # (node_id, name) -> the most recent error, for the UI to surface. The brain's
        # compile/run error keys on (node_id, _BRAIN_FILE); a per-key shape/orphan error on
        # (node_id, uniform_name).
        self.errors: dict[tuple[str, str], ScriptError] = {}
        # Engine-owned uniform names (u_time/u_aspect/u_resolution + table uniforms) — render()
        # hardcodes these, so a script on one would silently no-op. Passed in by ProjectSession (NOT
        # imported from core, which pulls in glfw — the headless boundary). Empty in a bare test engine.
        self._engine_driven = engine_driven

    def script_driven_uniforms(self, node_id: str) -> set[str]:
        # The uniform names the brain drove on its last tick (decision 10 — only known after a tick).
        # Used by the copilot set_uniform reject + the UI's play/stop button gate (a name here is
        # script-targeted: playing or stopped).
        node = self._nodes.get(node_id)
        return set(node.last_driven) if node is not None else set()

    def brain_status(self, node_id: str) -> "BrainStatus | None":
        # The node-brain's UI status (feature 042), or None when the node has no script.py. Whether a
        # brain is bound, its sentinel compile/run error (drives zero rows when set), the count of real
        # uniforms it drove last tick, and its HOMELESS soft-key errors (typo/orphan keys naming no row).
        node = self._nodes.get(node_id)
        if node is None or node.brain is None:
            return None
        sentinel = self.errors.get((node_id, _BRAIN_FILE))
        soft = [
            (name, err)
            for name in sorted(node.last_skipped)
            if (err := self.errors.get((node_id, name))) is not None
        ]
        return BrainStatus(
            sentinel_error=sentinel,
            driven_count=len(node.last_driven),
            soft_errors=soft,
        )

    def has_script(self, node_id: str) -> bool:
        # True when the node has a bound brain (script.py exists + compiled, error or not).
        node = self._nodes.get(node_id)
        return node is not None and node.brain is not None

    def reload(self, node_id: str, scripts_dir: Path, node: EngineNode) -> None:
        # Discover + (re)compile the node's `script.py` if its mtime changed (a recompile makes a
        # FRESH instance — state resets on edit), drop it if the file vanished. The brain binds by
        # EXISTENCE (048 — no active flag; the file IS the binding). Cheap when nothing changed: a stat.
        scripts = self._nodes.get(node_id)
        if scripts is None or scripts.scripts_dir != scripts_dir:
            scripts = NodeScripts(scripts_dir)
            self._nodes[node_id] = scripts

        path = scripts_dir / _BRAIN_FILE
        if not path.is_file():
            if scripts.brain is not None:
                self._drop_brain(node_id, scripts)
            return

        try:
            mtime = path.stat().st_mtime
            body = path.read_text(encoding="utf-8")
        except (OSError, ValueError):
            # A vanished / half-saved / non-UTF8 file mid-edit — keep the cached brain, never raise
            # into the frame loop (ValueError covers UnicodeDecodeError).
            return
        if scripts.mtime == mtime:
            return
        behavior = PythonBehavior(_BRAIN_FILE, body)
        scripts.brain = behavior
        scripts.mtime = mtime
        scripts.source = body
        key = (node_id, _BRAIN_FILE)
        if behavior.error is not None:
            self.errors[key] = behavior.error
        else:
            self.errors.pop(key, None)

    def _drop_brain(self, node_id: str, scripts: "NodeScripts") -> None:
        # Tear down a removed brain: free its last-good + every per-key error (a coercion-failed key
        # or a bad-key soft error records under (node_id, name), so popping the sentinel isn't enough)
        # and clear the cached sets so script_driven_uniforms reports nothing.
        scripts.brain = None
        scripts.mtime = None
        scripts.source = None
        self.errors.pop((node_id, _BRAIN_FILE), None)
        for stale in scripts.last_driven | scripts.last_skipped:
            scripts.last_good.pop(stale, None)
            self.errors.pop((node_id, stale), None)
        scripts.last_driven = set()
        scripts.last_skipped = set()
        scripts.warned = set()

    def _binding_reject(
        self,
        name: str,
        active: dict[str, moderngl.Uniform | moderngl.UniformBlock],
    ) -> str | None:
        # Why a brain key can't bind to `name` (None = it can). An engine-owned key (u_time…) is
        # dropped SILENTLY upstream (decision 5), so it never reaches this — this covers orphan/typo
        # and sampler/block keys.
        uniform = active.get(name)
        if uniform is None:
            return f"no active uniform '{name}' (orphan key)"
        if not is_scriptable(uniform):
            return f"'{name}' is a sampler/block — not a scriptable value"
        return None

    def reset(self, node_id: str) -> None:
        # Re-instantiate the live brain (re-run __init__) without recompiling — the manual "restart".
        # Sync the engine's recorded error to the behavior's post-reset state (a recovered __init__
        # clears it; a still-raising one re-records) so a consumer reading `errors` off-tick sees the
        # truth immediately.
        scripts = self._nodes.get(node_id)
        if scripts is None or scripts.brain is None:
            return
        scripts.brain.reset()
        key = (node_id, _BRAIN_FILE)
        if scripts.brain.error is not None:
            self.errors[key] = scripts.brain.error
        else:
            self.errors.pop(key, None)

    def fresh_behavior_for(self, node_id: str) -> PythonBehavior | None:
        # A NEW brain instance, independent of the live registry's instance — recompiled from the
        # live registry's CACHED source (not a fresh disk read, so an export never sees a half-saved
        # mid-edit file). The export path ticks THIS so an exported integrator starts from a clean
        # __init__ regardless of live state.
        scripts = self._nodes.get(node_id)
        if scripts is None or scripts.source is None:
            return None
        return PythonBehavior(_BRAIN_FILE, scripts.source)

    def tick(
        self,
        node_id: str,
        node: EngineNode,
        ctx: EngineContext,
        stopped: frozenset[str] = frozenset(),
    ) -> None:
        # Tick the LIVE brain: it writes node.uniform_values[name] before Node.render() reads it. A
        # name in `stopped` (the user froze it for manual edit, 048) still ticks the brain + counts as
        # driven, but its WRITE is skipped so the manual value sticks. A runtime/shape error freezes
        # the uniform at last-good and records a ScriptError; the frame always continues.
        scripts = self._nodes.get(node_id)
        if scripts is None or scripts.brain is None:
            return
        self._tick_brain(
            node_id,
            node,
            ctx,
            scripts.brain,
            scripts.last_good,
            self.errors,
            scripts.last_driven,
            scripts.last_skipped,
            scripts.warned,
            stopped,
        )

    def tick_export(
        self, node_id: str, node: EngineNode, ctx: EngineContext, brain: PythonBehavior
    ) -> None:
        # Tick an EXTERNAL brain (the export's fresh instance) against the node. EVERY sink is a
        # per-call throwaway, so an export never touches the live brain's recorded error/caches/warn
        # dedup (structurally isolated). NO stopped set — an export always plays the script.
        self._tick_brain(
            node_id, node, ctx, brain, {}, {}, set(), set(), set(), frozenset()
        )

    def dry_run(
        self,
        node_id: str,
        node: EngineNode,
        sample_times: tuple[float, ...],
        fps: int,
    ) -> ScriptProbe:
        # Synchronous copilot feedback (043): compile verdict from the ALREADY-LIVE state (the caller
        # reloaded the file at write time — no reload here, which would mutate live state), then an
        # ISOLATED dry-tick. ONE fresh brain is stepped CONTINUOUSLY through the export-clock frames so
        # self.* accumulates (an integrator animates correctly); every write lands in a per-call sink,
        # so the live node + live engine state are byte-identical afterward. Returns the driven set,
        # per-key + orphan errors, and the driven uniforms' VALUES at each sample time.
        compile_error = self.errors.get((node_id, _BRAIN_FILE))
        brain = self.fresh_behavior_for(node_id)
        if brain is None or compile_error is not None:
            return ScriptProbe(compile_error, set(), [], [], [])

        dt = 1.0 / fps
        max_frame = max((round(t * fps) for t in sample_times), default=0)
        # frame -> the first sample time landing on it; setdefault keeps the earliest so two close
        # times rounding to one frame don't silently drop a sample (the dict-comp would keep the last).
        want: dict[int, float] = {}
        for t in sample_times:
            want.setdefault(round(t * fps), t)
        sink: dict[str, Any] = {}
        errors: dict[tuple[str, str], ScriptError] = {}
        driven: set[str] = set()
        skipped: set[str] = set()
        samples: list[tuple[float, dict[str, Any]]] = []
        # The probe reports "did this EVER fail across the window", not the final-frame snapshot: the
        # live engine's `errors` dict SELF-HEALS (a good tick pops a key), so a TRANSIENT raise/coercion/
        # orphan that recovers before the last sampled frame would be lost. Accumulate each category
        # right after each tick, before the next one can pop it.
        seen_driven: set[str] = set()
        seen_skipped: set[str] = set()
        worst: dict[tuple[str, str], ScriptError] = {}
        for frame in range(max_frame + 1):
            ctx = EngineContext(t=frame * dt, dt=dt, frame=frame)
            self._tick_brain(
                node_id,
                node,
                ctx,
                brain,
                {},
                errors,
                driven,
                skipped,
                set(),
                frozenset(),
                values_sink=sink,
            )
            seen_driven |= driven
            seen_skipped |= skipped
            for key, err in errors.items():
                worst.setdefault(key, err)  # first failure across the window wins
            if frame in want:
                samples.append(
                    (want[frame], {name: sink[name] for name in driven if name in sink})
                )

        per_key = [
            (name, err)
            for name in sorted(seen_driven)
            if (err := worst.get((node_id, name))) is not None
        ]
        orphan = [
            (name, err)
            for name in sorted(seen_skipped)
            if (err := worst.get((node_id, name))) is not None
        ]
        # A behavior-level error seen at ANY frame = `update` raised / returned a non-dict at some point
        # (the script compiled but CRASHES at runtime). Surface it so the verdict isn't a false ANIMATING
        # off the values a recovered-by-the-last-frame crash leaves in the sink.
        runtime_error = worst.get((node_id, _BRAIN_FILE))
        # The probe is the SINGLE source of truth for the driven set in the headless/copilot path, where
        # no live tick warms NodeScripts.last_driven. Stash it there so script_driven_uniforms (the
        # working-set marker + the set_uniform reject) agrees with this write's verdict. Safe: last_driven
        # is metadata, the next live tick overwrites it; the byte-identical invariant covers uniform_values.
        scripts = self._nodes.get(node_id)
        if scripts is not None:
            scripts.last_driven = set(seen_driven)
        return ScriptProbe(
            None,
            set(seen_driven),
            per_key,
            orphan,
            samples,
            runtime_error=runtime_error,
        )

    def _tick_brain(
        self,
        node_id: str,
        node: EngineNode,
        ctx: EngineContext,
        brain: PythonBehavior,
        last_good: dict[str, Any],
        errors: dict[tuple[str, str], ScriptError],
        last_driven: set[str],
        last_skipped: set[str],
        warned: set[str],
        stopped: frozenset[str],
        values_sink: dict[str, Any] | None = None,
    ) -> None:
        # `values_sink` (the dry-run path): every uniform-value WRITE lands there + the freeze-fallback
        # READ consults it, so the LIVE node is never written. None = the live tick (write the node).
        write_target = node.uniform_values if values_sink is None else values_sink
        active = {u.name: u for u in node.get_active_uniforms()}
        behavior_key = (node_id, _BRAIN_FILE)
        drove_last = set(last_driven)

        if brain.error is not None:  # cached compile error — freeze, recorded at reload
            errors[behavior_key] = brain.error
            _freeze(drove_last, node, last_good, values_sink)
            return
        try:
            raw = brain.run(ctx)
        except _RuntimeScriptError as e:  # no instance — authored message
            errors[behavior_key] = e.error
            _freeze(drove_last, node, last_good, values_sink)
            return
        except Exception as e:  # a raw throw is behavior-level
            errors[behavior_key] = ScriptError(
                _BRAIN_FILE,
                "runtime",
                f"{type(e).__name__}: {e}",
                _user_error_line(brain.label, e),
            )
            _freeze(drove_last, node, last_good, values_sink)
            return
        if not isinstance(raw, dict):
            errors[behavior_key] = ScriptError(
                _BRAIN_FILE,
                "runtime",
                f"update must return a dict[str, value], got {type(raw).__name__}",
            )
            _freeze(drove_last, node, last_good, values_sink)
            return
        errors.pop(
            behavior_key, None
        )  # the run succeeded; clear a stale behavior-level error

        driven: set[str] = set()
        skipped: set[str] = set()
        for name, value in raw.items():
            key = (node_id, name)
            frozen = write_target.get(
                name, last_good.get(name, node.uniform_values.get(name))
            )
            uniform = active.get(name)
            # An engine-owned key (u_time…) is SILENTLY dropped (decision 5): the renderer owns that
            # slot and a brain can't be expected to avoid naming it.
            if name in self._engine_driven:
                continue
            reason = self._binding_reject(name, active)
            if reason is not None:
                # An orphan/typo/sampler key: record a soft error under (node,name) so the UI surfaces
                # it, warn-once, SKIP with no write. It goes in `skipped` NOT `driven` (it names no
                # scriptable uniform, so script_driven_uniforms must not claim ownership).
                errors[key] = ScriptError(name, "runtime", reason)
                skipped.add(name)
                if name not in warned:
                    logger.warning(
                        f"Node-brain on node {node_id}: key {reason} — skipped"
                    )
                    warned.add(name)
                continue
            assert is_scriptable(
                uniform
            )  # _binding_reject(None) implies a scriptable uniform
            driven.add(
                name
            )  # driven BEFORE coerce/write, so a stopped/failed key still counts
            try:
                coerced = coerce_one(value, uniform, name)
            except (
                _RuntimeScriptError
            ) as e:  # per-KEY shape mismatch — freeze ONLY this key
                errors[key] = e.error
                # A STOPPED key keeps the user's manual value (don't clobber it with stale last-good);
                # a playing key freezes at last-good, the freeze-as-data behavior.
                if name not in stopped:
                    write_target[name] = frozen
                continue
            errors.pop(key, None)
            last_good[name] = coerced
            # Play/stop (048): a STOPPED uniform's value is NOT written (the manual value sticks); the
            # brain still ran + the name still counts as driven (so the row keeps its play/stop button).
            if name not in stopped:
                write_target[name] = coerced

        # An omitted-after-failing key's stale error: clear any key TOUCHED last frame (driven OR a
        # bad/skipped key) but NOT touched this frame (decision 8 — no zombie; an omitted real key
        # keeps its last value).
        touched = driven | skipped
        for name in (last_driven | last_skipped) - touched:
            errors.pop((node_id, name), None)
        last_driven.clear()
        last_driven.update(driven)
        last_skipped.clear()
        last_skipped.update(skipped)

    def drop_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        for key in [k for k in self.errors if k[0] == node_id]:
            self.errors.pop(key, None)
