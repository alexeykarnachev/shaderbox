"""The CPU-script engine (feature 041, redesigned by 048 to ONE script per document) — repo code owned
by the headless ProjectSession.

Per document it resolves a single `documents/<id>/scripts/script.py` (the document script) whose
`update(self, ctx) -> dict` drives MANY uniforms across MANY passes from ONE stateful instance. The
engine compiles it once (cached by `(path, mtime)`, holding its own state instance), and on each `tick`
calls `update`, routes the returned dict per 069 D3, coerces each value against the live uniform of the
target pass, and writes it into that pass's `uniform_values` BEFORE `Pass.render()` reads them. A broken
script never raises into the frame loop: the uniform freezes at last-good and a `ScriptError` is recorded.

Routing (069 D3): the VALUE's type decides. A `dict` value is a PASS BLOCK — `{"paint": {"u_b": v}}`
drives that pass alone. Any other value is a BROADCAST — a bare key drives that uniform on every pass
declaring it. Broadcasts apply first and pass blocks second, so specific beats general regardless of the
author's insertion order. An unknown pass, or a key no target declares, is a soft error the UI's strip
shows. `coerce_one` rejects a dict outright, which is what keeps the dispatch unambiguous.

Play/stop (048, pass-qualified by 069): the live tick takes a `stopped` set of `(pass, name)` keys the
user has frozen for manual edit — a stopped key still ticks the script (state advances, the key stays
"driven") but its WRITE is skipped, so the manual value sticks. Export ticks a fresh per-export instance
with NO stopped set (an export always plays the script).

The engine imports no imgui/glfw/App and no concrete type — it works against the `ScriptTarget`
protocol (a document's `passes` by name, each a `ScriptPass`), so it stays in the 025 headless core.
A real `Document` satisfies it structurally.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeGuard

import moderngl
from OpenGL.GL import GL_INT, GL_SAMPLER_2D, GL_UNSIGNED_INT

from shaderbox.paths import DOCUMENT_SCRIPT_BASENAME
from shaderbox.scripting.behavior import (
    PythonBehavior,
    _RuntimeScriptError,
    _user_error_line,
    coerce_one,
)
from shaderbox.scripting.context import EngineContext
from shaderbox.scripting.errors import ScriptError
from shaderbox.scripting.keys import StoppedKey
from shaderbox.uniform_coerce import is_text_array

# The single document script: one stateful class whose update returns a dict driving many uniforms.
# One script per document (048 — the per-uniform `u_<name>__<tag>.py` scheme of 044/047 is removed). The
# error key is `(document_id, "", _SCRIPT_FILE)`.
_SCRIPT_FILE = DOCUMENT_SCRIPT_BASENAME


def normalize_script_tabs(text: str) -> str:
    # Tabs in a script are banned at the boundary: every tab -> 4 spaces (the project standard), so the
    # on-disk script + the copilot's old_str both live in a spaces-only world (Python indentation is
    # significant; mixed tabs/spaces break the indent-aware edit matcher + are a footgun in general).
    # Also normalize CRLF/CR -> LF so an edit's old_str can't miss on a line-ending mismatch.
    return text.replace("\t", "    ").replace("\r\n", "\n").replace("\r", "\n")


@dataclass(frozen=True)
class ScriptStatus:
    # The document script's UI-facing state (feature 042's strip). sentinel_error is the script's
    # compile/run failure (it drives nothing when set); soft_errors are (pass, key, error) for
    # homeless keys (typo/orphan) that name no real uniform row. A bare key no pass declares
    # carries "" as its pass.
    sentinel_error: "ScriptError | None"
    driven_count: int
    soft_errors: list[tuple[str, str, "ScriptError"]]


@dataclass(frozen=True)
class ScriptProbe:
    # The synchronous feedback a copilot write_script reads back (feature 043). compile_error is the
    # live reload verdict (None = clean). The rest come from an ISOLATED dry-tick (the live document + the
    # live engine state are untouched): runtime_error = the script RAN but `update` raised / returned a
    # non-dict at some frame (the uniform freezes from there — distinct from a compile error and from a
    # per-key shape error); driven = the real (pass, uniform) pairs the script drove; per_key_errors =
    # shape/coercion failures on real uniforms AND keys the engine refused (a sampler/block, an
    # unknown pass); orphan_keys = (pass, key) for a key naming no active uniform, which since 079
    # D5 is a normal authoring state and carries no error — the agent still needs the fact, because
    # such a key drives nothing; samples = (t, {(pass, name): value}) at each sample time — the
    # motion signal (values differ across t). Both lists carry "" as the pass for a bare key.
    compile_error: "ScriptError | None"
    driven: set[tuple[str, str]]
    per_key_errors: list[tuple[str, str, "ScriptError"]]
    orphan_keys: list[tuple[str, str]]
    samples: list[tuple[float, dict[tuple[str, str], Any]]]
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


class ScriptPass(Protocol):
    # The slice of ONE render pass the engine writes into — nothing GL-program-specific.
    uniform_values: dict[str, Any]

    # False only while the pass has NEVER ATTEMPTED a compile: the engine skips it this tick rather
    # than forcing a compile from inside the script tick (066 D1). True once a compile was attempted,
    # whether it succeeded (route normally) or FAILED (an empty active map, so its keys take the
    # ordinary orphan path and the user reads why on the strip). Read-only, so a `Pass` satisfying it
    # with a property is accepted.
    @property
    def script_ready(self) -> bool: ...

    def get_active_uniforms(
        self,
    ) -> list[moderngl.Uniform | moderngl.UniformBlock]: ...


class ScriptTarget(Protocol):
    # The slice of a DOCUMENT the engine routes across: its passes by name. The engine reads no
    # graph, no output notion — which is what makes "what you look at decides what the script
    # drives" structurally impossible. A read-only Mapping, not a dict: the engine never inserts a
    # pass, and a mutable dict is invariant in its value, so a real `dict[str, Pass]` would not
    # satisfy the protocol.
    @property
    def passes(self) -> Mapping[str, ScriptPass]: ...


def _freeze(
    keys: set[tuple[str, str]],
    document: ScriptTarget,
    last_good: dict[tuple[str, str], Any],
    sink: dict[tuple[str, str], Any] | None = None,
) -> None:
    # Hold each (pass, name) at its last-good value (a behavior-level failure freezes its whole key
    # set). A dry-run passes a `sink`: writes (and the fallback read) land there, keyed by the same
    # pairs, never on the live document.
    for key in keys:
        pass_name, name = key
        render_pass = document.passes.get(pass_name)
        live = render_pass.uniform_values.get(name) if render_pass is not None else None
        value = last_good.get(key, live)
        if sink is not None:
            sink[key] = value
        elif render_pass is not None:
            render_pass.uniform_values[name] = value


class DocumentScripts:
    # One document's compiled script + its (path, mtime) cache + the raw source (a fresh export behavior
    # recompiles from this cached source, never a fresh disk read) + per-uniform-name last-good.
    def __init__(self, scripts_dir: Path) -> None:
        self.scripts_dir = scripts_dir
        self.behavior: PythonBehavior | None = None
        self.mtime: float | None = None
        self.source: str | None = None
        self.last_good: dict[tuple[str, str], Any] = {}
        # The (pass, name) pairs the script drove on its last tick (every key it routed to a real
        # scriptable uniform, a coercion-failed key included) — dynamic (the routing result), cached
        # for script_driven_uniforms + the behavior-level freeze. Persists across reset() so a
        # reset-time __init__ failure can still freeze the prior frame's keys.
        self.last_driven: set[tuple[str, str]] = set()
        # Bad script keys (typo/orphan/unknown pass) that recorded a soft error on the last tick —
        # tracked separately from last_driven (a bad key must NOT claim ownership in
        # script_driven_uniforms) so the stale-clear can pop its error once the key stops being
        # returned. A bare key no pass declares carries "" as its pass.
        self.last_skipped: set[tuple[str, str]] = set()


def _stub_kind(uniform: moderngl.Uniform) -> tuple[str, str]:
    # The (output-type name, coercion-valid default expression) for one uniform's shape. Drives the
    # script stub's commented example lines + the explicit-import set.
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


# Scoped docstrings (045 E1; 048 documents the dict contract + play/stop; 079 D3 sets the style:
# PEP 257 with the Google layout, one fact per line, no `;`-joined lists). The per-frame ctx
# reference is NOT repeated here — `K` on `ctx` or on a field is the one authored home
# (`api_doc.py`), and a copy in the stub is a copy that drifts.
_UPDATE_DOC = (
    '        """Return this frame\'s uniform values.\n'
    "\n"
    "        Called once per drawn frame. Keep state on self between frames; a value that is\n"
    "        only a function of ctx.t belongs in the shader instead.\n"
    "\n"
    "        Args:\n"
    "            ctx: The clock and the cursor. Press K on `ctx` or on one of its fields for\n"
    "                what each one means.\n"
    "\n"
    "        Returns:\n"
    "            A dict of uniform names to values. A bare key drives that uniform on every\n"
    "            pass declaring it. A key mapped to a dict is a pass block, driving that one\n"
    "            pass and winning over the bare key. A uniform you return plays, meaning the\n"
    "            script owns it; one you omit or map to None stays manual.\n"
    "\n"
    "            Values are float or int, Vec2 / Vec3 / Vec4, Array([...]) for an array\n"
    '            uniform, or Text("...") for a text one.\n'
    '        """\n'
)
_INIT_DOC = (
    '        """Set up state that survives across frames.\n'
    "\n"
    "        Runs once: at app start, before the first render, and on every reload.\n"
    '        """\n'
)


def _script_import_line(annotations: Iterable[str]) -> str:
    # The explicit import line atop the stub (048 decision 8): `ScriptBehavior` + `Ctx` always, plus
    # only the output types the document's uniforms reference. Visible so the user sees what's available;
    # the engine also injects these names as a fallback (behavior.py::_build_globals).
    names = ["ScriptBehavior", "Ctx"]
    for ann in annotations:
        if ann in ("Vec2", "Vec3", "Vec4", "Array", "Text") and ann not in names:
            names.append(ann)
    return f"from shaderbox.scripting import {', '.join(names)}\n"


def script_stub_for(uniforms_by_pass: dict[str, list[moderngl.Uniform]]) -> str:
    # The ready-to-edit document script (044; 048; 069 routes it across passes): one stateful class
    # whose update returns a dict driving MANY uniforms across MANY passes. update returns an EMPTY
    # dict by default (a fresh script drives nothing — every uniform stays manual); each pass's
    # scriptable uniforms are listed as a COMMENTED block so the user sees what's available + the two
    # addressing forms. The annotation is BARE `-> dict` (never `dict[str, Any]`: `Any` isn't in the
    # exec globals, so the eager annotation eval would freeze it).
    kinds_by_pass: dict[str, list[tuple[str, str, str]]] = {
        pass_name: [(u.name, *_stub_kind(u)) for u in uniforms if is_scriptable(u)]
        for pass_name, uniforms in uniforms_by_pass.items()
    }
    import_line = _script_import_line(
        ann for kinds in kinds_by_pass.values() for _, ann, _ in kinds
    )
    if any(kinds_by_pass.values()):
        blocks = ""
        for pass_name, kinds in kinds_by_pass.items():
            # Double quotes, matching the bare-key example above and the design note's snippet, so
            # everything a user copies out of the stub is quoted one way.
            blocks += f'            # "{pass_name}": {{\n'
            if kinds:
                blocks += "".join(
                    f'            #     "{name}": {default},  # {ann}\n'
                    for name, ann, default in kinds
                )
            else:
                blocks += "            #     (no scriptable uniforms)\n"
            blocks += "            # },\n"
        body = (
            "        return {\n"
            "            # A bare key drives that uniform on EVERY pass declaring it:\n"
            '            #     "u_time_scale": 0.5,\n'
            "            # A pass block drives one pass only, and wins over a bare key:\n"
            f"{blocks}"
            "        }\n"
        )
    else:
        body = "        return {}\n"
    return (
        f"import math\n\n"
        f"{import_line}\n\n"
        f"class Behavior(ScriptBehavior):\n"
        f'    """Drive this document\'s uniforms, one object for the whole document.\n'
        f"\n"
        f"    `update` runs once per drawn frame and returns what to drive. State that has to\n"
        f"    survive between frames lives on self.\n"
        f'    """\n'
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
        self._documents: dict[str, DocumentScripts] = {}
        # (document_id, pass_name, name) -> the most recent error, for the UI to surface. The
        # script's compile/run error keys on (document_id, "", _SCRIPT_FILE); a per-key
        # shape/orphan error on (document_id, pass_name, uniform_name), with "" as the pass for a
        # document-level error and for a bare key no pass declares.
        self.errors: dict[tuple[str, str, str], ScriptError] = {}
        # Engine-owned uniform names (u_time/u_aspect/u_resolution + table uniforms) — render()
        # hardcodes these, so a script on one would silently no-op. Passed in by ProjectSession (NOT
        # imported from core, which pulls in glfw — the headless boundary). Empty in a bare test engine.
        self._engine_driven = engine_driven

    def cached_source(self, document_id: str) -> tuple[str, float] | None:
        """The script text the engine last loaded and the mtime it loaded it at, or None:
        what the editor's intelligence reads on a shader tab when the script tab is closed."""
        scripts = self._documents.get(document_id)
        if scripts is None or scripts.source is None or scripts.mtime is None:
            return None
        return scripts.source, scripts.mtime

    def script_driven_uniforms(self, document_id: str) -> set[tuple[str, str]]:
        # The (pass, name) pairs the script drove on its last tick (decision 10 — only known after a
        # tick). Used by the copilot set_uniform reject + the UI's play/stop button gate (a pair here
        # is script-targeted: playing or stopped).
        document = self._documents.get(document_id)
        return set(document.last_driven) if document is not None else set()

    def script_status(self, document_id: str) -> "ScriptStatus | None":
        # The document script's UI status (feature 042), or None when the document has no script.py. Whether a
        # script is bound, its sentinel compile/run error (drives zero rows when set), the count of real
        # uniforms it drove last tick, and its HOMELESS soft-key errors (typo/orphan keys naming no row).
        document = self._documents.get(document_id)
        if document is None or document.behavior is None:
            return None
        sentinel = self.errors.get((document_id, "", _SCRIPT_FILE))
        soft = [
            (pass_name, name, err)
            for pass_name, name in sorted(document.last_skipped)
            if (err := self.errors.get((document_id, pass_name, name))) is not None
        ]
        return ScriptStatus(
            sentinel_error=sentinel,
            driven_count=len(document.last_driven),
            soft_errors=soft,
        )

    def has_script(self, document_id: str) -> bool:
        # True when the document has a bound script (script.py exists + compiled, error or not).
        document = self._documents.get(document_id)
        return document is not None and document.behavior is not None

    def reload(
        self, document_id: str, scripts_dir: Path, document: ScriptTarget
    ) -> None:
        # Discover + (re)compile the document's `script.py` if its mtime changed (a recompile makes a
        # FRESH instance — state resets on edit), drop it if the file vanished. The script binds by
        # EXISTENCE (048 — no active flag; the file IS the binding). Cheap when nothing changed: a stat.
        scripts = self._documents.get(document_id)
        if scripts is None or scripts.scripts_dir != scripts_dir:
            scripts = DocumentScripts(scripts_dir)
            self._documents[document_id] = scripts

        path = scripts_dir / _SCRIPT_FILE
        if not path.is_file():
            if scripts.behavior is not None:
                self._drop_script(document_id, scripts)
            return

        try:
            mtime = path.stat().st_mtime
            body = path.read_text(encoding="utf-8")
        except (OSError, ValueError):
            # A vanished / half-saved / non-UTF8 file mid-edit — keep the cached script, never raise
            # into the frame loop (ValueError covers UnicodeDecodeError).
            return
        if scripts.mtime == mtime:
            return
        behavior = PythonBehavior(_SCRIPT_FILE, body)
        scripts.behavior = behavior
        scripts.mtime = mtime
        scripts.source = body
        key = (document_id, "", _SCRIPT_FILE)
        if behavior.error is not None:
            self.errors[key] = behavior.error
        else:
            self.errors.pop(key, None)

    def _drop_script(self, document_id: str, scripts: "DocumentScripts") -> None:
        # Tear down a removed script: free its last-good + every per-key error (a coercion-failed key
        # or a bad-key soft error records under (document_id, pass, name), so popping the sentinel
        # isn't enough) and clear the cached sets so script_driven_uniforms reports nothing. The pair
        # is UNPACKED into the three-tuple error key — composing (document_id, pair) would build a
        # key that never matches, leaving every per-key error behind.
        scripts.behavior = None
        scripts.mtime = None
        scripts.source = None
        self.errors.pop((document_id, "", _SCRIPT_FILE), None)
        for pass_name, name in scripts.last_driven | scripts.last_skipped:
            scripts.last_good.pop((pass_name, name), None)
            self.errors.pop((document_id, pass_name, name), None)
        scripts.last_driven = set()
        scripts.last_skipped = set()

    def _binding_reject(
        self,
        pass_name: str,
        name: str,
        active: dict[str, moderngl.Uniform | moderngl.UniformBlock],
    ) -> str | None:
        # Why a script key can't bind to `name` on `pass_name` (None = it can, or it is simply
        # absent — `_binds` separates those two). An engine-owned key (u_time…) is dropped SILENTLY
        # upstream (decision 5), so it never reaches this. Writing the script before the shader
        # declares the uniform is a normal authoring step (079 D5), so an absent name is no error;
        # a name that IS declared but is a sampler/block is one. The pass is named in the message:
        # the same uniform name can be legal on one pass and a sampler on another.
        uniform = active.get(name)
        if uniform is not None and not is_scriptable(uniform):
            return f"pass '{pass_name}': '{name}' is a sampler/block — not a scriptable value"
        return None

    def _binds(
        self,
        pass_name: str,
        name: str,
        active: dict[str, moderngl.Uniform | moderngl.UniformBlock],
    ) -> bool:
        # Whether a write to `name` on `pass_name` would land: declared AND scriptable.
        return self._binding_reject(pass_name, name, active) is None and name in active

    def reset(self, document_id: str) -> None:
        # Re-instantiate the live script (re-run __init__) without recompiling — the manual "restart".
        # Sync the engine's recorded error to the behavior's post-reset state (a recovered __init__
        # clears it; a still-raising one re-records) so a consumer reading `errors` off-tick sees the
        # truth immediately.
        scripts = self._documents.get(document_id)
        if scripts is None or scripts.behavior is None:
            return
        scripts.behavior.reset()
        key = (document_id, "", _SCRIPT_FILE)
        if scripts.behavior.error is not None:
            self.errors[key] = scripts.behavior.error
        else:
            self.errors.pop(key, None)

    def fresh_behavior_for(self, document_id: str) -> PythonBehavior | None:
        # A NEW script instance, independent of the live registry's instance — recompiled from the
        # live registry's CACHED source (not a fresh disk read, so an export never sees a half-saved
        # mid-edit file). The export path ticks THIS so an exported integrator starts from a clean
        # __init__ regardless of live state.
        scripts = self._documents.get(document_id)
        if scripts is None or scripts.source is None:
            return None
        return PythonBehavior(_SCRIPT_FILE, scripts.source)

    def tick(
        self,
        document_id: str,
        document: ScriptTarget,
        ctx: EngineContext,
        stopped: frozenset[StoppedKey] = frozenset(),
    ) -> None:
        # Tick the LIVE script: it routes the returned dict across the document's passes and writes
        # each target pass's uniform_values before Pass.render() reads them. A pair in `stopped` (the
        # user froze it for manual edit, 048) still ticks the script + counts as driven, but its WRITE
        # is skipped so the manual value sticks. A runtime/shape error freezes the uniform at last-good
        # and records a ScriptError; the frame always continues.
        scripts = self._documents.get(document_id)
        if scripts is None or scripts.behavior is None:
            return
        self._tick_script(
            document_id,
            document,
            ctx,
            scripts.behavior,
            scripts.last_good,
            self.errors,
            scripts.last_driven,
            scripts.last_skipped,
            stopped,
        )

    def tick_export(
        self,
        document_id: str,
        document: ScriptTarget,
        ctx: EngineContext,
        behavior: PythonBehavior,
    ) -> None:
        # Tick an EXTERNAL script (the export's fresh instance) against the document. EVERY sink is a
        # per-call throwaway, so an export never touches the live script's recorded error/caches
        # (structurally isolated). NO stopped set — an export always plays the script.
        self._tick_script(
            document_id,
            document,
            ctx,
            behavior,
            {},
            {},
            set(),
            set(),
            frozenset(),
        )

    def dry_run(
        self,
        document_id: str,
        document: ScriptTarget,
        sample_times: tuple[float, ...],
        fps: int,
    ) -> ScriptProbe:
        # Synchronous copilot feedback (043): compile verdict from the ALREADY-LIVE state (the caller
        # reloaded the file at write time — no reload here, which would mutate live state), then an
        # ISOLATED dry-tick. ONE fresh script is stepped CONTINUOUSLY through the export-clock frames so
        # self.* accumulates (an integrator animates correctly); every write lands in a per-call sink,
        # so the live document + live engine state are byte-identical afterward. Returns the driven
        # (pass, name) pairs, per-key + orphan errors, and the driven uniforms' VALUES at each sample.
        compile_error = self.errors.get((document_id, "", _SCRIPT_FILE))
        behavior = self.fresh_behavior_for(document_id)
        if behavior is None or compile_error is not None:
            return ScriptProbe(compile_error, set(), [], [], [])

        # Compile every pass FIRST. 066 D1 forbids compiling from inside the FRAME LOOP; this is a
        # synchronous agent call on the main thread, the same context `_scriptable_uniforms_for`
        # already compiles in. Without it a probe of a document whose passes have never rendered
        # holds every key and hands the agent three false facts at once — an empty driven set (the
        # deliberate "loud no-op"), an orphan naming a uniform the shader does declare, and STATIC
        # from empty samples — so the agent debugs a script that is correct.
        for render_pass in document.passes.values():
            render_pass.get_active_uniforms()

        dt = 1.0 / fps
        max_frame = max((round(t * fps) for t in sample_times), default=0)
        # frame -> the first sample time landing on it; setdefault keeps the earliest so two close
        # times rounding to one frame don't silently drop a sample (the dict-comp would keep the last).
        want: dict[int, float] = {}
        for t in sample_times:
            want.setdefault(round(t * fps), t)
        sink: dict[tuple[str, str], Any] = {}
        errors: dict[tuple[str, str, str], ScriptError] = {}
        driven: set[tuple[str, str]] = set()
        skipped: set[tuple[str, str]] = set()
        samples: list[tuple[float, dict[tuple[str, str], Any]]] = []
        # The probe reports "did this EVER fail across the window", not the final-frame snapshot: the
        # live engine's `errors` dict SELF-HEALS (a good tick pops a key), so a TRANSIENT raise/coercion/
        # orphan that recovers before the last sampled frame would be lost. Accumulate each category
        # right after each tick, before the next one can pop it.
        seen_driven: set[tuple[str, str]] = set()
        seen_skipped: set[tuple[str, str]] = set()
        worst: dict[tuple[str, str, str], ScriptError] = {}
        for frame in range(max_frame + 1):
            ctx = EngineContext(t=frame * dt, dt=dt, frame=frame)
            self._tick_script(
                document_id,
                document,
                ctx,
                behavior,
                {},
                errors,
                driven,
                skipped,
                frozenset(),
                values_sink=sink,
            )
            seen_driven |= driven
            seen_skipped |= skipped
            for key, err in errors.items():
                worst.setdefault(key, err)  # first failure across the window wins
            if frame in want:
                samples.append(
                    (want[frame], {key: sink[key] for key in driven if key in sink})
                )

        # A driven key's shape failure and a skipped key's refusal (a sampler/block, an unknown
        # pass) are both "this key names something and it did not work" — one list. A skipped key
        # with NO error is the 079 D5 state: the shader has yet to declare it.
        per_key = [
            (pass_name, name, err)
            for pass_name, name in sorted(seen_driven | seen_skipped)
            if (err := worst.get((document_id, pass_name, name))) is not None
        ]
        orphan = [
            (pass_name, name)
            for pass_name, name in sorted(seen_skipped)
            if worst.get((document_id, pass_name, name)) is None
        ]
        # A behavior-level error seen at ANY frame = `update` raised / returned a non-dict at some point
        # (the script compiled but CRASHES at runtime). Surface it so the verdict isn't a false ANIMATING
        # off the values a recovered-by-the-last-frame crash leaves in the sink.
        runtime_error = worst.get((document_id, "", _SCRIPT_FILE))
        # The probe is the SINGLE source of truth for the driven set in the headless/copilot path, where
        # no live tick warms DocumentScripts.last_driven. Stash it there so script_driven_uniforms (the
        # working-set marker + the set_uniform reject) agrees with this write's verdict. Safe: last_driven
        # is metadata, the next live tick overwrites it; the byte-identical invariant covers uniform_values.
        scripts = self._documents.get(document_id)
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

    def _active_by_pass(
        self, document: ScriptTarget
    ) -> dict[str, dict[str, moderngl.Uniform | moderngl.UniformBlock]]:
        # One active-uniform map per READY pass, built once per tick. A pass that has never
        # attempted a compile is absent from the map: `get_active_uniforms` would compile it from
        # inside the tick, which 066 D1 forbids. Its keys are then simply not declared anywhere,
        # which since 079 D5 is a silent skip whatever the reason — mid-compile, broken, or a
        # uniform the author has yet to write.
        return {
            pass_name: {u.name: u for u in render_pass.get_active_uniforms()}
            for pass_name, render_pass in document.passes.items()
            if render_pass.script_ready
        }

    def _write_one(
        self,
        *,
        document_id: str,
        pass_name: str,
        name: str,
        value: object,
        render_pass: ScriptPass,
        active: dict[str, moderngl.Uniform | moderngl.UniformBlock],
        errors: dict[tuple[str, str, str], ScriptError],
        last_good: dict[tuple[str, str], Any],
        driven: set[tuple[str, str]],
        stopped: frozenset[StoppedKey],
        values_sink: dict[tuple[str, str], Any] | None,
    ) -> None:
        # Route ONE key at ONE pass: coerce against that pass's own uniform and write, unless the pair
        # is stopped. The caller has already resolved the pass and checked `_binding_reject`.
        key = (document_id, pass_name, name)
        pair = (pass_name, name)
        uniform = active[name]
        assert is_scriptable(
            uniform
        )  # _binding_reject(None) implies a scriptable uniform
        frozen = (
            values_sink.get(
                pair, last_good.get(pair, render_pass.uniform_values.get(name))
            )
            if values_sink is not None
            else last_good.get(pair, render_pass.uniform_values.get(name))
        )
        driven.add(
            pair
        )  # driven BEFORE coerce/write, so a stopped/failed key still counts
        is_stopped = StoppedKey(pass_name=pass_name, name=name) in stopped
        try:
            coerced = coerce_one(value, uniform, name)
        except (
            _RuntimeScriptError
        ) as e:  # per-KEY shape mismatch — freeze ONLY this key
            e.error.pass_name = pass_name
            errors[key] = e.error
            # A STOPPED key keeps the user's manual value (don't clobber it with stale last-good);
            # a playing key freezes at last-good, the freeze-as-data behavior.
            if not is_stopped:
                if values_sink is not None:
                    values_sink[pair] = frozen
                else:
                    render_pass.uniform_values[name] = frozen
            return
        errors.pop(key, None)
        last_good[pair] = coerced
        # Play/stop (048): a STOPPED uniform's value is NOT written (the manual value sticks); the
        # script still ran + the pair still counts as driven (so the row keeps its play/stop button).
        if not is_stopped:
            if values_sink is not None:
                values_sink[pair] = coerced
            else:
                render_pass.uniform_values[name] = coerced

    def _tick_script(
        self,
        document_id: str,
        document: ScriptTarget,
        ctx: EngineContext,
        behavior: PythonBehavior,
        last_good: dict[tuple[str, str], Any],
        errors: dict[tuple[str, str, str], ScriptError],
        last_driven: set[tuple[str, str]],
        last_skipped: set[tuple[str, str]],
        stopped: frozenset[StoppedKey],
        values_sink: dict[tuple[str, str], Any] | None = None,
    ) -> None:
        # `values_sink` (the dry-run path): every uniform-value WRITE lands there, keyed by the (pass,
        # name) pair, + the freeze-fallback READ consults it, so the LIVE document is never written.
        # None = the live tick (write each target pass).
        behavior_key = (document_id, "", _SCRIPT_FILE)
        drove_last = set(last_driven)

        # cached compile error — freeze, recorded at reload
        if behavior.error is not None:
            errors[behavior_key] = behavior.error
            _freeze(drove_last, document, last_good, values_sink)
            return
        try:
            raw = behavior.run(ctx)
        except _RuntimeScriptError as e:  # no instance — authored message
            errors[behavior_key] = e.error
            _freeze(drove_last, document, last_good, values_sink)
            return
        except Exception as e:  # a raw throw is behavior-level
            errors[behavior_key] = ScriptError(
                _SCRIPT_FILE,
                "runtime",
                f"{type(e).__name__}: {e}",
                _user_error_line(behavior.label, e),
            )
            _freeze(drove_last, document, last_good, values_sink)
            return
        if not isinstance(raw, dict):
            errors[behavior_key] = ScriptError(
                _SCRIPT_FILE,
                "runtime",
                f"update must return a dict, got {type(raw).__name__}",
            )
            _freeze(drove_last, document, last_good, values_sink)
            return
        errors.pop(
            behavior_key, None
        )  # the run succeeded; clear a stale behavior-level error

        active_by_pass = self._active_by_pass(document)
        driven: set[tuple[str, str]] = set()
        skipped: set[tuple[str, str]] = set()

        # 069 D3: the VALUE's type dispatches — a dict is a pass block, anything else broadcasts. Not
        # "is the key a pass name", which a pass called u_brush would make ambiguous. Broadcasts run
        # FIRST and blocks SECOND, in two phases rather than one loop, so "specific over general" holds
        # regardless of the author's insertion order (a dict preserves it, and it is not a precedence
        # the author meant to express).
        broadcasts = {k: v for k, v in raw.items() if not isinstance(v, dict)}
        blocks = {k: v for k, v in raw.items() if isinstance(v, dict)}

        for name, value in broadcasts.items():
            # An engine-owned key (u_time…) is SILENTLY dropped (decision 5): the renderer owns that
            # slot and a script can't be expected to avoid naming it.
            if name in self._engine_driven:
                continue
            targets = [
                (pass_name, active)
                for pass_name, active in active_by_pass.items()
                if self._binds(pass_name, name, active)
            ]
            if not targets:
                # No pass declares this key. A normal authoring step (079 D5): the script names a
                # uniform the shader has yet to declare, and the shader side already offers the
                # declaration. Skipped silently, no row — including while a pass is mid-compile or
                # broken, where the pass's own compile error is the thing to read.
                skipped.add(("", name))
                continue
            for pass_name, active in targets:
                self._write_one(
                    document_id=document_id,
                    pass_name=pass_name,
                    name=name,
                    value=value,
                    render_pass=document.passes[pass_name],
                    active=active,
                    errors=errors,
                    last_good=last_good,
                    driven=driven,
                    stopped=stopped,
                    values_sink=values_sink,
                )

        for pass_name, block in blocks.items():
            if pass_name not in document.passes:
                real = ", ".join(sorted(document.passes))
                errors[(document_id, "", pass_name)] = ScriptError(
                    pass_name,
                    "runtime",
                    f"no pass named '{pass_name}' in this document (passes: {real})",
                )
                skipped.add(("", pass_name))
                continue
            active = active_by_pass.get(pass_name)
            if active is None:
                # Never attempted a compile: HELD for this tick, no error, nothing written. The next
                # tick recomputes; the first-render sweep admits one such pass per document per frame.
                continue
            for name, value in block.items():
                if name in self._engine_driven:
                    continue
                reason = self._binding_reject(pass_name, name, active)
                if reason is not None:
                    # A sampler/block key: record a soft error so the strip surfaces it on the
                    # script tab AND on this pass's shader tab, then SKIP with no write. It goes in
                    # `skipped` NOT `driven` (it names no scriptable uniform, so
                    # script_driven_uniforms must not claim ownership).
                    errors[(document_id, pass_name, name)] = ScriptError(
                        name, "runtime", reason, pass_name=pass_name
                    )
                    skipped.add((pass_name, name))
                    continue
                if name not in active:
                    # The pass compiles and does not declare this name: the same normal step as
                    # a broadcast orphan (079 D5), skipped silently.
                    skipped.add((pass_name, name))
                    continue
                self._write_one(
                    document_id=document_id,
                    pass_name=pass_name,
                    name=name,
                    value=value,
                    render_pass=document.passes[pass_name],
                    active=active,
                    errors=errors,
                    last_good=last_good,
                    driven=driven,
                    stopped=stopped,
                    values_sink=values_sink,
                )

        # An omitted-after-failing key's stale error: clear any key TOUCHED last frame (driven OR a
        # bad/skipped key) but NOT touched this frame (decision 8 — no zombie; an omitted real key
        # keeps its last value).
        touched = driven | skipped
        for pass_name, name in (last_driven | last_skipped) - touched:
            errors.pop((document_id, pass_name, name), None)
        last_driven.clear()
        last_driven.update(driven)
        last_skipped.clear()
        last_skipped.update(skipped)

    def drop_document(self, document_id: str) -> None:
        self._documents.pop(document_id, None)
        for key in [k for k in self.errors if k[0] == document_id]:
            self.errors.pop(key, None)
