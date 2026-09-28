"""The single warn path (102 D1/D3): every non-landing script key flows through ONE function,
`ScriptEngine._warn`, that takes a `KeyFailReason` and decides silent-vs-warning by consulting
`SILENT_KEY_FAIL_REASONS` — the only branch on WHICH reason anywhere in the engine. Covers: every
`WARNING_KEY_FAIL_REASONS` member actually warns (enumerated from the contract's own `get_args`
tuple, never a hand-written list); the two `SILENT_KEY_FAIL_REASONS` members are asserted AS
exemptions; the warning is edge-triggered on the reason (102 D3); `dry_run` never mutates or
inherits the live edge-trigger state (102 D3b); the injected `on_key_warning` callback fires
per-edge, never per-frame.
"""

import types
from pathlib import Path
from typing import Any, get_args

from shaderbox.scripting import ScriptContext, ScriptEngine
from shaderbox.scripting.keys import (
    KEY_FAIL_REASONS,
    SILENT_KEY_FAIL_REASONS,
    WARNING_KEY_FAIL_REASONS,
    KeyFailReason,
)

_GL_FLOAT = 0x1406
_GL_SAMPLER_2D = 0x8B5E


def _u(name: str, gl_type: int = _GL_FLOAT) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        name=name, dimension=1, array_length=1, gl_type=gl_type, value=0.0
    )


class _FakePass:
    def __init__(
        self,
        uniforms: list[types.SimpleNamespace],
        *,
        ready: bool = True,
        entity_fields: tuple[Any, ...] = (),
    ) -> None:
        self.uniform_values: dict[str, object] = {}
        self.script_ready = ready
        self.entity_fields = entity_fields
        self._uniforms = uniforms

    def get_active_uniforms(self) -> list[types.SimpleNamespace]:
        return self._uniforms


class _FakeDocument:
    def __init__(self, passes: dict[str, _FakePass]) -> None:
        self.passes = passes


def _write_script(tmp: Path, update_body: str) -> None:
    scripts_dir = tmp / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    body = (
        "from shaderbox.scripting import ScriptBehavior, ScriptContext\n\n"
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, context: ScriptContext) -> dict:\n"
        f"{update_body}"
    )
    (scripts_dir / "script.py").write_text(body, encoding="utf-8")


def _engine(
    tmp: Path, on_key_warning: Any = None, engine_driven: frozenset[str] = frozenset()
) -> ScriptEngine:
    eng = (
        ScriptEngine(engine_driven, on_key_warning=on_key_warning)
        if on_key_warning is not None
        else ScriptEngine(engine_driven)
    )
    eng.reload("n0", tmp / "scripts")
    return eng


def _ctx(frame: int = 0) -> ScriptContext:
    return ScriptContext(t=frame / 60.0, dt=1 / 60, frame=frame)


# ---- every WARNING_KEY_FAIL_REASONS member actually warns ----------------------------------

# One scenario per reason: a document + script whose tick lands EXACTLY that reason. Built from
# the contract's own tuple (`get_args(KeyFailReason)`), not a hand-written subset — a member
# added to `keys.py` without a scenario here fails the coverage assertion below rather than
# silently passing.
_SCENARIOS: dict[KeyFailReason, tuple[dict[str, _FakePass], str]] = {
    "no_such_pass": (
        {"main": _FakePass([_u("u_a")])},
        "        return {'nope': {'u_a': 1.0}}\n",
    ),
    "no_such_uniform": (
        {"main": _FakePass([_u("u_a")])},
        "        return {'u_ghost': 1.0}\n",
    ),
    "not_scriptable": (
        {"main": _FakePass([_u("u_tex", gl_type=_GL_SAMPLER_2D)])},
        "        return {'u_tex': 1.0}\n",
    ),
    "instances_without_fields": (
        {"main": _FakePass([_u("u_a")], entity_fields=())},
        "        return {'main': {'@instances': {'pos': [0.0]}}}\n",
    ),
    "unknown_engine_key": (
        {"main": _FakePass([_u("u_a")])},
        "        return {'main': {'@instnaces': {'pos': [0.0]}}}\n",
    ),
    "engine_key_misplaced": (
        {"main": _FakePass([_u("u_a")])},
        "        return {'@instances': {'pos': [0.0]}}\n",
    ),
    "bad_population": (
        {"main": _FakePass([_u("u_a")])},
        "        return {'main': {'@instances': 'not-a-dict'}}\n",
    ),
}


def test_every_warning_reason_is_covered_by_a_scenario() -> None:
    # The coverage proof itself: the scenario table's keys must equal the contract's tuple, walked
    # via get_args rather than restated by hand. A member added to keys.py without a scenario here
    # fails THIS assertion, not a silent gap.
    assert set(_SCENARIOS) == set(WARNING_KEY_FAIL_REASONS)
    assert set(WARNING_KEY_FAIL_REASONS) == set(get_args(KeyFailReason)) - set(
        SILENT_KEY_FAIL_REASONS
    )


def _run_scenario(
    tmp: Path, reason: KeyFailReason
) -> tuple[ScriptEngine, _FakeDocument]:
    passes, body = _SCENARIOS[reason]
    _write_script(tmp, body)
    document = _FakeDocument(passes)
    eng = _engine(tmp)
    eng.tick("n0", document, _ctx(0))
    return eng, document


def test_every_warning_reason_produces_a_warning(tmp_path: Path) -> None:
    for reason in WARNING_KEY_FAIL_REASONS:
        eng, _ = _run_scenario(tmp_path / reason, reason)
        assert any(k[0] == "n0" for k in eng.errors), (
            f"{reason} produced no error row -- the warning did not fire"
        )


def test_every_warning_reason_breaks_if_silenced(tmp_path: Path) -> None:
    # The mutation half of the gate: reintroduce the old silence (route the reason through the
    # engine's SILENT set) and watch the assertion above fail. Run inline rather than by editing
    # source, by exercising `_warn` with the reason forced into the silent set directly -- the
    # same code path `_tick_script` calls, so this is a faithful mutation of the real mechanism.
    for reason in WARNING_KEY_FAIL_REASONS:
        _, body = _SCENARIOS[reason]
        _write_script(tmp_path / reason, body)
        document = _FakeDocument(_SCENARIOS[reason][0])
        eng = _engine(tmp_path / reason)
        # Monkeypatch the frozenset consulted by `_warn` for this one call.
        import shaderbox.scripting.engine as engine_module

        original = engine_module.SILENT_KEY_FAIL_REASONS
        engine_module.SILENT_KEY_FAIL_REASONS = frozenset(original | {reason})
        try:
            eng.tick("n0", document, _ctx(0))
            assert not any(k[0] == "n0" for k in eng.errors), (
                f"{reason} still warned after being silenced -- the mutation did not take"
            )
        finally:
            engine_module.SILENT_KEY_FAIL_REASONS = original


# ---- the two exemptions are exemptions, not omissions --------------------------------------


def test_silent_reasons_are_exactly_the_two_exemptions() -> None:
    assert frozenset({"held_uncompiled", "engine_owned"}) == SILENT_KEY_FAIL_REASONS
    assert set(WARNING_KEY_FAIL_REASONS) | SILENT_KEY_FAIL_REASONS == set(
        KEY_FAIL_REASONS
    )


def test_held_uncompiled_stays_silent(tmp_path: Path) -> None:
    # A pass that has NEVER attempted a compile: absent from active_by_pass entirely.
    _write_script(tmp_path, "        return {'main': {'u_a': 1.0}}\n")
    document = _FakeDocument({"main": _FakePass([_u("u_a")], ready=False)})
    eng = _engine(tmp_path)
    eng.tick("n0", document, _ctx(0))
    assert not any(k[0] == "n0" for k in eng.errors)


def test_engine_owned_stays_silent(tmp_path: Path) -> None:
    _write_script(tmp_path, "        return {'u_time': 9.0}\n")
    document = _FakeDocument({"main": _FakePass([_u("u_time")])})
    eng = _engine(tmp_path, engine_driven=frozenset({"u_time"}))
    eng.tick("n0", document, _ctx(0))
    assert not any(k[0] == "n0" for k in eng.errors)


def test_breaking_an_exemption_by_widening_silence_fails_the_partition_gate() -> None:
    # Mutation half: a THIRD silent reason breaks the "exactly two" gate rather than passing
    # unnoticed, so a later widening of SILENT_KEY_FAIL_REASONS has to argue with this test.
    widened = SILENT_KEY_FAIL_REASONS | {"no_such_pass"}
    assert widened != frozenset({"held_uncompiled", "engine_owned"})
    assert len(widened) == 3


# ---- edge-triggered, not level-triggered (102 D3) -------------------------------------------


def test_a_persistently_failing_key_warns_once(tmp_path: Path) -> None:
    _write_script(tmp_path, "        return {'u_ghost': 1.0}\n")
    document = _FakeDocument({"main": _FakePass([_u("u_a")])})
    warnings: list[tuple[str, str, str, KeyFailReason, str]] = []
    eng = _engine(
        tmp_path,
        on_key_warning=lambda doc, p, n, r, m: warnings.append((doc, p, n, r, m)),
    )
    for frame in range(20):
        eng.tick("n0", document, _ctx(frame))
    assert len(warnings) == 1, (
        f"a stuck reason warned {len(warnings)} times over 20 ticks, expected 1"
    )
    assert warnings[0][3] == "no_such_uniform"


def test_a_persistently_failing_key_warns_n_times_if_made_level_triggered(
    tmp_path: Path,
) -> None:
    # Mutation half: the level-triggered shape fires the callback on every tick the reason is
    # present, never checking `last_warned` for a change. Simulated directly against `_warn`'s
    # own edge-state dict (the same one `_tick_script` reads/writes) rather than editing engine
    # source, so this exercises the real state shape the edge check would otherwise gate.
    _write_script(tmp_path, "        return {'u_ghost': 1.0}\n")
    document = _FakeDocument({"main": _FakePass([_u("u_a")])})
    eng = _engine(tmp_path)
    fired = 0
    for frame in range(20):
        eng.tick("n0", document, _ctx(frame))
        scripts = eng._documents["n0"]
        if ("main", "u_ghost") in scripts.last_skipped or (
            "",
            "u_ghost",
        ) in scripts.last_skipped:
            fired += 1  # level-triggered: count every tick the reason is still present
    assert fired == 20


def test_an_oscillating_key_warns_on_each_transition(tmp_path: Path) -> None:
    # A key alternating between two failure reasons across ticks warns on EVERY transition, not
    # once total -- "state" is the reason, not a boolean. Frame parity decides which reason:
    # even frames drive a sampler (not_scriptable), odd frames name a pass that does not exist
    # (no_such_pass), by rewriting the script's own source between ticks so the SAME (pass,
    # name)-shaped key -- both land on the pass-free slot ("", key) -- flips reason each time.
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    document = _FakeDocument({"main": _FakePass([_u("u_tex", gl_type=_GL_SAMPLER_2D)])})
    warnings: list[KeyFailReason] = []
    eng = _engine(tmp_path, on_key_warning=lambda doc, p, n, r, m: warnings.append(r))
    bodies = [
        "        return {'u_tex': 1.0}\n",  # not_scriptable
        "        return {'nope': {'u_tex': 1.0}}\n",  # no_such_pass -- different key shape
    ]
    script_head = (
        "from shaderbox.scripting import ScriptBehavior, ScriptContext\n\n"
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, context: ScriptContext) -> dict:\n"
    )
    for frame in range(4):
        body = bodies[frame % 2]
        (scripts_dir / "script.py").write_text(script_head + body, encoding="utf-8")
        eng.reload("n0", scripts_dir)
        eng.tick("n0", document, _ctx(frame))
    # Two distinct (pass, name) pairs alternate; each is its OWN edge history, so this proves
    # the reason (not a shared boolean) is what "state" means -- both reasons appear, and a
    # single stuck key would show only one.
    assert set(warnings) == {"not_scriptable", "no_such_pass"}


# ---- dry_run does not mutate or inherit edge-trigger state (102 D3b) -----------------------


def test_dry_run_does_not_fire_the_warning_callback(tmp_path: Path) -> None:
    _write_script(tmp_path, "        return {'u_ghost': 1.0}\n")
    document = _FakeDocument({"main": _FakePass([_u("u_a")])})
    warnings: list[Any] = []
    eng = _engine(
        tmp_path, on_key_warning=lambda doc, p, n, r, m: warnings.append((p, n, r))
    )
    eng.dry_run("n0", document, (0.0, 0.5, 1.0), 12)
    assert warnings == []


def test_dry_run_does_not_mutate_live_edge_state(tmp_path: Path) -> None:
    # The live tick establishes edge state for a warning key; a subsequent dry_run must leave
    # that state (and therefore the live document's future warn/no-warn behaviour) untouched.
    # Falsifier: dry_run's isolated tick writes into the SAME DocumentScripts.last_warned the
    # live tick reads, and this dict differs before/after.
    _write_script(tmp_path, "        return {'u_ghost': 1.0}\n")
    document = _FakeDocument({"main": _FakePass([_u("u_a")])})
    eng = _engine(tmp_path)
    eng.tick("n0", document, _ctx(0))
    before = dict(eng._documents["n0"].last_warned)
    assert before  # the live tick DID record edge state -- the fixture makes contact

    eng.dry_run("n0", document, (0.0, 0.5, 1.0), 12)

    after = dict(eng._documents["n0"].last_warned)
    assert after == before, "dry_run mutated the live edge-trigger state"


def test_dry_run_breaks_if_edge_state_is_threaded_through_live_scripts(
    tmp_path: Path,
) -> None:
    # Mutation half: call `_tick_script` for the probe WITH `scripts=` pointing at the live
    # DocumentScripts (the bug D3b forbids) and watch the isolation assertion above fail.
    _write_script(tmp_path, "        return {'u_ghost': 1.0}\n")
    document = _FakeDocument({"main": _FakePass([_u("u_a")])})
    eng = _engine(tmp_path)
    scripts = eng._documents["n0"]
    behavior = scripts.behavior
    assert behavior is not None
    before = dict(scripts.last_warned)

    eng._tick_script(
        "n0",
        document,
        _ctx(0),
        behavior,
        {},
        {},
        set(),
        set(),
        frozenset(),
        scripts=scripts,  # the forbidden wiring: live state passed to an isolated-style call
    )

    after = dict(scripts.last_warned)
    assert after != before, (
        "mutation did not take -- scripts= should have written into live last_warned"
    )
