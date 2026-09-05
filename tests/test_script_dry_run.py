"""ScriptEngine.dry_run — the synchronous copilot feedback probe (feature 043). Pure, no GL (a
SimpleNamespace stands in for moderngl.Uniform). Covers the make-or-break canaries: live state is
byte-identical after a dry_run (no corruption), an integrator's sampled values ADVANCE across the
sample times (the continuous tick accumulates self.* — the false-STATIC class), a closed-form motion
is captured, and the four facts (compile error with no tick, driven set, per-key coercion error,
orphan key) surface."""

import types
from pathlib import Path

from loguru import logger

from shaderbox.scripting import ScriptEngine

_GL_FLOAT = 0x1406
_GL_SAMPLER_2D = 0x8B5E

_SAMPLE_TIMES = (0.0, 0.5, 1.0)
_FPS = 12


def _u(
    name: str, dim: int = 1, n: int = 1, gl_type: int = _GL_FLOAT
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        name=name, dimension=dim, array_length=n, gl_type=gl_type, value=0.0
    )


class _FakePass:
    # Mirrors the real `Pass` contract the engine leans on: `get_active_uniforms` COMPILES a
    # never-attempted pass, and that attempt is what makes it `script_ready`. A stand-in that
    # reported readiness independently of the call would let a probe pass while the real lazy
    # compile did nothing.
    def __init__(self, uniforms: list[types.SimpleNamespace]) -> None:
        self.uniform_values: dict[str, object] = {}
        self.script_ready = True
        self._uniforms = uniforms

    def get_active_uniforms(self) -> list[types.SimpleNamespace]:
        self.script_ready = True
        return self._uniforms


class _FakeDocument:
    # The ScriptTarget slice: passes by name. A bare uniform list is the one-pass shorthand.
    def __init__(
        self,
        uniforms: list[types.SimpleNamespace] | dict[str, list[types.SimpleNamespace]],
    ) -> None:
        by_pass = {"main": uniforms} if isinstance(uniforms, list) else uniforms
        self.passes: dict[str, _FakePass] = {
            name: _FakePass(us) for name, us in by_pass.items()
        }

    @property
    def uniform_values(self) -> dict[str, object]:
        return self.passes["main"].uniform_values


def _write_script(tmp: Path, body: str) -> None:
    scripts_dir = tmp / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    (scripts_dir / "script.py").write_text(body, encoding="utf-8")


def _script(*, update_body: str, init_body: str = "") -> str:
    head = "class Behavior(ScriptBehavior):\n"
    init = f"    def __init__(self) -> None:\n{init_body}" if init_body else ""
    return f"{head}{init}    def update(self, context: ScriptContext) -> dict:\n{update_body}"


def _engine(tmp: Path, document: _FakeDocument) -> ScriptEngine:
    eng = ScriptEngine()
    eng.reload("n0", tmp / "scripts", document)
    return eng


def test_dry_run_does_not_corrupt_live_state(tmp_path: Path) -> None:
    # The no-corruption canary: a dry_run ticks an isolated script; the live document + live engine state
    # must be byte-identical afterward. Falsifier: any of them changes -> the sink leaked.
    _write_script(
        tmp_path,
        _script(update_body="        return {'u_x': 0.5 + 0.3 * context.t}\n"),
    )
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, types.SimpleNamespace(t=0.0, dt=0.0, frame=0, mouse=None))

    live_values = dict(document.uniform_values)
    live_driven = eng.script_driven_uniforms("n0")
    live_errors = dict(eng.errors)

    eng.dry_run("n0", document, _SAMPLE_TIMES, _FPS)

    assert document.uniform_values == live_values  # live document untouched
    assert eng.script_driven_uniforms("n0") == live_driven
    assert dict(eng.errors) == live_errors


def test_dry_run_integrator_advances_across_samples(tmp_path: Path) -> None:
    # THE make-or-break canary: an integrator (self.* accumulates) sampled by dry_run must show its
    # value ADVANCING across t. Falsifier: identical samples -> the probe did N independent single
    # ticks instead of one continuous tick (the figure-8-drift false-STATIC).
    _write_script(
        tmp_path,
        _script(
            init_body="        self.v = 0.0\n",
            update_body="        self.v += context.dt\n        return {'u_x': self.v}\n",
        ),
    )
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)

    probe = eng.dry_run("n0", document, _SAMPLE_TIMES, _FPS)

    assert probe.compile_error is None
    assert probe.driven == {("main", "u_x")}
    vals = [s[1][("main", "u_x")] for s in probe.samples]
    assert len(vals) == 3
    assert vals[0] < vals[1] < vals[2]  # accumulating, not frozen
    # Each frame (incl. frame 0) ticks dt=1/12, matching the export loop: at the t=1.0 sample (frame
    # 12) the integrator has summed 13 steps. The point is monotone advance, not the exact endpoint.
    assert vals[2] > vals[0] + 0.5


def test_dry_run_closed_form_motion_captured(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        _script(update_body="        return {'u_x': context.t, 'u_c': 0.7}\n"),
    )
    document = _FakeDocument([_u("u_x"), _u("u_c")])
    eng = _engine(tmp_path, document)

    probe = eng.dry_run("n0", document, _SAMPLE_TIMES, _FPS)

    assert probe.driven == {("main", "u_x"), ("main", "u_c")}
    moved = [s[1][("main", "u_x")] for s in probe.samples]
    held = [s[1][("main", "u_c")] for s in probe.samples]
    assert moved[0] < moved[2]  # u_x varies with t
    assert held[0] == held[2] == 0.7  # u_c constant


def test_dry_run_compile_error_no_tick(tmp_path: Path) -> None:
    # A syntax error: dry_run returns the live compile verdict with NO tick (driven empty, no samples).
    _write_script(
        tmp_path, "class Behavior(ScriptBehavior)\n    pass\n"
    )  # missing colon
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)

    probe = eng.dry_run("n0", document, _SAMPLE_TIMES, _FPS)

    assert probe.compile_error is not None
    assert probe.compile_error.kind == "compile"
    assert probe.driven == set()
    assert probe.samples == []


def test_dry_run_orphan_and_per_key_errors(tmp_path: Path) -> None:
    # u_typo names no active uniform (orphan); u_v is a vec2 the script drives with a bare float
    # (per-key coercion error). u_x is fine.
    _write_script(
        tmp_path,
        _script(
            update_body=("        return {'u_x': 0.5, 'u_typo': 1.0, 'u_v': 0.3}\n")
        ),
    )
    document = _FakeDocument([_u("u_x"), _u("u_v", dim=2)])
    eng = _engine(tmp_path, document)

    # The probe surfaces orphans via the RETURN value, never the console: it ticks the script across
    # ~N frames, so a console warning would spam once per frame (the pong-script regression). 069
    # deleted the warning outright, so the sink must stay empty for every caller, not just this one.
    records: list[str] = []
    handle = logger.add(lambda m: records.append(str(m)), level="WARNING")
    try:
        probe = eng.dry_run("n0", document, _SAMPLE_TIMES, _FPS)
    finally:
        logger.remove(handle)

    assert ("main", "u_x") in probe.driven and ("main", "u_v") in probe.driven
    assert any(
        (p, name) == ("main", "u_v") for p, name, _ in probe.per_key_errors
    )  # bad shape
    assert ("", "u_typo") in probe.orphan_keys  # no such uniform, and no error (079 D5)
    assert not [r for r in records if "shaderbox.scripting.engine" in r]


def test_dry_run_runtime_raise_surfaces(tmp_path: Path) -> None:
    # A script that COMPILES but `update` raises at a later frame (an integrator blow-up): the probe
    # must carry the runtime_error, NOT report a false ANIMATING off the crash-frozen values. Falsifier:
    # runtime_error is None -> the crash is swallowed and the verdict lies.
    _write_script(
        tmp_path,
        _script(
            init_body="        self.n = 0\n",
            update_body=(
                "        self.n += 1\n"
                "        if self.n >= 3:\n"
                "            raise ValueError('boom')\n"
                "        return {'u_x': float(self.n)}\n"
            ),
        ),
    )
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)

    probe = eng.dry_run("n0", document, _SAMPLE_TIMES, _FPS)

    assert probe.compile_error is None  # it DID compile
    assert probe.runtime_error is not None  # but it crashes at runtime
    assert "ValueError" in probe.runtime_error.message


def test_dry_run_transient_runtime_raise_surfaces(tmp_path: Path) -> None:
    # A TRANSIENT crash: update raises only at frame 3, then recovers. The live errors dict self-heals
    # (a good tick pops the key), so a final-frame snapshot would SWALLOW it. The probe must accumulate
    # "did it EVER fail across the window". Falsifier: runtime_error is None -> the transient is lost.
    _write_script(
        tmp_path,
        _script(
            init_body="        self.n = 0\n",
            update_body=(
                "        self.n += 1\n"
                "        if self.n == 3:\n"
                "            raise ValueError('blip')\n"
                "        return {'u_x': float(self.n)}\n"
            ),
        ),
    )
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)

    probe = eng.dry_run("n0", document, _SAMPLE_TIMES, _FPS)

    assert probe.runtime_error is not None  # the transient raise was NOT swallowed
    assert "ValueError" in probe.runtime_error.message


def test_dry_run_transient_per_key_error_surfaces(tmp_path: Path) -> None:
    # u_v (a vec2) gets a bad bare-float ONLY at frame 3, then a valid vec2 after. The per-key error
    # must survive to the probe (accumulated), not be popped by the recovering tick.
    _write_script(
        tmp_path,
        _script(
            init_body="        self.n = 0\n",
            update_body=(
                "        self.n += 1\n"
                "        v = 0.5 if self.n == 3 else (0.1, 0.2)\n"
                "        return {'u_v': v}\n"
            ),
        ),
    )
    document = _FakeDocument([_u("u_v", dim=2)])
    eng = _engine(tmp_path, document)

    probe = eng.dry_run("n0", document, _SAMPLE_TIMES, _FPS)

    assert any(
        (p, name) == ("main", "u_v") for p, name, _ in probe.per_key_errors
    )  # transient bad shape kept


def test_dry_run_colliding_sample_times_keeps_earliest(tmp_path: Path) -> None:
    # Two sample times rounding to the SAME frame must not drop a sample; setdefault keeps the
    # earliest. Falsifier: the first sample's t is 0.04 (a dict-comp would keep the last).
    _write_script(tmp_path, _script(update_body="        return {'u_x': context.t}\n"))
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)

    probe = eng.dry_run("n0", document, (0.0, 0.04, 0.08), 12)  # frames [0, 0, 1]

    assert probe.samples[0][0] == 0.0  # earliest of the colliding pair, not 0.04


def test_dry_run_empty_dict_drives_nothing(tmp_path: Path) -> None:
    _write_script(tmp_path, _script(update_body="        return {}\n"))
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)

    probe = eng.dry_run("n0", document, _SAMPLE_TIMES, _FPS)

    assert probe.compile_error is None
    assert probe.driven == set()  # the loud no-op fact source


def test_dry_run_reports_the_pass_in_orphan_keys(tmp_path: Path) -> None:
    # The copilot's write feedback is the only place a headless caller learns the ROUTING verdict, so
    # an orphan inside a pass block must name that pass. Falsifier: report the bare name — the pass
    # half of the assertion goes red.
    _write_script(
        tmp_path,
        _script(
            update_body=("        return {'paint': {'u_x': 0.5, 'u_typo': 1.0}}\n")
        ),
    )
    document = _FakeDocument({"paint": [_u("u_x")], "composite": [_u("u_x")]})
    eng = _engine(tmp_path, document)

    probe = eng.dry_run("n0", document, _SAMPLE_TIMES, _FPS)

    assert probe.driven == {("paint", "u_x")}  # the block drove paint ALONE
    assert probe.orphan_keys == [("paint", "u_typo")]


def test_a_cold_dry_run_compiles_and_reports_the_real_driven_set(
    tmp_path: Path,
) -> None:
    # `write_script_source` reloads then probes with no render in between, so on a document whose
    # passes have never rendered the probe used to hand the agent three false facts at once: an empty
    # driven set (the deliberate "loud no-op"), an orphan naming a uniform the shader DOES declare,
    # and STATIC from empty samples. `dry_run` is a synchronous agent call, not the frame loop, so it
    # compiles first (066 D1 constrains the frame loop only). Falsifier: drop that compile — driven
    # goes empty, orphan_keys grows the row, and every sample dict is empty.
    _write_script(tmp_path, _script(update_body="        return {'u_x': context.t}\n"))
    document = _FakeDocument({"seed": [_u("u_x")], "out": [_u("u_x")]})
    for render_pass in document.passes.values():
        render_pass.script_ready = False  # never rendered
    eng = _engine(tmp_path, document)

    probe = eng.dry_run("n0", document, _SAMPLE_TIMES, _FPS)

    assert probe.driven == {("seed", "u_x"), ("out", "u_x")}
    assert probe.orphan_keys == []
    assert all(s[1] for s in probe.samples)  # non-empty, so the motion verdict is real
    moved = [s[1][("seed", "u_x")] for s in probe.samples]
    assert moved[0] != moved[-1]


def test_a_refused_key_is_a_per_key_error_and_an_undeclared_one_is_not(
    tmp_path: Path,
) -> None:
    # 079 D5 splits the two states the probe used to report as one list. A key naming a SAMPLER is
    # refused and stays an error the agent must fix; a key naming nothing yet is a fact, no error.
    # Falsifier: report the sampler under orphan_keys again — it would reach the agent as "declare
    # it in the shader first", advice that cannot work for a sampler.
    _write_script(
        tmp_path,
        _script(
            update_body="        return {'u_x': 0.5, 'main': {'u_tex': 1.0, 'u_soon': 2.0}}\n"
        ),
    )
    document = _FakeDocument([_u("u_x"), _u("u_tex", gl_type=_GL_SAMPLER_2D)])
    eng = _engine(tmp_path, document)

    probe = eng.dry_run("n0", document, _SAMPLE_TIMES, _FPS)

    assert [(p, name) for p, name, _ in probe.per_key_errors] == [("main", "u_tex")]
    assert probe.orphan_keys == [("main", "u_soon")]
