"""ScriptEngine.dry_run — the synchronous copilot feedback probe (feature 043). Pure, no GL (a
SimpleNamespace stands in for moderngl.Uniform). Covers the make-or-break canaries: live state is
byte-identical after a dry_run (no corruption), an integrator's sampled values ADVANCE across the
sample times (the continuous tick accumulates self.* — the false-STATIC class), a closed-form motion
is captured, and the four facts (compile error with no tick, driven set, per-key coercion error,
orphan key) surface."""

import types
from pathlib import Path

from shaderbox.scripting import ScriptEngine

_GL_FLOAT = 0x1406

_SAMPLE_TIMES = (0.0, 0.5, 1.0)
_FPS = 12


def _u(name: str, dim: int = 1, n: int = 1) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        name=name, dimension=dim, array_length=n, gl_type=_GL_FLOAT, value=0.0
    )


class _FakeNode:
    def __init__(self, uniforms: list[types.SimpleNamespace]) -> None:
        self.uniform_values: dict[str, object] = {}
        self._uniforms = uniforms

    def get_active_uniforms(self) -> list[types.SimpleNamespace]:
        return self._uniforms


def _write_brain(tmp: Path, body: str) -> None:
    scripts_dir = tmp / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    (scripts_dir / "script.py").write_text(body, encoding="utf-8")


def _brain(*, update_body: str, init_body: str = "") -> str:
    head = "class Behavior(ScriptBehavior):\n"
    init = f"    def __init__(self) -> None:\n{init_body}" if init_body else ""
    return f"{head}{init}    def update(self, ctx: Ctx) -> dict:\n{update_body}"


def _engine(tmp: Path, node: _FakeNode) -> ScriptEngine:
    eng = ScriptEngine()
    eng.reload("n0", tmp / "scripts", node)
    return eng


def test_dry_run_does_not_corrupt_live_state(tmp_path: Path) -> None:
    # The no-corruption canary: a dry_run ticks an isolated brain; the live node + live engine state
    # must be byte-identical afterward. Falsifier: any of them changes -> the sink leaked.
    _write_brain(
        tmp_path,
        _brain(update_body="        return {'u_x': 0.5 + 0.3 * ctx.t}\n"),
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, types.SimpleNamespace(t=0.0, dt=0.0, frame=0, mouse=None))  # type: ignore[arg-type]

    live_values = dict(node.uniform_values)
    live_driven = eng.script_driven_uniforms("n0")
    live_errors = dict(eng.errors)

    eng.dry_run("n0", node, _SAMPLE_TIMES, _FPS)

    assert node.uniform_values == live_values  # live node untouched
    assert eng.script_driven_uniforms("n0") == live_driven
    assert dict(eng.errors) == live_errors


def test_dry_run_integrator_advances_across_samples(tmp_path: Path) -> None:
    # THE make-or-break canary: an integrator (self.* accumulates) sampled by dry_run must show its
    # value ADVANCING across t. Falsifier: identical samples -> the probe did N independent single
    # ticks instead of one continuous tick (the figure-8-drift false-STATIC).
    _write_brain(
        tmp_path,
        _brain(
            init_body="        self.v = 0.0\n",
            update_body="        self.v += ctx.dt\n        return {'u_x': self.v}\n",
        ),
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)

    probe = eng.dry_run("n0", node, _SAMPLE_TIMES, _FPS)

    assert probe.compile_error is None
    assert probe.driven == {"u_x"}
    vals = [s[1]["u_x"] for s in probe.samples]
    assert len(vals) == 3
    assert vals[0] < vals[1] < vals[2]  # accumulating, not frozen
    # Each frame (incl. frame 0) ticks dt=1/12, matching the export loop: at the t=1.0 sample (frame
    # 12) the integrator has summed 13 steps. The point is monotone advance, not the exact endpoint.
    assert vals[2] > vals[0] + 0.5


def test_dry_run_closed_form_motion_captured(tmp_path: Path) -> None:
    _write_brain(
        tmp_path,
        _brain(update_body="        return {'u_x': ctx.t, 'u_c': 0.7}\n"),
    )
    node = _FakeNode([_u("u_x"), _u("u_c")])
    eng = _engine(tmp_path, node)

    probe = eng.dry_run("n0", node, _SAMPLE_TIMES, _FPS)

    assert probe.driven == {"u_x", "u_c"}
    moved = [s[1]["u_x"] for s in probe.samples]
    held = [s[1]["u_c"] for s in probe.samples]
    assert moved[0] < moved[2]  # u_x varies with t
    assert held[0] == held[2] == 0.7  # u_c constant


def test_dry_run_compile_error_no_tick(tmp_path: Path) -> None:
    # A syntax error: dry_run returns the live compile verdict with NO tick (driven empty, no samples).
    _write_brain(
        tmp_path, "class Behavior(ScriptBehavior)\n    pass\n"
    )  # missing colon
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)

    probe = eng.dry_run("n0", node, _SAMPLE_TIMES, _FPS)

    assert probe.compile_error is not None
    assert probe.compile_error.kind == "compile"
    assert probe.driven == set()
    assert probe.samples == []


def test_dry_run_orphan_and_per_key_errors(tmp_path: Path) -> None:
    # u_typo names no active uniform (orphan); u_v is a vec2 the script drives with a bare float
    # (per-key coercion error). u_x is fine.
    _write_brain(
        tmp_path,
        _brain(
            update_body=("        return {'u_x': 0.5, 'u_typo': 1.0, 'u_v': 0.3}\n")
        ),
    )
    node = _FakeNode([_u("u_x"), _u("u_v", dim=2)])
    eng = _engine(tmp_path, node)

    probe = eng.dry_run("n0", node, _SAMPLE_TIMES, _FPS)

    assert "u_x" in probe.driven and "u_v" in probe.driven
    assert any(name == "u_v" for name, _ in probe.per_key_errors)  # bad shape
    assert any(name == "u_typo" for name, _ in probe.orphan_keys)  # no such uniform


def test_dry_run_empty_dict_drives_nothing(tmp_path: Path) -> None:
    _write_brain(tmp_path, _brain(update_body="        return {}\n"))
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)

    probe = eng.dry_run("n0", node, _SAMPLE_TIMES, _FPS)

    assert probe.compile_error is None
    assert probe.driven == set()  # the loud no-op fact source
