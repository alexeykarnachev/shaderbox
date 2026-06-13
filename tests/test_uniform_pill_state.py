"""The per-uniform row pill state (feature 045 fix wave): the row must reflect the NODE-BRAIN
driving a uniform (not just the uniform's own `u_<name>.py`) and a script ERROR, else the row shows
a misleading `+ script` pill over a script-owned value + an editable widget that fights the tick.
Pure: the three ProjectSession methods compose `self.script_engine` facts + disk presence, so they
bind onto a light stub (the test_script_driven_reject `__get__` idiom) with a tmp scripts dir — no
GL, no full session."""

import types
from pathlib import Path
from typing import Any

from shaderbox.project_session import ProjectSession
from shaderbox.scripting import ScriptError


def _stub(
    tmp: Path,
    *,
    driven: set[str],
    errors: dict[tuple[str, str], ScriptError] | None = None,
) -> Any:
    scripts_dir = tmp / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    engine = types.SimpleNamespace(
        errors=errors or {},
        script_driven_uniforms=lambda node_id: driven,
    )
    paths = types.SimpleNamespace(scripts_dir_for=lambda node_id: scripts_dir)
    stub = types.SimpleNamespace(script_engine=engine, paths=paths)
    for meth in ("script_path_for", "script_state_for", "uniform_pill_state", "is_uniform_script_owned"):
        setattr(stub, meth, getattr(ProjectSession, meth).__get__(stub))
    return stub


def _write_script(tmp: Path, name: str, *, disabled: bool = False) -> None:
    scripts_dir = tmp / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    path = scripts_dir / f"{name}.py"
    path.write_text("x = 1\n", encoding="utf-8")
    if disabled:
        path.with_name(path.name + ".disabled").write_text("", encoding="utf-8")


def test_brain_driven_uniform_reads_active_not_absent(tmp_path: Path) -> None:
    # The 045 regression: a brain-driven uniform with NO own file must read 'active' (script owns the
    # value), NOT 'absent' (which drew a '+ script' pill over a live-fighting widget).
    stub = _stub(tmp_path, driven={"u_color"})
    assert stub.uniform_pill_state("n0", "u_color") == "active"
    assert stub.is_uniform_script_owned("n0", "u_color") is True


def test_brain_driven_uniform_with_error_reads_error(tmp_path: Path) -> None:
    err = ScriptError("u_color", "runtime", "boom")
    stub = _stub(tmp_path, driven={"u_color"}, errors={("n0", "u_color"): err})
    assert stub.uniform_pill_state("n0", "u_color") == "error"
    assert stub.is_uniform_script_owned("n0", "u_color") is True  # broken still owns the slot


def test_own_file_wins_over_brain(tmp_path: Path) -> None:
    # A uniform's own `u_x.py` wins over the brain (the 044 conflict rule) — its own state shows.
    _write_script(tmp_path, "u_x")
    stub = _stub(tmp_path, driven={"u_x"})
    assert stub.uniform_pill_state("n0", "u_x") == "active"


def test_own_inactive_file_no_brain_reads_inactive_manual(tmp_path: Path) -> None:
    # An explicitly-disabled own file with no brain reads 'inactive' + the widget is manual.
    _write_script(tmp_path, "u_x", disabled=True)
    stub = _stub(tmp_path, driven=set())
    assert stub.uniform_pill_state("n0", "u_x") == "inactive"
    assert stub.is_uniform_script_owned("n0", "u_x") is False  # disabled, nothing drives -> manual


def test_inactive_own_override_shadowing_brain_stays_locked(tmp_path: Path) -> None:
    # A disabled own override does NOT bind, so a brain that drives the name owns the value. The pill
    # shows 'inactive' (the user sees their off override) but the widget stays LOCKED (brain owns it).
    _write_script(tmp_path, "u_x", disabled=True)
    stub = _stub(tmp_path, driven={"u_x"})
    assert stub.uniform_pill_state("n0", "u_x") == "inactive"
    assert stub.is_uniform_script_owned("n0", "u_x") is True  # brain owns the slot


def test_own_file_error_reads_error(tmp_path: Path) -> None:
    _write_script(tmp_path, "u_x")
    err = ScriptError("u_x", "compile", "syntax", 2)
    stub = _stub(tmp_path, driven=set(), errors={("n0", "u_x"): err})
    assert stub.uniform_pill_state("n0", "u_x") == "error"


def test_undriven_no_file_reads_absent(tmp_path: Path) -> None:
    stub = _stub(tmp_path, driven=set())
    assert stub.uniform_pill_state("n0", "u_x") == "absent"
    assert stub.is_uniform_script_owned("n0", "u_x") is False


def test_brain_sentinel_error_reads_error(tmp_path: Path) -> None:
    # The node-brain pill (name=None): a sentinel compile error reads 'error', not 'active'.
    err = ScriptError("script.py", "compile", "syntax", 1)
    stub = _stub(tmp_path, driven=set(), errors={("n0", "script.py"): err})
    (tmp_path / "scripts" / "script.py").write_text("x = 1\n", encoding="utf-8")
    assert stub.script_state_for("n0", None) == "error"
