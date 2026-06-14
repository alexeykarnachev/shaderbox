"""The per-uniform row glyph state (feature 045 fix wave, re-keyed for 047): the row must reflect the
NODE-BRAIN driving a uniform (not just the uniform's own per-uniform script) and a script ERROR, else
the row shows a misleading `absent` glyph over a script-owned value. Active/inactive is now a MODEL
FLAG (047) on the UIUniform / UINodeState, and the per-uniform file is `u_<name>__<tag>.py`. Pure: the
ProjectSession state methods compose `self.script_engine` facts + disk presence + the flags, so they
bind onto a light stub (the `__get__` idiom) with a tmp scripts dir + a fake node — no GL, no full
session."""

import types
from pathlib import Path
from typing import Any

from shaderbox.project_session import ProjectSession
from shaderbox.scripting import ScriptError
from shaderbox.ui_models import UIUniform
from shaderbox.util import get_uniform_hash

_GL_FLOAT = 0x1406


def _u(name: str) -> types.SimpleNamespace:
    # A scalar-float uniform stand-in (tag "float") — enough for get_uniform_hash + is_scriptable +
    # uniform_tag (the state methods never read a GL value here). `size` lets get_uniform_hash's
    # non-moderngl.Uniform branch hash it; the same object hashes identically at both call sites.
    return types.SimpleNamespace(
        name=name, dimension=1, array_length=1, gl_type=_GL_FLOAT, size=1
    )


def _stub(
    tmp: Path,
    *,
    driven: set[str],
    errors: dict[tuple[str, str], ScriptError] | None = None,
    uniforms: tuple[str, ...] = ("u_x", "u_color"),
    active: set[str] = frozenset(),
    brain_active: bool = False,
) -> Any:
    # `active` is the set of uniform names whose per-uniform script flag is True (047). `uniforms` are
    # the node's live scriptable uniform names. A node-brain flag is `brain_active`.
    scripts_dir = tmp / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    engine = types.SimpleNamespace(
        errors=errors or {},
        script_driven_uniforms=lambda node_id: driven,
    )
    paths = types.SimpleNamespace(scripts_dir_for=lambda node_id: scripts_dir)
    live = [_u(n) for n in uniforms]
    ui_uniforms = {
        get_uniform_hash(u): UIUniform(name=u.name, is_script_active=u.name in active)
        for u in live
    }
    ui_state = types.SimpleNamespace(
        ui_uniforms=ui_uniforms, is_brain_active=brain_active
    )
    node = types.SimpleNamespace(get_active_uniforms=lambda: live)
    ui_node = types.SimpleNamespace(node=node, ui_state=ui_state)
    stub = types.SimpleNamespace(
        script_engine=engine, paths=paths, ui_nodes={"n0": ui_node}
    )
    for meth in (
        "_ui_uniform_for",
        "_script_filename",
        "_is_script_active",
        "script_path_for",
        "script_state_for",
        "uniform_pill_state",
        "is_uniform_script_owned",
    ):
        setattr(stub, meth, getattr(ProjectSession, meth).__get__(stub))
    return stub


def _write_script(tmp: Path, name: str) -> None:
    # Write a per-uniform script under the 047 tagged scheme (`u_<name>__float.py` for the scalar _u).
    scripts_dir = tmp / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    (scripts_dir / f"{name}__float.py").write_text("x = 1\n", encoding="utf-8")


def test_brain_driven_uniform_reads_active_not_absent(tmp_path: Path) -> None:
    # The 045 regression: a brain-driven uniform with NO own file must read 'active' (script owns the
    # value), NOT 'absent' (which drew an absent glyph over a live-fighting widget).
    stub = _stub(tmp_path, driven={"u_color"})
    assert stub.uniform_pill_state("n0", "u_color") == "active"
    assert stub.is_uniform_script_owned("n0", "u_color") is True


def test_brain_driven_uniform_with_error_reads_error(tmp_path: Path) -> None:
    err = ScriptError("u_color", "runtime", "boom")
    stub = _stub(tmp_path, driven={"u_color"}, errors={("n0", "u_color"): err})
    assert stub.uniform_pill_state("n0", "u_color") == "error"
    assert (
        stub.is_uniform_script_owned("n0", "u_color") is True
    )  # broken still owns the slot


def test_own_file_wins_over_brain(tmp_path: Path) -> None:
    # A uniform's own active script wins over the brain (the 044 conflict rule) — its own state shows.
    _write_script(tmp_path, "u_x")
    stub = _stub(tmp_path, driven={"u_x"}, active={"u_x"})
    assert stub.uniform_pill_state("n0", "u_x") == "active"


def test_own_inactive_file_no_brain_reads_inactive_manual(tmp_path: Path) -> None:
    # An own file with the flag OFF and no brain reads 'inactive' + the widget is manual.
    _write_script(tmp_path, "u_x")
    stub = _stub(tmp_path, driven=set(), active=set())
    assert stub.uniform_pill_state("n0", "u_x") == "inactive"
    assert (
        stub.is_uniform_script_owned("n0", "u_x") is False
    )  # inactive, nothing drives -> manual


def test_inactive_own_override_shadowing_brain_stays_locked(tmp_path: Path) -> None:
    # An inactive own override does NOT bind, so a brain that drives the name owns the value. The glyph
    # shows 'inactive' (the user sees their off override) but the widget stays LOCKED (brain owns it).
    _write_script(tmp_path, "u_x")
    stub = _stub(tmp_path, driven={"u_x"}, active=set())
    assert stub.uniform_pill_state("n0", "u_x") == "inactive"
    assert stub.is_uniform_script_owned("n0", "u_x") is True  # brain owns the slot


def test_own_file_error_reads_error(tmp_path: Path) -> None:
    _write_script(tmp_path, "u_x")
    err = ScriptError("u_x", "compile", "syntax", 2)
    stub = _stub(tmp_path, driven=set(), errors={("n0", "u_x"): err}, active={"u_x"})
    assert stub.uniform_pill_state("n0", "u_x") == "error"


def test_undriven_no_file_reads_absent(tmp_path: Path) -> None:
    stub = _stub(tmp_path, driven=set())
    assert stub.uniform_pill_state("n0", "u_x") == "absent"
    assert stub.is_uniform_script_owned("n0", "u_x") is False


def test_brain_sentinel_error_reads_error(tmp_path: Path) -> None:
    # The node-brain glyph (name=None): a sentinel compile error reads 'error', not 'active'.
    err = ScriptError("script.py", "compile", "syntax", 1)
    stub = _stub(
        tmp_path,
        driven=set(),
        errors={("n0", "script.py"): err},
        brain_active=True,
    )
    (tmp_path / "scripts" / "script.py").write_text("x = 1\n", encoding="utf-8")
    assert stub.script_state_for("n0", None) == "error"
