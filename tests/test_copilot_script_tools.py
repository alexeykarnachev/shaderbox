"""read_script / write_script handler messages (feature 043), over the fake caps. The engine probe
is exercised in test_script_dry_run.py; here the tool LAYER — that each ScriptWriteResult shape
becomes the right agent-facing fact (compile error, the loud drives-0 no-op, the motion verdict, the
orphan/per-key lines), and that read_script surfaces the stub."""

from typing import Any

from shaderbox.copilot.capabilities import ScriptView, ScriptWriteResult
from shaderbox.copilot.tools.registry import ToolRegistry, build_registry
from tests._caps import minimal_caps


def _registry(**overrides: Any) -> ToolRegistry:
    return build_registry(minimal_caps(**overrides))


def test_read_script_stub_is_flagged() -> None:
    reg = _registry(
        read_script=lambda _node: ScriptView(
            "f90f", "Wave", "1  # stub\n2  return {}", [], is_stub=True
        )
    )
    ok, msg, payload = reg.execute("read_script", {"node": ""})
    assert ok is True
    assert "no brain yet" in msg and "STUB" in msg
    assert payload == {"node": "f90f", "is_stub": True}


def test_read_script_no_node_is_error() -> None:
    reg = _registry(
        read_script=lambda _node: ScriptView(
            "",
            "",
            "",
            [type("E", (), {"path": "", "line": 0, "message": "no node found"})()],
            is_stub=False,
        )
    )
    ok, msg, _ = reg.execute("read_script", {"node": "bad"})
    assert ok is False
    assert "no node found" in msg


def test_write_script_compile_error() -> None:
    reg = _registry(
        write_script=lambda _t, _node: ScriptWriteResult(
            ok=True, compile_error="script.py:3: SyntaxError: invalid syntax"
        )
    )
    ok, msg, _ = reg.execute("write_script", {"new_text": "broken"})
    assert ok is True  # the tool ran; the script just doesn't compile
    assert "compiled with errors" in msg
    assert "script.py:3: SyntaxError" in msg


def test_write_script_drives_nothing_is_loud() -> None:
    reg = _registry(
        write_script=lambda _t, _node: ScriptWriteResult(
            ok=True,
            driven=[],
            motion_facts="drives 0 uniforms (update returned an empty dict / only "
            "orphan keys). Nothing animates and every uniform stays manual.",
        )
    )
    ok, msg, _ = reg.execute("write_script", {"new_text": "return {}"})
    assert ok is True
    assert "drives 0 uniforms" in msg and "Nothing animates" in msg


def test_write_script_animating_verdict() -> None:
    reg = _registry(
        write_script=lambda _t, _node: ScriptWriteResult(
            ok=True,
            driven=["u_center", "u_radius"],
            motion_facts="values@t=0.0: u_center=(0.3,0.5) u_radius=0.2\n"
            "-> u_center, u_radius CHANGE across t (ANIMATING)",
        )
    )
    ok, msg, payload = reg.execute("write_script", {"new_text": "..."})
    assert ok is True
    assert "drives u_center, u_radius" in msg
    assert "ANIMATING" in msg
    assert payload == {"driven": ["u_center", "u_radius"]}


def test_write_script_surfaces_orphan_and_per_key() -> None:
    reg = _registry(
        write_script=lambda _t, _node: ScriptWriteResult(
            ok=True,
            driven=["u_x"],
            per_key_errors=["u_v: expected a vec2, got a float"],
            orphan_keys=["u_typo: no active uniform"],
            motion_facts="-> u_x CHANGE across t (ANIMATING)",
        )
    )
    ok, msg, _ = reg.execute("write_script", {"new_text": "..."})
    assert ok is True
    assert "u_v: expected a vec2" in msg
    assert "u_typo" in msg


def test_write_script_unresolved_node_is_error() -> None:
    reg = _registry(
        write_script=lambda _t, _node: ScriptWriteResult(
            ok=False, error="no node found for 'bad'"
        )
    )
    ok, msg, _ = reg.execute("write_script", {"new_text": "...", "node": "bad"})
    assert ok is False
    assert "no node found" in msg
