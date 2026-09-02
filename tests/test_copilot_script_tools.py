"""read_script / write_script handler messages (feature 043), over the fake caps. The engine probe
is exercised in test_script_dry_run.py; here the tool LAYER — that each ScriptWriteResult shape
becomes the right agent-facing fact (compile error, the loud drives-0 no-op, the motion verdict, the
orphan/per-key lines), and that read_script surfaces the stub."""

import types
from typing import Any

from shaderbox.copilot.backend import CopilotBackend
from shaderbox.copilot.capabilities import ScriptView, ScriptWriteResult
from shaderbox.copilot.tools.registry import ToolRegistry, build_registry
from tests._caps import minimal_caps


def _registry(**overrides: Any) -> ToolRegistry:
    return build_registry(minimal_caps(**overrides))


def test_read_script_stub_is_flagged() -> None:
    reg = _registry(
        read_script=lambda _document: ScriptView(
            "f90f", "Wave", "1  # stub\n2  return {}", [], is_stub=True
        )
    )
    ok, msg, payload = reg.execute("read_script", {"document": ""})
    assert ok is True
    assert "no script yet" in msg and "STUB" in msg
    assert payload == {"document": "f90f", "is_stub": True}


def test_read_script_of_a_scripted_document_does_not_repeat_the_listing() -> None:
    # The source rides the working set (mirroring read_shader) — returning it here too billed the
    # same bytes twice on every script read.
    reg = _registry(
        read_script=lambda _document: ScriptView(
            "f90f", "Wave", "1  class Behavior(ScriptBehavior):", [], is_stub=False
        )
    )
    ok, msg, _payload = reg.execute("read_script", {"document": ""})
    assert ok is True
    assert "class Behavior" not in msg  # the listing stays in the working set
    assert "working set" in msg and "compiles clean" in msg


def test_read_script_no_document_is_error() -> None:
    reg = _registry(
        read_script=lambda _document: ScriptView(
            "",
            "",
            "",
            [type("E", (), {"path": "", "line": 0, "message": "no document found"})()],
            is_stub=False,
        )
    )
    ok, msg, _ = reg.execute("read_script", {"document": "bad"})
    assert ok is False
    assert "no document found" in msg


def test_write_script_compile_error() -> None:
    reg = _registry(
        write_script=lambda _t, _document: ScriptWriteResult(
            ok=True, compile_error="script.py:3: SyntaxError: invalid syntax"
        )
    )
    ok, msg, payload = reg.execute("write_script", {"new_text": "broken"})
    assert ok is True  # the tool ran; the script just doesn't compile
    assert "compiled with errors" in msg
    assert "script.py:3: SyntaxError" in msg
    # A broken edit is NOT clean: the errors payload is what the engine's applies-but-broken
    # counter reads, and its LIST shape is what the card renders as "1 compile error".
    assert payload == {"errors": ["script.py:3: SyntaxError: invalid syntax"]}


def test_force_restore_is_not_an_applied_with_errors_result() -> None:
    # The restore is a SUCCESSFUL write of the last clean source — an errors payload here would
    # count the recovery itself as thrash and re-arm the very loop it just broke.
    reg = _registry(
        write_script=lambda _t, _document: ScriptWriteResult(
            ok=True,
            restored_note="SCRIPT RESTORED -- 6 broken script edits in a row, so the script "
            "was reverted to its last clean-running state.",
        )
    )
    ok, msg, payload = reg.execute("write_script", {"new_text": "broken again"})
    assert ok is True
    assert "SCRIPT RESTORED" in msg and "compiled with errors" not in msg
    assert payload is None


def test_write_script_drives_nothing_is_loud() -> None:
    reg = _registry(
        write_script=lambda _t, _document: ScriptWriteResult(
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
        write_script=lambda _t, _document: ScriptWriteResult(
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
        write_script=lambda _t, _document: ScriptWriteResult(
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
    assert "declare it in the SHADER first" in msg  # the orphan steer


def test_write_script_runtime_error_is_surfaced() -> None:
    reg = _registry(
        write_script=lambda _t, _document: ScriptWriteResult(
            ok=True, compile_error="ran, then script.py:5: ValueError: boom"
        )
    )
    ok, msg, _ = reg.execute("write_script", {"new_text": "..."})
    assert ok is True
    assert "ran, then" in msg and "ValueError: boom" in msg


def test_edit_script_routes_through_apply_and_formats_like_write() -> None:
    # edit_script produces the SAME ScriptWriteResult shape as write_script -> identical agent message.
    reg = _registry(
        apply_script_edit=lambda _o, _n, _r, _document: ScriptWriteResult(
            ok=True,
            driven=["u_radius"],
            motion_facts="-> u_radius CHANGE across t (ANIMATING)",
        )
    )
    ok, msg, payload = reg.execute("edit_script", {"old_str": "0.3", "new_str": "0.5"})
    assert ok is True
    assert "drives u_radius" in msg and "ANIMATING" in msg
    assert payload == {"driven": ["u_radius"]}


def test_edit_script_not_found_is_error() -> None:
    reg = _registry(
        apply_script_edit=lambda _o, _n, _r, _document: ScriptWriteResult(
            ok=False, error="old_str not found in the script -- re-read it"
        )
    )
    ok, msg, _ = reg.execute("edit_script", {"old_str": "nope", "new_str": "x"})
    assert ok is False
    assert "old_str not found" in msg


def test_edit_script_compile_error_surfaces() -> None:
    # An edit that breaks the script returns the compile error the same as write_script.
    reg = _registry(
        apply_script_edit=lambda _o, _n, _r, _document: ScriptWriteResult(
            ok=True, compile_error="script.py:4: SyntaxError: invalid syntax"
        )
    )
    ok, msg, _ = reg.execute("edit_script", {"old_str": "a", "new_str": "b("})
    assert ok is True
    assert "compiled with errors" in msg and "SyntaxError" in msg


def test_write_script_unresolved_document_is_error() -> None:
    reg = _registry(
        write_script=lambda _t, _document: ScriptWriteResult(
            ok=False, error="no document found for 'bad'"
        )
    )
    ok, msg, _ = reg.execute("write_script", {"new_text": "...", "document": "bad"})
    assert ok is False
    assert "no document found" in msg


def _fake_pass(name: str, uniforms: list[Any]) -> Any:
    # The six attributes _pass_views + _format_uniforms read. A SimpleNamespace, NOT the GL-backed
    # two-pass helper in test_document_graph.py: that one takes a moderngl.Context and skips without
    # a standalone one, which would put this file's headline assertion behind a GL skip.
    return types.SimpleNamespace(
        source=types.SimpleNamespace(text=f"// {name}\n"),
        compile_unit=types.SimpleNamespace(errors=[]),
        get_active_uniforms=lambda: uniforms,
        uniform_values={u.name: 0.25 for u in uniforms},
    )


def _uniform(name: str) -> Any:
    return types.SimpleNamespace(
        name=name, dimension=1, array_length=1, gl_type=0x1406, value=0.0
    )


def test_pass_views_marks_driven_only_on_the_pass_that_declares_it() -> None:
    # The site 069 W-G genuinely fixes: _pass_views loops every pass against ONE document-scoped
    # driven set, so before the pass filter a uniform driven on `paint` was marked
    # `<driven by script.py>` on every pass declaring that NAME. Falsifier: keep the document-scoped
    # set (drop the per-pass filter) — the composite assertion goes red.
    u = _uniform("u_wave")
    document = types.SimpleNamespace(
        passes={
            "paint": _fake_pass("paint", [u]),
            "composite": _fake_pass("composite", [u]),
        },
        graph=types.SimpleNamespace(passes={}, output="composite"),
        # `_pass_views` reads the effective graph (069 D9), which a real Document derives from
        # its compiled programs; this stub has no GL, so it hands back an empty wiring.
        effective_graph=lambda: types.SimpleNamespace(passes={}),
    )
    stub = types.SimpleNamespace(
        _get_script_driven_uniforms=lambda _id: {("paint", "u_wave")},
    )
    pass_views = CopilotBackend._pass_views.__get__(stub)

    views = {v.name: v for v in pass_views("n0", {}, document)}

    assert views["paint"].uniforms == ["u_wave float = <driven by script.py>"]
    assert views["composite"].uniforms == ["u_wave float = 0.25"]
