"""Edit-result message surfaces (dogfood forensic fixes). Failures NAME the resolved
target (an empty target silently meaning "current node" was a giveup cause); lib reads
stop claiming a clean compile; a node edit whose compile errors live in a spliced lib
is labelled cross-file instead of getting a misleading brace hint. GL-free — the tool
layer is driven through the registry over the _caps fakes.
"""

import types
from pathlib import Path
from typing import Any

import pytest

from shaderbox.copilot.backend import (
    CopilotBackend,
    _CopilotEditTarget,
    _cross_file_note,
    _edit_error_hints,
)
from shaderbox.copilot.capabilities import CompileErrorInfo, EditResult, ShaderView
from shaderbox.copilot.tools.registry import ToolRegistry, build_registry
from shaderbox.copilot.tools.shader import _applied_result
from tests._caps import minimal_caps

_LABEL = "node 'Text Rendering' (f90f)"
_EMPTY_LABEL = _LABEL + " — target was empty, so this hit the CURRENT node"


def _registry(**overrides: Any) -> ToolRegistry:
    return build_registry(minimal_caps(**overrides))


# ---- target naming in failure / success messages ----


def test_edit_not_found_names_resolved_target() -> None:
    reg = _registry(
        apply_shader_edit=lambda _o, _n, _r, _t: EditResult(
            matches=0, errors=[], target_label=_EMPTY_LABEL
        )
    )
    ok, msg, _ = reg.execute("edit_shader", {"old_str": "x", "new_str": "y"})
    assert ok is False
    assert "old_str not found in node 'Text Rendering' (f90f)" in msg
    assert "target was empty, so this hit the CURRENT node" in msg


def test_line_check_failure_names_checked_target() -> None:
    reg = _registry(
        apply_line_edit=lambda _s, _e, _t, _tg, _f, _l: EditResult(
            matches=0,
            errors=[],
            unresolved=True,
            unresolved_reason="line check failed — nothing was applied: ...",
            target_label=_LABEL,
        )
    )
    ok, msg, _ = reg.execute(
        "replace_lines",
        {
            "start_line": 1,
            "end_line": 2,
            "first_line": "a",
            "last_line": "b",
            "new_text": "x",
        },
    )
    assert ok is False
    assert f"(checked {_LABEL})" in msg


def test_out_of_range_names_target() -> None:
    reg = _registry(
        apply_line_edit=lambda _s, _e, _t, _tg, _f, _l: EditResult(
            matches=0, errors=[], target_label="lib:draw/neon_ring.glsl"
        )
    )
    ok, msg, _ = reg.execute("insert_after", {"line": 99, "new_text": "x"})
    assert ok is False
    assert "out of range in lib:draw/neon_ring.glsl" in msg


def test_applied_result_success_names_target() -> None:
    ok, msg, _ = _applied_result(EditResult(matches=1, errors=[], target_label=_LABEL))
    assert ok is True
    assert f"ok — compiled clean ({_LABEL})" in msg


# ---- the backend's label resolution ----


def _resolve_stub() -> types.SimpleNamespace:
    node = types.SimpleNamespace(source=types.SimpleNamespace(text="void main() {}"))
    ui_node = types.SimpleNamespace(
        ui_state=types.SimpleNamespace(ui_name="Text Rendering"), node=node
    )
    nodes = {"f90f1111": ui_node}
    stub = types.SimpleNamespace(
        _get_ui_nodes=lambda: nodes, _get_current_node_id=lambda: "f90f1111"
    )
    stub._copilot_resolve_node_id = CopilotBackend._copilot_resolve_node_id.__get__(
        stub
    )
    stub._copilot_short_ids = CopilotBackend._copilot_short_ids.__get__(stub)
    return stub


def test_resolve_target_label_marks_empty_target_as_current() -> None:
    resolve = CopilotBackend._copilot_resolve_target.__get__(_resolve_stub())
    tgt = resolve("", allow_create=False)
    assert isinstance(tgt, _CopilotEditTarget)
    assert tgt.label == _EMPTY_LABEL


def test_resolve_target_label_plain_for_explicit_target() -> None:
    resolve = CopilotBackend._copilot_resolve_target.__get__(_resolve_stub())
    tgt = resolve("f90f", allow_create=False)
    assert isinstance(tgt, _CopilotEditTarget)
    assert tgt.label == _LABEL


# ---- cross-file error labelling ----


def test_cross_file_note_names_the_lib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "lib"
    monkeypatch.setattr("shaderbox.copilot.backend.shader_lib_root", lambda: root)
    edited = tmp_path / "proj" / "shader.frag.glsl"
    err = CompileErrorInfo(
        path=str(root / "draw" / "neon_ring.glsl"), line=3, message="boom"
    )
    note = _cross_file_note(edited, [err])
    assert "lib:draw/neon_ring.glsl" in note
    assert "the file you edited may be fine" in note


def test_cross_file_note_silent_when_any_error_is_local(tmp_path: Path) -> None:
    edited = tmp_path / "shader.frag.glsl"
    own = CompileErrorInfo(path=str(edited), line=1, message="boom")
    foreign = CompileErrorInfo(path=str(tmp_path / "x.glsl"), line=1, message="boom")
    assert _cross_file_note(edited, [own]) == ""
    assert _cross_file_note(edited, [foreign, own]) == ""
    assert _cross_file_note(edited, []) == ""


def test_edit_error_hints_suppress_brace_hint_on_foreign_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "lib"
    monkeypatch.setattr("shaderbox.copilot.backend.shader_lib_root", lambda: root)
    edited = tmp_path / "proj" / "shader.frag.glsl"
    err = CompileErrorInfo(path=str(root / "a.glsl"), line=1, message="syntax error")
    hints = _edit_error_hints(edited, "void main() {", [err])
    assert hints and "the error is in lib:a.glsl" in hints[0]
    assert not any("'{'" in h for h in hints)


def test_edit_error_hints_keep_brace_hint_for_local_errors(tmp_path: Path) -> None:
    edited = tmp_path / "shader.frag.glsl"
    err = CompileErrorInfo(path=str(edited), line=1, message="syntax error")
    hints = _edit_error_hints(edited, "void main() {", [err])
    assert any("'{'" in h for h in hints)


# ---- honest lib-read summaries ----


def _lib_view(address: str) -> ShaderView:
    return ShaderView(
        node_id=address, name=address, listing="1  // x", uniforms=[], errors=[]
    )


def _node_view() -> ShaderView:
    return ShaderView(
        node_id="ab12", name="Wave", listing="1  void main() {}", uniforms=[], errors=[]
    )


def test_read_shader_lib_only_is_not_compiled_clean() -> None:
    reg = _registry(read_shaders=lambda _ids: [_lib_view("lib:noise.glsl")])
    ok, msg, payload = reg.execute("read_shader", {"nodes": ["lib:noise.glsl"]})
    assert ok is True
    assert "compiled clean" not in msg
    assert "no standalone compile" in msg
    assert payload is not None and "no standalone compile" in payload["display"]


def test_read_shader_mixed_separates_node_state_from_lib_note() -> None:
    reg = _registry(
        read_shaders=lambda _ids: [_node_view(), _lib_view("lib:noise.glsl")]
    )
    ok, msg, _ = reg.execute("read_shader", {"nodes": ["ab12", "lib:noise.glsl"]})
    assert ok is True
    assert "all compiled clean." in msg
    assert "(lib:noise.glsl: library file — no standalone compile)" in msg
