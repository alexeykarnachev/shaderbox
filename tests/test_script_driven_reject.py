"""The copilot set_uniform must REJECT a script-driven uniform (feature 040, decision 5): a
behavior script computes its value each frame, so a set would be silently overwritten next tick.
Mirrors the engine-driven reject. Pure: the reject branch touches no GL, so the real set_uniform
method is bound onto a light stub (the test_content_editing __get__ idiom) with a synchronous
bridge; no GL context needed.

Since 069 the driven set is (pass, name) pairs and the gate asks the OUTPUT pass's pair — so the
stub carries a real `render_pass.source.path`, which is where the output pass's NAME comes from.
"""

import types
from pathlib import Path
from typing import Any

from shaderbox.copilot.backend import CopilotBackend


class _SyncBridge:
    # run_on_main runs the op inline on the calling thread (the production bridge marshals it to
    # the main thread + blocks); the reject path returns before touching GL, so inline is faithful.
    def run_on_main(
        self, fn: Any, timeout: float | None = None, defer: bool = False
    ) -> Any:
        return fn()


def _document(uniforms: list[Any]) -> Any:
    # The slice set_uniform reads: the output pass's active uniforms + its source path (the pass's
    # NAME, which the driven-pair check needs). "main" is the output pass here.
    return types.SimpleNamespace(
        document=types.SimpleNamespace(
            render_pass=types.SimpleNamespace(
                get_active_uniforms=lambda: uniforms,
                source=types.SimpleNamespace(path=Path("passes/main.frag.glsl")),
            )
        )
    )


def _stub(
    script_driven: set[tuple[str, str]], uniforms: list[Any] | None = None
) -> Any:
    ui_documents = {"n0": _document(uniforms if uniforms is not None else [])}
    stub = types.SimpleNamespace(
        _bridge=_SyncBridge(),
        _get_ui_documents=lambda: ui_documents,
        _get_current_document_id=lambda: "n0",
        _get_script_driven_uniforms=lambda document_id: script_driven,
    )
    stub._copilot_resolve_document_id = (
        CopilotBackend._copilot_resolve_document_id.__get__(stub)
    )
    return stub


def _uniform(name: str) -> Any:
    return types.SimpleNamespace(
        name=name, dimension=1, array_length=1, gl_type=0x1406, value=0.0
    )


def test_set_uniform_rejects_script_driven() -> None:
    # One script per document (048): a driven uniform's reject points at the script edit TOOLS, never
    # at the file's path (059 D2 — the agent gets handles, not implementation detail).
    stub = _stub({("main", "u_wave")}, [_uniform("u_wave")])
    set_uniform = CopilotBackend.set_uniform.__get__(stub)
    result = set_uniform("u_wave", 0.5, "n0")
    assert not result.ok
    assert "script-driven" in result.error
    assert "edit_script/write_script" in result.error
    assert "documents/" not in result.error


def test_set_uniform_does_not_reject_a_non_script_uniform() -> None:
    # A name absent from the script-driven set passes the reject branch and reaches the normal
    # path; with no matching active uniform it returns the ordinary "no active uniform" error,
    # NOT the script-driven one — proving the reject is scoped to the script set.
    stub = _stub({("main", "u_wave")})  # u_wave is script-driven; u_x is not
    set_uniform = CopilotBackend.set_uniform.__get__(stub)
    result = set_uniform("u_x", 0.5, "n0")
    assert "script-driven" not in (result.error or "")
    assert "no active uniform" in (result.error or "")


def test_set_uniform_does_not_reject_a_uniform_driven_on_another_pass() -> None:
    # The tool addresses the OUTPUT pass, so a uniform of the same NAME driven on a sibling pass is
    # a legitimate manual edit here (069). Falsifier: test the name against the whole document's
    # pairs — this goes red with the script-driven reject.
    stub = _stub({("paint", "u_wave")}, [_uniform("u_wave")])
    set_uniform = CopilotBackend.set_uniform.__get__(stub)
    result = set_uniform("u_wave", 0.5, "n0")
    assert "script-driven" not in (result.error or "")
