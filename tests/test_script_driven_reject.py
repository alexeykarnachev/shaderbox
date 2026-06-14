"""The copilot set_uniform must REJECT a script-driven uniform (feature 040, decision 5): a
behavior script computes its value each frame, so a set would be silently overwritten next tick.
Mirrors the engine-driven reject. Pure: the reject branch fires before any GL, so the real
set_uniform method is bound onto a light stub (the test_content_editing __get__ idiom) with a
synchronous bridge; no GL context needed."""

import types
from typing import Any

from shaderbox.copilot.backend import CopilotBackend


class _SyncBridge:
    # run_on_main runs the op inline on the calling thread (the production bridge marshals it to
    # the main thread + blocks); the reject path returns before touching GL, so inline is faithful.
    def run_on_main(
        self, fn: Any, timeout: float | None = None, defer: bool = False
    ) -> Any:
        return fn()


def _stub(script_driven: set[str]) -> Any:
    ui_nodes = {
        "n0": object()
    }  # the node only needs to EXIST; the reject returns before .node
    stub = types.SimpleNamespace(
        _bridge=_SyncBridge(),
        _get_ui_nodes=lambda: ui_nodes,
        _get_current_node_id=lambda: "n0",
        _get_script_driven_uniforms=lambda node_id: script_driven,
    )
    stub._copilot_resolve_node_id = CopilotBackend._copilot_resolve_node_id.__get__(
        stub
    )
    return stub


def test_set_uniform_rejects_script_driven() -> None:
    # One script per node (048): a driven uniform's reject always points at scripts/script.py.
    stub = _stub({"u_wave"})
    set_uniform = CopilotBackend.set_uniform.__get__(stub)
    result = set_uniform("u_wave", 0.5, "n0")
    assert not result.ok
    assert "script-driven" in result.error
    assert "scripts/script.py" in result.error


def test_set_uniform_does_not_reject_a_non_script_uniform() -> None:
    # A name absent from the script-driven set passes the reject branch and reaches the normal
    # path; with no matching active uniform it returns the ordinary "no active uniform" error,
    # NOT the script-driven one — proving the reject is scoped to the script set.
    node = types.SimpleNamespace(
        node=types.SimpleNamespace(get_active_uniforms=lambda: [])
    )
    stub = _stub({"u_wave"})  # u_wave is script-driven; u_x is not
    stub._get_ui_nodes = lambda: {"n0": node}
    set_uniform = CopilotBackend.set_uniform.__get__(stub)
    result = set_uniform("u_x", 0.5, "n0")
    assert "script-driven" not in (result.error or "")
    assert "no active uniform" in (result.error or "")
