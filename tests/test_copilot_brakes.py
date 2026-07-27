"""Feature 056 slice B: the edit brakes cover the SCRIPT tools too. The streak key is a
(kind, target) tuple (a node's GLSL and its script.py are two files), a clean whole-file write
resets its OWN kind, a broken script edit counts as broken, and one write_script per node per
batch. Deterministic — scripted fake client + a stubbed backend, no GL."""

import threading
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

from shaderbox.copilot.agent import (
    AgentError,
    AgentTurnDone,
    _edit_target_key,
    run_turn,
)
from shaderbox.copilot.backend import CopilotBackend
from shaderbox.copilot.capabilities import EditResult, ScriptWriteResult
from shaderbox.copilot.config import COPILOT_CONFIG, CopilotConfig
from shaderbox.copilot.gate import GateChannel
from shaderbox.copilot.llm.api import LLMDone, LLMStreamEvent, LLMTextDelta
from shaderbox.copilot.tools.registry import build_registry
from shaderbox.copilot.trace import TraceLog
from tests._caps import minimal_caps
from tests.test_copilot_loop import _fake_context, _FakeClient, _tool_call

_EDIT_SHADER = _tool_call(
    "cs", "edit_shader", '{"old_str": "a", "new_str": "b", "target": "7f3a"}'
)
_EDIT_SCRIPT = _tool_call(
    "cp", "edit_script", '{"old_str": "a", "new_str": "b", "node": "7f3a"}'
)
_WRITE_SCRIPT = _tool_call("cw", "write_script", '{"new_text": "x", "node": "7f3a"}')
_DONE = [LLMTextDelta("done"), LLMDone("stop")]


class _KindTrace(TraceLog):
    def __init__(self) -> None:
        super().__init__(Path())
        self.kinds: list[str] = []

    def event(self, kind: str, **fields: Any) -> None:
        self.kinds.append(kind)


def _run(
    scripts: list[list[LLMStreamEvent]], config: CopilotConfig, **caps_overrides: Any
) -> tuple[list[Any], _KindTrace]:
    trace = _KindTrace()
    events = list(
        run_turn(
            _FakeClient(scripts),
            build_registry(minimal_caps(**caps_overrides)),
            config,
            _fake_context(),
            history=[],
            user_text="drive it",
            gate=GateChannel(),
            cancel=threading.Event(),
            trace=trace,
        )
    )
    return events, trace


# ---- B1: the key ----


def test_edit_target_key_namespaces_kind_and_reads_the_right_arg() -> None:
    # The script trio's arg is `node`, the shader pair's is `target` — a `target`-only read keyed
    # every script edit to the current-node sentinel and merged all nodes into one streak.
    assert _edit_target_key("edit_shader", {"target": "7f3a"}) == ("shader", "7f3a")
    assert _edit_target_key("edit_script", {"node": "7f3a"}) == ("script", "7f3a")
    assert _edit_target_key("edit_script", {}) == ("script", "<current>")
    # Same node, two files -> two different streaks.
    assert _edit_target_key("edit_shader", {"target": "7f3a"}) != _edit_target_key(
        "edit_script", {"node": "7f3a"}
    )


def test_a_nodes_shader_and_script_streaks_do_not_merge() -> None:
    # Two clean edits to each of a node's TWO files: neither streak reaches the threshold of 3,
    # so no nudge. Under one shared key the four edits would trip it.
    config = replace(COPILOT_CONFIG, clean_edit_soft_streak=3, clean_edit_hard_streak=0)
    _events, trace = _run(
        [_EDIT_SHADER, _EDIT_SCRIPT, _EDIT_SHADER, _EDIT_SCRIPT, _DONE],
        config,
        apply_shader_edit=lambda _o, _n, _r, _t: EditResult(matches=1, errors=[]),
        apply_script_edit=lambda _o, _n, _r, _node: ScriptWriteResult(
            ok=True, driven=["u_x"]
        ),
    )
    assert trace.kinds.count("clean_streak_nudge") == 0


# ---- B2: a clean write resets its own kind ----


def test_clean_write_script_resets_the_script_streak() -> None:
    # write_script is the sanctioned whole-file convergence for a script — finishing in one write
    # must never be the straw that trips the hard stop.
    config = replace(COPILOT_CONFIG, clean_edit_soft_streak=2, clean_edit_hard_streak=3)
    clean = ScriptWriteResult(ok=True, driven=["u_x"])
    events, _trace = _run(
        [_EDIT_SCRIPT, _EDIT_SCRIPT, _WRITE_SCRIPT, _EDIT_SCRIPT, _EDIT_SCRIPT, _DONE],
        config,
        apply_script_edit=lambda _o, _n, _r, _node: clean,
        write_script=lambda _t, _node: clean,
    )
    assert isinstance(events[-1], AgentTurnDone)


def test_script_edits_still_hit_the_hard_stop_without_a_write() -> None:
    # The counterpart falsifier: without the reset the same run force-ends, so the test above is
    # not passing for want of a working brake.
    config = replace(COPILOT_CONFIG, clean_edit_soft_streak=2, clean_edit_hard_streak=3)
    events, _trace = _run(
        [_EDIT_SCRIPT] * 5 + [_DONE],
        config,
        apply_script_edit=lambda _o, _n, _r, _node: ScriptWriteResult(
            ok=True, driven=["u_x"]
        ),
    )
    assert isinstance(events[-1], AgentError)


# ---- B3: a broken script edit is not clean ----


def test_broken_script_edits_reach_the_compile_thrash_nudge() -> None:
    # A script edit that applies but does not compile returns an `errors` payload, so the
    # applies-but-broken counter sees it — without it the nudge was unreachable for scripts AND
    # the broken edit counted toward the CLEAN streak.
    config = replace(COPILOT_CONFIG, max_compile_failures=2)
    _events, trace = _run(
        [_EDIT_SCRIPT, _EDIT_SCRIPT, _EDIT_SCRIPT, _DONE],
        config,
        apply_script_edit=lambda _o, _n, _r, _node: ScriptWriteResult(
            ok=True, compile_error="script.py:3: SyntaxError: invalid syntax"
        ),
    )
    assert trace.kinds.count("compile_thrash_nudge") == 1


# ---- B4: one write_script per node per batch ----


def _backend_stub() -> tuple[types.SimpleNamespace, list[tuple[str, str]]]:
    # CopilotBackend bound onto a namespace carrying only what the script write path touches, so
    # the REAL guard + the REAL shared write tail run without GL.
    written: list[tuple[str, str]] = []
    probe = types.SimpleNamespace(
        compile_error=None,
        runtime_error=None,
        driven={"u_x"},
        samples=[(0.0, {"u_x": 1.0})],
        per_key_errors=[],
        orphan_keys=[],
    )

    def _write(node_id: str, text: str) -> Any:
        written.append((node_id, text))
        return probe

    stub = types.SimpleNamespace(
        _bridge=types.SimpleNamespace(
            run_on_main=lambda fn, timeout=None, defer=False: fn()
        ),
        _resolve_node_or_current=lambda node: node or "n1",
        _batch_mutated=set(),
        _working_set_add=lambda address: None,
        _capture_script=lambda node_id: None,
        _write_script_source=_write,
        _script_broken_streak={},
        _script_last_clean={},
        _script_render_line=lambda node, samples: "",
        _get_ui_nodes=lambda: {"n1": types.SimpleNamespace(node=object())},
        _read_script_source=lambda node_id: ("old body\n", False),
    )
    stub._apply_script_text = CopilotBackend._apply_script_text.__get__(stub)
    return stub, written


def test_second_write_script_in_one_batch_is_rejected() -> None:
    stub, written = _backend_stub()
    write = CopilotBackend.write_script.__get__(stub)
    assert write("first", "").ok is True
    second = write("second", "")
    assert second.ok is False and "already edited earlier in this same step" in second.error
    assert "edit_script" in second.error  # steers at the script editor, not edit_shader
    assert written == [("n1", "first")]


def test_edit_script_after_a_write_in_the_same_batch_still_applies() -> None:
    # The check lives at the WRITE level only (mirroring the shader pair): a substring edit
    # re-matches the current text, so it is never stale.
    stub, written = _backend_stub()
    assert CopilotBackend.write_script.__get__(stub)("new body\n", "").ok is True
    result = CopilotBackend.apply_script_edit.__get__(stub)("old", "fresh", False, "")
    assert result.ok is True
    assert len(written) == 2


def test_batch_begin_clears_the_script_guard() -> None:
    stub, _written = _backend_stub()
    assert CopilotBackend.write_script.__get__(stub)("first", "").ok is True
    CopilotBackend.batch_begin.__get__(stub)()
    assert CopilotBackend.write_script.__get__(stub)("second", "").ok is True
