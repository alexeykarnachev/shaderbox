"""The engine-derived NL turn-summary (feature 020·25): history is NL-only, so each committed turn
collapses to one summary that MUST preserve four facts (each a real corpus failure moment): the new value
of a mutation (dial-back), the agent's stated assumption (correction), the irreversible-action ledger with
identity (don't re-publish on continue), and the failure note (why-did-that-fail). Pure: scripted fake
client, no GL."""

import threading
import time

from shaderbox.copilot.agent import (
    AgentError,
    AgentTurnDone,
    _build_turn_summary,
    _RunLog,
    run_turn,
)
from shaderbox.copilot.capabilities import (
    CompileErrorInfo,
    EditResult,
    PublishResult,
    SetUniformResult,
    ShaderView,
)
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.gate import GateChannel, GateResponse
from shaderbox.copilot.llm.api import LLMDone, LLMStreamEvent, LLMTextDelta, LLMUsage
from shaderbox.copilot.tools.registry import build_registry

from tests._caps import minimal_caps
from tests.test_copilot_loop import _FakeClient, _fake_context, _tool_call


def _run(caps, scripts, *, gate=None, cancel=None):
    return list(
        run_turn(
            _FakeClient(scripts),
            build_registry(caps),
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="go",
            gate=gate or GateChannel(),
            cancel=cancel or threading.Event(),
        )
    )


def test_fact2_set_uniform_new_value_lands_in_ledger() -> None:
    # The "not so much" dial-back needs the value the agent set last turn.
    caps = minimal_caps(
        set_uniform=lambda _n, _v, _node: SetUniformResult(ok=True, type_label="float")
    )
    scripts: list[list[LLMStreamEvent]] = [
        _tool_call(
            "c1", "set_uniform", '{"name": "u_speed", "value": 2.5, "node": "ab"}'
        ),
        [LLMTextDelta("Raised the speed."), LLMDone("stop", LLMUsage())],
    ]
    done = next(e for e in _run(caps, scripts) if isinstance(e, AgentTurnDone))
    joined = " ".join(done.summary.ledger)
    assert "u_speed" in joined and "2.5" in joined, joined
    assert "ab" in done.summary.nodes


def test_fact3_assumption_carried_in_reply() -> None:
    # The "no, center is bot-left" correction needs the agent's STATED assumption from the prior reply.
    caps = minimal_caps()
    scripts: list[list[LLMStreamEvent]] = [
        _tool_call("c1", "read_shader", '{"nodes": ["ab"]}'),
        [
            LLMTextDelta(
                "I assumed the text origin is screen-center and offset from there."
            ),
            LLMDone("stop", LLMUsage()),
        ],
    ]
    done = next(e for e in _run(caps, scripts) if isinstance(e, AgentTurnDone))
    assert "origin is screen-center" in done.summary.reply


def test_fact4_irreversible_ledger_carries_identity() -> None:
    # "continue" after a cutoff must not re-publish: the ledger must name WHAT was published (the
    # url/pack), which lives in payload, not the tool msg. publish_telegram is ALWAYS-gated -> approve.
    caps = minimal_caps(
        publish_telegram=lambda _n, _e: PublishResult(
            ok=True, url="t.me/packX", kind="telegram"
        ),
        telegram_connected=lambda: True,
        telegram_has_default_pack=lambda: True,
        has_current_node=lambda: True,
    )
    gate = GateChannel()
    approver = threading.Thread(target=_approve_when_asked, args=(gate,), daemon=True)
    approver.start()
    scripts: list[list[LLMStreamEvent]] = [
        _tool_call("c1", "publish_telegram", '{"emoji": "x"}'),
        [LLMTextDelta("Published."), LLMDone("stop", LLMUsage())],
    ]
    done = next(e for e in _run(caps, scripts, gate=gate) if isinstance(e, AgentTurnDone))
    approver.join(timeout=1.0)
    joined = " ".join(done.summary.ledger)
    assert "publish_telegram" in joined
    assert "packX" in joined, f"published identity missing from ledger: {joined}"


def _approve_when_asked(gate: GateChannel) -> None:
    # Mirror the UI thread: poll for the worker's pending gate, then approve it.
    for _ in range(500):
        if gate.take_pending() is not None:
            gate.answer(GateResponse(approved=True))
            return
        time.sleep(0.002)


def test_failure_note_lands_on_giveup() -> None:
    # A giveup (edit kept failing) must persist WHY, so "why did that fail" works next turn.
    caps = minimal_caps(
        read_shaders=lambda ids: [
            ShaderView(
                node_id="node-1", name="s", listing="1  x", uniforms=[], errors=[]
            )
        ],
        apply_shader_edit=lambda o, n, r, t: EditResult(matches=0, errors=[]),
    )
    scripts: list[list[LLMStreamEvent]] = [_tool_call("c0", "read_shader", "{}")]
    for i in range(COPILOT_CONFIG.max_edit_retries):
        scripts.append(
            _tool_call(f"e{i}", "edit_shader", '{"old_str": "zz", "new_str": "y"}')
        )
    error = next(e for e in _run(caps, scripts) if isinstance(e, AgentError))
    assert "couldn't apply" in error.summary.reply.lower()


def test_failed_edit_is_noted_in_ledger() -> None:
    # A failed mutating call is recorded (with FAILED) so the agent can see it tried and missed.
    caps = build_registry(minimal_caps())
    rl = _RunLog()
    rl.record("edit_shader", False, "error: old_str not found", {"target": "ab"}, None)
    summary = _build_turn_summary("", rl, caps)
    assert any("FAILED" in line for line in summary.ledger)


def test_cancel_partial_ledger_is_sane() -> None:
    # A mid-turn cancel produces a summary from the calls that DID execute, with no crash.
    caps = build_registry(minimal_caps())
    rl = _RunLog()
    rl.record("set_uniform", True, "set u_x (float) = 1.0", {"node": "ab"}, None)
    summary = _build_turn_summary("", rl, caps)
    assert "u_x" in " ".join(summary.ledger)
    assert "ab" in summary.nodes


def test_ledger_soft_caps_non_irreversible_lines() -> None:
    # A many-edit turn can't bloat history: non-irreversible mutating lines are capped.
    caps = build_registry(minimal_caps())
    rl = _RunLog()
    for i in range(30):
        rl.record("edit_shader", True, f"edit {i}", {"target": "ab"}, None)
    summary = _build_turn_summary("", rl, caps)
    assert len(summary.ledger) < 30
    assert any("more edits" in line for line in summary.ledger)
