"""Mid-turn stream tear containment: a network error on a later iteration (tools already
ran) must terminate the turn as an AgentError CARRYING the accumulated summary + stats
(the ledger keeps a continue from re-doing actions; the cost reaches the session
accounting), and the session must drop the partially-streamed ghost text."""

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from shaderbox.copilot.agent import AgentError, AgentTextDelta, run_turn
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.gate import GateChannel
from shaderbox.copilot.llm.api import (
    LLMDone,
    LLMMessage,
    LLMStreamEvent,
    LLMTextDelta,
    LLMToolCallCompleted,
    LLMToolCallStarted,
    LLMToolSpec,
    LLMUsage,
)
from shaderbox.copilot.llm.openrouter import OpenRouterLLMClient
from shaderbox.copilot.session import CopilotSession
from shaderbox.copilot.tools.registry import build_registry
from tests.test_copilot_loop import _fake_caps, _fake_context


class _TornSecondStreamClient:
    # First stream yields one applying edit; the second dies mid-iteration.
    def __init__(self) -> None:
        self._calls = 0

    def stream(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[LLMToolSpec] | None = None,
        max_tokens: int,
    ) -> Iterator[LLMStreamEvent]:
        _ = (messages, tools, max_tokens)
        self._calls += 1
        if self._calls == 1:
            return iter(
                [
                    LLMToolCallStarted(index=0, id="c1", name="edit_shader"),
                    LLMToolCallCompleted(
                        index=0,
                        id="c1",
                        name="edit_shader",
                        arguments='{"old_str": "vec3 p = u_pos;",'
                        ' "new_str": "vec3 p = u_pos * 2.0;"}',
                    ),
                    LLMDone(
                        finish_reason="tool_calls",
                        usage=LLMUsage(output_tokens=7, cost_usd=0.01),
                    ),
                ]
            )
        return self._torn()

    def _torn(self) -> Iterator[LLMStreamEvent]:
        yield LLMTextDelta("partial")
        raise RuntimeError("connection reset")


def test_torn_main_stream_keeps_summary_and_stats() -> None:
    caps = _fake_caps(edit_errors=[[]])
    registry = build_registry(caps)
    events = list(
        run_turn(
            _TornSecondStreamClient(),
            registry,
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="double the position",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )
    err = events[-1]
    assert isinstance(err, AgentError)
    assert "mid-turn" in err.message
    assert err.stats is not None
    assert err.stats.cost_usd == 0.01
    assert any(line.startswith("edit_shader") for line in err.summary.ledger)


def test_error_terminal_clears_streaming_ghost(tmp_path: Path) -> None:
    sess = CopilotSession(
        _fake_caps(edit_errors=[]),
        cast(OpenRouterLLMClient, object()),
        get_project_slug=lambda: "test",
        get_checkpoints_root=lambda: tmp_path / "checkpoints",
    )
    sess.enqueue_turn("go")
    try:
        sess._apply_event(AgentTextDelta("ghost text"))
        assert sess.state.streaming_text == "ghost text"
        sess._apply_event(AgentError(message="boom"))
        assert sess.state.streaming_text == ""
        assert sess.state.in_flight is False
    finally:
        sess.release()
