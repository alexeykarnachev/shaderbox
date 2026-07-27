"""Feature 056 slices C3 + E: what the USER is shown. A torn stream is not an incompatible model,
a cancelled turn's screen keeps what history keeps, an all-empty turn still persists an assistant
message, a precheck handoff is a visible neutral card, and the engine's own look is attributed.
Bare objects — no app fixture, no GL, no LLM."""

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from shaderbox.copilot.agent import (
    AgentCancelled,
    AgentError,
    AgentToolCard,
    TurnSummary,
    run_turn,
)
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.gate import GateChannel
from shaderbox.copilot.llm.api import (
    LLMDone,
    LLMMessage,
    LLMStreamEvent,
    LLMTextDelta,
    LLMToolSpec,
)
from shaderbox.copilot.llm.openrouter import OpenRouterLLMClient
from shaderbox.copilot.session import CopilotSession, _tool_card_outcome
from shaderbox.copilot.tools.registry import build_registry
from tests._caps import minimal_caps
from tests.test_copilot_loop import _fake_caps, _fake_context, _FakeClient, _tool_call


class _NoDoneClient:
    # A stream that runs one tool, then ends with NO terminal event at all (no LLMDone, no text).
    model = "test-model"

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
            return iter(_tool_call("c1", "read_shader", "{}"))
        return iter([])  # torn: no text, no LLMDone, nothing at all


def test_torn_stream_is_not_diagnosed_as_an_incompatible_model() -> None:
    events = list(
        run_turn(
            _NoDoneClient(),
            build_registry(_fake_caps(edit_errors=[])),
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="read it",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )
    err = events[-1]
    assert isinstance(err, AgentError)
    assert "connection dropped" in err.message.lower()
    assert "Settings" not in err.message  # the model is fine; the connection wasn't


def test_precheck_handoff_is_a_visible_neutral_card() -> None:
    # A deflected publish (no credentials) used to yield NOTHING — the user saw an empty step.
    events = list(
        run_turn(
            _FakeClient(
                [
                    _tool_call("c1", "publish_telegram", '{"emoji": "x"}'),
                    [LLMTextDelta("Telegram isn't connected yet."), LLMDone("stop")],
                ]
            ),
            build_registry(minimal_caps()),
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="publish it",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )
    cards = [e for e in events if isinstance(e, AgentToolCard)]
    handoff = [c for c in cards if (c.payload or {}).get("handoff")]
    assert len(handoff) == 1
    assert handoff[0].name == "publish_telegram"
    assert handoff[0].ok is True  # a deflection is not a red failure


def _session(tmp_path: Path) -> CopilotSession:
    return CopilotSession(
        minimal_caps(),
        cast(OpenRouterLLMClient, object()),
        get_project_slug=lambda: "test",
        get_checkpoints_root=lambda: tmp_path / "checkpoints",
    )


def test_cancelled_turn_keeps_the_partial_reply_on_screen(tmp_path: Path) -> None:
    # The cancel terminals carry text_buf into the committed summary, so dropping it from the
    # screen left the user missing text the history keeps.
    sess = _session(tmp_path)
    try:
        sess.enqueue_turn("go")
        sess._apply_event(
            AgentCancelled(summary=TurnSummary(reply="I was halfway through"))
        )
        assert sess.state.streaming_text == ""
        assistant = [m for m in sess.state.messages if m.role == "assistant"]
        assert len(assistant) == 1 and "halfway" in assistant[0].text
    finally:
        sess.release()


def test_empty_turn_persists_a_placeholder_assistant_message(tmp_path: Path) -> None:
    # A null assistant content reaches the wire as `content: null` (some providers 400) and
    # persists silently; skipping the message alone would break the user/assistant pairing.
    sess = _session(tmp_path)
    try:
        sess._commit_turn("go", TurnSummary(), "")
        assert len(sess.history) == 2
        assert sess.history[0].role == "user"
        assert sess.history[1].role == "assistant"
        assert sess.history[1].content == "(turn ended with no reply)"
    finally:
        sess.release()


def test_engine_look_card_renders_an_attributed_line(tmp_path: Path) -> None:
    # C3: the engine's look is a billed step the user never asked for — without the line it is an
    # anonymous square. The payload SHAPE is the trigger, not the tool name.
    sess = _session(tmp_path)
    try:
        sess.enqueue_turn("go")
        sess._apply_event(
            AgentToolCard(
                "probe_render",
                True,
                {"engine_look": True, "vision_ok": True},
                result="render@t=0.0s",
            )
        )
        lines = [m for m in sess.state.messages if m.role == "tool_status"]
        assert len(lines) == 1 and "engine checked the render" in lines[0].text
        # A model-called probe stays an anonymous square...
        sess._apply_event(AgentToolCard("probe_render", True, None, result="x"))
        # ...and a BLIND engine look claims no check happened (vision off / outage).
        sess._apply_event(
            AgentToolCard(
                "probe_render",
                True,
                {"engine_look": True, "vision_ok": False},
                result="facts only",
            )
        )
        assert len([m for m in sess.state.messages if m.role == "tool_status"]) == 1
    finally:
        sess.release()


def test_handoff_card_reads_as_handed_off_not_failed(tmp_path: Path) -> None:
    card = AgentToolCard("publish_telegram", True, {"handoff": True}, result="no creds")
    assert _tool_card_outcome(card) == "handed off"
    # ...and the outcome is REACHABLE: a handoff has no widget, so without its own branch the
    # line never renders and the user still sees nothing.
    sess = _session(tmp_path)
    try:
        sess.enqueue_turn("publish")
        sess._apply_event(card)
        lines = [m for m in sess.state.messages if m.role == "tool_status"]
        assert len(lines) == 1 and "handed off" in lines[0].text
    finally:
        sess.release()


def test_eye_verdict_reaches_the_persisted_summary(tmp_path: Path) -> None:
    sess = _session(tmp_path)
    try:
        sess._commit_turn(
            "go", TurnSummary(reply="done", eye="eye: ask not-met after 2 looks -- x"), ""
        )
        assert "ask not-met" in (sess.history[1].content or "")
    finally:
        sess.release()
