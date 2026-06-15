"""The load-bearing 020·28 invariant: a committed NL turn-summary is actually RECEIVED by the agent on a
LATER turn (not merely built). Every other test checks the summary's content in isolation; this drives two
turns through a real CopilotSession and asserts the second turn's LLM request carries the first turn's
summary (target name + new value) in its messages. Without this, "source-fidelity traded for resolvability"
is unverified end-to-end. Pure: a capturing fake client, fake caps, no GL."""

from collections.abc import Iterator
from pathlib import Path

from shaderbox.copilot.capabilities import SetUniformResult
from shaderbox.copilot.llm.api import (
    LLMDone,
    LLMMessage,
    LLMStreamEvent,
    LLMTextDelta,
    LLMUsage,
)
from shaderbox.copilot.session import CopilotSession
from tests._caps import minimal_caps
from tests.test_copilot_loop import _tool_call


class _CapturingClient:
    # Records the `messages` of every stream() call so a later turn can assert what it received.
    # `model` mirrors the real client — the session records it at turn_start.
    model = "test-model"

    def __init__(self, scripts: list[list[LLMStreamEvent]]) -> None:
        self._scripts = scripts
        self._i = 0
        self.requests: list[list[LLMMessage]] = []

    def stream(
        self, messages: list[LLMMessage], *, tools=None, max_tokens: int
    ) -> Iterator[LLMStreamEvent]:
        self.requests.append(list(messages))
        script = self._scripts[self._i]
        self._i += 1
        return iter(script)


def _drain(session: CopilotSession) -> None:
    # Run the queued worker turn to completion: poll the main-thread event pump until in_flight clears
    # (the terminal event flips it). Mirrors the session-level pattern in test_copilot_loop.
    import time

    for _ in range(2000):
        session.pump_events()
        if not session.state.in_flight:
            return
        time.sleep(0.002)


def test_second_turn_receives_first_turn_summary(tmp_path: Path) -> None:
    caps = minimal_caps(
        set_uniform=lambda _n, _v, _node: SetUniformResult(ok=True, type_label="float")
    )
    # Turn 1: set u_speed = 2.5 on node "ab", then reply. Turn 2: a plain reply (we only inspect input).
    scripts: list[list[LLMStreamEvent]] = [
        _tool_call(
            "c1", "set_uniform", '{"name": "u_speed", "value": 2.5, "node": "ab"}'
        ),
        [LLMTextDelta("Raised the speed."), LLMDone("stop", LLMUsage())],
        [LLMTextDelta("Okay."), LLMDone("stop", LLMUsage())],
    ]
    client = _CapturingClient(scripts)
    session = CopilotSession(  # type: ignore[arg-type]
        caps,
        client,
        get_project_slug=lambda: "test",
        get_checkpoints_root=lambda: tmp_path / "checkpoints",
    )
    try:
        session.enqueue_turn("set the speed to 2.5")
        _drain(session)
        session.enqueue_turn("now what?")
        _drain(session)
    finally:
        session.release()

    # The LAST request is turn 2's prompt. It must contain the NL summary of turn 1 — NO tool messages,
    # but the target name + new value carried in an assistant message.
    turn2 = client.requests[-1]
    assert all(m.role != "tool" for m in turn2), (
        "a tool message leaked into NL-only history"
    )
    assert all(m.tool_calls is None for m in turn2), (
        "a tool_call leaked into NL-only history"
    )
    blob = "\n".join(m.content or "" for m in turn2)
    assert "u_speed" in blob and "2.5" in blob, "turn-1 summary not visible to turn 2"
    assert "ab" in blob, "turn-1 referenced node not visible to turn 2"
