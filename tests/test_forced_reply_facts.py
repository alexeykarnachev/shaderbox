"""Feature 058: a force-ended turn's closing reply carries a FRESH measurement of the frame.

All three observed dishonest limit-endings happened on the forced-reply path, which summarized from
intentions alone. When the turn authored a render, the engine probes the last-authored document once
(free, numeric) and splices the facts into the nudge — so the model states the net result from the
measurements. Falsifier for each path: cut the splice and these go red. Deterministic (scripted fake
client, faked probe) — no GL, no live tokens."""

import threading
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from shaderbox.copilot.agent import AgentError, AgentTurnDone, run_turn
from shaderbox.copilot.capabilities import EditResult
from shaderbox.copilot.config import COPILOT_CONFIG, CopilotConfig
from shaderbox.copilot.edit_hints import STAMPED_FACTS_PREFIX
from shaderbox.copilot.gate import GateChannel
from shaderbox.copilot.llm.api import (
    LLMDone,
    LLMMessage,
    LLMStreamEvent,
    LLMTextDelta,
    LLMToolSpec,
    LLMUsage,
)
from shaderbox.copilot.prompt import _RENDER_FACTS_LEGEND
from shaderbox.copilot.tools.registry import build_registry
from tests._caps import minimal_caps
from tests.test_copilot_loop import _fake_context, _tool_call

_FACTS = (
    f"{STAMPED_FACTS_PREFIX}0.0s: ink 34% | bbox x 0.08-0.92, y 0.11-0.88 (y=0 bottom) | ink mean "
    "rgb(31,64,140) cool | luma 0-9 top/mid/bottom rows: 112/223 441/552 663/774"
)
_MEASURED = "the frame currently measures:"

_EDIT = _tool_call(
    "c1", "edit_shader", '{"old_str": "u_pos", "new_str": "u_pos", "target": ""}'
)
_READ = _tool_call("c2", "read_shader", "{}")


class _NudgeSpy:
    # Scripted like _FakeClient, but records the closing NO-TOOLS request's nudge (tools=None).
    model = "test-model"

    def __init__(self, scripts: list[list[LLMStreamEvent]]) -> None:
        self._scripts = scripts
        self._i = 0
        self.nudges: list[str] = []

    def stream(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[LLMToolSpec] | None = None,
        max_tokens: int,
    ) -> Iterator[LLMStreamEvent]:
        _ = max_tokens
        if tools is None:
            self.nudges.append(messages[-1].content or "")
            return iter(
                [
                    LLMTextDelta("I stopped at my limit."),
                    LLMDone("stop", LLMUsage(output_tokens=5)),
                ]
            )
        script = self._scripts[self._i]
        self._i += 1
        return iter(script)


def _drive(
    scripts: list[list[LLMStreamEvent]],
    probe: str = _FACTS,
    config: CopilotConfig = COPILOT_CONFIG,
    edit_facts: str = "",
) -> tuple[_NudgeSpy, list[tuple[str, float]], list[Any]]:
    probes: list[tuple[str, float]] = []

    def _probe(document: str, t: float) -> str:
        probes.append((document, t))
        return probe

    caps = minimal_caps(
        apply_shader_edit=lambda _o, _n, _r, _t: EditResult(
            matches=1, errors=[], render_facts=edit_facts
        ),
        probe_render=_probe,
    )
    client = _NudgeSpy(scripts)
    events = list(
        run_turn(
            client,
            build_registry(caps),
            config,
            _fake_context(),
            history=[],
            user_text="make a red circle",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )
    return client, probes, events


def test_token_budget_forced_reply_carries_the_facts() -> None:
    # The empty-text `length` cutoff after a tool ran: the model produced nothing, so the forced
    # reply is ALL the user gets — it must be measured, not imagined.
    scripts: list[list[LLMStreamEvent]] = [
        _EDIT,
        [LLMDone("length", LLMUsage(output_tokens=900))],
    ]
    client, probes, events = _drive(scripts)
    assert isinstance(events[-1], AgentTurnDone)
    assert len(client.nudges) == 1
    assert _MEASURED in client.nudges[0] and _FACTS in client.nudges[0]
    assert probes == [
        ("", 0.0)
    ]  # the last-authored document (empty = current), export clock


def test_max_iterations_forced_reply_carries_the_facts() -> None:
    config = replace(COPILOT_CONFIG, max_iterations=2)
    client, probes, events = _drive([_EDIT, _EDIT], config=config)
    assert isinstance(events[-1], AgentTurnDone)
    assert _MEASURED in client.nudges[0] and _FACTS in client.nudges[0]
    assert len(probes) == 1


def test_time_budget_forced_reply_carries_the_facts(monkeypatch: Any) -> None:
    clock = iter(range(0, 100_000, 200))
    monkeypatch.setattr(
        "shaderbox.copilot.agent.time.monotonic", lambda: float(next(clock))
    )
    config = replace(COPILOT_CONFIG, turn_time_budget_s=100)
    client, probes, events = _drive([_EDIT], config=config)
    assert isinstance(events[-1], AgentTurnDone)
    assert _MEASURED in client.nudges[0] and _FACTS in client.nudges[0]
    assert len(probes) == 1


def test_clean_streak_hard_stop_note_stays_human_prose() -> None:
    # This path is USER-facing engine text and the user can see the live render — raw probe
    # telemetry is model food, not prose. The facts clause rides only the model-facing nudges.
    config = replace(COPILOT_CONFIG, clean_edit_soft_streak=0, clean_edit_hard_streak=2)
    _client, probes, events = _drive([_EDIT, _EDIT], config=config)
    assert isinstance(events[-1], AgentError)
    assert _MEASURED not in events[-1].message
    assert probes == []


def test_probe_error_still_delivers_the_forced_reply_without_the_line() -> None:
    # A failed measurement must never block the reply — the clause is simply absent.
    scripts: list[list[LLMStreamEvent]] = [
        _EDIT,
        [LLMDone("length", LLMUsage(output_tokens=900))],
    ]
    client, probes, events = _drive(scripts, probe="error: no such document 'x'")
    assert isinstance(events[-1], AgentTurnDone)
    assert len(client.nudges) == 1 and _MEASURED not in client.nudges[0]
    assert "I stopped at my limit." in events[-1].summary.reply
    assert len(probes) == 1


def test_no_render_authored_means_no_probe_and_no_line() -> None:
    # A read-only turn changed nothing on screen; measuring it would be a pointless render and a
    # misleading clause about a frame this turn never touched.
    config = replace(COPILOT_CONFIG, max_iterations=2)
    client, probes, events = _drive([_READ, _READ], config=config)
    assert isinstance(events[-1], AgentTurnDone)
    assert probes == []
    assert _MEASURED not in client.nudges[0]


def test_forced_reply_facts_carry_the_legend_when_they_are_the_turns_first() -> None:
    # D7 (059): the legend routes through the SAME per-turn flag as the tool-result splice, so a
    # forced-end reply whose probe is the turn's first facts line is not left glossing nothing.
    scripts: list[list[LLMStreamEvent]] = [
        _EDIT,
        [LLMDone("length", LLMUsage(output_tokens=900))],
    ]
    client, _probes, events = _drive(scripts)
    assert isinstance(events[-1], AgentTurnDone)
    assert _FACTS in client.nudges[0] and _RENDER_FACTS_LEGEND in client.nudges[0]


def test_forced_reply_does_not_repeat_a_legend_an_earlier_result_carried() -> None:
    # The flag is per TURN, not per emission site: the edit's own facts line already took the
    # legend, so the forced reply's facts must arrive bare.
    scripts: list[list[LLMStreamEvent]] = [
        _EDIT,
        [LLMDone("length", LLMUsage(output_tokens=900))],
    ]
    client, _probes, events = _drive(scripts, edit_facts=_FACTS)
    assert isinstance(events[-1], AgentTurnDone)
    assert _FACTS in client.nudges[0] and _RENDER_FACTS_LEGEND not in client.nudges[0]
