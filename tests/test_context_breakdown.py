"""The per-request context breakdown (075 W-1): what each prompt tier contributed to ONE LLM
request, measured where the request is assembled.

Falsifiers, each named: (1) a block added to `build_blocks` without a breakdown entry -- the
coverage is enumerated from the block list, never from a hardcoded set; (2) a breakdown emitted
at build time -- the working set renders empty then and is spliced live per iteration, so the
entry must carry the LIVE splice and change when the splice changes; (3) the breakdown altering
what is sent -- the request the client receives is byte-identical with a listener attached and
without; (4) a trimmed dialogue reported as whole."""

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from shaderbox.copilot.agent import run_turn
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.context_breakdown import (
    EXCHANGE_BLOCK,
    ContextBreakdown,
    breakdown_request,
)
from shaderbox.copilot.gate import GateChannel
from shaderbox.copilot.llm.api import (
    LLMDone,
    LLMMessage,
    LLMStreamEvent,
    LLMTextDelta,
    LLMToolSpec,
)
from shaderbox.copilot.prompt import (
    DIALOGUE_BLOCK,
    WORKING_SET_BLOCK,
    build_blocks,
    build_messages,
    render_blocks,
)
from shaderbox.copilot.tools.registry import build_registry
from shaderbox.copilot.trace import TraceLog
from tests._caps import minimal_caps
from tests.test_copilot_loop import _fake_context, _tool_call


class _RecordingClient:
    # Keeps every request as received, so two runs can be compared byte for byte.
    model = "test-model"

    def __init__(self, scripts: list[list[LLMStreamEvent]]) -> None:
        self._scripts = scripts
        self.requests: list[tuple[list[LLMMessage], list[str]]] = []

    def stream(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[LLMToolSpec] | None = None,
        max_tokens: int,
    ) -> Iterator[LLMStreamEvent]:
        _ = max_tokens
        self.requests.append((list(messages), [t.name for t in tools or []]))
        return iter(self._scripts[len(self.requests) - 1])


def _breakdowns(trace_events: list[tuple[str, dict[str, Any]]]) -> list[ContextBreakdown]:
    return [f["breakdown"] for k, f in trace_events if k == "context_breakdown"]


def _drive(
    scratchpad_texts: list[str], with_listener: bool
) -> tuple[_RecordingClient, list[ContextBreakdown]]:
    # Two iterations: a read_shader call, then a text reply. The scratchpad differs per call so a
    # per-iteration emission is distinguishable from a one-shot one.
    scripts: list[list[LLMStreamEvent]] = [
        _tool_call("c1", "read_shader", '{"documents": ["document-1"]}'),
        [LLMTextDelta("read it"), LLMDone("stop")],
    ]
    calls = {"n": 0}

    def scratchpad() -> list[LLMMessage]:
        text = scratchpad_texts[min(calls["n"], len(scratchpad_texts) - 1)]
        calls["n"] += 1
        return [LLMMessage(role="user", content=text)]

    seen: list[tuple[str, dict[str, Any]]] = []
    listeners = [lambda k, f: seen.append((k, f))] if with_listener else []
    client = _RecordingClient(scripts)
    events = run_turn(
        client,
        build_registry(minimal_caps()),
        COPILOT_CONFIG,
        _fake_context(),
        history=[LLMMessage(role="user", content="earlier"), LLMMessage(role="assistant", content="ok")],
        user_text="read the shader",
        gate=GateChannel(),
        cancel=threading.Event(),
        trace=TraceLog(Path(), listeners),
        scratchpad_render=scratchpad,
    )
    list(events)
    return client, _breakdowns(seen)


def test_every_built_block_appears_enumerated_from_the_block_list() -> None:
    _, breakdowns = _drive(["WORKING SET -- live"], with_listener=True)
    assert len(breakdowns) == 2, "one breakdown per LLM request"
    names_in_prompt = [b.name for b in build_blocks(_fake_context(), [], "x")]
    for bd in breakdowns:
        reported = [b.name for b in bd.blocks]
        for name in names_in_prompt:
            assert name in reported, f"block {name!r} missing from the breakdown"
        assert EXCHANGE_BLOCK in reported


def test_working_set_is_the_live_splice_not_the_build_time_placeholder() -> None:
    _, breakdowns = _drive(["WS iteration zero", "WS iteration one, longer"], True)
    ws0 = next(b for b in breakdowns[0].blocks if b.name == WORKING_SET_BLOCK)
    ws1 = next(b for b in breakdowns[1].blocks if b.name == WORKING_SET_BLOCK)
    assert ws0.chars > 0 and "iteration zero" in ws0.text
    assert "iteration one" in ws1.text and ws1.chars > ws0.chars
    # The second request carries the read_shader exchange; the first does not.
    ex0 = next(b for b in breakdowns[0].blocks if b.name == EXCHANGE_BLOCK)
    ex1 = next(b for b in breakdowns[1].blocks if b.name == EXCHANGE_BLOCK)
    assert ex0.messages == 0 and ex1.messages == 2 and "read_shader" in ex1.text


def test_breakdown_does_not_change_what_is_sent() -> None:
    plain, _ = _drive(["same ws"], with_listener=False)
    observed, breakdowns = _drive(["same ws"], with_listener=True)
    assert plain.requests == observed.requests
    # And the measured sizes ARE the request's: block chars sum to the request's message chars.
    for (messages, tool_names), bd in zip(observed.requests, breakdowns, strict=True):
        request_chars = sum(len(m.content or "") for m in messages) + sum(
            len(tc.name) + len(tc.arguments) for m in messages for tc in m.tool_calls or ()
        )
        assert sum(b.chars for b in bd.blocks) == request_chars
        assert list(bd.tools) == tool_names


def test_trimmed_dialogue_is_reported_as_trimmed() -> None:
    big = COPILOT_CONFIG.max_input_tokens * 4
    history = [
        m
        for i in range(12)
        for m in (
            LLMMessage(role="user", content=f"u{i} " + "x" * big),
            LLMMessage(role="assistant", content=f"a{i}"),
        )
    ]
    rendered = render_blocks(build_blocks(_fake_context(), history, "now"))
    bd = breakdown_request(rendered, [], [], [], iteration=0, history_len=len(history))
    dialogue = next(b for b in bd.blocks if b.name == DIALOGUE_BLOCK)
    assert dialogue.trimmed and dialogue.dropped_messages > 0
    assert dialogue.messages + dialogue.dropped_messages == len(history)
    whole = breakdown_request(
        render_blocks(build_blocks(_fake_context(), history[:2], "now")),
        [],
        [],
        [],
        iteration=0,
        history_len=2,
    )
    assert not next(b for b in whole.blocks if b.name == DIALOGUE_BLOCK).trimmed


def test_build_messages_is_the_flattened_block_render() -> None:
    history = [LLMMessage(role="user", content="hi")]
    rendered = render_blocks(build_blocks(_fake_context(), history, "go"))
    assert [m for rb in rendered for m in rb.messages] == build_messages(
        _fake_context(), history, "go"
    )


def test_tools_block_is_measured_and_grows_with_loaded_tools() -> None:
    registry = build_registry(minimal_caps())
    eager = registry.assemble_specs(set())
    lazy_name = next(d.name for d in registry.definitions() if registry.is_lazy(d.name))
    grown = registry.assemble_specs({lazy_name})
    a = breakdown_request([], [], [], eager, iteration=0, history_len=0)
    b = breakdown_request([], [], [], grown, iteration=1, history_len=0)
    assert lazy_name in b.tools and lazy_name not in a.tools
    assert b.tools_chars > a.tools_chars and b.est_total_tokens > a.est_total_tokens
