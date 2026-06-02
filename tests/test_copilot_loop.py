"""The copilot agent loop against a fake LLM client — no GL, no live tokens.

Reproduces the §16.5 worked trace (read shader -> edit breaks compile -> read the
source-mapped error -> fix -> clean) and asserts the loop calls the tools in order,
feeds results back, and terminates on the clean compile with the final text.
"""

import threading
from collections.abc import Iterator

from shaderbox.copilot.agent import (
    AgentError,
    AgentTextDelta,
    AgentToolCard,
    AgentTurnDone,
    run_turn,
)
from shaderbox.copilot.capabilities import (
    CompileErrorInfo,
    CopilotCapabilities,
    EditResult,
    SetUniformResult,
    ShaderView,
)
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.context import CopilotContext
from shaderbox.copilot.gate import GateChannel
from shaderbox.copilot.glsl_lex import token_match
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
from shaderbox.copilot.tools.registry import build_registry


def _tool_call(call_id: str, name: str, args: str) -> list[LLMStreamEvent]:
    return [
        LLMToolCallStarted(index=0, id=call_id, name=name),
        LLMToolCallCompleted(index=0, id=call_id, name=name, arguments=args),
        LLMDone(finish_reason="tool_calls", usage=LLMUsage()),
    ]


class _FakeClient:
    # One scripted event-list per stream() call.
    def __init__(self, scripts: list[list[LLMStreamEvent]]) -> None:
        self._scripts = scripts
        self._i = 0

    def stream(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[LLMToolSpec] | None = None,
        max_tokens: int,
    ) -> Iterator[LLMStreamEvent]:
        _ = (messages, tools, max_tokens)
        script = self._scripts[self._i]
        self._i += 1
        return iter(script)


def _fake_context() -> CopilotContext:
    return CopilotContext(
        current_node_id="node-1",
        node_tree="- shader (id: node-1)  [current]",
        lib_catalog="(library is empty)",
        conventions="",
    )


def _fake_caps(edit_errors: list[list[CompileErrorInfo]]) -> CopilotCapabilities:
    # Models the App: apply_shader_edit matches old_str against the LIVE source via the REAL
    # token matcher (not a re-rolled str.count), replaces, and (on a real match) advances the
    # source — so a second edit sees the first's result and the fake stays faithful to prod.
    # The `target` arg is accepted and ignored (the fake models the current node only).
    state = {"text": "vec3 p = u_pos;", "n": 0}

    def apply_edit(
        old_str: str, new_str: str, replace_all: bool, target: str
    ) -> EditResult:
        _ = target
        spans = token_match(state["text"], old_str)
        if not spans or (len(spans) > 1 and not replace_all):
            return EditResult(matches=len(spans), errors=[])
        out: list[str] = []
        cursor = 0
        for start, end in spans:
            out.append(state["text"][cursor:start])
            out.append(new_str)
            cursor = end
        out.append(state["text"][cursor:])
        state["text"] = "".join(out)
        errors = edit_errors[state["n"]]
        state["n"] += 1
        return EditResult(matches=len(spans), errors=errors)

    def apply_line(
        start_line: int, end_line: int, new_text: str, target: str
    ) -> EditResult:
        # Faithful to App: split-on-newline list edit (§14 L2). matches==0 is the
        # out-of-range fail-soft signal the tool layer turns into the range error.
        _ = target
        lines = state["text"].split("\n")
        n = len(lines)
        is_insert = end_line == start_line - 1
        if start_line < 1 or end_line > n or (start_line > end_line and not is_insert):
            return EditResult(matches=0, errors=[])
        repl = new_text.split("\n") if new_text != "" else []
        state["text"] = "\n".join(lines[: start_line - 1] + repl + lines[end_line:])
        errors = edit_errors[state["n"]]
        state["n"] += 1
        return EditResult(matches=1, errors=errors)

    def read_shaders(node_ids: list[str]) -> list[ShaderView]:
        return [
            ShaderView(
                node_id="node-1",
                name="shader",
                listing=f"1  {state['text']}",
                uniforms=["u_pos vec3 = (0.0, 0.0, 0.0)"],
                errors=[],
            )
        ]

    return CopilotCapabilities(
        list_nodes=lambda: [],
        get_node_summary=lambda _nid: None,
        get_shader_source=lambda _nid: None,
        get_compile_errors=lambda _nid: [],
        current_node_id=lambda: "node-1",
        node_tree=lambda: [],
        lib_catalog=lambda: [],
        read_shaders=read_shaders,
        grep=lambda _q: [],
        read_lib=lambda _names: [],
        apply_shader_edit=apply_edit,
        apply_line_edit=apply_line,
        set_uniform=lambda _n, _v, _node: SetUniformResult(ok=True),
        create_node=lambda _n, _s, _sw: "node-new",
    )


def test_edit_compile_feedback_self_correction() -> None:
    scripts: list[list[LLMStreamEvent]] = [
        _tool_call("c1", "read_shader", "{}"),
        _tool_call(
            "c2",
            "edit_shader",
            '{"old_str": "vec3 p = u_pos;", '
            '"new_str": "vec3 p = u_pos + vec3(sin(u_time), 0.0, 0.0);"}',
        ),
        _tool_call(
            "c3",
            "edit_shader",
            '{"old_str": "vec3 p = u_pos + vec3(sin(u_time), 0.0, 0.0);", '
            '"new_str": "vec3 p = u_pos;"}',
        ),
        [LLMTextDelta("Added a sine wobble and declared u_time."), LLMDone("stop")],
    ]
    caps = _fake_caps(
        edit_errors=[
            [
                CompileErrorInfo(
                    path="node.frag.glsl", line=14, message="'u_time' : undeclared"
                )
            ],
            [],
        ]
    )
    registry = build_registry(caps)

    events = list(
        run_turn(
            _FakeClient(scripts),
            registry,
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="animate the position uniform",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )

    cards = [e for e in events if isinstance(e, AgentToolCard)]
    assert [c.name for c in cards] == [
        "read_shader",
        "edit_shader",
        "edit_shader",
    ]
    # First edit applied + compiled WITH errors (ok=True — the tool succeeded); second clean.
    assert cards[1].ok
    assert cards[1].payload == {
        "errors": [
            {"path": "node.frag.glsl", "line": 14, "message": "'u_time' : undeclared"}
        ]
    }
    assert cards[2].payload == {"errors": []}

    text = "".join(e.text for e in events if isinstance(e, AgentTextDelta))
    assert "u_time" in text
    assert isinstance(events[-1], AgentTurnDone)


def test_edit_applies_despite_whitespace_divergence() -> None:
    # Feature 020 · 13: an old_str that differs from the source ONLY in inter-token
    # whitespace (the 6-vs-4-space spiral) now token-MATCHES and applies — it does not
    # 0-match and fall to the hint path. The seeded source is "vec3 p = u_pos;"; the
    # model sends it with collapsed/extra spaces.
    scripts: list[list[LLMStreamEvent]] = [
        _tool_call(
            "c1",
            "edit_shader",
            '{"old_str": "vec3   p=u_pos ;", "new_str": "vec3 p = u_pos * 2.0;"}',
        ),
        [LLMTextDelta("Doubled the position."), LLMDone("stop")],
    ]
    caps = _fake_caps(edit_errors=[[]])
    registry = build_registry(caps)

    events = list(
        run_turn(
            _FakeClient(scripts),
            registry,
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="double the position",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )
    cards = [e for e in events if isinstance(e, AgentToolCard)]
    assert len(cards) == 1
    assert cards[0].name == "edit_shader"
    assert cards[0].ok  # applied + compiled clean, NOT a 0-match
    assert cards[0].payload == {"errors": []}


def test_edit_not_found_is_a_tool_error() -> None:
    scripts: list[list[LLMStreamEvent]] = [
        _tool_call("c1", "edit_shader", '{"old_str": "not present", "new_str": "x"}'),
        [LLMTextDelta("I could not find that text."), LLMDone("stop")],
    ]
    caps = _fake_caps(edit_errors=[])
    registry = build_registry(caps)

    events = list(
        run_turn(
            _FakeClient(scripts),
            registry,
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="change the thing",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )
    cards = [e for e in events if isinstance(e, AgentToolCard)]
    assert len(cards) == 1
    assert cards[0].name == "edit_shader"
    assert cards[0].ok is False


def test_repeated_failed_edits_stop_at_retry_cap() -> None:
    # §I2 self-correction cap (feature 020 · 12 B/C): a model that keeps sending an
    # old_str that never matches must STOP at max_edit_retries, not loop to
    # max_iterations, and the turn must end with an AgentError the user sees.
    fail_edit = _tool_call(
        "cx", "edit_shader", '{"old_str": "never present", "new_str": "x"}'
    )
    # Far more failing edits than the cap — if the cap weren't enforced the loop would
    # run to max_iterations and exhaust this list (then IndexError on the fake client).
    scripts: list[list[LLMStreamEvent]] = [fail_edit] * (
        COPILOT_CONFIG.max_iterations + 5
    )
    caps = _fake_caps(edit_errors=[])
    registry = build_registry(caps)

    events = list(
        run_turn(
            _FakeClient(scripts),
            registry,
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="do the thing",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )

    failed_cards = [e for e in events if isinstance(e, AgentToolCard) and not e.ok]
    # Exactly the retry budget of failed edits, then it gives up.
    assert len(failed_cards) == COPILOT_CONFIG.max_edit_retries
    assert isinstance(events[-1], AgentError)
    assert "couldn't apply" in events[-1].message
    # Did NOT run to the hard ceiling.
    assert len(failed_cards) < COPILOT_CONFIG.max_iterations


def test_max_iterations_cutoff_surfaces_as_error() -> None:
    # A non-edit tool that keeps getting called (read loop) must hit max_iterations and
    # surface an AgentError so the chat shows why it stopped (feature 020 · 12 C).
    read = _tool_call("cr", "read_shader", "{}")
    scripts: list[list[LLMStreamEvent]] = [read] * (COPILOT_CONFIG.max_iterations + 5)
    caps = _fake_caps(edit_errors=[])
    registry = build_registry(caps)

    events = list(
        run_turn(
            _FakeClient(scripts),
            registry,
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="read forever",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )
    assert isinstance(events[-1], AgentError)
    assert "stopped after" in events[-1].message.lower()


def test_stale_shutdown_sentinel_does_not_strand_turn() -> None:
    # Regression: a reused CopilotSession could hold a leftover _SHUTDOWN sentinel in
    # its turn queue (from a prior release()); the worker would dequeue it and exit
    # before processing the real turn, leaving in_flight=True forever ("thinking…").
    import time

    from shaderbox.copilot.llm.api import LLMDone, LLMTextDelta, LLMUsage
    from shaderbox.copilot.session import _SHUTDOWN, CopilotSession

    class _PlainClient:
        def stream(
            self,
            messages: list[LLMMessage],
            *,
            tools: list[LLMToolSpec] | None = None,
            max_tokens: int,
        ) -> Iterator[LLMStreamEvent]:
            _ = (messages, tools, max_tokens)
            return iter([LLMTextDelta("hi back"), LLMDone("stop", LLMUsage())])

    sess = CopilotSession(
        _fake_caps(edit_errors=[]),  # type: ignore[arg-type]
        _PlainClient(),  # type: ignore[arg-type]
        get_project_slug=lambda: "test",
    )
    sess._turn_queue.put(_SHUTDOWN)  # simulate the stale sentinel
    sess.enqueue_turn("hey")
    for _ in range(100):
        sess.pump_events()
        if not sess.state.in_flight:
            break
        time.sleep(0.02)
    sess.release()

    assert not sess.state.in_flight, "turn stranded by a stale shutdown sentinel"
    assert any(m.role == "assistant" for m in sess.state.messages)
