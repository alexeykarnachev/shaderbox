"""The copilot agent loop against a fake LLM client — no GL, no live tokens.

Reproduces the §16.5 worked trace (read shader -> edit breaks compile -> read the
source-mapped error -> fix -> clean) and asserts the loop calls the tools in order,
feeds results back, and terminates on the clean compile with the final text.
"""

import json
import threading
from collections.abc import Iterator
from pathlib import Path

from shaderbox.copilot.agent import (
    AgentError,
    AgentEvent,
    AgentTextDelta,
    AgentToolCard,
    AgentTurnDone,
    run_turn,
)
from shaderbox.copilot.capabilities import (
    CompileErrorInfo,
    CopilotCapabilities,
    EditResult,
    ShaderView,
)
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.gate import GateChannel, GateRequest, GateResponse
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
from shaderbox.copilot.prompt_context import CopilotContext
from shaderbox.copilot.tools.registry import build_registry
from shaderbox.copilot.trace import TraceLog
from tests._caps import minimal_caps


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
        node_tree="- shader (id: node-1)  [current]",
        lib_catalog="(library is empty)",
        template_catalog="(no templates)",
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

    return minimal_caps(
        read_shaders=read_shaders,
        apply_shader_edit=apply_edit,
        apply_line_edit=apply_line,
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


def test_applies_but_broken_thrash_nudges_not_giveup() -> None:
    # Broken-compile circuit-breaker: an edit that APPLIES but compiles WITH errors returns
    # ok=True, so it never trips the failed-edit cap (which counts edits that fail to apply).
    # max_compile_failures consecutive such edits splice a one-time rewrite nudge, NOT a giveup.
    edit = _tool_call(
        "cx",
        "edit_shader",
        '{"old_str": "vec3 p = u_pos;", "new_str": "vec3 p = u_pos;"}',
    )
    cap = COPILOT_CONFIG.max_compile_failures
    # MORE than `cap` broken-but-applying edits, then a clean reply. The latch must keep the nudge
    # to ONCE even though the thrash continues past the cap (no re-nudge every `cap` steps).
    n_broken = cap + 3
    scripts: list[list[LLMStreamEvent]] = [edit] * n_broken + [
        [LLMTextDelta("Rewrote the block."), LLMDone("stop")]
    ]
    one_error = [
        CompileErrorInfo(path="node.frag.glsl", line=1, message="'x' : undeclared")
    ]
    caps = _fake_caps(edit_errors=[one_error] * n_broken)
    registry = build_registry(caps)

    nudge_events: list[str] = []

    class _RecordingTrace(TraceLog):
        def __init__(self) -> None:
            super().__init__(Path())

        def event(self, kind: str, **fields: object) -> None:
            nudge_events.append(kind)

    events = list(
        run_turn(
            _FakeClient(scripts),
            registry,
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="keep breaking it",
            gate=GateChannel(),
            cancel=threading.Event(),
            trace=_RecordingTrace(),
        )
    )

    cards = [e for e in events if isinstance(e, AgentToolCard)]
    # All edits applied (ok=True) — none counted as a failed edit, so no giveup.
    assert len(cards) == n_broken
    assert all(c.ok for c in cards)
    # The nudge fired exactly ONCE despite the thrash running past the cap (latch held).
    assert nudge_events.count("compile_thrash_nudge") == 1
    # The turn ended on the clean reply, NOT an AgentError giveup/cutoff.
    assert isinstance(events[-1], AgentTurnDone)


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


def test_max_iterations_streams_a_final_no_tools_reply() -> None:
    # 033: budget exhaustion must end with the model addressing the user, not a canned
    # error — the engine runs one extra NO-TOOLS stream and commits a normal TurnDone.
    read = _tool_call("cr", "read_shader", "{}")
    scripts: list[list[LLMStreamEvent]] = [read] * COPILOT_CONFIG.max_iterations
    scripts.append(
        [LLMTextDelta("I kept re-reading; nothing changed yet."), LLMDone("stop")]
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
            user_text="read forever",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )
    assert isinstance(events[-1], AgentTurnDone)
    assert "nothing changed yet" in events[-1].summary.reply


def test_empty_length_cutoff_gets_a_final_reply() -> None:
    # 033: a hidden-reasoning burn (finish=length, zero text, zero tools) forces the
    # same no-tools reply instead of ending the turn silent.
    scripts: list[list[LLMStreamEvent]] = [
        [LLMDone("length")],
        [
            LLMTextDelta("Budget burned before I could act - ask me to continue."),
            LLMDone("stop"),
        ],
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
            user_text="fix the layout",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )
    assert isinstance(events[-1], AgentTurnDone)
    assert "continue" in events[-1].summary.reply


def test_stale_shutdown_sentinel_does_not_strand_turn(tmp_path: Path) -> None:
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
        get_checkpoints_root=lambda: tmp_path / "checkpoints",
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


def test_turn_snippet_collects_steps_not_status_lines(tmp_path: Path) -> None:
    # F06: tool cards fold into ONE turn_snippet (square per call), NOT a tool_status line each.
    # A card carrying an actionable widget keeps its own visible tool_status line.
    from shaderbox.copilot.agent import AgentTurnDone
    from shaderbox.copilot.session import CopilotSession
    from shaderbox.copilot.state import ResultWidget, TurnStats

    sess = CopilotSession(
        _fake_caps(edit_errors=[]),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]  # client unused — we feed events directly
        get_project_slug=lambda: "test",
        get_checkpoints_root=lambda: tmp_path / "checkpoints",
    )
    sess.enqueue_turn("do stuff")  # appends the user msg + the empty turn_snippet
    sess._apply_event(AgentToolCard(name="read_shader", ok=True, payload=None))
    sess._apply_event(AgentToolCard(name="edit_shader", ok=False, payload=None))
    sess._apply_event(
        AgentToolCard(
            name="render_image",
            ok=True,
            payload=None,
            widget=ResultWidget(kind="open_path", label="Reveal", target="/x.png"),
        )
    )
    sess._apply_event(
        AgentTurnDone(
            stats=TurnStats(context_tokens=500, reply_tokens=80, cost_usd=0.002)
        )
    )

    snippets = [m for m in sess.state.messages if m.role == "turn_snippet"]
    assert len(snippets) == 1, "exactly one snippet per turn"
    snip = snippets[0]
    assert [(s.name, s.ok) for s in snip.steps] == [
        ("read_shader", True),
        ("edit_shader", False),
        ("render_image", True),
    ]
    assert snip.snippet_stats is not None and snip.snippet_stats.reply_tokens == 80
    # Only the widget-bearing card keeps a visible tool_status line; the other two do NOT.
    tool_lines = [m for m in sess.state.messages if m.role == "tool_status"]
    assert len(tool_lines) == 1
    assert tool_lines[0].result_widget is not None
    sess.release()


def test_errored_turn_leaves_snippet_finished_not_live(tmp_path: Path) -> None:
    # Reviewer-caught bug: AgentError/AgentCancelled set no snippet_stats. The renderer must NOT
    # treat a statless snippet as live (frozen "thinking...") — that's gated on state.in_flight,
    # which the terminal event clears. Assert the post-error invariant the renderer relies on.
    from shaderbox.copilot.agent import AgentCancelled, AgentError
    from shaderbox.copilot.session import CopilotSession

    def _mk() -> CopilotSession:
        return CopilotSession(
            _fake_caps(edit_errors=[]),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            get_project_slug=lambda: "test",
            get_checkpoints_root=lambda: tmp_path / "checkpoints",
        )

    for terminal in (AgentError(message="boom"), AgentCancelled()):
        sess = _mk()
        sess.enqueue_turn("go")
        sess._apply_event(AgentToolCard(name="edit_shader", ok=True, payload=None))
        sess._apply_event(terminal)
        snip = next(m for m in sess.state.messages if m.role == "turn_snippet")
        assert snip.snippet_stats is None  # error/cancel set no stats
        assert (
            sess.state.in_flight is False
        )  # so the renderer's `live` is False -> no thinking
        assert len(snip.steps) == 1  # the squares still record what happened
        sess.release()


def test_snippet_square_language() -> None:
    # F09: the bar's (color, pulse) list across turn states. A pulsing gray head is shown from
    # frame 1 (no jitter); a clean turn caps with an info-blue answer square (so a zero-tool reply
    # is one blue square); an error/cancel turn (no stats) adds no cap.
    from shaderbox.copilot.state import Message, StepRecord, TurnStats
    from shaderbox.theme import COLOR
    from shaderbox.widgets.copilot_chat import _snippet_squares

    # zero-tool, still running -> just the pulsing gray head.
    sq = _snippet_squares(Message(role="turn_snippet"), live=True)
    assert sq == [(COLOR.FG_DIM, True)]

    # zero-tool, clean done -> one solid info-blue answer square.
    done = Message(role="turn_snippet", snippet_stats=TurnStats(0, 10, 0.001))
    assert _snippet_squares(done, live=False) == [(COLOR.STATE_INFO, False)]

    # multi-tool, running -> green/red done squares + the pulsing head.
    running = Message(
        role="turn_snippet",
        steps=[StepRecord("a", True), StepRecord("b", False)],
    )
    assert _snippet_squares(running, live=True) == [
        (COLOR.STATE_OK, False),
        (COLOR.STATE_ERROR, False),
        (COLOR.FG_DIM, True),
    ]

    # multi-tool, clean done -> done squares + the answer cap (no head).
    multi_done = Message(
        role="turn_snippet",
        steps=[StepRecord("a", True)],
        snippet_stats=TurnStats(0, 10, 0.001),
    )
    assert _snippet_squares(multi_done, live=False) == [
        (COLOR.STATE_OK, False),
        (COLOR.STATE_INFO, False),
    ]

    # error/cancel (steps, no stats, not live) -> bar only, NO head and NO answer cap.
    errored = Message(role="turn_snippet", steps=[StepRecord("a", True)])
    assert _snippet_squares(errored, live=False) == [(COLOR.STATE_OK, False)]


def test_terminal_carries_nl_summary_not_tool_tail() -> None:
    # feature 020·28: the terminal event carries an engine-derived NL TurnSummary, NOT the tool tail.
    # A read_shader turn (non-mutating) yields the agent's reply + the read node in `nodes`, no ledger.
    caps = _fake_caps(edit_errors=[[]])
    registry = build_registry(caps)
    scripts: list[list[LLMStreamEvent]] = [
        _tool_call("c1", "read_shader", '{"nodes": ["abcd"]}'),
        [LLMTextDelta("Read it."), LLMDone("stop", LLMUsage())],
    ]
    events = list(
        run_turn(
            _FakeClient(scripts),
            registry,
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="read the shader",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )
    from shaderbox.copilot.agent import AgentTurnDone

    done = next(e for e in events if isinstance(e, AgentTurnDone))
    assert done.summary.reply == "Read it."
    assert done.summary.ledger == [], "a non-mutating read must add no ledger line"
    assert (
        "abcd" in done.summary.nodes
    ), "the read node must be referenced for cross-turn binding"


class _AlwaysDeclineGate(GateChannel):
    # ask() returns a decline without blocking — no UI thread in the test.
    def ask(self, request: GateRequest) -> GateResponse:
        _ = request
        return GateResponse(approved=False, option="No")


class _ApproveGate(GateChannel):
    def ask(self, request: GateRequest) -> GateResponse:
        _ = request
        return GateResponse(approved=True, option="Yes")


class _DeleteThenStopClient:
    # Emits a `delete_node` tool call for the first `n_deletes` stream() calls, then a final
    # plain-text reply. Captures the messages list it is handed each call so the test can
    # assert tool-result integrity after the turn.
    def __init__(self, n_deletes: int) -> None:
        self._n_deletes = n_deletes
        self._calls = 0
        self.last_messages: list[LLMMessage] = []

    def stream(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[LLMToolSpec] | None = None,
        max_tokens: int,
    ) -> Iterator[LLMStreamEvent]:
        _ = (tools, max_tokens)
        self.last_messages = list(messages)
        self._calls += 1
        if self._calls <= self._n_deletes:
            call_id = f"call_{self._calls}"
            yield LLMToolCallStarted(index=0, id=call_id, name="delete_node")
            yield LLMToolCallCompleted(
                index=0,
                id=call_id,
                name="delete_node",
                arguments=json.dumps({"node": f"n{self._calls}"}),
            )
            yield LLMDone(finish_reason="tool_calls", usage=LLMUsage(output_tokens=1))
        else:
            yield LLMTextDelta("You said no, so I've cancelled the deletion.")
            yield LLMDone(finish_reason="stop", usage=LLMUsage(output_tokens=1))


def test_declined_gates_no_giveup_no_orphaned_tool_calls() -> None:
    # A declined delete_node must NOT trip the edit-retry cap (it ends in AgentTurnDone, the
    # model's comment), AND every assistant tool_call must still get a matching tool result
    # message — an orphaned tool_call_id 400s the real provider on the next stream.
    n = COPILOT_CONFIG.max_edit_retries + 1  # one MORE than the cap
    registry = build_registry(minimal_caps())
    client = _DeleteThenStopClient(n_deletes=n)

    events = list(
        run_turn(
            client,
            registry,
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="delete everything",
            gate=_AlwaysDeclineGate(),
            cancel=threading.Event(),
        )
    )

    errors = [e for e in events if isinstance(e, AgentError)]
    assert (
        not errors
    ), f"a declined delete tripped a giveup: {[e.message for e in errors]}"
    assert isinstance(events[-1], AgentTurnDone)

    open_ids: set[str] = set()
    result_ids: set[str] = set()
    for m in client.last_messages:
        if m.role == "assistant" and m.tool_calls:
            open_ids.update(tc.id for tc in m.tool_calls)
        if m.role == "tool" and m.tool_call_id:
            result_ids.add(m.tool_call_id)
    assert open_ids, "expected at least one tool_call in the replayed messages"
    assert open_ids <= result_ids, f"orphaned tool_call_id(s): {open_ids - result_ids}"


class _DeleteThenEmptyFinishClient:
    # One delete, then an empty (no text, no tool calls) reply ending with the given
    # finish_reason — a reasoning model that ran the tool then emitted no visible content.
    # A native call already executed, so this must NOT be flagged tool-incompatible.
    def __init__(self, finish_reason: str) -> None:
        self._finish_reason = finish_reason
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
            yield LLMToolCallStarted(index=0, id="c1", name="delete_node")
            yield LLMToolCallCompleted(
                index=0, id="c1", name="delete_node", arguments='{"node": "n1"}'
            )
            yield LLMDone(finish_reason="tool_calls", usage=LLMUsage(output_tokens=1))
        else:
            yield LLMDone(
                finish_reason=self._finish_reason, usage=LLMUsage(output_tokens=57)
            )


def _run_empty_finish(finish_reason: str) -> list[AgentEvent]:
    return list(
        run_turn(
            _DeleteThenEmptyFinishClient(finish_reason),
            build_registry(minimal_caps()),
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="delete n1",
            gate=_ApproveGate(),
            cancel=threading.Event(),
        )
    )


def _is_incompatible_error(events: list[AgentEvent]) -> bool:
    return any(
        isinstance(e, AgentError) and "compatible with tool calling" in e.message
        for e in events
    )


def test_silent_finish_after_tool_is_not_flagged_incompatible() -> None:
    # A clean `stop` after a successful tool ends in AgentTurnDone (no error at all);
    # `length` / `content_filter` after a tool are terminal provider signals, NOT
    # tool-incompatibility — the model proved it can call tools.
    stop_events = _run_empty_finish("stop")
    assert not [e for e in stop_events if isinstance(e, AgentError)]
    assert [e for e in stop_events if isinstance(e, AgentTurnDone)]
    assert not _is_incompatible_error(_run_empty_finish("length"))
    assert not _is_incompatible_error(_run_empty_finish("content_filter"))
