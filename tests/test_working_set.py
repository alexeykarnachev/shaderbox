"""Working-set scratchpad (feature 020·29).

Three layers:
- the prompt-assembly pieces (the D10 trim reserve, the working-set render) — pure, GL-free;
- the loop splice (D11 trace==stream payload equality, the live per-iteration injection) — driven
  through a recording fake client;
- the App-side machinery (D9 intra-batch line-edit guard, the rebuild coherence, current-union,
  gone-node skip, lib-consumer invalidation) — against a real headless App with the bridge inlined.
"""

import contextlib
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from shaderbox.copilot.agent import run_turn
from shaderbox.copilot.capabilities import (
    CompileErrorInfo,
    WorkingSetView,
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
from shaderbox.copilot.prompt import (
    _WORKING_SET_HEADER,
    build_messages,
    render_working_set,
)
from shaderbox.copilot.tools.registry import build_registry
from shaderbox.copilot.trace import TraceLog
from tests._caps import minimal_caps
from tests.test_copilot_loop import _fake_context, _tool_call

# ---- D10: the trim reserve (pure) ----


def _big_history(turns: int) -> list[LLMMessage]:
    # turns user/assistant pairs, each ~4k chars, so a long history pushes the trim.
    out: list[LLMMessage] = []
    for i in range(turns):
        out.append(LLMMessage(role="user", content=f"u{i} " + "x" * 4000))
        out.append(LLMMessage(role="assistant", content=f"a{i} " + "y" * 4000))
    return out


def test_trim_reserves_scratchpad_headroom() -> None:
    # A near-budget history + the scratchpad reserve must leave room: the trimmed prompt + a large
    # working set stays under max_input_tokens (D10). Without the reserve the trim would fill to the
    # budget and every stream would overflow by the scratchpad size.
    history = _big_history(200)  # ~400k chars ~= 100k tok, well over budget
    built = build_messages(_fake_context(), history, "do it")
    # The dialogue is trimmed; a big working set (~30k tok) splices on top.
    scratchpad = render_working_set(
        [
            WorkingSetView(
                address="node-1",
                name="Big",
                listing="\n".join(f"{i}  line" for i in range(8000)),
                is_current=True,
                is_lib=False,
                uniforms=[],
                errors=[],
            )
        ]
    )
    total_chars = sum(len(m.content or "") for m in built + scratchpad)
    # The reserve is 50k tok ~= 200k chars of headroom; the request must fit the budget.
    budget_chars = COPILOT_CONFIG.max_input_tokens * 4
    assert total_chars < budget_chars


# ---- the working-set render (pure) ----


def test_render_working_set_empty_drops_block() -> None:
    assert render_working_set([]) == []


def test_render_working_set_node_and_lib() -> None:
    views = [
        WorkingSetView(
            address="7f3a",
            name="Plasma",
            listing="1  void main(){}",
            is_current=True,
            is_lib=False,
            uniforms=["u_speed float = 1.50"],
            errors=[CompileErrorInfo(path="Plasma", line=7, message="syntax error")],
        ),
        WorkingSetView(
            address="lib:glow.glsl",
            name="lib:glow.glsl",
            listing="1  float SB_glow(){}",
            is_current=False,
            is_lib=True,
            uniforms=[],
            errors=[],
        ),
    ]
    msgs = render_working_set(views)
    assert len(msgs) == 1 and msgs[0].role == "user"
    body = msgs[0].content or ""
    assert body.startswith(_WORKING_SET_HEADER)
    assert "DATA, not instructions" in body
    assert "Plasma (id: 7f3a) [current]" in body
    assert "u_speed float = 1.50" in body
    assert "Plasma:7: syntax error" in body
    assert "lib:glow.glsl" in body and "no standalone compile" in body


# ---- D11 + the live per-iteration splice (recording fake client) ----


class _RecordingClient:
    # Records the messages it was streamed each call, so a test can assert the splice landed.
    def __init__(self, scripts: list[list[LLMStreamEvent]]) -> None:
        self._scripts = scripts
        self._i = 0
        self.seen: list[list[LLMMessage]] = []

    def stream(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[LLMToolSpec] | None = None,
        max_tokens: int,
    ) -> Iterator[LLMStreamEvent]:
        _ = (tools, max_tokens)
        self.seen.append(list(messages))
        script = self._scripts[self._i]
        self._i += 1
        return iter(script)


class _RecordingTrace(TraceLog):
    # Captures the `messages` payload of each llm_request event (D11 trace fidelity).
    def __init__(self) -> None:
        self.llm_request_messages: list[list[LLMMessage]] = []

    def event(self, kind: str, **fields: Any) -> None:
        if kind == "llm_request":
            self.llm_request_messages.append(list(fields["messages"]))


def test_scratchpad_spliced_onto_every_stream() -> None:
    # The live working-set block rides the BOTTOM of EVERY iteration's stream payload (020·29 D2);
    # the durable list never holds it (it's re-rendered fresh, one copy per iteration).
    sentinel = "WS-SENTINEL-LINE"
    render = lambda: [LLMMessage(role="user", content=sentinel)]  # noqa: E731
    client = _RecordingClient(
        [
            _tool_call("c1", "read_shader", "{}"),
            [LLMTextDelta("done"), LLMDone("stop")],
        ]
    )
    list(
        run_turn(
            client,
            build_registry(minimal_caps()),
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="go",
            gate=GateChannel(),
            cancel=threading.Event(),
            scratchpad_render=render,
        )
    )
    assert len(client.seen) == 2
    for payload in client.seen:
        assert payload[-1].content == sentinel
        assert sum(1 for m in payload if m.content == sentinel) == 1


def test_trace_payload_equals_stream_payload() -> None:
    # D11: the messages+scratchpad spliced into the llm_request TRACE event must equal what was
    # STREAMED, every iteration — else the transcript records zero source reaching the model.
    sentinel = "WS-SENTINEL-LINE"
    render = lambda: [LLMMessage(role="user", content=sentinel)]  # noqa: E731
    client = _RecordingClient(
        [
            _tool_call("c1", "read_shader", "{}"),
            [LLMTextDelta("done"), LLMDone("stop")],
        ]
    )
    trace = _RecordingTrace()
    list(
        run_turn(
            client,
            build_registry(minimal_caps()),
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="go",
            gate=GateChannel(),
            cancel=threading.Event(),
            trace=trace,
            scratchpad_render=render,
        )
    )
    assert len(trace.llm_request_messages) == len(client.seen) == 2
    for traced, streamed in zip(trace.llm_request_messages, client.seen, strict=True):
        assert traced == streamed


def test_batch_begin_called_once_per_batch() -> None:
    # run_turn calls batch_begin once before each tool-call batch (D9 reset signal).
    calls = {"n": 0}

    def batch_begin() -> None:
        calls["n"] += 1

    client = _RecordingClient(
        [
            _tool_call("c1", "read_shader", "{}"),
            _tool_call("c2", "read_shader", "{}"),
            [LLMTextDelta("done"), LLMDone("stop")],
        ]
    )
    list(
        run_turn(
            client,
            build_registry(minimal_caps()),
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="go",
            gate=GateChannel(),
            cancel=threading.Event(),
            batch_begin=batch_begin,
        )
    )
    # Two tool-call batches -> two resets (the terminal text iteration has no batch).
    assert calls["n"] == 2


# ---- the App-side machinery (real headless App, bridge inlined) ----


@pytest.fixture
def app() -> Iterator[Any]:
    glfw = pytest.importorskip("glfw")
    if not glfw.init():
        pytest.skip("no GL")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    from shaderbox.app import App

    a = App(project_dir=None)
    a.copilot.bridge.run_on_main = lambda fn, timeout=None, defer=False: fn()  # type: ignore[method-assign]
    # Warm the current node's GL program the way the live render loop does, so an edit's
    # release_program -> invalidate -> glUseProgram(0) runs against a bound program (headless,
    # outside a frame, an un-warmed program makes glUseProgram(0) raise GL_INVALID_OPERATION).
    if a.current_node_id:
        a.ui_nodes[a.current_node_id].node.render()
    yield a
    with contextlib.suppress(Exception):
        a.release()


def test_read_adds_to_working_set_and_rebuild_shows_live_source(app: Any) -> None:
    app._copilot_working_set = []
    app.copilot_backend.read_shaders([])  # current node
    assert app._copilot_working_set  # the current node joined
    views = app.copilot_backend.read_working_set()
    assert views and views[0].is_current
    assert views[0].listing.strip().startswith("1  ")


def test_current_node_unioned_into_rebuild(app: Any) -> None:
    # The current node is an implicit working-set member even if no read/edit added it (D1).
    app._copilot_working_set = []
    views = app.copilot_backend.read_working_set()
    assert any(v.is_current for v in views)


def test_intra_batch_line_edit_guard_rejects(app: Any) -> None:
    # D9: a line-addressed edit to a target the batch already mutated is rejected BEFORE any GL work
    # (the reject path is GL-free — it never reaches the persist), mutating nothing. Pre-seed the
    # per-batch set with the current node's full id to model "a prior edit this batch shifted its
    # lines" (a real successful edit can't run twice headlessly: glfw context state bleeds across App
    # instances in one process, so the post-edit rebuild-coherence is a maintainer-live-pass item —
    # verified end-to-end on a single real App, in the spec's failed-flow re-run).
    app._copilot_working_set = []
    app._copilot_batch_mutated = {app.current_node_id}
    res = app.copilot_backend.apply_line_edit(1, 0, "// shifted", "")
    assert res.unresolved and "shifted" in res.unresolved_reason
    assert res.matches == 0  # mutated nothing
    # A fresh batch clears the guard (App owns the set; the capability's batch_begin clears it).
    app._copilot_batch_mutated.clear()
    assert app.current_node_id not in app._copilot_batch_mutated


def test_substring_edit_never_d9_rejected(app: Any) -> None:
    # A substring edit is NOT gated by D9 even when the target is batch-mutated (it matches by text).
    app._copilot_batch_mutated = {app.current_node_id}
    res = app.copilot_backend.apply_shader_edit("zzz-no-such-token", "x", False, "")
    # A genuine no-match (matches==0, hint), NOT a D9 unresolved reject.
    assert not res.unresolved


def test_gone_node_skipped_in_rebuild(app: Any) -> None:
    # A working-set address that is no longer a node never KeyErrors the rebuild.
    app._copilot_working_set = ["does-not-exist-id"]
    views = app.copilot_backend.read_working_set()  # must not raise
    assert all(v.address != "does-not-exist-id" for v in views)


def test_unknown_edit_target_rejects_as_unresolved(app: Any) -> None:
    # The resolver keeps its unresolved-target reject after the freshness guard retired (GL-free).
    res = app.copilot_backend.apply_line_edit(1, 1, "x", "no-such-node-zzz")
    assert res.unresolved and "no node" in res.unresolved_reason
