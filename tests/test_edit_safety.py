"""Copilot edit-safety — the source-freshness guard (feature 020 · 15, Half B).

GL-free: drives run_turn against a fake App that models the freshness stamp/check
(_copilot_freshness_reject + the stamp/re-stamp). Asserts a stale edit is rejected with a
`stale` marker, mutates nothing, and is kept OUT of the edit-retry cap. The editor-lock half
(A) is GL/imgui-bound and verified in-app + by smoke, not here.
"""

import threading
import types
from collections.abc import Iterator

from shaderbox.app import App, _shader_revision
from shaderbox.copilot.agent import AgentError, AgentToolCard, run_turn
from shaderbox.copilot.capabilities import (
    CopilotCapabilities,
    CurrentShaderView,
    EditResult,
)
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.context import CopilotContext
from shaderbox.copilot.gate import GateChannel
from shaderbox.copilot.llm.api import (
    LLMDone,
    LLMMessage,
    LLMStreamEvent,
    LLMTextDelta,
    LLMToolSpec,
)
from shaderbox.copilot.tools.registry import build_registry
from tests.test_copilot_loop import _FakeClient, _tool_call


def _revision(node_id: str, text: str) -> tuple[str, str]:
    return (node_id, text)  # text itself is a fine content key for the fake


class _FreshnessApp:
    # Models App's freshness guard: get_current_shader stamps (node_id, text); a mutating
    # tool rejects unless the stamp matches the LIVE (current_node_id, source). Mirrors
    # app.py::_copilot_freshness_reject + the stamp/re-stamp in the apply closures.
    def __init__(self) -> None:
        self.current_node_id: str = "node-1"
        self.sources: dict[str, str] = {
            "node-1": "vec3 p = u_pos;",
            "node-2": "float x;",
        }
        self._read_revision: tuple[str, str] | None = None

    def view(self) -> CurrentShaderView | None:
        nid = self.current_node_id
        if nid not in self.sources:
            return None
        text = self.sources[nid]
        self._read_revision = _revision(nid, text)
        return CurrentShaderView(
            text=text, listing=f"1  {text}", uniforms=[], errors=[]
        )

    def _freshness_reject(self) -> EditResult | None:
        nid = self.current_node_id
        rev = self._read_revision
        if rev is None:
            return EditResult(
                matches=0, errors=[], stale=True, stale_reason="read first"
            )
        if nid not in self.sources or rev[0] != nid:
            return EditResult(
                matches=0, errors=[], stale=True, stale_reason="you switched nodes"
            )
        if rev != _revision(nid, self.sources[nid]):
            return EditResult(
                matches=0, errors=[], stale=True, stale_reason="source changed"
            )
        return None

    def apply_shader_edit(
        self, old_str: str, new_str: str, replace_all: bool
    ) -> EditResult:
        if (reject := self._freshness_reject()) is not None:
            return reject
        nid = self.current_node_id
        src = self.sources[nid]
        if old_str not in src:
            return EditResult(matches=0, errors=[])
        new_text = src.replace(old_str, new_str)
        self.sources[nid] = new_text
        self._read_revision = _revision(nid, new_text)  # re-stamp on success
        return EditResult(matches=1, errors=[])

    def apply_line_edit(
        self, start_line: int, end_line: int, new_text: str
    ) -> EditResult:
        if (reject := self._freshness_reject()) is not None:
            return reject
        nid = self.current_node_id
        self.sources[nid] = new_text
        self._read_revision = _revision(nid, new_text)
        return EditResult(matches=1, errors=[])

    def caps(self) -> CopilotCapabilities:
        return CopilotCapabilities(
            list_nodes=lambda: [],
            get_node_summary=lambda _nid: None,
            get_shader_source=lambda _nid: None,
            get_compile_errors=lambda _nid: [],
            current_node_id=lambda: self.current_node_id,
            get_current_shader_view=self.view,
            apply_shader_edit=self.apply_shader_edit,
            apply_line_edit=self.apply_line_edit,
            get_compile_errors_current=lambda: [],
        )


def _run(app: _FreshnessApp, scripts: list[list[LLMStreamEvent]]) -> list:
    return list(
        run_turn(
            _FakeClient(scripts),
            build_registry(app.caps()),
            COPILOT_CONFIG,
            CopilotContext(current_node_id="node-1"),
            history=[],
            user_text="edit",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )


def _read() -> list[LLMStreamEvent]:
    return _tool_call("r", "get_current_shader", "{}")


def _edit(old: str, new: str) -> list[LLMStreamEvent]:
    return _tool_call("e", "edit_shader", f'{{"old_str": "{old}", "new_str": "{new}"}}')


def test_never_read_is_rejected_as_stale() -> None:
    # A mutating tool as the FIRST action (no get_current_shader) → stale reject, no mutation.
    app = _FreshnessApp()
    events = _run(
        app,
        [_edit("vec3 p = u_pos;", "X"), [LLMTextDelta("ok"), LLMDone("stop")]],
    )
    cards = [e for e in events if isinstance(e, AgentToolCard)]
    assert cards[0].name == "edit_shader"
    assert cards[0].ok is False
    assert cards[0].payload == {"stale": True}
    assert app.sources["node-1"] == "vec3 p = u_pos;"  # unchanged


def test_content_drift_then_reread_succeeds() -> None:
    # read → (source mutated out-of-band) → edit rejects stale → read → edit succeeds.
    app = _FreshnessApp()

    class _DriftClient:
        def __init__(self) -> None:
            self._i = 0
            self._scripts = [
                _read(),
                _edit("vec3 p = u_pos;", "Y"),  # rejected (drift injected below)
                _read(),
                _edit("vec3 p = u_pos;", "Y"),  # now succeeds
                [LLMTextDelta("done"), LLMDone("stop")],
            ]

        def stream(
            self,
            messages: list[LLMMessage],
            *,
            tools: list[LLMToolSpec] | None = None,
            max_tokens: int,
        ) -> Iterator[LLMStreamEvent]:
            _ = (messages, tools, max_tokens)
            # Inject a drift right before the FIRST edit attempt (after the first read).
            if self._i == 1:
                app.sources["node-1"] = "vec3 p = u_pos; // touched"
            script = self._scripts[self._i]
            self._i += 1
            return iter(script)

    events = list(
        run_turn(
            _DriftClient(),
            build_registry(app.caps()),
            COPILOT_CONFIG,
            CopilotContext(current_node_id="node-1"),
            history=[],
            user_text="edit",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )
    cards = [e for e in events if isinstance(e, AgentToolCard)]
    # read, edit(stale-reject), read, edit(ok)
    assert [c.name for c in cards] == [
        "get_current_shader",
        "edit_shader",
        "get_current_shader",
        "edit_shader",
    ]
    assert cards[1].ok is False and cards[1].payload == {"stale": True}
    assert cards[3].ok is True
    assert app.sources["node-1"] == "Y // touched"


def test_identity_drift_rejected() -> None:
    # read node-1 → (switch to node-2) → edit rejects with the switched-nodes reason.
    app = _FreshnessApp()

    class _SwitchClient:
        def __init__(self) -> None:
            self._i = 0
            self._scripts = [
                _read(),
                _edit("vec3 p = u_pos;", "Z"),
                [LLMTextDelta("done"), LLMDone("stop")],
            ]

        def stream(
            self,
            messages: list[LLMMessage],
            *,
            tools: list[LLMToolSpec] | None = None,
            max_tokens: int,
        ) -> Iterator[LLMStreamEvent]:
            _ = (messages, tools, max_tokens)
            if self._i == 1:
                app.current_node_id = "node-2"  # switch between read and edit
            script = self._scripts[self._i]
            self._i += 1
            return iter(script)

    events = list(
        run_turn(
            _SwitchClient(),
            build_registry(app.caps()),
            COPILOT_CONFIG,
            CopilotContext(current_node_id="node-1"),
            history=[],
            user_text="edit",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )
    cards = [e for e in events if isinstance(e, AgentToolCard)]
    assert cards[-1].name == "edit_shader"
    assert cards[-1].ok is False and cards[-1].payload == {"stale": True}
    assert app.sources["node-2"] == "float x;"  # node-2 untouched


def test_chain_after_success_without_reread() -> None:
    # read → edit (success, re-stamps) → second edit WITHOUT a read → succeeds.
    app = _FreshnessApp()
    events = _run(
        app,
        [
            _read(),
            _edit("vec3 p = u_pos;", "A"),
            _edit("A", "B"),
            [LLMTextDelta("done"), LLMDone("stop")],
        ],
    )
    cards = [e for e in events if isinstance(e, AgentToolCard)]
    assert [c.ok for c in cards] == [True, True, True]
    assert app.sources["node-1"] == "B"


def test_stale_rejects_do_not_trip_retry_cap() -> None:
    # A run of genuine STALE rejects (source drifts under each edit) must NOT accumulate
    # toward max_edit_retries — a stale reject is a benign re-read signal, not a failed edit.
    app = _FreshnessApp()
    n = COPILOT_CONFIG.max_edit_retries + 2  # more stale edits than the cap

    class _AlwaysStaleClient:
        def __init__(self) -> None:
            self._i = 0
            self._scripts: list[list[LLMStreamEvent]] = [_read()]
            self._scripts += [_edit("vec3 p = u_pos;", "Q")] * n
            self._scripts.append([LLMTextDelta("done"), LLMDone("stop")])

        def stream(
            self,
            messages: list[LLMMessage],
            *,
            tools: list[LLMToolSpec] | None = None,
            max_tokens: int,
        ) -> Iterator[LLMStreamEvent]:
            _ = (messages, tools, max_tokens)
            # After the read (i==0), drift the source before EVERY edit so each one is stale.
            if self._i >= 1:
                app.sources["node-1"] = f"vec3 p = u_pos; // {self._i}"
            script = self._scripts[self._i]
            self._i += 1
            return iter(script)

    events = list(
        run_turn(
            _AlwaysStaleClient(),
            build_registry(app.caps()),
            COPILOT_CONFIG,
            CopilotContext(current_node_id="node-1"),
            history=[],
            user_text="edit",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )
    stale_cards = [
        e
        for e in events
        if isinstance(e, AgentToolCard) and e.payload == {"stale": True}
    ]
    # MORE stale rejects than the cap occurred, yet no edit-giveup AgentError fired.
    assert len(stale_cards) > COPILOT_CONFIG.max_edit_retries
    assert not any(
        isinstance(e, AgentError) and "couldn't apply" in e.message for e in events
    )


def test_malformed_mutating_args_do_not_crash_cap() -> None:
    # Regression (review BLOCKER): a mutating tool with malformed args → registry.execute
    # returns payload=None → the cap predicate must null-guard (payload or {}).get(...).
    app = _FreshnessApp()
    bad = _tool_call(
        "b", "edit_shader", '{"old_str": 123}'
    )  # missing new_str, wrong type
    events = _run(app, [bad, [LLMTextDelta("done"), LLMDone("stop")]])
    cards = [e for e in events if isinstance(e, AgentToolCard)]
    assert cards[0].ok is False  # validation error, did not crash the worker
    assert (
        not isinstance(events[-1], AgentError)
        or "stopped" in events[-1].message.lower()
    )


# ---- the REAL app.py freshness primitives (pure, GL-free) ----
def _node_stub(text: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        node=types.SimpleNamespace(source=types.SimpleNamespace(text=text))
    )


def _app_stub(node_id: str, sources: dict[str, str], rev) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        current_node_id=node_id,
        ui_nodes={k: _node_stub(v) for k, v in sources.items()},
        _copilot_read_revision=rev,
    )


def test_shader_revision_is_identity_plus_content() -> None:
    # Same (id, text) -> equal; different text -> different; different node -> different.
    assert _shader_revision("n", "abc") == _shader_revision("n", "abc")
    assert _shader_revision("n", "abc") != _shader_revision("n", "abcd")
    assert _shader_revision("n1", "abc") != _shader_revision("n2", "abc")
    assert _shader_revision("n", "x")[0] == "n"  # identity preserved


def test_real_freshness_reject_branches() -> None:
    # Exercise the REAL App._copilot_freshness_reject (not the fake) on a minimal stand-in,
    # so a future reorder of its branches/short-circuit is caught.
    src = "vec3 p = u_pos;"
    fresh = _shader_revision("n1", src)

    # None stamp -> read-first reject.
    stub = _app_stub("n1", {"n1": src}, None)
    r = App._copilot_freshness_reject(stub)  # type: ignore[arg-type]
    assert r is not None and r.stale and "before editing" in r.stale_reason

    # Stamp matches live source -> fresh (None = proceed).
    stub = _app_stub("n1", {"n1": src}, fresh)
    assert App._copilot_freshness_reject(stub) is None  # type: ignore[arg-type]

    # Node switched (stamp node != current) -> identity reject; must NOT KeyError even if
    # the current node is absent.
    stub = _app_stub("n2", {"n1": src}, fresh)
    r = App._copilot_freshness_reject(stub)  # type: ignore[arg-type]
    assert r is not None and r.stale and "switched" in r.stale_reason

    # Same node, content drifted -> content reject.
    stub = _app_stub("n1", {"n1": src + " // touched"}, fresh)
    r = App._copilot_freshness_reject(stub)  # type: ignore[arg-type]
    assert r is not None and r.stale and "changed since you read it" in r.stale_reason
