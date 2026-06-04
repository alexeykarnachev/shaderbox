"""Copilot edit-safety — the source-freshness guard (feature 020 · 15, Half B).

GL-free: drives run_turn against a fake App that models the freshness stamp/check
(_copilot_freshness_reject + the stamp/re-stamp). Asserts a stale edit is rejected with a
`stale` marker, mutates nothing, and is kept OUT of the edit-retry cap. The editor-lock half
(A) is GL/imgui-bound and verified in-app + by smoke, not here.
"""

import threading
import types
from collections.abc import Iterator

from shaderbox.app import App, _shader_digest
from shaderbox.copilot.agent import AgentError, AgentToolCard, run_turn
from shaderbox.copilot.capabilities import (
    CopilotCapabilities,
    EditResult,
    ShaderView,
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
from shaderbox.copilot.tools.registry import build_registry
from tests._caps import minimal_caps
from tests.test_copilot_loop import _fake_context, _FakeClient, _tool_call


class _FreshnessApp:
    # Models App's PER-NODE freshness guard (feature 020·16 Decision 2): read_shader stamps
    # _read_revision[node_id]=text; a mutating tool on a resolved target rejects unless the
    # node's stamp matches its LIVE source. Mirrors app.py::_copilot_freshness_reject + the
    # per-node stamp/re-stamp. The edit caps resolve target ""->current, like prod.
    def __init__(self) -> None:
        self.current_node_id: str = "node-1"
        self.sources: dict[str, str] = {
            "node-1": "vec3 p = u_pos;",
            "node-2": "float x;",
        }
        self._read_revision: dict[str, str] = {}

    def read_shaders(self, node_ids: list[str]) -> list[ShaderView]:
        ids = node_ids or [self.current_node_id]
        views: list[ShaderView] = []
        for nid in ids:
            if nid not in self.sources:
                continue
            text = self.sources[nid]
            self._read_revision[nid] = text
            views.append(
                ShaderView(
                    node_id=nid, name=nid, listing=f"1  {text}", uniforms=[], errors=[]
                )
            )
        return views

    def _freshness_reject(self, node_id: str) -> EditResult | None:
        stamped = self._read_revision.get(node_id)
        if stamped is None:
            reason = "you switched nodes" if self._read_revision else "read first"
            return EditResult(matches=0, errors=[], stale=True, stale_reason=reason)
        if node_id not in self.sources:
            return EditResult(
                matches=0, errors=[], stale=True, stale_reason="no longer exists"
            )
        if stamped != self.sources[node_id]:
            return EditResult(
                matches=0, errors=[], stale=True, stale_reason="source changed"
            )
        return None

    def apply_shader_edit(
        self, old_str: str, new_str: str, replace_all: bool, target: str
    ) -> EditResult:
        nid = target or self.current_node_id
        if (reject := self._freshness_reject(nid)) is not None:
            return reject
        src = self.sources[nid]
        if old_str not in src:
            return EditResult(matches=0, errors=[])
        new_text = src.replace(old_str, new_str)
        self.sources[nid] = new_text
        self._read_revision[nid] = new_text  # re-stamp the TARGET on success
        return EditResult(matches=1, errors=[])

    def apply_line_edit(
        self, start_line: int, end_line: int, new_text: str, target: str
    ) -> EditResult:
        nid = target or self.current_node_id
        if (reject := self._freshness_reject(nid)) is not None:
            return reject
        self.sources[nid] = new_text
        self._read_revision[nid] = new_text
        return EditResult(matches=1, errors=[])

    def caps(self) -> CopilotCapabilities:
        return minimal_caps(
            read_shaders=self.read_shaders,
            apply_shader_edit=self.apply_shader_edit,
            apply_line_edit=self.apply_line_edit,
        )


def _run(app: _FreshnessApp, scripts: list[list[LLMStreamEvent]]) -> list:
    return list(
        run_turn(
            _FakeClient(scripts),
            build_registry(app.caps()),
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="edit",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )


def _read() -> list[LLMStreamEvent]:
    return _tool_call("r", "read_shader", "{}")


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
            _fake_context(),
            history=[],
            user_text="edit",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )
    cards = [e for e in events if isinstance(e, AgentToolCard)]
    # read, edit(stale-reject), read, edit(ok)
    assert [c.name for c in cards] == [
        "read_shader",
        "edit_shader",
        "read_shader",
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
            _fake_context(),
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
            _fake_context(),
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


# ---- the REAL app.py per-node freshness primitives (pure, GL-free; feature 020·16) ----
def _node_stub(text: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        node=types.SimpleNamespace(source=types.SimpleNamespace(text=text))
    )


def _app_stub(
    node_id: str, sources: dict[str, str], rev: dict[str, bytes]
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        current_node_id=node_id,
        ui_nodes={k: _node_stub(v) for k, v in sources.items()},
        _copilot_read_revision=rev,
    )


def test_shader_digest_is_content_only() -> None:
    # Per-node keying (020·16 Decision 2): the digest is content-only — identity is the dict
    # key, not part of the token. Same text -> equal; different text -> different.
    assert _shader_digest("abc") == _shader_digest("abc")
    assert _shader_digest("abc") != _shader_digest("abcd")


def test_real_freshness_reject_branches() -> None:
    # Exercise the REAL App._copilot_freshness_reject (not the fake) on a minimal stand-in,
    # keyed by the resolved TARGET node id (020·16 Decision 2/2c), so a future reorder of its
    # branches/short-circuit is caught.
    src = "vec3 p = u_pos;"
    fresh = _shader_digest(src)

    # Empty dict -> never-read reject (read-first).
    stub = _app_stub("n1", {"n1": src}, {})
    r = App._copilot_freshness_reject(stub, "n1")  # type: ignore[arg-type]
    assert r is not None and r.stale and "before editing" in r.stale_reason

    # Target stamped + matches live source -> fresh (None = proceed).
    stub = _app_stub("n1", {"n1": src}, {"n1": fresh})
    assert App._copilot_freshness_reject(stub, "n1") is None  # type: ignore[arg-type]

    # Read a DIFFERENT node, edit n1 (pin 2c) -> "switched nodes" reject; no KeyError.
    stub = _app_stub("n1", {"n1": src}, {"n2": _shader_digest("float x;")})
    r = App._copilot_freshness_reject(stub, "n1")  # type: ignore[arg-type]
    assert r is not None and r.stale and "switched nodes" in r.stale_reason

    # Stamped but the node was deleted -> "no longer exists".
    stub = _app_stub("n1", {}, {"n1": fresh})
    r = App._copilot_freshness_reject(stub, "n1")  # type: ignore[arg-type]
    assert r is not None and r.stale and "no longer exists" in r.stale_reason

    # Same node, content drifted -> content reject.
    stub = _app_stub("n1", {"n1": src + " // touched"}, {"n1": fresh})
    r = App._copilot_freshness_reject(stub, "n1")  # type: ignore[arg-type]
    assert r is not None and r.stale and "changed since you read it" in r.stale_reason
