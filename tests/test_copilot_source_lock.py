"""Feature 083: the copilot's per-session source lock.

A locked session confirms each project-changing call through the SAME gate funnel every
destructive tool already uses; an unlocked one asks nothing. The three answers differ in what
they do to the lock AFTER approving, which is why the answer is not a bool. Deterministic —
scripted fake client, stubbed backend, a gate answered from a helper thread, no GL.
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shaderbox.copilot.agent import AgentGateOpened, run_turn
from shaderbox.copilot.capabilities import EditResult
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.gate import GateChannel, GateKind, GateResponse, LockAnswer
from shaderbox.copilot.llm.api import LLMDone, LLMStreamEvent, LLMTextDelta
from shaderbox.copilot.session import CopilotSession
from shaderbox.copilot.tools.registry import build_registry
from tests._caps import minimal_caps
from tests.test_copilot_loop import _fake_context, _FakeClient, _tool_call

_EDIT = _tool_call(
    "c1", "edit_shader", '{"old_str": "a", "new_str": "b", "target": "7f3a"}'
)
_DONE = [LLMTextDelta("done"), LLMDone("stop")]


@dataclass
class _GatePump:
    """The UI half of the gate, on its own thread: it answers what the worker blocks on."""

    thread: threading.Thread
    stop: threading.Event
    opened: list[int]

    def finish(self) -> int:
        self.stop.set()
        self.thread.join(timeout=5.0)
        return self.opened[0]


def _answer_with(gate: GateChannel, answers: list[GateResponse]) -> _GatePump:
    # The worker blocks in gate.ask(); the UI answers on the main thread. Here a helper thread
    # plays the UI, popping one prepared answer per request.
    #
    # It keeps answering past the prepared list, with DENY. That matters: a regression that opens
    # MORE gates than expected would otherwise leave the worker blocked on a request nobody
    # answers, and the test would hang forever instead of failing -- which is exactly what
    # happened while mutation-testing this file. A count assert then reports the surplus.
    opened = [0]
    stop = threading.Event()

    def pump() -> None:
        while not stop.is_set():
            if gate.take_pending() is None:
                # A bare spin burns a core for the whole turn (measured: ~600k iterations in half
                # a second of idle). The sleep is shorter than a gate round-trip, so it costs the
                # tests nothing.
                time.sleep(0.001)
                continue
            i = opened[0]
            opened[0] = i + 1
            gate.answer(
                answers[i]
                if i < len(answers)
                else GateResponse(False, lock_answer=LockAnswer.DENY)
            )

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    return _GatePump(thread=thread, stop=stop, opened=opened)


def _run(
    scripts: list[list[LLMStreamEvent]],
    *,
    locked: bool,
    answers: list[GateResponse],
    **caps_overrides: Any,
) -> tuple[list[Any], list[tuple[str, str]], Any]:
    edits: list[tuple[str, str]] = []

    def _record(old: str, new: str, replace_all: bool, target: str) -> EditResult:
        # The capability the edit_shader tool calls. Recording HERE rather than reading the tool's
        # return is what makes "the edit never happened" checkable: a declined call must not reach
        # the backend at all, which a message-text assert could not tell from a failed one.
        edits.append((old, new))
        return EditResult(matches=1, errors=[])

    caps = minimal_caps(apply_shader_edit=_record, **caps_overrides)
    registry = build_registry(caps)
    registry.source_locked = locked
    gate = GateChannel()
    pump = _answer_with(gate, answers)
    events = list(
        run_turn(
            _FakeClient(scripts),
            registry,
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="drive it",
            gate=gate,
            cancel=threading.Event(),
            unlock_source=lambda: setattr(registry, "source_locked", False),
        )
    )
    pump.finish()
    return events, edits, registry


def _gates(events: list[Any]) -> list[AgentGateOpened]:
    return [e for e in events if isinstance(e, AgentGateOpened)]


def test_a_locked_session_confirms_a_source_edit_before_it_runs() -> None:
    # The falsifier for the whole feature: with the lock on and the answer DENY, the edit must not
    # reach the capability at all. Unlock (the next test) and the same script edits without asking.
    events, edits, _ = _run(
        [_EDIT, _DONE],
        locked=True,
        answers=[GateResponse(False, lock_answer=LockAnswer.DENY)],
    )
    opened = _gates(events)
    assert len(opened) == 1
    assert opened[0].request.kind is GateKind.SOURCE_LOCK
    assert not edits, "a denied edit must never reach the capability"


def test_an_unlocked_session_asks_nothing() -> None:
    # The complement, and the check that keeps the widened condition from becoming "gate always":
    # without it, a must_confirm that returned True unconditionally would pass the test above.
    events, edits, _ = _run([_EDIT, _DONE], locked=False, answers=[])
    assert not _gates(events)
    assert edits, "an unlocked session edits without asking"


def test_allow_this_session_stops_the_asking_and_allow_once_does_not() -> None:
    # The two answers that both approve THIS call and differ in everything after it. Folding them
    # into `approved: bool` makes these two runs identical, so this cannot pass under that design.
    events, edits, registry = _run(
        [_EDIT, _EDIT, _DONE],
        locked=True,
        answers=[GateResponse(True, lock_answer=LockAnswer.SESSION)],
    )
    assert len(_gates(events)) == 1, "the session unlock must not ask a second time"
    assert len(edits) == 2
    assert not registry.source_locked

    events, edits, registry = _run(
        [_EDIT, _EDIT, _DONE],
        locked=True,
        answers=[
            GateResponse(True, lock_answer=LockAnswer.ONCE),
            GateResponse(True, lock_answer=LockAnswer.ONCE),
        ],
    )
    assert len(_gates(events)) == 2, "allow-once approves one call, not the turn"
    assert len(edits) == 2
    assert registry.source_locked, "allow-once leaves the session locked"


def _session() -> CopilotSession:
    return CopilotSession(
        caps=minimal_caps(),
        client=_FakeClient([]),
        get_project_slug=lambda: "p",
        get_checkpoints_root=lambda: Path("/tmp"),
    )


def test_a_headless_session_is_locked_like_any_other() -> None:
    # The lock is a property of a SESSION, so a session built without an App has it too -- the
    # dogfood harness and any headless driver gate exactly as the app does. That is deliberate:
    # a headless default of "unlocked" would make the one place nobody is watching the one place
    # the copilot edits freely, and the harness already has auto_approve_gates for when that is
    # what is wanted. The cost is that a headless test calling a source tool must unlock or
    # answer, which is the right way round.
    session = _session()
    assert session.registry.must_confirm("set_uniform")
    assert not session.registry.must_confirm("read_shader")


def test_the_locks_two_fields_agree_at_every_lifecycle_point() -> None:
    # V4, and the check that was specified and then not written -- which is exactly how a session
    # shipped drawing a closed padlock over a registry that confirmed nothing. The two fields are
    # born from two DIFFERENT defaults (ChatState locked, a bare registry unlocked), so agreement
    # is something the one writer must establish, never something construction gives for free.
    #
    # Three lifecycle points, because each reaches the pair by a different route: construction
    # (two defaults meeting), an explicit toggle (the icon), and a reset (which rebuilds ChatState
    # and NOT the registry).
    session = _session()
    assert session.state.source_locked == session.registry.source_locked, (
        "born diverged: the icon and the gate disagree before anything has happened"
    )
    assert session.registry.must_confirm("edit_shader"), (
        "a fresh session draws a locked icon, so it must actually confirm"
    )

    session.set_source_locked(False)
    assert session.state.source_locked == session.registry.source_locked
    assert not session.registry.must_confirm("edit_shader")

    session.reset_conversation()
    assert session.state.source_locked == session.registry.source_locked, (
        "reset rebuilds ChatState but not the registry; without a re-seed they diverge"
    )
    assert session.registry.must_confirm("edit_shader"), (
        "a cleared chat is a new session, so it locks again"
    )
