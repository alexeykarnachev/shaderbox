"""Feature 083: the copilot's per-session source lock.

A locked session confirms each project-changing call through the SAME gate funnel every
destructive tool already uses; an unlocked one asks nothing. The three answers differ in what
they do to the lock AFTER approving, which is why the answer is not a bool. Deterministic —
scripted fake client, stubbed backend, a gate answered from a helper thread, no GL.
"""

import threading
from typing import Any

from shaderbox.copilot.agent import AgentGateOpened, run_turn
from shaderbox.copilot.capabilities import EditResult
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.gate import GateChannel, GateKind, GateResponse, LockAnswer
from shaderbox.copilot.llm.api import LLMDone, LLMStreamEvent, LLMTextDelta
from shaderbox.copilot.tools.registry import build_registry
from tests._caps import minimal_caps
from tests.test_copilot_loop import _fake_context, _FakeClient, _tool_call

_EDIT = _tool_call(
    "c1", "edit_shader", '{"old_str": "a", "new_str": "b", "target": "7f3a"}'
)
_DONE = [LLMTextDelta("done"), LLMDone("stop")]


def _answer_with(
    gate: GateChannel, answers: list[GateResponse], opened: list[int]
) -> threading.Thread:
    # The worker blocks in gate.ask(); the UI answers on the main thread. Here a helper thread
    # plays the UI, popping one prepared answer per request.
    #
    # It keeps answering past the prepared list, with DENY. That matters: a regression that opens
    # MORE gates than expected would otherwise leave the worker blocked on a request nobody
    # answers, and the test would hang forever instead of failing -- which is exactly what
    # happened while mutation-testing this file. A count assert then reports the surplus.
    def pump() -> None:
        while not _stop.is_set():
            if gate.take_pending() is None:
                continue
            i = opened[0]
            opened[0] = i + 1
            gate.answer(
                answers[i]
                if i < len(answers)
                else GateResponse(False, lock_answer=LockAnswer.DENY)
            )

    _stop = threading.Event()
    t = threading.Thread(target=pump, daemon=True)
    t.stop_flag = _stop  # type: ignore[attr-defined]
    t.start()
    return t


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
    opened_count = [0]
    thread = _answer_with(gate, answers, opened_count)
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
    thread.stop_flag.set()  # type: ignore[attr-defined]
    thread.join(timeout=5.0)
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
