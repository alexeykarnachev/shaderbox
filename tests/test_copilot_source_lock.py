"""Feature 083: the copilot's per-session source lock.

A locked session confirms each project-changing call through the SAME gate funnel every
destructive tool already uses; an unlocked one asks nothing. The three answers differ in what
they do to the lock AFTER approving, which is why the answer is not a bool. Deterministic —
scripted fake client, stubbed backend, a gate answered from a helper thread, no GL.
"""

import inspect
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shaderbox.copilot.agent import (
    AgentGateOpened,
    AgentToolCard,
    AgentTurnDone,
    run_turn,
)
from shaderbox.copilot.capabilities import EditResult
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.gate import (
    GateChannel,
    GateKind,
    GateResponse,
    LockAnswer,
    SourceLock,
)
from shaderbox.copilot.llm.api import (
    LLMDone,
    LLMMessage,
    LLMStreamEvent,
    LLMTextDelta,
)
from shaderbox.copilot.prompt import _context_block
from shaderbox.copilot.prompt_context import build_context
from shaderbox.copilot.session import _LOCK_OUTCOMES, CopilotSession
from shaderbox.copilot.tools.registry import build_registry
from shaderbox.ui_models import UIAppState
from shaderbox.widgets.copilot_chat import _LOCK_LABELS
from tests._caps import minimal_caps
from tests.test_copilot_loop import _fake_context, _FakeClient, _tool_call

_EDIT = _tool_call(
    "c1", "edit_shader", '{"old_str": "a", "new_str": "b", "target": "7f3a"}'
)
_RENDER = _tool_call("c0", "render_image", "{}")
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


class _Recording:
    """Wraps a scripted client and keeps each request's messages, so a test can assert what the
    NEXT request would have carried -- the only place a missing tool result is observable without
    a real provider to 400."""

    def __init__(self, inner: _FakeClient, sink: list[list[LLMMessage]]) -> None:
        self._inner = inner
        self._sink = sink

    def stream(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[Any] | None = None,
        max_tokens: int,
    ) -> Any:
        self._sink.append(list(messages))
        return self._inner.stream(messages, tools=tools, max_tokens=max_tokens)


def _run(
    scripts: list[list[LLMStreamEvent]],
    *,
    lock: SourceLock,
    answers: list[GateResponse],
    record_messages: list[list[LLMMessage]] | None = None,
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
    registry.source_lock = lock
    gate = GateChannel()
    pump = _answer_with(gate, answers)
    client = _FakeClient(scripts)
    events = list(
        run_turn(
            client if record_messages is None else _Recording(client, record_messages),
            registry,
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="drive it",
            gate=gate,
            cancel=threading.Event(),
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
        lock=SourceLock.ASK,
        answers=[GateResponse(False, lock_answer=LockAnswer.DENY)],
    )
    opened = _gates(events)
    assert len(opened) == 1
    assert opened[0].request.kind is GateKind.SOURCE_LOCK
    assert not edits, "a denied edit must never reach the capability"


def test_an_unlocked_session_asks_nothing() -> None:
    # The complement, and the check that keeps the widened condition from becoming "gate always":
    # without it, a must_confirm that returned True unconditionally would pass the test above.
    events, edits, _ = _run([_EDIT, _DONE], lock=SourceLock.ALLOW, answers=[])
    assert not _gates(events)
    assert edits, "an unlocked session edits without asking"


def test_allow_answers_one_call_and_leaves_the_mode_alone() -> None:
    # ALLOW is per-CALL: two edits, two answers, two gates. The mode is the top bar's, and no gate
    # answer changes it -- 086 removed the session-wide answer precisely because a blocking card is
    # the wrong place to set a horizon.
    events, edits, registry = _run(
        [_EDIT, _EDIT, _DONE],
        lock=SourceLock.ASK,
        answers=[
            GateResponse(True, lock_answer=LockAnswer.ALLOW),
            GateResponse(True, lock_answer=LockAnswer.ALLOW),
        ],
    )
    assert len(_gates(events)) == 2, "allow approves one call, not the turn"
    assert len(edits) == 2
    assert registry.source_lock is SourceLock.ASK, "an answer must not change the mode"


def test_each_answer_shows_the_user_what_they_chose() -> None:
    # The card's resolved text is the user's own record of the decision, so a swapped mapping
    # would tell them they denied a call they allowed. Enumerated so a new answer needs a token.
    assert set(_LOCK_OUTCOMES) == set(LockAnswer)
    assert _LOCK_OUTCOMES[LockAnswer.ALLOW] == "Yes"
    assert _LOCK_OUTCOMES[LockAnswer.DENY] == "No"


def test_the_card_offers_exactly_two_answers() -> None:
    # The vocabulary itself: a third member is a third button, and the horizon it would carry is
    # the mode's job. Pinned as a count so re-adding one is a deliberate act with a red test.
    assert {a.name for a in LockAnswer} == {"ALLOW", "DENY"}


def test_no_gate_answer_can_change_the_mode() -> None:
    # The structural half of the same rule: run_turn no longer takes a way to write the lock, so a
    # future answer cannot quietly re-acquire one. A signature check, because a deleted parameter
    # that is merely unused would leave the plumbing wired and green.
    assert "unlock_source" not in inspect.signature(run_turn).parameters


def test_one_deny_answers_for_the_rest_of_the_turn() -> None:
    # 085's headline: the user's single "no" covers the turn, not the call. Two edits in two
    # SEPARATE responses (each script entry is its own batch), so a latch wrongly scoped per-batch
    # fails here while a single-batch test would pass either way.
    events, edits, _ = _run(
        [_EDIT, _EDIT, _DONE],
        lock=SourceLock.ASK,
        answers=[GateResponse(False, lock_answer=LockAnswer.DENY)],
    )
    assert len(_gates(events)) == 1, "a denied turn must not ask again"
    assert not edits, "neither edit may reach the capability"


def test_the_deny_latch_dies_with_its_turn() -> None:
    # The latch is turn state, not session state. Two turns on ONE registry: the first is denied,
    # the second must ask for itself. A latch hoisted onto the registry or ChatState passes every
    # other test in this file and fails only here.
    edits: list[tuple[str, str]] = []

    def _record(old: str, new: str, replace_all: bool, target: str) -> EditResult:
        edits.append((old, new))
        return EditResult(matches=1, errors=[])

    registry = build_registry(minimal_caps(apply_shader_edit=_record))
    registry.source_lock = SourceLock.ASK
    gate = GateChannel()
    pump = _answer_with(gate, [GateResponse(False, lock_answer=LockAnswer.DENY)])
    gates_per_turn: list[int] = []
    for _ in range(2):
        events = list(
            run_turn(
                _FakeClient([_EDIT, _DONE]),
                registry,
                COPILOT_CONFIG,
                _fake_context(),
                history=[],
                user_text="drive it",
                gate=gate,
                cancel=threading.Event(),
            )
        )
        gates_per_turn.append(len(_gates(events)))
    pump.finish()
    assert gates_per_turn == [1, 1], "the second turn asks for itself"
    assert not edits, "the pump denies both turns, so no edit ever lands"


def test_a_confirm_no_on_another_tool_does_not_latch_the_source_lock() -> None:
    # The latch's discriminator is `lock_answer is DENY`, not `not approved` -- and those are
    # different sets. An always-gated tool (render_image) answered plain "No" lands in
    # `not approved` carrying no lock_answer; latching there would let one refused render
    # silence the source lock for the rest of the turn, a decision the user never made.
    events, edits, _ = _run(
        [_RENDER, _EDIT, _DONE],
        lock=SourceLock.ASK,
        answers=[
            GateResponse(False),
            GateResponse(True, lock_answer=LockAnswer.ALLOW),
        ],
    )
    opened = _gates(events)
    assert len(opened) == 2, "the edit must still get its own question"
    assert opened[0].request.kind is GateKind.CONFIRM
    assert opened[1].request.kind is GateKind.SOURCE_LOCK
    assert len(edits) == 1, "the approved edit still runs"


def test_an_armed_lock_declines_without_asking() -> None:
    # The maintainer's second sentence: a lock he set deliberately is already the answer. No gate
    # card, no edit. ASK (the session default) still asks -- that is the other tests above.
    events, edits, _ = _run([_EDIT, _DONE], lock=SourceLock.READ_ONLY, answers=[])
    assert not _gates(events), "an armed lock is the answer; asking re-asks it"
    assert not edits


def test_a_call_declined_without_asking_still_returns_a_tool_result() -> None:
    # An assistant message whose tool_call_id has no matching tool result makes the provider 400
    # the NEXT stream, so a refuse path that records only a card breaks the turn instead of
    # declining a call. A fake client cannot 400, so the assert reads what the second request
    # actually carried: every tool_call_id the assistant emitted must be answered. Asserting the
    # card alone would pass with the tool message deleted, which is the bug this stands for.
    seen: list[list[LLMMessage]] = []
    events, edits, _ = _run(
        [_EDIT, _DONE], lock=SourceLock.READ_ONLY, answers=[], record_messages=seen
    )
    cards = [e for e in events if isinstance(e, AgentToolCard)]
    assert len(cards) == 1, "a refused call still reports a card"
    assert not cards[0].ok
    assert not edits

    assert len(seen) > 1, (
        "a turn that made one request cannot show the 400 this stands for"
    )
    for i, request in enumerate(seen[1:], start=1):
        asked = {
            call.id
            for message in request
            if message.role == "assistant"
            for call in (message.tool_calls or [])
        }
        answered = {m.tool_call_id for m in request if m.role == "tool"}
        assert asked <= answered, (
            f"request {i} carries tool_call_id(s) {sorted(asked - answered)} with no result; "
            "the provider 400s on that"
        )
    assert any(
        message.role == "assistant" and message.tool_calls for message in seen[1]
    ), "the refused call must be in the history at all, or the check above is vacuous"


def test_the_lock_never_reaches_a_tool_that_does_not_write_source() -> None:
    # The refuse branch is guarded by locks_source, and without that guard an armed lock would
    # silently swallow render/delete/publish -- tools with their OWN gate, which the user answers
    # separately. Two directions, because each fails on its own: an armed lock, and a latch already
    # set by a denied edit.
    events, _, _ = _run([_RENDER, _DONE], lock=SourceLock.READ_ONLY, answers=[])
    opened = _gates(events)
    assert len(opened) == 1, "an armed lock must not swallow an always-gated tool"
    assert opened[0].request.kind is GateKind.CONFIRM

    events, _, _ = _run(
        [_EDIT, _RENDER, _DONE],
        lock=SourceLock.ASK,
        answers=[
            GateResponse(False, lock_answer=LockAnswer.DENY),
            GateResponse(True),
        ],
    )
    opened = _gates(events)
    assert len(opened) == 2, "a denied edit does not answer for a render"
    assert opened[1].request.kind is GateKind.CONFIRM


def test_a_refused_call_reaches_the_turn_ledger() -> None:
    # The refused call must survive into the turn summary the NEXT turn reads: a ledger that
    # forgets it would tell the model the copilot did nothing at all, and drop the document
    # address a "do the same to C" follow-up needs. The live-decline path records this already;
    # nothing was holding the refuse path to it.
    events, _, _ = _run([_EDIT, _DONE, _DONE], lock=SourceLock.READ_ONLY, answers=[])
    done = [e for e in events if isinstance(e, AgentTurnDone)]
    assert done, "the turn ended without a summary"
    ledger = done[-1].summary.ledger
    assert any("edit_shader" in line for line in ledger), (
        f"the refused call left no ledger entry: {ledger}"
    )


def test_each_mode_does_what_its_chip_says() -> None:
    # The three positions, each its own assert so a collapse of any two names which.
    events, edits, _ = _run([_EDIT, _DONE], lock=SourceLock.ALLOW, answers=[])
    assert not _gates(events), "ALLOW asks nothing"
    assert edits, "ALLOW runs the edit"

    events, edits, _ = _run(
        [_EDIT, _DONE],
        lock=SourceLock.ASK,
        answers=[GateResponse(True, lock_answer=LockAnswer.ALLOW)],
    )
    assert len(_gates(events)) == 1, "ASK asks once"
    assert edits, "an approved ASK runs the edit"

    events, edits, _ = _run([_EDIT, _DONE], lock=SourceLock.READ_ONLY, answers=[])
    assert not _gates(events), "DENY asks nothing"
    assert not edits, "DENY runs nothing"


def test_deny_reaches_the_refusal_rather_than_skipping_the_gate_block() -> None:
    # The trap D4a names. The refusal lives INSIDE `if must_confirm(...)` and `execute` sits
    # outside it, so a must_confirm that excused DENY would route the call past the refusal and
    # RUN the edit. This drives the predicate directly, because the behavioural test above would
    # also go red for the opposite bug and could not tell the two apart.
    registry = build_registry(minimal_caps())
    registry.source_lock = SourceLock.READ_ONLY
    assert registry.must_confirm("edit_shader"), (
        "DENY must still claim the call, or the loop never reaches the refusal"
    )


def test_the_decline_message_is_one_template_on_both_paths() -> None:
    # D4: the model reads the SAME fact whether it was refused or declined live. Two copied
    # literals drift; one template cannot. The assert is on the messages the model actually
    # received, not on the constant, so inlining a second copy still fails here.
    def _declines(lock: SourceLock, answers: list[GateResponse]) -> list[str]:
        seen: list[list[LLMMessage]] = []
        _run([_EDIT, _DONE], lock=lock, answers=answers, record_messages=seen)
        return [
            m.content or ""
            for request in seen
            for m in request
            if m.role == "tool" and "declined" in (m.content or "")
        ]

    refused = _declines(SourceLock.READ_ONLY, [])
    declined = _declines(
        SourceLock.ASK, [GateResponse(False, lock_answer=LockAnswer.DENY)]
    )
    assert refused and declined, "both paths must produce a decline the model can read"
    assert refused[0] == declined[0], (
        f"the two decline paths tell the model different things:\n{refused[0]!r}\n{declined[0]!r}"
    )


def test_the_mode_round_trips_through_the_project_file(tmp_path: Path) -> None:
    # It is a MODE the user sets, so it must outlive the session. Driven through the real model and
    # a real file rather than by reading the field back off the object.
    path = tmp_path / "app_state.json"
    state = UIAppState()
    assert state.copilot_source_lock is SourceLock.ASK, (
        "a project that never answered asks"
    )
    state.copilot_source_lock = SourceLock.READ_ONLY
    state.save(path)
    assert UIAppState.load(path).copilot_source_lock is SourceLock.READ_ONLY


def test_a_project_file_from_before_the_mode_existed_asks(tmp_path: Path) -> None:
    # Every app_state.json on disk predates this key, so the DEFAULT is what those projects get.
    # ASK keeps their behaviour exactly as it was; ALLOW would have quietly switched the copilot to
    # editing unasked in projects that ask today.
    path = tmp_path / "app_state.json"
    path.write_text('{"current_document_id": "abc"}')
    assert UIAppState.load(path).copilot_source_lock is SourceLock.ASK


def test_a_reset_takes_the_projects_mode_not_a_fresh_default() -> None:
    # reset_conversation serves two callers that want opposite things: a project SWITCH must take
    # the incoming project's mode, and CLEAR must leave the current one alone. Seeding from the
    # project satisfies both with no branch -- seeding from a fresh ChatState() would hand Clear
    # the default and silently discard the user's setting.
    mode = SourceLock.READ_ONLY
    session = CopilotSession(
        caps=minimal_caps(),
        client=_FakeClient([]),
        get_project_slug=lambda: "p",
        get_checkpoints_root=lambda: Path("/tmp"),
        get_source_lock=lambda: mode,
        set_project_source_lock=lambda _lock: None,
    )
    assert session.state.source_lock is SourceLock.READ_ONLY, (
        "born with the project's mode"
    )

    session.reset_conversation()
    assert session.state.source_lock is SourceLock.READ_ONLY, (
        "a reset that seeds from a fresh ChatState() discards the project's mode"
    )
    assert session.registry.source_lock is SourceLock.READ_ONLY


def test_every_mode_has_a_chip_label() -> None:
    # The chip reads its label by lookup, so a mode without one is a KeyError at DRAW time -- in
    # a frame callback, where it surfaces as a dead panel rather than a traceback anyone reads.
    # Enumerated from the enum so a fourth mode fails here instead of on screen.
    assert set(_LOCK_LABELS) == set(SourceLock)
    assert all(_LOCK_LABELS[mode].strip() for mode in SourceLock)


def test_read_only_withholds_the_source_tools_from_the_request() -> None:
    # The point of the mode: the model does not SEE the editing tools, so it never spends a call
    # discovering it is barred. The withheld set is computed from the registry's own roster, not
    # listed here, so a new source tool joins it without anyone remembering.
    registry = build_registry(minimal_caps())
    registry.source_lock = SourceLock.ALLOW
    everything = {spec.name for spec in registry.assemble_specs(set())}

    registry.source_lock = SourceLock.READ_ONLY
    offered = {spec.name for spec in registry.assemble_specs(set())}

    locked = {d.name for d in registry.definitions() if d.locks_source and d.eager}
    assert locked, "the roster is empty; this test would pass vacuously"
    assert everything - offered == locked, (
        "READ_ONLY must withhold exactly the source-writing tools"
    )
    assert "read_shader" in offered, "reading still works"


def test_the_other_modes_offer_every_tool() -> None:
    # A mode nobody set must cost nothing: ALLOW and ASK see the same list as before the filter.
    registry = build_registry(minimal_caps())
    registry.source_lock = SourceLock.ALLOW
    allow = [spec.name for spec in registry.assemble_specs(set())]
    registry.source_lock = SourceLock.ASK
    ask = [spec.name for spec in registry.assemble_specs(set())]
    assert allow == ask, "ASK gates calls; it does not withhold tools"
    assert "edit_shader" in allow


def test_the_prompt_says_why_the_tools_are_missing() -> None:
    # Hiding the tools without saying why leaves a model that has quietly lost an ability and
    # answers an edit request with confusion. The fact rides the project block, and ONLY in the
    # mode it describes -- every other mode's prefix is unchanged, so the cache is untouched.
    plain = _context_block(build_context(minimal_caps(), source_read_only=False))
    noticed = _context_block(build_context(minimal_caps(), source_read_only=True))
    marker = "SOURCE IS READ-ONLY IN THIS PROJECT"
    assert marker not in plain, "no other mode pays for this sentence"
    assert marker in noticed
    assert noticed.startswith(plain), (
        "the notice is APPENDED; the cacheable prefix is unchanged"
    )


def test_load_tools_cannot_bring_back_a_withheld_tool() -> None:
    # The lazy path calls assemble_specs again, so the filter holds by construction -- but the
    # model must be TOLD, or it retries a load that silently did nothing.
    registry = build_registry(minimal_caps())
    registry.source_lock = SourceLock.READ_ONLY
    withheld = [d.name for d in registry.definitions() if d.locks_source]
    assert withheld
    for name in withheld:
        assert registry.is_withheld(name)
    offered = {spec.name for spec in registry.assemble_specs(set(withheld))}
    assert not (offered & set(withheld)), (
        "a lazy load must not reintroduce a tool the mode withholds"
    )


def _session() -> CopilotSession:
    return CopilotSession(
        caps=minimal_caps(),
        client=_FakeClient([]),
        get_project_slug=lambda: "p",
        get_checkpoints_root=lambda: Path("/tmp"),
        get_source_lock=lambda: SourceLock.ASK,
        set_project_source_lock=lambda _lock: None,
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
    # ASK specifically, not merely "not OFF": must_confirm answers True for ARMED too, so a
    # default flipped to ARMED would pass every other assertion here while making a fresh session
    # refuse every edit in silence -- the exact failure D2 exists to prevent.
    assert session.state.source_lock is SourceLock.ASK, (
        "a fresh session asks; it does not refuse"
    )


def test_the_locks_two_fields_agree_at_every_lifecycle_point() -> None:
    # V4, and the check that was specified and then not written -- which is exactly how a session
    # shipped drawing a closed padlock over a registry that confirmed nothing. The two fields are
    # born from two DIFFERENT defaults (ChatState at ASK, a bare registry OFF), so agreement
    # is something the one writer must establish, never something construction gives for free.
    #
    # Three lifecycle points, because each reaches the pair by a different route: construction
    # (two defaults meeting), an explicit toggle (the icon), and a reset (which rebuilds ChatState
    # and NOT the registry).
    session = _session()
    assert session.state.source_lock is session.registry.source_lock, (
        "born diverged: the icon and the gate disagree before anything has happened"
    )
    assert session.registry.must_confirm("edit_shader"), (
        "a fresh session draws a locked icon, so it must actually confirm"
    )

    session.set_source_lock(SourceLock.ALLOW)
    assert session.state.source_lock is session.registry.source_lock
    assert not session.registry.must_confirm("edit_shader")

    session.reset_conversation()
    assert session.state.source_lock is session.registry.source_lock, (
        "reset rebuilds ChatState but not the registry; without a re-seed they diverge"
    )
    assert session.registry.must_confirm("edit_shader"), (
        "a cleared chat is a new session, so it locks again"
    )
