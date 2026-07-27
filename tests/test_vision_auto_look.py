"""Feature 056 half 1: the bounded convergence loop. When a turn CHANGED the render, the engine
takes an AIMED look FOR the model (unconditional on the model's own probes), injects the
observation as data with round/target-aware provenance, and re-looks while further mutation
lands — up to `copilot_convergence_max_looks`. Deterministic (scripted fake client, faked probe)."""

import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from shaderbox.copilot.agent import (
    AgentTextDelta,
    AgentToolCard,
    AgentTurnDone,
    _auto_look_fact,
    _turn_intent_look_for,
    run_turn,
)
from shaderbox.copilot.capabilities import EditResult, ProbeResult
from shaderbox.copilot.config import COPILOT_CONFIG, CopilotConfig
from shaderbox.copilot.gate import GateChannel
from shaderbox.copilot.llm.api import LLMDone, LLMStreamEvent, LLMTextDelta, LLMUsage
from shaderbox.copilot.tools.registry import build_registry
from shaderbox.copilot.trace import TraceLog
from shaderbox.copilot.vision_contract import (
    ASK_MET,
    ASK_NOT_MET,
    ASK_UNCLEAR,
    VisionUsage,
)
from tests._caps import minimal_caps
from tests.test_copilot_loop import _fake_context, _FakeClient, _tool_call

# A PRODUCTION-LENGTH facts line (~150 chars): it precedes the eye's read in `msg`, so anything
# that quotes `msg` instead of the read alone truncates the read away.
_FACTS = (
    "render@t=0.0s: ink 34% | bbox x 0.08-0.92, y 0.11-0.88 (y=0 bottom) | ink mean "
    "rgb(31,64,140) cool | luma 0-9 top/mid/bottom rows: 112/223 441/552 663/774"
)
_READ = (
    "coherent: a blue field with white stripes | readability: main subject clear | "
    "orientation: upright | framing: fully in frame | text: none | artifacts: none | "
    "look_for: no stars are visible anywhere on the canton"
)


def _seen(verdict: str, usage: VisionUsage | None = None) -> ProbeResult:
    # A working vision look: the ASK line is already parsed + stripped backend-side.
    return ProbeResult(
        msg=f"{_FACTS}\nvisual (correctness only, NOT beauty): {_READ}",
        vision_ok=True,
        verdict=verdict,
        ask_line=f"ASK: {verdict}",
        read=_READ,
        usage=usage,
    )


_BLIND = ProbeResult(msg=_FACTS)  # vision off / outage / non-vision model

_EDIT = _tool_call(
    "c1", "edit_shader", '{"old_str": "u_pos", "new_str": "u_pos", "target": ""}'
)
_PROBE = _tool_call("c2", "probe_render", '{"node": "", "t": 0.0, "look_for": "round"}')


def _edit_on(target: str) -> list[LLMStreamEvent]:
    return _tool_call(
        "ce",
        "edit_shader",
        f'{{"old_str": "u_pos", "new_str": "u_pos", "target": "{target}"}}',
    )


def _stop(text: str) -> list[LLMStreamEvent]:
    return [
        LLMTextDelta(text),
        LLMDone(finish_reason="stop", usage=LLMUsage(output_tokens=5)),
    ]


class _RecordingTrace(TraceLog):
    def __init__(self) -> None:
        super().__init__(Path())
        self.events: list[tuple[str, dict[str, Any]]] = []

    def event(self, kind: str, **fields: Any) -> None:
        self.events.append((kind, fields))


def _drive(
    scripts: list[list[LLMStreamEvent]],
    probe: ProbeResult | list[ProbeResult] = _BLIND,
    config: CopilotConfig = COPILOT_CONFIG,
    trace: TraceLog | None = None,
) -> tuple[list[dict], list[Any]]:
    calls: list[dict] = []
    results = probe if isinstance(probe, list) else [probe] * 32

    def _probe(node: str, t: float, look_for: str = "") -> ProbeResult:
        calls.append({"node": node, "t": t, "look_for": look_for})
        return results[len(calls) - 1]

    caps = minimal_caps(
        apply_shader_edit=lambda _o, _n, _r, _t: EditResult(matches=1, errors=[]),
        probe_render=_probe,
    )
    events = list(
        run_turn(
            _FakeClient(scripts),
            build_registry(caps),
            config,
            _fake_context(),
            history=[],
            user_text="make a red circle",
            gate=GateChannel(),
            cancel=threading.Event(),
            trace=trace,
        )
    )
    return calls, events


def _replies(events: list[Any]) -> list[str]:
    return [e.text for e in events if isinstance(e, AgentTextDelta)]


# ---- the loop: gate, rounds, cap ----


def test_engine_look_fires_and_reopens_the_turn() -> None:
    # edit -> stop -> ENGINE look -> the model gets one more iteration. The 3rd script is ONLY
    # consumed if the injection re-opened the turn, so its reply proves the reaction happened.
    calls, events = _drive(
        [_EDIT, _stop("Done, circle drawn."), _stop("Confirmed: centered and round.")],
        probe=_seen(ASK_MET),
    )
    assert len(calls) == 1
    assert calls[0]["node"] == ""
    assert "make a red circle" in calls[0]["look_for"]  # aimed at the ask (054)
    assert "Confirmed: centered and round." in _replies(events)


def test_engine_look_is_unconditional_on_the_models_own_look() -> None:
    # The model probed itself (its look answers ITS question, not the user's ask) — the engine
    # still takes its own aimed look. The 053 opt-out is closed.
    calls, _ = _drive(
        [_EDIT, _PROBE, _stop("Looks right, done."), _stop("Rechecked.")],
        probe=_seen(ASK_MET),
    )
    assert len(calls) == 2  # the model's own probe + the engine's aimed look


def test_multi_round_relooks_only_while_mutation_lands() -> None:
    # Round 1 look -> the model edits again -> round 2 look -> the model replies WITHOUT further
    # mutation -> no third look, turn ends. Both the since-index gate and the multi-round loop.
    calls, events = _drive(
        [
            _EDIT,
            _stop("First pass."),
            _EDIT,
            _stop("Second pass."),
            _stop("Nothing else to change."),
        ],
        probe=_seen(ASK_NOT_MET),
    )
    assert len(calls) == 2
    assert isinstance(events[-1], AgentTurnDone)
    assert "Nothing else to change." in _replies(events)


def test_cap_one_stops_after_a_single_look() -> None:
    # Falsifier for the counter: at cap=1 the second mutation buys NO second look.
    config = replace(COPILOT_CONFIG, copilot_convergence_max_looks=1)
    calls, _ = _drive(
        [_EDIT, _stop("First pass."), _EDIT, _stop("Second pass.")],
        probe=_seen(ASK_NOT_MET),
        config=config,
    )
    assert len(calls) == 1


def test_cap_zero_is_the_master_switch() -> None:
    config = replace(COPILOT_CONFIG, copilot_convergence_max_looks=0)
    calls, _ = _drive([_EDIT, _stop("Done.")], probe=_seen(ASK_NOT_MET), config=config)
    assert calls == []


def test_no_look_without_a_render_mutation() -> None:
    calls, _ = _drive([_stop("Nothing to change.")])
    assert calls == []


def test_vision_off_skips_the_block_whole() -> None:
    # A vision-less user pays nothing new: no probe render, no card, no line — not even a blind
    # look that renders a frame for nothing.
    config = replace(COPILOT_CONFIG, copilot_vision_enabled=False)
    calls, events = _drive([_EDIT, _stop("Done.")], config=config)
    assert calls == []
    assert not [
        e
        for e in events
        if isinstance(e, AgentToolCard) and (e.payload or {}).get("engine_look")
    ]


def test_unclear_verdict_still_injects_and_reopens_one_round() -> None:
    # Decision 1's 053-parity path: a WORKING look injects for EVERY verdict; only the not-met
    # FRAMING and the summary line ride an affirmative not-met.
    _calls, events = _drive(
        [_EDIT, _stop("Done."), _stop("Acknowledged the read.")],
        probe=_seen(ASK_UNCLEAR),
    )
    assert "Acknowledged the read." in _replies(events)
    done = events[-1]
    assert isinstance(done, AgentTurnDone) and done.summary.eye == ""


def test_no_injection_when_vision_absent_from_probe() -> None:
    # Vision off / outage / non-vision model: the probe carries facts only. The engine may render
    # once but MUST NOT inject a blind look, so the turn ends without a reaction round (only 2
    # scripts — an injection would IndexError on a 3rd stream).
    calls, events = _drive([_EDIT, _stop("Done.")], probe=_BLIND)
    assert len(calls) == 1
    replies = _replies(events)
    assert "Done." in replies and "Confirmed" not in " ".join(replies)


# ---- the verdict: framing, summary, trace ----


def test_not_met_framing_rides_only_the_not_met_verdict() -> None:
    met = _auto_look_fact("read", 1, "", not_met=False)
    not_met = _auto_look_fact("read", 1, "", not_met=True)
    assert "NOT met" not in met
    assert "NOT met" in not_met and "do NOT claim it is done" in not_met


def test_not_met_verdict_lands_in_the_turn_summary() -> None:
    _calls, events = _drive(
        [_EDIT, _stop("Done."), _stop("Still missing the stars.")],
        probe=_seen(ASK_NOT_MET),
    )
    done = events[-1]
    assert isinstance(done, AgentTurnDone)
    assert done.summary.eye.startswith("eye: ask not-met after 1 look")
    # The line must quote the EYE's read, not the facts line that precedes it in `msg` — quoting
    # `msg` truncates the actual observation away before the 200-char cut.
    assert "no stars are visible" in done.summary.eye
    assert "ink mean rgb" not in done.summary.eye


def test_a_later_look_clears_an_earlier_not_met_note() -> None:
    # Round 1 not-met, round 2 blind (a vision outage mid-turn): the stale not-met must NOT be
    # recorded as the turn's final eye state.
    _calls, events = _drive(
        [_EDIT, _stop("First."), _EDIT, _stop("Second."), _stop("Done.")],
        probe=[_seen(ASK_NOT_MET), _BLIND],
    )
    done = events[-1]
    assert isinstance(done, AgentTurnDone)
    assert done.summary.eye == ""


def test_met_verdict_leaves_no_summary_line() -> None:
    _calls, events = _drive(
        [_EDIT, _stop("Done."), _stop("Confirmed.")], probe=_seen(ASK_MET)
    )
    done = events[-1]
    assert isinstance(done, AgentTurnDone)
    assert done.summary.eye == ""


def test_final_verdict_wins_the_summary() -> None:
    # not-met on round 1, met on round 2 -> the record carries the FINAL state, not the worst one.
    _calls, events = _drive(
        [_EDIT, _stop("First."), _EDIT, _stop("Second."), _stop("Done.")],
        probe=[_seen(ASK_NOT_MET), _seen(ASK_MET)],
    )
    done = events[-1]
    assert isinstance(done, AgentTurnDone)
    assert done.summary.eye == ""


def test_every_verdict_emits_a_trace_event() -> None:
    # The inertness detector must not itself ship inert: met / not-met / unclear / garbled (which
    # the backend parses as unclear) each emit ask_verdict carrying the raw line.
    for verdict, raw in (
        (ASK_MET, "ASK: met"),
        (ASK_NOT_MET, "ASK: not-met"),
        (ASK_UNCLEAR, "ASK: unclear"),
        (ASK_UNCLEAR, "ASK: probably? maybe"),
    ):
        trace = _RecordingTrace()
        result = replace(_seen(verdict), ask_line=raw)
        _drive([_EDIT, _stop("Done."), _stop("Ack.")], probe=result, trace=trace)
        emitted = [f for kind, f in trace.events if kind == "ask_verdict"]
        assert len(emitted) == 1
        assert emitted[0]["verdict"] == verdict
        assert emitted[0]["ask_line"] == raw


def test_a_blind_engine_look_still_emits_a_trace_event() -> None:
    # "Looked and came back blind" must be greppable in a trace, not a silent gap.
    trace = _RecordingTrace()
    _drive([_EDIT, _stop("Done.")], probe=_BLIND, trace=trace)
    emitted = [f for kind, f in trace.events if kind == "ask_verdict"]
    assert len(emitted) == 1
    assert emitted[0]["verdict"] == "none"
    assert emitted[0]["vision_ok"] is False


def test_every_round_reply_reaches_the_turn_record() -> None:
    # The screen accumulates each round's streamed text; the persisted summary must carry the same
    # prose, or history is short by every reply the CONV loop re-opened past.
    _calls, events = _drive(
        [_EDIT, _stop("Stage one done."), _EDIT, _stop("Stage two done.")],
        probe=[_seen(ASK_NOT_MET), _BLIND],
    )
    done = events[-1]
    assert isinstance(done, AgentTurnDone)
    assert "Stage one done." in done.summary.reply
    assert "Stage two done." in done.summary.reply


def test_model_probe_vision_cost_also_folds() -> None:
    # C1 covers the WHOLE turn's vision spend: the model's own probe_render is billed too.
    _calls, events = _drive(
        [_EDIT, _PROBE, _stop("Looks right."), _stop("Ack.")],
        probe=[
            _seen(ASK_MET, usage=VisionUsage(cost_usd=0.003)),  # the model's own look
            _seen(ASK_MET, usage=VisionUsage(cost_usd=0.002)),  # the engine's look
        ],
    )
    done = events[-1]
    assert isinstance(done, AgentTurnDone)
    assert done.stats is not None
    assert done.stats.cost_usd == 0.005


def test_vision_cost_folds_into_the_turn_stats() -> None:
    # C1: the look's billed cost reaches the turn stats; its TOKENS do not touch reply_tokens
    # (that gauge means the MAIN model's reply).
    _calls, events = _drive(
        [_EDIT, _stop("Done."), _stop("Ack.")],
        probe=_seen(
            ASK_MET,
            usage=VisionUsage(input_tokens=900, output_tokens=60, cost_usd=0.002),
        ),
    )
    done = events[-1]
    assert isinstance(done, AgentTurnDone)
    assert done.stats is not None
    assert done.stats.cost_usd == 0.002
    assert done.stats.reply_tokens == 10  # 2 model replies x 5, no vision tokens


# ---- provenance + targeting ----


def test_auto_look_fact_states_provenance_and_carries_the_read() -> None:
    # The self-awareness contract: the injected text must make clear the ENGINE looked (not the
    # model), and must carry the actual observation.
    fact = _auto_look_fact(_READ, 1, "", not_met=False)
    lowered = fact.lower()
    assert "did not call this" in lowered
    assert "for you" in lowered
    assert "not a look you performed" in lowered
    assert "no stars are visible anywhere" in fact
    assert fact.isascii()  # engine text on the user channel stays ASCII


def test_provenance_is_round_and_target_aware() -> None:
    # The engine must never state a falsehood: round 2 is not "ONE look", and a probed node that
    # is not the current one is NAMED instead of claiming the current frame.
    first = _auto_look_fact(_READ, 1, "", not_met=False)
    second = _auto_look_fact(_READ, 2, "7f3a", not_met=False)
    assert "ONE vision look" in first and "on the current frame" in first
    assert "vision look #2" in second and "at node 7f3a" in second
    assert "on the current frame" not in second


def test_look_targets_the_node_the_turn_changed() -> None:
    calls, _ = _drive(
        [_edit_on("7f3a"), _stop("Done."), _stop("Ack.")], probe=_seen(ASK_MET)
    )
    assert calls[0]["node"] == "7f3a"


def test_lib_target_falls_back_to_the_current_node() -> None:
    # A lib: address is not probeable (the resolver returns None) — the look falls back to the
    # current frame instead of erroring a useful look away.
    calls, _ = _drive(
        [_edit_on("lib:glow.glsl"), _stop("Done."), _stop("Ack.")], probe=_seen(ASK_MET)
    )
    assert calls[0]["node"] == ""


# ---- visibility ----


def test_engine_look_yields_a_visible_card() -> None:
    _calls, events = _drive(
        [_EDIT, _stop("Done."), _stop("Ack.")], probe=_seen(ASK_MET)
    )
    cards = [e for e in events if isinstance(e, AgentToolCard)]
    look_cards = [c for c in cards if (c.payload or {}).get("engine_look")]
    assert len(look_cards) == 1
    assert (
        look_cards[0].name == "probe_render"
    )  # keeps label_for + the dogfood analyzer


def test_turn_intent_look_for_carries_ask_and_handles_empty() -> None:
    lf = _turn_intent_look_for("Make a waving US flag with 50 stars")
    assert "waving US flag with 50 stars" in lf and lf.lower().startswith("does")
    assert _turn_intent_look_for("   ") == ""  # empty ask -> no fabricated intent
    assert len(_turn_intent_look_for("x" * 500)) < 320  # bounded
