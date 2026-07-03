"""Feature 053 slice B: the turn-end auto vision look. When a turn CHANGED the render but the model
never looked, the engine takes ONE look FOR it and injects the observation as data (with explicit
provenance) so a visual result is never declared blind. Deterministic (scripted fake client)."""

import threading

from shaderbox.copilot.agent import AgentTextDelta, _auto_look_fact, run_turn
from shaderbox.copilot.capabilities import EditResult
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.gate import GateChannel
from shaderbox.copilot.llm.api import LLMDone, LLMStreamEvent, LLMTextDelta, LLMUsage
from shaderbox.copilot.tools.registry import build_registry
from tests._caps import minimal_caps
from tests.test_copilot_loop import _fake_context, _FakeClient, _tool_call

_VISION_LINE = (
    "render@t=0.0s: ink 5% | bbox x 0.4-0.6\n"
    "visual (correctness only, NOT beauty): coherent: clear subject | orientation: upright | "
    "framing: fully in frame | text: none | artifacts: none"
)
_FACTS_ONLY = "render@t=0.0s: ink 5% | bbox x 0.4-0.6"  # vision off / outage -> no `visual (` line

_EDIT = _tool_call(
    "c1", "edit_shader", '{"old_str": "u_pos", "new_str": "u_pos", "target": ""}'
)
_PROBE = _tool_call("c2", "probe_render", '{"node": "", "t": 0.0, "look_for": "round"}')


def _stop(text: str) -> list[LLMStreamEvent]:
    return [
        LLMTextDelta(text),
        LLMDone(finish_reason="stop", usage=LLMUsage(output_tokens=5)),
    ]


def _drive(
    scripts: list[list[LLMStreamEvent]], probe_line: str = _VISION_LINE
) -> tuple[list[dict], list[str]]:
    calls: list[dict] = []

    def _probe(node: str, t: float, look_for: str = "") -> str:
        calls.append({"node": node, "t": t, "look_for": look_for})
        return probe_line

    caps = minimal_caps(
        apply_shader_edit=lambda _o, _n, _r, _t: EditResult(matches=1, errors=[]),
        probe_render=_probe,
    )
    events = list(
        run_turn(
            _FakeClient(scripts),
            build_registry(caps),
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="make a red circle",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )
    replies = [e.text for e in events if isinstance(e, AgentTextDelta)]
    return calls, replies


def test_auto_look_fires_when_render_changed_and_model_never_looked() -> None:
    # edit (visual mutation) -> stop -> ENGINE auto-looks -> model reacts -> stop. The 3rd script is
    # ONLY consumed if the injection re-opened the turn, so its reply proves the reaction happened.
    calls, replies = _drive(
        [_EDIT, _stop("Done, circle drawn."), _stop("Confirmed: centered and round.")]
    )
    assert (
        len(calls) == 1
    )  # the ENGINE's single auto-look (the model itself never called probe)
    assert calls[0]["node"] == ""  # the current frame
    assert (
        "Confirmed: centered and round." in replies
    )  # the model got the extra reaction iteration


def test_no_auto_look_when_model_already_looked() -> None:
    # The model mutated AND looked (its own probe_render) -> no engine auto-look (no double look).
    calls, _ = _drive([_EDIT, _PROBE, _stop("Looks right, done.")])
    assert len(calls) == 1  # only the model's own look; NO engine second look


def test_no_auto_look_when_no_visual_mutation() -> None:
    # The turn changed nothing on the render (a bare reply) -> nothing to look at.
    calls, _ = _drive([_stop("Nothing to change.")])
    assert calls == []


def test_no_injection_when_vision_absent_from_probe() -> None:
    # Vision off / outage / non-vision model: probe returns a facts-only line (no `visual (`). The
    # engine may render once but MUST NOT inject a blind auto-look, so the turn ends without reaction.
    # Only 2 scripts -> if an injection re-opened the turn, the loop would IndexError on a 3rd stream.
    calls, replies = _drive([_EDIT, _stop("Done.")], probe_line=_FACTS_ONLY)
    assert len(calls) == 1  # it tried the look
    assert "Done." in replies and "Confirmed" not in " ".join(
        replies
    )  # no reaction turn


def test_auto_look_fact_states_provenance_and_carries_the_read() -> None:
    # The self-awareness contract: the injected text must make clear the ENGINE looked (not the model),
    # three ways, and must carry the actual observation. Guards the wording the whole slice hinges on.
    fact = _auto_look_fact(_VISION_LINE)
    lowered = fact.lower()
    assert "did not call this" in lowered
    assert "for you" in lowered
    assert "not a look you performed" in lowered
    assert "coherent: clear subject" in fact  # the real observation is embedded
    assert fact.isascii()  # engine text on the user channel stays ASCII
