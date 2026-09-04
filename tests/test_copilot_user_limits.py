"""User-tunable copilot limits (034 F12): store fields <-> live config seam."""

import threading
from dataclasses import fields, replace
from pathlib import Path

from shaderbox.copilot.agent import AgentTurnDone, run_turn
from shaderbox.copilot.config import COPILOT_CONFIG, CopilotConfig
from shaderbox.copilot.gate import GateChannel
from shaderbox.copilot.llm.api import LLMDone, LLMStreamEvent, LLMTextDelta
from shaderbox.copilot.tools.registry import build_registry
from shaderbox.copilot.trace import TraceLog
from shaderbox.integrations import CopilotIntegration
from tests.test_copilot_loop import (
    _fake_caps,
    _fake_context,
    _FakeClient,
    _model_cards,
    _tool_call,
)

_LIMIT_FIELDS = (
    "max_iterations",
    "max_input_tokens",
    "max_tokens_per_turn",
    "max_edit_retries",
    "max_compile_failures",
    "clean_edit_soft_streak",
    "clean_edit_hard_streak",
    "noop_edit_soft_streak",
    "noop_edit_hard_streak",
    "auto_revert_after_failed_edits",
    "turn_time_budget_s",
)


def _snapshot() -> dict[str, int]:
    return {f: getattr(COPILOT_CONFIG, f) for f in _LIMIT_FIELDS}


def _restore(snap: dict[str, int]) -> None:
    for f, v in snap.items():
        setattr(COPILOT_CONFIG, f, v)


def test_copilot_config_holds_exactly_the_settings_tunable_knobs() -> None:
    # CopilotConfig is the USER surface: every field must be Settings-tunable (reachable through
    # apply_user_limits + persisted on CopilotIntegration). An untunable knob belongs on
    # CopilotEngineConfig — landing it here would silently widen the Settings contract.
    assert {f.name for f in fields(CopilotConfig)} == set(_LIMIT_FIELDS)


def test_integration_defaults_mirror_config_defaults() -> None:
    # CopilotConfig is the single source of truth; the persisted store must not drift.
    cfg = CopilotIntegration()
    defaults = CopilotConfig()
    for f in _LIMIT_FIELDS:
        assert getattr(cfg, f) == getattr(defaults, f), f


def test_apply_limits_reaches_the_live_config_with_floors() -> None:
    snap = _snapshot()
    try:
        CopilotIntegration(
            max_iterations=0,  # floored to 1
            max_input_tokens=50_000,
            max_tokens_per_turn=8_000,
            max_edit_retries=2,
            max_compile_failures=0,  # 0 = off, legal
            clean_edit_soft_streak=9,
            clean_edit_hard_streak=15,
            noop_edit_soft_streak=4,
            noop_edit_hard_streak=7,
            auto_revert_after_failed_edits=0,
        ).apply_limits()
        assert COPILOT_CONFIG.max_iterations == 1
        assert COPILOT_CONFIG.max_input_tokens == 50_000
        assert COPILOT_CONFIG.max_tokens_per_turn == 8_000
        assert COPILOT_CONFIG.max_edit_retries == 2
        assert COPILOT_CONFIG.max_compile_failures == 0
        assert COPILOT_CONFIG.clean_edit_soft_streak == 9
        assert COPILOT_CONFIG.clean_edit_hard_streak == 15
        assert COPILOT_CONFIG.noop_edit_soft_streak == 4
        assert COPILOT_CONFIG.noop_edit_hard_streak == 7
        assert COPILOT_CONFIG.auto_revert_after_failed_edits == 0
    finally:
        _restore(snap)


def test_zero_disables_clean_streak_nudge() -> None:
    edit = _tool_call(
        "cx",
        "edit_shader",
        '{"old_str": "vec3 p = u_pos;", "new_str": "vec3 p = u_pos;"}',
    )
    n_clean = COPILOT_CONFIG.clean_edit_soft_streak + 4
    scripts: list[list[LLMStreamEvent]] = [edit] * n_clean + [
        [LLMTextDelta("done"), LLMDone("stop")]
    ]
    caps = _fake_caps(edit_errors=[[]] * n_clean)

    nudge_events: list[str] = []

    class _RecordingTrace(TraceLog):
        def __init__(self) -> None:
            super().__init__(Path())

        def event(self, kind: str, **fields: object) -> None:
            nudge_events.append(kind)

    config = replace(
        COPILOT_CONFIG,
        clean_edit_soft_streak=0,
        clean_edit_hard_streak=0,
        noop_edit_soft_streak=0,
        noop_edit_hard_streak=0,
    )
    events = list(
        run_turn(
            _FakeClient(scripts),
            build_registry(caps),
            config,
            _fake_context(),
            history=[],
            user_text="tweak forever",
            gate=GateChannel(),
            cancel=threading.Event(),
            trace=_RecordingTrace(),
        )
    )
    assert nudge_events.count("clean_streak_nudge") == 0
    assert len(_model_cards(events)) == n_clean
    assert isinstance(events[-1], AgentTurnDone)
