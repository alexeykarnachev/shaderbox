"""History-window trim (resolves the unbounded-history BLOCKER). build_messages drops whole leading
TURNS once the request estimate exceeds max_input_tokens, always keeping the last _MIN_KEPT_TURNS and
never splitting an assistant->tool_call->tool group (an orphaned tool_call_id 400s the provider). Pure:
exercises _trim_history / _split_turns directly with hand-built LLMMessages."""

from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.llm.api import LLMMessage, LLMToolCall
from shaderbox.copilot.prompt import (
    _MIN_KEPT_TURNS,
    _split_turns,
    _trim_history,
)


def _turn(idx: int, body_chars: int = 0) -> list[LLMMessage]:
    # A user message + an assistant turn that called a tool + the tool result — the pairing the trim
    # must never split. body_chars pads the tool result so we can push a turn over budget on demand.
    tc = LLMToolCall(id=f"call_{idx}", name="read_shader", arguments="{}")
    return [
        LLMMessage(role="user", content=f"u{idx}"),
        LLMMessage(role="assistant", content=None, tool_calls=[tc]),
        LLMMessage(role="tool", tool_call_id=f"call_{idx}", content="x" * body_chars),
    ]


def _flat(turns: list[list[LLMMessage]]) -> list[LLMMessage]:
    return [m for t in turns for m in t]


def test_short_history_untouched() -> None:
    history = _flat([_turn(i) for i in range(3)])
    assert _trim_history(history, fixed_overhead_tokens=0) is history


def test_split_turns_groups_by_user_boundary() -> None:
    history = _flat([_turn(0), _turn(1)])
    turns = _split_turns(history)
    assert len(turns) == 2
    assert [m.role for m in turns[0]] == ["user", "assistant", "tool"]
    assert turns[1][0].content == "u1"


def test_over_budget_drops_leading_turns_keeps_min() -> None:
    # Each turn carries a fat tool result so a handful blow past the budget. 20 turns, big bodies.
    big = COPILOT_CONFIG.max_input_tokens * 4  # chars; /4 ~= max_input_tokens per turn
    history = _flat([_turn(i, body_chars=big) for i in range(20)])
    trimmed = _trim_history(history, fixed_overhead_tokens=0)
    kept_turns = _split_turns(trimmed)
    # Trimmed down to the floor (each turn alone already saturates the budget).
    assert len(kept_turns) == _MIN_KEPT_TURNS
    # The KEPT turns are the most RECENT ones (oldest dropped from the front).
    assert kept_turns[-1][0].content == "u19"
    assert kept_turns[0][0].content == f"u{20 - _MIN_KEPT_TURNS}"


def test_trim_never_orphans_a_tool_call() -> None:
    big = COPILOT_CONFIG.max_input_tokens * 4
    history = _flat([_turn(i, body_chars=big) for i in range(12)])
    trimmed = _trim_history(history, fixed_overhead_tokens=0)
    # Every assistant tool_call id has its matching tool result still present, and vice versa.
    call_ids = {c.id for m in trimmed for c in (m.tool_calls or ())}
    result_ids = {m.tool_call_id for m in trimmed if m.tool_call_id is not None}
    assert call_ids == result_ids, "trim orphaned a tool_call_id (provider would 400)"
    # And the head of the trimmed history is a user message (a clean turn boundary).
    assert trimmed[0].role == "user"
