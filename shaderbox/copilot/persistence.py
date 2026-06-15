"""On-disk persistence for a copilot conversation.

Durable per-project state in the project dir. Owns the versioned `ConversationStore`
pydantic model + load/save/archive (`extra="forbid"`, fail-soft `load`), and
maps the runtime dataclasses (`Message` / `LLMMessage` / `TurnStats`) to/from their
persisted shape so `state.py` stays free of disk concerns.
"""

import json
from pathlib import Path
from typing import Literal, Self, cast

from loguru import logger
from pydantic import BaseModel, ValidationError

from shaderbox.copilot.gate import GateKind
from shaderbox.copilot.llm.api import LLMMessage
from shaderbox.copilot.state import (
    RESULT_WIDGET_KINDS,
    ChatState,
    Message,
    MessageRole,
    RecoverInfo,
    ResultWidget,
    ResultWidgetKind,
    StepRecord,
    TurnStats,
)

_VERSION = 10


def _gate_kind_or_confirm(value: str) -> GateKind:
    # Unknown GateKind (future member on an older build) loads as CONFIRM instead of failing
    # the file.
    try:
        return GateKind(value)
    except ValueError:
        return GateKind.CONFIRM


def _result_widget_or_none(model: "_ResultWidgetModel | None") -> ResultWidget | None:
    # Unknown widget kind or empty target drops to None instead of failing the file or
    # rendering a dead button.
    if model is None or model.kind not in RESULT_WIDGET_KINDS or not model.target:
        return None
    return ResultWidget(
        kind=cast(ResultWidgetKind, model.kind), label=model.label, target=model.target
    )


def _stats_model(stats: TurnStats | None) -> "_TurnStatsModel | None":
    if stats is None:
        return None
    return _TurnStatsModel(
        context_tokens=stats.context_tokens,
        reply_tokens=stats.reply_tokens,
        cost_usd=stats.cost_usd,
    )


def _stats_or_none(model: "_TurnStatsModel | None") -> TurnStats | None:
    if model is None:
        return None
    return TurnStats(
        context_tokens=model.context_tokens,
        reply_tokens=model.reply_tokens,
        cost_usd=model.cost_usd,
    )


class _RecoverModel(BaseModel):
    # Stores the trash dir-NAME, not an absolute path — the project dir is relocatable;
    # re-anchored via App.trash_dir at click time.
    node_id: str
    node_name: str = ""
    trash_name: str
    done: bool = False
    model_config = {"extra": "forbid"}


class _ResultWidgetModel(BaseModel):
    # kind is loose-str so a future widget kind round-trips (the renderer skips an unknown one).
    kind: str
    label: str = "Open"
    target: str = ""
    model_config = {"extra": "forbid"}


class _StepRecordModel(BaseModel):
    name: str
    ok: bool = True
    model_config = {"extra": "forbid"}


class _TurnStatsModel(BaseModel):
    context_tokens: int = 0
    reply_tokens: int = 0
    cost_usd: float = 0.0
    model_config = {"extra": "forbid"}


class _MessageModel(BaseModel):
    # role / gate_kind are loose-str so a future member loads instead of failing the file.
    # gate_input (the typed secret buffer) is deliberately NOT a field — never persisted.
    role: str
    text: str = ""
    resolved: bool = False
    recover: _RecoverModel | None = None
    gate_kind: str = "confirm"
    # Optional + defaulted so a pre-field file loads as None.
    result_widget: _ResultWidgetModel | None = None
    # The user message's turn id, keying its rollback checkpoint (feature 020·30). Defaulted so a
    # pre-field file loads with no turn_id (no Revert button — the checkpoint dirs are gone anyway).
    turn_id: str = ""
    # role == "turn_snippet": the per-tool step list + this turn's own stats (feature 028 F06).
    # Defaulted so a pre-v9 file loads with an empty snippet / no stats.
    steps: list[_StepRecordModel] = []
    snippet_stats: _TurnStatsModel | None = None
    # Resolved gate's outcome token (feature 034 F03). Defaulted so a pre-v10 file (whose
    # outcome is baked into text) loads with no structured outcome — renders as plain text.
    gate_outcome: str = ""
    model_config = {"extra": "forbid"}


class _HistoryModel(BaseModel):
    # LLM replay context — NATURAL-LANGUAGE only: user messages + one assistant turn-summary
    # each, no tool messages. A pre-v5 store with tool_call_id / tool_calls hits extra="forbid"
    # and fail-softs to an empty chat (no backward-compat migration by design).
    role: str
    content: str | None = None
    model_config = {"extra": "forbid"}


class ConversationStore(BaseModel):
    version: int = _VERSION
    messages: list[_MessageModel] = []
    history: list[_HistoryModel] = []
    session_cost_usd: float = 0.0
    last_turn: _TurnStatsModel | None = None
    model_config = {"extra": "forbid"}

    @classmethod
    def from_runtime(
        cls, state: ChatState, history: list[LLMMessage]
    ) -> "ConversationStore":
        return cls(
            version=_VERSION,
            messages=[
                _MessageModel(
                    role=m.role,
                    text=m.text,
                    resolved=m.resolved,
                    gate_kind=m.gate_kind.value,
                    recover=(
                        _RecoverModel(
                            node_id=m.recover.node_id,
                            node_name=m.recover.node_name,
                            trash_name=m.recover.trash_name,
                            done=m.recover.done,
                        )
                        if m.recover is not None
                        else None
                    ),
                    result_widget=(
                        _ResultWidgetModel(
                            kind=m.result_widget.kind,
                            label=m.result_widget.label,
                            target=m.result_widget.target,
                        )
                        if m.result_widget is not None
                        else None
                    ),
                    turn_id=m.turn_id,
                    steps=[_StepRecordModel(name=s.name, ok=s.ok) for s in m.steps],
                    snippet_stats=_stats_model(m.snippet_stats),
                    gate_outcome=m.gate_outcome,
                )
                for m in state.messages
            ],
            history=[_HistoryModel(role=h.role, content=h.content) for h in history],
            session_cost_usd=state.session_cost_usd,
            last_turn=_stats_model(state.last_turn),
        )

    def to_messages(self) -> list[Message]:
        # role is a free str on disk; cast back to the Literal — the renderer treats an
        # unknown role as plain text anyway.
        return [
            Message(
                role=cast(MessageRole, m.role),
                text=m.text,
                resolved=m.resolved,
                gate_kind=_gate_kind_or_confirm(m.gate_kind),
                recover=(
                    RecoverInfo(
                        node_id=m.recover.node_id,
                        node_name=m.recover.node_name,
                        trash_name=m.recover.trash_name,
                        done=m.recover.done,
                    )
                    if m.recover is not None
                    else None
                ),
                result_widget=_result_widget_or_none(m.result_widget),
                turn_id=m.turn_id,
                steps=[StepRecord(name=s.name, ok=s.ok) for s in m.steps],
                snippet_stats=_stats_or_none(m.snippet_stats),
                gate_outcome=m.gate_outcome,
            )
            for m in self.messages
        ]

    def to_history(self) -> list[LLMMessage]:
        # NL-only: every persisted history message is a plain user/assistant message.
        return [
            LLMMessage(
                role=cast(Literal["system", "user", "assistant", "tool"], h.role),
                content=h.content,
            )
            for h in self.history
        ]

    def to_session_cost(self) -> float:
        return self.session_cost_usd

    def to_last_turn(self) -> TurnStats | None:
        return _stats_or_none(self.last_turn)

    def save(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load(cls, file_path: Path) -> Self:
        # Fail-soft: a missing file is a first-use empty chat; an unreadable / incompatible
        # one degrades to empty + a WARNING, never a crash and never a lost project.
        if not file_path.exists():
            return cls()
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Unreadable copilot conversation ({e}); starting empty")
            return cls()
        if not isinstance(data, dict):
            logger.warning(
                "Incompatible copilot conversation (not a JSON object); starting empty"
            )
            return cls()
        try:
            return cls(**data)
        except ValidationError as e:
            logger.warning(f"Incompatible copilot conversation ({e}); starting empty")
            return cls()


def archive_conversation(conversation_path: Path, stamp: str) -> None:
    # Move the live conversation into copilot/archive/conversation_<stamp>.json on clear
    # (recoverable, not destroyed; archive is uncapped).
    if not conversation_path.exists():
        return
    archive_dir = conversation_path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    conversation_path.replace(archive_dir / f"conversation_{stamp}.json")
