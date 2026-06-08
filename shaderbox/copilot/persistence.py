"""On-disk persistence for a copilot conversation.

Durable per-project state in the project dir. Owns the versioned `ConversationStore`
pydantic model + load/save/archive (`extra="forbid"`, fail-soft `load_and_migrate`), and
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
    ChatState,
    Message,
    MessageRole,
    RecoverInfo,
    ResultWidget,
    ResultWidgetKind,
    TurnStats,
)

_VERSION = 8

_RESULT_WIDGET_KINDS: frozenset[str] = frozenset({"open_url", "open_path"})


def _migrate_pre_v7(data: dict[str, object]) -> None:
    # In place: pre-v7 stored a cumulative `usage` block + a `last_turn_usage` (input/output/cost).
    # v7 keeps only the running cost (`session_cost_usd`) + a `last_turn` (context/reply/cost). The
    # model forbids extra keys, so remap the old keys before construction or the file is rejected.
    old_usage = data.pop("usage", None)
    if isinstance(old_usage, dict) and "session_cost_usd" not in data:
        data["session_cost_usd"] = old_usage.get("cost_usd", 0.0)
    old_last = data.pop("last_turn_usage", None)
    if isinstance(old_last, dict) and "last_turn" not in data:
        data["last_turn"] = {
            "context_tokens": old_last.get("input_tokens", 0),
            "reply_tokens": old_last.get("output_tokens", 0),
            "cost_usd": old_last.get("cost_usd", 0.0),
        }


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
    if model is None or model.kind not in _RESULT_WIDGET_KINDS or not model.target:
        return None
    return ResultWidget(
        kind=cast(ResultWidgetKind, model.kind), label=model.label, target=model.target
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
    model_config = {"extra": "forbid"}


class _HistoryModel(BaseModel):
    # LLM replay context — NATURAL-LANGUAGE only: user messages + one assistant turn-summary
    # each, no tool messages. A pre-v5 store with tool_call_id / tool_calls hits extra="forbid"
    # and fail-softs to an empty chat (no backward-compat migration by design).
    role: str
    content: str | None = None
    model_config = {"extra": "forbid"}


class _TurnStatsModel(BaseModel):
    context_tokens: int = 0
    reply_tokens: int = 0
    cost_usd: float = 0.0
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
                )
                for m in state.messages
            ],
            history=[_HistoryModel(role=h.role, content=h.content) for h in history],
            session_cost_usd=state.session_cost_usd,
            last_turn=(
                _TurnStatsModel(
                    context_tokens=state.last_turn.context_tokens,
                    reply_tokens=state.last_turn.reply_tokens,
                    cost_usd=state.last_turn.cost_usd,
                )
                if state.last_turn is not None
                else None
            ),
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
        if self.last_turn is None:
            return None
        return TurnStats(
            context_tokens=self.last_turn.context_tokens,
            reply_tokens=self.last_turn.reply_tokens,
            cost_usd=self.last_turn.cost_usd,
        )

    def save(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load_and_migrate(cls, file_path: Path) -> Self:
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
        _migrate_pre_v7(data)
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
