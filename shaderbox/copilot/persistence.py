"""On-disk persistence for a copilot conversation.

Durable per-project state in the project dir. Owns the versioned `ConversationStore`
pydantic model + load/save/archive (`extra="forbid"`, fail-soft `load_and_migrate`), and
maps the runtime dataclasses (`Message` / `LLMMessage` / `SessionUsage`) to/from their
persisted shape so `state.py` stays free of disk concerns.
"""

import json
from pathlib import Path
from typing import Literal, Self, cast

from loguru import logger
from pydantic import BaseModel, ValidationError

from shaderbox.copilot.gate import GateKind
from shaderbox.copilot.llm.api import LLMMessage, LLMUsage
from shaderbox.copilot.state import (
    ChatState,
    Message,
    MessageRole,
    RecoverInfo,
    ResultWidget,
    ResultWidgetKind,
    SessionUsage,
)

_VERSION = 6

_RESULT_WIDGET_KINDS: frozenset[str] = frozenset({"open_url", "open_path"})


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
    model_config = {"extra": "forbid"}


class _HistoryModel(BaseModel):
    # LLM replay context — NATURAL-LANGUAGE only: user messages + one assistant turn-summary
    # each, no tool messages. A pre-v5 store with tool_call_id / tool_calls hits extra="forbid"
    # and fail-softs to an empty chat (no backward-compat migration by design).
    role: str
    content: str | None = None
    model_config = {"extra": "forbid"}


class _UsageModel(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model_config = {"extra": "forbid"}


class ConversationStore(BaseModel):
    version: int = _VERSION
    messages: list[_MessageModel] = []
    history: list[_HistoryModel] = []
    usage: _UsageModel = _UsageModel()
    last_turn_usage: _UsageModel | None = None
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
                )
                for m in state.messages
            ],
            history=[_HistoryModel(role=h.role, content=h.content) for h in history],
            usage=_UsageModel(
                input_tokens=state.usage.input_tokens,
                output_tokens=state.usage.output_tokens,
                cost_usd=state.usage.cost_usd,
            ),
            last_turn_usage=(
                _UsageModel(
                    input_tokens=state.last_turn_usage.input_tokens,
                    output_tokens=state.last_turn_usage.output_tokens,
                    cost_usd=state.last_turn_usage.cost_usd,
                )
                if state.last_turn_usage is not None
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

    def to_usage(self) -> SessionUsage:
        return SessionUsage(
            input_tokens=self.usage.input_tokens,
            output_tokens=self.usage.output_tokens,
            cost_usd=self.usage.cost_usd,
        )

    def to_last_turn_usage(self) -> LLMUsage | None:
        if self.last_turn_usage is None:
            return None
        return LLMUsage(
            input_tokens=self.last_turn_usage.input_tokens,
            output_tokens=self.last_turn_usage.output_tokens,
            cost_usd=self.last_turn_usage.cost_usd,
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
