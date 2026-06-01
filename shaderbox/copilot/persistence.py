"""On-disk persistence for a copilot conversation (feature 022).

The conversation is durable, per-project state — same class as `app_state.json` (021
decision 9) — so it lives in the project dir and travels with it. This module owns the
versioned `ConversationStore` pydantic model + its load/save/archive, mirroring
`UIAppState`'s discipline (`extra="forbid"`, fail-soft `load_and_migrate`). It maps the
runtime dataclasses (`Message` / `LLMMessage` / `SessionUsage`, which `state.py` keeps
runtime-pure) to/from their persisted shape here, so `state.py` stays free of disk concerns.
"""

import json
from pathlib import Path
from typing import Literal, Self, cast

from loguru import logger
from pydantic import BaseModel, ValidationError

from shaderbox.copilot.llm.api import LLMMessage, LLMToolCall
from shaderbox.copilot.state import ChatState, Message, MessageRole, SessionUsage

_VERSION = 1


class _ToolCallModel(BaseModel):
    id: str
    name: str
    arguments: str
    model_config = {"extra": "forbid"}


class _MessageModel(BaseModel):
    # The UI render stream (ChatState.messages). role is validated loosely (str) so a
    # future MessageRole member loads on an older build instead of failing the whole file.
    role: str
    text: str = ""
    resolved: bool = False
    model_config = {"extra": "forbid"}


class _HistoryModel(BaseModel):
    # The LLM replay context (CopilotSession.history). Optional tool fields tolerate older
    # shapes (the read-first / line-edit waves added tool calls).
    role: str
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[_ToolCallModel] | None = None
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
    model_config = {"extra": "forbid"}

    @classmethod
    def from_runtime(
        cls, state: ChatState, history: list[LLMMessage]
    ) -> "ConversationStore":
        return cls(
            version=_VERSION,
            messages=[
                _MessageModel(role=m.role, text=m.text, resolved=m.resolved)
                for m in state.messages
            ],
            history=[
                _HistoryModel(
                    role=h.role,
                    content=h.content,
                    tool_call_id=h.tool_call_id,
                    tool_calls=(
                        [
                            _ToolCallModel(id=t.id, name=t.name, arguments=t.arguments)
                            for t in h.tool_calls
                        ]
                        if h.tool_calls is not None
                        else None
                    ),
                )
                for h in history
            ],
            usage=_UsageModel(
                input_tokens=state.usage.input_tokens,
                output_tokens=state.usage.output_tokens,
                cost_usd=state.usage.cost_usd,
            ),
        )

    def to_messages(self) -> list[Message]:
        # role is a free str on disk (loose so a future MessageRole member survives an
        # older build); cast back to the Literal — the renderer treats an unknown role as
        # plain text anyway, so the value is safe regardless.
        return [
            Message(role=cast(MessageRole, m.role), text=m.text, resolved=m.resolved)
            for m in self.messages
        ]

    def to_history(self) -> list[LLMMessage]:
        out: list[LLMMessage] = []
        for h in self.history:
            calls = (
                [
                    LLMToolCall(id=t.id, name=t.name, arguments=t.arguments)
                    for t in h.tool_calls
                ]
                if h.tool_calls is not None
                else None
            )
            role = cast(Literal["system", "user", "assistant", "tool"], h.role)
            out.append(
                LLMMessage(
                    role=role,
                    content=h.content,
                    tool_call_id=h.tool_call_id,
                    tool_calls=calls,
                )
            )
        return out

    def to_usage(self) -> SessionUsage:
        return SessionUsage(
            input_tokens=self.usage.input_tokens,
            output_tokens=self.usage.output_tokens,
            cost_usd=self.usage.cost_usd,
        )

    def save(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load_and_migrate(cls, file_path: Path) -> Self:
        # Fail-soft, mirroring UIAppState: a missing file is a first-use empty chat; an
        # unreadable / incompatible one degrades to empty + a file-only WARNING, never a
        # crash and never a lost project (021 leveling).
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
    # (recoverable, not destroyed — Q1: uncapped, projects + clears are both rare).
    if not conversation_path.exists():
        return
    archive_dir = conversation_path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    conversation_path.replace(archive_dir / f"conversation_{stamp}.json")
