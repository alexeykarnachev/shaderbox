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

from shaderbox.copilot.gate import GateKind
from shaderbox.copilot.llm.api import LLMMessage
from shaderbox.copilot.state import (
    ChatState,
    Message,
    MessageRole,
    RecoverInfo,
    ResultWidget,
    ResultWidgetKind,
    SessionUsage,
)

_VERSION = 5

_RESULT_WIDGET_KINDS: frozenset[str] = frozenset({"open_url", "open_path"})


def _gate_kind_or_confirm(value: str) -> GateKind:
    # A future GateKind member on an older build loads as CONFIRM (the renderer falls back to
    # Yes/No), instead of failing the whole conversation file (feature 020·19).
    try:
        return GateKind(value)
    except ValueError:
        return GateKind.CONFIRM


def _result_widget_or_none(model: "_ResultWidgetModel | None") -> ResultWidget | None:
    # An unknown widget kind (a future kind on an older build) or an empty target drops to None
    # instead of failing the file or rendering a dead button (feature 020·21).
    if model is None or model.kind not in _RESULT_WIDGET_KINDS or not model.target:
        return None
    return ResultWidget(
        kind=cast(ResultWidgetKind, model.kind), label=model.label, target=model.target
    )


class _RecoverModel(BaseModel):
    # The persisted Recover affordance on a resolved-Yes delete card (feature 020·17). Stores
    # the trash dir-NAME (NOT an absolute path — the project dir is relocatable; re-anchored
    # via App.trash_dir at click time).
    node_id: str
    node_name: str = ""
    trash_name: str
    done: bool = False
    model_config = {"extra": "forbid"}


class _ResultWidgetModel(BaseModel):
    # A persisted first-class result widget (feature 020·21). kind is loose-str so a future widget
    # kind on an older build round-trips (the renderer skips an unknown kind), like gate_kind/role.
    kind: str
    label: str = "Open"
    target: str = ""
    model_config = {"extra": "forbid"}


class _MessageModel(BaseModel):
    # The UI render stream (ChatState.messages). role is validated loosely (str) so a
    # future MessageRole member loads on an older build instead of failing the whole file.
    # gate_kind is loose-str for the same reason (feature 020·19); gate_input (the typed
    # secret buffer) is deliberately NOT a field — it is never persisted.
    role: str
    text: str = ""
    resolved: bool = False
    recover: _RecoverModel | None = None
    gate_kind: str = "confirm"
    # The result widget on a tool_status card (feature 020·21). Optional + defaulted so a v3 file
    # (no field) loads as None.
    result_widget: _ResultWidgetModel | None = None
    model_config = {"extra": "forbid"}


class _HistoryModel(BaseModel):
    # The LLM replay context (CopilotSession.history) — NATURAL-LANGUAGE only (feature 020·28):
    # user messages + one engine-derived assistant turn-summary each. No tool messages. A pre-v5
    # store carrying tool_call_id / tool_calls hits extra="forbid" -> ValidationError ->
    # load_and_migrate fail-softs it to an empty chat (no backward-compat migration by design).
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
        )

    def to_messages(self) -> list[Message]:
        # role is a free str on disk (loose so a future MessageRole member survives an
        # older build); cast back to the Literal — the renderer treats an unknown role as
        # plain text anyway, so the value is safe regardless.
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
        # NL-only (feature 020·28): every persisted history message is a plain user/assistant message.
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
