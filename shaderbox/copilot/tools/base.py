from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any

from pydantic import BaseModel

from shaderbox.copilot.gate import GateKind
from shaderbox.copilot.llm.api import LLMToolSpec

# Leaf tool types: the ToolDefinition value object + the gate policy. Lives apart from
# registry.py so the per-domain tool modules (tools/shader.py, …) import these without
# cycling through the registry (which imports those modules to build the catalog).

# (ok, message_for_llm, payload_for_ui). Sync — handlers run on the worker thread.
ToolHandler = Callable[[dict[str, Any]], tuple[bool, str, dict[str, Any] | None]]
# A credential tool's handler (feature 020·19): receives the gate's typed secret as a SECOND
# arg, kept out of `args` (which the trace + debug log print). It sets the secret via its
# capability and returns a REDACTED message — that message is the only thing reaching LLM history.
CredentialToolHandler = Callable[
    [dict[str, Any], str], tuple[bool, str, dict[str, Any] | None]
]
# A pre-gate guard (feature 020·18): args -> a guided-handoff message, or None to proceed.
ToolPrecheck = Callable[[dict[str, Any]], str | None]


def mask_secret(s: str) -> str:
    # Redact a secret for any human-readable surface (feature 020·19): show the leading 6 chars
    # so two secrets are distinguishable, hide the rest. A Telegram token's prefix is the
    # public-ish bot id, so this never exposes the secret tail.
    return f"{s[:6]}…" if len(s) > 6 else "…"


class GatePolicy(StrEnum):
    # When a tool call must block on the user before running (§F4 / §2.3).
    NONE = auto()  # single reversible edits / reads — flow free
    BULK = auto()  # confirm when an arg count exceeds bulk_gate_threshold
    ALWAYS = auto()  # destructive ops + external publish — always confirm


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    args_model: type[BaseModel]  # pydantic — schema + validation from one definition
    handler: ToolHandler | CredentialToolHandler
    mutating: bool  # gates the "what I did" note + the UI confirm
    needs_gl: bool  # True => the handler's capability marshals to the main thread
    category: str  # the catalogue tree (§4); single source of truth for grouping
    eager: bool  # True => carried in eager_specs() (turn-start tools=) (§4)
    # Shader-SOURCE edit tools only (edit_shader / replace_lines / insert_after) — drives the
    # consecutive-failed-edits giveup cap. Deliberately NARROWER than `mutating`: a failed
    # render/publish must not trip the edit-retry cap.
    is_edit: bool = False
    gate_policy: GatePolicy = GatePolicy.NONE
    # The gate WIDGET (feature 020·19): CONFIRM = Yes/No, CREDENTIAL = a masked secret input.
    # Orthogonal to gate_policy (WHEN to gate vs WHICH widget). A CREDENTIAL tool's handler is a
    # CredentialToolHandler (gets the secret as a 2nd arg); secret_field is its marker/grep-anchor.
    gate_kind: GateKind = GateKind.CONFIRM
    secret_field: str = ""
    # A pre-gate guard (feature 020·18): returns a guided-handoff message when the call CAN'T
    # run (a publish with no credentials / no pack), else None. The loop runs it BEFORE the
    # gate, so a publish that can't proceed never pops a confirm. None => no precheck.
    precheck: ToolPrecheck | None = None

    def spec(self) -> LLMToolSpec:
        return LLMToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.args_model.model_json_schema(),
        )
