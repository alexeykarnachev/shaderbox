from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any

from pydantic import BaseModel

from shaderbox.copilot.llm.api import LLMToolSpec

# Leaf tool types: the ToolDefinition value object + the gate policy. Lives apart from
# registry.py so the per-domain tool modules (tools/shader.py, …) import these without
# cycling through the registry (which imports those modules to build the catalog).

# (ok, message_for_llm, payload_for_ui). Sync — handlers run on the worker thread.
ToolHandler = Callable[[dict[str, Any]], tuple[bool, str, dict[str, Any] | None]]


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
    handler: ToolHandler
    mutating: bool  # gates the "what I did" note + the UI confirm
    needs_gl: bool  # True => the handler's capability marshals to the main thread
    category: str  # the catalogue tree (§4); single source of truth for grouping
    eager: bool  # True => carried in eager_specs() (turn-start tools=) (§4)
    gate_policy: GatePolicy = GatePolicy.NONE

    def spec(self) -> LLMToolSpec:
        return LLMToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.args_model.model_json_schema(),
        )
