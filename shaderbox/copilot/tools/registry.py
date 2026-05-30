from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pydantic import BaseModel, ValidationError

from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.llm.api import LLMToolSpec

# (ok, message_for_llm, payload_for_ui). Sync — handlers run on the worker thread.
ToolHandler = Callable[[dict[str, Any]], tuple[bool, str, dict[str, Any] | None]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    args_model: type[BaseModel]  # pydantic — schema + validation from one definition
    handler: ToolHandler
    mutating: bool  # gates the "what I did" note + the UI confirm
    needs_gl: bool  # True => the handler's capability marshals to the main thread

    def spec(self) -> LLMToolSpec:
        return LLMToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.args_model.model_json_schema(),
        )


def _validation_message(exc: ValidationError) -> str:
    first = exc.errors()[0] if exc.errors() else {}
    return f"error: invalid arguments - {first.get('msg', 'invalid')}"


class ToolRegistry:
    def __init__(self, definitions: list[ToolDefinition]) -> None:
        self._by_name: dict[str, ToolDefinition] = {d.name: d for d in definitions}

    def specs(self) -> list[LLMToolSpec]:
        return [d.spec() for d in self._by_name.values()]

    def describe(self) -> str:
        # Single source of truth for the system-prompt capability block + a future
        # user-facing "what the copilot can do" surface.
        return "\n".join(f"- {d.name}: {d.description}" for d in self._by_name.values())

    def execute(
        self, name: str, raw_args: dict[str, Any]
    ) -> tuple[bool, str, dict[str, Any] | None]:
        tool = self._by_name.get(name)
        if tool is None:
            return False, f"error: unknown tool '{name}'", None
        try:
            args = tool.args_model.model_validate(raw_args)
        except ValidationError as exc:
            return False, _validation_message(exc), None
        try:
            return tool.handler(args.model_dump())
        except Exception:
            # Generic message only — never leak exception internals into LLM context.
            logger.exception(f"copilot tool failed: {name}")
            return False, f"error: tool '{name}' failed", None


def build_registry(caps: CopilotCapabilities) -> ToolRegistry:
    # The tool CATALOG is the later capability brainstorm (§0 #8). This wave ships the
    # registry machinery with an empty catalog; tool groups register here as they land,
    # each as a closure over `caps`.
    _ = caps
    definitions: list[ToolDefinition] = []
    return ToolRegistry(definitions)
