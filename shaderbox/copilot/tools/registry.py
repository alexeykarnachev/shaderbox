from typing import Any

from loguru import logger
from pydantic import ValidationError

from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.config import CopilotConfig
from shaderbox.copilot.llm.api import LLMToolSpec
from shaderbox.copilot.tools.base import GatePolicy, ToolDefinition
from shaderbox.copilot.tools.publish import publish_tools
from shaderbox.copilot.tools.shader import shader_tools


def _validation_message(exc: ValidationError) -> str:
    first = exc.errors()[0] if exc.errors() else {}
    return f"error: invalid arguments - {first.get('msg', 'invalid')}"


class ToolRegistry:
    def __init__(self, definitions: list[ToolDefinition]) -> None:
        self._by_name: dict[str, ToolDefinition] = {d.name: d for d in definitions}

    def eager_specs(self) -> list[LLMToolSpec]:
        # The turn-start tools= set: eager-core only. Lazy long-tail loads via
        # search_tools/list_tools (§4) — a later slice.
        return [d.spec() for d in self._by_name.values() if d.eager]

    def specs_for(self, names: list[str]) -> list[LLMToolSpec]:
        return [self._by_name[n].spec() for n in names if n in self._by_name]

    def describe(self) -> str:
        # Single source of truth for the system-prompt capability block + a future
        # user-facing "what the copilot can do" surface.
        return "\n".join(f"- {d.name}: {d.description}" for d in self._by_name.values())

    def is_mutating(self, name: str) -> bool:
        tool = self._by_name.get(name)
        return tool is not None and tool.mutating

    def requires_gate_always(self, name: str) -> bool:
        tool = self._by_name.get(name)
        return tool is not None and tool.gate_policy is GatePolicy.ALWAYS

    def precheck(self, name: str, args: dict[str, Any]) -> str | None:
        # Pre-gate guard (feature 020·18): a guided-handoff message when the call can't run
        # (a publish with no creds/pack), else None. None for any tool without a precheck.
        tool = self._by_name.get(name)
        if tool is None or tool.precheck is None:
            return None
        return tool.precheck(args)

    def requires_gate(
        self, name: str, args: dict[str, Any], config: CopilotConfig
    ) -> bool:
        tool = self._by_name.get(name)
        if tool is None:
            return False
        if tool.gate_policy is GatePolicy.ALWAYS:
            return True
        if tool.gate_policy is GatePolicy.BULK:
            counts = [len(v) for v in args.values() if isinstance(v, list)]
            return bool(counts) and max(counts) > config.bulk_gate_threshold
        return False

    def status_for(self, name: str, args: dict[str, Any] | None) -> str:
        # Per-tool human phrase for the status pill (§8/§G). Falls back to the bare
        # name. The catalog is small enough in slice 1 that the bare name reads fine;
        # richer templates land as the catalog grows.
        _ = args
        tool = self._by_name.get(name)
        return tool.name if tool is not None else name

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
            return False, f"error: {name} failed", None


def build_registry(caps: CopilotCapabilities) -> ToolRegistry:
    # Each tool group is a local addition here: the cross-project edit/read set (shader_tools,
    # §16) + the render/publish set (publish_tools, §18, all GatePolicy.ALWAYS).
    definitions: list[ToolDefinition] = [*shader_tools(caps), *publish_tools(caps)]
    return ToolRegistry(definitions)
