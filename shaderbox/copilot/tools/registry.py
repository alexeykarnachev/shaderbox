from typing import Any, cast

from loguru import logger
from pydantic import ValidationError

from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.config import CopilotConfig
from shaderbox.copilot.errors import CopilotToolError
from shaderbox.copilot.gate import GateKind
from shaderbox.copilot.llm.api import LLMToolSpec
from shaderbox.copilot.tools.base import (
    CredentialToolHandler,
    GatePolicy,
    ToolDefinition,
    ToolHandler,
)
from shaderbox.copilot.tools.publish import publish_tools
from shaderbox.copilot.tools.shader import shader_tools
from shaderbox.copilot.tools.telegram import telegram_tools
from shaderbox.copilot.tools.youtube import youtube_tools


def _validation_message(exc: ValidationError) -> str:
    first = exc.errors()[0] if exc.errors() else {}
    return f"error: invalid arguments - {first.get('msg', 'invalid')}"


class ToolRegistry:
    def __init__(self, definitions: list[ToolDefinition]) -> None:
        self._by_name: dict[str, ToolDefinition] = {d.name: d for d in definitions}

    def eager_specs(self) -> list[LLMToolSpec]:
        # Turn-start tools= set: eager-core only (long-tail loads lazily).
        return [d.spec() for d in self._by_name.values() if d.eager]

    def specs_for(self, names: list[str]) -> list[LLMToolSpec]:
        return [self._by_name[n].spec() for n in names if n in self._by_name]

    def describe(self) -> str:
        return "\n".join(f"- {d.name}: {d.description}" for d in self._by_name.values())

    def is_mutating(self, name: str) -> bool:
        tool = self._by_name.get(name)
        return tool is not None and tool.mutating

    def is_edit_tool(self, name: str) -> bool:
        tool = self._by_name.get(name)
        return tool is not None and tool.is_edit

    def requires_gate_always(self, name: str) -> bool:
        tool = self._by_name.get(name)
        return tool is not None and tool.gate_policy is GatePolicy.ALWAYS

    def definition_for(self, name: str) -> ToolDefinition | None:
        return self._by_name.get(name)

    def definitions(self) -> list[ToolDefinition]:
        return list(self._by_name.values())

    def label_for(self, name: str) -> str:
        # Past-tense card/hover label. Raw-name fallback: persisted StepRecords may carry a
        # renamed/removed tool.
        tool = self._by_name.get(name)
        return tool.label_done if tool is not None else name

    def precheck(self, name: str, args: dict[str, Any]) -> str | None:
        # Pre-gate guard: a handoff message when the call can't run (e.g. publish with
        # no creds/pack), else None.
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
        # Live status-pill phrase. `args` is the seam for arg-aware phrasing
        # ("Editing gradient...", 020/11 §2.3) — unused until that lands.
        _ = args
        tool = self._by_name.get(name)
        return tool.label_live if tool is not None else name

    def execute(
        self, name: str, raw_args: dict[str, Any], secret: str = ""
    ) -> tuple[bool, str, dict[str, Any] | None]:
        # `secret`: the gate's typed key for a CREDENTIAL tool. Kept OUT of args, which
        # the trace + debug log print.
        tool = self._by_name.get(name)
        if tool is None:
            return False, f"error: unknown tool '{name}'", None
        try:
            args = tool.args_model.model_validate(raw_args)
        except ValidationError as exc:
            return False, _validation_message(exc), None
        try:
            if tool.gate_kind is GateKind.CREDENTIAL:
                return cast(CredentialToolHandler, tool.handler)(
                    args.model_dump(), secret
                )
            return cast(ToolHandler, tool.handler)(args.model_dump())
        except CopilotToolError as exc:
            # A deliberate domain reject: the message is authored for the model. Log at warning
            # (expected control flow, not a bug) and surface it verbatim.
            logger.warning(f"copilot tool rejected: {name}: {exc}")
            return False, f"error: {exc}", None
        except Exception as exc:
            # An unexpected bug: surface only the class name (never the message/traceback — those
            # can carry paths/secrets); the full traceback goes to the debug log.
            logger.exception(f"copilot tool failed: {name}")
            return False, f"error: {name} failed ({type(exc).__name__})", None


def build_registry(caps: CopilotCapabilities) -> ToolRegistry:
    definitions: list[ToolDefinition] = [
        *shader_tools(caps),
        *publish_tools(caps),
        *telegram_tools(caps),
        *youtube_tools(caps),
    ]
    return ToolRegistry(definitions)
