from typing import Any, cast

from loguru import logger
from pydantic import Field, ValidationError

from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.config import COPILOT_ENGINE
from shaderbox.copilot.errors import CopilotToolError
from shaderbox.copilot.gate import GateKind
from shaderbox.copilot.llm.api import LLMToolSpec
from shaderbox.copilot.tools.base import (
    CredentialToolHandler,
    GatePolicy,
    ToolArgs,
    ToolDefinition,
    ToolHandler,
)
from shaderbox.copilot.tools.inspect import inspect_tools
from shaderbox.copilot.tools.media import media_tools
from shaderbox.copilot.tools.node_ops import node_ops_tools
from shaderbox.copilot.tools.publish import publish_tools
from shaderbox.copilot.tools.script import script_tools
from shaderbox.copilot.tools.shader import shader_tools
from shaderbox.copilot.tools.telegram import telegram_tools
from shaderbox.copilot.tools.youtube import youtube_tools

LOAD_TOOLS_NAME = "load_tools"


class _LoadToolsArgs(ToolArgs):
    names: list[str] = Field(
        description="the lazy tool names to load for the rest of this turn (from the catalogue "
        "in this tool's description)"
    )


_LOAD_TOOLS_DESC = (
    "Load extra tools you need for THIS turn by name. To keep the toolset lean, the tools below are "
    "NOT loaded by default — call load_tools with their names to make them available for the rest of "
    "the turn. Load a tool BEFORE you need to call it. Available:\n"
)


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

    def assemble_specs(self, loaded: set[str]) -> list[LLMToolSpec]:
        # The `tools=` for a turn iteration: the eager core + any lazily-loaded tools, SORTED by name
        # so the block is byte-stable (prefix-cacheable) regardless of load order (feature 052 §3).
        chosen = [d for d in self._by_name.values() if d.eager or d.name in loaded]
        return [d.spec() for d in sorted(chosen, key=lambda d: d.name)]

    def is_lazy(self, name: str) -> bool:
        # A real, lazily-loadable tool (not eager, not the load_tools meta-tool itself).
        tool = self._by_name.get(name)
        return tool is not None and not tool.eager and name != LOAD_TOOLS_NAME

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

    def requires_gate(self, name: str, args: dict[str, Any]) -> bool:
        tool = self._by_name.get(name)
        if tool is None:
            return False
        if tool.gate_policy is GatePolicy.ALWAYS:
            return True
        if tool.gate_policy is GatePolicy.BULK:
            counts = [len(v) for v in args.values() if isinstance(v, list)]
            return bool(counts) and max(counts) > COPILOT_ENGINE.bulk_gate_threshold
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
        *script_tools(caps),
        *inspect_tools(caps),
        *node_ops_tools(caps),
        *media_tools(caps),
        *publish_tools(caps),
        *telegram_tools(caps),
        *youtube_tools(caps),
    ]

    def load_tools_handler(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        # Never actually invoked: run_turn intercepts load_tools before execute (it mutates the
        # turn's tools= set, engine state the handler can't reach). Present for schema + a benign
        # fallback if ever called directly.
        _ = args
        return True, "tools loaded", None

    lazy = sorted((d for d in definitions if not d.eager), key=lambda d: d.name)
    catalog = "\n".join(f"- {d.name}: {d.catalog_summary}" for d in lazy)
    load_tools_def = ToolDefinition(
        name=LOAD_TOOLS_NAME,
        label_live="Loading tools",
        label_done="Loaded tools",
        description=_LOAD_TOOLS_DESC + catalog,
        args_model=_LoadToolsArgs,
        handler=load_tools_handler,
        mutating=False,
        eager=True,
        gate_policy=GatePolicy.NONE,
    )
    return ToolRegistry([*definitions, load_tools_def])
