from typing import Any, cast

from loguru import logger
from pydantic import Field, ValidationError

from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.errors import CopilotToolError
from shaderbox.copilot.gate import GateKind, SourceLock
from shaderbox.copilot.llm.api import LLMToolSpec
from shaderbox.copilot.tools.base import (
    CredentialToolHandler,
    GatePolicy,
    ToolArgs,
    ToolDefinition,
    ToolHandler,
)
from shaderbox.copilot.tools.document_ops import document_ops_tools
from shaderbox.copilot.tools.inspect import inspect_tools
from shaderbox.copilot.tools.media import media_tools
from shaderbox.copilot.tools.passes import pass_tools
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
    # The offending field name rides `loc`; without it "Extra inputs are not permitted" reads as a
    # complaint about the VALUE and a model retries the same shape with different content.
    where = ".".join(str(part) for part in first.get("loc", ()))
    detail = first.get("msg", "invalid")
    return f"error: invalid arguments - {f'{where}: ' if where else ''}{detail}"


class ToolRegistry:
    def __init__(self, definitions: list[ToolDefinition]) -> None:
        self._by_name: dict[str, ToolDefinition] = {d.name: d for d in definitions}
        # The source lock (083, a user-set mode since 086), read by `must_confirm` on the WORKER
        # thread. Defaults ALLOW here on purpose, where a SESSION defaults ASK: a registry built
        # bare (every test does) is an inert catalogue, and gating it by default would silently
        # change what those tests measure.
        self.source_lock: SourceLock = SourceLock.ALLOW

    def eager_specs(self) -> list[LLMToolSpec]:
        # Turn-start tools= set: eager-core only (long-tail loads lazily).
        return [d.spec() for d in self._by_name.values() if d.eager]

    def assemble_specs(self, loaded: set[str]) -> list[LLMToolSpec]:
        # The `tools=` for a turn iteration: the eager core + any lazily-loaded tools, SORTED by name
        # so the block is byte-stable (prefix-cacheable) regardless of load order (feature 052 §3).
        #
        # READ_ONLY withholds the source-writing tools entirely rather than refusing their calls
        # afterwards: a model that cannot see a tool does not spend a call discovering it is barred,
        # and the prompt says WHY they are absent so it can tell the user instead of failing
        # mutely. Byte-stability is unaffected -- the mode is per project and cannot change without
        # a project switch, so the list is stable across a session's turns, which is all the cache
        # needs.
        chosen = [
            d
            for d in self._by_name.values()
            if (d.eager or d.name in loaded) and not self.is_withheld(d.name)
        ]
        return [d.spec() for d in sorted(chosen, key=lambda d: d.name)]

    def is_withheld(self, name: str) -> bool:
        """Is this tool absent from the request entirely, rather than merely gated?

        The roster is `locks_source` -- 083 D6's enumerated set, so what READ_ONLY withholds and
        what the lock covers cannot drift apart.
        """
        return self.source_lock is SourceLock.READ_ONLY and self.locks_source(name)

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

    def requires_gate(self, name: str) -> bool:
        # "Is this tool ALWAYS-gated?" -- and, because always-gated here means destructive or
        # external, ALSO the irreversibility test `_RunLog.summary_lines` asks when it decides
        # whether a ledger line carries its identity verbatim and uncapped. Two readers, one
        # meaning. The session lock deliberately does NOT widen this: a locked edit is confirmed,
        # not irreversible, and folding it in here would push every source edit of a locked
        # session into that uncapped branch of the turn summary.
        tool = self._by_name.get(name)
        return tool is not None and tool.gate_policy is GatePolicy.ALWAYS

    def locks_source(self, name: str) -> bool:
        tool = self._by_name.get(name)
        return tool is not None and tool.locks_source

    def must_confirm(self, name: str) -> bool:
        """Does this call need the user's permission at all? The agent loop's one question.

        Deliberately a BOOL over three lock states (085): whether the permission is ASKED for or
        refused outright also depends on the turn's deny latch, which is the loop's state and not
        the registry's -- a three-valued answer here could only ever be half of one.

        DENY answers True here, not False: the refusal happens INSIDE the loop's gate block, so a
        predicate that excused DENY would route the call past that block to `execute` and run the
        very edit the mode forbids."""
        return self.requires_gate(name) or (
            self.source_lock is not SourceLock.ALLOW and self.locks_source(name)
        )

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
        *document_ops_tools(caps),
        *pass_tools(caps),
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
