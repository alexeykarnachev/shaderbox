from typing import Any

from pydantic import Field

from shaderbox.copilot.capabilities import (
    CompileErrorInfo,
    CopilotCapabilities,
    ScriptView,
    ScriptWriteResult,
)
from shaderbox.copilot.tools.base import GatePolicy, ToolArgs, ToolDefinition

# The node-brain authoring surface (feature 043): read_script / write_script. Mirrors shader_tools —
# thin handlers calling a capability closure that owns the bridge round-trip + the dry-run probe.

_NODE_DESC = (
    "node id (from the project map); empty = the node you are currently working on"
)


class _ReadScriptArgs(ToolArgs):
    node: str = Field(default="", description=_NODE_DESC)


class _WriteScriptArgs(ToolArgs):
    new_text: str = Field(
        description="the brain's COMPLETE new source — this replaces the whole script.py "
        "(a `class Behavior(ScriptBehavior)` with `update(self, ctx) -> dict`). Anything "
        "omitted is gone."
    )
    node: str = Field(default="", description=_NODE_DESC)


_READ_SCRIPT_DESC = (
    "Read a node's Python brain script (nodes/<id>/scripts/script.py) — the `update(self, ctx)` "
    "that drives uniforms over time. Returns the source line-numbered. A node with NO brain yet "
    "returns a STUB (the node's drivable uniforms + their value shapes + one ctx.t example to "
    "ADAPT) — read it, then write_script a real body. Read this before editing a script you did "
    "not just write."
)

_WRITE_SCRIPT_DESC = (
    "Create or replace a node's Python brain (script.py): a `class Behavior(ScriptBehavior)` whose "
    "`update(self, ctx) -> dict` returns {uniform_name: value} to DRIVE those uniforms every frame "
    "(use it for ANIMATION / state over time — ctx.t/dt/frame; self.* persists). Send the COMPLETE "
    "script. After the write I compile it and probe it: I return the compile error (fix it like a "
    "shader compile), the uniforms it now drives (drives 0 = it animates NOTHING — a real problem to "
    "fix, not success), and a MOTION verdict (which values change across t: ANIMATING / STATIC). "
    "ctx.mouse is frozen in the probe — drive from ctx.t to verify motion. You cannot see the image "
    "beyond these facts."
)


def _fmt_errors(errors: list[CompileErrorInfo]) -> str:
    return "\n".join(f"{e.path}:{e.line}: {e.message}" for e in errors)


def script_tools(caps: CopilotCapabilities) -> list[ToolDefinition]:
    def read_script(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        view: ScriptView = caps.read_script(args["node"])
        if not view.node_id:
            return False, f"error: {_fmt_errors(view.errors)}", None
        lines = view.listing.count("\n") + 1 if view.listing else 0
        if view.is_stub:
            body = (
                f"{view.name} has no brain yet — here is the STUB to adapt + write_script "
                f"(its drivable uniforms + one ctx.t example):\n{view.listing}"
            )
        else:
            state = (
                f"{len(view.errors)} error(s):\n{_fmt_errors(view.errors)}"
                if view.errors
                else "compiles clean"
            )
            body = (
                f"read {view.name}'s script.py — {lines} lines, {state}\n{view.listing}"
            )
        return True, body, {"node": view.node_id, "is_stub": view.is_stub}

    def write_script(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        result: ScriptWriteResult = caps.write_script(args["new_text"], args["node"])
        if not result.ok:
            return False, f"error: {result.error}", None
        if result.compile_error:
            return (
                True,
                f"compiled with errors:\n{result.compile_error}\n-> fix the compile "
                "first (no uniforms driven, no motion probe). Same as a shader compile.",
                None,
            )
        if not result.driven:
            return True, f"ok -- {result.motion_facts}", None
        head = f"ok -- script compiled clean, drives {', '.join(result.driven)}"
        tail: list[str] = [result.motion_facts]
        for line in result.per_key_errors:
            tail.append(f"-> 1 key skipped: {line}")
        for line in result.orphan_keys:
            tail.append(
                f"-> '{line.split(':', 1)[0]}' is not an active uniform ({line})"
            )
        return (
            True,
            head + "\n" + "\n".join(t for t in tail if t),
            {"driven": result.driven},
        )

    return [
        ToolDefinition(
            name="read_script",
            label_live="Reading script",
            label_done="Read script",
            description=_READ_SCRIPT_DESC,
            args_model=_ReadScriptArgs,
            handler=read_script,
            mutating=False,
            eager=True,
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="write_script",
            label_live="Writing script",
            label_done="Wrote script",
            description=_WRITE_SCRIPT_DESC,
            args_model=_WriteScriptArgs,
            handler=write_script,
            mutating=True,
            is_edit=True,
            eager=True,
            gate_policy=GatePolicy.NONE,
        ),
    ]
