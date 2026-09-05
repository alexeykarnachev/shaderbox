from typing import Any

from pydantic import Field

from shaderbox.copilot.capabilities import (
    CopilotCapabilities,
    ScriptView,
    ScriptWriteResult,
)
from shaderbox.copilot.error_render import format_compile_errors
from shaderbox.copilot.tools.base import GatePolicy, ToolArgs, ToolDefinition

# The document script authoring surface (feature 043): read_script / write_script. Mirrors shader_tools —
# thin handlers calling a capability closure that owns the bridge round-trip + the dry-run probe.

_NODE_DESC = "document id (from the project map); empty = the document you are currently working on"


class _ReadScriptArgs(ToolArgs):
    document: str = Field(default="", description=_NODE_DESC)


class _WriteScriptArgs(ToolArgs):
    new_text: str = Field(
        description="the script's COMPLETE new source — this replaces the whole script.py "
        "(a `class Behavior(ScriptBehavior)` with `update(self, ctx) -> dict`). Anything "
        "omitted is gone."
    )
    document: str = Field(default="", description=_NODE_DESC)


class _EditScriptArgs(ToolArgs):
    old_str: str = Field(
        description="exact substring of the script.py source to replace (copied VERBATIM "
        "from read_script / the working set)"
    )
    new_str: str = Field(description="the replacement text (empty deletes the region)")
    replace_all: bool = Field(
        default=False,
        description="replace every occurrence (resolves a non-unique old_str)",
    )
    document: str = Field(default="", description=_NODE_DESC)


_READ_SCRIPT_DESC = (
    "Read a document's Python script — the `update(self, ctx)` that drives uniforms from CPU state. "
    "Returns the source line-numbered. A document with NO script yet returns a STUB (its drivable "
    "uniforms + their value shapes + an empty `update` to fill in) — read it, then write_script a "
    "real body. Read this before editing a script you did not just write."
)

_WRITE_SCRIPT_DESC = (
    "Create or replace a document's Python script: a `class Behavior(ScriptBehavior)` whose "
    "`update(self, ctx) -> dict` returns {uniform_name: value} to drive those uniforms every "
    "frame. For STATE the shader cannot hold — a value that depends on the PREVIOUS frame "
    "(`self.*` persists): an integrator, an accumulator, a phase machine, a score. A pure function "
    "of time belongs in the shader (u_time), not here. BEST FOR a fresh script or a full rewrite; "
    "for a localized change prefer edit_script. Send the COMPLETE script — I compile + motion-probe "
    "it and return the verdict."
)

_EDIT_SCRIPT_DESC = (
    "THE partial-edit tool for a script — the mirror of edit_shader, for script.py instead of GLSL. "
    "Replace an exact substring (old_str = the region copied VERBATIM from read_script / the working "
    "set; new_str = its replacement; empty new_str deletes; non-unique old_str fails — add context or "
    "replace_all=true). For a localized tweak; use write_script for a fresh script or a full rewrite. "
    "I re-compile + motion-probe and return the same verdict as write_script."
)


def _format_write_result(result: ScriptWriteResult) -> tuple[bool, str, dict | None]:
    # The shared agent-facing message for a write_script OR edit_script result (identical feedback).
    if not result.ok:
        return False, f"error: {result.error}", None
    if result.restored_note:
        # A force-restore is a SUCCESSFUL write of the last clean source — no errors payload,
        # so it never counts as applied-with-errors.
        return True, result.restored_note, None
    if result.compile_error:
        return (
            True,
            f"compiled with errors:\n{result.compile_error}\n-> fix the compile "
            "first (no uniforms driven, no motion probe). Same as a shader compile.",
            {"errors": [result.compile_error]},
        )
    if not result.driven:
        return True, f"ok -- {result.motion_facts}", None
    head = f"ok -- script compiled clean, drives {', '.join(result.driven)}"
    tail: list[str] = [result.motion_facts]
    for line in result.per_key_errors:
        tail.append(f"-> 1 key skipped: {line}")
    for name in result.orphan_keys:
        tail.append(
            f"-> '{name}' is not an active uniform -- declare it in the SHADER first "
            "(edit_shader), or fix the name."
        )
    return (
        True,
        head + "\n" + "\n".join(t for t in tail if t),
        {"driven": result.driven},
    )


def script_tools(caps: CopilotCapabilities) -> list[ToolDefinition]:
    def read_script(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        view: ScriptView = caps.read_script(args["document"])
        if not view.document_id:
            return False, f"error: {format_compile_errors(view.errors)}", None
        lines = view.listing.count("\n") + 1 if view.listing else 0
        if view.is_stub:
            # The stub is NOT persisted, so the working set can't render it — inline is its
            # only channel (a document WITH a script rides the working set, mirroring read_shader).
            body = (
                f"{view.name} has no script yet — here is the STUB to adapt + write_script "
                f"(its drivable uniforms + their value shapes + an empty `update` to fill "
                f"in):\n{view.listing}"
            )
        else:
            state = (
                f"{len(view.errors)} error(s):\n{format_compile_errors(view.errors)}"
                if view.errors
                else "compiles clean"
            )
            body = (
                f"added {view.name}'s script.py to your working set — {lines} lines, {state} "
                "(its live source is shown below; don't expect it in this return)"
            )
        return True, body, {"document": view.document_id, "is_stub": view.is_stub}

    def write_script(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        return _format_write_result(
            caps.write_script(args["new_text"], args["document"])
        )

    def edit_script(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        return _format_write_result(
            caps.apply_script_edit(
                args["old_str"], args["new_str"], args["replace_all"], args["document"]
            )
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
        ToolDefinition(
            name="edit_script",
            label_live="Editing script",
            label_done="Edited script",
            description=_EDIT_SCRIPT_DESC,
            args_model=_EditScriptArgs,
            handler=edit_script,
            mutating=True,
            is_edit=True,
            eager=True,
            gate_policy=GatePolicy.NONE,
        ),
    ]
