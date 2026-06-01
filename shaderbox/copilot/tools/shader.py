from typing import Any

from pydantic import BaseModel, Field

from shaderbox.copilot.capabilities import (
    CompileErrorInfo,
    CopilotCapabilities,
    EditResult,
)
from shaderbox.copilot.tools.base import GatePolicy, ToolDefinition

# Slice 1 — the edit / compile-feedback vertical (spec §16). Three eager, current-node
# tools: read the shader, edit-and-recompile, read compile errors. All non-destructive
# (gate_policy NONE). The handlers do the GL-free text work on the worker thread and call
# a capability closure for anything GL-touching (the closure owns the bridge round-trip;
# marshalling is invisible here — leaf-seam rule §3.2).


class _NoArgs(BaseModel):
    model_config = {"extra": "forbid"}


class _EditArgs(BaseModel):
    old_str: str = Field(description="exact substring of the current source to replace")
    new_str: str = Field(description="the replacement text")
    replace_all: bool = Field(
        default=False,
        description="replace every occurrence (resolves a non-unique old_str)",
    )
    model_config = {"extra": "forbid"}


class _ReplaceLinesArgs(BaseModel):
    start_line: int = Field(description="first line to replace (1-based, inclusive)")
    end_line: int = Field(description="last line to replace (1-based, inclusive)")
    new_text: str = Field(
        description="replacement text (no trailing newline needed); empty string deletes "
        "the range"
    )
    model_config = {"extra": "forbid"}


class _InsertAfterArgs(BaseModel):
    line: int = Field(
        description="insert after this 1-based line; 0 inserts at the top of the file"
    )
    new_text: str = Field(description="text to insert (no trailing newline needed)")
    model_config = {"extra": "forbid"}


_GET_CURRENT_SHADER_DESC = (
    "Return the source of the shader you are currently working on, with line numbers "
    "(for your orientation only — when you edit, you match on text content, NOT line "
    "numbers), plus its active uniforms and any current compile errors. ALWAYS call this "
    "before editing — you cannot edit a shader you have not read this turn."
)

_EDIT_SHADER_DESC = (
    "Replace an exact substring of the current shader's source with new text, then "
    "recompile. old_str must match the file EXACTLY, including whitespace and "
    "indentation. If old_str appears more than once, the edit fails — provide a larger "
    "old_str with surrounding context to make it unique, or set replace_all=true to "
    "replace every occurrence. After the edit I recompile and return any compile errors "
    "at the exact line they occur; if there are none, the edit compiled clean. You "
    "cannot see the rendered image — never claim a visual result, only that it compiled."
)

_GET_COMPILE_ERRORS_DESC = (
    "Return the current shader's compile errors as path:line: message, or none if it "
    "compiles. Use this to inspect errors without editing (e.g. when the user reports a "
    "render error)."
)

_REPLACE_LINES_DESC = (
    "Replace an inclusive range of lines [start_line, end_line] (1-based, the line numbers "
    "shown by get_current_shader) with new_text, then recompile. ALWAYS call "
    "get_current_shader first — the line numbers must be current. new_text is inserted "
    "verbatim, so include the exact indentation you want (it does not have to match the old "
    "lines). An empty new_text deletes the range. To insert without replacing, use "
    "insert_after instead. After the edit I recompile and return any compile errors plus a "
    "snippet of the changed region; if there are none, it compiled clean."
)

_INSERT_AFTER_DESC = (
    "Insert new_text as new line(s) AFTER the given 1-based line (the numbers shown by "
    "get_current_shader); pass 0 to insert at the very top, or the last line number to "
    "append at the end. ALWAYS call get_current_shader first. new_text is inserted verbatim "
    "— include the indentation you want. Existing lines shift down; nothing is replaced. "
    "After the edit I recompile and return any compile errors plus a snippet of the changed "
    "region; if there are none, it compiled clean."
)

_OUT_OF_RANGE = (
    "error: line number out of range — re-read with get_current_shader and use a line "
    "number it shows"
)


def _format_errors(errors: list[CompileErrorInfo]) -> str:
    return "\n".join(f"{e.path}:{e.line}: {e.message}" for e in errors)


def _stale_result(result: EditResult) -> tuple[bool, str, dict] | None:
    # A freshness reject (feature 020 · 15): the source moved since the agent read it.
    # payload["stale"] lets run_turn keep it OUT of the edit-retry cap (re-read, don't spiral).
    if result.stale:
        return False, f"error: {result.stale_reason}", {"stale": True}
    return None


def _applied_result(result: EditResult) -> tuple[bool, str, dict]:
    # The shared success/compile-error message for any applied edit (edit_shader /
    # replace_lines / insert_after). Appends the "what changed" excerpt when present, or a
    # region count for a multi-span replace_all (no single excerpt — §14 D5).
    if result.errors:
        head = "compiled with errors:\n" + _format_errors(result.errors)
    else:
        head = "ok — compiled clean"
    if result.changed_excerpt:
        head += f"\nchanged lines:\n{result.changed_excerpt}"
    elif result.matches > 1:
        head += f" ({result.matches} regions changed)"
    return True, head, {"errors": [e.__dict__ for e in result.errors]}


def shader_tools(caps: CopilotCapabilities) -> list[ToolDefinition]:
    def get_current_shader(_args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        view = caps.get_current_shader_view()
        if view is None:
            return False, "error: no shader is currently open", None
        uniforms = (
            "\n".join(view.uniforms)
            if view.uniforms
            else "(none — shader does not compile)"
            if view.errors
            else "(none)"
        )
        errors = _format_errors(view.errors) if view.errors else "none"
        msg = f"{view.listing}\n\nuniforms:\n{uniforms}\n\nerrors:\n{errors}"
        return True, msg, {"errors": [e.__dict__ for e in view.errors]}

    def edit_shader(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        # The match + replace + recompile run against the node's CURRENT source on the
        # main thread (caps.apply_shader_edit, §16.3) — no source re-read here, so a unique
        # source-of-truth and no staleness window. `matches` decides the §16.4 string.
        result = caps.apply_shader_edit(
            args["old_str"], args["new_str"], args["replace_all"]
        )
        if (stale := _stale_result(result)) is not None:
            return stale
        if result.matches == 0:
            base = (
                "error: old_str not found in the shader — re-read with "
                "get_current_shader and copy an exact substring"
            )
            if result.hint:
                # A unique region matches ignoring whitespace — the model's old_str only
                # differs in indentation/spacing. Hand it the exact bytes to copy.
                base += (
                    "\nThe closest region differs only in whitespace. Copy this EXACTLY "
                    f"as old_str:\n{result.hint}"
                )
            return False, base, None
        if result.matches > 1 and not args["replace_all"]:
            return (
                False,
                f"error: old_str is not unique ({result.matches} matches) — add "
                "surrounding context to make it unique, or set replace_all=true",
                None,
            )
        return _applied_result(result)

    def replace_lines(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        start, end = args["start_line"], args["end_line"]
        if start > end:
            return (
                False,
                f"error: start_line ({start}) is after end_line ({end}) — to insert "
                "without replacing, use insert_after",
                None,
            )
        result = caps.apply_line_edit(start, end, args["new_text"])
        if (stale := _stale_result(result)) is not None:
            return stale
        if result.matches == 0:
            return False, _OUT_OF_RANGE, None
        return _applied_result(result)

    def insert_after(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        line = args["line"]
        result = caps.apply_line_edit(line + 1, line, args["new_text"])
        if (stale := _stale_result(result)) is not None:
            return stale
        if result.matches == 0:
            return False, _OUT_OF_RANGE, None
        return _applied_result(result)

    def get_compile_errors(_args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        errors = caps.get_compile_errors_current()
        if errors:
            return (
                True,
                _format_errors(errors),
                {"errors": [e.__dict__ for e in errors]},
            )
        return True, "none — compiles clean", {"errors": []}

    return [
        ToolDefinition(
            name="get_current_shader",
            description=_GET_CURRENT_SHADER_DESC,
            args_model=_NoArgs,
            handler=get_current_shader,
            mutating=False,
            needs_gl=True,
            category="shader",
            eager=True,
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="edit_shader",
            description=_EDIT_SHADER_DESC,
            args_model=_EditArgs,
            handler=edit_shader,
            mutating=True,
            needs_gl=True,
            category="shader",
            eager=True,
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="replace_lines",
            description=_REPLACE_LINES_DESC,
            args_model=_ReplaceLinesArgs,
            handler=replace_lines,
            mutating=True,
            needs_gl=True,
            category="shader",
            eager=True,
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="insert_after",
            description=_INSERT_AFTER_DESC,
            args_model=_InsertAfterArgs,
            handler=insert_after,
            mutating=True,
            needs_gl=True,
            category="shader",
            eager=True,
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="get_compile_errors",
            description=_GET_COMPILE_ERRORS_DESC,
            args_model=_NoArgs,
            handler=get_compile_errors,
            mutating=False,
            needs_gl=True,
            category="shader",
            eager=True,
            gate_policy=GatePolicy.NONE,
        ),
    ]
