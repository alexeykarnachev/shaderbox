from typing import Any

from pydantic import BaseModel, Field

from shaderbox.copilot.capabilities import (
    CompileErrorInfo,
    CopilotCapabilities,
    EditResult,
    GrepHit,
    ShaderView,
)
from shaderbox.copilot.tools.base import GatePolicy, ToolDefinition

# The shader tool surface (spec §16 + feature 020·16 cross-project wave). read_shader is the
# one read (source + uniforms/values + errors, for one or many nodes); the three edit tools
# match/line/insert against a target (current node by default); grep + read_lib are discovery.
# All non-destructive (gate_policy NONE). Handlers do GL-free text work on the worker thread
# and call a capability closure for anything GL-touching (the closure owns the bridge
# round-trip; marshalling is invisible here — leaf-seam rule §3.2).


class _ReadShaderArgs(BaseModel):
    nodes: list[str] = Field(
        default_factory=list,
        description="node ids to read (from the project map); empty = the shader you are "
        "currently working on. NEVER means 'all'.",
    )
    model_config = {"extra": "forbid"}


_TARGET_DESC = (
    "what to edit: empty = the shader you're currently working on; a node id (from the "
    "project map) = that node; a 'lib:<path>' address (from the library catalogue) = that "
    "library file"
)


class _EditArgs(BaseModel):
    old_str: str = Field(description="exact substring of the source to replace")
    new_str: str = Field(description="the replacement text")
    replace_all: bool = Field(
        default=False,
        description="replace every occurrence (resolves a non-unique old_str)",
    )
    target: str = Field(default="", description=_TARGET_DESC)
    model_config = {"extra": "forbid"}


class _ReplaceLinesArgs(BaseModel):
    start_line: int = Field(description="first line to replace (1-based, inclusive)")
    end_line: int = Field(description="last line to replace (1-based, inclusive)")
    new_text: str = Field(
        description="replacement text (no trailing newline needed); empty string deletes "
        "the range"
    )
    target: str = Field(default="", description=_TARGET_DESC)
    model_config = {"extra": "forbid"}


class _InsertAfterArgs(BaseModel):
    line: int = Field(
        description="insert after this 1-based line; 0 inserts at the top of the file"
    )
    new_text: str = Field(description="text to insert (no trailing newline needed)")
    target: str = Field(
        default="",
        description=_TARGET_DESC
        + " — a 'lib:<path>' that doesn't exist yet is CREATED here (the way to add a new "
        "library function)",
    )
    model_config = {"extra": "forbid"}


class _SetUniformArgs(BaseModel):
    name: str = Field(description="the uniform's name (e.g. u_color)")
    value: float | int | list[float] = Field(
        description="a number for a scalar uniform, or a list of numbers for a vector "
        "(e.g. [1.0, 0.0, 0.0] for a vec3 color)"
    )
    node: str = Field(
        default="",
        description="node id (from the project map); empty = the node you're working on",
    )
    model_config = {"extra": "forbid"}


class _CreateNodeArgs(BaseModel):
    name: str = Field(description="a display name for the new node")
    source: str = Field(
        default="",
        description="initial GLSL source; empty = a ready-made starter shader you then edit, "
        "or full GLSL following the project conventions",
    )
    switch_to: bool = Field(
        default=True,
        description="switch the user's view to the new node (true), or create it in the "
        "background and keep editing via its returned id (false)",
    )
    model_config = {"extra": "forbid"}


class _GrepArgs(BaseModel):
    query: str = Field(
        description="substring to find across every node's source and the library"
    )
    model_config = {"extra": "forbid"}


class _ReadLibArgs(BaseModel):
    names: list[str] = Field(
        description="library function names (SB_*) whose full body you want to read"
    )
    model_config = {"extra": "forbid"}


class _DeleteNodeArgs(BaseModel):
    node: str = Field(
        description="node id (from the project map) to delete — REQUIRED, never empty "
        "(deleting is destructive, so the target must be explicit, not the current node)"
    )
    model_config = {"extra": "forbid"}


_READ_SHADER_DESC = (
    "Read shader nodes: returns each node's source with line numbers (for orientation — you "
    "match on text content, NOT line numbers, when editing), its uniforms (name, type, and "
    "current value), and any compile errors. Pass a list of node ids to read several at once "
    "(e.g. to compare two shaders); leave nodes empty to read the one you are currently working "
    "on. ALWAYS read a node before editing it — you cannot edit a shader you have not read this "
    "turn. You cannot see the rendered image — never claim a visual result."
)

_EDIT_SHADER_DESC = (
    "BEST FOR a SMALL, localized change to a unique snippet. For replacing a whole "
    "block/function prefer replace_lines, and for ADDING new lines prefer insert_after — "
    "both let you skip re-typing a large old_str. Replace an exact substring of the source "
    "with new text, then recompile. old_str must match the file EXACTLY, including whitespace "
    "and indentation. If old_str appears more than once, the edit fails — provide a larger "
    "old_str with surrounding context, or set replace_all=true. After the edit I recompile and "
    "return any compile errors at the exact line; if there are none, the edit compiled clean. "
    "You cannot see the rendered image — never claim a visual result, only that it compiled."
)

_REPLACE_LINES_DESC = (
    "BEST FOR replacing a whole block or function — you give the line numbers, so you never "
    "re-type the old lines (cheaper + no exact-match risk vs a large edit_shader old_str). "
    "Replace an inclusive range of lines [start_line, end_line] (1-based, the line numbers "
    "shown by read_shader) with new_text, then recompile. ALWAYS read_shader first — the line "
    "numbers must be current. new_text is inserted verbatim, so include the exact indentation "
    "you want. An empty new_text deletes the range. To insert without replacing, use "
    "insert_after. After the edit I recompile and return any compile errors plus a snippet of "
    "the changed region; if there are none, it compiled clean."
)

_INSERT_AFTER_DESC = (
    "BEST FOR adding new lines (a uniform, a helper function, a statement) — you quote no "
    "anchor text at all, just the line to insert after. Insert new_text as new line(s) AFTER "
    "the given 1-based line (the numbers shown by read_shader); pass 0 to insert at the very "
    "top, or the last line number to append at the end. ALWAYS read_shader first. new_text is "
    "inserted verbatim — include the indentation you want. Existing lines shift down; nothing "
    "is replaced. After the edit I recompile and return any compile errors plus a snippet of "
    "the changed region; if there are none, it compiled clean."
)

_SET_UNIFORM_DESC = (
    "Change a uniform's runtime VALUE — for tweaking a number/vector the user controls live "
    "(brightness, speed, a color), WITHOUT editing code. Read the node first so you know the "
    "uniform's type and current value. Only scalar and vector uniforms can be set; samplers, "
    "uniform blocks, and engine-driven uniforms (u_time, u_aspect, u_resolution) cannot. To "
    "change the shader's LOGIC, or to add/remove a uniform, edit the SOURCE instead. The value "
    "is in memory until the user saves the project; you cannot see the result — describe it."
)

_CREATE_NODE_DESC = (
    "Create a new shader node, then compile it and return the result — compile errors at their "
    "exact line, or that it compiled clean (same feedback as an edit). Leave source empty for a "
    "ready-made starter shader you then edit; or pass full GLSL (follow the project shader "
    "conventions so it compiles). By default the new node becomes the user's active node (so a "
    "follow-up edit with no target lands on it); pass switch_to=false to create it in the "
    "background and edit it via the node id this returns. Returns the new node's id and its "
    "compile result."
)

_GREP_DESC = (
    "Find where a substring occurs across every shader node and the whole library. Returns "
    "origin-labeled file:line hits (a node id, or a lib: address). Use it to LOCATE something "
    "(e.g. which shaders use u_time, or where a helper is called); then read_shader / read_lib "
    "to read the full thing. Substring match (a comment can match too)."
)

_READ_LIB_DESC = (
    "Read the full body of one or more library functions by name (the catalogue in your "
    "context lists signatures only). Use this when you need to see how a SB_* helper works "
    "before calling it, or to read it before editing it."
)

_DELETE_NODE_DESC = (
    "Delete a shader node. Pass the node id (from the project map) — required, never empty. "
    "This is destructive, so the user is shown a Yes/No confirmation before it happens; if they "
    "decline you'll get 'user declined' and should stop and explain. The node moves to the project "
    "trash and the user can recover it, so reassure them it's not permanently lost. After a delete "
    "the node is gone from the project map; do not read or edit it again."
)

_OUT_OF_RANGE = "error: line number out of range — re-read with read_shader and use a line number it shows"


def _format_errors(errors: list[CompileErrorInfo]) -> str:
    return "\n".join(f"{e.path}:{e.line}: {e.message}" for e in errors)


def _format_view(view: ShaderView) -> str:
    uniforms = "\n".join(view.uniforms) if view.uniforms else "(none)"
    errors = _format_errors(view.errors) if view.errors else "none"
    return (
        f"=== {view.name} (id: {view.node_id}) ===\n{view.listing}\n\n"
        f"uniforms:\n{uniforms}\n\nerrors:\n{errors}"
    )


def _format_hits(hits: list[GrepHit]) -> str:
    return "\n".join(f"{h.location}:{h.line}: {h.text}  [{h.origin}]" for h in hits)


def _stale_result(result: EditResult) -> tuple[bool, str, dict] | None:
    # A freshness reject (feature 020 · 15): the source moved since the agent read it.
    # payload["stale"] lets run_turn keep it OUT of the edit-retry cap (re-read, don't spiral).
    if result.stale:
        return False, f"error: {result.stale_reason}", {"stale": True}
    return None


def _applied_result(result: EditResult) -> tuple[bool, str, dict]:
    # The shared success/compile-error message for any applied edit (edit_shader /
    # replace_lines / insert_after). A LIB edit returns the honest "no standalone compile"
    # note (020·16 Decision 4) instead of a compile result; a NODE edit returns compile errors
    # or "compiled clean". Appends the "what changed" excerpt, or a region count for a
    # multi-span replace_all (no single excerpt — §14 D5).
    if result.lib_note:
        head = result.lib_note
    elif result.errors:
        head = "compiled with errors:\n" + _format_errors(result.errors)
    else:
        head = "ok — compiled clean"
    if result.changed_excerpt:
        head += f"\nchanged lines:\n{result.changed_excerpt}"
    elif result.matches > 1:
        head += f" ({result.matches} regions changed)"
    return True, head, {"errors": [e.__dict__ for e in result.errors]}


def shader_tools(caps: CopilotCapabilities) -> list[ToolDefinition]:
    def read_shader(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        # nodes are short ids from the project map; [] = current node (resolved App-side so the
        # tool layer never touches a full id). The returned ShaderViews carry short ids too.
        node_ids: list[str] = list(args["nodes"])
        views = caps.read_shaders(node_ids)
        if not views:
            return False, "error: no such node(s) — check the project map for ids", None
        found = {v.node_id for v in views}
        missing = [nid for nid in node_ids if nid not in found]
        body = "\n\n".join(_format_view(v) for v in views)
        if missing:
            body += f"\n\n(no node found for: {', '.join(missing)})"
        payload = {
            "errors": [e.__dict__ for v in views for e in v.errors],
            "read": [v.node_id for v in views],
        }
        return True, body, payload

    def edit_shader(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        result = caps.apply_shader_edit(
            args["old_str"], args["new_str"], args["replace_all"], args["target"]
        )
        if (stale := _stale_result(result)) is not None:
            return stale
        if result.matches == 0:
            base = (
                "error: old_str not found in the shader — re-read with read_shader and copy "
                "an exact substring"
            )
            if result.hint:
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
        result = caps.apply_line_edit(start, end, args["new_text"], args["target"])
        if (stale := _stale_result(result)) is not None:
            return stale
        if result.matches == 0:
            return False, _OUT_OF_RANGE, None
        return _applied_result(result)

    def insert_after(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        line = args["line"]
        result = caps.apply_line_edit(line + 1, line, args["new_text"], args["target"])
        if (stale := _stale_result(result)) is not None:
            return stale
        if result.matches == 0:
            return False, _OUT_OF_RANGE, None
        return _applied_result(result)

    def set_uniform(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        result = caps.set_uniform(args["name"], args["value"], args["node"])
        if not result.ok:
            return False, f"error: {result.error}", None
        return (
            True,
            f"set {args['name']} ({result.type_label}) = {args['value']} — look at the "
            "preview to confirm (not saved until you save the project)",
            None,
        )

    def create_node(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        node_id, errors = caps.create_node(
            args["name"], args["source"], args["switch_to"]
        )
        where = "now active" if args["switch_to"] else "in the background"
        # Same compile-result vocabulary as the edit tools (success stays True even with errors —
        # the node IS created; the agent reads the errors and fixes them with an edit).
        status = (
            "compiled with errors:\n" + _format_errors(errors)
            if errors
            else "compiled clean"
        )
        return (
            True,
            f"created node '{args['name']}' (id: {node_id}), {where} — {status}. "
            "Read it before editing.",
            {"created": node_id, "errors": [e.__dict__ for e in errors]},
        )

    def grep(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        hits = caps.grep(args["query"])
        if not hits:
            return True, "no matches", {"hits": 0}
        return True, _format_hits(hits), {"hits": len(hits)}

    def read_lib(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        names: list[str] = list(args["names"])
        bodies = caps.read_lib(names)
        if not bodies:
            return (
                False,
                "error: no library function found with that name — check the library "
                "catalogue in your context",
                None,
            )
        found = {b.name for b in bodies}
        body = "\n\n".join(f"// {b.lib_address}\n{b.body}" for b in bodies)
        missing = [n for n in names if n not in found]
        if missing:
            body += f"\n\n(no function found for: {', '.join(missing)})"
        return True, body, {"read": list(found)}

    def delete_node(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        # The gate (GatePolicy.ALWAYS) has already cleared a user Yes by the time this runs.
        # The payload carries node_id + trash_name so the chat can attach a Recover affordance.
        result = caps.delete_node(args["node"])
        if not result.ok:
            return False, f"error: {result.error}", None
        return (
            True,
            f"deleted node '{result.deleted_name}' — moved to the project trash (recoverable)",
            {
                "node_id": result.node_id,
                "trash_name": result.trash_name,
                "deleted_name": result.deleted_name,
            },
        )

    return [
        ToolDefinition(
            name="read_shader",
            description=_READ_SHADER_DESC,
            args_model=_ReadShaderArgs,
            handler=read_shader,
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
            name="set_uniform",
            description=_SET_UNIFORM_DESC,
            args_model=_SetUniformArgs,
            handler=set_uniform,
            mutating=True,
            needs_gl=True,
            category="shader",
            eager=True,
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="create_node",
            description=_CREATE_NODE_DESC,
            args_model=_CreateNodeArgs,
            handler=create_node,
            mutating=True,
            needs_gl=True,
            category="shader",
            eager=True,
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="grep",
            description=_GREP_DESC,
            args_model=_GrepArgs,
            handler=grep,
            mutating=False,
            needs_gl=False,
            category="shader",
            eager=True,
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="read_lib",
            description=_READ_LIB_DESC,
            args_model=_ReadLibArgs,
            handler=read_lib,
            mutating=False,
            needs_gl=False,
            category="shader",
            eager=True,
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="delete_node",
            description=_DELETE_NODE_DESC,
            args_model=_DeleteNodeArgs,
            handler=delete_node,
            mutating=True,
            needs_gl=True,
            category="shader",
            eager=True,
            gate_policy=GatePolicy.ALWAYS,
        ),
    ]
