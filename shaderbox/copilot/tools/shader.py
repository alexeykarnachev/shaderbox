from typing import Any

from pydantic import Field

from shaderbox.copilot.address import is_lib_address
from shaderbox.copilot.capabilities import (
    CompileErrorInfo,
    CopilotCapabilities,
    EditResult,
    GrepHit,
    ShaderView,
)
from shaderbox.copilot.tools.base import GatePolicy, ToolArgs, ToolDefinition

# The shader tool surface. Handlers do GL-free text work on the worker thread and call a
# capability closure for anything GL-touching (the closure owns the bridge round-trip).


class _ReadShaderArgs(ToolArgs):
    nodes: list[str] = Field(
        default_factory=list,
        description="node ids (from the project map) and/or lib: addresses (from the "
        "catalogue or grep) to read; empty = the shader you are currently working on. "
        "NEVER means 'all'.",
    )


_TARGET_DESC = (
    "what to edit: empty = the shader you're currently working on; a node id (from the "
    "project map) = that node; a 'lib:<path>' address (from the library catalogue) = that "
    "library file"
)


class _EditArgs(ToolArgs):
    old_str: str = Field(description="exact substring of the source to replace")
    new_str: str = Field(description="the replacement text")
    replace_all: bool = Field(
        default=False,
        description="replace every occurrence (resolves a non-unique old_str)",
    )
    target: str = Field(default="", description=_TARGET_DESC)


class _ReplaceLinesArgs(ToolArgs):
    start_line: int | None = Field(
        default=None,
        description="first line to replace (1-based, inclusive); OMIT both start_line "
        "and end_line to replace the ENTIRE file",
    )
    end_line: int | None = Field(
        default=None, description="last line to replace (1-based, inclusive)"
    )
    first_line: str | None = Field(
        default=None,
        description="ranged replace only: the exact CURRENT content of start_line, "
        "copied verbatim from the working set — checked before applying",
    )
    last_line: str | None = Field(
        default=None,
        description="ranged replace only: the exact CURRENT content of end_line, "
        "copied verbatim from the working set — checked before applying",
    )
    new_text: str = Field(
        description="replacement text (no trailing newline needed); empty string deletes "
        "the range"
    )
    target: str = Field(default="", description=_TARGET_DESC)


class _InsertAfterArgs(ToolArgs):
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


class _SetUniformArgs(ToolArgs):
    name: str = Field(description="the uniform's name (e.g. u_color)")
    value: float | int | str | list[float | int] = Field(
        description="the value, shaped to the uniform: a NUMBER for a scalar; a LIST of numbers for "
        "a vector ([1.0, 0.0, 0.0] for a vec3) or a numeric array; a STRING for a uint text array "
        '(e.g. "Hello\\nWorld" for u_text — ShaderBox converts it to codepoints, the same control '
        "the user has in the UI). To change displayed TEXT, set the text uniform with a string — "
        "do NOT edit the source (a uniform can't be default-initialized)."
    )
    node: str = Field(
        default="",
        description="node id (from the project map); empty = the node you're working on",
    )


class _CreateNodeArgs(ToolArgs):
    name: str = Field(description="a display name for the new node")
    template: str = Field(
        default="",
        description="a template: handle from the TEMPLATE LIBRARY to start from (e.g. for a "
        "text-rendering shader, pick the text template) — prefer this over writing source blind; "
        "empty = the default blank-canvas starter",
    )
    source: str = Field(
        default="",
        description="initial GLSL source; empty = the chosen template (or the starter), or full "
        "GLSL following the project conventions (overrides the template body)",
    )
    switch_to: bool = Field(
        default=True,
        description="switch the user's view to the new node (true), or create it in the "
        "background and keep editing via its returned id (false)",
    )


class _GrepArgs(ToolArgs):
    query: str = Field(
        description="substring to find across every node's source and the library"
    )


class _ReadLibArgs(ToolArgs):
    names: list[str] = Field(
        description="library function names (SB_*) whose full body you want to read"
    )


class _DeleteNodeArgs(ToolArgs):
    node: str = Field(
        description="node id (from the project map) to delete — REQUIRED, never empty "
        "(deleting is destructive, so the target must be explicit, not the current node)"
    )


class _SwitchNodeArgs(ToolArgs):
    node: str = Field(
        description="node id (from the project map) to make the current shader — REQUIRED"
    )


_READ_SHADER_DESC = (
    "Bring shader nodes into your WORKING SET — their full live source (line-numbered), uniforms, "
    "and compile errors then appear in the working-set block at the bottom of the conversation, "
    "rebuilt every step with CURRENT line numbers. Pass a list of node ids to add several at once "
    "(e.g. to compare two shaders); a `lib:` address reads that library file whole the same way; "
    "leave nodes empty for the node you are currently working on (it is already in your working "
    "set). Use this to add a DIFFERENT node before editing it. The node you are currently working "
    "on needs no read before you edit it — its source is already in the working set. You cannot "
    "see the rendered image — never claim a visual result."
)

_EDIT_SHADER_DESC = (
    "BEST FOR a SMALL, localized change to a short unique snippet — no line numbers involved "
    "at all. For replacing a whole block/function prefer replace_lines in WHOLE-FILE mode "
    "(ranged only when the file is large), and for ADDING new lines prefer insert_after — "
    "both let you skip re-typing a large old_str. Replace an exact substring of the source "
    "with new text, then recompile. old_str must match the file EXACTLY, including whitespace "
    "and indentation. If old_str appears more than once, the edit fails — provide a larger "
    "old_str with surrounding context, or set replace_all=true. After the edit I recompile and "
    "return any compile errors at the exact line; if there are none, the edit compiled clean. "
    "You cannot see the rendered image — never claim a visual result, only that it compiled."
)

_REPLACE_LINES_DESC = (
    "Replace a line range, or the WHOLE file. WHOLE-FILE mode is the DEFAULT for replacing a "
    "function/block in a small-to-medium file: OMIT start_line/end_line entirely — the working "
    "set already shows the whole file, and if it is roughly <=150 lines just rewrite it whole; "
    "new_text replaces the whole file, no line numbers to get wrong. RANGED mode is ONLY for a "
    "large block inside a LARGE file, where a whole-file rewrite would be wasteful: pass "
    "[start_line, end_line] (1-based, inclusive, from the WORKING SET block — current for THIS "
    "step) AND first_line + last_line: the exact current content of those two boundary lines, "
    "copied verbatim from the working set. If they "
    "don't match what is really there, NOTHING is applied and the result shows the actual lines — "
    "fix the range and resubmit. The range must cover EVERYTHING new_text replaces. new_text is "
    "inserted verbatim (include the indentation you want); an empty new_text deletes the range. "
    "To insert without replacing, use insert_after. (One line-addressed edit per file per step — "
    "see HOW TO WORK.) After the edit I recompile and return any compile errors; if there are "
    "none, it compiled clean."
)

_INSERT_AFTER_DESC = (
    "BEST FOR adding new lines (a uniform, a helper function, a statement) — you quote no "
    "anchor text at all, just the line to insert after. Insert new_text as new line(s) AFTER "
    "the given 1-based line (use the WORKING SET block's current numbers); pass 0 to insert at the "
    "very top, or the last line number to append at the end. new_text is inserted verbatim — "
    "include the indentation you want. Existing lines shift down; nothing is replaced. (One "
    "line-addressed edit per file per step — see HOW TO WORK.) After the edit I recompile and "
    "return any compile errors; if there are none, it compiled clean."
)

_SET_UNIFORM_DESC = (
    "Change a uniform's runtime VALUE — for tweaking a number/vector the user controls live "
    "(brightness, speed, a color), WITHOUT editing code. Read the node first so you know the "
    "uniform's type and current value. Only scalar and vector uniforms can be set; samplers, "
    "uniform blocks, and engine-driven uniforms (u_time, u_aspect, u_resolution) cannot. To "
    "change the shader's LOGIC, or to add/remove a uniform, edit the SOURCE instead. The value "
    "is in memory until the user saves the project; you cannot see the result — report the "
    "value you set, not how it looks."
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

_SWITCH_NODE_DESC = (
    "Make a node the CURRENT shader (the one the user is viewing). The publish and render tools, "
    "and edits with no target, all act on the current shader — so to publish/render a DIFFERENT "
    "node, switch to it first. Pass the node id from the project map. Non-destructive: the user's "
    "view switches to it."
)


def _out_of_range(result: EditResult) -> str:
    where = f" in {result.target_label}" if result.target_label else ""
    return (
        f"error: line number out of range{where} — use a line number from the WORKING "
        "SET block below, which has this file's current line numbers"
    )


def _format_errors(errors: list[CompileErrorInfo]) -> str:
    return "\n".join(f"{e.path}:{e.line}: {e.message}" for e in errors)


def _view_summary(view: ShaderView) -> str:
    # The terse chat line for a read: name + size + uniform count + compile state (the full
    # listing goes to the agent's context, not the chat).
    lines = view.listing.count("\n") + 1 if view.listing else 0
    if is_lib_address(view.node_id):
        return (
            f"read {view.name} — {lines} lines (library file — no standalone compile)"
        )
    state = f"{len(view.errors)} compile error(s)" if view.errors else "compiled clean"
    return f"read {view.name} — {lines} lines, {len(view.uniforms)} uniforms, {state}"


def _format_hits(hits: list[GrepHit]) -> str:
    return "\n".join(f"{h.location}:{h.line}: {h.text}  [{h.origin}]" for h in hits)


def _unresolved_result(result: EditResult) -> tuple[bool, str, None] | None:
    # An unresolvable reject (bad node id / lib path, read-only template, failed lib write,
    # intra-batch line-edit guard). Counts toward the edit-retry cap.
    if result.unresolved:
        msg = f"error: {result.unresolved_reason}"
        if result.target_label:
            msg += f" (checked {result.target_label})"
        return False, msg, None
    return None


def _applied_result(result: EditResult) -> tuple[bool, str, dict]:
    # The shared success/compile-error message for any applied edit. A LIB edit returns the
    # "no standalone compile" note instead of a compile result; a NODE edit returns compile
    # errors or "compiled clean". Region count only for a multi-span replace_all.
    if result.restored_note:
        head = result.restored_note
        if result.render_facts:
            head += "\n" + result.render_facts
    elif result.lib_note:
        head = result.lib_note
    elif result.errors:
        head = "compiled with errors:\n" + _format_errors(result.errors)
        if result.error_hints:
            head += "\n" + "\n".join(result.error_hints)
    else:
        head = "ok — compiled clean"
        if result.target_label:
            head += f" ({result.target_label})"
        if result.render_facts:
            head += "\n" + result.render_facts
    if result.matches > 1:
        head += f" ({result.matches} regions changed)"
    return True, head, {"errors": [e.__dict__ for e in result.errors]}


def shader_tools(caps: CopilotCapabilities) -> list[ToolDefinition]:
    def read_shader(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        # nodes are short ids from the project map; [] = current node (resolved App-side). The
        # read ADDS each node to the working set, whose per-turn scratchpad carries the full
        # source — so the agent body is a short confirmation + compile errors, not the listing.
        node_ids: list[str] = list(args["nodes"])
        views = caps.read_shaders(node_ids)
        if not views:
            return (
                False,
                "error: no such node(s) — check the project map for ids "
                "(library files: a lib: address from the catalogue or grep)",
                None,
            )
        # A handle is "found" if it matches a returned view's SHORT id under the resolver's
        # prefix rule (either is a prefix of the other) — a direct compare would mis-report a
        # full-id/long-prefix read as missing.
        short_ids = [v.node_id for v in views]
        missing = [
            nid
            for nid in node_ids
            if not any(s.startswith(nid) or nid.startswith(s) for s in short_ids)
        ]
        names = ", ".join(v.name for v in views)
        body = (
            f"added to your working set: {names} (their live source is shown below). "
        )
        # Lib views carry no compile (errors=[] by construction) — kept OUT of the
        # compiled-clean claim, which covers nodes only.
        node_views = [v for v in views if not is_lib_address(v.node_id)]
        lib_views = [v for v in views if is_lib_address(v.node_id)]
        all_errors = [e for v in node_views for e in v.errors]
        states: list[str] = []
        if all_errors:
            states.append("compile errors:\n" + _format_errors(all_errors))
        elif node_views:
            states.append("all compiled clean.")
        if lib_views:
            lib_names = ", ".join(v.name for v in lib_views)
            kind = "library files" if len(lib_views) > 1 else "library file"
            states.append(f"({lib_names}: {kind} — no standalone compile)")
        body += " ".join(states)
        if missing:
            body += f"\n(no node found for: {', '.join(missing)})"
        payload = {
            "errors": [e.__dict__ for e in all_errors],
            "read": [v.node_id for v in views],
            "display": "\n".join(_view_summary(v) for v in views),
        }
        return True, body, payload

    def edit_shader(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        result = caps.apply_shader_edit(
            args["old_str"], args["new_str"], args["replace_all"], args["target"]
        )
        if (unresolved := _unresolved_result(result)) is not None:
            return unresolved
        if result.comment_loss:
            return (
                False,
                "error: that region spans a comment your old_str doesn't reproduce, so "
                "replacing it verbatim would delete the comment. Use replace_lines (addressed "
                "by line number) so the surrounding lines stay intact.",
                None,
            )
        if result.matches == 0:
            where = result.target_label or "the shader"
            base = (
                f"error: old_str not found in {where} — re-read with read_shader and "
                "copy an exact substring"
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
        first, last = args["first_line"], args["last_line"]
        if (start is None) != (end is None):
            return (
                False,
                "error: provide BOTH start_line and end_line for a ranged replace, "
                "or NEITHER to replace the whole file",
                None,
            )
        if start is None or end is None:
            result = caps.apply_line_edit(
                0, 0, args["new_text"], args["target"], None, None
            )
            if (unresolved := _unresolved_result(result)) is not None:
                return unresolved
            return _applied_result(result)
        if start > end:
            return (
                False,
                f"error: start_line ({start}) is after end_line ({end}) — to insert "
                "without replacing, use insert_after",
                None,
            )
        if first is None or last is None:
            return (
                False,
                "error: a ranged replace requires first_line AND last_line — copy the "
                "CURRENT content of those two lines verbatim from the working set (or "
                "omit the range entirely to replace the whole file)",
                None,
            )
        result = caps.apply_line_edit(
            start, end, args["new_text"], args["target"], first, last
        )
        if (unresolved := _unresolved_result(result)) is not None:
            return unresolved
        if result.matches == 0:
            return False, _out_of_range(result), None
        return _applied_result(result)

    def insert_after(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        line = args["line"]
        result = caps.apply_line_edit(
            line + 1, line, args["new_text"], args["target"], None, None
        )
        if (unresolved := _unresolved_result(result)) is not None:
            return unresolved
        if result.matches == 0:
            return False, _out_of_range(result), None
        return _applied_result(result)

    def set_uniform(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        result = caps.set_uniform(args["name"], args["value"], args["node"])
        if not result.ok:
            return False, f"error: {result.error}", None
        msg = (
            f"set {args['name']} ({result.type_label}) = {args['value']} "
            "(not saved until the project is saved)"
        )
        if result.render_facts:
            msg += "\n" + result.render_facts
        return True, msg, None

    def create_node(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        node_id, errors, extra = caps.create_node(
            args["name"], args["source"], args["template"], args["switch_to"]
        )
        where = "now active" if args["switch_to"] else "in the background"
        # success stays True even with compile errors — the node IS created; the agent reads
        # the errors and fixes them with an edit.
        status = (
            "compiled with errors:\n" + _format_errors(errors)
            if errors
            else "compiled clean"
        )
        body = (
            f"created node '{args['name']}' (id: {node_id}), {where} — {status}. "
            "Read it before editing."
        )
        if extra:
            body += "\n" + extra
        return (
            True,
            body,
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

    def _node_display(node: str) -> str:
        # Display NAME for a gate prompt, resolved via the project map with read_shader's
        # prefix rule. GL-free + worker-safe (node_tree is the per-iteration context read).
        # Falls back to the raw arg — a wrong id still shows the user what was asked.
        if not node:
            return "?"
        for e in caps.node_tree():
            if e.node_id.startswith(node) or node.startswith(e.node_id):
                return e.name
        return node

    def delete_node(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        # The gate (GatePolicy.ALWAYS) has already cleared a user Yes by the time this runs.
        # payload carries node_id + trash_name for the chat's Recover affordance.
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

    def switch_node(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        result = caps.switch_node(args["node"])
        if not result.ok:
            return False, f"error: {result.error}", None
        return (
            True,
            f"switched the current shader to '{result.name}'. Publish/render/edits with no "
            "target now act on it.",
            {"switched": args["node"]},
        )

    return [
        ToolDefinition(
            name="read_shader",
            label_live="Reading shader",
            label_done="Read shader",
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
            label_live="Editing shader",
            label_done="Edited shader",
            description=_EDIT_SHADER_DESC,
            args_model=_EditArgs,
            handler=edit_shader,
            mutating=True,
            is_edit=True,
            needs_gl=True,
            category="shader",
            eager=True,
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="replace_lines",
            label_live="Editing shader",
            label_done="Edited shader",
            description=_REPLACE_LINES_DESC,
            args_model=_ReplaceLinesArgs,
            handler=replace_lines,
            mutating=True,
            is_edit=True,
            needs_gl=True,
            category="shader",
            eager=True,
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="insert_after",
            label_live="Editing shader",
            label_done="Edited shader",
            description=_INSERT_AFTER_DESC,
            args_model=_InsertAfterArgs,
            handler=insert_after,
            mutating=True,
            is_edit=True,
            needs_gl=True,
            category="shader",
            eager=True,
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="set_uniform",
            label_live="Setting uniform",
            label_done="Set uniform",
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
            label_live="Creating node",
            label_done="Created node",
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
            label_live="Searching",
            label_done="Searched",
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
            label_live="Reading library",
            label_done="Read library",
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
            label_live="Deleting node",
            label_done="Deleted node",
            description=_DELETE_NODE_DESC,
            args_model=_DeleteNodeArgs,
            handler=delete_node,
            mutating=True,
            needs_gl=True,
            category="shader",
            eager=True,
            gate_policy=GatePolicy.ALWAYS,
            gate_prompt=lambda a: (
                f"Delete node `{_node_display(a.get('node', ''))}`?"
            ),
        ),
        ToolDefinition(
            name="switch_node",
            label_live="Switching node",
            label_done="Switched node",
            description=_SWITCH_NODE_DESC,
            args_model=_SwitchNodeArgs,
            handler=switch_node,
            mutating=False,
            needs_gl=True,
            category="shader",
            eager=True,
            gate_policy=GatePolicy.NONE,
        ),
    ]
