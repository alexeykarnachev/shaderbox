from typing import Any

from pydantic import Field

from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.error_render import format_compile_errors
from shaderbox.copilot.tools.base import GatePolicy, ToolArgs, ToolDefinition

# Node file-management tools (feature 052 slice 3): rename / resize-canvas / duplicate. All mutate
# node.json (checkpoint-revertable via the backend's _capture_node / mark_created), handle-addressed
# by node id, no gate.

_NODE_DESC = "node id (from the project map)"


class _RenameNodeArgs(ToolArgs):
    new_name: str = Field(description="the node's new display name")
    node: str = Field(default="", description=_NODE_DESC)


class _SetCanvasSizeArgs(ToolArgs):
    width: int = Field(description="canvas width in pixels (16-4096)")
    height: int = Field(description="canvas height in pixels (16-4096)")
    node: str = Field(default="", description=_NODE_DESC)


class _DuplicateNodeArgs(ToolArgs):
    node: str = Field(default="", description=_NODE_DESC)
    new_name: str = Field(
        default="", description="name for the copy; empty = '<original> copy'"
    )
    switch_to: bool = Field(default=False, description="make the copy the current node")


class _ImportNodeArgs(ToolArgs):
    switch_to: bool = Field(
        default=False, description="make the imported node the current node"
    )


def node_ops_tools(caps: CopilotCapabilities) -> list[ToolDefinition]:
    def rename_node(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        res = caps.rename_node(args["node"], args["new_name"])
        if not res.ok:
            return False, f"error: {res.error}", None
        return True, f"renamed to '{res.name}'.", None

    def set_canvas_size(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        res = caps.set_canvas_size(args["node"], args["width"], args["height"])
        if not res.ok:
            return False, f"error: {res.error}", None
        return True, f"canvas set to {res.width}x{res.height}.", None

    def duplicate_node(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        new_id, errors, extra = caps.duplicate_node(
            args["node"], args["new_name"], args["switch_to"]
        )
        head = f"duplicated to node {new_id}"
        if errors:
            body = format_compile_errors(errors)
            return True, f"{head} ({len(errors)} compile error(s)):\n{body}", None
        return True, f"{head}.\n{extra}" if extra else f"{head}.", None

    def import_node(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        res = caps.import_node(args["switch_to"])
        if res.cancelled:
            return (
                True,
                "the user dismissed the file picker — nothing was imported.",
                None,
            )
        if not res.ok:
            return False, f"error: {res.error}", None
        head = f"imported {res.basename} -> node {res.node_id}"
        if res.errors:
            body = format_compile_errors(res.errors)
            return True, f"{head} ({len(res.errors)} compile error(s)):\n{body}", None
        return True, f"{head}.", None

    return [
        ToolDefinition(
            name="rename_node",
            label_live="Renaming node",
            label_done="Renamed node",
            description="Rename a node's display name (the id is unchanged). Reversible.",
            args_model=_RenameNodeArgs,
            handler=rename_node,
            mutating=True,
            eager=False,
            catalog_summary="rename a node's display name",
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="set_canvas_size",
            label_live="Resizing canvas",
            label_done="Resized canvas",
            description=(
                "Set a node's native canvas resolution (its render size, shown as `canvas WxH` in "
                "the working-set header). Clamped to 16-4096. Use it when the user wants a specific "
                "resolution or the render looks low-res."
            ),
            args_model=_SetCanvasSizeArgs,
            handler=set_canvas_size,
            mutating=True,
            eager=False,
            catalog_summary="set a node's render resolution (canvas WxH)",
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="duplicate_node",
            label_live="Duplicating node",
            label_done="Duplicated node",
            description=(
                "Fork a node into a new one (copies its shader, script, and bound media) so the "
                "user can try a variant without losing the original."
            ),
            args_model=_DuplicateNodeArgs,
            handler=duplicate_node,
            mutating=True,
            eager=False,
            catalog_summary="fork a node into a variant (copies shader/script/media)",
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="import_node",
            label_live="Importing shader",
            label_done="Imported shader",
            description=(
                "Import a .glsl/.frag shader file from the user's disk into the project as a new "
                "node. Opens the USER's native file picker (you never type a path — they choose the "
                "file). A broken import still creates the node and returns its compile errors to fix."
            ),
            args_model=_ImportNodeArgs,
            handler=import_node,
            mutating=True,
            eager=False,
            catalog_summary="import a .glsl shader file from the user's disk as a new node",
            gate_policy=GatePolicy.NONE,
        ),
    ]
