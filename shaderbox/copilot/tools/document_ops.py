from typing import Any

from pydantic import Field

from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.error_render import format_compile_errors
from shaderbox.copilot.tools.base import GatePolicy, ToolArgs, ToolDefinition

# Document file-management tools (feature 052 slice 3): rename / resize-canvas / duplicate. All mutate
# document.json (checkpoint-revertable via the backend's _capture_document / mark_created), handle-addressed
# by document id, no gate.

_NODE_DESC = "document id (from the project map)"


class _RenameDocumentArgs(ToolArgs):
    new_name: str = Field(description="the document's new display name")
    document: str = Field(default="", description=_NODE_DESC)


class _SetCanvasSizeArgs(ToolArgs):
    width: int = Field(description="canvas width in pixels (16-4096)")
    height: int = Field(description="canvas height in pixels (16-4096)")
    document: str = Field(default="", description=_NODE_DESC)


class _DuplicateDocumentArgs(ToolArgs):
    document: str = Field(default="", description=_NODE_DESC)
    new_name: str = Field(
        default="", description="name for the copy; empty = '<original> copy'"
    )
    switch_to: bool = Field(
        default=False, description="make the copy the current document"
    )


class _ImportDocumentArgs(ToolArgs):
    switch_to: bool = Field(
        default=False, description="make the imported document the current document"
    )


def document_ops_tools(caps: CopilotCapabilities) -> list[ToolDefinition]:
    def rename_document(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        res = caps.rename_document(args["document"], args["new_name"])
        if not res.ok:
            return False, f"error: {res.error}", None
        return True, f"renamed to '{res.name}'.", None

    def set_canvas_size(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        res = caps.set_canvas_size(args["document"], args["width"], args["height"])
        if not res.ok:
            return False, f"error: {res.error}", None
        return True, f"canvas set to {res.width}x{res.height}.", None

    def duplicate_document(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        new_id, errors, extra = caps.duplicate_document(
            args["document"], args["new_name"], args["switch_to"]
        )
        head = f"duplicated to document {new_id}"
        if errors:
            body = format_compile_errors(errors)
            return True, f"{head} ({len(errors)} compile error(s)):\n{body}", None
        return True, f"{head}.\n{extra}" if extra else f"{head}.", None

    def import_document(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        res = caps.import_document(args["switch_to"])
        if res.cancelled:
            return (
                True,
                "the user dismissed the file picker — nothing was imported.",
                None,
            )
        if not res.ok:
            return False, f"error: {res.error}", None
        head = f"imported {res.basename} -> document {res.document_id}"
        if res.errors:
            body = format_compile_errors(res.errors)
            return True, f"{head} ({len(res.errors)} compile error(s)):\n{body}", None
        return True, f"{head}.", None

    return [
        ToolDefinition(
            name="rename_document",
            label_live="Renaming document",
            label_done="Renamed document",
            description="Rename a document's display name (the id is unchanged). Reversible.",
            args_model=_RenameDocumentArgs,
            handler=rename_document,
            mutating=True,
            eager=False,
            catalog_summary="rename a document's display name",
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="set_canvas_size",
            label_live="Resizing canvas",
            label_done="Resized canvas",
            description=(
                "Set a document's native canvas resolution (its render size, shown as `canvas WxH` in "
                "the working-set header). Clamped to 16-4096. Use it when the user wants a specific "
                "resolution or the render looks low-res."
            ),
            args_model=_SetCanvasSizeArgs,
            handler=set_canvas_size,
            mutating=True,
            eager=False,
            catalog_summary="set a document's render resolution (canvas WxH)",
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="duplicate_document",
            label_live="Duplicating document",
            label_done="Duplicated document",
            description=(
                "Fork a document into a new one (copies its shader, script, and bound media) so the "
                "user can try a variant without losing the original."
            ),
            args_model=_DuplicateDocumentArgs,
            handler=duplicate_document,
            mutating=True,
            eager=False,
            catalog_summary="fork a document into a variant (copies shader/script/media)",
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="import_document",
            label_live="Importing shader",
            label_done="Imported shader",
            description=(
                "Import a .glsl/.frag shader file from the user's disk into the project as a new "
                "document. Opens the USER's native file picker (you never type a path — they choose the "
                "file). A broken import still creates the document and returns its compile errors to fix."
            ),
            args_model=_ImportDocumentArgs,
            handler=import_document,
            mutating=True,
            eager=False,
            catalog_summary="import a .glsl shader file from the user's disk as a new document",
            gate_policy=GatePolicy.NONE,
        ),
    ]
