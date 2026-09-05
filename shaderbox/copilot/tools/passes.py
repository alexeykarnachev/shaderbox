from typing import Any

from pydantic import Field

from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.tools.base import GatePolicy, ToolArgs, ToolDefinition
from shaderbox.pass_graph import MAX_ITERATIONS

# The pass list's verbs for the copilot (feature 076): add / configure / delete a pass of a
# multi-pass document. All lazy (the long tail beside the document-file ops), handle-addressed by
# document id; delete is gated. A pass's SOURCE is edited through the ordinary edit tools at its
# `<id>#<name>` address — these tools only shape the graph around it.

_NODE_DESC = "document id (from the project map); empty = the current document"
_NAME_DESC = "pass name: starts with a letter, letters/digits/underscores"
_RUNS_DESC = (
    f"how many times the pass draws per frame (1-{MAX_ITERATIONS}); the shader reads "
    "`uniform float u_pass_iteration;` / `uniform float u_pass_iterations;` (float, never int), "
    "and `u_prev` reads the previous run"
)
_DTYPE_DESC = "target texel format: f1 (8-bit), f2 (half float), f4 (float)"
_SCALE_DESC = "target size as a fraction of the canvas (0 < scale <= 1)"
_FILTER_DESC = "linear (true) or nearest (false) sampling of this pass's target"
_WRAP_DESC = "repeat (true) or clamp (false) at the target's edges"
_OUTPUT_DESC = "make this pass the document's output (what the viewer and exports show)"


class _AddPassArgs(ToolArgs):
    name: str = Field(description=_NAME_DESC)
    document: str = Field(default="", description=_NODE_DESC)
    runs: int | None = Field(default=None, description=_RUNS_DESC)
    dtype: str | None = Field(default=None, description=_DTYPE_DESC)
    scale: float | None = Field(default=None, description=_SCALE_DESC)
    filter_linear: bool | None = Field(default=None, description=_FILTER_DESC)
    wrap: bool | None = Field(default=None, description=_WRAP_DESC)
    output: bool = Field(default=False, description=_OUTPUT_DESC)


class _SetPassArgs(ToolArgs):
    name: str = Field(description="the pass to configure")
    document: str = Field(default="", description=_NODE_DESC)
    runs: int | None = Field(default=None, description=_RUNS_DESC)
    dtype: str | None = Field(default=None, description=_DTYPE_DESC)
    scale: float | None = Field(default=None, description=_SCALE_DESC)
    filter_linear: bool | None = Field(default=None, description=_FILTER_DESC)
    wrap: bool | None = Field(default=None, description=_WRAP_DESC)
    output: bool = Field(default=False, description=_OUTPUT_DESC)
    new_name: str = Field(default="", description="rename the pass (empty = keep)")


class _DeletePassArgs(ToolArgs):
    name: str = Field(description="the pass to delete")
    document: str = Field(default="", description=_NODE_DESC)


def pass_tools(caps: CopilotCapabilities) -> list[ToolDefinition]:
    def add_pass(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        res = caps.add_pass(
            args["document"],
            args["name"],
            args["runs"],
            args["dtype"],
            args["scale"],
            args["filter_linear"],
            args["wrap"],
            args["output"],
        )
        if not res.ok:
            return False, f"error: {res.error}", None
        return (
            True,
            f"added pass '{args['name']}' from a black stub — write_shader its source next.\n{res.table}",
            None,
        )

    def set_pass(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        res = caps.set_pass(
            args["document"],
            args["name"],
            args["runs"],
            args["dtype"],
            args["scale"],
            args["filter_linear"],
            args["wrap"],
            args["output"],
            args["new_name"],
        )
        if not res.ok:
            return False, f"error: {res.error}", None
        return True, f"configured pass '{args['name']}'.\n{res.table}", None

    def delete_pass(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        res = caps.delete_pass(args["document"], args["name"])
        if not res.ok:
            return False, f"error: {res.error}", None
        return True, f"deleted pass '{args['name']}'.\n{res.table}", None

    return [
        ToolDefinition(
            name="add_pass",
            label_live="Adding pass",
            label_done="Added pass",
            description=(
                "Add a pass to a document: a new shader with its own main() and render target, "
                "created as a black stub at `passes/<name>.glsl`. Then write_shader / edit_shader it "
                "at the `<id>#<name>` address. Configure it here in the same call (runs per frame, "
                "target format, output) or later with set_pass. A sampler named `u_<pass>` in "
                "another pass reads this pass's output; `u_prev` reads its own previous run/frame."
            ),
            args_model=_AddPassArgs,
            handler=add_pass,
            mutating=True,
            eager=True,
            catalog_summary="add a pass (its own shader + target) to a document",
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="set_pass",
            label_live="Configuring pass",
            label_done="Configured pass",
            description=(
                "Configure an existing pass: runs per frame (an iterated pass — a jump flood, a "
                "cascade stack), the target's dtype/scale/filter/wrap, make it the output, or "
                "rename it (every sampler naming it follows). Only the given fields change."
            ),
            args_model=_SetPassArgs,
            handler=set_pass,
            mutating=True,
            eager=True,
            catalog_summary="set a pass's runs / target / output, or rename it",
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="delete_pass",
            label_live="Deleting pass",
            label_done="Deleted pass",
            description=(
                "Delete a pass and its shader file; samplers that read it go back to unfilled "
                "(reading black). A document keeps at least one pass."
            ),
            args_model=_DeletePassArgs,
            handler=delete_pass,
            mutating=True,
            eager=False,
            catalog_summary="delete a pass of a document",
            gate_policy=GatePolicy.ALWAYS,
            gate_prompt=lambda a: f"Delete pass `{a.get('name', '')}`?",
        ),
    ]
