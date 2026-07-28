from typing import Any

from pydantic import Field

from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.tools.base import GatePolicy, ToolArgs, ToolDefinition

# Read-side inspection tools: the agent looks WITHOUT mutating or producing a deliverable. The
# read counterpart to the gated render/publish tools (feature 050).


class _ProbeRenderArgs(ToolArgs):
    node: str = Field(
        default="",
        description="node id (from the project map); empty = the node you're working on",
    )
    t: float = Field(
        default=0.0,
        description="the animation time (seconds) to render at; 0.0 = the export clock the "
        "user renders to a file. Aim it at a specific moment to inspect an animated shader "
        "past t=0 (e.g. t=2.5 to see the flame mid-rise).",
    )


_PROBE_RENDER_DESC = (
    "A read-only MEASUREMENT of a shader's frame at a chosen time `t`: the facts line (ink %, "
    "bbox, ink mean colour, luma rows, or FLAT — one uniform colour). Use it to check an animated "
    "shader past t=0, to re-measure after a set_uniform, and before you state what the frame "
    "contains. Unlike render_image (a heavy, gated, file-writing deliverable) it never confirms or "
    "writes a file, and it is free. It measures — it does not SEE: the numbers are all you get, "
    "and how the result LOOKS stays the user's judgment."
)


def inspect_tools(caps: CopilotCapabilities) -> list[ToolDefinition]:
    def probe_render(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        msg = caps.probe_render(args["node"], args["t"])
        return not msg.startswith("error:"), msg, None

    return [
        ToolDefinition(
            name="probe_render",
            label_live="Probing render",
            label_done="Probed render",
            description=_PROBE_RENDER_DESC,
            args_model=_ProbeRenderArgs,
            handler=probe_render,
            mutating=False,
            eager=True,
            gate_policy=GatePolicy.NONE,
        ),
    ]
