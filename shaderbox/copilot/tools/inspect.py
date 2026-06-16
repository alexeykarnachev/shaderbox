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
    "A FREE read-only look at a shader's frame at a chosen time `t`: returns one measured facts "
    "line (ink %, bbox, luma, or FLAT), the same an edit returns but on demand and at ANY t. Use "
    "it to check an animated shader past t=0, or to re-look after a set_uniform, without editing. "
    "Unlike render_image (a heavy, gated, file-writing deliverable) it never confirms or renders a "
    "file - glance as often as you need."
)


def inspect_tools(caps: CopilotCapabilities) -> list[ToolDefinition]:
    def probe_render(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        facts = caps.probe_render(args["node"], args["t"])
        ok = not facts.startswith("error:")
        return ok, facts, None

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
