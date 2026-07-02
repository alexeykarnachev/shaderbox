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
    "A read-only look at a shader's frame at a chosen time `t`: returns the measured facts line "
    "(ink %, bbox, mean colour, luma, or FLAT) AND a VISION read of the frame — a real inspection of "
    "its CORRECTNESS (coherent structure vs noise/speckle, orientation/mirroring, content off-frame, "
    "text legibility, obvious artifacts), NOT beauty (that stays the user's eye). This is your only "
    "actual SIGHT of your output: use it to check an animated shader past t=0, re-look after a "
    "set_uniform, and ESPECIALLY before you claim a visual result or report a visual task done. "
    "Unlike render_image (a heavy, gated, file-writing deliverable) it never confirms or writes a "
    "file. It does cost a little (a vision call) — glance when you need to SEE, not every step."
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
