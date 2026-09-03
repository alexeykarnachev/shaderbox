from typing import Any

from pydantic import Field

from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.tools.base import GatePolicy, ToolArgs, ToolDefinition

# Texture/media tools (feature 052 slice 2). bind_media opens the USER's OS file picker (its own FILE
# gate — the model never types a path); unbind_media resets a sampler to the default image.

_UNIFORM_DESC = "the sampler2D uniform's name (e.g. u_tex) — see the working-set row"
_NODE_DESC = (
    "document id (from the project map); empty = the document you're working on"
)


class _BindMediaArgs(ToolArgs):
    uniform: str = Field(description=_UNIFORM_DESC)
    document: str = Field(default="", description=_NODE_DESC)


def media_tools(caps: CopilotCapabilities) -> list[ToolDefinition]:
    def bind_media(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        res = caps.bind_media(args["document"], args["uniform"])
        if res.cancelled:
            return True, "the user dismissed the file picker — nothing was bound.", None
        if not res.ok:
            return False, f"error: {res.error}", None
        kind = "video" if res.is_video else "image"
        return (
            True,
            f"bound {res.basename} -> {args['uniform']} ({res.width}x{res.height}, {kind}).",
            None,
        )

    def unbind_media(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        res = caps.unbind_media(args["document"], args["uniform"])
        if not res.ok:
            return False, f"error: {res.error}", None
        return (
            True,
            f"unbound {args['uniform']}: it is undecided again (its name wires it, or it "
            "reads BLACK).",
            None,
        )

    return [
        ToolDefinition(
            name="bind_media",
            label_live="Binding media",
            label_done="Bound media",
            description=(
                "Bind an image or video to a sampler2D uniform. Opens the USER's native file picker "
                "(you never see or type a path — the user chooses the file). The document must already "
                "declare the sampler; add `uniform sampler2D u_tex;` via edit_shader first if not. "
                "Check the working-set row afterwards to confirm the binding."
            ),
            args_model=_BindMediaArgs,
            handler=bind_media,
            mutating=True,
            eager=False,
            catalog_summary="bind an image/video to a sampler2D uniform (opens the user's file picker)",
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="unbind_media",
            label_live="Unbinding media",
            label_done="Unbound media",
            description=(
                "Remove the media bound to a sampler2D uniform. The sampler is undecided again: "
                "a `u_<pass>` name reads that pass, otherwise its working-set row reads "
                "'(nothing; reads BLACK)'."
            ),
            args_model=_BindMediaArgs,
            handler=unbind_media,
            mutating=True,
            eager=False,
            catalog_summary="reset a sampler2D uniform to no bound media",
            gate_policy=GatePolicy.NONE,
        ),
    ]
