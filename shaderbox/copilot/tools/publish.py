from typing import Any

from pydantic import BaseModel, Field

from shaderbox.copilot.capabilities import (
    CopilotCapabilities,
    PublishResult,
    RenderResult,
)
from shaderbox.copilot.tools.base import GatePolicy, ToolDefinition

# The render + publish tool surface (feature 020·18). render_image/render_video write a file
# under the project renders dir (GL => the capability marshals via the bridge with the longer
# render_op_timeout_s). publish_telegram/publish_youtube render-then-upload, returning the
# pack/Studio URL. ALL are GatePolicy.ALWAYS (a render freezes the UI + costs a confirm; a
# publish is external + irreversible). The publish tools carry a `precheck` that returns a
# guided handoff (connect in Settings / pick a pack) BEFORE the gate when they can't run.

# Dimensions snap to the codec alignment, so the result reports the ACTUAL rendered size.
_DIM_DESC = "pixels; omit (0) to use the node's current canvas size. Snaps up to the codec alignment."


class _RenderImageArgs(BaseModel):
    node: str = Field(
        default="",
        description="node id (from the project map); empty = the shader you're working on",
    )
    width: int = Field(default=0, ge=0, le=4096, description=_DIM_DESC)
    height: int = Field(default=0, ge=0, le=4096, description=_DIM_DESC)
    model_config = {"extra": "forbid"}


class _RenderVideoArgs(BaseModel):
    node: str = Field(
        default="",
        description="node id (from the project map); empty = the shader you're working on",
    )
    seconds: float = Field(
        gt=0.0,
        le=60.0,
        description="how many seconds of the animation to render, from t=0 (required)",
    )
    fps: int = Field(default=30, ge=1, le=120, description="frames per second")
    width: int = Field(default=0, ge=0, le=4096, description=_DIM_DESC)
    height: int = Field(default=0, ge=0, le=4096, description=_DIM_DESC)
    model_config = {"extra": "forbid"}


class _PublishTelegramArgs(BaseModel):
    emoji: str = Field(
        default="🎨",
        description="the emoji to associate with the sticker (one character)",
    )
    model_config = {"extra": "forbid"}


class _PublishYoutubeArgs(BaseModel):
    title: str = Field(description="the video title")
    description: str = Field(default="", description="the video description")
    is_short: bool = Field(
        default=False,
        description="True = a YouTube Short (9:16, <=60s); False = a normal video",
    )
    model_config = {"extra": "forbid"}


_RENDER_IMAGE_DESC = (
    "Render the node's CURRENT frame to an image file (PNG) under the project's renders "
    "folder. The app pauses briefly while it encodes, and the user confirms first. Returns the "
    "file path + the actual size. You render the live source — land your edits before rendering."
)
_RENDER_VIDEO_DESC = (
    "Render `seconds` of the node's animation (always from t=0) to a video file (WebM) under "
    "the project's renders folder. The app pauses while it encodes; the user confirms first. "
    "Returns the path + duration. You cannot pick a start time other than 0."
)
_PUBLISH_TELEGRAM_DESC = (
    "Render the current shader as a Telegram sticker (3s WebM) and add it to the user's "
    "selected sticker pack. EXTERNAL + LIVE — the user confirms first. Needs Telegram connected "
    "+ a pack selected — both of which YOU do (set_telegram_token, create/select_telegram_pack), "
    "not the user in Settings. Returns the pack URL."
)
_PUBLISH_YOUTUBE_DESC = (
    "Render the current shader as a video and upload it to the user's YouTube channel (as a "
    "private video they publish from Studio). EXTERNAL + LIVE — the user confirms first. "
    "YouTube must be connected in Settings. Returns the Studio URL."
)


def _render_image_result_msg(r: RenderResult, kind: str) -> str:
    if not r.ok:
        return f"error: {r.error}"
    if r.is_video:
        return (
            f"rendered a {r.width}x{r.height} {r.duration:.1f}s video -> {r.path}. "
            "Tell the user where it is — you can't see it."
        )
    return (
        f"rendered a {r.width}x{r.height} {kind} -> {r.path}. "
        "Tell the user where it is — you can't see it."
    )


def _publish_result_msg(r: PublishResult, target: str) -> str:
    if not r.ok:
        return f"error: {r.error}"
    where = f" ({r.url})" if r.url else ""
    return f"published to {target}{where}. Give the user the link."


def publish_tools(caps: CopilotCapabilities) -> list[ToolDefinition]:
    def render_image(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        r = caps.render_image(args["node"], args["width"], args["height"])
        return (
            r.ok,
            _render_image_result_msg(r, "image"),
            {"path": r.path} if r.ok else None,
        )

    def render_video(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        r = caps.render_video(
            args["node"], args["seconds"], args["fps"], args["width"], args["height"]
        )
        return (
            r.ok,
            _render_image_result_msg(r, "video"),
            {"path": r.path} if r.ok else None,
        )

    def publish_telegram(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        r = caps.publish_telegram("", args["emoji"])
        return (
            r.ok,
            _publish_result_msg(r, "Telegram"),
            {"url": r.url} if r.ok else None,
        )

    def publish_youtube(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        r = caps.publish_youtube(
            "", args["title"], args["description"], args["is_short"]
        )
        return r.ok, _publish_result_msg(r, "YouTube"), {"url": r.url} if r.ok else None

    def telegram_precheck(args: dict[str, Any]) -> str | None:
        _ = args
        if not caps.has_current_node():
            return "There's no shader open to publish. Switch to the node first (switch_node)."
        if not caps.telegram_connected():
            return (
                "Telegram isn't connected. YOU connect it: call set_telegram_token (it opens "
                "the secure paste field for the user) — do NOT send them to Settings."
            )
        if not caps.telegram_has_default_pack():
            return (
                "No Telegram pack is selected. YOU handle it: list_telegram_packs to see what "
                "exists and select_telegram_pack, or create_telegram_pack to make a new one — "
                "do NOT send the user to the Share tab."
            )
        return None

    def youtube_precheck(args: dict[str, Any]) -> str | None:
        _ = args
        if not caps.has_current_node():
            return "There's no shader open to publish. Tell the user to select a node first."
        if not caps.youtube_connected():
            return (
                "YouTube isn't connected. Tell the user to open Settings -> Integrations -> "
                "YouTube and connect their channel, then ask you again."
            )
        return None

    return [
        ToolDefinition(
            name="render_image",
            description=_RENDER_IMAGE_DESC,
            args_model=_RenderImageArgs,
            handler=render_image,
            mutating=True,
            needs_gl=True,
            category="render",
            eager=True,
            gate_policy=GatePolicy.ALWAYS,
        ),
        ToolDefinition(
            name="render_video",
            description=_RENDER_VIDEO_DESC,
            args_model=_RenderVideoArgs,
            handler=render_video,
            mutating=True,
            needs_gl=True,
            category="render",
            eager=True,
            gate_policy=GatePolicy.ALWAYS,
        ),
        ToolDefinition(
            name="publish_telegram",
            description=_PUBLISH_TELEGRAM_DESC,
            args_model=_PublishTelegramArgs,
            handler=publish_telegram,
            mutating=True,
            needs_gl=False,
            category="publish",
            eager=True,
            gate_policy=GatePolicy.ALWAYS,
            precheck=telegram_precheck,
        ),
        ToolDefinition(
            name="publish_youtube",
            description=_PUBLISH_YOUTUBE_DESC,
            args_model=_PublishYoutubeArgs,
            handler=publish_youtube,
            mutating=True,
            needs_gl=False,
            category="publish",
            eager=True,
            gate_policy=GatePolicy.ALWAYS,
            precheck=youtube_precheck,
        ),
    ]
