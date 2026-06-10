from typing import Any

from pydantic import Field

from shaderbox.copilot.capabilities import (
    CopilotCapabilities,
    PublishResult,
    RenderResult,
)
from shaderbox.copilot.tools.base import GatePolicy, ToolArgs, ToolDefinition

# Render tools marshal GL via the bridge; all are GatePolicy.ALWAYS (render freezes the UI,
# publish is external + irreversible). Publish tools precheck (handoff) before the gate.

# Dimensions snap to the codec alignment, so the result reports the ACTUAL rendered size.
_DIM_DESC = "pixels; omit (0) to use the node's current canvas size. Snaps up to the codec alignment."


class _RenderImageArgs(ToolArgs):
    node: str = Field(
        default="",
        description="node id (from the project map); empty = the shader you're working on",
    )
    width: int = Field(default=0, ge=0, le=4096, description=_DIM_DESC)
    height: int = Field(default=0, ge=0, le=4096, description=_DIM_DESC)


class _RenderVideoArgs(ToolArgs):
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


class _PublishTelegramArgs(ToolArgs):
    emoji: str = Field(
        default="🎨",
        description="the emoji to associate with the sticker (one character)",
    )


class _PublishYoutubeArgs(ToolArgs):
    title: str = Field(description="the video title")
    description: str = Field(default="", description="the video description")
    is_short: bool = Field(
        default=False,
        description="True = a YouTube Short (9:16, <=60s); False = a normal video",
    )


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


# The URL/path goes in the widget, never embedded in the result string (the model-facing message).
# The "a button is shown" line is conditional on a non-empty target, so no dead-button promise.


def _render_result(r: RenderResult, kind: str) -> tuple[bool, str, dict | None]:
    if not r.ok:
        return False, f"error: {r.error}", None
    what = (
        f"{r.width}x{r.height} {r.duration:.1f}s video"
        if r.is_video
        else f"{r.width}x{r.height} {kind}"
    )
    if not r.path:
        return True, f"rendered a {what} (the file path wasn't captured).", None
    msg = (
        f"rendered a {what}. A 'Reveal render' button is shown to the user — point them to it "
        "in words; you can't see the result and don't have the path."
    )
    widget = {"kind": "open_path", "label": "Reveal render", "target": r.path}
    return True, msg, {"path": r.path, "widget": widget}


def _publish_result(r: PublishResult, target: str) -> tuple[bool, str, dict | None]:
    if not r.ok:
        return False, f"error: {r.error}", None
    if not r.url:
        return (
            True,
            f"published to {target} (the link wasn't captured — tell the user to check {target}).",
            None,
        )
    msg = (
        f"published to {target}. An 'Open in {target}' button is shown to the user — point them "
        "to it in words; you don't have the link."
    )
    widget = {"kind": "open_url", "label": f"Open in {target}", "target": r.url}
    return True, msg, {"url": r.url, "widget": widget}


def publish_tools(caps: CopilotCapabilities) -> list[ToolDefinition]:
    def render_image(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        r = caps.render_image(args["node"], args["width"], args["height"])
        return _render_result(r, "image")

    def render_video(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        r = caps.render_video(
            args["node"], args["seconds"], args["fps"], args["width"], args["height"]
        )
        return _render_result(r, "video")

    def publish_telegram(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        r = caps.publish_telegram(args["emoji"])
        return _publish_result(r, "Telegram")

    def publish_youtube(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        r = caps.publish_youtube(args["title"], args["description"], args["is_short"])
        return _publish_result(r, "YouTube")

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
                "YouTube isn't connected. YOU connect it: call set_youtube_credentials (it opens "
                "the inline setup panel for the user) — do NOT just send them to Settings."
            )
        return None

    return [
        ToolDefinition(
            name="render_image",
            label_live="Rendering image",
            label_done="Rendered image",
            description=_RENDER_IMAGE_DESC,
            args_model=_RenderImageArgs,
            handler=render_image,
            mutating=True,
            needs_gl=True,
            category="render",
            eager=True,
            gate_policy=GatePolicy.ALWAYS,
            gate_prompt=lambda a: (
                "Render an image of this shader? The app pauses while it encodes."
            ),
        ),
        ToolDefinition(
            name="render_video",
            label_live="Rendering video",
            label_done="Rendered video",
            description=_RENDER_VIDEO_DESC,
            args_model=_RenderVideoArgs,
            handler=render_video,
            mutating=True,
            needs_gl=True,
            category="render",
            eager=True,
            gate_policy=GatePolicy.ALWAYS,
            gate_prompt=lambda a: (
                f"Render a {a.get('seconds', '?')}s video of this shader? "
                "The app pauses while it encodes."
            ),
        ),
        ToolDefinition(
            name="publish_telegram",
            label_live="Publishing to Telegram",
            label_done="Published to Telegram",
            description=_PUBLISH_TELEGRAM_DESC,
            args_model=_PublishTelegramArgs,
            handler=publish_telegram,
            mutating=True,
            needs_gl=False,
            category="publish",
            eager=True,
            gate_policy=GatePolicy.ALWAYS,
            gate_prompt=lambda a: (
                "Publish this shader to your Telegram sticker pack? "
                "This uploads the sticker (external + live)."
            ),
            precheck=telegram_precheck,
        ),
        ToolDefinition(
            name="publish_youtube",
            label_live="Publishing to YouTube",
            label_done="Published to YouTube",
            description=_PUBLISH_YOUTUBE_DESC,
            args_model=_PublishYoutubeArgs,
            handler=publish_youtube,
            mutating=True,
            needs_gl=False,
            category="publish",
            eager=True,
            gate_policy=GatePolicy.ALWAYS,
            gate_prompt=lambda a: (
                f"Publish this shader to YouTube as '{a.get('title', '')}'? "
                "The video goes live on your channel (private; external)."
            ),
            precheck=youtube_precheck,
        ),
    ]
