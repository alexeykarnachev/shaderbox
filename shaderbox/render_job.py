"""Node render-to-file: the GL render job behind BOTH the Share tab and the copilot's render
tools. UI-free on purpose — it lives here, not in `tabs/`, because the headless copilot core
(`copilot/backend.py`) and the dogfood harness drive it with no imgui in the process.

`render_to` owns the render try/except + partial-file cleanup + artifact construction; the caller
mints the path (`render_for` a scratch uuid, the copilot a renders-dir name), so this stays
path-agnostic."""

from pathlib import Path
from uuid import uuid4

from loguru import logger

from shaderbox.constants import DEFAULT_FPS
from shaderbox.document import Node
from shaderbox.exporters.base import RenderedArtifact
from shaderbox.media import MediaDetails
from shaderbox.render_preset import RenderPreset


def preset_ext(preset: RenderPreset) -> str:
    is_video: bool = preset.is_video if preset.is_video is not None else True
    return (preset.container or ".webm").lstrip(".") if is_video else "png"


def render_to(
    node: Node, preset: RenderPreset, duration: float, out_path: Path
) -> RenderedArtifact | None:
    """Render the node into `out_path` bounded by the outlet preset.

    Owns the render try/except + partial-file cleanup + artifact value construction.
    The caller mints the path (`render_for` a scratch uuid; the copilot a renders-dir
    name) so this stays path-agnostic.
    """
    is_video: bool = preset.is_video if preset.is_video is not None else True

    capped_duration: float = duration
    if preset.duration_max is not None:
        capped_duration = min(capped_duration, preset.duration_max)

    details = MediaDetails(
        is_video=is_video,
        fps=preset.fps if preset.fps is not None else DEFAULT_FPS,
        duration=capped_duration,
    )
    details.file_details.path = str(out_path)

    try:
        rendered: MediaDetails = node.render_media(details, preset)
    except Exception as e:
        logger.error(f"Failed to render artifact: {e}")
        if out_path.exists():
            try:
                out_path.unlink()
            except OSError as cleanup_err:
                logger.warning(f"Failed to cleanup partial render: {cleanup_err}")
        return None

    return RenderedArtifact(
        path=out_path,
        is_video=is_video,
        duration=rendered.duration,
        size=(rendered.resolution_details.width, rendered.resolution_details.height),
    )


def render_for(
    node: Node, preset: RenderPreset, duration: float, scratch_dir: Path
) -> RenderedArtifact | None:
    """Render the node into a scratch artifact bounded by the outlet preset.

    Mints the scratch path, then delegates the render to `render_to`.
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    artifact_path: Path = scratch_dir / f"{uuid4()}.{preset_ext(preset)}"
    return render_to(node, preset, duration, artifact_path)
