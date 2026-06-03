from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

from shaderbox.constants import DEFAULT_FPS
from shaderbox.core import Node
from shaderbox.exporters.base import ExportProgress, RenderedArtifact
from shaderbox.media import Image, MediaDetails, MediaWithTexture, Video
from shaderbox.render_preset import RenderPreset


@dataclass
class OutletRenderState:
    """Per-outlet render config + its last rendered artifact.

    Each outlet renders with its own preset-bounded params, so the artifact
    can't be shared across outlets (different size/duration caps).
    """

    duration: float = 3.0
    current_artifact: RenderedArtifact | None = None
    artifact_is_fresh: bool = False
    preview: MediaWithTexture | None = None
    notified_progress: ExportProgress | None = None  # last terminal event surfaced
    # Free-form per-exporter UI scratch (e.g. Telegram's pending new-sticker
    # emoji). The outlet stays exporter-agnostic; only the owning exporter reads it.
    extra_state: dict[str, Any] = field(default_factory=dict)

    def set_artifact(self, artifact: RenderedArtifact | None) -> None:
        self._release_preview()
        self.current_artifact = artifact
        self.artifact_is_fresh = artifact is not None

    def preview_media(self) -> MediaWithTexture | None:
        art: RenderedArtifact | None = self.current_artifact
        if art is None or not art.path.exists():
            return None
        if self.preview is None:
            self.preview = Video(art.path) if art.is_video else Image(art.path)
        return self.preview

    def _release_preview(self) -> None:
        if self.preview is not None:
            self.preview.release()
            self.preview = None


@dataclass
class TabState:
    scratch_dir: Path
    outlets: dict[str, OutletRenderState] = field(default_factory=dict)

    def outlet(self, exporter_id: str) -> OutletRenderState:
        if exporter_id not in self.outlets:
            self.outlets[exporter_id] = OutletRenderState()
        return self.outlets[exporter_id]

    def release(self) -> None:
        for outlet in self.outlets.values():
            outlet.set_artifact(None)
        self.outlets.clear()


def make_state(scratch_dir: Path) -> TabState:
    return TabState(scratch_dir=scratch_dir)


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
