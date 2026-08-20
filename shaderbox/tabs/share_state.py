from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shaderbox.exporters.base import ExportProgress, RenderedArtifact
from shaderbox.media import Image, MediaWithTexture, Video


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
