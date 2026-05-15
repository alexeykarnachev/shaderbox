from dataclasses import dataclass, field
from pathlib import Path

from shaderbox.exporters.base import RenderedArtifact
from shaderbox.media import MediaDetails


@dataclass
class TabState:
    scratch_dir: Path
    media_details: MediaDetails = field(
        default_factory=lambda: MediaDetails(is_video=True)
    )
    current_artifact: RenderedArtifact | None = None


def make_state(scratch_dir: Path) -> TabState:
    return TabState(scratch_dir=scratch_dir)
