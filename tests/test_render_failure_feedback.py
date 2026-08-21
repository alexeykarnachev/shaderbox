"""A failed render must not leave a stale artifact looking publishable.

`artifact_is_fresh` is set only by `set_artifact`, so an early return on a failed re-render
left the PREVIOUS render's `True` in place — and the publish buttons gate on that flag
(`exporters/youtube.py`). The user re-renders, sees nothing happen, and publishes the old
artifact believing it is the new one.
"""

from pathlib import Path

from shaderbox.exporters.base import RenderedArtifact
from shaderbox.tabs.share_state import OutletRenderState


def test_a_successful_render_marks_the_artifact_fresh(tmp_path: Path) -> None:
    outlet = OutletRenderState()
    artifact = RenderedArtifact(
        path=tmp_path / "a.mp4", is_video=True, duration=1.0, size=(64, 64)
    )

    outlet.set_artifact(artifact)

    assert outlet.artifact_is_fresh


def test_clearing_the_artifact_clears_freshness(tmp_path: Path) -> None:
    # The invariant the fix depends on: freshness tracks the artifact, so a failed render
    # clearing the flag can never read as "ready to publish".
    outlet = OutletRenderState()
    outlet.set_artifact(
        RenderedArtifact(
            path=tmp_path / "a.mp4", is_video=True, duration=1.0, size=(64, 64)
        )
    )
    assert outlet.artifact_is_fresh

    outlet.set_artifact(None)

    assert not outlet.artifact_is_fresh
