"""A failed re-render must not leave the previous artifact looking publishable.

`artifact_is_fresh` is set only by `set_artifact`, so `_render`'s failure path clears it
explicitly (the stale-artifact guard in `tabs/share.py`) — without that line the publish
buttons stay armed on the OLD artifact after a failed re-render, and the user publishes the
previous render believing it is the new one. This drives the real path: `render_for` fails
and the guard must fire. Falsifier: delete the guard line and this goes red.
"""

import types
from pathlib import Path
from typing import Any, cast

from shaderbox.app import App
from shaderbox.exporters.base import RenderedArtifact
from shaderbox.render_preset import RenderPreset
from shaderbox.tabs import share
from shaderbox.tabs.share_state import OutletRenderState, TabState
from shaderbox.ui_models import UIDocument


def test_a_failed_render_clears_freshness_and_reports(
    monkeypatch: Any, tmp_path: Path
) -> None:
    outlet = OutletRenderState()
    outlet.set_artifact(
        RenderedArtifact(
            path=tmp_path / "a.mp4", is_video=True, duration=1.0, size=(64, 64)
        )
    )
    assert outlet.artifact_is_fresh

    monkeypatch.setattr(share, "render_for", lambda *args, **kwargs: None)
    pushed: list[str] = []
    app_stub = types.SimpleNamespace(
        notifications=types.SimpleNamespace(
            push=lambda message, color: pushed.append(message)
        )
    )

    share._render(
        cast(App, app_stub),
        outlet,
        cast(RenderPreset, object()),
        cast(UIDocument, types.SimpleNamespace(document=None)),
        TabState(scratch_dir=tmp_path),
    )

    assert not outlet.artifact_is_fresh
    assert pushed and "Render failed" in pushed[0]
