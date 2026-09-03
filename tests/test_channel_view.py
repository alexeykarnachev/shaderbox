"""The viewer's channel view (073 W-C): Color / Color+Alpha / Alpha, default unchanged.

The Alpha view is a separate blit, so the output texture that feedback reads and exports
sample is never touched; the view cycles through one command and persists with the app state.
"""

from pathlib import Path
from typing import Any

import numpy as np

from shaderbox.commands import CommandId
from shaderbox.ui_models import UIAppState
from shaderbox.ui_regions import (
    CHANNEL_VIEW_LABELS,
    ChannelView,
    next_channel_view,
)


def test_the_cycle_visits_every_view_and_wraps() -> None:
    seen = [ChannelView.COLOR]
    for _ in range(len(ChannelView)):
        seen.append(next_channel_view(seen[-1]))
    assert seen[:-1] == list(ChannelView)
    assert seen[-1] == ChannelView.COLOR


def test_every_view_has_a_label() -> None:
    assert set(CHANNEL_VIEW_LABELS) == set(ChannelView)


def test_every_label_is_within_the_control_budget() -> None:
    # The chip is a control label: one or two words (imgui-ui § 2).
    for label in CHANNEL_VIEW_LABELS.values():
        assert len(label.replace("+", " ").split()) <= 2, label


def test_the_default_is_color_and_the_choice_persists(tmp_path: Path) -> None:
    state = UIAppState()
    assert state.channel_view == ChannelView.COLOR
    state.channel_view = ChannelView.ALPHA
    state.save(tmp_path / "app_state.json")
    assert UIAppState.load(tmp_path / "app_state.json").channel_view == ChannelView.ALPHA


def test_the_command_cycles_the_view(app: Any) -> None:
    assert app.app_state.channel_view == ChannelView.COLOR
    app.command_callbacks[CommandId.CYCLE_CHANNEL_VIEW]()
    assert app.app_state.channel_view == ChannelView.COLOR_ALPHA


def test_the_alpha_view_is_the_alpha_channel_as_grayscale(app: Any) -> None:
    # A 2x1 texture: left texel opaque red, right texel transparent green. The view must
    # show the ALPHA (white / black), never the color, and leave the source untouched.
    source = app.alpha_view._gl.texture(
        (2, 1), 4, data=bytes([255, 0, 0, 255, 0, 255, 0, 0]), dtype="f1"
    )
    shown = app.alpha_view.render(source)
    assert shown is not source
    pixels = np.frombuffer(shown.read(), dtype=np.uint8).reshape(1, 2, 4)
    assert pixels[0, 0].tolist() == [255, 255, 255, 255]
    assert pixels[0, 1].tolist() == [0, 0, 0, 255]
    assert np.frombuffer(source.read(), dtype=np.uint8).tolist() == [
        255, 0, 0, 255, 0, 255, 0, 0,
    ]
    source.release()
