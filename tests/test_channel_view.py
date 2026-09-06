"""The viewer's channel view (073 W-C): Color / Alpha / RGB, default unchanged.

The Alpha and RGB views are separate blits, so the output texture that feedback reads and
exports sample is never touched; the view cycles through one command and persists with the
app state.
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
    assert (
        UIAppState.load(tmp_path / "app_state.json").channel_view == ChannelView.ALPHA
    )


def test_the_command_cycles_the_view(app: Any) -> None:
    assert app.app_state.channel_view == ChannelView.COLOR
    app.command_callbacks[CommandId.CYCLE_CHANNEL_VIEW]()
    assert app.app_state.channel_view == ChannelView.ALPHA


def test_the_alpha_view_is_the_alpha_channel_as_grayscale(app: Any) -> None:
    # A 2x1 texture: left texel dim red at full alpha, right texel bright green at none.
    # The view must show the ALPHA (white / black), never the color, and leave the source
    # untouched. Every channel differs from alpha in at least one texel, so a blit reading
    # the wrong one cannot pass: red 64/0 and alpha 255/0 were both 255/0 under the earlier
    # fixture, which let `.r` stand in for `.a` unnoticed.
    source = app.alpha_view._gl.texture(
        (2, 1), 4, data=bytes([64, 0, 0, 255, 0, 255, 128, 0]), dtype="f1"
    )
    shown = app.alpha_view.render(source)
    assert shown is not source
    pixels = np.frombuffer(shown.read(), dtype=np.uint8).reshape(1, 2, 4)
    assert pixels[0, 0].tolist() == [255, 255, 255, 255]
    assert pixels[0, 1].tolist() == [0, 0, 0, 255]
    assert np.frombuffer(source.read(), dtype=np.uint8).tolist() == [
        64,
        0,
        0,
        255,
        0,
        255,
        128,
        0,
    ]
    source.release()


def test_the_rgb_view_ignores_alpha_entirely(app: Any) -> None:
    # The maintainer's report: a shader writing `vec4(background, 0.0)` showed only the
    # checker in every view, because the compositing ones are honouring an alpha of 0. The
    # RGB view answers what color is THERE. Same 2x1 source as the alpha test: left opaque
    # red, right TRANSPARENT green -- the green must come through at full strength, and the
    # frame must be opaque so no checker shows under it.
    source = app.rgb_view._gl.texture(
        (2, 1), 4, data=bytes([255, 0, 0, 255, 0, 255, 0, 0]), dtype="f1"
    )
    shown = app.rgb_view.render(source)
    assert shown is not source
    pixels = np.frombuffer(shown.read(), dtype=np.uint8).reshape(1, 2, 4)
    assert pixels[0, 0].tolist() == [255, 0, 0, 255]
    assert pixels[0, 1].tolist() == [0, 255, 0, 255], (
        "a transparent texel keeps its color"
    )
    source.release()


def test_the_alpha_and_rgb_views_are_separate_blits(app: Any) -> None:
    # One class, two shaders (each its own GL program): a shared canvas would make the two
    # views overwrite each other's texture the frame both were asked for.
    assert app.alpha_view is not app.rgb_view
    assert app.alpha_view.canvas is not app.rgb_view.canvas
    assert app.alpha_view.program is not app.rgb_view.program
