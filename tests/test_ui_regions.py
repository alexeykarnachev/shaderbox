"""The Document tab's passes view (092 D2): strip or graph, default unchanged, persisted with
the app state. The `ChannelView` trio's shape."""

from pathlib import Path

from shaderbox.ui_models import UIAppState
from shaderbox.ui_regions import PASSES_VIEW_LABELS, PassesView


def test_every_passes_view_has_a_label() -> None:
    assert set(PASSES_VIEW_LABELS) == set(PassesView)


def test_every_passes_view_label_is_within_the_control_budget() -> None:
    for label in PASSES_VIEW_LABELS.values():
        assert len(label.split()) <= 2, label


def test_the_default_is_the_strip_and_the_choice_persists(tmp_path: Path) -> None:
    state = UIAppState()
    assert state.passes_view == PassesView.STRIP
    state.passes_view = PassesView.GRAPH
    state.save(tmp_path / "app_state.json")
    assert UIAppState.load(tmp_path / "app_state.json").passes_view == PassesView.GRAPH
