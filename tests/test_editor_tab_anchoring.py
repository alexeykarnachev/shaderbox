"""Closing a tab keeps the SAME file active.

`active_tab_index` used to be only CLAMPED after a removal, which keeps the index valid but
not the identity: removing a tab to the LEFT shifts every later tab down one, so the index
silently addresses a different file. `flush_current_editor()` then flushes the wrong tab on
Ctrl+S and on quit — the user loses edits to the file they were looking at.

Both removal paths (an explicit close, and a document deletion sweeping its tabs) go through the
same re-anchor helper, so they cannot drift apart.
"""

from pathlib import Path
from typing import Any

import pytest

from shaderbox.editor_types import EditorTab


def _tabs(app: Any) -> list[str]:
    return [t.path.name for t in app.editor_tabs]


def _seed(app: Any, names: list[str]) -> None:
    app.editor_tabs = [EditorTab(path=Path(f"/tmp/{n}"), kind="lib") for n in names]


@pytest.mark.parametrize(
    "closing,active_idx,expected",
    [
        # The clamp only DIFFERS from anchoring when the active tab is not the last one:
        # min(active, len-1) happens to be right for a trailing tab. These three are the
        # cases that actually distinguish the two policies over four tabs.
        (0, 1, "b.glsl"),
        (0, 2, "c.glsl"),
        (1, 2, "c.glsl"),
    ],
)
def test_closing_a_tab_to_the_left_keeps_the_same_file_active(
    app: Any, closing: int, active_idx: int, expected: str
) -> None:
    _seed(app, ["a.glsl", "b.glsl", "c.glsl", "d.glsl"])
    app.active_tab_index = active_idx
    active_before = app.editor_tabs[active_idx]

    app.close_tab(closing)

    assert app.editor_tabs[app.active_tab_index] is active_before
    assert app.editor_tabs[app.active_tab_index].path.name == expected


def test_closing_a_tab_to_the_right_keeps_the_same_file_active(app: Any) -> None:
    _seed(app, ["a.glsl", "b.glsl", "c.glsl"])
    app.active_tab_index = 0
    active_before = app.editor_tabs[0]

    app.close_tab(2)

    assert app.editor_tabs[app.active_tab_index] is active_before


def test_closing_the_active_tab_falls_back_in_range(app: Any) -> None:
    _seed(app, ["a.glsl", "b.glsl", "c.glsl"])
    app.active_tab_index = 2

    app.close_tab(2)

    assert 0 <= app.active_tab_index < len(app.editor_tabs)
    assert _tabs(app) == ["a.glsl", "b.glsl"]


def test_closing_the_last_tab_leaves_a_valid_index(app: Any) -> None:
    _seed(app, ["only.glsl"])
    app.active_tab_index = 0

    app.close_tab(0)

    assert app.editor_tabs == []
    assert app.active_tab_index == -1


def test_deleting_a_document_keeps_an_unrelated_tab_active(app: Any) -> None:
    # The sibling removal path: _on_document_deleted sweeps that document's tabs out of the list.
    # A lib tab the user is editing must stay active, not be swapped by an index shift.
    document_id = app.current_document_id
    document_source = app.ui_documents[document_id].document.render_pass.source.path
    app.editor_tabs = [
        EditorTab(path=document_source, kind="shader", document_id=document_id),
        EditorTab(path=Path("/tmp/keep.glsl"), kind="lib"),
    ]
    app.active_tab_index = 1
    active_before = app.editor_tabs[1]

    app._on_document_deleted(document_id, document_source)

    assert _tabs(app) == ["keep.glsl"]
    assert app.editor_tabs[app.active_tab_index] is active_before
