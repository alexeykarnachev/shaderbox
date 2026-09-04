"""The pass-settings modal's create mode (078 D5): a draft that becomes a pass only on
`Create`; closing the modal with a draft open makes nothing."""

from typing import Any

from shaderbox.app import PopupState
from shaderbox.pass_graph import PassEntry


def test_opening_add_pass_creates_nothing(app: Any) -> None:
    document = app.ui_documents[app.current_document_id].document
    before = set(document.passes)
    app.open_add_pass()
    assert app.pass_draft is not None
    assert app.popup_state == PopupState.PASS_SETTINGS
    assert set(document.passes) == before
    app.close_pass_settings()
    assert app.pass_draft is None
    assert set(document.passes) == before


def test_create_refuses_an_empty_or_taken_name_and_keeps_the_draft(app: Any) -> None:
    document = app.ui_documents[app.current_document_id].document
    before = set(document.passes)
    app.open_add_pass()
    assert app.create_pass_from_draft() is False
    app.pass_draft.name_buf = sorted(before)[0]
    assert app.create_pass_from_draft() is False
    assert app.pass_draft is not None
    assert set(document.passes) == before


def test_create_lands_the_draft_name_target_and_runs(app: Any) -> None:
    document = app.ui_documents[app.current_document_id].document
    app.open_add_pass()
    draft = app.pass_draft
    draft.name_buf = "  glow "
    draft.entry = PassEntry(
        target=PassEntry().target.model_copy(update={"scale": 0.5}), iterations=3
    )
    assert app.create_pass_from_draft() is True
    assert "glow" in document.passes
    entry = document.graph.passes["glow"]
    assert entry.iterations == 3
    assert entry.target.scale == 0.5
    assert app.pass_draft is None
    # The new pass is what the document shows.
    assert app.panel_pass(app.current_document_id) is document.passes["glow"]
