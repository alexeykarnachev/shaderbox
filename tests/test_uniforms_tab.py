"""Feature 083: the Uniforms tab and its pass selector.

The panel used to reach only a pass that was open in the editor or on screen; tuning a third one
meant opening its tab first. An explicit pick now wins over that derivation, and a stale pick
falls back to it rather than erroring.
"""

import json
from pathlib import Path
from typing import Any

from shaderbox.model_salvage import load_model
from shaderbox.ui_models import UIAppState, UIDocumentState
from shaderbox.ui_regions import DocumentTab


def test_the_tab_choice_round_trips_and_a_bad_one_costs_only_itself(
    tmp_path: Path,
) -> None:
    # The persisted enum gains a member. Round-trip first, then hand the loader a value no member
    # claims: per-key salvage must drop THAT key and keep the rest of the file.
    state = UIAppState(active_document_tab=DocumentTab.UNIFORMS)
    assert (
        UIAppState.model_validate(state.model_dump()).active_document_tab
        is DocumentTab.UNIFORMS
    )

    path = tmp_path / "app_state.json"
    path.write_text(
        json.dumps({"active_document_tab": "not_a_tab", "show_cheatsheet": False})
    )
    salvaged = load_model(UIAppState, path, "app_state")
    assert salvaged.active_document_tab is DocumentTab.DOCUMENT
    assert salvaged.show_cheatsheet is False, "one bad key must not cost a sibling"


def _two_pass_document(app: Any) -> str:
    # The fixture seeds the single-pass starter; the selector only means anything with two.
    document_id = app.current_document_id
    assert not app.session.add_pass(document_id, "second")
    return document_id


def test_an_explicit_pass_pick_wins_over_the_active_tab(app: Any) -> None:
    # The whole point of W-D: reach a pass that is neither the output nor open in the editor.
    document_id = _two_pass_document(app)
    document = app.ui_documents[document_id].document
    derived = app.panel_pass(document_id)
    other = next(p for n, p in document.passes.items() if p is not derived)
    other_name = next(n for n, p in document.passes.items() if p is other)

    app.set_panel_pass(document_id, other_name)
    assert app.panel_pass(document_id) is other

    # A name no pass carries (a rename, a deleted pass) falls through to the derived answer.
    app.set_panel_pass(document_id, "gone")
    assert app.panel_pass(document_id) is derived


def test_opening_a_shader_tab_retires_the_pick(app: Any) -> None:
    # The common path — click a tile, edit its uniforms — must keep working without the user
    # knowing the override exists, so opening a pass in the editor clears it.
    document_id = _two_pass_document(app)
    document = app.ui_documents[document_id].document
    name = sorted(document.passes)[-1]
    app.set_panel_pass(document_id, name)
    assert app.ui_documents[document_id].ui_state.panel_pass == name

    app.ensure_shader_tab(document_id, name)
    assert app.ui_documents[document_id].ui_state.panel_pass == ""


def test_the_write_seam_ignores_an_unknown_document(app: Any) -> None:
    # set_panel_pass resolves the document's OWN ui_state; a write for a document that is not
    # there is a no-op rather than a write into a throwaway that is discarded in silence.
    app.set_panel_pass("no-such-document", "whatever")
    assert UIDocumentState().panel_pass == ""
