"""The Document tab's canvas W x H pair, driven through real imgui frames (069 W-A).

Commit-on-deactivate needs a real focus transition across frames, so these drive
`tabs/document.py::draw` in a headless imgui frame the way `test_lib_files.py` drives the
picker's inline inputs, rather than asserting against the code.

The property under test is the one a post-implementation review found broken: only the field
the user is actually in holds a pending value. The other half mirrors the document every frame,
so a write that lands mid-edit (the copilot, a disk sync) is not reverted by a stale number the
user never touched.
"""

from typing import Any

from imgui_bundle import imgui

from shaderbox.tabs import document as document_tab
from tests.conftest import seed_extra_document


def _run_row_frames(app: Any, external_write: tuple[int, int] | None) -> None:
    """Focus W, type into it, land an external write while it is STILL active, then leave it.

    Focus index 1, not 0: the row's first submitted item is the document-name input, so index 0
    focuses that and the keystroke never reaches the width field.
    """
    # One imgui context serves the whole test session, so a previous test can leave an item
    # focused: clear it, or this row starts with a field already active and its stale half
    # commits on the first frame.
    imgui.set_window_focus(None)
    app.canvas_w_editing = False
    app.canvas_h_editing = False

    for frame in range(9):
        if frame == 3:
            imgui.get_io().add_input_character(ord("8"))
        if frame == 4 and external_write is not None:
            app.ui_documents[app.current_document_id].document.set_canvas_size(
                external_write
            )
        imgui.new_frame()
        imgui.begin("rig")
        # Offsets count focusable items from the cursor: 079 D7 moved the presets combo and
        # Reset onto the caption row ABOVE the inputs, so both shifted by two.
        if frame in (0, 1, 2):
            imgui.set_keyboard_focus_here(3)
        if frame == 5:
            imgui.set_keyboard_focus_here(5)
        document_tab.draw(app)
        imgui.end()
        imgui.end_frame()


def test_a_write_during_an_active_field_survives_the_commit(app: Any) -> None:
    # With W active and H untouched, an external 800x600 must keep its 600: the commit pair is
    # (the pending width, the document's CURRENT height). Falsifier: a buffer whose inactive half
    # is frozen at edit-start commits (new_w, stale_h) and reverts the write on the axis the user
    # never edited.
    document = app.ui_documents[app.current_document_id].document
    document.set_canvas_size((1280, 960))

    _run_row_frames(app, external_write=(800, 600))

    # The typed width clamps to 16; the height is the external write's, not the pre-edit 960.
    assert document.canvas_size == (16, 600), (
        f"the untouched height was clobbered: {document.canvas_size}"
    )


def test_the_active_field_keeps_its_own_pending_digits(app: Any) -> None:
    # The mirror must not steal what the user is typing: the width that lands is the edited one,
    # not the document's. Falsifier: mirroring an ACTIVE field's half every frame overwrites the
    # digits and the commit writes the document's own width straight back.
    document = app.ui_documents[app.current_document_id].document
    document.set_canvas_size((1280, 960))
    assert document.canvas_size == (1280, 960)

    _run_row_frames(app, external_write=None)

    # 16 is the typed 8 clamped: the digits reached the document. The height, whose field was
    # never touched, is the one this test's own setup put there.
    assert document.canvas_size == (16, 960), (
        f"the active field's edit never reached the document: {document.canvas_size}"
    )


def test_a_document_switch_mid_edit_does_not_resize_the_new_document(app: Any) -> None:
    # Clearing the editing flags is not enough on its own: imgui keeps the ITEM active across the
    # switch, so an unscoped `##canvas_w` id lets the outgoing document's half-typed digit re-latch
    # onto the incoming one and commit to it on click-away. The row's ids are scoped per document,
    # so the new document's field is a different item and cannot inherit that activeness.
    other_id = seed_extra_document(app, "bbbbbbbb-0000-4000-8000-00000000beef")
    original_id = app.current_document_id
    app.ui_documents[original_id].document.set_canvas_size((1280, 960))
    app.ui_documents[other_id].document.set_canvas_size((333, 444))

    imgui.set_window_focus(None)
    app.canvas_w_editing = False
    app.canvas_h_editing = False

    for frame in range(9):
        if frame == 3:
            imgui.get_io().add_input_character(ord("5"))
        if frame == 4:
            app.set_current_document_id(other_id)
        imgui.new_frame()
        imgui.begin("rig")
        if frame in (0, 1, 2):
            imgui.set_keyboard_focus_here(1)
        if frame == 6:
            imgui.set_keyboard_focus_here(3)
        document_tab.draw(app)
        imgui.end()
        imgui.end_frame()

    assert app.ui_documents[other_id].document.canvas_size == (333, 444), (
        "a digit typed into another document resized this one"
    )
    assert app.ui_documents[original_id].document.canvas_size == (1280, 960), (
        "the document being edited was resized by a switch that should have discarded the edit"
    )
