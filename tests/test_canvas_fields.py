"""The Document tab's W x H pair, driven through real imgui frames (069 W-A, 090 D5).

Commit-on-deactivate needs a real focus transition across frames, so these drive
`tabs/document.py::draw` in a headless imgui frame the way `test_lib_files.py` drives the
picker's inline inputs, rather than asserting against the code.

The property under test is the one a post-implementation review found broken: only the field
the user is actually in holds a pending value. The other half mirrors the document every frame,
so a write that lands mid-edit (the copilot, a disk sync) is not reverted by a stale number the
user never touched.

What the pair WRITES is the document's stored `resolution` (090 D5), and the live canvas
follows on the next tick through `App.pending_resolution` -- a resize inside the draw phase
would release textures imgui is still holding. So the assertions here read `resolution`.

The fields exist only under FIXED (090 revision 1): under Auto the same slot draws the aspect
control instead, since a pair the mode never reads is a control that lies about what it does.
Every rig here therefore puts its document in Fixed first.
"""

from typing import Any

from imgui_bundle import imgui

from shaderbox.render_shape import ResolutionMode
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
    # The W x H fields are the FIXED-mode control; under Auto the row draws aspect chips and
    # this rig's keystrokes would land on a different widget entirely.
    app.ui_documents[
        app.current_document_id
    ].document.resolution_mode = ResolutionMode.FIXED

    for frame in range(9):
        if frame == 3:
            imgui.get_io().add_input_character(ord("8"))
        if frame == 4 and external_write is not None:
            app.ui_documents[
                app.current_document_id
            ].document.resolution = external_write
        imgui.new_frame()
        imgui.begin("rig")
        # Offsets count focusable items from the cursor: 079 D7 put Reset on the caption row
        # ABOVE the inputs, and 090 revision 1 made the mode control a two-button segment
        # where it was one toggle -- each shifted everything after it by one.
        if frame in (0, 1, 2):
            imgui.set_keyboard_focus_here(4)
        if frame == 5:
            imgui.set_keyboard_focus_here(6)
        document_tab.draw(app)
        imgui.end()
        imgui.end_frame()


def test_a_write_during_an_active_field_survives_the_commit(app: Any) -> None:
    # With W active and H untouched, an external 800x600 must keep its 600: the commit pair is
    # (the pending width, the document's CURRENT height). Falsifier: a buffer whose inactive half
    # is frozen at edit-start commits (new_w, stale_h) and reverts the write on the axis the user
    # never edited.
    document = app.ui_documents[app.current_document_id].document
    document.resolution = (1280, 960)

    _run_row_frames(app, external_write=(800, 600))

    # The typed width clamps to 16; the height is the external write's, not the pre-edit 960.
    assert document.resolution == (16, 600), (
        f"the untouched height was clobbered: {document.resolution}"
    )


def test_the_active_field_keeps_its_own_pending_digits(app: Any) -> None:
    # The mirror must not steal what the user is typing: the width that lands is the edited one,
    # not the document's. Falsifier: mirroring an ACTIVE field's half every frame overwrites the
    # digits and the commit writes the document's own width straight back.
    document = app.ui_documents[app.current_document_id].document
    document.resolution = (1280, 960)

    _run_row_frames(app, external_write=None)

    # 16 is the typed 8 clamped: the digits reached the document. The height, whose field was
    # never touched, is the one this test's own setup put there.
    assert document.resolution == (16, 960), (
        f"the active field's edit never reached the document: {document.resolution}"
    )
    # ... and the commit is DEFERRED, not applied inside the draw (090 D4/R2): the pair the
    # next tick will apply is parked, and the live canvas has not moved yet.
    assert app.pending_resolution[app.current_document_id] == (16, 960)


def test_a_document_switch_mid_edit_does_not_resize_the_new_document(app: Any) -> None:
    # Clearing the editing flags is not enough on its own: imgui keeps the ITEM active across the
    # switch, so an unscoped `##canvas_w` id lets the outgoing document's half-typed digit re-latch
    # onto the incoming one and commit to it on click-away. The row's ids are scoped per document,
    # so the new document's field is a different item and cannot inherit that activeness.
    other_id = seed_extra_document(app, "bbbbbbbb-0000-4000-8000-00000000beef")
    original_id = app.current_document_id
    app.ui_documents[original_id].document.resolution = (1280, 960)
    app.ui_documents[other_id].document.resolution = (333, 444)

    imgui.set_window_focus(None)
    app.canvas_w_editing = False
    app.canvas_h_editing = False
    for document_id in (original_id, other_id):
        app.ui_documents[document_id].document.resolution_mode = ResolutionMode.FIXED

    for frame in range(9):
        if frame == 3:
            imgui.get_io().add_input_character(ord("5"))
        if frame == 4:
            app.set_current_document_id(other_id)
        imgui.new_frame()
        imgui.begin("rig")
        if frame in (0, 1, 2):
            imgui.set_keyboard_focus_here(3)
        if frame == 6:
            imgui.set_keyboard_focus_here(5)
        document_tab.draw(app)
        imgui.end()
        imgui.end_frame()

    assert app.ui_documents[other_id].document.resolution == (333, 444), (
        "a digit typed into another document resized this one"
    )
    assert app.ui_documents[original_id].document.resolution == (1280, 960), (
        "the document being edited was resized by a switch that should have discarded the edit"
    )


def _draw_tab_once(app: Any) -> None:
    imgui.new_frame()
    imgui.begin("rig")
    document_tab.draw(app)
    imgui.end()
    imgui.end_frame()


def _captions(app: Any, monkeypatch: Any) -> list[str]:
    """Every `small_caption` string the tab drew, in order.

    The readout follows the caption row's two labels and precedes the Passes caption (092
    moved that one into this tab). Captured
    rather than asserted against the pixels, because what this pins is the string the mode
    decides -- a layout swap must be free, which is why the control lives in one function.
    """
    seen: list[str] = []
    real = document_tab.small_caption

    def spy(font: Any, text: str) -> None:
        seen.append(text)
        real(font, text)

    monkeypatch.setattr(document_tab, "small_caption", spy)
    _draw_tab_once(app)
    return seen


def test_an_auto_documents_readout_is_its_live_size(app: Any, monkeypatch: Any) -> None:
    # Row 2 under Auto shows the number the control does NOT carry: the control names a ratio,
    # so the readout is the pixels. Falsifier: read `resolution` there and it reports a pair
    # the mode never renders at.
    document = app.ui_documents[app.current_document_id].document
    document.resolution_mode = ResolutionMode.AUTO
    document.aspect = (16, 9)
    document.resolution = (111, 222)  # a stale pair the readout must not show
    document.set_canvas_size((1214, 683))

    captions = _captions(app, monkeypatch)
    assert captions[-2] == "1214x683", captions
    assert "Aspect" in captions, "the caption row does not name the aspect under Auto"


def test_a_fixed_documents_readout_is_the_aspect_of_its_pair(
    app: Any, monkeypatch: Any
) -> None:
    # The mirror: the control names a size, so the readout is the shape. Falsifier: print the
    # size again and row 2 repeats what the row above already says.
    document = app.ui_documents[app.current_document_id].document
    document.resolution_mode = ResolutionMode.FIXED
    document.resolution = (1280, 720)

    captions = _captions(app, monkeypatch)
    assert captions[-2] == "16:9", captions
    assert "Canvas" in captions, "the caption row does not name the canvas under Fixed"


def test_the_aspect_chips_write_the_reduced_ratio(app: Any) -> None:
    # The control's whole contract with the rest of the app: whatever the user picks arrives
    # reduced. Driven through `_apply_aspect`, the one seam every chip and both fields reach.
    # Falsifier: store the pair as typed and a `32:18` custom ratio never lights the 16:9 chip.
    from shaderbox.tabs.document import _apply_aspect

    ui_document = app.ui_documents[app.current_document_id]
    ui_document.document.resolution_mode = ResolutionMode.AUTO
    _apply_aspect(app, ui_document, (32, 18))
    assert ui_document.ui_state.aspect == (16, 9)
    assert ui_document.document.aspect == (16, 9)
