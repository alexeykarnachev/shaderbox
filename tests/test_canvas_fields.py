"""The Document tab's canvas control, driven through real imgui frames (093 W8).

The W x H pair and the Auto-side aspect chips are gone: one combo now names both the mode and
its value, grouped by aspect with `Auto` leading each group (the maintainer's call -- "просто в
списке пресетов появится auto"). So what these tests pin moved with it. The row's per-document
id scoping is still tested here, since imgui keeps an ITEM active across a document switch and
an unscoped id would let the outgoing document's open popup land on the incoming one.

What a pick WRITES is the document's stored `resolution` (090 D5), and the live canvas follows
on the next tick through `App.pending_resolution` -- a resize inside the draw phase would
release textures imgui is still holding. So the assertions here read `resolution`.
"""

from typing import Any

from shaderbox.render_shape import ASPECT_PRESETS, ResolutionMode
from shaderbox.widgets.canvas_control import (
    CanvasChoiceKind,
    _apply_aspect,
    apply_canvas_choice,
    canvas_caption,
    canvas_choice_groups,
    canvas_choice_label,
)


def test_an_auto_documents_readout_is_its_live_size(app: Any, monkeypatch: Any) -> None:
    # The caption under Auto carries the number the control does NOT: the control names a
    # ratio, so the caption shows the pixels. Falsifier: read `resolution` there and it
    # reports a pair the mode never renders at.
    document = app.ui_documents[app.current_document_id].document
    document.resolution_mode = ResolutionMode.AUTO
    document.aspect = (16, 9)
    document.resolution = (111, 222)  # a stale pair the readout must not show
    document.set_canvas_size((1214, 683))

    assert (
        canvas_caption(app.ui_documents[app.current_document_id]) == "Canvas  1214x683"
    )


def test_a_fixed_documents_readout_is_the_aspect_of_its_pair(
    app: Any, monkeypatch: Any
) -> None:
    # The mirror: the control names a size, so the caption is the shape. Falsifier: print the
    # size again and the caption repeats what the combo below it already says.
    document = app.ui_documents[app.current_document_id].document
    document.resolution_mode = ResolutionMode.FIXED
    document.resolution = (1280, 720)

    assert canvas_caption(app.ui_documents[app.current_document_id]) == "Canvas  16:9"


def test_an_aspect_pick_writes_the_reduced_ratio(app: Any) -> None:
    # The control's whole contract with the rest of the app: whatever the user picks arrives
    # reduced. Driven through `_apply_aspect`, the one seam every Auto row reaches. Falsifier:
    # store the pair as typed and a `32:18` ratio never matches the 16:9 group.
    ui_document = app.ui_documents[app.current_document_id]
    ui_document.document.resolution_mode = ResolutionMode.AUTO
    _apply_aspect(app, ui_document, (32, 18))
    assert ui_document.ui_state.aspect == (16, 9)
    assert ui_document.document.aspect == (16, 9)


def test_every_aspect_group_leads_with_auto(app: Any) -> None:
    """Each group is one ratio and its first row is that ratio's `Auto` (093 W8).

    Falsifier: append the Auto row instead of leading with it, or drop the group for a ratio
    no fixed size covers -- 21:9 and 3:4 then have no Auto to pick.
    """
    ui_document = app.ui_documents[app.current_document_id]
    groups = canvas_choice_groups(ui_document)

    captions = [caption for caption, _ in groups]
    for aspect in ASPECT_PRESETS:
        assert f"{aspect[0]}:{aspect[1]}" in captions, (
            f"{aspect} has no group, so its Auto is unreachable"
        )

    for caption, rows in groups:
        assert rows[0].kind is CanvasChoiceKind.AUTO, (
            f"{caption} does not lead with Auto"
        )
        assert rows[0].label == "Auto"
        assert f"{rows[0].aspect[0]}:{rows[0].aspect[1]}" == caption
        for row in rows[1:]:
            assert row.kind is CanvasChoiceKind.FIXED
            assert row.size is not None


def test_a_fixed_row_sits_in_the_group_a_reader_would_name_it(app: Any) -> None:
    """An encoder-aligned 1920x1088 groups under 16:9, not under its exact 30:17.

    Falsifier: group by `aspect_of` (the exact reduction) -- the three Wide shapes then scatter
    across three captions that name the same shape.
    """
    ui_document = app.ui_documents[app.current_document_id]
    groups = dict(canvas_choice_groups(ui_document))
    wide = {row.size for row in groups["16:9"] if row.size is not None}
    assert (1920, 1088) in wide, "the 1080p shape left the 16:9 group"
    assert (1280, 720) in wide and (2560, 1440) in wide


def test_picking_auto_sets_the_mode_and_its_ratio(app: Any) -> None:
    """An Auto row is one click: the mode AND the aspect it names.

    Falsifier: set the mode and leave the aspect -- picking `4:3 Auto` from a 16:9 document
    then renders the old shape.
    """
    ui_document = app.ui_documents[app.current_document_id]
    ui_document.document.resolution_mode = ResolutionMode.FIXED
    ui_document.document.resolution = (1280, 720)

    groups = dict(canvas_choice_groups(ui_document))
    apply_canvas_choice(app, ui_document, groups["4:3"][0])

    assert ui_document.document.resolution_mode is ResolutionMode.AUTO
    assert ui_document.document.aspect == (4, 3)
    assert ui_document.ui_state.aspect == (4, 3)


def test_picking_a_size_sets_fixed_and_that_pair(app: Any) -> None:
    """A size row is the mirror: Fixed, at exactly the pair named.

    Falsifier: apply the size without the mode -- an Auto document keeps following the viewer
    and the pair it was just given never renders.
    """
    ui_document = app.ui_documents[app.current_document_id]
    ui_document.document.resolution_mode = ResolutionMode.AUTO

    groups = dict(canvas_choice_groups(ui_document))
    row = next(r for r in groups["1:1"] if r.size == (512, 512))
    apply_canvas_choice(app, ui_document, row)

    assert ui_document.document.resolution_mode is ResolutionMode.FIXED
    assert ui_document.document.resolution == (512, 512)


def test_the_closed_chip_names_the_mode_and_its_value(app: Any) -> None:
    """The chip carries what the popup would otherwise have to be opened to learn.

    Falsifier: print the pair under Auto -- the chip then shows pixels the mode recomputes
    every time the viewer resizes.
    """
    ui_document = app.ui_documents[app.current_document_id]

    ui_document.document.resolution_mode = ResolutionMode.FIXED
    ui_document.document.resolution = (1280, 720)
    assert canvas_choice_label(ui_document) == "1280x720"

    ui_document.document.resolution_mode = ResolutionMode.AUTO
    ui_document.ui_state.aspect = (21, 9)
    assert canvas_choice_label(ui_document) == "Auto 21:9"


def test_a_row_does_not_repeat_the_ratio_its_group_names(app: Any) -> None:
    """`Wide 720p`, not `Wide 720p (16:9)`, under a caption that already reads 16:9.

    Falsifier: pass the shape table's label through unchanged -- every row then carries its
    group's own caption in brackets.
    """
    ui_document = app.ui_documents[app.current_document_id]
    for caption, rows in canvas_choice_groups(ui_document):
        for row in rows:
            assert not row.label.endswith(f"({caption})"), row.label
