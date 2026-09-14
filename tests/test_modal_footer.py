"""A modal's content ends where its footer begins, so nothing overflows into a scrollbar.

The shape the primitives replace: each modal reserved its own room by hand, and the
documentation modal reserved one frame height while its footer drew a `SPACE.MD` spacer plus
the row -- the difference was a vertical scrollbar on a modal whose content fits.

Frame-driven, because the question is about the window imgui actually laid out: a body's AST
says which primitives it calls (`test_modal_chrome`), never how tall the result came out.
"""

from dataclasses import replace
from typing import Any

import pytest
from imgui_bundle import imgui

from shaderbox.app import ModalId
from shaderbox.popups import registry
from shaderbox.popups.registry import BY_ID, draw_modal

# Real frames, so its own worker: the imgui font atlas is per PROCESS and a second App that
# renders a full frame in one interpreter dies on a texture the first released.
pytestmark = pytest.mark.xdist_group("gl_frames_modal_footer")


def _scroll_max_of(
    app: Any, modal_id: ModalId, monkeypatch: Any, frames: int = 3
) -> float:
    """Draw the open modal for real and report its POPUP window's vertical overflow.

    The reading is taken at the end of the body, where the cursor is back in the popup window
    itself rather than in one of the children it draws -- a popup cannot be re-entered with
    `begin` from the outside.
    """
    row = BY_ID[modal_id]
    seen: list[float] = []

    def body(target: Any) -> bool:
        keep_open = row.body(target)
        seen.append(imgui.get_scroll_max_y())
        return keep_open

    monkeypatch.setitem(registry.BY_ID, modal_id, replace(row, body=body))
    for _ in range(frames):
        imgui.new_frame()
        draw_modal(app)
        imgui.end_frame()
    assert seen, "the modal never drew"
    return seen[-1]


# The modals whose content is a `modal_content` region: their own body can never be the
# reason the popup window scrolls, since the region ends exactly where the footer begins.
_CONTENT_MODALS: tuple[tuple[ModalId, str], ...] = (
    (ModalId.HELP, "open_help"),
    (ModalId.SETTINGS, "open_settings"),
    (ModalId.PROJECTS, "open_projects"),
)


@pytest.mark.parametrize(
    ("modal_id", "opener"), _CONTENT_MODALS, ids=[m.value for m, _ in _CONTENT_MODALS]
)
def test_a_modal_with_a_content_region_does_not_overflow(
    app: Any, monkeypatch: Any, modal_id: ModalId, opener: str
) -> None:
    """The content region is sized from `modal_footer_height`, which is the same number the
    footer occupies -- so the modal at its own size has nothing to scroll.

    Falsifier: restore the hand reservation (`list_h = -imgui.get_frame_height_with_spacing()`
    in `help.py`, with the row drawn after a `SPACE.MD` spacer) -- the overflow comes back.
    """
    getattr(app, opener)()
    assert app.modal is modal_id
    assert _scroll_max_of(app, modal_id, monkeypatch) == 0.0
