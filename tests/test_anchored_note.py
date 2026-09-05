"""The `K` / `F8` note's geometry (079 W-A, D1 D13).

The note used to be `always_auto_resize`, which imgui sizes from the PREVIOUS frame's content:
moving `K` from one symbol to another drew the new note inside the old note's box for one frame
and settled on the next. That is the blink the maintainer filed. The size is now measured, so a
note's box is a function of its own content and nothing else — which is what these assert.
"""

from typing import Any

from imgui_bundle import imgui

from shaderbox.theme import SIZE
from shaderbox.ui_primitives import _ellipsize, anchored_note


def _note_size(title: str, body: str, value: str = "") -> tuple[float, float]:
    # The window's size the frame it is drawn, not the frame after.
    imgui.new_frame()
    anchored_note("##probe", (100.0, 100.0), title, body, value)
    imgui.begin("##probe")
    size = imgui.get_window_size()
    imgui.end()
    imgui.end_frame()
    return (round(size.x, 1), round(size.y, 1))


def test_a_notes_size_does_not_lag_a_frame_behind_its_content(app: Any) -> None:
    # Falsifier: restore `always_auto_resize` — the tall note's FIRST frame reports the short
    # note's height, so `tall_first` and `tall_settled` differ.
    short = "vec3 mix(vec3 a, vec3 b, float t)"
    tall = "\n".join(f"a wrapped line of documentation number {i}" for i in range(8))

    _note_size(short, "")
    tall_first = _note_size(tall, "")
    tall_settled = _note_size(tall, "")
    assert tall_first == tall_settled, (
        f"the note measured {tall_first} on the frame its content changed and "
        f"{tall_settled} on the next — a one-frame lag is the blink"
    )
    assert tall_first[1] > _note_size(short, "")[1], "the rig must span two heights"


def test_a_long_value_is_cut_rather_than_widening_the_note(app: Any) -> None:
    # A long array uniform's value must not overflow the note (finding 3). Falsifier: drop the
    # `_ellipsize` and the note grows past its width token.
    long_value = ", ".join(str(i / 7.0) for i in range(40))
    imgui.new_frame()
    imgui.begin("rig")
    padding = imgui.get_style().window_padding
    wrap = float(SIZE.NOTE_W) - 2.0 * padding.x
    cut = _ellipsize(long_value, wrap)
    imgui.end()
    imgui.end_frame()
    assert cut.endswith("...") and len(cut) < len(long_value)

    width = _note_size("u_wave", "", long_value)[0]
    assert width <= float(SIZE.NOTE_W), (
        f"the note is {width}px wide against a {SIZE.NOTE_W}px token"
    )
