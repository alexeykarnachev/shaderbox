"""The `K` / `F8` note's geometry (079 W-A, D1 D13).

The note used to be `always_auto_resize`, which imgui sizes from the PREVIOUS frame's content:
moving `K` from one symbol to another drew the new note inside the old note's box for one frame
and settled on the next. That is the blink the maintainer filed. The size is now measured, so a
note's box is a function of its own content and nothing else — which is what these assert.
"""

from typing import Any

from imgui_bundle import imgui

from shaderbox.scripting.api_doc import API_NAMES, api_symbol_doc
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


def _note_geometry(
    anchor: tuple[float, float], title: str, body: str
) -> tuple[float, float, float, float]:
    """(top, height, where the content ended, how far it can scroll) for one drawn note.

    Two frames: one imgui window id serves every note, so `scroll_max` on the first frame after
    the content changes still describes the PREVIOUS note.
    """
    for _ in range(2):
        imgui.new_frame()
        anchored_note("##geometry", anchor, title, body)
        imgui.begin("##geometry")
        top = imgui.get_window_pos().y
        height = imgui.get_window_size().y
        used = imgui.get_cursor_screen_pos().y - top
        scroll_max = imgui.get_scroll_max_y()
        imgui.end()
        imgui.end_frame()
    return top, height, used, scroll_max


def test_a_note_is_tall_enough_for_every_api_docstring(app: Any) -> None:
    # `K` on `ScriptContext` clipped its last lines. imgui advances the cursor by an item's height PLUS
    # `item_spacing.y` after EVERY item, the last one included, so N items cost N gaps — the
    # measurement counted N-1 and came up exactly one gap short. Falsifier: drop the trailing
    # gap from any one branch and the note that uses it clips.
    padding = imgui.get_style().window_padding.y
    for name in sorted(API_NAMES):
        title, body = api_symbol_doc(name)
        _, height, used, scroll_max = _note_geometry((50.0, 50.0), title, body)
        assert height >= used + padding - 0.5, (
            f"`K` on {name} is clipped by {used + padding - height:.1f}px"
        )
        assert scroll_max == 0.0, f"`K` on {name} needs a scrollbar to show its text"


def test_a_tall_note_stays_on_screen(app: Any) -> None:
    # A note grows to fit, so a long docstring near an edge would run off it.
    screen = imgui.get_io().display_size.y
    tall = "\n".join(f"line {i} of a long docstring" for i in range(80))
    for label, anchor_y in (
        ("low", screen - 200.0),
        ("high", 100.0),
        ("mid", screen / 2),
    ):
        top, height, _, _ = _note_geometry((50.0, anchor_y), "Title", tall)
        assert top >= 0.0, f"the {label} note starts above the screen at {top:.0f}"
        assert top + height <= screen + 0.5, (
            f"the {label} note ends {top + height - screen:.0f}px below the screen"
        )


def test_a_note_anchored_low_flips_above_the_caret(app: Any) -> None:
    # Capping alone keeps a note on screen but leaves it a sliver when the caret is near the
    # bottom: the room BELOW is what the cap is measured against. Flipping uses the room above
    # instead. Falsifier: drop the flip and a note two rows from the bottom is ~40px tall rather
    # than showing its text.
    screen = imgui.get_io().display_size.y
    tall = "\n".join(f"line {i} of a long docstring" for i in range(80))
    top, height, _, _ = _note_geometry((50.0, screen - 40.0), "Title", tall)
    assert top + height <= screen - 20.0, "the flipped note must clear the caret's row"
    assert height > screen / 2, (
        f"the note is {height:.0f}px tall with {screen - 40.0:.0f}px of room above it — "
        "it was capped to the room below instead of flipping"
    )
