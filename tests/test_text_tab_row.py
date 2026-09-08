"""Feature 087 W-B: the pass selector's row of clickable names.

The row carries its state by color alone, so the hover color is chosen BEFORE the item is
submitted -- a `selectable` has no pre-hover to read. That makes the predicted rect part of
the behavior rather than an implementation detail, and it is what these tests pin, along with
the wrap that keeps a long row inside the content region.
"""

from collections.abc import Callable, Sequence
from typing import Any

from imgui_bundle import imgui

from shaderbox import ui_primitives
from shaderbox.theme import COLOR

_Rect = tuple[tuple[float, float], tuple[float, float]]


def _drive(
    names: Sequence[str],
    window_w: float,
    mouse: tuple[float, float],
    before: Callable[[], None] | None = None,
) -> dict[str, tuple[tuple[float, ...], _Rect]]:
    """Run `text_tab_row` for four frames with the pointer parked, and report, per name, the
    text color pushed for it and the rect imgui gave its item on the last frame."""
    seen: dict[str, tuple[tuple[float, ...], _Rect]] = {}
    real_push, real_selectable = imgui.push_style_color, imgui.selectable
    pending: dict[str, tuple[float, ...]] = {}

    def push_spy(index: Any, color: Any) -> Any:
        if index == imgui.Col_.text:
            pending["text"] = tuple(round(v, 3) for v in color)
        return real_push(index, color)

    def selectable_spy(label: str, *args: Any, **kwargs: Any) -> Any:
        result = real_selectable(label, *args, **kwargs)
        seen[label.split("##")[0]] = (
            pending["text"],
            (
                tuple(imgui.get_item_rect_min()),
                tuple(imgui.get_item_rect_max()),
            ),
        )
        return result

    io = imgui.get_io()
    for frame in range(4):
        io.add_mouse_pos_event(*mouse)
        imgui.new_frame()
        imgui.set_next_window_pos((0.0, 0.0))
        imgui.set_next_window_size((window_w, 300.0))
        imgui.begin("text_tab_row_rig")
        if before is not None:
            before()
        if frame == 3:
            imgui.push_style_color, imgui.selectable = push_spy, selectable_spy
        ui_primitives.text_tab_row("rig", names, names[0])
        if frame == 3:
            imgui.push_style_color, imgui.selectable = real_push, real_selectable
        imgui.end()
        imgui.end_frame()
    return seen


def _rounded(color: tuple[float, float, float, float]) -> tuple[float, ...]:
    return tuple(round(v, 3) for v in color)


def test_the_hover_color_covers_the_whole_rect_imgui_gave_the_item(app: Any) -> None:
    # The rect a `selectable` claims is the text outset by half an item spacing on each side,
    # so a prediction of the bare text rect leaves a live margin that answers clicks while
    # reading dim. Falsifier: predict `origin -> origin + (width, line_height_with_spacing)`
    # and the left edge measures FG_DIM while imgui reports the item hovered.
    names = ["main", "second"]
    parked = _drive(names, 600.0, (500.0, 250.0))
    (_, (rect_min, rect_max)) = parked["second"]
    assert parked["second"][0] == _rounded(COLOR.FG_DIM), "an unhovered name is dim"

    middle_y = (rect_min[1] + rect_max[1]) / 2
    for x, edge in ((rect_min[0] + 1.0, "left"), (rect_max[0] - 1.0, "right")):
        color = _drive(names, 600.0, (x, middle_y))["second"][0]
        assert color == _rounded(COLOR.FG_SECONDARY), (
            f"the {edge} edge of the item's own rect read {color}, not the hover color"
        )

    # And the gap between two names stays dark: the rects do not run together.
    gap_x = (parked["main"][1][1][0] + rect_min[0]) / 2
    assert _drive(names, 600.0, (gap_x, middle_y))["second"][0] == _rounded(
        COLOR.FG_DIM
    ), "the gap between the names lit one of them"


def test_a_row_too_long_for_the_panel_wraps_inside_the_content_region(app: Any) -> None:
    # Falsifier: make the `same_line` unconditional -- every name then stays on line one and
    # the row runs past the window's right edge.
    names = ["alpha", "bravo", "charlie", "delta"]
    drawn = _drive(names, 200.0, (500.0, 250.0))
    tops = [drawn[name][1][0][1] for name in names]
    assert len(set(tops)) > 1, f"the row never wrapped: every name at y {tops[0]}"

    # Every name ends inside the region the window offers its content.
    for _frame in range(2):
        imgui.new_frame()
        imgui.set_next_window_pos((0.0, 0.0))
        imgui.set_next_window_size((200.0, 300.0))
        imgui.begin("text_tab_row_rig")
        right_edge = (
            imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
        )
        imgui.end()
        imgui.end_frame()
    for name, (_, (_, rect_max)) in drawn.items():
        assert rect_max[0] <= right_edge, (
            f"`{name}` ends at {rect_max[0]}, past the content region's {right_edge}"
        )
