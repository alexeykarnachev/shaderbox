"""The code panel's cursor follow and its tab order (071 W-B).

The follow and the order permutation are imgui-free helpers, tested against the vendored
editor and a bare tab list; Ctrl+Tab's focus rule is driven through the real frame loop with
injected imgui input, the way the walk's repro was."""

from types import SimpleNamespace
from typing import Any

from imgui_bundle import imgui

from shaderbox.editor.ffi import Editor
from shaderbox.tabs import code as code_tab
from shaderbox.ui import update_and_draw


def _editor_with_status_row(lines: int) -> tuple[Editor, tuple[float, float], int]:
    e = Editor("\n".join(f"line{i}" for i in range(lines)))
    e.set_draw_chrome(True)
    e.layout((400.0, 200.0), 16.0)  # learn the cell size
    cell_h = e.get_cell_size()[1]
    size = (400.0, cell_h * 6)  # five text rows and the status row
    rows = int(size[1] / cell_h) - 1
    return e, size, rows


def test_the_follow_brings_a_line_the_edit_just_added_into_view() -> None:
    # `o` on the last line: the new line is not in the previous layout, so a follow that runs
    # before this frame's layout clamps to the old last line and leaves the caret under the
    # status row. Falsifier: move the follow back in front of the layout.
    e, size, rows = _editor_with_status_row(10)
    last = None
    e.feed("G")
    last = code_tab.layout_following_cursor(e, size, 16.0, rows, last)
    assert e.get_scroll() == 5 and last.line == 9
    e.feed("o")
    cursor = code_tab.layout_following_cursor(e, size, 16.0, rows, last)
    first = e.get_scroll()
    assert cursor.line == 10
    assert first <= cursor.line < first + rows, (
        f"caret row {cursor.line - first} of {rows}"
    )
    e.close()


def test_the_follow_leaves_an_idle_caret_alone() -> None:
    # Wheel-scrolling away from a caret that did not move must not snap back.
    e, size, rows = _editor_with_status_row(30)
    last = code_tab.layout_following_cursor(e, size, 16.0, rows, None)
    e.set_scroll(12)
    code_tab.layout_following_cursor(e, size, 16.0, rows, last)
    assert e.get_scroll() == 12
    e.close()


def test_the_display_order_permutes_the_model_and_keeps_the_active_tab() -> None:
    tabs = [SimpleNamespace(path=f"/p{i}") for i in range(3)]
    app = SimpleNamespace(
        editor_tabs=list(tabs), active_tab_index=2, active_tab=tabs[2]
    )
    code_tab._apply_display_order(app, [2, 0, 1])  # the eye sees p2 first
    assert app.editor_tabs == [tabs[2], tabs[0], tabs[1]]
    assert app.active_tab_index == 0
    code_tab._apply_display_order(app, [0, 1, 2])  # identity: untouched
    assert app.editor_tabs == [tabs[2], tabs[0], tabs[1]]


def _frames(app: Any, n: int) -> None:
    for _ in range(n):
        update_and_draw(app)


def _ctrl_tab(app: Any) -> None:
    io = imgui.get_io()
    io.add_key_event(imgui.Key.left_ctrl, True)
    io.add_key_event(imgui.Key.mod_ctrl, True)
    _frames(app, 1)
    io.add_key_event(imgui.Key.tab, True)
    _frames(app, 2)
    io.add_key_event(imgui.Key.tab, False)
    io.add_key_event(imgui.Key.left_ctrl, False)
    io.add_key_event(imgui.Key.mod_ctrl, False)
    _frames(app, 3)


def test_ctrl_tab_focuses_an_unfocused_editor_first_then_cycles(
    app: Any, monkeypatch: Any
) -> None:
    # The walk's repro: working in the app panel, Ctrl+Tab switched the code tab. Now the first
    # press only focuses the editor; the next one cycles.
    orders: list[list[int]] = []
    real_apply = code_tab._apply_display_order
    monkeypatch.setattr(
        code_tab,
        "_apply_display_order",
        lambda a, order: (orders.append(order), real_apply(a, order)),
    )
    _frames(app, 5)
    app.open_script_for(app.current_document_id)
    _frames(app, 5)
    assert len(app.editor_tabs) == 2
    io = imgui.get_io()
    x, y, w, _h = app.editor_rect
    io.add_mouse_pos_event(x + w + 200, y + 300)  # the app panel, right of the editor
    _frames(app, 2)
    io.add_mouse_button_event(0, True)
    _frames(app, 2)
    io.add_mouse_button_event(0, False)
    _frames(app, 3)
    assert not app.editor_focused
    before = app.active_tab_index

    _ctrl_tab(app)
    assert app.active_tab_index == before, "the first press must not switch the tab"
    assert app.editor_focused, "the first press focuses the editor"

    _ctrl_tab(app)
    assert app.active_tab_index != before, "a press while focused cycles"
    # The order read-back resolved every tab name to a model index: once the second tab
    # existed, every frame reported the two-tab display order.
    assert [0, 1] in orders
    assert all(sorted(o) == list(range(len(o))) for o in orders)
