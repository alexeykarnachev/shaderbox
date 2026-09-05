"""The shader-library picker's inline inputs commit when focus leaves them (069 D11).

A rename / new-file / new-dir input holds a REQUEST that has not happened yet, so an Enter-only
commit silently discards the work of anyone who clicks away instead of pressing Enter. Driven
through a real imgui frame: focus the input, type a character, move focus off it — which is what
makes the query `is_item_deactivated_after_edit` (an edit, not a bare focus move) read True.

The `app` fixture points `SHADERBOX_DATA_DIR` at a tmp dir, which is what keeps these renames off
the developer's own shader library — `shader_lib_root()` resolves under `app_data_dir()`.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from shaderbox.paths import shader_lib_root
from shaderbox.popups.lib_picker import tree

_FN = """float sb_probe(float x) {
    return x;
}
"""


def _run_frames(
    body: Callable[[int], None],
    *,
    type_char: str | None,
    cancel_on: int | None = None,
) -> None:
    # The five-frame recipe: focus, a REAL keystroke, then a focus move. Without the keystroke
    # only `is_item_deactivated` fires, never the after-edit form D11 rests on.
    #
    # `cancel_on` makes the row's `x` button report a click on that frame. imgui's hover test
    # never fires for a synthetic mouse in a headless context (/imgui-ui § 0), so the click is
    # injected at the button rather than at the mouse — the drawn function still runs its own
    # real branches, in their real order, against a real deactivate.
    for frame in range(6):
        if frame == 2 and type_char is not None:
            imgui.get_io().add_input_character(ord(type_char))
        imgui.new_frame()
        imgui.begin("rig")
        if frame in (0, 1):
            imgui.set_keyboard_focus_here(0)
        if frame == 3:
            # A real cancel click both defocuses the input and presses the button; the focus
            # move is what makes the deactivate fire, one frame ahead of the click.
            imgui.set_keyboard_focus_here(1)
        real_button = tree.standard_button

        def stub(
            label: str,
            *args: Any,
            _real: Any = real_button,
            _clicking: bool = frame == cancel_on,
            **kwargs: Any,
        ) -> bool:
            _real(label, *args, **kwargs)
            return _clicking and label.startswith("x##")

        tree.standard_button = stub
        try:
            body(frame)
        finally:
            tree.standard_button = real_button
        imgui.input_text("##sink", "sink")
        imgui.end()
        imgui.end_frame()


def test_a_picker_inline_input_commits_on_click_away(app: Any) -> None:
    root = shader_lib_root()
    original = root / "probe.glsl"
    original.write_text(_FN, encoding="utf-8")
    app.shader_lib_files.begin_file_rename(original)

    _run_frames(
        lambda _frame: tree._draw_file_rename_input(app, original),
        type_char="r",
    )

    assert not original.exists(), "the click-away committed nothing"
    assert (root / "r.glsl").is_file()
    assert not app.shader_lib_files.file_rename.is_open


def test_a_focus_move_with_no_edit_commits_nothing(app: Any) -> None:
    # `is_item_deactivated` fires on any focus move; only the after-edit form means the user
    # typed something. Committing on the bare form would rename on a stray click.
    root = shader_lib_root()
    original = root / "probe.glsl"
    original.write_text(_FN, encoding="utf-8")
    app.shader_lib_files.begin_file_rename(original)
    app.shader_lib_files.file_rename.buf = "moved.glsl"

    _run_frames(
        lambda _frame: tree._draw_file_rename_input(app, original),
        type_char=None,
    )

    assert original.is_file(), "a focus move with no edit renamed the file"
    assert not (root / "moved.glsl").exists()


def test_the_cancel_button_cancels_instead_of_committing(app: Any) -> None:
    # The `x` click is itself what deactivates the input, so a commit read before the button is
    # drawn fires the very transaction the click was cancelling. Capture-then-apply is what
    # makes the click mean cancel.
    root = shader_lib_root()
    original = root / "probe.glsl"
    original.write_text(_FN, encoding="utf-8")
    app.shader_lib_files.begin_file_rename(original)

    _run_frames(
        lambda _frame: tree._draw_file_rename_input(app, original),
        type_char="c",
        cancel_on=4,
    )

    assert original.is_file(), "the cancel click renamed the file"
    assert not (root / "c.glsl").exists()
    assert not app.shader_lib_files.file_rename.is_open


def test_the_new_file_cancel_button_creates_nothing(app: Any) -> None:
    dir_rel: tuple[str, ...] = ()
    app.shader_lib_files.begin_file_new_in(Path(*dir_rel))
    created: list[Path] = []

    def draw(_frame: int) -> None:
        tree._draw_inline_new_input(
            state=app.shader_lib_files.file_new,
            label="New file:",
            id_prefix="new_file_in",
            dir_rel=dir_rel,
            commit=app.shader_lib_files.commit_file_new,
            cancel=app.shader_lib_files.cancel_file_new,
            on_create=created.append,
        )

    _run_frames(draw, type_char="n", cancel_on=4)

    assert created == [], f"the cancel click created {created}"
    assert not (shader_lib_root() / "n.glsl").exists()
    assert not app.shader_lib_files.file_new.is_open
