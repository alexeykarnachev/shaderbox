"""The shader-library picker's inline inputs commit when focus leaves them (069 D11).

A rename / new-file / new-dir input holds a REQUEST that has not happened yet, so an Enter-only
commit silently discards the work of anyone who clicks away instead of pressing Enter. Driven
through a real imgui frame: focus the input, type a character, move focus off it — which is what
makes the query `is_item_deactivated_after_edit` (an edit, not a bare focus move) read True.
"""

import contextlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from imgui_bundle import imgui

from conftest import seed_tmp_project
from shaderbox.paths import shader_lib_root
from shaderbox.popups.lib_picker import tree

_FN = """float sb_probe(float x) {
    return x;
}
"""


@pytest.fixture
def lib_app(monkeypatch: Any, tmp_path: Path) -> Iterator[Any]:
    # The lib root lives under app_data_dir(), so the env override is what keeps this test off
    # the maintainer's real library.
    glfw = pytest.importorskip("glfw")
    if not glfw.init():
        pytest.skip("no GL")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    monkeypatch.setenv("SHADERBOX_DATA_DIR", str(tmp_path / "data"))
    from shaderbox.app import App

    a = App(project_dir=seed_tmp_project(tmp_path))
    a.copilot.bridge.run_on_main = lambda fn, timeout=None, defer=False: fn()
    yield a
    with contextlib.suppress(Exception):
        a.release()


def _run_frames(body: Callable[[int], None], *, type_char: str | None) -> None:
    # The five-frame recipe: focus, a REAL keystroke, then a focus move. Without the keystroke
    # only `is_item_deactivated` fires, never the after-edit form D11 rests on.
    for frame in range(6):
        if frame == 2 and type_char is not None:
            imgui.get_io().add_input_character(ord(type_char))
        imgui.new_frame()
        imgui.begin("rig")
        if frame in (0, 1):
            imgui.set_keyboard_focus_here(0)
        if frame == 3:
            imgui.set_keyboard_focus_here(1)
        body(frame)
        imgui.input_text("##sink", "sink")
        imgui.end()
        imgui.end_frame()


def test_a_picker_inline_input_commits_on_click_away(lib_app: Any) -> None:
    root = shader_lib_root()
    original = root / "probe.glsl"
    original.write_text(_FN, encoding="utf-8")
    lib_app.shader_lib_files.begin_file_rename(original)

    _run_frames(
        lambda _frame: tree._draw_file_rename_input(lib_app, original),
        type_char="r",
    )

    assert not original.exists(), "the click-away committed nothing"
    assert (root / "r.glsl").is_file()
    assert not lib_app.shader_lib_files.file_rename.is_open


def test_a_focus_move_with_no_edit_commits_nothing(lib_app: Any) -> None:
    # `is_item_deactivated` fires on any focus move; only the after-edit form means the user
    # typed something. Committing on the bare form would rename on a stray click.
    root = shader_lib_root()
    original = root / "probe.glsl"
    original.write_text(_FN, encoding="utf-8")
    lib_app.shader_lib_files.begin_file_rename(original)
    lib_app.shader_lib_files.file_rename.buf = "moved.glsl"

    _run_frames(
        lambda _frame: tree._draw_file_rename_input(lib_app, original),
        type_char=None,
    )

    assert original.is_file(), "a focus move with no edit renamed the file"
    assert not (root / "moved.glsl").exists()
