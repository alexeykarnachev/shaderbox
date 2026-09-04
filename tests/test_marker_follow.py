"""The host's line markers survive edits (078, the maintainer's Ctrl+Shift+I finding): the
library moves a marker with the text it marks, so a whole-buffer replace that keeps the cursor
on its line must still re-push the cursor-line band."""

import ctypes
from pathlib import Path

from shaderbox.editor.ffi import Editor, Language, ensure_loaded
from shaderbox.tabs.code import _apply_markers


class _Markers:
    # App-shaped holder for the fingerprint cache `_apply_markers` reads.
    def __init__(self) -> None:
        self.editor_marker_state: dict[Path, tuple] = {}


def _marked_lines(editor: Editor) -> list[int]:
    lib = ensure_loaded()
    floats = [ctypes.c_float() for _ in range(4)]
    glyph = ctypes.c_int32()
    found: list[int] = []
    for line in range(editor.get_line_count()):
        if lib.ed_marker_gutter(
            editor._h, line, 0, *(ctypes.byref(f) for f in floats), ctypes.byref(glyph)
        ):
            found.append(line)
    return found


def test_the_cursor_band_is_re_pushed_after_a_whole_buffer_replace() -> None:
    editor = Editor("a\nb\nc\nd\n")
    editor.set_language(Language.GLSL)
    editor.set_cursor(1, 0)
    app = _Markers()
    path = Path("x.glsl")
    _apply_markers(app, editor, [], None, path, 1)
    assert _marked_lines(editor) == [1]

    lines = editor.get_text().split("\n")
    editor.set_selection((0, 0), (len(lines) - 1, len(lines[-1])))
    editor.replace_selection("A\nB\nC\nD\n")
    editor.set_cursor(1, 0)
    assert _marked_lines(editor) != [1], "the library moved the band with the edit"
    _apply_markers(app, editor, [], None, path, 1)
    assert _marked_lines(editor) == [1]
    editor.close()


def test_the_cursor_band_follows_dd_then_u() -> None:
    editor = Editor("a\nb\nc\n")
    editor.set_language(Language.GLSL)
    app = _Markers()
    path = Path("x.glsl")
    editor.feed("j")
    _apply_markers(app, editor, [], None, path, 1)
    editor.feed("dd")
    _apply_markers(
        app, editor, [], None, path, editor.get_current_cursor_position().line
    )
    editor.feed("u")
    line = editor.get_current_cursor_position().line
    _apply_markers(app, editor, [], None, path, line)
    assert _marked_lines(editor) == [line]
    editor.close()
