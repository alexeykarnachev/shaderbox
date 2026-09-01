"""Feature 067: the libeditor binding, the input translation, the drain gates,
and the redraw-gate domain. All headless — the .so needs no GL."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import glfw
import numpy as np
from imgui_bundle import imgui

from shaderbox.commands import COMMAND_SPECS, CommandId
from shaderbox.editor.ffi import Editor, KeyCode, KeyMod, Kind, Language, Mode
from shaderbox.editor.input import KeyEvent, translate_char, translate_key
from shaderbox.editor.render import PRIM_DTYPE, build_vertices, render_state, should_redraw
from shaderbox.editor_types import EditorSession
from shaderbox.hotkeys import _drain_editor_input
from shaderbox.shader_source import ShaderSource


def _editor(text: str = "one\ntwo\nthree\n") -> Editor:
    e = Editor(text)
    e.set_language(Language.GLSL)
    return e


# --- binding basics ---------------------------------------------------------


def test_revision_rises_across_set_text() -> None:
    e = _editor()
    r0 = e.get_undo_index()
    e.set_text("replaced")
    assert e.get_undo_index() > r0, (
        "ed_revision must rise across a whole-buffer replace — the dirty flag "
        "depends on it (ABI: Lifecycle and text)"
    )
    e.close()


def test_vim_edit_undo_redo() -> None:
    e = _editor()
    e.feed("dd")
    assert e.get_text() == "two\nthree\n"
    assert e.undo()
    assert e.get_text() == "one\ntwo\nthree\n"
    assert e.redo()
    assert e.get_text() == "two\nthree\n"
    e.close()


def test_only_two_ctrl_chords_are_consumed() -> None:
    # The collision domain decision 5 builds on: Ctrl+R (normal redo) and Ctrl+N
    # (insert completion) — every other Ctrl chord falls through as False.
    e = _editor()
    assert e.key(KeyCode.CHAR, KeyMod.CTRL, "r") is True
    for ch in "abcdefghijklmopqstuvwxyz":  # every letter except r/n
        assert e.key(KeyCode.CHAR, KeyMod.CTRL, ch) is False, f"Ctrl+{ch} claimed"
    e.feed("i")
    assert e.get_mode() == Mode.INSERT
    assert e.key(KeyCode.CHAR, KeyMod.CTRL, "n") is True
    e.close()


def test_read_only_blocks_user_not_host() -> None:
    e = _editor()
    e.set_read_only_enabled(True)
    e.feed("iX")
    assert "X" not in e.get_text(), "insert entry must be refused under read-only"
    e.set_text("host write")
    assert e.get_text() == "host write", "a host set_text is unaffected by read-only"
    e.close()


def test_replace_text_in_current_cursor_falls_back_to_caret() -> None:
    e = _editor("abc")
    e.replace_text_in_current_cursor("X")
    assert "X" in e.get_text()
    e.close()


# --- input translation ------------------------------------------------------


def test_translate_key_specials_and_chords() -> None:
    assert translate_key(glfw.KEY_ESCAPE, glfw.PRESS, 0) == KeyEvent(KeyCode.ESCAPE, 0)
    assert translate_key(glfw.KEY_LEFT, glfw.REPEAT, 0) == KeyEvent(KeyCode.LEFT, 0)
    assert translate_key(glfw.KEY_A, glfw.RELEASE, 0) is None
    # Plain printables arrive via the char callback, never the key callback.
    assert translate_key(glfw.KEY_A, glfw.PRESS, 0) is None
    # A special carrying a chord mod is the registry's, not the editor's.
    assert translate_key(glfw.KEY_TAB, glfw.PRESS, glfw.MOD_CONTROL) is None


def test_translate_key_ctrl_letter_synthesizes_char() -> None:
    event = translate_key(glfw.KEY_R, glfw.PRESS, glfw.MOD_CONTROL)
    assert event is not None
    assert event.code == KeyCode.CHAR
    assert event.text == "r"
    assert event.mods == KeyMod.CTRL
    # The chord crosses in the REGISTRY's comparison space: the same int
    # commands.py builds for its specs.
    assert event.imgui_chord == int(imgui.Key.r) | int(imgui.Key.mod_ctrl)


def test_translate_char() -> None:
    assert translate_char(ord("d")) == KeyEvent(KeyCode.CHAR, 0, "d")


# --- drain gates ------------------------------------------------------------


def _drain_app(editor: Editor, focused: bool = True, popup: bool = False) -> Any:
    session = EditorSession(
        editor=editor,
        source=ShaderSource(path=Path("/tmp/x.glsl"), text="", mtime=0.0),
        saved_undo=editor.get_undo_index(),
    )
    return SimpleNamespace(
        editor_consumed_chords=set(),
        editor_esc_forwarded=False,
        editor_key_events=[],
        editor_focused=focused,
        any_popup_open=lambda: popup,
        get_current_session_if_exists=lambda: session,
        window=None,
    )


def test_unfocused_editor_receives_nothing() -> None:
    # The bare-`d`-is-an-edit hazard: an unfocused editor must be deaf.
    e = _editor()
    app = _drain_app(e, focused=False)
    app.editor_key_events = [translate_char(ord("d")), translate_char(ord("d"))]
    _drain_editor_input(app)
    assert e.get_text() == "one\ntwo\nthree\n"
    assert app.editor_key_events == [], "the stale queue must not leak into next frame"
    e.close()


def test_focused_editor_consumes_and_records_chords() -> None:
    e = _editor()
    e.feed("dd")  # something to redo after undo
    e.undo()
    app = _drain_app(e)
    ctrl_r = translate_key(glfw.KEY_R, glfw.PRESS, glfw.MOD_CONTROL)
    assert ctrl_r is not None
    app.editor_key_events = [ctrl_r]
    _drain_editor_input(app)
    assert e.get_text() == "two\nthree\n", "Ctrl+R must redo"
    open_script_chord = next(
        int(imgui.Key.r) | int(imgui.Key.mod_ctrl)
        for s in COMMAND_SPECS
        if s.id == CommandId.OPEN_SCRIPT
    )
    assert open_script_chord in app.editor_consumed_chords, (
        "the consumed chord must be recorded in registry space — it is the ONLY "
        "guard against Ctrl+R double-dispatch (decision 5)"
    )
    e.close()


def test_esc_in_insert_is_forwarded_not_defocused() -> None:
    e = _editor()
    e.feed("i")
    app = _drain_app(e)
    app.editor_key_events = [KeyEvent(KeyCode.ESCAPE, 0)]
    _drain_editor_input(app)
    assert e.get_mode() == Mode.NORMAL, "Esc must leave insert mode"
    assert app.editor_esc_forwarded is True, (
        "the forward mark is what keeps _handle_escape's defocus branch quiet "
        "(single-consumer rule, decision 6)"
    )
    e.close()


def test_esc_with_pending_phrase_is_forwarded() -> None:
    e = _editor()
    e.feed("3")
    assert e.is_pending()
    app = _drain_app(e)
    app.editor_key_events = [KeyEvent(KeyCode.ESCAPE, 0)]
    _drain_editor_input(app)
    assert app.editor_esc_forwarded is True
    assert not e.is_pending()
    e.close()


def test_idle_esc_drops_the_queue_tail() -> None:
    # [Esc, 'd'] in one frame: the Esc defocuses, so the 'd' belongs to a
    # defocused editor and must never arrive.
    e = _editor()
    app = _drain_app(e)
    app.editor_key_events = [KeyEvent(KeyCode.ESCAPE, 0), translate_char(ord("d")), translate_char(ord("d"))]
    _drain_editor_input(app)
    assert app.editor_esc_forwarded is False, "idle-NORMAL Esc is the host's"
    assert e.get_text() == "one\ntwo\nthree\n", "the tail 'dd' must not edit"
    e.close()


def test_popup_open_blocks_the_drain() -> None:
    e = _editor()
    app = _drain_app(e, popup=True)
    app.editor_key_events = [translate_char(ord("x"))]
    _drain_editor_input(app)
    assert e.get_text() == "one\ntwo\nthree\n"
    e.close()


# --- redraw gate domain -----------------------------------------------------


def _state(e: Editor, **overrides: Any) -> tuple:
    kwargs: dict[str, Any] = {
        "size": (640, 480),
        "px_per_em": 16.0,
        "marker_fingerprint": (),
        "settings_fingerprint": (),
        "focused": True,
    }
    kwargs.update(overrides)
    return render_state(e, **kwargs)


def test_render_state_reacts_to_every_editor_dimension() -> None:
    # The checker-narrowing guard: every input the drawn frame depends on must
    # move the tuple. Walks each dimension with a real mutation.
    e = _editor()
    e.layout((640.0, 480.0), 16.0)
    base = _state(e)
    assert not should_redraw(base, _state(e)), "an untouched editor must not redraw"

    e.feed("j")  # cursor
    s = _state(e)
    assert should_redraw(base, s)
    base = s

    e.feed("v")  # mode + selection
    s = _state(e)
    assert should_redraw(base, s)
    e.key(KeyCode.ESCAPE)
    base = _state(e)

    e.feed("ix")  # revision
    s = _state(e)
    assert should_redraw(base, s)
    e.key(KeyCode.ESCAPE)
    base = _state(e)

    e.key(KeyCode.CHAR, 0, "/")  # command line opens
    s = _state(e)
    assert should_redraw(base, s)
    e.key(KeyCode.ESCAPE)
    base = _state(e)

    assert should_redraw(base, _state(e, size=(320, 480)))
    assert should_redraw(base, _state(e, px_per_em=18.0))
    assert should_redraw(base, _state(e, marker_fingerprint=((3, "boom"),)))
    assert should_redraw(base, _state(e, settings_fingerprint=(True,)))
    assert should_redraw(base, _state(e, focused=False))
    e.close()


def test_scroll_moves_the_state() -> None:
    e = _editor("\n".join(f"line {i}" for i in range(200)))
    e.layout((640.0, 100.0), 16.0)
    base = _state(e)
    e.set_scroll(5)
    assert should_redraw(base, _state(e))
    e.close()


# --- vertex building --------------------------------------------------------


def test_build_vertices_expands_quads_and_flags_glyphs() -> None:
    e = _editor("ab")
    n = e.layout((640.0, 480.0), 16.0)
    assert n > 0
    arr, count = e.prims_array()
    prims = np.frombuffer(arr, dtype=PRIM_DTYPE, count=count)
    verts = build_vertices(prims)
    assert len(verts) == count * 6
    glyph_rows = prims["kind"] == int(Kind.GLYPH)
    assert glyph_rows.sum() == 2, "two glyphs for 'ab'"
    textured = verts[:, 8].reshape(count, 6)
    for i in range(count):
        expected = 1.0 if glyph_rows[i] else 0.0
        assert (textured[i] == expected).all()
    e.close()
