"""Feature 067: the libeditor binding, the input translation, the drain gates,
and the redraw-gate domain. All headless — the .so needs no GL."""

import ast
import ctypes
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import glfw
import numpy as np
import pytest
from imgui_bundle import imgui

from shaderbox.commands import SPEC_BY_ID, CommandId
from shaderbox.editor.ffi import (
    _SIG,
    EDITOR_RESOURCES_DIR,
    ChromeFlag,
    Editor,
    KeyCode,
    KeyMod,
    Kind,
    Language,
    Mode,
    Prim,
    Slot,
)
from shaderbox.editor.input import KeyEvent, translate_char, translate_key
from shaderbox.editor.render import (
    PRIM_DTYPE,
    build_vertices,
    render_state,
    should_redraw,
)
from shaderbox.editor_types import EditorSession
from shaderbox.hotkeys import _delete_word_back, _drain_editor_input, spec_eligible
from shaderbox.shader_source import ShaderSource


@pytest.fixture(autouse=True)
def _fake_clipboard(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    # The drain syncs the OS clipboard with the editor register; tests run
    # windowless, so glfw's clipboard is faked with a dict.
    store: dict[str, str] = {"text": ""}
    monkeypatch.setattr(glfw, "get_clipboard_string", lambda _w: store["text"].encode())
    monkeypatch.setattr(
        glfw, "set_clipboard_string", lambda _w, t: store.__setitem__("text", t)
    )
    return store


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


def test_the_consumed_ctrl_chord_domain_is_eight() -> None:
    # The collision domain decision 5 builds on. Editor 4b110f0 grew it from two
    # to eight: Ctrl+R (redo) + the six scroll motions in normal mode, Ctrl+N in
    # insert. Every OTHER Ctrl chord falls through as False — the registry's food.
    e = _editor("\n".join(f"line {i}" for i in range(50)))
    e.layout((640.0, 420.0), 16.0)
    consumed_normal = "rdufbey"
    for ch in consumed_normal:
        assert e.key(KeyCode.CHAR, KeyMod.CTRL, ch) is True, f"Ctrl+{ch} unbound?"
    for ch in "acghijklmnopqstvwxz":
        assert e.key(KeyCode.CHAR, KeyMod.CTRL, ch) is False, f"Ctrl+{ch} claimed"
    e.feed("i")
    assert e.get_mode() == Mode.INSERT
    assert e.key(KeyCode.CHAR, KeyMod.CTRL, "n") is True
    # The six scrolls are normal/visual motions; in insert they stay unbound
    # (the drain's insert-protection fallback depends on it).
    assert e.key(KeyCode.CHAR, KeyMod.CTRL, "d") is False
    e.close()


def test_read_only_blocks_user_not_host() -> None:
    e = _editor()
    e.set_read_only_enabled(True)
    e.feed("iX")
    assert "X" not in e.get_text(), "insert entry must be refused under read-only"
    e.set_text("host write")
    assert e.get_text() == "host write", "a host set_text is unaffected by read-only"
    e.close()


def test_read_only_refuses_every_editing_key_not_just_insert() -> None:
    # The library decided what read-only refuses from a hand-written list of editing keys,
    # so a key added to the keymap later reached the buffer through a locked editor. `>` and
    # `<` had drifted out of it and DID edit under read-only at the sha this repo shipped
    # before f738744 (measured: `>>` indented a locked line). tabs/code.py locks the editor
    # for the whole copilot turn, so that was a live path to a buffer the host believed
    # frozen. One key per shape rather than one key: the point is the CLASS.
    text = "  alpha beta\n"
    for keys in ("~", ">>", "<<", "RXY", "x", "dd", "J", "S"):
        e = _editor(text)
        e.set_read_only_enabled(True)
        e.set_cursor(
            0, 2
        )  # off the leading whitespace, or ~ is a no-op and proves nothing
        e.feed(keys)
        assert e.get_text() == text, (
            f"{keys!r} edited a read-only buffer: {e.get_text()!r}"
        )
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
        editor_key_events=[],
        editor_focused=focused,
        any_popup_open=lambda: popup,
        get_current_session_if_exists=lambda: session,
        window=None,
        copilot_turn_active=False,
        editor_completion_requested=False,
        editor_visible_rows=20,
        save=lambda: None,
        flush_current_editor=None,
        notifications=SimpleNamespace(push=lambda *_a, **_k: None),
        app_state=SimpleNamespace(editor_settings=SimpleNamespace(keymap="vim")),
        editor_tabs=[],
        active_tab_index=0,
        close_tab=lambda _i: None,
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


def test_a_reserved_chord_the_editor_consumes_is_recorded_in_registry_space() -> None:
    # Ctrl+R redoes and is recorded, but after 069 W-E no command owns Ctrl+R (OPEN_SCRIPT is
    # Alt+R), so this half no longer exercises the double-dispatch guard.
    e = _editor()
    e.feed("dd")  # something to redo after undo
    e.undo()
    app = _drain_app(e)
    ctrl_r = translate_key(glfw.KEY_R, glfw.PRESS, glfw.MOD_CONTROL)
    assert ctrl_r is not None
    app.editor_key_events = [ctrl_r]
    _drain_editor_input(app)
    assert e.get_text() == "two\nthree\n", "Ctrl+R must redo"
    assert int(imgui.Key.r) | int(imgui.Key.mod_ctrl) in app.editor_consumed_chords


def test_insert_ctrl_w_strikes_the_command_that_owns_the_chord() -> None:
    # The live instance of the double-dispatch guard (067 D15): insert-mode Ctrl+W is the
    # host's word-delete AND Ctrl+W is CLOSE_CODE_TAB's chord, so the consumed-set entry is
    # the only thing stopping one press from both deleting a word and closing the tab. The
    # chord is READ FROM THE SPEC, so a future move breaks this test instead of passing
    # against a hand-built int the filter never constrained.
    spec = SPEC_BY_ID[CommandId.CLOSE_CODE_TAB]
    e = Editor("hello world foo")
    e.set_language(Language.GLSL)
    e.feed("A")  # append: insert mode, caret at end of line
    app = _drain_app(e)
    ctrl_w = translate_key(glfw.KEY_W, glfw.PRESS, glfw.MOD_CONTROL)
    assert ctrl_w is not None
    app.editor_key_events = [ctrl_w]
    _drain_editor_input(app)
    assert e.get_text() == "hello world ", (
        "insert-mode Ctrl+W must delete the word back"
    )
    assert spec.default_chord in app.editor_consumed_chords, (
        "the consumed chord must be recorded in registry space, or the same press also "
        "fires CLOSE_CODE_TAB"
    )
    e.close()


def test_esc_in_insert_leaves_insert_mode() -> None:
    e = _editor()
    e.feed("i")
    app = _drain_app(e)
    app.editor_key_events = [KeyEvent(KeyCode.ESCAPE, 0)]
    _drain_editor_input(app)
    assert e.get_mode() == Mode.NORMAL, "Esc must leave insert mode"
    e.close()


def test_esc_with_pending_phrase_is_forwarded() -> None:
    e = _editor()
    e.feed("3")
    assert e.is_pending()
    app = _drain_app(e)
    app.editor_key_events = [KeyEvent(KeyCode.ESCAPE, 0)]
    _drain_editor_input(app)
    assert not e.is_pending()
    e.close()


def test_popup_open_blocks_the_drain() -> None:
    e = _editor()
    app = _drain_app(e, popup=True)
    app.editor_key_events = [translate_char(ord("x"))]
    _drain_editor_input(app)
    assert e.get_text() == "one\ntwo\nthree\n"
    e.close()


def test_esc_never_defocuses_the_editor() -> None:
    # Esc is vim's modal key: the editor owns it UNCONDITIONALLY while focused
    # (maintainer decision, 067 manual pass) — idle-NORMAL Esc included, and the
    # queue keeps flowing: [Esc, d, d] still deletes a line.
    e = _editor()
    app = _drain_app(e)
    app.editor_key_events = [
        KeyEvent(KeyCode.ESCAPE, 0),
        translate_char(ord("d")),
        translate_char(ord("d")),
    ]
    _drain_editor_input(app)
    assert e.get_text() == "two\nthree\n", "the dd after Esc still edits — no defocus"
    e.close()


def _ctrl(ch: str) -> KeyEvent:
    event = translate_key(
        glfw.KEY_A + (ord(ch) - ord("a")), glfw.PRESS, glfw.MOD_CONTROL
    )
    assert event is not None
    return event


def test_ctrl_d_moves_the_cursor_and_suppresses_delete_document() -> None:
    # Editor 4b110f0: Ctrl+D is a real keymap MOTION (cursor moves half a page,
    # nvim-measured); the consumed chord still guards DELETE_DOCUMENT.
    e = _editor("\n".join(f"line {i}" for i in range(200)))
    e.layout((640.0, 420.0), 16.0)
    app = _drain_app(e)
    app.editor_key_events = [_ctrl("d")]
    _drain_editor_input(app)
    assert e.get_current_cursor_position().line == 10, "half a 20-row page down"
    spec = SPEC_BY_ID[CommandId.DELETE_DOCUMENT]
    chord = int(imgui.Key.d) | int(imgui.Key.mod_ctrl)
    assert spec_eligible(app, spec, chord, popup_open=False) is False, (
        "the motion consumed the chord — DELETE_DOCUMENT must not also fire"
    )
    app.editor_key_events = [_ctrl("u")]
    _drain_editor_input(app)
    assert e.get_current_cursor_position().line == 0, "Ctrl+U back up"
    e.close()


def test_follow_the_cursor_brings_the_view_along() -> None:
    # ed_layout clamps but never follows: without host follow logic a Ctrl+D
    # (or a bare j below the fold) walks the caret off-screen invisibly.
    e = _editor("\n".join(f"line {i}" for i in range(200)))
    e.layout((640.0, 420.0), 16.0)
    for _ in range(3):
        e.key(KeyCode.CHAR, KeyMod.CTRL, "d")
    cursor = e.get_current_cursor_position()
    assert cursor.line == 30
    assert e.get_scroll() == 0, "the layout alone must NOT have followed (premise)"
    e.scroll_to_line(cursor.line, align_middle=False)
    e.layout((640.0, 420.0), 16.0)
    first = e.get_scroll()
    assert first <= cursor.line < first + 20, "the host follow puts the caret in view"
    e.close()


def test_ctrl_f_b_move_cursor_and_e_y_request_view_scroll() -> None:
    e = _editor("\n".join(f"line {i}" for i in range(200)))
    e.layout((640.0, 420.0), 16.0)
    app = _drain_app(e)
    app.editor_key_events = [_ctrl("f")]
    _drain_editor_input(app)
    # nvim reports "line 19" 1-based at a 20-row window; our columns/lines are 0-based.
    assert e.get_current_cursor_position().line == 18, "Ctrl+F pages the cursor (nvim)"
    app.editor_key_events = [_ctrl("b")]
    _drain_editor_input(app)
    assert e.get_current_cursor_position().line == 0
    # Ctrl+E/Y are VIEW-only: reading the request consumes it, returns the
    # ABSOLUTE target row, and applies the scroll (measured contract —
    # code.draw's whole job is one take per frame).
    e.set_scroll(5)
    e.layout((640.0, 420.0), 16.0)
    app.editor_key_events = [_ctrl("e")]
    _drain_editor_input(app)
    assert e.take_scroll_request() == 6, "Ctrl+E: one row down, absolute"
    assert e.get_scroll() == 6, "the read applied it"
    assert e.take_scroll_request() is None, "consumed by reading"
    e.layout((640.0, 420.0), 16.0)
    app.editor_key_events = [_ctrl("y")]
    _drain_editor_input(app)
    assert e.take_scroll_request() == 5, "Ctrl+Y: one row up, absolute"
    e.close()


def test_ctrl_o_reaches_the_app_while_focused() -> None:
    # 069 W-E: Ctrl+O is in NEITHER keymap's chord list, so the ownership rule gives it to
    # OPEN_PROJECT in all three states; the host consuming it would be inventing a binding.
    e = _editor()
    app = _drain_app(e)
    app.editor_key_events = [_ctrl("o")]
    _drain_editor_input(app)
    assert e.get_text() == "one\ntwo\nthree\n"
    assert (
        int(imgui.Key.o) | int(imgui.Key.mod_ctrl)
    ) not in app.editor_consumed_chords, (
        "Ctrl+O belongs to no keymap; the host must not swallow it"
    )
    e.close()


def test_insert_ctrl_w_deletes_word_back_not_the_tab() -> None:
    e = Editor("hello world")
    e.set_language(Language.GLSL)
    e.feed("A")  # append: insert mode, caret at end of line
    app = _drain_app(e)
    app.editor_key_events = [_ctrl("w")]
    _drain_editor_input(app)
    assert e.get_text() == "hello ", "Ctrl+W deleted one word back"
    assert e.get_mode() == Mode.INSERT, "still typing — the tab did not close"
    e.close()


def test_insert_ctrl_u_deletes_to_line_start() -> None:
    e = Editor("hello world")
    e.set_language(Language.GLSL)
    e.feed("A")
    app = _drain_app(e)
    app.editor_key_events = [_ctrl("u")]
    _drain_editor_input(app)
    assert e.get_text() == "", "Ctrl+U in insert deletes to line start"
    assert e.get_mode() == Mode.INSERT
    e.close()


def _type_ex(app: Any, text: str) -> None:
    app.editor_key_events = [translate_char(ord(c)) for c in text] + [
        KeyEvent(KeyCode.ENTER, 0)
    ]
    _drain_editor_input(app)


def test_colon_w_saves_through_the_host_command_surface() -> None:
    # The parser validates the spelling and hands the host the result (editor
    # c5fabc8) — :w saves, and force/argument spellings arrive typed.
    e = _editor()
    app = _drain_app(e)
    saved: list[bool] = []
    app.save = lambda: saved.append(True)
    _type_ex(app, ":w")
    assert saved, ":w must save through App.save (disk + busy gate)"
    assert e.get_command_line() is None, "the command line closed"
    assert e.get_text() == "one\ntwo\nthree\n", "no Enter leaked into the buffer"
    e.close()


def test_colon_q_refuses_dirty_and_bang_discards(tmp_path: Path) -> None:
    disk = tmp_path / "x.glsl"
    disk.write_text("one\ntwo\nthree\n")
    e = _editor()
    app = _drain_app(e)
    app.get_current_session_if_exists().source = ShaderSource(
        path=disk, text="one\ntwo\nthree\n", mtime=0.0
    )
    closed: list[int] = []
    app.editor_tabs = [object()]
    app.close_tab = lambda i: closed.append(i)
    notes: list[str] = []
    app.notifications = SimpleNamespace(push=lambda t, **_k: notes.append(t))
    e.feed("dd")  # dirty
    _type_ex(app, ":q")
    assert not closed, ":q with unsaved changes refuses, as vim does"
    assert any("Unsaved" in n for n in notes)
    _type_ex(app, ":q!")
    assert closed, ":q! closes"
    assert e.get_text() == "one\ntwo\nthree\n", ":q! discarded the edit from disk"
    e.close()


# --- redraw gate domain -----------------------------------------------------


def _state(e: Editor, **overrides: Any) -> tuple:
    kwargs: dict[str, Any] = {
        "identity": Path("/tmp/x.glsl"),
        "size": (640, 480),
        "px_per_em": 16.0,
        "text_origin": (0.0, 0.0),
        "completion_prefix": None,
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

    # The command MESSAGE: the library draws it into the status row, and the
    # line itself is closed again by the time the frame is gated.
    e.feed(":zzz<CR>")
    s = _state(e)
    assert e.get_command_line() is None
    assert should_redraw(base, s), "a failed ex command must repaint the status row"
    base = s

    # The command PROMPT rune: the bar text is prompt + input, so two different
    # prompts over the same input paint different glyphs.
    e.key(KeyCode.CHAR, 0, ":")
    e.key(KeyCode.CHAR, 0, "a")
    colon = _state(e)
    e.key(KeyCode.ESCAPE)
    e.key(KeyCode.CHAR, 0, "/")
    e.key(KeyCode.CHAR, 0, "a")
    slash = _state(e)
    assert e.get_command_line() == "a"
    assert should_redraw(colon, slash), "':a' and '/a' paint different bars"
    e.key(KeyCode.ESCAPE)
    base = _state(e)

    assert should_redraw(base, _state(e, size=(320, 480)))
    assert should_redraw(base, _state(e, px_per_em=18.0))
    assert should_redraw(base, _state(e, text_origin=(40.0, 0.0)))
    assert should_redraw(base, _state(e, completion_prefix="SB_"))
    assert should_redraw(base, _state(e, marker_fingerprint=((3, "boom"),)))
    assert should_redraw(base, _state(e, settings_fingerprint=(True,)))
    assert should_redraw(base, _state(e, focused=False))
    e.close()


def test_render_state_distinguishes_two_fresh_editors() -> None:
    # Tabs share ONE panel. Two freshly opened files agree on revision, cursor,
    # mode and everything else the editor reports — identity is what keeps a tab
    # switch from showing the previous file's texture (post-impl BLOCKER 1).
    a, b = _editor("file a"), _editor("file b")
    sa = _state(a, identity=Path("/a.glsl"))
    sb = _state(b, identity=Path("/b.glsl"))
    assert should_redraw(sa, sb)
    # The falsifier: without identity the two states really would collide.
    assert sa[1:] == sb[1:]
    a.close()
    b.close()


def test_scroll_moves_the_state() -> None:
    e = _editor("\n".join(f"line {i}" for i in range(200)))
    e.layout((640.0, 100.0), 16.0)
    base = _state(e)
    e.set_scroll(5)
    assert should_redraw(base, _state(e))
    e.close()


# --- host-driven completion (feature 067, editor commit f57e8d0) --------------


def test_completion_flow_open_navigate_accept() -> None:
    e = _editor("")
    e.set_host_completion(True)
    e.feed("iSB_n")
    assert e.complete_prefix() == "SB_n"
    assert not e.complete_open(), "host_completion: nothing opens until the host pushes"
    e.complete_begin()
    e.complete_push("SB_noise")
    e.complete_push("SB_normal")
    assert e.complete_open(), "pushing IS opening"
    assert e.complete_count() == 2
    first = e.complete_selected()
    e.key(KeyCode.DOWN)
    assert e.complete_selected() != first, "Down must move the selection"
    e.key(KeyCode.ENTER)
    assert "SB_normal" in e.get_text() or "SB_noise" in e.get_text()
    assert e.get_mode() == Mode.INSERT
    e.close()


def test_ctrl_n_with_host_completion_consumes_but_opens_nothing() -> None:
    # The drain's detection contract: Ctrl+N stays consumed (so the chord skip
    # still suppresses NEW_DOCUMENT) while the popup stays closed until we push.
    e = _editor("")
    e.set_host_completion(True)
    e.feed("i")
    assert e.key(KeyCode.CHAR, KeyMod.CTRL, "n") is True
    assert not e.complete_open()
    e.close()


def test_complete_cancel_stays_in_insert() -> None:
    e = _editor("")
    e.set_host_completion(True)
    e.feed("iSB_n")
    e.complete_begin()
    e.complete_push("SB_noise")
    assert e.complete_open()
    e.complete_cancel()
    assert not e.complete_open()
    assert e.get_mode() == Mode.INSERT, "cancel must not act as Escape"
    e.close()


def test_completion_state_moves_the_redraw_tuple() -> None:
    e = _editor("")
    e.feed("iSB_n")
    base = _state(e)
    e.complete_begin()
    e.complete_push("SB_noise")
    e.complete_push("SB_normal")
    opened = _state(e)
    assert should_redraw(base, opened), "an opening popup must repaint"
    e.key(KeyCode.DOWN)
    assert should_redraw(opened, _state(e)), "moving the selection must repaint"
    e.close()


# --- register-clipboard unification (editor commit 4befeaf) -------------------


def test_yank_reaches_the_system_clipboard(_fake_clipboard: dict[str, str]) -> None:
    e = _editor()
    app = _drain_app(e)
    app.editor_key_events = [translate_char(ord("y")), translate_char(ord("y"))]
    _drain_editor_input(app)
    assert _fake_clipboard["text"] == "one\n", (
        "a yy must land in the OS clipboard — one slot, not two clipboards"
    )
    e.close()


def test_p_pastes_the_system_clipboard(_fake_clipboard: dict[str, str]) -> None:
    _fake_clipboard["text"] = "FROM_OUTSIDE"
    e = _editor()
    app = _drain_app(e)
    app.editor_key_events = [translate_char(ord("p"))]
    _drain_editor_input(app)
    assert "FROM_OUTSIDE" in e.get_text(), (
        "p must paste what the user copied elsewhere (sync-in precedes the keys)"
    )
    e.close()


def test_dd_then_p_round_trips() -> None:
    e = _editor()
    e.feed("ddp")
    assert e.get_text() == "two\none\nthree\n", "dd+p moves the line down (vim)"
    e.close()


# --- registry eligibility (the consumed-chord skip) --------------------------


def test_consumed_chord_suppresses_the_registry_spec() -> None:
    spec = SPEC_BY_ID[CommandId.OPEN_SCRIPT]
    chord = int(imgui.Key.r) | int(imgui.Key.mod_ctrl)
    app = SimpleNamespace(
        editor_consumed_chords={chord},
        editor_focused=True,
        copilot_focused=False,
    )
    assert spec_eligible(app, spec, chord, popup_open=False) is False, (
        "a chord the editor consumed must not also dispatch OPEN_SCRIPT"
    )
    app.editor_consumed_chords = set()
    assert spec_eligible(app, spec, chord, popup_open=False) is True


# --- text getters -----------------------------------------------------------


def test_get_text_grows_past_the_scratch_buffer() -> None:
    # A >1 MiB buffer must round-trip, not save truncated (post-impl MINOR 8).
    big = "x" * (1 << 21)
    e = Editor(big)
    assert len(e.get_text()) == len(big)
    e.close()


def test_key_refuses_multichar_text() -> None:
    e = _editor()
    assert e.key(KeyCode.CHAR, 0, "ab") is False
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


# --- the sweep round's added coverage (067 review workflow) -------------------


def test_cross_tab_yank_pastes_via_the_clipboard_bridge(
    _fake_clipboard: dict[str, str],
) -> None:
    # Registers are per-handle; the drain seeds the ACTIVE tab's register from
    # the clipboard by comparing against that register (an app-global "seen"
    # left tab B unseeded and a cross-tab yy/p pasted nothing).
    a, b = _editor("alpha\n"), _editor("one\ntwo\n")
    app_a = _drain_app(a)
    app_a.editor_key_events = [translate_char(ord("y")), translate_char(ord("y"))]
    _drain_editor_input(app_a)
    assert _fake_clipboard["text"] == "alpha\n"
    app_b = _drain_app(b)
    app_b.editor_key_events = [translate_char(ord("p"))]
    _drain_editor_input(app_b)
    assert "alpha" in b.get_text(), "the yank from tab A pastes in tab B"
    a.close()
    b.close()


def test_insert_ctrl_r_is_reserved_not_open_script() -> None:
    e = _editor()
    e.feed("i")
    app = _drain_app(e)
    app.editor_key_events = [_ctrl("r")]
    _drain_editor_input(app)
    assert e.get_text() == "one\ntwo\nthree\n"
    assert (int(imgui.Key.r) | int(imgui.Key.mod_ctrl)) in app.editor_consumed_chords, (
        "insert-mode Ctrl+R must not fire OPEN_SCRIPT mid-typing"
    )
    e.close()


def test_visual_mode_scroll_chords_are_reserved() -> None:
    # The keymap consumes the six scrolls in NORMAL mode only; in VISUAL they
    # fell through to the registry (Ctrl+D = delete document, from visual mode).
    e = _editor("\n".join(f"line {i}" for i in range(50)))
    e.layout((640.0, 420.0), 16.0)
    e.feed("v")
    app = _drain_app(e)
    app.editor_key_events = [_ctrl("d")]
    _drain_editor_input(app)
    assert (int(imgui.Key.d) | int(imgui.Key.mod_ctrl)) in app.editor_consumed_chords
    assert e.get_mode() == Mode.VISUAL, "still in visual — nothing app-side fired"
    e.close()


def test_normal_ctrl_n_p_j_h_are_vim_motions() -> None:
    e = _editor("\n".join(f"line {i}" for i in range(10)))
    e.layout((640.0, 420.0), 16.0)
    app = _drain_app(e)
    for ch, expect in (("n", 1), ("j", 2), ("p", 1), ("h", 1)):
        app.editor_key_events = [_ctrl(ch)]
        _drain_editor_input(app)
        assert e.get_current_cursor_position().line == expect, f"Ctrl+{ch}"
    e.close()


def test_insert_ctrl_h_backspaces_and_ctrl_j_breaks_the_line() -> None:
    e = Editor("ab")
    e.set_language(Language.GLSL)
    e.feed("A")
    app = _drain_app(e)
    app.editor_key_events = [_ctrl("h")]
    _drain_editor_input(app)
    assert e.get_text() == "a", "insert Ctrl+H = backspace"
    app.editor_key_events = [_ctrl("j")]
    _drain_editor_input(app)
    assert e.get_text() == "a\n", "insert Ctrl+J = newline"
    e.close()


def test_ctrl_left_bracket_is_escape() -> None:
    event = translate_key(glfw.KEY_LEFT_BRACKET, glfw.PRESS, glfw.MOD_CONTROL)
    assert event == KeyEvent(KeyCode.ESCAPE, 0), "vim's second Esc"
    e = _editor()
    e.feed("i")
    app = _drain_app(e)
    app.editor_key_events = [event]
    _drain_editor_input(app)
    assert e.get_mode() == Mode.NORMAL
    e.close()


def test_repeated_ctrl_n_advances_the_completion_selection() -> None:
    e = _editor("")
    e.set_host_completion(True)
    e.feed("iSB_")
    e.complete_begin()
    for cand in ("SB_a", "SB_b", "SB_c"):
        e.complete_push(cand)
    assert e.complete_open()
    first = e.complete_selected()
    app = _drain_app(e)
    app.editor_key_events = [_ctrl("n")]
    _drain_editor_input(app)
    assert e.complete_selected() != first, (
        "Ctrl+N with the popup open advances instead of re-offering from zero"
    )
    assert app.editor_completion_requested is False, "no re-offer was queued"
    e.close()


def test_spec_eligible_scope_and_popup_gates() -> None:
    editor_spec = SPEC_BY_ID[CommandId.CLOSE_CODE_TAB]  # scope EDITOR
    copilot_spec = SPEC_BY_ID[CommandId.CYCLE_COPILOT_LAYOUT]  # scope COPILOT
    global_spec = SPEC_BY_ID[CommandId.SAVE]
    app = SimpleNamespace(
        editor_consumed_chords=set(), editor_focused=False, copilot_focused=False
    )
    assert spec_eligible(app, editor_spec, 1, popup_open=False) is False
    assert spec_eligible(app, copilot_spec, 1, popup_open=False) is False
    assert spec_eligible(app, global_spec, 1, popup_open=True) is False, (
        "a modal suppresses every scope"
    )
    app.editor_focused = True
    app.copilot_focused = True
    assert spec_eligible(app, editor_spec, 1, popup_open=False) is True
    assert spec_eligible(app, copilot_spec, 1, popup_open=False) is True


def test_delete_word_back_whitespace_and_punct_runs() -> None:
    e = Editor("word   ")
    e.set_language(Language.GLSL)
    e.feed("A")
    _delete_word_back(e)
    assert e.get_text() == "", "trailing whitespace run + the word both go"
    e.close()
    e = Editor("a ==")
    e.set_language(Language.GLSL)
    e.feed("A")
    _delete_word_back(e)
    assert e.get_text() == "a ", "a punctuation run deletes as one unit"
    e.close()


def test_translate_key_carries_shift_and_alt_mods() -> None:
    event = translate_key(glfw.KEY_R, glfw.PRESS, glfw.MOD_CONTROL | glfw.MOD_SHIFT)
    assert event is not None
    assert event.mods == KeyMod.CTRL | KeyMod.SHIFT
    assert event.text == "R", "shift resolves the synthesized char's case"
    alt = translate_key(glfw.KEY_X, glfw.PRESS, glfw.MOD_ALT)
    assert alt is not None
    assert alt.mods == KeyMod.ALT


def test_render_state_members_with_isolated_pairs() -> None:
    # Checker-domain guard: for each member a co-varying sibling could mask,
    # a state pair differing in (nearly) only that member.
    e = _editor("abc")
    e.layout((640.0, 480.0), 16.0)
    base = _state(e)
    e.set_text("abc")  # same text, same cursor (0,0) — ONLY revision moves
    only_revision = _state(e)
    assert should_redraw(base, only_revision)
    e2 = _editor("abc")
    b2 = _state(e2)
    e2.feed("i")  # cursor stays (0,0), revision untouched — ONLY mode moves
    assert should_redraw(b2, _state(e2))
    e2.key(KeyCode.ESCAPE)
    b3 = _state(e2)
    e2.set_selection((0, 0), (0, 2))  # host selection: revision untouched
    assert should_redraw(b3, _state(e2))
    e.close()
    e2.close()


def test_drain_clears_the_consumed_set_each_frame() -> None:
    e = _editor()
    app = _drain_app(e)
    app.editor_consumed_chords = {12345}
    app.editor_key_events = [translate_char(ord("j"))]
    _drain_editor_input(app)
    assert 12345 not in app.editor_consumed_chords, (
        "a stale chord from last frame would suppress a live command forever"
    )
    e.close()


def _upstream_sig(path: Path) -> dict[str, tuple[object, list[object]]]:
    # Parsed, never imported: the vendored probe opens a session at import time.
    # Upstream's OWN `Prim` is exec'd out of the same AST rather than ours being
    # substituted: POINTER memoizes per type object, so lending them our class
    # would make the two POINTER(Prim) entries compare against themselves and a
    # diverged struct — a stride mismatch across the whole array — read as equal.
    tree = ast.parse(path.read_text())
    namespace: dict[str, Any] = {"ctypes": ctypes}
    prim_def = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Prim"
    )
    exec(compile(ast.Module([prim_def], []), str(path), "exec"), namespace)
    their_prim: type[ctypes.Structure] = namespace["Prim"]
    assert ctypes.sizeof(their_prim) == ctypes.sizeof(Prim), (
        f"vendored Prim is {ctypes.sizeof(their_prim)} bytes, "
        f"ours is {ctypes.sizeof(Prim)} — the primitive stride diverged"
    )
    node = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "_SIG" for t in n.targets)
    )
    assert isinstance(node, ast.Dict)
    out: dict[str, tuple[object, list[object]]] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        assert key is not None
        restype, argtypes = eval(
            compile(ast.Expression(value), str(path), "eval"), namespace
        )
        out[ast.literal_eval(key)] = (
            _normalise(restype, their_prim),
            [_normalise(a, their_prim) for a in argtypes],
        )
    return out


def _normalise(ctype: object, prim: type[ctypes.Structure]) -> object:
    """POINTER(Prim) is a different object per Prim class, so the two tables can
    only be compared once each side's pointer-to-Prim is named the same thing.
    The struct itself is compared by size in `_upstream_sig`."""
    return "POINTER(Prim)" if ctype is ctypes.POINTER(prim) else ctype


def test_the_binding_mirrors_every_export_of_the_vendored_binary() -> None:
    # The mirror rule, in both directions: an unbound export is the rule's own
    # violation, a bound-but-absent name is the re-vendor regression, named here
    # instead of surfacing as an AttributeError out of the binding loop.
    lib_path = EDITOR_RESOURCES_DIR / "libeditor.so"
    if not lib_path.exists() or shutil.which("nm") is None:
        pytest.skip("vendored binary or nm unavailable")
    out = subprocess.run(
        ["nm", "-D", "--defined-only", str(lib_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    exported = {
        parts[2]
        for line in out.splitlines()
        if len(parts := line.split()) == 3
        and parts[1] == "T"
        and parts[2].startswith("ed_")
    }
    assert exported == set(_SIG), (
        f"unbound: {sorted(exported - set(_SIG))}; "
        f"bound but absent: {sorted(set(_SIG) - exported)}"
    )


def test_the_binding_mirrors_the_upstream_signature_table() -> None:
    # Names are the easy half. A short or wrong argtype is silent: ctypes pushes
    # what the binding declares and the callee reads its tail off the stack.
    probe = EDITOR_RESOURCES_DIR / "abi_probe.py"
    if not probe.exists():
        pytest.skip("vendored abi_probe.py unavailable")
    upstream = _upstream_sig(probe)
    ours = {
        name: (
            _normalise(restype, Prim),
            [_normalise(a, Prim) for a in argtypes],
        )
        for name, (restype, argtypes) in _SIG.items()
    }
    assert ours == upstream


def test_the_mode_enum_covers_every_value_upstream_can_return() -> None:
    # The Mode IntEnum is constructed from a raw int by `get_mode`, and an IntEnum RAISES
    # on a value it lacks -- so a mode appended upstream is a CRASH here the first time a
    # user reaches it, not a stale reading. Anchored to the vendored probe's own MODES
    # table rather than to a number written here, so the next re-vendor fails this test
    # instead of shipping a binding that throws.
    probe = EDITOR_RESOURCES_DIR / "abi_probe.py"
    if not probe.exists():
        pytest.skip("vendored abi_probe.py unavailable")
    for line in probe.read_text(encoding="utf-8").splitlines():
        if line.startswith("MODES = "):
            upstream_modes = ast.literal_eval(line.split("=", 1)[1].strip())
            break
    else:
        pytest.fail("the vendored abi_probe.py no longer declares a MODES table")
    assert len(Mode) >= len(upstream_modes), (
        f"upstream enumerates {len(upstream_modes)} modes {upstream_modes}, the binding "
        f"declares {len(Mode)}: {[m.name for m in Mode]}. Add the missing member -- "
        "get_mode() raises ValueError on a value the enum lacks."
    )
    for value in range(len(upstream_modes)):
        Mode(value)  # constructs, or ValueError names the gap


def test_a_marker_follows_a_line_inserted_above_it() -> None:
    # Markers anchor like nvim extmarks, so an error band tracks its code
    # between compiles instead of pointing at the line the code used to be on.
    e = _editor("\n".join(f"line {i}" for i in range(20)))
    e.add_marker(9, fill=(1.0, 0.0, 0.0, 0.2))
    e.set_cursor(6, 0)
    e.feed("O")
    e.feed("x")
    e.key(KeyCode.ESCAPE)
    marked = [line for line in range(25) if e.get_marker_gutter(line) is not None]
    assert marked == [10], f"marker should have moved down to 10, found {marked}"
    e.close()


def test_draw_chrome_adds_a_gutter_and_a_status_frame() -> None:
    # A GLYPH-count comparison is NOT a valid falsifier here: the gutter narrows
    # the text viewport, so a wide buffer emits FEWER glyphs under chrome.
    e = _editor("\n".join(f"line {i}" for i in range(20)))
    e.set_chrome_flag(ChromeFlag.LINE_NUMBERS, True)
    e.layout((600.0, 300.0), 16.0)
    off = [p.kind for p in e.prims_list()]
    assert int(Kind.FRAME) not in off
    assert e.get_text_origin()[0] == 0.0

    e.set_draw_chrome(True)
    e.layout((600.0, 300.0), 16.0)
    on = [p.kind for p in e.prims_list()]
    assert on.count(int(Kind.FRAME)) == 1, "the status row is one Frame"
    assert on.count(int(Kind.POPUP_GLYPH)) > 0, "the status row draws its text"
    cell_w, _cell_h = e.get_cell_size()
    text_x = e.get_text_origin()[0]
    assert text_x > 0.0
    assert text_x == e.get_gutter_cells() * cell_w
    e.close()


def test_text_origin_moves_right_by_the_gutter_under_chrome() -> None:
    # The host passes the WHOLE widget to ed_layout now; double-offsetting by a
    # host-computed gutter would break this identity.
    e = _editor("\n".join(f"line {i}" for i in range(20)))
    e.set_chrome_flag(ChromeFlag.LINE_NUMBERS, True)
    e.layout((600.0, 300.0), 16.0)
    assert e.get_text_origin() == (0.0, 0.0)
    e.set_draw_chrome(True)
    e.layout((600.0, 300.0), 16.0)
    cell_w, _cell_h = e.get_cell_size()
    x, _y = e.get_text_origin()
    assert x > 0.0
    assert x == e.get_gutter_cells() * cell_w
    e.close()


def test_a_marker_text_color_reaches_the_glyph_at_column_0() -> None:
    # Finding #14's own repro: a keyword in the first cell of an error line.
    # STATE_ERROR and SYN_KEYWORD are one palette entry, so a column-0 glyph
    # left at its syntax color is the exact red-on-red case the override fixes.
    # The caret sits on the second line: the glyph under a normal-mode caret is recolored
    # AFTER markers (reverse video), so it would mask the very fact this test pins.
    marker_rgb = (0.92, 0.86, 0.70)
    e = _editor("vec3 c = fn(x);\nx")
    e.set_language(Language.GLSL)
    e.feed("j")
    e.set_chrome_flag(ChromeFlag.LINE_NUMBERS, True)
    e.set_draw_chrome(True)
    e.add_marker(0, fill=(0.8, 0.1, 0.1, 0.2), text=(*marker_rgb, 1.0))
    e.layout((600.0, 300.0), 16.0)
    origin_x, origin_y = e.get_text_origin()
    first_row_bottom = origin_y + e.get_cell_size()[1]
    row = sorted(
        (p.x0, (round(p.r, 2), round(p.g, 2), round(p.b, 2)))
        for p in e.prims_list()
        if p.kind == int(Kind.GLYPH) and p.x1 > origin_x and p.y0 < first_row_bottom
    )
    assert len(row) == len("vec3 c = fn(x);".replace(" ", ""))
    # The first glyph's ink overhangs its cell to the left, past the origin —
    # which is what made a left-edge test skip it.
    assert row[0][0] < origin_x
    assert {color for _, color in row} == {marker_rgb}, (
        f"column 0 kept its syntax color: {row[0]}"
    )
    e.close()


def test_the_status_band_sits_below_the_interactive_height() -> None:
    # The host shrinks its invisible_button by one cell so clicks and hovers on
    # the status bar never reach the editor. That is only correct while the band
    # really is the bottom cell: the hit tests extrapolate rows past the last
    # drawn one, so any part of the band left inside the rect answers with a
    # line hidden behind it.
    e = _editor("\n".join(f"uAA{i}" for i in range(40)))
    e.set_chrome_flag(ChromeFlag.LINE_NUMBERS, True)
    e.set_draw_chrome(True)
    e.layout((600.0, 300.0), 16.0)
    cell_h = e.get_cell_size()[1]
    band = [p for p in e.prims_list() if p.kind == int(Kind.FRAME)]
    assert len(band) == 1
    interactive_h = 300.0 - cell_h
    assert band[0].y0 >= interactive_h, (
        f"status band starts at {band[0].y0}, inside the interactive {interactive_h}"
    )
    # And the band really does over-reach: probed directly, it answers glyphs.
    origin_x = e.get_text_origin()[0]
    assert e.is_mouse_pos_over_glyph((origin_x + 1.0, band[0].y0 + 1.0)), (
        "the band answering no glyph would make the host's shrink unnecessary"
    )
    e.close()


# --- 071 W-A re-vendor (editor aa8c6719): the four walk items, pinned through the ABI ----------


def test_dd_removes_an_empty_last_line() -> None:
    # 071 #1: the empty last line yielded an empty linewise range and the delete returned before
    # its own last-line rule; a non-empty last line always worked.
    e = _editor("a\n")
    e.feed("jdd")
    assert e.get_text() == "a"
    assert e.get_current_cursor_position().line == 0
    e.close()
    e = _editor("a\n")
    e.feed("jVd")
    assert e.get_text() == "a"
    e.close()


def test_shift_operators_move_lines_by_one_indent() -> None:
    e = _editor("a\nb")
    e.feed(">>")
    assert e.get_text() == "    a\nb"
    e.feed("<<")
    assert e.get_text() == "a\nb"
    e.feed("Vj>")
    assert e.get_text() == "    a\n    b"
    e.feed("u")
    assert e.get_text() == "a\nb", "a visual shift is one undo step"
    e.close()


def test_star_searches_the_whole_word_under_the_cursor() -> None:
    # `foobar` contains `foo` and must be skipped: vim's `*` is `\<foo\>`.
    e = _editor("foo bar\nfoobar foo")
    e.feed("*")
    assert e.get_current_cursor_position() == (1, 7)
    e.feed("n")
    assert e.get_current_cursor_position() == (0, 0), "n wraps with *'s whole-word rule"
    e.close()


def _search_bands(e: Editor) -> int:
    e.layout((640.0, 480.0), 16.0)
    return sum(1 for p in e.prims_list() if p.kind == int(Kind.SEARCH_MATCH))


def test_search_matches_are_lit_and_esc_puts_them_out() -> None:
    # D10: hlsearch on by default, incsearch while the line is open, Esc clears and keeps the
    # pattern, n lights again.
    e = _editor("foo bar foo\nbaz")
    assert _search_bands(e) == 0
    e.feed("/fo")
    assert _search_bands(e) == 2, "the typed prefix is lit while the / line is open"
    e.feed("o<CR>")
    assert _search_bands(e) == 2
    e.feed("<Esc>")
    assert _search_bands(e) == 0
    e.feed("n")
    assert _search_bands(e) == 2, "n keeps the pattern and lights it again"
    e.close()


def test_render_state_reacts_to_the_search_highlight_going_out() -> None:
    # Esc in normal mode clears the search bands and changes no other dimension the fingerprint
    # reads; the primitive count carries it. Falsifier: drop the count from render_state.
    e = _editor("foo bar foo")
    e.feed("/foo<CR>")
    e.layout((640.0, 480.0), 16.0)
    lit = _state(e)
    e.feed("<Esc>")
    e.layout((640.0, 480.0), 16.0)
    assert should_redraw(lit, _state(e)), "the highlight went out and nothing repainted"
    e.close()


def test_search_bands_stop_at_the_status_row() -> None:
    # Editor d2f19556: a decoration drawn after the status Frame is clipped to the text viewport,
    # so the band on the partial last row ends where the status row begins instead of painting
    # over it. Falsifier: the sixth band ends at 126, past the frame's top at 110.
    e = Editor("\n".join(f"foo {i}" for i in range(30)))
    e.set_draw_chrome(True)
    e.layout((400.0, 200.0), 16.0)
    cell_h = e.get_cell_size()[1]
    height = cell_h * 6 + 5.0  # five text rows, the status row, a five-pixel remainder
    e.feed("/foo<CR>")
    e.layout((400.0, height), 16.0)
    text_bottom = height - cell_h
    bands = [p for p in e.prims_list() if p.kind == int(Kind.SEARCH_MATCH)]
    assert bands, "the pattern is lit"
    assert all(p.y1 <= text_bottom + 0.01 for p in bands), [
        (p.y0, p.y1) for p in bands if p.y1 > text_bottom
    ]
    e.close()


def test_complete_select_minus_one_leaves_nothing_highlighted() -> None:
    # editor 469eec4: the noselect state for host-pushed batches. Enter then acts as with no
    # popup and closes it; Down picks row 0.
    e = _editor("")
    e.set_host_completion(True)
    e.feed("iSB_n")
    e.complete_begin()
    e.complete_push("SB_noise")
    e.complete_push("SB_normal")
    assert e.complete_selected() == 0
    assert e.complete_select(-1)
    assert e.complete_selected() == -1 and e.complete_open()
    e.key(KeyCode.ENTER)
    assert e.get_text() == "SB_n\n" and not e.complete_open()
    e.feed("SB_n")
    e.complete_begin()
    e.complete_push("SB_noise")
    e.complete_select(-1)
    e.key(KeyCode.DOWN)
    assert e.complete_selected() == 0
    e.key(KeyCode.ENTER)
    assert e.get_text() == "SB_n\nSB_noise"
    assert not e.complete_select(5), "an index past the list is refused"
    e.close()


def test_the_caret_glyph_is_emitted_in_the_caret_text_slot() -> None:
    # editor 469eec4: reverse video. In normal mode the glyph under the caret takes slot
    # CARET_TEXT's color and the caret quad is opaque; in insert mode the glyph is untouched.
    e = _editor("abc")
    e.set_palette(
        {
            Slot.CARET_TEXT: (0.1, 0.2, 0.3, 1.0),
            Slot.TEXT: (0.9, 0.9, 0.9, 1.0),
            Slot.SYNTAX_6: (0.9, 0.9, 0.9, 1.0),
        }
    )
    e.layout((300.0, 100.0), 16.0)
    glyphs = sorted(
        (p.x0, (round(p.r, 2), round(p.g, 2), round(p.b, 2)))
        for p in e.prims_list()
        if p.kind == int(Kind.GLYPH)
    )
    assert glyphs[0][1] == (0.1, 0.2, 0.3), glyphs
    assert glyphs[1][1] != (0.1, 0.2, 0.3), glyphs
    carets = [p for p in e.prims_list() if p.kind == int(Kind.CARET)]
    assert carets and all(round(p.a, 2) == 1.0 for p in carets)
    e.feed("i")
    e.layout((300.0, 100.0), 16.0)
    glyphs = sorted(
        (p.x0, (round(p.r, 2), round(p.g, 2), round(p.b, 2)))
        for p in e.prims_list()
        if p.kind == int(Kind.GLYPH)
    )
    assert glyphs[0][1] != (0.1, 0.2, 0.3), "insert mode leaves the glyph alone"
    e.close()
