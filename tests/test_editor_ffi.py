"""Feature 067: the libeditor binding, the input translation, the drain gates,
and the redraw-gate domain. All headless — the .so needs no GL."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import glfw
import numpy as np
import pytest
from imgui_bundle import imgui

from shaderbox.commands import COMMAND_SPECS, CommandId
from shaderbox.editor.ffi import Editor, KeyCode, KeyMod, Kind, Language, Mode
from shaderbox.editor.input import KeyEvent, translate_char, translate_key
from shaderbox.editor.render import (
    PRIM_DTYPE,
    build_vertices,
    render_state,
    should_redraw,
)
from shaderbox.editor_types import EditorSession
from shaderbox.hotkeys import _drain_editor_input
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
        copilot_turn_active=False,
        editor_completion_requested=False,
        editor_clipboard_seen="",
        editor_visible_rows=20,
        flush_current_editor=None,
        notifications=SimpleNamespace(push=lambda *_a, **_k: None),
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
    assert app.editor_esc_forwarded is True, (
        "every focused Esc forwards — _handle_escape's defocus branch stays quiet"
    )
    assert e.get_text() == "two\nthree\n", "the dd after Esc still edits"
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
    from shaderbox.commands import SPEC_BY_ID
    from shaderbox.hotkeys import spec_eligible

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


def test_ctrl_o_is_consumed_noop_while_focused() -> None:
    e = _editor()
    app = _drain_app(e)
    app.editor_key_events = [_ctrl("o")]
    _drain_editor_input(app)
    assert e.get_text() == "one\ntwo\nthree\n"
    assert (int(imgui.Key.o) | int(imgui.Key.mod_ctrl)) in app.editor_consumed_chords, (
        "the jump-back reflex must not open the project dialog"
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
    flushed: list[bool] = []
    app.flush_current_editor = lambda: flushed.append(True)
    _type_ex(app, ":w")
    assert flushed, ":w must save"
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
        "gutter_px": 0.0,
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
    assert should_redraw(base, _state(e, gutter_px=40.0))
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
    app.editor_clipboard_seen = ""
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
    app.editor_clipboard_seen = ""
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
    from shaderbox.commands import SPEC_BY_ID
    from shaderbox.hotkeys import spec_eligible

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
