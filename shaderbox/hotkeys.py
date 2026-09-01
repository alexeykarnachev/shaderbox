import glfw
from imgui_bundle import imgui

from shaderbox.app import App, PopupState
from shaderbox.commands import (
    COMMAND_SPECS,
    CommandScope,
    CommandSpec,
    popup_suppresses,
    route_flag,
)
from shaderbox.editor.ffi import Editor, KeyCode, KeyMod, Mode
from shaderbox.editor.input import KeyEvent
from shaderbox.popups.lib_picker import inline_input_owns_esc


def process_hotkeys(app: App) -> None:
    # Pre-frame ONLY: must run before imgui.new_frame(). imgui.shortcut() asserts
    # outside an active frame, so registry dispatch lives in dispatch_commands.
    glfw.poll_events()
    app.imgui_renderer.process_inputs()


def dispatch_commands(app: App) -> None:
    # In-frame, before the editor child draws, so ESC's defocus is consumed this frame.
    # The editor drain runs FIRST: chords it consumes are struck from this frame's
    # registry dispatch (feature 067 — the one guard against Ctrl+R double-dispatch).
    _drain_editor_input(app)
    _dispatch_registry(app)
    _handle_escape(app)


def _drain_editor_input(app: App) -> None:
    # Feed the frame's queued glfw key events into the focused editor. The focus gate
    # reads LAST frame's editor_focused (written after the editor draws); the
    # newly-focused-deaf-one-frame direction is safe, and the defocus direction is
    # closed by dropping the queue remainder once Esc decides to defocus.
    app.editor_consumed_chords.clear()
    app.editor_esc_forwarded = False
    events = app.editor_key_events
    app.editor_key_events = []
    if not events or not app.editor_focused or app.any_popup_open():
        return
    session = app.get_current_session_if_exists()
    if session is None:
        return
    editor = session.editor
    # Clipboard <-> register unification (editor commit 4befeaf): ONE slot, so
    # `p` pastes what the OS clipboard holds and a `yy` is Ctrl+V-able anywhere.
    # Synced at the drain boundaries — at most one OS round-trip per keyed frame,
    # and a tab switch reconciles on its first keypress (registers are per-handle).
    clip = _read_clipboard(app)
    if clip and clip != app.editor_clipboard_seen:
        editor.set_register(clip, linewise=clip.endswith("\n"))
        app.editor_clipboard_seen = clip
    for event in events:
        if event.code == KeyCode.ESCAPE:
            # Esc is vim's modal key — the editor owns it UNCONDITIONALLY while
            # focused (maintainer decision, 067 manual pass). Defocus lives on
            # CYCLE_REGION and the mouse, never on Esc.
            editor.key(KeyCode.ESCAPE)
            app.editor_esc_forwarded = True
            continue
        if _handle_ex_command(app, editor, event):
            continue
        if _handle_clipboard(app, editor, event):
            continue
        consumed = editor.key(event.code, event.mods, event.text)
        if not consumed and _handle_vim_chord(app, editor, event):
            continue
        if consumed and event.imgui_chord:
            app.editor_consumed_chords.add(event.imgui_chord)
            # An insert-mode Ctrl+N is the deliberate completion ask — the keymap
            # consumes it but opens nothing (host_completion); code.draw offers.
            if (
                event.text == "n"
                and event.mods == KeyMod.CTRL
                and editor.get_mode() == Mode.INSERT
            ):
                app.editor_completion_requested = True
    register = editor.get_register()
    if register and register != app.editor_clipboard_seen:
        glfw.set_clipboard_string(app.window, register)
        app.editor_clipboard_seen = register


# Vim-reserved Ctrl chords while the editor is focused: each either does the vim
# thing or nothing — NEVER an app command. FALLBACK ONLY: runs after ed_key
# returned unconsumed, so the day the keymap grows a real Ctrl+D motion (with
# cursor semantics — asked for), the host approximation yields automatically.
# Scroll steps are in visible rows; the app half of each chord (DELETE_DOCUMENT on
# Ctrl+D, OPEN_SHADER on Ctrl+E, OPEN_PROJECT on Ctrl+O) stays reachable unfocused.
_VIM_SCROLL_CHORDS: frozenset[str] = frozenset("dufbey")

_WORD_CHARS: str = "_"


def _is_word_char(c: str) -> bool:
    return c.isalnum() or c in _WORD_CHARS


def _delete_word_back(editor: Editor) -> None:
    # vim's insert-mode Ctrl+W: whitespace run, then one word (or punct run).
    pos = editor.get_current_cursor_position()
    if pos.column == 0:
        return
    line = editor.get_text().split("\n")[pos.line]
    i = pos.column
    while i > 0 and line[i - 1].isspace():
        i -= 1
    if i > 0:
        if _is_word_char(line[i - 1]):
            while i > 0 and _is_word_char(line[i - 1]):
                i -= 1
        else:
            while (
                i > 0 and not line[i - 1].isspace() and not _is_word_char(line[i - 1])
            ):
                i -= 1
    editor.delete_range((pos.line, i), (pos.line, pos.column))


def _handle_vim_chord(app: App, editor: Editor, event: KeyEvent) -> bool:
    if event.code != KeyCode.CHAR or event.mods != KeyMod.CTRL:
        return False
    ch = event.text
    insert = editor.get_mode() == Mode.INSERT
    handled = False
    if ch in _VIM_SCROLL_CHORDS:
        handled = True
        if insert:
            # vim's insert-mode meanings differ (dedent, char-below, ...); the one
            # worth having is Ctrl+U = delete to line start. The rest consume-noop
            # so no app command fires mid-typing.
            if ch == "u":
                pos = editor.get_current_cursor_position()
                if pos.column > 0:
                    editor.delete_range((pos.line, 0), (pos.line, pos.column))
        else:
            rows = max(1, app.editor_visible_rows)
            step = {
                "d": max(1, rows // 2),
                "u": -max(1, rows // 2),
                "f": rows,
                "b": -rows,
                "e": 1,
                "y": -1,
            }[ch]
            editor.set_scroll(editor.get_scroll() + step)
    elif ch == "o":
        # vim's jump-back reflex: consume-noop (no jumplist yet); OPEN_PROJECT
        # must not fire mid-editing.
        handled = True
    elif ch == "w" and insert:
        # vim's insert-mode Ctrl+W deletes the word back — it must NOT close the tab.
        _delete_word_back(editor)
        handled = True
    elif ch == "p" and insert:
        # vim's insert-mode Ctrl+P: previous candidate when the popup shows, else
        # trigger completion (the lib picker keeps Ctrl+P outside insert mode).
        if editor.complete_open():
            editor.key(KeyCode.UP)
        else:
            app.editor_completion_requested = True
        handled = True
    if handled and event.imgui_chord:
        app.editor_consumed_chords.add(event.imgui_chord)
    return handled


def _read_clipboard(app: App) -> str:
    # glfw raises on a clipboard holding no convertible text (X11) — read as "empty".
    try:
        raw = glfw.get_clipboard_string(app.window)
    except glfw.GLFWError:
        return ""
    if not raw:
        return ""
    return raw.decode() if isinstance(raw, bytes) else raw


def _handle_ex_command(app: App, editor: Editor, event: KeyEvent) -> bool:
    # INTERIM host intercept for the ex commands whose OBJECT the host owns (the
    # file): `:w` saves, `:wq`/`:x` save and close the tab. The editor executes
    # what it owns (`:s`, search); a file write can only happen here. Pending an
    # ABI host-command surface (filed editor-side) — this branch deletes itself
    # when that lands.
    if event.code != KeyCode.ENTER or event.mods != 0:
        return False
    if editor.get_command_line_prompt() != ":":
        return False
    command = (editor.get_command_line() or "").strip()
    if command not in ("w", "wq", "x"):
        return False
    editor.key(KeyCode.ESCAPE)  # close the command line without executing
    app.flush_current_editor()
    app.notifications.push("Saved")
    if command in ("wq", "x") and app.editor_tabs:
        app.close_tab(app.active_tab_index)
    return True


def _handle_clipboard(app: App, editor: Editor, event: KeyEvent) -> bool:
    # Host-wired clipboard (the keymap has no registers): Ctrl+C/X/V against the
    # system clipboard. Runs before ed_key — the editor leaves these unbound.
    if event.code != KeyCode.CHAR or event.mods != KeyMod.CTRL:
        return False
    if event.text not in ("c", "x", "v"):
        return False
    if event.text in ("c", "x"):
        selected = editor.get_selection_text()
        if selected:
            glfw.set_clipboard_string(app.window, selected)
            editor.set_register(selected, linewise=selected.endswith("\n"))
            app.editor_clipboard_seen = selected
            if event.text == "x" and not app.copilot_turn_active:
                editor.replace_selection("")
    elif not app.copilot_turn_active:
        raw = glfw.get_clipboard_string(app.window)
        if raw:
            text = raw.decode() if isinstance(raw, bytes) else raw
            editor.replace_text_in_current_cursor(text)
    if event.imgui_chord:
        app.editor_consumed_chords.add(event.imgui_chord)
    return True


def spec_eligible(app: App, spec: CommandSpec, chord: int, popup_open: bool) -> bool:
    # Whether this spec may dispatch this frame. The consumed-chord test is the
    # one guard against a focused-editor Ctrl+R both redoing and opening the
    # script tab (feature 067); the rest are the scope/popup gates.
    if chord == 0 or chord in app.editor_consumed_chords:
        return False
    if spec.scope == CommandScope.EDITOR and not app.editor_focused:
        return False
    if spec.scope == CommandScope.COPILOT and not app.copilot_focused:
        return False
    return not (popup_suppresses(spec.scope) and popup_open)


def _dispatch_registry(app: App) -> None:
    popup_open = app.any_popup_open()
    for spec in COMMAND_SPECS:
        chord = app.effective_bindings.get(spec.id, 0)
        if not spec_eligible(app, spec, chord, popup_open):
            continue
        flags = route_flag(spec.scope, chord)
        if spec.repeat:
            flags |= imgui.InputFlags_.repeat
        if imgui.shortcut(chord, flags=flags):
            app.command_callbacks[spec.id]()


def _handle_escape(app: App) -> None:
    if not imgui.is_key_pressed(imgui.Key.escape, repeat=False):
        return
    # While rebinding, Esc cancels the capture (settings draw) — don't also close the modal.
    if app.rebinding_command is not None:
        return
    # Jobless Esc is already swallowed at the glfw layer (App._install_escape_filter);
    # gate defensively on the same condition.
    if not app.escape_has_job():
        return
    # Editor settings apply at the one close funnel, not per-edit while the
    # modal is open.
    was_settings_open = app.popup_state == PopupState.SETTINGS
    # Esc dismisses ONE thing, most-modal first: the revert confirm, else an open popup, else
    # the palette, else the chat focus, else the editor caret. Dismissing a popup/palette must
    # NOT also defocus the editor or chat — App.reconcile_popup_focus restores focus to whoever
    # the popup stole it from.
    if app.copilot_revert_target is not None:
        app.copilot_revert_target = None
    elif app.any_popup_open():
        # The lib picker's inline inputs (rename / new-file / new-dir / add-tag) own Esc:
        # their per-input cancel runs later this frame — leave the picker open.
        if app.popup_state != PopupState.SHADER_LIB_PICKER or not inline_input_owns_esc(
            app
        ):
            app.popup_state = PopupState.CLOSED
    elif app.is_palette_open:
        app.is_palette_open = False
    elif app.copilot_focused:
        # Esc defocuses the chat but leaves it open.
        app.copilot_defocus_requested = True
    elif not app.editor_esc_forwarded:
        # Single-consumer rule (feature 067): when the drain forwarded this press into
        # the editor (insert/visual exit, pending cancel), the defocus stays quiet.
        app.editor_defocus_requested = True
    if was_settings_open:
        app.apply_editor_settings()
