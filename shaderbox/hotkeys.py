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
from shaderbox.editor.ffi import Editor, HostCommandKind, KeyCode, KeyMod, Mode
from shaderbox.editor.input import KeyEvent
from shaderbox.editor_types import EditorSession
from shaderbox.popups.lib_picker import inline_input_owns_esc
from shaderbox.theme import COLOR


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
    # Synced at the drain boundaries against the ACTIVE session's own register
    # (registers are per-handle — an app-global "last seen" left a switched-to
    # tab's register unseeded, so a cross-tab yy/p silently pasted nothing).
    clip = _read_clipboard(app)
    if clip and clip != editor.get_register():
        editor.set_register(clip, linewise=clip.endswith("\n"))
    for event in events:
        if event.code == KeyCode.ESCAPE:
            # Esc is vim's modal key — the editor owns it UNCONDITIONALLY while
            # focused (maintainer decision, 067 manual pass); the glfw filter
            # already keeps the press away from imgui. Defocus lives on
            # CYCLE_REGION and the mouse, never on Esc.
            editor.key(KeyCode.ESCAPE)
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
            # Only when the popup is CLOSED: with it open the keymap advances the
            # selection itself (editor 3f3a11b), and queuing an offer here would
            # re-push the list and reset that selection to zero on every press.
            if (
                event.text == "n"
                and event.mods == KeyMod.CTRL
                and editor.get_mode() == Mode.INSERT
                and not editor.complete_open()
            ):
                app.editor_completion_requested = True
    register = editor.get_register()
    if register and register != clip:
        glfw.set_clipboard_string(app.window, register)
    _serve_host_command(app, session)


def _serve_host_command(app: App, session: EditorSession) -> None:
    # The ex commands whose OBJECT the host owns (editor c5fabc8): :w saves,
    # :q closes the tab (refusing on unsaved changes unless forced, as vim
    # does), :wq/:x both. A path argument names a host feature we don't have.
    command = session.editor.take_host_command()
    if command is None:
        return
    if command.arg:
        app.notifications.push(
            "Path argument not supported", color=COLOR.STATE_WARN[:3]
        )
        return
    dirty = session.editor.get_undo_index() != session.saved_undo
    if command.kind in (HostCommandKind.WRITE, HostCommandKind.WRITE_QUIT):
        # App.save is the one funnel: flush + document write to DISK + the
        # copilot busy gate. A memory-only flush made :w then :q! revert past
        # the save the user was just told about.
        app.save()
        app.notifications.push("Saved")
    if command.kind in (HostCommandKind.QUIT, HostCommandKind.WRITE_QUIT):
        if command.kind == HostCommandKind.QUIT and dirty and not command.force:
            app.notifications.push(
                "Unsaved changes (:q! discards)", color=COLOR.STATE_WARN[:3]
            )
            return
        if command.kind == HostCommandKind.QUIT and dirty and command.force:
            # :q! — discard: reload the file's on-disk text into the session.
            try:
                disk_text = session.source.path.read_text()
            except OSError:
                disk_text = None
            if disk_text is not None:
                session.editor.set_text(disk_text)
                session.saved_undo = session.editor.get_undo_index()
        if app.editor_tabs:
            app.close_tab(app.active_tab_index)


# Vim-reserved Ctrl chords while the editor is focused: each either does the vim
# thing or nothing — NEVER an app command. FALLBACK ONLY: runs after ed_key
# returned unconsumed, so the day the keymap grows a real Ctrl+D motion (with
# cursor semantics — asked for), the host approximation yields automatically.
# Scroll steps are in visible rows; the app half of each chord (DELETE_DOCUMENT on
# Ctrl+D, OPEN_SHADER on Ctrl+E, OPEN_PROJECT on Ctrl+O) stays reachable unfocused.
# The full reserved set: the six scrolls + redo + jumplist + word/line kills +
# completion nav + left/down. NORMAL-mode Ctrl+W is the one deliberate carve-out
# (see _handle_vim_chord). The app half of every reserved chord stays reachable
# while the editor is unfocused.
_VIM_RESERVED_CHORDS: frozenset[str] = frozenset("dufbeyrownphj")

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
    if ch not in _VIM_RESERVED_CHORDS:
        return False
    mode = editor.get_mode()
    insert = mode == Mode.INSERT
    # One exception keeps an app command reachable: NORMAL-mode Ctrl+W falls
    # through to CLOSE_CODE_TAB — vim's own normal Ctrl+W is only a window
    # prefix and we have no windows. Insert/visual Ctrl+W stays vim's.
    if ch == "w" and mode == Mode.NORMAL:
        return False
    if insert:
        if ch == "u":
            pos = editor.get_current_cursor_position()
            if pos.column > 0:
                editor.delete_range((pos.line, 0), (pos.line, pos.column))
        elif ch == "w":
            _delete_word_back(editor)
        elif ch == "p":
            # previous candidate when the popup shows, else trigger completion
            # (the lib picker keeps Ctrl+P outside the editor's focus).
            if editor.complete_open():
                editor.key(KeyCode.UP)
            else:
                app.editor_completion_requested = True
        elif ch == "h":
            editor.key(KeyCode.BACKSPACE)
        elif ch == "j":
            editor.key(KeyCode.ENTER)
        # d/f/b/e/y/r/o/n: vim meanings we don't implement — consume-noop so no
        # app command fires mid-typing (Ctrl+R = OPEN_SCRIPT was reachable here).
    else:
        # NORMAL/VISUAL, unconsumed by the keymap. The six scrolls are keymap
        # motions in BOTH modes as of editor 3f3a11b (visual extends the
        # selection), so they never reach here; what lands is the rest.
        if ch in ("n", "j"):
            editor.key(KeyCode.DOWN)
        elif ch == "p":
            editor.key(KeyCode.UP)
        elif ch == "h":
            editor.key(KeyCode.LEFT)
        # r (visual), o: consume-noop, so no app command fires.
    if event.imgui_chord:
        app.editor_consumed_chords.add(event.imgui_chord)
    return True


def _read_clipboard(app: App) -> str:
    # glfw raises on a clipboard holding no convertible text (X11) — read as "empty".
    try:
        raw = glfw.get_clipboard_string(app.window)
    except glfw.GLFWError:
        return ""
    if not raw:
        return ""
    return raw.decode() if isinstance(raw, bytes) else raw


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
        if app.popup_state == PopupState.PASS_SETTINGS:
            # The gear's name field can hold an uncommitted edit; the close funnel commits it.
            app.close_pass_settings()
        elif (
            app.popup_state != PopupState.SHADER_LIB_PICKER
            or not inline_input_owns_esc(app)
        ):
            app.popup_state = PopupState.CLOSED
    elif app.is_palette_open:
        app.is_palette_open = False
    elif app.copilot_focused:
        # Esc defocuses the chat but leaves it open.
        app.copilot_defocus_requested = True
    # No editor branch: a focused editor's Esc never reaches imgui (the glfw
    # filter swallows it), so is_key_pressed above is False on those frames.
    if was_settings_open:
        app.apply_editor_settings()
