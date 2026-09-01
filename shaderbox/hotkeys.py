import glfw
from imgui_bundle import imgui

from shaderbox.app import App, PopupState
from shaderbox.commands import (
    COMMAND_SPECS,
    CommandScope,
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
    for event in events:
        if event.code == KeyCode.ESCAPE:
            if editor.is_pending() or editor.get_mode() != Mode.NORMAL:
                editor.key(KeyCode.ESCAPE)
                app.editor_esc_forwarded = True
                continue
            # Idle NORMAL: Esc is the host's — clear any selection remnant and let
            # _handle_escape defocus. Drop the queue tail: keys typed after this
            # press belong to a defocused editor.
            editor.clear_selection()
            break
        if _handle_clipboard(app, editor, event):
            continue
        consumed = editor.key(event.code, event.mods, event.text)
        if consumed and event.imgui_chord:
            app.editor_consumed_chords.add(event.imgui_chord)


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
            if event.text == "x":
                editor.replace_selection("")
    else:
        raw = glfw.get_clipboard_string(app.window)
        if raw:
            text = raw.decode() if isinstance(raw, bytes) else raw
            editor.replace_text_in_current_cursor(text)
    if event.imgui_chord:
        app.editor_consumed_chords.add(event.imgui_chord)
    return True


def _dispatch_registry(app: App) -> None:
    popup_open = app.any_popup_open()
    for spec in COMMAND_SPECS:
        chord = app.effective_bindings.get(spec.id, 0)
        if chord == 0:
            continue
        if chord in app.editor_consumed_chords:
            continue
        if spec.scope == CommandScope.EDITOR and not app.editor_focused:
            continue
        if spec.scope == CommandScope.COPILOT and not app.copilot_focused:
            continue
        if popup_suppresses(spec.scope) and popup_open:
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
    # Apply editor settings on close, not while open — avoids the modal-open FPE
    # (conventions.md ## Known quirks).
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
