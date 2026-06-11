import glfw
from imgui_bundle import imgui

from shaderbox.app import App, PopupState
from shaderbox.commands import (
    COMMAND_SPECS,
    CommandScope,
    popup_suppresses,
    route_flag,
)
from shaderbox.popups.lib_picker import inline_input_owns_esc


def process_hotkeys(app: App) -> None:
    # Pre-frame ONLY: must run before imgui.new_frame(). imgui.shortcut() asserts
    # outside an active frame, so registry dispatch lives in dispatch_commands.
    glfw.poll_events()
    app.imgui_renderer.process_inputs()


def dispatch_commands(app: App) -> None:
    # In-frame, before the editor child draws, so ESC's defocus is consumed this frame.
    _dispatch_registry(app)
    _handle_escape(app)


def _dispatch_registry(app: App) -> None:
    popup_open = app.any_popup_open()
    for spec in COMMAND_SPECS:
        chord = app.effective_bindings.get(spec.id, 0)
        if chord == 0:
            continue
        if spec.scope == CommandScope.EDITOR and not app.editor_focused:
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
    else:
        app.editor_defocus_requested = True
    if was_settings_open:
        app.apply_editor_settings()
