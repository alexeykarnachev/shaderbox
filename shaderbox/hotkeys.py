import glfw
from imgui_bundle import imgui

from shaderbox.app import App, PopupState
from shaderbox.commands import (
    COMMAND_SPECS,
    CommandScope,
    popup_suppresses,
    route_flag,
)


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
        flags = route_flag(spec.scope)
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
    app.popup_state = PopupState.CLOSED
    app.is_palette_open = False
    app.editor_defocus_requested = True
    # Esc defocuses the chat but leaves it open.
    if app.copilot_focused:
        app.copilot_defocus_requested = True
    if was_settings_open:
        app.apply_editor_settings()
