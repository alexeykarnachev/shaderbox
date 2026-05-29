import glfw
from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.commands import (
    COMMAND_SPECS,
    CommandScope,
    popup_suppresses,
    route_flag,
)


def process_hotkeys(app: App) -> None:
    # Pre-frame ONLY: these must run before imgui.new_frame(). Registry dispatch
    # lives in dispatch_commands (in-frame) because imgui.shortcut() asserts
    # outside an active frame.
    glfw.poll_events()
    app.imgui_renderer.process_inputs()


def dispatch_commands(app: App) -> None:
    # In-frame: called at the TOP of the main-window block, before the editor
    # child draws, so ESC's defocus request is consumed the same frame.
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
    # While the rebinder is capturing, Esc cancels the capture (handled in the
    # settings draw) — don't also close the modal.
    if app.rebinding_command is not None:
        return
    # Esc with no job (no popup/editor) is swallowed at the glfw layer before imgui
    # sees it (App._install_escape_filter), so this in-frame handler shouldn't even
    # receive it then — but gate defensively on the same condition.
    if not app.escape_has_job():
        return
    # ESC returns the app to its default state: close any popup, drop editor focus.
    # Never quits. Settings holds the editor options — push them on close (apply-on-
    # close avoids the modal-open FPE, conventions.md ## Known quirks).
    was_settings_open = app.is_settings_open
    app.is_node_creator_open = False
    app.is_settings_open = False
    app.is_palette_open = False
    app.editor_defocus_requested = True
    if was_settings_open:
        app.apply_editor_settings()
