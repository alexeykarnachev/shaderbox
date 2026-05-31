from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.commands import (
    COMMAND_SPECS,
    SPEC_BY_ID,
    CommandId,
    CommandScope,
    capture_chord,
    chord_needs_modifier,
    chord_to_str,
)
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_primitives import (
    chord_row,
    ghost_button,
    label_row,
    labeled_text_input,
    modal_window,
)

_LABEL = "Settings##popup"


def draw_settings(app: App) -> None:
    if not app.is_settings_open:
        return
    with modal_window(
        _LABEL, (float(SIZE.SETTINGS_W), float(SIZE.SETTINGS_H))
    ) as visible:
        if not visible:
            return
        if not _draw_body(app):
            # Editor settings apply on close only — set_*() while the modal is open
            # FPE-crashes the editor (conventions.md ## Known quirks).
            app.apply_editor_settings()
            app.is_settings_open = False
            imgui.close_current_popup()


def _draw_body(app: App) -> bool:
    ctrl_w = float(SIZE.SETTINGS_CTRL_W)
    label_w = float(SIZE.SETTINGS_LABEL_W)

    imgui.separator_text("General")
    label_row(app.font_12, "Target FPS", ctrl_w, label_w)
    app.app_state.global_target_fps = imgui.drag_int(
        "##global_target_fps", app.app_state.global_target_fps, v_min=30, v_max=240
    )[1]
    app.app_state.show_cheatsheet = imgui.checkbox(
        "Show keyboard cheatsheet", app.app_state.show_cheatsheet
    )[1]

    imgui.dummy((0.0, SPACE.MD))
    imgui.separator_text("Editor")

    settings = app.app_state.editor_settings
    settings.show_whitespace = imgui.checkbox(
        "Show whitespace", settings.show_whitespace
    )[1]
    settings.show_line_numbers = imgui.checkbox(
        "Show line numbers", settings.show_line_numbers
    )[1]
    settings.show_matching_brackets = imgui.checkbox(
        "Highlight matching brackets", settings.show_matching_brackets
    )[1]

    imgui.dummy((0.0, SPACE.SM))
    label_row(app.font_12, "Font size", ctrl_w, label_w)
    settings.font_size = imgui.drag_int(
        "##font_size", settings.font_size, v_min=8, v_max=48
    )[1]
    label_row(app.font_12, "Tab size", ctrl_w, label_w)
    settings.tab_size = imgui.drag_int(
        "##tab_size", settings.tab_size, v_min=1, v_max=8
    )[1]
    label_row(app.font_12, "Line spacing", ctrl_w, label_w)
    settings.line_spacing = imgui.drag_float(
        "##line_spacing", settings.line_spacing, v_min=1.0, v_max=2.0, v_speed=0.01
    )[1]

    imgui.dummy((0.0, SPACE.MD))
    imgui.separator_text("Integrations")

    for exporter in app.exporter_registry.all():
        if exporter.is_available:
            if imgui.tree_node(exporter.display_name):
                exporter.draw_config_ui()
                imgui.tree_pop()
        else:
            imgui.text_colored(
                COLOR.FG_DIM, f"{exporter.display_name} — {exporter.unavailable_reason}"
            )

    if imgui.tree_node("Copilot"):
        _draw_copilot_config(app)
        imgui.tree_pop()

    imgui.dummy((0.0, SPACE.MD))
    imgui.separator_text("Keyboard")
    _draw_keybindings(app)

    imgui.dummy((0.0, SPACE.MD))

    is_keep_opened: bool = True
    if ghost_button("Close", width=float(SIZE.BTN_SM_W)):
        is_keep_opened = False

    return is_keep_opened


def _draw_copilot_config(app: App) -> None:
    # The OpenRouter key + model for the in-app copilot (feature 020). Edit-on-change +
    # save, matching the exporter credential blocks. The client reads both live via
    # getters, so a key/model entered here is seen on the next turn.
    field_w = float(SIZE.SETTINGS_CTRL_W) * 2.0
    cfg = app.integrations_store.copilot
    new_key = labeled_text_input(
        "OpenRouter key", cfg.openrouter_key, field_w, password=True
    )
    new_model = labeled_text_input("Model", cfg.model, field_w)
    if new_key != cfg.openrouter_key or new_model != cfg.model:
        cfg.openrouter_key = new_key
        cfg.model = new_model
        app.integrations_store.save()


def _chord_in_use(
    app: App, chord: int, scope: CommandScope, exclude: CommandId
) -> bool:
    return any(
        spec.id != exclude
        and spec.scope == scope
        and app.effective_bindings.get(spec.id, 0) == chord
        for spec in COMMAND_SPECS
    )


def _draw_keybindings(app: App) -> None:
    # One row per command (the same set the cheatsheet shows): name + current
    # chord + a Rebind button that arms one-frame capture. While armed, the next
    # pressed chord commits (unless it duplicates another command in the same
    # scope or lacks a modifier), Esc cancels. Fixed-key commands (arrow nav) show
    # their chord with the Rebind button disabled.
    if app.rebinding_command is not None:
        chord = capture_chord()
        if imgui.is_key_pressed(imgui.Key.escape, repeat=False):
            app.rebinding_command = None
        elif chord is not None:
            command_id = app.rebinding_command
            scope = SPEC_BY_ID[command_id].scope
            if chord_needs_modifier(chord):
                app.notifications.push(
                    f"{chord_to_str(chord)} needs a modifier (Ctrl/Shift/Alt)",
                    COLOR.STATE_WARN[:3],
                )
            elif _chord_in_use(app, chord, scope, command_id):
                app.notifications.push(
                    f"{chord_to_str(chord)} is already bound in this scope",
                    COLOR.STATE_WARN[:3],
                )
            else:
                app.rebind_command(command_id, chord)
            app.rebinding_command = None

    label_w = float(SIZE.SETTINGS_W) * 0.4
    for spec in COMMAND_SPECS:
        capturing = app.rebinding_command == spec.id
        chord_str = (
            "press a key..."
            if capturing
            else chord_to_str(app.effective_bindings.get(spec.id, 0))
        )
        chord_row(spec.label, chord_str, label_w, highlight=capturing)
        imgui.same_line(label_w + float(SIZE.UNIFORM_CTRL_W) * 0.5)
        if not spec.rebindable:
            imgui.begin_disabled()
        if ghost_button(f"Rebind##{spec.id.value}", width=float(SIZE.BTN_SM_W)):
            app.rebinding_command = spec.id
        if not spec.rebindable:
            imgui.end_disabled()
