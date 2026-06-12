from enum import StrEnum

from imgui_bundle import imgui

from shaderbox.app import App, PopupState
from shaderbox.commands import (
    COMMAND_SPECS,
    SPEC_BY_ID,
    CommandId,
    CommandScope,
    capture_chord,
    chord_needs_modifier,
    chord_to_str,
)
from shaderbox.constants import SHADER_LIB_SEED_DIR
from shaderbox.paths import shader_lib_root
from shaderbox.shader_lib.seed import reset_to_shipped
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_primitives import (
    caption_text,
    chord_row,
    danger_button,
    ghost_button,
    help_marker,
    label_row,
    labeled_text_input,
    modal_window,
)


class SettingsField(StrEnum):
    """A focusable settings field. Pass to `app.open_settings(focus=...)` to expand its
    owning section + keyboard-focus the field on open (e.g. a gate's 'Open Settings' jumps
    straight to the missing key). To add one: a member here, then either branch on it in the
    owning section's draw (the Copilot key) or — for an exporter field — have that exporter
    return this value from `Exporter.config_field` and pass `focus` to its field's
    `focus_field` (the Integrations loop matches `config_field` and force-opens the node).
    The string VALUES are the cross-layer contract (exporters echo them without importing this
    enum)."""

    COPILOT_KEY = "copilot.openrouter_key"
    TELEGRAM_TOKEN = "telegram.bot_token"
    YOUTUBE_CLIENT = "youtube.client"


_LABEL = "Settings##popup"


def draw_settings(app: App) -> None:
    if app.popup_state != PopupState.SETTINGS:
        return
    with modal_window(
        _LABEL, (float(SIZE.SETTINGS_W), float(SIZE.SETTINGS_H))
    ) as visible:
        if not visible:
            return
        if not _draw_body(app):
            # Apply on close only — set_*() while the modal is open FPE-crashes
            # the editor (conventions.md ## Known quirks).
            app.apply_editor_settings()
            app.popup_state = PopupState.CLOSED
            imgui.close_current_popup()


def _draw_body(app: App) -> bool:
    ctrl_w = float(SIZE.SETTINGS_CTRL_W)
    label_w = float(SIZE.SETTINGS_LABEL_W)

    imgui.separator_text("General")
    label_row(app.font_12, "Target FPS", ctrl_w, label_w)
    app.app_state.global_target_fps = imgui.drag_int(
        "##global_target_fps",
        app.app_state.global_target_fps,
        v_min=30,
        v_max=240,
        flags=imgui.SliderFlags_.always_clamp,
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
        "##font_size",
        settings.font_size,
        v_min=8,
        v_max=48,
        flags=imgui.SliderFlags_.always_clamp,
    )[1]
    label_row(app.font_12, "Tab size", ctrl_w, label_w)
    settings.tab_size = imgui.drag_int(
        "##tab_size",
        settings.tab_size,
        v_min=1,
        v_max=8,
        flags=imgui.SliderFlags_.always_clamp,
    )[1]
    label_row(app.font_12, "Line spacing", ctrl_w, label_w)
    settings.line_spacing = imgui.drag_float(
        "##line_spacing",
        settings.line_spacing,
        v_min=1.0,
        v_max=2.0,
        v_speed=0.01,
        flags=imgui.SliderFlags_.always_clamp,
    )[1]

    imgui.dummy((0.0, SPACE.MD))
    imgui.separator_text("Integrations")

    # A pending focus target (app.settings_focus) force-opens its owning section so the field
    # is visible before focus_field scrolls to it; consumed one-shot below.
    focus = app.settings_focus

    for exporter in app.exporter_registry.all():
        if exporter.is_available:
            wants = focus != "" and focus == exporter.config_field
            if wants:
                imgui.set_next_item_open(True, imgui.Cond_.always)
            if imgui.tree_node(exporter.display_name):
                exporter.draw_config_ui(focus=wants)
                imgui.tree_pop()
        else:
            imgui.text_colored(
                COLOR.FG_DIM, f"{exporter.display_name} — {exporter.unavailable_reason}"
            )

    copilot_focus = focus == SettingsField.COPILOT_KEY
    if copilot_focus:
        imgui.set_next_item_open(True, imgui.Cond_.always)
    if imgui.tree_node("Copilot"):
        _draw_copilot_config(app, focus=copilot_focus)
        imgui.tree_pop()

    # One-shot: the focus request fired into this frame's draws; clear so it doesn't re-grab
    # focus every frame (which a modal reads as a dismiss — /imgui-ui §8).
    app.settings_focus = ""

    imgui.dummy((0.0, SPACE.MD))
    imgui.separator_text("Library")
    _draw_library_reset(app)

    imgui.dummy((0.0, SPACE.MD))
    imgui.separator_text("Keyboard")
    _draw_keybindings(app)

    imgui.dummy((0.0, SPACE.MD))

    is_keep_opened: bool = True
    if ghost_button("Close", width=float(SIZE.BTN_SM_W)):
        is_keep_opened = False

    return is_keep_opened


def _draw_library_reset(app: App) -> None:
    # Factory reset of the shipped shader library. Armed confirm (one extra click);
    # the index rebuild rides the mtime watcher, no explicit poke needed.
    if not app.lib_reset_armed:
        if danger_button("Reset library to shipped..."):
            app.lib_reset_armed = True
        return
    imgui.push_text_wrap_pos(0.0)
    imgui.text_colored(
        COLOR.STATE_WARN,
        "Every shipped library file returns to its factory version; your edited "
        "copies move to .trash/ first. Files you created yourself are kept.",
    )
    imgui.pop_text_wrap_pos()
    if danger_button("Confirm reset"):
        written, trashed = reset_to_shipped(SHADER_LIB_SEED_DIR, shader_lib_root())
        app.notifications.push(
            f"Library reset: {written} file(s) restored, {trashed} edited "
            "copy(ies) moved to .trash/"
        )
        app.lib_reset_armed = False
    imgui.same_line()
    if ghost_button("Cancel"):
        app.lib_reset_armed = False


# (label, field, hint, min value, input step) per user-tunable agent limit. 0 = off where
# the floor is 0; the hint is the durable explanation surfaced via help_marker.
_COPILOT_LIMITS: list[tuple[str, str, str, int, int]] = [
    (
        "Context cap (tokens)",
        "max_input_tokens",
        "Max input tokens per LLM request — the context gauge's budget. Older chat "
        "history is trimmed to fit under it. Bigger = more memory, higher cost per turn.",
        10_000,
        5_000,
    ),
    (
        "Reply cap (tokens)",
        "max_tokens_per_turn",
        "Max output tokens per LLM step: the visible reply + tool arguments + the "
        "model's hidden reasoning. Too low truncates big shader rewrites mid-edit.",
        1_000,
        1_000,
    ),
    (
        "Max steps per turn",
        "max_iterations",
        "Tool-call steps the agent may take for one of your messages before it is "
        "cut off with an error reply.",
        1,
        1,
    ),
    (
        "Failed-edit giveup",
        "max_edit_retries",
        "Consecutive edits that FAIL to apply (bad match / bad range) before the "
        "turn stops and the agent reports it is stuck.",
        1,
        1,
    ),
    (
        "Broken-compile hint after",
        "max_compile_failures",
        "Consecutive edits that apply but compile broken before a one-time 'stop "
        "patching, rewrite the whole block' hint. 0 = off.",
        0,
        1,
    ),
    (
        "Clean-edit hint after",
        "max_clean_edit_streak",
        "Consecutive clean edits in one turn before a one-time 'stop and let the "
        "user look' hint — brakes blind aesthetic tweak sprees. 0 = off.",
        0,
        1,
    ),
    (
        "Auto-restore after",
        "auto_revert_after_failed_edits",
        "Consecutive broken-compile edits on one file before the engine restores "
        "its last clean-compiling state and tells the agent. 0 = off.",
        0,
        1,
    ),
]


def _draw_copilot_config(app: App, focus: bool = False) -> None:
    field_w = float(SIZE.SETTINGS_CTRL_W) * 2.0
    cfg = app.integrations_store.copilot
    new_key = labeled_text_input(
        "OpenRouter key", cfg.openrouter_key, field_w, password=True, focus=focus
    )
    new_model = labeled_text_input("Model", cfg.model, field_w)
    if new_key != cfg.openrouter_key or new_model != cfg.model:
        cfg.openrouter_key = new_key
        cfg.model = new_model
        app.integrations_store.save()

    imgui.dummy((0.0, SPACE.SM))
    caption_text("Agent limits")
    changed_any = False
    label_w = max(
        imgui.calc_text_size(label).x for label, *_ in _COPILOT_LIMITS
    ) + float(SPACE.LG)
    for label, field, hint, min_v, step in _COPILOT_LIMITS:
        imgui.text_colored(COLOR.FG_DIM, label)
        imgui.same_line(label_w)
        imgui.set_next_item_width(float(SIZE.SETTINGS_CTRL_W))
        value = int(getattr(cfg, field))
        changed, new_value = imgui.input_int(f"##cp_{field}", value, step=step)
        if changed and max(min_v, new_value) != value:
            setattr(cfg, field, max(min_v, new_value))
            changed_any = True
        imgui.same_line()
        help_marker(hint)
    if changed_any:
        app.integrations_store.save()
        cfg.apply_limits()


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
    # Rebind arms one-frame capture: the next chord commits unless it lacks a
    # modifier or duplicates a command in the same scope; Esc cancels.
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
