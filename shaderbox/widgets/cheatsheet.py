import glfw
from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.commands import (
    COMMAND_SPECS,
    CommandId,
    CommandScope,
    chord_to_str,
)
from shaderbox.theme import CHEATSHEET_ALPHA, COLOR, SIZE, SPACE, fade


def _is_active(scope: CommandScope, app: App) -> bool:
    if scope == CommandScope.EDITOR:
        return app.editor_focused
    return not app.any_popup_open()


def draw(app: App) -> None:
    # Drawn onto the foreground draw list (absolute screen coords) rather than a
    # window — it renders on top of the full-screen main window unconditionally,
    # immune to window z-order. Top-right, faint, listing the chords valid now.
    if not app.app_state.show_cheatsheet:
        return

    rows: list[tuple[str, str]] = [
        (spec.label, chord_to_str(app.effective_bindings[spec.id]))
        for spec in COMMAND_SPECS
        if app.effective_bindings.get(spec.id, 0) != 0 and _is_active(spec.scope, app)
    ]
    if not rows:
        return

    hide_chord = chord_to_str(app.effective_bindings[CommandId.TOGGLE_CHEATSHEET])
    footer = f"press {hide_chord} to hide"

    pad = float(SPACE.MD)
    line_h = imgui.get_text_line_height_with_spacing()
    gap = float(SPACE.LG)  # min gap between the label column and the chord column

    label_w = max(imgui.calc_text_size(label).x for label, _ in rows)
    chord_w = max(imgui.calc_text_size(chord).x for _, chord in rows)
    title_h = line_h + float(SPACE.XS)
    footer_h = line_h + float(SPACE.XS)
    content_w = label_w + gap + chord_w
    content_w = max(content_w, imgui.calc_text_size(footer).x)
    box_w = pad * 2.0 + content_w
    box_h = pad * 2.0 + title_h + line_h * len(rows) + footer_h

    margin = float(SIZE.CHEATSHEET_MARGIN)
    win_w, _ = glfw.get_window_size(app.window)
    x0 = win_w - box_w - margin
    y0 = margin

    dl = imgui.get_foreground_draw_list()
    bg = imgui.color_convert_float4_to_u32(fade(COLOR.BG_POPUP, CHEATSHEET_ALPHA))
    border = imgui.color_convert_float4_to_u32(fade(COLOR.BORDER, CHEATSHEET_ALPHA))
    title_col = imgui.color_convert_float4_to_u32(COLOR.FG_DIM)
    label_col = imgui.color_convert_float4_to_u32(COLOR.FG_SECONDARY)
    chord_col = imgui.color_convert_float4_to_u32(COLOR.ACCENT_PRIMARY)
    footer_col = imgui.color_convert_float4_to_u32(COLOR.FG_MUTED)

    dl.add_rect_filled((x0, y0), (x0 + box_w, y0 + box_h), bg, rounding=4.0)
    dl.add_rect((x0, y0), (x0 + box_w, y0 + box_h), border, rounding=4.0)

    dl.add_text((x0 + pad, y0 + pad), title_col, "Keyboard")
    chord_x = x0 + pad + label_w + gap
    y = y0 + pad + title_h
    for label, chord in rows:
        dl.add_text((x0 + pad, y), label_col, label)
        dl.add_text((chord_x, y), chord_col, chord)
        y += line_h
    dl.add_text((x0 + pad, y + float(SPACE.XS)), footer_col, footer)
