"""Throwaway visual spike: imgui native tab bar as a replacement for the editor tab row.

Uses the SAME bare glfw + GlfwRenderer + apply_theme path as the real app (no hello_imgui,
which clobbers the style), so the gruvbox theme actually shows.

Run: uv run python scripts/_tabbar_spike.py
Delete when done — not part of the app.
"""

import glfw
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
from OpenGL import GL

from shaderbox.theme import COLOR, apply_theme

# Fake editor tabs mirroring the real EditorTab kinds + states.
# (label, kind, dirty, error)
TABS = [
    ("shader", "shader", False, False),
    ("node script", "node_script", True, False),
    ("script . u_position", "uniform_script", False, True),
    ("script . u_color", "uniform_script", False, False),
    ("SB_perlin_noise.glsl", "lib", False, False),
    ("script . u_velocity", "uniform_script", True, False),
    ("script . u_scale", "uniform_script", False, False),
    ("script . u_rotation", "uniform_script", False, False),
    ("script . u_offset", "uniform_script", False, False),
    ("script . u_phase", "uniform_script", False, False),
]

# Open-flags, one per tab (the native close `x` drives these False).
_open = [True] * len(TABS)
# The script-active toggle (pinned right) for the active tab.
_script_active = True


def _gui() -> None:
    global _script_active

    imgui.set_next_window_pos(imgui.ImVec2(20, 20), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size(imgui.ImVec2(820, 360), imgui.Cond_.first_use_ever)
    imgui.begin("editor tab-row spike")

    imgui.text_colored(
        COLOR.FG_DIM,
        "Native begin_tab_bar: drag to reorder, x to close, unsaved-dot, error tab tinted, overflow scrolls + dropdown.",
    )
    imgui.dummy(imgui.ImVec2(0, 8))

    # Reserve room on the right for the script-active toggle, like the real row.
    toggle_w = 100.0
    bar_w = imgui.get_content_region_avail().x - toggle_w - 8.0

    active_kind = "shader"
    if imgui.begin_child("##bar_region", imgui.ImVec2(bar_w, imgui.get_frame_height() + 8)):
        flags = (
            imgui.TabBarFlags_.reorderable.value
            | imgui.TabBarFlags_.fitting_policy_scroll.value
            | imgui.TabBarFlags_.tab_list_popup_button.value
            | imgui.TabBarFlags_.draw_selected_overline.value
        )
        if imgui.begin_tab_bar("##editor_tabs", flags):
            for i, (label, kind, dirty, error) in enumerate(TABS):
                if not _open[i]:
                    continue
                item_flags = 0
                if dirty:
                    item_flags |= imgui.TabItemFlags_.unsaved_document.value

                tinted = error
                if tinted:
                    imgui.push_style_color(imgui.Col_.tab, COLOR.STATE_ERROR)
                    imgui.push_style_color(imgui.Col_.tab_hovered, COLOR.STATE_ERROR)
                    imgui.push_style_color(imgui.Col_.tab_selected, COLOR.STATE_ERROR)

                opened, keep = imgui.begin_tab_item(f"{label}##tab{i}", _open[i], item_flags)
                if keep is not None:
                    _open[i] = keep
                if opened:
                    active_kind = kind
                    imgui.end_tab_item()

                if tinted:
                    imgui.pop_style_color(3)
            imgui.end_tab_bar()
    imgui.end_child()

    # The script-active toggle, pinned right (only for a script tab) — same role as the real row.
    if active_kind in ("node_script", "uniform_script"):
        imgui.same_line()
        if _script_active:
            imgui.push_style_color(imgui.Col_.button, COLOR.ACCENT_PRIMARY)
            imgui.push_style_color(imgui.Col_.button_hovered, COLOR.ACCENT_ACTIVE)
            imgui.push_style_color(imgui.Col_.button_active, COLOR.ACCENT_ACTIVE)
            imgui.push_style_color(imgui.Col_.text, COLOR.BG_APP)
            if imgui.button("active", imgui.ImVec2(toggle_w, 0)):
                _script_active = False
            imgui.pop_style_color(4)
        else:
            imgui.push_style_color(imgui.Col_.button, COLOR.TRANSPARENT)
            imgui.push_style_color(imgui.Col_.text, COLOR.FG_SECONDARY)
            if imgui.button("inactive", imgui.ImVec2(toggle_w, 0)):
                _script_active = True
            imgui.pop_style_color(2)

    imgui.dummy(imgui.ImVec2(0, 16))
    imgui.separator()
    imgui.text_colored(COLOR.FG_DIM, f"(active tab kind: {active_kind})")
    imgui.text("<-- editor body would render here -->")

    imgui.end()


def main() -> None:
    if not glfw.init():
        raise RuntimeError("glfw init failed")
    window = glfw.create_window(900, 460, "tabbar spike", None, None)
    glfw.make_context_current(window)

    imgui.create_context()
    apply_theme(imgui.get_style())
    renderer = GlfwRenderer(window)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        renderer.process_inputs()
        imgui.new_frame()
        _gui()
        imgui.render()
        GL.glClearColor(COLOR.BG_APP[0], COLOR.BG_APP[1], COLOR.BG_APP[2], 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        renderer.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    renderer.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()
