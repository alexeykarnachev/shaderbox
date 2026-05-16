import time

import glfw
import moderngl
import numpy as np
from imgui_bundle import imgui, imgui_ctx
from loguru import logger

from shaderbox.app import App
from shaderbox.hotkeys import process_hotkeys
from shaderbox.popups.node_creator import draw_node_creator
from shaderbox.popups.settings import draw_settings
from shaderbox.tabs import node as node_tab
from shaderbox.tabs import render as render_tab
from shaderbox.tabs import share as share_tab
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_models import UINode
from shaderbox.ui_utils import adjust_size
from shaderbox.widgets.node_grid import draw_node_preview_grid

_FONT_14_SIZE = 14.0
_FONT_18_SIZE = 18.0
_MAIN_WINDOW_FLAGS = (
    imgui.WindowFlags_.no_collapse
    | imgui.WindowFlags_.no_title_bar
    | imgui.WindowFlags_.no_scrollbar
    | imgui.WindowFlags_.no_scroll_with_mouse
    | imgui.WindowFlags_.no_resize
    | imgui.WindowFlags_.no_move
)


def run(app: App) -> None:
    while not glfw.window_should_close(app.window):
        start_time = glfw.get_time()

        update_and_draw(app)

        elapsed_time = glfw.get_time() - start_time
        target_fps = app.app_state.global_target_fps
        time.sleep(max(0.0, 1.0 / target_fps - elapsed_time))

        fps = 1.0 / (glfw.get_time() - start_time)
        if app.global_fps <= 0.0:
            app.global_fps = fps
        else:
            app.global_fps = 0.95 * app.global_fps + 0.05 * fps

    app.save()
    app.release()


def update_and_draw(app: App) -> None:
    # ----------------------------------------------------------------
    # Check for shader file changes and reload nodes
    for name in list(app.ui_nodes.keys()):
        fs_file_path = app.nodes_dir / name / "shader.frag.glsl"

        if not fs_file_path.exists():
            return

        fs_file_mtime = fs_file_path.lstat().st_mtime
        if fs_file_mtime != app.ui_nodes[name].mtime:
            logger.info(f"Reloading node {name} due to shader file change")
            ui_node = app.ui_nodes[name]
            ui_node.node.release_program(fs_file_path.read_text())
            ui_node.mtime = fs_file_mtime

    # ----------------------------------------------------------------
    # Render previews
    if app.current_node_id in app.ui_nodes:
        ui_node = app.ui_nodes[app.current_node_id]
        preview_size = adjust_size(
            ui_node.node.canvas.texture.size, width=SIZE.PREVIEW_W
        )

        app.preview_canvas.set_size(preview_size)
        ui_node.node.render(canvas=app.preview_canvas)

        try:
            share_tab.update(app)
        except Exception as e:
            logger.exception(f"Error in share-tab update: {e}")

    # ----------------------------------------------------------------
    # Render nodes
    if not app.any_popup_open():
        for ui_node in app.ui_nodes.values():
            if (
                app.app_state.is_render_all_nodes
                or ui_node == app.ui_nodes.get(app.current_node_id)
                or app.frame_idx == 0
            ):
                ui_node.node.render()
    elif app.is_node_creator_open:
        for ui_node in app.ui_node_templates.values():
            ui_node.node.render()

    # ----------------------------------------------------------------
    # Process hotkeys
    process_hotkeys(app)

    # ----------------------------------------------------------------
    # Age notification stack (rendered later inside the status bar)
    app.notifications.update()

    # ----------------------------------------------------------------
    # Prepare new frame
    imgui.new_frame()
    imgui.push_font(app.font_14, _FONT_14_SIZE)

    # ----------------------------------------------------------------
    # Main window
    window_width, window_height = glfw.get_window_size(app.window)
    imgui.set_next_window_size((window_width, window_height))
    imgui.set_next_window_pos((0, 0))
    with imgui_ctx.begin("ShaderBox - UI", flags=_MAIN_WINDOW_FLAGS):
        _draw_topbar(app)
        _draw_main_split(app)
        _draw_statusbar(app)

        draw_node_creator(app)
        draw_settings(app)

    imgui.pop_font()

    # ----------------------------------------------------------------
    # Finalize and draw the frame
    imgui.render()

    glfw.make_context_current(app.window)
    moderngl.get_context().clear_errors()

    gl = moderngl.get_context()
    gl.screen.use()
    gl.clear()

    app.imgui_renderer.render(imgui.get_draw_data())

    glfw.swap_buffers(app.window)

    app.frame_idx += 1


# ============================================================================
# Top bar (~32 px)
# ============================================================================


def _draw_topbar(app: App) -> None:
    if imgui.button("Open project"):
        app.open_project()

    imgui.same_line()
    if imgui.button("Settings"):
        app.open_settings()

    # Right-aligned tweaks placeholder (feature 009 will wire this up)
    imgui.same_line(imgui.get_content_region_avail().x - SIZE.BTN_SM_W)
    if imgui.button("Tweaks", size=(SIZE.BTN_SM_W, 0)):
        logger.info("Tweaks toggle — not yet implemented (feature 009)")


# ============================================================================
# Main split: editor placeholder | right pane
# ============================================================================


def _draw_main_split(app: App) -> None:
    avail = imgui.get_content_region_avail()
    gap_x = imgui.get_style().item_spacing.x
    statusbar_total_h = SIZE.STATUSBAR + SPACE.SM
    split_h = avail.y - statusbar_total_h
    if split_h < 1.0:
        return
    left_w = (avail.x - gap_x) / 2.0
    right_w = avail.x - left_w - gap_x

    with imgui_ctx.begin_child(
        "left_pane",
        size=imgui.ImVec2(left_w, split_h),
        child_flags=imgui.ChildFlags_.borders,
    ):
        _draw_editor_placeholder(app)

    imgui.same_line()

    with imgui_ctx.begin_child(
        "right_pane",
        size=imgui.ImVec2(right_w, split_h),
    ):
        _draw_render_card(app)
        _draw_node_panel(app)


def _draw_editor_placeholder(app: App) -> None:
    avail = imgui.get_content_region_avail()
    msg_a = "GLSL editor"
    msg_b = "coming in feature 006"
    imgui.push_font(app.font_18, _FONT_18_SIZE)
    sz_a = imgui.calc_text_size(msg_a)
    imgui.pop_font()
    sz_b = imgui.calc_text_size(msg_b)
    cx = avail.x / 2.0
    cy = avail.y / 2.0

    imgui.set_cursor_pos((cx - sz_a.x / 2.0, cy - sz_a.y))
    imgui.push_font(app.font_18, _FONT_18_SIZE)
    imgui.text_colored(COLOR.FG_PRIMARY, msg_a)
    imgui.pop_font()

    imgui.set_cursor_pos((cx - sz_b.x / 2.0, cy + float(SPACE.SM)))
    imgui.text_colored(COLOR.FG_DIM, msg_b)


# ============================================================================
# Render card (capped at SIZE.RENDER_MAX_H + padding)
# ============================================================================


def _draw_render_card(app: App) -> None:
    card_h = SIZE.RENDER_MAX_H + 2 * SPACE.MD
    imgui.push_style_color(imgui.Col_.child_bg, (0.0, 0.0, 0.0, 1.0))
    with imgui_ctx.begin_child(
        "render_card",
        size=imgui.ImVec2(0.0, card_h),
        child_flags=imgui.ChildFlags_.borders,
    ):
        ui_node = app.ui_nodes.get(app.current_node_id)
        if ui_node is None:
            _draw_render_empty_state(app)
        else:
            _draw_render_image(ui_node)
    imgui.pop_style_color()


def _draw_render_empty_state(app: App) -> None:
    avail = imgui.get_content_region_avail()
    msg = "Press  Ctrl+N  to create a new node"
    imgui.push_font(app.font_18, _FONT_18_SIZE)
    sz = imgui.calc_text_size(msg)
    imgui.set_cursor_pos((avail.x / 2.0 - sz.x / 2.0, avail.y / 2.0 - sz.y / 2.0))
    imgui.text_colored(COLOR.FG_DIM, msg)
    imgui.pop_font()


def _draw_render_image(ui_node: UINode) -> None:
    canvas = ui_node.node.canvas
    tex = canvas.texture
    shader_error = ui_node.node.shader_error
    has_error = shader_error != ""

    avail = imgui.get_content_region_avail()
    image_aspect = np.divide(*tex.size)
    img_w = min(avail.x, avail.y * image_aspect)
    img_h = min(avail.y, avail.x / image_aspect)

    # Center the image inside the card
    cursor = imgui.get_cursor_pos()
    imgui.set_cursor_pos(
        (cursor.x + (avail.x - img_w) / 2.0, cursor.y + (avail.y - img_h) / 2.0)
    )

    image_pos = imgui.get_cursor_screen_pos()
    imgui.image_with_bg(
        imgui.ImTextureRef(tex.glo),
        image_size=(img_w, img_h),
        uv0=(0, 1),
        uv1=(1, 0),
        tint_col=(0.2, 0.2, 0.2, 1.0) if has_error else (1.0, 1.0, 1.0, 1.0),
    )

    if has_error:
        # Custom-drawn error overlay (replaced by a real banner in feature 008).
        draw_list = imgui.get_window_draw_list()
        draw_list.add_text(
            (image_pos.x + float(SPACE.MD), image_pos.y + float(SPACE.MD)),
            imgui.color_convert_float4_to_u32(COLOR.STATE_ERROR),
            shader_error,
        )


# ============================================================================
# Node panel (node grid + 3-tab bar) — preserved from pre-redesign for PR 4
# ============================================================================


def _draw_node_panel(app: App) -> None:
    region = imgui.get_content_region_avail()
    panel_h = region.y - SPACE.SM
    panel_w = region.x

    with imgui_ctx.begin_child("node_panel", size=imgui.ImVec2(panel_w, panel_h)):
        grid_w = panel_w / 2.6
        draw_node_preview_grid(app, grid_w, panel_h)
        imgui.same_line()
        try:
            _draw_node_settings(app)
        except Exception as e:
            logger.error(f"Error in node settings: {e}")
            app.notifications.push(
                f"Error in node settings: {e!s}", COLOR.STATE_ERROR[:3]
            )


def _draw_node_settings(app: App) -> None:
    with (
        imgui_ctx.begin_child("node_settings", child_flags=imgui.ChildFlags_.borders),
        imgui_ctx.begin_tab_bar("node_settings_tabs") as bar,
    ):
        if bar:
            with imgui_ctx.begin_tab_item("Node") as tab:
                if tab:
                    node_tab.draw(app)

            with imgui_ctx.begin_tab_item("Render") as tab:
                if tab:
                    render_tab.draw(app)

            with imgui_ctx.begin_tab_item("Share") as tab:
                if tab:
                    share_tab.draw(app)


# ============================================================================
# Status bar (~24 px, pinned to the bottom of the main window)
# ============================================================================


def _draw_statusbar(app: App) -> None:
    imgui.push_style_color(imgui.Col_.child_bg, COLOR.BG_SURFACE)
    with imgui_ctx.begin_child(
        "statusbar",
        size=imgui.ImVec2(0.0, float(SIZE.STATUSBAR)),
    ):
        _draw_statusbar_state(app)
        imgui.same_line()
        _statusbar_sep()
        imgui.same_line()
        _draw_statusbar_node(app)
        imgui.same_line()
        _statusbar_sep()
        imgui.same_line()
        _draw_statusbar_resolution(app)
        imgui.same_line()
        _statusbar_sep()
        imgui.same_line()
        _draw_statusbar_fps(app)
        imgui.same_line()
        _statusbar_sep()
        imgui.same_line()
        _draw_statusbar_glsl()

        # Toast slot — the head of the notification stack, inline
        if (toast := app.notifications.head) is not None:
            imgui.same_line()
            _statusbar_sep()
            imgui.same_line()
            text, color = toast
            imgui.text_colored((*color, 1.0), text)

        # Right-aligned: shortcut legend
        legend_w = SIZE.BTN_MD_W * 3
        imgui.same_line(imgui.get_content_region_avail().x - legend_w)
        _draw_shortcut_legend()
    imgui.pop_style_color()


def _draw_statusbar_state(app: App) -> None:
    ui_node = app.ui_nodes.get(app.current_node_id)
    if ui_node is None:
        imgui.text_colored(COLOR.FG_DIM, "no node")
    elif ui_node.node.shader_error:
        imgui.text_colored(COLOR.STATE_ERROR, "compile error")
    else:
        imgui.text_colored(COLOR.STATE_OK, "ready")


def _draw_statusbar_node(app: App) -> None:
    imgui.text_colored(COLOR.FG_DIM, "node")
    imgui.same_line()
    ui_node = app.ui_nodes.get(app.current_node_id)
    name = ui_node.ui_state.ui_name if ui_node else "—"
    imgui.text_colored(COLOR.FG_PRIMARY, name)


def _draw_statusbar_resolution(app: App) -> None:
    ui_node = app.ui_nodes.get(app.current_node_id)
    if ui_node is None:
        imgui.text_colored(COLOR.FG_DIM, "—")
        return
    w, h = ui_node.node.canvas.texture.size
    imgui.text_colored(COLOR.FG_PRIMARY, f"{w}x{h}")


def _draw_statusbar_fps(app: App) -> None:
    imgui.text_colored(COLOR.FG_PRIMARY, f"{app.global_fps:.1f}")
    imgui.same_line()
    imgui.text_colored(COLOR.FG_DIM, "fps")


def _statusbar_sep() -> None:
    imgui.text_colored(COLOR.FG_DIM, "·")


def _draw_statusbar_glsl() -> None:
    imgui.text_colored(COLOR.FG_DIM, "GLSL")
    imgui.same_line()
    imgui.text_colored(COLOR.FG_PRIMARY, "4.60 core")


def _draw_shortcut_legend() -> None:
    chips = [
        ("Ctrl+N", "new"),
        ("Ctrl+S", "save"),
        ("Ctrl+E", "edit"),
        ("Ctrl+D", "del"),
        ("←→", "nav"),
        ("Alt+S", "setup"),
    ]
    for i, (kbd, label) in enumerate(chips):
        if i:
            imgui.same_line(0.0, float(SPACE.SM))
        imgui.text_colored(COLOR.ACCENT_PRIMARY, kbd)
        imgui.same_line(0.0, float(SPACE.XS))
        imgui.text_colored(COLOR.FG_DIM, label)


def main() -> None:
    app = App()
    run(app)


if __name__ == "__main__":
    main()
