import contextlib
import time

import glfw
import imgui
import moderngl
import numpy as np
from loguru import logger
from OpenGL.GL import GLError

from shaderbox.app import App
from shaderbox.hotkeys import process_hotkeys
from shaderbox.popups.node_creator import draw_node_creator
from shaderbox.popups.settings import draw_settings
from shaderbox.tabs import node as node_tab
from shaderbox.tabs import render as render_tab
from shaderbox.tabs import share as share_tab
from shaderbox.ui_utils import adjust_size
from shaderbox.widgets.node_grid import draw_node_preview_grid


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
        preview_size = adjust_size(ui_node.node.canvas.texture.size, width=200)

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
    # Prepare new frame
    imgui.new_frame()
    imgui.push_font(app.font_14)

    # ----------------------------------------------------------------
    # Main window
    window_width, window_height = glfw.get_window_size(app.window)
    imgui.set_next_window_size(window_width, window_height)
    imgui.set_next_window_position(0, 0)
    imgui.begin(
        "ShaderBox - UI",
        flags=imgui.WINDOW_NO_COLLAPSE
        | imgui.WINDOW_ALWAYS_AUTO_RESIZE
        | imgui.WINDOW_NO_TITLE_BAR
        | imgui.WINDOW_NO_SCROLLBAR
        | imgui.WINDOW_NO_SCROLL_WITH_MOUSE,
    )

    control_panel_min_height = 600

    # ----------------------------------------------------------------
    # Main menu bar
    if imgui.button("Open project"):
        app.open_project()

    imgui.same_line()
    if imgui.button("Settings"):
        app.open_settings()

    imgui.same_line()
    imgui.text(f"Global FPS: {round(app.global_fps)}")

    # ----------------------------------------------------------------
    # Current node image
    cursor_pos = imgui.get_cursor_screen_pos()

    if app.current_node_id in app.ui_nodes:
        ui_node = app.ui_nodes[app.current_node_id]
        min_image_height = 100
        max_image_height = max(
            min_image_height,
            imgui.get_content_region_available()[1] - control_panel_min_height - 10,
        )
        max_image_width = imgui.get_content_region_available()[0]
        image_aspect = np.divide(*ui_node.node.canvas.texture.size)
        image_width = min(max_image_width, max_image_height * image_aspect)
        image_height = min(max_image_height, max_image_width / image_aspect)

        has_error = ui_node.node.shader_error != ""
        imgui.image(
            ui_node.node.canvas.texture.glo,
            width=image_width,
            height=image_height,
            uv0=(0, 1),
            uv1=(1, 0),
            tint_color=(0.2, 0.2, 0.2, 1.0) if has_error else (1.0, 1.0, 1.0, 1.0),
            border_color=(0.2, 0.2, 0.2, 1.0),
        )

        if has_error:
            draw_list = imgui.get_window_draw_list()
            text_size = imgui.calc_text_size(ui_node.node.shader_error)
            text_x = cursor_pos[0] + 10.0
            text_y = cursor_pos[1] + 10.0
            draw_list.add_text(
                text_x,
                text_y,
                imgui.color_convert_float4_to_u32(1.0, 0.0, 0.0, 1.0),
                ui_node.node.shader_error,
            )
    else:
        image_width, image_height = imgui.get_content_region_available()
        image_height = max(image_height - control_panel_min_height, 400)

        message = "To create a new node, press Ctrl+N"
        text_size = imgui.calc_text_size(message)
        text_x = cursor_pos[0] + (image_width - text_size[0]) / 2
        text_y = cursor_pos[1] + (image_height - text_size[1]) / 2

        draw_list = imgui.get_window_draw_list()
        draw_list.add_text(
            text_x,
            text_y,
            imgui.color_convert_float4_to_u32(1.0, 1.0, 0.0, 1.0),
            message,
        )

    imgui.set_cursor_screen_pos((cursor_pos[0], cursor_pos[1] + image_height + 10))  # type: ignore

    # ----------------------------------------------------------------
    # Control panel
    region_width, region_height = imgui.get_content_region_available()
    control_panel_height = max(control_panel_min_height, region_height)
    control_panel_width = region_width
    with imgui.begin_child(
        "control_panel",
        width=control_panel_width,
        height=control_panel_height,
        border=False,
    ):
        node_preview_width = control_panel_width / 2.6
        draw_node_preview_grid(app, node_preview_width, control_panel_height)
        imgui.same_line()
        try:
            _draw_node_settings(app)
        except Exception as e:
            logger.error(f"Error in node settings: {e}")
            app.notifications.push(f"Error in node settings: {e!s}", (1.0, 0.0, 0.0))

    # ----------------------------------------------------------------
    # Popups and notifications
    draw_node_creator(app)
    draw_settings(app)

    imgui.push_font(app.font_18)
    app.notifications.update_and_draw()

    imgui.pop_font()
    imgui.pop_font()

    # ----------------------------------------------------------------
    # Finalize and draw the frame
    imgui.end()
    imgui.render()

    glfw.make_context_current(app.window)
    moderngl.get_context().clear_errors()

    gl = moderngl.get_context()
    gl.screen.use()
    gl.clear()

    with contextlib.suppress(GLError):
        app.imgui_renderer.render(imgui.get_draw_data())

    glfw.swap_buffers(app.window)

    app.frame_idx += 1


def _draw_node_settings(app: App) -> None:
    with imgui.begin_child("node_settings", border=True):
        if imgui.begin_tab_bar("node_settings_tabs").opened:
            if imgui.begin_tab_item("Node").selected:  # type: ignore
                node_tab.draw(app)
                imgui.end_tab_item()

            if imgui.begin_tab_item("Render").selected:  # type: ignore
                render_tab.draw(app)
                imgui.end_tab_item()

            if imgui.begin_tab_item("Share").selected:  # type: ignore
                share_tab.draw(app)
                imgui.end_tab_item()

            imgui.end_tab_bar()


def main() -> None:
    app = App()
    run(app)


if __name__ == "__main__":
    main()
