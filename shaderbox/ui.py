import time

import glfw
import moderngl
import numpy as np
from imgui_bundle import imgui, imgui_ctx
from loguru import logger

from shaderbox.app import App, app_data_dir
from shaderbox.hotkeys import process_hotkeys
from shaderbox.popups.node_creator import draw_node_creator
from shaderbox.popups.settings import draw_settings
from shaderbox.tabs import code as code_tab
from shaderbox.tabs import node as node_tab
from shaderbox.tabs import render as render_tab
from shaderbox.tabs import share as share_tab
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_utils import adjust_size
from shaderbox.widgets.node_grid import draw_node_preview_grid

_FONT_14_SIZE = 14.0
_FONT_18_SIZE = 18.0
_EDITOR_MIN_W = 320.0
_APP_PANEL_MIN_W = 360.0
_SPLITTER_W = 6.0
_MAIN_WINDOW_FLAGS = (
    imgui.WindowFlags_.no_collapse
    | imgui.WindowFlags_.always_auto_resize
    | imgui.WindowFlags_.no_title_bar
    | imgui.WindowFlags_.no_scrollbar
    | imgui.WindowFlags_.no_scroll_with_mouse
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
            try:
                ui_node = app.ui_nodes[name]
                new_source = fs_file_path.read_text()
                ui_node.node.release_program(new_source)
                ui_node.mtime = fs_file_mtime
                app.sync_editor_from_disk(name, new_source)
            except Exception as e:
                logger.error(f"Failed to reload node {name}: {e}")
                app.ui_nodes[name].mtime = fs_file_mtime

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
    # Prepare new frame
    imgui.new_frame()
    imgui.push_font(app.font_14, _FONT_14_SIZE)

    # ----------------------------------------------------------------
    # Main window
    window_width, window_height = glfw.get_window_size(app.window)
    imgui.set_next_window_size((window_width, window_height))
    imgui.set_next_window_pos((0, 0))
    with imgui_ctx.begin("ShaderBox - UI", flags=_MAIN_WINDOW_FLAGS):
        # ------------------------------------------------------------
        # Main menu bar
        if imgui.button("Open project"):
            app.open_project()

        imgui.same_line()
        if imgui.button("Settings"):
            app.open_settings()

        imgui.same_line()
        imgui.text(f"Global FPS: {round(app.global_fps)}")

        # ------------------------------------------------------------
        # Left editor / right app split
        split_region = imgui.get_content_region_avail()
        editor_width = max(
            _EDITOR_MIN_W,
            min(
                split_region.x - _APP_PANEL_MIN_W,
                split_region.x * app.app_state.editor_split_fraction,
            ),
        )

        with imgui_ctx.begin_child(
            "code_editor", size=imgui.ImVec2(editor_width, split_region.y)
        ):
            code_tab.draw(app)

        imgui.same_line(spacing=0.0)
        _draw_splitter(app, split_region.x, split_region.y)
        imgui.same_line(spacing=0.0)

        with imgui_ctx.begin_child("app_panel", size=imgui.ImVec2(0, split_region.y)):
            try:
                _draw_app_panel(app)
            except Exception as e:
                logger.error(f"Error in app panel: {e}")
                app.notifications.push(
                    f"Error in app panel: {e!s}", COLOR.STATE_ERROR[:3]
                )

        # ------------------------------------------------------------
        # Popups and notifications
        draw_node_creator(app)
        draw_settings(app)

        imgui.push_font(app.font_18, _FONT_18_SIZE)
        app.notifications.update_and_draw()
        imgui.pop_font()

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


def _draw_splitter(app: App, total_width: float, height: float) -> None:
    imgui.invisible_button("##editor_splitter", imgui.ImVec2(_SPLITTER_W, height))
    if imgui.is_item_hovered() or imgui.is_item_active():
        # glfw cursor — imgui cursors are no-op in this backend (conventions.md ## Known quirks)
        glfw.set_cursor(app.window, app.resize_ew_cursor)
    if imgui.is_item_active():
        delta_x = imgui.get_io().mouse_delta.x
        if delta_x and total_width > 0.0:
            fraction = app.app_state.editor_split_fraction + delta_x / total_width
            app.app_state.editor_split_fraction = max(0.15, min(0.85, fraction))


def _draw_app_panel(app: App) -> None:
    control_panel_min_height = SIZE.PANEL_CTRL_MINH

    # ----------------------------------------------------------------
    # Current node image
    cursor_pos = imgui.get_cursor_screen_pos()

    if app.current_node_id in app.ui_nodes:
        ui_node = app.ui_nodes[app.current_node_id]
        min_image_height = 100
        avail = imgui.get_content_region_avail()
        max_image_height = max(
            min_image_height,
            avail.y - control_panel_min_height - 10,
        )
        max_image_width = avail.x
        image_aspect = np.divide(*ui_node.node.canvas.texture.size)
        image_width = min(max_image_width, max_image_height * image_aspect)
        image_height = min(max_image_height, max_image_width / image_aspect)

        has_error = ui_node.node.shader_error != ""
        imgui.image_with_bg(
            imgui.ImTextureRef(ui_node.node.canvas.texture.glo),
            image_size=(image_width, image_height),
            uv0=(0, 1),
            uv1=(1, 0),
            tint_col=(0.2, 0.2, 0.2, 1.0) if has_error else (1.0, 1.0, 1.0, 1.0),
        )

        if has_error:
            draw_list = imgui.get_window_draw_list()
            text_x = cursor_pos.x + float(SPACE.MD)
            text_y = cursor_pos.y + float(SPACE.MD)
            draw_list.add_text(
                (text_x, text_y),
                imgui.color_convert_float4_to_u32(COLOR.STATE_ERROR),
                ui_node.node.shader_error,
            )
    else:
        avail = imgui.get_content_region_avail()
        image_width = avail.x
        image_height = max(avail.y - control_panel_min_height, 400)

        message = "To create a new node, press Ctrl+N"
        text_size = imgui.calc_text_size(message)
        text_x = cursor_pos.x + (image_width - text_size.x) / 2
        text_y = cursor_pos.y + (image_height - text_size.y) / 2

        draw_list = imgui.get_window_draw_list()
        draw_list.add_text(
            (text_x, text_y),
            imgui.color_convert_float4_to_u32(COLOR.STATE_WARN),
            message,
        )

    imgui.set_cursor_screen_pos(
        (cursor_pos.x, cursor_pos.y + image_height + float(SPACE.MD))
    )

    # ----------------------------------------------------------------
    # Control panel
    region = imgui.get_content_region_avail()
    control_panel_height = max(control_panel_min_height, region.y)
    control_panel_width = region.x
    with imgui_ctx.begin_child(
        "control_panel",
        size=imgui.ImVec2(control_panel_width, control_panel_height),
    ):
        node_preview_width = control_panel_width / 2.6
        draw_node_preview_grid(app, node_preview_width, control_panel_height)
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


def main() -> None:
    log_dir = app_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / "shaderbox_{time}.log", rotation="5 MB", retention=5)

    try:
        app = App()
        run(app)
    except Exception:
        logger.exception("ShaderBox crashed")
        logger.error(f"A crash log was written to {log_dir}")
        raise


if __name__ == "__main__":
    main()
