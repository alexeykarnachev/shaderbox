import time
from collections.abc import Callable
from dataclasses import replace

import glfw
import moderngl
import numpy as np
from imgui_bundle import imgui, imgui_ctx
from imgui_bundle import imgui_command_palette as imcmd
from loguru import logger

from shaderbox.app import App
from shaderbox.commands import ActiveRegion, CommandId, NodeTab, chord_to_str
from shaderbox.hotkeys import dispatch_commands, process_hotkeys
from shaderbox.logging_setup import configure_logging
from shaderbox.paths import log_dir, shader_lib_root
from shaderbox.popups.emoji_picker import draw_emoji_picker
from shaderbox.popups.lib_picker import draw_lib_picker
from shaderbox.popups.node_creator import draw_node_creator
from shaderbox.popups.settings import draw_settings
from shaderbox.shader_lib import is_shader_lib_path
from shaderbox.tabs import code as code_tab
from shaderbox.tabs import node as node_tab
from shaderbox.tabs import render as render_tab
from shaderbox.tabs import share as share_tab
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_models import UINode
from shaderbox.ui_primitives import (
    active_region_outline,
    fps_overlay,
    rendering_overlay,
    toggle_button,
)
from shaderbox.util import adjust_size
from shaderbox.widgets import cheatsheet, copilot_chat
from shaderbox.widgets.node_grid import draw_node_preview_grid

_FONT_14_SIZE = 14.0
_FONT_18_SIZE = 18.0
_EDITOR_MIN_W = 320.0
_APP_PANEL_MIN_W = 360.0
_SPLITTER_W = 6.0
_COPILOT_BAR_H = 34.0
_MAIN_WINDOW_FLAGS = (
    imgui.WindowFlags_.no_collapse
    | imgui.WindowFlags_.always_auto_resize
    | imgui.WindowFlags_.no_title_bar
    | imgui.WindowFlags_.no_scrollbar
    | imgui.WindowFlags_.no_scroll_with_mouse
    | imgui.WindowFlags_.menu_bar
    # Skip the sole top-level window in imgui's built-in Ctrl+Tab window-cycle (which
    # is on regardless of nav_enable_keyboard) — otherwise Ctrl+Tab pops a 1-item
    # window switcher and highlights the whole window (a 2nd outline over ours). This
    # also frees Ctrl+Tab for our CYCLE_REGION command (commands.py).
    | imgui.WindowFlags_.no_nav_focus
    # Don't raise the full-screen main window above the floating copilot chat when the
    # editor/panel is clicked — else focusing a region would z-order the main window
    # over an open-but-unfocused chat and bury it (the chat must stay visible while open,
    # independent of focus). The chat is submitted after this window so it stays on top.
    | imgui.WindowFlags_.no_bring_to_front_on_focus
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
    app.save_imgui_ini()
    app.release()


def _reload_if_changed(app: App, name: str, ui_node: UINode) -> None:
    # Walk every source the last compile depended on (root + included libs).
    # sources[0] is the root by construction (resolve_includes seeds it first);
    # sources[1:] are lib files in first-seen order. Each kind has a different
    # reload action — see comments inline.
    for i, src in enumerate(ui_node.node.compile_unit.sources):
        path = src.path
        if not path.exists():
            continue
        disk_mtime = path.lstat().st_mtime
        if disk_mtime == src.mtime:
            continue

        if i == 0:
            # Root reload: read text, replace, re-sync the open editor session.
            logger.debug(f"Reloading node {name} (root shader changed)")
            try:
                new_text = path.read_text()
                ui_node.node.release_program(new_text)
                ui_node.node.source = replace(ui_node.node.source, mtime=disk_mtime)
                app.sync_editor_from_disk(name, new_text)
            except Exception as e:
                logger.error(f"Failed to reload node {name}: {e}")
                ui_node.node.source = replace(ui_node.node.source, mtime=disk_mtime)
            # `sources` was rebuilt by release_program() via CompileUnit.empty;
            # don't keep iterating the stale list.
            return

        # Lib reload: bump the cached mtime so we don't re-fire next frame, then
        # invalidate the node so the next render's compile re-resolves the
        # include from disk. If a session is open on this lib file AND its text
        # diverges from disk, re-sync it (external edit); if the texts already
        # match, the user just saved in-app — don't clobber their undo history.
        logger.debug(f"Reloading node {name} (lib changed: {path.name})")
        ui_node.node.compile_unit.sources[i] = replace(src, mtime=disk_mtime)
        ui_node.node.invalidate()
        session = app.editor_sessions.get(path)
        if session is not None:
            try:
                new_text = path.read_text()
                if session.editor.get_text() != new_text:
                    session.editor.set_text(new_text)
                    session.saved_undo = session.editor.get_undo_index()
                session.source = replace(
                    session.source, text=new_text, mtime=disk_mtime
                )
            except Exception as e:
                logger.error(f"Failed to sync lib editor for {path}: {e}")


def _maybe_rebuild_lib_index(app: App) -> bool:
    # Detect lib-root changes: file added, removed, or mtime bumped. If anything
    # changed, rebuild the index. Cheap: one glob + N stats per frame.
    # `is_shader_lib_path` keeps `.trash/` (and any future dot-dir) out — must match
    # the filter ShaderLibIndex.build applies, or current vs cached would diverge
    # every frame on trashed files and loop rebuilds forever.
    root = shader_lib_root()
    current: dict[str, float] = {}
    for path in root.glob("**/*.glsl"):
        if not is_shader_lib_path(path, root):
            continue
        try:
            current[str(path)] = path.lstat().st_mtime
        except OSError:
            continue
    cached = {str(p): s.mtime for p, s in app.shader_lib_index.sources.items()}
    if current == cached:
        return False
    app.rebuild_shader_lib_index()
    # Any node whose last compile pulled in a lib file might now need a fresh
    # compile (the function it referenced may have changed body or disappeared).
    # We invalidate every node that has lib files in its sources; the next render
    # of each will recompile with the new index.
    for ui_node in app.ui_nodes.values():
        if len(ui_node.node.compile_unit.sources) > 1:
            ui_node.node.invalidate()
    return True


def update_and_draw(app: App) -> None:
    # ----------------------------------------------------------------
    # Rebuild the lib index if any lib file changed (added / removed / edited).
    # Walks shader_lib_root each frame; for small libs the cost is microseconds. If the
    # index changed we invalidate every node that depended on a lib file so its
    # next render recompiles against the fresh index.
    _maybe_rebuild_lib_index(app)

    # ----------------------------------------------------------------
    # Run any GL ops the copilot worker is blocked on (the worker->main bridge), EARLY
    # so a freshly recompiled node renders this same frame. Each op is isolated inside
    # drain(); the wrapper is belt-and-suspenders for the frame loop.
    try:
        app.copilot.drain_bridge()
    except Exception as e:
        logger.exception(f"Error draining copilot bridge: {e}")

    # ----------------------------------------------------------------
    # Check for per-node shader file changes (root + every prepended lib file).
    for name in list(app.ui_nodes.keys()):
        ui_node = app.ui_nodes[name]
        if not ui_node.node.source.path.exists():
            return
        _reload_if_changed(app, name, ui_node)

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
    # Drain copilot worker events into the chat render-state (no GL; single-writer).
    app.copilot.pump_events()
    # The editor lock follows the turn: set on send, lifted when the turn ends (§11).
    # A True->False transition means a turn just completed — persist it (feature 022: the
    # completed turn is the durable unit, so a later crash never loses it). project_dir is
    # always set here (the frame loop only runs after the first _init).
    if app.copilot_turn_active and not app.copilot.state.in_flight:
        app.copilot.save_conversation(app.copilot_conversation_path)
    app.copilot_turn_active = app.copilot.state.in_flight

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
        # Keyboard command dispatch — in-frame (imgui.shortcut() asserts outside
        # a frame), at the top of the block so ESC's editor-defocus is consumed
        # by code_tab.draw the same frame.
        dispatch_commands(app)

        # ------------------------------------------------------------
        # Main menu bar
        _draw_menu_bar(app)

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

        # Same-frame splitter-drag test (computed BEFORE the editor draws): the
        # splitter sits at x = editor's right edge, width _SPLITTER_W. Latch on the
        # press over that rect, hold while the button is down. code.py reads
        # app.splitter_dragging to neutralize the editor's mouse during the drag.
        io = imgui.get_io()
        split_origin = imgui.get_cursor_screen_pos()
        on_splitter = (
            split_origin.x + editor_width
            <= io.mouse_pos.x
            <= split_origin.x + editor_width + _SPLITTER_W
        )
        if imgui.is_mouse_clicked(imgui.MouseButton_.left):
            app._splitter_press_on_splitter = on_splitter
        if not imgui.is_mouse_down(imgui.MouseButton_.left):
            app._splitter_press_on_splitter = False
        app.splitter_dragging = app._splitter_press_on_splitter

        # The editor column = a code-editor child (top) + a fixed copilot bar (bottom),
        # stacked in one group so the splitter to their right spans the full height. The
        # bar OWNS the launcher button (always visible, always clickable — a reserved
        # region, never overlapped by the floating chat, which anchors above it).
        editor_height = split_region.y - _COPILOT_BAR_H
        with imgui_ctx.begin_group():
            with imgui_ctx.begin_child(
                "code_editor",
                size=imgui.ImVec2(editor_width, editor_height),
                child_flags=imgui.ChildFlags_.borders,
                window_flags=imgui.WindowFlags_.no_nav_inputs,
            ):
                # Capture the editor child's screen rect so the chat window anchors to the
                # coding area (above the bar), not the whole glfw window. get_window_pos/
                # size inside the child give its real screen rect (correct live).
                ed_pos = imgui.get_window_pos()
                ed_size = imgui.get_window_size()
                app.editor_rect = (ed_pos.x, ed_pos.y, ed_size.x, ed_size.y)
                # active_region is corrected from LIVE focus (so a mouse click that moves
                # focus updates it) — but NOT while a chord move is in flight (focus still
                # reads as the old region for a frame; correcting then would revert the
                # chord target + break cycling), and NOT while the chat owns focus (the
                # chat is a separate top-level window; the editor must yield to it).
                if (
                    imgui.is_window_focused(imgui.FocusedFlags_.child_windows)
                    and not app.focus_move_in_flight()
                    and not app.copilot_focused
                ):
                    app.active_region = ActiveRegion.EDITOR
                # Not while a modal is open: the outline is on the foreground draw list
                # (immune to window clip) so it would render OVER the popup.
                if (
                    app.active_region == ActiveRegion.EDITOR
                    and not app.any_popup_open()
                    and not app.copilot_focused
                ):
                    active_region_outline()
                code_tab.draw(app)

            _draw_copilot_bar(app, editor_width)

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
        draw_emoji_picker(app)
        draw_lib_picker(app)

        if app.is_palette_open:
            app.is_palette_open = imcmd.command_palette_window(
                "CommandPalette", app.is_palette_open
            )

        imgui.push_font(app.font_18, _FONT_18_SIZE)
        app.notifications.update_and_draw()
        imgui.pop_font()

    imgui.pop_font()

    # The cheatsheet + the copilot chat are their OWN top-level windows — drawn after
    # the full-screen main window closes so they aren't obscured by it.
    imgui.push_font(app.font_14, _FONT_14_SIZE)
    cheatsheet.draw(app)
    copilot_chat.draw(app)
    # The "Rendering..." cue. Two producers: the copilot bridge (a render op held one frame before
    # the encode, feature 020·20 D3) and the Render-tab button (a deferred main-thread encode). The
    # Render-tab encode runs AFTER this frame's swap (below) so its cue is provably on the glass
    # before the encode freezes the loop; the copilot bridge handles its own deferral.
    run_render_now = app.render_request is not None and app.render_request_shown
    if app.copilot.bridge.render_pending() or app.render_request is not None:
        rendering_overlay("Rendering... the app pauses while it encodes.")
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

    # Run the Render-tab encode HERE, after the swap, so the cue frame is presented before the
    # encode blocks. glFinish forces the GPU to actually display it (a queued buffer can otherwise
    # never composite while the main thread blocks for seconds inside the encode).
    if run_render_now:
        gl.finish()
        request = app.render_request
        app.render_request = None
        app.render_request_shown = False
        if request is not None:
            request()
    elif app.render_request is not None:
        # First sight of the request: hold it one frame so the cue paints + swaps above next loop.
        app.render_request_shown = True
    else:
        # No request (or it was cleared externally): re-arm so the next one gets its own hold frame.
        app.render_request_shown = False

    app.frame_idx += 1


def _hint(app: App, command_id: CommandId) -> str:
    return chord_to_str(app.effective_bindings[command_id])


def _draw_menu_bar(app: App) -> None:
    with imgui_ctx.begin_menu_bar() as bar:
        if not bar:
            return
        with imgui_ctx.begin_menu("File") as file_menu:
            if file_menu:
                if imgui.menu_item("New node", _hint(app, CommandId.NEW_NODE), False)[
                    0
                ]:
                    app.open_node_creator()
                if imgui.menu_item(
                    "Open project...", _hint(app, CommandId.OPEN_PROJECT), False
                )[0]:
                    app.open_project()
                imgui.separator()
                if imgui.menu_item("Quit", _hint(app, CommandId.QUIT), False)[0]:
                    glfw.set_window_should_close(app.window, True)
        with imgui_ctx.begin_menu("Edit") as edit_menu:
            if (
                edit_menu
                and imgui.menu_item(
                    "Settings...", _hint(app, CommandId.OPEN_SETTINGS), False
                )[0]
            ):
                app.open_settings()
        with imgui_ctx.begin_menu("Library") as lib_menu:
            if (
                lib_menu
                and imgui.menu_item(
                    "Browse...", _hint(app, CommandId.OPEN_LIB_PICKER), False
                )[0]
            ):
                app.open_shader_lib_picker()


def _draw_copilot_bar(app: App, width: float) -> None:
    # A fixed bar at the bottom of the editor column. It holds the editor's status chrome
    # (file path, dirty/compiled state, Open dir, back links — code_tab.draw_chrome) on
    # the left and the Copilot CTA on the right. The bar is its OWN region (sibling of the
    # editor child, not an overlay), so the button is always visible + clickable and the
    # floating chat anchors ABOVE it without ever covering it.
    with imgui_ctx.begin_child(
        "copilot_bar",
        size=imgui.ImVec2(width, _COPILOT_BAR_H),
        window_flags=imgui.WindowFlags_.no_nav_inputs,
    ):
        code_tab.draw_chrome(app)
        # Right-align the Copilot toggle on the same row — same label both states; the
        # style carries the state (filled accent when open, bordered ghost when closed).
        label = "Copilot"
        btn_w = imgui.calc_text_size(label).x + 2.0 * float(SPACE.MD)
        imgui.same_line(width - btn_w - float(SPACE.MD))
        if toggle_button(label, active=app.is_copilot_open):
            app.toggle_copilot_open()


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

        # On compile failure the last-good program stays bound, so the render is a
        # live reference — kept bright; the error surfaces in the editor pane strip.
        imgui.image_with_bg(
            imgui.ImTextureRef(ui_node.node.canvas.texture.glo),
            image_size=(image_width, image_height),
            uv0=(0, 1),
            uv1=(1, 0),
            tint_col=(1.0, 1.0, 1.0, 1.0),
        )
    else:
        avail = imgui.get_content_region_avail()
        image_width = avail.x
        image_height = max(avail.y - control_panel_min_height, 400)

        message = (
            f"To create a new node, press "
            f"{chord_to_str(app.effective_bindings[CommandId.NEW_NODE])}"
        )
        text_size = imgui.calc_text_size(message)
        text_x = cursor_pos.x + (image_width - text_size.x) / 2
        text_y = cursor_pos.y + (image_height - text_size.y) / 2

        draw_list = imgui.get_window_draw_list()
        draw_list.add_text(
            (text_x, text_y),
            imgui.color_convert_float4_to_u32(COLOR.STATE_WARN),
            message,
        )

    if app.current_node_id in app.ui_nodes:
        app.fps_details_open = fps_overlay(
            anchor_x=cursor_pos.x + image_width,
            anchor_y=cursor_pos.y,
            fps=round(app.global_fps),
            target_fps=app.app_state.global_target_fps,
            is_open=app.fps_details_open,
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


_NODE_TABS: list[tuple[str, NodeTab, Callable[[App], None]]] = [
    ("Node", NodeTab.NODE, node_tab.draw),
    ("Render", NodeTab.RENDER, render_tab.draw),
    ("Share", NodeTab.SHARE, share_tab.draw),
]


def _draw_node_settings(app: App) -> None:
    panel_active = app.active_region == ActiveRegion.PANEL
    # Each region consumes (reads + clears) its OWN region_focus_pending — clearing
    # it here unconditionally would eat a request the GRID set after it already drew
    # this frame (e.g. select_node re-latching grid focus), so the grid never sees it.
    focus_panel = app.region_focus_pending and panel_active
    if focus_panel:
        app.region_focus_pending = False
    # Capture the jump target NOW — the loop rewrites active_node_tab from the
    # visible tab, which would otherwise clobber the target before set_selected
    # reads it (set_selected only takes effect next frame).
    tab_select_target = app.active_node_tab if app.node_tab_select_pending else None
    app.node_tab_select_pending = False

    if focus_panel:
        imgui.set_next_window_focus()
    panel_flags = (
        imgui.WindowFlags_.none if panel_active else imgui.WindowFlags_.no_nav_inputs
    )
    with imgui_ctx.begin_child(
        "node_settings",
        child_flags=imgui.ChildFlags_.borders,
        window_flags=panel_flags,
    ):
        # active_region corrected from live focus (mouse clicks) except during a chord
        # move, and except while the chat owns focus (it's a separate window the regions
        # must yield to). See the editor pane.
        if (
            imgui.is_window_focused(imgui.FocusedFlags_.child_windows)
            and not app.focus_move_in_flight()
            and not app.copilot_focused
        ):
            app.active_region = ActiveRegion.PANEL
        if (
            app.active_region == ActiveRegion.PANEL
            and not app.any_popup_open()
            and not app.copilot_focused
        ):
            active_region_outline()
        with imgui_ctx.begin_tab_bar("node_settings_tabs") as bar:
            if bar:
                visible_tab = app.active_node_tab
                for label, tab_id, draw_tab in _NODE_TABS:
                    # set_selected drives the tab on the next frame after a Ctrl+digit
                    # jump; imgui then remembers the selection.
                    flags = (
                        imgui.TabItemFlags_.set_selected
                        if tab_select_target == tab_id
                        else imgui.TabItemFlags_.none
                    )
                    with imgui_ctx.begin_tab_item(label, flags=flags) as tab:
                        if tab:
                            visible_tab = tab_id
                            draw_tab(app)
                # The visible tab IS the selected one — commit after the loop so the
                # mid-loop write can't clobber the jump target read above.
                app.active_node_tab = visible_tab


def main() -> None:
    configure_logging()

    try:
        app = App()
        run(app)
    except Exception:
        logger.exception("ShaderBox crashed")
        logger.error(f"A crash log was written to {log_dir()}")
        raise


if __name__ == "__main__":
    main()
