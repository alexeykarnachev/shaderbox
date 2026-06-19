import time
from collections.abc import Callable

import glfw
import moderngl
import numpy as np
from imgui_bundle import imgui, imgui_ctx
from imgui_bundle import imgui_command_palette as imcmd
from loguru import logger

from shaderbox.app import App, PopupState
from shaderbox.commands import ActiveRegion, CommandId, NodeTab, chord_to_str
from shaderbox.hotkeys import dispatch_commands, process_hotkeys
from shaderbox.logging_setup import configure_logging
from shaderbox.paths import log_dir
from shaderbox.popups.emoji_picker import draw_emoji_picker
from shaderbox.popups.lib_picker import draw_lib_picker
from shaderbox.popups.node_creator import draw_node_creator
from shaderbox.popups.settings import draw_settings
from shaderbox.scripting import MouseState
from shaderbox.tabs import code as code_tab
from shaderbox.tabs import node as node_tab
from shaderbox.tabs import render as render_tab
from shaderbox.tabs import share as share_tab
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_primitives import (
    active_region_outline,
    fps_overlay,
    item_normalized_mouse,
    rendering_overlay,
    toggle_button,
)
from shaderbox.util import adjust_size
from shaderbox.watch import maybe_rebuild_lib_index, reload_node_if_changed
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
    # Skip imgui's Ctrl+Tab window-cycle (would pop a 1-item switcher + 2nd outline);
    # also frees Ctrl+Tab for our Cycle code tab command.
    | imgui.WindowFlags_.no_nav_focus
    # Don't z-order this full-screen window above the floating copilot chat on focus —
    # the chat must stay visible while open. It's submitted after this window to stay on top.
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


def update_and_draw(app: App) -> None:
    # ----------------------------------------------------------------
    # Rebuild the lib index if any lib file changed.
    maybe_rebuild_lib_index(app)

    # ----------------------------------------------------------------
    # Run any GL ops the copilot worker is blocked on (worker->main bridge), EARLY so a
    # freshly recompiled node renders this same frame.
    try:
        app.copilot.drain_bridge()
    except Exception as e:
        logger.exception(f"Error draining copilot bridge: {e}")

    # ----------------------------------------------------------------
    # Reconcile ui_nodes to disk: pick up node dirs added/removed/edited externally. Skipped while a
    # copilot turn is in flight — the worker is mutating nodes/node.json on its own thread, so a sync
    # here would race its writes (it rebaselines via save_ui_node; the post-turn frame syncs the rest).
    if not app.copilot.state.in_flight:
        app.session.sync_nodes_from_disk()

    # ----------------------------------------------------------------
    # Check for per-node shader file changes (root + every prepended lib file).
    for name in list(app.ui_nodes.keys()):
        ui_node = app.ui_nodes[name]
        if not ui_node.node.source.path.exists():
            # Degenerate frame (a node's shader file vanished): nothing draws/swaps, so the cue
            # can't paint — but a parked copilot render must still fire to unblock its waiting
            # worker, or the turn stalls until the op times out.
            app.copilot.bridge.run_deferred_render()
            return
        reload_node_if_changed(app, name, ui_node)

    # ----------------------------------------------------------------
    # Tick the CPU-script engine (feature 040) BEFORE render: hot-reload changed scripts, then
    # compute scripted uniform values for exactly the nodes this frame renders — the current
    # node's preview renders unconditionally (above), and the node-render block renders all nodes
    # when is_render_all_nodes or frame_idx==0 (and no popup). Matching that set keeps a scripted
    # uniform animating identically live and in export.
    app.session.reload_scripts()
    now = glfw.get_time()
    dt = (
        now - app.last_tick_time
        if app.last_tick_time
        else 1.0 / app.app_state.global_target_fps
    )
    app.last_tick_time = now
    tick_nodes = [app.current_node_id] if app.current_node_id in app.ui_nodes else []
    renders_all = app.app_state.is_render_all_nodes or app.frame_idx == 0
    if not app.any_popup_open() and renders_all:
        tick_nodes = list(app.ui_nodes.keys())
    app.session.tick(tick_nodes, now, dt, app.frame_idx, mouse=app.script_mouse)

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
    # A True->False transition means a turn just completed — persist it (a completed turn
    # is the durable unit, so a later crash never loses it) + re-focus the input, which was
    # disabled (unfocusable) for the whole turn so the on-send focus latch was lost.
    if app.copilot_turn_active and not app.copilot.state.in_flight:
        app.copilot.seal_checkpoint()  # finalize the turn's rollback snapshot (feature 020·30)
        app.copilot.save_conversation(app.paths.copilot_conversation_path)
        app.copilot_focus_pending = True
    app.copilot_turn_active = app.copilot.state.in_flight

    # Restore focus to the surface a just-closed modal stole it from (snapshot on open / restore
    # on close). Runs before new_frame so the restored latch is consumed this same frame.
    app.reconcile_popup_focus()

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
    elif app.popup_state == PopupState.NODE_CREATOR:
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
        # Keyboard command dispatch — in-frame (imgui.shortcut() asserts outside a frame),
        # at the top so ESC's editor-defocus reaches code_tab.draw the same frame.
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

        # Splitter hit-test computed BEFORE the editor draws (splitter is at the editor's right
        # edge, width _SPLITTER_W). The press-latch lives on App; code.py reads
        # app.splitter_dragging to neutralize the editor's mouse mid-drag.
        split_origin = imgui.get_cursor_screen_pos()
        on_splitter = (
            split_origin.x + editor_width
            <= imgui.get_io().mouse_pos.x
            <= split_origin.x + editor_width + _SPLITTER_W
        )
        app.update_splitter_drag(on_splitter)

        # Editor column = code-editor child (top) + fixed copilot bar (bottom), grouped so
        # the splitter to their right spans the full height. The bar owns the launcher
        # button (a reserved region the floating chat anchors above, never covers).
        editor_height = split_region.y - _COPILOT_BAR_H
        with imgui_ctx.begin_group():
            with imgui_ctx.begin_child(
                "code_editor",
                size=imgui.ImVec2(editor_width, editor_height),
                child_flags=imgui.ChildFlags_.borders,
                window_flags=imgui.WindowFlags_.no_nav_inputs,
            ):
                # Capture the editor child's screen rect so the chat anchors to the coding
                # area (above the bar), not the whole glfw window.
                ed_pos = imgui.get_window_pos()
                ed_size = imgui.get_window_size()
                app.editor_rect = (ed_pos.x, ed_pos.y, ed_size.x, ed_size.y)
                # Adopt EDITOR as the active region from live focus (App.region_derive_allowed
                # owns the "is this legal now" guard — chord move / chat focus).
                if (
                    imgui.is_window_focused(imgui.FocusedFlags_.child_windows)
                    and app.region_derive_allowed()
                ):
                    app.active_region = ActiveRegion.EDITOR
                # The editor is a focus-stop, not a sticky region: its outline tracks LIVE focus
                # (incl. a pending chord-move into it, matching the unfocused-dim in code.py) so
                # clicking the render canvas dims AND un-outlines it, not leaving a stale outline.
                editor_lit = app.editor_focused or app.editor_focus_requested
                if editor_lit and app.region_outline_visible(ActiveRegion.EDITOR):
                    active_region_outline()
                code_tab.draw(app)

            _draw_copilot_bar(app, editor_width)

        imgui.same_line(spacing=0.0)
        _draw_splitter(app, split_region.x, split_region.y)
        imgui.same_line(spacing=0.0)

        with imgui_ctx.begin_child("app_panel", size=imgui.ImVec2(0, split_region.y)):
            # Freeze the panel (uniform sliders, tab controls, share) while a copilot turn
            # runs — its inputs would race the values the worker reads. The editor has its own
            # read-only lock; the chat (Stop) stays live in its own window.
            imgui.begin_disabled(app.copilot_turn_active)
            try:
                _draw_app_panel(app)
            except Exception as e:
                logger.error(f"Error in app panel: {e}")
                app.notifications.push(
                    f"Error in app panel: {e!s}", COLOR.STATE_ERROR[:3]
                )
            finally:
                imgui.end_disabled()

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

    # Cheatsheet + copilot chat are their OWN top-level windows — drawn after the
    # full-screen main window closes so they aren't obscured by it.
    imgui.push_font(app.font_14, _FONT_14_SIZE)
    cheatsheet.draw(app)
    copilot_chat.draw(app)
    # Apply the frame's requested cursor ONCE, only on change — every glfw.set_cursor re-poke
    # flickers on X11. Surfaces set app.want_cursor; None = default arrow. (Done after all draws
    # so the topmost surface's request wins; reset for next frame.)
    if app.want_cursor is not app.cur_cursor:
        glfw.set_cursor(app.window, app.want_cursor)
        app.cur_cursor = app.want_cursor
    app.want_cursor = None
    # The "Rendering..." cue. Every render encode — Render tab, Share-tab outlet, copilot — runs
    # AFTER this frame's swap + gl.finish (below), the one point the cue is provably on the glass
    # before the encode freezes the loop.
    run_render_now = app.render_defer.ready_to_fire()
    if app.copilot.bridge.render_pending() or app.render_defer.has_request():
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

    # Run every deferred render encode HERE, after the swap, so the cue frame is presented before
    # the encode blocks. glFinish forces the GPU to display it (a queued buffer can't composite
    # while the main thread blocks for seconds inside the encode). Both producers share this one
    # firing point: RenderDefer (Render/Share tabs, a two-frame latch) and the copilot bridge's
    # parked render op (parked this frame in drain_bridge, fired now).
    bridge_render_pending: bool = app.copilot.bridge.render_pending()
    if run_render_now or bridge_render_pending:
        gl.finish()
    if run_render_now:
        request = app.render_defer.fire_and_clear()
        if request is not None:
            request()
    elif app.render_defer.has_request():
        # First sight of the request: hold one frame so the cue paints + swaps next loop.
        app.render_defer.mark_shown()
    else:
        app.render_defer.shown = False
    if bridge_render_pending:
        app.copilot.bridge.run_deferred_render()

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
    # Fixed bar at the bottom of the editor column: editor status chrome (code_tab.
    # draw_chrome) on the left, Copilot CTA on the right. Its OWN region (sibling of the
    # editor child, not an overlay), so the button stays clickable and the chat anchors above.
    with imgui_ctx.begin_child(
        "copilot_bar",
        size=imgui.ImVec2(width, _COPILOT_BAR_H),
        window_flags=imgui.WindowFlags_.no_nav_inputs,
    ):
        code_tab.draw_chrome(app)
        # Right-align the Copilot toggle — same label both states; the style carries state.
        label = "Copilot"
        btn_w = imgui.calc_text_size(label).x + 2.0 * float(SPACE.MD)
        imgui.same_line(width - btn_w - float(SPACE.MD))
        if toggle_button(label, active=app.is_copilot_open):
            app.toggle_copilot_open()


def _draw_splitter(app: App, total_width: float, height: float) -> None:
    imgui.invisible_button("##editor_splitter", imgui.ImVec2(_SPLITTER_W, height))
    if imgui.is_item_hovered() or imgui.is_item_active():
        app.want_cursor = app.resize_ew_cursor
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

        # On compile failure the last-good program stays bound — kept bright; the error
        # surfaces in the editor pane strip.
        img_min = imgui.get_cursor_screen_pos()
        imgui.image_with_bg(
            imgui.ImTextureRef(ui_node.node.canvas.texture.glo),
            image_size=(image_width, image_height),
            uv0=(0, 1),
            uv1=(1, 0),
            tint_col=(1.0, 1.0, 1.0, 1.0),
        )
        # Feed the cursor over the preview into the script tick as ctx.mouse (feature 042).
        # image_with_bg submits no interactive item, so hit-test the captured rect explicitly.
        hit = item_normalized_mouse(
            img_min,
            imgui.ImVec2(img_min.x + image_width, img_min.y + image_height),
        )
        if hit is not None and hit[2]:
            app.script_mouse = MouseState(hit[0], hit[1])
    else:
        # Same height budget as the with-node branch (incl. its gap slack) — an
        # oversized empty-state area overflows the panel into a phantom scrollbar.
        avail = imgui.get_content_region_avail()
        image_width = avail.x
        image_height = max(avail.y - control_panel_min_height - 10, 100)

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
    # Consume region_focus_pending only when this region is active — clearing it
    # unconditionally would eat a request the grid set later this same frame.
    focus_panel = app.region_focus_pending and panel_active
    if focus_panel:
        app.region_focus_pending = False
    # Capture the jump target NOW — the loop rewrites active_node_tab from the visible tab,
    # which would clobber the target before set_selected reads it (takes effect next frame).
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
        # Adopt PANEL as the active region from live focus. See the editor pane.
        if (
            imgui.is_window_focused(imgui.FocusedFlags_.child_windows)
            and app.region_derive_allowed()
        ):
            app.active_region = ActiveRegion.PANEL
        if app.region_outline_visible(ActiveRegion.PANEL):
            active_region_outline()
        with imgui_ctx.begin_tab_bar("node_settings_tabs") as bar:
            if bar:
                visible_tab = app.active_node_tab
                for label, tab_id, draw_tab in _NODE_TABS:
                    # set_selected drives the tab the frame after a Ctrl+digit jump.
                    flags = (
                        imgui.TabItemFlags_.set_selected
                        if tab_select_target == tab_id
                        else imgui.TabItemFlags_.none
                    )
                    with imgui_ctx.begin_tab_item(label, flags=flags) as tab:
                        if tab:
                            visible_tab = tab_id
                            draw_tab(app)
                # Commit the visible tab after the loop so the mid-loop write can't
                # clobber the jump target read above.
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
