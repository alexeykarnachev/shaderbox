import glfw
from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.theme import COLOR, EDITOR_UNFOCUSED_ALPHA, SIZE, SPACE
from shaderbox.ui_utils import draw_copyable_text


def draw(app: App) -> None:
    if not (ui_node := app.ui_nodes.get(app.current_node_id)):
        glfw.set_cursor(app.window, None)
        app.editor_focused = False
        imgui.text_colored(COLOR.FG_DIM, "No node selected")
        return

    full_file_path = app.nodes_dir / ui_node.id / "shader.frag.glsl"
    local_file_path = full_file_path.relative_to(app.project_dir)

    if draw_copyable_text(str(local_file_path), copy_value=str(full_file_path)):
        app.notifications.push("Copied to clipboard!")

    if app.is_current_editor_dirty():
        imgui.same_line()
        imgui.text_colored(COLOR.STATE_WARN, "(unsaved)")

    imgui.same_line(spacing=float(SPACE.LG))
    if imgui.button("Open dir", size=(SIZE.BTN_SM_W, 0)):
        app.open_current_node_dir()

    imgui.spacing()

    # The editor's render() FPEs while a popup is open (conventions.md ## Known quirks),
    # so it's simply not drawn then — it reappears when the popup closes.
    if app.any_popup_open():
        glfw.set_cursor(app.window, None)
        app.editor_focused = False
        return

    editor = app.get_editor(ui_node.id)
    settings = app.app_state.editor_settings
    imgui.push_font(app.font_14, float(settings.font_size))

    editor_pos = imgui.get_cursor_screen_pos()
    editor_size = imgui.get_content_region_avail()
    editor_max = imgui.ImVec2(
        editor_pos.x + editor_size.x, editor_pos.y + editor_size.y
    )
    hovering = imgui.is_mouse_hovering_rect(editor_pos, editor_max)

    # Consume the Ctrl+scroll wheel BEFORE render() so the editor doesn't also scroll on it
    io = imgui.get_io()
    if hovering and io.key_ctrl and io.mouse_wheel != 0.0:
        new_size = settings.font_size + int(io.mouse_wheel)
        settings.font_size = max(8, min(48, new_size))
        io.mouse_wheel = 0.0

    # Dim the whole pane while the editor lacks focus. app.editor_focused is last
    # frame's value (computed below, after render) — a one-frame lag on transitions.
    dim = not app.editor_focused
    if dim:
        imgui.push_style_var(imgui.StyleVar_.alpha, EDITOR_UNFOCUSED_ALPHA)

    editor.render("##glsl_editor", size=editor_size)

    if dim:
        imgui.pop_style_var()

    # The editor exposes no is-focused getter, so read imgui's real focus state for
    # this pane (the editor renders its own focusable child window). hotkeys.py reads
    # app.editor_focused to suppress arrow nav while the caret is active.
    app.editor_focused = imgui.is_window_focused(imgui.FocusedFlags_.child_windows)

    # Esc / arrow-nav request defocus (hotkeys.py, app.select_next_current_node).
    # A freshly-rendered editor auto-grabs focus, so the focus must be cleared AFTER
    # render — clearing before is undone by the editor's own first-render focus grab.
    if app.editor_defocus_requested:
        imgui.set_window_focus(None)
        app.editor_defocus_requested = False
        app.editor_focused = False

    # glfw cursor driven directly — editor isn't a hoverable imgui item and imgui cursors are no-op here (conventions.md ## Known quirks)
    glfw.set_cursor(app.window, app.ibeam_cursor if hovering else None)

    imgui.pop_font()
