import keyword
from pathlib import Path

from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.core import Pass
from shaderbox.editor.ffi import EDITOR_RESOURCES_DIR, Editor, Mode
from shaderbox.editor.render import (
    EditorPanel,
    EditorRenderer,
    render_state,
    should_redraw,
)
from shaderbox.editor_types import EditorTab, HoverMark, JumpRequest
from shaderbox.paths import pass_name_of
from shaderbox.shader_errors import ShaderError
from shaderbox.theme import COLOR, EDITOR_UNFOCUSED_ALPHA, SIZE, SPACE, fade
from shaderbox.ui_primitives import draw_copyable_text
from shaderbox.util import format_auto_value

_MAX_ERROR_ROWS = 3

_MODE_BADGES: dict[Mode, tuple[str, tuple[float, float, float, float]]] = {
    Mode.NORMAL: ("NORMAL", COLOR.FG_DIM),
    Mode.INSERT: ("INSERT", COLOR.STATE_OK),
    Mode.VISUAL: ("VISUAL", COLOR.SELECT),
    Mode.VISUAL_LINE: ("V-LINE", COLOR.SELECT),
}


def _is_script_tab(tab: EditorTab | None) -> bool:
    return tab is not None and tab.kind == "script"


def tab_label(app: App, tab: EditorTab) -> str:
    # The display label for a tab (048): document-derived so two documents' tabs are distinguishable
    # ("<document> (shader)" / "<document> (script)"), a lib by "library - <file>". The on-disk filename is
    # the same constant for every document, so the bare name can't tell tabs apart. Falls back to a short
    # id slice when the document has no name. The imgui ##id keys on the stable path/index, NOT this label.
    #
    # A MULTI-pass document names the pass instead of "shader" (065) — otherwise its tabs are all
    # "<document> (shader)" and tell each other apart by nothing. Taken from the tab's own PATH,
    # which is its identity, so the label cannot disagree with the file the tab opens and a rename
    # carries it along for free. A single-pass document keeps "(shader)": there is nothing to
    # disambiguate, and "(main)" would say less.
    if tab.kind == "lib":
        return f"library - {tab.path.stem}"
    ui_document = app.ui_documents.get(tab.document_id)
    document_name = (
        ui_document.ui_state.ui_name if ui_document else ""
    ) or tab.document_id[:8]
    if tab.kind == "script":
        return f"{document_name} (script)"
    multi_pass = ui_document is not None and len(ui_document.document.passes) > 1
    suffix = pass_name_of(tab.path) if multi_pass else "shader"
    return f"{document_name} ({suffix})"


def _draw_tab_row(app: App) -> None:
    # The editor's tab row (047): a native imgui tab bar — drag-reorder, x-close, unsaved dot, an
    # error-tinted tab, and overflow scroll + a ▾ list popup. Labels come from tab_label, the one
    # funnel this row and the chrome both use.
    # A genuine click is read back into active_tab_index; a PROGRAMMATIC switch (glyph open /
    # document-select / lib-jump / close) DRIVES imgui's selection via set_selected — imgui ignores a
    # model-side index change otherwise and reverts to the old tab. The target is read BEFORE the
    # loop so the mid-loop read-back can't clobber it (the ui.py document-settings-bar pattern).
    select_target = app.active_tab_index if app.tab_select_pending else None
    app.tab_select_pending = False
    flags = (
        imgui.TabBarFlags_.reorderable.value
        | imgui.TabBarFlags_.fitting_policy_scroll.value
        | imgui.TabBarFlags_.tab_list_popup_button.value
        | imgui.TabBarFlags_.draw_selected_overline.value
    )
    close_index: int | None = None
    if imgui.begin_tab_bar("##editor_tabs", flags):
        for i, tab in enumerate(app.editor_tabs):
            item_flags = 0
            if app.is_tab_dirty(tab):
                item_flags |= imgui.TabItemFlags_.unsaved_document.value
            if i == select_target:
                item_flags |= imgui.TabItemFlags_.set_selected.value
            tinted = _is_script_tab(tab) and _tab_has_error(app, tab)
            if tinted:
                imgui.push_style_color(imgui.Col_.tab, COLOR.STATE_ERROR)
                imgui.push_style_color(imgui.Col_.tab_hovered, COLOR.STATE_ERROR)
                imgui.push_style_color(imgui.Col_.tab_selected, COLOR.STATE_ERROR)
            opened, keep = imgui.begin_tab_item(
                f"{tab_label(app, tab)}##tab{i}", True, item_flags
            )
            if keep is not None and not keep:
                close_index = i
            if opened:
                # Read back a genuine user click — but NOT while we're driving a programmatic
                # switch (imgui still reports the old tab opened on the drive frame).
                if select_target is None and i != app.active_tab_index:
                    app.set_active_tab(i)
                imgui.end_tab_item()
            if tinted:
                imgui.pop_style_color(3)
        imgui.end_tab_bar()
    if close_index is not None:
        app.close_tab(close_index)


def _tab_has_error(app: App, tab: EditorTab) -> bool:
    return app.session.script_has_error(tab.document_id)


def _pass_for_tab(app: App, tab: EditorTab) -> Pass | None:
    """The pass a shader tab is editing, matched by PATH — not the document's output.

    A document has one file per pass, so "the current document's shader" stopped naming one file
    the moment a second pass existed.
    """
    ui_document = app.ui_documents.get(tab.document_id)
    if ui_document is None:
        return None
    return next(
        (
            render_pass
            for render_pass in ui_document.document.passes.values()
            if render_pass.source.path == tab.path
        ),
        None,
    )


def _script_errors_for(app: App, tab: EditorTab) -> list[ShaderError]:
    # Adapt the active script tab's engine errors into the shader-error shape so they render through
    # the SAME bottom strip as compile errors (045 decision 7), click-to-jump into the script file.
    # The document script shows its sentinel + every homeless soft-key error (typo/orphan keys that name
    # no uniform row).
    out: list[ShaderError] = []
    status = app.session.get_script_status(tab.document_id)
    if status is not None:
        if status.sentinel_error is not None:
            e = status.sentinel_error
            out.append(ShaderError(tab.path, e.line, e.message))
        for key, e in status.soft_errors:
            out.append(ShaderError(tab.path, e.line, f"{key}: {e.message}"))
    return out


def _apply_markers(
    app: App,
    editor: Editor,
    errors: list[ShaderError],
    hover: HoverMark | None,
    current_path: Path,
) -> tuple:
    """Push error line-fills + the hover mark into the editor, only on change.

    Returns the marker fingerprint — a render_state member, so a marker change
    triggers exactly one redraw."""
    hover_line = (
        hover.line if hover is not None and hover.path == current_path else None
    )
    fingerprint = (
        tuple(
            (err.line, err.message)
            for err in errors
            if err.line >= 0 and err.path == current_path
        ),
        hover_line,
    )
    if app.editor_marker_state.get(current_path) == fingerprint:
        return fingerprint
    app.editor_marker_state[current_path] = fingerprint
    editor.clear_markers()
    # Marker fills are translucent by necessity — they draw behind the glyphs.
    err_fill = fade(COLOR.STATE_ERROR, 0.35)
    for line, message in fingerprint[0]:
        editor.add_marker(line, err_fill, err_fill, message)
    if hover_line is not None:
        accent = fade(COLOR.ACCENT_PRIMARY, 0.15)
        editor.add_marker(hover_line, accent, accent)
    return fingerprint


def _consume_jump(app: App, editor: Editor, current_path: Path) -> bool:
    req = app.editor_jump_request
    if req is None:
        return False
    # A request for a different file is stale (one editor only); clear it.
    if req.path != current_path:
        app.editor_jump_request = None
        return False
    app.editor_jump_request = None
    editor.set_cursor(req.line, req.column)
    editor.scroll_to_line(req.line, align_middle=True)
    return True


def _visible_error_rows(app: App, n: int) -> int:
    # How many error rows the strip shows: all of them when expanded (047 F6), else capped.
    return n if app.errors_expanded else min(n, _MAX_ERROR_ROWS)


def _draw_error_strip(app: App, errors: list[ShaderError], height: float) -> None:
    imgui.push_style_color(imgui.Col_.child_bg, COLOR.BG_SURFACE)
    if imgui.begin_child("##shader_errors", size=(0.0, height)):
        n = len(errors)
        if n > 1:
            imgui.text_colored(COLOR.FG_DIM, f"{n} errors  (F8: next)")
        shown = _visible_error_rows(app, n)
        for i, err in enumerate(errors[:shown]):
            label = (
                err.message
                if err.line < 0
                else f"Line {err.line + 1}  ·  {err.message}"
            )
            imgui.push_style_color(imgui.Col_.text, COLOR.STATE_ERROR)
            clicked = imgui.selectable(f"{label}##err{i}", False)[0]
            imgui.pop_style_color(1)
            if clicked and err.line >= 0:
                app.editor_jump_request = JumpRequest(err.path, err.line, 0)
        if n > _MAX_ERROR_ROWS:
            # Clickable toggle (F6): expand to all errors / collapse back to the cap.
            extra = n - _MAX_ERROR_ROWS
            more = "show less" if app.errors_expanded else f"+{extra} more"
            imgui.push_style_color(imgui.Col_.text, COLOR.FG_DIM)
            toggled = imgui.selectable(f"{more}##errmore", False)[0]
            imgui.pop_style_color(1)
            if toggled:
                app.errors_expanded = not app.errors_expanded
    imgui.end_child()
    imgui.pop_style_color(1)


# GLSL completion seeds beyond the live lib index + uniforms: the keywords and
# builtins the lexer knows are a fine floor for a fragment shader.
_GLSL_WORDS: tuple[str, ...] = (
    "attribute",
    "bool",
    "break",
    "const",
    "continue",
    "discard",
    "else",
    "float",
    "for",
    "highp",
    "if",
    "in",
    "int",
    "ivec2",
    "ivec3",
    "ivec4",
    "lowp",
    "mat2",
    "mat3",
    "mat4",
    "mediump",
    "out",
    "return",
    "sampler2D",
    "uniform",
    "uint",
    "varying",
    "vec2",
    "vec3",
    "vec4",
    "void",
    "while",
    "abs",
    "ceil",
    "clamp",
    "cos",
    "cross",
    "distance",
    "dot",
    "exp",
    "floor",
    "fract",
    "length",
    "max",
    "min",
    "mix",
    "mod",
    "normalize",
    "pow",
    "reflect",
    "sin",
    "smoothstep",
    "sqrt",
    "step",
    "tan",
    "texture",
)


def _completion_vocabulary(app: App, tab: EditorTab) -> list[str]:
    if tab.kind == "script":
        return list(keyword.kwlist)
    words: list[str] = list(app.shader_lib_index.functions)
    ui_document = app.ui_documents.get(tab.document_id)
    if ui_document is not None:
        edited = _pass_for_tab(app, tab)
        if edited is not None:
            words.extend(edited.uniform_values)
    words.extend(_GLSL_WORDS)
    return words


def _drive_completion(app: App, editor: Editor, tab: EditorTab) -> None:
    # Host-driven autocomplete on the deliberate-offer rule (pushing IS opening):
    # the drain marks a consumed insert-mode Ctrl+N; this offers the filtered
    # vocabulary in response, re-filters while the popup stays open and the
    # prefix moves, and cancels when nothing matches. The built-in buffer-word
    # source is suppressed at session creation, so the popup shows only this.
    # Runs BEFORE layout so the popup primitives appear the same frame.
    if app.editor_completion_requested:
        app.editor_completion_requested = False
        _offer_completion(app, editor, tab)
    elif editor.complete_open():
        prefix = editor.complete_prefix()
        if prefix != app.editor_completion_prefix:
            _offer_completion(app, editor, tab)
    else:
        app.editor_completion_prefix = None


def _offer_completion(app: App, editor: Editor, tab: EditorTab) -> None:
    prefix = editor.complete_prefix()
    matches = [
        word
        for word in _completion_vocabulary(app, tab)
        if word.startswith(prefix) and word != prefix
    ]
    if not matches or not prefix:
        editor.complete_cancel()
        app.editor_completion_prefix = None
        return
    editor.complete_begin()
    for word in matches[:50]:
        editor.complete_push(word)
    app.editor_completion_prefix = prefix


def _draw_gutter(editor: Editor, origin: imgui.ImVec2, height: float) -> None:
    # ed_layout draws no furniture: line numbers are the host's, placed with the
    # layout's own cell metrics so row N's number sits at row N's y.
    text_x, _text_y = editor.get_text_origin()
    if text_x <= 0.0:
        return
    cell_w, cell_h = editor.get_cell_size()
    if cell_h <= 0.0:
        return
    draw_list = imgui.get_window_draw_list()
    draw_list.push_clip_rect(
        origin, imgui.ImVec2(origin.x + text_x, origin.y + height), True
    )
    first = editor.get_scroll()
    rows = int(height / cell_h) + 1
    line_count = editor.get_line_count()
    current = editor.get_current_cursor_position().line
    dim = imgui.get_color_u32(COLOR.FG_DIM)
    lit = imgui.get_color_u32(COLOR.FG_SECONDARY)
    right_pad = cell_w * 0.5
    for row in range(rows):
        line = first + row
        if line >= line_count:
            break
        label = str(line + 1)
        label_w = imgui.calc_text_size(label).x
        pos = imgui.ImVec2(
            origin.x + text_x - right_pad - label_w,
            origin.y + row * cell_h + (cell_h - imgui.get_text_line_height()) * 0.5,
        )
        draw_list.add_text(pos, lit if line == current else dim, label)
    draw_list.pop_clip_rect()


def draw_chrome(app: App) -> None:
    # The editor's status chrome — mode badge + caret + the vim command line, then the
    # active tab's file + dirty/compiled state + Open dir.
    tab = app.active_tab
    if tab is None:
        imgui.text_colored(COLOR.FG_DIM, "No file open")
        return
    if app.current_document_id not in app.ui_documents:
        imgui.text_colored(COLOR.FG_DIM, "No document selected")
        return
    session = app.editor_sessions.get(tab.path)
    if session is not None:
        editor = session.editor
        badge, badge_color = _MODE_BADGES[editor.get_mode()]
        imgui.text_colored(badge_color, badge)
        imgui.same_line()
        cursor = editor.get_current_cursor_position()
        imgui.text_colored(COLOR.FG_DIM, f"{cursor.line + 1}:{cursor.column + 1}")
        imgui.same_line(spacing=float(SPACE.MD))
        # The `:`/`/`/`?` line renders here — the editor owns the state, the host
        # the pixels (feature 067).
        command = editor.get_command_line()
        if command is not None:
            prompt = editor.get_command_line_prompt() or ""
            imgui.text_colored(COLOR.ACCENT_PRIMARY, f"{prompt}{command}")
            imgui.same_line(spacing=float(SPACE.MD))
        else:
            message = editor.get_command_message()
            if message:
                imgui.text_colored(COLOR.FG_DIM, message)
                imgui.same_line(spacing=float(SPACE.MD))
    if tab.kind == "shader":
        edited_pass = _pass_for_tab(app, tab)
        full_file_path = (
            edited_pass.source.path if edited_pass is not None else tab.path
        )
        local_file_path = full_file_path.relative_to(app.project_dir)
        if draw_copyable_text(str(local_file_path), copy_value=str(full_file_path)):
            app.notifications.push("Copied to clipboard!")
        if app.is_current_editor_dirty():
            imgui.same_line()
            imgui.text_colored(COLOR.STATE_WARN, "(unsaved)")
        elif edited_pass is not None and not edited_pass.compile_unit.error_raw:
            imgui.same_line()
            imgui.text_colored(COLOR.STATE_OK, "compiled")
        imgui.same_line(spacing=float(SPACE.LG))
        if imgui.button("Open dir", size=(SIZE.BTN_SM_W, 0)):
            app.open_current_document_dir()
    else:
        imgui.text_colored(COLOR.FG_DIM, tab_label(app, tab))
        if app.is_current_editor_dirty():
            imgui.same_line()
            imgui.text_colored(COLOR.STATE_WARN, "(unsaved)")


def _handle_mouse(
    app: App,
    editor: Editor,
    origin: imgui.ImVec2,
    hovered: bool,
) -> None:
    # Host-owned mouse: press places the caret (and anchors), drag extends the
    # selection, double-click selects the word. Coordinates are widget-space —
    # the same space the layout's primitives and hit tests answer in.
    if app.splitter_dragging or app.copilot_hovered:
        return
    mouse = imgui.get_mouse_pos()
    rel = (mouse.x - origin.x, mouse.y - origin.y)
    if imgui.is_item_activated():
        pos = editor.pixel_to_cursor(rel)
        if hovered and imgui.is_mouse_double_clicked(0):
            if editor.get_mode() == Mode.NORMAL:
                editor.set_cursor(pos.line, pos.column)
                editor.feed("viw")
        else:
            editor.clear_selection()
            editor.set_cursor(pos.line, pos.column)
            app.editor_drag_anchor = (pos.line, pos.column)
    elif imgui.is_item_active() and imgui.is_mouse_dragging(0):
        anchor = app.editor_drag_anchor
        if anchor is not None:
            head = editor.pixel_to_cursor(rel)
            if (head.line, head.column) != anchor:
                editor.set_selection(anchor, (head.line, head.column))
    if not imgui.is_item_active():
        app.editor_drag_anchor = None


def _handle_wheel(app: App, editor: Editor, hovered: bool) -> None:
    io = imgui.get_io()
    if not hovered or io.mouse_wheel == 0.0:
        return
    settings = app.app_state.editor_settings
    if io.key_ctrl:
        new_size = settings.font_size + int(io.mouse_wheel)
        settings.font_size = max(8, min(48, new_size))
    else:
        editor.set_scroll(editor.get_scroll() - int(io.mouse_wheel) * 3)
    io.mouse_wheel = 0.0


def draw(app: App) -> None:
    app.code_hovered_uniform = ""
    ui_document = app.ui_documents.get(app.current_document_id)

    _draw_tab_row(app)

    tab = app.active_tab
    current_path = app.current_editor_path
    if tab is None or current_path is None or ui_document is None:
        app.editor_focused = False
        app.editor_focus_requested = False
        return

    session = app.editor_sessions.get(current_path)
    if session is None:
        # Always keyed by the TAB's own path. A shader tab used to load the document's output
        # pass instead, which was the same file back when a document had one — with several it
        # meant every pass tab showed the output's source.
        session = (
            app.open_shader_lib_file(current_path)
            if tab.kind == "lib"
            else app.get_session_for_path(current_path)
        )
    editor = session.editor
    settings = app.app_state.editor_settings

    # Lock read-only during a copilot turn. Set every frame — the active session can change.
    # Host writes (set_text / insert) are unaffected by read-only.
    editor.set_read_only_enabled(app.copilot_turn_active)

    # The error strip shows the active tab's errors in ONE place + style: a shader/lib tab's
    # compile errors, or a script tab's engine errors adapted to the same shape (045 decision 7).
    errors = (
        _script_errors_for(app, tab)
        if tab.kind == "script"
        else edited.compile_unit.errors
        if (edited := _pass_for_tab(app, tab)) is not None
        else ui_document.document.render_pass.compile_unit.errors
    )
    marker_fingerprint = _apply_markers(
        app, editor, errors, app.editor_hover_line, current_path
    )
    app.editor_hover_line = None
    strip_height = 0.0
    if errors:
        n = len(errors)
        # The rows actually drawn: the "N errors" header (n>1) + the visible error rows (capped or
        # all when expanded) + the "+N more"/"show less" toggle (n>_MAX). Matches _draw_error_strip
        # exactly, so no dead trailing row (047 F5).
        rows = (n > 1) + _visible_error_rows(app, n) + (n > _MAX_ERROR_ROWS)
        # Measure in the strip's own font (font_12), not the ambient UI font.
        imgui.push_font(app.font_12, app.font_12.legacy_size)
        strip_height = (
            rows * imgui.get_text_line_height_with_spacing()
            + 2.0 * imgui.get_style().window_padding.y
        )
        imgui.pop_font()

    # Jump/focus requests latch for the upcoming layout; must run before it.
    jumped = _consume_jump(app, editor, current_path)
    if app.editor_focus_requested and not app.any_popup_open():
        # ui.py consumed the imgui half (set_next_window_focus before the child);
        # clear the latch here so the outline saw it this frame.
        app.editor_focus_requested = False
        app.editor_was_ever_focused = True

    editor_pos = imgui.get_cursor_screen_pos()
    avail = imgui.get_content_region_avail()
    editor_size = imgui.ImVec2(avail.x, max(1.0, avail.y - strip_height))
    size_px = (max(1, int(editor_size.x)), max(1, int(editor_size.y)))
    px_per_em = float(settings.font_size)

    # Layout runs every visible frame (hit tests + scroll clamping answer against
    # it); the GL redraw below is gated.
    # The text origin is the host-chosen gutter width: the layout reserves
    # nothing itself (the reference UI offsets the same way). Width converges a
    # frame behind the line count — cell metrics answer against the last layout.
    cell_w, cell_h = editor.get_cell_size()
    gutter_px = (
        editor.get_gutter_cells() * cell_w if settings.show_line_numbers else 0.0
    )
    app.editor_visible_rows = int(size_px[1] / cell_h) if cell_h > 0 else 0
    # The interaction surface FIRST: every editor mutation (mouse, wheel) must
    # precede this frame's layout, or the redraw gate records a state the
    # painted texture doesn't show and the next frame skips the repaint. Hit
    # tests answer against LAST frame's layout — same widget geometry.
    imgui.set_cursor_screen_pos(editor_pos)
    imgui.invisible_button("##editor_surface", editor_size)
    hovering = imgui.is_item_hovered()
    if imgui.is_item_activated():
        app.editor_was_ever_focused = True
    _handle_wheel(app, editor, hovering)
    _handle_mouse(app, editor, editor_pos, hovering)

    # Focus state: the child window owns it, exactly as before — the invisible
    # button focuses the child on click.
    focused = imgui.is_window_focused(imgui.FocusedFlags_.child_windows)

    # View reconciliation, then layout (editor 4b110f0): apply a keymap-issued
    # view scroll (Ctrl+E/Y, zz/zt/zb — consumed by reading), then follow the
    # cursor when a MOTION moved it outside the view (Ctrl+D, a jump, plain j
    # below the fold; ed_layout clamps but never follows). Only a cursor CHANGE
    # follows — wheel-scrolling away from an idle caret must not snap back —
    # and only once the panel has real metrics (rows > 0): a fresh session's
    # first frame must not record the cursor with rows=0, or a jump into it
    # would never be followed.
    editor.take_scroll_request()  # absolute target; applied by the read itself
    rows = app.editor_visible_rows
    if rows > 0:
        cursor = editor.get_current_cursor_position()
        if cursor != app.editor_last_cursor.get(current_path):
            app.editor_last_cursor[current_path] = cursor
            first = editor.get_scroll()
            if not (first <= cursor.line < first + rows):
                editor.scroll_to_line(cursor.line, align_middle=False)
    _drive_completion(app, editor, tab)
    editor.layout(
        (float(size_px[0]), float(size_px[1])), px_per_em, origin=(gutter_px, 0.0)
    )

    settings_fingerprint = (
        settings.show_whitespace,
        settings.show_line_numbers,
        settings.show_matching_brackets,
        settings.tab_size,
        settings.line_spacing,
    )
    state = render_state(
        editor,
        current_path,
        size_px,
        px_per_em,
        gutter_px,
        app.editor_completion_prefix,
        marker_fingerprint,
        settings_fingerprint,
        focused,
    )
    if app.editor_renderer is None:
        app.editor_renderer = EditorRenderer(
            EDITOR_RESOURCES_DIR / "atlas.png", EDITOR_RESOURCES_DIR / "atlas.json"
        )
        app.editor_panel = EditorPanel(app.editor_renderer)
    panel = app.editor_panel
    assert panel is not None
    if should_redraw(panel.last_state, state):
        panel.render(editor, size_px, px_per_em, COLOR.BG_SURFACE)
        panel.last_state = state
        app.editor_redraw_count += 1
    if panel.texture is not None:
        # Unfocused pane dims via the image tint (style alpha can't reach a texture).
        alpha = 1.0 if focused or jumped else EDITOR_UNFOCUSED_ALPHA
        imgui.get_window_draw_list().add_image(
            imgui.ImTextureRef(panel.texture.glo),
            editor_pos,
            imgui.ImVec2(editor_pos.x + size_px[0], editor_pos.y + size_px[1]),
            imgui.ImVec2(0, 0),
            imgui.ImVec2(1, 1),
            imgui.get_color_u32((1.0, 1.0, 1.0, alpha)),
        )
    if settings.show_line_numbers:
        imgui.push_font(app.font_12, app.font_12.legacy_size)
        _draw_gutter(editor, editor_pos, float(size_px[1]))
        imgui.pop_font()

    app.editor_focused = focused
    if app.editor_focused:
        # Sticky: stays True across popups/menus until an explicit defocus.
        app.editor_was_ever_focused = True

    if app.editor_defocus_requested:
        imgui.set_window_focus(None)
        app.editor_defocus_requested = False
        app.editor_focused = False
        app.editor_was_ever_focused = False

    # Drive the glfw cursor directly — imgui cursors are no-op here (conventions.md ## Known
    # quirks). is_item_hovered respects popup-blocking. NOT over the editor when the floating
    # chat is hovered: it owns its own cursor.
    cursor_over_editor = hovering and not app.copilot_hovered
    if cursor_over_editor:
        app.want_cursor = app.ibeam_cursor

    # Cursor-following tooltip for words that are live uniforms; also lights up the panel row.
    if cursor_over_editor:
        mouse = imgui.get_mouse_pos()
        rel = (mouse.x - editor_pos.x, mouse.y - editor_pos.y)
        if editor.is_mouse_pos_over_glyph(rel):
            word = editor.get_word_at_mouse_pos(rel)
            if word and word in ui_document.document.render_pass.uniform_values:
                value = ui_document.document.render_pass.uniform_values[word]
                imgui.set_tooltip(f"{word}: {format_auto_value(value)}")
                app.code_hovered_uniform = word

    if errors:
        imgui.push_font(app.font_12, app.font_12.legacy_size)
        _draw_error_strip(app, errors, strip_height)
        imgui.pop_font()
