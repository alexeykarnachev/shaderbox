from pathlib import Path

from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.completion import (
    CompletionContext,
    candidate_doc,
    offer,
    symbol_doc,
    word_at,
)
from shaderbox.core import Pass
from shaderbox.editor.ffi import EDITOR_RESOURCES_DIR, CursorPos, Editor, Mode
from shaderbox.editor.render import (
    EditorPanel,
    EditorRenderer,
    render_state,
    should_redraw,
)
from shaderbox.editor_types import EditorTab, HoverMark, JumpRequest, LookupPopup
from shaderbox.paths import pass_name_of
from shaderbox.shader_errors import ShaderError
from shaderbox.theme import (
    COLOR,
    EDITOR_CURSOR_LINE_ALPHA,
    EDITOR_UNFOCUSED_ALPHA,
    SIZE,
    SPACE,
    fade,
)
from shaderbox.ui_primitives import anchored_note, draw_copyable_text
from shaderbox.util import format_auto_value

_MAX_ERROR_ROWS = 3


def _is_script_tab(tab: EditorTab | None) -> bool:
    return tab is not None and tab.kind == "script"


def tab_label(app: App, tab: EditorTab) -> str:
    # The display label for a tab (048): document-derived so two documents' tabs are distinguishable
    # ("<document> (shader)" / "<document> (script)"), a lib by "library - <file>". The on-disk filename is
    # the same constant for every document, so the bare name can't tell tabs apart. Falls back to a short
    # id slice when the document has no name. The imgui ##id keys on the stable path, NOT this label.
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
    display_order: list[int] = []
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
                f"{tab_label(app, tab)}{_tab_id_suffix(tab)}", True, item_flags
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
        display_order = _display_order(app)
        imgui.end_tab_bar()
    if close_index is not None:
        app.close_tab(close_index)
    elif display_order:
        _apply_display_order(app, display_order)


def _tab_id_suffix(tab: EditorTab) -> str:
    # The imgui id is the PATH, not the list index: the model list follows imgui's drag order
    # (below), and an index-keyed id would make every moved tab a new tab to imgui, appended at
    # the end and scrambling the order it had just been read from.
    return f"##{tab.path}"


def _display_order(app: App) -> list[int]:
    """Model indices in the order the tab bar SHOWS them, read from imgui's own tab state.

    A drag reorders imgui's list only; nothing else reports it. Empty when imgui's list and the
    model disagree in size or names (the frame a tab is born or closed), so the caller skips
    that frame. Valid only between begin_tab_bar and end_tab_bar."""
    bar = imgui.internal.get_current_tab_bar()
    by_suffix = {_tab_id_suffix(tab): i for i, tab in enumerate(app.editor_tabs)}
    order: list[int] = []
    for k in range(len(app.editor_tabs)):
        item = imgui.internal.tab_bar_find_tab_by_order(bar, k)
        if item is None:
            return []
        index = _model_index(imgui.internal.tab_bar_get_tab_name(bar, item), by_suffix)
        if index is None:
            return []
        order.append(index)
    return order


def _model_index(tab_name: str, by_suffix: dict[str, int]) -> int | None:
    # Matched by suffix, never by splitting: a document's display name is free text and may
    # itself contain `##`, and the id suffix is what the name ENDS with.
    return next(
        (i for suffix, i in by_suffix.items() if tab_name.endswith(suffix)), None
    )


def _apply_display_order(app: App, order: list[int]) -> None:
    """Permute the model tab list to `order` (model indices, display-ordered), keeping the
    active tab's identity, so cycle / close / every index-based verb addresses what the eye
    sees."""
    if order == list(range(len(app.editor_tabs))):
        return
    active = app.active_tab
    app.editor_tabs = [app.editor_tabs[i] for i in order]
    if active is not None:
        app.active_tab_index = app.editor_tabs.index(active)


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
    # The script is ONE file, so its tab shows EVERY soft error whatever pass it names — its author
    # wants all of them — each prefixed with the pass it concerns (069).
    out: list[ShaderError] = []
    status = app.session.get_script_status(tab.document_id)
    if status is not None:
        if status.sentinel_error is not None:
            e = status.sentinel_error
            out.append(ShaderError(tab.path, e.line, e.message))
        for pass_name, key, e in status.soft_errors:
            label = f"{pass_name}.{key}" if pass_name else key
            out.append(ShaderError(tab.path, e.line, f"{label}: {e.message}"))
    return out


def _script_errors_for_pass(
    app: App, tab: EditorTab, pass_name: str
) -> list[ShaderError]:
    # The script's soft errors that name THIS shader tab's pass, shown under the pass's own compile
    # errors (069). No pass prefix — the tab already says which pass it is. The rows carry the SCRIPT
    # path, not the shader's, because that is where the fix is; the click branch opens it first (a
    # jump request for a non-current file is discarded as stale). The sentinel stays on the script
    # tab: it belongs to no pass, and repeating it on every shader tab is noise.
    status = app.session.get_script_status(tab.document_id)
    if status is None:
        return []
    script_path = app.session.script_path_for(tab.document_id)
    return [
        ShaderError(script_path, e.line, f"{key}: {e.message}")
        for err_pass, key, e in status.soft_errors
        if err_pass == pass_name
    ]


def _apply_markers(
    app: App,
    editor: Editor,
    errors: list[ShaderError],
    hover: HoverMark | None,
    current_path: Path,
    cursor_line: int,
) -> tuple:
    """Push the cursor-line band, the error line-fills and the hover mark into the editor,
    only on change. The cursor line goes FIRST: two markers on one line stack their fills and
    the later text color wins, so an error line keeps its flipped text.

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
        cursor_line,
    )
    if app.editor_marker_state.get(current_path) == fingerprint:
        return fingerprint
    app.editor_marker_state[current_path] = fingerprint
    editor.clear_markers()
    editor.add_marker(cursor_line, fill=fade(COLOR.BORDER, EDITOR_CURSOR_LINE_ALPHA))
    # Marker fills are translucent by necessity — they draw behind the glyphs.
    # STATE_ERROR and SYN_KEYWORD are the same palette entry, so the text color
    # is replaced rather than left to the lexer: red on red is unreadable.
    err_fill = fade(COLOR.STATE_ERROR, 0.20)
    for line, message in fingerprint[0]:
        editor.add_marker(
            line,
            fill=err_fill,
            gutter=COLOR.STATE_ERROR,
            text=COLOR.FG_PRIMARY,
            gutter_glyph="E",
            tooltip=message,
        )
    if hover_line is not None:
        accent = fade(COLOR.ACCENT_PRIMARY, 0.15)
        editor.add_marker(hover_line, fill=accent, gutter=accent)
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


def _draw_error_strip(
    app: App, tab: EditorTab, errors: list[ShaderError], height: float
) -> None:
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
                # A shader tab's strip can carry a SCRIPT error, whose row points at the script
                # file. `_consume_jump` discards a request whose path is not the CURRENT tab's, so
                # open the script first — exactly as the two other cross-file jumps do.
                if err.path != app.current_editor_path:
                    app.open_script_for(tab.document_id)
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


def _completion_context(
    app: App, editor: Editor, tab: EditorTab, explicit: bool
) -> CompletionContext:
    cursor = editor.get_current_cursor_position()
    lines = editor.get_text().split("\n")
    line = lines[cursor.line] if cursor.line < len(lines) else ""
    edited = _pass_for_tab(app, tab)
    return CompletionContext(
        tab_kind=tab.kind,
        line_before_caret=line[: cursor.column],
        prefix=editor.complete_prefix(),
        lib_functions=tuple(app.shader_lib_index.functions),
        pass_uniforms=tuple(edited.uniform_values) if edited is not None else (),
        explicit=explicit,
    )


def _drive_completion(app: App, editor: Editor, tab: EditorTab) -> None:
    # Host-driven autocomplete (pushing IS opening; the built-in buffer-word source is
    # suppressed at session creation, so the popup shows only what the providers say).
    # Three ways in: the deliberate Ctrl+N / Ctrl+P (explicit), a keystroke in insert mode
    # that changed the buffer (auto, 073 W-B), and a re-filter while the popup stays open
    # and the prefix moves. Runs BEFORE layout so the popup primitives appear the same frame.
    revision = editor.get_undo_index()
    edited = app.editor_completion_seen != (tab.path, revision)
    app.editor_completion_seen = (tab.path, revision)
    was_open = app.editor_completion_was_open
    if app.editor_completion_requested:
        app.editor_completion_requested = False
        _offer_completion(app, editor, tab, explicit=True)
    elif editor.complete_open():
        prefix = editor.complete_prefix()
        if prefix != app.editor_completion_prefix:
            _offer_completion(app, editor, tab, explicit=not app.editor_completion_auto)
    elif edited and editor.get_mode() == Mode.INSERT:
        if was_open:
            # The edit closed the popup (a typed character does, under host-driven
            # completion). An accept, or a character that ended the word, re-offers
            # nothing; a character that CONTINUED the word, or a backspace inside it,
            # re-offers the same batch with the same asked-for-ness, so a Ctrl+N list
            # keeps its row-0 highlight while the user keeps typing or corrects.
            prefix = editor.complete_prefix()
            cached = app.editor_completion_prefix
            continued = (
                bool(prefix)
                and bool(cached)
                and prefix not in app.editor_completion_items
                and (prefix.startswith(cached) or cached.startswith(prefix))
            )
            if continued:
                _offer_completion(
                    app, editor, tab, explicit=not app.editor_completion_auto
                )
            else:
                app.editor_completion_prefix = None
        else:
            _offer_completion(app, editor, tab, explicit=False)
    else:
        app.editor_completion_prefix = None
    # Recorded after the offer, so the frame that opens a popup is seen as open by the next.
    app.editor_completion_was_open = editor.complete_open()


def _consume_lookup_request(app: App, editor: Editor) -> None:
    if not app.editor_lookup_requested:
        return
    app.editor_lookup_requested = False
    cursor = editor.get_current_cursor_position()
    lines = editor.get_text().split("\n")
    line = lines[cursor.line] if cursor.line < len(lines) else ""
    word = word_at(line, cursor.column)
    found = symbol_doc(word, app.shader_lib_index.functions) if word else None
    app.editor_lookup = (
        LookupPopup(word=word, signature=found[0], doc=found[1]) if found else None
    )


def _draw_lookup_popup(app: App, editor: Editor) -> None:
    lookup = app.editor_lookup
    if lookup is None:
        return
    if imgui.is_mouse_clicked(0) or imgui.is_mouse_clicked(1):
        app.editor_lookup = None
        return
    # Under the caret's cell: the editor rect, the text origin inside it, then the cell.
    cursor = editor.get_current_cursor_position()
    origin_x, origin_y = editor.get_text_origin()
    cell_w, cell_h = editor.get_cell_size()
    rect_x, rect_y = app.editor_rect[0], app.editor_rect[1]
    anchor = (
        rect_x + origin_x + cursor.column * cell_w,
        rect_y + origin_y + (cursor.line - editor.get_scroll() + 1) * cell_h,
    )
    anchored_note("##lookup", anchor, lookup.signature, lookup.doc)


def _draw_candidate_doc(app: App, editor: Editor) -> None:
    """The doc for the completion popup's highlighted row, beside the popup.

    The library draws the list and owns the selection; the detail is the host's, read from
    the same tables `K` uses. Nothing is drawn while the row is unhighlighted (an unasked
    batch) or the candidate has no doc.
    """
    if not editor.complete_open():
        return
    index = editor.complete_selected()
    if index < 0:
        return
    candidate = editor.complete_item(index)
    if candidate is None:
        return
    found = candidate_doc(candidate, app.shader_lib_index.functions)
    if found is None:
        return
    # Beside the popup rather than under it: the popup itself hangs under the caret, and a
    # note under BOTH would cover the code the user is completing into.
    cursor = editor.get_current_cursor_position()
    origin_x, origin_y = editor.get_text_origin()
    cell_w, cell_h = editor.get_cell_size()
    rect_x, rect_y = app.editor_rect[0], app.editor_rect[1]
    widest = max(
        (len(editor.complete_item(i) or "") for i in range(editor.complete_count())),
        default=0,
    )
    anchor = (
        rect_x + origin_x + (cursor.column + widest + 3) * cell_w,
        rect_y + origin_y + (cursor.line - editor.get_scroll() + 1) * cell_h,
    )
    anchored_note("##candidate_doc", anchor, found[0], found[1])


def _offer_completion(app: App, editor: Editor, tab: EditorTab, explicit: bool) -> None:
    context = _completion_context(app, editor, tab, explicit)
    matches = offer(context)
    if not matches:
        editor.complete_cancel()
        app.editor_completion_prefix = None
        app.editor_completion_items = []
        app.editor_completion_auto = False
        return
    editor.complete_begin()
    for word in matches:
        editor.complete_push(word)
    app.editor_completion_prefix = context.prefix
    app.editor_completion_items = matches
    app.editor_completion_auto = not explicit
    if not explicit:
        # Unasked, so nothing is highlighted: Enter stays a newline until the user moves
        # into the list. Per batch -- every push batch starts at row 0.
        editor.complete_select(-1)


def draw_chrome(app: App) -> None:
    # Host things only: the active tab's file + dirty/compiled state + Open dir.
    # The mode badge, the ruler and the `:`/`/`/`?` line are drawn by the library
    # inside the editor rect (feature 069 W-F).
    tab = app.active_tab
    if tab is None:
        imgui.text_colored(COLOR.FG_DIM, "No file open")
        return
    if app.current_document_id not in app.ui_documents:
        imgui.text_colored(COLOR.FG_DIM, "No document selected")
        return
    if tab.kind == "shader":
        edited_pass = _pass_for_tab(app, tab)
        full_file_path = (
            edited_pass.source.path if edited_pass is not None else tab.path
        )
        # A path outside the project shows in full rather than raising. `relative_to` throws on
        # any non-descendant, and this runs inside the frame draw -- so one unexpected path
        # (a shipped example not yet copied, a lib file, a hand-opened file) took the whole app
        # down instead of showing a longer label.
        local_file_path = (
            full_file_path.relative_to(app.project_dir)
            if full_file_path.is_relative_to(app.project_dir)
            else full_file_path
        )
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


def layout_following_cursor(
    editor: Editor,
    size: tuple[float, float],
    px_per_em: float,
    rows: int,
    last_cursor: CursorPos | None,
) -> CursorPos:
    """Lay the editor out, bring a MOVED cursor into view, and lay out again if that scrolled.

    The follow runs AFTER this frame's layout: `scroll_to_line` answers against the last
    `ed_layout`, and only this frame's layout knows a line the edit just added. `rows` is the
    text rows the host shows (the status row excluded); 0 means no metrics yet, and nothing
    follows.
    """
    editor.layout(size, px_per_em)
    cursor = editor.get_current_cursor_position()
    if rows > 0 and cursor != last_cursor:
        first = editor.get_scroll()
        if not (first <= cursor.line < first + rows):
            editor.scroll_to_line(cursor.line, align_middle=False)
            editor.layout(size, px_per_em)
    return cursor


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
    # Compile errors FIRST: a shader that does not compile is why nothing else works, and the strip
    # caps its visible rows, so the ordering decides what an unexpanded strip shows.
    errors = (
        _script_errors_for(app, tab)
        if tab.kind == "script"
        else edited.compile_unit.errors
        + _script_errors_for_pass(app, tab, pass_name_of(tab.path))
        if (edited := _pass_for_tab(app, tab)) is not None
        else ui_document.document.render_pass.compile_unit.errors
    )
    marker_fingerprint = _apply_markers(
        app,
        editor,
        errors,
        app.editor_hover_line,
        current_path,
        editor.get_current_cursor_position().line,
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
        # ui.py consumed the imgui half (set_next_window_focus before the child), so the
        # latch has done its job and is cleared here.
        app.editor_focus_requested = False
        app.editor_was_ever_focused = True

    editor_pos = imgui.get_cursor_screen_pos()
    avail = imgui.get_content_region_avail()
    editor_size = imgui.ImVec2(avail.x, max(1.0, avail.y - strip_height))
    size_px = (max(1, int(editor_size.x)), max(1, int(editor_size.y)))
    px_per_em = float(settings.font_size)

    # Layout runs every visible frame (hit tests + scroll clamping answer against
    # it); the GL redraw below is gated.
    cell_h = editor.get_cell_size()[1]
    # Minus one row for the status line the library draws on the widget's bottom
    # edge: a cursor behind it is off-screen for the cursor-follow below.
    app.editor_visible_rows = max(0, int(size_px[1] / cell_h) - 1) if cell_h > 0 else 0
    # The interaction surface FIRST: every editor mutation (mouse, wheel) must
    # precede this frame's layout, or the redraw gate records a state the
    # painted texture doesn't show and the next frame skips the repaint. Hit
    # tests answer against LAST frame's layout — same widget geometry.
    # It stops one row ABOVE the image's bottom: the library's status band is
    # opaque chrome, and the hit tests extrapolate rows past the last drawn one,
    # so a click on the mode badge would otherwise place the caret on a line
    # hidden behind it.
    imgui.set_cursor_screen_pos(editor_pos)
    imgui.invisible_button(
        "##editor_surface",
        imgui.ImVec2(editor_size.x, max(1.0, editor_size.y - cell_h)),
    )
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
    # cursor when a MOTION or an EDIT moved it outside the view (Ctrl+D, a jump,
    # plain j below the fold, `o` on the last line; ed_layout clamps but never
    # follows). Only a cursor CHANGE follows — wheel-scrolling away from an idle
    # caret must not snap back — and only once the panel has real metrics
    # (rows > 0): a fresh session's first frame must not record the cursor with
    # rows=0, or a jump into it would never be followed.
    editor.take_scroll_request()  # absolute target; applied by the read itself
    _drive_completion(app, editor, tab)
    _consume_lookup_request(app, editor)
    rows = app.editor_visible_rows
    cursor = layout_following_cursor(
        editor,
        (float(size_px[0]), float(size_px[1])),
        px_per_em,
        rows,
        app.editor_last_cursor.get(current_path),
    )
    if rows > 0:
        app.editor_last_cursor[current_path] = cursor

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
        editor.get_text_origin(),
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

    _draw_lookup_popup(app, editor)
    _draw_candidate_doc(app, editor)
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
        _draw_error_strip(app, tab, errors, strip_height)
        imgui.pop_font()
