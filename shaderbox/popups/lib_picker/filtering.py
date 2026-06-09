"""Candidate filtering, selection, keyboard nav, and the insert/open actions.

`filter_functions` applies the search query + tag filters + favs toggle;
`selected_function` resolves the current selection against the filtered set;
the rest are the actions the tree context menu and the action row invoke.
"""

import pyperclip
from imgui_bundle import imgui
from loguru import logger

from shaderbox.app import App
from shaderbox.editor_types import JumpRequest
from shaderbox.popups.lib_picker.search import parse_query_tags
from shaderbox.shader_lib import ShaderLibFunction


def handle_arrow_nav(app: App, visible: list[ShaderLibFunction]) -> None:
    if not visible:
        app.shader_lib_files.picker_selected_function = ""
        return
    down = imgui.is_key_pressed(imgui.Key.down_arrow, repeat=True)
    up = imgui.is_key_pressed(imgui.Key.up_arrow, repeat=True)
    if not (down or up):
        # Clamp: if the current selection is no longer visible (filter change),
        # fall back to the first visible leaf.
        names = [fn.name for fn in visible]
        if app.shader_lib_files.picker_selected_function not in names:
            app.shader_lib_files.picker_selected_function = visible[0].name
        return
    names = [fn.name for fn in visible]
    try:
        idx = names.index(app.shader_lib_files.picker_selected_function)
    except ValueError:
        idx = 0
    step = 1 if down else -1
    app.shader_lib_files.picker_selected_function = names[(idx + step) % len(names)]


def selected_function(
    app: App, candidates: list[ShaderLibFunction]
) -> ShaderLibFunction | None:
    if not app.shader_lib_files.picker_selected_function:
        return None
    for fn in candidates:
        if fn.name == app.shader_lib_files.picker_selected_function:
            return fn
    # The selected function is hidden by current filters. Fall back to None so
    # the preview shows "(no selection)" rather than a stale function.
    return None


def filter_functions(app: App) -> list[ShaderLibFunction]:
    raw_query: str = app.shader_lib_files.picker_query.strip().lower()
    tag_prefixes, text_query = parse_query_tags(raw_query)
    disabled = app.shader_lib_files.picker_disabled_tags

    def passes(fn: ShaderLibFunction) -> bool:
        # Fetch tags ONCE per function; three filters share the result.
        fn_tags = app.shader_lib_tags.tags_for(fn.name)
        # tag-bar: untagged always visible; otherwise at least one enabled tag.
        if fn_tags and not (fn_tags - disabled):
            return False
        # #prefix tokens in the search bar: each must match at least one tag.
        if tag_prefixes and not all(
            any(t.startswith(p) for t in fn_tags) for p in tag_prefixes
        ):
            return False
        # free-text query: matches name, doc, or any tag substring.
        if text_query:
            name_lower = fn.name.lower()
            if text_query in name_lower:
                return True
            if fn.doc and text_query in fn.doc.lower():
                return True
            return any(text_query in t for t in fn_tags)
        return True

    candidates = [
        fn
        for fn in app.shader_lib_index.functions.values()
        if fn.name.startswith("SB_")
    ]
    if app.shader_lib_files.picker_favs_only:
        candidates = [
            fn for fn in candidates if app.shader_lib_favorites.is_favorite(fn.name)
        ]
    return [fn for fn in candidates if passes(fn)]


def insert_name(app: App, fn: ShaderLibFunction) -> None:
    session = app.get_current_session_if_exists()
    if session is None:
        logger.warning("No editor session active; can't insert lib name")
        return
    # Inserts at the caret; the editor leaves the cursor right after the text.
    session.editor.replace_text_in_current_cursor(fn.name)
    # The picker closes this frame (editor not drawn behind the modal), so request
    # the editor to re-grab focus on the next render — caret stays where insert ended.
    app.editor_focus_requested = True


def open_at_decl(app: App, fn: ShaderLibFunction) -> None:
    # Retained for hotkeys / callers that may invoke "Open function file"
    # directly (e.g. a future shortcut).
    app.open_shader_lib_file(fn.file)
    app.editor_jump_request = JumpRequest(fn.file, fn.line_in_file, 0)


def copy_to_clipboard(text: str) -> None:
    try:
        pyperclip.copy(text)
    except pyperclip.PyperclipException as e:
        logger.warning(f"Could not copy: {e}")
