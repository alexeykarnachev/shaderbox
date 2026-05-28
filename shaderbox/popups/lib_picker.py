"""Library function picker — Ctrl+P opens a fuzzy-search modal over SB_* lib functions.

Each row: function name, signature, docstring (one line, ellipsized). A right-hand
preview pane shows the selected function's full body. Two actions:
  - Enter / "Insert" — insert just the function name at the user's editor caret.
  - "Open file" — switch the editor to the lib file at the function's declaration line.

A "New library file" action at the bottom lets the user create an empty lib file
and start editing it (in step 4 + tab bar) immediately.
"""

from imgui_bundle import imgui, imgui_ctx
from loguru import logger

from shaderbox.app import App, JumpRequest
from shaderbox.lib_index import LibFunction
from shaderbox.paths import lib_root
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_primitives import button, ghost_button, primary_button

_LABEL = "Library##picker"
_POPUP_W = 760.0
_POPUP_H = 480.0
_LIST_W = 320.0
_NEW_FILE_NAME_KEY = "##new_lib_file_name"


def draw_lib_picker(app: App) -> None:
    if not app.is_lib_picker_open:
        return

    if not imgui.is_popup_open(_LABEL):
        imgui.open_popup(_LABEL)

    imgui.set_next_window_size(imgui.ImVec2(_POPUP_W, _POPUP_H), imgui.Cond_.appearing)
    with imgui_ctx.begin_popup_modal(_LABEL) as popup:
        if not popup.visible:
            return
        keep_open = _draw_body(app)
        if not keep_open:
            app.is_lib_picker_open = False
            imgui.close_current_popup()


def _draw_body(app: App) -> bool:
    keep_open: bool = True

    # Search row.
    imgui.set_next_item_width(_LIST_W)
    _, app.lib_picker_query = imgui.input_text("##q", app.lib_picker_query)
    imgui.same_line()
    imgui.text_colored(COLOR.FG_DIM, f"{len(app.lib_index.functions)} SB_* functions")

    matches = _matching_functions(app)

    # Left list + right preview.
    avail = imgui.get_content_region_avail()
    body_h = avail.y - SIZE.BTN_SM_H - float(SPACE.MD) * 2.0
    with imgui_ctx.begin_child("##list", size=imgui.ImVec2(_LIST_W, body_h)):
        for fn in matches:
            label = f"{fn.name}##row_{fn.name}"
            selected = fn.name == _selected_name(app)
            if imgui.selectable(label, selected)[0]:
                app.lib_picker_query = fn.name  # remember as selection
            if imgui.is_item_hovered():
                imgui.set_tooltip(fn.signature)
            # Doc line under the name, dim.
            if fn.doc:
                first_doc_line = fn.doc.splitlines()[0]
                imgui.same_line()
                imgui.text_colored(COLOR.FG_DIM, f"  - {first_doc_line[:50]}")

    imgui.same_line(spacing=float(SPACE.MD))

    with imgui_ctx.begin_child("##preview", size=imgui.ImVec2(0.0, body_h)):
        selected = _resolve_selected(app, matches)
        if selected is not None:
            imgui.text_colored(COLOR.FG_DIM, str(selected.file.name))
            imgui.text(selected.signature)
            if selected.doc:
                imgui.spacing()
                imgui.text_colored(COLOR.FG_DIM, selected.doc)
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            # Body in a scrollable read-only region.
            with imgui_ctx.begin_child(
                "##body",
                child_flags=imgui.ChildFlags_.borders,
            ):
                imgui.text_unformatted(selected.body)
        else:
            imgui.text_colored(COLOR.FG_DIM, "(select a function)")

    imgui.spacing()

    # Action row: Insert / Open file / New file / Close.
    can_act = _resolve_selected(app, matches) is not None
    if primary_button("Insert at caret", width=SIZE.BTN_SM_W) and can_act:
        sel = _resolve_selected(app, matches)
        if sel is not None:
            _insert_name(app, sel)
            keep_open = False
    imgui.same_line()
    if button("Open file", width=SIZE.BTN_SM_W) and can_act:
        sel = _resolve_selected(app, matches)
        if sel is not None:
            _open_at_decl(app, sel)
            keep_open = False
    imgui.same_line()
    if ghost_button("+ New library file"):
        imgui.open_popup("##new_lib_popup")
    imgui.same_line()
    if ghost_button("Close"):
        keep_open = False

    # Sub-popup for naming a new file.
    if imgui.begin_popup("##new_lib_popup"):
        imgui.text("New file name (.glsl):")
        imgui.set_next_item_width(220.0)
        # Reuse the query field as the new-file name buffer to avoid yet another
        # state field; the query gets reset to "" on next open anyway.
        _, app.lib_picker_query = imgui.input_text(
            _NEW_FILE_NAME_KEY, app.lib_picker_query
        )
        if primary_button("Create", width=SIZE.BTN_SM_W):
            _create_new(app)
            keep_open = False
            imgui.close_current_popup()
        imgui.same_line()
        if ghost_button("Cancel"):
            imgui.close_current_popup()
        imgui.end_popup()

    return keep_open


def _matching_functions(app: App) -> list[LibFunction]:
    # Filter strict on SB_ prefix; fuzzy on name + doc. Sort: exact-prefix matches
    # first, then by name.
    query: str = app.lib_picker_query.strip().lower()
    candidates = [
        fn for fn in app.lib_index.functions.values() if fn.name.startswith("SB_")
    ]
    if not query:
        return sorted(candidates, key=lambda f: f.name)

    def score(fn: LibFunction) -> tuple[int, str]:
        # Lower score is better. Exact-prefix beats substring beats doc-only.
        name_lower = fn.name.lower()
        if name_lower.startswith(query):
            return (0, name_lower)
        if query in name_lower:
            return (1, name_lower)
        if fn.doc and query in fn.doc.lower():
            return (2, name_lower)
        return (3, name_lower)

    scored = [(score(fn), fn) for fn in candidates]
    scored.sort(key=lambda s: s[0])
    return [fn for (rank, _), fn in scored if rank < 3]


def _selected_name(app: App) -> str:
    # The selectable list highlights whatever's in the query field that EXACTLY
    # matches a function name. Otherwise nothing is highlighted (first match in
    # _resolve_selected wins for actions).
    return app.lib_picker_query.strip()


def _resolve_selected(app: App, matches: list[LibFunction]) -> LibFunction | None:
    # If the query exactly matches a function name, use that. Otherwise default
    # to the first match — the user typing "perl" should be able to hit Enter
    # and get SB_perlin_noise_3 without arrow-keying.
    if not matches:
        return None
    query = app.lib_picker_query.strip()
    for fn in matches:
        if fn.name == query:
            return fn
    return matches[0]


def _insert_name(app: App, fn: LibFunction) -> None:
    # Insert the function name at the caret of the currently-active editor.
    # `replace_text_in_current_cursor` with a zero-width selection inserts.
    session = app.get_current_session_if_exists()
    if session is None:
        logger.warning("No editor session active; can't insert lib name")
        return
    session.editor.replace_text_in_current_cursor(fn.name)


def _open_at_decl(app: App, fn: LibFunction) -> None:
    app.open_lib_file(fn.file)
    # Jump to the declaration line in the now-open lib editor session.
    app.editor_jump_request = JumpRequest(fn.file, fn.line_in_file, 0)


def _create_new(app: App) -> None:
    name: str = app.lib_picker_query.strip()
    if not name:
        return
    if not name.endswith(".glsl"):
        name += ".glsl"
    path = lib_root() / name
    if path.exists():
        logger.warning(f"Lib file already exists: {path}")
        app.open_lib_file(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "/// new SB_* function — write a doc here\n"
        "// float SB_my_function(float x) {\n"
        "//     return x;\n"
        "// }\n",
        encoding="utf-8",
    )
    logger.info(f"Created new lib file: {path}")
    # Rebuild the index so the new file is visible right away (the watcher would
    # rebuild on the next frame anyway, but the user is about to edit, not wait).
    app.rebuild_lib_index()
    app.open_lib_file(path)
