"""Shader-library picker (Ctrl+P).

Unified tree-on-the-left + preview-on-the-right modal:
  - Left column = `/` root + nested dirs + files + function leaves, all
    collapsible. Right-click a directory / file / function for its actions
    (New file / New subdir / Rename / Delete / Reveal / Insert / Open at decl /
    Copy name / Toggle favorite).
  - Right column = the preview pane for the selected function (path, signature,
    doc, tags, body).
  - Above both: search input, then Favs/Reset pills, then tag bar.
  - Bottom: Insert at caret + Close.

Submodules: `search` (top bar + query parsing), `tree` (left column),
`preview` (right pane), `filtering` (candidate filter + selection + actions).
"""

from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.paths import shader_lib_root
from shaderbox.popups.lib_picker import filtering, preview, search, tree
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_primitives import ghost_button, modal_window, primary_button

_LABEL = "Shader Library##picker"
_POPUP_W = 1420.0
_POPUP_H = 1130.0
_TREE_FRAC = 0.40
_TREE_W_MIN = 280.0


def draw_lib_picker(app: App) -> None:
    if not app.is_shader_lib_picker_open:
        return
    with modal_window(_LABEL, (_POPUP_W, _POPUP_H)) as visible:
        if not visible:
            return
        if not _draw_body(app):
            app.is_shader_lib_picker_open = False
            imgui.close_current_popup()


def _draw_body(app: App) -> bool:
    keep_open: bool = True
    root = shader_lib_root()

    # imgui owns the "is this the first frame the modal is visible?" signal —
    # we use it for auto-select-first-leaf, search-input autofocus, and
    # suppression of the leaf autoscroll on the first frame.
    just_opened = imgui.is_window_appearing()
    app.shader_lib_picker_just_opened = (
        just_opened  # widgets called from sub-draws read this
    )

    # Build the visible-leaves list once per frame: this drives the preview
    # pane, the arrow-key nav, and the auto-select-first-on-open.
    candidates = filtering.filter_functions(app)
    node_tree = tree.build_tree(candidates, root)
    visible_leaves = tree.flatten_visible_leaves(node_tree, ())

    rename_active = app.shader_lib_file_rename.target is not None
    new_file_active = app.shader_lib_file_new.target is not None
    new_dir_active = app.shader_lib_dir_new.target is not None
    input_owns_keys = rename_active or new_file_active or new_dir_active

    if not input_owns_keys:
        filtering.handle_arrow_nav(app, visible_leaves)

    if just_opened and not app.shader_lib_picker_selected_function and visible_leaves:
        app.shader_lib_picker_selected_function = visible_leaves[0].name

    # The add-tag input also captures Enter (via `enter_returns_true`). When it
    # had focus last frame, suppress the outer Enter → Insert + close — otherwise
    # committing a tag would also close the picker. The flag is reset here and
    # re-set during the tag-editor draw if the input is focused this frame.
    tag_input_was_focused = app.shader_lib_picker_tag_input_focused
    app.shader_lib_picker_tag_input_focused = False
    pressed_enter = (
        not input_owns_keys
        and not tag_input_was_focused
        and imgui.is_key_pressed(imgui.Key.enter, repeat=False)
    )

    # ---------- Top bar: search first, then filter pills, then tag bar ----------
    search.draw_search_row(
        app, len(visible_leaves), len(app.shader_lib_index.functions)
    )
    search.draw_favs_and_reset_row(app)
    search.draw_tag_bar(app)

    imgui.spacing()

    # ---------- Body: tree | preview ----------
    avail = imgui.get_content_region_avail()
    body_h = max(220.0, avail.y - SIZE.BTN_SM_H - float(SPACE.MD) * 2.0)

    tree_w = max(_TREE_W_MIN, avail.x * _TREE_FRAC)
    with imgui_ctx.begin_child("##tree_col", size=imgui.ImVec2(tree_w, body_h)):
        imgui.text_colored(COLOR.FG_DIM, "Right-click for actions")
        if tree.draw_tree(app, node_tree, root):
            keep_open = False
    imgui.same_line(spacing=float(SPACE.MD))
    with imgui_ctx.begin_child("##preview", size=imgui.ImVec2(0.0, body_h)):
        preview.draw_preview(app, filtering.selected_function(app, candidates), root)

    imgui.spacing()

    # ---------- Action row ----------
    selected = filtering.selected_function(app, candidates)
    # Insert is meaningful only when the editor was actually being used (not
    # merely "a node is selected"). `editor_was_ever_focused` stays True
    # across popups/menus and is cleared only by explicit defocus or by a
    # target switch — so a freshly-selected node with an un-touched caret
    # correctly disables insert.
    has_editor = app.editor_was_ever_focused and app.current_editor_path is not None
    can_insert = selected is not None and has_editor
    imgui.begin_disabled(not can_insert)
    btn_clicked = primary_button("Insert at caret")
    imgui.end_disabled()
    if not has_editor and imgui.is_item_hovered(
        imgui.HoveredFlags_.allow_when_disabled
    ):
        imgui.set_tooltip(
            "Click into the code editor first (so the caret is positioned)"
        )
    if (btn_clicked or pressed_enter) and can_insert:
        assert selected is not None
        filtering.insert_name(app, selected)
        keep_open = False
    imgui.same_line()
    if ghost_button("Close"):
        keep_open = False

    # Esc closes the picker — except when an inline input owns it (rename /
    # new-file / new-dir / add-tag).
    if (
        not input_owns_keys
        and not tag_input_was_focused
        and imgui.is_key_pressed(imgui.Key.escape, repeat=False)
    ):
        keep_open = False

    return keep_open
