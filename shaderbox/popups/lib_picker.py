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
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pyperclip
from imgui_bundle import imgui, imgui_ctx
from loguru import logger

from shaderbox.app import App, InlineInput, JumpRequest
from shaderbox.paths import shader_lib_root
from shaderbox.shader_lib.index import ShaderLibFunction
from shaderbox.theme import COLOR, SIZE, SPACE, fade
from shaderbox.ui_primitives import (
    _ellipsize,
    context_menu_style,
    draw_copyable_text,
    ghost_button,
    modal_window,
    pill_button,
    primary_button,
)

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
    candidates = _filter_functions(app)
    tree = _build_tree(candidates, root)
    visible_leaves = _flatten_visible_leaves(tree, ())

    rename_active = app.shader_lib_file_rename.target is not None
    new_file_active = app.shader_lib_file_new.target is not None
    new_dir_active = app.shader_lib_dir_new.target is not None
    input_owns_keys = rename_active or new_file_active or new_dir_active

    if not input_owns_keys:
        _handle_arrow_nav(app, visible_leaves)

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
    _draw_search_row(app, len(visible_leaves), len(app.shader_lib_index.functions))
    _draw_favs_and_reset_row(app)
    _draw_tag_bar(app)

    imgui.spacing()

    # ---------- Body: tree | preview ----------
    avail = imgui.get_content_region_avail()
    body_h = max(220.0, avail.y - SIZE.BTN_SM_H - float(SPACE.MD) * 2.0)

    tree_w = max(_TREE_W_MIN, avail.x * _TREE_FRAC)
    with imgui_ctx.begin_child("##tree_col", size=imgui.ImVec2(tree_w, body_h)):
        imgui.text_colored(COLOR.FG_DIM, "Right-click for actions")
        if _draw_tree(app, tree, root):
            keep_open = False
    imgui.same_line(spacing=float(SPACE.MD))
    with imgui_ctx.begin_child("##preview", size=imgui.ImVec2(0.0, body_h)):
        _draw_preview(app, _selected_function(app, candidates), root)

    imgui.spacing()

    # ---------- Action row ----------
    selected = _selected_function(app, candidates)
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
        _insert_name(app, selected)
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


# ----------------------------------------------------------------------------
# Top bar
# ----------------------------------------------------------------------------


def _draw_favs_and_reset_row(app: App) -> None:
    # Yellow Favs pill (active=solid, inactive=faded). Pink Reset appears
    # right after when any tag is disabled.
    if pill_button(
        "Favs##favs_toggle", color=COLOR.FAVS, active=app.shader_lib_picker_favs_only
    ):
        app.shader_lib_picker_favs_only = not app.shader_lib_picker_favs_only
    if app.shader_lib_picker_disabled_tags:
        imgui.same_line()
        if pill_button("Reset##reset_pill", color=COLOR.RESET_PILL, active=True):
            app.shader_lib_picker_disabled_tags.clear()


def _draw_search_row(app: App, visible_count: int, total: int) -> None:
    if app.shader_lib_picker_just_opened:
        imgui.set_keyboard_focus_here(0)
    imgui.set_next_item_width(imgui.get_content_region_avail().x - 200.0)
    _, app.shader_lib_picker_query = imgui.input_text(
        "##q", app.shader_lib_picker_query
    )
    imgui.same_line()
    imgui.text_colored(COLOR.FG_DIM, f"{visible_count} / {total}")
    # Live tag-prefix feedback (`#noi` → "matching: #noise").
    tag_prefixes, _ = _parse_query_tags(app.shader_lib_picker_query.strip().lower())
    if tag_prefixes:
        matched = _resolve_tag_prefix_matches(app, tag_prefixes)
        imgui.same_line()
        if matched:
            imgui.text_colored(
                COLOR.FG_DIM, "matching: " + " ".join(f"#{t}" for t in matched)
            )
        else:
            imgui.text_colored(COLOR.STATE_WARN, "no tags match")


def _draw_tag_bar(app: App) -> None:
    # Blue pills (active = blue fill, disabled = dim). Wraps onto multiple rows.
    # Tag bar shows every tag in the lib; #prefix in the search bar narrows it.
    tag_prefixes, _ = _parse_query_tags(app.shader_lib_picker_query.strip().lower())
    all_tags_set = app.shader_lib_tags.all_tags()
    if tag_prefixes:
        bar_tags = sorted(
            t for t in all_tags_set if any(t.startswith(p) for p in tag_prefixes)
        )
    else:
        bar_tags = sorted(all_tags_set)
    if not bar_tags:
        return

    style = imgui.get_style()
    btn_pad = style.frame_padding.x * 2.0
    item_spacing = style.item_spacing.x
    row_width = imgui.get_content_region_avail().x
    cursor_x = 0.0

    for tag in bar_tags:
        label = f"#{tag}"
        full_label_id = f"#{tag}##bar_{tag}"
        w = imgui.calc_text_size(label).x + btn_pad
        if cursor_x > 0.0:
            if cursor_x + item_spacing + w <= row_width:
                imgui.same_line()
                cursor_x += item_spacing + w
            else:
                cursor_x = w
        else:
            cursor_x = w
        is_enabled = tag not in app.shader_lib_picker_disabled_tags
        if is_enabled:
            clicked = pill_button(
                full_label_id, color=COLOR.TAG, active=False, inactive_alpha=0.7
            )
        else:
            clicked = pill_button(
                full_label_id,
                color=COLOR.FG_DIM,
                active=False,
                inactive_alpha=0.15,
                text_color=COLOR.FG_DIM,
            )
        if clicked:
            if imgui.get_io().key_ctrl:
                # Isolate ACROSS the WHOLE lib (not just visible) — `#prefix`
                # may have narrowed the bar, but isolate means "only this tag".
                app.shader_lib_picker_disabled_tags = {
                    t for t in app.shader_lib_tags.all_tags() if t != tag
                }
            elif is_enabled:
                app.shader_lib_picker_disabled_tags.add(tag)
            else:
                app.shader_lib_picker_disabled_tags.discard(tag)


# ----------------------------------------------------------------------------
# Tree
# ----------------------------------------------------------------------------


@dataclass
class _TreeNode:
    subdirs: set[str] = field(default_factory=set)
    files: list[Path] = field(default_factory=list)
    # Functions PER FILE — populated only for files that pass the filter.
    # Empty list = file has no visible functions (still drawn if the file
    # itself matched, e.g. via filename search later).
    functions_by_file: dict[Path, list[ShaderLibFunction]] = field(default_factory=dict)


def _build_tree(
    visible: list[ShaderLibFunction], root: Path
) -> dict[tuple[str, ...], _TreeNode]:
    # Build the dir tree keyed by tuple-of-parts. Only files containing at
    # least one VISIBLE function appear in the tree (so search/favs/tags filter
    # the tree naturally). The visible functions for each file are also stored
    # so the leaf walker can emit them in deterministic order.
    tree: dict[tuple[str, ...], _TreeNode] = {(): _TreeNode()}
    by_file: dict[Path, list[ShaderLibFunction]] = {}
    for fn in visible:
        by_file.setdefault(fn.file, []).append(fn)

    for file_path, fns in by_file.items():
        parts = file_path.relative_to(root).parts
        cur: tuple[str, ...] = ()
        for part in parts[:-1]:
            child = (*cur, part)
            tree.setdefault(child, _TreeNode())
            tree[cur].subdirs.add(part)
            cur = child
        tree[cur].files.append(file_path)
        tree[cur].functions_by_file[file_path] = sorted(fns, key=lambda f: f.name)
    return tree


def _flatten_visible_leaves(
    tree: dict[tuple[str, ...], _TreeNode], dir_rel: tuple[str, ...]
) -> list[ShaderLibFunction]:
    # Emit leaves in the SAME order the draw walks them — files-before-subdirs,
    # alphabetical at each level. Arrow-nav relies on this for natural up/down.
    out: list[ShaderLibFunction] = []
    node = tree.get(dir_rel)
    if node is None:
        return out
    for f in sorted(node.files, key=lambda p: p.name.lower()):
        out.extend(node.functions_by_file.get(f, []))
    for subname in sorted(node.subdirs, key=str.lower):
        out.extend(_flatten_visible_leaves(tree, (*dir_rel, subname)))
    return out


def _draw_tree(app: App, tree: dict[tuple[str, ...], _TreeNode], root: Path) -> bool:
    # Wraps the whole tree in a `/` root node (collapsible). Right-clicking the
    # root opens the dir context menu (New file / New subdir / Reveal).
    # Returns True iff a "Open at decl" action fired and the picker should close.
    flags = imgui.TreeNodeFlags_.default_open | imgui.TreeNodeFlags_.span_avail_width
    open_root = imgui.tree_node_ex("/##dirnode_root", flags)
    _draw_dir_context_menu(app, (), is_root=True)
    if not open_root:
        return False
    _draw_dir_new_inputs_for(app, ())
    close_picker = _draw_tree_children(app, tree, (), root)
    imgui.tree_pop()
    return close_picker


def _draw_tree_children(
    app: App,
    tree: dict[tuple[str, ...], _TreeNode],
    dir_rel: tuple[str, ...],
    root: Path,
) -> bool:
    # Returns True iff a function-leaf "Open at decl" closed the picker.
    close_picker = False
    node = tree.get(dir_rel)
    if node is None:
        return False
    # Files first (alphabetical), then subdirs.
    for file_path in sorted(node.files, key=lambda p: p.name.lower()):
        if _draw_file_node(app, file_path, node.functions_by_file.get(file_path, [])):
            close_picker = True
    for subname in sorted(node.subdirs, key=str.lower):
        child = (*dir_rel, subname)
        flags = (
            imgui.TreeNodeFlags_.default_open | imgui.TreeNodeFlags_.span_avail_width
        )
        node_id = f"{subname}##dirnode_{'/'.join(child)}"
        # Armed state is stored as the ABSOLUTE path (the context menu arms with
        # `shader_lib_root() / dir_rel`); compare like-for-like or the red tint never shows.
        is_armed = app.shader_lib_dir_delete_armed == root / Path(*child)
        # Force-open if a new-file / new-dir input is targeted at this dir or
        # any descendant — otherwise the inline input would render inside an
        # invisible collapsed branch.
        if _dir_contains_pending_input(app, child):
            imgui.set_next_item_open(True, imgui.Cond_.always)
        if is_armed:
            imgui.push_style_color(imgui.Col_.text, COLOR.STATE_ERROR)
        open_dir = imgui.tree_node_ex(node_id, flags)
        if is_armed:
            imgui.pop_style_color(1)
        _draw_dir_context_menu(app, child, is_root=False)
        if open_dir:
            _draw_dir_new_inputs_for(app, child)
            if _draw_tree_children(app, tree, child, root):
                close_picker = True
            imgui.tree_pop()
    return close_picker


def _dir_contains_pending_input(app: App, dir_rel: tuple[str, ...]) -> bool:
    # True iff the inline new-file or new-dir input is targeted at this dir
    # or one of its descendants. Drives auto-expand so the input is visible.
    dir_path = Path(*dir_rel)

    def _is_descendant(target: Path | None) -> bool:
        if target is None:
            return False
        try:
            target.relative_to(dir_path)
        except ValueError:
            return False
        return True

    return _is_descendant(app.shader_lib_file_new.target) or _is_descendant(
        app.shader_lib_dir_new.target
    )


def _draw_dir_context_menu(app: App, dir_rel: tuple[str, ...], is_root: bool) -> None:
    # `begin_popup_context_item` opens on right-click of the last submitted
    # item (the tree_node_ex header). Unique id per dir.
    popup_id = f"##dirctx_{'/'.join(dir_rel) if dir_rel else 'root'}"
    with context_menu_style():
        if imgui.begin_popup_context_item(popup_id):
            dir_path_rel = Path(*dir_rel) if dir_rel else Path()
            if imgui.menu_item_simple("New file here"):
                app.begin_shader_lib_file_new_in(dir_path_rel)
            if imgui.menu_item_simple("New subdirectory"):
                app.begin_shader_lib_dir_new_in(dir_path_rel)
            if imgui.menu_item_simple("Reveal in file manager"):
                abs_path = (
                    shader_lib_root() / dir_path_rel if dir_rel else shader_lib_root()
                )
                app.reveal_shader_lib_file_in_manager(abs_path)
            if not is_root:
                imgui.separator()
                abs_path = shader_lib_root() / dir_path_rel
                is_armed = app.shader_lib_dir_delete_armed == abs_path
                label = (
                    "Confirm delete (recursive)"
                    if is_armed
                    else "Delete directory (recursive)"
                )
                imgui.push_style_color(imgui.Col_.text, COLOR.STATE_ERROR)
                clicked = imgui.menu_item_simple(label)
                imgui.pop_style_color(1)
                if clicked:
                    if is_armed:
                        app.delete_shader_lib_dir(abs_path)
                    else:
                        app.arm_shader_lib_dir_delete(abs_path)
            imgui.end_popup()


def _draw_dir_new_inputs_for(app: App, dir_rel: tuple[str, ...]) -> None:
    # Both "New file:" and "New dir:" inline inputs share the same chrome —
    # one helper, two configs.
    _draw_inline_new_input(
        state=app.shader_lib_file_new,
        label="New file:",
        id_prefix="new_file_in",
        dir_rel=dir_rel,
        commit=app.commit_shader_lib_file_new,
        cancel=app.cancel_shader_lib_file_new,
        on_create=app.open_shader_lib_file,
    )
    _draw_inline_new_input(
        state=app.shader_lib_dir_new,
        label="New dir:",
        id_prefix="new_dir_in",
        dir_rel=dir_rel,
        commit=app.commit_shader_lib_dir_new,
        cancel=app.cancel_shader_lib_dir_new,
        on_create=None,
    )


def _draw_inline_new_input(
    *,
    state: InlineInput,
    label: str,
    id_prefix: str,
    dir_rel: tuple[str, ...],
    commit: Callable[[], Path | None],
    cancel: Callable[[], None],
    on_create: Callable[[Path], object] | None,
) -> None:
    if state.target is None or state.target != Path(*dir_rel):
        return
    imgui.indent(float(SPACE.MD))
    imgui.text_colored(COLOR.FG_DIM, label)
    imgui.same_line()
    cancel_w = imgui.calc_text_size("x").x + float(SPACE.MD) * 2.0
    imgui.set_next_item_width(imgui.get_content_region_avail().x - cancel_w)
    if state.needs_focus:
        imgui.set_keyboard_focus_here(0)
        state.needs_focus = False
    changed, state.buf = imgui.input_text(
        f"##{id_prefix}_{dir_rel}",
        state.buf,
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    input_focused = imgui.is_item_focused()
    if changed:
        created = commit()
        if created is not None and on_create is not None:
            on_create(created)
    if input_focused and imgui.is_key_pressed(imgui.Key.escape, repeat=False):
        cancel()
    imgui.same_line()
    if ghost_button(f"x##cancel_{id_prefix}_{dir_rel}"):
        cancel()
    imgui.unindent(float(SPACE.MD))


def _draw_file_node(app: App, path: Path, fns: list[ShaderLibFunction]) -> bool:
    # File becomes a tree_node_ex (collapsible) with its functions as children.
    # Returns True iff a child function's "Open at decl" closed the picker.
    is_renaming = app.shader_lib_file_rename.target == path
    if is_renaming:
        _draw_file_rename_input(app, path)
        return False

    is_armed = app.shader_lib_file_delete_armed == path
    if is_armed:
        imgui.push_style_color(imgui.Col_.text, COLOR.STATE_ERROR)
    flags = imgui.TreeNodeFlags_.default_open | imgui.TreeNodeFlags_.span_avail_width
    open_file = imgui.tree_node_ex(f"{path.name}##filenode_{path}", flags)
    if is_armed:
        imgui.pop_style_color(1)
    close_picker = False
    _draw_file_context_menu(app, path)
    if open_file:
        for fn in fns:
            if _draw_function_leaf(app, fn):
                close_picker = True
        imgui.tree_pop()
    return close_picker


def _draw_file_context_menu(app: App, path: Path) -> None:
    with context_menu_style():
        if imgui.begin_popup_context_item(f"##filectx_{path}"):
            if imgui.menu_item_simple("Rename"):
                app.begin_shader_lib_file_rename(path)
            if imgui.menu_item_simple("Reveal in file manager"):
                app.reveal_shader_lib_file_in_manager(path)
            imgui.separator()
            is_armed = app.shader_lib_file_delete_armed == path
            label = "Confirm delete" if is_armed else "Delete"
            imgui.push_style_color(imgui.Col_.text, COLOR.STATE_ERROR)
            clicked = imgui.menu_item_simple(label)
            imgui.pop_style_color(1)
            if clicked:
                if is_armed:
                    app.delete_shader_lib_file(path)
                else:
                    app.arm_shader_lib_file_delete(path)
            imgui.end_popup()


def _draw_file_rename_input(app: App, path: Path) -> None:
    imgui.indent(float(SPACE.MD))
    # Reserve room on the right for the cancel `x` button.
    cancel_w = imgui.calc_text_size("x").x + float(SPACE.MD) * 2.0
    imgui.set_next_item_width(imgui.get_content_region_avail().x - cancel_w)
    if app.shader_lib_file_rename.needs_focus:
        imgui.set_keyboard_focus_here(0)
        app.shader_lib_file_rename.needs_focus = False
    changed, app.shader_lib_file_rename.buf = imgui.input_text(
        f"##ren_in_{path}",
        app.shader_lib_file_rename.buf,
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    input_focused = imgui.is_item_focused()
    if input_focused and imgui.is_key_pressed(imgui.Key.escape, repeat=False):
        app.cancel_shader_lib_file_rename()
    if changed:
        app.rename_shader_lib_file(path, app.shader_lib_file_rename.buf)
    imgui.same_line()
    if ghost_button(f"x##cancel_ren_{path}"):
        app.cancel_shader_lib_file_rename()
    imgui.unindent(float(SPACE.MD))


def _draw_function_leaf(app: App, fn: ShaderLibFunction) -> bool:
    # Function leaf: favs star + selectable. Right-click → context menu
    # (Insert / Open at decl / Copy / Toggle favorite). Returns True if the
    # "Open at decl" action fired and the picker should close.
    is_selected = app.shader_lib_picker_selected_function == fn.name
    is_fav = app.shader_lib_favorites.is_favorite(fn.name)
    star_label = ("*" if is_fav else "o") + f"##fav_{fn.name}"
    imgui.push_style_color(imgui.Col_.text, COLOR.FAVS if is_fav else COLOR.FG_DIM)
    if imgui.small_button(star_label):
        app.shader_lib_favorites.toggle(fn.name)
    imgui.pop_style_color(1)
    imgui.same_line()
    label = f"{fn.name}##leaf_{fn.name}"
    if imgui.selectable(label, is_selected)[0]:
        app.shader_lib_picker_selected_function = fn.name
        # Clicking a function disarms any pending file/dir delete — moving on
        # to inserting/exploring is a clear "I'm not deleting" signal.
        app.arm_shader_lib_file_delete(None)
        app.arm_shader_lib_dir_delete(None)
    close_picker = _draw_function_context_menu(app, fn)
    if (
        is_selected
        and not app.shader_lib_picker_just_opened
        and not imgui.is_item_visible()
    ):
        imgui.set_scroll_here_y(0.5)
    if fn.doc:
        imgui.same_line()
        sep = "  - "
        sep_w = imgui.calc_text_size(sep).x
        avail = imgui.get_content_region_avail().x - sep_w
        if avail > imgui.calc_text_size("...").x:
            first_doc_line = fn.doc.splitlines()[0]
            imgui.text_colored(COLOR.FG_DIM, sep + _ellipsize(first_doc_line, avail))
    return close_picker


def _draw_function_context_menu(app: App, fn: ShaderLibFunction) -> bool:
    close_picker = False
    has_editor = app.editor_was_ever_focused and app.current_editor_path is not None
    with context_menu_style():
        if imgui.begin_popup_context_item(f"##fnctx_{fn.name}"):
            if imgui.menu_item_simple("Insert at caret", enabled=has_editor):
                _insert_name(app, fn)
                close_picker = True
            if imgui.menu_item_simple("Open file at declaration"):
                _open_at_decl(app, fn)
                close_picker = True
            if imgui.menu_item_simple("Copy name"):
                _copy_to_clipboard(fn.name)
            is_fav = app.shader_lib_favorites.is_favorite(fn.name)
            if imgui.menu_item_simple("Unfavorite" if is_fav else "Favorite"):
                app.shader_lib_favorites.toggle(fn.name)
            imgui.end_popup()
    return close_picker


def _copy_to_clipboard(text: str) -> None:
    try:
        pyperclip.copy(text)
    except pyperclip.PyperclipException as e:
        logger.warning(f"Could not copy: {e}")


# ----------------------------------------------------------------------------
# Preview pane
# ----------------------------------------------------------------------------


def _draw_preview(app: App, selected: ShaderLibFunction | None, root: Path) -> None:
    if selected is None:
        imgui.text_colored(COLOR.FG_DIM, "(no function selected)")
        return

    # Click-to-copy file path. The label shows the relative path; the copy
    # value is the absolute on-disk path (more useful for "open in another tool").
    rel = selected.file.relative_to(root)
    draw_copyable_text(
        str(rel),
        copy_value=str(selected.file),
        color=COLOR.FG_DIM,
        tooltip="Click to copy file path",
    )

    imgui.text_colored(COLOR.ACCENT_PRIMARY, selected.signature)

    if selected.doc:
        imgui.spacing()
        imgui.push_text_wrap_pos(0.0)
        imgui.text_colored(COLOR.FG_DIM, selected.doc)
        imgui.pop_text_wrap_pos()

    imgui.spacing()
    _draw_function_tag_editor(app, selected)

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    imgui.push_style_color(imgui.Col_.child_bg, fade(COLOR.BG_SURFACE, 0.5))
    with imgui_ctx.begin_child(
        "##body",
        child_flags=imgui.ChildFlags_.borders,
    ):
        imgui.text_unformatted(selected.body)
    imgui.pop_style_color(1)


def _draw_function_tag_editor(app: App, fn: ShaderLibFunction) -> None:
    # Blue pills (active tag fill) with an `x` on the right to remove. Add-tag
    # input + suggestions row below.
    current_tags = sorted(app.shader_lib_tags.tags_for(fn.name))
    for tag in current_tags:
        if pill_button(
            f"#{tag} x##rmtag_{fn.name}_{tag}",
            color=COLOR.TAG,
            active=False,
            inactive_alpha=0.5,
        ):
            app.shader_lib_tags.remove(fn.name, tag)
        imgui.same_line()

    imgui.set_next_item_width(140.0)
    changed, app.shader_lib_picker_new_tag_buf = imgui.input_text(
        f"##newtag_{fn.name}",
        app.shader_lib_picker_new_tag_buf,
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    # Update the "input owns Enter" flag immediately so the outer Enter check
    # on the NEXT frame skips Insert+close when the user is still typing here.
    app.shader_lib_picker_tag_input_focused = imgui.is_item_focused()
    imgui.same_line()
    add_pressed = ghost_button(f"+ Add##addtag_{fn.name}")
    buf = app.shader_lib_picker_new_tag_buf.strip().lstrip("#").lower()
    if (changed or add_pressed) and buf:
        app.shader_lib_tags.add(fn.name, buf)
        app.shader_lib_picker_new_tag_buf = ""
        return

    if buf:
        suggestions = _autocomplete_tags(app, buf, exclude=set(current_tags))
        if suggestions:
            imgui.push_font(app.font_12, app.font_12.legacy_size)
            imgui.text_colored(COLOR.FG_DIM, "Existing:")
            for s in suggestions[:8]:
                imgui.same_line()
                if pill_button(
                    f"#{s}##sugg_{fn.name}_{s}",
                    color=COLOR.FG_DIM,
                    active=False,
                    inactive_alpha=0.2,
                    text_color=COLOR.FG_PRIMARY,
                ):
                    app.shader_lib_tags.add(fn.name, s)
                    app.shader_lib_picker_new_tag_buf = ""
            imgui.pop_font()


def _autocomplete_tags(app: App, buf: str, exclude: set[str]) -> list[str]:
    all_tags = app.shader_lib_tags.all_tags() - exclude

    def rank(tag: str) -> tuple[int, str]:
        if tag.startswith(buf):
            return (0, tag)
        if buf in tag:
            return (1, tag)
        return (2, tag)

    candidates = [(rank(t), t) for t in all_tags if buf in t]
    candidates.sort(key=lambda x: x[0])
    return [t for (r, _), t in candidates if r < 2]


# ----------------------------------------------------------------------------
# Filtering, selection, keyboard
# ----------------------------------------------------------------------------


def _handle_arrow_nav(app: App, visible: list[ShaderLibFunction]) -> None:
    if not visible:
        app.shader_lib_picker_selected_function = ""
        return
    down = imgui.is_key_pressed(imgui.Key.down_arrow, repeat=True)
    up = imgui.is_key_pressed(imgui.Key.up_arrow, repeat=True)
    if not (down or up):
        # Clamp: if the current selection is no longer visible (filter change),
        # fall back to the first visible leaf.
        names = [fn.name for fn in visible]
        if app.shader_lib_picker_selected_function not in names:
            app.shader_lib_picker_selected_function = visible[0].name
        return
    names = [fn.name for fn in visible]
    try:
        idx = names.index(app.shader_lib_picker_selected_function)
    except ValueError:
        idx = 0
    step = 1 if down else -1
    app.shader_lib_picker_selected_function = names[(idx + step) % len(names)]


def _selected_function(
    app: App, candidates: list[ShaderLibFunction]
) -> ShaderLibFunction | None:
    if not app.shader_lib_picker_selected_function:
        return None
    for fn in candidates:
        if fn.name == app.shader_lib_picker_selected_function:
            return fn
    # The selected function is hidden by current filters. Fall back to None so
    # the preview shows "(no selection)" rather than a stale function.
    return None


def _filter_functions(app: App) -> list[ShaderLibFunction]:
    raw_query: str = app.shader_lib_picker_query.strip().lower()
    tag_prefixes, text_query = _parse_query_tags(raw_query)
    disabled = app.shader_lib_picker_disabled_tags

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
    if app.shader_lib_picker_favs_only:
        candidates = [
            fn for fn in candidates if app.shader_lib_favorites.is_favorite(fn.name)
        ]
    return [fn for fn in candidates if passes(fn)]


def _parse_query_tags(query: str) -> tuple[list[str], str]:
    prefixes: list[str] = []
    parts: list[str] = []
    for tok in query.split():
        if tok.startswith("#") and len(tok) > 1:
            prefixes.append(tok[1:])
        else:
            parts.append(tok)
    return prefixes, " ".join(parts).strip()


def _resolve_tag_prefix_matches(app: App, prefixes: list[str]) -> list[str]:
    if not prefixes:
        return []
    all_tags = app.shader_lib_tags.all_tags()
    matched: set[str] = set()
    for p in prefixes:
        matched.update(t for t in all_tags if t.startswith(p))
    return sorted(matched)


def _insert_name(app: App, fn: ShaderLibFunction) -> None:
    session = app.get_current_session_if_exists()
    if session is None:
        logger.warning("No editor session active; can't insert lib name")
        return
    session.editor.replace_text_in_current_cursor(fn.name)


def _open_at_decl(app: App, fn: ShaderLibFunction) -> None:
    # Retained for hotkeys / callers that may invoke "Open function file"
    # directly (e.g. a future shortcut).
    app.open_shader_lib_file(fn.file)
    app.editor_jump_request = JumpRequest(fn.file, fn.line_in_file, 0)
