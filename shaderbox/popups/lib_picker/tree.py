"""The left-hand tree column: dir/file/function nodes, context menus, inline inputs.

`build_tree` / `flatten_visible_leaves` turn the filtered candidates into a
keyed dir tree; the `draw_*` functions render it with right-click context menus
(new / rename / delete / reveal) and the inline create/rename inputs.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.editor_types import InlineInput
from shaderbox.paths import shader_lib_root
from shaderbox.popups.lib_picker.filtering import (
    copy_to_clipboard,
    insert_name,
    open_at_decl,
)
from shaderbox.shader_lib import ShaderLibFunction
from shaderbox.theme import COLOR, SPACE
from shaderbox.ui_primitives import _ellipsize, context_menu_style, ghost_button


@dataclass
class TreeNode:
    subdirs: set[str] = field(default_factory=set)
    files: list[Path] = field(default_factory=list)
    # Functions PER FILE — populated only for files that pass the filter.
    # Empty list = file has no visible functions (still drawn if the file
    # itself matched, e.g. via filename search later).
    functions_by_file: dict[Path, list[ShaderLibFunction]] = field(default_factory=dict)


def build_tree(
    visible: list[ShaderLibFunction], root: Path
) -> dict[tuple[str, ...], TreeNode]:
    # Build the dir tree keyed by tuple-of-parts. Only files containing at
    # least one VISIBLE function appear in the tree (so search/favs/tags filter
    # the tree naturally). The visible functions for each file are also stored
    # so the leaf walker can emit them in deterministic order.
    tree: dict[tuple[str, ...], TreeNode] = {(): TreeNode()}
    by_file: dict[Path, list[ShaderLibFunction]] = {}
    for fn in visible:
        by_file.setdefault(fn.file, []).append(fn)

    for file_path, fns in by_file.items():
        parts = file_path.relative_to(root).parts
        cur: tuple[str, ...] = ()
        for part in parts[:-1]:
            child = (*cur, part)
            tree.setdefault(child, TreeNode())
            tree[cur].subdirs.add(part)
            cur = child
        tree[cur].files.append(file_path)
        tree[cur].functions_by_file[file_path] = sorted(fns, key=lambda f: f.name)
    return tree


def flatten_visible_leaves(
    tree: dict[tuple[str, ...], TreeNode], dir_rel: tuple[str, ...]
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
        out.extend(flatten_visible_leaves(tree, (*dir_rel, subname)))
    return out


def draw_tree(app: App, tree: dict[tuple[str, ...], TreeNode], root: Path) -> bool:
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
    tree: dict[tuple[str, ...], TreeNode],
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
                insert_name(app, fn)
                close_picker = True
            if imgui.menu_item_simple("Open file at declaration"):
                open_at_decl(app, fn)
                close_picker = True
            if imgui.menu_item_simple("Copy name"):
                copy_to_clipboard(fn.name)
            is_fav = app.shader_lib_favorites.is_favorite(fn.name)
            if imgui.menu_item_simple("Unfavorite" if is_fav else "Favorite"):
                app.shader_lib_favorites.toggle(fn.name)
            imgui.end_popup()
    return close_picker
