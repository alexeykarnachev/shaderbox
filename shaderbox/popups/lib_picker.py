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
from shaderbox.theme import COLOR, SIZE, SPACE, fade
from shaderbox.ui_primitives import _ellipsize, button, ghost_button, primary_button

_LABEL = "Library##picker"
_POPUP_W = 760.0
_POPUP_H = 480.0
_LIST_FRAC = 0.45  # fraction of body width for the left list; preview takes the rest
_LIST_W_MIN = 220.0  # below this width the rows look cramped — clamp the fraction
_NEW_FILE_NAME_KEY = "##new_lib_file_name"


def draw_lib_picker(app: App) -> None:
    if not app.is_lib_picker_open:
        return

    if not imgui.is_popup_open(_LABEL):
        imgui.open_popup(_LABEL)

    # `first_use_ever` (not `appearing`): seed once, then imgui.ini persists the
    # user's manual resize across re-opens. `appearing` would clobber it.
    imgui.set_next_window_size(
        imgui.ImVec2(_POPUP_W, _POPUP_H), imgui.Cond_.first_use_ever
    )
    with imgui_ctx.begin_popup_modal(_LABEL) as popup:
        if not popup.visible:
            return
        keep_open = _draw_body(app)
        if not keep_open:
            app.is_lib_picker_open = False
            imgui.close_current_popup()


def _draw_body(app: App) -> bool:
    keep_open: bool = True
    matches = _matching_functions(app)

    # Keyboard nav: arrow up/down wraps through the result list. The input
    # field stays focused for the entire modal lifetime (so the user can type
    # at any moment); arrows are intercepted before the input consumes them.
    _handle_arrow_nav(app, len(matches))

    # Enter activates whatever's selected (or first match).
    pressed_enter = imgui.is_key_pressed(imgui.Key.enter, repeat=False)

    # Search row. Autofocus on the first frame after open; after that imgui
    # keeps focus on the input as long as nothing else grabs it.
    if app.lib_picker_just_opened:
        imgui.set_keyboard_focus_here(0)
    # Width tracks the list column (fractional of the body width).
    list_w = max(_LIST_W_MIN, imgui.get_content_region_avail().x * _LIST_FRAC)
    imgui.set_next_item_width(list_w)
    _, app.lib_picker_query = imgui.input_text("##q", app.lib_picker_query)
    imgui.same_line()
    # Favs toggle: same label always (no leading star / no leading spaces — the
    # text would otherwise visibly shift inside the button area on toggle, since
    # space widths != star widths). Active state is the filled accent color.
    if app.lib_picker_favs_only:
        imgui.push_style_color(imgui.Col_.button, COLOR.ACCENT_PRIMARY)
        imgui.push_style_color(imgui.Col_.text, COLOR.BG_APP)
        if imgui.button("Favs"):
            app.lib_picker_favs_only = False
        imgui.pop_style_color(2)
    else:
        if ghost_button("Favs"):
            app.lib_picker_favs_only = True
    # Gap so the count doesn't read as a value attached to Favs.
    imgui.same_line(spacing=float(SPACE.LG))
    imgui.text_colored(
        COLOR.FG_DIM, f"{len(matches)} of {len(app.lib_index.functions)}"
    )

    # Show which tags the user's `#prefix` tokens resolve to as they type. This
    # is the live feedback for partial-prefix matches (e.g. `#nois` -> #noise).
    tag_prefixes, _ = _parse_query_tags(app.lib_picker_query.strip().lower())
    if tag_prefixes:
        matched = _resolve_tag_prefix_matches(app, tag_prefixes)
        imgui.same_line()
        if matched:
            imgui.text_colored(
                COLOR.FG_DIM, "matching: " + " ".join(f"#{t}" for t in matched)
            )
        else:
            imgui.text_colored(COLOR.STATE_WARN, "no tags match")

    # Global tag bar: every tag in the lib as a clickable pill. Enabled (accent
    # color) by default — click to disable (dim color), click again to re-enable.
    # Ctrl+click isolates this tag (disables all others). A "Reset" pill clears
    # the disabled set. A function passes the tag filter if it has at least one
    # still-enabled tag, OR if it has no tags at all (always visible).
    #
    # `#prefix` tokens in the search bar narrow the visible pills to just the
    # matches — clear the search to see the full bar / disabled-state.
    all_tags_set = app.lib_tags.all_tags()
    if tag_prefixes:
        bar_tags = sorted(
            t for t in all_tags_set if any(t.startswith(p) for p in tag_prefixes)
        )
    else:
        bar_tags = sorted(all_tags_set)

    if bar_tags:
        imgui.dummy(imgui.ImVec2(0.0, float(SPACE.MD)))
        imgui.text_colored(
            COLOR.FG_DIM,
            "Click to toggle a tag. Ctrl+click to keep only this tag.",
        )
        imgui.dummy(imgui.ImVec2(0.0, float(SPACE.XS)))
        _draw_tag_bar(app, bar_tags)
        imgui.dummy(imgui.ImVec2(0.0, float(SPACE.MD)))

    selected = _selected_function(app, matches)

    # Left list + right preview. Fractional split: the list takes ~45% of the
    # body width (clamped to a sensible minimum), the preview takes the rest.
    avail = imgui.get_content_region_avail()
    body_h = avail.y - SIZE.BTN_SM_H - float(SPACE.MD) * 2.0
    body_list_w = max(_LIST_W_MIN, avail.x * _LIST_FRAC)
    with imgui_ctx.begin_child("##list", size=imgui.ImVec2(body_list_w, body_h)):
        _draw_list(app, matches)

    imgui.same_line(spacing=float(SPACE.MD))

    with imgui_ctx.begin_child("##preview", size=imgui.ImVec2(0.0, body_h)):
        _draw_preview(app, selected)

    imgui.spacing()

    # Action row: Insert / Open file / New file / Close.
    can_act = selected is not None
    if (
        primary_button("Insert at caret", width=SIZE.BTN_SM_W) or pressed_enter
    ) and can_act:
        assert selected is not None  # gated by can_act
        _insert_name(app, selected)
        keep_open = False
    imgui.same_line()
    if button("Open file", width=SIZE.BTN_SM_W) and can_act:
        assert selected is not None
        _open_at_decl(app, selected)
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

    # Esc closes the picker.
    if imgui.is_key_pressed(imgui.Key.escape, repeat=False):
        keep_open = False

    app.lib_picker_just_opened = False
    return keep_open


def _draw_tag_bar(app: App, all_tags: list[str]) -> None:
    # Pills wrap onto multiple rows when running out of horizontal space.
    #
    # Wrap math: track row width manually. `get_content_region_avail().x` and
    # `get_cursor_pos_x()` after a row-advance don't compose cleanly inside a
    # popup (cursor jumps to a new line as soon as you don't call same_line(),
    # so post-pill queries report next-row coordinates, not current-row). The
    # only fix is to accumulate row width ourselves.
    items: list[str] = [*all_tags, "__RESET__"]
    style = imgui.get_style()
    btn_pad = style.frame_padding.x * 2.0
    item_spacing = style.item_spacing.x
    row_width = imgui.get_content_region_avail().x  # captured before any pill draws

    def width_for(label: str) -> float:
        return imgui.calc_text_size(label).x + btn_pad

    cursor_x = 0.0  # accumulated width of pills (incl. spacings) on the current row
    for item in items:
        is_reset = item == "__RESET__"
        if is_reset and not app.lib_picker_disabled_tags:
            continue
        label = "Reset" if is_reset else f"#{item}"
        full_label_id = "Reset##bar_reset" if is_reset else f"#{item}##bar_{item}"
        w = width_for(label)
        # If this is NOT the first pill on a row, decide whether to inline it
        # or let it drop to a new row.
        if cursor_x > 0.0:
            if cursor_x + item_spacing + w <= row_width:
                imgui.same_line()
                cursor_x += item_spacing + w
            else:
                # Wrap: do NOT call same_line; the natural cursor advance from
                # the previous pill already moved to a new line. Reset the
                # accumulator.
                cursor_x = w
        else:
            cursor_x = w

        # Draw the pill.
        if is_reset:
            imgui.push_style_color(imgui.Col_.button, fade(COLOR.STATE_WARN, 0.4))
            imgui.push_style_color(imgui.Col_.text, COLOR.BG_APP)
            if imgui.small_button(full_label_id):
                app.lib_picker_disabled_tags.clear()
            imgui.pop_style_color(2)
        else:
            tag = item
            is_enabled = tag not in app.lib_picker_disabled_tags
            if is_enabled:
                imgui.push_style_color(
                    imgui.Col_.button, fade(COLOR.ACCENT_PRIMARY, 0.6)
                )
                imgui.push_style_color(imgui.Col_.text, COLOR.BG_APP)
            else:
                imgui.push_style_color(imgui.Col_.button, fade(COLOR.FG_DIM, 0.15))
                imgui.push_style_color(imgui.Col_.text, COLOR.FG_DIM)
            clicked = imgui.small_button(full_label_id)
            imgui.pop_style_color(2)
            if clicked:
                if imgui.get_io().key_ctrl:
                    # Isolate ACROSS THE WHOLE LIB, not just visible pills —
                    # the bar may be narrowed by a `#prefix` search but the
                    # isolate operation always means "show only this tag."
                    app.lib_picker_disabled_tags = {
                        t for t in app.lib_tags.all_tags() if t != tag
                    }
                elif is_enabled:
                    app.lib_picker_disabled_tags.add(tag)
                else:
                    app.lib_picker_disabled_tags.discard(tag)


def _draw_list(app: App, matches: list[LibFunction]) -> None:
    for i, fn in enumerate(matches):
        is_selected = i == app.lib_picker_selected_idx
        is_fav = app.lib_favorites.is_favorite(fn.name)
        # Star toggle on the left of each row — click to mark favorite.
        star_label = ("*" if is_fav else "o") + f"##fav_{fn.name}"
        imgui.push_style_color(
            imgui.Col_.text, COLOR.ACCENT_PRIMARY if is_fav else COLOR.FG_DIM
        )
        if imgui.small_button(star_label):
            app.lib_favorites.toggle(fn.name)
        imgui.pop_style_color(1)
        imgui.same_line()
        # Row selectable. ASCII-only (RUF001 trips on em-dash).
        label = f"{fn.name}##row_{i}"
        if imgui.selectable(label, is_selected)[0]:
            app.lib_picker_selected_idx = i
        if imgui.is_item_hovered():
            imgui.set_tooltip(fn.signature)
        # Auto-scroll to keep the selected row visible when nav-keyed. Only nudge
        # scroll when off-screen — calling every frame would lock the viewport.
        if (
            is_selected
            and not app.lib_picker_just_opened
            and not imgui.is_item_visible()
        ):
            imgui.set_scroll_here_y(0.5)
        # Doc on the same row, dim, pixel-ellipsized to whatever space remains
        # after the name. No tag chips here — tags live in the global tag bar
        # and per-function in the preview pane.
        if fn.doc:
            imgui.same_line()
            sep = "  - "
            sep_w = imgui.calc_text_size(sep).x
            avail = imgui.get_content_region_avail().x - sep_w
            if avail > imgui.calc_text_size("...").x:
                first_doc_line = fn.doc.splitlines()[0]
                imgui.text_colored(
                    COLOR.FG_DIM, sep + _ellipsize(first_doc_line, avail)
                )


def _draw_preview(app: App, selected: LibFunction | None) -> None:
    if selected is None:
        imgui.text_colored(COLOR.FG_DIM, "(no match)")
        return

    # Hierarchy: small dim file caption, accent-colored signature, dim doc,
    # bordered body block with a subtle bg tint to make it pop as code.
    imgui.push_font(app.font_12, app.font_12.legacy_size)
    imgui.text_colored(COLOR.FG_DIM, str(selected.file.name))
    imgui.pop_font()

    imgui.text_colored(COLOR.ACCENT_PRIMARY, selected.signature)

    if selected.doc:
        imgui.spacing()
        imgui.push_text_wrap_pos(0.0)
        imgui.text_colored(COLOR.FG_DIM, selected.doc)
        imgui.pop_text_wrap_pos()

    # Per-function tags: current tags as removable pills + inline "add tag" input.
    imgui.spacing()
    _draw_function_tag_editor(app, selected)

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    # Body in a bordered, slightly-tinted child so it reads as a code block,
    # distinct from the header chrome above.
    imgui.push_style_color(imgui.Col_.child_bg, fade(COLOR.BG_SURFACE, 0.5))
    with imgui_ctx.begin_child(
        "##body",
        child_flags=imgui.ChildFlags_.borders,
    ):
        imgui.text_unformatted(selected.body)
    imgui.pop_style_color(1)


def _draw_function_tag_editor(app: App, fn: LibFunction) -> None:
    # Row 1: current tags as removable pills (`#tag x` — click `x` to remove).
    current_tags = sorted(app.lib_tags.tags_for(fn.name))
    for tag in current_tags:
        imgui.push_style_color(imgui.Col_.button, fade(COLOR.ACCENT_PRIMARY, 0.4))
        imgui.push_style_color(imgui.Col_.text, COLOR.BG_APP)
        if imgui.small_button(f"#{tag} x##rmtag_{fn.name}_{tag}"):
            app.lib_tags.remove(fn.name, tag)
        imgui.pop_style_color(2)
        imgui.same_line()

    # Row 1 (continued): add-tag input + "Add" button.
    imgui.set_next_item_width(140.0)
    changed, app.lib_picker_new_tag_buf = imgui.input_text(
        f"##newtag_{fn.name}",
        app.lib_picker_new_tag_buf,
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    imgui.same_line()
    add_pressed = ghost_button(f"+ Add##addtag_{fn.name}")
    buf = app.lib_picker_new_tag_buf.strip().lstrip("#").lower()
    if (changed or add_pressed) and buf:
        app.lib_tags.add(fn.name, buf)
        app.lib_picker_new_tag_buf = ""
        return

    # Row 2: autocomplete suggestions — existing tags whose names contain the
    # current buffer (prefix-first ranking). Click a suggestion to adopt it.
    if buf:
        suggestions = _autocomplete_tags(app, buf, exclude=set(current_tags))
        if suggestions:
            imgui.push_font(app.font_12, app.font_12.legacy_size)
            imgui.text_colored(COLOR.FG_DIM, "Existing:")
            for s in suggestions[:8]:
                imgui.same_line()
                imgui.push_style_color(imgui.Col_.button, fade(COLOR.FG_DIM, 0.2))
                imgui.push_style_color(imgui.Col_.text, COLOR.FG_PRIMARY)
                if imgui.small_button(f"#{s}##sugg_{fn.name}_{s}"):
                    app.lib_tags.add(fn.name, s)
                    app.lib_picker_new_tag_buf = ""
                imgui.pop_style_color(2)
            imgui.pop_font()


def _autocomplete_tags(app: App, buf: str, exclude: set[str]) -> list[str]:
    # Existing tags ranked by: prefix-match first, then substring, alpha tie-break.
    all_tags = app.lib_tags.all_tags() - exclude

    def rank(tag: str) -> tuple[int, str]:
        if tag.startswith(buf):
            return (0, tag)
        if buf in tag:
            return (1, tag)
        return (2, tag)

    candidates = [(rank(t), t) for t in all_tags if buf in t]
    candidates.sort(key=lambda x: x[0])
    return [t for (r, _), t in candidates if r < 2]


def _handle_arrow_nav(app: App, n_matches: int) -> None:
    # The input field stays keyboard-focused for the whole modal lifetime, so
    # arrow keys can wrap freely through the result list — there's no "input
    # is the selection" sentinel to leave or return to. Down past last wraps
    # to the first; up before first wraps to the last.
    if n_matches == 0:
        app.lib_picker_selected_idx = 0
        return
    down = imgui.is_key_pressed(imgui.Key.down_arrow, repeat=True)
    up = imgui.is_key_pressed(imgui.Key.up_arrow, repeat=True)
    if not (down or up):
        # Clamp the existing selection into range (matches may have shrunk).
        if app.lib_picker_selected_idx < 0 or app.lib_picker_selected_idx >= n_matches:
            app.lib_picker_selected_idx = 0
        return
    idx = max(0, app.lib_picker_selected_idx)
    step = 1 if down else -1
    app.lib_picker_selected_idx = (idx + step) % n_matches


def _selected_function(app: App, matches: list[LibFunction]) -> LibFunction | None:
    if not matches:
        return None
    idx = app.lib_picker_selected_idx
    if 0 <= idx < len(matches):
        return matches[idx]
    return matches[0]


def _matching_functions(app: App) -> list[LibFunction]:
    # Filter pipeline:
    #   1. strict on SB_ prefix (anything else is a private helper)
    #   2. favs-only: only starred functions
    #   3. tag-bar filter: function passes if it has at least one ENABLED tag,
    #      OR has no tags at all.
    #   4. `#prefix` tokens IN THE SEARCH BAR: every `#prefix` must match (by
    #      prefix) at least one tag of the function. AND across `#`-tokens.
    #      This fires live as the user types, so `#nois` already filters down.
    #   5. Remaining query tokens: fuzzy across name / doc / tags.
    raw_query: str = app.lib_picker_query.strip().lower()
    tag_prefixes, text_query = _parse_query_tags(raw_query)
    disabled = app.lib_picker_disabled_tags

    def passes_tag_bar(fn: LibFunction) -> bool:
        fn_tags = app.lib_tags.tags_for(fn.name)
        if not fn_tags:
            return True  # untagged functions are always visible
        return bool(fn_tags - disabled)

    def passes_tag_prefixes(fn: LibFunction) -> bool:
        if not tag_prefixes:
            return True
        fn_tags = app.lib_tags.tags_for(fn.name)
        # Each prefix must match at least one of this function's tags (by prefix).
        return all(any(t.startswith(p) for t in fn_tags) for p in tag_prefixes)

    candidates = [
        fn for fn in app.lib_index.functions.values() if fn.name.startswith("SB_")
    ]
    if app.lib_picker_favs_only:
        candidates = [fn for fn in candidates if app.lib_favorites.is_favorite(fn.name)]
    candidates = [
        fn for fn in candidates if passes_tag_bar(fn) and passes_tag_prefixes(fn)
    ]

    if not text_query:
        return sorted(candidates, key=lambda f: f.name)

    def score(fn: LibFunction) -> tuple[int, str]:
        # Lower score is better. Exact-prefix beats substring beats doc/tag-only.
        name_lower = fn.name.lower()
        if name_lower.startswith(text_query):
            return (0, name_lower)
        if text_query in name_lower:
            return (1, name_lower)
        if fn.doc and text_query in fn.doc.lower():
            return (2, name_lower)
        if any(text_query in t for t in app.lib_tags.tags_for(fn.name)):
            return (2, name_lower)
        return (3, name_lower)

    scored = [(score(fn), fn) for fn in candidates]
    scored.sort(key=lambda s: s[0])
    return [fn for (rank, _), fn in scored if rank < 3]


def _parse_query_tags(query: str) -> tuple[list[str], str]:
    # Split the query into (list of `#`-prefixed tag PREFIXES, remaining free-text).
    # `#noi hash` -> (["noi"], "hash"); `hash #col #pal` -> (["col", "pal"], "hash").
    # A bare `#` (no character after) is ignored — the user is just opening a
    # token; once they type a letter the filter fires live.
    prefixes: list[str] = []
    parts: list[str] = []
    for tok in query.split():
        if tok.startswith("#") and len(tok) > 1:
            prefixes.append(tok[1:])
        else:
            parts.append(tok)
    return prefixes, " ".join(parts).strip()


def _resolve_tag_prefix_matches(app: App, prefixes: list[str]) -> list[str]:
    # For each `#prefix` token in the query, return the union of tags that
    # actually match — used by the UI to show "you're filtering by [#x, #y]".
    if not prefixes:
        return []
    all_tags = app.lib_tags.all_tags()
    matched: set[str] = set()
    for p in prefixes:
        matched.update(t for t in all_tags if t.startswith(p))
    return sorted(matched)


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
