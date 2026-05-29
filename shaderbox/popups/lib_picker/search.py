"""Top bar of the shader-library picker: search input, favs/reset pills, tag bar.

`#prefix` tokens in the search query narrow the tag bar; free text matches
name/doc/tags (the match itself lives in `filtering`).
"""

from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.theme import COLOR
from shaderbox.ui_primitives import pill_button


def parse_query_tags(query: str) -> tuple[list[str], str]:
    prefixes: list[str] = []
    parts: list[str] = []
    for tok in query.split():
        if tok.startswith("#") and len(tok) > 1:
            prefixes.append(tok[1:])
        else:
            parts.append(tok)
    return prefixes, " ".join(parts).strip()


def resolve_tag_prefix_matches(app: App, prefixes: list[str]) -> list[str]:
    if not prefixes:
        return []
    all_tags = app.shader_lib_tags.all_tags()
    matched: set[str] = set()
    for p in prefixes:
        matched.update(t for t in all_tags if t.startswith(p))
    return sorted(matched)


def draw_favs_and_reset_row(app: App) -> None:
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


def draw_search_row(app: App, visible_count: int, total: int) -> None:
    if app.shader_lib_picker_just_opened:
        imgui.set_keyboard_focus_here(0)
    imgui.set_next_item_width(imgui.get_content_region_avail().x - 200.0)
    _, app.shader_lib_picker_query = imgui.input_text(
        "##q", app.shader_lib_picker_query
    )
    imgui.same_line()
    imgui.text_colored(COLOR.FG_DIM, f"{visible_count} / {total}")
    # Live tag-prefix feedback (`#noi` → "matching: #noise").
    tag_prefixes, _ = parse_query_tags(app.shader_lib_picker_query.strip().lower())
    if tag_prefixes:
        matched = resolve_tag_prefix_matches(app, tag_prefixes)
        imgui.same_line()
        if matched:
            imgui.text_colored(
                COLOR.FG_DIM, "matching: " + " ".join(f"#{t}" for t in matched)
            )
        else:
            imgui.text_colored(COLOR.STATE_WARN, "no tags match")


def draw_tag_bar(app: App) -> None:
    # Blue pills (active = blue fill, disabled = dim). Wraps onto multiple rows.
    # Tag bar shows every tag in the lib; #prefix in the search bar narrows it.
    tag_prefixes, _ = parse_query_tags(app.shader_lib_picker_query.strip().lower())
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
