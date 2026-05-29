"""Right-hand preview pane: path, signature, doc, tag editor, body."""

from pathlib import Path

from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.shader_lib import ShaderLibFunction
from shaderbox.theme import COLOR, fade
from shaderbox.ui_primitives import draw_copyable_text, ghost_button, pill_button


def draw_preview(app: App, selected: ShaderLibFunction | None, root: Path) -> None:
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
