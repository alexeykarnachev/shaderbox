import contextlib
from collections.abc import Sequence
from pathlib import Path

import moderngl
import numpy as np
from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd
from OpenGL.GL import GL_FLOAT, GL_UNSIGNED_INT

from shaderbox.app import App
from shaderbox.constants import MEDIA_EXTENSIONS
from shaderbox.core import UniformValue
from shaderbox.editor_types import HoverMark, JumpRequest
from shaderbox.intel.symbols import SymbolKind
from shaderbox.media import MediaWithTexture, Video, media_class_for
from shaderbox.pass_graph import AutoSource, NoSource, PassSource, wired_pass
from shaderbox.paths import pass_name_of
from shaderbox.shader_errors import find_uniform_declaration_line
from shaderbox.theme import COLOR, SIZE, SPACE, kind_color
from shaderbox.ui_models import UIUniform
from shaderbox.ui_primitives import (
    ComboRow,
    button,
    caption_text,
    chip_button,
    clickable_label,
    clipped_caption,
    grouped_combo,
    play_stop_toggle,
)
from shaderbox.util import (
    format_auto_value,
    get_resolution_str,
    pfd_block,
    str_to_unicode,
    try_to_release,
    unicode_to_str,
)
from shaderbox.widgets.media_ops import draw_video_filters

_NAME_X = SIZE.CHIP_W + SPACE.MD
_CTRL_X = _NAME_X + SIZE.UNIFORM_NAME_W + SPACE.MD


def uniform_name_label(
    app: App,
    name: str,
    width: float,
    *,
    text_color: tuple[float, float, float, float] | None = None,
    accent: tuple[float, float, float, float] | None = None,
) -> None:
    """Clickable uniform-name cell — jump-to-declaration on click, code<->panel
    hover bridge on hover. `text_color`/`accent` override the row's color.
    """
    clicked = clickable_label(
        name,
        width,
        id_=f"uname_{name}",
        tooltip="Jump to declaration",
        highlight=(name == app.code_hovered_uniform),
        text_color=text_color,
        accent=accent,
    )
    if not (clicked or imgui.is_item_hovered()):
        return
    located = _locate_uniform_declaration(app, name)
    if located is None:
        return
    path, line = located
    if clicked:
        if path != app.current_editor_path:
            app.open_shader_lib_file(path)
        app.editor_jump_request = JumpRequest(path, line, 0)
    elif path == app.current_editor_path:
        # Hover only marks the active editor; a lib-declared uniform shows nothing to highlight.
        app.editor_hover_line = HoverMark(path, line)


def _locate_uniform_declaration(app: App, name: str) -> tuple[Path, int] | None:
    # The active editor first (it carries unsaved edits); then every file in the document's
    # compile unit, so a uniform declared in a resolved lib file is jump-reachable.
    session = app.get_current_session()
    if session is not None:
        line = find_uniform_declaration_line(session.editor.get_text(), name)
        if line is not None:
            return session.source.path, line
    if app.current_document_id in app.ui_documents:
        active_path = session.source.path if session is not None else None
        for source in app.panel_pass(app.current_document_id).compile_unit.sources:
            if source.path == active_path:
                continue
            line = find_uniform_declaration_line(source.text, name)
            if line is not None:
                return source.path, line
    return None


def _begin_ctrl(
    app: App, name: str, count_suffix: str = "", *, playing: bool = False
) -> None:
    """Lay out a uniform row: chip (already drawn) -> clickable name -> control.

    Call after the chip; positions the cursor at the control column and sets
    the next item's width. The control's own imgui label must be hidden (##).
    `count_suffix` (text/array `len/cap`) renders dim in the name column (045 B6 —
    out of the trailing column the play/stop button now owns). `playing` colors the
    name `STATE_INFO` blue — the at-a-glance "the script drives this" cue (048).
    """
    imgui.same_line(_NAME_X)
    name_color = COLOR.STATE_INFO if playing else None
    uniform_name_label(
        app, name, SIZE.UNIFORM_NAME_W, text_color=name_color, accent=name_color
    )
    if count_suffix:
        # Right-anchor the caption against the control column (047 F13), so it never overlaps the
        # input: a long name used to push a flowed caption past _CTRL_X (same_line to a smaller
        # offset is a no-op), so the control drew over it. Placing it caption-width left of _CTRL_X
        # keeps it dim in the name column's tail and clear of the control.
        caption_w = imgui.calc_text_size(count_suffix).x
        imgui.same_line(_CTRL_X - caption_w - float(SPACE.SM))
        caption_text(count_suffix)
    imgui.same_line(_CTRL_X)
    imgui.set_next_item_width(SIZE.UNIFORM_CTRL_W)


def draw_input_type_selector(ui_uniform: UIUniform) -> None:
    """The single seam for input-shape selection — swap cycle<->dropdown here alone."""
    valid = ui_uniform.valid_input_types()
    locked = len(valid) == 1

    label = f"{ui_uniform.input_type}##input_type_{ui_uniform.name}"
    if chip_button(label, width=SIZE.CHIP_W, disabled=locked):
        current_idx = valid.index(ui_uniform.input_type)
        ui_uniform.input_type = valid[(current_idx + 1) % len(valid)]


def _count_suffix(ui_uniform: UIUniform, current_value: UniformValue) -> str:
    # The text/array len/cap caption (045 B6): shown dim in the name column now the trailing column
    # is the script pill's. Empty for every other input type.
    cap = ui_uniform.array_length
    if ui_uniform.input_type == "text" and isinstance(current_value, Sequence):
        text = unicode_to_str([int(c) for c in current_value])
        return f"({len(text[:cap])}/{cap})"
    if ui_uniform.input_type == "array" and isinstance(current_value, Sequence):
        py_type = {GL_FLOAT: float, GL_UNSIGNED_INT: int}.get(ui_uniform.gl_type)
        if py_type is not None:
            return f"({len(current_value)}/{cap})"
        return f"({cap})"
    return ""


def _draw_play_stop(
    app: App, pass_name: str, name: str, *, driven: bool, playing: bool
) -> None:
    # The trailing per-row play/stop affordance (048): drawn ONLY for a uniform the script TARGETS
    # (driven — playing OR stopped); a never-scripted MANUAL uniform shows nothing. `stop` (accent)
    # when playing, `play` (dim) when stopped — the toggle flips the document-scoped stopped state.
    # Disabled while the whole document is stopped: a per-uniform play is meaningless then (nothing
    # writes), and a full stop->play resets every uniform to playing anyway.
    if not driven:
        return
    document_id = app.current_document_id
    document_stopped = app.current_document_ui_state_or_default.all_stopped
    imgui.same_line()
    imgui.begin_disabled(app.copilot_turn_active or document_stopped)
    tooltip = (
        "Whole script is stopped"
        if document_stopped
        else "Stop this uniform"
        if playing
        else "Resume this uniform"
    )
    if play_stop_toggle(f"u_{pass_name}_{name}", playing, tooltip=tooltip):
        app.set_uniform_stopped(document_id, pass_name, name, playing)
    imgui.end_disabled()


def _thumb_size(texture: moderngl.Texture) -> tuple[int, int]:
    height = int(SIZE.THUMB_SM)
    return int(height * texture.width / max(texture.height, 1)), height


def _draw_pass_source(texture: moderngl.Texture, source: str) -> None:
    # A live thumbnail of the producing pass, captioned with its name.
    imgui.set_cursor_pos_x(_CTRL_X)
    imgui.image(
        imgui.ImTextureRef(texture.glo),
        image_size=_thumb_size(texture),
        uv0=(0, 1),
        uv1=(1, 0),
    )
    imgui.same_line()
    caption_text(source)


def _pick_media_file() -> Path:
    # Both cases: Linux glob filters are case-sensitive, phone cameras emit .MOV/.JPG.
    patterns = " ".join(f"*{ext} *{ext.upper()}" for ext in MEDIA_EXTENSIONS)
    results = pfd_block(
        pfd.open_file(
            "Select image or video", default_path=".", filters=["Media", patterns]
        )
    )
    return Path(results[0]) if results else Path()


def _draw_black_swatch() -> None:
    side = int(SIZE.THUMB_SM)
    pos = imgui.get_cursor_screen_pos()
    imgui.get_window_draw_list().add_rect_filled(
        (pos.x, pos.y),
        (pos.x + side, pos.y + side),
        imgui.color_convert_float4_to_u32(COLOR.BLACK),
    )
    imgui.dummy((side, side))


def draw_ui_uniform(app: App, ui_uniform: UIUniform) -> None:
    if app.current_document_id not in app.ui_documents:
        return

    panel_pass = app.panel_pass(app.current_document_id)
    # The script drives a uniform ON A PASS (069 D3), so every driven/stopped question is asked
    # about the pass the panel is showing; `Pass` carries its source path, not its name.
    panel_pass_name = pass_name_of(panel_pass.source.path)
    current_value: UniformValue = panel_pass.uniform_values[ui_uniform.name]
    new_value = None
    name = ui_uniform.name
    hidden = f"##{name}"

    # Play/stop state (048): a uniform the script TARGETS is `driven` (playing OR stopped); PLAYING =
    # driven and not stopped (the engine writes it each tick). The value widget stays EDITABLE while
    # playing — grabbing it AUTO-STOPS (below), so the manual edit sticks instead of snapping back.
    document_id = app.current_document_id
    driven = app.session.uniform_is_driven(document_id, panel_pass_name, name)
    playing = driven and not app.session.is_uniform_stopped(
        document_id, panel_pass_name, name
    )

    draw_input_type_selector(ui_uniform)
    _begin_ctrl(app, name, _count_suffix(ui_uniform, current_value), playing=playing)

    if ui_uniform.input_type == "auto":
        clipped_caption(format_auto_value(current_value), SIZE.UNIFORM_CTRL_W)

    elif ui_uniform.input_type == "buffer":
        assert isinstance(current_value, moderngl.Buffer)

        if button("Randomize" + hidden):
            data = np.random.rand(current_value.size // 4).astype(np.float32)
            current_value.write(data)

        imgui.same_line()
        caption_text(f"{current_value.size} B")

    elif ui_uniform.input_type == "array":
        assert isinstance(current_value, Sequence)

        py_type = {GL_FLOAT: float, GL_UNSIGNED_INT: int}.get(ui_uniform.gl_type)

        cap = ui_uniform.array_length
        if py_type is not None:
            value_str = ", ".join(map(str, current_value))
            is_changed, value_str = imgui.input_text(hidden, value_str)
            if is_changed:
                with contextlib.suppress(Exception):
                    parsed = [py_type(x.strip()) for x in value_str.split(",")]
                    new_value = parsed[:cap]
        else:
            clipped_caption(format_auto_value(current_value), SIZE.UNIFORM_CTRL_W)

    elif ui_uniform.input_type == "text":
        assert isinstance(current_value, Sequence)
        cap = ui_uniform.array_length
        text = unicode_to_str([int(c) for c in current_value])
        is_changed, text = imgui.input_text_multiline(
            hidden, text, size=(SIZE.UNIFORM_CTRL_W, SIZE.UNIFORM_TEXT_H)
        )
        text = text[:cap]

        if is_changed:
            new_value = str_to_unicode(text, ui_uniform.array_length)

    elif ui_uniform.input_type == "texture":
        # The row is where a sampler's SOURCE is chosen (072): a pass, black, the name rule, or
        # a file. Under the combo, what it reads: the pass's live thumbnail, the bound media, or
        # the black swatch every unfilled sampler binds.
        document = app.ui_documents[document_id].document
        passes = sorted(document.passes)
        # One flat list (079 D6): `none`, the passes, `file...` — no auto row and no captions.
        # `AutoSource` stays the VALUE a fresh sampler holds; the closed control shows what it
        # resolves to, so the user reads the pass it reads and picking one writes it explicitly.
        pass_rows: list[ComboRow] = [
            (p, kind_color(SymbolKind.PASS_SAMPLER)) for p in passes
        ]
        none_row: ComboRow = ("none", COLOR.FG_DIM)
        file_row: ComboRow = ("file...", COLOR.FG_SECONDARY)
        choices: list[ComboRow] = [none_row, *pass_rows, file_row]
        file_item = len(choices) - 1
        if isinstance(current_value, PassSource):
            resolved = current_value.name
        elif isinstance(current_value, AutoSource):
            resolved = wired_pass(AutoSource(), name, panel_pass_name, passes)
        else:
            resolved = None
        if isinstance(current_value, PassSource | AutoSource):
            index = 1 + passes.index(resolved) if resolved in passes else 0
        elif isinstance(current_value, NoSource):
            index = 0
        else:
            index = file_item
        picked = grouped_combo(
            f"##source_{name}",
            choices[index],
            [("", choices)],
            SIZE.UNIFORM_CTRL_W,
        )
        changed = picked is not None and picked != index
        if changed and picked == file_item:
            file_path = _pick_media_file()
            if file_path.suffix.lower() in MEDIA_EXTENSIONS:
                new_value = media_class_for(file_path.suffix)(file_path)
        elif changed and picked is not None:
            source = NoSource() if picked == 0 else PassSource(passes[picked - 1])
            error = app.session.set_sampler_source(
                document_id, panel_pass_name, name, source
            )
            if error:
                app.notifications.push(error)
            current_value = panel_pass.uniform_values[name]

        source_pass = document.sampler_source(panel_pass_name, name)
        if source_pass is not None:
            _draw_pass_source(
                document.input_texture(panel_pass_name, source_pass), source_pass
            )
        elif isinstance(current_value, MediaWithTexture | moderngl.Texture):
            texture = (
                current_value.texture
                if isinstance(current_value, MediaWithTexture)
                else current_value
            )
            imgui.set_cursor_pos_x(_CTRL_X)
            imgui.image(
                imgui.ImTextureRef(texture.glo),
                image_size=_thumb_size(texture),
                uv0=(0, 1),
                uv1=(1, 0),
            )
            imgui.same_line()
            caption_text(get_resolution_str(None, *texture.size))
            if isinstance(current_value, Video):
                imgui.same_line(spacing=float(SPACE.LG))
                video_value = draw_video_filters(app, current_value)
                if video_value is not current_value:
                    new_value = video_value
        else:
            imgui.set_cursor_pos_x(_CTRL_X)
            _draw_black_swatch()

    elif ui_uniform.input_type == "color":
        assert isinstance(current_value, Sequence)

        fn = getattr(imgui, f"color_edit{ui_uniform.dimension}")
        new_value = fn(hidden, list(current_value))[1]

    elif ui_uniform.input_type == "drag":
        change_speed = 0.01
        if ui_uniform.dimension == 1:
            assert isinstance(current_value, float | int)
            if isinstance(current_value, int) and not isinstance(current_value, bool):
                new_value = imgui.drag_int(hidden, current_value)[1]
            else:
                new_value = imgui.drag_float(
                    hidden, current_value, v_speed=change_speed
                )[1]
        else:
            assert isinstance(current_value, Sequence)
            fn = getattr(imgui, f"drag_float{ui_uniform.dimension}")
            new_value = fn(hidden, list(current_value), change_speed)[1]

    # Auto-stop on grab (048 D6): `is_item_activated()` fires ONCE when the user grabs the value
    # widget (not per drag-frame). Gated on `playing` — only a PLAYING uniform auto-stops, which
    # defuses the per-branch trailing-item hazard (a texture is non-scriptable → never playing). The
    # manual edit then applies + sticks (the slot is no longer written by the tick).
    if playing and imgui.is_item_activated():
        app.set_uniform_stopped(document_id, panel_pass_name, name, True)
        playing = False

    _draw_play_stop(app, panel_pass_name, name, driven=driven, playing=playing)

    # A PLAYING uniform's value is owned by the script's tick — but a manual edit auto-stopped it
    # above (playing is now False), so this write applies + sticks. A still-playing slot is never
    # written back here (the tick wins); a stopped/manual slot's edit always applies.
    if new_value is not None and not playing:
        try_to_release(current_value)
        panel_pass.uniform_values[ui_uniform.name] = new_value
