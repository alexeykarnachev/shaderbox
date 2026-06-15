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
from shaderbox.media import MediaWithTexture, Video, media_class_for
from shaderbox.shader_errors import find_uniform_declaration_line
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_models import UIUniform
from shaderbox.ui_primitives import (
    button,
    caption_text,
    chip_button,
    clickable_label,
    clipped_caption,
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
    hover bridge on hover. `text_color`/`accent` override the row's colour.
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
    # The active editor first (it carries unsaved edits); then every file in the node's
    # compile unit, so a uniform declared in a resolved lib file is jump-reachable.
    session = app.get_current_session()
    if session is not None:
        line = find_uniform_declaration_line(session.editor.get_text(), name)
        if line is not None:
            return session.source.path, line
    if ui_node := app.ui_nodes.get(app.current_node_id):
        active_path = session.source.path if session is not None else None
        for source in ui_node.node.compile_unit.sources:
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
    out of the trailing column the play/stop button now owns). `playing` colours the
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


def _draw_play_stop(app: App, name: str, *, driven: bool, playing: bool) -> None:
    # The trailing per-row play/stop affordance (048): drawn ONLY for a uniform the script TARGETS
    # (driven — playing OR stopped); a never-scripted MANUAL uniform shows nothing. `stop` (accent)
    # when playing, `play` (dim) when stopped — the toggle flips the node-scoped stopped state.
    # Disabled while the whole node is stopped: a per-uniform play is meaningless then (nothing
    # writes), and a full stop->play resets every uniform to playing anyway.
    if not driven:
        return
    node_id = app.current_node_id
    node_stopped = app.current_node_ui_state_or_default.all_stopped
    imgui.same_line()
    imgui.begin_disabled(app.copilot_turn_active or node_stopped)
    tooltip = (
        "Can't play a single uniform while the whole script is stopped"
        if node_stopped
        else "Stop the script driving this uniform (edit it by hand)"
        if playing
        else "Resume the script driving this uniform"
    )
    if play_stop_toggle(f"u_{name}", playing, tooltip=tooltip):
        app.set_uniform_stopped(node_id, name, playing)
    imgui.end_disabled()


def draw_ui_uniform(app: App, ui_uniform: UIUniform) -> None:
    if not (ui_node := app.ui_nodes.get(app.current_node_id)):
        return

    current_value: UniformValue = ui_node.node.uniform_values[ui_uniform.name]
    new_value = None
    name = ui_uniform.name
    hidden = f"##{name}"

    # Play/stop state (048): a uniform the script TARGETS is `driven` (playing OR stopped); PLAYING =
    # driven and not stopped (the engine writes it each tick). The value widget stays EDITABLE while
    # playing — grabbing it AUTO-STOPS (below), so the manual edit sticks instead of snapping back.
    node_id = app.current_node_id
    driven = app.session.uniform_is_driven(node_id, name)
    playing = driven and not app.session.is_uniform_stopped(node_id, name)

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
        assert isinstance(current_value, MediaWithTexture)

        image_height = SIZE.THUMB_SM
        image_width = int(
            image_height
            * current_value.texture.width
            / max(current_value.texture.height, 1)
        )

        if button("Load" + hidden):
            patterns = " ".join("*" + ext for ext in MEDIA_EXTENSIONS)
            results = pfd_block(
                pfd.open_file(
                    "Select image or video",
                    default_path=".",
                    filters=["Media", patterns],
                )
            )
            file_path = Path(results[0]) if results else Path()

            if file_path.suffix in MEDIA_EXTENSIONS:
                new_value = media_class_for(file_path.suffix)(file_path)

        imgui.same_line()
        caption_text(get_resolution_str(None, *current_value.texture.size))

        imgui.set_cursor_pos_x(_CTRL_X)
        imgui.image(
            imgui.ImTextureRef(current_value.texture.glo),
            image_size=(image_width, image_height),
            uv0=(0, 1),
            uv1=(1, 0),
        )

        if isinstance(current_value, Video):
            imgui.same_line(spacing=float(SPACE.LG))
            video_value = draw_video_filters(app, current_value)
            if video_value is not current_value:
                new_value = video_value

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
        app.set_uniform_stopped(node_id, name, True)
        playing = False

    _draw_play_stop(app, name, driven=driven, playing=playing)

    # A PLAYING uniform's value is owned by the script's tick — but a manual edit auto-stopped it
    # above (playing is now False), so this write applies + sticks. A still-playing slot is never
    # written back here (the tick wins); a stopped/manual slot's edit always applies.
    if new_value is not None and not playing:
        try_to_release(current_value)
        ui_node.node.uniform_values[ui_uniform.name] = new_value
