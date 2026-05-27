import contextlib
from collections.abc import Iterable, Sequence
from pathlib import Path

import moderngl
import numpy as np
from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd
from OpenGL.GL import GL_FLOAT, GL_UNSIGNED_INT

from shaderbox.app import App
from shaderbox.constants import (
    IMAGE_EXTENSIONS,
    MEDIA_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from shaderbox.core import UniformValue
from shaderbox.media import Image, MediaWithTexture, Video
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_models import UIUniform
from shaderbox.ui_primitives import button, caption_text, chip_button
from shaderbox.util import (
    get_resolution_str,
    pfd_block,
    str_to_unicode,
    try_to_release,
    unicode_to_str,
)
from shaderbox.widgets.media_ops import draw_video_filters

_NAME_X = SIZE.CHIP_W + SPACE.MD
_CTRL_X = _NAME_X + SIZE.UNIFORM_NAME_W + SPACE.MD


def _begin_ctrl(name: str) -> None:
    """Lay out a uniform row: chip (already drawn) -> dim name column -> control.

    Call after the chip; positions the cursor at the control column and sets
    the next item's width. The control's own imgui label must be hidden (##).
    """
    imgui.same_line(_NAME_X)
    imgui.align_text_to_frame_padding()
    imgui.text_colored(COLOR.FG_DIM, name)
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


def draw_ui_uniform(app: App, ui_uniform: UIUniform) -> None:
    if not (ui_node := app.ui_nodes.get(app.current_node_id)):
        return

    current_value: UniformValue = ui_node.node.uniform_values[ui_uniform.name]
    new_value = None
    name = ui_uniform.name
    hidden = f"##{name}"

    draw_input_type_selector(ui_uniform)
    _begin_ctrl(name)

    if ui_uniform.input_type == "auto":
        if isinstance(current_value, float | int):
            caption_text(f"{current_value:.3f}")
        elif isinstance(current_value, Iterable):
            value_str = ", ".join(f"{v:.3f}" for v in current_value)
            caption_text(f"[{value_str}]")
        else:
            caption_text(str(current_value))

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
            imgui.same_line()
            caption_text(f"{len(current_value)}/{cap}")
            if is_changed:
                with contextlib.suppress(Exception):
                    parsed = [py_type(x.strip()) for x in value_str.split(",")]
                    new_value = parsed[:cap]
        else:
            value_str = ", ".join(f"{v:.3f}" for v in current_value)
            caption_text(f"[{value_str}]  ({cap})")

    elif ui_uniform.input_type == "text":
        assert isinstance(current_value, Sequence)
        cap = ui_uniform.array_length
        text = unicode_to_str([int(c) for c in current_value])
        is_changed, text = imgui.input_text_multiline(
            hidden, text, size=(SIZE.UNIFORM_CTRL_W, SIZE.UNIFORM_TEXT_H)
        )
        text = text[:cap]

        imgui.same_line()
        caption_text(f"{len(text)}/{cap}")

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

            media_cls: type[Image] | type[Video] | None = None
            if file_path.suffix in IMAGE_EXTENSIONS:
                media_cls = Image
            elif file_path.suffix in VIDEO_EXTENSIONS:
                media_cls = Video

            if media_cls:
                new_value = media_cls(file_path)

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

    if new_value is not None:
        try_to_release(current_value)
        ui_node.node.uniform_values[ui_uniform.name] = new_value
