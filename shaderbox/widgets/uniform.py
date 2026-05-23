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
from shaderbox.theme import COLOR, SIZE
from shaderbox.ui_models import UIUniform
from shaderbox.ui_utils import (
    get_resolution_str,
    pfd_block,
    str_to_unicode,
    try_to_release,
    unicode_to_str,
)
from shaderbox.widgets.media_ops import draw_video_filters


def draw_input_type_selector(ui_uniform: UIUniform) -> None:
    """The single seam for input-shape selection — swap cycle<->dropdown here alone."""
    valid = ui_uniform.valid_input_types()
    locked = len(valid) == 1

    imgui.push_style_var(imgui.StyleVar_.frame_rounding, SIZE.CHIP_ROUNDING)
    imgui.push_style_color(imgui.Col_.button, COLOR.CHIP_BG)
    imgui.push_style_color(imgui.Col_.button_hovered, COLOR.CHIP_BG_HOVER)
    imgui.push_style_color(imgui.Col_.button_active, COLOR.CHIP_BG_HOVER)
    imgui.push_style_color(imgui.Col_.text, COLOR.CHIP_FG)

    if locked:
        imgui.begin_disabled()

    label = f"{ui_uniform.input_type}##input_type_{ui_uniform.name}"
    if imgui.button(label, size=(SIZE.CHIP_W, 0.0)) and not locked:
        current_idx = valid.index(ui_uniform.input_type)
        ui_uniform.input_type = valid[(current_idx + 1) % len(valid)]

    if locked:
        imgui.end_disabled()

    imgui.pop_style_color(4)
    imgui.pop_style_var()
    imgui.same_line()


def draw_ui_uniform(app: App, ui_uniform: UIUniform) -> None:
    if not (ui_node := app.ui_nodes.get(app.current_node_id)):
        return

    current_value: UniformValue = ui_node.node.uniform_values[ui_uniform.name]
    new_value = None

    draw_input_type_selector(ui_uniform)

    if ui_uniform.input_type == "auto":
        if isinstance(current_value, float | int):
            imgui.text(f"{ui_uniform.name}: {current_value:.3f}")
        elif isinstance(current_value, Iterable):
            value_str = ", ".join(f"{v:.3f}" for v in current_value)
            imgui.text(f"{ui_uniform.name}: [{value_str}]")
        else:
            imgui.text(f"{ui_uniform.name}: {current_value}")

    elif ui_uniform.input_type == "buffer":
        assert isinstance(current_value, moderngl.Buffer)

        if imgui.button("Randomize"):
            data = np.random.rand(current_value.size // 4).astype(np.float32)
            current_value.write(data)

        imgui.same_line()
        imgui.text(f"{ui_uniform.name}  ({current_value.size} B)")

    elif ui_uniform.input_type == "array":
        assert isinstance(current_value, Sequence)

        py_type = {GL_FLOAT: float, GL_UNSIGNED_INT: int}.get(ui_uniform.gl_type)

        if py_type is not None:
            value_str = ", ".join(map(str, current_value))
            is_changed, value_str = imgui.input_text(ui_uniform.name, value_str)
            if is_changed:
                with contextlib.suppress(Exception):
                    new_value = [py_type(x.strip()) for x in value_str.split(",")]
        else:
            value_str = ", ".join(f"{v:.3f}" for v in current_value)
            imgui.text(f"{ui_uniform.name}[{ui_uniform.array_length}]: [{value_str}]")

    elif ui_uniform.input_type == "text":
        assert isinstance(current_value, Sequence)
        text = unicode_to_str([int(c) for c in current_value])
        is_changed, text = imgui.input_text(ui_uniform.name, text)

        imgui.same_line()
        imgui.text(f"({len(text)})")

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

        imgui.text(get_resolution_str(ui_uniform.name, *current_value.texture.size))

        imgui.image(
            imgui.ImTextureRef(current_value.texture.glo),
            image_size=(image_width, image_height),
            uv0=(0, 1),
            uv1=(1, 0),
        )

        imgui.same_line()
        if imgui.button(f"Load##{ui_uniform.name}"):
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

        if isinstance(current_value, Video):
            video_value = draw_video_filters(app, current_value)
            if video_value is not current_value:
                new_value = video_value

    elif ui_uniform.input_type == "color":
        assert isinstance(current_value, Sequence)

        fn = getattr(imgui, f"color_edit{ui_uniform.dimension}")
        new_value = fn(ui_uniform.name, list(current_value))[1]

    elif ui_uniform.input_type == "drag":
        change_speed = 0.01
        if ui_uniform.dimension == 1:
            assert isinstance(current_value, float | int)
            if isinstance(current_value, int) and not isinstance(current_value, bool):
                new_value = imgui.drag_int(ui_uniform.name, current_value)[1]
            else:
                new_value = imgui.drag_float(
                    ui_uniform.name, current_value, v_speed=change_speed
                )[1]
        else:
            assert isinstance(current_value, Sequence)
            fn = getattr(imgui, f"drag_float{ui_uniform.dimension}")
            new_value = fn(ui_uniform.name, list(current_value), change_speed)[1]

    if new_value is not None:
        try_to_release(current_value)
        ui_node.node.uniform_values[ui_uniform.name] = new_value
