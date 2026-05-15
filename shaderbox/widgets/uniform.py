import contextlib
from collections.abc import Iterable, Sequence
from pathlib import Path

import crossfiledialog
import imgui
import moderngl
import numpy as np
from OpenGL.GL import GL_FLOAT, GL_UNSIGNED_INT

from shaderbox.app import App
from shaderbox.constants import (
    IMAGE_EXTENSIONS,
    MEDIA_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from shaderbox.core import UniformValue
from shaderbox.media import Image, MediaWithTexture, Video
from shaderbox.ui_models import UIUniform
from shaderbox.ui_utils import (
    get_resolution_str,
    get_uniform_hash,
    str_to_unicode,
    try_to_release,
    unicode_to_str,
)
from shaderbox.widgets.media_ops import draw_media_models, draw_video_filters


def draw_selected_ui_uniform_settings(app: App) -> None:
    ui_node = app.ui_nodes.get(app.current_node_id)

    if (
        not ui_node
        or not ui_node.node.program
        or not ui_node.ui_state.selected_uniform_name
        or ui_node.ui_state.selected_uniform_name not in ui_node.node.program
    ):
        return

    uniform = ui_node.node.program[ui_node.ui_state.selected_uniform_name]
    if not isinstance(uniform, moderngl.Uniform | moderngl.UniformBlock):
        return

    ui_uniform = ui_node.ui_state.ui_uniforms[get_uniform_hash(uniform)]
    imgui.text(f"{ui_uniform.name} - {ui_uniform.input_type}")
    imgui.separator()
    imgui.spacing()

    current_value = ui_node.node.uniform_values.get(uniform.name)

    if ui_uniform.input_type == "texture":
        assert isinstance(current_value, MediaWithTexture)

        imgui.text(get_resolution_str(ui_uniform.name, *current_value.texture.size))

        max_image_width = imgui.get_content_region_available()[0]
        max_image_height = 0.5 * imgui.get_content_region_available()[1]
        image_aspect = np.divide(*current_value.texture.size)
        image_width = min(max_image_width, max_image_height * image_aspect)
        image_height = min(max_image_height, max_image_width / image_aspect)

        imgui.image(
            current_value.texture.glo,
            width=image_width,
            height=image_height,
            uv0=(0, 1),
            uv1=(1, 0),
        )

        imgui.new_line()
        imgui.separator()
        imgui.spacing()

        new_value = draw_media_models(app, current_value)  # type: ignore
        if isinstance(new_value, Video):
            new_value = draw_video_filters(app, new_value)

        if current_value != new_value:
            try_to_release(current_value)
            ui_node.node.uniform_values[ui_uniform.name] = new_value

    elif ui_uniform.input_type in ("drag", "color"):
        current_idx = 0 if ui_uniform.input_type == "drag" else 1
        new_idx = imgui.combo(
            "Input type##ui_uniform", current_idx, items=["drag", "color"]
        )[1]

        ui_uniform.input_type = "drag" if new_idx == 0 else "color"

    elif (
        ui_uniform.input_type in ("array", "text")
        and ui_uniform.gl_type == GL_UNSIGNED_INT
    ):
        current_idx = 0 if ui_uniform.input_type == "array" else 1
        new_idx = imgui.combo(
            "Input type##ui_uniform", current_idx, items=["array", "text"]
        )[1]

        ui_uniform.input_type = "array" if new_idx == 0 else "text"

    elif ui_uniform.input_type == "text":
        text = unicode_to_str(current_value)  # type: ignore
        imgui.text_colored(text, *(0.5, 0.5, 0.5))
        imgui.text(f"Length: {len(text)}")

    elif ui_uniform.input_type == "buffer":
        imgui.text(f"Size: {uniform.size} B")  # type: ignore


def draw_ui_uniform(app: App, ui_uniform: UIUniform) -> None:
    if not (ui_node := app.ui_nodes.get(app.current_node_id)):
        return

    current_value: UniformValue = ui_node.node.uniform_values[ui_uniform.name]
    new_value = None

    if ui_uniform.input_type == "auto":
        if ui_uniform.dimension == 1:
            imgui.text(f"{ui_uniform.name}: {current_value:.3f}")
        else:
            if isinstance(current_value, Iterable):
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
        imgui.text(ui_uniform.name)

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
            imgui.text(
                f"{ui_uniform.name}[{ui_uniform.array_length}]: [{value_str}]"
            )

    elif ui_uniform.input_type == "text":
        text = unicode_to_str(current_value)  # type: ignore
        is_changed, text = imgui.input_text(ui_uniform.name, text)

        if is_changed:
            new_value = str_to_unicode(text, ui_uniform.array_length)

    elif ui_uniform.input_type == "texture":
        assert isinstance(current_value, MediaWithTexture)

        n_styles = 0
        if ui_node.ui_state.selected_uniform_name == ui_uniform.name:
            color = (0.0, 1.0, 0.0, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON, *color)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *color)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *color)
            n_styles += 3

        button_texture_id = 0
        image_height = 90
        image_width = 90

        if current_value is not None:
            image_width = int(
                image_height
                * current_value.texture.width
                / max(current_value.texture.height, 1)
            )

            imgui.text(ui_uniform.name)

            button_texture_id = current_value.texture.glo

        if imgui.image_button(
            button_texture_id,
            width=image_width,
            height=image_height,
            uv0=(0, 1),
            uv1=(1, 0),
        ):
            ui_node.ui_state.selected_uniform_name = ui_uniform.name

        imgui.pop_style_color(n_styles)

        imgui.same_line()
        if imgui.button(f"Load##{ui_uniform.name}"):
            filter = ["*" + ext for ext in MEDIA_EXTENSIONS]
            file_path = Path(
                crossfiledialog.open_file(
                    title="Select image or video", filter=filter
                )
                or ""
            )

            media_cls = None
            if file_path.suffix in IMAGE_EXTENSIONS:
                media_cls = Image
            elif file_path.suffix in VIDEO_EXTENSIONS:
                media_cls = Video  # type: ignore

            if media_cls:
                new_value = media_cls(file_path)  # type: ignore
                ui_node.ui_state.selected_uniform_name = ui_uniform.name

    elif ui_uniform.input_type == "color":
        assert isinstance(current_value, Sequence)

        imgui.text(ui_uniform.name)
        fn = getattr(imgui, f"color_edit{ui_uniform.dimension}")
        new_value = fn(f"##{ui_uniform.name}", *current_value)[1]

    elif ui_uniform.input_type == "drag":
        imgui.text(ui_uniform.name)

        change_speed = 0.01
        if ui_uniform.dimension == 1:
            assert isinstance(current_value, float)
            new_value = imgui.drag_float(
                f"##{ui_uniform.name}", current_value, change_speed
            )[1]
        else:
            assert isinstance(current_value, Sequence)
            fn = getattr(imgui, f"drag_float{ui_uniform.dimension}")
            new_value = fn(f"##{ui_uniform.name}", *current_value, change_speed)[1]

    if new_value is not None:
        try_to_release(current_value)
        ui_node.node.uniform_values[ui_uniform.name] = new_value

    if ui_uniform.input_type != "auto" and (
        imgui.is_item_clicked() or imgui.is_item_active()
    ):
        ui_node.ui_state.selected_uniform_name = ui_uniform.name
