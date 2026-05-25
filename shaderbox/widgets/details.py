from collections.abc import Sequence
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd
from loguru import logger

from shaderbox.app import App
from shaderbox.constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from shaderbox.media import FileDetails, MediaDetails, ResolutionDetails
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_primitives import (
    button,
    caption_text,
    draw_copyable_text,
    ghost_button,
    label_row,
    row_label,
)
from shaderbox.util import adjust_size, pfd_block


def draw_file_details(
    app: App,
    details: FileDetails,
    extensions: Sequence[str] | None = None,
) -> FileDetails:
    details = details.model_copy()

    if button("Choose file..."):
        file_path = pfd_block(pfd.save_file("File path", default_path="."))
        if file_path:
            extension = Path(file_path).suffix
            if extensions and extension not in extensions:
                details.path = ""
                err = f"Can't select {extension} file, available extensions are: {extensions}"
                logger.warning(err)
                app.notifications.push(err, COLOR.STATE_ERROR[:3])
            else:
                details.path = file_path

    if details.path:
        draw_copyable_text(details.path)
        caption_text(f"{details.size // 1024} KB")
    else:
        text = "No output file selected"
        if extensions:
            text += f" ({', '.join(extensions)})"
        caption_text(text)

    return details


def draw_resolution_details(
    app: App,
    details: ResolutionDetails,
    aspect: float | None = None,
) -> ResolutionDetails:
    details = details.model_copy()

    if not (ui_node := app.ui_nodes.get(app.current_node_id)):
        return details

    label_row(app.font_12, "Width", SIZE.RENDER_CTRL_W)
    is_width_changed, new_width = imgui.drag_int(
        "##width", details.width, v_min=16, v_max=2560
    )
    label_row(app.font_12, "Height", SIZE.RENDER_CTRL_W)
    is_height_changed, new_height = imgui.drag_int(
        "##height", details.height, v_min=16, v_max=2560
    )

    if aspect is not None:
        if is_height_changed:
            new_width = int(new_height * aspect)
        elif is_width_changed:
            new_height = int(new_width / aspect)

    details.width = new_width
    details.height = new_height

    full_w, full_h = ui_node.node.canvas.texture.size
    half_w, half_h = adjust_size(ui_node.node.canvas.texture.size, max_size=512)

    row_label(app.font_12, "Presets")
    if ghost_button(f"{full_w}x{full_h}") or not details.width or not details.height:
        details.width, details.height = full_w, full_h
    imgui.same_line()
    if ghost_button(f"{half_w}x{half_h}"):
        details.width, details.height = half_w, half_h

    return details


def draw_media_details(
    app: App,
    details: MediaDetails,
) -> MediaDetails:
    details = details.model_copy()
    aspect = None

    if ui_node := app.ui_nodes.get(app.current_node_id):
        aspect = np.divide(*ui_node.node.canvas.texture.size)

    output_type_name = "video" if details.is_video else "image"
    options = ["video", "image"]
    label_row(app.font_12, "Output", SIZE.RENDER_CTRL_W)
    idx = imgui.combo(
        "##render_output_type",
        options.index(output_type_name),
        items=options,
    )[1]
    details.is_video = idx == 0

    if details.is_video:
        label_row(app.font_12, "Quality", SIZE.RENDER_CTRL_W)
        details.quality = imgui.combo(
            "##video_quality",
            details.quality,
            items=["low", "medium-low", "medium-high", "high"],
        )[1]
        label_row(app.font_12, "FPS", SIZE.RENDER_CTRL_W)
        details.fps = imgui.drag_int("##video_fps", details.fps, 1, 10, 60)[1]
        label_row(app.font_12, "Duration", SIZE.RENDER_CTRL_W)
        details.duration = imgui.drag_float(
            "##video_duration", details.duration, 0.1, 1.0, 60.0, "%.1f s"
        )[1]

    imgui.dummy((0, SPACE.SM))
    details.resolution_details = draw_resolution_details(
        app,
        details.resolution_details,
        aspect=aspect,
    )
    imgui.dummy((0, SPACE.SM))
    details.file_details = draw_file_details(
        app,
        details.file_details,
        extensions=VIDEO_EXTENSIONS if details.is_video else IMAGE_EXTENSIONS,
    )

    return details
