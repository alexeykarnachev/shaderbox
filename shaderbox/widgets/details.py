from collections.abc import Sequence
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd
from loguru import logger

from shaderbox.app import App
from shaderbox.constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from shaderbox.media import FileDetails, MediaDetails, ResolutionDetails
from shaderbox.theme import COLOR
from shaderbox.util import adjust_size, pfd_block


def draw_file_details(
    app: App,
    details: FileDetails,
    extensions: Sequence[str] | None = None,
    is_changeable: bool = True,
) -> FileDetails:
    details = details.model_copy()

    file_path: str | None = None
    if is_changeable and imgui.button("File:##file_path"):
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
    elif not is_changeable:
        imgui.text("File:")

    imgui.same_line()
    if details.path:
        imgui.text_colored(COLOR.FG_DIM, str(details.path))
        imgui.text(f"File size: {details.size // 1024} KB")
    else:
        text = "Select file path"
        if extensions:
            text += f" ({', '.join(extensions)})"
        imgui.text_colored(COLOR.FG_DIM, text)

    return details


def draw_resolution_details(
    app: App,
    details: ResolutionDetails,
    aspect: float | None = None,
    is_changeable: bool = True,
) -> ResolutionDetails:
    details = details.model_copy()

    if not (ui_node := app.ui_nodes.get(app.current_node_id)):
        return details

    if not is_changeable:
        imgui.text(f"Resolution: {details.width}x{details.height}")
        return details

    is_width_changed, new_width = imgui.drag_int(
        "Width", details.width, v_min=16, v_max=2560
    )
    is_height_changed, new_height = imgui.drag_int(
        "Height", details.height, v_min=16, v_max=2560
    )

    if aspect is not None:
        if is_height_changed:
            new_width = int(new_height * aspect)
        elif is_width_changed:
            new_height = int(new_width / aspect)

    details.width = new_width
    details.height = new_height

    width, height = ui_node.node.canvas.texture.size
    if imgui.button(f"{width}x{height}") or not details.width or not details.height:
        details.width = width
        details.height = height

    imgui.same_line()
    width, height = adjust_size(ui_node.node.canvas.texture.size, max_size=512)
    if imgui.button(f"{width}x{height}") or not details.width or not details.height:
        details.width = width
        details.height = height

    return details


def draw_media_details(
    app: App,
    details: MediaDetails,
    is_changeable: bool = True,
) -> MediaDetails:
    details = details.model_copy()
    aspect = None

    if ui_node := app.ui_nodes.get(app.current_node_id):
        aspect = np.divide(*ui_node.node.canvas.texture.size)

    output_type_name = "video" if details.is_video else "image"
    if is_changeable:
        options = ["video", "image"]
        idx = imgui.combo(
            "Output type##render_output_type_idx",
            options.index(output_type_name),
            items=options,
        )[1]
        details.is_video = idx == 0
    else:
        imgui.text(f"Output type: {output_type_name}")

    if details.is_video and is_changeable:
        details.quality = imgui.combo(
            "Quality##video_quality_idx",
            details.quality,
            items=["low", "medium-low", "medium-high", "high"],
        )[1]
        details.fps = imgui.drag_int("FPS##video_fps", details.fps, v_min=10, v_max=60)[
            1
        ]
        details.duration = imgui.drag_float(
            "Duration, sec##video_duration",
            details.duration,
            v_speed=0.1,
            v_min=1.0,
            v_max=60.0,
        )[1]
    elif details.is_video:
        imgui.text(f"FPS: {details.fps}")
        imgui.text(f"Duration: {details.duration} sec")

    details.resolution_details = draw_resolution_details(
        app,
        details.resolution_details,
        aspect=aspect,
        is_changeable=is_changeable,
    )
    details.file_details = draw_file_details(
        app,
        details.file_details,
        extensions=VIDEO_EXTENSIONS if details.is_video else IMAGE_EXTENSIONS,
        is_changeable=is_changeable,
    )

    return details
