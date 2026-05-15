from pathlib import Path
from typing import TypeVar

import imgui
from loguru import logger

from shaderbox import modelbox
from shaderbox.app import App
from shaderbox.media import Image, Video

T = TypeVar("T", Image, Video)


def draw_media_models(app: App, input_media: T) -> T:
    output_media: Image | Video | None = None
    media_model_names = app.modelbox_info.get("media_model_names")

    if not media_model_names:
        imgui.text_colored("Media models are not available", *(1.0, 1.0, 0.0))
        if imgui.button("Refresh##media"):
            app.fetch_modelbox_info()
    else:
        imgui.text("Media model")

        app.app_state.media_model_idx = min(
            app.app_state.media_model_idx,
            len(media_model_names) - 1,
        )

        app.app_state.media_model_idx = imgui.combo(
            "##media_model_idx",
            app.app_state.media_model_idx,
            media_model_names,
        )[1]

        model_name = media_model_names[app.app_state.media_model_idx]

        imgui.same_line()
        if imgui.button("Apply##media_model"):
            try:
                output_media = modelbox.infer_media_model(
                    modelbox_url=app.app_state.modelbox_url,
                    media=input_media,
                    model_name=model_name,
                    output_dir=app.media_dir,
                )
            except Exception as e:
                err = "Failed to infer ModelBox media model"
                logger.error(f"{err}: {e}")
                app.notifications.push(err, (1, 0, 0))

    return output_media or input_media  # type: ignore


def draw_video_filters(app: App, input_video: Video) -> Video:
    output_video: Video | None = None

    imgui.text("Smoothing")

    node_ui_state = app.current_node_ui_state_or_default
    node_ui_state.video_to_video_smoothing_window = imgui.drag_int(
        "Window",
        node_ui_state.video_to_video_smoothing_window,
        min_value=3,
    )[1]

    node_ui_state.video_to_video_smoothing_sigma = imgui.drag_float(
        "Sigma",
        node_ui_state.video_to_video_smoothing_sigma,
        min_value=0.01,
        change_speed=0.01,
    )[1]

    if imgui.button("Apply##video_to_video_smoothing"):
        input_file_path = Path(input_video.details.file_details.path)
        name = input_file_path.stem

        w = node_ui_state.video_to_video_smoothing_window
        s = node_ui_state.video_to_video_smoothing_sigma
        name = f"{name}_w:{w}_s:{s}"
        output_file_path = (app.trash_dir / name).with_suffix(input_file_path.suffix)

        input_video.apply_temporal_smoothing(
            output_file_path=output_file_path,
            window_size=w,
            sigma=s,
        )
        output_video = Video(output_file_path)

    return output_video or input_video
