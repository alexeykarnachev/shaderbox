from pathlib import Path

from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.media import Video


def draw_video_filters(app: App, input_video: Video) -> Video:
    output_video: Video | None = None

    imgui.text("Smoothing")

    node_ui_state = app.current_node_ui_state_or_default
    node_ui_state.video_to_video_smoothing_window = imgui.drag_int(
        "Window",
        node_ui_state.video_to_video_smoothing_window,
        v_min=3,
    )[1]

    node_ui_state.video_to_video_smoothing_sigma = imgui.drag_float(
        "Sigma",
        node_ui_state.video_to_video_smoothing_sigma,
        v_min=0.01,
        v_speed=0.01,
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
