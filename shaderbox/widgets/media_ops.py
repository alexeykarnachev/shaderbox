from pathlib import Path

from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.media import Video
from shaderbox.theme import SIZE, SPACE
from shaderbox.ui_primitives import ghost_button, row_label, small_caption


def draw_video_filters(app: App, input_video: Video) -> Video:
    """A compact temporal-smoothing block, drawn as a small column of sliders
    meant to sit on the same line to the right of the video thumbnail."""
    output_video: Video | None = None
    node_ui_state = app.current_node_ui_state_or_default

    with imgui_ctx.begin_group():
        small_caption(app.font_12, "Smoothing")
        row_label(app.font_12, "Window", float(SIZE.SMOOTHING_LABEL_W))
        imgui.set_next_item_width(SIZE.SMOOTHING_DRAG_W)
        node_ui_state.video_to_video_smoothing_window = imgui.drag_int(
            "##smoothing_window",
            node_ui_state.video_to_video_smoothing_window,
            v_min=3,
        )[1]

        row_label(app.font_12, "Sigma", float(SIZE.SMOOTHING_LABEL_W))
        imgui.set_next_item_width(SIZE.SMOOTHING_DRAG_W)
        node_ui_state.video_to_video_smoothing_sigma = imgui.drag_float(
            "##smoothing_sigma",
            node_ui_state.video_to_video_smoothing_sigma,
            v_min=0.01,
            v_speed=0.01,
        )[1]

        imgui.dummy((0, SPACE.XS))
        if ghost_button("Apply##video_to_video_smoothing", width=SIZE.BTN_SM_W):
            input_file_path = Path(input_video.details.file_details.path)
            w = node_ui_state.video_to_video_smoothing_window
            s = node_ui_state.video_to_video_smoothing_sigma
            name = f"{input_file_path.stem}_w:{w}_s:{s}"
            output_file_path = (app.trash_dir / name).with_suffix(
                input_file_path.suffix
            )
            input_video.apply_temporal_smoothing(
                output_file_path=output_file_path,
                window_size=w,
                sigma=s,
            )
            output_video = Video(output_file_path)

    return output_video or input_video
