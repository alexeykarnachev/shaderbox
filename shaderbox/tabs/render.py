import moderngl
from imgui_bundle import imgui, imgui_ctx
from loguru import logger

from shaderbox.app import App
from shaderbox.media import MediaDetails
from shaderbox.theme import SIZE, SPACE
from shaderbox.ui_primitives import caption_text, primary_button
from shaderbox.widgets.details import draw_media_details


def draw(app: App) -> None:
    if not (ui_node := app.ui_nodes.get(app.current_node_id)):
        return

    imgui.spacing()

    preview_col_w = float(SIZE.RENDER_PREVIEW_W)
    row_start = imgui.get_cursor_pos()

    # Controls drawn first: their measured height sizes the preview box beside them.
    imgui.set_cursor_pos((row_start.x + preview_col_w + SPACE.XL, row_start.y))
    with imgui_ctx.begin_group():
        ui_node.ui_state.render_media_details = draw_media_details(
            app,
            ui_node.ui_state.render_media_details,
        )
        ui_node.ui_state.render_media_details = _draw_render_button(
            app,
            ui_node.ui_state.render_media_details,
        )
    controls_h = imgui.get_item_rect_size().y

    imgui.set_cursor_pos(row_start)
    _draw_preview(app.preview_canvas.texture, preview_col_w, controls_h)
    imgui.set_cursor_pos(row_start)
    imgui.dummy((0.0, controls_h))


def _draw_preview(texture: moderngl.Texture, box_w: float, box_h: float) -> None:
    aspect = texture.size[0] / texture.size[1]
    w, h = box_h * aspect, box_h
    if w > box_w:
        w, h = box_w, box_w / aspect
    # Letterbox centered in the [box_w x box_h] cell; the cell is box_h tall for any aspect.
    origin = imgui.get_cursor_pos()
    imgui.set_cursor_pos((origin.x + (box_w - w) * 0.5, origin.y + (box_h - h) * 0.5))
    imgui.image(
        imgui.ImTextureRef(texture.glo),
        image_size=(w, h),
        uv0=(0, 1),
        uv1=(1, 0),
    )


def _draw_render_button(app: App, details: MediaDetails) -> MediaDetails:
    if not (ui_node := app.ui_nodes.get(app.current_node_id)):
        return details

    media_type = "video" if details.is_video else "image"
    has_path = bool(details.file_details.path)

    imgui.begin_disabled(not has_path)
    if primary_button("Render"):
        try:
            details = ui_node.node.render_media(details)
        except Exception as e:
            logger.error(f"Failed to render media: {e}")
    imgui.end_disabled()

    if not has_path:
        imgui.same_line()
        caption_text(f"Select an output file to render the {media_type}")

    return details
