from imgui_bundle import imgui, imgui_ctx
from loguru import logger

from shaderbox.app import App
from shaderbox.media import MediaDetails
from shaderbox.theme import SIZE, SPACE
from shaderbox.ui_primitives import caption_text, centered_image, primary_button
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
    tex = app.preview_canvas.texture
    centered_image(tex.glo, tex.size, preview_col_w, controls_h)
    imgui.set_cursor_pos(row_start)
    imgui.dummy((0.0, controls_h))


def _draw_render_button(app: App, details: MediaDetails) -> MediaDetails:
    if app.current_node_id not in app.ui_nodes:
        return details

    media_type = "video" if details.is_video else "image"
    has_path = bool(details.file_details.path)

    imgui.begin_disabled(not has_path)
    if primary_button("Render"):
        # Defer the encode one frame so the "Rendering..." cue paints before it freezes the
        # loop (update_and_draw runs the request, then writes the result back). Capture the
        # node id, not ui_node, so a node switch before the run frame can't render the wrong one.
        node_id = app.current_node_id
        pending = details

        def _run_render() -> None:
            target = app.ui_nodes.get(node_id)
            if target is None:
                return
            try:
                target.ui_state.render_media_details = target.node.render_media(pending)
            except Exception as e:
                logger.error(f"Failed to render media: {e}")

        app.render_defer.submit(_run_render)
    imgui.end_disabled()

    if not has_path:
        imgui.same_line()
        caption_text(f"Select an output file to render the {media_type}")

    return details
