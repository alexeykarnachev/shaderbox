from imgui_bundle import imgui, imgui_ctx
from loguru import logger

from shaderbox.app import App
from shaderbox.media import MediaDetails
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_primitives import caption_text, centered_image, primary_button
from shaderbox.widgets.details import draw_media_details


def draw(app: App) -> None:
    if not (ui_document := app.ui_documents.get(app.current_document_id)):
        return

    imgui.spacing()

    preview_col_w = float(SIZE.RENDER_PREVIEW_W)
    row_start = imgui.get_cursor_pos()

    # Controls drawn first: their measured height sizes the preview box beside them.
    imgui.set_cursor_pos((row_start.x + preview_col_w + SPACE.XL, row_start.y))
    with imgui_ctx.begin_group():
        ui_document.ui_state.render_media_details = draw_media_details(
            app,
            ui_document.ui_state.render_media_details,
        )
        ui_document.ui_state.render_media_details = _draw_render_button(
            app,
            ui_document.ui_state.render_media_details,
        )
    controls_h = imgui.get_item_rect_size().y

    imgui.set_cursor_pos(row_start)
    tex = app.preview_canvas.texture
    centered_image(tex.glo, tex.size, preview_col_w, controls_h)
    imgui.set_cursor_pos(row_start)
    imgui.dummy((0.0, controls_h))


def _draw_render_button(app: App, details: MediaDetails) -> MediaDetails:
    if app.current_document_id not in app.ui_documents:
        return details

    media_type = "video" if details.is_video else "image"
    has_path = bool(details.file_details.path)

    imgui.begin_disabled(not has_path)
    if primary_button("Render"):
        # Defer the encode one frame so the "Rendering..." cue paints before it freezes the
        # loop (update_and_draw runs the request, then writes the result back). Capture the
        # document id, not ui_document, so a document switch before the run frame can't render the wrong one.
        document_id = app.current_document_id
        pending = details

        def _run_render() -> None:
            target = app.ui_documents.get(document_id)
            if target is None:
                return
            try:
                target.ui_state.render_media_details = target.document.render_media(
                    pending
                )
            except Exception as e:
                # A toast, not just a log: the "Rendering..." cue clears either way, so a
                # silent failure is indistinguishable from a finished render.
                logger.error(f"Failed to render media: {e}")
                app.notifications.push(f"Render failed: {e!s}", COLOR.STATE_ERROR[:3])

        app.render_defer.submit(_run_render)
    imgui.end_disabled()

    if not has_path:
        imgui.same_line()
        caption_text(f"no output {media_type} file")

    return details
