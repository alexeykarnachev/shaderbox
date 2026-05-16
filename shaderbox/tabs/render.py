import moderngl
from imgui_bundle import imgui
from loguru import logger

from shaderbox.app import App
from shaderbox.media import MediaDetails
from shaderbox.widgets.details import draw_media_details


def draw(app: App) -> None:
    if not (ui_node := app.ui_nodes.get(app.current_node_id)):
        return

    ui_node.ui_state.render_media_details = draw_media_details(
        app,
        ui_node.ui_state.render_media_details,
    )
    _, ui_node.ui_state.render_media_details = _draw_render_button(
        app,
        ui_node.ui_state.render_media_details,
        app.preview_canvas.texture,
    )


def _draw_render_button(
    app: App,
    details: MediaDetails,
    preview_texture: moderngl.Texture | None,
) -> tuple[bool, MediaDetails]:
    if preview_texture is not None:
        imgui.text("Render preview:")
        imgui.image(
            imgui.ImTextureRef(preview_texture.glo),
            image_size=(preview_texture.size[0], preview_texture.size[1]),
            uv0=(0, 1),
            uv1=(1, 0),
        )

    is_rendered = False
    media_type = "video" if details.is_video else "image"
    if details.file_details.path:
        if not (ui_node := app.ui_nodes.get(app.current_node_id)):
            imgui.text_colored(
                (1.0, 1.0, 0.0, 1.0), "Node is not selected, nothing to render"
            )
        elif imgui.button("Render##media"):
            try:
                details = ui_node.node.render_media(details)
                is_rendered = True
            except Exception as e:
                logger.error(f"Failed to render media: {e}")
    else:
        imgui.text_colored(
            (1.0, 1.0, 0.0, 1.0),
            f"Select output file path to render the {media_type}",
        )

    return is_rendered, details
