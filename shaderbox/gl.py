from pathlib import Path
from typing import Tuple

import glfw
import moderngl
from loguru import logger
from PIL import Image

context: moderngl.Context
window: glfw._GLFWwindow | None = None


def initialize(
    is_headless: bool = False,
    window_size: Tuple[int, int] | None = None,
):
    global context, window

    if is_headless and window_size is not None:
        logger.warning("window_size provided in headless mode, ignoring")

    if not is_headless:
        glfw.init()
        monitor = None

        if window_size is None:
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            window_size = (mode.size.width, mode.size.height)

        window = glfw.create_window(
            window_size[0], window_size[1], "ShaderBox", monitor, None
        )
        glfw.make_context_current(window)

    context = moderngl.create_context(standalone=is_headless)


def load_texture(source: Path | str | Image.Image) -> moderngl.Texture:
    if isinstance(source, Image.Image):
        image, source = source, "image"
    else:
        image = Image.open(source).convert("RGBA")
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    texture = context.texture(image.size, 4, image.tobytes())
    logger.info(f"Loaded texture from {source} with size {image.size}")

    return texture
