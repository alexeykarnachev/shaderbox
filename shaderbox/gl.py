from pathlib import Path

import glfw
import moderngl
from loguru import logger
from PIL import Image


class GLContext:
    def __init__(
        self,
        is_headless: bool = False,
        window_size: tuple[int, int] | None = None,
        window_title: str = "ShaderBox",
    ) -> None:
        self.is_headless = is_headless
        self.window_title = window_title
        self.window: glfw._GLFWwindow | None = None
        self.context: moderngl.Context

        if is_headless and window_size is not None:
            logger.warning("window_size provided in headless mode, ignoring")

        if not is_headless:
            glfw.init()

            if window_size is None:
                monitor = glfw.get_primary_monitor()
                mode = glfw.get_video_mode(monitor)
                window_size = (mode.size.width, mode.size.height)

            self.window = glfw.create_window(
                width=window_size[0],
                height=window_size[1],
                title=self.window_title,
                monitor=None,
                share=None,
            )
            glfw.make_context_current(self.window)

        self.context = moderngl.create_context(standalone=is_headless)

    def load_texture(self, source: Path | str | Image.Image) -> moderngl.Texture:
        if isinstance(source, Image.Image):
            image, source_name = source, "image"
        else:
            image = Image.open(source).convert("RGBA")
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            source_name = str(source)

        texture = self.context.texture(image.size, 4, image.tobytes())
        logger.info(f"Loaded texture from {source_name} with size {image.size}")

        return texture

    def cleanup(self) -> None:
        if self.window:
            glfw.terminate()
