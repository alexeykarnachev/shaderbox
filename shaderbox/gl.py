from pathlib import Path

import glfw
import moderngl
from loguru import logger
from PIL import Image


class GLContext:
    """Manages OpenGL context and window creation."""

    def __init__(
        self,
        is_headless: bool = False,
        window_size: tuple[int, int] | None = None,
        window_title: str = "ShaderBox",
    ) -> None:
        """Initialize the OpenGL context and window if needed.

        Args:
            is_headless: Whether to create a headless context (no window)
            window_size: Size of the window to create (width, height)
            window_title: Title of the window
        """
        self.is_headless = is_headless
        self.window_title = window_title
        self.window: glfw._GLFWwindow | None = None
        self.context: moderngl.Context

        self._initialize(is_headless, window_size)

    def _initialize(
        self,
        is_headless: bool,
        window_size: tuple[int, int] | None,
    ) -> None:
        """Initialize the OpenGL context and window.

        Args:
            is_headless: Whether to create a headless context (no window)
            window_size: Size of the window to create (width, height)
        """
        if is_headless and window_size is not None:
            logger.warning("window_size provided in headless mode, ignoring")

        if not is_headless:
            glfw.init()
            monitor = None

            if window_size is None:
                monitor = glfw.get_primary_monitor()
                mode = glfw.get_video_mode(monitor)
                window_size = (mode.size.width, mode.size.height)

            self.window = glfw.create_window(
                window_size[0], window_size[1], self.window_title, monitor, None
            )
            glfw.make_context_current(self.window)

        self.context = moderngl.create_context(standalone=is_headless)

    def load_texture(self, source: Path | str | Image.Image) -> moderngl.Texture:
        """Load a texture from a file or PIL image.

        Args:
            source: Path to image file or PIL Image object

        Returns:
            moderngl.Texture: OpenGL texture object
        """
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
        """Release all OpenGL resources."""
        if self.window:
            glfw.terminate()
