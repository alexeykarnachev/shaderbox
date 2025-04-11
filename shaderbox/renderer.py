import time
from pathlib import Path
from typing import Any

import glfw
import imageio
import moderngl
import numpy as np
from loguru import logger
from PIL import Image

from shaderbox.gl import GLContext
from shaderbox.graph import Node, OutputSize, RenderGraph
from shaderbox.ui import UI


class Renderer:
    def __init__(
        self, is_headless: bool = False, window_size: tuple[int, int] | None = None
    ) -> None:
        self.fps: int = 60
        self.frame_idx: int = 0
        self.render_time: float = 0.0

        self.gl_context = GLContext(is_headless=is_headless, window_size=window_size)
        self.graph = RenderGraph()

    def create_node(
        self,
        fs_source: str,
        output_size: OutputSize,
        uniforms: dict[str, Any],
        name: str | None = None,
    ) -> Node:
        node = Node(self.gl_context, fs_source, output_size, uniforms, name)
        self.graph.add_node(node)
        return node

    def load_texture(self, source: Path | str | Image.Image) -> moderngl.Texture:
        return self.gl_context.load_texture(source)

    def run_editor(self, fps: int) -> None:
        if self.gl_context.window is None:
            logger.error("Can't run editor if is_headless=True")
            exit(0)

        target_frame_time = 1.0 / fps
        self.fps = fps

        ui = UI(self.gl_context)

        while not glfw.window_should_close(self.gl_context.window):
            start_time = glfw.get_time()
            self.frame_idx += 1
            self.render_time = self.frame_idx / fps

            glfw.poll_events()

            self.gl_context.context.clear(0.0, 0.0, 0.0, 1.0)
            self.graph.render()
            self.gl_context.context.screen.use()

            ui.update_and_render(self.graph)

            glfw.swap_buffers(self.gl_context.window)

            elapsed_time = glfw.get_time() - start_time
            time.sleep(max(0.0, target_frame_time - elapsed_time))

            if glfw.get_key(self.gl_context.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break

    def render_image(self, output_node: Node, file_path: str) -> None:
        self.graph.render()

        width, height = output_node.output_size
        data = output_node._fbo.read(viewport=(0, 0, width, height), components=3)
        image = np.frombuffer(data, np.uint8).reshape(height, width, 3)[::-1]

        logger.info(f"Saving image to {file_path}")
        imageio.imwrite(file_path, image)

    def _render_frames(
        self, output_node: Node, duration: float, fps: float
    ) -> np.ndarray:
        n_frames = int(duration * fps)
        width, height = output_node.output_size
        frames = np.empty((n_frames, height, width, 3), dtype=np.uint8)

        for frame_idx in range(n_frames):
            self.frame_idx = frame_idx
            self.render_time = frame_idx / fps if fps > 0 else 0.0
            self.graph.render()

            data = output_node._fbo.read(viewport=(0, 0, width, height), components=3)
            frames[frame_idx] = np.frombuffer(data, np.uint8).reshape(height, width, 3)[
                ::-1
            ]

        return frames

    def render_gif(
        self, output_node: Node, duration: float, fps: float, file_path: str
    ) -> None:
        frames = self._render_frames(output_node, duration, fps)

        logger.info(f"Saving GIF to {file_path}")
        imageio.mimwrite(file_path, frames, fps=fps)  # type: ignore

    def render_video(
        self, output_node: Node, duration: float, fps: float, file_path: str
    ) -> None:
        frames = self._render_frames(output_node, duration, fps)

        logger.info(f"Saving video to {file_path}")
        imageio.mimwrite(file_path, frames, fps=fps)  # type: ignore

    def cleanup(self) -> None:
        self.graph.release()
        self.gl_context.cleanup()
