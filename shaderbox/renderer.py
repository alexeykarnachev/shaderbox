import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import glfw
import imageio
import moderngl
import numpy as np
from loguru import logger
from PIL import Image

from shaderbox.gl import GLContext

Size = tuple[int, int]


class Node:
    """A node in the render graph representing a shader operation."""

    def __init__(
        self,
        gl_context: GLContext,
        fs_source: str,
        output_size: Size,
        uniforms: dict[str, Any],
        name: str | None = None,
    ) -> None:
        self._fs_source = fs_source
        self._output_size = output_size
        self._uniforms = uniforms
        self._name = name or str(id(self))
        self._graph: RenderGraph | None = None
        self._gl_context = gl_context

        self._program: moderngl.Program = self._gl_context.context.program(
            vertex_shader="""
            #version 460
            in vec2 a_pos;
            out vec2 vs_uv;
            void main() {
                gl_Position = vec4(a_pos, 0.0, 1.0);
                vs_uv = a_pos * 0.5 + 0.5;
            }
            """,
            fragment_shader=fs_source,
        )

        self._texture = self._gl_context.context.texture(output_size, 4)
        self._fbo = self._gl_context.context.framebuffer(
            color_attachments=[self._texture]
        )
        self._vbo = self._gl_context.context.buffer(
            np.array(
                [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
                dtype="f4",
            )
        )
        self._vao = self._gl_context.context.vertex_array(
            self._program, [(self._vbo, "2f", "a_pos")]
        )

    def render(self) -> None:
        """Render this node, binding all uniforms and executing the shader."""
        texture_unit = 0
        for u_name, u_value in self._uniforms.items():
            u_value = u_value() if callable(u_value) else u_value

            if isinstance(u_value, moderngl.Texture):
                u_value.use(texture_unit)
                u_value = texture_unit
                texture_unit += 1
            elif isinstance(u_value, Node):
                u_value._texture.use(texture_unit)
                u_value = texture_unit
                texture_unit += 1

            self._program[u_name] = u_value

        self._fbo.use()
        self._vao.render(moderngl.TRIANGLES)

    def release(self) -> None:
        """Release all OpenGL resources used by this node."""
        self._vbo.release()
        self._vao.release()
        self._fbo.release()
        self._texture.release()
        self._program.release()

        if self._graph:
            self._graph.remove_node(self)

    def get_parents(self) -> list["Node"]:
        """Get all nodes that are direct dependencies of this node."""
        return [n for n in self._uniforms.values() if isinstance(n, Node)]

    @property
    def name(self) -> str:
        """Get the name of this node."""
        return self._name

    @property
    def output_size(self) -> Size:
        """Get the output size of this node."""
        return self._output_size

    @property
    def uniforms(self) -> dict[str, Any]:
        """Get the uniforms dictionary of this node."""
        return self._uniforms


class RenderGraph:
    """Manages a directed acyclic graph of shader nodes."""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}

    def add_node(self, node: Node) -> None:
        """Add a node to the render graph."""
        if node.name in self._nodes:
            raise KeyError(f"Node with name {node.name} already exists")

        self._nodes[node.name] = node
        node._graph = self

    def remove_node(self, node: Node) -> None:
        """Remove a node from the render graph."""
        if node.name in self._nodes:
            del self._nodes[node.name]
            node._graph = None

    def get_node(self, name: str) -> Node | None:
        """Get a node by name."""
        return self._nodes.get(name)

    def render(self) -> None:
        """Render all nodes in the graph in topological order."""
        in_degree = {node: len(node.get_parents()) for node in self._nodes.values()}
        children = defaultdict(list)

        for node in self._nodes.values():
            for parent in node.get_parents():
                children[parent].append(node)

        queue = deque([node for node in self._nodes.values() if in_degree[node] == 0])

        while queue:
            current = queue.popleft()
            current.render()

            for child in children[current]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    def release(self) -> None:
        """Release all nodes in the graph."""
        for node in list(self._nodes.values()):
            node.release()

        self._nodes.clear()


class Renderer:
    """Main renderer class handling all rendering operations."""

    def __init__(
        self, is_headless: bool = False, window_size: Size | None = None
    ) -> None:
        """Initialize the renderer.

        Args:
            is_headless: Whether to run in headless mode (no window)
            window_size: Size of the window if not headless
        """
        self.fps: int = 60
        self.frame_idx: int = 0
        self.render_time: float = 0.0

        self.gl_context = GLContext(is_headless=is_headless, window_size=window_size)
        self.graph = RenderGraph()

    def create_node(
        self,
        fs_source: str,
        output_size: Size,
        uniforms: dict[str, Any],
        name: str | None = None,
    ) -> Node:
        """Create and register a new node in the render graph."""
        node = Node(self.gl_context, fs_source, output_size, uniforms, name)
        self.graph.add_node(node)
        return node

    def load_texture(self, source: Path | str | Image.Image) -> moderngl.Texture:
        """Load a texture using the renderer's OpenGL context."""
        return self.gl_context.load_texture(source)

    def render_to_screen(self, output_node: Node, fps: int) -> None:
        """Render the output of a node to the screen in a loop."""
        if self.gl_context.window is None:
            logger.error("Can't render to screen if is_headless=True")
            exit(0)

        target_frame_time = 1.0 / fps
        self.fps = fps

        while not glfw.window_should_close(self.gl_context.window):
            start_time = glfw.get_time()
            self.frame_idx += 1
            self.render_time = self.frame_idx / fps

            glfw.poll_events()

            self.gl_context.context.clear(0.0, 0.0, 0.0, 1.0)
            self.graph.render()
            self.gl_context.context.screen.use()
            self.gl_context.context.copy_framebuffer(
                self.gl_context.context.screen, output_node._fbo
            )

            glfw.swap_buffers(self.gl_context.window)

            elapsed_time = glfw.get_time() - start_time
            time.sleep(max(0.0, target_frame_time - elapsed_time))

            if glfw.get_key(self.gl_context.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break

    def render_image(self, output_node: Node, file_path: str) -> None:
        """Render a single frame to an image file."""
        self.graph.render()

        width, height = output_node.output_size
        data = output_node._fbo.read(viewport=(0, 0, width, height), components=3)
        image = np.frombuffer(data, np.uint8).reshape(height, width, 3)[::-1]

        logger.info(f"Saving image to {file_path}")
        imageio.imwrite(file_path, image)

    def _render_frames(
        self, output_node: Node, duration: float, fps: float
    ) -> np.ndarray:
        """Render multiple frames and return them as a numpy array."""
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
        """Render an animated GIF and save it to a file."""
        frames = self._render_frames(output_node, duration, fps)

        logger.info(f"Saving GIF to {file_path}")
        imageio.mimwrite(file_path, frames, fps=fps)  # type: ignore

    def render_video(
        self, output_node: Node, duration: float, fps: float, file_path: str
    ) -> None:
        """Render a video and save it to a file."""
        frames = self._render_frames(output_node, duration, fps)

        logger.info(f"Saving video to {file_path}")
        imageio.mimwrite(file_path, frames, fps=fps)  # type: ignore

    def cleanup(self) -> None:
        """Clean up all OpenGL resources."""
        self.graph.release()
        self.gl_context.cleanup()
