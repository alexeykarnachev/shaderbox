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


if __name__ == "__main__":
    parallax_fs = """
    #version 460
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform sampler2D u_base_texture;
    uniform sampler2D u_depth_map;
    uniform float u_time;
    uniform vec2 u_texture_size;
    uniform float u_focal_length = 1480.0;
    uniform float u_parallax_amount = 0.05;
    void main() {
        vec2 uv = vs_uv;
        float depth = texture(u_depth_map, uv).r;
        vec2 camera_move = vec2(sin(u_time), cos(u_time)) * u_focal_length * u_parallax_amount;
        vec2 offset = -camera_move * depth / u_texture_size;
        uv += offset;
        uv = clamp(uv, 0.0, 1.0);
        vec4 color = texture(u_base_texture, uv);
        fs_color = vec4(color.rgb, 1.0);
    }
    """

    bright_pass_fs = """
    #version 460
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform sampler2D u_source_texture;
    uniform float u_threshold = 0.5;
    void main() {
        vec4 color = texture(u_source_texture, vs_uv);
        float brightness = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
        fs_color = brightness > u_threshold ? color : vec4(0.0, 0.0, 0.0, 0.0);
    }
    """

    downscale_fs = """
    #version 460
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform sampler2D u_input_texture;
    void main() {
        fs_color = texture(u_input_texture, vs_uv);
    }
    """

    blur_fs = """
    #version 460
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform sampler2D u_blur_input;
    uniform vec2 u_pixel_size;
    void main() {
        vec4 color = vec4(0.0);
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                color += texture(u_blur_input, vs_uv + vec2(float(i), float(j)) * u_pixel_size);
        fs_color = color / 9.0;
    }
    """

    outline_fs = """
    #version 460
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform sampler2D u_outline_source;
    uniform vec2 u_pixel_size;
    uniform float u_outline_thickness = 1.0;
    void main() {
        vec4 center = texture(u_outline_source, vs_uv);
        float edge = 0.0;
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0) continue;
                vec4 neighbor = texture(u_outline_source, vs_uv + vec2(float(i), float(j)) * u_pixel_size);
                edge += length(center.rgb - neighbor.rgb);
            }
        edge = smoothstep(0.0, 1.0, edge) * u_outline_thickness;
        fs_color = vec4(edge, 0.0, 0.0, 1.0);
    }
    """

    combine_fs = """
    #version 460
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform sampler2D u_base_image;
    uniform sampler2D u_bloom_pass1;
    uniform sampler2D u_bloom_pass2;
    uniform sampler2D u_outline_pass;
    void main() {
        vec4 base = texture(u_base_image, vs_uv);
        vec4 bloom1 = texture(u_bloom_pass1, vs_uv);
        vec4 bloom2 = texture(u_bloom_pass2, vs_uv);
        vec4 outline = texture(u_outline_pass, vs_uv);
        vec3 color = base.rgb + bloom1.rgb + bloom2.rgb;
        color = outline.r * color * vec3(1.0, 0.8, 0.7) + 1.0 * color;
        fs_color = vec4(color, 1.0);
    }
    """

    # Example usage of the refactored code
    renderer = Renderer(is_headless=False)

    photo_texture = renderer.load_texture("photo.jpeg")
    depth_texture = renderer.load_texture("depth.png")

    parallax_node = renderer.create_node(
        name="Parallax",
        fs_source=parallax_fs,
        output_size=(800, 608),
        uniforms={
            "u_base_texture": photo_texture,
            "u_depth_map": depth_texture,
            "u_texture_size": (800.0, 608.0),
            "u_parallax_amount": 0.02,
            "u_focal_length": 1480.0,
            "u_time": lambda: renderer.render_time,
        },
    )

    bright_pass_node = renderer.create_node(
        name="Bright pass",
        fs_source=bright_pass_fs,
        output_size=(800, 608),
        uniforms={
            "u_source_texture": parallax_node,
            "u_threshold": 0.5,
        },
    )

    downscale_node1 = renderer.create_node(
        name="Downscale 1",
        fs_source=downscale_fs,
        output_size=(400, 304),
        uniforms={
            "u_input_texture": bright_pass_node,
        },
    )

    blur_node1 = renderer.create_node(
        name="Blur 1",
        fs_source=blur_fs,
        output_size=(400, 304),
        uniforms={
            "u_blur_input": downscale_node1,
            "u_pixel_size": (1.0 / 400, 1.0 / 304),
        },
    )

    downscale_node2 = renderer.create_node(
        name="Downscale 2",
        fs_source=downscale_fs,
        output_size=(200, 152),
        uniforms={"u_input_texture": blur_node1},
    )

    blur_node2 = renderer.create_node(
        name="Blur 2",
        fs_source=blur_fs,
        output_size=(200, 152),
        uniforms={
            "u_blur_input": downscale_node2,
            "u_pixel_size": (1.0 / 200, 1.0 / 152),
        },
    )

    outline_node = renderer.create_node(
        name="Outline",
        fs_source=outline_fs,
        output_size=(800, 608),
        uniforms={
            "u_outline_source": parallax_node,
            "u_pixel_size": (8.0 / 800, 8.0 / 608),
            "u_outline_thickness": 8.0,
        },
    )

    combine_node = renderer.create_node(
        name="Combine",
        fs_source=combine_fs,
        output_size=(800, 608),
        uniforms={
            "u_base_image": parallax_node,
            "u_bloom_pass1": blur_node1,
            "u_bloom_pass2": blur_node2,
            "u_outline_pass": outline_node,
        },
    )

    try:
        renderer.render_gif(combine_node, 1.0, 12.0, "output.gif")
        renderer.render_video(combine_node, 1.0, 30.0, "output.mp4")
        renderer.render_to_screen(combine_node, 60)
    finally:
        renderer.cleanup()
