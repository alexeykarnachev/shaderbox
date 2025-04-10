import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple

import glfw
import imageio
import moderngl
import numpy as np
from loguru import logger

from shaderbox import gl

Size = Tuple[int, int]


fps: int = 60
frame_idx: int = 0
render_time: float = 0.0


class Node:
    _registry: Dict[str, "Node"] = {}

    def __init__(
        self,
        fs_source: str,
        output_size: Size,
        uniforms: Dict[str, Any],
        name: str | None = None,
    ) -> None:
        self._fs_source = fs_source
        self._output_size = output_size
        self._uniforms = uniforms
        self._name = name or str(id(self))

        if self._name in self._registry:
            raise KeyError(f"Node with name {self._name} already exists")

        self._program: moderngl.Program = gl.context.program(
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

        self._texture = gl.context.texture(output_size, 4)
        self._fbo = gl.context.framebuffer(color_attachments=[self._texture])
        self._vbo = gl.context.buffer(
            np.array(
                [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
                dtype="f4",
            )
        )
        self._vao = gl.context.vertex_array(self._program, [(self._vbo, "2f", "a_pos")])

        self._registry[self._name] = self

    def render(self):
        texture_unit = 0
        for u_name, u_value in self._uniforms.items():
            u_value = u_value() if callable(u_value) else u_value

            if isinstance(u_value, moderngl.Texture):
                u_value.use(texture_unit)
                u_value = texture_unit
                texture_unit += 1
            elif isinstance(u_value, Node):
                self._registry[u_value._name]._texture.use(texture_unit)
                u_value = texture_unit
                texture_unit += 1

            self._program[u_name] = u_value

        self._fbo.use()
        self._vao.render(moderngl.TRIANGLES)

    def release(self):
        del self._registry[self._name]

        self._vbo.release()
        self._vao.release()
        self._fbo.release()
        self._texture.release()
        self._program.release()

    def get_parents(self) -> List["Node"]:
        return [n for n in self._uniforms.values() if isinstance(n, Node)]

    @classmethod
    def render_graph(cls):
        in_degree = {node: len(node.get_parents()) for node in cls._registry.values()}
        children = defaultdict(list)

        for node in cls._registry.values():
            for parent in node.get_parents():
                children[parent].append(node)

        queue = deque([node for node in cls._registry.values() if in_degree[node] == 0])

        while queue:
            current = queue.popleft()
            current.render()

            for child in children[current]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    @classmethod
    def release_graph(cls):
        for node in cls._registry.values():
            node.release()


def render_to_screen(node: Node, fps: int):
    global frame_idx, render_time

    if gl.window is None:
        logger.error("Can't render to screen if is_headless=True")
        exit(0)

    target_frame_time = 1.0 / fps

    while not glfw.window_should_close(gl.window):
        start_time = glfw.get_time()
        frame_idx += 1
        render_time = frame_idx / fps

        glfw.poll_events()

        gl.context.clear(0.0, 0.0, 0.0, 1.0)
        Node.render_graph()
        gl.context.screen.use()
        gl.context.copy_framebuffer(gl.context.screen, node._fbo)

        ui.render()

        glfw.swap_buffers(gl.window)

        elapsed_time = glfw.get_time() - start_time
        time.sleep(max(0.0, target_frame_time - elapsed_time))

        if glfw.get_key(gl.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break


def render_image(node: Node, file_path: str):
    Node.render_graph()

    width, height = node._output_size
    data = node._fbo.read(viewport=(0, 0, width, height), components=3)
    image = np.frombuffer(data, np.uint8).reshape(height, width, 3)[::-1]

    logger.info(f"Saving image to {file_path}")
    imageio.imwrite(file_path, image)


def render_gif(node: Node, duration: float, fps: float, file_path: str):
    global render_time, frame_idx

    n_frames = int(duration * fps)
    width, height = node._output_size
    frames = np.empty((n_frames, height, width, 3), dtype=np.uint8)

    for frame_idx in range(n_frames):
        render_time = frame_idx / fps if fps > 0 else 0.0
        Node.render_graph()

        data = node._fbo.read(viewport=(0, 0, width, height), components=3)
        frames[frame_idx] = np.frombuffer(data, np.uint8).reshape(height, width, 3)[
            ::-1
        ]

    logger.info(f"Saving GIF to {file_path}")
    imageio.mimwrite(file_path, frames, fps=fps)  # type: ignore


def render_video(node: Node, duration: float, fps: float, file_path: str):
    global render_time, frame_idx

    n_frames = int(duration * fps)
    width, height = node._output_size
    frames = np.empty((n_frames, height, width, 3), dtype=np.uint8)

    for frame_idx in range(n_frames):
        render_time = frame_idx / fps if fps > 0 else 0.0
        Node.render_graph()

        data = node._fbo.read(viewport=(0, 0, width, height), components=3)
        frames[frame_idx] = np.frombuffer(data, np.uint8).reshape(height, width, 3)[
            ::-1
        ]

    logger.info(f"Saving video to {file_path}")
    imageio.mimwrite(file_path, frames, fps=fps)  # type: ignore


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

    gl.initialize(is_headless=False)

    photo_texture = gl.load_texture("photo.jpeg")
    depth_texture = gl.load_texture("depth.png")

    parallax_node = Node(
        name="Parallax",
        fs_source=parallax_fs,
        output_size=(800, 608),
        uniforms={
            "u_base_texture": photo_texture,
            "u_depth_map": depth_texture,
            "u_texture_size": (800.0, 608.0),
            "u_parallax_amount": 0.02,
            "u_focal_length": 1480.0,
            "u_time": lambda: render_time,
        },
    )

    bright_pass_node = Node(
        name="Bright pass",
        fs_source=bright_pass_fs,
        output_size=(800, 608),
        uniforms={
            "u_source_texture": parallax_node,
            "u_threshold": 0.5,
        },
    )

    downscale_node1 = Node(
        name="Downscale 1",
        fs_source=downscale_fs,
        output_size=(400, 304),
        uniforms={
            "u_input_texture": bright_pass_node,
        },
    )

    blur_node1 = Node(
        name="Blur 1",
        fs_source=blur_fs,
        output_size=(400, 304),
        uniforms={
            "u_blur_input": downscale_node1,
            "u_pixel_size": (1.0 / 400, 1.0 / 304),
        },
    )

    downscale_node2 = Node(
        name="Downscale 2",
        fs_source=downscale_fs,
        output_size=(200, 152),
        uniforms={"u_input_texture": blur_node1},
    )

    blur_node2 = Node(
        name="Blur 2",
        fs_source=blur_fs,
        output_size=(200, 152),
        uniforms={
            "u_blur_input": downscale_node2,
            "u_pixel_size": (1.0 / 200, 1.0 / 152),
        },
    )

    outline_node = Node(
        name="Outline",
        fs_source=outline_fs,
        output_size=(800, 608),
        uniforms={
            "u_outline_source": parallax_node,
            "u_pixel_size": (8.0 / 800, 8.0 / 608),
            "u_outline_thickness": 8.0,
        },
    )

    combine_node = Node(
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

    render_gif(combine_node, 1.0, 12.0, "output.gif")
    render_video(combine_node, 1.0, 30.0, "output.mp4")
    render_to_screen(combine_node, 60)
