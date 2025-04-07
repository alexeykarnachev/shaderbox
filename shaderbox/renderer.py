import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Union

import glfw
import imageio
import moderngl
import numpy as np
from loguru import logger
from PIL import Image

# Configure loguru
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
)

# Size type alias for clarity
Size = Tuple[int, int]


# Abstract base class for uniforms
class UniformLike(ABC):
    @abstractmethod
    def get_value(self, renderer: "Renderer") -> Any:
        """Retrieve the uniform's value during rendering."""
        pass


# Concrete uniform implementations
class ValueUniform(UniformLike):
    def __init__(self, value: Any):
        self.value = value

    def get_value(self, renderer: "Renderer") -> Any:
        assert renderer
        return self.value


class DynamicUniform(UniformLike):
    def __init__(self, callback: Callable[["Renderer"], Any]):
        self.callback = callback

    def get_value(self, renderer: "Renderer") -> Any:
        return self.callback(renderer)


class LazyTexture(UniformLike):
    def __init__(self, source: Union[str, Image.Image, moderngl.Texture]):
        self.source = source
        self._texture: moderngl.Texture | None = (
            None if not isinstance(source, moderngl.Texture) else source
        )

    def get_value(self, renderer: "Renderer") -> moderngl.Texture:
        if self._texture is None:
            ctx = renderer.context
            if isinstance(self.source, str):
                img = Image.open(self.source).convert("RGBA")
            elif isinstance(self.source, Image.Image):
                img = self.source.convert("RGBA")
            else:
                raise ValueError("Invalid source type for LazyTexture")
            self._texture = ctx.texture(img.size, 4, img.tobytes())
            logger.info(f"Loaded texture with size {img.size}")
        return self._texture


class ShaderNodeUniform(UniformLike):
    def __init__(self, node: "ShaderNode"):
        self.node = node

    def get_value(self, renderer: "Renderer") -> moderngl.Texture:
        self.node.render(renderer)
        return self.node._node_context.texture


# Data-only class for node context
@dataclass
class NodeContext:
    program: moderngl.Program
    texture: moderngl.Texture
    fbo: moderngl.Framebuffer
    vao: moderngl.VertexArray


class ShaderNode:
    VS_SOURCE = """
    #version 460
    in vec2 a_pos;
    out vec2 vs_uv;
    void main() {
        gl_Position = vec4(a_pos, 0.0, 1.0);
        vs_uv = a_pos * 0.5 + 0.5;
    }
    """

    def __init__(
        self,
        fs_source: str,
        output_size: Size,
        uniforms: Dict[str, Union[UniformLike, Any]],
    ):
        self.fs_source = fs_source
        self.output_size = output_size
        self.uniforms: Dict[str, UniformLike] = {}
        for name, value in uniforms.items():
            if isinstance(value, UniformLike):
                self.uniforms[name] = value
            elif isinstance(value, ShaderNode):
                self.uniforms[name] = ShaderNodeUniform(value)
            else:
                self.uniforms[name] = ValueUniform(value)
        self._node_context: NodeContext  # Set in _initialize

    def __setitem__(self, name: str, value: Union[UniformLike, Any]):
        if isinstance(value, UniformLike):
            self.uniforms[name] = value
        elif isinstance(value, ShaderNode):
            self.uniforms[name] = ShaderNodeUniform(value)
        else:
            self.uniforms[name] = ValueUniform(value)

    def _initialize(self, renderer: "Renderer"):
        if hasattr(self, "_node_context"):
            return
        ctx = renderer.context
        program = ctx.program(
            vertex_shader=self.VS_SOURCE, fragment_shader=self.fs_source
        )
        texture = ctx.texture(self.output_size, 4)
        fbo = ctx.framebuffer(color_attachments=[texture])
        vao = ctx.vertex_array(program, [(renderer.vbo, "2f", "a_pos")])
        self._node_context = NodeContext(program, texture, fbo, vao)
        logger.debug(f"Initialized ShaderNode with shader: {self.fs_source[:50]}...")

    def render(self, renderer: "Renderer") -> moderngl.Framebuffer:
        self._initialize(renderer)
        node_context = self._node_context

        # Resolve and set uniforms
        texture_unit = 0
        for name, uniform in self.uniforms.items():
            if name not in node_context.program:
                logger.warning(f"Uniform '{name}' not found in shader program.")
                continue
            value = uniform.get_value(renderer)
            if isinstance(value, moderngl.Texture):
                value.use(texture_unit)
                uniform_value = texture_unit
                texture_unit += 1
            else:
                uniform_value = value
            node_context.program[name] = uniform_value

        # Render
        node_context.fbo.use()
        renderer.context.clear(0.0, 0.0, 0.0, 1.0)
        node_context.vao.render(moderngl.TRIANGLES)
        return node_context.fbo


class Renderer:
    def __init__(self, screen_size: Size | None = None):
        self.is_headless = screen_size is None

        if screen_size is not None:
            # Initialize GLFW and create window
            glfw.init()
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            self._window = glfw.create_window(
                screen_size[0], screen_size[1], "ShaderBox", None, None
            )
            glfw.make_context_current(self._window)

        # Initialize ModernGL context
        self._ctx = moderngl.create_context(standalone=self.is_headless, require=460)
        logger.info(f"Created ModernGL context (is_headless={self.is_headless})")

        # Set up vertex buffer
        self.vbo = self._ctx.buffer(
            np.array(
                [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
                dtype="f4",
            )
        )

        # Frame tracking
        self._frame_idx = 0
        self._fps = 0.0

    @property
    def context(self) -> moderngl.Context:
        return self._ctx

    @property
    def time(self) -> float:
        return self._frame_idx / self._fps if self._fps > 0 else 0.0

    def render_to_screen(self, node: ShaderNode, fps: float = 60.0):
        if self.is_headless:
            logger.warning("Cannot render to screen in headless mode.")
            return

        self._fps = fps
        target_frame_time = 1.0 / fps
        logger.info(f"Starting screen rendering loop at {fps} FPS")

        while not glfw.window_should_close(self._window):
            start_time = glfw.get_time()
            self._frame_idx += 1
            self._ctx.clear(0.0, 0.0, 0.0, 1.0)
            fbo = node.render(self)
            self._ctx.screen.use()
            self._ctx.copy_framebuffer(self._ctx.screen, fbo)
            glfw.swap_buffers(self._window)

            elapsed_time = glfw.get_time() - start_time
            sleep_time = max(0.0, target_frame_time - elapsed_time)
            time.sleep(sleep_time)

            glfw.poll_events()
            if glfw.get_key(self._window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break

        glfw.destroy_window(self._window)
        glfw.terminate()

    def _render_frames(self, node: ShaderNode, num_frames: int, fps: float) -> list:
        if not self.is_headless:
            logger.warning(
                "Rendering offscreen in non-headless mode; output may not match screen."
            )
        images = []
        self._fps = fps
        logger.info(f"Rendering {num_frames} frames at {fps} FPS")
        fbo = self._ctx.framebuffer(
            color_attachments=[self._ctx.texture(node.output_size, 4)]
        )

        for frame_idx in range(num_frames):
            self._frame_idx = frame_idx
            node_fbo = node.render(self)
            fbo.use()
            self._ctx.copy_framebuffer(fbo, node_fbo)
            width, height = node.output_size
            image_data = fbo.read(viewport=(0, 0, width, height), components=4)
            image = np.frombuffer(image_data, dtype=np.uint8).reshape(height, width, 4)
            image = image[::-1, :, :]
            images.append(image)
        fbo.release()
        return images

    def render_gif(self, node: ShaderNode, duration: float, fps: float, filename: str):
        num_frames = int(duration * fps)
        images = self._render_frames(node, num_frames, fps)
        rgb_images = [image[:, :, :3] for image in images]
        logger.info(f"Saving GIF to {filename}")
        imageio.mimwrite(filename, rgb_images, fps=fps)

    def render_image(self, node: ShaderNode, filename: str):
        images = self._render_frames(node, 1, 30.0)
        logger.info(f"Saving image to {filename}")
        imageio.imwrite(filename, images[0])

    def render_video(
        self, node: ShaderNode, duration: float, fps: float, filename: str
    ):
        num_frames = int(duration * fps)
        images = self._render_frames(node, num_frames, fps)
        rgb_images = [image[:, :, :3] for image in images]
        logger.info(f"Saving video to {filename}")
        imageio.mimwrite(filename, rgb_images, fps=fps)

    def destroy(self):
        self.vbo.release()
        logger.info("Renderer resources cleaned up")


if __name__ == "__main__":
    # Shader definitions
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
        uv.y = 1.0 - uv.y;
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

    # Initialize renderer
    renderer = Renderer(screen_size=None)

    # Create nodes with meaningful names and raw values
    parallax_node = ShaderNode(
        parallax_fs,
        (800, 608),
        {
            "u_base_texture": LazyTexture("photo.jpeg"),
            "u_depth_map": LazyTexture("depth.png"),
            "u_texture_size": (800.0, 608.0),  # Raw tuple
            "u_parallax_amount": 0.02,  # Raw float
            "u_focal_length": 1480.0,  # Raw float
            "u_time": DynamicUniform(lambda r: r.time),
        },
    )

    bright_pass_node = ShaderNode(
        bright_pass_fs,
        (800, 608),
        {"u_source_texture": parallax_node, "u_threshold": 0.5},  # Raw float
    )

    downscale_node1 = ShaderNode(
        downscale_fs,
        (400, 304),
        {"u_input_texture": bright_pass_node},
    )

    blur_node1 = ShaderNode(
        blur_fs,
        (400, 304),
        {
            "u_blur_input": downscale_node1,
            "u_pixel_size": (1.0 / 400, 1.0 / 304),  # Raw tuple
        },
    )

    downscale_node2 = ShaderNode(
        downscale_fs,
        (200, 152),
        {"u_input_texture": blur_node1},
    )

    blur_node2 = ShaderNode(
        blur_fs,
        (200, 152),
        {
            "u_blur_input": downscale_node2,
            "u_pixel_size": (1.0 / 200, 1.0 / 152),  # Raw tuple
        },
    )

    outline_node = ShaderNode(
        outline_fs,
        (800, 608),
        {
            "u_outline_source": parallax_node,
            "u_pixel_size": (8.0 / 800, 8.0 / 608),  # Raw tuple
            "u_outline_thickness": 8.0,  # Raw float
        },
    )

    combine_node = ShaderNode(
        combine_fs,
        (800, 608),
        {
            "u_base_image": parallax_node,
            "u_bloom_pass1": blur_node1,
            "u_bloom_pass2": blur_node2,
            "u_outline_pass": outline_node,
        },
    )

    # Render outputs
    renderer.render_image(combine_node, "output.png")
    renderer.render_gif(combine_node, 5.0, 30, "output.gif")
    renderer.render_video(combine_node, 5.0, 30, "output.mp4")
    renderer.render_to_screen(combine_node, fps=60.0)

    renderer.destroy()
