import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Union

import glfw
import imageio
import imgui
import moderngl
import numpy as np
from imgui.integrations.glfw import GlfwRenderer as ImguiGlfwRenderer
from loguru import logger
from PIL import Image

logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
)

Size = Tuple[int, int]


class UniformLike(ABC):
    @abstractmethod
    def get_value(self, renderer: "Renderer") -> Any:
        pass


class ValueUniform(UniformLike):
    def __init__(self, value: Any):
        self.value = value

    def get_value(self, renderer: "Renderer") -> Any:
        return self.value


class DynamicUniform(UniformLike):
    def __init__(self, callback: Callable[["Renderer"], Any]):
        self.callback = callback

    def get_value(self, renderer: "Renderer") -> Any:
        return self.callback(renderer)


class LazyTexture(UniformLike):
    def __init__(self, source: Union[str, Image.Image, moderngl.Texture]):
        self.source = source
        self._texture = None if not isinstance(source, moderngl.Texture) else source

    def get_value(self, renderer: "Renderer") -> moderngl.Texture:
        if self._texture is None:
            if isinstance(self.source, str):
                img = Image.open(self.source).convert("RGBA")
            elif isinstance(self.source, Image.Image):
                img = self.source.convert("RGBA")
            else:
                raise ValueError("Invalid source type for LazyTexture")
            self._texture = renderer.context.texture(img.size, 4, img.tobytes())
            logger.info(f"Loaded texture with size {img.size}")
        return self._texture


class ShaderNodeUniform(UniformLike):
    def __init__(self, node: "ShaderNode"):
        self.node = node

    def get_value(self, renderer: "Renderer") -> moderngl.Texture:
        renderer.render_node(self.node)
        return renderer.node_contexts[self.node]["texture"]


@dataclass
class ShaderNode:
    fs_source: str
    output_size: Size
    uniforms: Dict[str, Union[UniformLike, Any]]

    def __hash__(self):
        return int(hashlib.md5(self.fs_source.encode("utf-8")).hexdigest(), 16)


class Renderer:
    def __init__(self, is_headless: bool, window_size: Size | None):
        self.is_headless = is_headless
        self.window_size = window_size
        if is_headless and window_size is not None:
            logger.warning("window_size provided in headless mode, ignoring")

        if not is_headless:
            glfw.init()
            self.window = glfw.create_window(
                window_size[0], window_size[1], "ShaderBox", None, None
            )
            glfw.make_context_current(self.window)

        self.context = moderngl.create_context(standalone=is_headless)
        self.node_contexts: Dict[ShaderNode, Dict[str, Any]] = {}
        self.render_time = 0.0
        self.fps = 0.0
        self.frame_idx = 0

    def render_node(self, node: ShaderNode):
        if node not in self.node_contexts:
            program = self.context.program(
                vertex_shader="""
                #version 460
                in vec2 a_pos;
                out vec2 vs_uv;
                void main() {
                    gl_Position = vec4(a_pos, 0.0, 1.0);
                    vs_uv = a_pos * 0.5 + 0.5;
                }
                """,
                fragment_shader=node.fs_source,
            )
            texture = self.context.texture(node.output_size, 4)
            fbo = self.context.framebuffer(color_attachments=[texture])
            vbo = self.context.buffer(
                np.array(
                    [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
                    dtype="f4",
                )
            )
            vao = self.context.vertex_array(program, [(vbo, "2f", "a_pos")])
            self.node_contexts[node] = {
                "program": program,
                "texture": texture,
                "fbo": fbo,
                "vao": vao,
                "vbo": vbo,
            }

        ctx = self.node_contexts[node]
        for uniform in node.uniforms.values():
            if isinstance(uniform, ShaderNodeUniform):
                self.render_node(uniform.node)

        texture_unit = 0
        for name, uniform in node.uniforms.items():
            if isinstance(uniform, ShaderNodeUniform):
                value = self.node_contexts[uniform.node]["texture"]
            else:
                value = (
                    uniform.get_value(self)
                    if isinstance(uniform, UniformLike)
                    else uniform
                )
            if isinstance(value, moderngl.Texture):
                value.use(texture_unit)
                ctx["program"][name] = texture_unit
                texture_unit += 1
            else:
                ctx["program"][name] = value

        ctx["fbo"].use()
        ctx["vao"].render(moderngl.TRIANGLES)

    def cleanup(self):
        for ctx in self.node_contexts.values():
            ctx["vao"].release()
            ctx["fbo"].release()
            ctx["texture"].release()
            ctx["program"].release()
            ctx["vbo"].release()
        self.node_contexts.clear()

    def shutdown(self):
        self.cleanup()
        if not self.is_headless:
            glfw.terminate()

    def render_to_screen(self, node: ShaderNode, fps: float):
        if self.is_headless:
            logger.warning("Cannot render to screen in headless mode")
            return
        self.fps = fps
        imgui.create_context()
        imgui_impl = ImguiGlfwRenderer(self.window)
        target_frame_time = 1.0 / fps
        logger.info(f"Starting screen rendering loop at {fps} FPS")

        while not glfw.window_should_close(self.window):
            start_time = glfw.get_time()
            self.frame_idx += 1
            self.render_time = self.frame_idx / fps if fps > 0 else 0.0
            self.context.clear(0.0, 0.0, 0.0, 1.0)
            self.render_node(node)
            self.context.screen.use()
            self.context.copy_framebuffer(
                self.context.screen, self.node_contexts[node]["fbo"]
            )
            imgui.new_frame()
            imgui.begin("Test Window 1")
            imgui.text("Hello, ImGui!")
            imgui.end()
            imgui.begin("Test Window 2")
            imgui.text(f"Frame: {self.frame_idx}")
            imgui.text(f"Render time: {self.render_time:.2f}s")
            imgui.end()
            imgui.render()
            imgui_impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)
            elapsed_time = glfw.get_time() - start_time
            time.sleep(max(0.0, target_frame_time - elapsed_time))
            glfw.poll_events()
            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break
        self.shutdown()

    def render_image(self, node: ShaderNode, file_path: str):
        self.render_time = 0.0
        self.render_node(node)
        width, height = node.output_size
        data = self.node_contexts[node]["fbo"].read(
            viewport=(0, 0, width, height), components=3
        )
        image = np.frombuffer(data, np.uint8).reshape(height, width, 3)[::-1]
        logger.info(f"Saving image to {file_path}")
        imageio.imwrite(file_path, image)
        self.cleanup()

    def render_gif(self, node: ShaderNode, duration: float, fps: float, file_path: str):
        self.fps = fps
        n_frames = int(duration * fps)
        width, height = node.output_size
        frames = np.empty((n_frames, height, width, 3), dtype=np.uint8)
        for self.frame_idx in range(n_frames):
            self.render_time = self.frame_idx / fps if fps > 0 else 0.0
            self.render_node(node)
            data = self.node_contexts[node]["fbo"].read(
                viewport=(0, 0, width, height), components=3
            )
            frames[self.frame_idx] = np.frombuffer(data, np.uint8).reshape(
                height, width, 3
            )[::-1]
        logger.info(f"Saving GIF to {file_path}")
        imageio.mimwrite(file_path, frames, fps=fps)
        self.cleanup()

    def render_video(
        self, node: ShaderNode, duration: float, fps: float, file_path: str
    ):
        self.fps = fps
        n_frames = int(duration * fps)
        width, height = node.output_size
        frames = np.empty((n_frames, height, width, 3), dtype=np.uint8)
        for self.frame_idx in range(n_frames):
            self.render_time = self.frame_idx / fps if fps > 0 else 0.0
            self.render_node(node)
            data = self.node_contexts[node]["fbo"].read(
                viewport=(0, 0, width, height), components=3
            )
            frames[self.frame_idx] = np.frombuffer(data, np.uint8).reshape(
                height, width, 3
            )[::-1]
        logger.info(f"Saving video to {file_path}")
        imageio.mimwrite(file_path, frames, fps=fps)
        self.cleanup()


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

    parallax_node = ShaderNode(
        fs_source=parallax_fs,
        output_size=(800, 608),
        uniforms={
            "u_base_texture": LazyTexture("photo.jpeg"),
            "u_depth_map": LazyTexture("depth.png"),
            "u_texture_size": ValueUniform((800.0, 608.0)),
            "u_parallax_amount": ValueUniform(0.02),
            "u_focal_length": ValueUniform(1480.0),
            "u_time": DynamicUniform(lambda r: r.render_time),
        },
    )

    bright_pass_node = ShaderNode(
        fs_source=bright_pass_fs,
        output_size=(800, 608),
        uniforms={
            "u_source_texture": ShaderNodeUniform(parallax_node),
            "u_threshold": ValueUniform(0.5),
        },
    )

    downscale_node1 = ShaderNode(
        fs_source=downscale_fs,
        output_size=(400, 304),
        uniforms={"u_input_texture": ShaderNodeUniform(bright_pass_node)},
    )

    blur_node1 = ShaderNode(
        fs_source=blur_fs,
        output_size=(400, 304),
        uniforms={
            "u_blur_input": ShaderNodeUniform(downscale_node1),
            "u_pixel_size": ValueUniform((1.0 / 400, 1.0 / 304)),
        },
    )

    downscale_node2 = ShaderNode(
        fs_source=downscale_fs,
        output_size=(200, 152),
        uniforms={"u_input_texture": ShaderNodeUniform(blur_node1)},
    )

    blur_node2 = ShaderNode(
        fs_source=blur_fs,
        output_size=(200, 152),
        uniforms={
            "u_blur_input": ShaderNodeUniform(downscale_node2),
            "u_pixel_size": ValueUniform((1.0 / 200, 1.0 / 152)),
        },
    )

    outline_node = ShaderNode(
        fs_source=outline_fs,
        output_size=(800, 608),
        uniforms={
            "u_outline_source": ShaderNodeUniform(parallax_node),
            "u_pixel_size": ValueUniform((8.0 / 800, 8.0 / 608)),
            "u_outline_thickness": ValueUniform(8.0),
        },
    )

    combine_node = ShaderNode(
        fs_source=combine_fs,
        output_size=(800, 608),
        uniforms={
            "u_base_image": ShaderNodeUniform(parallax_node),
            "u_bloom_pass1": ShaderNodeUniform(blur_node1),
            "u_bloom_pass2": ShaderNodeUniform(blur_node2),
            "u_outline_pass": ShaderNodeUniform(outline_node),
        },
    )

    renderer_headless = Renderer(is_headless=True, window_size=None)
    renderer_headless.render_image(combine_node, "output.png")
    renderer_headless.render_gif(combine_node, 1.0, 15, "output.gif")
    renderer_headless.render_video(combine_node, 1.0, 15, "output.mp4")
    renderer_headless.shutdown()

    # renderer_windowed = Renderer(is_headless=False, window_size=(800, 600))
    # renderer_windowed.render_to_screen(combine_node, 60.0)
