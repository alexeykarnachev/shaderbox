import re
from dataclasses import dataclass

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


def create_context(width, height, headless=False):
    if not headless:
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        window = glfw.create_window(width, height, "ShaderBox", None, None)
        glfw.make_context_current(window)
    ctx = moderngl.create_context(standalone=headless, require=460)
    logger.info(f"Created ModernGL context (headless={headless})")
    return ctx


def load_texture(ctx, filename):
    try:
        img = Image.open(filename).convert("RGBA")
        texture = ctx.texture(img.size, 4, img.tobytes())
        logger.info(f"Loaded texture {filename} with size {img.size}")
        return texture
    except Exception as e:
        logger.error(f"Failed to load texture {filename}: {e}")
        raise


class Uniform:
    def __init__(self, value):
        self.callback = value if callable(value) else lambda _: value

    def get_value(self, renderer):
        return self.callback(renderer)


@dataclass
class GLContext:
    """Holds all OpenGL-related resources for a ShaderNode."""

    ctx: moderngl.Context
    program: moderngl.Program
    texture: moderngl.Texture
    fbo: moderngl.Framebuffer
    vao: moderngl.VertexArray


class ShaderNode:
    def __init__(self, fs_source, width, height, inputs=None, **uniforms):
        self.fs_source = fs_source
        self.width = width
        self.height = height
        self.inputs = inputs or []
        self.uniforms = {}

        # Pre-extract uniform names from fragment shader
        self.valid_uniforms = set()
        uniform_pattern = re.compile(r"uniform\s+\w+\s+(\w+)\s*[;=]")
        for match in uniform_pattern.finditer(fs_source):
            self.valid_uniforms.add(match.group(1))

        # Set initial uniforms, including textures
        for name, value in uniforms.items():
            self[name] = value

        # Lazy initialization placeholder
        self._gl_context: GLContext | None = None

    def __setitem__(self, name, value):
        if name not in self.valid_uniforms:
            raise ValueError(f"Uniform '{name}' is not defined in the shader.")
        self.uniforms[name] = Uniform(value)

    def _initialize(self, renderer):
        """Initialize OpenGL resources if not already done."""
        if self._gl_context is not None:
            return

        ctx = renderer.context
        vs_source = """
        #version 460
        in vec2 a_pos;
        out vec2 vs_uv;
        void main() {
            gl_Position = vec4(a_pos, 0.0, 1.0);
            vs_uv = a_pos * 0.5 + 0.5;
        }
        """
        program = ctx.program(vertex_shader=vs_source, fragment_shader=self.fs_source)
        texture = ctx.texture((self.width, self.height), 4)
        fbo = ctx.framebuffer(color_attachments=[texture])
        vao = ctx.vertex_array(program, [(renderer.vbo, "2f", "a_pos")])
        self._gl_context = GLContext(ctx, program, texture, fbo, vao)
        logger.debug(f"Initialized ShaderNode with shader: {self.fs_source[:50]}...")

    def render(self, renderer):
        """Render the node, initializing if necessary."""
        self._initialize(renderer)
        assert self._gl_context is not None  # For type checkers

        # Render inputs first
        for input_node in self.inputs:
            input_node.render(renderer)

        # Check input uniform mismatches
        input_uniforms = {
            name for name in self.valid_uniforms if name.startswith("u_input")
        }
        expected_input_count = len(input_uniforms)
        actual_input_count = len(self.inputs)
        if expected_input_count > actual_input_count:
            missing_inputs = {
                f"u_input{i}" for i in range(actual_input_count, expected_input_count)
            }
            logger.warning(
                f"Node expects {expected_input_count} input textures but only {actual_input_count} provided. "
                f"Missing uniforms: {missing_inputs} will use default textures."
            )
        elif actual_input_count > expected_input_count:
            logger.warning(
                f"Node has {actual_input_count} input nodes but only {expected_input_count} input uniforms defined. "
                f"Excess inputs will be ignored."
            )

        # Resolve uniforms
        combined_uniforms = {
            name: uniform.get_value(renderer) for name, uniform in self.uniforms.items()
        }

        # Check for unset non-input uniforms
        non_input_uniforms = self.valid_uniforms - input_uniforms
        unset_uniforms = non_input_uniforms - set(combined_uniforms.keys())
        if unset_uniforms:
            logger.warning(
                f"Node has unset non-input uniforms: {unset_uniforms}. These will use default values."
            )

        # Set uniforms in the shader
        texture_unit = 0
        for name, value in combined_uniforms.items():
            if name in self._gl_context.program:
                if isinstance(value, moderngl.Texture):
                    value.use(texture_unit)
                    self._gl_context.program[name] = texture_unit
                    texture_unit += 1
                else:
                    self._gl_context.program[name] = value

        # Bind inputs as textures
        for i, input_node in enumerate(self.inputs):
            if f"u_input{i}" in self._gl_context.program:
                input_node._gl_context.texture.use(i)
                self._gl_context.program[f"u_input{i}"] = i

        self._gl_context.fbo.use()
        self._gl_context.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._gl_context.vao.render(moderngl.TRIANGLES)
        return self._gl_context.fbo


class Renderer:
    def __init__(self, width=800, height=608, headless=False):
        self._width, self._height = width, height
        self._headless = headless
        self._ctx = create_context(width, height, headless)
        self._window = None if headless else glfw.get_current_context()
        quad = np.array(
            [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
            dtype="f4",
        )
        self._vbo = self._ctx.buffer(quad)
        # Unified FBO, always created for offscreen rendering
        self._fbo = self._ctx.framebuffer(
            color_attachments=[self._ctx.texture((width, height), 4)]
        )
        self._start_time = glfw.get_time() if not headless else 0
        self._frame_count = 0
        self._fps = 0.0

    @property
    def context(self):
        return self._ctx

    @property
    def vbo(self):
        return self._vbo

    @property
    def time(self):
        return (
            self._frame_count / self._fps
            if self._headless and self._fps > 0
            else glfw.get_time() - self._start_time
        )

    @property
    def frame_count(self):
        return self._frame_count

    @property
    def fps(self):
        return self._fps

    def render_to_screen(self, node):
        if self._headless:
            logger.warning("Cannot render to screen in headless mode.")
            return
        self._fps = 60.0
        logger.info("Starting screen rendering loop")
        while not glfw.window_should_close(self._window):
            self._frame_count += 1
            self._ctx.clear(0.0, 0.0, 0.0, 1.0)
            node.render(self)
            self._ctx.screen.use()
            self._ctx.copy_framebuffer(self._ctx.screen, node.render(self))
            glfw.swap_buffers(self._window)
            glfw.poll_events()
            if glfw.get_key(self._window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break
        self.destroy()

    def _render_frames(self, node, num_frames, fps):
        if not self._headless:
            logger.warning(
                "Rendering offscreen (GIF/video/image) in non-headless mode; output may not match screen."
            )
        images = []
        self._fps = fps
        logger.info(f"Rendering {num_frames} frames at {fps} FPS")
        for frame in range(num_frames):
            self._frame_count = frame
            fbo = node.render(self)
            self._fbo.use()
            self._ctx.copy_framebuffer(self._fbo, fbo)
            width, height = self._fbo.viewport[2], self._fbo.viewport[3]
            image_data = self._fbo.read(viewport=(0, 0, width, height), components=4)
            image = np.frombuffer(image_data, dtype=np.uint8).reshape(height, width, 4)
            image = image[::-1, :, :]
            images.append(image)
        return images

    def render_gif(self, node, duration, fps, filename):
        num_frames = int(duration * fps)
        images = self._render_frames(node, num_frames, fps)
        rgb_images = [image[:, :, :3] for image in images]
        logger.info(f"Saving GIF to {filename}")
        imageio.mimwrite(filename, rgb_images, fps=fps)

    def render_image(self, node, filename):
        images = self._render_frames(node, 1, 30.0)
        logger.info(f"Saving image to {filename}")
        imageio.imwrite(filename, images[0])

    def render_video(self, node, duration, fps, filename):
        num_frames = int(duration * fps)
        images = self._render_frames(node, num_frames, fps)
        rgb_images = [image[:, :, :3] for image in images]
        logger.info(f"Saving video to {filename}")
        imageio.mimwrite(filename, rgb_images, fps=fps)

    def destroy(self):
        self._vbo.release()
        self._fbo.release()
        if not self._headless:
            glfw.terminate()
        logger.info("Renderer resources cleaned up")


if __name__ == "__main__":
    # Shader sources (unchanged)
    parallax_fs = """
    #version 460
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform sampler2D u_texture;
    uniform sampler2D u_depth;
    uniform float u_time;
    uniform vec2 u_texture_size;
    uniform float u_focal_px = 1480.0;
    uniform float u_parallax_strength = 0.05;
    void main() {
        vec2 uv = vs_uv;
        uv.y = 1.0 - uv.y;
        float depth = texture(u_depth, uv).r;
        vec2 camera_move = vec2(sin(u_time), cos(u_time)) * u_focal_px * u_parallax_strength;
        vec2 offset = -camera_move * depth / u_texture_size;
        uv += offset;
        uv = clamp(uv, 0.0, 1.0);
        vec4 color = texture(u_texture, uv);
        fs_color = vec4(color.rgb, 1.0);
    }
    """
    bright_pass_fs = """
    #version 460
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform sampler2D u_input0;
    uniform float u_threshold = 0.5;
    void main() {
        vec4 color = texture(u_input0, vs_uv);
        float brightness = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
        fs_color = brightness > u_threshold ? color : vec4(0.0, 0.0, 0.0, 0.0);
    }
    """
    downscale_fs = """
    #version 460
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform sampler2D u_input0;
    void main() {
        fs_color = texture(u_input0, vs_uv);
    }
    """
    blur_fs = """
    #version 460
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform sampler2D u_input0;
    uniform vec2 u_pixel_size;
    void main() {
        vec4 color = vec4(0.0);
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                color += texture(u_input0, vs_uv + vec2(float(i), float(j)) * u_pixel_size);
        fs_color = color / 9.0;
    }
    """
    outline_fs = """
    #version 460
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform sampler2D u_input0;
    uniform vec2 u_pixel_size;
    uniform float u_outline_thickness = 1.0;
    void main() {
        vec4 center = texture(u_input0, vs_uv);
        float edge = 0.0;
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0) continue;
                vec4 neighbor = texture(u_input0, vs_uv + vec2(float(i), float(j)) * u_pixel_size);
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
    uniform sampler2D u_input0;  // Parallax (base)
    uniform sampler2D u_input1;  // Bloom 1
    uniform sampler2D u_input2;  // Bloom 2
    uniform sampler2D u_input3;  // Outline
    void main() {
        vec4 base = texture(u_input0, vs_uv);
        vec4 bloom1 = texture(u_input1, vs_uv);
        vec4 bloom2 = texture(u_input2, vs_uv);
        vec4 outline = texture(u_input3, vs_uv);
        vec3 color = base.rgb + bloom1.rgb + bloom2.rgb;
        color = outline.r * color * vec3(1.0, 0.8, 0.7) + 1.0 * color;
        fs_color = vec4(color, 1.0);
    }
    """

    # Load textures first
    renderer = Renderer(width=800, height=608, headless=False)
    image_texture = load_texture(renderer.context, "photo.jpeg")
    depth_texture = load_texture(renderer.context, "depth.png")

    # Create nodes with textures at initialization
    node0 = ShaderNode(
        parallax_fs,
        800,
        608,
        u_texture=image_texture,
        u_depth=depth_texture,
        u_texture_size=(800.0, 608.0),
        u_parallax_strength=0.02,
        u_focal_px=1480.0,
        u_time=lambda r: r.time,
    )

    node1 = ShaderNode(bright_pass_fs, 800, 608, inputs=[node0], u_threshold=0.5)

    node2 = ShaderNode(downscale_fs, 400, 304, inputs=[node1])

    node3 = ShaderNode(
        blur_fs, 400, 304, inputs=[node2], u_pixel_size=(1.0 / 400, 1.0 / 304)
    )

    node4 = ShaderNode(downscale_fs, 200, 152, inputs=[node3])

    node5 = ShaderNode(
        blur_fs, 200, 152, inputs=[node4], u_pixel_size=(1.0 / 200, 1.0 / 152)
    )

    node6 = ShaderNode(
        outline_fs,
        800,
        608,
        inputs=[node0],
        u_pixel_size=(8.0 / 800, 8.0 / 608),
        u_outline_thickness=8.0,
    )

    node7 = ShaderNode(combine_fs, 800, 608, inputs=[node0, node3, node5, node6])

    # Render outputs
    renderer.render_gif(node7, 1.0, 30, "output.gif")
    renderer.render_video(node7, 5.0, 30, "output.mp4")
    renderer.render_image(node7, "output.png")
    renderer.render_to_screen(node7)

    renderer.destroy()
