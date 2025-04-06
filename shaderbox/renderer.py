import re

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


class DynamicUniform:
    def __init__(self, callback):
        self.callback = callback

    def get_value(self, renderer):
        return self.callback(renderer)


class ShaderNode:
    def __init__(self, ctx, fs_source, vbo, width, height, inputs=None):
        self._ctx = ctx
        self._width, self._height = width, height
        self._inputs = inputs or []
        self._uniforms = {}

        # Extract uniform names from fragment shader
        self._valid_uniforms = set()
        uniform_pattern = re.compile(r"uniform\s+\w+\s+(\w+)\s*[;=]")
        for match in uniform_pattern.finditer(fs_source):
            self._valid_uniforms.add(match.group(1))

        # Vertex shader
        self._vs_source = """
        #version 460
        in vec2 a_pos;
        out vec2 vs_uv;
        void main() {
            gl_Position = vec4(a_pos, 0.0, 1.0);
            vs_uv = a_pos * 0.5 + 0.5;
        }
        """
        self._program = ctx.program(
            vertex_shader=self._vs_source, fragment_shader=fs_source
        )
        self._texture = ctx.texture((width, height), 4)
        self._fbo = ctx.framebuffer(color_attachments=[self._texture])
        self._vao = ctx.vertex_array(self._program, [(vbo, "2f", "a_pos")])

    def __setitem__(self, name, value):
        """Set a uniform value (raw or DynamicUniform)."""
        if name not in self._valid_uniforms:
            raise ValueError(f"Uniform '{name}' is not defined in the shader.")
        self._uniforms[name] = value

    def render(self, renderer, uniforms=None):
        uniforms = uniforms or {}
        for input_node in self._inputs:
            input_node.render(renderer, uniforms)

        # Identify input uniforms
        input_uniforms = {
            name for name in self._valid_uniforms if name.startswith("u_input")
        }
        expected_input_count = len(input_uniforms)
        actual_input_count = len(self._inputs)

        # Check input uniform mismatches
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

        # Resolve all uniforms to their current values
        combined_uniforms = {}
        for name, value in {**self._uniforms, **uniforms}.items():
            if isinstance(value, DynamicUniform):
                combined_uniforms[name] = value.get_value(renderer)
            else:
                combined_uniforms[name] = value

        # Check for unset non-input uniforms
        non_input_uniforms = self._valid_uniforms - input_uniforms
        unset_uniforms = non_input_uniforms - set(combined_uniforms.keys())
        if unset_uniforms:
            logger.warning(
                f"Node has unset non-input uniforms: {unset_uniforms}. These will use default values."
            )

        # Set uniforms in the shader
        texture_unit = 0
        for name, value in combined_uniforms.items():
            if name in self._program:
                if isinstance(value, moderngl.Texture):
                    value.use(texture_unit)
                    self._program[name] = texture_unit
                    texture_unit += 1
                else:
                    self._program[name] = value

        # Bind inputs as textures
        for i, input_node in enumerate(self._inputs):
            if f"u_input{i}" in self._program:
                input_node._texture.use(i)
                self._program[f"u_input{i}"] = i

        self._fbo.use()
        self._ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._vao.render(moderngl.TRIANGLES)
        return self._fbo


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
        if headless:
            self._offscreen_fbo = self._ctx.framebuffer(
                color_attachments=[self._ctx.texture((width, height), 4)]
            )
        self._start_time = glfw.get_time() if not headless else 0
        self._frame_count = 0
        self._fps = 0.0

    def get_context(self):
        return self._ctx

    def get_vbo(self):
        return self._vbo

    def get_time(self):
        return (
            self._frame_count / self._fps
            if self._headless and self._fps > 0
            else glfw.get_time() - self._start_time
        )

    def get_frame_count(self):
        return self._frame_count

    def get_fps(self):
        return self._fps

    def render_to_screen(self, terminal_node):
        if self._headless:
            logger.warning("Cannot render to screen in headless mode.")
            return
        self._fps = 60.0
        logger.info("Starting screen rendering loop")
        while not glfw.window_should_close(self._window):
            self._frame_count += 1
            self._ctx.clear(0.0, 0.0, 0.0, 1.0)
            terminal_node.render(self)
            self._ctx.screen.use()
            self._ctx.copy_framebuffer(self._ctx.screen, terminal_node.render(self))
            glfw.swap_buffers(self._window)
            glfw.poll_events()
            if glfw.get_key(self._window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break
        self.destroy()

    def _render_frames(self, terminal_node, num_frames, fps):
        images = []
        self._fps = fps
        logger.info(f"Rendering {num_frames} frames at {fps} FPS")
        for frame in range(num_frames):
            self._frame_count = frame
            fbo = terminal_node.render(self)
            self._offscreen_fbo.use()
            self._ctx.copy_framebuffer(self._offscreen_fbo, fbo)
            width, height = (
                self._offscreen_fbo.viewport[2],
                self._offscreen_fbo.viewport[3],
            )
            image_data = self._offscreen_fbo.read(
                viewport=(0, 0, width, height), components=4
            )
            image = np.frombuffer(image_data, dtype=np.uint8).reshape(height, width, 4)
            image = image[::-1, :, :]
            images.append(image)
        return images

    def render_gif(self, terminal_node, duration, fps, filename):
        num_frames = int(duration * fps)
        images = self._render_frames(terminal_node, num_frames, fps)
        rgb_images = [image[:, :, :3] for image in images]
        logger.info(f"Saving GIF to {filename}")
        imageio.mimwrite(filename, rgb_images, fps=fps)

    def render_image(self, terminal_node, filename):
        images = self._render_frames(terminal_node, 1, 30.0)
        logger.info(f"Saving image to {filename}")
        imageio.imwrite(filename, images[0])

    def render_video(self, terminal_node, duration, fps, filename):
        num_frames = int(duration * fps)
        images = self._render_frames(terminal_node, num_frames, fps)
        rgb_images = [image[:, :, :3] for image in images]
        logger.info(f"Saving video to {filename}")
        imageio.mimwrite(filename, rgb_images, fps=fps)

    def destroy(self):
        self._vbo.release()
        if self._headless and hasattr(self, "_offscreen_fbo"):
            self._offscreen_fbo.release()
        if not self._headless:
            glfw.terminate()
        logger.info("Renderer resources cleaned up")


if __name__ == "__main__":
    renderer = Renderer(width=800, height=608, headless=True)
    image_texture = load_texture(renderer.get_context(), "photo.jpeg")
    depth_texture = load_texture(renderer.get_context(), "depth.png")

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

    # Create nodes with static and dynamic uniforms
    node0 = ShaderNode(
        renderer.get_context(), parallax_fs, renderer.get_vbo(), 800, 608
    )
    node0["u_texture"] = image_texture
    node0["u_depth"] = depth_texture
    node0["u_texture_size"] = (800.0, 608.0)
    node0["u_parallax_strength"] = 0.02
    node0["u_focal_px"] = 1480.0
    node0["u_time"] = DynamicUniform(lambda r: r.get_time())  # Dynamic uniform

    node1 = ShaderNode(
        renderer.get_context(),
        bright_pass_fs,
        renderer.get_vbo(),
        800,
        608,
        inputs=[node0],
    )
    node1["u_threshold"] = 0.5

    node2 = ShaderNode(
        renderer.get_context(),
        downscale_fs,
        renderer.get_vbo(),
        400,
        304,
        inputs=[node1],
    )
    node3 = ShaderNode(
        renderer.get_context(), blur_fs, renderer.get_vbo(), 400, 304, inputs=[node2]
    )
    node3["u_pixel_size"] = (1.0 / 400, 1.0 / 304)

    node4 = ShaderNode(
        renderer.get_context(),
        downscale_fs,
        renderer.get_vbo(),
        200,
        152,
        inputs=[node3],
    )
    node5 = ShaderNode(
        renderer.get_context(), blur_fs, renderer.get_vbo(), 200, 152, inputs=[node4]
    )
    node5["u_pixel_size"] = (1.0 / 200, 1.0 / 152)

    node6 = ShaderNode(
        renderer.get_context(), outline_fs, renderer.get_vbo(), 800, 608, inputs=[node0]
    )
    node6["u_pixel_size"] = (8.0 / 800, 8.0 / 608)
    node6["u_outline_thickness"] = 8.0

    node7 = ShaderNode(
        renderer.get_context(),
        combine_fs,
        renderer.get_vbo(),
        800,
        608,
        inputs=[node0, node3, node5, node6],
    )

    # Render outputs (no uniforms_callback needed)
    renderer.render_gif(node7, 1.0, 30, "output.gif")
    renderer.render_video(node7, 5.0, 30, "output.mp4")
    renderer.render_image(node7, "output.png")

    renderer.destroy()
