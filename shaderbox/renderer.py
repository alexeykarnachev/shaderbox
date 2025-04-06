import re

import glfw
import imageio
import moderngl
import numpy as np
from PIL import Image


def create_context(width, height, headless=False):
    """Utility function to create a ModernGL context."""
    if not headless:
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        window = glfw.create_window(width, height, "ShaderBox", None, None)
        glfw.make_context_current(window)
    ctx = moderngl.create_context(standalone=headless, require=460)
    return ctx


def load_texture(ctx, filename):
    """Utility function to load an image as a texture."""
    img = Image.open(filename).convert("RGBA")
    texture = ctx.texture(img.size, 4, img.tobytes())
    print(f"Loaded {filename} with size {img.size}")
    return texture


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

        # Vertex shader (same for all nodes)
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

        # Set up rendering target
        self._texture = ctx.texture((width, height), 4)
        self._fbo = ctx.framebuffer(color_attachments=[self._texture])
        self._vao = ctx.vertex_array(self._program, [(vbo, "2f", "a_pos")])

    def __setitem__(self, name, value):
        """Set a uniform value using node['u_name'] = value syntax."""
        if name not in self._valid_uniforms:
            raise ValueError(f"Uniform '{name}' is not defined in the shader.")
        self._uniforms[name] = value

    def render(self, uniforms=None):
        """Render this node and its dependencies."""
        uniforms = uniforms or {}
        # Render all input nodes first
        for input_node in self._inputs:
            input_node.render(uniforms)

        # Merge node's uniforms with provided uniforms (provided take precedence)
        combined_uniforms = {**self._uniforms, **uniforms}

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

        # Render to this node's FBO
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

        # Full-screen quad VBO
        quad = np.array(
            [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
            dtype="f4",
        )
        self._vbo = self._ctx.buffer(quad)

        # Offscreen FBO for headless rendering
        if headless:
            self._offscreen_fbo = self._ctx.framebuffer(
                color_attachments=[self._ctx.texture((width, height), 4)]
            )

        # Runtime tracking
        self._start_time = glfw.get_time() if not headless else 0
        self._frame_count = 0
        self._fps = 0.0  # Will be set by render methods

    def get_context(self):
        """Get the rendering context."""
        return self._ctx

    def get_vbo(self):
        """Get the vertex buffer object."""
        return self._vbo

    def get_time(self):
        """Get current time based on frame count and FPS."""
        if self._headless:
            return self._frame_count / self._fps if self._fps > 0 else 0.0
        return glfw.get_time() - self._start_time

    def get_frame_count(self):
        """Get the current frame count."""
        return self._frame_count

    def get_fps(self):
        """Get the fixed FPS value."""
        return self._fps

    def render_to_screen(self, terminal_node, uniforms_callback=None):
        """Render to screen in a loop (non-headless only)."""
        if self._headless:
            print("Cannot render to screen in headless mode.")
            return
        self._fps = 60.0  # Default FPS for screen rendering
        while not glfw.window_should_close(self._window):
            self._frame_count += 1
            uniforms = uniforms_callback(self) if uniforms_callback else {}
            self._ctx.clear(0.0, 0.0, 0.0, 1.0)
            terminal_node.render(uniforms)
            self._ctx.screen.use()
            self._ctx.copy_framebuffer(self._ctx.screen, terminal_node.render(uniforms))
            glfw.swap_buffers(self._window)
            glfw.poll_events()
            if glfw.get_key(self._window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break
        self.destroy()

    def _render_frames(self, terminal_node, uniforms_callback, num_frames, fps):
        """Render a sequence of frames to memory."""
        images = []
        self._fps = fps
        for frame in range(num_frames):
            self._frame_count = frame  # Reset to frame index
            uniforms = uniforms_callback(self) if uniforms_callback else {}
            fbo = terminal_node.render(uniforms)
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
            image = image[::-1, :, :]  # Flip vertically
            images.append(image)
        return images

    def render_gif(self, terminal_node, uniforms_callback, duration, fps, filename):
        """Render to a GIF file."""
        num_frames = int(duration * fps)
        images = self._render_frames(terminal_node, uniforms_callback, num_frames, fps)
        rgb_images = [image[:, :, :3] for image in images]
        imageio.mimwrite(filename, rgb_images, fps=fps)

    def render_image(self, terminal_node, uniforms_callback, filename):
        """Render to a single image file."""
        images = self._render_frames(
            terminal_node, uniforms_callback, 1, 30.0
        )  # Default FPS for single frame
        imageio.imwrite(filename, images[0])

    def render_video(self, terminal_node, uniforms_callback, duration, fps, filename):
        """Render to a video file."""
        num_frames = int(duration * fps)
        images = self._render_frames(terminal_node, uniforms_callback, num_frames, fps)
        rgb_images = [image[:, :, :3] for image in images]
        imageio.mimwrite(filename, rgb_images, fps=fps)

    def destroy(self):
        """Clean up resources."""
        self._vbo.release()
        if self._headless and hasattr(self, "_offscreen_fbo"):
            self._offscreen_fbo.release()
        if not self._headless:
            glfw.terminate()


if __name__ == "__main__":
    # Initialize renderer
    renderer = Renderer(width=800, height=608, headless=True)

    # Load textures
    image_texture = load_texture(renderer.get_context(), "photo.jpeg")
    depth_texture = load_texture(renderer.get_context(), "depth.png")

    # Shader sources
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

    # Create nodes
    node0 = ShaderNode(
        renderer.get_context(), parallax_fs, renderer.get_vbo(), 800, 608
    )
    node0["u_texture"] = image_texture
    node0["u_depth"] = depth_texture
    node0["u_texture_size"] = (800.0, 608.0)
    node0["u_parallax_strength"] = 0.02

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

    # Uniforms callback using renderer
    def uniforms_callback(renderer):
        time = renderer.get_time()
        return {
            "u_time": time,
            "u_focal_px": 1480.0,
            # Example: Toggle based on frame count
            # "u_brightness": 1.0 if renderer.get_frame_count() % 60 < 30 else 0.5
        }

    # Render outputs
    renderer.render_gif(node7, uniforms_callback, 5.0, 30, "output.gif")
    renderer.render_video(node7, uniforms_callback, 5.0, 30, "output.mp4")
    renderer.render_image(node7, uniforms_callback, "output.png")

    renderer.destroy()
