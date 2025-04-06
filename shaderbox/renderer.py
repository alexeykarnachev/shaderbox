import os
import subprocess
import time

import glfw
import imageio
import moderngl
import numpy as np
from PIL import Image


class ShaderNode:
    def __init__(self, ctx, fs_source, vbo, width, height, inputs=None):
        self._vs_source = """
        #version 460
        in vec2 a_pos;
        out vec2 vs_uv;
        void main() {
            gl_Position = vec4(a_pos, 0.0, 1.0);
            vs_uv = a_pos * 0.5 + 0.5;
        }
        """
        self.program = ctx.program(
            vertex_shader=self._vs_source, fragment_shader=fs_source
        )
        self.inputs = inputs or []
        self.texture = ctx.texture((width, height), 4)
        self.fbo = ctx.framebuffer(color_attachments=[self.texture])
        self.vao = ctx.vertex_array(self.program, [(vbo, "2f", "a_pos")])
        self.width, self.height = width, height


class Program:
    def __init__(self, ctx, vbo, nodes, external_inputs=None):
        self.ctx = ctx
        self.external_inputs = external_inputs or {}
        self.nodes = {}
        for node_id, (fs_source, input_ids, width, height) in nodes.items():
            inputs = [self.nodes[i] for i in input_ids if i in self.nodes]
            self.nodes[node_id] = ShaderNode(ctx, fs_source, vbo, width, height, inputs)
        self.terminal_node = max(nodes.keys()) if nodes else None

    def execute(self, output_node_id=None, uniforms=None, target_fbo=None):
        if uniforms is None:
            uniforms = {}
        uniforms.update(self.external_inputs)
        output_node = self.nodes.get(output_node_id, self.nodes.get(self.terminal_node))
        if not output_node:
            return None

        rendered = set()

        def render_node(node, fbo):
            if node in rendered:
                return
            for input_node in node.inputs:
                render_node(input_node, input_node.fbo)
            for i, input_node in enumerate(node.inputs):
                if f"u_input{i}" in node.program:
                    node.program[f"u_input{i}"] = i
                    input_node.texture.use(i)
            external_slot = len(node.inputs)
            max_slots = self.ctx.max_texture_units
            for name, value in uniforms.items():
                if name in node.program:
                    if isinstance(value, moderngl.Texture):
                        if external_slot >= max_slots:
                            raise ValueError(
                                f"Exceeded max texture units ({max_slots})"
                            )
                        node.program[name] = external_slot
                        value.use(external_slot)
                        external_slot += 1
                    else:
                        node.program[name] = value
            fbo.use()
            node.vao.render(moderngl.TRIANGLES)
            rendered.add(node)

        render_node(output_node, target_fbo or output_node.fbo)
        return output_node


class Renderer:
    def __init__(self, width=800, height=608, headless=False):
        self.headless = headless
        self.width, self.height = width, height
        self._xvfb_process = None

        if self.headless:
            try:
                self._ctx = moderngl.create_context(standalone=True, backend="egl")
                print(
                    f"Created headless EGL context. Version: {self._ctx.version_code}"
                )
            except Exception as e:
                print(f"EGL failed: {e}. Falling back to Xvfb.")
                os.environ["DISPLAY"] = ":99.0"
                self._xvfb_process = subprocess.Popen(
                    ["Xvfb", ":99", "-screen", "0", f"{width}x{height}x24"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                time.sleep(1)
                self._ctx = moderngl.create_context(standalone=True)
                print(
                    f"Created headless Xvfb context. Version: {self._ctx.version_code}"
                )
        else:
            glfw.init()
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            self._window = glfw.create_window(width, height, "ShaderBox", None, None)
            glfw.make_context_current(self._window)
            self._ctx = moderngl.create_context()

        self._quad = np.array(
            [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
            dtype="f4",
        )
        self._vbo = self._ctx.buffer(self._quad)
        if self.headless:
            self._offscreen_fbo = self._ctx.framebuffer(
                color_attachments=[self._ctx.texture((self.width, self.height), 4)]
            )

    def load_texture(self, filename):
        img = Image.open(filename).convert("RGBA")
        texture = self._ctx.texture(img.size, 4, img.tobytes())
        print(f"Loaded {filename} with size {img.size}")
        return texture

    def render_frame(
        self, program, output_node_id=None, uniforms=None, target_fbo=None
    ):
        target_fbo = target_fbo or (
            self._offscreen_fbo if self.headless else self._ctx.screen
        )
        if target_fbo:
            target_fbo.use()
            self._ctx.clear(0.0, 0.0, 0.0, 1.0)  # Clear to black
        return program.execute(output_node_id, uniforms, target_fbo)

    def render_screen(self, program, output_node_id=None, uniforms_callback=None):
        if self.headless:
            print("Cannot render to screen in headless mode.")
            return
        start_time = glfw.get_time()
        while not glfw.window_should_close(self._window):
            time = glfw.get_time() - start_time
            uniforms = uniforms_callback(time) if uniforms_callback else None
            self._ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.render_frame(program, output_node_id, uniforms, self._ctx.screen)
            glfw.swap_buffers(self._window)
            glfw.poll_events()
            if glfw.get_key(self._window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break
        self.destroy()

    def _render_frames(self, program, output_node_id, uniforms_callback, times):
        images = []
        for t in times:
            uniforms = uniforms_callback(t) if uniforms_callback else None
            self.render_frame(program, output_node_id, uniforms, self._offscreen_fbo)
            width, height = (
                self._offscreen_fbo.viewport[2],
                self._offscreen_fbo.viewport[3],
            )
            image_data = self._offscreen_fbo.read(
                viewport=(0, 0, width, height), components=4
            )
            image = np.frombuffer(image_data, dtype=np.uint8).reshape(height, width, 4)
            image = image[::-1, :, :]  # Flip vertically if needed
            images.append(image)
        return images

    def render_gif(
        self, program, output_node_id, uniforms_callback, duration, fps, filename
    ):
        num_frames = int(duration * fps)
        times = [i / fps for i in range(num_frames)]
        images = self._render_frames(program, output_node_id, uniforms_callback, times)
        rgb_images = [image[:, :, :3] for image in images]
        imageio.mimwrite(filename, rgb_images, fps=fps)

    def render_image(self, program, output_node_id, uniforms_callback, time, filename):
        images = self._render_frames(program, output_node_id, uniforms_callback, [time])
        imageio.imwrite(filename, images[0])

    def render_video(
        self, program, output_node_id, uniforms_callback, duration, fps, filename
    ):
        num_frames = int(duration * fps)
        times = [i / fps for i in range(num_frames)]
        images = self._render_frames(program, output_node_id, uniforms_callback, times)
        rgb_images = [image[:, :, :3] for image in images]
        imageio.mimwrite(filename, rgb_images, fps=fps)

    def destroy(self):
        self._vbo.release()
        if self.headless and hasattr(self, "_offscreen_fbo"):
            self._offscreen_fbo.release()
        if self._xvfb_process:
            self._xvfb_process.terminate()
        if not self.headless:
            glfw.terminate()


if __name__ == "__main__":
    renderer = Renderer(width=800, height=608, headless=True)  # Enable headless mode

    image_texture = renderer.load_texture("photo.jpeg")
    depth_texture = renderer.load_texture("depth.png")

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
        uv = clamp(uv, 0.0, 1.0);  // Clamp UVs instead of discard
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
        fs_color = brightness > u_threshold ? color : vec4(0.0, 0.0, 0.0, 0.0);  // Explicit alpha
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

    nodes = {
        0: (parallax_fs, [], renderer.width, renderer.height),
        1: (bright_pass_fs, [0], renderer.width, renderer.height),
        2: (downscale_fs, [1], renderer.width // 2, renderer.height // 2),
        3: (blur_fs, [2], renderer.width // 2, renderer.height // 2),
        4: (downscale_fs, [3], renderer.width // 4, renderer.height // 4),
        5: (blur_fs, [4], renderer.width // 4, renderer.height // 4),
        6: (outline_fs, [0], renderer.width, renderer.height),
        7: (combine_fs, [0, 3, 5, 6], renderer.width, renderer.height),
    }

    external_inputs = {"u_texture": image_texture, "u_depth": depth_texture}

    def uniforms_callback(time):
        return {
            "u_time": time,
            "u_texture_size": (float(renderer.width), float(renderer.height)),
            "u_focal_px": 1480.0,
            "u_parallax_strength": 0.02,
            "u_threshold": 0.5,
            "u_pixel_size": (8.0 / renderer.width, 8.0 / renderer.height),
            "u_outline_thickness": 8.0,
        }

    program = Program(renderer._ctx, renderer._vbo, nodes, external_inputs)
    # Set per-node pixel sizes for blur nodes
    program.nodes[3].program["u_pixel_size"] = (
        1.0 / (renderer.width // 2),
        1.0 / (renderer.height // 2),
    )
    program.nodes[5].program["u_pixel_size"] = (
        1.0 / (renderer.width // 4),
        1.0 / (renderer.height // 4),
    )

    # Uncomment to test screen rendering (non-headless mode only)
    # renderer.render_screen(program, 7, uniforms_callback)

    renderer.render_gif(program, 7, uniforms_callback, 5.0, 30, "output.gif")
    renderer.render_video(program, 7, uniforms_callback, 5.0, 30, "output.mp4")
    renderer.render_image(program, 7, uniforms_callback, 8.0, "output.png")

    renderer.destroy()
