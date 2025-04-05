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
        out vec2 vs_pos;
        void main() {
            gl_Position = vec4(a_pos, 0.0, 1.0);
            vs_pos = a_pos;
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
        """nodes: dict of {node_id: (fs_source, [input_ids], width, height)}
        external_inputs: dict of {uniform_name: texture_object}"""
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
        uniforms.update(self.external_inputs)  # Add external textures to uniforms
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
                node.program[f"u_input{i}"] = i
                input_node.texture.use(i)
            external_slot = len(node.inputs)
            for name, value in uniforms.items():
                if name in node.program:
                    if isinstance(value, moderngl.Texture):
                        node.program[name].value = external_slot
                        value.use(external_slot)
                        external_slot += 1
                    else:
                        node.program[name].value = value
            fbo.use()
            node.vao.render(moderngl.TRIANGLES)
            rendered.add(node)

        render_node(output_node, target_fbo or output_node.fbo)
        return output_node


class Renderer:
    def __init__(self, width=800, height=608):  # Changed to 608 (16x38)
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self._window = glfw.create_window(width, height, "ShaderBox", None, None)
        glfw.make_context_current(self._window)
        self._ctx = moderngl.create_context()
        self.width, self.height = width, height

        self._quad = np.array(
            [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
            dtype="f4",
        )
        self._vbo = self._ctx.buffer(self._quad)

    def _default_uniforms(self, time=0.0, flip_y=False):
        return {
            "u_time": time,
            "u_aspect": self.width / self.height,
            "u_flip_y": float(flip_y),
        }

    def load_texture(self, filename):
        img = Image.open(filename).convert("RGBA")
        texture = self._ctx.texture(img.size, 4, img.tobytes())
        print(f"Loaded {filename} with size {img.size}")
        return texture

    def render_frame(
        self, program, output_node_id=None, time=0.0, target_fbo=None, flip_y=False
    ):
        uniforms = self._default_uniforms(time, flip_y)
        return program.execute(output_node_id, uniforms, target_fbo)

    def render_image(
        self, program, output_node_id=None, time=0.0, filename="output.png"
    ):
        output_node = self.render_frame(program, output_node_id, time)
        if output_node:
            raw_data = output_node.fbo.read(components=4)
            img = Image.frombytes(
                "RGBA", (output_node.width, output_node.height), raw_data
            )
            # Removed transpose to keep PNG as it was (correct)
            img.save(filename)
            return img
        return None

    def render_animation(
        self, program, output_node_id=None, duration=1.0, fps=30, filename="output.mp4"
    ):
        frames = int(duration * fps)
        writer = (
            imageio.get_writer(filename, fps=fps, codec="libx264")
            if filename.endswith(".mp4")
            else imageio.get_writer(filename, fps=fps)
        )
        for i in range(frames):
            time = i / fps
            output_node = self.render_frame(program, output_node_id, time)
            if output_node:
                raw_data = output_node.fbo.read(components=4)
                frame = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                    output_node.height, output_node.width, 4
                )
                frame = frame[:, :, :3]  # Convert RGBA to RGB
                writer.append_data(frame)
        writer.close()

    def render_screen(self, program, output_node_id=None):
        start_time = glfw.get_time()
        while not glfw.window_should_close(self._window):
            time = glfw.get_time() - start_time
            self._ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.render_frame(
                program, output_node_id, time, self._ctx.screen, flip_y=True
            )
            glfw.swap_buffers(self._window)
            glfw.poll_events()
            if glfw.get_key(self._window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break
        self.destroy()

    def destroy(self):
        self._vbo.release()
        glfw.terminate()


if __name__ == "__main__":
    renderer = Renderer(width=800, height=608)  # Match new resolution

    # Load external image and depth map (replace with your own files)
    image_texture = renderer.load_texture("photo.jpeg")  # Example: a color photo
    depth_texture = renderer.load_texture("depth.png")  # Example: grayscale depth map

    # Declarative pipeline configuration
    nodes = {
        0: (
            """
            #version 460
            in vec2 vs_pos;
            out vec4 fs_color;
            uniform sampler2D u_texture;
            uniform sampler2D u_depth;
            uniform float u_time;
            uniform float u_aspect;
            void main() {
                vec2 uv = vs_pos * 0.5 + 0.5;
                float depth = texture(u_depth, uv).r;
                vec2 offset = vec2(sin(u_time) * 0.05 * depth, cos(u_time) * 0.05 * depth);
                vec4 color = texture(u_texture, uv + offset);
                fs_color = vec4(color.rgb, 1.0);
            }
            """,
            [],  # No internal node inputs, uses external textures
            800,  # Width
            608,  # Height (updated)
        ),
        1: (
            """
            #version 460
            in vec2 vs_pos;
            out vec4 fs_color;
            uniform sampler2D u_input0;
            uniform float u_time;
            uniform float u_aspect;
            void main() {
                vec2 uv = vs_pos * 0.5 + 0.5;
                vec4 color = texture(u_input0, uv);
                float bright = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
                if (bright > 0.5) {  // Lowered threshold for debug
                    fs_color = color;
                } else {
                    fs_color = vec4(0.0, 0.0, 0.0, 1.0);
                }
            }
            """,
            [0],  # Input from node 0 (parallax)
            400,  # Downscaled for bloom
            304,  # Half of 608 (updated)
        ),
        2: (
            """
            #version 460
            in vec2 vs_pos;
            out vec4 fs_color;
            uniform sampler2D u_input0;
            uniform sampler2D u_input1;
            uniform float u_time;
            uniform float u_aspect;
            uniform float u_flip_y;
            void main() {
                vec2 uv = vs_pos * 0.5 + 0.5;
                if (u_flip_y > 0.5) uv.y = 1.0 - uv.y;
                vec4 base = texture(u_input0, uv);
                vec4 bloom = texture(u_input1, uv);
                fs_color = vec4(base.rgb + bloom.rgb * 2.0, 1.0);  // Exaggerated bloom for debug
            }
            """,
            [0, 1],  # Inputs: original (0) and bloom (1)
            800,  # Back to full resolution
            608,  # Height (updated)
        ),
    }

    # External inputs for the pipeline
    external_inputs = {"u_texture": image_texture, "u_depth": depth_texture}

    # Create and run the program
    program = Program(renderer._ctx, renderer._vbo, nodes, external_inputs)

    # Examples
    renderer.render_image(program, 2, time=0.5, filename="output.png")  # Single frame
    renderer.render_animation(
        program, 2, duration=2.0, fps=30, filename="output.mp4"
    )  # MP4
    renderer.render_animation(
        program, 2, duration=2.0, fps=30, filename="output.gif"
    )  # GIF
    renderer.render_screen(program, 2)  # Real-time screen rendering
