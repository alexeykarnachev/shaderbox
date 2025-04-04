import glfw
import moderngl
import numpy as np
from PIL import Image


class ShaderNode:
    def __init__(self, ctx, vs_source, fs_source, vbo, inputs=None):
        self.program = ctx.program(vertex_shader=vs_source, fragment_shader=fs_source)
        self.inputs = inputs or []
        self.texture = ctx.texture((800, 600), 4)
        self.fbo = ctx.framebuffer(color_attachments=[self.texture])
        self.vao = ctx.vertex_array(self.program, [(vbo, "2f", "a_pos")])


class Program:
    def __init__(self, ctx, vs_source, vbo, nodes):
        """nodes: dict of {node_id: (fs_source, [input_node_ids])}"""
        self.ctx = ctx
        self.nodes = {}
        for node_id, (fs_source, input_ids) in nodes.items():
            inputs = [self.nodes[i] for i in input_ids if i in self.nodes]
            self.nodes[node_id] = ShaderNode(ctx, vs_source, fs_source, vbo, inputs)
        self.terminal_node = max(nodes.keys()) if nodes else None

    def execute(self, output_node_id=None, uniforms=None, target_fbo=None):
        if uniforms is None:
            uniforms = {}
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
            for name, value in uniforms.items():
                if name in node.program:
                    node.program[name].value = value
            fbo.use()
            node.vao.render(moderngl.TRIANGLES)
            rendered.add(node)

        render_node(output_node, target_fbo or output_node.fbo)
        return output_node


class Renderer:
    def __init__(self, width=800, height=600):
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self._window = glfw.create_window(width, height, "ShaderBox", None, None)
        glfw.make_context_current(self._window)

        self._ctx = moderngl.create_context()

        self._vs_source = """
        #version 460
        in vec2 a_pos;
        out vec2 vs_pos;
        void main() {
            gl_Position = vec4(a_pos, 0.0, 1.0);
            vs_pos = a_pos;
        }
        """
        self._quad = np.array(
            [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
            dtype="f4",
        )
        self._vbo = self._ctx.buffer(self._quad)

    def render_screen(self, program, output_node_id=None, uniforms=None):
        """Render the program to the screen."""
        if uniforms is None:
            uniforms = {}
        self._ctx.clear(0.0, 0.0, 0.0, 1.0)
        output_node = program.execute(
            output_node_id, uniforms, target_fbo=self._ctx.screen
        )
        if output_node:
            glfw.swap_buffers(self._window)
            while not glfw.window_should_close(self._window):
                glfw.poll_events()
                if glfw.get_key(self._window, glfw.KEY_ESCAPE) == glfw.PRESS:
                    break
        self.destroy()

    def render_image(self, program, output_node_id=None, uniforms=None):
        """Render the program to a Pillow Image."""
        if uniforms is None:
            uniforms = {}
        output_node = program.execute(output_node_id, uniforms)
        if output_node:
            raw_data = output_node.fbo.read(components=4)
            return Image.frombytes("RGBA", (800, 600), raw_data)
        return None

    def destroy(self):
        self._vbo.release()
        glfw.terminate()


if __name__ == "__main__":
    renderer = Renderer()
    nodes = {
        0: (
            """
            #version 460
            in vec2 vs_pos;
            out vec4 fs_color;
            void main() {
                fs_color = vec4(vs_pos.x * 0.5 + 0.5, vs_pos.y * 0.5 + 0.5, 0.0, 1.0);
            }
            """,
            [],
        ),
        1: (
            """
            #version 460
            in vec2 vs_pos;
            out vec4 fs_color;
            uniform sampler2D u_input0;
            void main() {
                vec4 color = texture(u_input0, vs_pos * 0.5 + 0.5);
                fs_color = vec4(color.r, color.g, 1.0, 1.0);
            }
            """,
            [0],
        ),
    }
    program = Program(renderer._ctx, renderer._vs_source, renderer._vbo, nodes)
    # Render to screen
    renderer.render_screen(program, 1)
    # Or render to image
    # img = renderer.render_image(program, 1)
    # img.save("output.png")
