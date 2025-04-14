import hashlib
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import moderngl
import numpy as np
from loguru import logger
from OpenGL.GL import GL_DOUBLE, GL_FLOAT, GL_INT, GL_UNSIGNED_INT

from shaderbox.gl import GLContext

NodeOutput = tuple[int, int] | str | tuple[str, float]


class FileSource:
    def __init__(self, source: str | Path) -> None:
        self._text: str = ""
        self._hash: str = ""
        self._path = Path(source) if FileSource._is_file_path(source) else None

    def reload_if_needed(self) -> bool:
        is_reloaded = False

        if self._path is None:
            return is_reloaded

        text = self._path.read_text()
        hash = FileSource._get_content_hash(text)
        if hash != self._hash:
            self._hash = hash
            self._text = text
            is_reloaded = True

        return is_reloaded

    @staticmethod
    def _is_file_path(source: str | Path) -> bool:
        return "\n" not in str(source).strip()

    @staticmethod
    def _get_content_hash(content: str | bytes) -> str:
        b = content.encode() if isinstance(content, str) else content
        return hashlib.sha256(b).hexdigest()


class NodeResources:
    _VERTEX_SHADER = """
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
        gl_context: GLContext,
        fs_source: str | Path,
        texture_size: tuple[int, int],
    ) -> None:
        self._gl_context: GLContext = gl_context
        self._fs_source: FileSource = FileSource(fs_source)
        self._texture_size: tuple[int, int] = texture_size

        self._error: str = ""

        self._program: moderngl.Program | None = None
        self._texture: moderngl.Texture | None = None
        self._fbo: moderngl.Framebuffer | None = None
        self._vbo: moderngl.Buffer | None = None
        self._vao: moderngl.VertexArray | None = None

    def release(self):
        if self._program is not None:
            self._program.release()

        if self._texture is not None:
            self._texture.release()

        if self._fbo is not None:
            self._fbo.release()

        if self._vbo is not None:
            self._vbo.release()

        if self._vao is not None:
            self._vao.release()

        self._program = None
        self._texture = None
        self._fbo = None
        self._vbo = None
        self._vao = None

    def try_reload_if_needed(self, new_texture_size: tuple[int, int] | None = None):
        if not self._fs_source.reload_if_needed() and not new_texture_size:
            return

        try:
            self.release()

            self._program = self._gl_context.context.program(
                vertex_shader=self._VERTEX_SHADER,
                fragment_shader=self._fs_source._text,
            )

            self._error = ""

            self._texture_size = new_texture_size or self._texture_size
            self._texture = self._gl_context.context.texture(self._texture_size, 4)
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
        except Exception as e:
            self._error = str(e)
            logger.warning(f"Failed to compile shader: {e}")


class Node:

    def __init__(
        self,
        gl_context: "GLContext",
        fs_source: str | Path,
        output: NodeOutput,
        uniforms: dict[str, Any],
        name: str | None = None,
        reload_period: int = 30,
    ) -> None:
        self._output = output
        self._uniforms = uniforms

        self._resources = NodeResources(
            gl_context=gl_context,
            fs_source=fs_source,
            texture_size=self.get_output_size(),
        )

        self._name = name
        self._reload_period = reload_period
        self._frame_idx = 0

        self._graph: RenderGraph | None = None

    @property
    def name(self) -> str:
        return self._name or str(id(self))

    def get_output_size(self) -> tuple[int, int]:
        size = self._output
        if isinstance(size, str):
            size = (size, 1.0)

        name_or_width, scale_or_height = size
        if isinstance(name_or_width, str):
            name, scale = name_or_width, scale_or_height
            texture = self._uniforms.get(name)
            if isinstance(texture, Node):
                texture = texture._resources._texture
            if not isinstance(texture, moderngl.Texture):
                raise ValueError(f"Uniform '{name}' is not a texture")
            width = max(1, round(texture.size[0] * scale))
            height = max(1, round(texture.size[1] * scale))
        else:
            width, height = name_or_width, scale_or_height

        return width, height  # type: ignore

    def render(self) -> None:
        if self._frame_idx % self._reload_period == 0:
            self._resources.try_reload_if_needed(self.get_output_size())

        self._frame_idx += 1

        if (
            self._resources._program is None
            or self._resources._fbo is None
            or self._resources._vao is None
            or self._resources._error
        ):
            return

        texture_unit = 0

        for u_name, u_value in self._uniforms.items():
            if u_name not in self._resources._program:
                continue

            u_value = u_value() if callable(u_value) else u_value
            if isinstance(u_value, moderngl.Texture):
                u_value.use(texture_unit)
                u_value = texture_unit
                texture_unit += 1
            elif isinstance(u_value, Node):
                if texture := u_value._resources._texture:
                    texture.use(texture_unit)
                    u_value = texture_unit
                    texture_unit += 1
            self._resources._program[u_name] = u_value

        self._resources._fbo.use()
        self._resources._vao.render(moderngl.TRIANGLES)

    def release(self) -> None:
        self._resources.release()
        if self._graph is not None:
            self._graph.remove_node(self)

    def get_uniforms(self):
        uniforms = {}
        are_used = {}

        if self._resources._program is not None:
            program_uniforms = [
                (k, u)
                for k, u in self._resources._program._members.items()
                if isinstance(u, moderngl.Uniform)
            ]
            for name, uniform in program_uniforms:
                value = self._get_default_uniform_value(uniform)
                uniforms[name] = self._uniforms.get(name, value)
                are_used[name] = True

        for name, uniform in self._uniforms.items():
            if name not in uniforms:
                uniforms[name] = uniform
                are_used[name] = False

        return uniforms, are_used

    def get_parents(self) -> list["Node"]:
        return [n for n in self._uniforms.values() if isinstance(n, Node)]

    @staticmethod
    def _get_default_uniform_value(uniform_obj: moderngl.Uniform) -> Any:
        dimension = uniform_obj.dimension
        array_length = uniform_obj.array_length
        is_matrix = uniform_obj.matrix  # type: ignore
        gl_type = uniform_obj.gl_type  # type: ignore

        # Handle matrix uniforms (always float-based)
        if is_matrix:
            # For mat4, dimension=4 (4x4 matrix), etc.
            size = dimension * dimension
            value = [0.0] * size
        # Handle vector uniforms (always float-based)
        elif dimension > 1:
            # For vec2, dimension=2; vec3, dimension=3, etc.
            value = [0.0] * dimension
        # Handle scalar uniforms
        else:
            # Scalar (float, int, etc.)
            if gl_type in [GL_FLOAT, GL_DOUBLE]:
                value = 0.0
            elif gl_type in [GL_INT, GL_UNSIGNED_INT]:
                value = 0
            else:
                value = 0  # Fallback for samplers or other types

        # Handle arrays
        if array_length > 1:
            # Create a list of the value repeated array_length times
            return [value for _ in range(array_length)]
        return tuple(value) if isinstance(value, list) else value


class RenderGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}

    def add_node(self, node: Node) -> None:
        if node.name in self._nodes:
            raise KeyError(f"Node with name {node.name} already exists")

        self._nodes[node.name] = node
        node._graph = self

    def remove_node(self, node: Node) -> None:
        if node.name in self._nodes:
            del self._nodes[node.name]
            node._graph = None

    def get_node(self, name: str) -> Node | None:
        return self._nodes.get(name)

    def iter_nodes(self):
        in_degree = {node: len(node.get_parents()) for node in self._nodes.values()}
        children = defaultdict(list)

        for node in self._nodes.values():
            for parent in node.get_parents():
                children[parent].append(node)

        queue = deque([node for node in self._nodes.values() if in_degree[node] == 0])

        while queue:
            current = queue.popleft()
            yield current

            for child in children[current]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    def render(self) -> None:
        for node in self.iter_nodes():
            node.render()

    def release(self) -> None:
        for node in list(self._nodes.values()):
            node.release()

        self._nodes.clear()
