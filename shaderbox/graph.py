from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import moderngl
import numpy as np

from shaderbox.gl import GLContext


@dataclass
class SizeFromIntTuple:
    width: int
    height: int


@dataclass
class SizeFromUniformTexture:
    name: str
    scale: float = 1.0


OutputSize = SizeFromIntTuple | SizeFromUniformTexture


class Node:
    def __init__(
        self,
        gl_context: GLContext,
        fs_source: str,
        output_size: OutputSize,
        uniforms: dict[str, Any],
        name: str | None = None,
    ) -> None:
        self._fs_source = fs_source
        self._output_size = output_size
        self._uniforms = uniforms
        self._name = name or str(id(self))
        self._graph: RenderGraph | None = None
        self._gl_context = gl_context

        self._program: moderngl.Program = self._gl_context.context.program(
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

        self._init()

    def _init(self):
        self._texture = self._gl_context.context.texture(self.output_size, 4)
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

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_size(self) -> tuple[int, int]:
        if isinstance(self._output_size, SizeFromIntTuple):
            return (self._output_size.width, self._output_size.height)
        elif isinstance(self._output_size, SizeFromUniformTexture):
            texture = self._uniforms.get(self._output_size.name)

            if isinstance(texture, Node):
                texture = texture._texture
            elif not isinstance(texture, moderngl.Texture):
                raise ValueError(f"Uniform '{self._output_size.name}' is not a texture")

            width, height = texture.size
            scaled_width = max(1, round(width * self._output_size.scale))
            scaled_height = max(1, round(height * self._output_size.scale))
            return (scaled_width, scaled_height)
        else:
            raise ValueError("Unreachable")

    def set_output_size(self, size: OutputSize):
        self._output_size = size

    @property
    def uniforms(self) -> dict[str, Any]:
        return self._uniforms

    @property
    def texture_id(self) -> int:
        return self._texture.glo

    def render(self) -> None:
        desired_size = self.output_size
        if desired_size != self._texture.size:
            self._texture.release()
            self._fbo.release()
            self._texture = self._gl_context.context.texture(desired_size, 4)
            self._fbo = self._gl_context.context.framebuffer(
                color_attachments=[self._texture]
            )

        texture_unit = 0
        for u_name, u_value in self._uniforms.items():
            u_value = u_value() if callable(u_value) else u_value

            if isinstance(u_value, moderngl.Texture):
                u_value.use(texture_unit)
                u_value = texture_unit
                texture_unit += 1
            elif isinstance(u_value, Node):
                u_value._texture.use(texture_unit)
                u_value = texture_unit
                texture_unit += 1

            self._program[u_name] = u_value

        self._fbo.use()
        self._vao.render(moderngl.TRIANGLES)

    def release(self) -> None:
        self._vbo.release()
        self._vao.release()
        self._fbo.release()
        self._texture.release()
        self._program.release()

        if self._graph:
            self._graph.remove_node(self)

    def get_parents(self) -> list["Node"]:
        return [n for n in self._uniforms.values() if isinstance(n, Node)]


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
