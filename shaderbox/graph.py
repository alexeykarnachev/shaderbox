import hashlib
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import moderngl
import numpy as np
from loguru import logger
from OpenGL.GL import GL_DOUBLE, GL_FLOAT, GL_INT, GL_UNSIGNED_INT

from shaderbox.gl import GLContext

OutputSize = tuple[int, int] | str | tuple[str, float]
FSSource = str | Path


class Node:
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
        gl_context: "GLContext",
        fs_source: "FSSource",
        output_size: "OutputSize",
        uniforms: dict[str, Any],
        name: str | None = None,
        check_interval: int = 30,
    ) -> None:
        self._gl_context = gl_context
        self._output_size = output_size
        self._uniforms = {}
        self._name = name or str(id(self))
        self._graph: RenderGraph | None = None
        self._fs_file_path: Path | None = None
        self._fs_source: str
        self._fs_file_hash: str | None = None
        self._check_interval = max(1, check_interval)
        self._frame_count = 0

        # ----------------------------------------------------------------
        # Resolve fragment shader source
        if isinstance(fs_source, str) and "\n" not in fs_source.strip():
            fs_source = Path(fs_source)

        if isinstance(fs_source, Path):
            self._fs_file_path = fs_source.resolve()
            self._fs_source = self._fs_file_path.read_text(encoding="utf-8")
            self._fs_file_hash = self._compute_file_hash(self._fs_file_path)
            logger.info(f"Loaded shader file: {self._fs_file_path}")
        else:
            self._fs_source = fs_source

        # ----------------------------------------------------------------
        # Setup the program and uniforms
        self._program = self._gl_context.context.program(
            vertex_shader=self._VERTEX_SHADER,
            fragment_shader=self._fs_source,
        )

        # Extract all uniforms from the program
        program_uniforms = {
            k: u
            for k, u in self._program._members.items()  # type: ignore
            if isinstance(u, moderngl.Uniform)
        }

        # Validate user-provided uniforms
        for u_name in uniforms:
            if u_name not in program_uniforms:
                raise ValueError(f"Uniform '{u_name}' not found in the shader program")

        # Populate self._uniforms with all program uniforms
        for uniform_name, uniform_obj in program_uniforms.items():
            if uniform_name in uniforms:
                self._uniforms[uniform_name] = uniforms[uniform_name]
            else:
                self._uniforms[uniform_name] = self._get_default_uniform_value(
                    uniform_obj
                )

        # ----------------------------------------------------------------
        # Setup OpenGL objects
        self._texture = self._gl_context.context.texture(self.get_output_size(), 4)
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

    @staticmethod
    def _compute_file_hash(file_path) -> str:
        if not file_path:
            return ""
        try:
            with Path.open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to compute hash for {file_path}: {e}")
            return ""

    def reload(self) -> None:
        if not self._fs_file_path:
            logger.warning("Cannot reload shader: no file path associated")
            return

        try:
            fs_source = self._fs_file_path.read_text(encoding="utf-8")
            program = self._gl_context.context.program(
                vertex_shader=self._VERTEX_SHADER,
                fragment_shader=fs_source,
            )

            self._fs_source = fs_source
            self._fs_file_hash = self._compute_file_hash(self._fs_source)

            # Get new uniforms from the program
            new_program_uniforms = {
                k: u
                for k, u in program._members.items()  # type: ignore
                if isinstance(u, moderngl.Uniform)
            }

            # Update self._uniforms
            new_uniforms = {}
            for uniform_name, uniform_obj in new_program_uniforms.items():
                if uniform_name in self._uniforms:
                    new_uniforms[uniform_name] = self._uniforms[uniform_name]
                else:
                    new_uniforms[uniform_name] = self._get_default_uniform_value(
                        uniform_obj
                    )
            self._uniforms = (
                new_uniforms  # Only keep uniforms that exist in the new program
            )

            # Release old resources
            self._vao.release()
            self._program.release()

            # Set new program and vao
            self._program = program
            self._vao = self._gl_context.context.vertex_array(
                self._program, [(self._vbo, "2f", "a_pos")]
            )

            logger.info(f"Successfully reloaded shader from {self._fs_file_path}")
        except Exception as e:
            logger.error(f"Failed to reload shader from {self._fs_file_path}: {e}")

    @property
    def name(self) -> str:
        return self._name

    def get_output_size(self) -> tuple[int, int]:
        s = self._output_size
        if isinstance(s, str):
            s = (s, 1.0)
        name_or_width, scale_or_height = s
        if isinstance(name_or_width, str):
            name, scale = name_or_width, scale_or_height
            texture = self._uniforms.get(name)
            if isinstance(texture, Node):
                texture = texture._texture
            if not isinstance(texture, moderngl.Texture):
                raise ValueError(f"Uniform '{name}' is not a texture")
            width = max(1, round(texture.size[0] * scale))
            height = max(1, round(texture.size[1] * scale))
        else:
            width, height = name_or_width, scale_or_height
        return width, height  # type: ignore

    def set_output_size(self, size: "OutputSize"):
        self._output_size = size

    @property
    def uniforms(self) -> dict[str, Any]:
        return self._uniforms

    @property
    def texture_id(self) -> int:
        return self._texture.glo

    def render(self) -> None:
        self._frame_count += 1
        if (
            self._fs_file_path
            and self._frame_count % self._check_interval == 0
            and self._fs_file_hash
        ):
            current_hash = self._compute_file_hash(self._fs_file_path)
            if current_hash and current_hash != self._fs_file_hash:
                logger.info(f"Detected change in {self._fs_file_path}, reloading...")
                self.reload()

        desired_size = self.get_output_size()
        if desired_size != self._texture.size:
            self._texture.release()
            self._fbo.release()
            self._texture = self._gl_context.context.texture(desired_size, 4)
            self._fbo = self._gl_context.context.framebuffer(
                color_attachments=[self._texture]
            )

        texture_unit = 0
        # Set all uniforms in self._uniforms
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
