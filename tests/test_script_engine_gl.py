"""GL-integration test for the CPU-script engine (feature 040): a scripted scalar + a scripted
vec2 are ticked then rendered on a real GL context, asserting the computed value reaches
node.uniform_values, that the rendered pixel CHANGES between two t values (the value reaches the
GPU), and that a shape-mismatch freezes + records a ScriptError.

Needs a real GL context. On the display-less dev box use the EGL backend + the MESA version
overrides (set at process top, read at context creation); skips cleanly if no context is available.
"""

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.6")
os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "460")

import moderngl
import pytest

from shaderbox.core import Node
from shaderbox.scripting import EngineContext, ScriptEngine

_SRC = """#version 460 core
in vec2 vs_uv;
uniform float u_wave;
uniform vec2 u_offset;
out vec4 fs_color;
void main() {
    fs_color = vec4(u_wave, u_offset.x, u_offset.y, 1.0);
}
"""


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator[moderngl.Context]:
    ctx = None
    for kwargs in ({"backend": "egl"}, {}):
        try:
            ctx = moderngl.create_standalone_context(**kwargs)  # type: ignore[arg-type]
            break
        except Exception:
            continue
    if ctx is None:
        pytest.skip("no standalone GL context available")
    yield ctx
    ctx.release()


def _node(gl: moderngl.Context) -> Node:
    node = Node(gl=gl)
    node.release_program(_SRC)
    node.compile()
    node.render(u_time=0.0)  # warm-up so get_active_uniforms is populated
    return node


def _write(scripts_dir: Path, name: str, body: str) -> None:
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / f"{name}.py").write_text(body, encoding="utf-8")


def _pixel(node: Node) -> tuple[int, int, int, int]:
    data = node.canvas.texture.read()
    return tuple(data[:4])  # type: ignore[return-value]


def test_scripted_value_reaches_gpu(gl_ctx: moderngl.Context, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    _write(scripts_dir, "u_wave", "out.set(0.5 + 0.5 * sin(ctx.t))")
    _write(scripts_dir, "u_offset", "out.set(0.25, 0.75)")
    node = _node(gl_ctx)
    eng = ScriptEngine()
    eng.reload("n", scripts_dir, node)

    eng.tick("n", node, EngineContext(t=0.0, dt=0.0, frame=0))
    assert abs(node.uniform_values["u_wave"] - 0.5) < 1e-6
    assert node.uniform_values["u_offset"] == (0.25, 0.75)
    node.render(u_time=0.0)
    px_a = _pixel(node)

    # A different t -> a different u_wave -> a different rendered red channel.
    eng.tick("n", node, EngineContext(t=1.5708, dt=0.0, frame=1))  # sin(pi/2)=1 -> u_wave≈1.0
    node.render(u_time=1.5708)
    px_b = _pixel(node)
    assert px_a[0] != px_b[0], "scripted u_wave did not reach the GPU"

    with contextlib.suppress(Exception):
        node.release()


def test_shape_mismatch_freezes_and_records(gl_ctx: moderngl.Context, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    _write(scripts_dir, "u_wave", "out.set(0.1, 0.2, 0.3)")  # vec3 into a float uniform
    node = _node(gl_ctx)
    node.seed_uniform_values()
    seeded = node.uniform_values.get("u_wave")
    eng = ScriptEngine()
    eng.reload("n", scripts_dir, node)
    eng.tick("n", node, EngineContext(t=0.0, dt=0.0, frame=0))
    assert node.uniform_values.get("u_wave") == seeded  # frozen, not corrupted
    assert eng.errors[("n", "u_wave")].kind == "runtime"

    with contextlib.suppress(Exception):
        node.release()
