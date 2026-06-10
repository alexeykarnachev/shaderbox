"""Regression: UINode.save must seed uniform_values for the live program, not rely on a prior
render() (the copilot create_node(source=...) path compiles then saves with no render in between).
Pre-fix this raised KeyError on the first non-engine uniform, or ValueError on a sampler (whose
GL default is an int texture-unit, not a usable value). Node.seed_uniform_values is the single home
for the per-type defaults; save calls it. These need a real GL context; skip if none is available."""

import contextlib
from collections.abc import Iterator
from pathlib import Path

import moderngl
import pytest

from shaderbox.core import Node
from shaderbox.ui_models import UINode

_SCALAR_SRC = """#version 460 core
in vec2 vs_uv;
uniform float u_aspect;
uniform vec3 u_bg_color;
uniform float u_radius;
out vec4 fs_color;
void main() {
    vec2 p = vs_uv - 0.5;
    p.x *= u_aspect;
    float d = length(p) - u_radius;
    fs_color = vec4(u_bg_color * step(0.0, d), 1.0);
}
"""

_SAMPLER_SRC = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_image;
out vec4 fs_color;
void main() { fs_color = texture(u_image, vs_uv); }
"""


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator[moderngl.Context]:
    try:
        ctx = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
    yield ctx
    ctx.release()


def _node_from_source(gl: moderngl.Context, source: str) -> Node:
    # Mirror create_node's path: swap in source, compile, but DO NOT render.
    node = Node(gl=gl)
    node.release_program(source)
    node.compile()
    return node


def _teardown(node: Node) -> None:
    with contextlib.suppress(Exception):
        node.release()


def test_save_seeds_scalar_uniforms_without_render(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    node = _node_from_source(gl_ctx, _SCALAR_SRC)
    assert "u_bg_color" not in node.uniform_values  # unseeded: no render happened
    ui_node = UINode(node=node, id="scalar")
    ui_node.save(tmp_path)  # pre-fix: KeyError: 'u_bg_color'
    reloaded, meta = Node.load_from_dir(tmp_path / "scalar", gl=gl_ctx)
    assert "u_bg_color" in meta["uniforms"]
    assert "u_radius" in meta["uniforms"]
    _teardown(node)
    _teardown(reloaded)


def test_save_seeds_sampler_uniform_without_render(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    node = _node_from_source(gl_ctx, _SAMPLER_SRC)
    ui_node = UINode(node=node, id="sampler")
    ui_node.save(
        tmp_path
    )  # pre-fix: ValueError (sampler default was an int, not a texture)
    reloaded, meta = Node.load_from_dir(tmp_path / "sampler", gl=gl_ctx)
    assert "u_image" in meta["uniforms"]  # serialized via the default Image
    _teardown(node)
    _teardown(reloaded)


def test_seed_skips_engine_uniforms(gl_ctx: moderngl.Context) -> None:
    node = _node_from_source(gl_ctx, _SCALAR_SRC)
    node.seed_uniform_values()
    assert (
        "u_aspect" not in node.uniform_values
    )  # engine-driven: valued only in render()
    _teardown(node)
