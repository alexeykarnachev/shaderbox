"""A `Pass` compiles and renders on its own, with no `Document` above it (065 stage 2).

The point of the split is that the unit which owns a shader, a program, a target and a set of
uniforms is constructible and drawable alone. Everything here builds a bare `Pass` — never a
`Document` — so a document concern leaking back down into the pass fails these rather than passing
because a `Document` happened to supply it.

Needs a real GL context. On the display-less dev box use the EGL backend + the MESA version
overrides (set at process top, read at context creation); skips cleanly if no context is available.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import moderngl
import pytest

from shaderbox.constants import DEFAULT_FS_FILE_PATH
from shaderbox.core import Pass
from shaderbox.pass_graph import TargetConfig
from shaderbox.shader_source import ShaderSource

_RED = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
void main() {
    fs_color = vec4(1.0, 0.0, 0.0, 1.0);
}
"""

_TIME = """#version 460 core
in vec2 vs_uv;
uniform float u_time;
out vec4 fs_color;
void main() {
    fs_color = vec4(sin(u_time), 0.0, 0.0, 1.0);
}
"""

_BROKEN = """#version 460 core
out vec4 fs_color;
void main() {
    fs_color = nonsense_symbol;
}
"""


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator[moderngl.Context]:
    os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.6")
    os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "460")
    # Default-backend like every other GL module's fixture — an EXPLICIT backend="egl" context
    # released here poisons the process's EGL display and the NEXT module's first program
    # compile segfaults (module-order-only; one context recipe per process is the rule).
    try:
        context = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
    yield context
    context.release()


def _pass(
    gl: moderngl.Context,
    src: str = _RED,
    target: TargetConfig | None = None,
    size: tuple[int, int] | None = (16, 16),
) -> Pass:
    render_pass = Pass(gl=gl, canvas_size=size, target=target)
    render_pass.release_program(src)
    render_pass.compile()
    return render_pass


def _pixel(render_pass: Pass) -> tuple[int, ...]:
    return tuple(render_pass.canvas.texture.read()[:4])


def test_a_bare_pass_compiles_and_draws(gl_ctx: moderngl.Context) -> None:
    render_pass = _pass(gl_ctx)
    assert render_pass.compile_unit.errors == []
    assert render_pass.program is not None
    render_pass.render(u_time=0.0)
    assert _pixel(render_pass) == (255, 0, 0, 255)
    render_pass.release()


def test_a_pass_owns_its_uniforms_and_its_clock(gl_ctx: moderngl.Context) -> None:
    render_pass = _pass(gl_ctx, _TIME)
    render_pass.render(u_time=0.0)
    at_zero = _pixel(render_pass)
    render_pass.render(u_time=1.5708)  # sin(pi/2) = 1
    assert _pixel(render_pass)[0] != at_zero[0]
    assert "u_time" in render_pass.uniform_values
    render_pass.release()


def test_a_pass_carries_its_own_compile_errors(gl_ctx: moderngl.Context) -> None:
    # Per-pass error locality is the whole reason a pass is a file: the error has to belong to
    # the pass, not to whatever document happens to hold it.
    render_pass = _pass(gl_ctx, _BROKEN)
    assert render_pass.compile_unit.errors
    assert render_pass.compile_unit.errors[0].path == render_pass.source.path
    render_pass.release()


def test_a_target_config_shapes_the_pass_target(gl_ctx: moderngl.Context) -> None:
    target = TargetConfig(dtype="f2", filter_linear=False, wrap=True)
    render_pass = _pass(gl_ctx, target=target)
    assert (
        render_pass.target is target
    )  # kept, so stage 3 can re-read what a pass was built with
    assert render_pass.canvas.texture.dtype == "f2"
    assert render_pass.canvas.filter == (moderngl.NEAREST, moderngl.NEAREST)
    assert render_pass.canvas.texture.repeat_x and render_pass.canvas.texture.repeat_y
    render_pass.release()


def test_clamp_is_the_default_wrap(gl_ctx: moderngl.Context) -> None:
    # moderngl defaults repeat_x/y to True, which is wrong for a feedback border (D9), so the
    # unset field has to actively turn it off rather than inherit the library's choice.
    render_pass = _pass(gl_ctx, target=TargetConfig())
    assert not render_pass.canvas.texture.repeat_x
    assert not render_pass.canvas.texture.repeat_y
    render_pass.release()


def test_no_target_config_keeps_the_eight_bit_canvas(gl_ctx: moderngl.Context) -> None:
    # TargetConfig's f2 default (D9) belongs to a pass IN A GRAPH. Applying it to every
    # unconfigured pass would silently reformat the canvas every export path reads as 8-bit.
    render_pass = _pass(gl_ctx)
    assert render_pass.target is None
    assert render_pass.canvas.texture.dtype == "f1"
    render_pass.release()


def test_two_passes_are_independent(gl_ctx: moderngl.Context) -> None:
    # The split's real claim: N passes coexist, each with its own program, target and uniforms.
    # One file, one CompileUnit — the container 064 could not give.
    a = _pass(gl_ctx, _RED, size=(8, 8))
    b = _pass(gl_ctx, _TIME, size=(4, 4))
    a.render(u_time=0.0)
    b.render(u_time=0.0)
    assert a.canvas.texture.size == (8, 8)
    assert b.canvas.texture.size == (4, 4)
    assert a.program is not b.program
    assert a.compile_unit is not b.compile_unit
    assert _pixel(a) == (255, 0, 0, 255)
    assert "u_time" in b.uniform_values and "u_time" not in a.uniform_values
    a.release()
    b.release()


def test_a_pass_source_is_its_own_identity(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # A source the FALLBACK could not have produced: asserting the default path back would pass
    # even if the parameter were dropped entirely.
    path = tmp_path / "mine.frag.glsl"
    path.write_text(_RED)
    render_pass = Pass(gl=gl_ctx, source=ShaderSource.load(path))
    assert render_pass.source.path == path != DEFAULT_FS_FILE_PATH
    assert render_pass.source.text == _RED
    render_pass.release()
