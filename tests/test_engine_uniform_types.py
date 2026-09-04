"""An engine-driven uniform declared with the wrong type is a compile error, not a silent zero.

The engine writes `u_time`, `u_aspect`, `u_resolution`, `u_pass_iteration` and
`u_pass_iterations` itself, and moderngl refuses a value of the wrong shape; before this check
the failed write was swallowed and the uniform stayed at zero every frame -- a jump flood whose
`int u_pass_iteration` never advanced (rc_full_build, attempt 2, turn 3). The check enumerates
`ENGINE_UNIFORM_TYPES`, so a builtin added without a wrong-type test still gets one here."""

from pathlib import Path

import moderngl
import pytest

from shaderbox.core import ENGINE_UNIFORM_TYPES, Pass
from shaderbox.pass_graph import TargetConfig
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.shader_source import ShaderSource

_WRONG: dict[str, str] = {"float": "int", "vec2": "ivec2"}


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    ctx = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(Path()))
    return ctx


def _compile(gl: moderngl.Context, tmp_path: Path, declaration: str, use: str) -> Pass:
    path = tmp_path / "p.frag.glsl"
    path.write_text(
        "#version 460 core\nin vec2 vs_uv;\nout vec4 fs_color;\n"
        f"{declaration}\nvoid main() {{ fs_color = vec4({use}); }}\n",
        encoding="utf-8",
    )
    render_pass = Pass(
        gl=gl, source=ShaderSource.load(path), canvas_size=(8, 8), target=TargetConfig()
    )
    render_pass.compile()
    return render_pass


@pytest.mark.parametrize("name", sorted(ENGINE_UNIFORM_TYPES))
def test_a_wrong_type_is_a_compile_error_on_its_line(
    gl: moderngl.Context, tmp_path: Path, name: str
) -> None:
    right = ENGINE_UNIFORM_TYPES[name]
    wrong = _WRONG[right]
    use = f"vec3(float({name}{'.x' if wrong.startswith('i') and 'vec' in wrong else ''})), 1.0"
    render_pass = _compile(gl, tmp_path, f"uniform {wrong} {name};", use)
    errors = render_pass.compile_unit.errors
    assert len(errors) == 1, [e.message for e in errors]
    assert name in errors[0].message and f"`{right}`" in errors[0].message
    assert f"`{wrong}`" in errors[0].message
    assert errors[0].line == 3  # the declaration line, 0-based
    assert render_pass.program is None


@pytest.mark.parametrize("name", sorted(ENGINE_UNIFORM_TYPES))
def test_the_right_type_compiles_clean(
    gl: moderngl.Context, tmp_path: Path, name: str
) -> None:
    right = ENGINE_UNIFORM_TYPES[name]
    use = f"vec3({name}{'.x' if right.startswith('vec') else ''}), 1.0"
    render_pass = _compile(gl, tmp_path, f"uniform {right} {name};", use)
    assert render_pass.compile_unit.errors == []
    assert render_pass.program is not None
