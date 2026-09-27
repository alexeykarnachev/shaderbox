"""The generated vertex stage of an instanced pass.

The author writes a fragment shader and declares each per-entity value as a `flat in`; the
engine writes the vertex shader. These pin what the author is promised: their own names on
both sides, a round shape where they asked for one, and an error naming their own field
rather than a compile failure in a file they never wrote.
"""

from pathlib import Path

import moderngl
import numpy as np
import pytest

from shaderbox.core import Pass
from shaderbox.instanced import (
    InstancedError,
    build,
    generate_vertex_source,
    locations_used,
    validate_fields,
)
from shaderbox.intel.glsl import EntityField, entity_fields
from shaderbox.shader_source import ShaderSource

# A driver's real budget is passed in, so these fix one to keep the arithmetic readable.
_ATTRIBUTES = 16


def _lit(render_pass: Pass) -> np.ndarray:
    width, height = render_pass.canvas.texture.size
    pixels = np.frombuffer(
        render_pass.canvas.fbo.read(components=4, dtype="f2"), dtype="f2"
    ).reshape(height, width, 4)
    return pixels[..., 3] > 0.5


def _fields(source: str) -> tuple[EntityField, ...]:
    return entity_fields(source)


_MINIMAL = "flat in vec2 pos;\nflat in float radius;\n"


def test_a_matrix_is_refused_rather_than_silently_unbindable() -> None:
    # A matrix links as an attribute but occupies one location per column, so it needs a
    # binding per column. Nothing asks for that yet -- a per-entity transform is its
    # columns -- so it is refused HERE, by name, rather than as a KeyError in the draw
    # path or a compile error inside source the author never wrote.
    source = "flat in vec2 pos;\nflat in float radius;\nflat in mat4 xform;"
    with pytest.raises(InstancedError, match="xform"):
        validate_fields(_fields(source), _ATTRIBUTES)


def test_the_location_budget_refuses_what_would_alias() -> None:
    # Above the driver's budget, locations ALIAS: two fields share one, with no error at
    # link, at VAO build or at draw -- just wrong values. The corner attribute takes one
    # of the budget alongside the fields, which is why this crosses at 16 rather than 17.
    source = "flat in vec2 pos;\nflat in float radius;\n" + "".join(
        f"flat in vec4 c{i};\n" for i in range(14)
    )
    assert locations_used(_fields(source)) == 16
    with pytest.raises(InstancedError, match="attribute locations"):
        validate_fields(_fields(source), _ATTRIBUTES)


def test_the_two_geometry_fields_are_required_by_name() -> None:
    # Nothing in a declaration says which field is the centre and which the half-extent.
    # Inferring by position draws every entity at its velocity when a vec2 is declared
    # first; inferring nothing draws every quad across the whole viewport. Both measured,
    # both silent -- so the names are fixed and their absence is an error the author reads.
    with pytest.raises(InstancedError, match="pos"):
        validate_fields(
            _fields("flat in float radius;\nflat in vec2 centre;"), _ATTRIBUTES
        )
    with pytest.raises(InstancedError, match="radius"):
        validate_fields(_fields("flat in vec2 pos;\nflat in float size;"), _ATTRIBUTES)


def test_a_geometry_field_of_the_wrong_type_is_named() -> None:
    with pytest.raises(InstancedError, match="must be a vec2"):
        validate_fields(
            _fields("flat in vec3 pos;\nflat in float radius;"), _ATTRIBUTES
        )
    with pytest.raises(InstancedError, match="float or a vec2"):
        validate_fields(_fields("flat in vec2 pos;\nflat in vec4 radius;"), _ATTRIBUTES)


def test_an_engine_name_is_refused_with_the_author_s_field_named() -> None:
    # These collide inside the GENERATED source, where the author has no line to look at,
    # so the error has to be raised here while their own field name is still in hand.
    for name in ("vs_uv", "vs_quad", "sb_instanced", "a_corner"):
        source = f"flat in vec2 pos;\nflat in float radius;\nflat in float {name};"
        with pytest.raises(InstancedError, match=name):
            validate_fields(_fields(source), _ATTRIBUTES)


def test_bool_is_refused_by_name_rather_than_by_the_compiler() -> None:
    source = "flat in vec2 pos;\nflat in float radius;\nflat in bool awake;"
    with pytest.raises(InstancedError, match="awake"):
        validate_fields(_fields(source), _ATTRIBUTES)


def test_a_pass_with_no_entity_fields_is_not_instanced() -> None:
    with pytest.raises(InstancedError, match="no entity fields"):
        validate_fields((), _ATTRIBUTES)


def test_a_scalar_radius_is_aspect_corrected_and_a_vec2_is_not() -> None:
    # A scalar half-extent is a LENGTH, so it must cover the same distance on both screen
    # axes. Clip space is not square: 0.03 is 58px across and 32px down at 16:9, so an
    # unconverted scalar draws an ellipse where the author wrote a circle. A vec2 is
    # componentwise by the author's own choice, which is how a rectangle is expressed.
    assert "u_aspect" in generate_vertex_source(_fields(_MINIMAL))
    rect = _fields("flat in vec2 pos;\nflat in vec2 radius;")
    assert "u_aspect" not in generate_vertex_source(rect).split("void main")[1]


def test_the_generated_stage_links_and_draws_one_quad_per_entity(
    gl_ctx: moderngl.Context,
) -> None:
    fragment = """#version 460 core
in vec2 vs_uv;
in vec2 vs_quad;
flat in vec2  pos;
flat in float radius;
flat in float energy;
out vec4 frag_color;
void main(){
    if (length(vs_quad) > 1.0) discard;
    frag_color = vec4(energy, 1.0 - energy, 0.0, 1.0);
}
"""
    fields = entity_fields(fragment)
    program = build(fields, gl_ctx.info["GL_MAX_VERTEX_ATTRIBS"])
    compiled = gl_ctx.program(
        vertex_shader=program.vertex_source, fragment_shader=fragment
    )

    corner = gl_ctx.buffer(
        np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], dtype="f4")
    )
    # Three entities at distinct x. A fixture with one entity cannot tell a per-instance
    # attribute from a per-vertex one: with divisor 0 every instance reads row 0 and the
    # single-entity picture is identical either way.
    positions = gl_ctx.buffer(np.array([[-0.6, 0], [0, 0], [0.6, 0]], dtype="f4"))
    radii = gl_ctx.buffer(np.array([0.10, 0.12, 0.14], dtype="f4"))
    energies = gl_ctx.buffer(np.array([0.0, 0.5, 1.0], dtype="f4"))
    vao = gl_ctx.vertex_array(
        compiled,
        [
            (corner, "2f", "a_corner"),
            (positions, "2f/i", "a_pos"),
            (radii, "1f/i", "a_radius"),
            (energies, "1f/i", "a_energy"),
        ],
    )
    width, height = 320, 180
    canvas = gl_ctx.texture((width, height), 4, dtype="f2")
    fbo = gl_ctx.framebuffer([canvas])
    compiled["u_aspect"].value = width / height
    compiled["sb_instanced"].value = True
    fbo.use()
    gl_ctx.clear()
    vao.render(moderngl.TRIANGLES, vertices=6, instances=3)

    pixels = np.frombuffer(fbo.read(components=4, dtype="f2"), dtype="f2").reshape(
        height, width, 4
    )
    lit = pixels[..., 3] > 0.5
    columns = np.where(lit.any(axis=0))[0]
    runs = 1 + len(np.where(np.diff(columns) > 1)[0])
    assert runs == 3, "three entities must draw at three distinct places"

    # The middle entity's scalar radius must cover the same distance both ways. Measured
    # without the aspect division this is 1.78x wider than tall.
    middle = lit[:, width // 2 - 20 : width // 2 + 20]
    rows = np.where(middle.any(axis=1))[0]
    cols = np.where(middle.any(axis=0))[0]
    span_x = cols.max() - cols.min() + 1
    span_y = rows.max() - rows.min() + 1
    assert abs(span_x - span_y) <= 2, f"a scalar radius drew {span_x}x{span_y}"

    # The same program draws fullscreen when the engine clears the flag -- one program,
    # one VAO, no relink, which is what lets a script decide per frame.
    compiled["sb_instanced"].value = False
    fbo.use()
    gl_ctx.clear()
    vao.render(moderngl.TRIANGLES, vertices=6, instances=1)
    after = np.frombuffer(fbo.read(components=4, dtype="f2"), dtype="f2").reshape(
        height, width, 4
    )
    assert (after[..., 3] > 0.5).sum() > lit.sum() * 4, "fullscreen must cover far more"

    for owned in (corner, positions, radii, energies, vao, compiled, fbo, canvas):
        owned.release()


def test_a_pass_draws_its_population_and_frees_it(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    """The whole seam through a real Pass: declare, upload, draw, switch mode, release."""
    source = tmp_path / "swarm.frag.glsl"
    source.write_text(
        """#version 460 core
in vec2 vs_uv;
in vec2 vs_quad;
flat in vec2  pos;
flat in float radius;
flat in float energy;
out vec4 frag_color;
void main(){
    if (length(vs_quad) > 1.0) discard;
    frag_color = vec4(energy, 1.0 - energy, 0.0, 1.0);
}
"""
    )
    render_pass = Pass(
        gl=gl_ctx, source=ShaderSource.load(source), canvas_size=(320, 180)
    )
    render_pass.compile()
    assert render_pass.program is not None, render_pass.compile_unit.error_raw
    assert [f.name for f in render_pass.entity_fields] == ["pos", "radius", "energy"]

    columns = {
        "pos": np.array([[-0.6, 0], [0, 0], [0.6, 0]], dtype="f4"),
        "radius": np.array([0.10, 0.12, 0.14], dtype="f4"),
        "energy": np.array([0.0, 0.5, 1.0], dtype="f4"),
    }
    render_pass.render(u_time=0.0, instances=columns)
    lit = _lit(render_pass)
    columns_hit = np.where(lit.any(axis=0))[0]
    runs = 1 + len(np.where(np.diff(columns_hit) > 1)[0])
    assert runs == 3, "three entities must draw at three distinct places"
    instanced_total = int(lit.sum())

    # The same program draws fullscreen when no population arrives. Asserted as a
    # DIFFERENT picture, not merely a non-empty one: before the mode uniform was set the
    # instanced draw silently took this branch and the two were byte-identical.
    render_pass.render(u_time=0.0)
    assert int(_lit(render_pass).sum()) > instanced_total * 4

    # A population past its capacity grows the buffers rather than truncating.
    bigger = {k: np.repeat(v, 400, axis=0).astype("f4") for k, v in columns.items()}
    render_pass.render(u_time=0.0, instances=bigger)
    assert int(_lit(render_pass).sum()) > 0

    render_pass.invalidate()
    assert render_pass.instance_buffers == {}
    assert render_pass.entity_fields == ()


def test_invalidate_frees_the_instance_buffers_it_allocated(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    """A shader edit must not leak a buffer per field, per edit.

    The sibling gate in `test_gl_lifetime_guards.py` runs on a fullscreen example, where
    `instance_buffers` is empty before and after -- so it asserts `{} == {}` and stays
    green with the release deleted. This one allocates buffers first and proves they
    were there, which is what makes the silence afterwards mean anything.
    """
    source = tmp_path / "swarm.frag.glsl"
    source.write_text(
        """#version 460 core
in vec2 vs_quad;
flat in vec2  pos;
flat in float radius;
out vec4 frag_color;
void main(){ if (length(vs_quad) > 1.0) discard; frag_color = vec4(1.0); }
"""
    )
    render_pass = Pass(
        gl=gl_ctx, source=ShaderSource.load(source), canvas_size=(64, 64)
    )
    render_pass.compile()
    render_pass.render(
        u_time=0.0,
        instances={
            "pos": np.zeros((3, 2), dtype="f4"),
            "radius": np.full(3, 0.2, dtype="f4"),
        },
    )
    allocated = list(render_pass.instance_buffers.values())
    assert len(allocated) == 2, "the fixture allocated nothing -- it never reached this"

    render_pass.invalidate()

    assert render_pass.instance_buffers == {}
    # Freed, not merely forgotten. GL hands released names back out, so a recompile that
    # re-uploads must land on the SAME names. Clearing the dict without releasing leaves
    # the old names allocated and the new buffers take fresh ones -- which is the leak,
    # and it is what this catches. Comparing the classes would not: moderngl's deferred
    # collection leaves a released Buffer answering as a Buffer.
    freed = sorted(buffer.glo for buffer in allocated)
    render_pass.compile()
    render_pass.render(
        u_time=0.0,
        instances={
            "pos": np.zeros((3, 2), dtype="f4"),
            "radius": np.full(3, 0.2, dtype="f4"),
        },
    )
    assert sorted(b.glo for b in render_pass.instance_buffers.values()) == freed


def test_build_refuses_invalid_fields_through_the_seam_compile_uses(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # Every other validation test calls `validate_fields` directly, so deleting the call
    # from `build` -- which is what `Pass.compile` uses -- leaves them all green. This
    # goes the way a real pass does and checks the author reads their own field's name.
    source = tmp_path / "bad.frag.glsl"
    source.write_text(
        """#version 460 core
in vec2 vs_quad;
flat in vec2  pos;
flat in float radius;
flat in mat4  xform;
out vec4 frag_color;
void main(){ frag_color = vec4(xform[0][0]); }
"""
    )
    render_pass = Pass(
        gl=gl_ctx, source=ShaderSource.load(source), canvas_size=(64, 64)
    )
    render_pass.compile()
    assert render_pass.program is None
    assert "xform" in render_pass.compile_unit.error_raw
