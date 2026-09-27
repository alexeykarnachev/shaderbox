"""The vertex stage of an instanced pass: what the engine writes so the author does not.

An instanced pass draws one quad per entity. The author writes only a fragment shader and
declares each per-entity value as a `flat in`; this module turns those declarations into the
vertex shader that feeds them, so a pass still has exactly one source file.

Two field names are RESERVED because the quad cannot be built without them: `pos` is the
entity's centre and `radius` its half-extent, both in clip space. Nothing in a declaration
says which field is geometry -- inferring it by position draws entities at their velocity
when a `vec2` happens to be declared first, and inferring nothing draws every quad across
the whole viewport. Both were measured, and both are silent, so the names are fixed and
their absence is an error.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from shaderbox.intel.glsl import EntityField

# The entity's centre and half-extent, in clip space. Reserved: the generated quad reads
# them by name, so a pass may not use these names for anything else.
POSITION_FIELD = "pos"
RADIUS_FIELD = "radius"

# The per-vertex corner of the unit quad, and the engine's mode switch. An author's field
# may not take either name: the corner would collide with the attribute the VAO binds, and
# the switch would collide with a uniform the engine writes -- both fail inside the
# GENERATED source, where the author has no line to look at.
CORNER_ATTRIBUTE = "a_corner"
MODE_UNIFORM = "sb_instanced"

# The quad-local coordinate, -1..1 from the entity's centre. A NEW name: `vs_uv` is 0..1
# across the canvas and the shader library's own helpers assume that range, so redefining
# it for an instanced pass would silently change what every one of them computes.
QUAD_VARYING = "vs_quad"

# Names the generated source already binds. A field taking one of these is rejected here,
# with the author's own field named, rather than surfacing as a compile error in a file
# they never wrote.
RESERVED_NAMES = frozenset({CORNER_ATTRIBUTE, MODE_UNIFORM, QUAD_VARYING, "vs_uv"})

# How many attribute locations one field of each type consumes. A matrix takes one per
# column while reporting as a single field, so a count of FIELDS does not bound the
# hardware's budget: three mat4 and three floats is six fields and seventeen locations,
# which aliases silently at link, at VAO build and at draw.
_LOCATIONS: dict[str, int] = {
    "mat2": 2,
    "mat3": 3,
    "mat4": 4,
    "dmat2": 2,
    "dmat3": 3,
    "dmat4": 4,
}

# Types a field may take. `bool` is absent because it does not compile as a vertex
# attribute. A matrix is absent for a different reason: it links, but it occupies one
# attribute location per column and so needs a binding per column, which nothing asks for
# yet -- a per-entity transform is expressible as its columns. Both are refused by name
# here rather than by a KeyError in the draw path or a compile error in generated source.
SUPPORTED_TYPES = frozenset(
    {
        "float",
        "vec2",
        "vec3",
        "vec4",
        "int",
        "ivec2",
        "ivec3",
        "ivec4",
        "uint",
        "uvec2",
        "uvec3",
        "uvec4",
    }
)


# Components per field type, and the dtype a column must already be in. Checked rather
# than converted: numpy's own assignment casts silently, so an f8 column would be accepted
# and 1e40 would arrive as inf.
_COMPONENTS: dict[str, int] = {
    "float": 1,
    "vec2": 2,
    "vec3": 3,
    "vec4": 4,
    "int": 1,
    "ivec2": 2,
    "ivec3": 3,
    "ivec4": 4,
    "uint": 1,
    "uvec2": 2,
    "uvec3": 3,
    "uvec4": 4,
}
_EXPECTED_DTYPE: dict[str, str] = {
    name: (
        "f4"
        if name.startswith(("float", "vec"))
        else "u4"
        if name.startswith("u")
        else "i4"
    )
    for name in _COMPONENTS
}


class InstancedError(Exception):
    """A pass's entity declarations cannot form a vertex stage. Carries the author's words."""


@dataclass(frozen=True)
class InstancedProgram:
    """The generated vertex source and the field list its attributes were built from."""

    vertex_source: str
    fields: tuple[EntityField, ...]


def locations_used(fields: tuple[EntityField, ...]) -> int:
    """Attribute locations these fields consume, counting a matrix once per column."""
    return sum(_LOCATIONS.get(field.glsl_type, 1) for field in fields)


def validate_fields(fields: tuple[EntityField, ...], max_attributes: int) -> None:
    """Raise `InstancedError` naming what an author must change, or return.

    `max_attributes` is the driver's `GL_MAX_VERTEX_ATTRIBS`, passed in rather than read
    here so this module needs no GL context and the bound is the real one.
    """
    if not fields:
        raise InstancedError(
            "an instanced pass declares no entity fields -- add e.g. "
            f"`flat in vec2 {POSITION_FIELD};`"
        )
    seen: set[str] = set()
    for field in fields:
        if field.name in RESERVED_NAMES or f"a_{field.name}" in RESERVED_NAMES:
            # Both spellings. A field called `corner` is not itself an engine name, but
            # it GENERATES `a_corner`, which collides with the quad attribute and fails
            # inside source the author never wrote, at a line number that means nothing
            # to them.
            raise InstancedError(
                f"`{field.name}` collides with an engine name -- rename the field"
            )
        if field.name in seen:
            raise InstancedError(f"`{field.name}` is declared more than once")
        seen.add(field.name)
        if field.glsl_type not in SUPPORTED_TYPES:
            raise InstancedError(
                f"`{field.name}` is a {field.glsl_type}, which cannot be a per-entity "
                f"field -- use one of {', '.join(sorted(SUPPORTED_TYPES))}"
            )
    by_name = {field.name: field for field in fields}
    position = by_name.get(POSITION_FIELD)
    if position is None:
        raise InstancedError(
            f"an instanced pass needs `flat in vec2 {POSITION_FIELD};` -- the entity's "
            "centre in clip space"
        )
    if position.glsl_type != "vec2":
        raise InstancedError(
            f"`{POSITION_FIELD}` is the entity's centre and must be a vec2, "
            f"not a {position.glsl_type}"
        )
    radius = by_name.get(RADIUS_FIELD)
    if radius is None:
        raise InstancedError(
            f"an instanced pass needs `flat in float {RADIUS_FIELD};` (or a vec2) -- "
            "the entity's half-extent"
        )
    if radius.glsl_type not in ("float", "vec2"):
        raise InstancedError(
            f"`{RADIUS_FIELD}` is the entity's half-extent and must be a float or a "
            f"vec2, not a {radius.glsl_type}"
        )
    # The corner attribute takes one location of the budget alongside the fields.
    used = locations_used(fields) + 1
    if used > max_attributes:
        raise InstancedError(
            f"these fields need {used} attribute locations and the driver allows "
            f"{max_attributes} -- a matrix costs one location per column"
        )


def _extent_expression(radius: EntityField) -> str:
    # A scalar half-extent is a length, so it must be the same on both axes ON SCREEN. In
    # clip space the axes are not the same length: 0.03 is 58px across and 32px down on a
    # 16:9 canvas, so an unconverted scalar draws an ellipse where the author wrote a
    # circle. Dividing x by the aspect ratio makes the two equal. A vec2 is componentwise
    # by the author's own choice and is left alone, which is how a rectangle is expressed.
    if radius.glsl_type == "vec2":
        return f"a_{RADIUS_FIELD}"
    return f"vec2(a_{RADIUS_FIELD} / u_aspect, a_{RADIUS_FIELD})"


def generate_vertex_source(fields: tuple[EntityField, ...]) -> str:
    """The vertex shader feeding `fields`, for a pass the author never sees the VS of.

    Emits one attribute per field and passes each through `flat`, so the fragment stage's
    own `flat in` matches by name and type. The mode uniform switches the same program
    between a per-entity quad and the ordinary fullscreen triangle pair, which is what lets
    a script decide per frame without a relink or a second VAO.
    """
    radius = next(field for field in fields if field.name == RADIUS_FIELD)
    attributes = "\n".join(f"in {f.glsl_type} a_{f.name};" for f in fields)
    varyings = "\n".join(f"flat out {f.glsl_type} {f.name};" for f in fields)
    assignments = "\n".join(f"    {f.name} = a_{f.name};" for f in fields)
    return f"""#version 460 core

in vec2 {CORNER_ATTRIBUTE};
{attributes}

uniform bool {MODE_UNIFORM};
uniform float u_aspect;

out vec2 vs_uv;
out vec2 {QUAD_VARYING};
{varyings}

void main() {{
{assignments}
    if ({MODE_UNIFORM}) {{
        vec2 extent = {_extent_expression(radius)};
        vec2 corner = {CORNER_ATTRIBUTE};
        {QUAD_VARYING} = corner;
        vec2 clip = a_{POSITION_FIELD} + corner * extent;
        vs_uv = clip * 0.5 + 0.5;
        gl_Position = vec4(clip, 0.0, 1.0);
    }} else {{
        {QUAD_VARYING} = {CORNER_ATTRIBUTE};
        vs_uv = {CORNER_ATTRIBUTE} * 0.5 + 0.5;
        gl_Position = vec4({CORNER_ATTRIBUTE}, 0.0, 1.0);
    }}
}}
"""


def build(fields: tuple[EntityField, ...], max_attributes: int) -> InstancedProgram:
    """Validate `fields` and return the vertex stage they imply."""
    validate_fields(fields, max_attributes)
    return InstancedProgram(generate_vertex_source(fields), fields)


def validate_population(
    fields: tuple[EntityField, ...], columns: Mapping[str, object]
) -> tuple[int, str | None]:
    """`(count, error)` for one frame's columns against what the pass declares.

    Every check is exact and costs O(columns) rather than O(entities) -- measured at
    1.15 us for three columns of 50k. Nothing is truncated or padded: a data array that
    silently loses its tail is corruption wearing a plausible picture, which is the same
    reason the uniform path refuses to pad a numeric array.
    """
    declared = {field.name for field in fields}
    supplied = set(columns)
    missing = declared - supplied
    if missing:
        return 0, f"no column for {', '.join(sorted(missing))}"
    extra = supplied - declared
    if extra:
        # This REFUSES the population rather than ignoring the extra column, and it is
        # deliberately asymmetric with its mirror: a field declared but not yet READ is
        # bound as nothing and the frame draws, because the driver strips it and the two
        # states are indistinguishable from here. A column with no declaration is the
        # other order of the same edit, and refusing it is what stops a renamed field
        # from silently keeping the old name's values.
        return (
            0,
            f"nothing declares {', '.join(sorted(extra))} -- add a `flat in` for it",
        )
    counts = set()
    for field in fields:
        column = columns[field.name]
        if not isinstance(column, np.ndarray):
            return 0, f"`{field.name}` is a {type(column).__name__}, not an array"
        if column.dtype != _EXPECTED_DTYPE[field.glsl_type]:
            return 0, (
                f"`{field.name}` is {column.dtype}, expected "
                f"{_EXPECTED_DTYPE[field.glsl_type]} -- numpy casts silently, so this "
                "is checked rather than converted"
            )
        if not column.flags["C_CONTIGUOUS"]:
            return 0, f"`{field.name}` is not contiguous -- use np.ascontiguousarray"
        width = _COMPONENTS[field.glsl_type]
        shape = column.shape
        if width == 1:
            if column.ndim != 1:
                return 0, f"`{field.name}` is a {field.glsl_type}, wants a 1-D array"
        elif column.ndim != 2 or shape[1] != width:
            return 0, (
                f"`{field.name}` is a {field.glsl_type}, wants (N, {width}), "
                f"got {shape}"
            )
        counts.add(shape[0])
    if len(counts) > 1:
        return 0, f"columns disagree on how many entities there are: {sorted(counts)}"
    return counts.pop() if counts else 0, None
