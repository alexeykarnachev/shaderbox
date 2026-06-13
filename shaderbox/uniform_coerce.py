"""Uniform-value coercion — the single home for shaping a Python value into the exact
form a `moderngl.Uniform` write wants, shared by the copilot `set_uniform` path and the
CPU-script engine's behavior output (feature 041/044 — `scripting.coerce_one` validates a
script's `update` return against the live uniform via this). A leaf module: it imports only
`moderngl` + the unicode helper, so both `copilot/backend.py` and `scripting/` can pull
it in without a cycle.
"""

from typing import TypeGuard

import moderngl
from OpenGL.GL import (
    GL_INT,
    GL_INT_VEC2,
    GL_INT_VEC3,
    GL_INT_VEC4,
    GL_UNSIGNED_INT,
    GL_UNSIGNED_INT_VEC2,
    GL_UNSIGNED_INT_VEC3,
    GL_UNSIGNED_INT_VEC4,
)

from shaderbox.util import str_to_unicode

# moderngl rejects a float written to any of these (`required argument is not an integer`),
# so an integer-typed uniform's value must be int-coerced before the write — shader math + a
# script naturally yield floats. GL_UNSIGNED_INT scalar/array is the text-glyph kind too.
_UINT_GL_TYPES = frozenset(
    {GL_UNSIGNED_INT, GL_UNSIGNED_INT_VEC2, GL_UNSIGNED_INT_VEC3, GL_UNSIGNED_INT_VEC4}
)
_SINT_GL_TYPES = frozenset({GL_INT, GL_INT_VEC2, GL_INT_VEC3, GL_INT_VEC4})
_INT_GL_TYPES = _UINT_GL_TYPES | _SINT_GL_TYPES


def is_number(v: object) -> TypeGuard[int | float]:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _is_int_uniform(uniform: moderngl.Uniform) -> bool:
    return uniform.gl_type in _INT_GL_TYPES  # type: ignore[attr-defined]


def is_text_array(uniform: moderngl.Uniform) -> bool:
    # The glyph text uniform: a `uint[N]` (scalar uint array) the engine fills from a string's
    # codepoints. The one home for this shape test (coercion + the stub generator share it).
    return (
        uniform.array_length > 1
        and uniform.dimension == 1
        and uniform.gl_type == GL_UNSIGNED_INT  # type: ignore[attr-defined]
    )


def gl_type_label(uniform: moderngl.Uniform) -> str:
    # A human GLSL-ish name for the uniform's shape — for the shape-mismatch message.
    dim = uniform.dimension
    n = uniform.array_length
    gl_type = uniform.gl_type  # type: ignore[attr-defined]
    base = (
        "uint"
        if gl_type in _UINT_GL_TYPES
        else "int"
        if gl_type in _SINT_GL_TYPES
        else "float"
    )
    scalar = base if dim == 1 else f"{base[0] if base != 'float' else ''}vec{dim}"
    return f"{scalar}[{n}]" if n > 1 else scalar


def coerce_uniform_value(
    value: object, uniform: moderngl.Uniform
) -> float | int | list[int] | tuple[float, ...] | list[tuple[float, ...]] | None:
    # Coerce a value to the exact shape moderngl's Uniform write wants (probe-pinned): scalar -> number;
    # vecN -> tuple of N; dim==1 array -> flat list of array_length; vecN[M] -> array_length nested
    # dim-tuples. An integer-typed uniform rounds to int (moderngl rejects a float write). None on any
    # mismatch (caller errors). Bools rejected.
    dim = uniform.dimension
    n = uniform.array_length
    is_int = _is_int_uniform(uniform)
    if n > 1:
        return coerce_array(value, uniform, dim, n)
    if dim == 1:
        if not is_number(value):
            return None
        return round(value) if is_int else value
    if not isinstance(value, list | tuple) or len(value) != dim:
        return None
    if not all(is_number(v) for v in value):
        return None
    return tuple(round(v) if is_int else float(v) for v in value)


def coerce_array(
    value: object, uniform: moderngl.Uniform, dim: int, n: int
) -> list[int] | tuple[float, ...] | list[tuple[float, ...]] | None:
    # uint[N] TEXT array (a glyph BUFFER): a str OR a codepoint list -> ints, truncated/null-padded to
    # N (a partial buffer is the norm — the rest are empty glyph slots). A NON-text numeric array is
    # EXACT-length, NO padding (padding a data array is silent corruption) — int-rounded for an integer
    # uniform, float otherwise.
    if is_text_array(uniform):
        if isinstance(value, str):
            return str_to_unicode(value, n)
        if isinstance(value, list | tuple) and all(is_number(v) for v in value):
            ints = [int(v) for v in value][:n]
            return ints + [0] * (n - len(ints))
        return None
    if not isinstance(value, list | tuple) or not all(is_number(v) for v in value):
        return None
    if len(value) != (n if dim == 1 else n * dim):
        return None
    conv = round if _is_int_uniform(uniform) else float
    if dim == 1:  # float[N] / int[N] -> flat tuple of exactly N
        return tuple(conv(v) for v in value)
    flat = [conv(v) for v in value]  # vecN[M] -> M rows of `dim`
    return [tuple(flat[i : i + dim]) for i in range(0, n * dim, dim)]


def uniform_shape_hint(uniform: moderngl.Uniform, label: str, value: object) -> str:
    # The shape-mismatch feedback: the exact shape `label` expects + what the rejected value was.
    dim = uniform.dimension
    n = uniform.array_length
    got = " (got a bool — use 1.0/0.0)" if isinstance(value, bool) else ""
    if is_text_array(uniform):
        return (
            f"value does not match {label} (a text array) — pass the text as a string e.g. "
            f'"Hello\\nWorld", or a list of up to {n} codepoint ints{got}'
        )
    if n > 1 and dim == 1:
        return (
            f"value does not match {label} — provide a list of exactly {n} numbers{got}"
        )
    if n > 1:
        return (
            f"value does not match {label} — provide a list of {n * dim} numbers "
            f"({n} groups of {dim}){got}"
        )
    if dim > 1:
        return f"value does not match {label} — provide a list of {dim} numbers for a vector{got}"
    return f"value does not match {label} — provide a number{got}"
