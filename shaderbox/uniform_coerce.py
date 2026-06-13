"""Uniform-value coercion — the single home for shaping a Python value into the exact
form a `moderngl.Uniform` write wants, shared by the copilot `set_uniform` path and the
CPU-script engine's `UniformOut.set` (feature 040). A leaf module: it imports only
`moderngl` + the unicode helper, so both `copilot/backend.py` and `scripting/` can pull
it in without a cycle.
"""

from typing import TypeGuard

import moderngl
from OpenGL.GL import GL_UNSIGNED_INT

from shaderbox.util import str_to_unicode


def is_number(v: object) -> TypeGuard[int | float]:
    return isinstance(v, int | float) and not isinstance(v, bool)


def coerce_uniform_value(
    value: object, uniform: moderngl.Uniform
) -> float | int | list[int] | tuple[float, ...] | list[tuple[float, ...]] | None:
    # Coerce a value to the exact shape moderngl's Uniform write wants (probe-pinned): scalar -> number;
    # vecN -> tuple of N; dim==1 array -> flat list of array_length; vecN[M] -> array_length nested
    # dim-tuples. None on any mismatch (caller errors). Bools rejected.
    dim = uniform.dimension
    n = uniform.array_length
    if n > 1:
        return coerce_array(value, uniform, dim, n)
    if dim == 1:
        return value if is_number(value) else None
    if not isinstance(value, list | tuple) or len(value) != dim:
        return None
    if not all(is_number(v) for v in value):
        return None
    return tuple(float(v) for v in value)


def coerce_array(
    value: object, uniform: moderngl.Uniform, dim: int, n: int
) -> list[int] | tuple[float, ...] | list[tuple[float, ...]] | None:
    # uint[N] TEXT array: a str (-> codepoints) or int list, truncated/null-padded to N. Numeric array:
    # exact length, NO padding (padding numeric data is silent corruption).
    gl_type = uniform.gl_type  # type: ignore
    if dim == 1 and gl_type == GL_UNSIGNED_INT:
        if isinstance(value, str):
            return str_to_unicode(value, n)
        if isinstance(value, list | tuple) and all(is_number(v) for v in value):
            ints = [int(v) for v in value][:n]
            return ints + [0] * (n - len(ints))
        return None
    # numeric array. A str is never valid here.
    if not isinstance(value, list | tuple) or not all(is_number(v) for v in value):
        return None
    if dim == 1:  # float[N] -> flat list of exactly N
        return tuple(float(v) for v in value) if len(value) == n else None
    if len(value) != n * dim:  # vecN[M] -> N rows of `dim`
        return None
    flat = [float(v) for v in value]
    return [tuple(flat[i : i + dim]) for i in range(0, n * dim, dim)]


def uniform_shape_hint(name: str, uniform: moderngl.Uniform, label: str) -> str:
    # The shape-mismatch feedback: the exact shape `name` expects.
    dim = uniform.dimension
    n = uniform.array_length
    gl_type = uniform.gl_type  # type: ignore
    if n > 1 and dim == 1 and gl_type == GL_UNSIGNED_INT:
        return (
            f"value does not match {label} (a text array) — pass the text as a string e.g. "
            f'"Hello\\nWorld", or a list of up to {n} codepoint ints'
        )
    if n > 1 and dim == 1:
        return f"value does not match {label} — provide a list of exactly {n} numbers"
    if n > 1:
        return (
            f"value does not match {label} — provide a list of {n * dim} numbers "
            f"({n} groups of {dim})"
        )
    if dim > 1:
        return f"value does not match {label} — provide a list of {dim} numbers for a vector"
    return f"value does not match {label} — provide a number"
