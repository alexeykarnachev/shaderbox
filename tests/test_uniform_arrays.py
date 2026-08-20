"""Array-uniform coercion (feature 020·23) — the probe-pinned shapes set_uniform must emit so a
uint text array (u_text), a float[N], and a vecN[M] write to moderngl correctly (or reject cleanly,
never silently corrupt). Pure: a SimpleNamespace stands in for moderngl.Uniform (the matrix is shape
logic, not a GL write — the GL write is verified in-app + by the headless probe)."""

import types

from shaderbox.uniform_coerce import coerce_uniform_value, gl_type_label

_GL_FLOAT = 0x1406
_GL_UINT = 0x1405
_GL_INT = 0x1404
_GL_INT_VEC3 = 0x8B54
_GL_UINT_VEC3 = 0x8DC7
_GL_FLOAT_VEC3 = 0x8B51
_GL_SAMPLER_2D = 0x8B5E


def _u(dim: int, n: int, gl_type: int = _GL_FLOAT) -> types.SimpleNamespace:
    return types.SimpleNamespace(dimension=dim, array_length=n, gl_type=gl_type)


# ---- uint text array (u_text: uint[64]) ----


def test_text_array_from_string_pads_to_length() -> None:
    out = coerce_uniform_value("Hi", _u(1, 4, _GL_UINT))
    assert out == [72, 105, 0, 0]  # "Hi" + null-pad to 4, ints not floats


def test_text_array_from_codepoint_list_pads() -> None:
    out = coerce_uniform_value([72, 105], _u(1, 4, _GL_UINT))
    assert out == [72, 105, 0, 0]


def test_text_array_truncates_overlong() -> None:
    assert coerce_uniform_value("Hello", _u(1, 3, _GL_UINT)) == [72, 101, 108]


# ---- numeric arrays ----


def test_float_array_flat_exact_length() -> None:
    assert coerce_uniform_value([1, 2, 3, 4], _u(1, 4)) == (1.0, 2.0, 3.0, 4.0)


def test_float_array_wrong_length_rejects_no_pad() -> None:
    assert (
        coerce_uniform_value([1, 2, 3], _u(1, 4)) is None
    )  # short -> None, NOT padded
    assert coerce_uniform_value([1, 2, 3, 4, 5], _u(1, 4)) is None


def test_vec_array_nests_rows() -> None:
    # vec3[2] wants 2 nested rows of 3 (a flat-6 list raises at the GL write — probe-confirmed).
    out = coerce_uniform_value([1, 0, 0, 0, 1, 0], _u(3, 2))
    assert out == [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]


def test_vec_array_wrong_length_rejects() -> None:
    assert coerce_uniform_value([1, 0, 0], _u(3, 2)) is None  # needs 6, got 3


# ---- cross-type rejects ----


def test_string_on_numeric_uniform_rejects() -> None:
    assert (
        coerce_uniform_value("fast", _u(1, 1)) is None
    )  # str only valid for a uint text array
    assert coerce_uniform_value("x", _u(1, 4)) is None  # float array, not text


def test_scalar_and_vec_unchanged() -> None:
    assert coerce_uniform_value(0.5, _u(1, 1)) == 0.5
    assert coerce_uniform_value([1, 0, 0], _u(3, 1)) == (1.0, 0.0, 0.0)
    assert coerce_uniform_value([1, 0], _u(3, 1)) is None


def test_uint_array_stays_int_for_node_json_round_trip() -> None:
    # A uint text array must coerce to INT elements (not float) so it survives node.json save/load:
    # the loader tuple-izes the JSON list, and moderngl's uint write needs ints (struct.pack('I')).
    out = coerce_uniform_value("Hi", _u(1, 4, _GL_UINT))
    assert isinstance(out, list) and all(isinstance(x, int) for x in out)


# ---- gl_type_label: the ONE label producer (feature 060) ----
# The copilot's project map and its set_uniform reject message both read this. A second producer in
# copilot/backend.py collapsed every non-uint type to "float", so an ivec3 was advertised as "vec3"
# and the agent wrote floats into an int uniform. Falsifier: reintroduce that fallthrough and the
# int/ivec3/uvec3/int[4] rows below go red.


def test_int_family_labels_are_not_collapsed_to_float() -> None:
    assert gl_type_label(_u(1, 1, _GL_INT)) == "int"
    assert gl_type_label(_u(3, 1, _GL_INT_VEC3)) == "ivec3"
    assert gl_type_label(_u(3, 1, _GL_UINT_VEC3)) == "uvec3"
    assert gl_type_label(_u(1, 4, _GL_INT)) == "int[4]"


def test_float_and_sampler_and_block_labels() -> None:
    assert gl_type_label(_u(3, 1, _GL_FLOAT_VEC3)) == "vec3"
    assert gl_type_label(_u(1, 4)) == "float[4]"
    # A sampler is a live Uniform but not settable — set_uniform matches this exact label to reject.
    assert gl_type_label(_u(1, 1, _GL_SAMPLER_2D)) == "sampler2D"
