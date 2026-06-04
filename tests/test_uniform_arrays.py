"""Array-uniform coercion (feature 020·23) — the probe-pinned shapes set_uniform must emit so a
uint text array (u_text), a float[N], and a vecN[M] write to moderngl correctly (or reject cleanly,
never silently corrupt). Pure: a SimpleNamespace stands in for moderngl.Uniform (the matrix is shape
logic, not a GL write — the GL write is verified in-app + by the headless probe)."""

import types

from shaderbox.app import _coerce_uniform_value

_GL_FLOAT = 0x1406
_GL_UINT = 0x1405


def _u(dim: int, n: int, gl_type: int = _GL_FLOAT) -> types.SimpleNamespace:
    return types.SimpleNamespace(dimension=dim, array_length=n, gl_type=gl_type)


# ---- uint text array (u_text: uint[64]) ----


def test_text_array_from_string_pads_to_length() -> None:
    out = _coerce_uniform_value("Hi", _u(1, 4, _GL_UINT))
    assert out == [72, 105, 0, 0]  # "Hi" + null-pad to 4, ints not floats


def test_text_array_from_codepoint_list_pads() -> None:
    out = _coerce_uniform_value([72, 105], _u(1, 4, _GL_UINT))
    assert out == [72, 105, 0, 0]


def test_text_array_truncates_overlong() -> None:
    assert _coerce_uniform_value("Hello", _u(1, 3, _GL_UINT)) == [72, 101, 108]


# ---- numeric arrays ----


def test_float_array_flat_exact_length() -> None:
    assert _coerce_uniform_value([1, 2, 3, 4], _u(1, 4)) == (1.0, 2.0, 3.0, 4.0)


def test_float_array_wrong_length_rejects_no_pad() -> None:
    assert (
        _coerce_uniform_value([1, 2, 3], _u(1, 4)) is None
    )  # short -> None, NOT padded
    assert _coerce_uniform_value([1, 2, 3, 4, 5], _u(1, 4)) is None


def test_vec_array_nests_rows() -> None:
    # vec3[2] wants 2 nested rows of 3 (a flat-6 list raises at the GL write — probe-confirmed).
    out = _coerce_uniform_value([1, 0, 0, 0, 1, 0], _u(3, 2))
    assert out == [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]


def test_vec_array_wrong_length_rejects() -> None:
    assert _coerce_uniform_value([1, 0, 0], _u(3, 2)) is None  # needs 6, got 3


# ---- cross-type rejects ----


def test_string_on_numeric_uniform_rejects() -> None:
    assert (
        _coerce_uniform_value("fast", _u(1, 1)) is None
    )  # str only valid for a uint text array
    assert _coerce_uniform_value("x", _u(1, 4)) is None  # float array, not text


def test_scalar_and_vec_unchanged() -> None:
    assert _coerce_uniform_value(0.5, _u(1, 1)) == 0.5
    assert _coerce_uniform_value([1, 0, 0], _u(3, 1)) == (1.0, 0.0, 0.0)
    assert _coerce_uniform_value([1, 0], _u(3, 1)) is None


def test_uint_array_stays_int_for_node_json_round_trip() -> None:
    # A uint text array must coerce to INT elements (not float) so it survives node.json save/load:
    # the loader tuple-izes the JSON list, and moderngl's uint write needs ints (struct.pack('I')).
    out = _coerce_uniform_value("Hi", _u(1, 4, _GL_UINT))
    assert isinstance(out, list) and all(isinstance(x, int) for x in out)
