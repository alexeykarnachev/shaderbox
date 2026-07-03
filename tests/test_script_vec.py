"""Feature 054: the scripting Vec2/3/4 are real little vectors (named components + component math),
so a physics/geometry script reads naturally. They stay tuple subclasses so uniform coercion accepts
them unchanged."""

from shaderbox.scripting.outputs import Vec2, Vec3, Vec4, normalize_output


def test_named_components() -> None:
    assert (Vec2(1, 2).x, Vec2(1, 2).y) == (1.0, 2.0)
    v = Vec3(1, 2, 3)
    assert (v.x, v.y, v.z) == (1.0, 2.0, 3.0)
    assert Vec4(1, 2, 3, 4).w == 4.0


def test_component_wise_and_scalar_math() -> None:
    a, b = Vec3(1, 2, 3), Vec3(4, 5, 6)
    assert a + b == Vec3(5, 7, 9)
    assert b - a == Vec3(3, 3, 3)
    assert a * 2 == Vec3(2, 4, 6)  # scalar scale -- NOT tuple repeat
    assert 2 * a == Vec3(2, 4, 6)  # rmul
    assert a * b == Vec3(4, 10, 18)  # component-wise
    assert b / 2 == Vec3(2, 2.5, 3)
    assert -a == Vec3(-1, -2, -3)
    # results keep the concrete type
    assert isinstance(a + b, Vec3) and isinstance(a * 2, Vec3)


def test_dot_length_normalized_cross() -> None:
    assert Vec3(1, 2, 3).dot(Vec3(4, 5, 6)) == 32.0
    assert Vec3(3, 4, 0).length() == 5.0
    n = Vec3(3, 4, 0).normalized()
    assert abs(n.length() - 1.0) < 1e-9
    assert Vec3(0, 0, 0).normalized() == Vec3(0, 0, 0)  # zero-safe
    assert Vec3(1, 0, 0).cross(Vec3(0, 1, 0)) == Vec3(0, 0, 1)


def test_still_a_coercible_tuple() -> None:
    v = Vec3(1, 2, 3)
    assert isinstance(v, tuple) and tuple(v) == (1.0, 2.0, 3.0)
    assert normalize_output(v) is v  # passes through unchanged for coercion


def test_array_auto_flattens_vec_rows() -> None:
    from shaderbox.scripting.outputs import Array

    # a vecN[M] built as a list of Vecs -> flattened row by row (the natural sim form)
    a = Array([Vec3(1, 2, 3), Vec3(4, 5, 6)])
    assert a.values == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # a plain flat list still works
    assert Array([1, 2, 3, 4]).values == [1.0, 2.0, 3.0, 4.0]
    # tuples/lists as rows also flatten
    assert Array([(1, 2), [3, 4]]).values == [1.0, 2.0, 3.0, 4.0]
