"""Typed return values a behavior's `update` hands back (feature 041). They correspond
directly to GLSL types and carry the shape the bare Python value can't. A scalar uniform
takes a bare `float`/`int` (no wrapper — convenience); only the shaped/capped kinds get a
type. Each normalizes to exactly what `uniform_coerce.coerce_uniform_value` accepts
(`list | tuple | str | number`) so the engine validates the result against the live
`moderngl.Uniform` with no special-casing.

`Vec2/3/4` subclass `tuple` (so coercion's `isinstance(value, list | tuple)` check passes on
the value as-is) AND carry component-wise vector math (`.x/.y/.z/.w`, `+ - *` and scalar `* /`,
`.dot/.length/.normalized`, `Vec3.cross`) so a physics/geometry script reads naturally — feature
054 found the bare-tuple form (no `.x`, `*n` repeats) made even a strong model abandon a real sim.
`Array` holds a FLAT numeric sequence (`vec2[3]` is `[x0,y0,x1,y1,x2,y2]`, not nested). `Text`
carries the raw string (coercion's `str_to_unicode` branch)."""

import math
from collections.abc import Sequence


class _Vec(tuple[float, ...]):
    # Component-wise vector math shared by Vec2/3/4. Subclasses fix the arity + named components in
    # __new__/properties; every op rebuilds the SAME concrete type via `type(self)(*components)`, so
    # the result stays a coercible tuple of the right length.
    __slots__ = ()

    def __new__(cls, *components: float) -> "_Vec":
        return tuple.__new__(cls, tuple(float(c) for c in components))

    def _same_len(self, other: object) -> bool:
        return isinstance(other, tuple) and len(other) == len(self)

    def __add__(self, other: object) -> "_Vec":
        if not self._same_len(other):
            return NotImplemented
        assert isinstance(other, tuple)
        return type(self)(*(a + b for a, b in zip(self, other, strict=True)))

    def __sub__(self, other: object) -> "_Vec":
        if not self._same_len(other):
            return NotImplemented
        assert isinstance(other, tuple)
        return type(self)(*(a - b for a, b in zip(self, other, strict=True)))

    def __mul__(self, other: object) -> "_Vec":
        if isinstance(other, int | float):
            return type(self)(*(a * other for a in self))
        if self._same_len(other):
            assert isinstance(other, tuple)
            return type(self)(*(a * b for a, b in zip(self, other, strict=True)))
        return NotImplemented

    def __rmul__(self, other: object) -> "_Vec":
        return self.__mul__(other)

    def __truediv__(self, other: object) -> "_Vec":
        if isinstance(other, int | float):
            return type(self)(*(a / other for a in self))
        if self._same_len(other):
            assert isinstance(other, tuple)
            return type(self)(*(a / b for a, b in zip(self, other, strict=True)))
        return NotImplemented

    def __neg__(self) -> "_Vec":
        return type(self)(*(-a for a in self))

    def dot(self, other: Sequence[float]) -> float:
        return float(sum(a * b for a, b in zip(self, other, strict=True)))

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> "_Vec":
        n = self.length()
        return self if n == 0.0 else self / n


class Vec2(_Vec):
    def __new__(cls, x: float, y: float) -> "Vec2":
        return tuple.__new__(cls, (float(x), float(y)))

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]


class Vec3(_Vec):
    def __new__(cls, x: float, y: float, z: float) -> "Vec3":
        return tuple.__new__(cls, (float(x), float(y), float(z)))

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]

    @property
    def z(self) -> float:
        return self[2]

    def cross(self, o: Sequence[float]) -> "Vec3":
        return Vec3(
            self[1] * o[2] - self[2] * o[1],
            self[2] * o[0] - self[0] * o[2],
            self[0] * o[1] - self[1] * o[0],
        )


class Vec4(_Vec):
    def __new__(cls, x: float, y: float, z: float, w: float) -> "Vec4":
        return tuple.__new__(cls, (float(x), float(y), float(z), float(w)))

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]

    @property
    def z(self) -> float:
        return self[2]

    @property
    def w(self) -> float:
        return self[3]


class Array:
    """A whole numeric uniform array (`float[N]` or `vecN[M]`). Accepts EITHER a flat sequence of
    numbers OR a sequence of Vec/tuple/list rows (a `vecN[M]` as `[Vec3(...), Vec3(...), ...]`) —
    rows are auto-flattened row by row (feature 054: a physics sim naturally builds a list of Vecs,
    and hand-flattening was a footgun). Coercion chunks the flat result by the uniform's `dim`."""

    def __init__(self, values: Sequence[object]) -> None:
        def _num(x: object) -> float:
            if isinstance(x, bool) or not isinstance(x, int | float | str):
                raise TypeError(f"not a number: {x!r}")
            return float(x)

        flat: list[float] = []
        try:
            for v in values:
                if isinstance(v, tuple | list):  # a Vec/tuple/list row -> flatten it
                    flat.extend(_num(c) for c in v)
                else:
                    flat.append(_num(v))  # a bare number
        except (TypeError, ValueError) as e:
            raise TypeError(
                "Array takes numbers or Vec/tuple rows — each element must be a number or a "
                "sequence of numbers (e.g. [Vec3(...), ...] or a flat [x0,y0,z0, x1,...])"
            ) from e
        self.values: list[float] = flat


class Text:
    """The text glyph uniform (`uint[N]`) — a string the engine turns into codepoints
    (truncated/padded to the uniform's cap via `str_to_unicode`)."""

    def __init__(self, text: str) -> None:
        self.text: str = text


def normalize_output(value: object) -> object:
    """Reduce an `update` return value to a form `coerce_uniform_value` accepts as-is.
    A bare number / Vec* tuple passes through; Array yields its flat list; Text yields its
    raw string. Any other type is handed back unchanged so coercion can reject it cleanly
    (a clear shape ScriptError, not a murky crash)."""
    if isinstance(value, Array):
        return value.values
    if isinstance(value, Text):
        return value.text
    return value
