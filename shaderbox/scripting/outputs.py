"""Typed return values a behavior's `update` hands back (feature 041). They correspond
directly to GLSL types and carry the shape the bare Python value can't. A scalar uniform
takes a bare `float`/`int` (no wrapper — convenience); only the shaped/capped kinds get a
type. Each normalizes to exactly what `uniform_coerce.coerce_uniform_value` accepts
(`list | tuple | str | number`) so the engine validates the result against the live
`moderngl.Uniform` with no special-casing.

`Vec2/3/4` subclass `tuple` so coercion's `isinstance(value, list | tuple)` check passes on
the value as-is. `Array` holds a FLAT numeric sequence (`vec2[3]` is `[x0,y0,x1,y1,x2,y2]`,
not nested). `Text` carries the raw string (coercion's `str_to_unicode` branch)."""

from collections.abc import Sequence


class Vec2(tuple[float, float]):
    def __new__(cls, x: float, y: float) -> "Vec2":
        return super().__new__(cls, (float(x), float(y)))


class Vec3(tuple[float, float, float]):
    def __new__(cls, x: float, y: float, z: float) -> "Vec3":
        return super().__new__(cls, (float(x), float(y), float(z)))


class Vec4(tuple[float, float, float, float]):
    def __new__(cls, x: float, y: float, z: float, w: float) -> "Vec4":
        return super().__new__(cls, (float(x), float(y), float(z), float(w)))


class Array:
    """A whole numeric uniform array (`float[N]` or `vecN[M]`) — a FLAT sequence of numbers.
    For `vecN[M]` pass the components flattened row by row; coercion chunks them by `dim`."""

    def __init__(self, values: Sequence[float]) -> None:
        try:
            self.values: list[float] = [float(v) for v in values]
        except (TypeError, ValueError) as e:
            # The common vecN[M] mistake is a list of tuples/lists; the float() error is cryptic.
            raise TypeError(
                "Array takes a FLAT sequence of numbers — for a vecN[M] uniform pass the "
                "components flattened row by row (e.g. [x0,y0, x1,y1, ...]), not a list of tuples"
            ) from e


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
