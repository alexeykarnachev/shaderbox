"""How each `BlendMode` maps to GL state. The one place the modes become numbers.

Separate from `pass_graph.py` because that module is pure document state and imports no
moderngl -- the persisted vocabulary and its GL meaning are two facts, and only this one
needs a GL context to be true.
"""

import moderngl

from shaderbox.pass_graph import BLEND_MODES, BlendMode

# src, dst factors per mode. `opaque` is the one that does not blend at all: it is
# expressed as a factor pair rather than as a `disable(BLEND)` branch so every mode is one
# shape and the draw has no per-mode control flow -- which is what keeps a sixth mode from
# needing a sixth `if`.
_FACTORS: dict[BlendMode, tuple[int, int]] = {
    "additive": (moderngl.ONE, moderngl.ONE),
    "alpha": (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA),
    "opaque": (moderngl.ONE, moderngl.ZERO),
    "screen": (moderngl.ONE, moderngl.ONE_MINUS_SRC_COLOR),
}

# Every mode has a mapping. A `Literal` member with no entry would raise at draw time on
# the one document that used it, which is the shape this check exists to make impossible.
assert set(_FACTORS) == set(BLEND_MODES), (
    f"blend modes without a GL mapping: {set(BLEND_MODES) - set(_FACTORS)}"
)


def blend_func_for(mode: BlendMode) -> tuple[int, int]:
    """The (src, dst) factor pair for a mode.

    Unconditional `KeyError` on an unknown mode rather than a default: a mode that reached
    here without a mapping is a `graph.json` holding a value the `Literal` should have
    rejected, and answering it with `additive` would draw a plausible picture for a
    document that is wrong.
    """
    return _FACTORS[mode]
