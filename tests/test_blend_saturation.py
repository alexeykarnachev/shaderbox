"""I7: `f1` (8-bit, clamps 0-1) plus additive is an unguarded saturation trap, and six of
seven shipped examples set `dtype: f1`. 102 D5 says D-D (blend becoming a real per-pass
choice) PROBABLY dissolves it, but the spec forbids un-hedging that claim without a gate:
this file is that gate.

**I7 is LIVE, measured with the real `blend.blend_func_for` (not the hardcoded `ONE, ONE`
`core.py` draws today):** a dense overlapping population (20 entities) on an `f1` target
saturates to exactly 1.0 under `additive`, and reads 0.996 (one 8-bit step below white)
under `screen`, both indistinguishable from a smaller dense population -- the correct,
per-mode blend factor pair does NOT dissolve the trap for either brightening mode. This
was measured directly against `blend.py`'s factor table, standalone, outside `Pass.render`
(since `core.py` does not yet read `self.target.blend`), so the result does not depend on
the F-core contract landing. `test_an_f1_target_does_not_saturate_under_each_blend_mode`
below is therefore EXPECTED TO FAIL for `additive` and `screen` even after `core.py` wires
per-mode blend -- that failure is the gate correctly reporting a live defect, not a fixture
bug, and it must not be weakened to pass. I7 needs its own fix (a warn on `f1` + a
brightening mode, or excluding `f1` from the combo for such a pass, or something else) --
deciding which is out of this flow's scope; this file's job is to make the defect
undeniable rather than to close it.
"""

from pathlib import Path

import moderngl
import numpy as np
import pytest

from shaderbox.core import Pass
from shaderbox.pass_graph import BLEND_MODES, BlendMode, TargetConfig
from shaderbox.shader_source import ShaderSource

# A dense cluster of entities, all overlapping at the canvas center, each writing a small
# additive contribution -- 20 entities x 0.5 alpha-equivalent is well past what an 8-bit
# additive accumulation survives (063 measured f1 saturating on the FIRST accumulate pass
# where f2 reached exactly 7.0; this fixture pushes harder still, deliberately).
_ENTITY_FRAGMENT = """#version 460 core
in vec2 vs_quad;
flat in vec2 pos;
flat in float radius;
out vec4 frag_color;
void main() {
    if (length(vs_quad) > 1.0) discard;
    frag_color = vec4(0.3, 0.3, 0.3, 0.5);
}
"""

_CANVAS = (8, 8)
_ENTITY_COUNT = 20


def _dense_overlap_instances() -> dict[str, np.ndarray]:
    # All at the canvas center with a radius large enough to fully overlap -- this is the
    # measurement I7 rests on: does the WORST-CASE population saturate.
    return {
        "pos": np.zeros((_ENTITY_COUNT, 2), dtype="f4"),
        "radius": np.full(_ENTITY_COUNT, 0.8, dtype="f4"),
    }


def _f1_pass(gl_ctx: moderngl.Context, tmp_path: Path, blend: BlendMode) -> Pass:
    source = tmp_path / f"{blend}.frag.glsl"
    source.write_text(_ENTITY_FRAGMENT)
    render_pass = Pass(
        gl=gl_ctx,
        source=ShaderSource.load(source),
        canvas_size=_CANVAS,
        target=TargetConfig(dtype="f1", blend=blend),
    )
    render_pass.compile()
    assert render_pass.program is not None, render_pass.compile_unit.error_raw
    render_pass.render(u_time=0.0, instances=_dense_overlap_instances())
    return render_pass


def _center_pixel(render_pass: Pass) -> np.ndarray:
    # `dtype="f1"` is moderngl's OWN format string for a normalized 8-bit texture (the
    # `Canvas`/`TargetConfig` vocabulary); the bytes it returns are plain unsigned bytes, so
    # the numpy read-back dtype is `"u1"`, not `"f1"` (numpy has no such dtype at all).
    w, h = render_pass.canvas.texture.size
    pixels = np.frombuffer(
        render_pass.canvas.fbo.read(components=4, dtype="f1"), dtype="u1"
    ).reshape(h, w, 4)
    return pixels[h // 2, w // 2].astype("f4") / 255.0


@pytest.mark.parametrize("blend", BLEND_MODES)
def test_an_f1_target_does_not_saturate_under_each_blend_mode(
    gl_ctx: moderngl.Context, tmp_path: Path, blend: BlendMode
) -> None:
    """The gate I7's spec text requires: a dense overlapping population on an `f1` target
    must not clamp every channel to 1.0 under any mode. A value of exactly 1.0 is not itself
    proof of saturation (a mode could legitimately reach full white) -- the falsifier is
    comparing the DENSE population against a SPARSE one of the same mode: saturation is
    "adding more entities stopped changing the result," which only a two-population
    comparison can show, matching the maintainer's rule against a single-sample extremum
    standing in for a distribution question.
    """
    dense = _center_pixel(_f1_pass(gl_ctx, tmp_path, blend))

    sparse_pass_source = tmp_path / f"{blend}_sparse.frag.glsl"
    sparse_pass_source.write_text(_ENTITY_FRAGMENT)
    sparse_pass = Pass(
        gl=gl_ctx,
        source=ShaderSource.load(sparse_pass_source),
        canvas_size=_CANVAS,
        target=TargetConfig(dtype="f1", blend=blend),
    )
    sparse_pass.compile()
    assert sparse_pass.program is not None
    sparse_pass.render(
        u_time=0.0,
        instances={
            "pos": np.zeros((2, 2), dtype="f4"),
            "radius": np.full(2, 0.8, dtype="f4"),
        },
    )
    sparse = _center_pixel(sparse_pass)

    if blend in ("additive", "screen"):
        # Both are monotonically brightening -- the mode a real I7 trap would hit. The gate:
        # 20 entities must read brighter than 2, or the extra 18 contributed NOTHING, which
        # is what saturation looks like.
        assert float(dense.max()) > float(sparse.max()) + 1.0 / 255.0, (
            f"{blend}: 20 entities ({dense.tolist()}) no brighter than 2 "
            f"({sparse.tolist()}) -- saturated"
        )
    # For opaque/alpha/multiply the count-independence is BY DESIGN (opaque and alpha never
    # exceed what a single covering fragment writes; multiply only darkens), so a dense and
    # a sparse population legitimately agreeing is not I7 -- only additive/screen accumulate
    # without bound and can run out of the 8-bit range.
