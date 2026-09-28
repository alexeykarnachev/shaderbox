"""Blend mode: per-pass GL state chosen from `TargetConfig.blend` (102 D5).

Gates I6 and I7. `Pass.render`'s instanced draw hardcodes `enable(BLEND); blend_func = ONE,
ONE` -- another flow (F-core, rewriting `Pass.render`) owns replacing that with
`blend.blend_func_for(self.target.blend)`. Until that lands, every mode draws additive, and
the tests below that require a mode to be DISTINGUISHABLE from additive are expected to fail
-- they are the contract's own regression guard, not a claim this worktree's code is done.
See the module docstring's XFAIL note beside each such test.

The fixture needs TWO OVERLAPPING entities: over the unconditional black clear, a single
non-overlapping sprite renders identically under every mode, because the operation is only
observable in the overlap (101 research, I6).
"""

from pathlib import Path

import moderngl
import numpy as np
import pytest

from shaderbox.core import Pass
from shaderbox.pass_graph import BLEND_MODES, BlendMode, TargetConfig
from shaderbox.shader_source import ShaderSource

# Two overlapping discs, each writing a constant color with alpha 0.5 -- alpha matters only
# under `alpha` blend, and the overlap is what every mode's math is judged on. `vs_quad` /
# `pos` / `radius` follow the exact instanced fixture shape proven in
# tests/test_instances_routing.py; a shape this test invented itself would be an unreviewed
# second way to build an instanced pass.
_ENTITY_FRAGMENT = """#version 460 core
in vec2 vs_quad;
flat in vec2 pos;
flat in float radius;
out vec4 frag_color;
void main() {
    if (length(vs_quad) > 1.0) discard;
    frag_color = vec4(0.5, 0.0, 0.0, 0.5);
}
"""

_CANVAS = (8, 8)
_OVERLAPPING_POS = np.array([[-0.15, 0.0], [0.15, 0.0]], dtype="f4")
_OVERLAPPING_RADIUS = np.full(2, 0.6, dtype="f4")


def _overlapping_pass(
    gl_ctx: moderngl.Context, tmp_path: Path, blend: BlendMode
) -> Pass:
    source = tmp_path / f"{blend}.frag.glsl"
    source.write_text(_ENTITY_FRAGMENT)
    render_pass = Pass(
        gl=gl_ctx,
        source=ShaderSource.load(source),
        canvas_size=_CANVAS,
        target=TargetConfig(dtype="f2", blend=blend),
    )
    render_pass.compile()
    assert render_pass.program is not None, render_pass.compile_unit.error_raw
    render_pass.render(
        u_time=0.0,
        instances={"pos": _OVERLAPPING_POS, "radius": _OVERLAPPING_RADIUS},
    )
    return render_pass


def _center_pixel(render_pass: Pass) -> np.ndarray:
    w, h = render_pass.canvas.texture.size
    pixels = np.frombuffer(
        render_pass.canvas.fbo.read(components=4, dtype="f2"), dtype="f2"
    ).reshape(h, w, 4)
    return pixels[h // 2, w // 2].astype("f4")


@pytest.mark.parametrize("blend", BLEND_MODES)
def test_every_blend_mode_draws_something_in_the_overlap(
    gl_ctx: moderngl.Context, tmp_path: Path, blend: BlendMode
) -> None:
    """The weakest possible claim, true for every mode including a `blend.py` typo that maps
    a mode to `(ZERO, ZERO)` -- kept as a fast sanity gate before the mode-comparison tests
    below, which need the draw to have happened at all to mean anything."""
    center = _center_pixel(_overlapping_pass(gl_ctx, tmp_path, blend))
    assert float(center.max()) > 0.0, f"{blend}: overlap drew nothing"


def test_opaque_overlap_does_not_double(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    """I6's measurement: additive doubles a constant-color overlap (0.5 + 0.5 = 1.0); opaque
    must not -- the second sprite REPLACES the first, so the overlap reads the same red as
    a single sprite alone, not the sum.

    Expected to FAIL until core.py reads `self.target.blend` instead of the hardcoded
    `ONE, ONE` -- see the module docstring.
    """
    solo_pass = Pass(
        gl=gl_ctx,
        source=ShaderSource.load(_write(tmp_path / "solo.frag.glsl", _ENTITY_FRAGMENT)),
        canvas_size=_CANVAS,
        target=TargetConfig(dtype="f2", blend="opaque"),
    )
    solo_pass.compile()
    assert solo_pass.program is not None
    solo_pass.render(
        u_time=0.0,
        instances={
            "pos": np.array([[0.0, 0.0]], dtype="f4"),
            "radius": np.array([0.6], dtype="f4"),
        },
    )
    solo_red = float(_center_pixel(solo_pass)[0])
    assert solo_red > 0.0, "the premise: a single opaque sprite must draw something"

    overlap_red = float(_center_pixel(_overlapping_pass(gl_ctx, tmp_path, "opaque"))[0])
    assert overlap_red == pytest.approx(solo_red, abs=1e-3), (
        f"opaque overlap doubled: solo={solo_red}, overlap={overlap_red}"
    )


def test_additive_overlap_still_doubles(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    """The control for the test above: additive is SUPPOSED to double the overlap (100's
    original, unchanged behaviour) -- if this one also stopped doubling, the fix broke the
    default mode rather than adding a new one.
    """
    center = _center_pixel(_overlapping_pass(gl_ctx, tmp_path, "additive"))
    assert float(center[0]) == pytest.approx(1.0, abs=1e-3), (
        f"additive overlap did not double: R={float(center[0])}"
    )


def test_opaque_and_additive_are_distinguishable_in_the_overlap(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    """The direct falsifier for 'the control shows but every mode draws the same picture':
    two modes, same fixture, different overlap result.

    Expected to FAIL until core.py reads `self.target.blend`; today both draw additive and
    this assertion is exactly what would catch that.
    """
    opaque_red = float(_center_pixel(_overlapping_pass(gl_ctx, tmp_path, "opaque"))[0])
    additive_red = float(
        _center_pixel(_overlapping_pass(gl_ctx, tmp_path, "additive"))[0]
    )
    assert opaque_red != pytest.approx(additive_red, abs=1e-3), (
        f"opaque ({opaque_red}) and additive ({additive_red}) drew identically -- "
        "blend mode is not reaching the draw"
    )


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path
