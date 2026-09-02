"""The shipped Radiance Cascades document renders bounced light (068).

The check that matters is NOT "it renders something": 063's own RC attempt rendered convincing
shadows while every merge direction read the wrong slot. So this asserts the shape of the
result -- light where nothing emits, darkness behind an occluder -- rather than a mean.

The numerical check against brute force lives in `ai_docs/features/068_radiance_cascades/
oracle.py`, which shares this document's scene and merge.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import moderngl
import pytest

from shaderbox.constants import EXAMPLE_ORDER
from shaderbox.document import Document
from shaderbox.media import texture_to_rgba8

_RC_ID = "77a84d27-2e5b-406d-8011-ee1cb1a9587c"
_DOC = (
    Path(__file__).resolve().parent.parent
    / "shaderbox"
    / "resources"
    / "document_examples"
    / _RC_ID
)


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator[moderngl.Context]:
    os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.6")
    os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "460")
    try:
        ctx = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
    yield ctx
    ctx.release()


def test_the_example_is_registered() -> None:
    assert _RC_ID in EXAMPLE_ORDER


def _render(ctx: moderngl.Context, frames: int = 24):
    doc, _ = Document.load_from_dir(_DOC, ctx)
    paint = doc.passes["paint"]
    # The script drives the brush in the app; here the same strokes are fed directly, so the
    # test needs no ScriptEngine and still paints the document's own scene.
    strokes = (
        [(0.28, 0.70, 1.0, (4.0, 3.3, 2.0)), (0.80, 0.80, 1.0, (0.5, 1.2, 4.0))]
        + [(0.50, 0.36 + 0.02 * i, 0.0, (0.0, 0.0, 0.0)) for i in range(15)]
        + [(0.22, 0.26, 0.0, (0.0, 0.0, 0.0))]
    )
    for frame, (x, y, emissive, colour) in enumerate(strokes):
        paint.uniform_values["u_brush"] = (x, y, 0.035 if emissive else 0.02, 1.0)
        paint.uniform_values["u_brush_color"] = colour
        paint.uniform_values["u_brush_emissive"] = emissive
        paint.uniform_values["u_clear"] = 0.0
        doc.begin_frame(frame)
        doc.render(u_time=frame / 30.0)
    paint.uniform_values["u_brush"] = (0.0, 0.0, 0.0, 0.0)
    for frame in range(len(strokes), len(strokes) + frames):
        doc.begin_frame(frame)
        doc.render(u_time=frame / 30.0)
    return doc


def test_every_pass_compiles_and_the_graph_is_clean(gl_ctx: moderngl.Context) -> None:
    doc = _render(gl_ctx, frames=2)
    broken = [n for n, p in doc.passes.items() if p.program is None]
    assert broken == [], f"passes failed to compile: {broken}"
    assert doc.graph_errors == []


def test_the_document_lights_pixels_that_emit_nothing(gl_ctx: moderngl.Context) -> None:
    # The claim of global illumination: a texel with no emitter on it is still lit, because
    # light reached it. Falsifier: a document that only drew its emitters leaves this black.
    doc = _render(gl_ctx)
    img = texture_to_rgba8(doc.render_pass.canvas.texture)
    # Left of the wall, below the warm light, well away from any emitter.
    lit = int(img[380, 120, :3].mean())
    assert lit > 12, f"no bounced light reached an unlit texel (got {lit})"


def test_an_occluder_casts_a_shadow(gl_ctx: moderngl.Context) -> None:
    # The wall at x=0.5 blocks the warm light on its right. Compare two texels at the same
    # height, one each side, both away from the emitters themselves.
    doc = _render(gl_ctx)
    img = texture_to_rgba8(doc.render_pass.canvas.texture)
    row = 300  # inside the wall's vertical span
    warm_side = int(img[row, 200, 0])  # red channel, left of the wall
    shadow_side = int(img[row, 290, 0])  # just right of it
    assert warm_side > shadow_side + 8, (
        f"the wall casts no warm shadow: left {warm_side} vs right {shadow_side}"
    )
