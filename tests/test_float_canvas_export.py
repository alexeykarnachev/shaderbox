"""A float canvas must export correct pixels, and `Canvas` must carry its own format.

`texture_to_pil` read every texture as 8-bit RGBA. On an `f2`/`f4` target the buffer is
2x/4x larger, so `PILImage.frombytes` consumed the first half and returned a
plausible-looking WRONG image rather than raising -- silent corruption, and the video
path had the same bug via `np.frombuffer(..., dtype=np.uint8)`.

Unreachable while `Canvas` was always `f1`. Multi-step render targets make float
canvases a first-class case, so the read has to be driven by the texture's dtype.
"""

from pathlib import Path

import moderngl
import numpy as np
import pytest

from shaderbox.core import Canvas
from shaderbox.media import texture_to_pil, texture_to_rgba8


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    return moderngl.create_standalone_context(require=460)


def test_canvas_defaults_to_clamped_linear_8bit(gl: moderngl.Context) -> None:
    canvas = Canvas(gl=gl, size=(8, 8))
    assert canvas.texture.dtype == "f1"
    # moderngl defaults repeat_x/y to True; a feedback border needs clamp.
    assert canvas.texture.repeat_x is False
    assert canvas.texture.repeat_y is False


@pytest.mark.parametrize("dtype", ["f1", "f2", "f4"])
def test_canvas_carries_its_format_through_a_resize(
    gl: moderngl.Context, dtype: str
) -> None:
    canvas = Canvas(gl=gl, size=(8, 8), dtype=dtype, wrap=True)
    assert canvas.texture.dtype == dtype
    assert canvas.set_size((16, 16)) is True
    # A resize reallocates; the format must survive it or a float target silently
    # reverts to 8-bit on the next canvas-size change.
    assert canvas.texture.dtype == dtype
    assert canvas.texture.size == (16, 16)
    assert canvas.texture.repeat_x is True


@pytest.mark.parametrize("dtype", ["f1", "f2", "f4"])
def test_texture_read_is_sized_by_dtype(gl: moderngl.Context, dtype: str) -> None:
    texture = gl.texture((4, 4), 4, dtype=dtype)
    frame = texture_to_rgba8(texture)
    assert frame.shape == (4, 4, 4)
    assert frame.dtype == np.uint8


def test_float_export_keeps_the_true_colours(gl: moderngl.Context) -> None:
    # Every texel mid-grey. Read as uint8 the f4 buffer truncates to a quarter of the
    # image and the rest is garbage, so a wrong read shows up as a wrong colour.
    payload = np.full((4, 4, 4), 0.5, dtype=np.float32)
    payload[:, :, 3] = 1.0
    texture = gl.texture((4, 4), 4, data=payload.tobytes(), dtype="f4")

    frame = texture_to_rgba8(texture)
    assert frame[0, 0, 0] == pytest.approx(127, abs=1)
    assert frame[3, 3, 0] == pytest.approx(127, abs=1)
    assert frame[3, 3, 3] == 255


def test_float_export_tonemaps_values_above_one(gl: moderngl.Context) -> None:
    # The reason float targets exist: 8-bit saturates on the first accumulate pass.
    # An HDR value must clamp to white on export, not wrap around to a dark pixel.
    payload = np.full((2, 2, 4), 7.0, dtype=np.float32)
    texture = gl.texture((2, 2), 4, data=payload.tobytes(), dtype="f4")

    frame = texture_to_rgba8(texture)
    assert (frame == 255).all()


def test_pil_roundtrip_of_a_float_texture(gl: moderngl.Context, tmp_path: Path) -> None:
    payload = np.full((4, 4, 4), 0.25, dtype=np.float32)
    payload[:, :, 3] = 1.0
    texture = gl.texture((4, 4), 4, data=payload.tobytes(), dtype="f4")

    image = texture_to_pil(texture)
    assert image.size == (4, 4)
    assert image.mode == "RGBA"
    out = tmp_path / "f.png"
    image.save(out)
    assert out.stat().st_size > 0
