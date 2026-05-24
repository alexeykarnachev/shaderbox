"""Pure unit tests for resolve_dims — no GL context needed.

Covers each ResolutionPolicy across square/landscape/portrait sources plus the
16-px alignment rounding that codec compatibility requires.
"""

import pytest

from shaderbox.render_preset import RenderPreset, ResolutionPolicy, resolve_dims


def test_free_keeps_source_but_aligns() -> None:
    # 100x70 → aligned up to the next multiple of 16.
    assert resolve_dims(RenderPreset(), (100, 70)) == (112, 80)


def test_free_already_aligned_unchanged() -> None:
    assert resolve_dims(RenderPreset(), (128, 64)) == (128, 64)


def test_longest_edge_landscape() -> None:
    preset = RenderPreset(
        resolution_policy=ResolutionPolicy.LONGEST_EDGE, longest_edge=512
    )
    # 1024x512 source: longest edge clamped to 512, height halved → 256.
    assert resolve_dims(preset, (1024, 512)) == (512, 256)


def test_longest_edge_portrait() -> None:
    preset = RenderPreset(
        resolution_policy=ResolutionPolicy.LONGEST_EDGE, longest_edge=512
    )
    assert resolve_dims(preset, (512, 1024)) == (256, 512)


def test_longest_edge_does_not_upscale() -> None:
    preset = RenderPreset(
        resolution_policy=ResolutionPolicy.LONGEST_EDGE, longest_edge=512
    )
    # Source smaller than the cap stays at source size (aligned).
    assert resolve_dims(preset, (128, 128)) == (128, 128)


def test_longest_edge_requires_param() -> None:
    preset = RenderPreset(resolution_policy=ResolutionPolicy.LONGEST_EDGE)
    with pytest.raises(ValueError):
        resolve_dims(preset, (256, 256))


def test_fixed_aspect_landscape() -> None:
    preset = RenderPreset(
        resolution_policy=ResolutionPolicy.FIXED_ASPECT,
        aspect=(16, 9),
        longest_edge=512,
    )
    # 16:9 with longest edge 512 → 512 x round(512*9/16)=288 (already aligned).
    assert resolve_dims(preset, (1000, 1000)) == (512, 288)


def test_fixed_aspect_portrait() -> None:
    preset = RenderPreset(
        resolution_policy=ResolutionPolicy.FIXED_ASPECT,
        aspect=(9, 16),
        longest_edge=512,
    )
    assert resolve_dims(preset, (1000, 1000)) == (288, 512)


def test_fixed_aspect_requires_params() -> None:
    preset = RenderPreset(
        resolution_policy=ResolutionPolicy.FIXED_ASPECT, aspect=(1, 1)
    )
    with pytest.raises(ValueError):
        resolve_dims(preset, (256, 256))


def test_fixed_dims_passthrough_with_alignment() -> None:
    preset = RenderPreset(
        resolution_policy=ResolutionPolicy.FIXED_DIMS, target_w=200, target_h=100
    )
    assert resolve_dims(preset, (640, 480)) == (208, 112)


def test_fixed_dims_requires_params() -> None:
    preset = RenderPreset(resolution_policy=ResolutionPolicy.FIXED_DIMS, target_w=200)
    with pytest.raises(ValueError):
        resolve_dims(preset, (256, 256))
