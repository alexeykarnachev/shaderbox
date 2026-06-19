"""The shared render-size vocabulary (RenderShape) and its lowering to a RenderPreset. RenderShape is
the ONE named-tier home the Share UI + copilot tools both speak; shape_to_preset is the single resolver
that turns a named tier into the transient RenderPreset the render funnel consumes."""

from shaderbox.render_preset import ResolutionPolicy, resolve_dims
from shaderbox.render_shape import (
    MENU_SHAPES,
    SHAPE_TABLE,
    RenderShape,
    is_short,
    shape_to_preset,
)


def test_every_member_is_in_the_table() -> None:
    # SHAPE_TABLE is the single source of truth — no member may lack a spec.
    assert set(SHAPE_TABLE) == set(RenderShape)


def test_native_lowers_to_free() -> None:
    preset = shape_to_preset(
        RenderShape.NATIVE, is_video=False, fps=None, container=None, duration_max=None
    )
    assert preset.resolution_policy is ResolutionPolicy.FREE
    # FREE renders at the source canvas size (a 16-aligned source passes through untouched).
    assert resolve_dims(preset, (1024, 768)) == (1024, 768)


def test_short_is_9_16_at_its_longest_edge() -> None:
    preset = shape_to_preset(
        RenderShape.SHORT_1080,
        is_video=True,
        fps=30,
        container=".mp4",
        duration_max=60.0,
    )
    assert preset.resolution_policy is ResolutionPolicy.FIXED_ASPECT
    assert preset.aspect == (9, 16)
    w, h = resolve_dims(preset, (100, 100))  # source ignored for FIXED_ASPECT
    assert h == 1920  # 9:16, longest edge (height) 1920
    assert w == 1088  # 1080 rounded up to the 16px codec alignment


def test_wide_is_16_9_at_its_longest_edge() -> None:
    preset = shape_to_preset(
        RenderShape.WIDE_720, is_video=True, fps=30, container=".mp4", duration_max=None
    )
    assert preset.aspect == (16, 9)
    w, h = resolve_dims(preset, (100, 100))
    assert w == 1280 and h == 720  # 16:9, longest edge (width) 1280


def test_is_short_groups() -> None:
    assert is_short(RenderShape.SHORT_720)
    assert not is_short(RenderShape.WIDE_1080)
    assert not is_short(RenderShape.NATIVE)


def test_menu_shapes_covers_every_member_with_native_first() -> None:
    # The Share-tab combo lists every shape exactly once; NATIVE leads (the safe default).
    assert set(MENU_SHAPES) == set(RenderShape)
    assert len(MENU_SHAPES) == len(RenderShape)
    assert MENU_SHAPES[0] is RenderShape.NATIVE


def test_every_member_has_a_distinct_menu_label() -> None:
    labels = [SHAPE_TABLE[s].menu_label for s in MENU_SHAPES]
    assert len(set(labels)) == len(labels)  # no ambiguous duplicate in a flat combo


def test_short_edges_match_the_pre_refactor_values() -> None:
    # Byte-identical Shorts output: the three longest edges that were SHORT_RES_PRESETS.
    assert SHAPE_TABLE[RenderShape.SHORT_720].longest_edge == 1280
    assert SHAPE_TABLE[RenderShape.SHORT_1080].longest_edge == 1920
    assert SHAPE_TABLE[RenderShape.SHORT_1440].longest_edge == 2560
