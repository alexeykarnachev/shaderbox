from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from shaderbox.render_preset import FitPolicy, RenderPreset, ResolutionPolicy

# The ONE named-size vocabulary the human Share UI and the copilot tools both speak. A member names a
# quality tier with its aspect baked in (a "Short at 1080p" is one token, so a copilot render matches
# what publish emits). It lowers into a RenderPreset via shape_to_preset — never a persisted field,
# never a RenderPreset member; RenderPreset stays the transient resolved form one layer below.


class RenderShape(StrEnum):
    NATIVE = "native"  # FREE -> the node's canvas size, any aspect
    SHORT_720 = "short_720"  # 9:16, longest edge 1280
    SHORT_1080 = "short_1080"  # 9:16, longest edge 1920
    SHORT_1440 = "short_1440"  # 9:16, longest edge 2560
    WIDE_720 = "wide_720"  # 16:9, longest edge 1280
    WIDE_1080 = "wide_1080"  # 16:9, longest edge 1920
    WIDE_1440 = "wide_1440"  # 16:9, longest edge 2560


ShapeGroup = Literal["native", "short", "wide"]


@dataclass(frozen=True)
class ShapeSpec:
    menu_label: str  # the picker entry ("Native (canvas)" / "Short 1080p" / ...)
    group: ShapeGroup
    aspect: tuple[int, int] | None  # None => FREE (NATIVE)
    longest_edge: int | None


SHAPE_TABLE: dict[RenderShape, ShapeSpec] = {
    RenderShape.NATIVE: ShapeSpec("Native (canvas)", "native", None, None),
    RenderShape.SHORT_720: ShapeSpec("Short 720p (9:16)", "short", (9, 16), 1280),
    RenderShape.SHORT_1080: ShapeSpec("Short 1080p (9:16)", "short", (9, 16), 1920),
    RenderShape.SHORT_1440: ShapeSpec("Short 1440p (9:16)", "short", (9, 16), 2560),
    RenderShape.WIDE_720: ShapeSpec("Wide 720p (16:9)", "wide", (16, 9), 1280),
    RenderShape.WIDE_1080: ShapeSpec("Wide 1080p (16:9)", "wide", (16, 9), 1920),
    RenderShape.WIDE_1440: ShapeSpec("Wide 1440p (16:9)", "wide", (16, 9), 2560),
}

# Picker order for the Share-tab resolution combo (native first, then shorts, then wide).
MENU_SHAPES: list[RenderShape] = [
    RenderShape.NATIVE,
    RenderShape.SHORT_720,
    RenderShape.SHORT_1080,
    RenderShape.SHORT_1440,
    RenderShape.WIDE_720,
    RenderShape.WIDE_1080,
    RenderShape.WIDE_1440,
]


def is_short(shape: RenderShape) -> bool:
    return SHAPE_TABLE[shape].group == "short"


def shape_to_preset(
    shape: RenderShape,
    *,
    is_video: bool,
    fps: int | None,
    container: str | None,
    duration_max: float | None,
) -> RenderPreset:
    # Lower a named shape to a transient RenderPreset. The shape owns ONLY size + aspect; fps /
    # container / duration_max are per-outlet facts the caller supplies.
    spec: ShapeSpec = SHAPE_TABLE[shape]
    if spec.aspect is None:
        return RenderPreset(
            is_video=is_video,
            fps=fps,
            container=container,
            duration_max=duration_max,
            resolution_policy=ResolutionPolicy.FREE,
            fit=FitPolicy.RENDER_AT_TARGET,
        )
    return RenderPreset(
        is_video=is_video,
        fps=fps,
        container=container,
        duration_max=duration_max,
        resolution_policy=ResolutionPolicy.FIXED_ASPECT,
        aspect=spec.aspect,
        longest_edge=spec.longest_edge,
        fit=FitPolicy.RENDER_AT_TARGET,
    )
