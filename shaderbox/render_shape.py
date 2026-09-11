from dataclasses import dataclass
from enum import StrEnum
from math import gcd
from typing import Literal

from shaderbox.render_preset import FitPolicy, RenderPreset, ResolutionPolicy

# The ONE named-size vocabulary the human Share UI and the copilot tools both speak. A member names a
# quality tier with its aspect baked in (a "Short at 1080p" is one token, so a copilot render matches
# what publish emits). It lowers into a RenderPreset via shape_to_preset — never a persisted field,
# never a RenderPreset member; RenderPreset stays the transient resolved form one layer below.


class ResolutionMode(StrEnum):
    """How a document's live render size is decided (090 D1, revision 1).

    AUTO: the document stores an ASPECT and no size at all; its live canvas is the viewer
    region fitted to that aspect, and it follows every window and panel resize. FIXED: the
    document stores a `resolution` pair and that pair IS the live size, which is what it always
    was. The pair is meaningful only under FIXED; the aspect only under AUTO.
    """

    AUTO = "auto"
    FIXED = "fixed"


# What a new document is shaped like before anyone chooses. 16:9 because a shader is written to
# be looked at and then posted, and both ends of that are wide.
DEFAULT_ASPECT: tuple[int, int] = (16, 9)

# The aspects the Document tab offers as chips, in the order they are drawn: wide to tall, with
# square in the middle, so the row reads as one axis rather than a set.
ASPECT_PRESETS: tuple[tuple[int, int], ...] = (
    (21, 9),
    (16, 9),
    (4, 3),
    (1, 1),
    (3, 4),
    (9, 16),
)

# A ratio's two numbers are bounded because they live in `document.json`, where nothing type
# checks them, and because the UI's own fields must not be able to write something the model
# would reject on the next load.
MIN_ASPECT_TERM: int = 1
MAX_ASPECT_TERM: int = 1000


def reduce_aspect(ratio: tuple[int, int]) -> tuple[int, int]:
    """`ratio` in lowest terms, clamped into the model's bounds.

    Stored reduced so one shape has one spelling: 1280x720, 32:18 and 16:9 are the same aspect,
    and a document that persisted the first would compare unequal to a preset chip naming the
    third. A non-positive term degrades to 1 rather than raising -- this runs over hand-edited
    JSON and over two spin fields the user can empty.

    The bound is applied AFTER the reduction, never before: clamping first changes the ratio
    itself, which turned a 1280x960 document into 25:24 instead of 4:3.
    """
    width = max(MIN_ASPECT_TERM, ratio[0])
    height = max(MIN_ASPECT_TERM, ratio[1])
    divisor = gcd(width, height)
    width, height = width // divisor, height // divisor
    if max(width, height) <= MAX_ASPECT_TERM:
        return (width, height)
    # A ratio that survives reduction and is still enormous (a prime-ish pixel pair a user
    # typed) is scaled down to the bound and re-reduced, so the stored pair stays a ratio the
    # model accepts rather than being rejected on the next load.
    scale = max(width, height) / MAX_ASPECT_TERM
    width = max(MIN_ASPECT_TERM, round(width / scale))
    height = max(MIN_ASPECT_TERM, round(height / scale))
    divisor = gcd(width, height)
    return (width // divisor, height // divisor)


def aspect_of(size: tuple[int, int]) -> tuple[int, int]:
    """The reduced integer ratio of a pixel size -- what Fixed -> Auto seeds the aspect from."""
    return reduce_aspect(size)


# How far a stored size's ratio may sit from a preset's and still be NAMED as that preset. A
# 1920x1088 render is 16:9 to anyone looking at it -- the 8 rows are the encoder's alignment,
# and reporting `120:68` would be arithmetically right and useless.
ASPECT_SNAP_TOLERANCE: float = 0.01


def aspect_label(size: tuple[int, int]) -> str:
    """A pixel size's aspect as a reader would say it: `16:9`, `4:3`, `607:341`.

    A ratio within `ASPECT_SNAP_TOLERANCE` of a preset is named as that preset; anything else
    is its own reduced integers, which is the honest answer when the shape is genuinely not a
    standard one. The snap is what makes a 1920x1088 render read as `16:9` rather than `30:17`
    -- the eight extra rows are the encoder's alignment, not a shape anyone chose.
    """
    exact = reduce_aspect(size)
    ratio = exact[0] / exact[1]
    for preset in ASPECT_PRESETS:
        preset_ratio = preset[0] / preset[1]
        if abs(ratio - preset_ratio) <= ASPECT_SNAP_TOLERANCE * preset_ratio:
            return f"{preset[0]}:{preset[1]}"
    return f"{exact[0]}:{exact[1]}"


def aspect_ratio(aspect: tuple[int, int]) -> float:
    """The aspect as width / height, for the one place that needs a float: fitting a region."""
    width, height = reduce_aspect(aspect)
    return width / height


def fit_to_aspect(
    region: tuple[float, float], aspect: tuple[int, int]
) -> tuple[int, int]:
    """The largest pixel size of `aspect` that fits inside `region`.

    The whole of Auto sizing: the viewer hands its own region, and the document renders at
    exactly what is shown. Whichever axis runs out first decides, and the other is derived, so
    the answer carries the aspect exactly rather than the region's.
    """
    ratio = aspect_ratio(aspect)
    width, height = region
    if width <= 0.0 or height <= 0.0:
        return (MIN_ASPECT_TERM, MIN_ASPECT_TERM)
    if width / height >= ratio:
        return (max(1, round(height * ratio)), max(1, round(height)))
    return (max(1, round(width)), max(1, round(width / ratio)))


class RenderShape(StrEnum):
    NATIVE = "native"  # FREE -> the document's canvas size, any aspect
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
