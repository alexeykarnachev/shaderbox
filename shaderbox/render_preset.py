from enum import StrEnum

from pydantic import BaseModel

from shaderbox.constants import VIDEO_RESOLUTION_ALIGNMENT


class ResolutionPolicy(StrEnum):
    FREE = "free"
    LONGEST_EDGE = "longest_edge"
    FIXED_ASPECT = "fixed_aspect"
    FIXED_DIMS = "fixed_dims"


class FitPolicy(StrEnum):
    RENDER_AT_TARGET = "render_at_target"
    SCALE_DISTORT = "scale_distort"


class RenderPreset(BaseModel):
    is_video: bool | None = None
    fps: int | None = None
    duration_max: float | None = None
    container: str | None = None
    resolution_policy: ResolutionPolicy = ResolutionPolicy.FREE
    longest_edge: int | None = None
    aspect: tuple[int, int] | None = None
    target_w: int | None = None
    target_h: int | None = None
    fit: FitPolicy = FitPolicy.SCALE_DISTORT
    max_bytes: int | None = None


def _align(value: int, alignment: int = VIDEO_RESOLUTION_ALIGNMENT) -> int:
    return max(alignment, (value + alignment - 1) // alignment * alignment)


def resolve_dims(preset: RenderPreset, source_size: tuple[int, int]) -> tuple[int, int]:
    src_w, src_h = source_size
    policy: ResolutionPolicy = preset.resolution_policy

    if policy is ResolutionPolicy.FIXED_DIMS:
        if preset.target_w is None or preset.target_h is None:
            raise ValueError("FIXED_DIMS requires target_w and target_h")
        w, h = preset.target_w, preset.target_h

    elif policy is ResolutionPolicy.FIXED_ASPECT:
        if preset.aspect is None or preset.longest_edge is None:
            raise ValueError("FIXED_ASPECT requires aspect and longest_edge")
        aw, ah = preset.aspect
        if aw >= ah:
            w = preset.longest_edge
            h = round(w * ah / aw)
        else:
            h = preset.longest_edge
            w = round(h * aw / ah)

    elif policy is ResolutionPolicy.LONGEST_EDGE:
        if preset.longest_edge is None:
            raise ValueError("LONGEST_EDGE requires longest_edge")
        edge: int = preset.longest_edge
        if src_w >= src_h:
            w = min(edge, src_w)
            h = round(w * src_h / src_w)
        else:
            h = min(edge, src_h)
            w = round(h * src_w / src_h)

    else:
        w, h = src_w, src_h

    return _align(w), _align(h)
