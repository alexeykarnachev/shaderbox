"""Pixel primitives for judging dogfood renders (feature 057).

NUMBERS OUT, never a verdict: every function returns a measurement (a centroid, a diff, a run
list) and nothing asserts — the judgment stays with the human reading the report. PIL + numpy
only, so importing this opens no GL context and needs no display.

"Bright" throughout means the pixel's MAX channel is at or above `thresh` — a saturated red ball
counts as bright, which a luminance test would miss.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

_DEFAULT_THRESH = 170


def load_rgb(path: str | Path) -> np.ndarray:
    """An (H, W, 3) int16 array — signed and wide so a channel difference never wraps."""
    with PILImage.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.int16)


def grid_cell(im: np.ndarray, row: int, col: int, rows: int, cols: int) -> np.ndarray:
    """The (row, col) cell of an evenly-split rows x cols grid, as a VIEW (row 0 = top)."""
    h, w = int(im.shape[0]), int(im.shape[1])
    y0, y1 = row * h // rows, (row + 1) * h // rows
    x0, x1 = col * w // cols, (col + 1) * w // cols
    return im[y0:y1, x0:x1]


def region_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute per-channel difference of two same-shape regions (0 = identical)."""
    return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())


def bright_centroid(
    im: np.ndarray, thresh: int = _DEFAULT_THRESH
) -> tuple[float, float] | None:
    """(x, y) pixel centroid of the bright pixels; None when nothing is bright."""
    ys, xs = np.nonzero(im.max(axis=2) >= thresh)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def color_mask_centroid(
    im: np.ndarray, rgb_min: tuple[int, int, int], rgb_max: tuple[int, int, int]
) -> tuple[float, float] | None:
    """(x, y) centroid of the pixels whose every channel is within [rgb_min, rgb_max]."""
    lo = np.array(rgb_min, dtype=np.int16)
    hi = np.array(rgb_max, dtype=np.int16)
    ys, xs = np.nonzero(np.all((im >= lo) & (im <= hi), axis=2))
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def column_runs(im: np.ndarray, thresh: int = _DEFAULT_THRESH) -> list[tuple[int, int]]:
    """Inclusive (x_start, x_end) x-extents of the contiguous column bands holding bright pixels.

    The circle-counter: N separated blobs on one row yield N runs.
    """
    on = np.any(im.max(axis=2) >= thresh, axis=0)
    runs: list[tuple[int, int]] = []
    start = -1
    for x in range(int(on.shape[0])):
        if bool(on[x]):
            if start < 0:
                start = x
        elif start >= 0:
            runs.append((start, x - 1))
            start = -1
    if start >= 0:
        runs.append((start, int(on.shape[0]) - 1))
    return runs


def farthest_bright_angle(
    im: np.ndarray, thresh: int = _DEFAULT_THRESH
) -> float | None:
    """Degrees from the image center to the FARTHEST bright pixel; None when nothing is bright.

    Image coordinates (y grows DOWNWARD), so a rising angle over a strip means clockwise
    on screen. Range (-180, 180]; 0 = due right.
    """
    ys, xs = np.nonzero(im.max(axis=2) >= thresh)
    if xs.size == 0:
        return None
    cy = (int(im.shape[0]) - 1) / 2.0
    cx = (int(im.shape[1]) - 1) / 2.0
    dx = xs.astype(np.float64) - cx
    dy = ys.astype(np.float64) - cy
    far = int(np.argmax(dx * dx + dy * dy))
    return math.degrees(math.atan2(float(dy[far]), float(dx[far])))
