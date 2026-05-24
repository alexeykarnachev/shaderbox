import hashlib
import math
from collections.abc import Sequence
from typing import Any, TypeVar

import moderngl

K = TypeVar("K")


def adjust_size(
    size: tuple[int, int],
    width: int | None = None,
    height: int | None = None,
    aspect: float | None = None,
    max_size: int | None = None,
) -> tuple[int, int]:
    if (width, height, aspect, max_size).count(None) != 3:
        return size

    original_width, original_height = size

    if width is not None:
        new_height = round(width * original_height / original_width)
        return (width, new_height)
    elif height is not None:
        new_width = round(height * original_width / original_height)
        return (new_width, height)
    elif aspect is not None:
        current_aspect = original_width / original_height
        if aspect > current_aspect:
            new_width = round(original_height * aspect)
            return (new_width, original_height)
        else:
            new_height = round(original_width / aspect)
            return (original_width, new_height)
    elif max_size is not None:
        if original_width >= original_height:
            new_width = max_size
            new_height = round(max_size * original_height / original_width)
            return (new_width, new_height)
        else:
            new_height = max_size
            new_width = round(max_size * original_width / original_height)
            return (new_width, new_height)
    else:
        return size


def select_next_value(
    values: Sequence[K],
    current_value: K | None,
    default_value: K,
    step: int = 1,
) -> K:
    if not values:
        return default_value

    idx = (
        0
        if not current_value or current_value not in values
        else values.index(current_value)
    )

    return values[(idx + step) % len(values)]


def get_resolution_str(name: str | None, w: int, h: int) -> str:
    g = math.gcd(w, h)
    w_ratio, h_ratio = w // g, h // g
    aspect = f"{w_ratio}:{h_ratio}"
    parts = [f"{w}x{h}", aspect]
    if name:
        parts.append(name)
    return " | ".join(parts)


def get_uniform_hash(u: moderngl.Uniform | moderngl.UniformBlock) -> int:
    if isinstance(u, moderngl.Uniform):
        key = f"{u.name}_{u.array_length}_{u.dimension}_{u.gl_type}"  # type: ignore
    else:
        key = f"{u.name}_{u.size}"

    hash = hashlib.md5(key.encode()).digest()
    return int.from_bytes(hash, "big")


def unicode_to_str(char_inds: list[int]) -> str:
    eos_idx = char_inds.index(0)
    chars = []
    for i in range(0, eos_idx):
        chars.append(chr(char_inds[i]))
    text = "".join(chars)

    return text


def str_to_unicode(text: str, max_n_chars: int) -> list[int]:
    if len(text) >= max_n_chars:
        text = text[:max_n_chars]
    char_inds = [ord(c) for c in text]
    pad_len = max_n_chars - len(char_inds)
    char_inds += [0] * pad_len

    return char_inds


def try_to_release(value: Any) -> bool:
    if release := getattr(value, "release", None):
        release()
        return True
    return False


def pfd_block(dialog: Any) -> Any:
    while not dialog.ready(20):
        pass
    return dialog.result()
