import hashlib
import math
from collections.abc import Sequence
from typing import Any, TypeVar

import moderngl
import pyperclip
from imgui_bundle import imgui
from loguru import logger

from shaderbox.theme import COLOR, SIZE, SPACE

K = TypeVar("K")


_TRANSPARENT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Button system — four tiers. Pick by the action's role, never by how it looks:
#   primary_button — the ONE call-to-action of a section (filled accent).
#   button         — an ordinary action (filled neutral grey, the imgui default).
#   ghost_button   — low-emphasis / secondary (text only, transparent).
#   danger_button  — destructive (text only, red); confirm chrome carries the weight.
# ---------------------------------------------------------------------------


def primary_button(label: str, width: float = 0.0) -> bool:
    imgui.push_style_color(imgui.Col_.button, COLOR.ACCENT_PRIMARY)
    imgui.push_style_color(imgui.Col_.button_hovered, COLOR.ACCENT_ACTIVE)
    imgui.push_style_color(imgui.Col_.button_active, COLOR.ACCENT_ACTIVE)
    imgui.push_style_color(imgui.Col_.text, COLOR.BG_APP)
    clicked: bool = imgui.button(label, size=(width, 0.0))
    imgui.pop_style_color(4)
    return clicked


def button(label: str, width: float = 0.0) -> bool:
    return imgui.button(label, size=(width, 0.0))


def ghost_button(label: str, width: float = 0.0) -> bool:
    imgui.push_style_color(imgui.Col_.button, _TRANSPARENT)
    imgui.push_style_color(imgui.Col_.button_hovered, COLOR.BG_FRAME)
    imgui.push_style_color(imgui.Col_.button_active, COLOR.BORDER)
    imgui.push_style_color(imgui.Col_.text, COLOR.FG_SECONDARY)
    clicked: bool = imgui.button(label, size=(width, 0.0))
    imgui.pop_style_color(4)
    return clicked


def danger_button(label: str, width: float = 0.0) -> bool:
    imgui.push_style_color(imgui.Col_.button, _TRANSPARENT)
    imgui.push_style_color(imgui.Col_.button_hovered, COLOR.BG_FRAME)
    imgui.push_style_color(imgui.Col_.button_active, COLOR.BORDER)
    imgui.push_style_color(imgui.Col_.text, COLOR.STATE_ERROR)
    clicked: bool = imgui.button(label, size=(width, 0.0))
    imgui.pop_style_color(4)
    return clicked


def caption_text(
    text: str, color: tuple[float, float, float, float] | None = None
) -> None:
    """Small, dim, secondary annotation (artifact stats, hints)."""
    imgui.text_colored(color or COLOR.FG_DIM, text)


def close_cross_button(id_: str, side: float) -> bool:
    """A red square with a crisp drawn ✕ — overlay close/delete affordance.

    The glyph is two draw-list lines (no font dependency / baseline guesswork),
    so it is always perfectly centred. Returns True on click.
    """
    origin = imgui.get_cursor_screen_pos()
    imgui.push_style_color(imgui.Col_.button, COLOR.STATE_ERROR)
    imgui.push_style_color(imgui.Col_.button_hovered, COLOR.STATE_ERROR)
    imgui.push_style_color(imgui.Col_.button_active, COLOR.STATE_ERROR)
    clicked: bool = imgui.button(f"##{id_}", size=(side, side))
    imgui.pop_style_color(3)
    pad: float = side * 0.3
    col = imgui.color_convert_float4_to_u32(COLOR.FG_TITLE)
    dl = imgui.get_window_draw_list()
    a = (origin.x + pad, origin.y + pad)
    b = (origin.x + side - pad, origin.y + side - pad)
    c = (origin.x + side - pad, origin.y + pad)
    d = (origin.x + pad, origin.y + side - pad)
    dl.add_line(a, b, col, 1.5)
    dl.add_line(c, d, col, 1.5)
    return clicked


def duration_slider(label: str, value: float, v_max: float, width: float) -> float:
    """A labelled seconds slider (label column + slider). Returns the new value."""
    imgui.align_text_to_frame_padding()
    caption_text(label)
    imgui.same_line(float(SIZE.LABEL_W) + SPACE.MD)
    imgui.set_next_item_width(width)
    return imgui.slider_float(f"##{label}", value, 0.5, v_max, "%.1f s")[1]


def draw_copyable_text(
    label: str,
    copy_value: str | None = None,
    color: tuple[float, float, float, float] | None = None,
) -> bool:
    """Click-to-copy text (the editor file-path / the pack link share this).

    Copies `copy_value` (defaults to `label`) to the clipboard on click; returns
    True iff the copy succeeded. Caller decides whether to surface a notification.
    """
    imgui.push_style_color(imgui.Col_.text, color or COLOR.FG_DIM)
    clicked: bool = imgui.selectable(
        label, False, size=(imgui.calc_text_size(label).x, 0)
    )[0]
    imgui.pop_style_color(1)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Click to copy")
    if not clicked:
        return False
    try:
        pyperclip.copy(copy_value if copy_value is not None else label)
        return True
    except pyperclip.PyperclipException:
        logger.warning("No clipboard backend (install xclip or xsel)")
        return False


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
