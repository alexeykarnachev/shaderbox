import pyperclip
from imgui_bundle import imgui
from loguru import logger

from shaderbox.theme import COLOR, SIZE, SPACE

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


def duration_slider(
    label: str,
    value: float,
    v_max: float,
    width: float,
    v_min: float = 0.5,
    fmt: str = "%.1f s",
    label_w: float = float(SIZE.LABEL_W),
) -> float:
    """A labelled numeric slider (label column + slider). Returns the new value."""
    imgui.align_text_to_frame_padding()
    caption_text(label)
    imgui.same_line(label_w + SPACE.MD)
    imgui.set_next_item_width(width)
    return imgui.slider_float(f"##{label}", value, v_min, v_max, fmt)[1]


def draw_copyable_text(
    label: str,
    copy_value: str | None = None,
    color: tuple[float, float, float, float] | None = None,
) -> bool:
    """Click-to-copy text (the editor file-path / a share link share this).

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
