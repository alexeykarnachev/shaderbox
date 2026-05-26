from collections.abc import Callable
from dataclasses import dataclass

import pyperclip
from imgui_bundle import imgui, imgui_ctx
from loguru import logger

from shaderbox.theme import COLOR, OVERLAY_ALPHA, SIZE, SPACE, fade

_TRANSPARENT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def _ellipsize(text: str, max_width: float) -> str:
    if imgui.calc_text_size(text).x <= max_width:
        return text
    ellipsis = "..."
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if imgui.calc_text_size(text[:mid] + ellipsis).x <= max_width:
            lo = mid
        else:
            hi = mid - 1
    if lo > 0:
        return text[:lo] + ellipsis
    return ellipsis if imgui.calc_text_size(ellipsis).x <= max_width else ""


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


def chip_button(
    label: str,
    width: float = 0.0,
    height: float = 0.0,
    disabled: bool = False,
    faded: bool = False,
) -> bool:
    """A rounded pill (input-type selector, the FPS overlay). `faded` makes the
    base semi-transparent for overlays drawn over the render image."""
    base = fade(COLOR.CHIP_BG, OVERLAY_ALPHA) if faded else COLOR.CHIP_BG
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, float(SIZE.CHIP_ROUNDING))
    imgui.push_style_color(imgui.Col_.button, base)
    imgui.push_style_color(imgui.Col_.button_hovered, COLOR.CHIP_BG_HOVER)
    imgui.push_style_color(imgui.Col_.button_active, COLOR.CHIP_BG_HOVER)
    imgui.push_style_color(imgui.Col_.text, COLOR.CHIP_FG)
    if disabled:
        imgui.begin_disabled()
    clicked: bool = imgui.button(label, size=(width, height))
    if disabled:
        imgui.end_disabled()
    imgui.pop_style_color(4)
    imgui.pop_style_var()
    return clicked


def centered_image(
    texture_glo: int, size: tuple[int, int], box_w: float, box_h: float
) -> None:
    """Letterbox a texture centered in a `box_w x box_h` cell (preview panels).

    Fits the texture into the box preserving aspect, then offsets the cursor to
    center it. Uses normal-flow `imgui.image` (cursor-positioned), distinct from
    `preview_cell`'s draw-list path."""
    w, h = size
    if w <= 0 or h <= 0:
        return
    scale: float = min(box_w / w, box_h / h)
    dw, dh = w * scale, h * scale
    origin = imgui.get_cursor_pos()
    imgui.set_cursor_pos((origin.x + (box_w - dw) / 2, origin.y + (box_h - dh) / 2))
    imgui.image(
        imgui.ImTextureRef(texture_glo),
        image_size=(dw, dh),
        uv0=(0, 1),
        uv1=(1, 0),
    )


def caption_text(
    text: str, color: tuple[float, float, float, float] | None = None
) -> None:
    """Small, dim, secondary annotation (artifact stats, hints)."""
    imgui.text_colored(color or COLOR.FG_DIM, text)


def wrapped_caption(
    text: str, color: tuple[float, float, float, float] | None = None
) -> None:
    """A `caption_text` that wraps at the window's right edge (multi-line hints).

    `imgui.text_colored` never wraps; push a wrap position at x=0 (= the content
    region's right edge) so a long instruction line folds instead of clipping.
    """
    imgui.push_text_wrap_pos(0.0)
    imgui.text_colored(color or COLOR.FG_DIM, text)
    imgui.pop_text_wrap_pos()


def small_caption(font: imgui.ImFont, text: str) -> None:
    """Dim caption in a smaller font (column labels, inline readouts).

    `font.legacy_size` is the rasterized size push_font wants (conventions.md
    ## Known quirks).
    """
    imgui.push_font(font, font.legacy_size)
    imgui.text_colored(COLOR.FG_DIM, text)
    imgui.pop_font()


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


def cell_delete_confirm(origin: imgui.ImVec2, avail: imgui.ImVec2) -> bool | None:
    """`Delete?` + [Yes][No] drawn over a grid cell, dimming its content.

    Positions absolutely within the caller's cell child. Returns True on Yes,
    False on No, None while still armed. Caller owns the armed state.
    """
    dl = imgui.get_window_draw_list()
    dl.add_rect_filled(
        (origin.x, origin.y),
        (origin.x + avail.x, origin.y + avail.y),
        imgui.color_convert_float4_to_u32(fade(COLOR.STATE_ERROR, 0.45)),
    )
    prompt = "Delete?"
    pw = imgui.calc_text_size(prompt)
    imgui.set_cursor_screen_pos(
        (origin.x + (avail.x - pw.x) / 2, origin.y + avail.y * 0.28)
    )
    imgui.text_colored(COLOR.FG_TITLE, prompt)

    # Neutral filled buttons read clearly on the red wash (red text would not).
    btn_w: float = (avail.x - 3 * SPACE.SM) / 2
    row_y: float = origin.y + avail.y * 0.55
    imgui.set_cursor_screen_pos((origin.x + SPACE.SM, row_y))
    if button("Yes", width=btn_w):
        return True
    imgui.set_cursor_screen_pos((origin.x + SPACE.SM + btn_w + SPACE.SM, row_y))
    if button("No", width=btn_w):
        return False
    return None


@dataclass
class PreviewCellResult:
    clicked: bool = False  # whole-cell click target hit
    delete_armed: bool = False  # the delete-✕ was pressed this frame
    delete_confirmed: bool = False  # `Yes` on the in-cell confirm wash
    delete_cancelled: bool = False  # `No` on the in-cell confirm wash


def preview_cell(
    id_: str,
    cell_w: float,
    texture_glo: int | None,
    texture_size: tuple[int, int],
    selected: bool,
    armed: bool,
    border_color: tuple[float, float, float, float] | None = None,
    bg_color: tuple[float, float, float, float] | None = None,
    footer: str = "",
    overlay: Callable[[float], None] | None = None,
) -> PreviewCellResult:
    """A bordered preview tile: a `cell_w`-wide square image + whole-cell click
    target + selection border + a top-right delete-✕ arming an in-cell `Delete?` wash.

    The cell is `cell_w` wide and grows below the image by exactly one text line when
    `footer` is set; the caller sizes only the width. `overlay` draws an extra top-LEFT
    control (e.g. an emoji button), receiving the standard overlay-button side; it's
    shown alongside the delete-✕ only while `selected` and not `armed`. The whole
    tile is its own child window so the overlays' absolute cursor moves can't perturb
    the parent (no jitter / SetCursorPos assert).
    """
    footer_h: float = imgui.get_text_line_height_with_spacing() if footer else 0.0
    cell_h: float = cell_w + footer_h
    result = PreviewCellResult()
    n_styles = 0
    if border_color is not None:
        imgui.push_style_color(imgui.Col_.border, border_color)
        n_styles += 1
    if bg_color is not None:
        imgui.push_style_color(imgui.Col_.child_bg, bg_color)
        n_styles += 1
    with imgui_ctx.begin_child(
        f"##preview_cell_{id_}",
        size=imgui.ImVec2(cell_w, cell_h),
        child_flags=imgui.ChildFlags_.borders,
        window_flags=imgui.WindowFlags_.no_scrollbar
        | imgui.WindowFlags_.no_scroll_with_mouse,
    ):
        imgui.pop_style_color(n_styles)
        origin = imgui.get_cursor_screen_pos()
        avail = imgui.get_content_region_avail()
        dl = imgui.get_window_draw_list()
        img_h: float = avail.y - footer_h  # footer occupies the last line

        if texture_glo is not None and min(texture_size) > 0:
            tw, th = texture_size
            scale: float = min(avail.x / tw, img_h / th)
            dw, dh = tw * scale, th * scale
            ix: float = origin.x + (avail.x - dw) / 2
            iy: float = origin.y + (img_h - dh) / 2
            dl.add_image(
                imgui.ImTextureRef(texture_glo),
                (ix, iy),
                (ix + dw, iy + dh),
                (0, 1),
                (1, 0),
            )

        imgui.set_next_item_allow_overlap()
        if imgui.invisible_button(f"##cell_{id_}", size=(avail.x, img_h)):
            result.clicked = True

        if footer:
            label: str = _ellipsize(footer, avail.x)
            fw = imgui.calc_text_size(label)
            fy: float = origin.y + img_h
            imgui.set_cursor_screen_pos((origin.x + (avail.x - fw.x) / 2, fy))
            imgui.text(label)

        if selected and armed:
            choice: bool | None = cell_delete_confirm(
                origin, imgui.ImVec2(avail.x, img_h)
            )
            if choice is True:
                result.delete_confirmed = True
            elif choice is False:
                result.delete_cancelled = True
        elif selected:
            x_side: float = float(SIZE.ROW_HEIGHT)
            if overlay is not None:
                imgui.set_next_item_allow_overlap()
                imgui.set_cursor_screen_pos((origin.x, origin.y))
                overlay(x_side)
            imgui.set_cursor_screen_pos((origin.x + avail.x - x_side, origin.y))
            imgui.set_next_item_allow_overlap()
            if close_cross_button(f"del_{id_}", x_side):
                result.delete_armed = True
            if imgui.is_item_hovered():
                imgui.set_tooltip("Delete")
    return result


def row_label(
    font: imgui.ImFont, label: str, label_w: float = float(SIZE.LABEL_W)
) -> None:
    """Draw a small-font dim label column and leave the cursor on the same line at
    the control column. Caller draws its widget(s) next."""
    imgui.align_text_to_frame_padding()
    small_caption(font, label)
    imgui.same_line(label_w + SPACE.MD)


def label_row(
    font: imgui.ImFont,
    label: str,
    item_width: float,
    label_w: float = float(SIZE.LABEL_W),
) -> None:
    """A `row_label` plus `set_next_item_width` for the caller's single widget,
    drawn immediately after. The caller passes a `##`-only id."""
    row_label(font, label, label_w)
    imgui.set_next_item_width(item_width)


def duration_drag(
    label: str,
    value: float,
    v_max: float,
    width: float,
    v_min: float = 0.5,
    fmt: str = "%.1f s",
    label_w: float = float(SIZE.LABEL_W),
) -> float:
    """A labelled numeric drag (full-size caption label + drag). Double-click to
    type an exact value. Returns the new value.

    Distinct from `label_row` (which takes a small-caption font): this is used by
    the exporter panels, which have no `App` handle to reach `font_12`.
    """
    imgui.align_text_to_frame_padding()
    caption_text(label)
    imgui.same_line(label_w + SPACE.MD)
    imgui.set_next_item_width(width)
    return imgui.drag_float(f"##{label}", value, 0.1, v_min, v_max, fmt)[1]


def fps_overlay(
    anchor_x: float,
    anchor_y: float,
    fps: int,
    target_fps: int,
    is_open: bool,
) -> bool:
    """A clickable FPS chip pinned to the top-right of a region, optionally
    unfolding a stats panel beneath it.

    `anchor_x` / `anchor_y` are the top-RIGHT screen corner of the region the
    overlay hugs (the pill's right edge sits `inset` left of `anchor_x`). Returns
    the new open state (toggled on a pill click). The pill is anchored in screen
    space independent of the detail panel, so opening it never shifts the pill.
    """
    label = f"{fps} FPS"
    pad: float = float(SPACE.MD)
    pill_w = imgui.calc_text_size(label).x + 2.0 * pad
    pill_h = imgui.get_frame_height()
    inset: float = float(SPACE.MD)

    pill_x = anchor_x - pill_w - inset
    pill_y = anchor_y + inset

    imgui.set_cursor_screen_pos((pill_x, pill_y))
    clicked: bool = chip_button(label, pill_w, pill_h, faded=True)

    if is_open:
        panel_w = float(SIZE.FPS_PANEL_W)
        imgui.set_cursor_screen_pos((anchor_x - panel_w - inset, pill_y + pill_h))
        imgui.push_style_color(imgui.Col_.child_bg, fade(COLOR.BG_POPUP, OVERLAY_ALPHA))
        with imgui_ctx.begin_child(
            "fps_details",
            size=imgui.ImVec2(panel_w, 0.0),
            child_flags=imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y,
            window_flags=imgui.WindowFlags_.no_scrollbar,
        ):
            caption_text(f"Current: {fps} FPS")
            caption_text(f"Target:  {target_fps} FPS")
        imgui.pop_style_color(1)

    return not is_open if clicked else is_open


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
