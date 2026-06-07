import webbrowser
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import pyperclip
from imgui_bundle import imgui, imgui_ctx
from loguru import logger

from shaderbox.theme import COLOR, OVERLAY_ALPHA, SIZE, SPACE, fade

_REGION_OUTLINE_THICKNESS: float = 2.0


def active_region_outline() -> None:
    """Stroke the accent around the CURRENT child window (keyboard-nav active-region cue).
    Call from INSIDE the region's `begin_child` — reads the child's own window rect
    (`get_item_rect_*` after `end_child` can report a collapsed rect). Drawn on the child's
    OWN window draw list, inset by the stroke width so the clip rect can't cut it — this way
    it is z-ordered beneath any later-drawn floating window instead of punching through it
    (a foreground-list outline ignores window stacking)."""
    pos = imgui.get_window_pos()
    size = imgui.get_window_size()
    inset = _REGION_OUTLINE_THICKNESS
    imgui.get_window_draw_list().add_rect(
        (pos.x + inset, pos.y + inset),
        (pos.x + size.x - inset, pos.y + size.y - inset),
        imgui.color_convert_float4_to_u32(COLOR.ACCENT_PRIMARY),
        rounding=imgui.get_style().child_rounding,
        thickness=_REGION_OUTLINE_THICKNESS,
    )


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
# Button system — pick by the action's role, never by how it looks:
#   primary_button — the ONE call-to-action of a section (filled accent).
#   button         — an ordinary action (filled neutral grey, the imgui default).
#   ghost_button   — low-emphasis / secondary (text only, transparent).
#   toggle_button  — a stateful on/off control: filled accent when active, bordered
#                    ghost when off (same label both states).
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
    imgui.push_style_color(imgui.Col_.button, COLOR.TRANSPARENT)
    imgui.push_style_color(imgui.Col_.button_hovered, COLOR.BG_FRAME)
    imgui.push_style_color(imgui.Col_.button_active, COLOR.BORDER)
    imgui.push_style_color(imgui.Col_.text, COLOR.FG_SECONDARY)
    clicked: bool = imgui.button(label, size=(width, 0.0))
    imgui.pop_style_color(4)
    return clicked


def toggle_button(label: str, active: bool, width: float = 0.0) -> bool:
    """A stateful on/off button: filled accent when `active` (like `primary_button`),
    a bordered ghost when off. Same label both states — the style carries the state."""
    if active:
        return primary_button(label, width)
    imgui.push_style_color(imgui.Col_.button, COLOR.TRANSPARENT)
    imgui.push_style_color(imgui.Col_.button_hovered, COLOR.BG_FRAME)
    imgui.push_style_color(imgui.Col_.button_active, COLOR.BORDER)
    imgui.push_style_color(imgui.Col_.text, COLOR.FG_SECONDARY)
    imgui.push_style_color(imgui.Col_.border, COLOR.BORDER)
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 1.0)
    clicked: bool = imgui.button(label, size=(width, 0.0))
    imgui.pop_style_var()
    imgui.pop_style_color(5)
    return clicked


def danger_button(label: str, width: float = 0.0) -> bool:
    imgui.push_style_color(imgui.Col_.button, COLOR.TRANSPARENT)
    imgui.push_style_color(imgui.Col_.button_hovered, COLOR.BG_FRAME)
    imgui.push_style_color(imgui.Col_.button_active, COLOR.BORDER)
    imgui.push_style_color(imgui.Col_.text, COLOR.STATE_ERROR)
    clicked: bool = imgui.button(label, size=(width, 0.0))
    imgui.pop_style_color(4)
    return clicked


def pill_button(
    label: str,
    *,
    color: tuple[float, float, float, float],
    active: bool,
    text_color: tuple[float, float, float, float] = COLOR.BG_APP,
    inactive_alpha: float = 0.4,
) -> bool:
    """A small filled pill whose `color` is the role (a tag, a favs toggle, a reset
    action). `active` fills it solid; otherwise a faded variant signals "off". Use
    when `chip_button`'s fixed palette doesn't fit (e.g. blue tags, yellow favs)."""
    fill = color if active else fade(color, inactive_alpha)
    imgui.push_style_color(imgui.Col_.button, fill)
    imgui.push_style_color(imgui.Col_.text, text_color)
    clicked: bool = imgui.small_button(label)
    imgui.pop_style_color(2)
    return clicked


def chip_button(
    label: str,
    width: float = 0.0,
    height: float = 0.0,
    disabled: bool = False,
    faded: bool = False,
    active: bool = False,
) -> bool:
    """A rounded pill (input-type selector, the FPS overlay, a binary mode toggle).
    `faded` makes the base semi-transparent for overlays drawn over the render
    image. `active` fills it with the accent (a selected mode in a chip group)."""
    if active:
        base = COLOR.ACCENT_PRIMARY
        hover = COLOR.ACCENT_ACTIVE
        text = COLOR.BG_APP
    else:
        base = fade(COLOR.CHIP_BG, OVERLAY_ALPHA) if faded else COLOR.CHIP_BG
        hover = COLOR.CHIP_BG_HOVER
        text = COLOR.CHIP_FG
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, float(SIZE.CHIP_ROUNDING))
    imgui.push_style_color(imgui.Col_.button, base)
    imgui.push_style_color(imgui.Col_.button_hovered, hover)
    imgui.push_style_color(imgui.Col_.button_active, hover)
    imgui.push_style_color(imgui.Col_.text, text)
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

    Normal-flow `imgui.image` (cursor-positioned), distinct from `preview_cell`'s
    draw-list path."""
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


def preview_box(
    id_: str,
    texture_glo: int | None,
    texture_size: tuple[int, int],
    box_w: float,
    box_h: float,
    overlay: Callable[[imgui.ImVec2], None] | None = None,
) -> None:
    """A bordered, surface-bg, letterboxed preview tile shared by exporter share
    panels. Fixed size — never measured — so it can't jitter. `overlay` (optional)
    draws one affordance at the box's top-left, receiving the box origin."""
    imgui.push_style_color(imgui.Col_.child_bg, COLOR.BG_SURFACE)
    with imgui_ctx.begin_child(
        id_,
        size=imgui.ImVec2(box_w, box_h),
        child_flags=imgui.ChildFlags_.borders,
        window_flags=imgui.WindowFlags_.no_scrollbar,
    ):
        origin = imgui.get_cursor_screen_pos()
        if texture_glo is not None:
            avail = imgui.get_content_region_avail()
            centered_image(texture_glo, texture_size, avail.x, avail.y)
        if overlay is not None:
            imgui.set_cursor_screen_pos((origin.x, origin.y))
            overlay(origin)
    imgui.pop_style_color(1)


@contextmanager
def modal_window(label: str, size: tuple[float, float]) -> Iterator[bool]:
    """Boilerplate-free modal-popup wrapper. Caller owns the `is_X_open` flag on `App`
    (allows per-modal cleanup on close); this owns the imgui dance: open by label,
    seed size once (`Cond_.first_use_ever`), enter the popup scope, yield visibility.
    Use as:

        if not app.is_X_open:
            return
        with modal_window(LABEL, (W, H)) as visible:
            if not visible:
                return
            if not _draw_body(app):
                app.is_X_open = False
                imgui.close_current_popup()
    """
    if not imgui.is_popup_open(label):
        imgui.open_popup(label)
    imgui.set_next_window_size(imgui.ImVec2(*size), imgui.Cond_.first_use_ever)
    # Center on the viewport (pivot at the window's own center). first_use_ever so a
    # user drag persists via imgui.ini.
    center = imgui.get_main_viewport().get_center()
    imgui.set_next_window_pos(
        center, imgui.Cond_.first_use_ever, imgui.ImVec2(0.5, 0.5)
    )
    with imgui_ctx.begin_popup_modal(label) as popup:
        yield popup.visible


def rendering_overlay(text: str) -> None:
    # A centered, non-interactive cue painted one frame before the encode freezes the
    # frame loop. NOT a begin_popup_modal — it stays off the popup ID stack + the
    # popup-mutex (any_popup_open) so it never gates node rendering.
    center = imgui.get_main_viewport().get_center()
    imgui.set_next_window_pos(center, imgui.Cond_.always, imgui.ImVec2(0.5, 0.5))
    imgui.set_next_window_bg_alpha(OVERLAY_ALPHA)
    flags = (
        imgui.WindowFlags_.no_decoration
        | imgui.WindowFlags_.no_inputs
        | imgui.WindowFlags_.no_nav
        | imgui.WindowFlags_.no_saved_settings
        | imgui.WindowFlags_.always_auto_resize
    )
    with imgui_ctx.begin("##copilot_rendering", flags=flags) as window:
        if window:
            imgui.text(text)


@contextmanager
def context_menu_style() -> Iterator[None]:
    """Style overrides to make a right-click context menu visually distinct from the
    modal/popup it lives over (both use `popup_bg` by default, so they otherwise blend).
    Wrap AROUND a `begin_popup_context_item` block — styles must be pushed before
    `begin_popup_context_item` so they apply when imgui materializes the popup window:

        with context_menu_style():
            if imgui.begin_popup_context_item(...):
                ...
                imgui.end_popup()
    """
    imgui.push_style_color(imgui.Col_.popup_bg, COLOR.BG_FRAME)
    imgui.push_style_color(imgui.Col_.border, fade(COLOR.SELECT, 0.6))
    imgui.push_style_color(imgui.Col_.header_hovered, fade(COLOR.SELECT, 0.6))
    imgui.push_style_color(imgui.Col_.header_active, COLOR.SELECT)
    imgui.push_style_var(imgui.StyleVar_.popup_border_size, 2.0)
    try:
        yield
    finally:
        imgui.pop_style_var(1)
        imgui.pop_style_color(4)


@contextmanager
def status_slot(id_: str, width: float) -> Iterator[None]:
    """A fixed-height (one frame-height) borderless child the caller draws the export
    status into — fixed size so the surrounding column never changes height between
    idle/uploading/uploaded (no jitter). Use as `with status_slot(id, w):`.

    Zero WindowPadding: an 18px-tall child with the default 8px padding leaves ~2px
    of usable content — the status content must fill the frame, not be squeezed."""
    with (
        imgui_ctx.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0, 0)),
        imgui_ctx.begin_child(
            id_,
            size=imgui.ImVec2(width, imgui.get_frame_height()),
            window_flags=imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_scroll_with_mouse,
        ),
    ):
        yield


def caption_text(
    text: str, color: tuple[float, float, float, float] | None = None
) -> None:
    """Small, dim, secondary annotation (artifact stats, hints)."""
    imgui.text_colored(color or COLOR.FG_DIM, text)


# ---------------------------------------------------------------------------
# Labelled fields — a dim caption on its own line above the control. One primitive
# per control type, used by the exporter panels.
# ---------------------------------------------------------------------------


def labeled_text_input(
    label: str, value: str, width: float, password: bool = False
) -> str:
    """Caption above a single-line text input. Returns the new value."""
    caption_text(label)
    imgui.set_next_item_width(width)
    flags = imgui.InputTextFlags_.password if password else imgui.InputTextFlags_.none
    return imgui.input_text(f"##{label}", value, flags=flags)[1]


def labeled_multiline_input(label: str, value: str, width: float, height: float) -> str:
    """Caption above a multi-line text input. Returns the new value."""
    caption_text(label)
    return imgui.input_text_multiline(
        f"##{label}", value, size=imgui.ImVec2(width, height)
    )[1]


def labeled_drag_float(
    label: str,
    value: float,
    v_min: float,
    v_max: float,
    width: float,
    fmt: str = "%.1f s",
    v_speed: float = 0.1,
) -> float:
    """Caption above a numeric drag (double-click to type). Returns the new value."""
    caption_text(label)
    imgui.set_next_item_width(width)
    return imgui.drag_float(f"##{label}", value, v_speed, v_min, v_max, fmt)[1]


def labeled_combo(
    label: str, current_idx: int, items: list[str], width: float
) -> tuple[bool, int]:
    """Caption above a combo. Returns (changed, new_index)."""
    caption_text(label)
    imgui.set_next_item_width(width)
    return imgui.combo(f"##{label}", current_idx, items)


def unconnected_gate(
    not_connected_msg: str,
    hint: str,
    action_label: str,
    on_action: Callable[[], None] | None,
) -> None:
    """The shared 'not connected to <service>' panel state: a warning, a hint, and a
    primary button that opens Settings. Drawn by an exporter's draw_target_panel when
    its credentials aren't set up yet; the caller returns after."""
    imgui.text_colored(COLOR.STATE_WARN, not_connected_msg)
    caption_text(hint)
    imgui.dummy(imgui.ImVec2(0, SPACE.SM))
    if on_action is not None and primary_button(action_label):
        on_action()


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


def setup_steps(steps: list[str | tuple[str, str]]) -> None:
    """A first-run setup checklist for an integration's config UI.

    Each item is either a plain step (`"1. Do the thing"`, ghost/dim wrapped text)
    or a `(step_text, copyable_url)` tuple — the step then a click-to-copy link.
    Numbering is the caller's (in the strings)."""
    for item in steps:
        if isinstance(item, tuple):
            text, url = item
            wrapped_caption(text)
            draw_link(url)
        else:
            wrapped_caption(item)


def connection_status(
    connected: bool,
    is_error: bool,
    message: str,
    who: str = "",
    on_disconnect: Callable[[], None] | None = None,
) -> None:
    """The shared 'Connected as … / Not connected.' status line for integrations.

    One color rule for every exporter: OK when connected, ERROR on an error state,
    else WARN. `who` labels the connected identity; `message` is an optional extra
    line (auth error / hint) in the same color. When connected and `on_disconnect`
    is given, a `Disconnect` danger button sits on the same line as the status."""
    color: tuple[float, float, float, float] = (
        COLOR.STATE_OK
        if connected
        else (COLOR.STATE_ERROR if is_error else COLOR.STATE_WARN)
    )
    if connected:
        imgui.align_text_to_frame_padding()
        imgui.text_colored(color, f"Connected as {who}" if who else "Connected.")
        if on_disconnect is not None:
            imgui.same_line()
            if danger_button("Disconnect"):
                on_disconnect()
    else:
        imgui.text_colored(color, "Not connected.")
    if message:
        imgui.text_colored(color, message)


def small_caption(font: imgui.ImFont, text: str) -> None:
    """Dim caption in a smaller font (column labels, inline readouts).

    `font.legacy_size` is the rasterized size push_font wants (conventions.md
    ## Known quirks)."""
    imgui.push_font(font, font.legacy_size)
    imgui.text_colored(COLOR.FG_DIM, text)
    imgui.pop_font()


def _glyph_button(
    id_: str,
    side: float,
    base: tuple[float, float, float, float],
    hovered: tuple[float, float, float, float],
    active: tuple[float, float, float, float],
) -> tuple[bool, imgui.ImVec2]:
    """A square `side`x`side` button hosting a draw-list glyph (no label, so the caller
    paints the icon over its rect). Returns (clicked, top-left screen origin). The one
    sanctioned spot to push button colors for a glyph button — call sites don't hand-roll."""
    origin = imgui.get_cursor_screen_pos()
    imgui.push_style_color(imgui.Col_.button, base)
    imgui.push_style_color(imgui.Col_.button_hovered, hovered)
    imgui.push_style_color(imgui.Col_.button_active, active)
    clicked: bool = imgui.button(f"##{id_}", size=(side, side))
    imgui.pop_style_color(3)
    return clicked, origin


def close_cross_button(id_: str, side: float) -> bool:
    """A red square with a crisp drawn ✕ — overlay close/delete affordance.

    The glyph is two draw-list lines (no font dependency), so it's always centred.
    Returns True on click."""
    clicked, origin = _glyph_button(
        id_, side, COLOR.STATE_ERROR, COLOR.STATE_ERROR, COLOR.STATE_ERROR
    )
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


def layout_icon_button(id_: str, variant: int, side: float) -> bool:
    """A square ghost button drawn as a box-in-frame glyph showing a panel layout.

    `variant`: 0 = corner (small rect bottom-right), 1 = strip (wide rect along the
    bottom), 2 = free (centred rect). The frame is the editor area; the filled sub-rect
    is where the panel sits. No font/emoji dependency. Returns True on click."""
    clicked, origin = _glyph_button(
        id_, side, COLOR.TRANSPARENT, COLOR.BG_FRAME, COLOR.BORDER
    )
    pad: float = side * 0.25
    fx0, fy0 = origin.x + pad, origin.y + pad
    fx1, fy1 = origin.x + side - pad, origin.y + side - pad
    fw, fh = fx1 - fx0, fy1 - fy0
    dl = imgui.get_window_draw_list()
    frame = imgui.color_convert_float4_to_u32(COLOR.BORDER)
    fill = imgui.color_convert_float4_to_u32(COLOR.FG_SECONDARY)
    dl.add_rect((fx0, fy0), (fx1, fy1), frame, thickness=1.0)
    if variant == 1:  # bottom strip
        sx0, sy0 = fx0, fy1 - fh * 0.32
        sx1, sy1 = fx1, fy1
    elif variant == 2:  # free / centred
        sx0, sy0 = fx0 + fw * 0.28, fy0 + fh * 0.28
        sx1, sy1 = fx1 - fw * 0.28, fy1 - fh * 0.28
    else:  # corner (bottom-right)
        sx0, sy0 = fx0 + fw * 0.45, fy0 + fh * 0.45
        sx1, sy1 = fx1, fy1
    dl.add_rect_filled((sx0, sy0), (sx1, sy1), fill)
    return clicked


def usage_bars(
    id_: str,
    fractions: tuple[float, float],
    tooltip: str,
    width: float,
) -> None:
    """Two thin stacked fill bars (top over bottom), each `fraction` pre-clamped to
    [0, 1] by the caller, drawn vertically centred within one frame-height row. Owns the
    hit rect + the shared hover tooltip. A prototype readout — geometry, colour, and the
    tooltip all live here so richer visuals can replace it without touching the caller."""
    top_frac, bottom_frac = fractions
    bar_h: float = float(SIZE.USAGE_BAR_H)
    gap: float = float(SPACE.XS)
    stack_h: float = 2.0 * bar_h + gap
    row_h: float = imgui.get_frame_height()
    origin = imgui.get_cursor_screen_pos()
    imgui.invisible_button(f"##{id_}", imgui.ImVec2(width, row_h))
    if imgui.is_item_hovered():
        imgui.set_tooltip(tooltip)
    y0: float = origin.y + (row_h - stack_h) / 2.0
    dl = imgui.get_window_draw_list()
    track = imgui.color_convert_float4_to_u32(COLOR.BG_FRAME)
    border = imgui.color_convert_float4_to_u32(COLOR.BORDER)
    rows: list[tuple[float, tuple[float, float, float, float]]] = [
        (top_frac, COLOR.ACCENT_PRIMARY),
        (bottom_frac, COLOR.SELECT),
    ]
    for i, (frac, color) in enumerate(rows):
        by0: float = y0 + i * (bar_h + gap)
        by1: float = by0 + bar_h
        dl.add_rect_filled((origin.x, by0), (origin.x + width, by1), track)
        if frac > 0.0:
            fill = imgui.color_convert_float4_to_u32(color)
            dl.add_rect_filled((origin.x, by0), (origin.x + width * frac, by1), fill)
        dl.add_rect((origin.x, by0), (origin.x + width, by1), border, thickness=1.0)


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
    nav_flatten: bool = False,
) -> PreviewCellResult:
    """A bordered preview tile: a `cell_w`-wide square image + whole-cell click
    target + selection border + a top-right delete-✕ arming an in-cell `Delete?` wash.

    The cell is `cell_w` wide and grows below the image by one text line when `footer`
    is set; the caller sizes only the width. `overlay` draws an extra top-LEFT control,
    shown alongside the delete-✕ only while `selected` and not `armed`. The whole tile
    is its own child window so the overlays' absolute cursor moves can't perturb the
    parent (no jitter / SetCursorPos assert).

    `nav_flatten` lets keyboard-nav cross the per-tile child border so a grid traverses
    as one ring; the click target is a `selectable` (a nav stop, unlike an
    `invisible_button`) with a transparent fill so the image/border carries the visual.
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
    child_flags = imgui.ChildFlags_.borders
    if nav_flatten:
        child_flags |= imgui.ChildFlags_.nav_flattened
    with imgui_ctx.begin_child(
        f"##preview_cell_{id_}",
        size=imgui.ImVec2(cell_w, cell_h),
        child_flags=child_flags,
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

        # selectable (not invisible_button) so keyboard-nav can land on the cell;
        # transparent fill; allow_overlap so the buttons drawn on top win the click.
        imgui.push_style_color(imgui.Col_.header, COLOR.TRANSPARENT)
        imgui.push_style_color(imgui.Col_.header_hovered, COLOR.TRANSPARENT)
        imgui.push_style_color(imgui.Col_.header_active, COLOR.TRANSPARENT)
        if imgui.selectable(
            f"##cell_{id_}",
            False,
            flags=imgui.SelectableFlags_.allow_overlap,
            size=imgui.ImVec2(avail.x, img_h),
        )[0]:
            result.clicked = True
        imgui.pop_style_color(3)

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


def fps_overlay(
    anchor_x: float,
    anchor_y: float,
    fps: int,
    target_fps: int,
    is_open: bool,
) -> bool:
    """A clickable FPS chip pinned to the top-right of a region, optionally
    unfolding a stats panel beneath it.

    `anchor_x` / `anchor_y` are the top-RIGHT screen corner of the region the overlay
    hugs (the pill's right edge sits `inset` left of `anchor_x`). Returns the new open
    state (toggled on a pill click). The pill is anchored in screen space independent
    of the detail panel, so opening it never shifts the pill.
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
    tooltip: str = "Click to copy",
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
        imgui.set_tooltip(tooltip)
    if not clicked:
        return False
    try:
        pyperclip.copy(copy_value if copy_value is not None else label)
        return True
    except pyperclip.PyperclipException:
        logger.warning("No clipboard backend (install xclip or xsel)")
        return False


def draw_link(
    label: str,
    url: str | None = None,
    color: tuple[float, float, float, float] | None = None,
) -> None:
    """A clickable URL: opens it in the browser AND copies it to the clipboard on
    click (the setup-step links, the share links). `url` defaults to `label`.

    Distinct from `draw_copyable_text` (copy-only — used for file paths, where a
    browser-open is meaningless)."""
    target: str = url if url is not None else label
    imgui.push_style_color(imgui.Col_.text, color or COLOR.STATE_INFO)
    clicked: bool = imgui.selectable(
        label, False, size=(imgui.calc_text_size(label).x, 0)
    )[0]
    imgui.pop_style_color(1)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Click to open + copy")
    if not clicked:
        return
    try:
        pyperclip.copy(target)
    except pyperclip.PyperclipException:
        logger.warning("No clipboard backend (install xclip or xsel)")
    open_target: str = target if "://" in target else f"https://{target}"
    try:
        webbrowser.open(open_target)
    except Exception as e:
        logger.warning(f"Could not open browser: {e}")


def open_url_button(label: str, url: str, *, id_: str = "") -> None:
    # An OPEN-ONLY web-link button: opens the browser, does NOT copy (distinct from
    # draw_link). The url must carry its own scheme.
    if ghost_button(f"{label}{id_}") and url:
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.warning(f"Could not open browser: {e}")


def open_path_button(
    label: str, path: str, on_open: Callable[[str], None], *, id_: str = ""
) -> None:
    # An OPEN-ONLY local-file button: reveals the file in the OS file manager via the
    # injected opener (the caller passes util.open_in_file_manager). NO clipboard.
    if ghost_button(f"{label}{id_}") and path:
        on_open(path)


def clickable_label(
    label: str,
    width: float,
    *,
    id_: str | None = None,
    tooltip: str | None = None,
    highlight: bool = False,
    text_color: tuple[float, float, float, float] | None = None,
    accent: tuple[float, float, float, float] | None = None,
) -> bool:
    """A fixed-width clickable text cell (the uniform name -> jump-to-code).

    Fixed width so a following `same_line(x)` column stays put; the hover affordance is
    a colour change only (jitter-free). `highlight` paints the translucent accent wash.
    `text_color` overrides the label colour (default: dim foreground); `accent`
    overrides the hover/active wash (default: ACCENT_PRIMARY). Returns True on click.
    """
    imgui.align_text_to_frame_padding()
    text = _ellipsize(label, width)
    fg = text_color or COLOR.FG_DIM
    base = accent or COLOR.ACCENT_PRIMARY
    imgui.push_style_color(imgui.Col_.text, fg)
    imgui.push_style_color(imgui.Col_.header, fade(base, 0.15))
    imgui.push_style_color(imgui.Col_.header_hovered, fade(base, 0.18))
    imgui.push_style_color(imgui.Col_.header_active, fade(base, 0.28))
    clicked: bool = imgui.selectable(
        f"{text}##{id_ or label}", highlight, size=(width, 0)
    )[0]
    imgui.pop_style_color(4)
    if tooltip and imgui.is_item_hovered():
        imgui.set_tooltip(tooltip)
    return clicked


def chord_row(
    label: str, chord_str: str, label_w: float, *, highlight: bool = False
) -> None:
    """One ``label    [chord]`` row — the action name (dim, left) and its keychord as
    a pill, the pill's left edge at `label_w`. Shared by the cheatsheet overlay and the
    rebinder rows. `highlight` paints the pill text in the accent (the rebinder's
    "press a key..." capture state)."""
    imgui.align_text_to_frame_padding()
    imgui.text_colored(COLOR.FG_SECONDARY, label)
    imgui.same_line(label_w)
    text = COLOR.ACCENT_PRIMARY if highlight else COLOR.CHIP_FG
    pill_w = imgui.calc_text_size(chord_str).x + 2.0 * float(SPACE.MD)
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, float(SIZE.CHIP_ROUNDING))
    imgui.push_style_color(imgui.Col_.button, COLOR.CHIP_BG)
    imgui.push_style_color(imgui.Col_.button_hovered, COLOR.CHIP_BG)
    imgui.push_style_color(imgui.Col_.button_active, COLOR.CHIP_BG)
    imgui.push_style_color(imgui.Col_.text, text)
    imgui.button(f"{chord_str}##chord_{label}", size=(pill_w, 0.0))
    imgui.pop_style_color(4)
    imgui.pop_style_var(1)
