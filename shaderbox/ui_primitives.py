import math
import re
import webbrowser
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace

import moderngl
import pyperclip
from imgui_bundle import imgui, imgui_ctx
from loguru import logger

from shaderbox.profiling import FrameProfile, Span, by_cost, headline_ms, other_ms
from shaderbox.render_plan import RenderPlan, document_id_of_span
from shaderbox.theme import (
    COLOR,
    OVERLAY_ALPHA,
    SIZE,
    SPACE,
    fade,
    load_color,
    throttle_color,
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


def faint_hline(
    dl: imgui.ImDrawList, x0: float, x1: float, y: float, alpha: float = 1.0
) -> None:
    """A thin horizontal rule in the faded border color, drawn onto the given draw list.
    Caller picks the list (window vs foreground) + coords + alpha, so it serves both a
    layout-flow divider and an absolute foreground overlay."""
    col = imgui.color_convert_float4_to_u32(fade(COLOR.BORDER, alpha))
    dl.add_line((x0, y), (x1, y), col)


# ---------------------------------------------------------------------------
# The button system (079 D12) — FOUR tiers, picked by the action's role, never by how
# it looks. Every button in the app is one of them; a site that wants a fifth is a
# design question, not a new primitive.
#
#   standard_button — an ordinary verb. Transparent fill, a 1 px BORDER frame,
#                     secondary text. This is the family's look; the other three are
#                     the same frame in a different color.
#   primary_button  — the ONE call-to-action of a section (filled accent).
#   danger_button   — a destructive verb (the standard frame in the error color).
#   toggle_button   — a stateful on/off control: the standard frame when off, filled
#                     accent when on. SAME label both states; the style carries it.
#
# Chips and pills (`chip_button`, `pill_button`) are NOT in the count: they are a
# different thing — a tag, a filter, a mode selector — and are their own primitive.
# ---------------------------------------------------------------------------


def _framed_button(
    label: str,
    width: float,
    text_color: tuple[float, float, float, float],
    border_color: tuple[float, float, float, float],
) -> bool:
    # The one shape the standard, danger and off-toggle tiers all draw: transparent
    # fill, a 1 px frame, colored text, and a fill appearing on hover.
    imgui.push_style_color(imgui.Col_.button, COLOR.TRANSPARENT)
    imgui.push_style_color(imgui.Col_.button_hovered, COLOR.BG_FRAME)
    imgui.push_style_color(imgui.Col_.button_active, COLOR.BORDER)
    imgui.push_style_color(imgui.Col_.text, text_color)
    imgui.push_style_color(imgui.Col_.border, border_color)
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 1.0)
    clicked: bool = imgui.button(label, size=(width, 0.0))
    imgui.pop_style_var()
    imgui.pop_style_color(5)
    return clicked


def standard_button(
    label: str,
    width: float = 0.0,
    *,
    text_color: tuple[float, float, float, float] = COLOR.FG_SECONDARY,
) -> bool:
    """An ordinary verb: `open`, `add pass`, `Close`, `Cancel`, `Apply`.

    `text_color` is for a site whose label carries a state of its own (the script row's
    `open`, which reads in the script's own color); the frame stays the family's.
    """
    return _framed_button(label, width, text_color, COLOR.BORDER)


def primary_button(label: str, width: float = 0.0) -> bool:
    """The ONE call-to-action of a section: `Create`, `Send`, `Render`, `Connect`."""
    imgui.push_style_color(imgui.Col_.button, COLOR.ACCENT_PRIMARY)
    imgui.push_style_color(imgui.Col_.button_hovered, COLOR.ACCENT_ACTIVE)
    imgui.push_style_color(imgui.Col_.button_active, COLOR.ACCENT_ACTIVE)
    imgui.push_style_color(imgui.Col_.text, COLOR.BG_APP)
    imgui.push_style_color(imgui.Col_.border, COLOR.ACCENT_PRIMARY)
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 1.0)
    clicked: bool = imgui.button(label, size=(width, 0.0))
    imgui.pop_style_var()
    imgui.pop_style_color(5)
    return clicked


def toggle_button(label: str, active: bool, width: float = 0.0) -> bool:
    """A stateful on/off control: filled accent when `active`, the standard frame when
    off. Same label both states — the style carries the state."""
    if active:
        return primary_button(label, width)
    return _framed_button(label, width, COLOR.FG_SECONDARY, COLOR.BORDER)


def segmented_choice(
    id_: str, options: Sequence[str], selected: int, width: float = 0.0
) -> int:
    """A two-or-more-way selector drawn as one joined strip; returns the chosen index.

    The chosen segment is filled accent and the rest carry the standard frame, so the strip
    reads as ONE control with a position rather than as several buttons -- which is what
    separates it from a row of toggles: these options are mutually exclusive and always
    exactly one is on. `width` sizes each segment; 0.0 lets each take its own label's width.
    """
    chosen = selected
    spacing = imgui.get_style().item_spacing
    imgui.push_style_var(imgui.StyleVar_.item_spacing, (0.0, spacing.y))
    for index, label in enumerate(options):
        if index:
            imgui.same_line()
        if toggle_button(f"{label}##{id_}_{index}", index == selected, width):
            chosen = index
    imgui.pop_style_var()
    return chosen


def danger_button(label: str, width: float = 0.0) -> bool:
    """A destructive verb: `Delete`, `Reset`, `Clear`. The standard frame in the error
    color — the confirm step carries the weight, not a filled-red fill."""
    return _framed_button(label, width, COLOR.STATE_ERROR, COLOR.STATE_ERROR)


def pill_button(
    label: str,
    *,
    color: tuple[float, float, float, float],
    active: bool,
    text_color: tuple[float, float, float, float] = COLOR.BG_APP,
    inactive_alpha: float = 0.4,
    width: float = 0.0,
) -> bool:
    """A small filled pill whose `color` is the role (a tag, a favs toggle, a reset
    action). `active` fills it solid; otherwise a faded variant signals "off". Use
    when `chip_button`'s fixed palette doesn't fit (e.g. blue tags, yellow favs).
    `width=0` is content-width (`small_button`); a fixed `width` aligns into a column."""
    fill = color if active else fade(color, inactive_alpha)
    imgui.push_style_color(imgui.Col_.button, fill)
    imgui.push_style_color(imgui.Col_.text, text_color)
    clicked: bool = (
        imgui.small_button(label)
        if width == 0.0
        else imgui.button(label, size=(width, 0.0))
    )
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


def play_stop_toggle(id_: str, playing: bool, *, tooltip: str = "") -> bool:
    """The per-uniform / whole-document play/stop control (feature 048). A small word button — `stop`
    (accent) when the script is driving the slot, `play` (dim) when stopped — clearer than an
    ambiguous icon (/imgui-ui §1: use words for icon-ambiguous controls). Returns True on click; the
    caller flips the stopped state. SAME slot for both states; the label + color carry the state."""
    label, color = ("stop", COLOR.ACCENT_PRIMARY) if playing else ("play", COLOR.FG_DIM)
    imgui.push_style_color(imgui.Col_.button, COLOR.TRANSPARENT)
    imgui.push_style_color(imgui.Col_.button_hovered, COLOR.BG_FRAME)
    imgui.push_style_color(imgui.Col_.button_active, COLOR.BG_FRAME)
    imgui.push_style_color(imgui.Col_.text, color)
    clicked: bool = imgui.small_button(f"{label}##play_stop_{id_}")
    imgui.pop_style_color(4)
    if tooltip and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
        imgui.set_tooltip(tooltip)
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
def message_bubble(
    id_: str,
    bg: tuple[float, float, float, float],
    bordered: bool = True,
) -> Iterator[imgui.ImVec2]:
    """A full-width, auto-height rounded bubble for one chat message. `bordered=True` draws
    `bg` fill + border (user/assistant); `bordered=False` is fully invisible (no fill, no
    border) but keeps the same inset + padding, so system lines (tool/error) align with the
    bubbles. Yields the content origin (screen pos) so the caller can pin a corner affordance.
    Bubbles separate by their gap, not a rule."""
    # The borders flag stays set in BOTH modes (an invisible bubble paints fill + border
    # transparent) so the border inset is identical and the text aligns with visible bubbles.
    imgui.push_style_color(imgui.Col_.child_bg, bg if bordered else COLOR.TRANSPARENT)
    imgui.push_style_color(
        imgui.Col_.border, COLOR.BORDER if bordered else COLOR.TRANSPARENT
    )
    imgui.push_style_var(imgui.StyleVar_.child_rounding, float(SIZE.BUBBLE_ROUNDING))
    imgui.push_style_var(
        imgui.StyleVar_.window_padding, imgui.ImVec2(SPACE.SM, SPACE.SM)
    )
    # Inset both edges by SPACE.SM so the side margin matches the inter-bubble vertical gap.
    imgui.indent(float(SPACE.SM))
    width = imgui.get_content_region_avail().x - float(SPACE.SM)
    with imgui_ctx.begin_child(
        id_,
        size=imgui.ImVec2(width, 0.0),
        child_flags=imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y,
        window_flags=imgui.WindowFlags_.no_scrollbar,
    ):
        origin = imgui.get_cursor_screen_pos()
        yield origin
    imgui.unindent(float(SPACE.SM))
    imgui.pop_style_var(2)
    imgui.pop_style_color(2)


@contextmanager
def modal_window(
    label: str,
    size: tuple[float, float],
    flags: int = 0,
    fixed_size: bool = False,
) -> Iterator[bool]:
    """Boilerplate-free modal-popup wrapper. Caller owns the `is_X_open` flag on `App`
    (allows per-modal cleanup on close); this owns the imgui dance: open by label,
    seed size, enter the popup scope, yield visibility. `flags` passes window flags
    through (e.g. `no_scrollbar` for a modal that sizes its own content). `fixed_size`
    forces `size` every frame (`Cond_.always`) for a non-resizable modal — pair it with
    `WindowFlags_.no_resize`; the default seeds `size` once (`Cond_.first_use_ever`) so a
    user resize persists via imgui.ini. Use as:

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
    size_cond = imgui.Cond_.always if fixed_size else imgui.Cond_.first_use_ever
    imgui.set_next_window_size(imgui.ImVec2(*size), size_cond)
    # Center on the viewport (pivot at the window's own center). first_use_ever so a
    # user drag persists via imgui.ini.
    center = imgui.get_main_viewport().get_center()
    imgui.set_next_window_pos(
        center, imgui.Cond_.first_use_ever, imgui.ImVec2(0.5, 0.5)
    )
    with imgui_ctx.begin_popup_modal(label, flags=flags) as popup:
        yield popup.visible


# The caret row a note's anchor already stepped past, so a note flipped ABOVE the caret clears
# it going the other way; and the floor a capped note keeps so it never collapses to a sliver.
_NOTE_ANCHOR_ROW: float = 20.0
_NOTE_MIN_H: float = 60.0


def anchored_note(
    id_: str,
    anchor: tuple[float, float],
    title: str,
    body: str,
    value: str = "",
    texture: moderngl.Texture | None = None,
) -> None:
    """A small non-interactive note pinned at a screen point (the `K` lookup under the caret):
    an accent title line, an optional value line, a wrapped dim body and an optional picture,
    capped at `SIZE.NOTE_W`.

    The size is MEASURED and set, never auto-resized: imgui sizes an auto-resize window from
    the PREVIOUS frame's content, so a note whose text changes while it is open draws one frame
    inside the old note's size — the blink the maintainer saw moving `K` from symbol to symbol
    (079 D1).
    """
    padding = imgui.get_style().window_padding
    # The WIDTH text wraps at, which is the note's content width. `push_text_wrap_pos` below
    # takes a POSITION in window space, so it gets this plus the left padding the content
    # starts at — measuring one and wrapping at the other makes the text wrap earlier than it
    # was measured for, and the note comes up short by however many lines that adds.
    wrap = float(SIZE.NOTE_W) - 2.0 * padding.x
    line_gap = imgui.get_style().item_spacing.y
    # imgui advances the cursor by an item's height PLUS `item_spacing.y` after EVERY item, the
    # last one included — so N items cost N gaps, not N-1. Counting N-1 leaves the note one
    # gap short and clips its bottom line.
    title_size = imgui.calc_text_size(title, wrap_width=wrap)
    width, height = title_size.x, title_size.y + line_gap
    if value:
        value = _ellipsize(value, wrap)
        value_size = imgui.calc_text_size(value)
        width = max(width, value_size.x)
        height += value_size.y + line_gap
    if body:
        body_size = imgui.calc_text_size(body, wrap_width=wrap)
        width = max(width, body_size.x)
        height += body_size.y + line_gap
    picture = _note_picture_size(texture, wrap) if texture is not None else None
    if picture is not None:
        width = max(width, picture[0])
        height += picture[1] + line_gap
    note_w = min(width, wrap) + 2.0 * padding.x
    note_h = height + 2.0 * padding.y
    # A note is anchored a row below the caret and grows to fit its content, so a long docstring
    # near the bottom of the screen would run off it. Put it ABOVE the caret when there is more
    # room there, and cap it to that room either way — the body then scrolls rather than being
    # cut. `SPACE.MD` keeps it off the screen edge; `_NOTE_ANCHOR_ROW` is the row the anchor
    # already skipped, which the flipped note has to clear going the other way.
    x, y = anchor
    margin = float(SPACE.MD)
    below = imgui.get_io().display_size.y - y - margin
    above = y - _NOTE_ANCHOR_ROW - margin
    if note_h > below and above > below:
        note_h = min(note_h, above)
        y = max(margin, y - _NOTE_ANCHOR_ROW - note_h)
    else:
        note_h = min(note_h, max(below, _NOTE_MIN_H))
    imgui.set_next_window_pos(imgui.ImVec2(x, y), imgui.Cond_.always)
    imgui.set_next_window_size(imgui.ImVec2(note_w, note_h))
    # Opaque: the note sits over code and a translucent one made both unreadable.
    imgui.set_next_window_bg_alpha(1.0)
    imgui.push_style_color(imgui.Col_.window_bg, COLOR.BG_POPUP)
    flags = (
        imgui.WindowFlags_.no_decoration
        | imgui.WindowFlags_.no_inputs
        | imgui.WindowFlags_.no_nav
        | imgui.WindowFlags_.no_saved_settings
        | imgui.WindowFlags_.no_focus_on_appearing
    )
    with imgui_ctx.begin(id_, flags=flags) as window:
        if window:
            imgui.push_text_wrap_pos(padding.x + wrap)
            imgui.text_colored(COLOR.ACCENT_PRIMARY, title)
            if value:
                imgui.text_colored(COLOR.FG_PRIMARY, value)
            if body:
                imgui.text_colored(COLOR.FG_SECONDARY, body)
            imgui.pop_text_wrap_pos()
            if texture is not None and picture is not None:
                imgui.image(
                    imgui.ImTextureRef(texture.glo),
                    image_size=picture,
                    uv0=(0, 1),
                    uv1=(1, 0),
                )
    imgui.pop_style_color()


def _note_picture_size(texture: moderngl.Texture, width: float) -> tuple[float, float]:
    # The note's content width, aspect preserved (079 D13): a thumbnail-sized picture in a
    # note this wide reads as a swatch rather than as the texture.
    tex_w, tex_h = texture.size
    return (width, width * tex_h / tex_w if tex_w else width)


def rendering_overlay(text: str) -> None:
    # A centered, non-interactive cue painted one frame before the encode freezes the
    # frame loop. NOT a begin_popup_modal — it stays off the popup ID stack + the
    # popup-mutex (any_popup_open) so it never gates document rendering.
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


def item_normalized_mouse(
    rect_min: imgui.ImVec2,
    rect_max: imgui.ImVec2,
    *,
    flip_y: bool = True,
) -> tuple[float, float, bool] | None:
    """Hit-test the mouse against an EXPLICIT screen rect (the canvas preview), returning
    (nx, ny, inside) normalized 0..1 — `flip_y` gives y-up (GLSL). The rect is passed in (not
    'the last item') because `image_with_bg` submits no interactive item. Popup-blocking is
    honoured by ANDing `is_window_hovered(child_windows)` (`is_mouse_hovering_rect` alone ignores
    it). Returns None when the mouse pos is invalid; clamps to the rect edge when outside it.

    ANY hovered imgui item takes the mouse, not merely one drawn over this rect: the term is
    `is_any_item_hovered`, which is global and answers for LAST frame, since the chips over
    the preview are submitted after this runs. Both facts are wider than the case they were
    added for -- a click on the channel-view or FPS chip used to cycle the view AND reach the
    script as a brush-down, painting a stroke under the chip.

    What the width costs, measured: a frame in which any item anywhere was hovered suppresses
    the hit even once the mouse is back over the preview, so canvas re-entry from a widget
    loses one frame. A press landing the same frame the mouse arrives on a chip still leaks
    through, since last frame nothing was hovered. Both are bounded by the mouse being in one
    place at a time; a rect-local test would close them, at the cost of this function knowing
    what is drawn on top of it. A drag already in progress is unaffected -- imgui hover-tests
    no widget while one is active."""
    w = rect_max.x - rect_min.x
    h = rect_max.y - rect_min.y
    if w <= 0.0 or h <= 0.0:
        return None
    pos = imgui.get_mouse_pos()
    if pos.x < 0.0 or pos.y < 0.0:  # imgui's "no valid mouse" sentinel (-FLT_MAX)
        return None
    inside = (
        imgui.is_mouse_hovering_rect(rect_min, rect_max)
        and imgui.is_window_hovered(imgui.HoveredFlags_.child_windows)
        and not imgui.is_any_item_hovered()
    )
    nx = min(max((pos.x - rect_min.x) / w, 0.0), 1.0)
    ny = min(max((pos.y - rect_min.y) / h, 0.0), 1.0)
    if flip_y:
        ny = 1.0 - ny
    return nx, ny, inside


def caption_text(
    text: str, color: tuple[float, float, float, float] | None = None
) -> None:
    """Small, dim, secondary annotation (artifact stats, hints)."""
    imgui.text_colored(color or COLOR.FG_DIM, text)


def clipped_caption(
    text: str, max_width: float, color: tuple[float, float, float, float] | None = None
) -> None:
    """A `caption_text` bounded to `max_width` — ellipsizes when it would overflow, full value on
    hover. For read-only value readouts in a fixed column (a vec-array uniform's `[[...], ...]` would
    otherwise overflow the row and shove trailing controls off-screen)."""
    shown = _ellipsize(text, max_width)
    imgui.text_colored(color or COLOR.FG_DIM, shown)
    if shown != text and imgui.is_item_hovered():
        imgui.set_tooltip(text)


# ---------------------------------------------------------------------------
# Labelled fields — a dim caption on its own line above the control. One primitive
# per control type, used by the exporter panels.
# ---------------------------------------------------------------------------


def help_marker(text: str) -> None:
    """Dim '(?)' that shows a wrapped explanation on hover — the settings help affordance."""
    imgui.text_colored(COLOR.FG_DIM, "(?)")
    if imgui.is_item_hovered(imgui.HoveredFlags_.delay_short):
        with imgui_ctx.begin_tooltip():
            imgui.push_text_wrap_pos(imgui.get_font_size() * 24.0)
            imgui.text_unformatted(text)
            imgui.pop_text_wrap_pos()


@dataclass(frozen=True)
class FieldFocus:
    """How a jump points at a settings field (`open_settings(focus=…)`): `keyboard` gives it
    keyboard focus and scrolls it into view, on ONE frame; `mark` outlines it with the accent
    at that alpha, for as long as the caller keeps it above zero, so the eye finds the field
    that a caret alone does not show."""

    keyboard: bool = False
    mark: float = 0.0


NO_FOCUS = FieldFocus()


ComboRow = tuple[str, tuple[float, float, float, float]]


def grouped_combo(
    id_: str,
    current: ComboRow,
    groups: Sequence[tuple[str, Sequence[ComboRow]]],
    width: float,
) -> int | None:
    """A combo whose list reads as groups: each group a dim caption (or none, for a bare
    run) over its rows, every row in its own color, a gap between groups. The closed control
    shows `current` in its color. Returns the picked row's index across all groups, else
    None."""
    imgui.set_next_item_width(width)
    imgui.push_style_color(imgui.Col_.text, current[1])
    opened = imgui.begin_combo(id_, current[0])
    imgui.pop_style_color(1)
    if not opened:
        return None
    picked: int | None = None
    index = 0
    for g, (caption, rows) in enumerate(groups):
        if g > 0:
            imgui.dummy(imgui.ImVec2(0.0, float(SPACE.XS)))
        if caption:
            caption_text(caption)
        for label, color in rows:
            imgui.push_style_color(imgui.Col_.text, color)
            chosen = imgui.selectable(f"{label}##{index}", label == current[0])[0]
            imgui.pop_style_color(1)
            if chosen:
                picked = index
            index += 1
    imgui.end_combo()
    return picked


@contextmanager
def focus_field(focus: FieldFocus) -> Iterator[None]:
    """Wrap the ONE widget a jump points at. The caller owns the timing: `keyboard` True only
    on the frame the request fires (a re-grab every frame reads as a modal dismiss), `mark`
    decaying to zero over `SETTINGS_MARK_S`."""
    if focus.keyboard:
        imgui.set_keyboard_focus_here()
        imgui.set_scroll_here_y()
    yield
    if focus.mark > 0.0:
        lo = imgui.get_item_rect_min()
        hi = imgui.get_item_rect_max()
        pad = float(SPACE.XS)
        imgui.get_window_draw_list().add_rect(
            imgui.ImVec2(lo.x - pad, lo.y - pad),
            imgui.ImVec2(hi.x + pad, hi.y + pad),
            imgui.get_color_u32(fade(COLOR.ACCENT_PRIMARY, focus.mark)),
            imgui.get_style().frame_rounding,
            thickness=2.0,
        )


def labeled_text_input(
    label: str,
    value: str,
    width: float,
    password: bool = False,
    focus: FieldFocus = NO_FOCUS,
) -> str:
    """Caption above a single-line text input. Returns the new value. `focus` is how a jump
    points at the input (see `focus_field`) — the caller owns its timing."""
    caption_text(label)
    imgui.set_next_item_width(width)
    with focus_field(focus):
        value = imgui.input_text(
            f"##{label}",
            value,
            flags=imgui.InputTextFlags_.password
            if password
            else imgui.InputTextFlags_.none,
        )[1]
    return value


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


MarkdownSpan = tuple[str, str]  # (style: "plain" | "bold" | "code", text)
MarkdownLine = tuple[bool, list[MarkdownSpan]]  # (is_code_block_line, spans)

# Bold content must not start/end with whitespace (CommonMark) — keeps a math-ish
# "2 ** 3" or a torn stream literal instead of bolding across the operators.
_MD_INLINE_RE = re.compile(r"(\*\*(?=\S)[^*]+?(?<=\S)\*\*|`[^`]+`)")
_MD_WORD_RE = re.compile(r"\S+\s*|\s+")


def parse_markdown_lines(text: str) -> list[MarkdownLine]:
    """Markdown-lite for chat prose: per line, inline `**bold**` + backtick code
    spans; ``` fences toggle whole-line code blocks (the fence lines are dropped).
    Unmatched markers stay literal, so a torn streaming preview renders safely."""
    out: list[MarkdownLine] = []
    in_block = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_block = not in_block
            continue
        if in_block:
            out.append((True, [("code", line)]))
            continue
        spans: list[MarkdownSpan] = []
        for part in _MD_INLINE_RE.split(line):
            if not part:
                continue
            if part.startswith("**") and part.endswith("**") and len(part) > 4:
                spans.append(("bold", part[2:-2]))
            elif part.startswith("`") and part.endswith("`") and len(part) > 2:
                spans.append(("code", part[1:-1]))
            else:
                spans.append(("plain", part))
        out.append((False, spans))
    return out


_TEXT_CHIP_PAD = 2.0


def text_chip(
    text: str, text_color: tuple[float, float, float, float], dim: bool = False
) -> None:
    """A word on a faded CHIP_BG rect (an inline code span, a pass's read). Drawn at the
    current cursor, `_TEXT_CHIP_PAD` wider than the text on each side; the caller owns line
    placement. Trailing whitespace rides outside the chip."""
    visible = text.rstrip()
    pos = imgui.get_cursor_screen_pos()
    size = imgui.calc_text_size(visible)
    imgui.get_window_draw_list().add_rect_filled(
        (pos.x - _TEXT_CHIP_PAD, pos.y),
        (pos.x + size.x + _TEXT_CHIP_PAD, pos.y + size.y),
        imgui.color_convert_float4_to_u32(fade(COLOR.CHIP_BG, 0.2 if dim else 0.4)),
        3.0,
    )
    imgui.text_colored(text_color, text)


def _code_chip(text: str) -> None:
    text_chip(text, COLOR.FG_PRIMARY)


def markdown_text(text: str, bold_font: imgui.ImFont) -> None:
    """Chat prose with markdown-lite styling: **bold** (bold font), backtick code
    (a chip), ``` blocks (dim code-colored lines). Mixed inline styles can't ride
    one wrapped text item, so styled lines word-wrap manually; marker-free text
    (the common case) takes the native-wrap fast path."""
    if "**" not in text and "`" not in text:
        imgui.push_text_wrap_pos(0.0)
        imgui.text_unformatted(text)
        imgui.pop_text_wrap_pos()
        return
    for is_block, spans in parse_markdown_lines(text):
        if is_block:
            imgui.push_text_wrap_pos(0.0)
            imgui.text_colored(COLOR.CHIP_FG, spans[0][1] or " ")
            imgui.pop_text_wrap_pos()
            continue
        if not spans:
            imgui.text_unformatted("")
            continue
        first = True
        for style, span_text in spans:
            # A code span is ONE wrap unit (a split chip reads broken); prose splits
            # into words so long sentences wrap normally.
            tokens = [span_text] if style == "code" else _MD_WORD_RE.findall(span_text)
            for tok in tokens:
                if style == "bold":
                    imgui.push_font(bold_font, bold_font.legacy_size)
                width = imgui.calc_text_size(tok.rstrip()).x
                if not first:
                    imgui.same_line(0.0, 0.0)
                    if imgui.get_content_region_avail().x < width:
                        imgui.new_line()
                if style == "code":
                    _code_chip(tok)
                else:
                    imgui.text_unformatted(tok)
                if style == "bold":
                    imgui.pop_font()
                first = False


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


def tune_icon_button(id_: str, side: float) -> bool:
    """A framed square with a drawn sliders glyph (three rails, offset knobs) — the
    settings affordance for a preview tile. The frame-colored fill keeps the glyph
    readable over any image. No font dependency. Returns True on click."""
    clicked, origin = _glyph_button(
        id_, side, COLOR.BG_FRAME, COLOR.BORDER, COLOR.BORDER
    )
    col = imgui.color_convert_float4_to_u32(COLOR.FG_TITLE)
    dl = imgui.get_window_draw_list()
    pad: float = side * 0.28
    x0, x1 = origin.x + pad, origin.x + side - pad
    for i, knob in enumerate((0.7, 0.3, 0.55)):
        y: float = origin.y + pad + (side - 2 * pad) * i / 2.0
        dl.add_line((x0, y), (x1, y), col, 1.2)
        dl.add_circle_filled((x0 + (x1 - x0) * knob, y), side * 0.09, col)
    return clicked


def copy_icon_button(id_: str, side: float) -> bool:
    """A ghost square with a drawn copy glyph (two offset rounded-rect outlines), for a
    corner copy affordance. No font dependency. Returns True on click."""
    clicked, origin = _glyph_button(
        id_, side, COLOR.TRANSPARENT, COLOR.BG_FRAME, COLOR.BORDER
    )
    col = imgui.color_convert_float4_to_u32(COLOR.FG_DIM)
    dl = imgui.get_window_draw_list()
    # Two overlapping sheets: a back rect up-right, a front rect down-left. The pair's
    # bounding box is (w + off) square; center it in the button.
    s = side
    w = s * 0.5  # sheet side
    off = s * 0.18  # diagonal offset between the two sheets
    bx = origin.x + (s - (w + off)) * 0.5
    by = origin.y + (s - (w + off)) * 0.5 + off
    dl.add_rect((bx + off, by - off), (bx + off + w, by - off + w), col, rounding=1.5)
    dl.add_rect((bx, by), (bx + w, by + w), col, rounding=1.5)
    return clicked


def revert_icon_button(id_: str, side: float) -> bool:
    """A ghost square with a drawn counter-clockwise undo arc + arrowhead, for the rollback
    affordance on a chat turn. No font dependency. Returns True on click."""
    clicked, origin = _glyph_button(
        id_, side, COLOR.TRANSPARENT, COLOR.BG_FRAME, COLOR.BORDER
    )
    col = imgui.color_convert_float4_to_u32(COLOR.FG_DIM)
    dl = imgui.get_window_draw_list()
    cx, cy = origin.x + side * 0.5, origin.y + side * 0.54
    r = side * 0.26
    # A ~270deg arc open at the top-left, with an arrowhead at the open (upper) end.
    dl.path_arc_to(imgui.ImVec2(cx, cy), r, math.radians(300), math.radians(120))
    dl.path_stroke(col, 1.5)
    tip = imgui.ImVec2(
        cx + r * math.cos(math.radians(300)), cy + r * math.sin(math.radians(300))
    )
    a = side * 0.16
    dl.add_line(tip, imgui.ImVec2(tip.x - a, tip.y - a * 0.2), col, 1.5)
    dl.add_line(tip, imgui.ImVec2(tip.x - a * 0.2, tip.y - a), col, 1.5)
    return clicked


def cycle_chip_width(labels: Sequence[str]) -> float:
    """How wide `cycle_chip` will draw for this label set.

    Public because a caller laying out a row has to RESERVE this: the chip sizes to its widest
    label, so a neighbour budgeting `SIZE.CHIP_W` instead is short by however much the longest
    label exceeds it, and whatever the row draws next overruns. One owner of the number, read by
    both the drawer and the layout."""
    return max(
        float(SIZE.CHIP_W),
        max(imgui.calc_text_size(label).x for label in labels) + 2.0 * float(SPACE.MD),
    )


def cycle_chip(id_: str, labels: Sequence[str], active: int) -> bool:
    """One chip showing where a setting IS; a click advances it to the next position.

    The caller owns the ordering and does the advancing — this is the drawn seam, matching the
    uniform panel's input-type selector. Width is the WIDEST label's, not the current one's, so
    the chip keeps one size across the cycle and the row cannot shift on click; sizing to the
    current label instead makes every neighbour jump."""
    return chip_button(f"{labels[active]}##{id_}", cycle_chip_width(labels))


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


def gauge_bar(id_: str, fraction: float, tooltip: str, width: float) -> None:
    """A thin horizontal fill bar (`fraction` pre-clamped [0, 1] by the caller), vertically
    centred within one frame-height row, with a hover tooltip. Owns the hit rect + the
    tooltip — geometry/color live here so richer visuals can replace it without touching the
    caller."""
    bar_h: float = float(SIZE.USAGE_BAR_H)
    row_h: float = imgui.get_frame_height()
    origin = imgui.get_cursor_screen_pos()
    imgui.invisible_button(f"##{id_}", imgui.ImVec2(width, row_h))
    if imgui.is_item_hovered():
        imgui.set_tooltip(tooltip)
    by0: float = origin.y + (row_h - bar_h) / 2.0
    by1: float = by0 + bar_h
    dl = imgui.get_window_draw_list()
    dl.add_rect_filled(
        (origin.x, by0),
        (origin.x + width, by1),
        imgui.color_convert_float4_to_u32(COLOR.BG_FRAME),
    )
    if fraction > 0.0:
        dl.add_rect_filled(
            (origin.x, by0),
            (origin.x + width * fraction, by1),
            imgui.color_convert_float4_to_u32(COLOR.ACCENT_PRIMARY),
        )
    dl.add_rect(
        (origin.x, by0),
        (origin.x + width, by1),
        imgui.color_convert_float4_to_u32(COLOR.BORDER),
        thickness=1.0,
    )


def step_squares(
    id_: str,
    squares: list[tuple[tuple[float, float, float, float], bool]],
) -> bool:
    """A row of small filled squares (a turn's compact progress bar) — one per `(color, pulse)`
    entry, wrapping at the content-region width. A `pulse=True` square breathes its alpha via
    `imgui.get_time()` (the live/pending head). Reserves its own layout height and overlays one
    hit-rect; returns True while hovered (the caller shows a breakdown tooltip). No font dependency.

    The caller owns the color language (e.g. done=ok/fail, a gray pulsing head, a final answer
    square) so this primitive stays feature-agnostic."""
    side: float = float(SPACE.MD)
    gap: float = float(SPACE.XS)
    avail_w: float = max(imgui.get_content_region_avail().x, side)
    per_row: int = max(1, int((avail_w + gap) // (side + gap)))
    n: int = len(squares)
    rows: int = max(1, -(-n // per_row)) if n else 1
    total_h: float = rows * side + max(0, rows - 1) * gap
    origin = imgui.get_cursor_screen_pos()
    # Reserve the block + own the hover hit-rect in one item. Height is fixed by the row count
    # (independent of WHICH squares fill it), so a turn's bar never changes height as steps land.
    imgui.invisible_button(f"##{id_}", imgui.ImVec2(avail_w, total_h))
    hovered: bool = imgui.is_item_hovered()
    # Triangle-wave alpha in [0.35, 1.0] at ~1.4 Hz for the pulsing head.
    t = imgui.get_time() * 1.4
    pulse_a: float = 0.35 + 0.65 * abs((t - math.floor(t)) * 2.0 - 1.0)
    dl = imgui.get_window_draw_list()
    for i, (color, pulse) in enumerate(squares):
        r, c = divmod(i, per_row)
        x0: float = origin.x + c * (side + gap)
        y0: float = origin.y + r * (side + gap)
        rgba = (color[0], color[1], color[2], color[3] * (pulse_a if pulse else 1.0))
        col = imgui.color_convert_float4_to_u32(rgba)
        dl.add_rect_filled((x0, y0), (x0 + side, y0 + side), col, rounding=2.0)
    return hovered


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

    # The wash already carries the danger, and red-on-red would not read: the confirm is
    # the primary action of this overlay, the dismissal the standard tier.
    btn_w: float = (avail.x - 3 * SPACE.SM) / 2
    row_y: float = origin.y + avail.y * 0.55
    imgui.set_cursor_screen_pos((origin.x + SPACE.SM, row_y))
    if primary_button("Yes", width=btn_w):
        return True
    imgui.set_cursor_screen_pos((origin.x + SPACE.SM + btn_w + SPACE.SM, row_y))
    if standard_button("No", width=btn_w):
        return False
    return None


@dataclass
class PreviewCellResult:
    clicked: bool = False  # whole-cell click target hit
    delete_armed: bool = False  # the delete-✕ was pressed this frame
    delete_confirmed: bool = False  # `Yes` on the in-cell confirm wash
    delete_cancelled: bool = False  # `No` on the in-cell confirm wash


def _chip_row(
    chips: Sequence[str], font: imgui.ImFont, max_width: float, dim: bool
) -> None:
    # As many chips as fit, then `+N` for the rest; the row is centered in `max_width`.
    imgui.push_font(font, font.legacy_size)
    gap: float = float(SPACE.SM)
    chip_w = [imgui.calc_text_size(c).x + 2 * _TEXT_CHIP_PAD for c in chips]
    shown: int = len(chips)
    while shown > 0:
        rest = f"+{len(chips) - shown}" if shown < len(chips) else ""
        rest_w = imgui.calc_text_size(rest).x + gap if rest else 0.0
        row_w = sum(chip_w[:shown]) + gap * (shown - 1) + rest_w
        if row_w <= max_width:
            break
        shown -= 1
    rest = f"+{len(chips) - shown}" if shown < len(chips) else ""
    row_w = (
        sum(chip_w[:shown])
        + gap * max(0, shown - 1)
        + (imgui.calc_text_size(rest).x + gap if rest else 0.0)
    )
    color = COLOR.FG_DIM if dim else COLOR.CHIP_FG
    pos = imgui.get_cursor_screen_pos()
    if not chips:
        # The row is reserved even when empty; an item must cover it or the cursor move
        # asserts as a window-boundary extension.
        imgui.dummy((max_width, imgui.get_text_line_height()))
    x: float = pos.x + max(0.0, (max_width - row_w) / 2) + _TEXT_CHIP_PAD
    for chip, w in zip(chips[:shown], chip_w[:shown], strict=True):
        imgui.set_cursor_screen_pos((x, pos.y))
        text_chip(chip, color, dim)
        x += w + gap
    if rest:
        imgui.set_cursor_screen_pos((x - _TEXT_CHIP_PAD, pos.y))
        imgui.text_colored(color, rest)
    imgui.pop_font()


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
    footer_font: imgui.ImFont | None = None,
    footer_color: tuple[float, float, float, float] | None = None,
    overlay: Callable[[float], None] | None = None,
    stale: bool = False,
    chips: Sequence[str] | None = None,
    chip_font: imgui.ImFont | None = None,
    bordered: bool = True,
) -> PreviewCellResult:
    """A bordered preview tile: a `cell_w`-wide square image + whole-cell click
    target + selection border + a top-right delete-✕ arming an in-cell `Delete?` wash.

    The cell is `cell_w` wide and grows below the image by one text line when `footer`
    is set; the caller sizes only the width. `overlay` draws an extra top-LEFT control,
    shown alongside the delete-✕ only while `selected` and not `armed`. The whole tile
    is its own child window so the overlays' absolute cursor moves can't perturb the
    parent (no jitter / SetCursorPos assert).

    `stale` marks a texture that is no longer being rendered — the footer and chips dim and
    the tile takes the corner tick; the picture itself is left as it is. `footer_font` /
    `footer_color` override the footer's face and color (the strip's live names are bold and
    bright, its dormant ones darker than the default dim).

    `chips` adds one more line under the footer, drawn in `chip_font`: each word on its own
    small chip, centered as a row. The line is reserved whenever `chips` is given (an empty
    row keeps every cell in a strip the same height); the chips that do not fit the width
    collapse into a `+N` count, so the row never clips.

    `bordered=False` drops the child's own border and keeps its padding: `ChildFlags_.borders`
    is what enables `WindowPadding`, so the plain flag would shift the picture by the padding.
    An explicit `border_color` still draws its border either way.
    """
    line_h: float = imgui.get_text_line_height_with_spacing()
    footer_h: float = line_h if footer else 0.0
    chips_h: float = 0.0
    if chips is not None and chip_font is not None:
        imgui.push_font(chip_font, chip_font.legacy_size)
        chips_h = imgui.get_text_line_height() + float(SPACE.XS)
        imgui.pop_font()
    cell_h: float = cell_w + footer_h + chips_h
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
        child_flags=imgui.ChildFlags_.borders
        if bordered or border_color is not None
        else imgui.ChildFlags_.always_use_window_padding,
        window_flags=imgui.WindowFlags_.no_scrollbar
        | imgui.WindowFlags_.no_scroll_with_mouse,
    ):
        imgui.pop_style_color(n_styles)
        origin = imgui.get_cursor_screen_pos()
        avail = imgui.get_content_region_avail()
        dl = imgui.get_window_draw_list()
        img_h: float = (
            avail.y - footer_h - chips_h
        )  # the text lines sit under the image

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
                imgui.color_convert_float4_to_u32(COLOR.WHITE),
            )

        # allow_overlap so the buttons drawn on top win the click; the transparent
        # header colors leave the image/border carrying the visual.
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
            if footer_font is not None:
                imgui.push_font(footer_font, footer_font.legacy_size)
            label: str = _ellipsize(footer, avail.x)
            fw = imgui.calc_text_size(label)
            fy: float = origin.y + img_h
            imgui.set_cursor_screen_pos((origin.x + (avail.x - fw.x) / 2, fy))
            color = footer_color or (COLOR.FG_DIM if stale else None)
            if color is not None:
                imgui.text_colored(color, label)
            else:
                imgui.text(label)
            if footer_font is not None:
                imgui.pop_font()

        if chips is not None and chip_font is not None:
            # The row spans the cell's width to a small inset, not the padded content region:
            # the padding frames the image, and three short names need every pixel of it.
            inset: float = float(SPACE.XS)
            imgui.set_cursor_screen_pos(
                (imgui.get_window_pos().x + inset, origin.y + img_h + footer_h)
            )
            _chip_row(chips, chip_font, imgui.get_window_size().x - 2 * inset, stale)

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


@dataclass(frozen=True)
class ProfileRow:
    """One row of the FPS panel, decided before any imgui call.

    `name` is the raw span name where the row is a span, so the draw clips it itself;
    `number` is the formatted readout and `color` the hue it draws in. `starts_tree` marks
    the row the panel puts a gap above, so the draw reads the boundary off the row rather
    than recounting the plan's leading rows. `tooltip`, where a row has one, carries what the
    compact number left out -- a throttled document's own milliseconds (090 D9c).
    """

    depth: int
    name: str
    count: int
    number: str
    color: tuple[float, float, float, float]
    starts_tree: bool = False
    tooltip: str = ""


def profile_rows_plan(
    profile: FrameProfile | None,
    fps: int,
    target_fps: int,
    plan: RenderPlan | None = None,
    titles: dict[str, str] | None = None,
    budget: float = 1.0,
) -> list[ProfileRow]:
    """The panel's rows in draw order, colored by each measurement's share of the budget.

    The tree's children are ordered by cost (`profiling.by_cost`); the profile itself is
    never reordered. Pure and imgui-free, so the order and the bands are testable without
    a window.

    A `document:<id>` span is rendered through `titles` and, when `plan` throttles it, reads as
    its effective fps, its interval and its share of wall time rather than as a millisecond
    count (090 D7/D9c); `budget` is the share of wall time all documents together may take,
    which that percentage is measured against. The match is by ID: two documents sharing a
    title are two rows with their own numbers, which matching through a title could not
    express.
    """
    budget_ms: float = 1e3 / target_fps
    rows: list[ProfileRow] = []
    frame_over_budget: bool = profile is not None and profile.cpu_ms > budget_ms
    if profile is not None:
        # CPU and GPU overlap, so neither alone is the frame's bound -- both are shown,
        # and which one is larger is what the reader is looking for.
        rows.append(_measured_row(0, "frame", 1, profile.cpu_ms, budget_ms))
        rows.append(_measured_row(0, "gpu", 1, profile.gpu_ms, budget_ms))
    rows.append(ProfileRow(0, "budget", 1, f"{budget_ms:.1f} ms", COLOR.FG_MUTED))
    rows.append(ProfileRow(0, "fps", 1, str(fps), COLOR.FG_MUTED))
    rows.append(ProfileRow(0, "target", 1, str(target_fps), COLOR.FG_MUTED))
    if profile is not None:
        first_tree_row = len(rows)
        _plan_tree(
            profile.root,
            0,
            budget_ms,
            rows,
            plan,
            titles or {},
            frame_over_budget,
            budget,
        )
        rows.append(_measured_row(0, "other", 1, other_ms(profile.root), budget_ms))
        rows[first_tree_row] = replace(rows[first_tree_row], starts_tree=True)
    return rows


def _measured_row(
    depth: int, name: str, count: int, ms: float, budget_ms: float
) -> ProfileRow:
    return ProfileRow(depth, name, count, f"{ms:.2f} ms", load_color(ms / budget_ms))


# How much of a closed document's uuid its row still shows, until the two-frame-late profile
# stops carrying its span. Enough to tell two rows apart, short enough not to clip the number.
_CLOSED_DOCUMENT_ID_CHARS: int = 8


def _document_row(
    depth: int,
    span: Span,
    budget_ms: float,
    plan: RenderPlan | None,
    titles: dict[str, str],
    frame_over_budget: bool,
    budget: float,
) -> ProfileRow:
    """One `document:<id>` span's row: its title, and its plan numbers where it is throttled."""
    document_id = document_id_of_span(span.name) or span.name
    # A profile is two frames behind (088 D2), so a document closed on frame N still has a span
    # on N+1 and N+2 while the title map no longer carries its id. A 36-character uuid clips
    # hard in a 280-px panel, so the fallback is a short handle rather than the whole id.
    title = titles.get(document_id) or document_id[:_CLOSED_DOCUMENT_ID_CHARS]
    ms = headline_ms(span)
    interval = plan.intervals.get(document_id, 1) if plan is not None else 1
    if plan is None or interval <= 1:
        return ProfileRow(
            depth, title, span.count, f"{ms:.2f} ms", load_color(ms / budget_ms)
        )
    document_fps = plan.document_fps.get(document_id, 0.0)
    # The share of WALL TIME this document takes at its own rate (D9c: cost x document fps),
    # against the budget all documents together may take. A converged one sits at ~100 %.
    share = ms * document_fps / 1e3
    ratio = share / budget if budget > 0.0 else 0.0
    return ProfileRow(
        depth,
        title,
        span.count,
        f"{document_fps:.0f} fps x{interval}  {ratio * 100:.0f}%",
        throttle_color(ratio, frame_over_budget),
        tooltip=f"{ms:.2f} ms",
    )


def _plan_tree(
    span: Span,
    depth: int,
    budget_ms: float,
    rows: list[ProfileRow],
    plan: RenderPlan | None = None,
    titles: dict[str, str] | None = None,
    frame_over_budget: bool = False,
    budget: float = 1.0,
) -> None:
    names = titles or {}
    for child in by_cost(span.children):
        if document_id_of_span(child.name) is not None:
            rows.append(
                _document_row(
                    depth, child, budget_ms, plan, names, frame_over_budget, budget
                )
            )
        else:
            rows.append(
                _measured_row(
                    depth, child.name, child.count, headline_ms(child), budget_ms
                )
            )
        _plan_tree(
            child, depth + 1, budget_ms, rows, plan, names, frame_over_budget, budget
        )


def _profile_number(
    number_font: imgui.ImFont, value: str, color: tuple[float, float, float, float]
) -> None:
    """The number half of a profiler row: right-aligned at the panel edge, in `number_font`.

    Separate from the label so every authored string in this panel stays one word -- a
    label-plus-number-plus-unit string is six, over the caption budget the prose gate scores
    (`tests/test_ui_prose_budget.py`).
    """
    imgui.push_font(number_font, number_font.legacy_size)
    width = imgui.calc_text_size(value).x
    imgui.same_line(
        imgui.get_content_region_avail().x - width + imgui.get_cursor_pos_x()
    )
    imgui.text_colored(color, value)
    imgui.pop_font()


def _number_width(number_font: imgui.ImFont, value: str) -> float:
    imgui.push_font(number_font, number_font.legacy_size)
    width = imgui.calc_text_size(value).x
    imgui.pop_font()
    return width


def _profile_rows(rows: list[ProfileRow], number_font: imgui.ImFont) -> None:
    """One planned row per line, indented by its depth.

    A span name is data (`pass:cascade_gather_and_merge`, a document's own title), so it is
    clipped to the room left before the number rather than running under it. The row that
    starts the tree takes a gap above it.
    """
    for row in rows:
        if row.starts_tree:
            imgui.dummy((0.0, float(SPACE.SM)))
        indent = float(SPACE.MD) * row.depth
        room = (
            imgui.get_content_region_avail().x
            - indent
            - _number_width(number_font, row.number)
            - float(SPACE.SM)
        )
        if row.count > 1:
            room -= imgui.calc_text_size(f"x{row.count}").x + float(SPACE.SM)
        imgui.dummy((indent, 0.0))
        imgui.same_line(0.0, 0.0)
        clipped_caption(row.name, max(0.0, room))
        if row.tooltip and imgui.is_item_hovered():
            imgui.set_tooltip(row.tooltip)
        if row.count > 1:
            imgui.same_line(0.0, float(SPACE.SM))
            caption_text(f"x{row.count}")
        _profile_number(number_font, row.number, row.color)


def fps_overlay(
    anchor_x: float,
    anchor_y: float,
    fps: int,
    target_fps: int,
    is_open: bool,
    profile: FrameProfile | None,
    number_font: imgui.ImFont,
    document_fps: int | None = None,
    plan: RenderPlan | None = None,
    titles: dict[str, str] | None = None,
    budget: float = 1.0,
) -> bool:
    """A clickable FPS chip pinned to the top-right of a region, optionally
    unfolding the last complete frame's profile beneath it.

    `anchor_x` / `anchor_y` are the top-RIGHT screen corner of the region the overlay
    hugs (the pill's right edge sits `inset` left of `anchor_x`). Returns the new open
    state (toggled on a pill click). The pill is anchored in screen space independent
    of the detail panel, so opening it never shifts the pill.

    `profile` is the last frame whose GPU queries have been read back (two frames behind
    live, 088 D2); `None` draws the headline alone, which is the first frame after opening.

    `document_fps` is the current document's own rate where the throttle gave it one (090 D9b):
    the chip then carries both numbers, since the UI's fps alone says nothing about how often
    the document the user is watching actually redraws. `None` draws today's single number.
    `plan` / `titles` / `budget` go to the panel's rows.
    """
    label = f"{fps} FPS" if document_fps is None else f"{fps} | doc {document_fps}"
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
            _profile_rows(
                profile_rows_plan(profile, fps, target_fps, plan, titles, budget),
                number_font,
            )
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
    if standard_button(f"{label}{id_}") and url:
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.warning(f"Could not open browser: {e}")


def open_path_button(
    label: str, path: str, on_open: Callable[[str], None], *, id_: str = ""
) -> None:
    # An OPEN-ONLY local-file button: reveals the file in the OS file manager via the
    # injected opener (the caller passes util.open_in_file_manager). NO clipboard.
    if standard_button(f"{label}{id_}") and path:
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
    a color change only (jitter-free). `highlight` paints the translucent accent wash.
    `text_color` overrides the label color (default: dim foreground); `accent`
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


def text_tab_row(id_: str, names: Sequence[str], active: str) -> str | None:
    """A row of clickable names selecting one of them (the Uniforms tab's pass pick).

    A selector, not a verb: each name is a frameless `selectable` sized to its own text,
    and the state is carried by color alone — `active` in the accent, the rest dim, brighter
    on hover. The row wraps to a new line when the next name would cross the content width.
    Returns the name clicked this frame, or None.
    """
    gap: float = float(SPACE.LG)
    pad = imgui.get_style().item_spacing
    clicked: str | None = None
    imgui.push_style_color(imgui.Col_.header, COLOR.TRANSPARENT)
    imgui.push_style_color(imgui.Col_.header_hovered, COLOR.TRANSPARENT)
    imgui.push_style_color(imgui.Col_.header_active, COLOR.TRANSPARENT)
    # Re-read per row, not once: `avail` is measured from the CURSOR, so a caller that
    # left it mid-line gives the first row less than a wrapped row gets.
    avail: float = imgui.get_content_region_avail().x
    x: float = 0.0
    for index, name in enumerate(names):
        width: float = imgui.calc_text_size(name).x
        if index:
            if x + gap + width <= avail:
                imgui.same_line(spacing=gap)
                x += gap
            else:
                x = 0.0
                avail = imgui.get_content_region_avail().x
        # The hover color is decided BEFORE the item is submitted, so the name is drawn
        # once at its final color; `imgui.selectable` has no pre-hover to read. The rect
        # imgui gives the item is the text outset by half an item spacing on each side.
        origin = imgui.get_cursor_screen_pos()
        hovered: bool = imgui.is_window_hovered(
            imgui.HoveredFlags_.child_windows
        ) and imgui.is_mouse_hovering_rect(
            (origin.x - pad.x / 2, origin.y - pad.y / 2),
            (
                origin.x + width + pad.x / 2,
                origin.y + imgui.get_text_line_height() + pad.y / 2,
            ),
        )
        if name == active:
            color = COLOR.ACCENT_PRIMARY
        else:
            color = COLOR.FG_SECONDARY if hovered else COLOR.FG_DIM
        imgui.push_style_color(imgui.Col_.text, color)
        if imgui.selectable(f"{name}##{id_}_{name}", False, size=(width, 0))[0]:
            clicked = name
        imgui.pop_style_color(1)
        x += width
    imgui.pop_style_color(3)
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
