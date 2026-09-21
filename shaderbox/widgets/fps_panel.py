"""The FPS panel: the frame profile as rows, and the overlay that draws them.

Split from `ui_primitives` so the widget kit can be imported by a host
that has no render plan and no profiler -- this was the only widget
there that knew what a shaderbox frame is made of.
"""

from dataclasses import dataclass, replace

from imgui_bundle import imgui, imgui_ctx

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
from shaderbox.ui_primitives import caption_text, chip_button, clipped_caption

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
    label-plus-number-plus-unit string is six, over the caption budget in
    `ai_docs/conventions.md`.
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


