"""shaderbox · gruvbox theme · v1

Drop-in style + color block for imgui-bundle 1.92.x.

After creating the imgui context and BEFORE the first frame::

    from imgui_bundle import imgui
    from shaderbox.theme import apply_theme, COLOR, SIZE, SPACE

    apply_theme(imgui.get_style(), accent="yellow", density="tight",
                rounding="subtle")

At runtime (Tweaks panel callbacks, future feature 009), call ``apply_theme(...)``
again with new args to re-skin. The function is idempotent.

This module is the live source of truth for the theme. (It originated from the
feature-005 design pass; the palette ramp has since diverged from that pass's
``ai_docs/design/tokens.json`` — that file is an archived snapshot, not a synced
source.)

Color framework (portable to a future non-gruvbox theme):
  - ``_P`` — the raw palette (named hues). The ONLY place literal colors live.
  - ``_ACCENTS`` — accent presets drawn from ``_P``; the user picks one. This is the
    single SWAPPABLE hue ("active / interactive / call-to-action"); ``apply_theme``
    rewrites ``COLOR.ACCENT_*`` from it.
  - ``_ColorBag`` (``COLOR``) — role tokens. Every role maps to a ``_P`` entry, never
    a literal. Roles are either swappable (``ACCENT_*``) or FIXED (everything else:
    ``SELECT`` / ``STATE_*`` / ``TAG`` / ...). The one fixed role that shares spatial
    context with the accent — ``SELECT`` (its outline nests inside the accent's) —
    must use a hue no accent preset and no state color uses; enforced by the
    import-time invariant below. (State colors are status text, so they may overlap
    an accent hue.)
  Adding a theme = supply a new ``_P`` + ``_ACCENTS`` + role mapping; the invariant
  check tells you at import whether the SELECT assignment is valid.
"""

from typing import Literal

from imgui_bundle import imgui

# ----------------------------------------------------------------------------
# Public types
# ----------------------------------------------------------------------------

AccentName = Literal["yellow", "aqua", "orange", "blue"]
DensityName = Literal["tight", "comfortable"]
RoundingName = Literal["sharp", "subtle", "rounded"]


# ============================================================================
# Palette — raw gruvbox-dark values (RGBA floats in 0..1)
# ============================================================================


def _hex(h: str, a: float = 1.0) -> tuple[float, float, float, float]:
    h = h.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b, a)


# Gruvbox-hard palette: near-black floor (bg_0h) backs surfaces, app bg is bg_0
_P: dict[str, tuple[float, float, float, float]] = {
    "bg_0h": _hex("#161819"),
    "bg_0": _hex("#1d2021"),
    "bg_0s": _hex("#222526"),
    "bg_1": _hex("#282828"),
    "bg_2": _hex("#3c3836"),
    "bg_3": _hex("#504945"),
    "bg_4": _hex("#665c54"),
    "gray": _hex("#928374"),
    "fg_4": _hex("#a89984"),
    "fg_3": _hex("#bdae93"),
    "fg_2": _hex("#d5c4a1"),
    "fg_1": _hex("#ebdbb2"),
    "fg_0": _hex("#fbf1c7"),
    "red_n": _hex("#cc241d"),
    "red_b": _hex("#fb4934"),
    "green_n": _hex("#98971a"),
    "green_b": _hex("#b8bb26"),
    "yellow_n": _hex("#d79921"),
    "yellow_b": _hex("#fabd2f"),
    "blue_n": _hex("#458588"),
    "blue_b": _hex("#83a598"),
    "purple_n": _hex("#b16286"),
    "purple_b": _hex("#d3869b"),
    "aqua_n": _hex("#689d6a"),
    "aqua_b": _hex("#8ec07c"),
    "orange_n": _hex("#d65d0e"),
    "orange_b": _hex("#fe8019"),
}


# Accent presets: (primary, active, alpha_fill)
_ACCENTS: dict[
    AccentName,
    tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ],
] = {
    "yellow": (_P["yellow_b"], _P["orange_b"], (250 / 255, 189 / 255, 47 / 255, 0.18)),
    "aqua": (_P["aqua_b"], _P["aqua_n"], (142 / 255, 192 / 255, 124 / 255, 0.18)),
    "orange": (_P["orange_b"], _P["orange_n"], (254 / 255, 128 / 255, 25 / 255, 0.18)),
    "blue": (_P["blue_b"], _P["blue_n"], (131 / 255, 165 / 255, 152 / 255, 0.22)),
}


# ============================================================================
# Public role tokens
# ============================================================================
# The codebase imports these instead of hardcoding hex strings.
# Accent fields are overwritten by apply_theme() when the user swaps accents.


class _ColorBag:
    # fully transparent fill (e.g. an invisible selectable carrying its own visual)
    TRANSPARENT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    # neutrals
    BG_APP: tuple[float, float, float, float] = _P["bg_0"]
    BG_SURFACE: tuple[float, float, float, float] = _P["bg_0h"]
    BG_POPUP: tuple[float, float, float, float] = _P["bg_0s"]
    BG_FRAME: tuple[float, float, float, float] = _P["bg_1"]
    BORDER: tuple[float, float, float, float] = _P["bg_2"]

    FG_PRIMARY: tuple[float, float, float, float] = _P["fg_1"]
    FG_SECONDARY: tuple[float, float, float, float] = _P["fg_2"]
    FG_MUTED: tuple[float, float, float, float] = _P["fg_4"]
    FG_DIM: tuple[float, float, float, float] = _P["gray"]
    FG_TITLE: tuple[float, float, float, float] = _P["fg_0"]

    CHIP_BG: tuple[float, float, float, float] = _P["bg_3"]
    CHIP_BG_HOVER: tuple[float, float, float, float] = _P["bg_4"]
    CHIP_FG: tuple[float, float, float, float] = _P["fg_3"]

    # accents — overwritten by apply_theme()
    ACCENT_PRIMARY: tuple[float, float, float, float] = _P["yellow_b"]
    ACCENT_ACTIVE: tuple[float, float, float, float] = _P["orange_b"]
    ACCENT_ALPHA: tuple[float, float, float, float] = (
        250 / 255,
        189 / 255,
        47 / 255,
        0.18,
    )

    # selection / context cue — FIXED (never written by apply_theme). The swappable
    # accent is the active-region/tab cue; selection must stay a distinct hue from it
    # under EVERY accent preset, so it can't be the accent. purple_b is used by no
    # accent preset and no state color.
    SELECT: tuple[float, float, float, float] = _P["purple_b"]

    # state semantics
    STATE_OK: tuple[float, float, float, float] = _P["aqua_b"]
    STATE_WARN: tuple[float, float, float, float] = _P["yellow_b"]
    STATE_ERROR: tuple[float, float, float, float] = _P["red_b"]
    STATE_INFO: tuple[float, float, float, float] = _P["blue_b"]

    # picker-specific roles
    TAG: tuple[float, float, float, float] = _P["blue_b"]
    FAVS: tuple[float, float, float, float] = _P["yellow_b"]
    RESET_PILL: tuple[float, float, float, float] = _P["purple_n"]

    # syntax (for imgui_color_text_edit palette wiring; feature 006)
    SYN_KEYWORD: tuple[float, float, float, float] = _P["red_b"]
    SYN_TYPE: tuple[float, float, float, float] = _P["yellow_b"]
    SYN_BUILTIN: tuple[float, float, float, float] = _P["green_b"]
    SYN_NUMBER: tuple[float, float, float, float] = _P["purple_b"]
    SYN_STRING: tuple[float, float, float, float] = _P["green_b"]
    SYN_COMMENT: tuple[float, float, float, float] = _P["gray"]
    SYN_PREPROC: tuple[float, float, float, float] = _P["aqua_b"]
    SYN_UNIFORM: tuple[float, float, float, float] = _P["blue_b"]
    SYN_IDENT: tuple[float, float, float, float] = _P["fg_1"]
    SYN_OP: tuple[float, float, float, float] = _P["fg_3"]


COLOR = _ColorBag()


# ----------------------------------------------------------------------------
# Theme-portability invariant (enforced at import).
#
# The color framework has TWO kinds of role: the SWAPPABLE accent (ACCENT_*, the
# one "active / interactive" hue, chosen per `_ACCENTS` preset and rewritten by
# `apply_theme`), and FIXED semantic hues (SELECT / STATE_* / TAG / FAVS / ...)
# that must read distinctly no matter which accent is picked. The whole scheme —
# and its portability to a future non-gruvbox palette — rests on one rule:
#
#   a FIXED hue may not equal ANY accent preset's primary, nor another fixed hue
#   it shares spatial context with.
#
# Break it (e.g. set SELECT to a hue some accent preset also uses) and the
# selection cue silently merges with the active-region cue the moment that accent
# is chosen — exactly the clash this framework exists to prevent. A new theme just
# supplies its own `_P` + `_ACCENTS` + role mapping; this check is what tells the
# author, at import, whether their assignment is valid. Keep it; extend the
# `_fixed` list when a new fixed cross-context role is added.
# ----------------------------------------------------------------------------
_accent_primaries: set[tuple[float, float, float, float]] = {
    primary for primary, _active, _alpha in _ACCENTS.values()
}
# SELECT is the one fixed role that shares SPATIAL context with the swappable accent
# — a selected-tile border can sit INSIDE an accent-outlined region, and the
# context-menu chrome floats over accent-bearing UI. So it must be distinct from
# every accent primary AND from every state hue (a selected-but-errored tile must
# still read 'error', not 'selected'). State colors are status TEXT in their own
# rows (not outlines nesting under the accent), so they MAY share a hue with an
# accent preset — that's a tolerable text/accent overlap, not the nested-outline
# clash. If a future fixed role gains accent-adjacent OUTLINE context, add it here.
assert COLOR.SELECT not in _accent_primaries, (
    f"theme invariant: SELECT={COLOR.SELECT} collides with an accent preset's "
    f"primary — pick a hue no accent uses, or the selection outline merges with the "
    f"active-region accent outline when that accent is selected."
)
assert COLOR.SELECT not in {
    COLOR.STATE_OK,
    COLOR.STATE_WARN,
    COLOR.STATE_ERROR,
    COLOR.STATE_INFO,
}, "theme invariant: SELECT must differ from every STATE_* hue."


# ============================================================================
# Size + spacing tokens
# ============================================================================


class SIZE:
    ROW_HEIGHT: int = 22

    BTN_SM_W: int = 80
    BTN_SM_H: int = 19

    CHIP_W: int = 64
    CHIP_ROUNDING: int = 8

    SORT_COMBO_W: int = 150
    TAB_MIN_W: int = 72
    NAME_INPUT_W: int = 180
    RES_COMBO_W: int = 200

    LABEL_W: int = 64
    RENDER_CTRL_W: int = 200
    RENDER_PREVIEW_W: int = 200

    UNIFORM_NAME_W: int = 140
    UNIFORM_CTRL_W: int = 320
    UNIFORM_TEXT_H: int = 72  # multiline text-uniform box (~3 rows)
    SMOOTHING_LABEL_W: int = 52  # label column for the video-smoothing sliders
    SMOOTHING_DRAG_W: int = 90  # the Window/Sigma drags beside a video thumbnail

    THUMB_SM: int = 90
    THUMB_LG: int = 150

    PREVIEW_W: int = 200
    PANEL_CTRL_MINH: int = 600

    # Shared share-panel preview box — every exporter's preview is this exact size
    # (one source of truth so all outlets match; fixed so it can't jitter). Height is
    # set generously so the preview is always taller than any outlet's control column
    # (no bottom-alignment math — controls just stack top-down beside it).
    SHARE_PREVIEW_W: int = 200
    SHARE_PREVIEW_H: int = 310

    FPS_PANEL_W: int = 160
    SETTINGS_W: int = 780
    SETTINGS_H: int = 484
    SETTINGS_LABEL_W: int = 92
    SETTINGS_CTRL_W: int = 120

    SCROLLBAR_W: int = 12
    GRAB_MIN: int = 10

    # Keyboard cheatsheet floating overlay (feature 018).
    CHEATSHEET_W: int = 230
    CHEATSHEET_MARGIN: int = 12  # gap from the window's bottom-right corner


class SPACE:
    XS: int = 2
    SM: int = 4
    MD: int = 8
    LG: int = 16
    XL: int = 24


def fade(
    color: tuple[float, float, float, float], a: float
) -> tuple[float, float, float, float]:
    """Same RGB, new alpha — for translucent washes off a solid token."""
    return (color[0], color[1], color[2], a)


# Whole-pane alpha for the code editor when it lacks keyboard focus; style.Alpha
# reaches the editor's glyph draws.
EDITOR_UNFOCUSED_ALPHA: float = 0.6

# Fill alpha for chrome floating over the render image (the FPS overlay).
OVERLAY_ALPHA: float = 0.7

# The keyboard cheatsheet floats top-right over live content — kept faint.
CHEATSHEET_ALPHA: float = 0.45


# ============================================================================
# apply_theme
# ============================================================================


def apply_theme(
    style: imgui.Style | None = None,
    *,
    accent: AccentName = "yellow",
    density: DensityName = "tight",
    rounding: RoundingName = "subtle",
) -> None:
    """Write the gruvbox theme into the given ImGuiStyle.

    Re-callable to swap accent/density/rounding at runtime.
    """
    if style is None:
        style = imgui.get_style()

    primary, active, alpha = _ACCENTS[accent]
    COLOR.ACCENT_PRIMARY = primary
    COLOR.ACCENT_ACTIVE = active
    COLOR.ACCENT_ALPHA = alpha

    if density == "comfortable":
        SPACE.XS, SPACE.SM, SPACE.MD, SPACE.LG, SPACE.XL = 3, 6, 10, 18, 28
        frame_pad = imgui.ImVec2(8.0, 4.0)
        SIZE.ROW_HEIGHT = 26
    else:
        SPACE.XS, SPACE.SM, SPACE.MD, SPACE.LG, SPACE.XL = 2, 4, 8, 16, 24
        frame_pad = imgui.ImVec2(6.0, 3.0)
        SIZE.ROW_HEIGHT = 22

    style.window_padding = imgui.ImVec2(SPACE.MD, SPACE.MD)
    style.frame_padding = frame_pad
    style.cell_padding = imgui.ImVec2(SPACE.SM, SPACE.XS)
    style.item_spacing = imgui.ImVec2(SPACE.MD, SPACE.SM)
    style.item_inner_spacing = imgui.ImVec2(SPACE.SM, SPACE.SM)
    style.indent_spacing = 18.0
    style.scrollbar_size = float(SIZE.SCROLLBAR_W)
    style.grab_min_size = float(SIZE.GRAB_MIN)
    style.columns_min_spacing = 6.0

    rd = {
        "sharp": {
            "frame": 0,
            "child": 0,
            "window": 0,
            "popup": 0,
            "grab": 0,
            "tab": 0,
            "scroll": 4,
        },
        "subtle": {
            "frame": 2,
            "child": 4,
            "window": 4,
            "popup": 4,
            "grab": 2,
            "tab": 2,
            "scroll": 6,
        },
        "rounded": {
            "frame": 4,
            "child": 6,
            "window": 6,
            "popup": 6,
            "grab": 4,
            "tab": 4,
            "scroll": 8,
        },
    }[rounding]
    style.window_rounding = float(rd["window"])
    style.child_rounding = float(rd["child"])
    style.frame_rounding = float(rd["frame"])
    style.popup_rounding = float(rd["popup"])
    style.scrollbar_rounding = float(rd["scroll"])
    style.grab_rounding = float(rd["grab"])
    style.tab_rounding = float(rd["tab"])

    style.window_border_size = 1.0
    style.child_border_size = 1.0
    style.frame_border_size = 0.0
    style.popup_border_size = 1.0
    style.tab_border_size = 0.0
    style.tab_bar_overline_size = 3.0
    style.tab_min_width_base = float(SIZE.TAB_MIN_W)

    style.window_title_align = imgui.ImVec2(0.0, 0.5)
    style.button_text_align = imgui.ImVec2(0.5, 0.5)
    style.selectable_text_align = imgui.ImVec2(0.0, 0.0)

    style.alpha = 1.0
    style.disabled_alpha = 0.5
    style.anti_aliased_lines = True
    style.anti_aliased_fill = True

    _set_colors(style, primary, active)


# ============================================================================
# Color slot assignments
# ============================================================================


def _set_colors(
    style: imgui.Style,
    accent_primary: tuple[float, float, float, float],
    accent_active: tuple[float, float, float, float],
) -> None:
    """Map every relevant ImGuiCol_* slot to a gruvbox palette token.

    All slot names verified against imgui-bundle 1.92.801's pyi stub.
    Pre-1.91 names (nav_highlight, tab_active, tab_unfocused*) are
    intentionally absent — they were renamed in Dear ImGui 1.91+.

    Uses imgui-bundle's `Style.set_color_(idx, ImVec4Like)` accessor since
    the C++ `Colors[]` array isn't directly exposed as a Python attribute.
    """
    col = imgui.Col_

    # text
    style.set_color_(col.text, COLOR.FG_PRIMARY)
    style.set_color_(col.text_disabled, COLOR.FG_DIM)

    # windows + frames
    style.set_color_(col.window_bg, _P["bg_0"])
    style.set_color_(col.child_bg, (0.0, 0.0, 0.0, 0.0))
    style.set_color_(col.popup_bg, _P["bg_0s"])
    style.set_color_(col.border, _P["bg_2"])
    style.set_color_(col.border_shadow, (0.0, 0.0, 0.0, 0.0))

    style.set_color_(col.frame_bg, _P["bg_1"])
    style.set_color_(col.frame_bg_hovered, _P["bg_2"])
    style.set_color_(col.frame_bg_active, _P["bg_3"])

    # title bars + menu bar
    style.set_color_(col.title_bg, _P["bg_0h"])
    style.set_color_(col.title_bg_active, _P["bg_1"])
    style.set_color_(col.title_bg_collapsed, _P["bg_0h"])
    style.set_color_(col.menu_bar_bg, _P["bg_1"])

    # scrollbar
    style.set_color_(col.scrollbar_bg, _P["bg_0h"])
    style.set_color_(col.scrollbar_grab, _P["bg_3"])
    style.set_color_(col.scrollbar_grab_hovered, _P["bg_4"])
    style.set_color_(col.scrollbar_grab_active, _P["gray"])

    # accent slots
    style.set_color_(col.check_mark, accent_primary)
    style.set_color_(col.slider_grab, accent_primary)
    style.set_color_(col.slider_grab_active, accent_active)
    style.set_color_(col.drag_drop_target, accent_primary)
    style.set_color_(col.text_selected_bg, fade(accent_primary, 0.40))

    # nav cursor (Dear ImGui 1.91+ name; replaces nav_highlight)
    style.set_color_(col.nav_cursor, accent_primary)
    style.set_color_(col.nav_windowing_highlight, fade(accent_primary, 0.70))
    style.set_color_(col.nav_windowing_dim_bg, (0.0, 0.0, 0.0, 0.20))

    # buttons
    style.set_color_(col.button, _P["bg_2"])
    style.set_color_(col.button_hovered, _P["bg_3"])
    style.set_color_(col.button_active, _P["bg_4"])

    # headers
    style.set_color_(col.header, _P["bg_2"])
    style.set_color_(col.header_hovered, _P["bg_3"])
    style.set_color_(col.header_active, _P["bg_4"])

    # separators
    style.set_color_(col.separator, _P["bg_3"])
    style.set_color_(col.separator_hovered, _P["bg_4"])
    style.set_color_(col.separator_active, accent_primary)

    # resize grips
    style.set_color_(col.resize_grip, _P["bg_3"])
    style.set_color_(col.resize_grip_hovered, fade(accent_primary, 0.60))
    style.set_color_(col.resize_grip_active, accent_primary)

    # tabs (Dear ImGui 1.91+ names)
    style.set_color_(col.tab, _P["bg_1"])
    style.set_color_(col.tab_hovered, fade(accent_primary, 0.32))
    style.set_color_(col.tab_selected, fade(accent_primary, 0.32))
    style.set_color_(col.tab_selected_overline, accent_primary)
    style.set_color_(col.tab_dimmed, _P["bg_0h"])
    style.set_color_(col.tab_dimmed_selected, fade(accent_primary, 0.20))
    style.set_color_(col.tab_dimmed_selected_overline, accent_primary)

    # docking (unused today, harmless)
    style.set_color_(col.docking_preview, fade(accent_primary, 0.40))
    style.set_color_(col.docking_empty_bg, _P["bg_0h"])

    # plots
    style.set_color_(col.plot_lines, _P["aqua_b"])
    style.set_color_(col.plot_lines_hovered, _P["orange_b"])
    style.set_color_(col.plot_histogram, accent_primary)
    style.set_color_(col.plot_histogram_hovered, _P["orange_b"])

    # modal veil
    style.set_color_(col.modal_window_dim_bg, (0.0, 0.0, 0.0, 0.55))
