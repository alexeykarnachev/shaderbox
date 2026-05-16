"""shaderbox · gruvbox theme · v1

Drop-in style + color block for imgui-bundle 1.92.x.

After creating the imgui context and BEFORE the first frame::

    from imgui_bundle import imgui
    from shaderbox.theme import apply_theme, COLOR, SIZE, SPACE

    apply_theme(imgui.get_style(), accent="yellow", density="tight",
                rounding="subtle")

At runtime (Tweaks panel callbacks, future feature 009), call ``apply_theme(...)``
again with new args to re-skin. The function is idempotent.

Every color / size / spacing value comes from ``ai_docs/design/tokens.json`` —
keep the two in sync.
"""

from pathlib import Path
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


_P: dict[str, tuple[float, float, float, float]] = {
    "bg_0h": _hex("#1d2021"),
    "bg_0": _hex("#282828"),
    "bg_0s": _hex("#32302f"),
    "bg_1": _hex("#3c3836"),
    "bg_2": _hex("#504945"),
    "bg_3": _hex("#665c54"),
    "bg_4": _hex("#7c6f64"),
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

    # accents — overwritten by apply_theme()
    ACCENT_PRIMARY: tuple[float, float, float, float] = _P["yellow_b"]
    ACCENT_ACTIVE: tuple[float, float, float, float] = _P["orange_b"]
    ACCENT_ALPHA: tuple[float, float, float, float] = (
        250 / 255,
        189 / 255,
        47 / 255,
        0.18,
    )

    # state semantics
    STATE_OK: tuple[float, float, float, float] = _P["aqua_b"]
    STATE_WARN: tuple[float, float, float, float] = _P["yellow_b"]
    STATE_ERROR: tuple[float, float, float, float] = _P["red_b"]
    STATE_INFO: tuple[float, float, float, float] = _P["blue_b"]

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


# ============================================================================
# Size + spacing tokens
# ============================================================================


class SIZE:
    ROW_HEIGHT: int = 22
    ROW_COMPACT: int = 19

    TOPBAR: int = 32
    STATUSBAR: int = 24

    BTN_SM_W: int = 80
    BTN_MD_W: int = 120
    BTN_SM_H: int = 19

    THUMB_SM: int = 90
    THUMB_MD: int = 110
    THUMB_LG: int = 150

    PREVIEW_W: int = 200
    PANEL_CTRL_MINH: int = 600
    PANEL_RIGHT_W: int = 720
    RENDER_MAX_H: int = 360
    RENDER_MIN_H: int = 220

    SCROLLBAR_W: int = 12
    GRAB_MIN: int = 10
    GUTTER_W: int = 48

    TG_THUMB_H: int = 90
    TG_GRID_COLS: int = 4

    NODE_CREATOR_COLS: int = 3


class SPACE:
    XS: int = 2
    SM: int = 4
    MD: int = 8
    LG: int = 16
    XL: int = 24


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
    style.child_border_size = 0.0
    style.frame_border_size = 0.0
    style.popup_border_size = 1.0
    style.tab_border_size = 0.0

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

    def fade(
        cl: tuple[float, float, float, float], a: float
    ) -> tuple[float, float, float, float]:
        return (cl[0], cl[1], cl[2], a)

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
    style.set_color_(col.tab_hovered, _P["bg_2"])
    style.set_color_(col.tab_selected, _P["bg_0"])
    style.set_color_(col.tab_selected_overline, accent_primary)
    style.set_color_(col.tab_dimmed, _P["bg_0h"])
    style.set_color_(col.tab_dimmed_selected, _P["bg_1"])
    style.set_color_(col.tab_dimmed_selected_overline, _P["bg_2"])

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


# ============================================================================
# Project paths (for future feature 009 — Tweaks panel persistence)
# ============================================================================

_FONT_DIR: Path = Path(__file__).parent / "resources" / "fonts"
