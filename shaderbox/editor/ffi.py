"""ctypes binding for the vendored libeditor.so (feature 067).

Leaf module: no imgui, no moderngl. Derived from the editor repo's
`examples/python/editor_widget.py` (the reference binding), adapted to repo
conventions. The library handle lives in a module global populated by the
idempotent `ensure_loaded()`; text getters share ONE module-level scratch
buffer (the frame loop is single-threaded).

Method names mirror the old `TextEditor` surface where call sites survive:
`get_undo_index` is `ed_revision` (monotonic, RISES across `set_text` — a
re-baseline must read it AFTER the set), `get_current_cursor_position`
returns a `.line`-carrying tuple, `replace_text_in_current_cursor` is the
host paste-at-caret.
"""

import ctypes
from collections.abc import Sequence
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple

from shaderbox.constants import RESOURCES_DIR

EDITOR_RESOURCES_DIR: Path = RESOURCES_DIR / "editor"

_LIB: ctypes.CDLL | None = None

# Shared scratch for text getters (the frame loop is single-threaded). Grows on
# demand: the ABI truncates on a codepoint boundary, so a result filling the
# buffer to within one codepoint is retried bigger rather than saved truncated.
_TEXT_BUF = (ctypes.c_ubyte * (1 << 20))()


def _grow_text_buf() -> None:
    global _TEXT_BUF
    _TEXT_BUF = (ctypes.c_ubyte * (len(_TEXT_BUF) * 2))()


class Prim(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int32)] + [
        (n, ctypes.c_float)
        for n in ("x0", "y0", "x1", "y1", "u0", "v0", "u1", "v1", "r", "g", "b", "a")
    ]


PRIM_STRIDE: int = ctypes.sizeof(Prim)


class Kind(IntEnum):
    BACKGROUND = 0
    SELECTION = 1
    GLYPH = 2
    CARET = 3
    FRAME = 4
    POPUP_PANEL = 5
    POPUP_GLYPH = 6
    MISSING_GLYPH = 7
    WHITESPACE = 8
    BRACKET_MATCH = 9


class Mode(IntEnum):
    NORMAL = 0
    INSERT = 1
    VISUAL = 2
    VISUAL_LINE = 3


class Language(IntEnum):
    NONE = 0
    PYTHON = 1
    GLSL = 2


class Slot(IntEnum):
    BACKGROUND = 0
    TEXT = 1
    CARET = 2
    CARET_INSERT = 3
    SELECTION = 4
    GUTTER_TEXT = 5
    GUTTER_CURRENT = 6
    FILLER = 7
    STATUS_BG = 8
    STATUS_TEXT = 9
    STATUS_ACCENT = 10
    POPUP_PANEL = 11
    POPUP_TEXT = 12
    POPUP_SELECTED = 13
    SYNTAX_1 = 14
    SYNTAX_2 = 15
    SYNTAX_3 = 16
    SYNTAX_4 = 17
    SYNTAX_5 = 18
    SYNTAX_6 = 19
    SYNTAX_7 = 20
    WHITESPACE = 21
    BRACKET_MATCH = 22


class ViewFlag(IntEnum):
    SHOW_SPACES = 0
    SHOW_TABS = 1
    SHOW_MATCHING_BRACKETS = 2


class ChromeFlag(IntEnum):
    LINE_NUMBERS = 0
    RELATIVE_NUMBERS = 1
    STATUS_LINE = 2
    STATUS_SHOWS_MODE = 3
    STATUS_SHOWS_RULER = 4


class KeyCode(IntEnum):
    CHAR = 1
    ESCAPE = 2
    ENTER = 3
    TAB = 4
    BACKSPACE = 5
    DELETE = 6
    LEFT = 7
    RIGHT = 8
    UP = 9
    DOWN = 10
    HOME = 11
    END = 12
    PAGE_UP = 13
    PAGE_DOWN = 14


class KeyMod(IntEnum):
    NONE = 0
    CTRL = 1
    ALT = 2
    SHIFT = 4
    SUPER = 8


class CursorPos(NamedTuple):
    line: int
    column: int


class HostCommandKind(IntEnum):
    WRITE = 1
    QUIT = 2
    WRITE_QUIT = 3


class HostCommand(NamedTuple):
    kind: HostCommandKind
    force: bool
    arg: str


_P = ctypes.POINTER
_SIG: dict[str, tuple[object, Sequence[object]]] = {
    "ed_new": (ctypes.c_void_p, [ctypes.c_char_p]),
    "ed_free": (None, [ctypes.c_void_p]),
    "ed_feed": (None, [ctypes.c_void_p, ctypes.c_char_p]),
    "ed_text": (ctypes.c_int32, [ctypes.c_void_p, _P(ctypes.c_ubyte), ctypes.c_int32]),
    "ed_set_text": (None, [ctypes.c_void_p, ctypes.c_char_p]),
    "ed_cursor": (None, [ctypes.c_void_p, _P(ctypes.c_int32), _P(ctypes.c_int32)]),
    "ed_set_cursor": (None, [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]),
    "ed_mode": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_revision": (ctypes.c_uint64, [ctypes.c_void_p]),
    "ed_line_count": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_set_read_only": (None, [ctypes.c_void_p, ctypes.c_bool]),
    "ed_load_atlas": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_char_p]),
    "ed_atlas_distance_range": (ctypes.c_float, [ctypes.c_void_p]),
    "ed_layout": (
        ctypes.c_int32,
        [ctypes.c_void_p] + [ctypes.c_float] * 5 + [ctypes.c_bool],
    ),
    "ed_primitives": (ctypes.c_int32, [ctypes.c_void_p, _P(Prim), ctypes.c_int32]),
    "ed_complete_prefix": (
        ctypes.c_int32,
        [ctypes.c_void_p, _P(ctypes.c_ubyte), ctypes.c_int32],
    ),
    "ed_complete_begin": (None, [ctypes.c_void_p]),
    "ed_complete_push": (None, [ctypes.c_void_p, ctypes.c_char_p]),
    "ed_complete_open": (ctypes.c_bool, [ctypes.c_void_p]),
    "ed_complete_count": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_complete_selected": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_complete_item": (
        ctypes.c_int32,
        [ctypes.c_void_p, ctypes.c_int32, _P(ctypes.c_ubyte), ctypes.c_int32],
    ),
    "ed_complete_cancel": (None, [ctypes.c_void_p]),
    "ed_set_host_completion": (None, [ctypes.c_void_p, ctypes.c_bool]),
    "ed_take_host_command": (
        ctypes.c_int32,
        [
            ctypes.c_void_p,
            _P(ctypes.c_bool),
            _P(ctypes.c_ubyte),
            ctypes.c_int32,
            _P(ctypes.c_int32),
        ],
    ),
    "ed_take_scroll_request": (ctypes.c_bool, [ctypes.c_void_p, _P(ctypes.c_int32)]),
    "ed_register": (
        ctypes.c_int32,
        [ctypes.c_void_p, _P(ctypes.c_ubyte), ctypes.c_int32],
    ),
    "ed_register_linewise": (ctypes.c_bool, [ctypes.c_void_p]),
    "ed_set_register": (None, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]),
    "ed_set_color": (
        ctypes.c_bool,
        [ctypes.c_void_p, ctypes.c_int32] + [ctypes.c_float] * 4,
    ),
    "ed_clear_markers": (None, [ctypes.c_void_p]),
    "ed_add_marker": (
        None,
        [ctypes.c_void_p, ctypes.c_int32]
        + [ctypes.c_float] * 8
        + [ctypes.c_int32, ctypes.c_char_p],
    ),
    "ed_marker_tooltip": (
        ctypes.c_int32,
        [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            _P(ctypes.c_ubyte),
            ctypes.c_int32,
        ],
    ),
    "ed_set_scroll": (None, [ctypes.c_void_p, ctypes.c_int32]),
    "ed_scroll": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_scroll_max": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_scroll_to_line": (None, [ctypes.c_void_p, ctypes.c_int32, ctypes.c_bool]),
    "ed_pixel_over_glyph": (
        ctypes.c_bool,
        [ctypes.c_void_p, ctypes.c_float, ctypes.c_float],
    ),
    "ed_word_at_pixel": (
        ctypes.c_int32,
        [
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_bool,
            _P(ctypes.c_ubyte),
            ctypes.c_int32,
        ],
    ),
    "ed_pixel_to_cursor": (
        None,
        [
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_float,
            _P(ctypes.c_int32),
            _P(ctypes.c_int32),
        ],
    ),
    "ed_set_language": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int32]),
    "ed_language_for_path": (ctypes.c_int32, [ctypes.c_char_p]),
    "ed_set_chrome_flag": (
        ctypes.c_bool,
        [ctypes.c_void_p, ctypes.c_int32, ctypes.c_bool],
    ),
    "ed_gutter_cells": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_key": (ctypes.c_bool, [ctypes.c_void_p] + [ctypes.c_int32] * 3),
    "ed_selection": (ctypes.c_bool, [ctypes.c_void_p] + [_P(ctypes.c_int32)] * 4),
    "ed_selection_text": (
        ctypes.c_int32,
        [ctypes.c_void_p, _P(ctypes.c_ubyte), ctypes.c_int32],
    ),
    "ed_select_line": (None, [ctypes.c_void_p, ctypes.c_int32]),
    "ed_clear_selection": (None, [ctypes.c_void_p]),
    "ed_undo": (ctypes.c_bool, [ctypes.c_void_p]),
    "ed_redo": (ctypes.c_bool, [ctypes.c_void_p]),
    "ed_insert_at_cursor": (None, [ctypes.c_void_p, ctypes.c_char_p]),
    "ed_delete_range": (None, [ctypes.c_void_p] + [ctypes.c_int32] * 4),
    "ed_replace_selection": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_char_p]),
    "ed_set_selection": (None, [ctypes.c_void_p] + [ctypes.c_int32] * 4),
    "ed_pending": (ctypes.c_bool, [ctypes.c_void_p]),
    "ed_command_line": (
        ctypes.c_int32,
        [ctypes.c_void_p, _P(ctypes.c_ubyte), ctypes.c_int32],
    ),
    "ed_command_line_prompt": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_command_message": (
        ctypes.c_int32,
        [ctypes.c_void_p, _P(ctypes.c_ubyte), ctypes.c_int32],
    ),
    "ed_cell_size": (None, [ctypes.c_void_p, _P(ctypes.c_float), _P(ctypes.c_float)]),
    "ed_text_origin": (None, [ctypes.c_void_p, _P(ctypes.c_float), _P(ctypes.c_float)]),
    "ed_set_view_flag": (
        ctypes.c_bool,
        [ctypes.c_void_p, ctypes.c_int32, ctypes.c_bool],
    ),
    "ed_set_show_whitespace": (None, [ctypes.c_void_p, ctypes.c_bool]),
    "ed_set_line_spacing": (None, [ctypes.c_void_p, ctypes.c_float]),
    "ed_set_tab_width": (None, [ctypes.c_void_p, ctypes.c_int32]),
}


def ensure_loaded() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(EDITOR_RESOURCES_DIR / "libeditor.so"))
        for name, (restype, argtypes) in _SIG.items():
            fn = getattr(lib, name)
            fn.restype = restype
            fn.argtypes = argtypes
        _LIB = lib
    return _LIB


def language_for_path(path: Path) -> Language:
    """Language by extension; unknown falls back to GLSL (host policy — every
    non-Python file in a document dir is shader source)."""
    lib = ensure_loaded()
    lang = Language(lib.ed_language_for_path(str(path).encode()))
    return Language.GLSL if lang == Language.NONE else lang


class Editor:
    """One editor instance bound to one file's text: the TextEditor replacement."""

    def __init__(self, text: str = "") -> None:
        lib = ensure_loaded()
        self._lib = lib
        self._h: int | None = lib.ed_new(text.encode())
        # Grown on demand by layout_bulk so the common case reuses one buffer.
        self._prims = (Prim * 4096)()
        self._prim_count: int = 0
        if not lib.ed_load_atlas(
            self._h, str(EDITOR_RESOURCES_DIR / "atlas.json").encode()
        ):
            raise RuntimeError("editor atlas failed to load")

    def close(self) -> None:
        if self._h:
            self._lib.ed_free(self._h)
            self._h = None

    # --- text -------------------------------------------------------------

    def get_text(self) -> str:
        n = self._lib.ed_text(self._h, _TEXT_BUF, len(_TEXT_BUF))
        while n >= len(_TEXT_BUF) - 4:
            _grow_text_buf()
            n = self._lib.ed_text(self._h, _TEXT_BUF, len(_TEXT_BUF))
        return bytes(_TEXT_BUF[:n]).decode()

    def set_text(self, text: str) -> None:
        self._lib.ed_set_text(self._h, text.encode())

    def get_line_count(self) -> int:
        return self._lib.ed_line_count(self._h)

    def get_undo_index(self) -> int:
        return self._lib.ed_revision(self._h)

    # --- cursor / mode ----------------------------------------------------

    def get_current_cursor_position(self) -> CursorPos:
        line, col = ctypes.c_int32(), ctypes.c_int32()
        self._lib.ed_cursor(self._h, ctypes.byref(line), ctypes.byref(col))
        return CursorPos(line.value, col.value)

    def set_cursor(self, line: int, col: int) -> None:
        self._lib.ed_set_cursor(self._h, line, col)

    def get_mode(self) -> Mode:
        return Mode(self._lib.ed_mode(self._h))

    def is_pending(self) -> bool:
        """True while the keymap is mid-phrase; ask BEFORE forwarding Escape."""
        return self._lib.ed_pending(self._h)

    def set_read_only_enabled(self, on: bool) -> None:
        self._lib.ed_set_read_only(self._h, on)

    # --- input ------------------------------------------------------------

    def feed(self, keys: str) -> None:
        """Vim notation — tests and scripts only; the app pumps key()."""
        self._lib.ed_feed(self._h, keys.encode())

    def key(self, code: KeyCode, mods: int = 0, text: str = "") -> bool:
        """One key press. True if the editor consumed it. A multi-codepoint
        `text` is refused (one platform event carries one codepoint)."""
        if len(text) > 1:
            return False
        return self._lib.ed_key(self._h, int(code), mods, ord(text) if text else 0)

    def replace_text_in_current_cursor(self, text: str) -> None:
        # Host insert: over the selection when one exists, else at the caret.
        if not self._lib.ed_replace_selection(self._h, text.encode()):
            self._lib.ed_insert_at_cursor(self._h, text.encode())

    def undo(self) -> bool:
        return self._lib.ed_undo(self._h)

    def redo(self) -> bool:
        return self._lib.ed_redo(self._h)

    # --- selection --------------------------------------------------------

    def get_selection(self) -> tuple[CursorPos, CursorPos] | None:
        a, b, c, d = (ctypes.c_int32() for _ in range(4))
        got = self._lib.ed_selection(
            self._h, ctypes.byref(a), ctypes.byref(b), ctypes.byref(c), ctypes.byref(d)
        )
        if not got:
            return None
        return CursorPos(a.value, b.value), CursorPos(c.value, d.value)

    def get_selection_text(self) -> str | None:
        n = self._lib.ed_selection_text(self._h, _TEXT_BUF, len(_TEXT_BUF))
        while n >= len(_TEXT_BUF) - 4:
            _grow_text_buf()
            n = self._lib.ed_selection_text(self._h, _TEXT_BUF, len(_TEXT_BUF))
        return None if n < 0 else bytes(_TEXT_BUF[:n]).decode()

    def set_selection(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        """Anchor at start, head at end; call repeatedly to drag."""
        self._lib.ed_set_selection(self._h, start[0], start[1], end[0], end[1])

    def select_line(self, line: int) -> None:
        self._lib.ed_select_line(self._h, line)

    def clear_selection(self) -> None:
        """The host's Escape: drops the selection AND any pending phrase."""
        self._lib.ed_clear_selection(self._h)

    def replace_selection(self, text: str) -> bool:
        return self._lib.ed_replace_selection(self._h, text.encode())

    def delete_range(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        """Host delete, one undo step; end is exclusive, columns are codepoints."""
        self._lib.ed_delete_range(self._h, start[0], start[1], end[0], end[1])

    # --- language / theme -------------------------------------------------

    def set_language(self, lang: Language) -> None:
        if not self._lib.ed_set_language(self._h, int(lang)):
            raise ValueError(f"unknown language: {lang}")

    def set_palette(
        self, palette: dict[Slot, tuple[float, float, float, float]]
    ) -> None:
        for slot, rgba in palette.items():
            if not self._lib.ed_set_color(self._h, int(slot), *rgba):
                raise ValueError(f"unknown theme slot: {slot}")

    # --- markers ----------------------------------------------------------

    def clear_markers(self) -> None:
        self._lib.ed_clear_markers(self._h)

    def add_marker(
        self,
        line: int,
        fill: tuple[float, float, float, float],
        gutter: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        tooltip: str = "",
    ) -> None:
        self._lib.ed_add_marker(
            self._h, line, *fill, *gutter, 0, tooltip.encode() if tooltip else None
        )

    # --- scrolling --------------------------------------------------------

    def get_scroll(self) -> int:
        return self._lib.ed_scroll(self._h)

    def set_scroll(self, rows: int) -> None:
        self._lib.ed_set_scroll(self._h, rows)

    def get_scroll_max(self) -> int:
        return self._lib.ed_scroll_max(self._h)

    def scroll_to_line(self, line: int, align_middle: bool = False) -> None:
        self._lib.ed_scroll_to_line(self._h, line, align_middle)

    # --- hit testing (widget-space pixels, against the last layout) --------

    def is_mouse_pos_over_glyph(self, pos: tuple[float, float]) -> bool:
        return self._lib.ed_pixel_over_glyph(self._h, pos[0], pos[1])

    def get_word_at_mouse_pos(
        self, pos: tuple[float, float], big: bool = False
    ) -> str | None:
        n = self._lib.ed_word_at_pixel(
            self._h, pos[0], pos[1], big, _TEXT_BUF, len(_TEXT_BUF)
        )
        return None if n < 0 else bytes(_TEXT_BUF[:n]).decode()

    def pixel_to_cursor(self, pos: tuple[float, float]) -> CursorPos:
        line, col = ctypes.c_int32(), ctypes.c_int32()
        self._lib.ed_pixel_to_cursor(
            self._h, pos[0], pos[1], ctypes.byref(line), ctypes.byref(col)
        )
        return CursorPos(line.value, col.value)

    # --- completion -------------------------------------------------------

    def complete_prefix(self) -> str:
        """The word prefix at the cursor. A BUFFER property, not a popup-state
        query (ask complete_open for that) — it reports whether or not a popup
        is open, which is exactly what a host deciding to offer needs."""
        n = self._lib.ed_complete_prefix(self._h, _TEXT_BUF, len(_TEXT_BUF))
        return "" if n <= 0 else bytes(_TEXT_BUF[:n]).decode()

    def complete_begin(self) -> None:
        self._lib.ed_complete_begin(self._h)

    def complete_push(self, text: str) -> None:
        """Pushing IS opening: feed candidates only on a deliberate offer."""
        self._lib.ed_complete_push(self._h, text.encode())

    def complete_open(self) -> bool:
        """Whether the popup is showing — i.e. what Enter will do."""
        return self._lib.ed_complete_open(self._h)

    def complete_count(self) -> int:
        return self._lib.ed_complete_count(self._h)

    def complete_selected(self) -> int:
        return self._lib.ed_complete_selected(self._h)

    def complete_item(self, index: int) -> str | None:
        n = self._lib.ed_complete_item(self._h, index, _TEXT_BUF, len(_TEXT_BUF))
        return None if n < 0 else bytes(_TEXT_BUF[:n]).decode()

    def complete_cancel(self) -> None:
        """Close the popup without touching the buffer or leaving INSERT."""
        self._lib.ed_complete_cancel(self._h)

    def get_register(self) -> str:
        """The unnamed register's text — what dd/yy/x wrote and p pastes."""
        n = self._lib.ed_register(self._h, _TEXT_BUF, len(_TEXT_BUF))
        while n >= len(_TEXT_BUF) - 4:
            _grow_text_buf()
            n = self._lib.ed_register(self._h, _TEXT_BUF, len(_TEXT_BUF))
        return "" if n <= 0 else bytes(_TEXT_BUF[:n]).decode()

    def set_register(self, text: str, linewise: bool = False) -> None:
        """Write the register — the host's way to make `p` paste the system
        clipboard (one slot, never two clipboards that disagree)."""
        self._lib.ed_set_register(self._h, text.encode(), linewise)

    def take_host_command(self) -> HostCommand | None:
        """A host-owned ex command (:w / :q / :wq family) the parser validated,
        or None. Reading CONSUMES it — poll once per frame after feeding keys."""
        force = ctypes.c_bool()
        n = ctypes.c_int32()
        kind = self._lib.ed_take_host_command(
            self._h, ctypes.byref(force), _TEXT_BUF, len(_TEXT_BUF), ctypes.byref(n)
        )
        if kind == 0:
            return None
        return HostCommand(
            HostCommandKind(kind), force.value, bytes(_TEXT_BUF[: n.value]).decode()
        )

    def take_scroll_request(self) -> int | None:
        """A view-only scroll the keymap asked for (Ctrl+E/Y, zz/zt/zb).
        MEASURED contract: reading consumes it, the returned value is the
        ABSOLUTE target row, and the scroll is applied by the read itself —
        the host just calls this once per frame after feeding keys."""
        rows = ctypes.c_int32()
        if not self._lib.ed_take_scroll_request(self._h, ctypes.byref(rows)):
            return None
        return rows.value

    def set_host_completion(self, on: bool) -> None:
        """Suppress the built-in buffer-word source: Ctrl+N stays consumed but
        opens nothing, so the host can detect the key and push its own list
        without a one-frame flash of buffer words."""
        self._lib.ed_set_host_completion(self._h, on)

    # --- chrome / view settings -------------------------------------------

    def set_chrome_flag(self, flag: ChromeFlag, on: bool) -> None:
        self._lib.ed_set_chrome_flag(self._h, int(flag), on)

    def set_view_flag(self, flag: ViewFlag, on: bool) -> None:
        self._lib.ed_set_view_flag(self._h, int(flag), on)

    def set_show_whitespace(self, on: bool) -> None:
        self._lib.ed_set_show_whitespace(self._h, on)

    def set_line_spacing(self, factor: float) -> None:
        self._lib.ed_set_line_spacing(self._h, factor)

    def set_tab_size(self, cells: int) -> None:
        self._lib.ed_set_tab_width(self._h, cells)

    def get_gutter_cells(self) -> int:
        return self._lib.ed_gutter_cells(self._h)

    def get_cell_size(self) -> tuple[float, float]:
        """Cell pixels the LAST layout used; (0, 0) before the first one."""
        w, h = ctypes.c_float(), ctypes.c_float()
        self._lib.ed_cell_size(self._h, ctypes.byref(w), ctypes.byref(h))
        return w.value, h.value

    def get_text_origin(self) -> tuple[float, float]:
        """Where the text starts, as the last layout placed it (right of the gutter)."""
        x, y = ctypes.c_float(), ctypes.c_float()
        self._lib.ed_text_origin(self._h, ctypes.byref(x), ctypes.byref(y))
        return x.value, y.value

    # --- command line -----------------------------------------------------

    def get_command_line(self) -> str | None:
        """Typed text without the prompt, or None while the line is closed."""
        n = self._lib.ed_command_line(self._h, _TEXT_BUF, len(_TEXT_BUF))
        return None if n < 0 else bytes(_TEXT_BUF[:n]).decode()

    def get_command_line_prompt(self) -> str | None:
        c = self._lib.ed_command_line_prompt(self._h)
        return None if c == 0 else chr(c)

    def get_command_message(self) -> str:
        n = self._lib.ed_command_message(self._h, _TEXT_BUF, len(_TEXT_BUF))
        return bytes(_TEXT_BUF[:n]).decode()

    # --- drawing ----------------------------------------------------------

    def get_atlas_distance_range(self) -> float:
        return self._lib.ed_atlas_distance_range(self._h)

    def layout(
        self,
        size: tuple[float, float],
        px_per_em: float,
        origin: tuple[float, float] = (0.0, 0.0),
        wrap: bool = False,
    ) -> int:
        """Lay the buffer out and pull the primitive array in one crossing.
        `origin` is where the TEXT starts in widget space — the host passes its
        gutter width as origin.x (ed_layout reserves nothing itself; the
        reference UI does the same). Returns the primitive count."""
        n = self._lib.ed_layout(
            self._h, origin[0], origin[1], size[0], size[1], px_per_em, wrap
        )
        if n <= 0:
            self._prim_count = 0
            return 0
        if len(self._prims) < n:
            self._prims = (Prim * n)()
        self._prim_count = self._lib.ed_primitives(self._h, self._prims, n)
        return self._prim_count

    def prims_array(self) -> tuple[ctypes.Array, int]:
        """The last layout's primitives as the raw ctypes array + valid count,
        good until the next layout(). numpy views it via the buffer protocol."""
        return self._prims, self._prim_count

    def prims_list(self) -> list[Prim]:
        """The last layout's primitives as objects — tests and probes, not the renderer."""
        return list(self._prims[: self._prim_count])
