#!/usr/bin/env python3
"""Exercises the whole C ABI from Python, the way a host would.

Not a test of the editor -- `odin test src` covers that. This checks the
BOUNDARY: that every exported function is callable through ctypes with no
bindings layer, and that state persists across separate calls.

    make ffi && python3 ffi/probe.py
"""

import ctypes
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LIB = ROOT / "libeditor.so"

if not LIB.exists():
    sys.exit(f"{LIB} missing -- run `make ffi` first")


class Prim(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int32)] + [
        (n, ctypes.c_float)
        for n in ("x0", "y0", "x1", "y1", "u0", "v0", "u1", "v1", "r", "g", "b", "a")
    ]


KINDS = [
    "Background", "Selection", "Glyph", "Caret",
    "Frame", "Popup_Panel", "Popup_Glyph", "Missing_Glyph",
    "Whitespace", "Bracket_Match", "Search_Match",
]
MODES = ["NORMAL", "INSERT", "VISUAL", "V-LINE"]
# Key codes and modifier bits, matching the ED_KEY_* / ED_MOD_* constants.
K_CHAR, K_ESC, K_ENTER, K_TAB, K_BS = 1, 2, 3, 4, 5
K_DEL, K_LEFT, K_RIGHT, K_UP, K_DOWN = 6, 7, 8, 9, 10
K_HOME, K_END, K_PAGE_UP, K_PAGE_DOWN = 11, 12, 13, 14
M_NONE, M_CTRL, M_ALT, M_SHIFT, M_SUPER = 0, 1, 2, 4, 8
LANGS = {"None": 0, "Python": 1, "GLSL": 2}
FLAGS = {
    n: i
    for i, n in enumerate(
        "Line_Numbers Relative_Numbers Status_Line Status_Shows_Mode "
        "Status_Shows_Ruler".split()
    )
}
STYLES = {"Vim": 0, "Standard": 1}
CLASSES = ["None", "Keyword", "String", "Comment", "Number", "Operator", "Builtin"]

# Theme_Slot, in enum order. Names are the contract; the numbers are what a
# host stores, which is why the enum only ever grows at the end.
SLOTS = {
    name: i
    for i, name in enumerate(
        "Background Text Caret Caret_Insert Selection Gutter_Text Gutter_Current "
        "Filler Status_Bg Status_Text Status_Accent Popup_Panel Popup_Text "
        "Popup_Selected Syntax_1 Syntax_2 Syntax_3 Syntax_4 Syntax_5 Syntax_6 "
        "Syntax_7 Whitespace Bracket_Match Search_Match".split()
    )
}

VIEW = {n: i for i, n in enumerate("Show_Spaces Show_Tabs Show_Matching_Brackets Highlight_Search".split())}

lib = ctypes.CDLL(str(LIB))
_SIG = {
    "ed_new": (ctypes.c_void_p, [ctypes.c_char_p]),
    "ed_free": (None, [ctypes.c_void_p]),
    "ed_feed": (None, [ctypes.c_void_p, ctypes.c_char_p]),
    "ed_text": (ctypes.c_int32, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]),
    "ed_set_text": (None, [ctypes.c_void_p, ctypes.c_char_p]),
    "ed_cursor": (None, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]),
    "ed_set_cursor": (None, [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]),
    "ed_mode": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_revision": (ctypes.c_uint64, [ctypes.c_void_p]),
    "ed_line_count": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_set_read_only": (None, [ctypes.c_void_p, ctypes.c_bool]),
    "ed_load_atlas": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_char_p]),
    "ed_atlas_distance_range": (ctypes.c_float, [ctypes.c_void_p]),
    "ed_layout": (ctypes.c_int32, [ctypes.c_void_p] + [ctypes.c_float] * 5 + [ctypes.c_bool]),
    "ed_primitive": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(Prim)]),
    "ed_complete_prefix": (ctypes.c_int32, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]),
    "ed_complete_begin": (None, [ctypes.c_void_p]),
    "ed_complete_push": (None, [ctypes.c_void_p, ctypes.c_char_p]),
    "ed_set_color": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int32] + [ctypes.c_float] * 4),
    "ed_primitives": (ctypes.c_int32, [ctypes.c_void_p, ctypes.POINTER(Prim), ctypes.c_int32]),
    "ed_set_host_completion": (None, [ctypes.c_void_p, ctypes.c_bool]),
    "ed_host_completion": (ctypes.c_bool, [ctypes.c_void_p]),
    "ed_complete_open": (ctypes.c_bool, [ctypes.c_void_p]),
    "ed_complete_count": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_complete_item": (ctypes.c_int32, [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]),
    "ed_complete_selected": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_complete_cancel": (None, [ctypes.c_void_p]),
    "ed_pending": (ctypes.c_bool, [ctypes.c_void_p]),
    "ed_command_line": (ctypes.c_int32, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]),
    "ed_command_line_prompt": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_command_message": (ctypes.c_int32, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]),
    "ed_cell_size": (None, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]),
    "ed_text_origin": (None, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]),
    "ed_set_view_flag": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int32, ctypes.c_bool]),
    "ed_view_flag": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_bool)]),
    "ed_set_show_whitespace": (None, [ctypes.c_void_p, ctypes.c_bool]),
    "ed_set_line_spacing": (None, [ctypes.c_void_p, ctypes.c_float]),
    "ed_line_spacing": (ctypes.c_float, [ctypes.c_void_p]),
    "ed_register": (ctypes.c_int32, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]),
    "ed_register_linewise": (ctypes.c_bool, [ctypes.c_void_p]),
    "ed_set_register": (None, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]),
    "ed_paste": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_bool, ctypes.c_int32]),
    "ed_take_scroll_request": (ctypes.c_bool, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]),
    "ed_take_host_command": (ctypes.c_int32, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool),
                                              ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32,
                                              ctypes.POINTER(ctypes.c_int32)]),
    "ed_find": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool]),
    "ed_find_next": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_bool]),
    "ed_find_count": (ctypes.c_int32, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]),
    "ed_replace_at_cursor": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool]),
    "ed_replace_all": (ctypes.c_int32, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool]),
    "ed_color": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int32] + [ctypes.POINTER(ctypes.c_float)] * 4),
    "ed_reset_theme": (None, [ctypes.c_void_p]),
    "ed_clear_markers": (None, [ctypes.c_void_p]),
    "ed_add_marker": (
        None,
        [ctypes.c_void_p, ctypes.c_int32] + [ctypes.c_float] * 12 + [ctypes.c_int32, ctypes.c_char_p],
    ),
    "ed_marker_count": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_key": (ctypes.c_bool, [ctypes.c_void_p] + [ctypes.c_int32] * 3),
    "ed_selection": (ctypes.c_bool, [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_int32)] * 4),
    "ed_selection_text": (ctypes.c_int32, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]),
    "ed_select_line": (None, [ctypes.c_void_p, ctypes.c_int32]),
    "ed_clear_selection": (None, [ctypes.c_void_p]),
    "ed_undo": (ctypes.c_bool, [ctypes.c_void_p]),
    "ed_redo": (ctypes.c_bool, [ctypes.c_void_p]),
    "ed_insert_at_cursor": (None, [ctypes.c_void_p, ctypes.c_char_p]),
    "ed_insert_at": (None, [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_char_p]),
    "ed_delete_range": (None, [ctypes.c_void_p] + [ctypes.c_int32] * 4),
    "ed_replace_selection": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_char_p]),
    "ed_set_selection": (None, [ctypes.c_void_p] + [ctypes.c_int32] * 4),
    "ed_set_line_selection": (None, [ctypes.c_void_p] + [ctypes.c_int32] * 2),
    "ed_set_tab_width": (None, [ctypes.c_void_p, ctypes.c_int32]),
    "ed_tab_width": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_set_chrome_flag": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int32, ctypes.c_bool]),
    "ed_chrome_flag": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_bool)]),
    "ed_set_number_width": (None, [ctypes.c_void_p, ctypes.c_int32]),
    "ed_set_filler_glyph": (None, [ctypes.c_void_p, ctypes.c_int32]),
    "ed_filler_glyph": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_gutter_cells": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_set_chrome_style": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int32]),
    "ed_set_draw_chrome": (None, [ctypes.c_void_p, ctypes.c_bool]),
    "ed_draw_chrome": (ctypes.c_bool, [ctypes.c_void_p]),
    "ed_set_style": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int32]),
    "ed_style": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_set_language": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int32]),
    "ed_language": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_language_for_path": (ctypes.c_int32, [ctypes.c_char_p]),
    "ed_class_at": (ctypes.c_int32, [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]),
    "ed_pixel_over_glyph": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]),
    "ed_word_at_pixel": (
        ctypes.c_int32,
        [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_bool,
         ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32],
    ),
    "ed_pixel_to_cursor": (
        None,
        [ctypes.c_void_p, ctypes.c_float, ctypes.c_float,
         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)],
    ),
    "ed_set_scroll": (None, [ctypes.c_void_p, ctypes.c_int32]),
    "ed_scroll": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_scroll_max": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_scroll_to_line": (None, [ctypes.c_void_p, ctypes.c_int32, ctypes.c_bool]),
    "ed_marker_gutter": (
        ctypes.c_bool,
        [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
        + [ctypes.POINTER(ctypes.c_float)] * 4
        + [ctypes.POINTER(ctypes.c_int32)],
    ),
    "ed_marker_tooltip": (
        ctypes.c_int32,
        [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32],
    ),
}
for name, (restype, argtypes) in _SIG.items():
    fn = getattr(lib, name)
    fn.restype, fn.argtypes = restype, argtypes

_buf = (ctypes.c_ubyte * 65536)()


def text(h):
    return bytes(_buf[: lib.ed_text(h, _buf, 65536)]).decode()


def cstr(n):
    return bytes(_buf[: max(0, n)]).decode()


def reg(h):
    return bytes(_buf[: lib.ed_register(h, _buf, 65536)]).decode()


def cmdline(h):
    n = lib.ed_command_line(h, _buf, 65536)
    return None if n < 0 else bytes(_buf[:n]).decode()


def message(h):
    return bytes(_buf[: lib.ed_command_message(h, _buf, 65536)]).decode()


def cursor(h):
    line, col = ctypes.c_int32(), ctypes.c_int32()
    lib.ed_cursor(h, ctypes.byref(line), ctypes.byref(col))
    return line.value, col.value


def color(h, slot):
    ch = [ctypes.c_float() for _ in range(4)]
    if not lib.ed_color(h, slot, *(ctypes.byref(c) for c in ch)):
        return None
    return tuple(round(c.value, 4) for c in ch)


def tooltip(h, line, index):
    n = lib.ed_marker_tooltip(h, line, index, _buf, 65536)
    return None if n < 0 else bytes(_buf[:n]).decode()


def word_at(h, x, y, big=False):
    n = lib.ed_word_at_pixel(h, x, y, big, _buf, 65536)
    return None if n < 0 else bytes(_buf[:n]).decode()


def chrome_flag(h, flag):
    out = ctypes.c_bool()
    if not lib.ed_chrome_flag(h, flag, ctypes.byref(out)):
        return None
    return out.value


def selection(h):
    a, b, c, d = (ctypes.c_int32() for _ in range(4))
    if not lib.ed_selection(h, *(ctypes.byref(v) for v in (a, b, c, d))):
        return None
    return (a.value, b.value), (c.value, d.value)


def selection_text(h):
    n = lib.ed_selection_text(h, _buf, 65536)
    return None if n < 0 else bytes(_buf[:n]).decode()


def prefix(h):
    return bytes(_buf[: lib.ed_complete_prefix(h, _buf, 65536)]).decode()


def main() -> int:
    failures = 0

    def check(label, got, want):
        nonlocal failures
        ok = got == want
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}" + ("" if ok else f" want {want!r}"))

    print("editing")
    h = lib.ed_new(b"void main() {\n    vec3 c = vec3(1.0);\n}")
    check("line count", lib.ed_line_count(h), 3)
    lib.ed_feed(h, b"jwciwcolor<Esc>")
    check("ciw", text(h), "void main() {\n    color c = vec3(1.0);\n}")
    check("cursor", cursor(h), (1, 8))
    check("mode", MODES[lib.ed_mode(h)], "NORMAL")

    print("revision and read-only")
    # The revision must be MONOTONIC across a whole-buffer replace. It is not
    # the buffer's own version, which restarts at 1: a host that saved at 2,
    # loaded a file and made one edit would read its saved number back over
    # different text, and the dirty dot would silently clear.
    seq = [lib.ed_revision(h)]
    lib.ed_feed(h, b"x")
    seq.append(lib.ed_revision(h))
    lib.ed_set_text(h, b"a whole new buffer")
    seq.append(lib.ed_revision(h))
    lib.ed_feed(h, b"x")
    seq.append(lib.ed_revision(h))
    check("revision only ever rises", all(b > a for a, b in zip(seq, seq[1:])), True)
    check("a saved revision never recurs", seq[1] in seq[2:], False)

    rev = lib.ed_revision(h)
    lib.ed_feed(h, b"x")
    check("edit moves revision", lib.ed_revision(h) != rev, True)
    lib.ed_set_read_only(h, True)
    rev = lib.ed_revision(h)
    lib.ed_feed(h, b"dd")
    check("read-only holds it", lib.ed_revision(h), rev)
    lib.ed_set_text(h, b"vec3 p;")
    check("host writes anyway", text(h), "vec3 p;")
    lib.ed_set_read_only(h, False)

    print("columns are codepoints, in both directions")
    # ed_cursor reports codepoint columns, so every column the ABI takes must
    # mean the same thing -- otherwise set_cursor(cursor()) moves the cursor,
    # and a selection cannot be compared against it. Tabs and multi-byte text
    # are where a byte or display-cell column diverges.
    for src in ("h\tabc", "h\u00e9llo w\u00f6rld", "plain"):
        lib.ed_set_text(h, src.encode())
        lib.ed_feed(h, b"ll")
        before = cursor(h)
        lib.ed_set_cursor(h, *before)
        check(f"set_cursor(cursor()) is the identity on {src!r}", cursor(h), before)

    # In visual mode the cursor may sit ON the newline, one past the last
    # character, which is how `vl` at the end of a line selects the break.
    lib.ed_set_text(h, b"abc\ndef")
    lib.ed_feed(h, b"$vl")
    before = cursor(h)
    check("a visual cursor may sit one past the last character", before, (0, 3))
    lib.ed_set_cursor(h, *before)
    check("and set_cursor(cursor()) is the identity there too", cursor(h), before)
    lib.ed_feed(h, b"<Esc>")
    check("leaving visual mode clamps it back", cursor(h), (0, 2))

    lib.ed_set_text(h, "h\u00e9llo w\u00f6rld".encode())
    lib.ed_feed(h, b"lllvll")
    sel = selection(h)
    check("a selection starts at the cursor's own column", sel[0], (0, 3))
    lib.ed_clear_selection(h)

    lib.ed_set_text(h, "h\u00e9llo".encode())
    lib.ed_insert_at(h, 0, 2, b"!")
    check("insert_at counts codepoints too", text(h), "h\u00e9!llo")

    print("navigation")
    lib.ed_set_text(h, b"a\nbb\nccc")
    lib.ed_set_cursor(h, 2, 1)
    check("set_cursor", cursor(h), (2, 1))

    print("drawing")
    check("atlas", lib.ed_load_atlas(h, str(ROOT / "assets/atlas.json").encode()), True)
    check("distance range", lib.ed_atlas_distance_range(h), 8.0)
    n = lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)
    check("primitives emitted", n > 0, True)
    seen, p = {}, Prim()
    for i in range(n):
        lib.ed_primitive(h, i, ctypes.byref(p))
        seen[KINDS[p.kind]] = seen.get(KINDS[p.kind], 0) + 1
    check("kinds", sorted(seen), ["Caret", "Glyph"])

    # The array is in DRAW order, because that is what this README tells a host
    # to do with it. Emission order is not: a selection is emitted after the
    # glyphs it covers and a marker's fill after everything, so a host walking
    # the array and drawing painted an opaque selection over its own text.
    lib.ed_set_text(h, b"hello world")
    lib.ed_feed(h, b"vlll")
    lib.ed_clear_markers(h)
    lib.ed_add_marker(h, 0, 0.8, 0.2, 0.2, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, None)
    n = lib.ed_layout(h, 0.0, 0.0, 400.0, 200.0, 16.0, False)
    kinds = []
    for i in range(n):
        lib.ed_primitive(h, i, ctypes.byref(p))
        kinds.append(p.kind)
    check("a selection and a marker are both present", len(set(kinds)) >= 3, True)
    check("and the array is in draw order", kinds, sorted(kinds))
    lib.ed_clear_selection(h)
    lib.ed_clear_markers(h)
    check("index past the end refuses", lib.ed_primitive(h, n, ctypes.byref(p)), False)

    print("theme")
    check("default text colour", color(h, SLOTS["Text"]), (0.878, 0.886, 0.918, 1.0))
    check("set", lib.ed_set_color(h, SLOTS["Text"], 1.0, 0.0, 0.5, 1.0), True)
    check("reads back", color(h, SLOTS["Text"]), (1.0, 0.0, 0.5, 1.0))
    check("slot past the end refuses", lib.ed_set_color(h, len(SLOTS), 1.0, 1.0, 1.0, 1.0), False)
    check("negative slot refuses", lib.ed_set_color(h, -1, 1.0, 1.0, 1.0, 1.0), False)
    check("unknown slot reads nothing", color(h, len(SLOTS)), None)

    # A setter that stores a colour nothing draws with would pass a round-trip
    # check and still be useless, so ask the primitives what they got.
    lib.ed_set_text(h, b"abc")
    n = lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)
    glyph = None
    for i in range(n):
        lib.ed_primitive(h, i, ctypes.byref(p))
        if KINDS[p.kind] == "Glyph":
            glyph = (p.r, p.g, p.b, p.a)
            break
    check("glyphs draw in the set colour", tuple(round(c, 4) for c in glyph), (1.0, 0.0, 0.5, 1.0))

    lib.ed_reset_theme(h)
    check("reset restores the default", color(h, SLOTS["Text"]), (0.878, 0.886, 0.918, 1.0))

    print("markers")
    lib.ed_set_text(h, b"a\nb\nc")
    lib.ed_reset_theme(h)
    lib.ed_clear_markers(h)
    check("empty to start", lib.ed_marker_count(h), 0)
    lib.ed_add_marker(h, 1, 0.8, 0.1, 0.1, 0.25, 1.0, 0.4, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0, ord(">"), b"undefined identifier")
    check("added", lib.ed_marker_count(h), 1)
    check("tooltip comes back", tooltip(h, 1, 0), "undefined identifier")
    check("unmarked line has none", tooltip(h, 0, 0), None)
    check("index past the end refuses", tooltip(h, 1, 1), None)

    # A marker must reach the drawn primitives, and its fill must sort BEFORE
    # the glyphs -- draw order is enum order, so a fill emitted as any later
    # kind would paint over the code it marks.
    n = lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)
    kinds = []
    for i in range(n):
        lib.ed_primitive(h, i, ctypes.byref(p))
        kinds.append(KINDS[p.kind])
    check("fill emitted", kinds.count("Background"), 1)
    check("fill sorts before glyphs", KINDS.index("Background") < KINDS.index("Glyph"), True)

    # ed_layout draws no gutter, so a marker may not place anything in one.
    # Giving the emit a Vim Chrome put the gutter mark four cells left of the
    # origin, off the widget entirely, where no host would ever see it. Glyph
    # ink legitimately overhangs its cell by a fraction of a pixel, so the
    # threshold is a cell, not zero.
    leftmost = 0.0
    for i in range(n):
        lib.ed_primitive(h, i, ctypes.byref(p))
        leftmost = min(leftmost, p.x0)
    check("nothing drawn in a gutter ed_layout never made", leftmost > -12.0, True)

    # The gutter mark crosses as data instead, for the host to draw in the
    # column it owns.
    gc = [ctypes.c_float() for _ in range(4)]
    gg = ctypes.c_int32()
    got = lib.ed_marker_gutter(h, 1, 0, *(ctypes.byref(c) for c in gc), ctypes.byref(gg))
    check("gutter mark crosses", got, True)
    check("gutter colour", tuple(round(c.value, 4) for c in gc), (1.0, 0.4, 0.4, 1.0))
    check("gutter glyph", chr(gg.value), ">")
    check("unmarked line has no gutter mark",
          lib.ed_marker_gutter(h, 0, 0, *(ctypes.byref(c) for c in gc), ctypes.byref(gg)), False)

    lib.ed_clear_markers(h)
    check("cleared", lib.ed_marker_count(h), 0)
    n = lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)
    fills = 0
    for i in range(n):
        lib.ed_primitive(h, i, ctypes.byref(p))
        fills += KINDS[p.kind] == "Background"
    check("and stops drawing", fills, 0)

    # A marker follows its line between rebuilds. shaderbox's report: an
    # error on line 9, a line inserted at 7, and the red band still on 9 while
    # the code that caused it sat on 10.
    lib.ed_set_text(h, "\n".join(f"line {i}" for i in range(20)).encode())
    lib.ed_clear_markers(h)
    lib.ed_add_marker(h, 9, 0.8, 0.1, 0.1, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, b"error")
    lib.ed_set_cursor(h, 7, 0)
    lib.ed_feed(h, b"Oinserted")
    lib.ed_key(h, K_ESC, 0, 0)
    check("the marker moved with its line", tooltip(h, 10, 0), "error")
    check("and left the old line", tooltip(h, 9, 0), None)
    lib.ed_feed(h, b"dd")
    check("deleting the line above moves it back", tooltip(h, 9, 0), "error")
    lib.ed_set_cursor(h, 9, 0)
    lib.ed_feed(h, b"dd")
    check("deleting the marked line lands it on the next", tooltip(h, 9, 0), "error")
    lib.ed_set_text(h, "\n".join(f"other {i}" for i in range(20)).encode())
    check("a whole-text replace keeps it where it was", tooltip(h, 9, 0), "error")
    lib.ed_set_cursor(h, 0, 0)
    lib.ed_feed(h, b"dd")
    check("and it follows edits in the new text", tooltip(h, 8, 0), "error")

    # The text colour replaces the syntax colour on the marked line only.
    lib.ed_set_text(h, b"ab\ncd")
    lib.ed_clear_markers(h)
    lib.ed_add_marker(h, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0, None)
    n = lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)
    cw, ch = ctypes.c_float(), ctypes.c_float()
    lib.ed_cell_size(h, ctypes.byref(cw), ctypes.byref(ch))
    rows = {}
    for i in range(n):
        lib.ed_primitive(h, i, ctypes.byref(p))
        if KINDS[p.kind] == "Glyph":
            row = int((p.y0 + p.y1) * 0.5 / ch.value)
            rows.setdefault(row, set()).add((round(p.r, 2), round(p.g, 2), round(p.b, 2)))
    check("the marked line's glyphs take the marker colour", rows.get(1), {(1.0, 0.0, 0.0)})
    check("the other line keeps its own", (1.0, 0.0, 0.0) in rows.get(0, set()), False)
    lib.ed_clear_markers(h)

    print("scrolling")
    lib.ed_set_text(h, "\n".join(f"line {i}" for i in range(100)).encode())
    lib.ed_clear_markers(h)
    lib.ed_set_scroll(h, 0)
    # 400px tall at 16px/em -> a viewport of some whole number of rows.
    n = lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)
    rows = lib.ed_scroll_max(h)
    check("a long buffer can scroll", rows > 0, True)

    def glyph_uvs():
        # The atlas cells of the first row of glyphs: which CHARACTERS are drawn
        # at the top of the view. Comparing y would prove nothing -- the top row
        # sits at the same y whether or not scrolling works.
        n = lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)
        rowtop, cells = None, []
        for i in range(n):
            lib.ed_primitive(h, i, ctypes.byref(p))
            if KINDS[p.kind] != "Glyph":
                continue
            if rowtop is None or p.y0 < rowtop - 0.5:
                rowtop, cells = p.y0, []
            if abs(p.y0 - rowtop) < 8.0:
                cells.append(round(p.u0, 5))
        return cells

    at_top = glyph_uvs()
    lib.ed_set_scroll(h, 10)
    check("the offset holds", lib.ed_scroll(h), 10)
    check("scrolling does not move the buffer", lib.ed_line_count(h), 100)
    check("a different line is now on top", glyph_uvs() != at_top, True)
    lib.ed_set_scroll(h, 0)
    check("and scrolling back restores it", glyph_uvs(), at_top)
    lib.ed_set_scroll(h, 10)
    lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)

    # Whatever the host asks for, it reads back what actually took effect.
    lib.ed_set_scroll(h, 100000)
    lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)
    check("clamped to the last screenful", lib.ed_scroll(h), rows)
    lib.ed_set_scroll(h, -5)
    check("negative clamps to the top", lib.ed_scroll(h), 0)

    # Jump-to-error, which is what this exists for.
    lib.ed_set_scroll(h, 0)
    lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)
    lib.ed_scroll_to_line(h, 3, False)
    check("an on-screen line does not move the view", lib.ed_scroll(h), 0)
    lib.ed_scroll_to_line(h, 80, True)
    lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)
    centred = lib.ed_scroll(h)
    check("centring puts the line inside the view", 0 < centred < rows, True)
    lib.ed_scroll_to_line(h, 99, False)
    lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)
    check("the last line is reachable", lib.ed_scroll(h), rows)

    print("hit testing")
    lib.ed_set_scroll(h, 0)
    lib.ed_set_text(h, b"vec3 u_time.xyz;\nsecond line")
    n = lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)
    # Cell width at 16px/em on the shipped atlas; the probe asks the layout
    # rather than hardcoding it, so a re-bake does not silently move the target.
    lib.ed_primitive(h, 0, ctypes.byref(p))
    cell_w = 800.0
    for i in range(n):
        lib.ed_primitive(h, i, ctypes.byref(p))
        if KINDS[p.kind] == "Glyph" and p.x0 > 0.5:
            cell_w = min(cell_w, p.x0)
    check("a cell is a sane width", 4.0 < cell_w < 20.0, True)

    inside = (7 * cell_w + cell_w / 2, 8.0)   # inside "u_time"
    check("over a glyph", lib.ed_pixel_over_glyph(h, *inside), True)
    check("word under the pointer", word_at(h, *inside), "u_time")
    check("WORD takes punctuation", word_at(h, *inside, True), "u_time.xyz;")

    # The cases pixel_to_buffer answers wrongly for a hover: it clamps, so it
    # returns a valid position in blank space.
    check("past the end of a line", lib.ed_pixel_over_glyph(h, 700.0, 8.0), False)
    check("and yields no word", word_at(h, 700.0, 8.0), None)
    check("below the last line", lib.ed_pixel_over_glyph(h, 10.0, 380.0), False)
    check("left of the origin", lib.ed_pixel_over_glyph(h, -10.0, 8.0), False)
    check("blank space between words", word_at(h, 4 * cell_w + cell_w / 2, 8.0), None)

    line, col = ctypes.c_int32(), ctypes.c_int32()
    lib.ed_pixel_to_cursor(h, *inside, ctypes.byref(line), ctypes.byref(col))
    check("pixel to cursor", (line.value, col.value), (0, 7))

    print("languages")
    check("unknown extension", lib.ed_language_for_path(b"notes.txt"), LANGS["None"])
    check("a .py file", lib.ed_language_for_path(b"/src/main.py"), LANGS["Python"])
    check("a .frag file", lib.ed_language_for_path(b"shader.frag"), LANGS["GLSL"])
    check("an unknown language refuses", lib.ed_set_language(h, 99), False)

    # The same text, two grammars: `#version` is a comment in Python and a
    # keyword in GLSL. That disagreement is the whole point of the selection,
    # and it is checked through the DRAWN class rather than the lexer directly.
    lib.ed_set_text(h, b"#version 330\nvec3 c;")
    for name, want in (("Python", "Comment"), ("GLSL", "Keyword")):
        check(f"set {name}", lib.ed_set_language(h, LANGS[name]), True)
        check("reads back", lib.ed_language(h), LANGS[name])
        lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)
        check(f"#version under {name}", CLASSES[lib.ed_class_at(h, 0, 2)], want)

    lib.ed_set_language(h, LANGS["GLSL"])
    lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)
    check("a GLSL type is a keyword", CLASSES[lib.ed_class_at(h, 1, 1)], "Keyword")

    # Highlighting must reach the drawn glyphs, not just the class query: a
    # syntax palette that never colours anything would pass every check above.
    lib.ed_set_color(h, SLOTS["Syntax_1"], 0.0, 1.0, 0.0, 1.0)
    n = lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)
    greens = 0
    for i in range(n):
        lib.ed_primitive(h, i, ctypes.byref(p))
        greens += KINDS[p.kind] == "Glyph" and (p.r, p.g, p.b) == (0.0, 1.0, 0.0)
    check("keywords draw in the syntax colour", greens > 0, True)

    # Highlighting is LEXICAL, and this pins its known limit rather than hiding
    # it: `mix` is classified by spelling, so a local declaration shadowing the
    # builtin is painted as the builtin. Telling those apart needs a parse, which
    # is what ed_set_language's override hook is for -- see ffi/README.md.
    lib.ed_set_text(h, b"float mix;\nvoid main(){ float a = mix(1.0, 2.0, mix); }")
    lib.ed_set_language(h, LANGS["GLSL"])
    lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)
    check("a call names a builtin", CLASSES[lib.ed_class_at(h, 1, 23)], "Builtin")
    check("a shadowing declaration reads as one too (lexical limit)",
          CLASSES[lib.ed_class_at(h, 0, 6)], "Builtin")

    lib.ed_set_language(h, LANGS["None"])
    lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)
    check("None turns highlighting off", lib.ed_class_at(h, 0, 2), 0)
    lib.ed_reset_theme(h)

    print("chrome")
    check("vim style", lib.ed_set_chrome_style(h, STYLES["Vim"]), True)
    check("vim names its mode", chrome_flag(h, FLAGS["Status_Shows_Mode"]), True)
    check("vim counts relative", chrome_flag(h, FLAGS["Relative_Numbers"]), True)
    check("vim fills with ~", chr(lib.ed_filler_glyph(h)), "~")

    check("standard style", lib.ed_set_chrome_style(h, STYLES["Standard"]), True)
    check("no mode to name", chrome_flag(h, FLAGS["Status_Shows_Mode"]), False)
    check("no filler column", lib.ed_filler_glyph(h), 0)
    check("an unknown style refuses", lib.ed_set_chrome_style(h, len(STYLES)), False)

    print("drawn chrome")
    lib.ed_set_text(h, b"a\nb\nc")
    lib.ed_set_chrome_style(h, STYLES["Vim"])
    lib.ed_clear_markers(h)
    check("off on a fresh handle", lib.ed_draw_chrome(h), False)
    lib.ed_set_draw_chrome(h, True)
    check("switches on", lib.ed_draw_chrome(h), True)
    lib.ed_set_cursor(h, 1, 0)
    W, H = 400.0, 200.0
    n = lib.ed_layout(h, 0.0, 0.0, W, H, 16.0, False)
    cw, ch = ctypes.c_float(), ctypes.c_float()
    lib.ed_cell_size(h, ctypes.byref(cw), ctypes.byref(ch))
    ox, oy = ctypes.c_float(), ctypes.c_float()
    lib.ed_text_origin(h, ctypes.byref(ox), ctypes.byref(oy))
    check("the text starts after the gutter", round(ox.value / cw.value), lib.ed_gutter_cells(h))
    prims = []
    for i in range(n):
        lib.ed_primitive(h, i, ctypes.byref(p))
        prims.append((KINDS[p.kind], p.x0, p.y0, p.x1, p.y1, (round(p.r, 2), round(p.g, 2), round(p.b, 2))))
    gutter_glyphs = [q for q in prims if q[0] == "Glyph" and q[3] <= ox.value + 0.5]
    # Three numbers -- 1, the cursor line's absolute 2, 1 -- and a filler on
    # every text row past the buffer, which stop above the status row.
    text_rows = int((H - ch.value) / ch.value)
    check("numbers and filler draw in the gutter", len(gutter_glyphs), 3 + (text_rows - 3))
    check("nothing drawn on the status row's height in the gutter",
          max(q[4] for q in gutter_glyphs) <= H - ch.value + 0.5, True)
    frames = [q for q in prims if q[0] == "Frame"]
    check("the status bar is one Frame on the bottom row",
          [(round(f[2]), round(f[4])) for f in frames], [(round(H - ch.value), round(H))])
    status_text = [q for q in prims if q[0] == "Popup_Glyph" and q[2] >= H - ch.value - 1]
    check("the mode badge and the ruler sit on it", len(status_text), len("NORMAL") + len("2,1"))
    # The command line replaces them while it is open.
    lib.ed_feed(h, b":wq")
    n = lib.ed_layout(h, 0.0, 0.0, W, H, 16.0, False)
    kinds = []
    for i in range(n):
        lib.ed_primitive(h, i, ctypes.byref(p))
        if p.y0 >= H - ch.value - 1:
            kinds.append(KINDS[p.kind])
    check("the command line takes the status row", kinds.count("Popup_Glyph"), len(":wq"))
    check("with a block caret after it", kinds.count("Popup_Panel"), 1)
    lib.ed_key(h, K_ESC, 0, 0)
    # A marker's mark lands in the gutter's separator cell, in its own colour.
    lib.ed_add_marker(h, 0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.4, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0, ord(">"), None)
    n = lib.ed_layout(h, 0.0, 0.0, W, H, 16.0, False)
    marks = []
    for i in range(n):
        lib.ed_primitive(h, i, ctypes.byref(p))
        if KINDS[p.kind] == "Glyph" and (round(p.r, 2), round(p.g, 2), round(p.b, 2)) == (1.0, 0.4, 0.4):
            marks.append(int((p.x0 + p.x1) * 0.5 / cw.value))
    check("the gutter mark is drawn, in the separator cell", marks, [lib.ed_gutter_cells(h) - 1])
    lib.ed_clear_markers(h)
    # A text colour reaches COLUMN 0 behind the gutter: the first glyph's ink
    # overhangs its cell to the left, and shaderbox measured column 0 keeping
    # its syntax colour on every line not starting with a space.
    lib.ed_set_text(h, b"int x;\nvec3 c = fn(x);\nAAAA")
    lib.ed_add_marker(h, 1, 0.8, 0.1, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.92, 0.86, 0.70, 1.0, 0, None)
    n = lib.ed_layout(h, 0.0, 0.0, W, H, 16.0, False)
    row = []
    for i in range(n):
        lib.ed_primitive(h, i, ctypes.byref(p))
        if KINDS[p.kind] == "Glyph" and ch.value <= (p.y0 + p.y1) * 0.5 < 2 * ch.value and p.x1 > ox.value:
            row.append((p.x0, (round(p.r, 2), round(p.g, 2), round(p.b, 2))))
    row.sort()
    check("the first glyph overhangs the origin", row[0][0] < ox.value, True)
    check("and takes the marker colour with the rest", sorted({c for _, c in row}), [(0.92, 0.86, 0.7)])
    check("every glyph of the line is there", len(row), len("vec3 c = fn(x);".replace(" ", "")))
    lib.ed_clear_markers(h)
    # Hit testing answers against the offset text, not the rect's corner.
    line, col = ctypes.c_int32(), ctypes.c_int32()
    lib.ed_pixel_to_cursor(h, ox.value + cw.value * 0.5, ch.value * 0.5, ctypes.byref(line), ctypes.byref(col))
    check("a pixel on the first glyph resolves there", (line.value, col.value), (0, 0))
    check("a pixel in the gutter is over no glyph", lib.ed_pixel_over_glyph(h, cw.value * 0.5, ch.value * 0.5), False)
    lib.ed_set_draw_chrome(h, False)
    n = lib.ed_layout(h, 0.0, 0.0, W, H, 16.0, False)
    lib.ed_text_origin(h, ctypes.byref(ox), ctypes.byref(oy))
    frames = 0
    for i in range(n):
        lib.ed_primitive(h, i, ctypes.byref(p))
        frames += KINDS[p.kind] == "Frame"
    check("off again: the text starts at the rect's corner", ox.value, 0.0)
    check("and no status bar is drawn", frames, 0)

    print("search highlight")
    lib.ed_set_text(h, b"foo bar\nbaz foo\nfoobar foo")
    lib.ed_set_draw_chrome(h, False)
    lib.ed_clear_markers(h)
    flag = ctypes.c_bool()
    lib.ed_view_flag(h, VIEW["Highlight_Search"], ctypes.byref(flag))
    check("on by default", flag.value, True)

    def bands():
        n = lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, False)
        out = []
        for i in range(n):
            lib.ed_primitive(h, i, ctypes.byref(p))
            if KINDS[p.kind] == "Search_Match":
                out.append((round(p.y0 / ch.value), round(p.x0 / cw.value), round(p.x1 / cw.value)))
        return sorted(out)
    lib.ed_cell_size(h, ctypes.byref(cw), ctypes.byref(ch))
    check("nothing lit before a search", bands(), [])
    check("a host find lights every match", (lib.ed_find(h, b"foo", False, False), bands())[1],
          [(0, 0, 3), (1, 4, 7), (2, 0, 3), (2, 7, 10)])
    check("Escape puts it out", (lib.ed_key(h, K_ESC, 0, 0), bands())[1], [])
    lib.ed_feed(h, b"n")
    check("and n brings it back", len(bands()), 4)
    # `*` searches the word whole: "foobar" stays dark.
    lib.ed_set_cursor(h, 0, 0)
    lib.ed_feed(h, b"*")
    check("* lands on the next whole word", cursor(h), (1, 4))
    check("and lights only whole words", bands(), [(0, 0, 3), (1, 4, 7), (2, 7, 10)])
    lib.ed_feed(h, b"n")
    check("n keeps *'s whole-word rule", cursor(h), (2, 7))
    lib.ed_set_cursor(h, 0, 0)
    lib.ed_feed(h, b"g*")
    check("g* takes substrings", cursor(h), (1, 4))
    lib.ed_feed(h, b"n")
    check("(n after g* too)", cursor(h), (2, 0))
    # Typing a / line lights the text so far, before Enter.
    lib.ed_key(h, K_ESC, 0, 0)
    lib.ed_feed(h, b"/ba")
    check("incsearch lights the typed prefix", bands(), [(0, 4, 6), (1, 0, 2), (2, 3, 5)])
    lib.ed_key(h, K_ESC, 0, 0)
    check("Escape on the line drops it", bands(), [])
    check("the flag turns it off", (lib.ed_set_view_flag(h, VIEW["Highlight_Search"], False),
                                    lib.ed_feed(h, b"n"), bands())[2], [])
    lib.ed_set_view_flag(h, VIEW["Highlight_Search"], True)
    check("the slot colours the band", lib.ed_set_color(h, SLOTS["Search_Match"], 0.1, 0.2, 0.3, 0.4), True)
    lib.ed_feed(h, b"n")
    n = lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, False)
    got = None
    for i in range(n):
        lib.ed_primitive(h, i, ctypes.byref(p))
        if KINDS[p.kind] == "Search_Match":
            got = tuple(round(c, 2) for c in (p.r, p.g, p.b, p.a))
            break
    check("and reaches the primitive", got, (0.1, 0.2, 0.3, 0.4))
    lib.ed_reset_theme(h)
    lib.ed_key(h, K_ESC, 0, 0)

    print("shift and the empty last line")
    lib.ed_set_text(h, b"a\nb\nc")
    lib.ed_feed(h, b">>")
    check(">> indents the line", text(h), "    a\nb\nc")
    lib.ed_feed(h, b"jVj>")
    check("visual > shifts the selection", text(h), "    a\n    b\n    c")
    lib.ed_feed(h, b"gg<<")
    check("<< takes it back", text(h), "a\n    b\n    c")
    lib.ed_feed(h, b"u")
    check("one undo step per shift", text(h), "    a\n    b\n    c")
    lib.ed_set_text(h, b"a\n")
    lib.ed_feed(h, b"jdd")
    check("dd on an empty last line removes it", text(h), "a")

    print("keymap style")
    # The keymap style is separate from the chrome style: ed_set_chrome_style
    # never moves the keymap, and ed_set_style moves both.
    check("a fresh handle is vim", lib.ed_style(h), STYLES["Vim"])
    check("chrome alone leaves the keymap", lib.ed_style(h), STYLES["Vim"])
    check("set the standard style", lib.ed_set_style(h, STYLES["Standard"]), True)
    check("reads back", lib.ed_style(h), STYLES["Standard"])
    check("and sets the chrome with it", chrome_flag(h, FLAGS["Status_Shows_Mode"]), False)
    check("the mode is insert", lib.ed_mode(h), 1)
    check("an out-of-range style refuses", lib.ed_set_style(h, len(STYLES)), False)
    check("a negative style refuses", lib.ed_set_style(h, -1), False)
    check("and left the style alone", lib.ed_style(h), STYLES["Standard"])
    lib.ed_set_text(h, b"  abc")
    check("the style survives set_text", lib.ed_style(h), STYLES["Standard"])
    check("as does the mode", lib.ed_mode(h), 1)
    check("with the caret on the first non-blank", cursor(h), (0, 2))
    lib.ed_set_text(h, b"   ")
    check("or at a blank line's end", cursor(h), (0, 3))
    lib.ed_set_text(h, b"   \nabcdefgh")
    lib.ed_key(h, K_DOWN, 0, 0)
    check("set_text leaves the desired column at the caret", cursor(h), (1, 3))
    for width, want in ((8, 8), (2, 2)):
        lib.ed_set_tab_width(h, width)
        lib.ed_set_text(h, b"\tab\nxxxxxxxxxxxx")  # the caret lands after the tab
        lib.ed_key(h, K_DOWN, 0, 0)
        check(f"set_text's desired column uses tab width {width}", cursor(h), (1, want))
    lib.ed_set_tab_width(h, 4)
    lib.ed_set_text(h, b"   ")
    lib.ed_set_text(h, b"a\nb\nc\nd")
    lib.ed_set_line_selection(h, 2, 1)
    check("a backwards line selection is the same lines", selection_text(h), "b\nc\n")
    lib.ed_set_line_selection(h, 3, 0)
    check("whichever way round", selection_text(h), "a\nb\nc\nd")
    lib.ed_set_text(h, b"   ")
    check("back to vim", lib.ed_set_style(h, STYLES["Vim"]), True)
    check("the mode is normal", lib.ed_mode(h), 0)
    check("and the caret is clamped", cursor(h), (0, 2))
    lib.ed_set_chrome_style(h, STYLES["Standard"])
    check("chrome alone still leaves the keymap", lib.ed_style(h), STYLES["Vim"])
    lib.ed_set_chrome_style(h, STYLES["Vim"])
    for width, want in ((8, 8), (2, 2)):
        lib.ed_set_tab_width(h, width)
        lib.ed_set_text(h, b"\tab\nxxxxxxxxxxxx")
        lib.ed_key(h, K_DOWN, 0, 0)
        check(f"under vim too, at width {width}", cursor(h), (1, want))
    lib.ed_set_tab_width(h, 4)
    lib.ed_set_chrome_style(h, STYLES["Vim"])
    lib.ed_set_text(h, b"vec3 p;")

    print("the standard keymap through the ABI")
    # Every bullet of feature 005's "What each existing ABI call does under
    # Standard", driven through ed_key with real modifiers.
    lib.ed_set_style(h, STYLES["Standard"])
    lib.ed_set_text(h, b"hello world\nsecond")
    check("mode is insert", lib.ed_mode(h), 1)
    check("pending idle is false", lib.ed_pending(h), False)
    check("a plain character types", lib.ed_key(h, K_CHAR, 0, ord("x")), True)
    check("into the text", text(h), "xhello world\nsecond")
    check("Ctrl+Z undoes", lib.ed_key(h, K_CHAR, M_CTRL, ord("z")), True)
    check("back", text(h), "hello world\nsecond")
    check("Shift+Right selects", lib.ed_key(h, K_RIGHT, M_SHIFT, 0), True)
    check("selection reads back", selection(h), ((0, 0), (0, 1)))
    check("with its text", selection_text(h), "h")
    check("pending with a selection is true", lib.ed_pending(h), True)
    lib.ed_key(h, K_LEFT, M_SHIFT, 0)
    check("a collapsed selection is none", selection_text(h), None)
    check("and not pending", lib.ed_pending(h), False)
    check("Ctrl+A selects all", lib.ed_key(h, K_CHAR, M_CTRL, ord("a")), True)
    check("to the end", selection(h), ((0, 0), (1, 6)))
    lib.ed_complete_begin(h)
    lib.ed_complete_push(h, b"cand")
    check("replace_selection", lib.ed_replace_selection(h, b"new"), True)
    check("closes the popup like every host edit", lib.ed_complete_open(h), False)
    check("replaces it", text(h), "new")
    check("caret after the text", cursor(h), (0, 3))
    check("mode still insert", lib.ed_mode(h), 1)
    check("set_cursor honours a line's length", (lib.ed_set_cursor(h, 0, 3), cursor(h))[1], (0, 3))
    check("Ctrl+X is the host's", lib.ed_key(h, K_CHAR, M_CTRL, ord("x")), False)
    check("Ctrl+C is the host's", lib.ed_key(h, K_CHAR, M_CTRL, ord("c")), False)
    check("Ctrl+V is the host's", lib.ed_key(h, K_CHAR, M_CTRL, ord("v")), False)
    check("Alt chords are the host's", lib.ed_key(h, K_CHAR, M_ALT, ord("z")), False)
    check("an idle Escape is the host's", lib.ed_key(h, K_ESC, 0, 0), False)
    check("page keys are bound", lib.ed_key(h, K_PAGE_DOWN, 0, 0), True)
    check("and no scroll request follows", lib.ed_take_scroll_request(h, ctypes.byref(ctypes.c_int32())), False)
    # The selection setters: exclusive at the head, the mode untouched.
    lib.ed_set_text(h, b"abc\ndef")
    lib.ed_set_selection(h, 0, 1, 0, 3)
    check("set_selection is exclusive at the head", selection_text(h), "bc")
    check("and a drag may end at a line's end", (lib.ed_set_selection(h, 0, 0, 0, 3), selection(h))[1], ((0, 0), (0, 3)))
    check("mode stays insert", lib.ed_mode(h), 1)
    lib.ed_set_line_selection(h, 0, 0)
    check("a line selection runs through the newline", selection_text(h), "abc\n")
    lib.ed_select_line(h, 1)
    check("select_line the last line runs to the end", selection_text(h), "def")
    lib.ed_clear_selection(h)
    check("clear_selection deactivates it", selection_text(h), None)
    # Host edits land the bar caret at a line's end.
    lib.ed_set_cursor(h, 0, 0)
    lib.ed_delete_range(h, 0, 1, 0, 3)
    check("delete_range to a line's end reports the length", cursor(h), (0, 1))
    lib.ed_set_register(h, b"XY", False)
    lib.ed_set_cursor(h, 0, 1)
    check("paste", lib.ed_paste(h, False, 1), True)
    check("lands after the text", (text(h), cursor(h)), ("aXY\ndef", (0, 3)))
    lib.ed_set_register(h, b"L", True)
    lib.ed_paste(h, False, 1)
    check("a linewise paste lands on the next line's start, newline supplied", (text(h), cursor(h)), ("aXY\nL\ndef", (2, 0)))
    check("feed types plain characters", (lib.ed_feed(h, b"dw"), text(h))[1], "aXY\nL\ndwdef")
    # Every host call that edits or moves ends the selection and closes the
    # popup: enumerated per call, since one call's assertion says nothing
    # about the others.
    def arm_selection():
        lib.ed_set_text(h, b"alpha beta\ngamma")
        lib.ed_set_selection(h, 0, 0, 0, 5)
    def arm_popup():
        lib.ed_set_text(h, b"alpha beta\nal")
        lib.ed_set_cursor(h, 1, 2)
        lib.ed_complete_begin(h)
        lib.ed_complete_push(h, b"alpha")
        lib.ed_key(h, K_CHAR, M_CTRL, ord(" "))
    lib.ed_set_register(h, b"R", False)
    host_edits = [
        ("insert_at_cursor", lambda: lib.ed_insert_at_cursor(h, b"Q")),
        ("insert_at", lambda: lib.ed_insert_at(h, 0, 1, b"Q")),
        ("delete_range", lambda: lib.ed_delete_range(h, 0, 0, 0, 1)),
        ("undo", lambda: lib.ed_undo(h)),
        ("redo", lambda: lib.ed_redo(h)),
        ("paste", lambda: lib.ed_paste(h, False, 1)),
        ("set_cursor", lambda: lib.ed_set_cursor(h, 1, 0)),
        ("replace_selection", lambda: lib.ed_replace_selection(h, b"Z")),
        ("find", lambda: lib.ed_find(h, b"a", False, False)),
        ("replace_all", lambda: lib.ed_replace_all(h, b"a", b"b", False)),
        ("clear_selection", lambda: lib.ed_clear_selection(h)),
    ]
    for name, call in host_edits:
        arm_selection()
        lib.ed_key(h, K_CHAR, 0, ord("x"))  # an open group too
        lib.ed_set_selection(h, 0, 0, 0, 5)
        call()
        check(f"{name} ends the selection", selection(h), None)
        if name != "replace_selection":  # nothing selected: it does nothing
            arm_popup()
            call()
            check(f"{name} closes the popup", lib.ed_complete_open(h), False)
    # ...and closes the open undo group: two characters, the call, two more,
    # one undo leaves the first two.
    for name, call in host_edits:
        if name in ("undo", "redo", "delete_range", "replace_all"):
            continue
        lib.ed_set_text(h, b"")
        lib.ed_key(h, K_CHAR, 0, ord("a"))
        lib.ed_key(h, K_CHAR, 0, ord("b"))
        call()
        after_call = text(h)
        lib.ed_set_cursor(h, 0, len(after_call.split("\n")[0]))
        lib.ed_key(h, K_CHAR, 0, ord("c"))
        lib.ed_key(h, K_CHAR, 0, ord("d"))
        lib.ed_undo(h)
        check(f"{name} closes the open group", text(h), after_call)
    lib.ed_set_text(h, b"alpha\nal")
    lib.ed_set_cursor(h, 1, 2)

    # The popup: a host feeds it, the keys navigate and accept.
    lib.ed_set_text(h, b"alpha\nal")
    lib.ed_set_cursor(h, 1, 2)
    check("Ctrl+Space opens", lib.ed_key(h, K_CHAR, M_CTRL, ord(" ")), True)
    check("popup open", lib.ed_complete_open(h), True)
    check("pending with a popup is true", lib.ed_pending(h), True)
    check("Enter accepts", lib.ed_key(h, K_ENTER, 0, 0), True)
    check("the word", text(h), "alpha\nalpha")
    check("read-only refuses an edit", (lib.ed_set_read_only(h, True), lib.ed_key(h, K_CHAR, 0, ord("q")))[1], True)
    check("and changes nothing", text(h), "alpha\nalpha")
    check("but moves", (lib.ed_key(h, K_LEFT, 0, 0), cursor(h))[1], (1, 4))
    check("mode still insert read-only", lib.ed_mode(h), 1)
    lib.ed_set_read_only(h, False)
    lib.ed_set_style(h, STYLES["Vim"])
    lib.ed_set_chrome_style(h, STYLES["Vim"])
    lib.ed_set_text(h, b"vec3 p;")

    check("set a flag", lib.ed_set_chrome_flag(h, FLAGS["Relative_Numbers"], True), True)
    check("reads back", chrome_flag(h, FLAGS["Relative_Numbers"]), True)
    check("flag past the end refuses", lib.ed_set_chrome_flag(h, len(FLAGS), True), False)
    check("negative flag refuses", lib.ed_set_chrome_flag(h, -1, True), False)
    check("unknown flag reads nothing", chrome_flag(h, len(FLAGS)), None)

    # The gutter width a host must match when it draws its own, and which the
    # marker placement is stated against.
    lib.ed_set_chrome_style(h, STYLES["Vim"])
    lib.ed_set_text(h, b"a\nb\nc")
    check("a short buffer uses numberwidth", lib.ed_gutter_cells(h), 4)
    lib.ed_set_text(h, ("x\n" * 5000).encode())
    check("and widens for a long one", lib.ed_gutter_cells(h), 5)
    lib.ed_set_number_width(h, 8)
    check("numberwidth is a MINIMUM", lib.ed_gutter_cells(h), 8)
    lib.ed_set_chrome_flag(h, FLAGS["Line_Numbers"], False)
    check("no numbers, no gutter", lib.ed_gutter_cells(h), 0)
    lib.ed_set_chrome_style(h, STYLES["Vim"])

    print("navigation keys")
    # A platform sends these as CODES, not characters. Every one was declared,
    # documented and forwarded by the reference UI while nothing read them.
    # Expectations are nvim's, from line 2 column 5.
    lib.ed_set_text(h, b"line one\nline two\nline three")
    for code, want in ((K_LEFT, (1, 3)), (K_RIGHT, (1, 5)), (K_UP, (0, 4)),
                       (K_DOWN, (2, 4)), (K_HOME, (1, 0)), (K_END, (1, 7))):
        lib.ed_set_cursor(h, 1, 4)
        check(f"code {code} moves", (lib.ed_key(h, code, M_NONE, 0), cursor(h))[1], want)
    lib.ed_set_cursor(h, 1, 4)
    lib.ed_key(h, K_DEL, M_NONE, 0)
    check("delete removes under the cursor", text(h), "line one\nlinetwo\nline three")

    print("raw key input")
    # A GUI host holds a key CODE and a modifier set each frame, never vim
    # notation, so the codes cross directly rather than being serialised.
    lib.ed_set_text(h, b"alpha beta")
    lib.ed_set_cursor(h, 0, 0)
    check("a bound key is consumed", lib.ed_key(h, K_CHAR, M_NONE, ord("w")), True)
    check("and it moved", cursor(h), (0, 6))
    lib.ed_key(h, K_CHAR, M_NONE, ord("i"))
    check("insert mode", MODES[lib.ed_mode(h)], "INSERT")
    lib.ed_key(h, K_CHAR, M_NONE, ord("X"))
    lib.ed_key(h, K_ESC, M_NONE, 0)
    check("typed through the code path", text(h), "alpha Xbeta")
    check("named keys work", MODES[lib.ed_mode(h)], "NORMAL")
    # A Char with no resolved codepoint is what a host sends for a bare
    # modifier press; it must be a no-op, not an edit.
    before = text(h)
    lib.ed_key(h, K_CHAR, M_CTRL, 0)
    check("a modifier-only press changes nothing", text(h), before)
    check("an out-of-range code refuses", lib.ed_key(h, 999, M_NONE, 0), False)

    print("selection, undo and redo")
    lib.ed_set_text(h, b"one two three")
    lib.ed_set_cursor(h, 0, 0)
    check("nothing selected in normal mode", selection(h), None)
    check("and no text", selection_text(h), None)
    lib.ed_feed(h, b"vee")
    check("a visual selection reports", selection(h) is not None, True)
    check("the selected text", selection_text(h), "one two")
    lib.ed_clear_selection(h)
    check("clearing it", selection(h), None)

    lib.ed_set_text(h, b"line one\nline two\nline three")
    lib.ed_select_line(h, 1)
    check("select_line enters visual-line", MODES[lib.ed_mode(h)], "V-LINE")
    check("on the line asked for", cursor(h)[0], 1)
    lib.ed_clear_selection(h)

    lib.ed_set_text(h, b"abc")
    lib.ed_feed(h, b"x")
    check("edited", text(h), "bc")
    check("undo reports it acted", lib.ed_undo(h), True)
    check("and restored", text(h), "abc")
    check("redo", lib.ed_redo(h), True)
    check("re-applied", text(h), "bc")
    lib.ed_undo(h)
    check("undo past the start refuses", lib.ed_undo(h), False)

    print("insert at the caret")
    # The seam a host uses for a picked path, a completion, a copilot answer.
    lib.ed_set_text(h, b"#include XX\nvoid main(){}")
    lib.ed_set_cursor(h, 0, 9)
    lib.ed_insert_at_cursor(h, b'"noise.glsl"')
    check("landed at the caret", text(h).split("\n")[0], '#include "noise.glsl"XX')
    check("cursor follows the text", cursor(h), (0, 21))
    check("one undo step removes it", (lib.ed_undo(h), text(h).split("\n")[0])[1], "#include XX")
    lib.ed_insert_at_cursor(h, b"")
    check("an empty insert is a no-op", text(h).split("\n")[0], "#include XX")

    # A normal-mode cursor may not rest past the last character, so ed_set_cursor
    # clamps one short of end-of-line and cannot address an append. ed_insert_at
    # reaches the line's length, which is what appending needs.
    lib.ed_set_text(h, b"#include\nvoid main(){}")
    lib.ed_set_cursor(h, 0, 99)
    check("the cursor clamps inside the line", cursor(h), (0, 7))
    lib.ed_insert_at(h, 0, 8, b" <noise>")
    check("but insert_at appends at the end", text(h).split("\n")[0], "#include <noise>")
    lib.ed_insert_at(h, 0, 999, b"!")
    check("and a column past the end clamps there", text(h).split("\n")[0], "#include <noise>!")

    print("edges a host will hit")
    # Each of these contradicted a written promise in ffi/README.md, and none
    # was caught by the surface tests -- they only ever asked well-formed
    # questions.

    # A column past a line's end has no token. Walking on returned the NEXT
    # line's classes, so a minimap painted every line with its successors'.
    lib.ed_set_text(h, b"x = 1\ndef f(): pass")
    lib.ed_set_language(h, LANGS["Python"])
    lib.ed_layout(h, 0.0, 0.0, 800.0, 400.0, 16.0, True)
    check("a class on line 1", CLASSES[lib.ed_class_at(h, 1, 0)], "Keyword")
    check("but not past line 0's end", lib.ed_class_at(h, 0, 6), 0)
    check("nor far past it", lib.ed_class_at(h, 0, 99), 0)

    # ed_clear_selection is the host's Escape, so it must discard the pending
    # state too. Writing only the mode left a count alive: v3 + clear + w
    # jumped three words.
    lib.ed_set_text(h, b"alpha beta gamma delta")
    lib.ed_feed(h, b"v3")
    lib.ed_clear_selection(h)
    lib.ed_feed(h, b"w")
    after_clear = cursor(h)
    lib.ed_set_text(h, b"alpha beta gamma delta")
    lib.ed_feed(h, b"v3<Esc>w")
    check("clear_selection matches Esc", after_clear, cursor(h))

    # An inserted snippet is ONE undo step, including what the user types after
    # it: undo_record_insert re-arms the open insert run, so the snippet
    # swallowed the next keystrokes without a second break.
    lib.ed_set_text(h, b"abc")
    lib.ed_feed(h, b"A12")
    lib.ed_insert_at_cursor(h, b"SNIP")
    lib.ed_feed(h, b"34")
    check("typed after a snippet", text(h), "abc12SNIP34")
    lib.ed_undo(h)
    check("undo takes only what was typed after", text(h), "abc12SNIP")
    lib.ed_feed(h, b"<Esc>")

    # A host forwarding a raw platform value can hand over anything; encoding a
    # value outside Unicode wrote bytes no decode could survive.
    lib.ed_set_text(h, b"")
    lib.ed_feed(h, b"i")
    check("a codepoint outside Unicode refuses", lib.ed_key(h, K_CHAR, M_NONE, 0x110000), False)
    check("and nothing was written", text(h), "")
    lib.ed_feed(h, b"<Esc>")

    # Getters truncate on a CODEPOINT boundary: a clipboard copy with a fixed
    # buffer must not come back with a split character.
    lib.ed_set_text(h, "\u043f\u0440\u0438\u0432\u0435\u0442".encode())
    lib.ed_feed(h, b"vll")
    n = lib.ed_selection_text(h, _buf, 3)
    got = bytes(_buf[:n])
    check("a 3-byte cap on 2-byte codepoints", got, "\u043f".encode())
    check("and it decodes", got.decode(), "\u043f")
    lib.ed_clear_selection(h)

    print("tab width")
    # One width for the drawn tab AND the columns motions count, or the cursor
    # lands on a different character than the one drawn beneath it. Measured
    # against nvim at tabstop 2, 4 and 8: `0llj` puts virtcol at 4, 6, 10.
    check("the default", lib.ed_tab_width(h), 4)
    for width, want_virtcol in ((2, 4), (4, 6), (8, 10)):
        lib.ed_set_text(h, b"\tabc\n\txyz")
        lib.ed_set_tab_width(h, width)
        check(f"set to {width}", lib.ed_tab_width(h), width)
        lib.ed_feed(h, b"0ll")
        before = cursor(h)
        lib.ed_feed(h, b"j")
        after = cursor(h)
        # `j` keeps the DISPLAY column, and on "\tabc"/"\txyz" both lines have
        # the same shape, so the codepoint column must be unchanged too. At
        # tab_width 2 this landed on column 1 instead of 2 while the layout drew
        # the tab two wide -- the cursor under a different character than the
        # one it named.
        check(f"j keeps the column at ts={width}", (after[0], after[1]), (1, before[1]))
    lib.ed_set_tab_width(h, 0)
    check("zero clamps rather than dividing by zero", lib.ed_tab_width(h), 1)
    lib.ed_set_tab_width(h, 99)
    check("and a huge width clamps too", lib.ed_tab_width(h), 16)
    lib.ed_set_tab_width(h, 4)

    # A negative cap writes nothing and returns 0, not the cap -- a negative
    # return would collide with the -1 that means "there is nothing here".
    lib.ed_set_text(h, b"some text")
    check("a negative cap returns nothing written", lib.ed_text(h, _buf, -3), 0)
    check("and a zero cap too", lib.ed_text(h, _buf, 0), 0)

    # Host settings must survive a whole-buffer replace. tab_width lives on the
    # editor rather than the handle, so the rebuild silently reset it to 4 --
    # changing both the drawn width and where `j` lands, with nothing said.
    lib.ed_set_tab_width(h, 8)
    lib.ed_set_read_only(h, True)
    lib.ed_set_language(h, LANGS["GLSL"])
    lib.ed_set_color(h, SLOTS["Text"], 1.0, 0.0, 0.5, 1.0)
    lib.ed_set_text(h, b"a whole new buffer")
    check("tab width survives set_text", lib.ed_tab_width(h), 8)
    check("read-only survives it", (lib.ed_feed(h, b"dd"), text(h))[1], "a whole new buffer")
    check("language survives it", lib.ed_language(h), LANGS["GLSL"])
    check("the theme survives it", color(h, SLOTS["Text"]), (1.0, 0.0, 0.5, 1.0))
    lib.ed_set_read_only(h, False)
    lib.ed_set_tab_width(h, 4)
    lib.ed_reset_theme(h)

    # An insert must record the position it ACTUALLY used. core_insert snaps to
    # a codepoint boundary and clamps into the buffer; recording the caller's
    # raw value described a range the edit never occupied, so undo removed the
    # wrong bytes and left text that was never typed. Four calls reach it: a
    # visual selection, a host edit that empties the buffer, `O` -- which moves
    # a stale anchor into the cursor unclamped -- then an insert.
    for op in (b"O", b"o"):
        lib.ed_set_text(h, b"ab\n")
        lib.ed_feed(h, b"Gv")
        lib.ed_delete_range(h, 0, 0, 9, 9)
        lib.ed_feed(h, op)
        lib.ed_insert_at_cursor(h, b"XYZ")
        for _ in range(4):
            lib.ed_undo(h)
        check(f"undo after a stale-cursor insert ({op.decode()})", text(h), "ab\n")
    lib.ed_clear_selection(h)

    print("dragging a selection")
    # A mouse drag is not a keystroke. Building one from set_cursor plus a
    # synthesized `v` works only from normal mode: begun in insert mode it typed
    # a literal `v` into the buffer, which made the host responsible for the
    # editor's mode -- the coupling this boundary exists to remove.
    lib.ed_set_text(h, b"void main() {\n    vec3 col;\n}")
    lib.ed_feed(h, b"i")
    check("in insert mode", MODES[lib.ed_mode(h)], "INSERT")
    lib.ed_set_selection(h, 1, 4, 1, 8)
    check("a drag from insert mode types nothing", text(h), "void main() {\n    vec3 col;\n}")
    check("and selects", selection_text(h), "vec3 ")

    # Dragging is the same call with a new head, forwards and backwards.
    lib.ed_set_selection(h, 1, 4, 1, 12)
    check("the head moves", selection_text(h), "vec3 col;")
    lib.ed_set_selection(h, 1, 8, 1, 4)
    check("and drags backwards past the anchor", selection_text(h), "vec3 ")
    # The head's own character is INSIDE the selection, as vim's `v` has it:
    # measured with nvim, `ggv2j0y` on this text yanks all three lines.
    lib.ed_set_selection(h, 0, 0, 2, 0)
    check("across lines, head included", selection_text(h), "void main() {\n    vec3 col;\n}")
    lib.ed_clear_selection(h)

    # Whole lines, vim's V: an operator on a linewise selection takes the
    # newline with it, which a character range starting at a line start does not.
    lib.ed_set_text(h, b"one\ntwo\nthree\nfour")
    lib.ed_set_line_selection(h, 1, 2)
    check("linewise mode", MODES[lib.ed_mode(h)], "V-LINE")
    check("whole lines", selection_text(h), "two\nthree\n")
    lib.ed_feed(h, b"d")
    check("and deleting takes the newline", text(h), "one\nfour")

    # A selection and an open command line are different states: leaving `:`
    # active sent the host's next key to the command line instead.
    lib.ed_set_text(h, b"alpha beta")
    lib.ed_feed(h, b":")
    lib.ed_set_selection(h, 0, 0, 0, 4)
    check("setting a selection closes the command line", MODES[lib.ed_mode(h)], "VISUAL")
    check("and the selection is live", selection_text(h), "alpha")
    lib.ed_clear_selection(h)

    print("delete and clipboard")
    # Registers are out of scope for the KEYMAP, which would become an
    # accidental limit on the ABI if a host could not reach its own clipboard.
    lib.ed_set_text(h, b"alpha beta gamma")
    lib.ed_delete_range(h, 0, 6, 0, 11)
    check("a range is deleted", text(h), "alpha gamma")
    check("one undo restores it", (lib.ed_undo(h), text(h))[1], "alpha beta gamma")

    lib.ed_set_text(h, b"alpha beta gamma")
    lib.ed_feed(h, b"wve")
    check("copy reads the selection", selection_text(h), "beta")
    check("paste over it", lib.ed_replace_selection(h, b"DELTA"), True)
    check("replaced", text(h), "alpha DELTA gamma")
    check("and the pair is one undo step", (lib.ed_undo(h), text(h))[1], "alpha beta gamma")
    check("with nothing selected it refuses", lib.ed_replace_selection(h, b"x"), False)

    print("the register")
    # A FRESH handle, because the register deliberately survives ed_set_text --
    # a host loading a second file can paste what it yanked in the first, as
    # vim does across `:e`.
    h2 = lib.ed_new(b"alpha\nbeta\ngamma")
    check("empty before anything", lib.ed_register(h2, _buf, 65536), 0)
    check("and a paste refuses", lib.ed_paste(h2, False, 1), False)
    lib.ed_free(h2)

    lib.ed_set_text(h, b"alpha\nbeta\ngamma")
    lib.ed_feed(h, b"yy")
    check("yank stores the line", reg(h), "alpha\n")
    check("as linewise", lib.ed_register_linewise(h), True)
    lib.ed_feed(h, b"dd")
    check("delete overwrites it", reg(h), "alpha\n")
    lib.ed_feed(h, b"dd")
    check("and again -- one slot, last write wins", reg(h), "beta\n")

    lib.ed_set_text(h, b"alpha beta")
    lib.ed_feed(h, b"yw")
    check("a word yank is charwise", lib.ed_register_linewise(h), False)
    check("holding the word", reg(h), "alpha ")

    # p pastes it, as the keymap does.
    lib.ed_set_text(h, b"alpha\nbeta")
    lib.ed_feed(h, b"yy")
    check("ed_paste acts", lib.ed_paste(h, False, 1), True)
    check("below the line", text(h), "alpha\nalpha\nbeta")
    check("one undo takes it", (lib.ed_undo(h), text(h))[1], "alpha\nbeta")
    check("a count repeats", (lib.ed_paste(h, False, 3), text(h))[1],
          "alpha\nalpha\nalpha\nalpha\nbeta")
    lib.ed_undo(h)

    # A host wiring its own clipboard writes the slot, and p then behaves as vim.
    lib.ed_set_text(h, b"alpha\nbeta")
    lib.ed_set_register(h, b"HOST\n", True)
    check("a host-set register reads back", reg(h), "HOST\n")
    check("and pastes linewise", (lib.ed_paste(h, False, 1), text(h))[1],
          "alpha\nHOST\nbeta")
    lib.ed_undo(h)
    lib.ed_set_register(h, b"X", False)
    check("charwise from the host too", lib.ed_register_linewise(h), False)
    check("pasting after the cursor", (lib.ed_paste(h, False, 1), text(h))[1],
          "aXlpha\nbeta")
    lib.ed_undo(h)

    lib.ed_set_read_only(h, True)
    check("read-only refuses a paste", lib.ed_paste(h, False, 1), False)
    lib.ed_set_read_only(h, False)

    # The register and the search pattern SURVIVE a whole-buffer replace, which
    # is what a host loading a second file does. Both were silently lost when
    # ed_set_text carried its fields by hand.
    lib.ed_set_text(h, b"alpha\nbeta")
    lib.ed_feed(h, b"yy")
    lib.ed_find(h, b"beta", False, False)
    lib.ed_set_text(h, b"one\ntwo\nbeta")
    check("the register survives a buffer replace", reg(h), "alpha\n")
    check("and pastes into the new buffer",
          (lib.ed_paste(h, False, 1), text(h))[1], "one\nalpha\ntwo\nbeta")
    lib.ed_undo(h)
    check("the search pattern survives too", lib.ed_find_next(h, False), True)

    print("host commands")
    _force, _n = ctypes.c_bool(), ctypes.c_int32()

    def host_cmd(h):
        k = lib.ed_take_host_command(h, ctypes.byref(_force), _buf, 65536, ctypes.byref(_n))
        return k, _force.value, bytes(_buf[: _n.value]).decode()

    lib.ed_set_text(h, b"abc")
    check("nothing pending", host_cmd(h)[0], 0)
    # 1 write, 2 quit, 3 write+quit.
    for typed, want in ((b":w", 1), (b":q", 2), (b":wq", 3), (b":x", 3)):
        lib.ed_feed(h, typed)
        lib.ed_key(h, K_ENTER, 0, 0)
        check(f"{typed.decode()} reports its kind", host_cmd(h)[0], want)
    # `!` is passed along rather than rejected -- the editor has nothing to
    # force, but the host might.
    lib.ed_feed(h, b":q!")
    lib.ed_key(h, K_ENTER, 0, 0)
    check("a forced command reports force", host_cmd(h)[1:], (True, ""))
    # The spellings a host-side allowlist dropped.
    lib.ed_feed(h, b":w out.glsl")
    lib.ed_key(h, K_ENTER, 0, 0)
    check("an argument reaches the host", host_cmd(h), (1, False, "out.glsl"))
    # Reading consumes it, so a `:w` is served once and not every frame.
    lib.ed_feed(h, b":w")
    lib.ed_key(h, K_ENTER, 0, 0)
    check("reported once", host_cmd(h)[0], 1)
    check("and consumed by reading", host_cmd(h)[0], 0)
    # An unknown command stays an error rather than reaching the host.
    lib.ed_feed(h, b":nope")
    lib.ed_key(h, K_ENTER, 0, 0)
    check("an unknown command is not handed over", host_cmd(h)[0], 0)
    check("it reports a message instead", message(h), "not an editor command")
    # `:s` acts on the BUFFER, which the editor owns.
    lib.ed_set_text(h, b"alpha beta")
    lib.ed_feed(h, b":s/beta/X/")
    lib.ed_key(h, K_ENTER, 0, 0)
    check("a substitute is served here", host_cmd(h)[0], 0)
    check("and it acted", text(h), "alpha X")

    print("scroll motions")
    lib.ed_set_text(h, ("\n".join(f"line {i}" for i in range(1, 41))).encode())
    lib.ed_load_atlas(h, b"assets/atlas.json")
    # Read the cell height, then lay out a viewport exactly 20 rows tall, which
    # is the height the nvim oracle pins. "Half a page" means nothing unless
    # both sides agree on the page.
    cw, chh = ctypes.c_float(), ctypes.c_float()
    lib.ed_layout(h, 0.0, 0.0, 600.0, 100.0, 13.0, True)
    lib.ed_cell_size(h, ctypes.byref(cw), ctypes.byref(chh))
    lib.ed_layout(h, 0.0, 0.0, 600.0, 20.0 * chh.value, 13.0, True)

    for key, want in (("d", 11), ("u", 1), ("f", 19), ("b", 1)):
        lib.ed_set_cursor(h, 0, 0)
        took = lib.ed_key(h, K_CHAR, M_CTRL, ord(key))
        check(f"Ctrl-{key.upper()} is consumed", took, True)
        check(f"Ctrl-{key.upper()} lands where nvim does", cursor(h)[0] + 1, want)

    # Ctrl-E and Ctrl-Y move the VIEW: the cursor stays, and an offset is
    # reported for the host to apply.
    lib.ed_set_cursor(h, 0, 0)
    rows = ctypes.c_int32()
    check("nothing requested yet", lib.ed_take_scroll_request(h, ctypes.byref(rows)), False)
    lib.ed_key(h, K_CHAR, M_CTRL, ord("e"))
    check("Ctrl-E requests a scroll", lib.ed_take_scroll_request(h, ctypes.byref(rows)), True)
    check("of one row", rows.value, 1)
    check("and the cursor did not move", cursor(h)[0], 0)
    check("the request is consumed by reading it",
          lib.ed_take_scroll_request(h, ctypes.byref(rows)), False)

    # The two things a host has to get right, and which a doc sentence alone
    # did not convey -- one host applied the returned value itself and scrolled
    # twice. Reading APPLIES the scroll, and the value is ABSOLUTE.
    lib.ed_set_scroll(h, 5)
    lib.ed_layout(h, 0.0, 0.0, 600.0, 20.0 * chh.value, 13.0, True)
    check("the offset starts where the host put it", lib.ed_scroll(h), 5)
    lib.ed_key(h, K_CHAR, M_CTRL, ord("e"))
    check("the key alone does not scroll yet", lib.ed_scroll(h), 5)
    lib.ed_take_scroll_request(h, ctypes.byref(rows))
    check("reading APPLIES it", lib.ed_scroll(h), 6)
    check("and reports the ABSOLUTE offset, not a delta", rows.value, 6)
    check("so the reported value equals ed_scroll", rows.value, lib.ed_scroll(h))
    lib.ed_set_scroll(h, 0)

    # zz/zt/zb place the cursor's line without moving it.
    lib.ed_set_cursor(h, 29, 0)
    lib.ed_layout(h, 0.0, 0.0, 600.0, 20.0 * chh.value, 13.0, True)
    lib.ed_feed(h, b"zt")
    check("zt requests a scroll", lib.ed_take_scroll_request(h, ctypes.byref(rows)), True)
    check("and the cursor stayed", cursor(h)[0], 29)

    print("host-driven completion")
    lib.ed_set_text(h, b"uniform float u_time;\nvoid main(){ u_ti }")
    lib.ed_feed(h, b"G$hi")
    check("prefix", prefix(h), "u_ti")
    lib.ed_complete_begin(h)
    for cand in ("u_time", "u_resolution"):
        if cand.startswith(prefix(h)):
            lib.ed_complete_push(h, cand.encode())
    lib.ed_feed(h, b"<Tab>")
    check("accepted", text(h), "uniform float u_time;\nvoid main(){ u_time }")

    print("view flags")
    flag = ctypes.c_bool(True)
    check("default off", lib.ed_view_flag(h, VIEW["Show_Spaces"], ctypes.byref(flag)), True)
    check("and it is off", flag.value, False)
    check("set", lib.ed_set_view_flag(h, VIEW["Show_Spaces"], True), True)
    lib.ed_view_flag(h, VIEW["Show_Spaces"], ctypes.byref(flag))
    check("reads back", flag.value, True)
    check("flag past the end refuses", lib.ed_set_view_flag(h, len(VIEW), True), False)
    check("negative flag refuses", lib.ed_set_view_flag(h, -1, True), False)
    # The one-call convenience sets BOTH whitespace flags, which is what a
    # host's single checkbox means.
    lib.ed_set_show_whitespace(h, True)
    lib.ed_view_flag(h, VIEW["Show_Tabs"], ctypes.byref(flag))
    check("show_whitespace sets tabs too", flag.value, True)
    lib.ed_set_show_whitespace(h, False)
    lib.ed_view_flag(h, VIEW["Show_Spaces"], ctypes.byref(flag))
    check("and clears both", flag.value, False)

    print("line spacing")
    check("defaults to 1", lib.ed_line_spacing(h), 1.0)
    lib.ed_set_line_spacing(h, 1.5)
    check("reads back", lib.ed_line_spacing(h), 1.5)
    # A non-positive factor would collapse every cell to zero height and blank
    # the widget, so it is refused at the boundary rather than downstream.
    lib.ed_set_line_spacing(h, 0.0)
    check("zero reads as 1", lib.ed_line_spacing(h), 1.0)
    lib.ed_set_line_spacing(h, -2.0)
    check("negative reads as 1", lib.ed_line_spacing(h), 1.0)
    # A taller cell means fewer primitives fit the same viewport, which is the
    # observable proof the setting reached the layout rather than only the
    # handle.
    lib.ed_set_line_spacing(h, 1.0)
    lib.ed_set_text(h, b"\n".join([b"line of text"] * 40))
    tight = lib.ed_layout(h, 0.0, 0.0, 400.0, 200.0, 13.0, True)
    lib.ed_set_line_spacing(h, 3.0)
    loose = lib.ed_layout(h, 0.0, 0.0, 400.0, 200.0, 13.0, True)
    check("taller cells emit fewer glyphs", loose < tight, True)
    lib.ed_set_line_spacing(h, 1.0)

    print("find and replace")
    lib.ed_set_text(h, b"alpha beta gamma beta")
    check("counts every occurrence", lib.ed_find_count(h, b"beta", False), 2)
    check("finds", lib.ed_find(h, b"beta", False, False), True)
    check("cursor on the first match", cursor(h), (0, 6))
    check("next advances", lib.ed_find_next(h, False), True)
    check("cursor on the second", cursor(h), (0, 17))
    check("and wraps", lib.ed_find_next(h, False), True)
    check("back to the first", cursor(h), (0, 6))
    check("backward too", lib.ed_find_next(h, True), True)
    check("absent pattern is not found", lib.ed_find(h, b"zzz", False, False), False)
    check("case matters by default", lib.ed_find(h, b"BETA", False, False), False)
    check("unless ignored", lib.ed_find(h, b"BETA", False, True), True)

    lib.ed_set_text(h, b"alpha beta gamma beta")
    lib.ed_find(h, b"beta", False, False)
    check("replace at the cursor", lib.ed_replace_at_cursor(h, b"beta", b"X", False), True)
    check("replaced just the one", text(h), "alpha X gamma beta")
    check("undo restores it", (lib.ed_feed(h, b"u"), text(h))[1], "alpha beta gamma beta")

    lib.ed_set_text(h, b"x y x y x")
    check("replace all", lib.ed_replace_all(h, b"x", b"LONG", False), 3)
    check("a longer replacement does not corrupt", text(h), "LONG y LONG y LONG")
    lib.ed_set_read_only(h, True)
    check("read-only refuses", lib.ed_replace_all(h, b"LONG", b"z", False), 0)
    check("and the text is untouched", text(h), "LONG y LONG y LONG")
    lib.ed_set_read_only(h, False)

    print("bulk primitives")
    lib.ed_set_text(h, b"vec3 shade(vec3 base) {\n    return base * 2.0;\n}")
    n = lib.ed_layout(h, 0.0, 0.0, 400.0, 200.0, 13.0, True)
    check("the layout emitted something", n > 0, True)
    buf = (Prim * n)()
    got = lib.ed_primitives(h, buf, n)
    check("bulk returns the same count", got, n)
    # Every field must match the one-at-a-time getter, or a host reading the
    # block draws something different from a host reading the array.
    one = Prim()
    same = True
    for i in range(n):
        lib.ed_primitive(h, i, ctypes.byref(one))
        for f, _ in Prim._fields_:
            if getattr(buf[i], f) != getattr(one, f):
                same = False
                break
        if not same:
            break
    check("and every field agrees with ed_primitive", same, True)
    # A short buffer fills what it can rather than overrunning it.
    half = (Prim * n)()
    check("a small cap writes only that many", lib.ed_primitives(h, half, 3), 3)
    check("a zero cap writes nothing", lib.ed_primitives(h, half, 0), 0)
    check("a negative cap writes nothing", lib.ed_primitives(h, half, -5), 0)

    print("cell geometry")
    cw, ch = ctypes.c_float(), ctypes.c_float()
    lib.ed_cell_size(h, ctypes.byref(cw), ctypes.byref(ch))
    check("cell width is positive", cw.value > 0, True)
    check("cell height is positive", ch.value > 0, True)
    # It must be the size the LAST layout used, so line spacing moves it.
    lib.ed_set_line_spacing(h, 2.0)
    lib.ed_layout(h, 0.0, 0.0, 400.0, 200.0, 13.0, True)
    ch2 = ctypes.c_float()
    lib.ed_cell_size(h, None, ctypes.byref(ch2))
    check("and it tracks line spacing", ch2.value > ch.value, True)
    lib.ed_set_line_spacing(h, 1.0)
    ox, oy = ctypes.c_float(), ctypes.c_float()
    lib.ed_layout(h, 7.0, 11.0, 400.0, 200.0, 13.0, True)
    lib.ed_text_origin(h, ctypes.byref(ox), ctypes.byref(oy))
    check("text origin reports where the text went", (ox.value, oy.value), (7.0, 11.0))

    print("command line")
    lib.ed_set_text(h, b"alpha beta gamma")
    check("closed reports -1", lib.ed_command_line(h, _buf, 65536), -1)
    check("and no prompt", lib.ed_command_line_prompt(h), 0)
    lib.ed_feed(h, b":")
    check("open but empty reports 0", lib.ed_command_line(h, _buf, 65536), 0)
    check("with a colon prompt", chr(lib.ed_command_line_prompt(h)), ":")
    lib.ed_feed(h, b"wq")
    check("the typed text reads back", cmdline(h), "wq")
    lib.ed_key(h, 2, 0, 0)  # Esc
    check("escape closes it", lib.ed_command_line(h, _buf, 65536), -1)
    lib.ed_feed(h, b"/")
    check("a search prompt is distinct", chr(lib.ed_command_line_prompt(h)), "/")
    lib.ed_feed(h, b"bet")
    check("and carries its own text", cmdline(h), "bet")
    lib.ed_key(h, 2, 0, 0)
    lib.ed_feed(h, b"?x")
    check("as does a backward search", chr(lib.ed_command_line_prompt(h)), "?")
    lib.ed_key(h, 2, 0, 0)
    # A failed search leaves a message on the same row.
    lib.ed_feed(h, b"/zzz<CR>")
    check("a failed search leaves a message", message(h), "pattern not found")

    print("pending")
    lib.ed_set_text(h, b"alpha beta gamma")
    check("idle is not pending", lib.ed_pending(h), False)
    lib.ed_feed(h, b"2")
    check("a count is pending", lib.ed_pending(h), True)
    lib.ed_key(h, 2, 0, 0)
    check("escape clears it", lib.ed_pending(h), False)
    lib.ed_feed(h, b"d")
    check("an operator is pending", lib.ed_pending(h), True)
    lib.ed_key(h, 2, 0, 0)
    lib.ed_feed(h, b"f")
    check("a half-typed find is pending", lib.ed_pending(h), True)
    lib.ed_key(h, 2, 0, 0)
    lib.ed_feed(h, b":")
    check("an open command line is pending", lib.ed_pending(h), True)
    lib.ed_key(h, 2, 0, 0)
    lib.ed_feed(h, b"v")
    check("visual mode is NOT pending", lib.ed_pending(h), False)
    lib.ed_key(h, 2, 0, 0)

    print("completion popup state")
    lib.ed_set_text(h, b"SB_n")
    lib.ed_feed(h, b"A")
    check("closed before any push", lib.ed_complete_open(h), False)
    check("and no items", lib.ed_complete_count(h), 0)
    check("selected is -1 while closed", lib.ed_complete_selected(h), -1)
    check("an out-of-range item refuses", lib.ed_complete_item(h, 0, _buf, 65536), -1)
    # The prefix is a property of the BUFFER, not of the popup: a host needs it
    # while nothing is open, in order to decide whether to offer anything.
    check("the prefix reads while closed", cstr(lib.ed_complete_prefix(h, _buf, 65536)), "SB_n")

    lib.ed_complete_begin(h)
    for cand in (b"SB_noise", b"SB_normal"):
        lib.ed_complete_push(h, cand)
    check("pushing opens it", lib.ed_complete_open(h), True)
    check("with both candidates", lib.ed_complete_count(h), 2)
    check("the first is selected", lib.ed_complete_selected(h), 0)
    check("and readable", cstr(lib.ed_complete_item(h, 0, _buf, 65536)), "SB_noise")
    check("as is the second", cstr(lib.ed_complete_item(h, 1, _buf, 65536)), "SB_normal")
    check("past the end refuses", lib.ed_complete_item(h, 2, _buf, 65536), -1)

    # Cancel closes it WITHOUT leaving insert mode, which ed_clear_selection
    # (an Escape) cannot do.
    lib.ed_complete_cancel(h)
    check("cancel closes the popup", lib.ed_complete_open(h), False)
    check("and stays in insert mode", MODES[lib.ed_mode(h)], "INSERT")
    check("and leaves the buffer alone", text(h), "SB_n")

    # The measured host failure: Enter accepted a candidate instead of opening a
    # line, because pushing had armed it and nothing said so.
    lib.ed_complete_begin(h)
    lib.ed_complete_push(h, b"SB_noise")
    check("Enter accepts while open", (lib.ed_key(h, K_ENTER, 0, 0), text(h))[1], "SB_noise")
    lib.ed_set_text(h, b"SB_n")
    lib.ed_feed(h, b"A")
    check("but inserts a newline while closed", lib.ed_complete_open(h), False)
    lib.ed_key(h, K_ENTER, 0, 0)
    check("a real newline", text(h), "SB_n\n")

    print("host-driven completion")
    lib.ed_set_text(h, b"SB_noise SB_normal\nSB_n")
    lib.ed_feed(h, b"GA")
    check("off by default", lib.ed_host_completion(h), False)
    # Ctrl-N is bound inside the keymap: it opens the popup from buffer words
    # before a host has seen the key, which is one frame of the wrong list for
    # a host feeding its own vocabulary.
    lib.ed_key(h, K_CHAR, M_CTRL, ord("n"))
    check("the built-in source opens a popup", lib.ed_complete_open(h), True)
    check("with buffer words in it", lib.ed_complete_count(h) > 0, True)

    lib.ed_set_text(h, b"SB_noise SB_normal\nSB_n")
    lib.ed_set_host_completion(h, True)
    check("reads back", lib.ed_host_completion(h), True)
    lib.ed_feed(h, b"GA")
    check("Ctrl-N is still consumed", lib.ed_key(h, K_CHAR, M_CTRL, ord("n")), True)
    check("but opens nothing", lib.ed_complete_open(h), False)
    check("and holds no items", lib.ed_complete_count(h), 0)
    # The host's own candidates still work.
    lib.ed_complete_begin(h)
    lib.ed_complete_push(h, b"SB_uniform")
    check("a pushed candidate still opens it", lib.ed_complete_open(h), True)
    check("with only the host's list", lib.ed_complete_count(h), 1)
    lib.ed_complete_cancel(h)
    lib.ed_set_host_completion(h, False)
    lib.ed_key(h, K_ESC, 0, 0)

    print("the popup reaches the primitive array")
    lib.ed_load_atlas(h, b"assets/atlas.json")
    lib.ed_set_text(h, b"SB_n")
    lib.ed_feed(h, b"A")
    lib.ed_complete_begin(h)
    for cand in (b"SB_noise", b"SB_normal"):
        lib.ed_complete_push(h, cand)
    n = lib.ed_layout(h, 0.0, 0.0, 400.0, 200.0, 13.0, True)
    kinds = {}
    pr = Prim()
    for i in range(n):
        lib.ed_primitive(h, i, ctypes.byref(pr))
        kinds[pr.kind] = kinds.get(pr.kind, 0) + 1
    # KINDS indexes the enum: 5 Popup_Panel, 6 Popup_Glyph.
    check("a panel is emitted", kinds.get(5, 0), 1)
    check("and its candidate glyphs", kinds.get(6, 0) > 0, True)
    lib.ed_complete_cancel(h)
    n2 = lib.ed_layout(h, 0.0, 0.0, 400.0, 200.0, 13.0, True)
    kinds2 = {}
    for i in range(n2):
        lib.ed_primitive(h, i, ctypes.byref(pr))
        kinds2[pr.kind] = kinds2.get(pr.kind, 0) + 1
    check("and nothing once closed", kinds2.get(5, 0) + kinds2.get(6, 0), 0)
    lib.ed_key(h, K_ESC, 0, 0)

    lib.ed_free(h)
    print("\nFAILED" if failures else "\nall boundary checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
