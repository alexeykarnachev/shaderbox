"""Generate the data-driven text/glyphs.glsl for the shader library.

Glyph geometry lives here as python tables (strokes on the segment lattice); the
emitted GLSL is engine-bound uniform-array declarations + one small evaluator loop,
with the values in the generated shaderbox/glyph_tables.py. This replaces the old
per-glyph-function + dispatch-switch shape, whose inlined code volume made driver
codegen explode (V3D spent ~20s of CPU per shader hunting a register allocation);
uniform arrays (not const — NVIDIA demotes dynamically-indexed const arrays to
local memory) keep both vendors fast. Same SDF semantics, pixel-identical strokes.

Usage: uv run python scripts/gen_glyphs.py
Outputs (repo-anchored, generated together): shaderbox/resources/shader_lib/text/glyphs.glsl
+ shaderbox/glyph_tables.py (the engine-bound table values).
"""

from pathlib import Path

Point = tuple[float, float]
Stroke = tuple  # ("seg", a, b) | ("arc"|"tall", c, sx, sy) | ("dot", c, r)

# Lattice anchors: cell x in [-0.5, 0.5], y in [-1, 1].
_A: Point = (-0.5, -1.0)
_B: Point = (-0.5, -0.5)
_C: Point = (-0.5, 0.0)
_D: Point = (-0.5, 0.5)
_E: Point = (-0.5, 1.0)
_F: Point = (0.0, 1.0)
_G: Point = (0.5, 1.0)
_H: Point = (0.5, 0.5)
_J: Point = (0.5, 0.0)
_K: Point = (0.5, -0.5)
_L: Point = (0.5, -1.0)
_M: Point = (0.0, -1.0)
_N: Point = (0.0, -0.5)
_O: Point = (0.0, 0.0)
_P: Point = (0.0, 0.5)

# Straight segments by the historical seg ids (kept for readable glyph defs).
SEGS: dict[int, tuple[Point, Point]] = {
    0: (_A, _B),
    1: (_B, _C),
    2: (_C, _D),
    3: (_D, _E),
    4: (_E, _F),
    5: (_F, _G),
    6: (_G, _H),
    7: (_H, _J),
    8: (_J, _K),
    9: (_K, _L),
    10: (_M, _L),
    11: (_A, _M),
    12: (_M, _N),
    14: (_N, _O),
    16: (_O, _P),
    18: (_P, _F),
    20: (_A, _O),
    21: (_E, _O),
    22: (_O, _G),
    23: (_O, _L),
    24: (_C, _O),
    25: (_O, _J),
}

# Quarter-circle arcs (radius 0.5): center + quadrant signs; endpoints derive as
# c + (sx*0.5, 0) and c + (0, sy*0.5).
ARCS: dict[int, tuple[Point, int, int]] = {
    26: ((0.0, -0.5), -1, -1),
    27: ((0.0, -0.5), -1, +1),
    28: ((0.0, -0.5), +1, +1),
    29: ((0.0, -0.5), +1, -1),
    30: ((0.0, 0.5), -1, -1),
    31: ((0.0, 0.5), -1, +1),
    32: ((0.0, 0.5), +1, +1),
    33: ((0.0, 0.5), +1, -1),
}


def s(*ids: int) -> list[Stroke]:
    out: list[Stroke] = []
    for i in ids:
        if i in SEGS:
            a, b = SEGS[i]
            out.append(("seg", a, b))
        else:
            c, sx, sy = ARCS[i]
            out.append(("arc", c, sx, sy))
    return out


def seg(a: Point, b: Point) -> list[Stroke]:
    return [("seg", a, b)]


def tall(c: Point, sx: int, sy: int) -> list[Stroke]:
    return [("tall", c, sx, sy)]


def dot(c: Point, r: float = 0.12) -> list[Stroke]:
    return [("dot", c, r)]


COMMA: list[Stroke] = dot((0.0, -0.5)) + seg((0.0, -0.5), (-0.125, -1.0))
SOFT_SIGN: list[Stroke] = s(0, 1, 2, 3, 24, 28, 29, 11)
SHA: list[Stroke] = s(0, 1, 2, 3, 12, 14, 16, 18, 6, 7, 8, 9, 11, 10)
EL: list[Stroke] = s(0, 1, 2, 31, 5, 6, 7, 8, 9)
TAIL: list[Stroke] = seg((0.5, -1.0), (0.62, -1.22))

GLYPHS: dict[str, list[Stroke]] = {
    "lat_A": s(0, 1, 2, 31, 32, 7, 8, 9, 24, 25),
    "lat_B": s(0, 1, 2, 3, 4, 32, 33, 24, 28, 29, 11),
    "lat_C": s(29, 26, 1, 2, 31, 32),
    "lat_D": s(0, 1, 2, 3, 4, 32, 7, 8, 29, 11),
    "lat_E": s(0, 1, 2, 3, 4, 5, 24, 25, 11, 10),
    "lat_F": s(0, 1, 2, 3, 4, 5, 24),
    "lat_G": s(26, 1, 2, 31, 5, 29, 8, 25),
    "lat_H": s(0, 1, 2, 3, 24, 25, 9, 8, 7, 6),
    "lat_I": s(12, 14, 16, 18, 4, 5, 11, 10),
    "lat_J": s(8, 7, 6, 5, 29, 26),
    "lat_K": s(0, 1, 2, 3, 24, 22, 23),
    "lat_L": s(0, 1, 2, 3, 11, 10),
    "lat_M": s(0, 1, 2, 3, 21, 22, 6, 7, 8, 9),
    "lat_N": s(0, 1, 2, 3, 21, 23, 6, 7, 8, 9),
    "lat_O": s(26, 1, 2, 31, 32, 7, 8, 29),
    "lat_P": s(0, 1, 2, 3, 4, 32, 33, 24),
    "lat_Q": s(26, 1, 2, 31, 32, 7, 8, 29, 23),
    "lat_R": s(0, 1, 2, 3, 4, 32, 33, 23, 24),
    "lat_S": s(32, 31, 30, 28, 29, 26),
    "lat_T": s(12, 14, 16, 18, 4, 5),
    "lat_U": s(1, 2, 3, 26, 29, 8, 7, 6),
    "lat_V": s(0, 1, 2, 3, 20, 22),
    "lat_W": s(0, 1, 2, 3, 20, 23, 9, 8, 7, 6),
    "lat_X": s(20, 22, 21, 23),
    "lat_Y": s(12, 14, 21, 22),
    "lat_Z": s(4, 5, 22, 20, 11, 10),
    "dig_0": s(26, 1, 2, 31, 32, 7, 8, 29, 20, 22),
    "dig_1": s(12, 14, 16, 18, 4, 11, 10),
    "dig_2": s(31, 32, 7, 25, 27, 0, 11, 10),
    "dig_3": s(31, 32, 33, 25, 8, 9, 10, 11),
    "dig_4": s(2, 3, 24, 25, 6, 7, 8, 9),
    "dig_5": s(4, 5, 2, 3, 24, 28, 29, 11),
    "dig_6": s(31, 2, 1, 26, 27, 28, 29),
    "dig_7": s(4, 5, 22, 14, 12),
    "dig_8": s(30, 31, 32, 33, 26, 27, 28, 29),
    "dig_9": s(30, 31, 32, 33, 7, 8, 29, 26),
    "exclaim": s(16, 18) + dot((0.0, -0.88)),
    "ampersand": s(30, 31, 32, 33, 2, 20, 25),
    "apostrophe": seg((0.0, 1.0), (-0.15, 0.6)),
    "comma": COMMA,
    "dash": s(24, 25),
    "period": dot((0.0, -0.88)),
    "colon": dot((0.0, -0.88)) + dot((0.0, 0.0)),
    "semicolon": dot((0.0, 0.0)) + COMMA,
    "question": s(31, 32, 33, 14) + dot((0.0, -0.88)),
    "cyr_BE": s(0, 1, 2, 3, 4, 5, 24, 28, 29, 11),
    "cyr_GHE": s(0, 1, 2, 3, 4, 5),
    "cyr_DE": EL
    + s(11, 10)
    + seg((-0.5, -1.0), (-0.5, -1.22))
    + seg((0.5, -1.0), (0.5, -1.22)),
    "cyr_YO": s(0, 1, 2, 3, 4, 5, 24, 25, 11, 10)
    + dot((-0.22, 1.32), 0.07)
    + dot((0.22, 1.32), 0.07),
    "cyr_ZHE": s(12, 14, 16, 18)
    + tall((0.0, 1.0), +1, -1)
    + tall((0.0, 1.0), -1, -1)
    + tall((0.0, -1.0), +1, +1)
    + tall((0.0, -1.0), -1, +1),
    "cyr_ZE": s(31, 32, 33, 28, 29, 26),
    "cyr_I": s(0, 1, 2, 3, 6, 7, 8, 9, 20, 22),
    "cyr_SHORT_I": s(0, 1, 2, 3, 6, 7, 8, 9, 20, 22)
    + seg((-0.26, 1.38), (0.0, 1.1))
    + seg((0.0, 1.1), (0.26, 1.38)),
    "cyr_EL": EL,
    "cyr_PE": s(0, 1, 2, 31, 32, 7, 8, 9),
    "cyr_U": s(3, 30, 25, 6, 7, 8, 29, 26),
    "cyr_EF": s(12, 14, 16, 18, 30, 31, 32, 33),
    "cyr_TSE": s(1, 2, 3, 26, 10, 6, 7, 8, 9) + TAIL,
    "cyr_CHE": s(3, 30, 25, 6, 7, 8, 9),
    "cyr_SHA": SHA,
    "cyr_SHCHA": SHA + TAIL,
    "cyr_HARD": SOFT_SIGN + seg((-0.5, 1.0), (-0.85, 1.0)),
    "cyr_YERU": SOFT_SIGN + s(6, 7, 8, 9),
    "cyr_SOFT": SOFT_SIGN,
    "cyr_E": s(4, 32, 7, 8, 29, 11, 25),
    "cyr_YU": s(0, 1, 2, 3, 24, 12, 14, 16, 18, 32, 7, 8, 29),
    "cyr_YA": s(6, 7, 8, 9, 5, 31, 30, 25, 20),
}

# Glyph-id order: ids 0..25 = A-Z (cp 65..90), 26..35 = digits (48..57), 36..44 =
# punctuation, 45 = YO (1025), 46..77 = Cyrillic А-Я (1040..1071). Must match
# sbt_glyph_id in the emitted GLSL.
ORDER: list[str] = (
    [f"lat_{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    + [f"dig_{i}" for i in range(10)]
    + [
        "exclaim",
        "ampersand",
        "apostrophe",
        "comma",
        "dash",
        "period",
        "colon",
        "semicolon",
        "question",
        "cyr_YO",
    ]
    + [
        "lat_A",
        "cyr_BE",
        "lat_B",
        "cyr_GHE",
        "cyr_DE",
        "lat_E",
        "cyr_ZHE",
        "cyr_ZE",
        "cyr_I",
        "cyr_SHORT_I",
        "lat_K",
        "cyr_EL",
        "lat_M",
        "lat_H",
        "lat_O",
        "cyr_PE",
        "lat_P",
        "lat_C",
        "lat_T",
        "cyr_U",
        "cyr_EF",
        "lat_X",
        "cyr_TSE",
        "cyr_CHE",
        "cyr_SHA",
        "cyr_SHCHA",
        "cyr_HARD",
        "cyr_YERU",
        "cyr_SOFT",
        "cyr_E",
        "cyr_YU",
        "cyr_YA",
    ]
)

# Strokes are stored sorted by kind within each glyph (segments, arcs, tall arcs,
# dots) so the evaluator needs no per-stroke kind fetch — the span carries counts.
KIND_ORDER: dict[str, int] = {"seg": 0, "arc": 1, "tall": 2, "dot": 3}


def _stroke_payload(st: Stroke) -> tuple[float, float, float, float]:
    kind = st[0]
    if kind == "seg":
        (ax, ay), (bx, by) = st[1], st[2]
        return (ax, ay, bx, by)
    if kind in ("arc", "tall"):
        (cx, cy), sx, sy = st[1], st[2], st[3]
        return (cx, cy, float(sx), float(sy))
    (cx, cy), r = st[1], st[2]
    return (cx, cy, r, 0.0)


def build_tables() -> (
    tuple[list[tuple[int, int, int, int]], list[tuple[float, float, float, float]]]
):
    # (spans, strokes): one ivec4-shaped span per ORDER entry (offset, segs, arcs,
    # (tall<<8)|dots), one vec4-shaped stroke row per stroke of each UNIQUE glyph.
    spans: dict[str, tuple[int, int, int, int, int]] = {}
    data: list[tuple[float, float, float, float]] = []
    for name in dict.fromkeys(ORDER):  # unique names, first-seen order
        strokes = sorted(GLYPHS[name], key=lambda st: KIND_ORDER[st[0]])
        counts = [
            sum(1 for st in strokes if st[0] == k)
            for k in ("seg", "arc", "tall", "dot")
        ]
        spans[name] = (len(data), counts[0], counts[1], counts[2], counts[3])
        for st in strokes:
            data.append(_stroke_payload(st))
    span_rows = [
        (
            spans[name][0],
            spans[name][1],
            spans[name][2],
            (spans[name][3] << 8) | spans[name][4],
        )
        for name in ORDER
    ]
    return span_rows, data


def generate() -> str:
    span_rows_v, data_v = build_tables()
    n = len(data_v)

    return f"""// Segment-glyph face, DATA-DRIVEN: glyph geometry lives in engine-bound uniform
// tables and one small evaluator loop walks the strokes — same lattice, same SDF
// semantics as the old per-glyph functions, but orders of magnitude less CODE (the
// inlined glyph switch made driver shader-codegen explode on register-starved GPUs).
// GENERATED by scripts/gen_glyphs.py — edit the python stroke tables, regenerate;
// do not hand-edit. Glyph-local cell: x in [-0.5,0.5], y in [-1,1].
// Public surface: SB_sd_char. sbt_* names are library-private (not catalogued;
// layout.glsl composes sbt_char_skel).

float sbt_arc(vec2 p, vec2 c, vec2 q) {{
    vec2 v = (p - c) / 0.5;
    float dist = abs(length(v) - 1.0) * 0.5;
    float in_q = step(0.0, v.x * q.x) * step(0.0, v.y * q.y);
    vec2 e1 = c + vec2(q.x * 0.5, 0.0);
    vec2 e2 = c + vec2(0.0, q.y * 0.5);
    return mix(min(distance(p, e1), distance(p, e2)), dist, in_q);
}}

float sbt_arc_tall(vec2 p, vec2 c, vec2 q) {{
    vec2 v = (p - c) / vec2(0.5, 1.0);
    float len = max(length(v), 0.0001);
    // First-order true distance |phi|/|grad phi| (the naive estimate rendered
    // arc-armed glyphs visibly bolder near the vertical endpoints).
    float dist = abs(len - 1.0) * len / max(length(vec2(v.x / 0.5, v.y)), 0.0001);
    float in_q = step(0.0, v.x * q.x) * step(0.0, v.y * q.y);
    vec2 e1 = c + vec2(q.x * 0.5, 0.0);
    vec2 e2 = c + vec2(0.0, q.y * 1.0);
    return mix(min(distance(p, e1), distance(p, e2)), dist, in_q);
}}

int sbt_glyph_id(uint cp) {{
    if (cp >= 97u && cp <= 122u) cp -= 32u;
    if (cp >= 1072u && cp <= 1103u) cp -= 32u;
    if (cp == 1105u) cp = 1025u;
    if (cp >= 65u && cp <= 90u) return int(cp) - 65;
    if (cp >= 48u && cp <= 57u) return 26 + int(cp) - 48;
    if (cp >= 1040u && cp <= 1071u) return 46 + int(cp) - 1040;
    if (cp == 1025u) return 45;
    if (cp == 33u) return 36;
    if (cp == 38u) return 37;
    if (cp == 39u) return 38;
    if (cp == 44u) return 39;
    if (cp == 45u) return 40;
    if (cp == 46u) return 41;
    if (cp == 58u) return 42;
    if (cp == 59u) return 43;
    if (cp == 63u) return 44;
    return -1;
}}

// Tables are ENGINE-BOUND uniform arrays on purpose: a dynamically-indexed
// const array (function-local OR global) is demoted to per-thread local memory
// on NVIDIA (~100x slower text stack); a uniform array lives in the constant
// bank on every driver tested, and keeps V3D codegen small (the original
// inlined glyph switch exploded its compile). Node.compile() writes the values
// from shaderbox/glyph_tables.py — generated together with this file, never
// node-settable (ENGINE_DRIVEN_UNIFORMS). Span per glyph: x = stroke offset,
// y = segment count, z = arc count, w = (tall_arc_count << 8) | dot_count.
// Strokes are pre-sorted by kind.
uniform ivec4 SBT_SPANS[{len(ORDER)}];
uniform vec4 SBT_STROKES[{n}];

ivec4 sbt_glyph_span(int g) {{
    return SBT_SPANS[g];
}}

vec4 sbt_stroke_data(int i) {{
    return SBT_STROKES[i];
}}

float sbt_char_skel(vec2 p, uint cp) {{
    int g = sbt_glyph_id(cp);
    if (g < 0) return 100000.0;
    ivec4 span = sbt_glyph_span(g);
    float d = 100000.0;
    int i = span.x;
    for (int k = 0; k < span.y; ++k) {{
        vec4 st = sbt_stroke_data(i++);
        d = min(d, SB_sd_segment(p, st.xy, st.zw));
    }}
    for (int k = 0; k < span.z; ++k) {{
        vec4 st = sbt_stroke_data(i++);
        d = min(d, sbt_arc(p, st.xy, st.zw));
    }}
    for (int k = 0; k < (span.w >> 8); ++k) {{
        vec4 st = sbt_stroke_data(i++);
        d = min(d, sbt_arc_tall(p, st.xy, st.zw));
    }}
    for (int k = 0; k < (span.w & 0xFF); ++k) {{
        vec4 st = sbt_stroke_data(i++);
        d = min(d, distance(p, st.xy) - st.z);
    }}
    return d;
}}

/// SIGNED distance (negative inside the ink) to one glyph in glyph-local coords
/// (cell x in [-0.5,0.5], y in [-1,1]). `weight` = stroke half-width in the same
/// LOCAL units (0.1 regular .. 0.25 bold). To place a glyph of height ch at uv
/// position c: SB_sd_char((uv - c) / (0.5*ch), cp, 0.1) * (0.5*ch) — the *0.5*ch
/// converts the result back to uv units. Lowercase folds to uppercase; supports
/// A-Z, Cyrillic А-Я/Ё, 0-9 and ! ? : ; , . - ' &. Unknown codepoints draw nothing.
float SB_sd_char(vec2 p, uint codepoint, float weight) {{
    return sbt_char_skel(p, codepoint) - weight;
}}
"""


def generate_tables_py() -> str:
    span_rows, data = build_tables()
    span_lines = "\n".join(f"    {row}," for row in span_rows)
    stroke_lines = "\n".join(f"    {row}," for row in data)
    return f'''"""Glyph stroke tables, engine-bound as uniform arrays.

`text/glyphs.glsl` declares `uniform ivec4 SBT_SPANS[..]` / `uniform vec4
SBT_STROKES[..]`; `Node.compile()` writes these values into any program that
uses them (and `ENGINE_DRIVEN_UNIFORMS` keeps them off every user surface).
GENERATED by scripts/gen_glyphs.py together with glyphs.glsl — do not hand-edit;
the two artifacts must come from the same generator run.
"""

import struct

_SPANS: list[tuple[int, int, int, int]] = [
{span_lines}
]

_STROKES: list[tuple[float, float, float, float]] = [
{stroke_lines}
]

TABLE_UNIFORMS: dict[str, bytes] = {{
    "SBT_SPANS": b"".join(struct.pack("<4i", *v) for v in _SPANS),
    "SBT_STROKES": b"".join(struct.pack("<4f", *v) for v in _STROKES),
}}
'''


def main() -> None:
    # Both artifacts come from one generator run and must stay in sync — outputs
    # are repo-anchored so the script works from any cwd.
    repo = Path(__file__).parent.parent
    out = repo / "shaderbox/resources/shader_lib/text/glyphs.glsl"
    tables_out = repo / "shaderbox/glyph_tables.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(generate(), encoding="utf-8")
    tables_out.write_text(generate_tables_py(), encoding="utf-8")
    n_strokes = sum(len(GLYPHS[n]) for n in dict.fromkeys(ORDER))
    print(
        f"wrote {out} + {tables_out} "
        f"({len(dict.fromkeys(ORDER))} unique glyphs, {n_strokes} strokes)"
    )


if __name__ == "__main__":
    main()
