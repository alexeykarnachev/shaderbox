"""Engine-computed facts appended to edit-tool results (feature 033).

Pure text/pixel analysis — no GL, no imgui. The compiler says WHAT broke; the
hints say WHY in terms the model can act on (the dominant dogfood failure class:
a line edit whose new_text duplicates content that survived outside the range).
Render facts are the agent's only sight: scalar truths about the latest clean
frame, computed from raw RGBA bytes the backend reads off a tiny probe canvas.
"""

import re

from shaderbox.shader_lib import parser

_REDECLARED_RE = re.compile(r"`(\w+)' redeclared")
_INITIALIZER_RE = re.compile(
    r"initializer of type (\w+)\[(\d+)\] cannot be assigned to .*?\[(\d+)\]"
)
# A declaration-ish line: optional qualifiers, a type (with optional array suffix),
# then the identifier followed by `=`, `;`, `[` or `(` (function defs redeclare too).
_DECL_HEAD = r"^\s*(?:const\s+)?(?:uniform\s+)?[A-Za-z_]\w*(?:\s*\[\s*\d*\s*\])?\s+"


def compile_hints(source: str, error_messages: list[str]) -> list[str]:
    """Structural hints for compile errors, derived from the post-edit source."""
    hints: list[str] = []
    stripped = parser.strip_comments(source)
    lines = stripped.splitlines()
    seen: set[str] = set()
    for msg in error_messages:
        m = _REDECLARED_RE.search(msg)
        if m and m.group(1) not in seen:
            name = m.group(1)
            seen.add(name)
            decl = re.compile(_DECL_HEAD + re.escape(name) + r"\s*[=;[(]")
            decl_lines = [i for i, ln in enumerate(lines, start=1) if decl.match(ln)]
            if len(decl_lines) >= 2:
                rows = ", ".join(str(i) for i in decl_lines)
                hints.append(
                    f"hint: '{name}' is declared on lines {rows} — the edit range "
                    "missed one copy; widen the range over it or delete the duplicate"
                )
        m = _INITIALIZER_RE.search(msg)
        if m:
            kind, got, want = m.group(1), m.group(2), m.group(3)
            hints.append(
                f"hint: the initializer has {got} elements, the array wants {want} — "
                f"or declare it unsized (`{kind}[] x = {kind}[](...)`), no counting"
            )
    opens, closes = stripped.count("{"), stripped.count("}")
    if opens != closes:
        hints.append(
            f"hint: the file has {opens} '{{' vs {closes} '}}' — an orphan tail "
            "survived below the edit range, or a brace went missing in new_text"
        )
    return hints


def render_facts(rgba: bytes, width: int, height: int) -> str:
    """One terse line of truth about a rendered frame.

    Background = the most common corner color; ink = pixels that differ from it.
    y follows vs_uv (0 = bottom) — GL framebuffer reads are already bottom-up.
    """
    n = width * height
    if len(rgba) < n * 4:
        return ""

    def at(i: int) -> tuple[int, int, int]:
        o = i * 4
        return (rgba[o], rgba[o + 1], rgba[o + 2])

    corners = [at(0), at(width - 1), at((height - 1) * width), at(n - 1)]
    bg = max(set(corners), key=corners.count)

    ink = 0
    min_x, min_y, max_x, max_y = width, height, -1, -1
    luma_sum = [0.0] * 9
    luma_cnt = [0] * 9
    for y in range(height):
        row = y * width
        cell_row = (y * 3 // height) * 3
        for x in range(width):
            r, g, b = at(row + x)
            cell = cell_row + (x * 3 // width)
            luma_sum[cell] += 0.2126 * r + 0.7152 * g + 0.0722 * b
            luma_cnt[cell] += 1
            if abs(r - bg[0]) + abs(g - bg[1]) + abs(b - bg[2]) > 24:
                ink += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

    if ink == 0:
        return "render: EMPTY — every pixel matches the background color"

    grid = [
        round((luma_sum[i] / luma_cnt[i]) / 255.0 * 9.0) if luma_cnt[i] else 0
        for i in range(9)
    ]
    # Cells were accumulated bottom-up; present top row first (reading order).
    rows = [grid[6:9], grid[3:6], grid[0:3]]
    grid_str = " / ".join(" ".join(str(v) for v in row) for row in rows)
    pct = round(ink / n * 100)
    return (
        f"render: ink {pct}% | bbox x {min_x / width:.2f}-{(max_x + 1) / width:.2f}, "
        f"y {min_y / height:.2f}-{(max_y + 1) / height:.2f} (y=0 bottom) | "
        f"luma 0-9 top/mid/bottom rows: {grid_str}"
    )
