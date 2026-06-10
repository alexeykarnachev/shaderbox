"""Engine-computed facts appended to edit-tool results (feature 033).

Pure text/pixel analysis — no GL, no imgui. The compiler says WHAT broke; the
hints say WHY in terms the model can act on (the dominant dogfood failure class:
a line edit whose new_text duplicates content that survived outside the range).
Render facts are the agent's only sight: scalar truths about the latest clean
frame, computed from raw RGBA bytes the backend reads off a tiny probe canvas.
"""

import re

from shaderbox.shader_lib import parser

# Duplicate-declaration wordings across the closed vendor set (Mesa GLSL-IR,
# glslang/Apple/Intel-Windows, NVIDIA, AMD) — the hint itself is source-verified
# (fires only when >=2 declaration lines actually exist), so broad matching is safe.
_REDECLARED_RES = (
    re.compile(r"`(\w+)' (?:redeclared|redefined)"),
    re.compile(r"'(\w+)'\s*:\s*(?:redefinition|already)"),
    re.compile(r'"(\w+)".*conflicts with previous declaration'),
    re.compile(r"[Rr]edeclaration of symbol:?\s*'?(\w+)'?"),
)
# Any wording that smells like an array-initializer size problem; the COUNT comes
# from the post-edit source (driver-agnostic), not from the message.
_INITIALIZER_SMELL_RE = re.compile(
    r"initializ|too many elements|element array", re.IGNORECASE
)
# Both GLSL spellings: `type[N] name = type[](...)` and `type name[N] = type[](...)`.
_ARRAY_INIT_RE = re.compile(
    r"(\w+)\s*(?:\[\s*(\d+)\s*\])?\s+\w+\s*(?:\[\s*(\d+)\s*\])?\s*="
    r"\s*\w+\s*\[\s*\d*\s*\]\s*\("
)
# A declaration-ish line: the FULL (closed, spec-defined) qualifier set + optional
# layout(...), a type (with optional array suffix), then the identifier followed by
# `=`, `;`, `[` or `(` (function defs redeclare too).
_DECL_HEAD = (
    r"^\s*(?:layout\s*\([^)]*\)\s*)?"
    r"(?:(?:const|uniform|in|out|inout|flat|smooth|noperspective|centroid|sample|"
    r"patch|invariant|precise|highp|mediump|lowp|varying|attribute|buffer|shared)\s+)*"
    r"[A-Za-z_]\w*(?:\s*\[\s*\d*\s*\])?\s+"
)


def compile_hints(source: str, error_messages: list[str]) -> list[str]:
    """Structural hints for compile errors, derived from the post-edit source."""
    hints: list[str] = []
    stripped = parser.strip_comments_keep_lines(source)
    lines = stripped.splitlines()
    seen: set[str] = set()
    # Drivers can deliver every error in ONE blob message — iterate ALL matches of
    # ALL vendor wordings per message, not just the first.
    init_smell = False
    for msg in error_messages:
        for rx in _REDECLARED_RES:
            for m in rx.finditer(msg):
                name = m.group(1)
                if name in seen:
                    continue
                seen.add(name)
                decl = re.compile(_DECL_HEAD + re.escape(name) + r"\s*[=;[(]")
                decl_lines = [
                    i for i, ln in enumerate(lines, start=1) if decl.match(ln)
                ]
                if len(decl_lines) >= 2:
                    rows = ", ".join(str(i) for i in decl_lines)
                    hints.append(
                        f"hint: '{name}' is declared on lines {rows} — the edit "
                        "range missed one copy; widen the range over it or delete "
                        "the duplicate"
                    )
        if _INITIALIZER_SMELL_RE.search(msg):
            init_smell = True
    if init_smell:
        # Count source-side (driver-agnostic): any sized array whose initializer
        # list length mismatches its declared size gets a quantified hint.
        for i, ln in enumerate(lines, start=1):
            m = _ARRAY_INIT_RE.search(ln)
            if not m or (m.group(2) is None and m.group(3) is None):
                continue
            kind = m.group(1)
            want = int(m.group(2) or m.group(3))
            got = _count_initializer_elements(stripped, i - 1)
            if got is not None and got != want:
                hints.append(
                    f"hint: line {i}: the initializer has {got} elements, the "
                    f"array wants {want} — declare it unsized "
                    f"(`{kind}[] x = {kind}[](...)`) to skip counting; for TEXT "
                    "skip const arrays entirely: uniform uint u_text[64] + "
                    "set_uniform"
                )
    opens, closes = stripped.count("{"), stripped.count("}")
    if opens != closes:
        loc = _brace_imbalance_location(stripped)
        hints.append(
            f"hint: the file has {opens} '{{' vs {closes} '}}'{loc} — an orphan "
            "tail survived below the edit range, or a brace went missing in new_text"
        )
    return hints


def _count_initializer_elements(stripped: str, decl_line_idx: int) -> int | None:
    # Count top-level commas inside the type[](...) initializer starting on the
    # given line; spans multiple lines until the matching ')'.
    text = "\n".join(stripped.splitlines()[decl_line_idx:])
    start = text.find("(")
    if start < 0:
        return None
    depth = 0
    count = 1
    empty = True
    for ch in text[start:]:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return 0 if empty else count
        elif depth == 1:
            if ch == ",":
                count += 1
            elif not ch.isspace():
                empty = False
    return None


def _brace_imbalance_location(stripped: str) -> str:
    # Locate the imbalance: the line where depth first goes negative (extra '}'),
    # or the opener line of the block left unclosed at EOF (missing '}').
    depth = 0
    open_stack: list[int] = []
    for i, ln in enumerate(stripped.splitlines(), start=1):
        for ch in ln:
            if ch == "{":
                depth += 1
                open_stack.append(i)
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    return f" (first unmatched '}}' on line {i})"
                if open_stack:
                    open_stack.pop()
    if open_stack:
        return f" (the block opened on line {open_stack[-1]} never closes)"
    return ""


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
        # Self-describing flat-frame verdict: a blank AND a full-screen fill both
        # land here — the color + max deviation lets the agent tell them apart.
        max_dev = 0
        for y in range(height):
            row = y * width
            for x in range(width):
                r, g, b = at(row + x)
                dev = abs(r - bg[0]) + abs(g - bg[1]) + abs(b - bg[2])
                if dev > max_dev:
                    max_dev = dev
        return (
            f"render: FLAT — one uniform color rgb({bg[0]},{bg[1]},{bg[2]}), "
            f"max pixel deviation {max_dev}/765 (a blank OR a full-screen fill)"
        )

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
