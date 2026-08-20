"""Content-addressed text matching + splicing for the copilot edit tools — the Python/plain-text
half of the job `glsl_lex.py` does for GLSL. Pure functions over `(source, old_str)`: no GL, no
project state, no `self`. Two matcher families, because the two languages need different tolerance:

- GLSL / free-form text: whitespace-normalized matching (`ws_normalize` + `whitespace_near_match`)
  and comment-only spans, layered on the GLSL token matcher in `glsl_lex`.
- Python scripts: exact substring first, then an indent-level structural fallback
  (`script_match_spans`) that forgives a re-typed leading indent, spliced back with `reindent`.

A leaf: imports only `glsl_lex` + stdlib, so `backend.py` pulls it in without a cycle."""

from shaderbox.copilot.glsl_lex import glsl_lex

def ws_normalize(text: str) -> tuple[str, list[int]]:
    # Collapse horizontal-whitespace runs to one space (dropped adjacent to a newline) and
    # return the normalized text + a per-char map back to original indices, so a normalized-space
    # match can be sliced back to exact original bytes.
    out: list[str] = []
    src_index: list[int] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in " \t":
            j = i
            while j < n and text[j] in " \t":
                j += 1
            # One space only between two non-newline chars; a run touching a newline collapses away.
            prev = out[-1] if out else "\n"
            nxt = text[j] if j < n else "\n"
            if prev != "\n" and nxt != "\n":
                out.append(" ")
                src_index.append(i)
            i = j
        else:
            out.append(c)
            src_index.append(i)
            i += 1
    src_index.append(n)  # sentinel: end of the last char maps past the source end
    return "".join(out), src_index


def whitespace_near_match(src: str, old_str: str) -> str:
    # The unique region of src matching old_str ignoring whitespace, as exact original bytes;
    # "" when no match or not unique.
    norm_src, src_index = ws_normalize(src)
    norm_old, _ = ws_normalize(old_str)
    if not norm_old:
        return ""
    first = norm_src.find(norm_old)
    if first == -1 or norm_src.find(norm_old, first + 1) != -1:
        return ""  # no match, or ambiguous — no safe single hint
    return src[src_index[first] : src_index[first + len(norm_old)]]


def comment_only_spans(src: str, old_str: str) -> list[tuple[int, int]] | None:
    # None = old_str has code tokens (the token matcher owns it). A COMMENT/whitespace-only
    # old_str is invisible to the token matcher (comments lex as trivia), so it matches by
    # whitespace-normalized TEXT instead — comments are editable content too.
    if glsl_lex(old_str):
        return None
    norm_src, src_index = ws_normalize(src)
    norm_old, _ = ws_normalize(old_str)
    if not norm_old.strip():
        return []
    spans: list[tuple[int, int]] = []
    pos = norm_src.find(norm_old)
    while pos != -1:
        spans.append((src_index[pos], src_index[pos + len(norm_old)]))
        pos = norm_src.find(norm_old, pos + len(norm_old))
    return spans


def _indent_levels(lines: list[str]) -> tuple[tuple[int, str], ...]:
    # The structural KEY of a Python block, indent-ABSOLUTE-agnostic: each non-blank line as
    # (indent LEVEL, stripped content), where level = the rank of its indent among the DISTINCT
    # indents in the block. So a 6-space block (indents {6,12} → levels {0,1}) and an 8-space block
    # ({8,16} → {0,1}) share a key — the agent's re-typed indent is forgiven — while a real nesting
    # difference (if-body vs flat sibling) changes the level pattern and does NOT match. Tabs are
    # already 4 spaces by the time source reaches disk (normalize_script_tabs); old_str is normalized
    # at the call site, so this is a spaces-only world.
    raw = [len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()]
    rank = {ind: r for r, ind in enumerate(sorted(set(raw)))}
    return tuple(
        (rank[len(ln) - len(ln.lstrip())], ln.strip()) for ln in lines if ln.strip()
    )


def script_match_spans(src: str, old_str: str) -> list[tuple[int, int, int]]:
    # Match old_str against the script source, returning (start, end, indent_shift) per occurrence.
    # FAST PATH: an exact substring (the common case — most edits are verbatim). FALLBACK: the
    # indent-aware structural match, so a re-typed leading indent still lands (the 043 breakout
    # 6-vs-8-space miss). `indent_shift` = the matched region's leading indent minus old_str's, so the
    # splice can re-indent new_str onto the real column. Empty old_str matches nothing.
    if not old_str:
        return []
    exact: list[tuple[int, int, int]] = []
    start = src.find(old_str)
    while start != -1:
        exact.append((start, start + len(old_str), 0))
        start = src.find(old_str, start + len(old_str))
    if exact:
        return exact
    # structural fallback: scan line windows, compare the indent-level key
    old_lines = old_str.rstrip("\n").split("\n")
    old_key = _indent_levels(old_lines)
    if not old_key:
        return []
    old_indent = next((len(ln) - len(ln.lstrip()) for ln in old_lines if ln.strip()), 0)
    src_lines = src.split("\n")
    offsets: list[int] = []
    pos = 0
    for ln in src_lines:
        offsets.append(pos)
        pos += len(ln) + 1
    n = len(old_lines)
    spans: list[tuple[int, int, int]] = []
    for i in range(len(src_lines) - n + 1):
        window = src_lines[i : i + n]
        if _indent_levels(window) != old_key:
            continue
        first = next((ln for ln in window if ln.strip()), "")
        win_indent = len(first) - len(first.lstrip())
        start = offsets[i] + (
            len(window[0]) - len(window[0].lstrip()) if window[0].strip() else 0
        )
        end = offsets[i + n - 1] + len(window[n - 1])
        spans.append((start, end, win_indent - old_indent))
    return spans


def reindent(text: str, shift: int) -> str:
    # Shift every non-blank line's leading indent by `shift` columns (>=0 add, <0 remove). Used to
    # re-indent new_str onto the absolute column of a structurally-matched (indent-shifted) region.
    if shift == 0:
        return text
    out: list[str] = []
    for ln in text.split("\n"):
        if not ln.strip():
            out.append(ln)
        elif shift > 0:
            out.append(" " * shift + ln)
        else:
            cut = min(-shift, len(ln) - len(ln.lstrip()))
            out.append(ln[cut:])
    return "\n".join(out)


def splice_script(src: str, spans: list[tuple[int, int, int]], new_str: str) -> str:
    # Replace each (start, end, indent_shift) span with new_str, re-indented by the span's shift so a
    # structural (indent-forgiven) match lands the replacement at the right column. Offset-stable.
    out: list[str] = []
    cursor = 0
    for start, end, shift in spans:
        out.append(src[cursor:start])
        out.append(reindent(new_str, shift))
        cursor = end
    out.append(src[cursor:])
    return "".join(out)


def splice(src: str, spans: list[tuple[int, int]], new_str: str) -> str:
    # Replace each non-overlapping (start, end) span with new_str. Offset-stable: spans don't overlap.
    out: list[str] = []
    cursor: int = 0
    for start, end in spans:
        out.append(src[cursor:start])
        out.append(new_str)
        cursor = end
    out.append(src[cursor:])
    return "".join(out)
