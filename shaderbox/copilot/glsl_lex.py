"""A minimal GLSL tokenizer + whitespace-invariant token-stream matcher.

`token_match` lexes both sides and compares token streams, ignoring inter-token
whitespace, so a whitespace-divergent copy still locates. Returned spans index the
EXACT original bytes — replacement splices in place without touching formatting.

Leaf: no GL, no imgui, no App, no copilot-package imports.
"""

import re
from dataclasses import dataclass
from enum import StrEnum


class TokenKind(StrEnum):
    IDENT = "ident"
    NUMBER = "number"
    PUNCT = "punct"
    PREPROC_HASH = "preproc_hash"


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    raw: str  # verbatim source slice — compared as-is, never canonicalized
    start: int  # byte offset into the lexed source
    end: int


# Number sub-grammars, hex-first (the `f`/`e` in 0xFe are hex digits, not a float suffix).
# Float `[+-]` is reachable only after `e`/`E`: a bare leading `-` stays subtraction
# (`a-1` -> a, -, 1). A lone `.` not followed by a digit falls through to PUNCT.
_HEX_RE = re.compile(r"0[xX][0-9a-fA-F]+[uU]?")
_FLOAT_RE = re.compile(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?(?:lf|LF|[uUfF])?")

_IDENT_RE = re.compile(r"[A-Za-z_]\w*")

# Multi-char operators, longest-first (maximal munch). Single chars handle the rest.
_MULTI_PUNCT: tuple[str, ...] = (
    "<<=",
    ">>=",
    "<=",
    ">=",
    "==",
    "!=",
    "&&",
    "||",
    "++",
    "--",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "<<",
    ">>",
)


def _skip_trivia(src: str, i: int, n: int) -> int:
    # Advance past whitespace/comments POSITIONALLY — a re.sub pre-strip would break spans.
    # GLSL comment rules (mirror shader_lib/parser.strip_comments): //...\n, /*...*/,
    # no strings, no nesting.
    while i < n:
        c: str = src[i]
        if c in " \t\r\n":
            i += 1
        elif c == "/" and i + 1 < n and src[i + 1] == "/":
            i += 2
            while i < n and src[i] != "\n":
                i += 1
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            i += 2
            while i < n and not (src[i] == "*" and i + 1 < n and src[i + 1] == "/"):
                i += 1
            i = min(i + 2, n)  # consume closing */ (or land at EOF if unterminated)
        else:
            break
    return i


def glsl_lex(src: str) -> list[Token]:
    tokens: list[Token] = []
    i: int = 0
    n: int = len(src)
    while True:
        i = _skip_trivia(src, i, n)
        if i >= n:
            break
        c: str = src[i]
        # Probe number first (hex before float); a `.` only starts a number when a digit
        # follows. Regexes are anchored at i, so a non-match falls through to ident/punct.
        num: re.Match[str] | None = None
        if c in "0123456789" or (c == "." and i + 1 < n and src[i + 1] in "0123456789"):
            num = _HEX_RE.match(src, i) or _FLOAT_RE.match(src, i)
        if num is not None:
            tokens.append(Token(TokenKind.NUMBER, num.group(), i, num.end()))
            i = num.end()
            continue
        ident: re.Match[str] | None = _IDENT_RE.match(src, i)
        if ident is not None:
            tokens.append(Token(TokenKind.IDENT, ident.group(), i, ident.end()))
            i = ident.end()
            continue
        if c == "#":
            tokens.append(Token(TokenKind.PREPROC_HASH, c, i, i + 1))
            i += 1
            continue
        op: str | None = next((p for p in _MULTI_PUNCT if src.startswith(p, i)), None)
        if op is not None:
            tokens.append(Token(TokenKind.PUNCT, op, i, i + len(op)))
            i += len(op)
            continue
        # Any single char (incl. an unrecognized byte) becomes its own PUNCT — never crash.
        tokens.append(Token(TokenKind.PUNCT, c, i, i + 1))
        i += 1
    return tokens


def comments_in(src: str, start: int = 0, end: int | None = None) -> list[str]:
    # The normalized text of every // or /* comment in src[start:end], in order. token_match
    # ignores comments, so a comment between two matched tokens is invisible to the match yet
    # inside the span. The caller compares the span's comments against old_str's: a comment in
    # the span but NOT in old_str would be silently deleted by a verbatim splice (reject); one
    # present in both is an intentional rewrite (allow). Normalized (stripped + inner whitespace
    # collapsed) so a reformat-but-same comment in old_str still counts as reproduced.
    n: int = len(src) if end is None else end
    out: list[str] = []
    i: int = start
    while i < n:
        c: str = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = i + 2
            while j < n and src[j] != "\n":
                j += 1
            out.append(_norm_comment(src[i:j]))
            i = j
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            j = i + 2
            while j < n and not (src[j] == "*" and j + 1 < n and src[j + 1] == "/"):
                j += 1
            j = min(j + 2, n)
            out.append(_norm_comment(src[i:j]))
            i = j
        else:
            i += 1
    return out


def _norm_comment(text: str) -> str:
    return " ".join(text.split())


def span_drops_comment(src: str, start: int, end: int, old_str: str) -> bool:
    # True iff the matched source span src[start:end] contains a comment that old_str does NOT
    # reproduce — i.e. a verbatim splice would SILENTLY delete it (reject). A comment present in
    # both is a deliberate rewrite (allow). Multiset-aware so two identical comments aren't
    # masked by one in old_str.
    span = comments_in(src, start, end)
    quoted = comments_in(old_str)
    for c in quoted:
        if c in span:
            span.remove(c)
    return bool(span)


def _edge_comments(old_str: str, leading: bool) -> list[str]:
    # The normalized comment lines old_str quotes BEFORE its first code token (leading=True) or
    # AFTER its last code token (leading=False). token_match anchors on code tokens only, so these
    # edge comments fall OUTSIDE the matched span — _grow_span_over_comments pulls the span over the
    # matching source comments so a splice replaces them instead of duplicating new_str's copies.
    toks = glsl_lex(old_str)
    if not toks:
        return []
    cut = toks[0].start if leading else toks[-1].end
    region = old_str[:cut] if leading else old_str[cut:]
    return comments_in(region)


def _lone_comment(line: str) -> str | None:
    # The single normalized comment of a line that is ONLY whitespace + one comment, else None.
    if not line.strip().startswith(("//", "/*")):
        return None
    cs = comments_in(line)
    return cs[0] if len(cs) == 1 else None


def _comment_line_above(src: str, start: int) -> tuple[int, str] | None:
    # The lone comment line immediately above the span, as (line_start_offset, normalized_comment),
    # or None. The span START is grown back to line_start so the splice replaces the whole line.
    line_end = src.rfind("\n", 0, start)
    if line_end == -1:
        return None
    line_start = src.rfind("\n", 0, line_end) + 1
    c = _lone_comment(src[line_start:line_end])
    return (line_start, c) if c is not None else None


def _comment_line_below(src: str, end: int) -> tuple[int, str] | None:
    # The lone comment line immediately below the span, as (line_end_offset, normalized_comment),
    # or None. The span END is grown forward to line_end so the splice replaces the whole line.
    nl = src.find("\n", end)
    if nl == -1:
        return None
    line_start = nl + 1
    line_end = src.find("\n", line_start)
    if line_end == -1:
        line_end = len(src)
    c = _lone_comment(src[line_start:line_end])
    return (line_end, c) if c is not None else None


def _grow_span_over_comments(
    src: str, start: int, end: int, old_str: str
) -> tuple[int, int]:
    # Extend [start, end) over the source comment lines old_str quotes at its leading/trailing edge,
    # so _splice REPLACES them instead of leaving the file's originals beside new_str's copies (the
    # comment-duplication bug). Grows ONLY over comments old_str actually reproduces, matched in
    # order against the immediately-adjacent source lines — never swallows an unquoted comment.
    # LONE single-line (// or one-line /* */) comments only: a MULTI-LINE /* */ block at the edge is
    # not grown (so a block-comment-led edit can still duplicate). Agents emit // section comments,
    # never multi-line blocks, so it's parked; extend _comment_line_above/below to walk a block if it
    # ever fires.
    for want in reversed(_edge_comments(old_str, leading=True)):
        above = _comment_line_above(src, start)
        if above is None or above[1] != want:
            break
        start = above[0]
    for want in _edge_comments(old_str, leading=False):
        below = _comment_line_below(src, end)
        if below is None or below[1] != want:
            break
        end = below[0]
    return start, end


def token_match(src: str, old_str: str) -> list[tuple[int, int]]:
    # Every contiguous source-token run whose (kind, raw) sequence equals old_str's, as a
    # (byte_start, byte_end) span into src. Anchored full-length, non-overlapping
    # left-to-right (matches str.replace). ALL matches returned so an ambiguous old_str
    # surfaces as ">1" rather than a silent wrong-region edit. Each span is grown over the
    # leading/trailing comment lines old_str quotes (comments lex as trivia, so they'd otherwise
    # sit outside the span and get duplicated by the splice).
    needle: list[Token] = glsl_lex(old_str)
    if not needle:
        return []  # empty/whitespace-only/comment-only old_str — never a zero-length span
    hay: list[Token] = glsl_lex(src)
    key: list[tuple[TokenKind, str]] = [(t.kind, t.raw) for t in needle]
    hay_keys: list[tuple[TokenKind, str]] = [(t.kind, t.raw) for t in hay]
    spans: list[tuple[int, int]] = []
    k: int = len(needle)
    i: int = 0
    while i + k <= len(hay):
        if hay_keys[i : i + k] == key:
            spans.append(
                _grow_span_over_comments(src, hay[i].start, hay[i + k - 1].end, old_str)
            )
            i += k  # non-overlapping: resume past the consumed run
        else:
            i += 1
    return spans
