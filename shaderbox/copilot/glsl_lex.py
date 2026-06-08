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


def token_match(src: str, old_str: str) -> list[tuple[int, int]]:
    # Every contiguous source-token run whose (kind, raw) sequence equals old_str's, as a
    # (byte_start, byte_end) span into src. Anchored full-length, non-overlapping
    # left-to-right (matches str.replace). ALL matches returned so an ambiguous old_str
    # surfaces as ">1" rather than a silent wrong-region edit.
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
            spans.append((hay[i].start, hay[i + k - 1].end))
            i += k  # non-overlapping: resume past the consumed run
        else:
            i += 1
    return spans
