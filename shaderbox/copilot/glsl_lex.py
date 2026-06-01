"""A minimal GLSL tokenizer + whitespace-invariant token-stream matcher.

The copilot's `edit_shader` matches an `old_str` against the live shader source. A
byte-exact match (`str.count`/`str.replace`) fails when the model reproduces a region
correctly token-for-token but indents it differently. `token_match` lexes both sides and
compares the token streams, ignoring only inter-token whitespace, so a whitespace-divergent
copy still locates. The returned spans index the EXACT original bytes — replacement splices
in place and never touches the file's own formatting.

Leaf: no GL, no imgui, no App, no copilot-package imports. Spec: feature 020 · 13.
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
    raw: str  # verbatim source slice — compared as-is, never canonicalized (I3)
    start: int  # byte offset into the lexed source
    end: int


# Number sub-grammars, tried hex-first (the `f`/`e` in 0xFe/0xFF are hex digits, never a
# float suffix/exponent). The float branch's `[+-]` is reachable ONLY after `e`/`E`, so a
# bare leading `-` stays subtraction (`a-1` -> a, -, 1) and `1.0e-3` keeps its exponent
# sign in one token. A lone `.` not followed by a digit falls through to PUNCT.
_HEX_RE = re.compile(r"0[xX][0-9a-fA-F]+[uU]?")
_FLOAT_RE = re.compile(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?(?:lf|LF|[uUfF])?")

_IDENT_RE = re.compile(r"[A-Za-z_]\w*")

# Multi-char operators, longest-first (maximal munch, I2). Single chars handle the rest.
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
    # Advance past whitespace and comments POSITIONALLY (never a re.sub pre-strip — that
    # would break token spans). GLSL comment rules (also in shader_lib/parser.strip_comments,
    # kept in sync if those rules ever change): //...\n, /*...*/, no strings, no nesting.
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
            i = min(i + 2, n)  # consume the closing */ (or land at EOF if unterminated)
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
        # Probe number first (hex before float). A `.` only starts a number when a digit
        # follows; the regexes are anchored at i, so a non-match (e.g. a unicode "digit"
        # that \d rejects) falls through to ident/punct rather than crashing.
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


def token_match(src: str, old_str: str) -> list[tuple[int, int]]:
    # Every contiguous source-token run whose (kind, raw) sequence equals old_str's, as a
    # (byte_start, byte_end) span into src. Anchored (full-length, no prefix/subsequence),
    # non-overlapping left-to-right (matches str.replace), ALL matches returned (I4 — a
    # token-ambiguous old_str must surface as ">1", never a silent wrong-region edit).
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
