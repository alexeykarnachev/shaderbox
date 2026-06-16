"""Unit tests for the GLSL lexer + token matcher (feature 020 · 13).

Pure — no GL, no App. This is the primary verification surface for edit_shader's
whitespace-invariant matcher; the adversarial reject tables (numbers, wrong-region) are
the load-bearing ones — they prove the matcher REJECTS what it must, not only that it
matches what it should.
"""

from shaderbox.copilot.backend import _splice
from shaderbox.copilot.glsl_lex import (
    TokenKind,
    comments_in,
    glsl_lex,
    span_drops_comment,
    token_match,
)


def _kinds_raws(src: str) -> list[tuple[TokenKind, str]]:
    return [(t.kind, t.raw) for t in glsl_lex(src)]


def test_lexer_basics_and_spans() -> None:
    src = "vec3 p = u_pos;"
    toks = glsl_lex(src)
    assert [(t.kind, t.raw) for t in toks] == [
        (TokenKind.IDENT, "vec3"),
        (TokenKind.IDENT, "p"),
        (TokenKind.PUNCT, "="),
        (TokenKind.IDENT, "u_pos"),
        (TokenKind.PUNCT, ";"),
    ]
    # Every token's span indexes the original source verbatim.
    for t in toks:
        assert src[t.start : t.end] == t.raw


def test_maximal_munch() -> None:
    assert _kinds_raws("a<=b") == [
        (TokenKind.IDENT, "a"),
        (TokenKind.PUNCT, "<="),
        (TokenKind.IDENT, "b"),
    ]
    assert _kinds_raws(">>=") == [(TokenKind.PUNCT, ">>=")]
    assert _kinds_raws("a >> b") == [
        (TokenKind.IDENT, "a"),
        (TokenKind.PUNCT, ">>"),
        (TokenKind.IDENT, "b"),
    ]


def test_comments_and_whitespace_skipped_spans_preserved() -> None:
    src = "  vec3 p; // trailing\n  /* block */ p = 1.0;"
    bare = "vec3 p; p = 1.0;"
    assert _kinds_raws(src) == _kinds_raws(bare)
    # Spans still index the ORIGINAL source (positional skip, not a pre-strip).
    for t in glsl_lex(src):
        assert src[t.start : t.end] == t.raw


def test_number_accept_table() -> None:
    for lexeme in [
        "1",
        "1u",
        "1.0",
        ".5",
        "1.",
        "1e3",
        "1.0e-3",
        "2.0f",
        "1.0lf",
        "0xFF",
        "0xFFu",
        "0xFe",
    ]:
        toks = glsl_lex(lexeme)
        assert len(toks) == 1, f"{lexeme!r} should be ONE token, got {toks}"
        assert toks[0].kind == TokenKind.NUMBER
        assert toks[0].raw == lexeme


def test_number_reject_table() -> None:
    # The `-` in subtraction is NOT absorbed into a signed number.
    assert _kinds_raws("a-1") == [
        (TokenKind.IDENT, "a"),
        (TokenKind.PUNCT, "-"),
        (TokenKind.NUMBER, "1"),
    ]
    # But the exponent sign stays inside the one number.
    assert _kinds_raws("1.0e-3") == [(TokenKind.NUMBER, "1.0e-3")]
    # A lone `.` (member access) is punctuation, never a number.
    assert _kinds_raws("a.b") == [
        (TokenKind.IDENT, "a"),
        (TokenKind.PUNCT, "."),
        (TokenKind.IDENT, "b"),
    ]
    # `.5` (one number) and `. 5` (dot then number) do NOT lex the same.
    assert _kinds_raws(".5") == [(TokenKind.NUMBER, ".5")]
    assert _kinds_raws(". 5") == [(TokenKind.PUNCT, "."), (TokenKind.NUMBER, "5")]
    assert not token_match(".5", ". 5")
    assert not token_match(". 5", ".5")


def test_raw_text_honesty_numbers_differ() -> None:
    # I3: 1.0 and 1.00 are different lexemes (raw differs) — must not match.
    assert glsl_lex("1.0")[0].raw != glsl_lex("1.00")[0].raw
    assert not token_match("x = 1.0;", "1.00")
    assert not token_match("x = 1.00;", "1.0")


def test_token_match_whitespace_invariance_preserves_source_bytes() -> None:
    src = "  vec3  p\t=\tu_pos ;"
    spans = token_match(src, "vec3 p = u_pos;")
    assert len(spans) == 1
    start, end = spans[0]
    # The span is the SOURCE's bytes (its odd spacing), so a splice preserves the rest.
    matched = src[start:end]
    assert matched.startswith("vec3")
    assert matched.endswith(";")
    spliced = src[:start] + "X" + src[end:]
    assert spliced == "  X"  # the two leading spaces survive; the whole run is replaced


def test_token_match_ambiguous_whitespace_variants() -> None:
    # I4: an old_str matching two whitespace-variant regions -> 2 spans (caller's >1 rule).
    src = "a = 1;\na  =  1;"
    spans = token_match(src, "a = 1;")
    assert len(spans) == 2


def test_token_match_non_overlapping_advance() -> None:
    # After a match the scan resumes PAST the consumed run (str.replace semantics).
    spans = token_match("a a a a", "a a")
    assert spans == [(0, 3), (4, 7)]  # disjoint, NOT (0,3),(2,5),(4,7)


def test_lexer_never_crashes_on_odd_input() -> None:
    # The "never crash" contract: a char str.isdigit() accepts but \d rejects (unicode
    # superscript), an unterminated block comment, and a stray byte must all lex without
    # raising — each odd char just becomes its own PUNCT (or is skipped, for the comment).
    for src in ["x = ² ;", "a /* unterminated", "x = ①;", "@#$"]:
        glsl_lex(src)  # must not raise


def test_division_is_not_a_comment() -> None:
    assert _kinds_raws("a / b") == [
        (TokenKind.IDENT, "a"),
        (TokenKind.PUNCT, "/"),
        (TokenKind.IDENT, "b"),
    ]


def test_unterminated_block_comment_terminates() -> None:
    # Everything after an unterminated /* is consumed as comment -> no tokens, no hang.
    assert glsl_lex("vec3 p; /* trailing") == glsl_lex("vec3 p;")


def test_replace_all_multi_span_splice() -> None:
    src = "f(x); g(y); f(x);"
    spans = token_match(src, "f(x)")
    assert len(spans) == 2
    out: list[str] = []
    cursor = 0
    for start, end in spans:
        out.append(src[cursor:start])
        out.append("h(z)")
        cursor = end
    out.append(src[cursor:])
    assert "".join(out) == "h(z); g(y); h(z);"


def test_empty_and_degenerate_needle() -> None:
    src = "vec3 p = u_pos;"
    assert token_match(src, "") == []
    assert token_match(src, "   ") == []
    assert token_match(src, "// just a comment") == []
    assert token_match(src, "/* block */") == []
    assert glsl_lex("") == []


def test_true_zero_match() -> None:
    # A token typo or dropped token -> no span (the hint path then runs upstream).
    assert token_match("vec3 p = u_pos;", "vec3 p = u_poss;") == []
    assert token_match("vec3 p = u_pos;", "p = u_pos = q;") == []


def test_comments_in_extracts_normalized() -> None:
    src = "a = 1; // first\nb = /*  inner  */ 2;"
    assert comments_in(src) == ["// first", "/* inner */"]
    assert comments_in("no comments here") == []


# The comment-loss guard: reject only a SILENT drop (comment in the matched span but not in
# old_str), allow a deliberate rewrite (comment quoted in old_str). The bug this fixes: the
# old guard rejected ANY span-with-a-comment, blocking legitimate rewrites (real trace case).
_SRC = (
    "uv.y /= u_aspect;\n\n// center-based polar coordinates\nvec2 p = uv * 2.0 - 1.0;"
)


def test_guard_allows_deliberate_comment_rewrite() -> None:
    # old_str reproduces the comment -> the rewrite is intentional, NOT a silent drop. (The
    # exact shape that wrongly failed in the 2026-06-08 trace.)
    old = (
        "uv.y /= u_aspect;\n// center-based polar coordinates\nvec2 p = uv * 2.0 - 1.0;"
    )
    (s, e), *_ = token_match(_SRC, old)
    assert span_drops_comment(_SRC, s, e, old) is False


def test_guard_rejects_silent_comment_drop() -> None:
    # old_str omits the comment that sits inside the matched span -> a verbatim splice would
    # delete it. Must still reject (the protection the guard exists for).
    old = "uv.y /= u_aspect;\nvec2 p = uv * 2.0 - 1.0;"
    (s, e), *_ = token_match(_SRC, old)
    assert span_drops_comment(_SRC, s, e, old) is True


def test_guard_no_comment_in_span_never_fires() -> None:
    src = "vec2 p = uv * 2.0 - 1.0;\nfloat r = length(p);"
    old = "float r = length(p);"
    (s, e), *_ = token_match(src, old)
    assert span_drops_comment(src, s, e, old) is False


# --- feature 050: leading-comment span-grow (the duplicate-comment spiral root) ---

_DUP_SRC = (
    "void main() {\n"
    "    vec2 p = fire_space(vs_uv, u_aspect);\n\n"
    "    // Step 2: animate\n"
    "    float heat = fire_heat_distorted(p, u_time);\n}\n"
)


def test_leading_comment_not_duplicated_by_splice() -> None:
    # An edit whose old_str LEADS with a comment above its code: the matched span must include the
    # comment so _splice REPLACES it, instead of leaving the file's original beside new_str's copy
    # (the 1->2 duplication that spiraled to 16). Verified end-to-end on the real splice.
    old = "    // Step 2: animate\n    float heat = fire_heat_distorted(p, u_time);\n"
    new = (
        "    // Step 2: animate\n"
        "    // Step 3: shape\n"
        "    float heat = fire_heat_distorted(p, u_time);\n"
    )
    spans = token_match(_DUP_SRC, old)
    assert len(spans) == 1
    out = _splice(_DUP_SRC, spans, new)
    assert out.count("// Step 2: animate") == 1  # NOT 2


def test_cleanup_of_dupes_converges_not_grows() -> None:
    # A cleanup edit on an already-2-copy file collapses to 1 (not 3) and a re-run does not climb -
    # the monotonic-growth spiral is structurally dead.
    src = "void main() {\n    // Step 2\n        // Step 2\n    float heat = h(p);\n}\n"
    old = "    // Step 2\n        // Step 2\n    float heat = h(p);\n"
    new = "    // Step 2\n    float heat = h(p);\n"
    out = _splice(src, token_match(src, old), new)
    assert out.count("// Step 2") == 1


def test_leading_comment_rename_still_lands() -> None:
    # The false-positive direction: a legitimate comment RENAME (old label -> new label) must
    # replace the label exactly once - no stray old label left above the span, no duplicate.
    src = "void f() {\n    // old label\n    int x = 1;\n}\n"
    old = "    // old label\n    int x = 1;\n"
    new = "    // new label\n    int x = 1;\n"
    out = _splice(src, token_match(src, old), new)
    assert "old label" not in out
    assert out.count("// new label") == 1


def test_guard_still_fires_when_grown_span_drops_a_unique_comment() -> None:
    # The grow must not weaken the guard: when old_str OMITS an interior comment the matched span
    # covers (a genuine silent-loss), span_drops_comment STILL fires on the (possibly grown) span.
    src = (
        "void f() {\n    // header\n    int a = 1;\n    // keep me\n    int b = 2;\n}\n"
    )
    # old_str leads with '// header' (grown) but OMITS the interior '// keep me' between the code.
    old = "    // header\n    int a = 1;\n    int b = 2;\n"
    (s, e), *_ = token_match(src, old)
    assert (
        span_drops_comment(src, s, e, old) is True
    )  # '// keep me' would be silently lost


def test_trailing_comment_grow_does_not_duplicate() -> None:
    # Symmetric to the leading case: old_str ending with a comment BELOW its last code token grows
    # the span END forward so the splice replaces it, not duplicates it.
    src = "void f() {\n    int x = 1;\n    // trailing note\n}\n"
    old = "    int x = 1;\n    // trailing note\n"
    new = "    int x = 2;\n    // trailing note\n"
    out = _splice(src, token_match(src, old), new)
    assert out.count("// trailing note") == 1
    assert "int x = 2;" in out


def test_block_comment_not_grown_known_limit() -> None:
    # KNOWN LIMIT: the grow walks LONE single-line comments only; a multi-line /* */ block leading
    # old_str is NOT grown, so a block-comment-led edit can still duplicate. Agents emit // section
    # comments (the spiral's shape), never /* */ blocks, so this is parked, not fixed. This test
    # PINS the current behavior so a future block-comment grow is a deliberate change, not a surprise.
    src = "void f() {\n    /* note line1\n       note line2 */\n    int x = 1;\n}\n"
    old = "    /* note line1\n       note line2 */\n    int x = 1;\n"
    new = "    /* note line1\n       note line2 */\n    int x = 2;\n"
    out = _splice(src, token_match(src, old), new)
    # Not grown -> the block survives above AND new_str re-inserts it -> duplicated (the known limit).
    assert out.count("note line1") == 2
