"""Unit tests for the GLSL lexer + token matcher (feature 020 · 13).

Pure — no GL, no App. This is the primary verification surface for edit_shader's
whitespace-invariant matcher; the adversarial reject tables (numbers, wrong-region) are
the load-bearing ones — they prove the matcher REJECTS what it must, not only that it
matches what it should.
"""

from shaderbox.copilot.glsl_lex import TokenKind, glsl_lex, token_match


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
