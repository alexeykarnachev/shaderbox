"""Unit tests for the chat markdown-lite parser (ui_primitives) — no GL, no imgui calls."""

from shaderbox.ui_primitives import parse_markdown_lines


def test_plain_text_single_span() -> None:
    assert parse_markdown_lines("hello world") == [
        (False, [("plain", "hello world")])
    ]


def test_bold_and_code_spans() -> None:
    lines = parse_markdown_lines("nodes **Blank** and `My Text` removed")
    assert lines == [
        (
            False,
            [
                ("plain", "nodes "),
                ("bold", "Blank"),
                ("plain", " and "),
                ("code", "My Text"),
                ("plain", " removed"),
            ],
        )
    ]


def test_unmatched_markers_stay_literal() -> None:
    # A torn streaming preview (unclosed **) must render as-is, not crash or eat text.
    lines = parse_markdown_lines("2 ** 3 is 8, and **unclosed")
    assert lines == [(False, [("plain", "2 ** 3 is 8, and **unclosed")])]
    assert parse_markdown_lines("a ` lone backtick") == [
        (False, [("plain", "a ` lone backtick")])
    ]


def test_fenced_block_lines_marked_code() -> None:
    lines = parse_markdown_lines("look:\n```glsl\nfloat d = 1.0;\n```\ndone")
    assert lines == [
        (False, [("plain", "look:")]),
        (True, [("code", "float d = 1.0;")]),
        (False, [("plain", "done")]),
    ]


def test_unclosed_fence_marks_rest_as_block() -> None:
    lines = parse_markdown_lines("```\nstill code")
    assert lines == [(True, [("code", "still code")])]


def test_empty_and_blank_lines_preserved() -> None:
    lines = parse_markdown_lines("a\n\nb")
    assert lines == [
        (False, [("plain", "a")]),
        (False, []),
        (False, [("plain", "b")]),
    ]


def test_markers_inside_code_span_not_nested() -> None:
    lines = parse_markdown_lines("`**not bold**`")
    assert lines == [(False, [("code", "**not bold**")])]
