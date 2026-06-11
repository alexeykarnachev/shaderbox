"""The chat display sanitizer (020·20 D2 + 034 F11 Cyrillic pass-through) — pure text."""

from shaderbox.copilot.sanitize import sanitize_display


def test_ascii_passes_verbatim() -> None:
    assert sanitize_display("plain -> text") == "plain -> text"


def test_known_glyphs_transliterate() -> None:
    assert sanitize_display("a → b — c…") == "a -> b -- c..."


def test_cyrillic_passes_through() -> None:
    assert sanitize_display("ШЕЙДЕР ok") == "ШЕЙДЕР ok"
    # Ё / ё live outside the А-я run but inside the block.
    assert sanitize_display("Ёё") == "Ёё"


def test_unrenderable_becomes_question_mark() -> None:
    assert sanitize_display("ok 中") == "ok ?"


def test_idempotent_on_own_output() -> None:
    once = sanitize_display("Ш → 中")
    assert sanitize_display(once) == once
