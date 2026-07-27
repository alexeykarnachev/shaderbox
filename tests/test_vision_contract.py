"""Feature 056: the eye's ASK contract — the prompt text that demands the final line, the parser
that reads it, and the strip that keeps it off every model-facing surface. Pure, no GL, no LLM."""

from shaderbox.copilot.llm.openrouter import _VISION_SYSTEM
from shaderbox.copilot.vision_contract import (
    ASK_CONTRACT_INSTRUCTION,
    ASK_MET,
    ASK_NOT_MET,
    ASK_UNCLEAR,
    find_ask_line,
    parse_ask_verdict,
    strip_ask_line,
)

_REPLY = (
    "coherent: clear subject | orientation: upright | artifacts: none\n"
    "look_for: no stars are visible on the blue field\n"
)


def test_vision_system_carries_the_contract_after_the_look_for_segment() -> None:
    # Two "final instruction" clauses must not interleave: the look_for segment is demanded first,
    # the ASK line last. Falsifier: the constant drifts out of the prompt, or lands above look_for.
    assert ASK_CONTRACT_INSTRUCTION in _VISION_SYSTEM
    assert _VISION_SYSTEM.index("`look_for:` segment") < _VISION_SYSTEM.index(
        ASK_CONTRACT_INSTRUCTION
    )


def test_vision_system_carves_the_ask_line_out_of_the_no_verdict_rule() -> None:
    # The standing "never a verdict" line and the mandated ASK line would contradict each other
    # without an explicit carve-out — the eye would have to disobey one of them.
    assert "exception" in _VISION_SYSTEM and "ASK line" in _VISION_SYSTEM


def test_parser_accepts_the_demanded_format() -> None:
    for verdict in (ASK_MET, ASK_NOT_MET, ASK_UNCLEAR):
        assert parse_ask_verdict(f"{_REPLY}ASK: {verdict}") == verdict
    assert parse_ask_verdict(f"{_REPLY}  ask:   NOT-MET  ") == ASK_NOT_MET


def test_missing_or_garbled_line_reads_as_unclear() -> None:
    # A format miss must never imply not-met — the eye can't fail its way to a negative verdict.
    assert parse_ask_verdict(_REPLY) == ASK_UNCLEAR
    assert parse_ask_verdict(f"{_REPLY}ASK: probably met?") == ASK_UNCLEAR
    assert parse_ask_verdict("") == ASK_UNCLEAR


def test_strip_removes_the_line_including_a_garbled_one() -> None:
    # The label never reaches a model-facing string — a garbled one carries the same done-ness
    # wording, so it is stripped too.
    for tail in ("ASK: not-met", "ASK: definitely not met, sorry"):
        stripped = strip_ask_line(f"{_REPLY}{tail}")
        assert "ASK" not in stripped
        assert "no stars are visible" in stripped
    assert strip_ask_line(_REPLY).endswith("blue field")


def test_find_ask_line_returns_the_raw_line_for_the_trace() -> None:
    assert find_ask_line(f"{_REPLY}ASK: met") == "ASK: met"
    assert find_ask_line(_REPLY) == ""


def test_the_last_ask_line_wins() -> None:
    # The contract puts the verdict LAST; an earlier stray mention must not outvote it (and both
    # are stripped, so neither reaches the model either way).
    reply = f"ASK: met (draft, ignore)\n{_REPLY}ASK: not-met"
    assert parse_ask_verdict(reply) == ASK_NOT_MET
    assert find_ask_line(reply) == "ASK: not-met"
    assert "ASK" not in strip_ask_line(reply)


def test_crlf_line_end_still_parses() -> None:
    assert parse_ask_verdict(f"{_REPLY}ASK: not-met\r\n") == ASK_NOT_MET


def test_strip_collapses_the_gap_it_leaves_behind() -> None:
    # A mid-body strip must not leave a blank hole in the read the model is shown.
    stripped = strip_ask_line("first line\n\nASK: met\n\nsecond line")
    assert stripped == "first line\n\nsecond line"


def test_the_ask_line_is_demanded_only_with_a_look_for_hint() -> None:
    # The engine's baseline (no hint) look must not draw a pointless verdict line.
    assert "ONLY when a 'Look for' hint was given" in ASK_CONTRACT_INSTRUCTION
