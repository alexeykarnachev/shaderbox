"""Slice-2 line-editing tools (feature 020 · 14).

Two layers, both GL-free:
- the pure line-edit math (app.py module functions);
- the tool-layer behavior (replace_lines / insert_after) driven through a fake App, incl.
  the start>end insert-vs-error disambiguation and the widened retry cap.
"""

import threading

from shaderbox.copilot.agent import AgentError, AgentToolCard, run_turn
from shaderbox.copilot.backend import _number_lines, _range_check_error
from shaderbox.copilot.capabilities import CompileErrorInfo, EditResult
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.gate import GateChannel
from shaderbox.copilot.llm.api import LLMDone, LLMStreamEvent, LLMTextDelta
from shaderbox.copilot.tools.registry import build_registry
from shaderbox.copilot.tools.shader import _applied_result
from tests.test_copilot_loop import _fake_caps, _fake_context, _FakeClient, _tool_call


# ---- the pure line-edit model (mirrors app.py::_copilot_apply_line_edit's list edit) ----
def _line_edit(src: str, start_line: int, end_line: int, new_text: str) -> str | None:
    lines = src.split("\n")
    n = len(lines)
    is_insert = end_line == start_line - 1
    if start_line < 1 or end_line > n or (start_line > end_line and not is_insert):
        return None  # out of range — caller fails soft
    repl = new_text.split("\n") if new_text != "" else []
    return "\n".join(lines[: start_line - 1] + repl + lines[end_line:])


def test_replace_lines_basic() -> None:
    assert _line_edit("a\nb\nc\nd", 2, 3, "X") == "a\nX\nd"
    # Multi-line replacement expands.
    assert _line_edit("a\nb\nc", 2, 2, "X\nY") == "a\nX\nY\nc"


def test_replace_lines_round_trip() -> None:
    # Replacing a range with its own current content is byte-identical (L2 inverse).
    src = "uniform float u;\nvoid main() {\n  gl_FragColor = vec4(u);\n}\n"
    assert _line_edit(src, 2, 3, "void main() {\n  gl_FragColor = vec4(u);") == src


def test_delete_via_empty_new_text() -> None:
    # Empty new_text deletes the range with no blank residue.
    assert _line_edit("a\nb\nc\nd", 2, 3, "") == "a\nd"


def test_insert_endpoints() -> None:
    # insert_after(k) -> _line_edit(k+1, k, ...): empty selection at position k+1.
    assert _line_edit("a\nb\nc\nd", 1, 0, "X") == "X\na\nb\nc\nd"  # top (k=0)
    assert _line_edit("a\nb\nc\nd", 5, 4, "X") == "a\nb\nc\nd\nX"  # append (k=4)
    assert _line_edit("a\nb\nc\nd", 3, 2, "X") == "a\nb\nX\nc\nd"  # middle (k=2)


def test_out_of_range_fails_soft() -> None:
    src = "a\nb\nc\nd"  # 4 lines
    assert _line_edit(src, 0, 2, "x") is None
    assert _line_edit(src, 2, 99, "x") is None
    assert _line_edit(src, 5, 3, "x") is None  # start>end, NOT the insert sentinel
    assert _line_edit(src, 0, -1, "x") is None  # insert_after(-1)
    assert _line_edit(src, 6, 5, "x") is None  # insert_after(5) on a 4-line file


def test_trailing_newline_line_count_matches_listing() -> None:
    # L1: the model's line model is split("\n"), so a trailing \n shows an empty last line;
    # the line-edit math must agree with what _number_lines displays.
    src = "a\nb\nc\n"
    assert src.split("\n") == ["a", "b", "c", ""]  # 4 lines, last empty
    assert _number_lines(src).count("\n") + 1 == 4
    # insert_after(4) appends after the empty trailing line; range 1..4 is valid.
    assert _line_edit(src, 5, 4, "X") == "a\nb\nc\n\nX"
    assert _line_edit(src, 4, 4, "z") == "a\nb\nc\nz"  # replace the empty trailing line


def test_applied_result_clean_compile() -> None:
    # A clean single-region apply (feature 020·29: no excerpt — the scratchpad shows the source).
    ok, msg, payload = _applied_result(EditResult(matches=1, errors=[]))
    assert ok is True
    assert "ok — compiled clean" in msg
    assert payload == {"errors": []}


def test_applied_result_multi_span_reports_count() -> None:
    # A multi-span replace_all reports the region count (D5).
    ok, msg, _payload = _applied_result(EditResult(matches=3, errors=[]))
    assert ok is True
    assert "3 regions changed" in msg


def test_applied_result_compile_errors_included() -> None:
    err = CompileErrorInfo(path="n.frag.glsl", line=4, message="boom")
    ok, msg, payload = _applied_result(EditResult(matches=1, errors=[err]))
    assert ok is True
    assert "compiled with errors" in msg and "boom" in msg
    assert payload == {"errors": [err.__dict__]}


# ---- tool-layer behavior through the loop ----
def _run(scripts: list[list[LLMStreamEvent]]) -> list:
    caps = _fake_caps(edit_errors=[[]] * 20)
    return list(
        run_turn(
            _FakeClient(scripts),
            build_registry(caps),
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="edit it",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )


def test_replace_lines_start_after_end_errors_not_inserts() -> None:
    # The B1 disambiguation: a user replace_lines(5,3) ERRORS at the tool layer (never
    # reaches the capability as a spurious insert).
    events = _run(
        [
            _tool_call(
                "c1",
                "replace_lines",
                '{"start_line": 5, "end_line": 3, "new_text": "x"}',
            ),
            [LLMTextDelta("done"), LLMDone("stop")],
        ]
    )
    card = next(e for e in events if isinstance(e, AgentToolCard))
    assert card.name == "replace_lines"
    assert card.ok is False


def test_insert_after_applies_via_same_capability() -> None:
    # insert_after(0) -> apply_line_edit(1, 0, ...) INSERTS (the empty-selection sentinel),
    # proving the same capability handles both replace and insert.
    events = _run(
        [
            _tool_call("c1", "insert_after", '{"line": 0, "new_text": "// header"}'),
            [LLMTextDelta("done"), LLMDone("stop")],
        ]
    )
    card = next(e for e in events if isinstance(e, AgentToolCard))
    assert card.name == "insert_after"
    assert card.ok is True


def test_retry_cap_fires_on_spiraling_replace_lines() -> None:
    # D6: the widened cap counts replace_lines failures too. An always-out-of-range range
    # must stop at max_edit_retries, not run to max_iterations.
    fail = _tool_call(
        "cx", "replace_lines", '{"start_line": 99, "end_line": 100, "new_text": "x"}'
    )
    scripts: list[list[LLMStreamEvent]] = [fail] * (COPILOT_CONFIG.max_iterations + 5)
    events = _run(scripts)
    failed = [e for e in events if isinstance(e, AgentToolCard) and not e.ok]
    assert len(failed) == COPILOT_CONFIG.max_edit_retries
    assert isinstance(events[-1], AgentError)
    assert len(failed) < COPILOT_CONFIG.max_iterations


# ---- whole-file mode + range checksums (034 wave: coordinates stop being guessed) ----


def test_range_check_passes_on_verbatim_quotes() -> None:
    lines = ["uniform float u;", "void main() {", "    x;", "}"]
    assert _range_check_error(lines, 2, 4, "void main() {", "}") == ""
    # Whitespace-stripped comparison: indentation differences don't reject.
    assert _range_check_error(lines, 3, 3, "x;", "x;") == ""


def test_range_check_names_the_actual_line_on_mismatch() -> None:
    lines = ["a", "float glow = 1.0;", "c", "}"]
    msg = _range_check_error(lines, 1, 2, "a", "}")
    assert "end_line 2" in msg
    assert 'is "float glow = 1.0;"' in msg
    assert 'you quoted "}"' in msg
    assert "nothing was applied" in msg


def test_range_check_skips_none_sides() -> None:
    lines = ["a", "b"]
    assert _range_check_error(lines, 1, 2, None, None) == ""
    assert "start_line 1" in _range_check_error(lines, 1, 2, "zzz", None)


def test_range_check_rejects_multiline_quote() -> None:
    lines = ["a", "    return vec2(mask, glow);", "}"]
    msg = _range_check_error(lines, 1, 2, "a", "    return vec2(mask, glow);\n}")
    assert "last_line must be exactly ONE line — you quoted 2 lines" in msg
    assert "nothing was applied" in msg


def test_range_check_suggests_unique_matching_line() -> None:
    lines = ["a", "b", "float glow = 1.0;", "d"]
    msg = _range_check_error(lines, 1, 2, "a", "float glow = 1.0;")
    assert "your quoted last_line matches line 3" in msg
    assert "did you mean end_line=3?" in msg


def test_range_check_no_suggestion_on_ambiguous_match() -> None:
    # A bare "}" matches many lines — no single-line suggestion, just the mismatch.
    lines = ["if (x) {", "}", "void main() {", "}", "float y;"]
    msg = _range_check_error(lines, 5, 5, "}", None)
    assert "start_line 5" in msg
    assert "did you mean" not in msg


def test_range_check_tail_range_suggests_whole_file() -> None:
    lines = ["void main() {", "    fs_color = c;", "}", ""]
    msg = _range_check_error(lines, 1, 3, "void main() {", "    fs_color = c;")
    assert "omit start_line/end_line entirely to replace the WHOLE file" in msg


def test_range_check_mid_file_has_no_whole_file_suggestion() -> None:
    lines = ["a", "b", "c", "d", "e"]
    msg = _range_check_error(lines, 1, 2, "a", "zzz")
    assert "WHOLE file" not in msg


def test_whole_file_replace_applies_without_range() -> None:
    events = _run(
        [
            _tool_call(
                "c1", "replace_lines", '{"new_text": "// rewritten whole file"}'
            ),
            [LLMTextDelta("done"), LLMDone("stop")],
        ]
    )
    card = next(e for e in events if isinstance(e, AgentToolCard))
    assert card.name == "replace_lines"
    assert card.ok is True


def test_ranged_replace_without_checksums_rejected() -> None:
    events = _run(
        [
            _tool_call(
                "c1",
                "replace_lines",
                '{"start_line": 1, "end_line": 2, "new_text": "x"}',
            ),
            [LLMTextDelta("done"), LLMDone("stop")],
        ]
    )
    card = next(e for e in events if isinstance(e, AgentToolCard))
    assert card.ok is False
    assert "first_line AND last_line" in card.result


def test_partial_range_rejected() -> None:
    events = _run(
        [
            _tool_call(
                "c1",
                "replace_lines",
                '{"start_line": 1, "new_text": "x", "first_line": "a", "last_line": "b"}',
            ),
            [LLMTextDelta("done"), LLMDone("stop")],
        ]
    )
    card = next(e for e in events if isinstance(e, AgentToolCard))
    assert card.ok is False
    assert "BOTH start_line and end_line" in card.result
