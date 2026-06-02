"""Slice-2 line-editing tools (feature 020 · 14).

Two layers, both GL-free:
- the pure line-edit math + apply-feedback helpers (app.py module functions);
- the tool-layer behavior (replace_lines / insert_after) driven through a fake App, incl.
  the start>end insert-vs-error disambiguation and the widened retry cap.
"""

import threading

from shaderbox.app import _changed_excerpt, _line_of_offset, _number_lines
from shaderbox.copilot.agent import AgentError, AgentToolCard, run_turn
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


def test_changed_excerpt_window() -> None:
    new_text = "l1\nl2\nl3\nl4\nl5\nl6\nl7"
    out = _changed_excerpt(new_text, (4, 4), context=1)
    # lines 3..5, 1-based, numbered against the new source.
    assert out == "3  l3\n4  l4\n5  l5"


def test_changed_excerpt_clamps_at_edges() -> None:
    out = _changed_excerpt("a\nb\nc", (1, 1), context=2)
    assert out == "1  a\n2  b\n3  c"


def test_line_of_offset() -> None:
    src = "a\nbb\nccc"
    assert _line_of_offset(src, 0) == 1
    assert _line_of_offset(src, 2) == 2  # first char of line 2
    assert _line_of_offset(src, len(src)) == 3


def _edit_shader_changed_range(
    new_text: str, start_off: int, new_str: str
) -> tuple[int, int]:
    # Mirrors _copilot_apply_shader_edit's single-span changed-range formula (the -1 keeps a
    # trailing newline in new_str off the next, unchanged line).
    end_off = start_off + max(len(new_str) - 1, 0)
    return (_line_of_offset(new_text, start_off), _line_of_offset(new_text, end_off))


def test_edit_shader_changed_range_no_trailing_newline() -> None:
    # src "line1\nline2\nline3", replace "line2" (off 6) with "LX" -> range stays on line 2.
    new_text = "line1\nLX\nline3"
    assert _edit_shader_changed_range(new_text, 6, "LX") == (2, 2)


def test_edit_shader_changed_range_trailing_newline_no_bleed() -> None:
    # The off-by-one guard: new_str ends in "\n" must NOT extend the range onto line 3.
    new_text = "line1\nLX\nline3"  # from replacing "line2\n" (off 6) with "LX\n"
    assert _edit_shader_changed_range(new_text, 6, "LX\n") == (2, 2)


def test_applied_result_appends_excerpt() -> None:
    # A clean apply with a changed_excerpt surfaces it in the tool message.
    ok, msg, payload = _applied_result(
        EditResult(matches=1, errors=[], changed_excerpt="2  LX", changed_range=(2, 2))
    )
    assert ok is True
    assert "ok — compiled clean" in msg
    assert "changed lines:\n2  LX" in msg
    assert payload == {"errors": []}


def test_applied_result_multi_span_reports_count() -> None:
    # A multi-span replace_all has no single excerpt -> the region count instead (D5).
    ok, msg, _payload = _applied_result(
        EditResult(matches=3, errors=[], changed_excerpt="", changed_range=None)
    )
    assert ok is True
    assert "3 regions changed" in msg


def test_applied_result_compile_errors_included() -> None:
    err = CompileErrorInfo(path="n.frag.glsl", line=4, message="boom")
    ok, msg, payload = _applied_result(
        EditResult(
            matches=1, errors=[err], changed_excerpt="4  bad", changed_range=(4, 4)
        )
    )
    assert ok is True
    assert "compiled with errors" in msg and "boom" in msg
    assert "changed lines:\n4  bad" in msg
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
                '{"start_line": 5, "end_line": 3, ' '"new_text": "x"}',
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
