"""Line-editing tools (features 020 · 14 + 036).

Two layers, both GL-free:
- the pure line-edit math + the 036 anchor resolution (backend module functions);
- the tool-layer behavior (replace_lines / insert_after) driven through a fake App, incl.
  the anchored ranged mode, the span echo, and the widened retry cap.
"""

import threading
import types
from collections.abc import Callable

from shaderbox.copilot.agent import AgentError, AgentToolCard, run_turn
from shaderbox.copilot.backend import (
    CopilotBackend,
    _CopilotEditTarget,
    _locate_anchor,
    _number_lines,
    _resolve_anchored_edit,
)
from shaderbox.copilot.capabilities import CompileErrorInfo, EditResult
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.gate import GateChannel
from shaderbox.copilot.llm.api import LLMDone, LLMStreamEvent, LLMTextDelta
from shaderbox.copilot.tools.registry import build_registry
from shaderbox.copilot.tools.shader import _applied_result
from tests.test_copilot_loop import _fake_caps, _fake_context, _FakeClient, _tool_call


# ---- the pure line-edit model (mirrors backend.py::_splice_lines + apply_line_edit's bounds) ----
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
def _run(scripts: list[list[LLMStreamEvent]], text: str = "vec3 p = u_pos;") -> list:
    caps = _fake_caps(edit_errors=[[]] * 20, text=text)
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
    # D6: the widened cap counts replace_lines failures too. An anchor that never matches
    # must stop at max_edit_retries, not run to max_iterations.
    fail = _tool_call(
        "cx",
        "replace_lines",
        '{"first_line": "no such line", "last_line": "no such line", "new_text": "x"}',
    )
    scripts: list[list[LLMStreamEvent]] = [fail] * (COPILOT_CONFIG.max_iterations + 5)
    events = _run(scripts)
    failed = [e for e in events if isinstance(e, AgentToolCard) and not e.ok]
    assert len(failed) == COPILOT_CONFIG.max_edit_retries
    assert isinstance(events[-1], AgentError)
    assert len(failed) < COPILOT_CONFIG.max_iterations


# ---- 036 anchor resolution (text locates, numbers are hints) ----


def test_anchor_unique_match_resolves() -> None:
    lines = ["uniform float u;", "void main() {", "    x;", "}"]
    assert _locate_anchor(lines, "void main() {", "first_line", None) == 2
    # Whitespace-stripped comparison: indentation differences still locate.
    assert _locate_anchor(lines, "x;", "first_line", None) == 3
    # near_line is consulted ONLY on a multi-match — a unique match ignores even a wild hint.
    assert _locate_anchor(lines, "void main() {", "first_line", 999) == 2


def test_anchor_garbled_quote_rejects_naming_the_anchor() -> None:
    lines = ["a", "float glow = 1.0;", "c"]
    msg = _locate_anchor(lines, "float glow = 2.0;", "last_line", None)
    assert isinstance(msg, str)
    assert "last_line" in msg
    assert "does not match any line" in msg
    assert "copy it verbatim from the working set" in msg


def test_anchor_blank_quote_rejects() -> None:
    lines = ["a", "", "c"]
    msg = _locate_anchor(lines, "   ", "first_line", None)
    assert isinstance(msg, str)
    assert "must be a content line" in msg


def test_anchor_multiline_quote_rejects() -> None:
    lines = ["a", "b", "c"]
    msg = _locate_anchor(lines, "a\nb", "first_line", None)
    assert isinstance(msg, str)
    assert "first_line must be exactly ONE line — you quoted 2 lines" in msg


def test_anchor_ambiguous_without_hint_lists_candidates() -> None:
    lines = ["if (x) {", "}", "void main() {", "}", "float y;"]
    msg = _locate_anchor(lines, "}", "last_line", None)
    assert isinstance(msg, str)
    assert "matches lines 2, 4" in msg
    assert "near_line" in msg


def test_anchor_ambiguous_near_line_picks_closest() -> None:
    lines = ["if (x) {", "}", "void main() {", "}", "float y;"]
    assert _locate_anchor(lines, "}", "last_line", 5) == 4
    assert _locate_anchor(lines, "}", "last_line", 1) == 2


def test_anchor_near_line_tie_rejects() -> None:
    lines = ["}", "x", "}"]
    msg = _locate_anchor(lines, "}", "first_line", 2)
    assert isinstance(msg, str)
    assert "equally near near_line 2" in msg


def test_anchored_edit_single_line_block() -> None:
    # first == last quoting the same unique line: a one-line block replaces in place.
    src = "a\nfloat glow = 1.0;\nc"
    resolved = _resolve_anchored_edit(
        src, "float glow = 1.0;", "float glow = 1.0;", None, "float glow = 2.0;"
    )
    assert resolved == ("a\nfloat glow = 2.0;\nc", "2-2")


def test_anchored_edit_reversed_anchors_reject() -> None:
    src = "a\nb\nc\nd"
    msg = _resolve_anchored_edit(src, "c", "b", None, "x")
    assert isinstance(msg, str)
    assert "anchors in reverse order" in msg
    assert "matched line 3" in msg and "matched line 2" in msg


def test_anchored_edit_deletes_block_on_empty_new_text() -> None:
    src = "a\nb\nc\nd"
    assert _resolve_anchored_edit(src, "b", "c", None, "") == ("a\nd", "2-3")


# ---- the two 2026-06-12 bundle-trace failures (the spec's regression fixtures) ----


def _trace_like_source() -> str:
    # Mirrors the failing trace's shape: the block sits on lines 37-43, line 44 is BLANK
    # (the +1-on-blank coordinate class), boundary-line text is unique.
    filler = [f"// filler {i}" for i in range(1, 37)]
    block = [
        "    vec3 base = mix(vec3(0.05, 0.2, 0.5), vec3(0.9, 0.6, 0.2), t);",
        "    float radius = length(uv);",
        "    vec3 color = base;",
        "    color += 0.1 * glow;",
        "    color = clamp(color, 0.0, 1.0);",
        "    color = pow(color, vec3(0.4545));",
        "    color *= 1.0 - 0.25 * pow(radius, 2.0);",
    ]
    tail = ["", "    fs_color = vec4(color, 1.0);", "}"]
    return "\n".join(filler + block + tail)


def test_trace_failure_1_off_by_one_now_applies() -> None:
    # The model quoted both boundary lines perfectly but sent end_line = correct + 1
    # (a blank line). With no numbers on the wire the text resolves 37..43 and applies.
    src = _trace_like_source()
    resolved = _resolve_anchored_edit(
        src,
        "    vec3 base = mix(vec3(0.05, 0.2, 0.5), vec3(0.9, 0.6, 0.2), t);",
        "    color *= 1.0 - 0.25 * pow(radius, 2.0);",
        None,
        "    vec3 color = vec3(0.0);",
    )
    assert not isinstance(resolved, str)
    new_full, span = resolved
    assert span == "37-43"
    lines = new_full.split("\n")
    assert lines[36] == "    vec3 color = vec3(0.0);"
    assert lines[37] == ""  # the blank line below the block survives
    assert lines[38] == "    fs_color = vec4(color, 1.0);"


def test_trace_failure_2_garbled_anchor_still_rejects() -> None:
    # The model corrupted the quote mid-string — intent is unverifiable, must reject.
    src = "void main() {\n    vec2 text_center_offset = vec2(0.0, 0.0);\n}"
    msg = _resolve_anchored_edit(
        src,
        "    vec2 text_center_offset = vecce of code?",
        "}",
        None,
        "x",
    )
    assert isinstance(msg, str)
    assert "first_line" in msg
    assert "does not match any line" in msg


# ---- the backend method's own branches (stubbed per the test_edit_messages idiom) ----


def _anchored_method(
    source: str, *, lib_create: bool = False
) -> Callable[[str, str, int | None, str, str], EditResult]:
    tgt = _CopilotEditTarget(
        kind="node",
        source=source,
        ws_address="n1",
        label="node 'X' (n1)",
        lib_create=lib_create,
    )
    stub = types.SimpleNamespace(
        _copilot_resolve_target=lambda _target, *, allow_create: tgt,
        _batch_mutated=set(),
        _bridge=types.SimpleNamespace(run_on_main=lambda fn: fn()),
        _copilot_persist_target=lambda _tgt, _new, matches: EditResult(
            matches=matches, errors=[]
        ),
    )
    return CopilotBackend.apply_anchored_edit.__get__(stub)


def test_apply_anchored_edit_empty_source_points_at_whole_file() -> None:
    res = _anchored_method("")("a", "b", None, "x", "")
    assert res.unresolved
    assert "omit first_line/last_line to write the whole file" in res.unresolved_reason
    assert res.target_label == "node 'X' (n1)"
    res = _anchored_method("a\nb", lib_create=True)("a", "b", None, "x", "lib:new.glsl")
    assert res.unresolved and "omit first_line/last_line" in res.unresolved_reason


def test_apply_anchored_edit_stamps_applied_span() -> None:
    res = _anchored_method("a\nb\nc")("a", "b", None, "x", "")
    assert res.applied_span == "1-2"


# ---- tool-layer ranged mode + whole-file mode ----


def test_anchored_replace_echoes_resolved_span() -> None:
    # The fake's source is the single line "vec3 p = u_pos;" — the success message
    # carries the resolved-span echo.
    events = _run(
        [
            _tool_call(
                "c1",
                "replace_lines",
                '{"first_line": "vec3 p = u_pos;", "last_line": "vec3 p = u_pos;", '
                '"new_text": "vec3 p = u_dir;"}',
            ),
            [LLMTextDelta("done"), LLMDone("stop")],
        ]
    )
    card = next(e for e in events if isinstance(e, AgentToolCard))
    assert card.name == "replace_lines"
    assert card.ok is True
    assert "replaced lines 1-1" in card.result


def test_applied_result_span_echo_leads_even_with_compile_errors() -> None:
    err = CompileErrorInfo(path="n.frag.glsl", line=4, message="boom")
    ok, msg, _ = _applied_result(
        EditResult(matches=1, errors=[err], applied_span="3-5")
    )
    assert ok is True
    assert msg.startswith("replaced lines 3-5 — ")
    assert "compiled with errors" in msg


def test_applied_result_span_echo_skipped_on_force_restore() -> None:
    ok, msg, _ = _applied_result(
        EditResult(
            matches=1, errors=[], restored_note="EDIT UNDONE", applied_span="3-5"
        )
    )
    assert ok is True
    assert "replaced lines" not in msg


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


def test_single_anchor_rejected() -> None:
    events = _run(
        [
            _tool_call(
                "c1",
                "replace_lines",
                '{"first_line": "vec3 p = u_pos;", "new_text": "x"}',
            ),
            [LLMTextDelta("done"), LLMDone("stop")],
        ]
    )
    card = next(e for e in events if isinstance(e, AgentToolCard))
    assert card.ok is False
    assert "BOTH first_line and last_line" in card.result


def test_reversed_anchors_rejected_through_the_loop() -> None:
    # Two-line fake source so the anchors resolve in reverse order; the reject names both.
    events = _run(
        [
            _tool_call(
                "c1",
                "replace_lines",
                '{"first_line": "second", "last_line": "first", "new_text": "x"}',
            ),
            [LLMTextDelta("done"), LLMDone("stop")],
        ],
        text="first\nsecond",
    )
    card = next(e for e in events if isinstance(e, AgentToolCard))
    assert card.name == "replace_lines"
    assert card.ok is False
    assert "anchors in reverse order" in card.result
