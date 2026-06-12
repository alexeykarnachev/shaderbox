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


def test_anchor_multiline_quote_resolves_to_boundary_line() -> None:
    # A multi-line quote is the model quoting a contiguous run of the block's lines; the boundary
    # is the OUTERMOST line (first_line -> first quoted line, last_line -> last quoted line).
    lines = ["a", "b", "c"]
    assert _locate_anchor(lines, "a\nb", "first_line", None) == 1
    assert _locate_anchor(lines, "a\nb", "last_line", None) == 2
    assert _locate_anchor(lines, "b\nc", "last_line", None) == 3


def test_anchor_multiline_quote_not_in_file_rejects() -> None:
    # The quoted run isn't in the source as written (stale/reversed) — reject, never fall back to
    # locating a bare boundary line somewhere else.
    lines = ["a", "b", "c"]
    msg = _locate_anchor(lines, "b\na", "first_line", None)
    assert isinstance(msg, str)
    assert "isn't in the file as quoted" in msg


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


def test_orphan_tail_absorbed_when_last_line_is_penultimate() -> None:
    # The model anchors last_line on the final STATEMENT, not the closing brace; new_text re-sends
    # the whole block. The duplicated "}" tail (a suffix of new_text) is absorbed into the range.
    source = "void main() {\n    x = 1.0;\n}\n"
    new_text = "void main() {\n    x = 2.0;\n}"
    result = _resolve_anchored_edit(source, "void main() {", "x = 1.0;", None, new_text)
    assert isinstance(result, tuple)
    new_full, span = result
    assert new_full.count("}") == 1
    assert span == "1-3"


def test_orphan_tail_absorbed_when_last_line_is_opening_brace() -> None:
    # The wide miss: last_line on the block's OPENING brace, new_text re-sends the whole block.
    # The entire original body+brace tail (a suffix of new_text) is absorbed.
    source = "void main() {\n    a;\n    fs_color = vec4(0.0);\n}\n"
    new_text = "void main() {\n    a;\n    fs_color = vec4(0.0);\n}"
    result = _resolve_anchored_edit(
        source, "void main() {", "void main() {", None, new_text
    )
    assert isinstance(result, tuple)
    new_full, span = result
    assert new_full.count("}") == 1
    assert span == "1-4"


def test_orphan_tail_absorbed_in_multi_function_shader() -> None:
    # The Grid-shape case (churn repro): helper functions before main, first_line=#version,
    # last_line on main's opening brace, new_text re-sends the whole shader.
    source = (
        "#version 460 core\nfloat grid() {\n    return 1.0;\n}\n"
        "void main() {\n    a;\n    fs_color = vec4(0.0);\n}\n"
    )
    new_text = (
        "#version 460 core\nfloat grid() {\n    return 1.0;\n}\n"
        "void main() {\n    a;\n    fs_color = vec4(0.0);\n}"
    )
    result = _resolve_anchored_edit(
        source, "#version 460 core", "void main() {", None, new_text
    )
    assert isinstance(result, tuple)
    new_full, _ = result
    assert new_full.count("{") == new_full.count("}")  # balanced, no orphan


def test_orphan_tail_not_absorbed_for_a_bare_statement_new_text() -> None:
    # new_text is a fragment INSIDE a function (no braces of its own), replacing a statement range.
    # The tail below (fs_color + the function "}") is NOT a duplicate of new_text — never absorbed.
    source = (
        "void main() {\n    vec3 base = mix(a, b, t);\n    color *= 1.0;\n"
        "    fs_color = vec4(color, 1.0);\n}\n"
    )
    new_text = "    vec3 color = vec3(0.0);"
    result = _resolve_anchored_edit(
        source, "    vec3 base = mix(a, b, t);", "    color *= 1.0;", None, new_text
    )
    assert isinstance(result, tuple)
    new_full, span = result
    assert span == "2-3"
    assert "fs_color = vec4(color, 1.0);" in new_full  # kept, not over-absorbed


def test_orphan_tail_does_not_eat_a_following_function() -> None:
    # Mis-anchor on foo's opening, but new_text only re-sends foo — bar() below must stay intact.
    source = "void foo() {\n    a;\n}\n\nvoid bar() {\n    b;\n}\n"
    new_text = "void foo() {\n    a;\n}"
    result = _resolve_anchored_edit(
        source, "void foo() {", "void foo() {", None, new_text
    )
    assert isinstance(result, tuple)
    new_full, _ = result
    assert "void bar()" in new_full
    assert new_full.count("{") == new_full.count("}")


def test_orphan_tail_not_absorbed_when_new_text_unbalanced() -> None:
    # A deliberate brace-changing edit (new_text leaves a block open) is left alone — the splice's
    # imbalance is the model's intent, not a duplicated tail.
    source = "if (a) {\n    x;\n}\nrest;\n"
    new_text = "if (a) {\n    y;"
    result = _resolve_anchored_edit(source, "if (a) {", "x;", None, new_text)
    assert isinstance(result, tuple)
    _, span = result
    assert span == "1-2"


def test_orphan_tail_not_absorbed_when_tail_diverges_rejects() -> None:
    # last_line on the opener (range = the opener line only, delta +1) with a balanced new_text whose
    # body DIFFERS from the tail below — absorb can't prove the tail is a duplicate (not a suffix), so
    # it doesn't extend. The range then opens a brace new_text re-supplies but doesn't account for the
    # original tail "}" — the brace-delta coherence guard rejects rather than apply a broken splice.
    source = "void foo() {\n    a;\n}\n"
    new_text = "void foo() {\n    DIFFERENT;\n}"
    result = _resolve_anchored_edit(
        source, "void foo() {", "void foo() {", None, new_text
    )
    assert isinstance(result, str)
    assert "new_text doesn't balance" in result


def test_orphan_tail_absorbed_when_tail_brace_is_re_indented() -> None:
    # The duplicated closing brace below the range is indented differently than new_text's brace.
    # A byte-exact suffix test would miss it and leave a doubled brace; the strip-invariant compare
    # (matching the anchor matcher) absorbs it.
    source = "void main() {\n    x = 1.0;\n    }\n"  # original close is indented
    new_text = "void main() {\n    x = 2.0;\n}"  # new close is flush
    result = _resolve_anchored_edit(source, "void main() {", "x = 1.0;", None, new_text)
    assert isinstance(result, tuple)
    new_full, span = result
    assert span == "1-3"
    assert new_full.count("}") == 1


def test_orphan_tail_not_absorbed_when_new_text_unbalanced_collides() -> None:
    # An unbalanced new_text (a deliberate structural change) must NEVER have a coincidental brace
    # below it eaten, even when that brace would rebalance the splice. The new_text-balance guard.
    source = "{\n{\n}\n"
    new_text = "{"  # unbalanced (+1)
    result = _resolve_anchored_edit(source, "{", "{", 1, new_text)
    assert isinstance(result, tuple)
    _, span = result
    assert span == "1-1"  # no absorb — the unbalanced new_text is left alone


def test_orphan_tail_absorbed_in_nested_block() -> None:
    # A re-sent block whose body itself nests braces (if/for) — the whole duplicated tail, inner
    # braces included, is absorbed when last_line anchors on the outer opening brace.
    source = (
        "void main() {\n    if (a) {\n        x;\n    }\n    fs_color = vec4(0.0);\n}\n"
    )
    new_text = (
        "void main() {\n    if (a) {\n        x;\n    }\n    fs_color = vec4(0.0);\n}"
    )
    result = _resolve_anchored_edit(
        source, "void main() {", "void main() {", None, new_text
    )
    assert isinstance(result, tuple)
    new_full, _ = result
    assert new_full.count("{") == new_full.count("}")  # balanced, no orphan


def test_orphan_tail_brace_in_comment_does_not_fool_balance() -> None:
    # A "}" inside a comment must not register as a real brace in the balance check.
    source = "void main() {\n    x = 1.0; // close }\n}\n"
    new_text = "void main() {\n    x = 2.0; // a }\n}"
    result = _resolve_anchored_edit(
        source, "void main() {", "x = 1.0; // close }", None, new_text
    )
    assert isinstance(result, tuple)
    new_full, _ = result
    assert new_full.count("}") == 2  # one real brace + one in the comment


# ---- multi-line boundary-anchor resolution (the 2026-06-12 T6 giveup fixture) ----


def test_multiline_last_line_resolves_to_closing_brace() -> None:
    # The T6 bug verbatim: last_line quotes the final statement + the closing brace as TWO lines;
    # new_text re-sends the whole block. The run "return total;\n}" locates the real block end (the
    # "}"), so the range ends there, the splice is balanced, and absorb NO-OPs.
    source = (
        "float fbm(vec2 p) {\n    float total = 0.0;\n    for (int i = 0; i < 6; ++i) {\n"
        "        total += noise(p);\n    }\n    return total;\n}\n"
    )
    new_text = (
        "float fbm(vec2 p) {\n    float total = 0.0;\n    for (int i = 0; i < 8; ++i) {\n"
        "        total += noise(p);\n        p = rotate2d(p, 0.3);\n    }\n    return total;\n}"
    )
    result = _resolve_anchored_edit(
        source, "float fbm(vec2 p) {", "    return total;\n}", 1, new_text
    )
    assert isinstance(result, tuple)
    new_full, span = result
    assert span == "1-7"  # last_line's last line ("}") is the block end at line 7
    assert new_full.count("{") == new_full.count("}")  # balanced — absorb no-ops
    assert "for (int i = 0; i < 8" in new_full


def test_multiline_last_line_partial_run_absorbs_tail() -> None:
    # The model under-quotes last_line (stops before the "}") as a multi-line run that DOES match
    # the source; new_text carries the full block, so absorb still swallows the duplicated "}".
    source = "void main() {\n    a;\n    b;\n    c;\n}\n"
    new_text = "void main() {\n    a;\n    b;\n    c;\n}"
    result = _resolve_anchored_edit(
        source, "void main() {", "    a;\n    b;", None, new_text
    )
    assert isinstance(result, tuple)
    new_full, span = result
    assert new_full.count("{") == new_full.count("}")
    assert span == "1-5"


def test_multiline_anchor_ambiguous_run_uses_near_line() -> None:
    # The same run appears twice; near_line disambiguates by the boundary line's position.
    source = "void f() {\n    x;\n}\nvoid g() {\n    x;\n}\n"
    msg = _resolve_anchored_edit(
        source, "void f() {", "    x;\n}", None, "void f() {\n    x;\n}"
    )
    assert isinstance(msg, str)
    assert "matches lines 3, 6" in msg  # the "}" boundary candidates listed
    result = _resolve_anchored_edit(
        source, "void f() {", "    x;\n}", 3, "void f() {\n    y;\n}"
    )
    assert isinstance(result, tuple)
    _, span = result
    assert span == "1-3"


def test_cross_block_span_rejected_not_silently_deleted() -> None:
    # The worst corner of multi-line anchoring: first_line in block f, last_line a run UNIQUE to a
    # LATER block g. The span would straddle both, a balanced new_text would clean-compile, and g
    # would be SILENTLY deleted. The straddle guard rejects instead of guessing.
    source = "void f() {\n    a();\n}\n\nvoid g() {\n    b();\n}\n\nvoid h() {\n    c();\n}\n"
    # last_line "    b();\n}" is unique (g's body+close) and lands the boundary in block g.
    msg = _resolve_anchored_edit(
        source, "void f() {", "    b();\n}", None, "void f() {\n    a2();\n}"
    )
    assert isinstance(msg, str)
    assert "DIFFERENT blocks" in msg


def test_cross_block_span_with_unbalanced_tail_also_rejected() -> None:
    # The straddle guard fires regardless of brace balance — a run ending on the NEXT block's
    # opening line straddles too.
    source = "void f() {\n    a();\n}\n\nvoid g() {\n    b();\n}\n"
    msg = _resolve_anchored_edit(
        source, "void f() {", "}\n\nvoid g() {", None, "void f() {\n    a2();\n}"
    )
    assert isinstance(msg, str)
    assert "DIFFERENT blocks" in msg


def test_multiblock_whole_file_resend_is_not_a_straddle() -> None:
    # A legit whole-file resend spans multiple blocks (first_line at file top, last_line on a later
    # block's opener) yet deletes nothing — new_text re-supplies every line the range covers. The
    # straddle guard must NOT reject it; only an UNRECOVERABLE multi-block span (new_text dropping
    # content) is the cross-block deletion corner.
    source = (
        "#version 460 core\nfloat grid() {\n    return 1.0;\n}\n"
        "void main() {\n    a;\n    fs_color = vec4(0.0);\n}\n"
    )
    new_text = (
        "#version 460 core\nfloat grid() {\n    return 2.0;\n}\n"
        "void main() {\n    a;\n    fs_color = vec4(0.0);\n}"
    )
    result = _resolve_anchored_edit(
        source, "#version 460 core", "void main() {", None, new_text
    )
    assert isinstance(result, tuple)
    new_full, _ = result
    assert new_full.count("{") == new_full.count("}")
    assert "return 2.0;" in new_full  # the edit landed
    assert "void main()" in new_full  # nothing deleted


def test_nested_block_range_is_not_a_straddle() -> None:
    # A legit edit of a function whose body nests (for/if) returns to depth 0 only ONCE, at the
    # function's own close — the straddle guard must NOT fire.
    source = (
        "float fbm(vec2 p) {\n    float t = 0.0;\n    for (int i = 0; i < 8; ++i) {\n"
        "        t += 1.0;\n    }\n    return t;\n}\n"
    )
    result = _resolve_anchored_edit(
        source,
        "float fbm(vec2 p) {",
        "    return t;\n}",
        None,
        "float fbm(vec2 p) {\n    return 9.0;\n}",
    )
    assert isinstance(result, tuple)
    new_full, _ = result
    assert new_full.count("{") == new_full.count("}")


# ---- bare-"}" last_line resolved by brace-matching the opener (the 2026-06-12 T6-rerun bug) ----


def _fbm_in_multi_function_source() -> str:
    # The live-trace shape: noise2d (closes line 8), then fbm (opens 11, NESTED for-loop closes 16,
    # fbm's own close 18), then a later function. A bare "}" last_line strip-matches lines 8, 16, 18.
    return (
        "float noise2d(vec2 p) {\n"  # 1
        "    vec2 i = floor(p);\n"  # 2
        "    vec2 f = fract(p);\n"  # 3
        "    return i.x + f.y;\n"  # 4
        "}\n"  # 5  (noise2d close)
        "\n"  # 6
        "\n"  # 7
        "// fbm\n"  # 8
        "// detail\n"  # 9
        "\n"  # 10
        "float fbm(vec2 p) {\n"  # 11  (opener)
        "    float total = 0.0;\n"  # 12
        "    for (int i = 0; i < 6; ++i) {\n"  # 13
        "        total += noise2d(p);\n"  # 14
        "        p *= 2.0;\n"  # 15
        "    }\n"  # 16  (NESTED for close)
        "    return total;\n"  # 17
        "}\n"  # 18  (fbm close)
        "\n"  # 19
        "float sd_circle(vec2 p, float r) {\n"  # 20
        "    return length(p) - r;\n"  # 21
        "}\n"  # 22
    )


def test_bare_brace_last_line_brace_matches_the_opener() -> None:
    # The T6-rerun bug: last_line="}" + near_line on the block START. near_line picks the PREVIOUS
    # function's close (line 5, |5-11|=6 < |18-11|=7) -> reverse-order reject. The fix brace-matches
    # fbm's opener and lands the real close (line 18), ignoring the misleading near_line.
    source = _fbm_in_multi_function_source()
    new_text = (
        "float fbm(vec2 p) {\n    float total = 0.0;\n"
        "    for (int i = 0; i < 8; ++i) {\n        total += noise2d(p);\n"
        "        p = p * 2.0 + 0.1;\n    }\n    return total;\n}"
    )
    result = _resolve_anchored_edit(source, "float fbm(vec2 p) {", "}", 11, new_text)
    assert isinstance(result, tuple)
    new_full, span = result
    assert (
        span == "11-18"
    )  # fbm's OWN close, not line 5 (prev fn) nor line 16 (nested for)
    assert new_full.count("{") == new_full.count("}")  # balanced; absorb no-ops
    assert "float sd_circle(vec2 p, float r) {" in new_full  # later fn intact
    assert "float noise2d(vec2 p) {" in new_full  # earlier fn intact
    assert "i < 8" in new_full  # the edit landed


def test_bare_brace_last_line_ignores_nested_inner_close() -> None:
    # near_line points INSIDE the block (past the nested for) — nearest-among-hits would pick the
    # nested for's "}" (line 16); brace-matching from the opener still resolves fbm's own close (18).
    source = _fbm_in_multi_function_source()
    new_text = "float fbm(vec2 p) {\n    return 0.0;\n}"
    result = _resolve_anchored_edit(source, "float fbm(vec2 p) {", "}", 14, new_text)
    assert isinstance(result, tuple)
    _, span = result
    assert span == "11-18"


def test_bare_brace_last_line_unique_close_defers_to_plain_path() -> None:
    # Only ONE "}" in the file: the brace-match disambiguator must NOT engage (nothing to
    # disambiguate); the plain single-match anchor path resolves it.
    source = "void main() {\n    x = 1.0;\n}\n"
    result = _resolve_anchored_edit(
        source, "void main() {", "}", None, "void main() {\n    x = 2.0;\n}"
    )
    assert isinstance(result, tuple)
    _, span = result
    assert span == "1-3"


def test_bare_brace_last_line_non_opener_first_line_defers() -> None:
    # first_line is NOT a brace opener (a plain statement), so brace-matching is impossible — the
    # bare "}" falls back to the near_line path and resolves the nearest closer as before. new_text
    # re-sends the closer so the block count is preserved (no straddle).
    source = "void f() {\n    a;\n}\nvoid g() {\n    b;\n}\n"
    # first_line="    a;" (line 2, not an opener); near_line=3 -> nearest "}" is line 3.
    result = _resolve_anchored_edit(source, "    a;", "}", 3, "    a2;\n}")
    assert isinstance(result, tuple)
    _, span = result
    assert span == "2-3"


def test_bare_brace_last_line_truncated_block_defers_then_reverse_rejects() -> None:
    # The opener's block never closes (a malformed/truncated source with no matching "}" forward) —
    # brace-match can't resolve, so it defers to the near_line path, which keeps its existing
    # behavior (here, the only "}" above start -> reverse-order reject).
    source = "}\nfloat fbm(vec2 p) {\n    return 0.0;\n"  # fbm never closes; one "}" at line 1
    msg = _resolve_anchored_edit(
        source, "float fbm(vec2 p) {", "}", 2, "float fbm(vec2 p) {\n    return 1.0;\n}"
    )
    assert isinstance(msg, str)
    assert "reverse order" in msg


def test_bare_brace_near_line_on_inner_close_partial_new_text_rejects() -> None:
    # Regression (the silent-corruption corner): first_line is the OUTER fbm opener, last_line a bare
    # "}", near_line points AT the nested for-loop's close (line 16), and new_text re-sends ONLY the
    # inner for-loop (a partial edit). Brace-matching to fbm's whole close (18) and splicing the
    # partial there would DELETE fbm's signature and "return total;", leaving a dangling top-level
    # for-block (brace-balanced, top-level count preserved -> straddle guard blind) and report SUCCESS.
    # near_line ON an inner closer defers to the near_line path, which stops at line 16; splicing 11-16
    # drops fbm's block -> the straddle guard rejects. A safe corrective reject, never a silent edit.
    source = _fbm_in_multi_function_source()
    partial = "    for (int i = 0; i < 8; ++i) {\n        total += noise2d(p);\n    }"
    msg = _resolve_anchored_edit(source, "float fbm(vec2 p) {", "}", 16, partial)
    assert isinstance(msg, str)
    assert "DIFFERENT blocks" in msg


def test_bare_brace_partial_inner_edit_via_inner_opener_brace_matches_for() -> None:
    # The supported partial inner-block edit: first_line is the INNER for opener, last_line a bare "}".
    # find_body_end from the for's opener brace-matches the for's OWN close (16), NOT fbm's close (18) —
    # so the partial edit applies to the loop only, fbm's body and signature untouched.
    source = _fbm_in_multi_function_source()
    new_text = (
        "    for (int i = 0; i < 8; ++i) {\n        total += noise2d(p);\n"
        "        p = p * 2.0 + 0.1;\n    }"
    )
    result = _resolve_anchored_edit(
        source, "    for (int i = 0; i < 6; ++i) {", "}", 13, new_text
    )
    assert isinstance(result, tuple)
    new_full, span = result
    assert span == "13-16"  # the for's own close, not fbm's close at 18
    assert "    return total;" in new_full  # fbm tail preserved
    assert "float fbm(vec2 p) {" in new_full  # fbm signature preserved
    assert "i < 8" in new_full


def test_range_opening_more_braces_than_new_text_rejects() -> None:
    # The brace-delta coherence guard catches the partial-corruption shape the straddle guard is
    # blind to: a single-function file where the range opens the fn + an inner block but last_line
    # lands on the inner close, and a partial new_text (one balanced inner block) would replace
    # signature+open — the spliced for-block REPLACES the fn as the lone top-level block (count
    # 1->1, straddle blind), brace-balanced, silently dropping the signature. The range/new_text
    # brace-delta mismatch (+1 vs 0) rejects it.
    source = (
        "float fbm(vec2 p) {\n    float total = 0.0;\n    for (int i = 0; i < 6; ++i) {\n"
        "        total += 1.0;\n    }\n    return total;\n}\n"
    )
    partial = "    for (int i = 0; i < 8; ++i) {\n        total += 2.0;\n    }"
    msg = _resolve_anchored_edit(source, "float fbm(vec2 p) {", "}", 5, partial)
    assert isinstance(msg, str)
    assert "new_text doesn't balance" in msg


def test_bare_brace_comment_brace_does_not_overshoot() -> None:
    # find_body_end counts braces raw; an unbalanced "{" in a body comment between the opener and its
    # real close would derail the depth walk and overshoot to a LATER function's "}". The brace-match
    # runs over comment-stripped lines, so the stray comment brace is invisible and the close resolves
    # to fbm's own "}" — a bare-"}" edit on a function whose comment carries a brace still succeeds.
    source = (
        "float fbm(vec2 p) {\n"  # 1
        "    // handle the { edge case\n"  # 2  (unbalanced brace in a comment)
        "    return 0.0;\n"  # 3
        "}\n"  # 4  (fbm's real close)
        "\n"  # 5
        "float tail() {\n"  # 6
        "    return 1.0;\n"  # 7
        "}\n"  # 8
    )
    new_text = "float fbm(vec2 p) {\n    // handle the { edge case\n    return 2.0;\n}"
    result = _resolve_anchored_edit(source, "float fbm(vec2 p) {", "}", 1, new_text)
    assert isinstance(result, tuple)
    new_full, span = result
    assert span == "1-4"  # fbm's own close, NOT tail's close at line 8
    assert "float tail() {" in new_full  # tail intact
    assert "return 2.0;" in new_full


def test_bare_brace_no_near_line_auto_resolves_block_close() -> None:
    # No near_line at all: the bare-"}" disambiguator still resolves via brace-matching (it needs no
    # hint), so the edit lands without the old "add near_line" reject. Strictly better than rejecting.
    source = _fbm_in_multi_function_source()
    new_text = "float fbm(vec2 p) {\n    return 0.0;\n}"
    result = _resolve_anchored_edit(source, "float fbm(vec2 p) {", "}", None, new_text)
    assert isinstance(result, tuple)
    _, span = result
    assert span == "11-18"
