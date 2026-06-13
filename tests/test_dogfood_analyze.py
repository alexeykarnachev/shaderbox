"""Regression tests for the dogfood run-analyzer's recovery detector (feature 027).

The detector must count an edit that APPLIES but compiles WITH errors (`ok: True` +
"compiled with errors" in the result — the `_applied_result` quirk) followed by a later clean
same-tool edit IN THE SAME TURN as a recovery. Run 3's analyzer first reported 0 recoveries for a
run that visibly self-corrected three times, because it only keyed on `ok: False`; this pins the fix.
"""

import textwrap
from pathlib import Path

from scripts.dogfood.analyze import analyze


def _write_trace(data_dir: Path, body: str) -> None:
    traces = data_dir / "copilot_traces"
    traces.mkdir(parents=True)
    (traces / "copilot_test_2026-01-01_00-00-00_000001.transcript").write_text(
        textwrap.dedent(body), encoding="utf-8"
    )


_RECOVERY_TRACE = """\
### turn_start  ·  2026-01-01T00:00:00.000
user_text: make it red

### llm_response  ·  2026-01-01T00:00:01.000
iteration: 1
finish_reason: tool_calls
usage: in=100 out=10 cost=$0.001000

    -> tool_call replace_lines(id=call_aaa1)

### tool_call  ·  2026-01-01T00:00:01.100
n: 1
name: replace_lines
ok: True
result: compiled with errors:

### llm_response  ·  2026-01-01T00:00:02.000
iteration: 2
finish_reason: tool_calls
usage: in=110 out=10 cost=$0.001100
    -> tool_call replace_lines(id=call_bbb2)

### tool_call  ·  2026-01-01T00:00:02.100
n: 2
name: replace_lines
ok: True
result: ok — compiled clean

### llm_response  ·  2026-01-01T00:00:03.000
iteration: 3
finish_reason: stop
usage: in=120 out=10 cost=$0.001200

### turn_done  ·  2026-01-01T00:00:03.100
iterations: 3
tool_calls: 2
reply: done
usage: in=330 out=30 cost=$0.003300
"""


_NO_RECOVERY_TRACE = """\
### turn_start  ·  2026-01-01T00:00:00.000
user_text: make it red

### llm_response  ·  2026-01-01T00:00:01.000
iteration: 1
finish_reason: tool_calls
usage: in=100 out=10 cost=$0.001000

### tool_call  ·  2026-01-01T00:00:01.100
n: 1
name: replace_lines
ok: True
result: ok — compiled clean

### llm_response  ·  2026-01-01T00:00:02.000
iteration: 2
finish_reason: stop
usage: in=110 out=10 cost=$0.001100

### turn_done  ·  2026-01-01T00:00:02.100
iterations: 2
tool_calls: 1
reply: done
usage: in=210 out=20 cost=$0.002100
"""


def test_applies_with_errors_then_clean_is_a_recovery(tmp_path: Path) -> None:
    data_dir = tmp_path / "data-x"
    _write_trace(data_dir, _RECOVERY_TRACE)
    an = analyze(data_dir, "")
    assert len(an.recoveries) == 1
    rec = an.recoveries[0]
    assert rec.tool == "replace_lines" and rec.fixer == "replace_lines"
    assert rec.turn_index == 1
    assert rec.error_attempt == 1 and rec.recovered_attempt == 2
    assert "compiled with errors" in rec.error_head.lower()
    assert an.turns[0].recovered is True and an.turns[0].errored is True


def test_clean_only_turn_has_no_recovery(tmp_path: Path) -> None:
    data_dir = tmp_path / "data-y"
    _write_trace(data_dir, _NO_RECOVERY_TRACE)
    an = analyze(data_dir, "")
    assert an.recoveries == []
    assert an.turns[0].recovered is False and an.turns[0].errored is False


def test_coverage_and_cost_roll_up(tmp_path: Path) -> None:
    data_dir = tmp_path / "data-z"
    _write_trace(data_dir, _RECOVERY_TRACE)
    an = analyze(data_dir, "")
    # Two deduped invocations of a HISTORICAL tool: counted + recognized (no
    # unknown-tool warning), but outside the current-registry coverage denominator.
    assert an.tool_counts.get("replace_lines") == 2
    assert "replace_lines" not in an.reachable_used
    assert not any("unknown tool" in w for w in an.warnings)
    assert abs(an.total_cost_usd - 0.0033) < 1e-6
    assert an.turns[0].billed_in_tokens == 330
    assert an.turns[0].peak_iter_in_tokens == 120


_CROSS_TOOL_RECOVERY_TRACE = """\
### turn_start  ·  2026-01-01T00:00:00.000
user_text: build a gallery

### llm_response  ·  2026-01-01T00:00:01.000
iteration: 1
finish_reason: tool_calls
usage: in=100 out=10 cost=$0.001000
    -> tool_call create_node(id=call_aaa1)

### tool_call  ·  2026-01-01T00:00:01.100
n: 1
name: create_node
ok: True
result: created node 'Gallery' — compiled with errors:

### llm_response  ·  2026-01-01T00:00:02.000
iteration: 2
finish_reason: stop
usage: in=110 out=10 cost=$0.001100
    -> tool_call replace_lines(id=call_bbb2)

### tool_call  ·  2026-01-01T00:00:02.100
n: 2
name: replace_lines
ok: True
result: ok — compiled clean

### turn_done  ·  2026-01-01T00:00:02.200
iterations: 2
tool_calls: 2
reply: done
usage: in=210 out=20 cost=$0.002100
"""


def test_cross_tool_recovery_is_counted(tmp_path: Path) -> None:
    # A broken create_node recovered by a DIFFERENT edit tool (replace_lines) is still a recovery —
    # the same-tool-only matcher missed this, the most common real self-correction (MUST-FIX 2).
    data_dir = tmp_path / "data-x2"
    _write_trace(data_dir, _CROSS_TOOL_RECOVERY_TRACE)
    an = analyze(data_dir, "")
    assert len(an.recoveries) == 1
    rec = an.recoveries[0]
    assert rec.tool == "create_node" and rec.fixer == "replace_lines"
    assert an.turns[0].recovered is True


# An edit-giveup terminal: no turn_done event, so cost/tokens must fall back to summed iterations.
_GIVEUP_TRACE = """\
### turn_start  ·  2026-01-01T00:00:00.000
user_text: tidy it

### llm_response  ·  2026-01-01T00:00:01.000
iteration: 1
finish_reason: tool_calls
usage: in=100 out=10 cost=$0.001000
    -> tool_call replace_lines(id=call_aaa1)

### tool_call  ·  2026-01-01T00:00:01.100
n: 1
name: replace_lines
ok: False
result: error: last_line must be exactly ONE line

### llm_response  ·  2026-01-01T00:00:02.000
iteration: 2
finish_reason: tool_calls
usage: in=110 out=20 cost=$0.002000
    -> tool_call replace_lines(id=call_bbb2)

### tool_call  ·  2026-01-01T00:00:02.100
n: 2
name: replace_lines
ok: False
result: error: last_line must be exactly ONE line

### edit_giveup  ·  2026-01-01T00:00:02.200
consecutive_failed_edits: 3
"""


def test_error_terminal_turn_costs_fall_back_to_iteration_sum(tmp_path: Path) -> None:
    # A giveup turn emits no turn_done, so the turn_done-keyed cost/token fields would stay 0 and
    # the rollup would understate spend. The fallback sums the per-iteration usage.
    data_dir = tmp_path / "data-giveup"
    _write_trace(data_dir, _GIVEUP_TRACE)
    an = analyze(data_dir, "")
    assert abs(an.turns[0].cost_usd - 0.003) < 1e-6
    assert an.turns[0].billed_in_tokens == 210
    assert an.turns[0].reply_tokens == 30
    assert an.turns[0].peak_iter_in_tokens == 110  # max per-iteration, not the sum
    assert abs(an.total_cost_usd - 0.003) < 1e-6


def test_trace_usage_line_round_trips_through_analyzer() -> None:
    # The trace's usage-line FORMAT and the analyzer's regex are one contract — render
    # with the real producer, parse with the real consumer (cache field included).
    from scripts.dogfood.analyze import _USAGE_RE
    from shaderbox.copilot.llm.api import LLMUsage
    from shaderbox.copilot.trace import _render_usage

    u = LLMUsage(
        input_tokens=100,
        output_tokens=20,
        reasoning_tokens=2,
        cached_tokens=64,
        cost_usd=0.001,
    )
    m = _USAGE_RE.match("usage: " + _render_usage(u))
    assert m is not None
    # groups: in, out, rsn, cache, cost
    assert (m.group(1), m.group(2), m.group(3), m.group(4)) == ("100", "20", "2", "64")
    assert float(m.group(5)) == 0.001
    total = u + LLMUsage(input_tokens=1, cached_tokens=6)
    assert total.cached_tokens == 70


_CACHED_TRACE = """\
### turn_start  ·  2026-01-01T00:00:00.000
user_text: tweak it

### llm_response  ·  2026-01-01T00:00:01.000
iteration: 1
finish_reason: stop
usage: in=100 out=10 rsn=0 cache=80 cost=$0.001000

### turn_done  ·  2026-01-01T00:00:02.000
iterations: 1
usage: in=100 out=10 cost=$0.001000
"""


def test_cache_share_parsed_and_reported(tmp_path: Path) -> None:
    from scripts.dogfood.analyze import _cache_share_line

    data_dir = tmp_path / "data-c"
    _write_trace(data_dir, _CACHED_TRACE)
    an = analyze(data_dir, "")
    assert an.turns[0].iterations[0].cached_tokens == 80
    assert "80/100" in _cache_share_line(an) and "80%" in _cache_share_line(an)


# A gate-declined call: the model EMITTED render_image (echoed in the next iteration's history) but
# the user declined, so execute() never ran and no ### tool_call block was written. Coverage must key
# on the execution block, so render_image must NOT count as used.
_DECLINED_TRACE = """\
### turn_start  ·  2026-01-01T00:00:00.000
model: x-ai/grok-4-fast
user_text: render it

### llm_response  ·  2026-01-01T00:00:01.000
iteration: 1
finish_reason: tool_calls
usage: in=100 out=10 cost=$0.001000
    -> tool_call render_image(id=call_aaa1)

### gate_open  ·  2026-01-01T00:00:01.050
name: render_image
prompt: Render the current node?

### gate_declined  ·  2026-01-01T00:00:01.100
name: render_image

### llm_response  ·  2026-01-01T00:00:02.000
iteration: 2
finish_reason: stop
usage: in=110 out=10 cost=$0.001100
    -> tool_call render_image(id=call_aaa1)

### turn_done  ·  2026-01-01T00:00:02.100
iterations: 2
tool_calls: 0
reply: ok, not rendered
usage: in=210 out=20 cost=$0.002100
"""


def test_declined_call_is_not_counted_as_used(tmp_path: Path) -> None:
    data_dir = tmp_path / "data-decl"
    _write_trace(data_dir, _DECLINED_TRACE)
    an = analyze(data_dir, "")
    # render_image was emitted + echoed but never executed -> not in coverage.
    assert "render_image" not in an.tool_counts
    assert "render_image" not in an.reachable_used
    assert an.turns[0].gate_declines == 1
    # the echo (1 deduped id) is fully explained by the 1 decline -> no drift warning.
    assert not any("invocation/block mismatch" in w for w in an.warnings)
    # model echo wins over the absent integrations.json default.
    assert an.model == "x-ai/grok-4-fast"


# An approved gate that executes: the ### tool_call block IS written, so it counts; the gate_approved
# event is recorded on the turn.
_APPROVED_TRACE = """\
### turn_start  ·  2026-01-01T00:00:00.000
model: x-ai/grok-4-fast
user_text: render it

### llm_response  ·  2026-01-01T00:00:01.000
iteration: 1
finish_reason: tool_calls
usage: in=100 out=10 cost=$0.001000
    -> tool_call render_image(id=call_aaa1)

### gate_open  ·  2026-01-01T00:00:01.050
name: render_image
prompt: Render the current node?

### gate_approved  ·  2026-01-01T00:00:01.080
name: render_image

### tool_call  ·  2026-01-01T00:00:01.100
n: 1
name: render_image
ok: True
result: ok — rendered 400x400

### llm_response  ·  2026-01-01T00:00:02.000
iteration: 2
finish_reason: stop
usage: in=110 out=10 cost=$0.001100

### turn_done  ·  2026-01-01T00:00:02.100
iterations: 2
tool_calls: 1
reply: rendered
usage: in=210 out=20 cost=$0.002100
"""


def test_approved_gate_executes_and_is_counted(tmp_path: Path) -> None:
    data_dir = tmp_path / "data-appr"
    _write_trace(data_dir, _APPROVED_TRACE)
    an = analyze(data_dir, "")
    assert an.tool_counts.get("render_image") == 1
    assert "render_image" in an.reachable_used
    assert an.turns[0].gate_approvals == 1
    assert an.turns[0].gate_declines == 0


# A probe turn: an attempt deliberately fails (bad node id) and the agent answers WITHOUT re-editing,
# ending on a clean turn_done. That is a PASS (✅), not a failed turn — the old glyph marked it 🔴.
_PROBE_TRACE = """\
### turn_start  ·  2026-01-01T00:00:00.000
user_text: edit node zzzz

### llm_response  ·  2026-01-01T00:00:01.000
iteration: 1
finish_reason: tool_calls
usage: in=100 out=10 cost=$0.001000
    -> tool_call edit_shader(id=call_aaa1)

### tool_call  ·  2026-01-01T00:00:01.100
n: 1
name: edit_shader
ok: False
result: error: no node with id 'zzzz'

### llm_response  ·  2026-01-01T00:00:02.000
iteration: 2
finish_reason: stop
usage: in=110 out=10 cost=$0.001100

### turn_done  ·  2026-01-01T00:00:02.100
iterations: 2
tool_calls: 1
reply: there is no node zzzz — which node did you mean?
usage: in=210 out=20 cost=$0.002100
"""


def test_handled_probe_failure_is_a_pass_not_red(tmp_path: Path) -> None:
    from scripts.dogfood.analyze import _result_glyph

    data_dir = tmp_path / "data-probe"
    _write_trace(data_dir, _PROBE_TRACE)
    an = analyze(data_dir, "")
    turn = an.turns[0]
    assert turn.terminal_kind == "turn_done"
    assert _result_glyph(turn) == "✅"  # clean terminal — NOT 🔴
    # the attempt still counts as errored-unrecovered-in-turn (no later clean edit), but that is an
    # attempt-level stat, not a turn failure.
    assert turn.errored is True and turn.recovered is False
    assert "0 failed turns" in an.recovery_summary


def test_giveup_terminal_is_red(tmp_path: Path) -> None:
    from scripts.dogfood.analyze import _result_glyph

    data_dir = tmp_path / "data-giveup-glyph"
    _write_trace(data_dir, _GIVEUP_TRACE)
    an = analyze(data_dir, "")
    assert an.turns[0].terminal_kind == "edit_giveup"
    assert _result_glyph(an.turns[0]) == "🔴"
    assert "1 failed turns" in an.recovery_summary


_REASONING_TRACE = """\
### turn_start  ·  2026-01-01T00:00:00.000
user_text: think hard

### llm_response  ·  2026-01-01T00:00:01.000
iteration: 1
finish_reason: stop
usage: in=100 out=200 rsn=180 cache=0 cost=$0.001000

### turn_done  ·  2026-01-01T00:00:02.000
iterations: 1
usage: in=100 out=200 cost=$0.001000
"""


def test_reasoning_share_parsed_and_reported(tmp_path: Path) -> None:
    from scripts.dogfood.analyze import _reasoning_share_line

    data_dir = tmp_path / "data-rsn"
    _write_trace(data_dir, _REASONING_TRACE)
    an = analyze(data_dir, "")
    assert an.turns[0].iterations[0].reasoning_tokens == 180
    line = _reasoning_share_line(an)
    assert "180/200" in line and "90%" in line


def test_trace_turn_start_and_gate_events_round_trip(tmp_path: Path) -> None:
    # The trace PRODUCER (TraceLog.event) and the analyzer PARSER are one contract for the new
    # turn_start `model` field and the gate_approved event — render with the real producer, parse
    # with the real consumer.
    from shaderbox.copilot.trace import TraceLog

    data_dir = tmp_path / "data-rt"
    traces = data_dir / "copilot_traces"
    traces.mkdir(parents=True)
    tr = TraceLog(traces / "copilot_test_2026-01-01_00-00-00_000001.transcript")
    tr.event(
        "turn_start",
        model="x-ai/grok-4-fast",
        user_text="go",
        history=[],
        eager_tools=[],
    )
    tr.event(
        "llm_response", iteration=1, finish_reason="tool_calls", text="", tool_calls=[]
    )
    # usage rendered by the producer is exercised elsewhere; emit a minimal usage line by hand-free
    # path: the analyzer reads iterations from llm_response sections, cost from the usage line.
    tr.event("gate_approved", name="render_image")
    tr.event(
        "tool_call",
        n=1,
        name="render_image",
        args={},
        ok=True,
        result="ok — rendered",
        payload={},
    )
    tr.event("turn_done", iterations=1, tool_calls=1, reply="done")
    tr.close()

    an = analyze(data_dir, "")
    assert an.model == "x-ai/grok-4-fast"
    assert an.turns[0].gate_approvals == 1
    assert an.tool_counts.get("render_image") == 1
