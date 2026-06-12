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
    assert (m.group(1), m.group(2), m.group(3)) == ("100", "20", "64")
    assert float(m.group(4)) == 0.001
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
