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
    # two deduped replace_lines invocations -> coverage 1/12, cost = turn_done rollup
    assert an.tool_counts.get("replace_lines") == 2
    assert "replace_lines" in an.reachable_used
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
