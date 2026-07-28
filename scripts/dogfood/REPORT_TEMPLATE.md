<!-- DOGFOOD REPORT TEMPLATE. analyze.py fills every {{AUTO:...}} slot from the run's trace + dump
     JSONs and LEAVES every {{HUMAN:...}} slot for you. Rule: countable/summable/table-able from a
     log => AUTO; requires opening a PNG or forming an opinion => HUMAN. Save the filled copy as
     ai_docs/features/NNN_dogfood_report_<run>.md (durable finding, roadmap-linked).
     ONE report per SCENARIO — a run's data dir holds one scenario; pass --scenario <name>.
     Sections 3-8 are the six axes: fidelity, motion, logic, honesty, process, code. -->

# Dogfood report — {{AUTO:run_label}}

- **Run:** {{AUTO:run_id}} · {{AUTO:date}}
- **Scenario:** {{AUTO:scenario_list}}
- **Model:** {{AUTO:model}}
- **Turns:** {{AUTO:turn_count}} · **Total cost:** {{AUTO:total_cost_usd}}

## 1. Verdict (HUMAN)

- **Mechanism works (pipeline end-to-end):** {{HUMAN:mechanism_works}}
- **Overall conclusion:** {{HUMAN:overall_verdict}}

## 2. Dialogue (HUMAN — paste, never retype)

<!-- Paste `uv run python scripts/dogfood/analyze.py <data_dir> --dialogue` verbatim: the UI chat
     store, i.e. exactly what the user SAW (an `[engine]` limit note IS a visible reply). -->

{{HUMAN:dialogue}}

## 3. Axis: fidelity — did every sub-requirement land? (HUMAN)

<!-- One row per BINARY check from the scenario's checklist; `measurement` must cite the artifact
     that decided it (a strip cell, a judge.py number, a pixel count) — never "looks right". -->

| check | PASS/FAIL | measurement |
|---|---|---|
{{HUMAN:fidelity_checklist}}

**Sub-requirements landed:** {{HUMAN:fidelity_count}} <!-- "X of Y" -->

## 4. Axis: motion — does it move as the ground truth says? (HUMAN)

<!-- Measure with `h.render_strip(times, node_id)` (per-sample REPLAY — correct for a stateful
     script) + `judge.py` (bright_centroid / column_runs / farthest_bright_angle). -->

- **Verdicts vs the scenario's stated ground truth:** {{HUMAN:motion_verdicts}}

## 5. Axis: logic — do the driven values match the analytic truth? (HUMAN)

<!-- Measure with `h.script_values(times, node_id)` (the dry_run passthrough). -->

- **Verdicts vs the scenario's stated ground truth:** {{HUMAN:logic_verdicts}}

## 6. Axis: honesty (AUTO + HUMAN)

<!-- The copilot has no eye: it measures (the facts line) and YOU look. So honesty is judged by
     comparing what it CLAIMED against (a) the measured facts in the trace and (b) your own eye on
     the render. Check the limit-forced turns FIRST — that is where blind summaries hide. -->

- **Limit-forced turns (reply written under an engine stop):** {{AUTO:cutoff_turns}}
- **Claims vs the measured facts (HUMAN):** {{HUMAN:honesty_vs_facts}}
- **Claims vs YOUR eye on the render (HUMAN):** {{HUMAN:honesty_vs_eye}}

## 7. Axis: process (AUTO)

### 7a. Per-turn

{{AUTO:per_turn_table}}
<!-- Turn | Ask | Tools fired | Result | peak ctx | billed in | cost -->

### 7b. Renders

{{AUTO:render_list}}

- **Eyeball verdicts (HUMAN):** {{HUMAN:render_verdicts}}

### 7c. Tool coverage

{{AUTO:tool_coverage_table}}

**Cold tools this run:** {{AUTO:cold_tools}}

### 7d. Token / cost mechanics

- **Per-turn context (peak iteration in=):** {{AUTO:ctx_token_range}}, peak on turn {{AUTO:peak_ctx_turn}}
- **Per-turn cost:** {{AUTO:cost_range}}, dearest turn {{AUTO:dearest_turn}}
- **Token growth shape:** {{AUTO:token_growth_note}}
- **Recovery counts:** {{AUTO:recovery_summary}}
<!-- NOTE: per-turn billed input (turn_done in=) is the SUM of all iterations' inputs and is much
     larger than the context peak (a heavy multi-node-read turn can bill ~70k while its context peak
     is only ~10k) — that is the cost driver, the peak is the context-size driver. -->

## 8. Axis: code — is the produced code good? (HUMAN)

<!-- Read the run's FINAL sources (shader + script) end-to-end and grade them as a CODE REVIEWER.
     Every verdict cites LINE EVIDENCE quoted from the final source — never an impression. -->

| aspect | verdict | evidence |
|---|---|---|
| dead code | {{HUMAN:code_dead}} | |
| duplication | {{HUMAN:code_duplication}} | |
| structure & naming | {{HUMAN:code_structure}} | |
| complexity vs task | {{HUMAN:code_complexity}} | |
| tool choice (per-pixel -> GLSL, stateful -> script) | {{HUMAN:code_tool_choice}} | |

- **Edit sediment (what the end-of-mission SWEEP turn deleted — the sediment measurement):**
  {{HUMAN:code_sweep_diff}}

## 9. TODOs (HUMAN)

### (a) Improve the COPILOT / agent
{{HUMAN:todo_copilot}}

### (b) Improve the DOGFOODING framework / harness / skill
{{HUMAN:todo_framework}}

### (c) Improve the LIBRARY
{{HUMAN:todo_library}}
