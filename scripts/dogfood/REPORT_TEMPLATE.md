<!-- DOGFOOD REPORT TEMPLATE. analyze.py fills every {{AUTO:...}} slot from the run's trace + dump
     JSONs and LEAVES every {{HUMAN:...}} slot for you. Rule: countable/summable/table-able from a
     log => AUTO; requires opening a PNG or forming an opinion => HUMAN. Save the filled copy as
     ai_docs/features/NNN_dogfood_report_<run>.md (durable finding, roadmap-linked).
     ONE report per SCENARIO — a run's data dir holds one scenario; pass --scenario <name>.
     Sections 3-7 are the five axes: fidelity, motion, logic, honesty, process. -->

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

{{AUTO:look_table}}

- **Per-turn FINAL verdict:** {{AUTO:final_verdicts}}
- **Parse rate (strict `ASK:` line / looks that saw a frame):** {{AUTO:parse_rate}}
- **Engine-look vision spend:** {{AUTO:engine_look_spend}}
  <!-- ENGINE looks only. A MODEL-initiated probe_render look emits no usage event, so its vision
       spend is absent by construction — a named copilot-wave gap, not a measurement error. -->
- **Limit-forced turns (the forced reply skips the eye):** {{AUTO:cutoff_turns}}
- **Over-claims / under-claims caught (HUMAN):** {{HUMAN:honesty_claims}}

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

## 8. TODOs (HUMAN)

### (a) Improve the COPILOT / agent
{{HUMAN:todo_copilot}}

### (b) Improve the DOGFOODING framework / harness / skill
{{HUMAN:todo_framework}}

### (c) Improve the LIBRARY
{{HUMAN:todo_library}}
