<!-- DOGFOOD REPORT TEMPLATE. analyze.py fills every {{AUTO:...}} slot from the run's trace + dump
     JSONs and LEAVES every {{HUMAN:...}} slot for you. Rule: countable/summable/table-able from a
     log => AUTO; requires opening a PNG or forming an opinion => HUMAN. Save the filled copy as
     ai_docs/features/NNN_dogfood_report_<run>.md (durable finding, roadmap-linked). -->

# Dogfood report — {{AUTO:run_label}}

- **Run:** {{AUTO:run_id}} · {{AUTO:date}}
- **Scenario(s):** {{AUTO:scenario_list}}
- **Model:** {{AUTO:model}}
- **Turns:** {{AUTO:turn_count}} · **Total cost:** {{AUTO:total_cost_usd}}

## 1. Verdict (HUMAN)

- **Mechanism works (pipeline end-to-end):** {{HUMAN:mechanism_works}}
- **Overall conclusion:** {{HUMAN:overall_verdict}}

## 2. Per-turn (AUTO)

{{AUTO:per_turn_table}}
<!-- Turn | Ask | Tools fired | Result | peak ctx | billed in | cost -->

## 3. Per-render visual eyeball (HUMAN)

{{AUTO:render_list}}

- **Eyeball verdicts:** {{HUMAN:render_verdicts}}

## 4. Tool coverage (AUTO)

{{AUTO:tool_coverage_table}}

**Cold tools this run:** {{AUTO:cold_tools}}

## 5. Token / cost mechanics (AUTO)

- **Per-turn context (peak iteration in=):** {{AUTO:ctx_token_range}}, peak on turn {{AUTO:peak_ctx_turn}}
- **Per-turn cost:** {{AUTO:cost_range}}, dearest turn {{AUTO:dearest_turn}}
- **Token growth shape:** {{AUTO:token_growth_note}}
- **Recovery counts:** {{AUTO:recovery_summary}}
<!-- NOTE: per-turn billed input (turn_done in=) is the SUM of all iterations' inputs and is much
     larger than the context peak (a heavy multi-node-read turn can bill ~70k while its context peak
     is only ~10k) — that is the cost driver, the peak is the context-size driver. -->

## 6. Honesty / visual-blindness (HUMAN)

- **Honesty judgment:** {{HUMAN:honesty_judgment}}

## 7. TODOs (HUMAN)

### (a) Improve the COPILOT / agent
{{HUMAN:todo_copilot}}

### (b) Improve the DOGFOODING framework / harness / skill
{{HUMAN:todo_framework}}

### (c) Improve the LIBRARY
{{HUMAN:todo_library}}
