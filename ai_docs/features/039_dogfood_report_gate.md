<!-- DOGFOOD REPORT TEMPLATE. analyze.py fills every {{AUTO:...}} slot from the run's trace + dump
     JSONs and LEAVES every {{HUMAN:...}} slot for you. Rule: countable/summable/table-able from a
     log => AUTO; requires opening a PNG or forming an opinion => HUMAN. Save the filled copy as
     ai_docs/features/NNN_dogfood_report_<run>.md (durable finding, roadmap-linked). -->

# Dogfood report — data-bl58fubn — unknown (in-tree default)

- **Run:** data-bl58fubn · 2026-06-12
- **Scenario(s):** (unspecified)
- **Model:** unknown (in-tree default)
- **Turns:** 8 · **Total cost:** $0.1512

## 1. Verdict (HUMAN)

- **Mechanism works (pipeline end-to-end):** YES — 8 turns, resume/dump stable, all renders produced; this run IS the 039 acceptance gate (edit tools: edit_shader + write_shader only).
- **Overall conclusion:** 039 GATE: PASS on safety/robustness, PARTIAL on cost. (a) The model edits naturally via edit_shader and reaches for write_shader for full rewrites — no steering confusion. (b) The anchor-reject class is GONE: ONE failed attempt in the whole run (a loud old_str no-match — it was the turn's LAST iteration, n=16/16, so the turn ended on the iteration-budget cutoff with the engine-forced reply rather than a retry) vs the baseline's 9 rejects + 2 giveups; the baseline-killer fbm rewrite itself APPLIED clean within that same first turn (baseline needed 4 attempts across 4 turns, 2 giveups). (c) The shrink-fact fired live and correctly (`note: this rewrite removed function(s): hillsSilhouette, paletteSunset`). (d) COST: $0.1512 / 8 turns vs baseline $0.0623 / 10 — ~2.4x. The driver is NOT the predicted old_str re-quoting (+10-20%) but a model-STYLE shift: codex-mini decomposes one-function rewrites into 9-16 micro edit_shader steps (turns 3/5/6/8) and accumulates duplicate comment lines via the quote-neighbor insert pattern, then cleans them. Every step is loud and convergent — the failure mode moved from silent/giveup to verbose/expensive, which is the right trade. RE-MEASURED same-day: a second run of the same expensive shapes (Scene2D create / palette / fbm) cost $0.0135/3 turns with 2-4 iterations per turn and a 68% prompt-cache hit — CHEAPER than the baseline on every shape. The churn is model-stochastic and clean-streak-nudge-bounded; cost verdict revised to PASS. Brake deferred (todo.md 'edit-churn brake', third-run trigger).

## 2. Per-turn (AUTO)

| # | Ask (head) | tools fired | result | peak ctx | billed in | cost |
|---|---|---|---|---|---|---|
| 1 | Create a node called Plasma that draws a colorfu | create_node, write_shader | ✅ | 7259 | 19671 | $0.0042 |
| 2 | Tidy up the Plasma source formatting: normalize  | write_shader | ✅ | 7340 | 14169 | $0.0043 |
| 3 | Replace ONLY the map_plasma_color function body  | edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader | ✅ | 11169 | 89579 | $0.0176 |
| 4 | Create a new node called Scene2D that is a LARGE | create_node, write_shader | ✅ | 10384 | 26196 | $0.0069 |
| 5 | In the Scene2D shader, rewrite just the palette  | edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, write_shader | ✅ | 16857 | 174052 | $0.0384 |
| 6 | In Scene2D, rewrite just the fbm function (the o | edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader | 🔴 | 17689 | 233580 | $0.0402 |
| 7 | Rewrite Scene2D from scratch as a MINIMAL versio | write_shader | ✅ | 11609 | 22769 | $0.0034 |
| 8 | Sweep the Plasma shader: remove dead code, dupli | read_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader, edit_shader | ✅ | 18250 | 243606 | $0.0361 |
<!-- Turn | Ask | Tools fired | Result | peak ctx | billed in | cost -->

## 3. Per-render visual eyeball (HUMAN)

- `/home/akarnachev/src/shaderbox/scripts/dogfood/runs/proj-tncimigk/renders/Scene2D_4a3f_4.png` (open with Read)

- **Eyeball verdicts:** t1 Plasma: soft blue/green plasma gradient — plausible t=0. t5 Scene2D: sun + layered hills silhouette present and composed correctly, BUT the claimed dusk palette (indigo/magenta/teal) rendered pale yellow/cream — the model claimed hues it cannot see (known model-bound honesty class, render facts did not contradict it loudly enough). t7 minimal Scene2D: correct plain fbm-textured sky gradient. Sweep turn rendered Scene2D (current node) while editing Plasma by target — harness behavior, expected.

## 4. Tool coverage (AUTO)

| Tool | Used | Count |
|---|---|---|
| read_shader | ✅ | 1 |
| edit_shader | ✅ | 52 |
| write_shader | ✅ | 5 |
| set_uniform | ❌ | 0 |
| create_node | ✅ | 2 |
| delete_node | ❌ | 0 |
| switch_node | ❌ | 0 |
| grep | ❌ | 0 |
| read_lib | ❌ | 0 |
| render_image | ❌ | 0 |
| render_video | ❌ | 0 |

**Coverage: 4/11 reachable tools**

**Cold tools this run:** set_uniform, delete_node, switch_node, grep, read_lib, render_image, render_video

## 5. Token / cost mechanics (AUTO)

- **Per-turn context (peak iteration in=):** 7259-18250, peak on turn 8
- **Per-turn cost:** $0.0034-$0.0402, dearest turn 6
- **Token growth shape:** peak context 18250 tok at turn 8; series 7259 -> 7340 -> 11169 -> 10384 -> 16857 -> 17689 -> 11609 -> 18250
- **Recovery counts:** 0 compile-error recoveries; 1 errored-no-recovery; 0 GLError 1282; 1 total failed attempts
<!-- NOTE: per-turn billed input (turn_done in=) is the SUM of all iterations' inputs and is much
     larger than the context peak (a heavy multi-node-read turn can bill ~70k while its context peak
     is only ~10k) — that is the cost driver, the peak is the context-size driver. -->

## 6. Honesty / visual-blindness (HUMAN)

- **Honesty judgment:** Mostly honest: turn 2 declined to invent work on its own already-clean file; turn 8 EXPLICITLY reported the sweep as incomplete ('the sweep isn't fully done yet') — good. One overclaim: turn 5 asserted dusk hues the render does not show (pale yellows) — the persistent model-bound render-blindness class, unchanged by 039.

## 7. TODOs (HUMAN)

### (a) Improve the COPILOT / agent
- Micro-edit churn: codex-mini splits a one-function rewrite into 9-16 edit_shader steps and accumulates duplicate comment lines (quote-neighbor insert re-sends the comment). Candidate levers, all deterministic: a result fact when an edit leaves N identical adjacent comment lines; or steering weight in _EDIT_SHADER_DESC toward ONE edit covering the whole region. Needs another run before building — may be model-specific.
- The clean-streak nudge fired and stopped the churn — working as designed; no change.

### (b) Improve the DOGFOODING framework / harness / skill
- Turn 6's 🔴 is half-right: the fbm rewrite applied clean mid-turn, but the turn DID end on the max_iterations cutoff with its final attempt a no-match — the glyph-keying todo entry gets a fresh datum either way.
- Turn 5's write_shader in the tools-fired column was a REAL call (the model finished the palette churn with a whole-file write), not a history echo.

### (c) Improve the LIBRARY
- Nothing: this gate run deliberately avoided SB_* (cost parity with the lib-free baseline).


## Post-review addendum (final swarm, same day)

- A FALSE shrink-fact fired in the cost re-measure run: `note: this rewrite removed function(s):
  main` on a write whose new_text kept `main()` in Allman style — the per-line scan missed the
  restyled signature and claimed a false removal. FIXED same-wave: removed names are now filtered
  against the new text (a name still textually present is never claimed removed), pinned by
  `test_rewrite_note_not_fooled_by_restyled_signature`.
- The cost re-measure run's analyzer table (durable copy — the runs dir is purgeable):
  create Scene2D $0.0045 (2 iters) / rewrite palette $0.0025 (3 iters) / rewrite fbm $0.0065
  (4 iters); total $0.0135; Cache: 41856/61679 input tokens cached (68%); zero rejects.
