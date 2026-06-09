<!-- DOGFOOD REPORT TEMPLATE. analyze.py fills every {{AUTO:...}} slot from the run's trace + dump
     JSONs and LEAVES every {{HUMAN:...}} slot for you. Rule: countable/summable/table-able from a
     log => AUTO; requires opening a PNG or forming an opinion => HUMAN. Save the filled copy as
     ai_docs/features/NNN_dogfood_report_<run>.md (durable finding, roadmap-linked). -->

# Dogfood report — data-9b11k08b — gpt-5.1-codex-mini

- **Run:** data-9b11k08b · 2026-06-09
- **Scenario(s):** (unspecified)
- **Model:** openai/gpt-5.1-codex-mini
- **Turns:** 10 · **Total cost:** $0.0932

## 1. Verdict (HUMAN)

- **Mechanism works (pipeline end-to-end):** YES. Run #4 was the verification pass after the run-3 fixes (B1 set_uniform, B2 create_node resolver, the analyzer recovery/dump fixes, C1/C2 scenario wording), and it itself was then mega-review-swarmed: every AUTO number re-verified against raw logs, 2 analyzer bugs found + fixed (cross-tool recovery, per-run dump filter) and this report regenerated against the fixed analyzer. The full feature-027 loop ran clean across 10 turns; analyzer warnings are now just the benign model-id note.
- **Overall conclusion:** Strong, fixes confirmed live. set_uniform fired ONCE (run 3 looped it 9x) — B1 holds. create_node worked with NO template="" hint (run 3 hard-failed) — B2 holds. grep fired 3x (cold every prior run) after the C2 map-unanswerable content question — C2 holds. The agent self-corrected compile errors 5x this run, including CROSS-tool (create_node-broke -> replace_lines-clean), all caught by the fixed analyzer. Gallery is visually correct. One fix did NOT take: C1 switch_node — codex-mini structurally avoids it (self-targets via read_shader + targeted edit); it is effectively vestigial for this model (see §7). Coverage 9/12.

## 2. Per-turn (AUTO)

| # | Ask (head) | tools fired | result | peak ctx | billed in | cost |
|---|---|---|---|---|---|---|
| 1 | Create a node named Circle: a solid white filled | create_node, replace_lines, set_uniform | 🔴 | 6980 | 26983 | $0.0033 |
| 2 | Create a node named Square: solid white filled s | create_node | ✅ | 7318 | 14192 | $0.0019 |
| 3 | Create a node named Triangle: solid white equila | create_node, replace_lines, replace_lines, replace_lines | ⚠️ | 8515 | 38440 | $0.0235 |
| 4 | Create a node named Ring: white annulus, outer 0 | read_shader, insert_after, replace_lines, replace_lines, create_node, replace_lines | ⚠️ | 9402 | 58757 | $0.0211 |
| 5 | Also create a node named Hexagon: white filled r | read_shader, read_lib, create_node, read_shader | ✅ | 8771 | 41486 | $0.0089 |
| 6 | Actually drop the Hexagon node, four shapes is e | delete_node | ✅ | 8379 | 16471 | $0.0014 |
| 7 | Fresh start. First tell me which nodes call SB_a | grep, grep, read_shader, create_node, replace_lines | ⚠️ | 8679 | 45297 | $0.0175 |
| 8 | Switch over to the Triangle node and work there: | read_shader, replace_lines, replace_lines | ⚠️ | 8383 | 32394 | $0.0033 |
| 9 | On the Gallery node add a uniform u_brightness ( | read_shader, grep, insert_after, read_shader, replace_lines, set_uniform | ⚠️ | 9127 | 58925 | $0.0111 |
| 10 | Render the Gallery node to a PNG so I have it sa | render_image | ✅ | 8097 | 16122 | $0.0012 |
<!-- Turn | Ask | Tools fired | Result | peak ctx | billed in | cost -->

## 3. Per-render visual eyeball (HUMAN)

- `/home/akarnachev/src/shaderbox/scripts/dogfood/runs/proj-z6p1s54n/renders/Gallery_f6f4_0.png` (open with Read)

- **Eyeball verdicts:** Circle/square/triangle/ring each correct as built. Final Gallery CORRECT — 2x2 grid, right quadrants, ring reuses SB_aa_disc. Minor: the triangle sits a touch low in its cell (cosmetic). codex-mini handled the coordinate mapping correctly.

## 4. Tool coverage (AUTO)

| Tool | Used | Count |
|---|---|---|
| read_shader | ✅ | 7 |
| edit_shader | ❌ | 0 |
| replace_lines | ✅ | 11 |
| insert_after | ✅ | 2 |
| set_uniform | ✅ | 2 |
| create_node | ✅ | 6 |
| delete_node | ✅ | 1 |
| switch_node | ❌ | 0 |
| grep | ✅ | 3 |
| read_lib | ✅ | 1 |
| render_image | ✅ | 1 |
| render_video | ❌ | 0 |

**Coverage: 9/12 reachable tools**

**Cold tools this run:** edit_shader, switch_node, render_video

## 5. Token / cost mechanics (AUTO)

- **Per-turn context (peak iteration in=):** 6980-9402, peak on turn 4
- **Per-turn cost:** $0.0012-$0.0235, dearest turn 3
- **Token growth shape:** peak context 9402 tok at turn 4; series 6980 -> 7318 -> 8515 -> 9402 -> 8771 -> 8379 -> 8679 -> 8383 -> 9127 -> 8097
- **Recovery counts:** 5 compile-error recoveries; 1 errored-no-recovery; 0 GLError 1282; 6 total failed attempts
<!-- NOTE: per-turn billed input (turn_done in=) is the SUM of all iterations' inputs and is much
     larger than the context peak (a heavy multi-node-read turn can bill ~70k while its context peak
     is only ~10k) — that is the cost driver, the peak is the context-size driver. -->

## 6. Honesty / visual-blindness (HUMAN)

- **Honesty judgment:** Good. Consistent hedging ("confirm in the preview"), no hallucinated visual claim. The B1 fix visibly changed behavior: after set_uniform the agent saw the cached 0.6 in the working set and stopped, rather than re-issuing the call.

## 7. TODOs (HUMAN)

### (a) Improve the COPILOT / agent
1. **switch_node never fires — C1 wording did NOT fix it (behavioral, confirmed across runs 3-4).** Even on an explicit no-target "switch over to the Triangle and work there", codex-mini does `read_shader(triangle)` + `replace_lines target=<id>` and never calls `switch_node`. It has a working alternate path and always prefers it. Decision: treat switch_node as vestigial for this model — either drop it from the reachable-coverage target (so coverage isn't permanently capped at 11/12) or only probe it with a request that genuinely cannot be done by targeted edit. Not a bug.
2. **set_uniform loop — FIXED (B1), confirmed 1 call this run vs 9 in run 3.**
3. **create_node template footgun — FIXED (B2), confirmed: no hint needed, no RuntimeError.**
4. **grep — now fires (C2), confirmed 3x.**

### (b) Improve the DOGFOODING framework / harness / skill
1. **analyzer fixes confirmed + regression-tested.** Recovery detector now counts CROSS-tool self-corrections (5 caught in run 4 vs 3 before the fix); the dump cross-check filters to THIS run's dumps (was mixing run-3/run-4 render paths + costs); the noisy 45-vs-46 + dump warnings are gone; `_growth_shape` dropped its dead peak-drop wipe heuristic. `tests/test_dogfood_analyze.py` (4 tests) pins the recovery detector incl. the cross-tool case.
2. **render_video still cold** — scenario 01 never asks for animation; a future mission can add a "make a short animated version" move if render_video coverage matters.
3. **Harder missions next** — code-quality grading + token-overflow provocation, building on the obkatan 01_shape_gallery mechanism.

### (c) Improve the LIBRARY
1. **Ship `SB_aa_disc` / `SB_aa_box` / `SB_aa_triangle`** — across runs 3+4 the agent re-authored these every time. A shipped antialiased-2D-primitive lib would cut tokens, give real `read_lib` targets, and make composites cheaper. The agent's own `sd_box`/`sd_equilateral_triangle` (run 4) are good reference implementations to lift.
