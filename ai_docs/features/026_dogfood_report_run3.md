<!-- DOGFOOD REPORT TEMPLATE. analyze.py fills every {{AUTO:...}} slot from the run's trace + dump
     JSONs and LEAVES every {{HUMAN:...}} slot for you. Rule: countable/summable/table-able from a
     log => AUTO; requires opening a PNG or forming an opinion => HUMAN. Save the filled copy as
     ai_docs/features/NNN_dogfood_report_<run>.md (durable finding, roadmap-linked). -->

# Dogfood report — data-t00w5yil — gpt-5.1-codex-mini

- **Run:** data-t00w5yil · 2026-06-09
- **Scenario(s):** (unspecified)
- **Model:** openai/gpt-5.1-codex-mini
- **Turns:** 10 · **Total cost:** $0.0794

## 1. Verdict (HUMAN)

- **Mechanism works (pipeline end-to-end):** YES. The full feature-027 loop — interactive resume/dump per turn, `clear_context()` memory wipe, and the new `analyze.py` auto-rollup + template fill — all worked end-to-end across a 10-turn run. This report had its AUTO sections (§2/§4/§5) filled by `analyze.py` from logs (zero hand-summing); only the HUMAN sections below were typed. A review swarm re-verified every AUTO number against the raw transcripts: all cost/token/coverage figures CONFIRMED.
- **Overall conclusion:** A strong run. Scenario 01's whole point landed: after the context wipe, a fresh zero-memory agent LOCATED + read the four nodes AND the `SB_aa_disc` library helper from disk (read_shader x3 + read_lib) and reused them — the Gallery source calls `SB_aa_disc` by name and the grid is correct. Coverage jumped to 9/12 (run 2 was 4/12) — the §1a "force the cold tools" discipline works. codex-mini writes solid GLSL and self-corrects compile errors (3 recoveries). Two behavioral oddities (set_uniform spam at turn 9, switch_node dodged via targeted-edit) + 3 still-cold tools (edit_shader/grep/render_video). NOTE: the copilot fixes for the set_uniform loop (B1) + create_node template footgun (B2) + the scenario wording for switch_node/grep have since landed — run #4 exercises them.

## 2. Per-turn (AUTO)

| # | Ask (head) | tools fired | result | peak ctx | billed in | cost |
|---|---|---|---|---|---|---|
| 1 | Create a node named Circle: a solid white filled | create_node, replace_lines, replace_lines, replace_lines, set_uniform | ⚠️ | 7431 | 42125 | $0.0068 |
| 2 | Create a node named Square: solid white filled s | create_node, switch_node | ✅ | 7431 | 21683 | $0.0041 |
| 3 | Create a node named Triangle: solid white equila | create_node | ✅ | 7778 | 14951 | $0.0073 |
| 4 | Create a node named Ring: white annulus, outer r | read_shader, insert_after, replace_lines, create_node | ✅ | 8315 | 40080 | $0.0078 |
| 5 | Also create a node named Hexagon: a white filled | create_node, replace_lines, replace_lines, replace_lines | ⚠️ | 8605 | 41599 | $0.0077 |
| 6 | Actually drop the Hexagon node, four shapes is e | delete_node, read_shader, read_lib | ✅ | 8314 | 32514 | $0.0029 |
| 7 | This project has several shader nodes drawing sh | read_shader, read_shader, read_lib, read_shader, create_node, read_shader, replace_lines, replace_lines | ⚠️ | 9777 | 72361 | $0.0227 |
| 8 | Bump the Triangle node circumradius to 0.50 (edi | read_shader, replace_lines, read_lib | ✅ | 8252 | 32152 | $0.0034 |
| 9 | On the Gallery node add a uniform u_brightness ( | insert_after, replace_lines, insert_after, set_uniform, set_uniform, set_uniform, set_uniform, set_uniform, set_uniform, set_uniform, set_uniform, set_uniform | 🔴 | 9638 | 107261 | $0.0124 |
| 10 | Render the Gallery node to a PNG file so I have  | render_image, set_uniform, set_uniform, set_uniform | ✅ | 9176 | 45205 | $0.0042 |
<!-- Turn | Ask | Tools fired | Result | peak ctx | billed in | cost -->

## 3. Per-render visual eyeball (HUMAN)

- `/home/akarnachev/src/shaderbox/scripts/dogfood/runs/proj-euh05wsb/renders/Gallery_959b_0.png` (open with Read)

- **Eyeball verdicts:** Each primitive verified correct as built (round aspect-correct circle, clean square, equilateral triangle, annulus with a clear hole — all white-on-dark). The final Gallery is CORRECT this run — 2x2 grid with circle TL, square TR, triangle BL, ring BR, exactly as requested (run 2 scrambled the quadrants + mis-described them; codex-mini nailed the coordinate mapping this time). The ring visibly reuses the shared `SB_aa_disc` edge.

## 4. Tool coverage (AUTO)

| Tool | Used | Count |
|---|---|---|
| read_shader | ✅ | 7 |
| edit_shader | ❌ | 0 |
| replace_lines | ✅ | 11 |
| insert_after | ✅ | 3 |
| set_uniform | ✅ | 12 |
| create_node | ✅ | 6 |
| delete_node | ✅ | 1 |
| switch_node | ✅ | 1 |
| grep | ❌ | 0 |
| read_lib | ✅ | 3 |
| render_image | ✅ | 1 |
| render_video | ❌ | 0 |

**Coverage: 9/12 reachable tools**

**Cold tools this run:** edit_shader, grep, render_video

## 5. Token / cost mechanics (AUTO)

- **Per-turn context (peak iteration in=):** 7431-9777, peak on turn 7
- **Per-turn cost:** $0.0029-$0.0227, dearest turn 7
- **Token growth shape:** peak context 9777 tok at turn 7; series 7431 -> 7431 -> 7778 -> 8315 -> 8605 -> 8314 -> 9777 -> 8252 -> 9638 -> 9176
- **Recovery counts:** 3 compile-error recoveries; 1 errored-no-recovery; 0 GLError 1282; 5 total failed attempts
<!-- NOTE: per-turn billed input (turn_done in=) is the SUM of all iterations' inputs and is much
     larger than the context peak (e.g. 72k billed vs ~10k peak on turn 7, the 4-node read turn) —
     that is the cost driver, the peak is the context-size driver. -->

## 6. Honesty / visual-blindness (HUMAN)

- **Honesty judgment:** Good this run. The agent consistently hedged ("compiles clean — check the live preview to confirm") rather than asserting a visual outcome it couldn't see, obeying the "you cannot see the render" discipline. No hallucinated-success instance like run 2's confident-but-wrong Gallery layout description (and here the layout was actually right). The set_uniform spam (turn 9, 9 identical `u_brightness=0.6` calls) is a tool-loop oddity, not dishonesty — root-caused to `_format_uniforms` reading live `u.value` (overwritten each frame) instead of the cache, fixed in B1.

## 7. TODOs (HUMAN)

### (a) Improve the COPILOT / agent
1. **set_uniform loop (turn 9: 9 identical `u_brightness=0.6` calls) — FIXED (B1).** Root cause: `_format_uniforms` showed live `u.value`, which `Node.render()` overwrites every frame, so the agent read its just-set value back as stale and retried. Fix: read `node.uniform_values.get(name, u.value)` (the same cache `tabs/node.py` reads).
2. **create_node template footgun — FIXED (B2).** Without a `template=""` hint codex-mini passed a display NAME and `create_node` hard-failed. Fix: `_copilot_resolve_template_id` now also matches a template by display name (casefold, unambiguous), falling back after the handle/prefix checks.
3. **grep never fires — WONTFIX (engine), addressed in the scenario instead.** The agent behaves optimally: the always-present project map + lib catalogue answer "what exists" with zero tool calls, so nudging grep in the prompt would be cargo-cult. Scenario 01 now adds a MAP-UNANSWERABLE content question ("which nodes call `SB_aa_disc` / use `fwidth`?") so grep gets a real reason to fire.

### (b) Improve the DOGFOODING framework / harness / skill
1. **analyze.py recovery detector — FIXED + regression-tested.** It missed applies-with-errors->clean recoveries (keyed only on `ok=False`). Now treats `ok=True` + "compiled with errors" as a broke-state; detects the 3 real recoveries. `tests/test_dogfood_analyze.py` pins it (positive + negative + rollup).
2. **45-vs-46 mismatch — RESOLVED + warning made directional.** Not a parser bug: turn 9 hit max_iterations, so its last set_uniform executed (46 blocks) but was never replayed as a history echo (45 ids) — same root as the set_uniform spam. The warning now only fires on the genuine anomaly (echo without a block).
3. **dump cross-check noise — FIXED.** It warned on every wiped run (dumps are a per-process subset). Now gated on `dumps_are_full_series` (one monotonic dump per turn) — silent on multi-process/wiped runs.
4. **switch_node hard to force — FIXED (scenario wording).** The agent did a targeted `replace_lines target=<id>` instead of switching. Scenario 01 §6 now phrases it as a no-target "switch over to the Triangle and work there", the only path that forces `switch_node`.

### (c) Improve the LIBRARY
1. **A shipped `SB_aa_disc` / `SB_aa_box` / `SB_grid_cell` antialiased-primitive set would be genuinely useful** and would give the copilot real `read_lib`-able primitives (it had to author `SB_aa_disc` itself this run). A small shipped 2D-primitive library lowers the bar for composite shaders and makes the "reuse the helper" probe land on real shipped code, not agent-authored scaffolding.
