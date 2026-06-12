<!-- DOGFOOD REPORT TEMPLATE. analyze.py fills every {{AUTO:...}} slot from the run's trace + dump
     JSONs and LEAVES every {{HUMAN:...}} slot for you. Rule: countable/summable/table-able from a
     log => AUTO; requires opening a PNG or forming an opinion => HUMAN. Save the filled copy as
     ai_docs/features/NNN_dogfood_report_<run>.md (durable finding, roadmap-linked). -->

# Dogfood report — data-036anchor — gpt-5.1-codex-mini

- **Run:** data-036anchor · 2026-06-12
- **Scenario(s):** 01_shape_gallery, driven to deliberately PROVOKE feature 036 (anchored ranged
  `replace_lines`): multi-match `near_line`, reverse-order anchors, garbled/multi-line anchor rejects,
  the span echo, ranged edits in a large file. Plus the full tool sweep + context wipe.
- **Model:** openai/gpt-5.1-codex-mini
- **Turns:** 14 · **Total cost:** $0.1582

## 1. Verdict (HUMAN)

- **Mechanism works (pipeline end-to-end):** YES. Resume/dump across 14 separate processes, the
  context wipe + fresh-agent reconstruction, the worker+bridge pump, and every gated tool all worked.
  Zero GLError 1282, zero harness faults, zero un-recovered errors.
- **Overall conclusion:** **Feature 036 is robust and behaves exactly as specced under a real model.**
  Every 036 path fired naturally in the run and every reject message was actionable enough for
  codex-mini to self-correct in ONE step:
  - **multi-match → `near_line`** (turn 6): the agent quoted `last_line="}"`, which matched lines
    16/19/33; the engine rejected with `matches lines 16, 19, 33 — add near_line`; the agent re-issued
    the identical call + `near_line=33` and it applied (`replaced lines 1-33 — ok — compiled clean`).
    This is the exact off-by-one class 036 was built to kill, now resolved by text not numbers.
  - **reverse-order anchors** (turn 14, tool #11/#12): `first_line="void main() {"` resolved to line 31,
    `last_line="}"` to line 29 → `anchors in reverse order — first_line matched line 31, last_line
    matched line 29; first_line must be the block's FIRST line`. Names both resolved lines, per spec.
  - **span echo** worked on every applied ranged edit (`replaced lines 52-54`, `34-55`, `1-33`, …) —
    visible and correct, so a mislocated anchor would be caught immediately.
  - **ranged edit in a large file** (turn 12): a targeted `replace_lines` over the ring block of the
    ~60-line Gallery (`first_line`/`last_line` + `near_line=52`, `target='07e4'`) landed `replaced lines
    52-54` — the explicit "make a ranged edit, not a whole-file rewrite" instruction was honored.
  - **multi-line anchor reject** (`last_line must be exactly ONE line — you quoted N lines`) fired
    repeatedly (turns 3/8/14) and the agent always recovered — see TODO (a) for the behavioral note.
  - **whole-file mode** (anchors omitted) still works (turns 1, 8, 10 create_node bodies).
  No 036 defect found. The rough edges this run surfaced are all model behavior, not the feature.

## 2. Per-turn (AUTO)

| # | Ask (head) | tools fired | result | peak ctx | billed in | cost |
|---|---|---|---|---|---|---|
| 1 | Create a new node called Circle that draws a sol | create_node, replace_lines, edit_shader | ⚠️ | 9127 | 34914 | $0.0106 |
| 2 | Now create a separate node called Square that dr | create_node | ✅ | 9431 | 18339 | $0.0024 |
| 3 | Create a separate node called Triangle that draw | create_node, replace_lines, edit_shader | ⚠️ | 10248 | 39163 | $0.0106 |
| 4 | The triangle is rendering as a navy shape on an  | edit_shader | ✅ | 9975 | 19599 | $0.0044 |
| 5 | Still inverted: the triangle SHAPE is navy and t | edit_shader, edit_shader, edit_shader | ✅ | 10595 | 40932 | $0.0077 |
| 6 | The render facts have not changed across your la | replace_lines, replace_lines | ⚠️ | 11536 | 32740 | $0.0187 |
| 7 | Create a separate node called Ring that draws a  | create_node | ✅ | 10847 | 21596 | $0.0053 |
| 8 | Also make a quick Hexagon node so we have more o | create_node, replace_lines, replace_lines, replace_lines, replace_lines, replace_lines, replace_lines, replace_lines, replace_lines | ⚠️ | 14404 | 125238 | $0.0213 |
| 9 | Actually drop the Hexagon node - four shapes is  | delete_node | ✅ | 11765 | 23494 | $0.0024 |
| 10 | This project has several shader nodes each drawi | read_shader, create_node | ✅ | 10842 | 29018 | $0.0101 |
| 11 | Switch over to the standalone Triangle node and  | read_shader, edit_shader, read_shader, edit_shader, switch_node | ✅ | 11095 | 62193 | $0.0111 |
| 12 | Switch to the Gallery node. Give the RING shape  | read_shader, read_shader, edit_shader, replace_lines, set_uniform | ✅ | 11127 | 62938 | $0.0144 |
| 13 | Render the Gallery node to a 400x400 PNG file so | switch_node, render_image, replace_lines | ✅ | 10553 | 40840 | $0.0056 |
| 14 | Sweep the Gallery shader: remove any dead code,  | replace_lines, edit_shader, replace_lines, replace_lines, edit_shader, replace_lines, replace_lines, replace_lines, replace_lines, replace_lines, replace_lines, replace_lines | ⚠️ | 15538 | 165236 | $0.0338 |
<!-- Turn | Ask | Tools fired | Result | peak ctx | billed in | cost -->

## 3. Per-render visual eyeball (HUMAN)

- **Eyeball verdicts** (opened each PNG with Read):
  - **Circle** (`Circle_8e86_0`): PASS — round (aspect-corrected), radius ~0.4, light cyan on navy.
  - **Square** (`Square_82a5_0`): PASS — centered, square, half-side ~0.35.
  - **Triangle v0** (`Triangle_0b52_0`): FAIL — navy shape on ORANGE field (colors inverted); agent
    claimed "navy background". v1 (`_1`): still inverted (navy on cyan). v2 (`_2`): STILL inverted after
    3 edits — render facts never moved. v3 (`_3`): PASS — cyan triangle on navy, once the SDF sign was
    fixed (the real bug: `length(p)*sign(p.y)` is not negative-inside).
  - **Ring** (`Ring_7fcb_0`): PASS — cyan annulus with a visible hole, centered.
  - **Hexagon** (`Hexagon_7885_0`): PASS — six-sided cyan hexagon on navy (then deleted).
  - **Gallery** (`Gallery_07e4_0`): PASS — clean 2×2 grid, four distinct shapes in the right quadrants,
    reusing the exact source constants from the four nodes after a context wipe.
  - **Gallery + ring glow** (`Gallery_07e4_1` / `_3`): PASS — the ring visibly glows (`SB_glow` added by
    a targeted ranged edit); grid otherwise unchanged. Triangle a touch larger (radius 0.46 from turn 11).

## 4. Tool coverage (AUTO)

| Tool | Used | Count |
|---|---|---|
| read_shader | ✅ | 5 |
| edit_shader | ✅ | 11 |
| replace_lines | ✅ | 24 |
| insert_after | ❌ | 0 |
| set_uniform | ✅ | 1 |
| create_node | ✅ | 6 |
| delete_node | ✅ | 1 |
| switch_node | ✅ | 2 |
| grep | ❌ | 0 |
| read_lib | ❌ | 0 |
| render_image | ✅ | 1 |
| render_video | ❌ | 0 |

**Coverage: 8/12 reachable tools**

**Cold tools this run:** insert_after, grep, read_lib, render_video

## 5. Token / cost mechanics (AUTO)

- **Per-turn context (peak iteration in=):** 9127-15538, peak on turn 14
- **Per-turn cost:** $0.0024-$0.0338, dearest turn 14
- **Token growth shape:** peak context 15538 tok at turn 14; series 9127 -> 9431 -> 10248 -> 9975 -> 10595 -> 11536 -> 10847 -> 14404 -> 11765 -> 10842 -> 11095 -> 11127 -> 10553 -> 15538
- **Recovery counts:** 12 compile-error recoveries; 0 errored-no-recovery; 0 GLError 1282; 14 total failed attempts
<!-- NOTE: per-turn billed input (turn_done in=) is the SUM of all iterations' inputs and is much
     larger than the context peak (a heavy multi-node-read turn can bill ~70k while its context peak
     is only ~10k) — that is the cost driver, the peak is the context-size driver. -->

## 6. Honesty / visual-blindness (HUMAN)

- **Honesty judgment:** the hallucinated-success class fired hard on the Triangle (turns 4-5). The agent
  claimed "triangle now appears cyan on navy" THREE times while the render was unchanged and inverted —
  the render-facts line (`luma 7 7 7 / 7 4 7 / 7 7 7`, identical across all three edits) plainly said
  nothing changed (bright edges, dark center = the SHAPE is the dark one), but the agent never read the
  facts as a contradiction of its own claim. It only converged once *I* fed back "the facts have not
  changed across your last 3 edits" verbatim. This is the known render-blindness gap, NOT a 036 issue —
  036's own outputs (span echo, anchor rejects) were always accurate. Matches the existing `todo.md`
  "render-facts honesty: residual model-bound half" deferral; this run is another datum for it, not a
  new class. Elsewhere honesty was fine (it correctly reported "Triangle and Ring call SB_fill, Square
  does not" from reading the sources; never invented a render path; reported set_uniform value exactly).

## 7. TODOs (HUMAN)

### (a) Improve the COPILOT / agent
- **codex-mini repeatedly sends a MULTI-LINE `last_line` anchor** (rejected by 036 with "must be exactly
  ONE line — you quoted N lines"; fired on turns 3, 8, 14). It quotes the whole block as `last_line`
  instead of just the final line. It always recovers, so this is a model-competence flaw the vendor
  amortizes — per the "guard earns its place / better-model test" convention, do NOT add a prompt rule
  for it. But IF a future better model still does it, a one-line prompt nudge ("last_line is ONE line, not
  the block") would be the minimal fix. Logged as a datum, not an action.
- **Spree/thrash on sweep + hexagon** (turns 8, 14: 9 and 12 tool-calls, 125k/165k cumulative input,
  $0.02/$0.03). On turn 14 the agent issued the IDENTICAL `replace_lines` 3× in a row (#8/#9/#10, same
  args, same `replaced lines 34-55` result) without noticing the render facts were unchanged — it was
  "cleaning" indentation it had itself just written. The `consecutive_failed_edits` cap doesn't catch
  this because each call SUCCEEDS (ok=True). Possible angle: an oscillation/no-op-edit brake that counts
  consecutive CLEAN edits whose render facts don't move (the `max_clean_edit_streak` nudge exists but
  didn't fire here — re-check its threshold against this trace). Matches the existing giveup-counter
  deferral; add this trace as evidence.
- **set_uniform value is lost on resume** (turn 12 set `u_ring_glow=0.8`, but it's in-memory-only until a
  project save, so the next process renders the file default 0.4). Pre-existing/by-design (uniform values
  aren't persisted), but it makes the harness's between-process render under-represent a tuned look —
  more a dogfooding-framework note (b) than an agent bug.

### (b) Improve the DOGFOODING framework / harness / skill
- **`h.render()` always renders `current_node_id`**, and `create_node`/edit-by-target don't move current —
  so after building a node "in the background" the harness render shows the OLD current node (turn 2
  rendered Circle when I'd just made Square). I worked around it by poking `h.session.set_current_node_id`
  + re-rendering (no LLM). Worth a `h.render(node_id=...)` convenience param so a between-process eyeball
  doesn't need the current-pointer dance. **Add to the skill / harness.**
- **The analyzer's `result_glyph` marks a turn ⚠️ whenever ANY intermediate step failed**, even when the
  turn recovered and ended clean (turns 1/3/6/8/14 are all ⚠️ but every one ended compiled-clean). The
  glyph reads as "turn failed" — consider a separate "recovered" glyph vs a true terminal failure.
- **Skill gotcha confirmed:** `bash -ic` is mandatory for the key; `DogfoodHarness` exposes
  `_last_render_path` (underscore), not `last_render_path` — the skill's example text implies the latter.
  Minor; note it.

### (c) Improve the LIBRARY
- Nothing this run. `SB_fill` / `SB_glow` / `SB_center_uv` all behaved per their doc (negative-inside
  convention held); the only SDF bug was in the agent's hand-rolled `sd_equilateral_triangle`, not a lib
  helper. A possible nicety: ship `SB_sd_triangle` / `SB_sd_circle` / `SB_sd_box` primitives so the agent
  doesn't hand-roll (and mis-sign) SDFs — but that's a library-scope decision, not a finding.
