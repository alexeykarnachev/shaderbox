# Dogfood report — run 2 (scenario 01 shape gallery, interactive resume/dump, codex-mini)

First run through the feature-027 interactive RESUME/DUMP flow (no server; one blocking `uv run` per
turn, state on disk between turns) and the FIRST run on the new default model
`openai/gpt-5.1-codex-mini`. Scenario: `scripts/dogfood/scenarios/01_shape_gallery.md` — build four 2D
shape nodes (circle/square/triangle/ring), wipe the agent's memory, then make a FRESH agent read them
from disk and combine into a 2×2 `Gallery`. Driven live, turn by turn, by Claude (me) — not scripted.

## 1. Bottom line

**The pipeline works end-to-end, and the interactive resume/dump + context-wipe mechanism is proven.**
Five nodes built, each shape visually correct; the memory-wiped fresh agent read all four sources from
disk and composed a 2×2 grid that compiled and rendered four distinct shapes. codex-mini writes
**competent GLSL** (every individual shape was correct first or second try) — a clear win over grok, which
the maintainer reports can't write a shader. Total cost: **$0.0487** for the whole 6-turn scenario. Two
real findings surfaced: a `create_node` template footgun (cost a wasted turn) and a 2×2 quadrant
mis-layout the agent confidently mis-described (visual blindness).

## 2. Per-turn summary

| Turn | Ask | Result | ctx tok | cost |
|---|---|---|---|---|
| 1 | Circle | 🔴 `create_node` failed — agent invented `template='UV Mango'` (a display name, not a `template:` handle); gave up + asked | 6397 | $0.0038 |
| 2 | (hint: `template=""`) Circle | ✅ created from starter; replace_lines errored once → read → fixed → clean; round circle | 6517 | $0.0025 |
| 3 | Square | ✅ clean; aspect-correct square | 7081 | $0.0056 |
| 4 | Triangle | ✅ edit errored once → fixed → clean; clean equilateral triangle | 7235 | $0.0183 |
| 5 | Ring + "what nodes exist?" | ✅ answered from project map (no list tool); clean annulus | 7803 | $0.0030 |
| 6 | **context wipe** → read 4 + build Gallery | ✅ read_shader ×4 → create_node → insert_after → replace_lines; 2×2 grid, 4 shapes; **but quadrants mis-placed** | 6757→`total_in=68311` | $0.0155 |

## 3. Pipeline mechanics that WORK (trace-confirmed)

- **Interactive resume/dump across separate processes.** Each turn was its own `uv run`; `dump` persisted
  the conversation + app_state, the next `create(project_dir=…)` resumed it with zero LLM calls. State
  (conversation, nodes, current-node) survived every process boundary. The `bash -ic` key-from-`~/.bashrc`
  path worked once the non-interactive-guard footgun was documented.
- **Context wipe (`clear_context` → `ProjectSession.clear_conversation`).** Turn 6: 23 messages / 10
  history BEFORE → **0 / 0** AFTER. The old conversation archived (recoverable), the nodes on disk
  untouched. The fresh agent had genuinely nothing in memory.
- **Tool-use under the wipe was LOAD-BEARING, not cosmetic.** The wiped agent called `read_shader` on all
  four nodes BEFORE writing Gallery, and the Gallery source reused each shape's exact size constant
  (circle 0.4, square 0.35, triangle 0.35, ring 0.25/0.4) — proof it read the numbers off disk rather than
  re-deriving from its weights. This is the headline result: the context-wipe + read-from-disk loop is the
  real tool-use test, and codex-mini passed it.
- **Agent-level error recovery.** Three separate compile errors during the shape builds (turns 2, 4, plus
  the Gallery), each read and fixed in one follow-up edit → clean. No thrash, no max_iterations.
- **Visual honesty (mostly good).** Every turn the agent hedged ("compiles clean — please check the live
  preview to confirm"), obeying the "you cannot see the render" discipline — EXCEPT it over-claimed the
  Gallery LAYOUT (see finding 5.2).

## 4. Tool coverage

Tracked across the run (real invocations, from the live drive log — NOT the per-iteration `tools=` block
which re-ships all 14 names every step):

| Tool | Used? | Notes |
|---|---|---|
| `create_node` | ✅ | every shape (5×) |
| `read_shader` | ✅ | 4× post-wipe (the core probe) + Gallery self-read |
| `replace_lines` | ✅ | shape-body edits |
| `insert_after` | ✅ | Gallery helper |
| `edit_shader` | ✅ | (surfaced in status cards) |
| `grep` | ❌ | the project map answered "what nodes exist?" so it never grepped |
| `read_lib` | ❌ | no library functions involved |
| `set_uniform` | ❌ | shapes hard-coded constants, no uniforms added |
| `switch_node` | ❌ | resume/wipe set current; never explicitly switched |
| `delete_node` | ❌ | nothing deleted |
| `render_image/video`, `publish_*` | ❌ | render is harness-side; publish precheck-fails (empty ExporterRegistry) |

**Coverage is THIN — only the create/read/edit cluster fired.** The whole navigation/value/integration
half (`grep`, `read_lib`, `set_uniform`, `switch_node`, `delete_node`) was never exercised. Per the
maintainer's note, this is itself a finding: scenario 01 doesn't provoke those tools. **Follow-up: future
scenarios must DELIBERATELY route through the cold tools** (a uniform to tune → `set_uniform`; a lib
function to reuse → `read_lib`+`grep`; a node to remove → `delete_node`; a multi-node project where the
target isn't current → `switch_node`).

## 5. Findings

### 5.1 🔴 `create_node` template footgun (cost a wasted turn)
On the empty project the agent called `create_node(template='UV Mango')` — a display NAME copied from the
template catalogue (`template:5372 | UV Mango`) instead of the handle `template:5372` or an empty string.
`_copilot_resolve_template_id` returned None → `RuntimeError: no template matching 'UV Mango'` → the tool
failed twice and the agent gave up and asked. The catalogue shows `template:<handle> | <name>`, and the
agent grabbed the human half. **Honest fix (copilot-side, NOT done here):** either (a) make
`_copilot_resolve_template_id` also match a template by display NAME (forgiving resolve), or (b) sharpen the
`create_node` tool description — "template = a `template:` handle or empty for the starter; NOT a display
name." (a) is more robust (matches how a human reads the catalogue). File as a copilot deferral.

### 5.2 🔴 2×2 quadrant mis-layout + confident mis-description (visual blindness)
The Gallery rendered four correct, distinct shapes in a clean 2×2 grid — but in the WRONG quadrants.
Requested: circle TL, square TR, triangle BL, ring BR. Rendered: triangle TL, ring TR, circle BL, square
BR. The agent's prose CONFIDENTLY said "circle top-left, square top-right, triangle bottom-left, ring
bottom-right" — matching the REQUEST, not the actual render. It cannot see the image, so it described what
it intended, and the GLSL coordinate mapping (almost certainly a flipped Y / cell-index→shape mapping)
disagreed. **This is the canonical visual-blindness class:** clean compile + confident layout claim +
wrong pixels. It's exactly what a machine-readable `inspect_render` affordance (the standing `todo.md`
deferral) would catch — a coarse "which quadrant has ink" probe would have let the agent self-correct.

### 5.3 ✅ codex-mini is the right model
Competent GLSL on every shape (round aspect-corrected circle, clean square, equilateral triangle, proper
annulus), good error recovery, good honesty discipline. Cheap ($0.0487 for 6 turns incl. a 4-node read).
The model swap (decision 9) is validated.

## 6. 💲 Cost
**$0.0487 total** for the full scenario (6 turns, 5 nodes built + 1 composite + a context wipe). Per-turn
$0.0025–$0.0183; the dearest were the ones with long replies (triangle math, Gallery composite). The
4-node read turn hit `total_in=68311` input tokens vs ~7k for single-node turns — a ~10× context jump when
four shaders enter the working set at once. Nowhere near codex-mini's 400k ceiling (no overflow risk at
this scale), but it's the baseline growth shape for the later token-overflow scenario.

## 7. Follow-up TODOs

### Improve the COPILOT (agent / engine)
1. **`create_node` template resolve should accept a display name** (or the tool description must forbid it
   explicitly). Today a name silently fails → wasted turn. (finding 5.1) — file as a copilot deferral.
2. **`inspect_render` machine-readable feedback** (standing deferral, now re-confirmed live by 5.2): a
   coarse per-region ink/luma probe so the agent can detect "my layout didn't land" instead of confidently
   mis-describing it. The 2×2 mis-layout is a clean motivating case.
3. **Layout/coordinate guidance in the prompt?** The agent mapped grid cells to the wrong shapes — worth
   checking whether a one-line "screen Y is [convention]" note in the prompt would cut coordinate-flip
   errors. Trace-gated: only if it recurs.

### Improve the DOGFOODING (framework / harness / skill)
4. **Tool coverage is a first-class metric now** (maintainer's ask). Only 5 of 14 tools fired. Future
   scenarios must deliberately provoke the cold half (`grep` / `read_lib` / `set_uniform` / `switch_node`
   / `delete_node`). Consider a coverage checklist in the report template + a `/dogfood` skill note.
5. **A `context_breakdown` / per-section token event** would turn the "68311 input tok on a 4-node read"
   observation into a real breakdown (system prompt vs project map vs working set vs `tools=` block).
   Standing deferral; the 4-node read makes it concrete.
6. **Scenario 01 should pre-seed the `create_node` hint** OR we leave it un-hinted to keep re-catching
   finding 5.1 — decide once 5.1 is fixed. (Right now I hinted `template=""` after turn 1 to not burn the
   whole run on it.)

### Improve the LIBRARY overall
7. **The grid mis-layout suggests a `Gallery`/layout helper or a `SB_grid_cell(uv, col, row, cols, rows)`
   library function** would be genuinely useful for composites — and would also give the copilot a
   `read_lib`-able primitive (helping coverage #4). Speculative; only if composite scenes become common.

## 8. Mechanism verdict
The feature-027 interactive resume/dump + context-wipe loop is **proven and pleasant to drive** — one
blocking call per turn, read the JSON, compose the next message, eyeball the PNG. The obkatka goal is met.
Ready for harder scenarios (code-quality grading, token-overflow provocation, multi-node targeting) that
also fix the thin tool coverage.
