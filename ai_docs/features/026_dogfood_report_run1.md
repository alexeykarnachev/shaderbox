# Dogfood report — first live run (2026-06-09)

First real end-to-end dogfood of the copilot ENGINE, headless on the Pi via `scripts/dogfood.py`
(feature 026). Model: **`x-ai/grok-4.3`** (the configured `x-ai/grok-4-fast` is DEPRECATED — 404 from
OpenRouter; first finding). 5 scenarios, 16 turns, **$0.207 total**. All renders eyeballed.

**Bottom line: the whole pipeline WORKS** — real LLM → tool calls → edit/create → compile → render →
correct image. Every scenario produced a visually-correct result. The run surfaced one real pipeline BUG
(a headless GL error — now FIXED), a dead default model (now FIXED), and one large EFFICIENCY problem (tool
catalogue sent twice per request — deferred), plus confirmed/refuted several behavioral weak-spots. A
follow-up review swarm corrected three trace-interpretation errors in an earlier draft of this report (see
the inline "corrected by the review" notes).

## What each scenario showed (pipeline mechanics)

| Scenario | Result | Pipeline verdict |
|---|---|---|
| smoke (solid red) | red rendered, edit→compile→render clean | ✅ full loop works |
| 01 visual-blindness (darker / vignette) | darker honest + visible; vignette in code but visually imperceptible | ⚠️ soft visual-blindness (below) |
| 02 wrong-node targeting | "make this green" correctly hit CURRENT (Red Quad), not name-match | ✅ targeting healthy (simple case) |
| 04 circle / square / morph | squircle rendered correctly; agent DID read both refs (but content not load-bearing) | ✅ render ok / ⚠️ multi-file read happened, content unproven |
| 05 set_uniform edges | u_time correctly REJECTED; u_color added+set→red rendered | ✅ uniform path + reject both correct |

Renders verified by eye: solid red, darkened gradient, white squircle, red-via-uniform — all correct.

## Pipeline mechanics that WORK (confirmed in the traces)

- **Edit → compile feedback loop**: every edit returns `ok — compiled clean` / source-mapped errors to
  the agent as the tool result. Confirmed.
- **WORKING SET live-rebuild**: the block at the conversation bottom shows current line-numbered source +
  uniforms + errors, rebuilt every iteration. After the red edit, `uniforms:` correctly dropped `u_time`
  (no longer used). Working as designed.
- **Infrastructure error recovery:** when `create_node` hit the GL error, the tool returned
  `error: … failed`, the agent re-submitted a byte-identical create, which succeeded. The loop never hung —
  the worker+bridge-pump drive is solid. (This was a BLIND retry of a transient infra failure, NOT agent
  comprehension — and the GL error itself is now fixed.)
- **✅ Agent-level compile-error recovery (the biggest gap — NOW CLOSED for the broken-compile class).**
  A follow-up live run (scenario 06 Part A) forced a GUARANTEED compile error: the agent was asked to call a
  nonexistent `plasma_wave(...)`. Trace evidence: `edit_shader` #1 → `compiled with errors: ... no function
  with name 'plasma_wave'`; the agent READ that error → `insert_after` #2 defined `vec3 plasma_wave(...)`
  inline → `compiled clean`. 2 tool calls, no giveup, no max_iterations; the render shows the correct plasma
  pattern. So the copilot DOES read a compile error and self-correct (a comprehending fix, not a blind
  retry) — for this class. STILL untested: `old_str` mismatch recovery, bad-node-id recovery, and the THRASH case
  (many consecutive applies-but-broken edits — scenario 03 + 06 Parts B/C, not yet run).
- **Gate flow**: render_image is always-gated; `drive_until_idle(auto_approve_gates=True)` answered them
  inline; no deadlock.
- **set_uniform type handling**: engine-driven `u_time` rejected with a clear message; a real `vec3`
  uniform added via 2 edits then set to `[1,0,0]` and rendered red.
- **"You cannot see" discipline**: the agent consistently hedged ("compiled clean", "look at the preview
  to confirm") rather than asserting a visual result. The prompt rule holds for grok-4.3.

## 🔴 PIPELINE BUG — `GLError 1282 (invalid operation) glUseProgram(0)` headless

**FIXED** (see Resolution at the end). During scenario 04 (multiple create_node in one session), some
`create_node` calls returned `error: create_node failed`. The underlying GL exception was:

```
OpenGL.error.GLError(err=1282, description=b'invalid operation', baseOperation=glUseProgram, cArguments=(0,))
```

**PROVENANCE (corrected by the review):** this stack came from STDERR (loguru `logger.exception` in
`registry.py::execute`), NOT the trace transcript — the traces only show `result: error: create_node
failed`. The stack above is from the live-run stderr, not trace evidence.

**CALL PATH (corrected by the review):** the report originally mis-cited this as
`_copilot_persist_shader → node.render()` and blamed "render/worker interleaving". That was wrong on both
counts. The real chain is `create_node`/edit → `Node.release_program()` → `Node.invalidate()` →
**`glUseProgram(0)` in `Node.invalidate` (core.py)**. `_copilot_persist_shader` never calls `render()`, and `render()`
contains no `glUseProgram`. The harness's `render()` blocks on `worker.join()` and runs only when the turn
is idle, so it cannot interleave with a worker compile — that "fix" was chasing a non-cause.

- `glUseProgram(0)` (unbind) is needed in the LIVE app (a deleted program left GL-current crashes the imgui
  end-of-frame restore — GLError 1281). Under a STANDALONE EGL context the same call raises GLError 1282
  (invalid operation) — the same headless GL-quirk class already suppressed for node teardown
  (`conventions.md ## Known quirks`, `test_render_for.py`). Spuradic in the run (timing-dependent), not on
  every create.
- Impact (pre-fix): the copilot RECOVERED (the tool returned `error: create_node failed`, the agent
  re-submitted a byte-identical create, which succeeded), so the user got a correct result — but it wasted
  a tool call + tokens + money, and a worse model might not recover.
- Fix (DONE): wrap the `glUseProgram(0)` in `Node.invalidate` in `contextlib.suppress(Exception)` — harmless in
  the live app (the call still runs), and under a standalone context the bind is pointless (no imgui
  restore to protect) so only its exception mattered. Verified: re-running the morph scenario after the fix
  created 3 nodes with `create_node/replace_lines failed count = 0` (was 3 before).

## 🔴 EFFICIENCY — the tool catalogue is sent TWICE per request, and 15/21 tools are irrelevant

Per-section breakdown of ONE simple request (iter-0, "make it red", 8055 input tokens):

| Section | ~tokens | note |
|---|---|---|
| system prompt (prose) | ~3550 | **abbreviated prose mentions of the tools** ("WHAT YOU CAN DO" walks edit/replace/insert/render/publish/… with explanations) — a SECOND, prose copy on top of the native defs |
| native `tools=` block | ~1950 | the 21 tools as full native tool-defs (the canonical copy) |
| WORKING SET (shader source) | ~2900 | the live source — legitimately needed |
| project map + lib + templates + conventions | ~490 | compact, useful |
| user message | ~17 | — |

So **tool descriptions ride in EVERY request twice** — once as prose in the system prompt, once in the
native `tools=` array (~5500 tok combined). And **15 of the 21 tools are telegram/youtube/publish** —
never relevant to a shader-edit turn, yet their full descriptions ship every request. The whole context
(8k tok) re-ships every iteration (the model re-reads it each step), so a 2-iteration turn = 16k input
tokens, and cost scales with it: a trivial one-line edit costs **$0.021**.

This is exactly the deferred `todo.md` "lazy tool catalogue" item, now with live evidence. The two cheap
wins: (1) DON'T duplicate the tool descriptions in the system prompt prose — the native `tools=` block
already carries them; the prose should be a short "you can edit/create/render/publish" overview, not a
re-listing. (2) Lazily load the telegram/youtube/publish tools (they carry `eager`/`category` already) so
a shader turn ships ~6 tools, not 21. Together these could roughly HALVE the per-turn input tokens.

## Behavioral findings (model quality — NOT pipeline; grok-4.3, a cheap model)

- **Visual-blindness (soft form):** the vignette edit landed real code (`1.0 - smoothstep(0.6,1.1,
  length(vs_uv-0.5))`) and the agent hedged ("subtle"), but the effect is visually imperceptible (weak
  smoothstep range + already-dark frame). NOT a hallucination — but the human can't confirm the claimed
  effect and neither can the agent. This is the milder shape of the `inspect_render` motivation: "can't
  verify the claimed effect", not "lied about it".
- **Multi-file read HAPPENED, but its content wasn't load-bearing (corrected):** the agent DID call
  `read_shader({"nodes":["<square>","<circle>"]})` — both reference shaders — the moment the user said
  "read both first" (verified in the trace; the earlier claim that it "did NOT read the refs / only read
  Rounded" was a misreading and is wrong — `read_shader` isn't even a valid recovery for a create_node GL
  failure). So the multi-file read MECHANISM works. What's unproven is whether the read CONTENT actually
  drove the synthesis — the squircle is simple enough the model could have written it from knowledge
  regardless. To probe that the content is load-bearing, the task must be UNSOLVABLE without the references
  (e.g. "use the EXACT color constant from node X"). The capability isn't broken; the probe was too weak.
- **Targeting healthy (simple case):** a bare "this one" correctly resolved to the CURRENT node, did not
  name-match a sibling. The harder bait (a request word matching a different node's name) wasn't run.
- **Tool selection sane:** create_node with `switch_to=False` for a background node; edit vs set_uniform
  chosen correctly; no redundant re-reads; stopped cleanly when done.

## Cost summary

16 turns, **$0.2072** total on grok-4.3 ($1.25/$2.50 per Mtok in/out). ~$0.013 per turn average; a
trivial edit ~$0.005-0.021 (dominated by the re-shipped context). The duplicated+irrelevant tool
catalogue is the biggest single lever on per-turn cost.

## Retrospective priorities

**Shaderbox engine:**
1. ✅ **DONE — bumped the default model** off the deprecated `grok-4-fast` → `x-ai/grok-4.3`
   (`exporters/integrations.py::CopilotIntegration.model`). Also fixed a SECOND hardcoded default the
   harness carried (`scripts/dogfood.py`'s `OPENROUTER_MODEL` fallback — now defers to the in-tree default).
2. ✅ **DONE — fixed the headless `glUseProgram(0)` GLError** (`Node.invalidate`, wrapped in
   `contextlib.suppress(Exception)`). Re-verified: the morph scenario now creates 3 nodes with 0 failures.
3. **De-duplicate + lazy-load the tool catalogue** (the `todo.md` lazy-catalogue item) — the biggest token/$
   win, ~halves per-turn input. NOT done (its own slice). Two levers: drop the prose tool-walk in the
   system prompt (the native `tools=` block is canonical); lazy-load the telegram/youtube/publish tools.
4. **An `inspect_render` affordance** for the visual-blindness gap (deferred in `todo.md`).

**Dogfooding (the harness + scenarios + skill):**
5. **Agent-level error recovery — PARTIALLY closed.** The broken-compile read→fix loop is now PROVEN (the
   scenario-06 plasma_wave run above: real compile error → agent reads it → defines the function → clean).
   STILL untested: the THRASH case (many consecutive applies-but-broken edits — scenario 03, never run, the
   one tied to the broken-compile circuit-breaker), `old_str` mismatch recovery, bad-node-id recovery,
   malformed args (scenario 06 Parts B/C). Run those + inspect the `edit_giveup`/`max_iterations`/
   `consecutive_failed_edits` trace events.
6. **Strengthen scenario 04** so the multi-file read content is LOAD-BEARING (the task must be unsolvable
   without the reference, e.g. "use the exact color constant from node X") — the current task is solvable
   from the model's own knowledge, so it only proves the read MECHANISM fires, not that the content matters.
7. **A `context_breakdown` trace event** (per-section token counts) so the §5 analysis is automatic, not
   hand-estimated — and a per-run cost ceiling so a runaway turn can't burn credits.
