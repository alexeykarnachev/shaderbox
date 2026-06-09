# Dogfood report — first live run (2026-06-09)

First real end-to-end dogfood of the copilot ENGINE, headless on the Pi via `scripts/dogfood.py`
(feature 026). Model: **`x-ai/grok-4.3`** (the configured `x-ai/grok-4-fast` is DEPRECATED — 404 from
OpenRouter; first finding). 5 scenarios, 16 turns, **$0.207 total**. All renders eyeballed.

**Bottom line: the whole pipeline WORKS** — real LLM → tool calls → edit/create → compile → render →
correct image. Every scenario produced a visually-correct result. But the run surfaced one real
pipeline BUG (a headless GL error) and one large EFFICIENCY problem (tool catalogue sent twice per
request), plus confirmed/refuted several behavioral weak-spots. None fixed yet (per maintainer: run
first, retrospect after).

## What each scenario showed (pipeline mechanics)

| Scenario | Result | Pipeline verdict |
|---|---|---|
| smoke (solid red) | red rendered, edit→compile→render clean | ✅ full loop works |
| 01 visual-blindness (darker / vignette) | darker honest + visible; vignette in code but visually imperceptible | ⚠️ soft visual-blindness (below) |
| 02 wrong-node targeting | "make this green" correctly hit CURRENT (Red Quad), not name-match | ✅ targeting healthy (simple case) |
| 04 circle / square / morph | squircle rendered correctly; but agent did NOT read the two refs | ✅ render ok / ⚠️ multi-file not exercised |
| 05 set_uniform edges | u_time correctly REJECTED; u_color added+set→red rendered | ✅ uniform path + reject both correct |

Renders verified by eye: solid red, darkened gradient, white squircle, red-via-uniform — all correct.

## Pipeline mechanics that WORK (confirmed in the traces)

- **Edit → compile feedback loop**: every edit returns `ok — compiled clean` / source-mapped errors to
  the agent as the tool result. Confirmed.
- **WORKING SET live-rebuild**: the block at the conversation bottom shows current line-numbered source +
  uniforms + errors, rebuilt every iteration. After the red edit, `uniforms:` correctly dropped `u_time`
  (no longer used). Working as designed.
- **Error recovery**: when `create_node` / `replace_lines` hit the GL error (below), the tool returned
  `error: … failed`, the agent received it, RETRIED, and the retry succeeded. The loop never hung — the
  worker+bridge-pump drive is solid.
- **Gate flow**: render_image is always-gated; `drive_until_idle(auto_approve_gates=True)` answered them
  inline; no deadlock.
- **set_uniform type handling**: engine-driven `u_time` rejected with a clear message; a real `vec3`
  uniform added via 2 edits then set to `[1,0,0]` and rendered red.
- **"You cannot see" discipline**: the agent consistently hedged ("compiled clean", "look at the preview
  to confirm") rather than asserting a visual result. The prompt rule holds for grok-4.3.

## 🔴 PIPELINE BUG — `GLError 1282 (invalid operation) glUseProgram(0)` headless

The single real bug. During scenario 04 (multiple create_node + a replace_lines in one session), two tool
calls failed with:

```
OpenGL.error.GLError(err=1282, description=b'invalid operation', baseOperation=glUseProgram, cArguments=(0,))
```

- `glUseProgram(0)` (unbind program) throws `invalid operation` under the standalone EGL context — the
  SAME headless GL-quirk class already documented for node teardown (`conventions.md ## Known quirks`,
  `test_render_for.py`'s Exception-suppressed `node.release()`). Here it surfaces in the
  create/replace → `_copilot_persist_shader` → `node.render()` path, not just teardown.
- A minimal repro (5 nodes compile+render in a loop) does NOT reproduce it — so it's not plain repeated
  compile; it's tied to the bridge-marshalled create/render interleaving (render_image between turns +
  the worker-drain timing). Spuradic, not deterministic.
- Impact: the copilot RECOVERS (retries, succeeds) so the user gets a correct result, but it wastes a
  tool call + tokens + money each time, and a worse model might not recover. The render path itself
  (render_image) never failed — only the create/edit persist path.
- Honest fix (when retrospecting): suppress / guard the `glUseProgram(0)` in the persist/render path
  under a standalone context (mirror the teardown suppression), OR make the harness's render not
  interleave with an in-flight worker compile. Needs a focused look at `core.py`'s render + the bridge
  drain ordering.

## 🔴 EFFICIENCY — the tool catalogue is sent TWICE per request, and 15/21 tools are irrelevant

Per-section breakdown of ONE simple request (iter-0, "make it red", 8055 input tokens):

| Section | ~tokens | note |
|---|---|---|
| system prompt (prose) | ~3550 | **includes a full prose description of every tool** ("WHAT YOU CAN DO" lists edit/replace/insert/… with explanations) |
| native `tools=` block | ~1950 | **the SAME 21 tools again**, full descriptions, as native tool-defs |
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
- **Multi-file read NOT exercised:** despite "read both of those shaders first", the agent wrote the
  squircle from its own knowledge and only `read_shader`'d the new Rounded node (for GL-error recovery).
  To actually probe multi-file read, the task must be UNSOLVABLE without the references (e.g. "use the
  exact color from node X"). The capability isn't broken — the task was too easy to need it.
- **Targeting healthy (simple case):** a bare "this one" correctly resolved to the CURRENT node, did not
  name-match a sibling. The harder bait (a request word matching a different node's name) wasn't run.
- **Tool selection sane:** create_node with `switch_to=False` for a background node; edit vs set_uniform
  chosen correctly; no redundant re-reads; stopped cleanly when done.

## Cost summary

16 turns, **$0.2072** total on grok-4.3 ($1.25/$2.50 per Mtok in/out). ~$0.013 per turn average; a
trivial edit ~$0.005-0.021 (dominated by the re-shipped context). The duplicated+irrelevant tool
catalogue is the biggest single lever on per-turn cost.

## Retrospective priorities (NOT done — for the next pass)

1. **Bump the default model** off the deprecated `grok-4-fast` (→ `x-ai/grok-4.3` or current). One-line
   config fix in `copilot/integrations` default + anywhere `grok-4-fast` is hardcoded.
2. **Fix the headless `glUseProgram(0)` GLError** in the create/persist/render path (suppress/guard under
   standalone context, mirroring the teardown fix).
3. **De-duplicate + lazy-load the tool catalogue** (the `todo.md` lazy-catalogue item) — biggest token/$
   win; ~halves per-turn input.
4. Optionally: a `context_breakdown` trace event (per-section token counts) so this analysis is automatic,
   not hand-estimated.
