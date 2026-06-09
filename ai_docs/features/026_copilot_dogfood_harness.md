# Feature 026 — Copilot dogfood harness

A headless driver that lets a HUMAN (Claude) exercise the copilot ENGINE end-to-end against a real
shader project, step by step — send a user message, read the trace, render a node to a small PNG,
eyeball it, take the next step. The judge is the human reading the trace + looking at the rendered
images; there are NO code assertions and no CI pass/fail. The point is dogfooding: surface where the
copilot is weak, where context wastes tokens, what's missing from context — before shipping the
copilot stack (020–024 + 030).

**Built on feature 025.** 025 extracts `ProjectSession` (the headless project + copilot core, App-free) and
lands C1–C4 + C6. This feature is its OWN feature (not a commit in the 025 sequence): the thin driver on
top — it constructs a `ProjectSession` on a standalone EGL context and drives `session.copilot` turn by
turn. It does NOT reconstruct or stub the backend — that's the whole point of 025. It also adds the
`ProjectSession.pump_until_idle()` helper (bridge mode 2b) here, since the harness is its only consumer.

## Goal

- A small `scripts/dogfood.py` the human imports and drives **function by function** from a REPL / a
  chat-driven loop. NOT a scenario-file parser, NOT a framework. The "scenario" is a free-form markdown
  checklist (or just a chat instruction) the human reads and translates into calls by hand.
- Runs the REAL copilot engine end-to-end via a real `ProjectSession` (feature 025) — the same
  edit/compile/working-set/checkpoint/render code that runs in the live app — on a standalone EGL GL
  context instead of a glfw window. Real renders, real compiles, real tool dispatch. Zero lambdas /
  reimplementation: the harness builds `ProjectSession(project_dir=tmp, notifier=<noop>)` (all `on_*`
  callbacks default to no-ops) and drives `session.copilot`.
- Renders IMAGES ONLY at 400×400 (no video — Pi is slow at video; a single 400×400 image is ~2.6 ms
  here). The human opens the PNG and judges the shape (circle vs square vs morph).
- Reuses the EXISTING copilot trace (`copilot/trace.py`) — the plain-text transcript already logs the
  full request messages (system prompt + context + tools=) and per-iteration usage (in/out/cost). No
  new trace instrumentation for v1.
- Multi-turn: one live `ProjectSession` + the copilot's own history accumulates across the human's
  messages, like a real chat session.
- A `pump_until_idle()` helper (lands in 025/C5 on `ProjectSession`) drives the worker→main bridge drain
  on the harness thread so a `run_on_main` never deadlocks — the harness either monkeypatches
  `bridge.run_on_main = fn()` inline (single-thread) or pumps via this helper.

## Out of scope

- **Video render** — `render_video` is never called by the harness. Trigger to revisit: a scenario
  genuinely needs animated output AND someone accepts the Pi's multi-frame+ffmpeg cost.
- **Publish / Telegram / YouTube tools** — not stubbed, not connected. On a fresh isolated data dir
  the precheck (`tools/publish.py`) returns ok=False on unconnected state WITHOUT network I/O, which is
  itself useful dogfood signal (does the copilot recover gracefully from a cred-miss handoff?). Trigger
  to revisit: a scenario specifically dogfooding the publish-handoff UX wants a fake connected exporter.
- **Per-section token-breakdown trace event** (`context_breakdown`: per-section char/approx-token counts
  for system / project-map / lib-catalog / template-catalog / working-set / history). Deferred to a
  follow-up — it's a nice-to-have AFTER the human finds eyeballing the existing `llm_request` blocks too
  tedious, NOT a v1 prerequisite. Trigger: `todo.md` entry filed at impl time.
- **App / glfw path (variant A)** — building the harness on `App(headless=True)` + the one-line bridge
  patch (`tests/conftest.py:44`) works on a machine WITH a display, but glfw cannot init on this
  display-less Pi (verified: `glfw.init()` returns 0, "X11: DISPLAY missing"). The harness targets the
  standalone-EGL path (variant B) only. Trigger to add A: dogfooding moves to a desktop with a display
  and the App-realistic threading path becomes worth testing.
- **Scenario-file DSL / auto-runner** — the harness is hand-driven. No structured markers, no parser.
- **Recorded/replay LLM client** — v1 uses the real OpenRouter client. A recorded client (for
  deterministic re-runs) is a possible later addition; the `LLMClient` Protocol already allows it.

## Design decisions

1. **Real `ProjectSession` on a standalone EGL context.** The harness sets the env (decisions 6-7), makes
   a `moderngl.create_standalone_context(backend='egl')` current on the harness thread, then constructs a
   real `ProjectSession(project_dir=tmp, notifier=<noop>)` (feature 025) — all `on_*` callbacks default to
   no-ops. Node/Canvas pick up the EGL context via `moderngl.get_context()` on the current thread (025
   decision 2's documented precondition). It does NOT import `App` (glfw-bound, won't init here — verified:
   `glfw.init()` fails on this Pi). The REAL `CopilotBackend`/`RevertExecutor`/`CopilotSession` come built
   inside `ProjectSession` — the harness reconstructs nothing. Verified this session: EGL → V3D GPU,
   `#version 460` compiles, 400×400 render+read ≈ 2.6 ms.

2. **Bridge drained on the harness thread (no deadlock).** `CopilotBackend` marshals every GL-affine verb
   through `bridge.run_on_main(fn)`, which BLOCKS the worker until the bridge is drained. Two valid headless
   modes: (a) single-thread — monkeypatch `session.copilot.bridge.run_on_main = lambda fn, timeout=None,
   defer=False: fn()` (the `tests/conftest.py:44` idiom; the current thread owns the GL context, `defer` is
   ignored); or (b) drive `session.copilot.enqueue_turn(prompt)` on its worker and pump via
   `ProjectSession.pump_until_idle()` (the 025/C5 helper: `while copilot.state.in_flight: drain_bridge();
   pump_events()`) so `run_on_main` never deadlocks. Default to (a) — simpler, and it's the path the tests
   already exercise.

3. **Driven via `session.copilot`, gate answered inline / by the human.** The harness drives the copilot
   through `ProjectSession`'s own `copilot` (the real `CopilotSession`), consuming the `AgentEvent` stream.
   On each `AgentGateOpened`, the harness surfaces the request to the human/driver (decision 4) — under
   single-thread mode (2a) the gate is answered between iterator steps, no polling thread. Multi-turn
   history is the copilot's own (accumulated in `CopilotSession`), exactly like the live app.

4. **The human IS the gate policy.** Per maintainer: Claude is the tester, drives a high-level scenario,
   and decides each gate by situation (approve a render, decline a delete to probe recovery, …). The
   harness surfaces every `AgentGateOpened` to the driver — prints the engine-authored gate prompt
   (`_GATE_PROMPTS` in agent.py — its clarity is itself dogfood signal) and lets the driver answer
   yes/no/secret in code. No fixed auto-policy, no scenario-file syntax for gates.

5. **FORCE 400×400 on every render.** `render_image(node, 0, 0)` falls back to the node's canvas size
   (`_copilot_render_dims` → `node.canvas.texture.size`), which defaults small (~64×64) — too small to
   judge a shape. The harness's render helper passes width=height=400 explicitly, regardless of what the
   copilot requested, so the human always gets an eyeball-able PNG.

6. **Isolated data dir + tmp project — no real-state pollution.** `shader_lib_root()` lives under
   `app_data_dir()`, shared across runs; a lib-editing scenario (`insert_after('lib:…')`) would WRITE into
   the maintainer's real library and leak between runs. The harness sets `SHADERBOX_DATA_DIR` to a tmp dir
   (paths.py honors it) BEFORE any import that reads it, and stands the throwaway project under a tmp dir
   too — never `projects/dev/`. (Note: `scripts/smoke.py` does NOT set `SHADERBOX_DATA_DIR`; do not copy
   its setup blindly.)

7. **MESA overrides set process-wide before context creation.** `#version 460` on v3d/llvmpipe needs
   `MESA_GL_VERSION_OVERRIDE=4.6` + `MESA_GLSL_VERSION_OVERRIDE=460` in the environment BEFORE the GL
   context is created (the driver reads them at creation). The harness asserts they're set (or sets them
   in `os.environ` early) and fails LOUDLY with a clear message — a missing override makes every shader
   compile fail identically, which reads as "the copilot is broken" when it's an env problem.

8. **Stable render path per step (anti-stale-PNG).** `render_image` mints auto-incrementing filenames
   (`<name>_<short-id>_<n>.png`); after an edit+re-render the human must open the NEW file, and an OS
   viewer may cache the prior one — a false "no visual change" that poisons the verdict (the human-side
   mirror of the copilot's visual blindness). The harness render helper RETURNS and PRINTS the exact new
   path each call, and the driver opens that path via `Read` (which never caches). (Considered: copy each
   render to a single stable `last_render.png` — rejected for v1, the explicit-path print + `Read` is
   enough and keeps the per-step history.)

9. **Node GL teardown is Exception-suppressed.** Under a standalone context, `node.release()` ends in a
   raw GL call with no bound context and raises on cleanup (documented in `tests/test_render_for.py`'s
   fixture). The harness wraps node/context teardown in `contextlib.suppress(Exception)` — copy the test
   fixture's pattern, the standalone context reclaims objects on its own `release()`.

10. **API key + model come from the integrations store / env, not a UI.** The OpenRouter client reads
    key+model through getters (`copilot/llm/openrouter.py`). The harness supplies them from the existing
    `integrations.json` under the (isolated) data dir, or from env vars, configured once at construction —
    no Settings UI. If absent, the harness fails loudly before the first turn.

## Backend construction — handled by feature 025

The harness builds NO backend wiring. `ProjectSession` (feature 025) owns the whole copilot cluster
(`CopilotBackend` + `RevertExecutor` + `CopilotSession`, the `_build_copilot_capabilities` body, the
`CheckpointStore`) constructed against `self`. The harness just calls
`ProjectSession(project_dir=tmp, node_templates_dir=…, starter_template_id=…, notifier=<noop>)` — the
`on_*` callbacks default to no-ops, so no UI seam needs supplying. The 18 backend injections that earlier
recon thought the harness must reproduce are exactly the state `ProjectSession` lifts out of `App`; the
harness inherits all of them for free. This is the entire reason 025 exists: zero lambdas, zero
reimplementation, the same prod code path.

## Tool coverage (image-only headless)

**Exercisable (drive in scenarios):** read_shader, edit_shader, replace_lines, insert_after, set_uniform,
create_node, delete_node (gated), switch_node, grep, read_lib, render_image (400×400). These are the
core editing / context / search / nav / image-render surface — the whole point of dogfooding.

**Skip execution:** render_video (out of scope), publish_telegram, publish_youtube (precheck blocks on
unconnected — the handoff message is signal, but no network), the 6 telegram/credential tools + youtube
CONFIG gate (require live API / UI gates). Don't write scenarios that need them; nothing to mock.

## Scenarios — what to dogfood (prioritize known weak spots)

Scenarios are markdown checklists the human reads + drives. The high-value ones deliberately probe the
`todo.md` copilot deferrals (these are the reasons we're NOT shipping yet):

- **Visual blindness / hallucinated success** (todo.md, the single most important target). Edit code that
  compiles clean + changes source but produces NO visual change; read the trace (copilot claims "made it
  red" / "added the text"); open the PNG and falsify the claim. This scenario is the proof-of-concept for
  why a machine-readable-render-feedback tool (`inspect_render`, deferred) is needed.
- **Wrong-node demonstrative targeting** (todo.md). Two nodes (e.g. current "Red Quad" + "Blue Sphere");
  say "make this a circle" (bare demonstrative, no name); check whether the agent resolves "this" → current
  or free-associates to the named node.
- **Broken-compile edit thrash** (todo.md). Push the agent toward edits that apply-but-compile-with-errors
  repeatedly; watch the iteration count approach `max_iterations` with no circuit-breaker.
- **Multi-file read + aggregate** (the maintainer's example): create node A (circle), node B (square), ask
  the copilot to merge/morph them into one (rounded square); check the trace shows BOTH nodes read into
  the working set, then render + eyeball.
- **grep → read_lib two-step**, **set_uniform edge cases** (vec3 / scalar / text string / reject u_time),
  **checkpoint data-integrity** (mutate → inspect the checkpoint dir captured the pre-turn state).

A scenario file reads like a STORY: `User:` lines (what the human types to the copilot), `Expect:` (which
tools should fire), `Human check:` (trace claim vs PNG eyeball). No tool/arg specs — the harness infers
those; the human judges.

## Files touched

- **`scripts/dogfood.py`** (new) — the harness module: env+MESA assertion, EGL context, tmp project +
  isolated data dir, `ProjectSession` construction (no-op `notifier`/callbacks), the bridge mode (decision
  2), a `send(user_text)` that drives `session.copilot` and yields/prints each `AgentEvent`, a
  `render(node_id?)` that forces 400×400 and returns+prints the PNG path, gate-answer hooks the driver
  calls inline. Designed to be imported + driven from a REPL, not run as `__main__`.
- **`ai_docs/scenarios/`** (new dir) — a handful of markdown scenario checklists (the weak-spot probes +
  the maintainer's circle/square/morph). Free-form; the human reads + drives them.
- **`shaderbox/project_session.py`** — add the `pump_until_idle()` helper (used by bridge mode 2b). The
  rest of `ProjectSession` is feature 025.
- **No new backend / capability wiring** — `ProjectSession` (025) owns it. No `copilot/` engine change for
  v1. (If the `context_breakdown` trace event lands later, that's a separate small change to
  `trace.py`/`agent.py`.)
- **`ai_docs/todo.md`** — file the deferred `context_breakdown` token-breakdown event with a trigger.
- **`ai_docs/roadmap.md`** — banner + a feature row at sweep time.

## Manual verification

This feature IS a manual-verification tool, so "verification" = the harness successfully drives a real
turn end-to-end on the Pi:
- `make check` green (ruff + pyright on the new module).
- A smoke drive: construct the harness, `send("create a shader that draws a filled white circle")`,
  confirm the trace shows create_node/edit + a clean compile, `render()` to a 400×400 PNG, open it,
  confirm a white circle. Then one weak-spot scenario (visual-blindness) to confirm the human-judge loop
  surfaces the gap.
- The token/context review loop works: the human can read a turn's `llm_request` block in the trace and
  see what the context is composed of.

## Review history

(Pre-impl recon: a 7-agent ultracode workflow — 6 lenses [backend-construction, bridge-gl-context,
project-fixture, trace-tokens-logging, gate-llm-session, scenarios-coverage] + an adversarial critic. The
critic overturned the recon's shared framing: 5 lenses assumed the harness must REBUILD the backend +
bridge + 20 injections from scratch, but `run_turn` consumes plain `CopilotCapabilities` and the real
`CopilotBackend` is App-free and constructable headless — so the harness drives the REAL engine, not a
reimplementation — which led to feature 025 (extract a real headless `ProjectSession` rather than rebuild
or stub the backend; this harness drives that). The critic also surfaced the load-bearing gotchas now
pinned as decisions 4–10: the
64×64 default-canvas trap, the global-shader-lib pollution risk, the stale-PNG judge trap, the
set-MESA-before-context requirement, and the standalone-context teardown suppression. Established facts
verified live this session: EGL→V3D compiles `#version 460`; 400×400 render ≈ 2.6 ms; `glfw.init()` fails
on this display-less Pi [App path excluded]. Pre-impl engine review TBD before plan-lock if requested.)

## Resolution (as-built — scaffold landed + verified, LLM turn blocked)

`scripts/dogfood.py` landed (commit `7b64bae`): `DogfoodHarness.create()` + `send` / `drive_until_idle` /
`render` / `approve` / `decline` / `nodes` / `release`. Built on a real `ProjectSession` (025), constructed
+ load + render + the human-eyeball loop VERIFIED live on the Pi (the UV-Mango gradient rendered headless on
V3D at 400×400 and read back correctly — the first runtime proof 025's headless core works).

As-built corrections to the plan above:
- **Threading: worker + bridge-pump, NOT a sync patch (decision 2 was wrong).** Verified against the code:
  `CopilotSession` ALWAYS spawns a worker thread (`_ensure_worker`), and `run_turn` runs there; the worker
  marshals GL via `bridge.run_on_main` which BLOCKS until the main thread drains it. A sync patch
  (`run_on_main = fn()`) would run GL on the WORKER thread → EGL thread-affinity violation. So the harness's
  `drive_until_idle` pumps `drain_bridge()` + `pump_events()` on the context-owning thread while the worker
  runs (the exact `App` frame-loop shape, `ui.py`). `render()` likewise runs `render_image` on a throwaway
  thread and pumps the bridge from the owner thread (a direct call deadlocks — confirmed).
- **Env via module-top side effect, not a `bootstrap_env()` call.** `SHADERBOX_DATA_DIR` + the MESA
  overrides + the written `integrations.json` are set at the top of `scripts/dogfood.py` BEFORE the
  shaderbox imports (the import-order constraint), reading the key from `OPENROUTER_API_KEY`.
- **No `make_current` call** — `moderngl.create_standalone_context(backend='egl')` leaves the context
  current on the creating thread (verified); `Node`/`Canvas` pick it up via `get_context()`. One sanctioned
  `# type: ignore[arg-type]` (moderngl's `**kwargs` stub gap, added to `conventions.md` allowlist).
- **glfw warning silenced.** `core.py`'s render reads `glfw.get_time()` for the default `u_time`; glfw is
  never `init()`'d (EGL path), so it warns + returns 0.0 — which is exactly the static t=0 frame the dogfood
  wants. The harness installs a no-op glfw error callback to keep output clean.
- **`ProjectSession` has no `release()`** (App owns lifecycle live); the harness tears down via
  `session.copilot.release()` then `ctx.release()`.

BLOCKED: a live LLM turn (`send` + `drive_until_idle`) needs `OPENROUTER_API_KEY` — none on this Pi (no
`integrations.json` at the standard paths, env unset). Run it on a box with the maintainer's key; then write
+ drive the weak-spot scenarios. The token/context-review and `context_breakdown` log are still deferred to
after the first real runs show eyeballing the existing trace is too tedious.)
