---
name: dogfood
description: "Run the headless copilot dogfood harness — drive the REAL copilot engine on the Pi (no App/glfw) against a real LLM, render images, eyeball them, and produce a findings report (cost + what to improve in ShaderBox + what to improve in the dogfooding itself). Use when: dogfooding the copilot, testing the copilot end-to-end, exercising the copilot engine, running scenarios against the copilot, checking the copilot pipeline, or 'докфудинг'/'прогони сценарии'/'протестируй копайлот'. Living skill — improve it each run."
user_invocable: true
---

<command-name>dogfood</command-name>

Drive the REAL copilot ENGINE end-to-end, headless on the Pi (no App, no glfw window), against a real
LLM. Create a `ProjectSession` on a standalone EGL context, send turns, watch the tool calls + compile
feedback, render images, OPEN them and judge by eye. The judge is YOU (reading the trace + the PNGs) —
there are NO code assertions. The point is to test the whole PIPELINE and find where the copilot is weak,
where context wastes tokens, and what's broken — not to make the copilot write good shaders (use a CHEAP
model and SIMPLE tasks; it will make mistakes, that's fine).

This is feature 026. The harness is `scripts/dogfood.py`; scenarios are in `ai_docs/scenarios/`. This
skill is the operating manual — the process + every gotcha already hit, so you don't re-discover them.

## 0. Prerequisites (the run fails without these)

- **`OPENROUTER_API_KEY`** — required, billed. The maintainer supplies it. Pass it in the COMMAND env
  (`env OPENROUTER_API_KEY=… uv run …`), not a permanent export — the harness reads it at import.
- **Model:** the in-tree default (`CopilotIntegration.model`) is `x-ai/grok-4.3` (cheap: ~$1.25/$2.50 per
  Mtok), used automatically — no `OPENROUTER_MODEL` override needed. Set `OPENROUTER_MODEL` only to try a
  different model. Models go deprecated (grok-4-fast 404'd a prior run) — if a run 404s on the model, check
  `curl -s https://openrouter.ai/api/v1/models | …` filter `x-ai/grok` for the current cheap grok and bump
  the in-tree default.
- **This is a display-less Pi.** `glfw.init()` FAILS here; `import glfw`/`import imgui` SUCCEED. The whole
  point of the headless harness is to bypass glfw via a standalone EGL context.

## 1. Run a scenario

The harness is a hand-driven library. Write a tiny throwaway driver (NOT committed — name it
`scripts/_dogfood_run.py`, delete it after) that imports `DogfoodHarness` and drives the steps, OR drive
it inline from a `python -c`. Pattern:

```python
from scripts.dogfood import DogfoodHarness
h = DogfoodHarness.create()                 # seeded tmp project (UV Mango / Media / Text)
# h = DogfoodHarness.create(seed_templates=False)   # empty project -> create_node from scratch
h.send("Make the current shader output solid red. Keep it simple.")
h.drive_until_idle(auto_approve_gates=True) # pump worker+bridge; prints each event; auto-yes gates
png = h.render(size=400)                     # 400x400 PNG of the current node (FORCED size)
# ...then Read the png and eyeball it...
print("cost $%.5f" % h.session_cost_usd, "trace", h.trace_path)
h.release()
```

Run it:
```
env OPENROUTER_API_KEY=… OPENROUTER_MODEL=x-ai/grok-4.3 uv run python scripts/_dogfood_run.py <scenario>
```

Then for each render: `cp <png> /tmp/dogfood_<tag>.png` and **Read it** — the visual check is the whole
point. Copy the trace too (`h.trace_path` → `/tmp/dogfood_<tag>.transcript`) — it's the per-turn context +
token/cost record you analyze.

The 5 scenarios live in `ai_docs/scenarios/` (visual-blindness, wrong-node targeting, compile-thrash,
circle/square/morph, grep+read_lib+uniforms). Read each `.md`, translate its `User:` lines into
`h.send(...)`, judge its `Human check:` against the trace + PNG.

## 2. The gotchas (hard-won — don't re-discover them)

- **Threading is worker + main-thread pump — NOT a sync bridge patch.** `CopilotSession` ALWAYS spawns a
  worker thread; the worker marshals GL ops to the main (context-owning) thread via `bridge.run_on_main`,
  which BLOCKS until drained. A sync patch (`run_on_main = fn()`) would run GL on the worker thread →
  EGL thread-affinity violation. The harness's `drive_until_idle` pumps `drain_bridge()` + `pump_events()`
  on the owning thread (mirrors `App`'s frame loop, `ui.py`). DON'T "simplify" this to a sync patch.
- **`render()` runs on a throwaway thread + pumps the bridge.** A DIRECT `render_image` call from the main
  thread DEADLOCKS (it enqueues a bridge op and blocks on a drain that never comes). The harness runs it on
  a helper thread and drains from the owner thread. Already handled — don't call `render_image` directly.
- **Env order: set BEFORE importing shaderbox.** `SHADERBOX_DATA_DIR` (isolation — never pollute the real
  library/creds) + the two MESA overrides (`MESA_GL_VERSION_OVERRIDE=4.6` / `MESA_GLSL_VERSION_OVERRIDE=460`,
  for `#version 460` on V3D) are set at the TOP of `scripts/dogfood.py` before the shaderbox imports. If
  you write a new entry point, preserve that order.
- **EGL context is already current after creation** — no `make_current` call; `Node`/`Canvas` pick it up
  via `moderngl.get_context()`. (moderngl's stub mistypes `backend=` — the one sanctioned `# type: ignore`.)
- **The `GLFWError: not initialized` warning is benign** — `core.py` reads `glfw.get_time()` for the default
  `u_time` (returns 0.0, the static t=0 frame we want). The harness installs a no-op glfw error callback.
- **🔴 `GLError 1282 (invalid operation) glUseProgram(0)` is a REAL pipeline bug, not harness noise.** It
  fires spuradically on bridge-marshalled create_node/replace_lines (the persist→render path) under the
  standalone context — the same headless GL-quirk as node teardown. The copilot RECOVERS (retries), so a
  run still completes, but log it as a finding. (Tracked in `todo.md`.) Don't mistake it for a harness fault.
- **Multi-file read needs an UNSOLVABLE-without-reading task.** "Merge node A and B" is solved from the
  model's own knowledge — a cheap model won't bother to `read_shader` the references. To actually exercise
  multi-file read, the task must REQUIRE the other node's content (e.g. "use the EXACT color/constant from
  node X"). Otherwise the probe is inconclusive.
- **`session_cost_usd` accumulates across turns**; `state.last_turn` has `context_tokens`/`reply_tokens`/
  `cost_usd` per turn. The trace's `llm_response` events have per-iteration `usage: in=/out=/cost=`.

## 3. Reading the trace (the context/token analysis)

`h.trace_path` → a plain-text transcript. Per turn it logs: `turn_start` (user_text + history +
eager_tools), each `llm_request` (the FULL messages array — system prompt + project map + working set +
the native `tools=` block — + max_tokens), each `llm_response` (finish_reason + text + tool_calls +
`usage: in/out/cost`), each `tool_call` (name + args + ok + result), `turn_done` (summed usage).

To estimate the per-section context cost, split one `llm_request` block: the system prompt (prose), the
project-map/lib/templates/conventions system block, the user message, the WORKING SET (shader source), and
the native `tools=` block. Rough tokens ≈ chars/4. (A `context_breakdown` trace event would automate this —
it's a deferred improvement.)

## 4. The report format (what the maintainer wants to see at the end)

Produce a markdown report (save it as `ai_docs/features/026_dogfood_report_<run>.md` — these are durable
findings). Sections, in this order:

1. **Bottom line** — does the whole pipeline work end-to-end? One paragraph. Note the model used.
2. **Per-scenario table** — scenario | result | pipeline verdict (✅/⚠️/🔴). One row each.
3. **Pipeline mechanics that WORK** — confirmed-in-the-trace: edit→compile feedback, WORKING SET
   live-rebuild, error recovery, gate flow, set_uniform handling, the "you cannot see" discipline. Cite the
   trace evidence, don't just assert.
4. **🔴 Pipeline BUGS** — anything that actually broke (the GLError, a deadlock, a wrong result). For each:
   the exact error, a minimal-repro attempt, the impact (did the copilot recover?), and the honest fix.
5. **🔴 Efficiency / token findings** — the per-section context breakdown (a table of ~tokens per section),
   where tokens are wasted (e.g. the tool catalogue shipped twice + irrelevant tools), and the lever to fix
   it. THIS IS A FIRST-CLASS DELIVERABLE — the maintainer specifically wants the context composition + token
   flow + the optimization.
6. **Behavioral findings (model, NOT pipeline)** — visual-blindness, multi-file read, targeting, tool
   selection. Be clear these are model-quality on a CHEAP model, separate from pipeline health.
7. **💲 Cost summary** — total $, per-turn average, the biggest cost lever. ALWAYS include cost.
8. **Retrospective priorities** — numbered, what to fix next, split into:
   - **What to improve in SHADERBOX** (the copilot engine itself — model default, the GLError, the tool
     catalogue, an `inspect_render` affordance, etc.).
   - **What to improve in the DOGFOODING** (the harness + this skill — a `context_breakdown` event, better
     scenario design, a cost ceiling, automating the per-section breakdown, anything that made this run
     awkward).

## 5. Clean up

Delete the throwaway driver (`scripts/_dogfood_run.py`). Keep the report + the traces/PNGs (copy traces
somewhere durable if a reviewer needs to verify your log interpretations). The harness + scenarios + this
skill stay. File the prioritized findings into `todo.md` with concrete triggers.

## 6. Improve this skill

This is a LIVING skill. Each run, if you hit a new gotcha or the report format wants a new section, ADD it
here so the next run is smoother. The maintainer wants the dogfooding itself to get more convenient over
time — that improvement loop lives in this file's §4 "improve the DOGFOODING" findings flowing back here.
