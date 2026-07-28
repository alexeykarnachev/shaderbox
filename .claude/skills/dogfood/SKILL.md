---
name: dogfood
description: "Run the headless copilot dogfood harness — drive the REAL copilot engine on the Pi (no App/glfw) against a real LLM, render images, eyeball them, and produce a findings report (cost + what to improve in ShaderBox + what to improve in the dogfooding itself). Use when: dogfooding the copilot, testing the copilot end-to-end, exercising the copilot engine, running scenarios against the copilot, checking the copilot pipeline, or 'докфудинг'/'прогони сценарии'/'протестируй копайлот'. Living skill — improve it each run."
user_invocable: true
---

<command-name>dogfood</command-name>

Drive the REAL copilot ENGINE end-to-end, headless on ANY display-less box (Pi V3D / WSL Mesa / CI —
NOT Pi-specific; all it needs is a standalone EGL context + `OPENROUTER_API_KEY` + network), against a
real LLM. Create a `ProjectSession` on a standalone EGL context, send turns, watch the tool calls + compile
feedback, render images, OPEN them and judge by eye. The judge is YOU (reading the trace + the PNGs) —
there are NO code assertions. The point is to test the whole PIPELINE and find where the copilot is weak,
where context wastes tokens, and what's broken — not to make the copilot write good shaders (use a CHEAP
model and SIMPLE tasks; it will make mistakes, that's fine).

Features 026 (the harness) + 027 (interactive resume/dump). Everything dogfood lives under ONE dir:
`scripts/dogfood/` — the harness is `scripts/dogfood/harness.py`, scenarios are
`scripts/dogfood/scenarios/`, and ALL run artifacts (per-run project dirs, the data dir, JSON dumps,
traces, PNGs) land in `scripts/dogfood/runs/` (gitignored). The public import is unchanged:
`from scripts.dogfood import DogfoodHarness`. This skill is the operating manual — the process + every
gotcha already hit, so you don't re-discover them.

## 0. Prerequisites (the run fails without these)

- **`OPENROUTER_API_KEY`** — required, billed. Must be in the process env before `uv run` (the harness
  reads it at import). **On the Pi** it's `export`ed in `~/.bashrc` — but below the standard "if not
  interactive, return" guard, so a NON-interactive shell (the default for tool Bash calls) comes back
  empty; run through an INTERACTIVE shell so the export fires: `bash -ic '<the uv run … one-liner>'`.
  **On another box (WSL / a fresh clone) the key may NOT be preset** — check `bash -ic 'echo
  ${OPENROUTER_API_KEY:+SET}'`; if empty, the maintainer must export it (or add it to that box's
  `~/.bashrc`). Don't assume the Pi's setup exists elsewhere.
- **Model:** the in-tree default (`CopilotIntegration.model`) is `openai/gpt-5.1-codex-mini` (cheap:
  ~USD 0.25 in / 2.00 out per Mtok, tool-call compatible, 400k ctx — no `$N` literals in this file:
  the skill runner substitutes `$0`/`$1`/… with invocation args), used automatically — no `OPENROUTER_MODEL`
  override needed. Chosen over grok: grok writes BAD GLSL (you can't dogfood the authoring pipeline on a
  model that can't write a shader); codex-mini is the cheap-but-competent-at-code pick. Set
  `OPENROUTER_MODEL` only to try a different model. Models go deprecated (grok-4-fast 404'd a prior run) —
  if a run 404s, `curl -s https://openrouter.ai/api/v1/models` and filter for the current cheap codex,
  confirm `tools` is in its `supported_parameters` (the agent rejects tool-incompatible models), bump the
  in-tree default.
- **Display-less box.** `glfw.init()` FAILS (no window); `import glfw`/`import imgui` SUCCEED. The whole
  point of the headless harness is to bypass glfw via a standalone EGL context — works on Pi V3D, WSL
  Mesa, a CI runner, anything with EGL. `h.render()` uses the DIRECT context-thread render
  (robust everywhere); the bridge-marshalled `render_image` can be slow-first-draw under software GL
  and hit the op timeout — that's a per-box quirk, not a harness fault (the agent's own render_image calls
  still exercise that path).
- **🔴 On WSL, set `GALLIUM_DRIVER=d3d12` — the real GPU, not llvmpipe (verified 2026-07-27).** Bare
  surfaceless EGL on WSL2 silently picks llvmpipe (software, minutes-long heavy-shader renders; a 560s
  flag turn barely fit); with `GALLIUM_DRIVER=d3d12` the SAME EGL path lands on the host GPU via WSLg's
  d3d12 gallium driver (`D3D12 (NVIDIA GeForce RTX 3090)`, native GL 4.6 — the MESA_*_OVERRIDE vars
  become unnecessary but are harmless). Prepend it to every harness command on a WSL box:
  `env GALLIUM_DRIVER=d3d12 SHADERBOX_DATA_DIR=... uv run ...`. Don't bake it into `harness.py` — it
  would misfire on non-WSL boxes; it's a per-box env, like the key.

## 1. Drive a scenario — ONE blocking `uv run` per turn (resume/dump)

> ⚠️ **DRIVE INTERACTIVELY — NEVER pre-script the reply sequence.** The scenarios are FREE-FORM GOALS
> with branch points (the `User:` / `if it does X, do Y` shape), not fixed dialogues. The dogfood tests
> whether YOU read each copilot reply and ADAPT the next message; a baked multi-turn `h.send(...)`
> progression replays a recording and defeats the entire point. Send ONE turn, READ the reply, THEN
> compose the next message. **A baked multi-turn driver is forbidden.**

The mechanism (feature 027): each turn is its OWN `uv run` process (inherently one blocking call). State
persists ON DISK — the harness `dump`s the conversation after the turn, and the NEXT process `resume`s it
via `create(project_dir=...)` with ZERO LLM calls (the conversation is NL-only-serialized; node edits are
already on disk). So you read turn N's JSON, think, then write turn N+1's command. No server, no background
process, no PID.

**Seeding the shader library (any mission that should exercise `SB_*` helpers):** the harness's
tmp data dir starts with an EMPTY lib — copy the canonical seed in BEFORE turn 1 and pass the SAME
`SHADERBOX_DATA_DIR` on every turn (the app's own startup seed-sync — `shader_lib/seed.py` — lives
in `App.__init__`, and the harness drives `ProjectSession` directly, so it does NOT fire here):
```
mkdir -p scripts/dogfood/runs/data-<run> \
  && cp -r shaderbox/resources/shader_lib scripts/dogfood/runs/data-<run>/shader_lib
env SHADERBOX_DATA_DIR=$PWD/scripts/dogfood/runs/data-<run> ... uv run ...
```
V3D shader-codegen cost (Pi): the driver compiles the final GPU code lazily at FIRST DRAW, on the
CPU — a heavy shader's first render pays it once (the old code-based glyphs paid ~20s; the
data-driven glyphs of 032 cut that to ~1s). Warm renders are fast (text 300x300 ~ tens of ms). If
a render burns 99% CPU for minutes it's first-draw codegen of an oversized shader, not a deadlock;
for time-sampled stills load the node directly on a standalone EGL context (no bridge timeout).

**Turn 1 (fresh project):**
```
env OPENROUTER_API_KEY=… uv run python -c '
from pathlib import Path
from scripts.dogfood import DogfoodHarness
h = DogfoodHarness.create()                          # seeded project (UV Mango / Media / Text)
# h = DogfoodHarness.create(seed_examples=False)     # empty -> create_node from scratch
h.send("Make the current shader output solid red. Keep it simple.")
h.drive_until_idle()                                 # pump worker+bridge; STOPS on a gate
h.render(size=400)                                   # 400x400 PNG (path echoed in the dump)
h.dump(Path("scripts/dogfood/runs/turn.json"))       # persist convo + write the JSON turn-result
h.release()'
cat scripts/dogfood/runs/turn.json                   # READ the result; note project_dir + data_dir
```
The JSON has `new_messages`, `assistant_text`, `open_gate`, `last_turn` (tokens/cost), `session_cost_usd`,
`last_render_path`, `trace_path`, and the two stable paths `project_dir` + `data_dir` to reuse next turn.
**Read the dumped `last_render_path` PNG** — the visual check is the whole point.

**Turn 2+ (resume — REUSE the same project_dir AND SHADERBOX_DATA_DIR from turn 1's dump):**
```
env OPENROUTER_API_KEY=… SHADERBOX_DATA_DIR=<data_dir from turn.json> uv run python -c '
from pathlib import Path
from scripts.dogfood import DogfoodHarness
h = DogfoodHarness.create(project_dir=Path("<project_dir from turn.json>"))  # resumes the convo
h.send("<the message YOU chose after reading turn 1>")
h.drive_until_idle(); h.render(size=400)
h.dump(Path("scripts/dogfood/runs/turn.json")); h.release()'
cat scripts/dogfood/runs/turn.json
```
🔴 **`SHADERBOX_DATA_DIR` MUST be set on the COMMAND LINE before `uv run`** — the harness reads it at
import (the env block runs when `scripts.dogfood` is imported, before any `create()` arg). Setting it
in-script after import loses to the already-run `setdefault`. Same for the resume project_dir: it's a
`create()` arg, but the data dir is env-only.

**Gates are answered WITHIN one process — a gate CANNOT span two turns.** A gate pauses the worker
mid-turn; the worker dies on process exit and a gated turn is never persisted, so there is no "dump the
gate, resume, answer it". Decide the gate answer UP FRONT when you compose that turn's command:
```python
h.send("delete the Media node")
h.drive_until_idle()                       # stops on the gate
if h._open_gate() is not None:
    h.decline()                            # or approve() — YOU decide per the scenario
    h.drive_until_idle()                   # let the copilot react to the decision
h.dump(Path("scripts/dogfood/runs/turn.json"))
```
For an unconditional yes, `h.drive_until_idle(auto_approve_gates=True)` is the shortcut. (Answering a gate
based on reading its OWN prompt text first is the one thing this can't do — reserved for a future server,
`027` Out-of-scope.)

The scenarios live in `scripts/dogfood/scenarios/`. Read each `.md`, drive its arc turn-by-turn, judge its
`Human check:` against the trace + the rendered PNG. (REPL note: for ad-hoc poking you can still drive the
harness from one long-lived `python` REPL — `send` / `drive_until_idle` / `render` / `approve` / `decline`
/ `reload` — but for a real scenario run the one-blocking-call-per-turn shape above is the discipline that
keeps you honest about reading each reply.)

## 1a. Tool-coverage discipline — DELIBERATELY route through the cold tools

Run 2 fired only 5 of ~12 reachable tools (`create_node` / `read_shader` / `edit_shader` +
the since-removed line tools); the whole navigation/value/integration half stayed COLD (`grep`,
`read_lib`, `set_uniform`, `switch_node`, `delete_node`, `render_image`/`render_video`). A cheap model
takes the lazy path — it answers "what nodes exist?" from the project map instead of grepping, hard-codes a
constant instead of adding a tunable uniform, edits the current node instead of switching. So coverage
won't happen by accident; YOU (the driver) have to provoke it.

**Before composing each turn, ask: "which cold tool can THIS turn legitimately force?"** Prefer the phrasing
that routes through an unexercised tool, as long as it stays a natural mission move (never a fake "now call
grep" instruction — the agent must have a real reason):

- **`grep` / `read_lib`** — ask the agent to REUSE something it must first LOCATE: a `SB_*` helper by
  behavior not name ("reuse the library edge helper"), or "which shaders use `u_time`?". A bare "what nodes
  exist?" loses to the project map — make the thing live in the LIBRARY (not in the always-present map) so
  it has to grep + read_lib to find and read it.
- **`set_uniform`** — demand an ADJUSTABLE look and then DIAL it ("turn the glow up", "make it dimmer").
  A hard-coded constant can't be tuned; the agent must introduce a uniform and `set_uniform` a value.
- **`switch_node`** — with node A current, ask to edit node B by name. Edits with no target hit the
  current node, so the agent must `switch_node` to B first.
- **`delete_node`** — give it a genuine throwaway to remove. It's GATED — decide approve/decline UP FRONT
  when you compose that turn (a gate can't span turns; `/dogfood` §1).
- **`render_image` / `render_video`** — these are the COPILOT's own gated tools, distinct from the harness
  `h.render()`. Ask the AGENT to save the result to a file ("render this to a PNG"); drive that turn with
  `drive_until_idle(auto_approve_gates=True)`.

**The analyzer reports coverage, and thin coverage is a first-class finding.** Every run's report MUST
include the per-tool fired/not-fired table (report §7c) and call out any tool that stayed
cold — distinguishing "the scenario never pressured it" from "a pressure move aimed at it but the agent
dodged" (the latter is a behavioral finding about the model). Treat full reachable-surface coverage as a
run goal alongside the scenario's visual goal: a beautiful render that touched 5 tools is a worse run than
a rougher one that exercised 11. (`publish_*` precheck-fails in the harness — empty `ExporterRegistry` —
so it's NOT reachable; don't count it as a missed cold tool.)

**End every mission with a SWEEP turn.** The final-source audits show edit sediment in every
multi-turn shader (dead clamps, duplicate predicates, no-op guards narrated as fixes, stale names).
Before closing a mission, send one last turn: "sweep the shader: remove dead code, duplicate
logic and leftovers from the editing session; change no behavior" — it both cleans the artifact
and probes the agent's self-review. **This turn is the CODE axis's standing probe (report §8): what
the sweep removes IS the edit-sediment measurement — record its diff.**

## 2. The gotchas (hard-won — don't re-discover them)

- **Threading is worker + main-thread pump — NOT a sync bridge patch.** `CopilotSession` ALWAYS spawns a
  worker thread; the worker marshals GL ops to the main (context-owning) thread via `bridge.run_on_main`,
  which BLOCKS until drained. A sync patch (`run_on_main = fn()`) would run GL on the worker thread →
  EGL thread-affinity violation. The harness's `drive_until_idle` pumps `drain_bridge()` + `pump_events()`
  on the owning thread (mirrors `App`'s frame loop, `ui.py`). DON'T "simplify" this to a sync patch.
- **`render()` runs on a throwaway thread + pumps the bridge.** A DIRECT `render_image` call from the main
  thread DEADLOCKS (it enqueues a bridge op and blocks on a drain that never comes). The harness runs it on
  a helper thread and drains from the owner thread. Already handled — don't call `render_image` directly.
- **`drive_until_idle` MUST fire `bridge.run_deferred_render()` each loop (fixed 2026-07-03).** The
  copilot's render tools — `probe_render`, `render_image`, `render_video` — call
  `run_on_main(..., defer=True)`, which PARKS the op for a post-swap firing point. The real App fires it in
  `ui.py` (`run_deferred_render()` after `drain_bridge`); the harness loop omitted it, so any of those tools
  BLOCKED until the worker's 60s `render_op_timeout_s` and returned `error: main-thread op timed out`. Symptom
  in a trace: a tool the model legitimately called (e.g. `probe_render`) failing 3× at exactly 60s apart.
  The pump now mirrors `ui.py`: `drain_bridge()` → `run_deferred_render()` → `pump_events()`. If you ever see
  a deferred render time out, this is the line.
- **Env order: set BEFORE importing shaderbox.** `SHADERBOX_DATA_DIR` (isolation — never pollute the real
  library/creds; also the RESUME seam) + the two MESA overrides (`MESA_GL_VERSION_OVERRIDE=4.6` /
  `MESA_GLSL_VERSION_OVERRIDE=460`, for `#version 460` on V3D) are set at the TOP of
  `scripts/dogfood/harness.py` before the shaderbox imports. A caller-set `SHADERBOX_DATA_DIR` WINS — so on
  resume you MUST pass it on the command line, never assign it in-script after import. That env block runs
  when `scripts.dogfood.harness` is imported, which `from scripts.dogfood import DogfoodHarness` triggers
  lazily (PEP 562 `__getattr__`); `import scripts.dogfood.analyze` / `.judge` deliberately does NOT touch
  it — they load no GL and cut no run dir.
- **🔴 `COPILOT_CONFIG` overrides must be set AFTER `DogfoodHarness.create()`, not before.**
  `ProjectSession.__init__` calls `integrations_store.copilot.apply_limits()`, which pushes the
  persisted integrations.json limits onto the shared `COPILOT_CONFIG` — clobbering any pre-create
  override. To force a limit for a run (e.g. a low `clean_edit_hard_streak` to provoke the churn
  hard-stop, or a low `max_iterations`): `h = DogfoodHarness.create()` THEN
  `COPILOT_CONFIG.clean_edit_hard_streak = 3`. A pre-create assignment silently runs with defaults
  (verified 2026-06-16 — cost a wasted turn). NOTE: a well-behaved cheap model self-limits on the soft
  nudge and won't naturally spiral, so the hard-stop / max_iterations / forced-turn-end paths REQUIRE
  such an override to exercise live.
- **Resume = same project_dir + same SHADERBOX_DATA_DIR.** `create(project_dir=<existing>)` skips seeding,
  reloads the shaders, restores the conversation from `<project_dir>/copilot/conversation.json` (zero LLM
  calls). The data dir (lib + integrations) is separate and env-only — both must point at turn 1's dirs or
  the resume is half-restored. `dump`'s JSON echoes both paths so you copy them forward without bookkeeping.
- **`dump` uses its own cursor.** `drive_until_idle` advances the PRINT cursor; `dump` slices on a SEPARATE
  cursor, so `new_messages` reports the turn's output even after the prints. Don't expect `dump` to re-emit
  the restored conversation on a resume — that's already-seen by design.
- **EGL context is already current after creation** — no `make_current` call; `Node`/`Canvas` pick it up
  via `moderngl.get_context()`. (moderngl's stub mistypes `backend=` — the one sanctioned `# type: ignore`.)
- **The `GLFWError: not initialized` warning is benign** — `core.py` reads `glfw.get_time()` for the default
  `u_time` (returns 0.0, the static t=0 frame we want). The harness installs a no-op glfw error callback.
- **White-on-transparent renders eyeball as BLANK (all white).** The Read tool / viewers flatten alpha onto
  white, so a sticker-style render (the flagship pattern) looks empty. Composite onto a dark background
  before judging: PIL `alpha_composite` onto e.g. (25,25,40), save, THEN Read. The render facts' `ink %`
  tells you whether there is alpha-carried content worth compositing.
- **Large canvas + many renders WITHOUT a per-frame `texture.read()` goes blank on V3D.** A Mesa/V3D
  driver quirk (NOT a script-engine or harness bug): rendering to a ≥256px canvas hundreds of times and
  reading the FBO only AT THE END yields a near-empty framebuffer (mean alpha ~7). The engine wrote every
  uniform correctly throughout — it's the accumulated GPU frame that's lost. Fix in a direct-engine driver:
  call `node.canvas.texture.read()` (or a flush) EACH frame, not just on the frames you keep (mean alpha
  jumps to 255). Cheap at the sizes dogfood uses; just don't batch the reads. (Found 2026-06-13 hudgame
  scene, 512px × 330 frames.)
- **🔴 `GLError 1282 (invalid operation) glUseProgram(0)` is a REAL pipeline bug, not harness noise.** It
  fires sporadically on bridge-marshalled create_node/write_shader (the persist→render path) under the
  standalone context — the same headless GL-quirk as node teardown. The copilot RECOVERS (retries), so a
  run still completes, but log it as a finding (a known headless-GL quirk; re-file in `todo.md` if it grows). Don't mistake it for a harness fault.
- **Multi-file read needs an UNSOLVABLE-without-reading task.** "Merge node A and B" is solved from the
  model's own knowledge — a cheap model won't bother to `read_shader` the references. To actually exercise
  multi-file read, the task must REQUIRE the other node's content (e.g. "use the EXACT color/constant from
  node X"). Otherwise the probe is inconclusive.
- **`session_cost_usd` accumulates across turns**; `state.last_turn` has `context_tokens`/`reply_tokens`/
  `cost_usd` per turn. The trace's `llm_response` events have per-iteration `usage: in=/out=/cost=`.
- **`h.render()` renders the CURRENT node only**, and `create_node(switch_to=False)` / edit-by-`target`
  do NOT move current — so after building or editing a node in the background, `h.render()` shows the OLD
  current node (e.g. you make Square, but the render is still Circle). To eyeball a non-current node
  WITHOUT spending an LLM turn: `h.session.set_current_node_id("<full-uuid>")` then `h.render()` (both
  GL-free, no `send`). The render path attr is `h._last_render_path` (underscore). NOTE: a uniform value
  set via `set_uniform` is in-memory-only until a project save, so a between-process render shows the
  file's inline default, not the tuned value.
- **Turns are engine-bounded since `turn_time_budget_s` (default 180s, `copilot/config.py`):** a
  slow reasoning grind force-ends with an honest final reply at the budget, so `timeout 300` on the
  command is enough again — if a turn still exceeds it, that's a REAL finding (a hung stream, not a
  long grind). Pre-budget history: debug-shaped correction turns legally ground past 10 minutes
  (14+ iterations x 30-70s reasoning streams; observed 2026-07-27).
- **🔴 ALWAYS wrap a turn process in `timeout` (`... timeout 300 uv run python -c …`).** A stalled LLM
  stream could leave the non-daemon copilot worker blocked, and interpreter `_shutdown` then hangs
  joining it — a process that never exits, never dumps. The per-delta stream cancel + the 120s client
  timeout (committed 2026-06-14) bound this to ~2 min, but `timeout 300` on the command is the belt — a
  turn that exceeds it is a finding, not a wait. Diagnose a hang with `py-spy dump --pid <pid>` (the
  `_shutdown`/worker-in-`get()` stack pins it instantly — far better than guessing).
- **MP4 for iOS/iPad (WebM won't open there): `h.render_video_mp4(seconds=, fps=, size=)`.** Renders
  H.264 directly via `share_state.render_to` through the export-isolation seam (a stateful script
  animates from a clean __init__). The webm `h.render_video()` is the other deliverable; both set
  `_last_render_path`. Keep `seconds`/`size` small on V3D. (A FREE `RenderPreset` yields a stray ffmpeg
  `-s 0x0` broken-pipe — the method uses FIXED_DIMS; don't hand-roll a bare preset.)
- **A sample-frame STRIP is the cheapest visual motion check — use `h.render_strip(times, node_id)`,
  never a hand-rolled `render_at` loop.** One call renders each `t`, composites onto (25,25,40), adds
  a `t=` label + 4px gutter, and writes ONE horizontal sheet into the project's renders dir (so
  `dump()`'s `last_render_path` finds it). 🔴 The reason it's a method and not a loop: each sample is a
  fresh-behavior REPLAY through the export-isolation seam, stepping frames `0..round(t*fps)`. A
  `render_at` loop LIVE-ticks one persistent script instance once per sample, so a stateful integrator
  (anything using `ctx.dt`) samples a trajectory that never happened — the 05 bounce script only
  survived it by self-deriving dt from `ctx.t`. Canvas size is saved/restored around the calls
  (`render_at`'s `set_size` persists into node.json and would overwrite an agent's `set_canvas_size`).
- **For NUMBERS, not pixels: `h.script_values(times, node_id)`** — the `ScriptEngine.dry_run`
  passthrough returning `(t, {uniform: value})` per sample, with the live node left byte-identical.
  That's the logic axis's ground-truth check (a trajectory, a period, a state machine's phase) without
  spending an LLM turn or squinting at a render.
- **For pixel measurements: `scripts/dogfood/judge.py`** (GL-free, PIL+numpy) — `load_rgb`, `grid_cell`
  (split a grid render into cells), `region_diff`, `bright_centroid`, `color_mask_centroid`,
  `column_runs` (the blob counter), `farthest_bright_angle` (rotation direction; image y grows DOWN,
  so a rising angle = clockwise). Numbers out, judgment yours. Pixel measurement beat the human eye
  twice in the cornerstone pilot (a false "bunched" call; a model's "still broken" claim disproved) —
  reach for it before arguing with a render.

## 3. Reading the trace (the context/token analysis)

`h.trace_path` → a plain-text transcript. Per turn it logs: `turn_start` (user_text + history +
eager_tools), each `llm_request` (the FULL messages array — system prompt + project map + working set +
the native `tools=` block — + max_tokens), each `llm_response` (finish_reason + text + tool_calls +
`usage: in/out/cost`), each `tool_call` (name + args + ok + result), `turn_done` (summed usage).

**Run `uv run python scripts/dogfood/analyze.py <data_dir> --scenario <name>` to auto-extract** tool
coverage, the per-turn iteration/token/cost table, recoveries, the token-growth shape, AND the honesty
axis's AUTO half — paste its markdown block into the report §6/§7 instead of hand-summing. (The
per-section context_breakdown — system prompt vs project map vs working set vs `tools=` block — remains a
separate deferred trace event, not yet automated; for that, split one `llm_request` block by hand,
~chars/4.)

**The honesty axis is YOURS to judge.** The copilot has no eye — it measures (the `render:` facts line)
and you look. The analyzer only hands you the LIMIT-FORCED turns (a `cutoff=` / giveup turn glyphs
⚠️/🔴 because its reply was written under an engine stop); those are where blind summaries hide, so
check them first. Then compare each claim against (a) the measured facts in the trace and (b) your own
eye on the render.

**`analyze.py <data_dir> --dialogue` prints the run's user-visible dialogue** — paste it into report §2.
Source is the UI chat store (`<project>/copilot/conversation.json` `messages`, resolved from the run's
dumps; `--project` overrides), i.e. what the user SAW, NOT what the model saw. `error`-role rows print
as **Copilot [engine]:** — an `[engine]` limit note IS a visible reply and dropping it erases the
under-claim evidence. On a context-wipe run `clear_context()` re-saves an EMPTY store to the same path
(it never deletes), so the printer falls back to the per-turn dumps whenever the store holds fewer rows
than the dumps recorded; the pre-wipe text also survives in
`<project>/copilot/archive/conversation_<stamp>.json`.

**Load-bearing token note:** the `turn_done` `in=` is the CUMULATIVE billed input (the SUM of every
iteration's input — e.g. 68k on the 4-node read turn); the real per-turn CONTEXT size is the max
per-iteration `in=` (analyze.py's `peak_iter_in_tokens`, ~10k on that turn). Don't report the cumulative
figure as "context size" — it's the cost driver, the peak is the context-size driver.

## 4. The report (template + analyzer flow)

The report is half AUTO (filled by the analyzer from logs — you never hand-sum), half HUMAN (your
judgment), structured as the **six axes**: `fidelity` · `motion` · `logic` · `honesty` · `process` ·
`code`. **ONE report per SCENARIO** (a run's data dir holds one scenario). Flow:

1. Copy `scripts/dogfood/REPORT_TEMPLATE.md` → `ai_docs/features/NNN_dogfood_report_<run>.md` (durable,
   roadmap-linked finding — stays in `ai_docs/features/`, NOT under `scripts/dogfood/`).
2. Run the analyzer to fill the **AUTO slots** (run label/model/turns/cost, per-turn table, render
   list, tool-coverage table, cold tools, token range/peak, cost range, token-growth, recovery summary,
   and the whole honesty block):
   ```
   uv run python scripts/dogfood/analyze.py <data_dir> --scenario <scenario name> \
       --template scripts/dogfood/REPORT_TEMPLATE.md \
       --report-out ai_docs/features/NNN_dogfood_report_<run>.md
   ```
   (Pass `--model <id>` if the run used a non-default model not recorded in the data dir's
   `integrations.json`.)
3. Write the **9 HUMAN sections** by hand — the things a log can't give you:
   - **§1 Verdict** — mechanism works Y/N, overall conclusion.
   - **§2 Dialogue** — paste `analyze.py <data_dir> --dialogue` verbatim (§3). Never retype it.
   - **§3 fidelity** — the scenario's checklist as a table (`| check | PASS/FAIL | measurement |`), each
     row citing the artifact that decided it, plus a counted "X of Y sub-requirements landed".
   - **§4 motion** — verdicts against the scenario's STATED ground truth, each citing its measurement (a
     `render_strip` sheet, a `judge.py` number).
   - **§5 logic** — same, off `script_values` / analytic truth.
   - **§6 honesty** — the limit-forced turns are filled for you; YOU write the two claim lines (claims
     vs the measured facts, claims vs your own eye). The agent CANNOT see its render, so any "it looks
     …" claim is unsupported by construction.
   - **§7b Per-render eyeball** — open each PNG with Read; correct/wrong, quadrants, did a tuned uniform
     visibly change anything. (NOT automatable — you have to look.)
   - **§8 code** — read the run's FINAL sources (shader + script) end-to-end as a CODE REVIEWER and fill
     the `| aspect | verdict | evidence |` table (dead code · duplication · structure & naming ·
     complexity vs task · tool choice), each verdict quoting line evidence from the final source. Record
     what the end-of-mission SWEEP turn (§1a) deleted as the edit-sediment number. This is the
     first-class axis: the copilot's job is EXCELLENT CODE; the visual call is the human's.
   - **§9 TODOs**, split: (a) improve the COPILOT/agent, (b) improve the DOGFOODING framework/harness/
     skill, (c) improve the LIBRARY.

The template's inline comments mark every `{{AUTO:...}}` vs `{{HUMAN:...}}` slot. Treat full reachable-tool
coverage (this skill's §1a) as a run goal — the report's §7c coverage table makes a thin run visible.

## 5. Clean up

No throwaway driver to delete (the one-blocking-call-per-turn shape has none). All run artifacts live under
`scripts/dogfood/runs/` (gitignored). To free disk between runs:
`rm -rf scripts/dogfood/runs/{data-*,proj-*,*.json}` (regenerable — the harness recreates `runs/` on the
next run; the dumps are the stray `*.json`). NOTE: these data dirs hold the LIVE OpenRouter key in their
`integrations.json`, so purging them is also key hygiene. Keep the report (durable, in `ai_docs/features/`);
if a reviewer needs your trace, copy the specific `trace_path` somewhere durable before the purge. The
harness + analyzer + template + scenarios + this skill stay. File prioritized findings into `todo.md` with
concrete triggers.

## 6. Improve this skill

This is a LIVING skill. Each run, if you hit a new gotcha or the report format wants a new section, ADD it
here so the next run is smoother. The maintainer wants the dogfooding itself to get more convenient over
time — the report's "improve the DOGFOODING framework" TODO bucket (report §8 (b)) is where those
findings start, and they flow back HERE (the skill) or into `scripts/dogfood/analyze.py` (the analyzer).
