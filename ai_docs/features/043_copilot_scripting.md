# 043 — Copilot scripting (the agent authors `script.py`)

Give the in-app copilot the ability to author a node's Python brain script
(`nodes/<id>/scripts/script.py`, feature 048) the same way it already authors GLSL shaders — so it can
build an ANIMATION end to end: declare uniforms in the shader, then write an `update(self, ctx) -> dict`
that drives them over `ctx.t`, and SEE (headlessly, on its token stream) whether the script compiled,
which uniforms it drives, and whether the result is actually moving.

This is the feature named NEXT on the roadmap after 048. The shape is **mirror the shader tools** — the
copilot is already built around content-addressed editing + a live working set + per-turn checkpoint
rollback; scripting plugs into all three. The whole design is driven by ONE practical test, not paper
aesthetics: *can the driver (the dogfood harness, a human-or-agent steering the copilot turn by turn)
take the agent to an interesting animation autonomously, and never have to eyeball a frame to know it
worked?*

Pre-spec research: three independent agent sweeps over the shader-tool seams, the scripting public
surface, and the checkpoint/prompt seams (2026-06-14). They converged; their findings are the Design
decisions below. The actor-model constitution (`.claude/skills/copilot-llm-agent-design`,
`conventions.md ## Design decisions` copilot bullets) governs every choice.

## The practical walkthrough this feature must satisfy

Concrete goal an agent should deliver autonomously: *"a glowing circle that pulses (radius oscillates)
and drifts across the screen in a figure-8, leaving a fading trail."* The shader exposes `u_center vec2`,
`u_radius float`, `u_glow float`. The agent's path, with today's coverage marked:

| # | Agent action | Tool | Today |
|---|---|---|---|
| 1 | create/read the shader node, see its uniforms | `create_node`/`read_shader` | EXISTS |
| 2 | write GLSL: SDF circle at `u_center`, radius `u_radius`, glow `u_glow` | `write_shader` | EXISTS (probe facts @t=0) |
| 3 | (optional) hand-tune a uniform to sanity-check the look | `set_uniform` | EXISTS — **rejects a script-driven uniform, pointing at `scripts/script.py`** |
| 4 | write the brain: figure-8 path + pulse over `ctx.t` | `write_script` | **MISSING** |
| 5 | read the script back (to edit it, or inspect the stub) | `read_script` | **MISSING** |
| 6 | confirm it compiled + drives `u_center,u_radius,u_glow` | (folded into write/read result) | **MISSING** |
| 7 | confirm the shape is in a DIFFERENT place at t=0 vs t=1 (motion) | multi-t render facts | **MISSING — the make-or-break** |
| 8 | render a small deliverable video | `render_video` | EXISTS (but returns no pixel facts) |

Steps 1-3 and 8 are covered. **Steps 4-7 have zero tool surface today** — the copilot package has no
script method at all (`grep` doesn't even walk `script.py`); the only place the agent meets a script is
the `set_uniform` reject at `backend.py` that tells it to *"edit the node script at
nodes/<id>/scripts/script.py instead"* — a file no tool can reach. That dead-end is the proof the gap is
real and load-bearing.

## Goal

The copilot can READ, WRITE, and EDIT a node's `script.py` (content-addressed, mirroring the shader
tools), SEES the script live in its working set every iteration, gets SYNCHRONOUS feedback after a write
(compile error | the driven-uniform set | per-key coercion errors | multi-t MOTION facts proving the
animation moves), and the whole write is checkpoint-revertible. After this lands, the dogfood driver can
steer the agent to an autonomous animation and read "it's animating" off the token stream — no human eye
in the verify loop.

## Out of scope

- **Script CREATE / DELETE / play-stop / node-execution-control tools.** The agent authors the brain;
  the brain plays every uniform by default (export ignores `stopped`), so for autonomous animation the
  agent never needs to stop a uniform or toggle node execution. `write_script` create-or-overwrites
  (no separate `create_script` tool), so no standalone create. **Trigger:** first dogfood/user scenario
  where the agent genuinely needs to STOP a uniform it drives (a "leave this one manual" ask) — add a
  `stop_uniform`/`play_uniform` pair then, mirroring the UI affordance.
- **`edit_script` (a substring edit tool) — ADDED 2026-06-14 on the maintainer's full-symmetry call**
  (reversing the original drop below). The script surface now MIRRORS the shader surface exactly:
  `read_script`/`write_script`/`edit_script` ↔ `read_shader`/`write_shader`/`edit_shader`. `edit_script`
  reuses the shared `_splice` + the 0/1/N-match contract, but matches PLAIN TEXT (`_plain_text_spans`),
  not the GLSL `token_match` (a script is Python). It routes through the same write tail
  (`_apply_script_text`) so an edit and a write give IDENTICAL feedback (compile / drives / motion).
  (It was briefly dropped as per-iteration tool tax; the maintainer's symmetry argument — asymmetry is
  itself a cost, an agent can't tweak a grown brain cleanly — overruled it. Decision 1 below retains the
  original drop rationale for the record; this bullet is the live decision.)
- **A scripts library / cross-node script reuse for the agent.** Same trigger as 048's: a script picker
  when whole-node reuse is wanted. Not now.
- **`render_video`/`render_image` returning pixel facts.** The motion fact lives on the `write_script`
  clean-compile result (cheaper — at authoring time, before a slow video render). Attaching a
  first/last-frame fact pair to `render_video`'s result is a clean follow-on but not required for the
  walkthrough. **Trigger:** a dogfood run where the agent renders a video and then can't tell if the
  deliverable is right (the write-result motion fact proved insufficient because the video sampled a
  different window).
- **Cross-uniform / cross-node shared script state.** Still 041/044's deferrals (`todo.md`) — one brain
  per node, `self.*`, no shared channel.
- **The agent ADDING uniforms from the script.** A brain writes VALUES only; it cannot declare a uniform
  or change a control's look. The agent declares a new uniform in the SHADER first (`edit_shader`/inline
  default), then drives it from the script. This is stated in the prompt block, not enforced by new code
  (the engine already silently drops an orphan key — a stub-listing guard already excludes it).

## Design decisions (locked)

### A. The tool shape

1. **A dedicated `read_script` / `write_script` PAIR — NOT an overload of `edit_shader`/`write_shader`
   via a `target:"script:"` scheme, and NO `edit_script` in this wave.** The decision is grounded in the
   walkthrough + the actor model, not code aesthetics:
   - **Why a separate pair, not `target:"script:"` (the rebuttal a reviewer WILL ask, given the existing
     `target:"lib:"` precedent):** the lib overload addresses the SAME artifact kind — `lib:` and a node
     target are both GLSL, both compile through the GLSL lexer, both return `EditResult`/`CompileErrorInfo`;
     the `target:` discriminator selects WHICH GLSL file, not which LANGUAGE. A script diverges in every
     dimension the result cares about: Python not GLSL (so the §1 GLSL token lexer/matcher does NOT
     apply — the content-addressed match must switch matchers), a `ScriptError(kind ∈ {compile, runtime})`
     not a `CompileErrorInfo`, plus a "now drives u_center, u_radius, u_glow" concept with NO GLSL analogue
     (a shader edit can't be "driving uniforms"). A `target:"script:"` overload would force a UNION return
     type (`EditResult | ScriptWriteResult`), a target-kind branch in every shader-result path, AND a
     matcher-language branch inside the content-addresser — the guard-pile the actor model's "structural
     impossibility over guard-piles" warns against. The clean structural split is the separate pair.
     (Note: the agent routes on tool NAME across heterogeneous result shapes fine today — `grep`/`read_lib`/
     `read_shader` all differ — so the rationale is artifact-kind divergence, NOT "the agent gets confused
     by a traceback.")
   - **The tool name matching the artifact makes the agent's walkthrough smoother:** it picks
     `write_script` when it wants to drive uniforms over time, `write_shader` when it wants to change
     pixels-from-uniforms — the same mental split 048 created in the UI (open-shader vs open-script are
     two affordances, 048 decision 12).
   - **`edit_script` is DROPPED from this wave** (devil's-advocate HOLDS): a brain is ~30 lines, the whole
     file is in the working set every step, and `write_script` whole-file covers every mutation. An
     `edit_script` description is re-billed every iteration (the most expensive description shape in the
     registry — `edit_shader` is ~200 tok) and earns NOTHING the agent can't do with `write_script` — it
     even resets `self.*` identically (both paths recompile → fresh `__init__`), so it isn't even a
     state-preserving edit. Per the actor model's "tool count must not grow casually" + "fewer tools is
     better if it earns nothing": ship `read_script` + `write_script` only. **Trigger to add `edit_script`
     later:** a dogfood run shows the agent thrashing on whole-file rewrites of a brain that grew past
     trivial.

2. **`write_script` create-or-overwrites; `read_script` is read-or-stub.** `write_script(node, new_text)`
   writes the complete `script.py`; if the node has none, it is created (the file's existence IS the
   binding, 048 — no activate step). `read_script(node)` returns the script source line-numbered; if the
   node has NO script, it returns the freshly-generated STUB WITHOUT persisting it — so the agent sees the
   available uniforms + the shape to return, then `write_script`s a real body. The stub the AGENT reads
   carries ONE un-commented `math.sin(ctx.t)` worked example (decision G), distinct from the UI's
   no-op-by-default stub — the actor copies verbatim, so give it a line that ANIMATES to copy.

3. **MUST-HAVE tool set (the agent is blocked without each):** `read_script`, `write_script`, and the
   MOTION fact on `write_script`'s clean-compile result (decision E). No `get_brain_status` standalone
   tool (the status fact is folded into read/write results — a lazy model won't call an optional inspect
   tool, per the actor model).

### B. The engine feedback entry point (the make-or-break)

4. **The engine needs a single "compile + dry-tick" entry point, because a script's facts are
   tick-gated.** The split (verified against `engine.py`):

   | Fact | From `reload()` alone (no tick)? | Why |
   |---|---|---|
   | compile error (syntax / no subclass / bad `update` / raising `__init__`) | **YES** | `reload` builds `PythonBehavior` synchronously + stores `behavior.error` under `(node_id, "script.py")`; `get_brain_status().sentinel_error` reads it |
   | driven set (`last_driven`) | **NO** | populated only inside `_tick_brain`; empty pre-tick |
   | per-key coercion errors + orphan/typo keys | **NO** | computed only in the `_tick_brain` write loop |
   | runtime error (`update` raises / returns non-dict) | **NO** | caught only inside `_tick_brain` |
   | the driven VALUES per t (the motion signal) | **NO** | written only in `_tick_brain` |

   So `reload()` (which the live `write_script` already runs at persist time) gives the compile verdict
   synchronously; everything else needs `update` to actually RUN, which headless has NO frame loop to do
   (`glfw.get_time()`=0, `_render_facts_for` renders once at t=0 without ticking the script — so the
   working set alone would show "compiled clean" with an EMPTY driven set, the agent hallucinating nothing
   is driven). So a synchronous dry-tick is load-bearing, NOT redundant with the next-iteration working
   set: the working set carries the standing COMPILE error; `dry_run` carries the tick-gated facts. They
   partition cleanly by tick-gating.

   **Add `ScriptEngine.dry_run(node_id, node, sample_times) -> ScriptProbe`** that:
   - does NOT call `reload()` (reload mutates live `scripts.brain`/`scripts.errors` and is mtime-gated —
     it is the LIVE mutator, not a force-recompile; calling it would violate the no-corruption guarantee).
     The compile verdict is read from the ALREADY-LIVE state (`write_script` persisted + reloaded the file
     before calling `dry_run`, so the live `errors[(node_id, "script.py")]` is current). If a compile error
     is live → return it, no tick.
   - else ticks ONE FRESH isolated brain (via `fresh_behavior_for`) **continuously through the sample
     frames, advancing `self.*`** (decision E) — NOT N independent single ticks. Records into call-local
     sinks: the driven set, per-key errors, orphan/skipped keys, and the driven uniforms' VALUES at each
     sample frame.
   - returns `ScriptProbe(compile_error, driven, per_key_errors, orphan_keys, samples)` where `samples` is
     per-sample-time `(t, {name: value})` for the driven uniforms.

5. **`dry_run` reads GL (it is bridge-marshalled, NOT worker-safe) and must not corrupt live state.**
   The "value-sampling is GL-free" claim is FALSE: `_tick_brain` reads `node.get_active_uniforms()`, which
   iterates the live GL program (a current-context read), and coercion reads `dimension`/`array_length`/
   `gl_type` off those GL `Uniform` objects. So the WHOLE `dry_run` marshals through the bridge `_on_main`
   (the value half AND the motion-render half) — mirroring how `set_uniform` already rides `_render_facts_for`
   on the main thread. To keep the live node untouched, `_tick_brain` gains a `values_sink: dict | None`
   param: when provided, every `node.uniform_values[name] = …` write (the success write, the per-key
   freeze, `_freeze`) redirects into `values_sink`, and the freeze-fallback READ reads
   `values_sink.get(name, live_snapshot.get(name))` against a one-shot read-only snapshot taken at entry —
   so the live `node.uniform_values` is NEVER written and NEVER alias-mutated (safer than snapshot+restore,
   which a future in-place `Array` write through a shared list reference would defeat). The error/cache/warn
   sinks are throwaways exactly as `tick_export` already passes (`{}, {}, set(), set(), set()`). After
   `dry_run` returns, the live `errors`/`last_driven`/`uniform_values` are byte-identical to before — the
   gating canary in Manual-verification.

### C. Working set — the script is in context, always

6. **`WorkingSetView` carries the node's script live, every iteration** (the maintainer's explicit call:
   "скрипт стоит положить в контекст сразу"). Add `script_listing: str` (cat -n of `script.py`; "" = no
   brain) + `script_errors: list[CompileErrorInfo]` to `WorkingSetView`. `_render_working_set_member`
   appends a `=== <node> SCRIPT (scripts/script.py) ===` sub-section after the node's uniforms/errors when
   `script_listing` is non-empty. The read is a plain-text `read_text` + a dict-membership error lookup —
   **GL-FREE and worker-safe** (it rides inside the existing `read_working_set` `_on_main` block only
   because the *shader* recompile there needs GL; the script additions add no main-thread requirement).
   This is the highest-leverage move: the agent SEES both shader and script source live, edits either
   directly, never needs a separate read for the current node. (The shader-side rule "the CURRENT node is
   already in your working set — edit directly, no read needed" now covers the script too.)

7. **The script error in the working set keys on the same data as the UI tint:** `(node_id, "script.py")`
   in `script_engine.errors` for the sentinel + `get_brain_status().soft_errors` for orphan/typo keys.
   The working-set script_errors render the sentinel (the compile/run freeze) — the per-key coercion
   errors on real uniforms ride the `write_script` RESULT (decision 4/B), not the standing working set
   (they are a write-time fact, not a persistent state).

7a. **The working-set `uniforms:` row marks a script-driven uniform instead of showing a phantom default.**
   `_format_uniforms` reads `node.uniform_values.get(name, u.value)` — but the copilot path never ticks the
   live brain (no `session.tick` in `copilot/`), so a script-driven uniform's row shows its MANUAL default
   (a number the agent's own write said it "drives" — a contradiction the agent burns a turn re-reading).
   Fix: when `name in script_engine.script_driven_uniforms(node_id)`, render the row as
   `u_radius float = <driven by script.py>` instead of a frozen scalar. This is the working-set twin of the
   048 UI blue-name marker (a number is the wrong mental model for a script-owned slot) and it doubles as
   the discoverability fix — the row tells the agent which uniforms it already drives vs which are manual.
   One membership check in the existing loop; GL-free.

### D. Checkpoint — the rollback correctness fix (real work, not a formality)

8. **`_capture_node` does NOT capture `scripts/script.py` today — and worse, a revert would DELETE the
   script.** Verified against `checkpoint.py`/`ui_models.py`/`revert.py`: `snapshot_node` re-serializes
   the in-memory `UINode` via `UINode.save` (which writes only `shader.frag.glsl` + `node.json` + sampler
   `media/`/`textures/` — NO `scripts/`), and restore is a full-dir swap (`copytree` snapshot →
   `rmtree` live → replace). So:
   - reverting a turn that EDITED an existing script → the swap deletes the script entirely (the snapshot
     never carried it) — WORSE than a no-op, it loses the user's brain;
   - reverting a turn that purely CREATED a script (no other node touch) → no `snapshotted_nodes` entry,
     so the swap never runs and the script SURVIVES the revert (escapes undo).

   This resolves the `todo.md` "copilot turn-rollback" deferral's 043 clause: it is REAL new work, not
   mostly-solved. **Add a `_capture_script(node_id)` seam** called in every script-mutating tool's
   `_on_main` BEFORE the write (mirroring `_capture_node` at the `set_uniform`/persist seams). Shape
   (mirrors the lib create-vs-edit asymmetry already handled in `checkpoint.py`/`revert.py`):
   - if the node had a pre-existing `script.py`, copy it into the per-node snapshot dir
     (`turn_dir/<node_id>/scripts/script.py`) so the full-dir swap restores it for free;
   - if the script did NOT exist (a pure create this turn), record a per-script "created" marker
     (analogous to `created_libs`) so revert DELETES `nodes/<id>/scripts/script.py` instead of relying on
     the swap.
   - Ordering hazard (confirmed safe): `UINode.save` uses `mkdir(exist_ok=True)` + additive file writes
     (it does NOT wipe the dir), so whether `_capture_node` or `_capture_script` writes the per-node
     `turn_dir/<node_id>/` first, the other's files survive and the full-dir swap restores both. Each
     capture must be independently first-touch-wins guarded (`_capture_script` checks its own marker;
     `snapshot_node` already does).
   - Double-revert guard: a script created on a node ALSO created this turn — the `created_nodes` revert
     loop already trashes the WHOLE node dir (incl. `scripts/`), so the `created_scripts` marker must SKIP
     recording when `node_id in cp.created_nodes` (mirroring `record_deleted`'s `if node_id in
     self.created_nodes: return`), and the `created_scripts` revert path must be path-absent-graceful (like
     `_revert_created_lib`'s `if not path.exists(): return False`).

   **This lands as a STANDALONE commit BEFORE the feature** (decision per the devil's-advocate + dev_flow):
   it is a live data-loss bug that exists TODAY independent of the copilot (a human edits a script via the
   UI, a copilot turn touches the node, revert deletes the script), it is testable with NO copilot tool,
   and landing it first de-risks the high-blast-radius feature diff. See `## Commit plan`.

### E. Motion verification — the headless animation fact (make-or-break)

9. **The motion VERDICT comes from the script's sampled VALUES across t (GL-free, exact), NOT from a
   pixel-bbox diff; ONE pixel render corroborates "it reaches the screen / is not FLAT".** Today no
   fact-bearing render lets the agent see motion: `render_image` is hard-coded to t=0 (`backend.py`
   `render_to(..., 0.0, ...)`); `render_video` returns only a path + "you can't see the result"; the
   edit/write probe (`_render_facts_for`) renders at `glfw.get_time()` (0.0 headless) and does NOT tick the
   script. So the agent CANNOT self-confirm "it's animating" — it would hallucinate motion.

   **The signal is the VALUES, not the pixels.** `dry_run` already returns the exact driven values per t
   (`samples`); `coerce_one` emits plain comparable scalars/tuples/lists. Comparing them across t is a dict
   diff — GL-free, exact, and it catches the cases bbox is BLIND to:
   - a **pulse in place** (`u_radius` oscillates, `u_center` fixed) — bbox center doesn't translate, so a
     pixel-bbox diff reads STATIC on the spec's OWN headline goal ("a glowing circle that pulses");
   - a **color cycle** (`u_color` changes, geometry fixed) — bbox is constant by construction;
   - **phase-aliasing** — three evenly-spaced pixel samples of a periodic shader can hit near-equal frames
     and false-STATIC; the value-diff would need EVERY driven value to alias simultaneously.

   So the verdict is per-uniform value motion, which also tells the agent WHICH uniform is static when it
   expected motion (a debugging signal bbox can't give):

   ```
   ok -- script compiled clean, drives u_center, u_radius, u_glow
   values@t=0.0: u_center=(0.30,0.50) u_radius=0.20 u_glow=0.80
   values@t=0.5: u_center=(0.55,0.42) u_radius=0.35 u_glow=0.80
   values@t=1.0: u_center=(0.42,0.62) u_radius=0.20 u_glow=0.80
   render@t=0.5: ink 9% | bbox x 0.40-0.64, y 0.30-0.54 | luma ...
   -> u_center, u_radius CHANGE across t (ANIMATING); u_glow constant; visible (ink 9%)
   ```

   **Value comparison rule (the verdict mechanics):** `coerce_one` emits scalars (`float`/`int`), tuples
   (Vec2/3/4), lists (flat Array), and lists-of-tuples (Array of vecs). A scalar epsilon is NOT enough and
   bare `==` false-positives on `sin` float-jitter, so the verdict uses a RECURSIVE epsilon compare
   (structurally like `behavior.py::_all_finite`): descend tuples/lists element-wise, `abs(x-y) > eps` at the
   leaves, shape-mismatch = differ. A uniform "changes" if ANY sample pair differs by > eps; "ANIMATING" =
   any driven uniform changes. `eps` is `COPILOT_CONFIG` (resolve the "exact diff" wording — it is
   epsilon-exact, not bit-exact).

   **The single pixel render does a DIFFERENT job, which is why it stays:** it answers "did the driven
   values actually produce visible ink, or is the shape off-screen / the shader ignores this uniform /
   it's FLAT?" — the one honesty case a value-diff alone gets WRONG (values move, but the GLSL never reads
   `u_center`, so the frame is static). The render is owned by the BACKEND, not the engine (the headless
   engine cannot import `Canvas`/`render_facts` — its own contract): `dry_run` returns `samples` ONLY (no
   render); the backend reads the mid-sample `{name: value}` dict, renders ONE frame with those values, and
   calls `render_facts` UNCHANGED for the FLAT/ink/bbox line. To keep the decision-5 canary green, the
   render REBINDS the dict reference rather than mutating it in place:
   `saved = node.uniform_values; node.uniform_values = {**entry_snapshot, **mid_samples}; try:
   node.render(t, probe_canvas) finally: node.uniform_values = saved` — the live dict OBJECT is never
   touched (so no Array-aliasing hazard), and after the finally the node is byte-identical. This does NOT
   route through `_make_export_isolation` (a frame-loop hook swapper that writes live `uniform_values` +
   ticks a SECOND fresh brain — the wrong primitive); what's reused from export is the CLOCK
   (`EngineContext` built with `EXPORT_MOUSE`, `dt=1/motion_fps`, `frame=step`), not the isolation manager.
   The whole probe marshals through `run_on_main(..., timeout=COPILOT_CONFIG.render_op_timeout_s)` (the
   render timeout, NOT the 5s default — 12 ticks + a V3D render can exceed 5s; mirrors render_image/video).
   So the cost is ONE GL probe per clean write (cheaper than 3), and the verdict is a 4-way honest signal:

   - **drives 0 uniforms** — `update` returned `{}` / only orphan keys → "drives NOTHING — nothing
     animates" (the no-op trap, decision G; must be loud, never silent).
   - **values constant across t** — "STATIC across samples — did you mean to use ctx.t?"
   - **values change AND ink present** — "ANIMATING".
   - **values change BUT render FLAT/blank** — "values animate but nothing is visible — the shader may
     ignore these uniforms or the shape is off-screen" (the case pure value-diff would falsely call
     ANIMATING — this is exactly what the one pixel render is FOR).

10. **The sampling ticks ONE fresh brain CONTINUOUSLY through the sample frames so `self.*` accumulates —
    integrator-safe.** The crux bug in N-independent-single-ticks: the export video loop (`core.py`) ticks
    frame-by-frame with `dt = 1/fps`, accumulating `self.*`; a fresh brain ticked ONCE at t=0.5 has
    `self.x = dt`, not half a second of accumulation. So an integrator (`self.x += ctx.dt`, the spec's own
    "drifts in a figure-8" goal) sampled by three discontinuous single ticks reads `x=dt, x=dt, x=dt` →
    **false STATIC on a script that clearly animates** — the exact make-or-break false-negative. The fix:
    `dry_run` steps ONE fresh brain through the frame sequence `0, 1, … round(t_max * fps)` exactly as the
    export loop does, captures the driven values at the frames nearest each sample time (at 12fps:
    t=0,0.5,1.0 → frames 0,6,12). `dry_run` does NOT render — it returns `samples` only; the backend owns
    the ONE corroborating render (decision 9), reproducing the mid-sample frame's values. The dry-tick clock
    is PINNED to the export clock — `dt = 1/motion_fps`, `frame = the step index`, `mouse = EXPORT_MOUSE`
    (the constant from `context.py`, NOT the `_make_export_isolation` manager) — so the sampled values are a
    faithful preview of what
    `render_video` (which rides the same loop) will produce; a different dt would make the fact lie about
    the deliverable. `motion_fps` is `COPILOT_CONFIG` (default a modest 12 — the probe steps ~12 frames to
    reach t=1.0, cheap, value-only for all but the one render frame). Sample times stay deterministic
    config constants (`(0.0, 0.5, 1.0)`) — never derived from the script (a derived set is non-reproducible
    across edits, defeating "did THIS edit change the motion").

### F. Frozen-mouse + ctx-API the agent is blind to

11. **`ctx.mouse` is FROZEN at center (0.5, 0.5) on export AND in the headless probe — the agent must be
    told, or a correct mouse-animation reads STATIC.** The motion probe ticks through export-isolation,
    which feeds `EXPORT_MOUSE = MouseState(0.5, 0.5)` (`context.py`). An agent that authors a valid
    `u_center = Vec2(ctx.mouse.x, ctx.mouse.y)` (a plausible "follow the cursor" reading) probes identical
    at every t → "STATIC" → the agent "fixes" a non-bug and burns turns. The SCRIPTING prompt block
    (decision 13) MUST state: for AUTONOMOUS animation drive from `ctx.t`, not `ctx.mouse`; `ctx.mouse` is
    frozen in the probe (live-only). This is a fact the agent is structurally blind to (it's in
    `context.py`, not its stream) — the highest-blast-radius single prompt line.

### G. The stub teaches motion + the no-op fact

12. **The stub the AGENT reads carries ONE un-commented `math.sin(ctx.t)` worked example; a drives-0
    result is a LOUD fact.** Today `brain_stub_for` emits every example COMMENTED OUT with zero/origin
    constants and no `ctx.t` anywhere (`math` imported, never used). The actor copies verbatim → it returns
    the empty dict (no-op, "I'm done") or uncommented static constants (drives, but STATIC). Two fixes:
    - **`read_script`'s stub for the agent** wraps `brain_stub_for`'s output with ONE copy-ready animating
      example block (`pulse = 0.2 + 0.1*math.sin(ctx.t*2.0)`; `cx = 0.5 + 0.3*math.sin(ctx.t)`; `return
      {'u_radius': pulse, 'u_center': Vec2(cx, cy)}`) — a verbatim `math.sin(ctx.t)` pattern to copy. Do
      NOT change `brain_stub_for` itself (the UI depends on its no-op-by-default contract — a fresh node
      must drive nothing); the agent-only example is added in `read_script`'s stub path. The stub is
      read-only/unpersisted, so it can't leak into a `write_script` (the agent sends its OWN new_text); the
      prompt frames the stub as a reference to ADAPT, not a body to save.
    - **The drives-0 result fact** (decision E's first branch): a clean script that drives nothing returns
      an explicit "compiled clean, but drives 0 uniforms (update returned an empty dict). Nothing animates
      — return {name: value}" — a distinct loud line, never the ABSENCE of a "drives" clause.

### H. Prompt

13. **One STATIC `SCRIPTING (node brains)` policy block in `_SYSTEM_PROMPT`, after VALUES, NODES, LIBRARY
    and before RENDER & PUBLISH** (the script is the other half of "how a uniform gets its value"). Pure
    policy (no per-project data) → STATIC tier, never busts the prefix cache. Per the actor model it carries
    the routing table (which of the 4 value-setting verbs to reach for), the frozen-mouse fact (decision
    11), the value-shape pointer, the set_uniform reject (so prompt + backend speak with one voice), and the
    motion/no-op verdict. Drafted, house style (terse, ASCII `->`/`--`, no Cyrillic):

    ```
    SCRIPTING (node brains -- driving uniforms over time)
    - A node can have ONE Python brain at nodes/<id>/scripts/script.py: its `update(self, ctx)` returns
      a dict {uniform_name: value} that drives those uniforms EVERY frame (feature 048). Omitted (or
      None) keys stay MANUAL. self.* persists across frames; ctx gives t (seconds), dt, frame, mouse.
    - WHICH tool sets a value -- pick by what the user wants:
        "make it pulse / drift / animate / react over time"   -> write_script (value from ctx.t)
        "make it brighter / bigger / slower" (one fixed value) -> set_uniform
        "add a u_glow uniform"        -> edit_shader to declare it, THEN write_script to drive it
        "change what the shader DOES with a value" (logic)    -> edit_shader (source)
      A script is for VALUES THAT CHANGE; set_uniform / an inline default is for a value that sits.
    - write_script(node?, new_text) create-or-overwrites the whole brain; read_script(node?) returns it
      line-numbered. A FRESH node returns the STUB -- its uniforms + each one's value shape (float /
      Vec2 / Vec3 / Array / Text) + an example to ADAPT (don't save it unchanged).
    - A script-DRIVEN uniform is NOT set_uniform-able (a set is overwritten next tick and rejected,
      pointing at scripts/script.py). To change a driven value, edit update -- not set_uniform, not the
      shader default. The brain writes VALUES only: it cannot add a uniform or change a control's look.
      Declare a new uniform in the SHADER first (inline default is fine -- once driven, the default only
      seeds the initial value), then drive it.
    - ctx.mouse is FROZEN at center (0.5, 0.5) on export and in the headless motion probe -- a
      mouse-driven uniform reads STATIC in the motion facts even when correct. For AUTONOMOUS animation
      drive from ctx.t; reserve ctx.mouse for live-only interactive motion.
    - You SEE a node's script live in the WORKING SET (its own SCRIPT sub-section, rebuilt every step) --
      no separate read for the current node. A write returns the compile verdict, the uniforms it now
      drives (a write that drives 0 uniforms animates NOTHING -- a loud fact, not silence), and a motion
      verdict (which values change across t: ANIMATING / STATIC). Fix a script compile error the same as
      a shader one.
    ```

14. **The result messages, drafted in shader.py house style** (the `tools/script.py` handler formats
    these, mirroring `_applied_result`): success-animating leads with `ok --` (the same token the shader
    path uses, so the agent's success classifier matches); success-static says `-> ... UNCHANGED across
    samples (STATIC) -- vary a value by ctx.t`; drives-0 is the loud `ok -- compiled clean, but drives 0
    uniforms ...`; a compile error mirrors `compiled with errors:\nscript.py:L: <msg>`; a runtime error
    distinguishes `compiled clean, but update() failed at runtime when probed: script.py:L: <type>: <msg>`;
    a per-key coercion error appends `-> 1 key skipped: 'u_glow' expected float, got Vec2`; an orphan key
    appends `-> 'u_gloww' is not an active uniform (typo?). Active uniforms: u_center, u_radius, u_glow`
    (listing the actives so the agent fixes the typo by COPYING the right name). The `set_uniform` tool
    description gains a script-driven clause so it and the reject agree on WHY a set is refused.

## Commit plan

1. **C1 — checkpoint script-capture fix (STANDALONE, lands FIRST).** `_capture_script` + `created_scripts`
   + the revert paths (decision D8). A live data-loss bug fix, testable with no copilot tool, de-risks the
   feature diff. Files: `backend.py` (`_capture_script` + the seam call sites at the existing
   `set_uniform`/persist captures), `checkpoint.py`, `revert.py`, `tests/test_revert_executor.py` (+ a new
   script round-trip test). `make check` + the revert tests green.
2. **C2 — the scripting feature** (everything else below), one coherent diff.

## Files touched

### C1 (checkpoint, first commit)
- `shaderbox/copilot/checkpoint.py` — `TurnCheckpoint`: a `created_scripts` list (skip-recording guarded on
  `node_id in created_nodes`, the double-revert guard); `_capture_script` copies a pre-existing `script.py`
  into the per-node `turn_dir/<node_id>/scripts/`; `has_changes` accounts for it.
- `shaderbox/copilot/revert.py` — a captured script restores via the per-node full-dir swap (it now lives
  in the snapshot dir); a `created_scripts` entry deletes `nodes/<id>/scripts/script.py` (path-absent-
  graceful, like `_revert_created_lib`).
- `shaderbox/copilot/backend.py` — `_capture_script(node_id)` + call it BEFORE the script write (in C2's
  write path); in C1 it is called at the existing `_capture_node` sites so a shader/uniform turn that also
  touches a script captures it.
- `tests/test_revert_executor.py` — a script edit reverts to the pre-edit body; a script create reverts to
  deletion; a script created on a node also created this turn reverts cleanly (double-revert).

### C2 (the feature)
- `shaderbox/scripting/engine.py` — NEW `dry_run(node_id, node, sample_times) -> ScriptProbe`: reads the
  ALREADY-LIVE compile verdict (does NOT call `reload` — decision 4); on clean, steps ONE fresh brain
  (`fresh_behavior_for`) CONTINUOUSLY through the export-clock frame sequence (`dt=1/motion_fps`,
  `frame=step`, `mouse=EXPORT_MOUSE`), recording driven set + per-key errors + orphan keys + per-sample
  values into call-local sinks (decision 10). NEW `ScriptProbe` dataclass (`compile_error: ScriptError |
  None`, `driven: set[str]`, `per_key_errors: list[tuple[str, ScriptError]]`, `orphan_keys: list[tuple[str,
  ScriptError]]`, `samples: list[tuple[float, dict[str, Any]]]`). `_tick_brain` gains a
  `values_sink: dict | None` param redirecting every uniform-value write so the LIVE node is never written
  (decision 5).
- `shaderbox/project_session.py` — NEW `read_script_source(node_id) -> tuple[str, list[CompileErrorInfo]]`
  (file text + sentinel error as CompileErrorInfo, or the AGENT stub when absent — `brain_stub_for` over the
  SAME filtered uniform list `create_script` uses, `is_scriptable(u) and u.name not in
  ENGINE_DRIVEN_UNIFORMS`, NOT the raw active list, else the stub lists engine-owned uniforms = a silent
  no-op trap; + the one animating example, decision 12); NEW `write_script_source(node_id, new_text) ->
  ScriptProbe` (mkdir + write + reload + `dry_run`); forward `dry_run`; the script working-set view getter
  (listing + errors) the backend injects.
- `shaderbox/copilot/capabilities.py` — NEW value objects: `ScriptView` (read_script — listing + status),
  `ScriptWriteResult` (compile_error | driven | per_key/orphan errors | the value-diff motion verdict + the
  one render-facts line). Extend `WorkingSetView` with `script_listing: str = ""` +
  `script_errors: list[CompileErrorInfo] = field(default_factory=list)` (APPENDED after the existing
  required fields — defaults-after-required ordering). NEW Protocol methods: `read_script`, `write_script`,
  `get_script_working_view`.
- `shaderbox/copilot/backend.py` — implement `read_script`/`write_script`/`get_script_working_view`
  (the whole script probe marshals through `run_on_main(..., timeout=render_op_timeout_s)` — `dry_run` reads
  the GL program for active uniforms, so it is NOT worker-safe, decision 5); the motion-facts builder
  (value-diff verdict from `ScriptProbe.samples` + ONE backend-owned render line via the dict-rebind render
  at the mid sample, decision 9 — `_script_render_line` reuses `_render_facts_for` with an explicit sample-time
  `t=mid[0]` so the render clock matches the sampled values, ONE stamp, no wall-clock-at-t=0 drift); populate
  `WorkingSetView.script_*` in BOTH `_copilot_node_working_view` and confirm `_copilot_lib_working_view`
  constructs the new defaulted fields. The script-driven uniforms-row marker (decision 7a) changes
  `_format_uniforms(node)` → `_format_uniforms(node, driven: set[str])` (it is a free function with no
  `node_id`); update BOTH call sites — `_copilot_node_working_view` AND `read_shaders` (the latter now also
  marks driven rows on a read_shader of a scripted node — intended, a sibling-surface behavior change to
  note).
- `shaderbox/copilot/tools/script.py` — NEW (mirror `shader.py`): `script_tools(caps)` returning
  `read_script` + `write_script` ToolDefinitions (args models + handlers formatting the
  ScriptView/ScriptWriteResult facts per decision 14). Register in `tools/registry.py::build_registry`.
  Both `eager=True` (the agent reaches for them in the core animation loop — defensible; re-measure with
  `scripts/token_probe.py` per the context-bloat deferral and record the delta).
- `shaderbox/copilot/prompt.py` — the SCRIPTING block in `_SYSTEM_PROMPT` (decision 13);
  `_render_working_set_member` renders the `=== <node> SCRIPT ===` sub-section gated on a non-empty
  `script_listing` (zero bytes for a script-less node).
- `shaderbox/copilot/config.py` — `COPILOT_CONFIG.motion_sample_times` (default `(0.0, 0.5, 1.0)`) +
  `motion_fps` (default 12) + a value-delta epsilon for the static/animating verdict.
- `tests/_caps.py` — add the 3 new Callable fields (`read_script`/`write_script`/`get_script_working_view`)
  to `_FakeCaps` + 3 entries to `minimal_caps`'s defaults (else every `build_registry` test breaks +
  pyright flags non-conformance). MANDATORY for `make check` green.
- `tests/` — NEW `test_copilot_script_tools.py` (read/write handlers over the fake caps);
  `test_script_dry_run.py` (compile error surfaces with NO tick; driven set + per-key error + orphan key +
  per-sample values after the dry-tick; an INTEGRATOR script's sampled values ADVANCE across samples — the
  decision-10 canary; live `errors`/`last_driven`/`uniform_values` byte-identical before/after — the
  decision-5 no-corruption canary, written FIRST); working-set rendering carries the script + the
  script-driven row marker. The `WorkingSetView` field additions are defaulted (additive) — confirm
  `tests/test_working_set.py`'s existing constructions still pass.
- `scripts/dogfood/harness.py` — NEW `render_video(node, seconds, fps, size)` driver affordance (PNG-only
  today): drives `render_video` off-thread, returns the webm path so the driver can send it (bake in the
  per-frame `texture.read()` V3D note). + the 5 scripting scenarios (decision: pulse-in-place, figure-8
  drift, color cycle, integrator/accumulator — the dry-run state canary, array particle field). The
  `/dogfood` skill gains the scripting walkthrough.
- `ai_docs/todo.md` — resolve the 043 clause of the "copilot turn-rollback" deferral (the `_capture_script`
  seam lands in C1); note the `render_video` pixel-facts follow-on trigger + the `edit_script` trigger.
- `ai_docs/roadmap.md` — banner + the 043 row; flip 043 from the "NEXT" mention.
- `ai_docs/conventions.md` — a copilot Design-decision bullet: the script tools mirror the shader tools
  (content-addressed, working-set-live, checkpoint-captured); the dry-run-for-synchronous-feedback rule
  (compile verdict from live state, tick-gated facts from an isolated continuous dry-tick); the value-diff
  motion verdict + 1 corroborating render.
- `projects/dev/` — a scripted-animation node for the dogfood scenario + sandbox sync.

## Manual verification (maintainer `make run` + a dogfood run — headless can't judge the visuals)

Per dev_flow step 7, each check falsifiable + names the consumer:

- **`make check` + `make smoke` green** — no regression. `tests/_caps.py` updated so `build_registry` tests
  pass with the new Protocol methods.
- **[C1] Checkpoint round-trip (test, first commit)** — a turn that EDITS an existing script then reverts →
  the PRE-EDIT body is restored, not deleted (the live data-loss bug, falsifier: revert deletes it). A turn
  that CREATES a script then reverts → `script.py` is GONE. A script created on a node also created this
  turn reverts cleanly. Falsifier: revert deletes an edited script / leaves a created one behind / errors
  on the double-create.
- **[C2] dry_run no-corruption canary (test, written FIRST)** — a clean script + a syntax-error script:
  assert the live `errors`, `last_driven`, and `node.uniform_values` are byte-identical before and after
  `dry_run`. Falsifier: any of the three differs (the canary goes red — the corruption class is open).
- **[C2] dry_run facts (test)** — a syntax error: `dry_run` returns the compile error, NO tick. A clean
  script driving `u_center,u_radius,u_glow`: `driven` = those three; the per-sample values DIFFER across t.
  A typo key surfaces in `orphan_keys`; a vec2-for-float surfaces in `per_key_errors`. Falsifier: driven
  set empty (tick didn't run), values identical across t (no sampling / wrong dt).
- **[C2] integrator canary (test)** — a script `self.x += ctx.dt; return {u_offset: self.x}` sampled by
  `dry_run` shows `u_offset` ADVANCING across t=0/0.5/1.0 (the continuous-tick accumulates `self.*`).
  Falsifier: `u_offset` identical across samples → the sampling does N independent single ticks (decision
  10 violated, the figure-8-drift false-STATIC).
- **[C2] Working set carries the script + marks driven rows (test)** — after the agent reads a scripted
  node, the working-set member has a `=== <node> SCRIPT ===` sub-section AND the `uniforms:` row of a
  driven uniform reads `<driven by script.py>`, not a phantom default. A script-LESS node adds ZERO bytes.
  Falsifier: script source absent, or a driven row shows a stale number, or a script-less node bloats.
- **[C2] set_uniform reject still fires** — `set_uniform` on a script-driven uniform is rejected naming
  `scripts/script.py`. Falsifier: silent no-op, or the message names a per-uniform file.
- **[C2] DOGFOOD autonomous animation (the headline)** — drive the copilot to the figure-8 pulsing circle
  (an INTEGRATOR + a pulse, exercising decision 10). The agent must: write the shader + uniforms, write a
  `script.py` driving them, read back `ok -- compiled clean, drives u_center,u_radius,u_glow` + a motion
  verdict whose VALUES change across t (ANIMATING) + the single ink-present render line, and render a small
  video. The driver renders the small video + a sample-frame strip and sends them to the maintainer.
  Falsifier: the agent can't tell its animation moves (verdict absent or STATIC when the values clearly
  vary), gets stuck authoring the script, or the rendered video is static.
- **[C2] Motion verdict honesty (test + dogfood)** — a STATIC script (constant dict) reads `-> STATIC`; an
  animated one `-> ANIMATING`; a script driving a uniform the shader IGNORES reads `-> values animate but
  nothing visible` (the value-diff would lie ANIMATING; the one render saves it); an empty-dict script
  reads `-> drives 0 uniforms`. Falsifier: any of these four reports the wrong verdict.
- **[C2] frozen-mouse prompt line lands** — the SCRIPTING block names `ctx.mouse` frozen in the probe.
  (Dogfood falsifier: the agent authors a mouse-driven animation, sees STATIC, and chases a non-bug.)
- **[C2] probe completes within budget on V3D** — a clean `write_script` on the Pi returns a motion verdict,
  NOT a `main-thread op timed out`. Falsifier: the probe rides the 5s default `bridge_op_timeout_s` instead
  of `render_op_timeout_s` and times out the headline walkthrough at the verify step itself.

## Open questions for the user (defaults chosen — maintainer said "just do something sensible, sort it out during dogfooding")

All defaulted per the maintainer's explicit call (msg 2026-06-14 12:58) to stop asking low-level questions
and decide sensibly, revisiting at dogfood time. Recorded here so a reviewer sees what was decided:

1. **`read_script` on a script-less node returns the AGENT stub** (brain_stub_for + one animating example),
   unpersisted — strictly more useful than an error.
2. **`edit_script` DROPPED this wave** — `write_script` whole-file covers a 30-line brain; add on a
   demonstrated whole-file-thrash trigger.
3. **Motion verdict = value-diff across t (primary) + ONE corroborating render** — cheaper and honester than
   3 pixel renders; sample times `(0.0,0.5,1.0)`, `motion_fps` 12, all `COPILOT_CONFIG`.
4. **The agent reaches for scripting PROACTIVELY** on an "animate / pulse / drift over time" ask (the prompt
   routing table) — the whole point is autonomous animation; tighten the prompt if it over-reaches.

## Review history

### Plan-draft research (2026-06-14)
Three parallel agent sweeps grounded the spec in real code: the shader-tool seams (the mirror pattern), the
scripting public surface (`dry_run`-able primitives), and the checkpoint/prompt seams (the D8 data-loss
bug). Findings became the Design decisions above.

### Brainstorm + review swarm (2026-06-14, 6 agents — 2 brainstorm + 4 review incl. a devil's advocate)
The first spec draft (a `read/write/edit` trio + `dry_run` with snapshot/restore + 3-pixel-render motion
facts) went through 2 generative + 4 adversarial agents (one anchored to the actor-model skill + real code,
the non-self-authored anchor). Converged; the accepted findings rewrote the spec:
- **BLOCKER (2 reviewers) — N-independent single ticks report a FALSE STATIC for any integrator** (the
  spec's own figure-8-drift goal). → decision 10: ONE fresh brain stepped CONTINUOUSLY through the export
  clock so `self.*` accumulates; the integrator canary added.
- **BLOCKER — "value-sampling is GL-free" is FALSE** (`get_active_uniforms` reads the live GL program). →
  decision 5: the whole `dry_run` is bridge-marshalled; the false worker-safe claim removed.
- **BLOCKER — `dry_run` calling `reload()` corrupts live state** (`reload` is the live mutator + mtime-
  gated). → decision 4: `dry_run` reads the already-live compile verdict, never reloads.
- **BLOCKER (integration) — the dogfood harness can't render a video** (PNG-only); the headline check was
  unrunnable. → a `render_video` harness affordance added to Files-touched.
- **BLOCKER — `tests/_caps.py` breaks** on the new Protocol methods. → added to Files-touched (mandatory).
- **STRONG (3 agents) — motion verdict should come from VALUES, not pixels** (GL-free, exact, catches the
  pulse-in-place + color-cycle cases bbox is blind to, the spec's own headline goal). → decision 9
  rewritten: value-diff primary + ONE corroborating render for the FLAT/visible honesty case; the 4-way
  verdict (nothing/static/animating/animates-but-FLAT).
- **HOLDS (devil's advocate) — `edit_script` is pure tool tax** (resets state identically to `write_script`,
  ~30-line file). → dropped from the wave (decision 1).
- **HOLDS — the checkpoint fix is a separable live data-loss bug** → split to commit C1, lands first.
- **STRONG (prompt reviewer) — frozen `ctx.mouse` is invisible to the agent + the stub teaches no motion +
  the no-op write is silent** → decisions 11, 12, the routing table + drives-0 loud fact in decisions 13/14;
  the phantom-uniform-row contradiction → decision 7a.
- **REFUTED (noise) — trio-vs-overload is backwards / `dry_run` is unnecessary.** Kept the separate tools
  (artifact-kind divergence forces a different matcher + result type — the rebuttal folded into decision 1)
  and `dry_run` (headless has no frame loop to run the tick the facts depend on — folded into decision 4).
- **CONFIRMED-TRUE** by the code-anchored reviewer: the D8 claim (a revert DELETES an edited script —
  verified against `UINode.save` + `_swap_in_snapshot`), the engine fact-split table, the set_uniform reject
  text, the working-set "" gate adding zero bytes, the prompt tier.

### Pre-implementation review (2026-06-14, 2 adversarial agents — correctness/design + verification/blast-radius)
Both anchored to real code + the actor-model skill (the non-self-authored anchor). Verdict PARTIAL; the
swarm's "value-diff + 1 render" rewrite introduced a contradiction the swarm missed, now fixed:
- **BLOCKER (both) — decision-5 no-corruption canary vs decision-9 "render the mid sample via the
  export-isolation tick" collide:** the render needs the sink'd values ON the node, but the sink keeps them
  OFF; and `_make_export_isolation` is the wrong primitive (writes live `uniform_values`, ticks a 2nd fresh
  brain, lives in `project_session` — the headless engine can't import `Canvas`/`render_facts`). → decision
  9 rewritten: `dry_run` returns `samples` only (no render, headless boundary preserved); the BACKEND owns
  ONE render via a dict-REBIND (`node.uniform_values = {**snapshot, **mid_samples}` in try/finally — the live
  dict object never mutated, canary stays byte-identical); only the export CLOCK is reused, not the manager.
- **MAJOR — `dry_run`'s `run_on_main` timeout unspecified** (5s default vs the render tools' 60s); a V3D
  probe (12 ticks + a render) could spuriously time out the headline walkthrough AT the verify step. →
  pinned to `render_op_timeout_s` (decisions 5/9) + a verification line.
- **MAJOR — the value-equality rule was named but undefined** for the tuple/list/nested shapes `coerce_one`
  emits (scalar epsilon insufficient, bare `==` false-positives on `sin` jitter). → a recursive epsilon
  compare (structurally like `_all_finite`), `eps` in `COPILOT_CONFIG`; "exact diff" reworded to
  epsilon-exact.
- **MINOR (folded) — `read_script`'s stub must filter `ENGINE_DRIVEN_UNIFORMS`** (like `create_script`, not
  raw `brain_stub_for`) else it lists engine-owned uniforms as a no-op trap; **`_format_uniforms` gains a
  `driven` param + BOTH call sites** (`_copilot_node_working_view` + `read_shaders`) update — the
  read_shader sibling now also marks driven rows (intended).
- **CONFIRMED correct (no change):** decision-4 ordering (write→reload→dry_run reads the live verdict
  without a tick); the values_sink redirect covers all 4 `_tick_brain` value sites; the integrator continuous
  accumulation (`run()` never re-inits, `fresh_behavior_for` makes one instance); the C1/C2 split (C1 closes
  the pre-existing data-loss bug, `write_script`'s own capture ships with the tool in C2 — not a half-fix);
  the eager-tools call (the context-bloat deferral itself scopes lazy-load to the telegram/youtube tail, ~262
  tok marginal for the 2 core tools is defensible — re-measure + record); no migration code (the
  `created_scripts` `.get(..., [])` additive-load is the sanctioned unreleased-format pattern, not a shim);
  smoke unaffected (drives `update_and_draw`, not the copilot).

**Spec is plan-lock-ready after these fixes.** Implementation pending the maintainer's go.
