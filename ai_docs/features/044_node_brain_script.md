# Feature 044 — node-brain script (one stateful class drives many uniforms)

> **PROMOTED to the SOLE scripting path by 048 (`ai_docs/features/048_single_script_play_stop.md`).** The
> node-brain (`script.py` → `update -> dict`) this spec introduced is now the ONLY script per node — 048
> deleted the per-uniform `u_*.py` path it was a sibling to. So 044's brain-vs-per-uniform CONFLICT rule
> (the two-pass write order, the broken-override-yields-to-brain fallback) is GONE (there is no per-uniform
> script left to conflict). Everything else (the stateful class, the dict return, freeze-as-data, the
> per-key vs sentinel error granularity, export isolation) carries forward as the live model. Read 048 for
> the current shape; this spec is the history of the brain's introduction.

> Status: SPEC — plan-locked with the maintainer (2026-06-13), pre-impl review pending. A SECOND,
> parallel scripting path beside 041's per-uniform script, sharing the SAME engine core + flow. NOT
> shipped (last release v0.15.0; 041 itself is unshipped on `dev`) — so NO migration burden, the
> contract is free to evolve. Headless engine ONLY (like 041); the UI affordance (a "New node-brain"
> template, the node-level error indicator) is feature **042**'s concern, noted but deferred.

## Goal

041 made a uniform's behavior a stateful class, but kept "one file = one uniform = one class". For a
game-like object whose position, speed, colour, and counters all derive from ONE piece of physics, that
forces the **same state + the same integration step to be copy-pasted across N script files** (proven
in the dogfood: a spring pendulum needed its Verlet step duplicated into `u_pos.py`, `u_speed.py`,
`u_trail.py`, `u_bounces.py`). The reason CPU scripting exists — state — is exactly what wants to live
in ONE object that projects into MANY uniforms.

044 adds a **node-brain** script: a single file `nodes/<id>/scripts/script.py` whose
`update(self, ctx) -> dict[str, <typed output>]` returns a map of uniform-name → value, driving as many
of the node's uniforms as it likes from ONE stateful instance. This is the in-scope, explicitly-blessed
resolution of 041's "one object, many uniforms (the pong-ball case)" intent (041 Out-of-scope lines
55-56) — NOT a reopening of the cross-uniform shared-state CHANNEL deferral (which is two SEPARATE
scripts sharing mutable state; that stays deferred).

The decisive design constraint (the maintainer's ask): node-brain must look **NATIVE, not bolted-on** —
both paths share the same base class, resolver, compile machinery, coercion, freeze-as-data store,
tick loop, `(path,mtime)` cache, and export-isolation seam. After the refactor there is ONE pipeline;
per-uniform is the **1-entry case** of the node-brain value-map, not a parallel implementation.

## The shared abstraction (the whole point)

Today `PythonBehavior.compute(ctx, uniform)` (behavior.py:264-283) FUSES two concerns: producing the
raw value (where the "one behavior = one uniform" assumption is baked) and coercing it (always
per-name). 044 splits that seam so the common logic becomes genuinely common:

```
discover source files  ->  compile each VERBATIM into a ScriptBehavior instance  (one compile leaf)
   ->  per tick, ask each behavior for its raw value(s)        (run(ctx) — cardinality-agnostic)
   ->  flatten into a (uniform_name, raw_value) PAIR STREAM     (_resolved_pairs — the 1->N fan-out)
   ->  coerce each pair against the live uniform                (coerce_one — one leaf, both paths)
   ->  freeze-or-write each by name into node.uniform_values    (existing per-name freeze body)
```

A `u_<name>.py` file contributes **exactly one** pair `[(name, run(ctx))]`. A `script.py` file
contributes `run(ctx).items()`. Below the `_resolved_pairs` fan-out the engine cannot tell the two
apart — same coercion, same last-good store, same `errors[(node_id, name)]` map, same writes. Deleting
the node-brain branch would mean RE-specializing the general loop back to cardinality 1 — proof the
general case is primary and per-uniform is the degenerate one.

## Out of scope (each deferral carries a trigger)

- **UI / discoverability (a "New node-brain" template button, a node-level error indicator distinct
  from a uniform-row indicator, the script-driven chip on brain-driven rows).** v1 brain scripts are
  hand-placed + hot-reloaded, errors log-only — the SAME posture 041 shipped per-uniform with. The
  node-level error key (`(node_id, "script.py")`, decision 6) is a NEW shape 042's UI must learn to
  render as a node-level error, not a uniform-row error. Trigger: feature **042** (the script UI).
- **`stub_for` for node-brain (a generated `return {...}` skeleton listing all the node's scriptable
  uniforms).** v1 brain scripts are hand-written, as 041 shipped per-uniform stubs into 042. When the
  brain stub IS written, it MUST NOT annotate `-> dict[str, Any]` (`Any` is not in the exec-globals →
  permanent compile-freeze per 041 decision 12); use `-> dict` bare or seed `Any`. Trigger: 042's "Add
  script" work (file a todo so 042 knows).
- **Cross-NODE state (object A's brain reads object B's brain).** A brain is node-scoped: its `self`
  covers ONE node's object. Two objects seeing each other needs a cross-node channel — a strictly
  larger design. Trigger: a real two-object-interaction workflow (the mini-game milestone).
- **Cross-SCRIPT shared-state channel (one script writes a value another script READS the same frame).**
  041's deferral (Out-of-scope lines 54-58) stands UNCHANGED. 044 does not add `ctx.shared`/`ctx.state`;
  the physics-copy-paste is solved by putting the shared state in ONE brain instance, not a side channel.
  Trigger: 041's original (two scripts need to share mutable state with a read/write ordering invariant).
- **Copilot write-behavior tool for brain scripts.** No copilot tool authors a `script.py` in v1; that
  is feature **043** (which must write the dict-return CLASS form). Trigger: 043.

## Design decisions (numbered; lock-in only)

> All six were plan-locked with the maintainer (2026-06-13) — see the *Plan-lock resolution log*.

1. **A node-brain script is `nodes/<id>/scripts/script.py` — INSIDE `scripts/`, beside `u_*.py`** (NOT
   a sibling one level up). This keeps `ScriptEngine.reload`'s call signature unchanged: it already
   receives `scripts_dir = paths.scripts_dir_for(node_id)` (project_session.py:298,350) and discovers
   `script.py` in the SAME reload pass, via a SEPARATE discovery branch (the existing `glob("u_*.py")`
   at engine.py:141 does NOT match `script.py` — the dot prevents it; the impl adds an explicit
   `scripts_dir / "script.py"` check beside the glob loop, `found.add(sentinel)` so the drop loop at
   engine.py:176-182 doesn't evict it). A sibling `nodes/<id>/script.py` would force widening `reload`'s
   signature to take the node root + threading it through `_resolve_scripts`/`reload_scripts` — exactly
   the "call sites unchanged" guarantee this design preserves. The filename `script.py` has a dot, so
   its sentinel key can never collide with a GLSL identifier (decision 6). Revisit only if a node ever
   needs MULTIPLE brains (it won't — one node = one object = one brain).

2. **Same base class `ScriptBehavior`, same resolver — NO new base, NO new Protocol member for the user.**
   A brain is `class Behavior(ScriptBehavior)` resolved by the existing `_resolve_behavior_class`
   (behavior.py:158-167) and arity-checked by the existing `_check_update_arity`. The ONLY difference
   from a per-uniform script is that `update` returns a `dict` instead of a single typed value. The
   strongest "native" signal: the user writes the same class, the engine reuses the same compile path.
   (The `update` return annotation `dict[str, Vec2]` evaluates EAGERLY at class-def — `dict` is in the
   exec builtins and subscriptable in 3.12, `Vec2` etc. are top-level globals, so a realistic brain
   annotation is safe; only `Any` is the trap, decision-12-of-041 carried into the stub deferral above.)

3. **Dispatch by FILENAME at reload, NEVER by return-type sniffing.** `script.py` → node-brain
   cardinality; `u_*.py` → per-uniform cardinality. The cardinality of a behavior is a pure function of
   its binding KEY, decided once at discovery — not by inspecting whether `update` returned a dict. A
   per-uniform script that legitimately returns a dict-shaped value (none today, but a future kind
   could) must NOT be re-routed, and the write key must stay STATICALLY known before the tick. This
   survived all three pre-design critiques unanimously and is non-negotiable.

4. **`PythonBehavior.compute` is SPLIT into `run` + a free `coerce_one`** (the seam relocation). `run(self,
   ctx) -> object` keeps only the no-instance guard + `self._instance.update(ctx)` (behavior.py:268-272),
   returning the raw value — cardinality-agnostic. `coerce_one(value, uniform, error_name) -> coerced` is
   a module-level free function holding the normalize+coerce+shape-hint tail (behavior.py:273-282). NAMING
   PIN (a reviewer caught the collision): the third arg is `error_name` — the uniform NAME the resulting
   `ScriptError`/`_RuntimeScriptError` is recorded under, NOT to be confused with `uniform_shape_hint`'s
   own `label` arg (the GLSL type string like `"vec2"`, which `coerce_one` still derives INTERNALLY via
   `gl_type_label(uniform)` exactly as compute does today at behavior.py:281). `error_name` is load-bearing
   because the free fn no longer has `self._uniform_name` in scope — per-uniform passes the uniform name;
   node-brain passes the dict KEY (the per-uniform name the value targets, NOT the sentinel). `compute` is
   DELETED. The exported `Behavior` Protocol (behavior.py:170-171) loses `compute` → gains `run(self, ctx)
   -> object`, with a docstring stating the dual contract (decision 6): a per-uniform behavior's `run`
   returns a single typed value; a node-brain's `run` returns a `dict[str, value]`; coercion is now the
   ENGINE's job (a future C backend produces raw values, the engine coerces). Per-uniform path: engine
   yields `[(uniform_name, behavior.run(ctx))]` then `coerce_one`; node-brain: engine yields
   `behavior.run(ctx).items()` then `coerce_one` per key. Identical coercion atom both paths.

5. **The per-tick loop iterates a (name, raw) PAIR STREAM.** `_tick_behaviors` (engine.py:251-301)
   becomes an OUTER loop over behaviors and an INNER loop over the pairs each yields:
   - **Outer** `for behavior in behaviors.values()`: call `run(ctx)` ONCE inside a try/except. A raw
     exception (or a non-dict return from a brain) is a BEHAVIOR-level failure → freeze ALL the names
     that behavior drove last frame (per-uniform: its 1 name; brain: last frame's keys) and record under
     the behavior's error key (decision 6).
   - **Inner** `for name, raw in _resolved_pairs(behavior, ...)`: the EXISTING body (engine.py:266-301)
     re-indented, NOT rewritten — `frozen`-lookup (engine.py:267), `is_scriptable` gate (275), `coerce_one`,
     freeze-on-shape-mismatch, `errors.pop` + `last_good[name]` + write. A key naming a NON-active/
     non-scriptable uniform is warn-once + a real `continue` **with NO write**. The CHANGE the critiques
     caught (precise lines): today the `is_scriptable`-fails branch at engine.py:277-278 writes
     `node.uniform_values[name] = frozen` then continues — for a per-uniform binding that's fine (`name`
     was a real uniform that went inactive, `frozen` is its last-good). But for an unknown BRAIN key
     `frozen` is `None` (no last-good for a name that was never a binding) and that write pollutes
     `uniform_values` with `None`. The inner loop must therefore SKIP (warn-once + `continue`) BEFORE the
     write for a name that was never an active scriptable uniform, distinct from the "went-inactive"
     freeze-to-last-good branch. (Per-uniform never reaches this: its bindings are resolved+dropped at
     reload, so it never ticks an unknown name.)

6. **Error keys + freeze granularity — per-key for coercion, behavior-level for a raw throw.** A per-KEY
   coercion mismatch records `(node_id, name)` and freezes ONLY that uniform (granular, matches the
   per-uniform path; sibling keys still write). A raw exception in `update()` OR a non-dict return is a
   behavior-level failure recorded under the SENTINEL key `(node_id, "script.py")` and freezes EVERY
   name the brain drove last frame (one object = one coherent state — a half-updated object is wrong).
   This asymmetry is correct for the model but MUST be documented (it amends 041 decision 9's "ScriptError
   shape UNCHANGED" — the key shape gains the sentinel). The `ScriptError` dataclass itself is unchanged.

7. **Conflict rule: BOTH apply in a locked write order — brain writes FIRST, `u_<name>.py` writes LAST
   and wins.** If a node has both `script.py` (driving `u_x`) and `u_x.py`, both run; the per-uniform
   value overwrites the brain's. Rationale: the brain sets an object's BASE behavior; an explicitly-named
   `u_<name>.py` is a point OVERRIDE of one parameter — "explicit beats general", and it lets a uniform
   be migrated OUT of the brain into its own file without editing the brain. **ENFORCEMENT PIN (a
   reviewer flagged the "or equivalently" as an unresolved fork): TWO-PASS, not a `claimed` set.**
   `_tick_behaviors` runs ALL brain behaviors' pairs first (one pass over the brain bindings), THEN all
   per-uniform behaviors' pairs (a second pass), so a per-uniform write lands AFTER and overwrites the
   brain's slot. This does NOT rely on `behaviors`-dict insertion order (which the `(path,mtime)`
   `continue`-on-unchanged at engine.py:163-164 could perturb on a reload) — the two-pass split is the
   guarantee. **SPEC-FIDELITY (a reviewer questioned whether this fires 041's compute-order deferral):
   it does NOT.** 041's trigger (Out-of-scope line 60-61) fires when "a script depends on another's
   same-frame OUTPUT" — i.e. a READ dependency (script B reads what A wrote this frame). 044 has no such
   read: brain and `u_x.py` both WRITE the slot, neither reads the other's value (there is no
   `ctx.uniforms` — removed in 041). This is write-precedence at one slot, not an update-order-with-read
   dependency, and it is within ONE node, not a cross-script shared-state CHANNEL. So 041's trigger stays
   unfired; 044 locks this as a deliberate, NEW within-node write-precedence decision (not slipped in).
   When the files target DIFFERENT uniforms (the normal case) the order is invisible — each owns its own slot.

8. **The dict is NON-exhaustive (partial scripting generalized).** A brain need not return every uniform;
   only returned names are written, the rest keep their slider/default value. A name driven on frame N
   then omitted on frame N+1 simply keeps its last written value (NOT frozen-to-last-good — that path is
   for a name the brain DID target but that errored; the loop must not conflate "omitted" with "frozen").
   This is the per-uniform path's partial-scripting affordance, generalized for free. An omitted-after-
   failing key must also have its stale `errors` entry CLEARED (no zombie error).

9. **Export isolation is PRESERVED with ZERO change to the factory — cardinality is recovered from the
   KEY.** `fresh_behaviors_for` (engine.py:214-225) rebuilds `{key: PythonBehavior(key, source)}` from
   `scripts.sources`; the sentinel `"script.py"` IS a key in `sources`, and `_resolved_pairs` decides
   fan-out by `key == "script.py"` (the SAME filename dispatch reload used). So the export tick path fans
   out the brain correctly with NO new data shape and NO flag threaded through `fresh_behaviors_for`. This
   was the single biggest hole the designs glossed (they assumed a flag on the cached binding, which the
   export set does not carry) — resolved by keeping the cardinality decision a pure function of the key.
   The isolation MECHANISM (`Node.export_isolation` factory, `_make_export_isolation`, the on_pre_render
   swap, `tick_behaviors` with throwaway last_good+errors sinks) stays BYTE-IDENTICAL.

10. **`script_driven_uniforms` becomes last-tick-accurate for a brain.** The per-uniform path reports
    `set(behaviors)` (stems) statically. A brain's driven names are DYNAMIC (the dict keys), so the
    reported set is the union of `u_*` stems + the brain's CACHED last-frame keys (`NodeScripts` gains a
    `last_driven: set[str]`, initialized empty, updated to the dict keys after each SUCCESSFUL brain tick;
    it PERSISTS across `reset()` so a reset-time `__init__` failure can still freeze the prior frame's
    driven names per decision 6), MINUS the sentinel. This is used only by the copilot `set_uniform` reject
    (a COURTESY — coercion is the real gate), so a one-frame cold-start window (a freshly-loaded brain
    before its first tick reports no keys) is accepted: a `set_uniform` could slip through on a uniform
    the brain overwrites next tick — the exact loop `backend.py` already warns about. A static declared
    set is impossible because brain keys are dynamic.

11. **Error MARKER threaded end-to-end (the second critique fix).** `_user_error_line` (behavior.py:30-39)
    takes a `uniform_name` and builds the frame marker `f"<u:{uniform_name}>"` INTERNALLY (line 34), then
    matches traceback frames against it. The compile filename is the same shape (behavior.py:190).
    WRAPPING PIN (a reviewer caught a double-wrap risk): the threaded value stays the UNWRAPPED label
    (`"script.py"` for a brain, the uniform name for per-uniform) — the function keeps building
    `f"<u:{label}>"` itself; we do NOT pass it an already-wrapped `"<u:script.py>"` (that would produce
    `<u:<u:script.py>>` and never match). Concretely: rename the ctor arg `uniform_name` → `label` and the
    `_user_error_line` param `uniform_name` → `marker_name` (still unwrapped), compile stays
    `f"<u:{label}>"`. The engine's generic-exception path (engine.py:295) today calls
    `_user_error_line(name, e)` with the per-PAIR uniform name; for a brain, `run()` raises BEFORE
    per-pair iteration, so the engine catches it at the BEHAVIOR level and calls
    `_user_error_line(behavior.label, e)` with the brain's compile label (`"script.py"`), which matches the
    `<u:script.py>` frame the brain was compiled under. Per-uniform is unaffected (label == uniform name).
    Without this, every brain runtime traceback maps to `line=-1` (lost user line).

## Files touched (mapped to the dev_flow module map)

- **`shaderbox/scripting/behavior.py`** — SPLIT `compute` (264-283) into `PythonBehavior.run(self, ctx)
  -> object` (~4 lines: the no-instance guard + `update` call) and module-level `coerce_one(value,
  uniform, error_name) -> coerced | raises _RuntimeScriptError` (~9 lines lifted from 273-282; `error_name`
  is the uniform name the error records under — `coerce_one` derives the GLSL `label` for the shape hint
  internally via `gl_type_label`, per decision 4). Delete `compute`. Rename ctor `uniform_name` → `label`
  (mechanical, keep `<u:{label}>`). `_user_error_line`'s param `uniform_name` → `marker_name` (unwrapped,
  decision 11). `Behavior` Protocol (170-171): `compute` → `run(self, ctx) -> object` + a docstring stating
  the dual contract (per-uniform `run` → a value; brain `run` → a `dict`; the engine coerces) — else
  pyright reds on the stale member. New exported symbol: `coerce_one`.
- **`shaderbox/scripting/engine.py`** — add a `script.py` discovery branch in `reload` keyed by the
  sentinel (decision 1; `found.add(sentinel)` so the drop loop at 176-182 doesn't evict it every poll);
  add `_resolved_pairs(behavior, key, active) -> Iterable[tuple[str, object]]` (the 1→N fan-out — KEY-driven
  per decision 3: `key == "script.py"` → `run().items()`, else `[(key, run())]`; ~10 lines); reshape
  `_tick_behaviors` (251-301) into the outer-behavior/inner-pair loop with the `run()`-level catch using
  the behavior label (decision 5/11), the unknown-key skip-`continue` BEFORE the write (decision 5), and
  the explicit TWO-PASS apply order (all brain pairs, then all per-uniform pairs overwrite — decision 7);
  a brief in-code comment at each of the three non-obvious seams (the two-pass apply order, the
  behavior-level vs per-key freeze, the unknown-key skip) per `conventions.md ## Code rules`;
  `script_driven_uniforms` unions the cached `last_driven` minus the sentinel (decision 10); `NodeScripts`
  gains `last_driven: set[str]`. `_binding_reject` (184-195) runs only on per-uniform keys; brain keys
  validate lazily at tick (the skip). `fresh_behaviors_for` UNCHANGED (decision 9). `stub_for` UNCHANGED
  (per-uniform only; brain stub deferred).
- **`shaderbox/scripting/__init__.py`** — export `coerce_one` if a test imports it; update the `Behavior`
  Protocol export note (`compute` → `run`).
- **`shaderbox/scripting/outputs.py`** — UNCHANGED. A brain returns the SAME typed outputs per dict value.
- **`shaderbox/scripting/context.py`** — UNCHANGED (`EngineContext` is `t/dt/frame`; no `ctx.shared`).
- **`shaderbox/scripting/errors.py`** — UNCHANGED (`ScriptError` dataclass; only the error-KEY shape gains
  the sentinel, which lives in the engine's `errors` dict, not in the dataclass).
- **`shaderbox/uniform_coerce.py`** — UNCHANGED. `coerce_uniform_value`/`normalize_output`/
  `uniform_shape_hint` are the coercion atoms; only their CALLER moves (`compute` → `coerce_one`).
- **`shaderbox/project_session.py`** — `_resolve_scripts`/`reload_scripts` UNCHANGED (reload discovers
  `script.py` inside the same `scripts_dir`). `get_script_driven_uniforms` UNCHANGED (delegates). The
  export-isolation factory + `_make_export_isolation` + `_wire_node_hooks` UNCHANGED (decision 9).
- **`shaderbox/copilot/backend.py`** — the `set_uniform` reject message branches on file kind
  (`scripts/{name}.py` vs `scripts/script.py`) so the user sees the right "this uniform is script-driven
  by …" message (~3 lines).
- **`shaderbox/core.py`** — UNCHANGED (confirm: `Node.render_media` + the `on_pre_render` seam keep their
  shape; the engine-free `Node` is untouched).
- **`tests/test_script_engine.py`** — ADD `_write_brain(tmp, body)` (writes `tmp/scripts/script.py`).
  PARAMETRIZE the cardinality-agnostic invariants over (per_uniform_body, node_brain_body): state
  accumulates, `reset()` clears, recompile-on-edit resets, and the export-isolation INTERLEAVE (assert
  the EXTRACTED per-key value is cold, NOT the raw dict — a dict is truthy and would mask a coercion fail).
  ADD node-brain-only tests (decision targets): (a) one brain drives 2+ uniforms in a tick; (b) per-KEY
  shape mismatch freezes only that key, siblings still write (the is-it-native falsifier); (c) a non-dict
  return is a clean runtime `ScriptError` under the sentinel, not a crash; (d) a raw `update()` exception
  freezes all last-frame names + records under the sentinel with the CORRECT user line (the decision-11
  falsifier — fails if `_user_error_line` still hardcodes `<u:name>`); (e) an unknown key is warn-once +
  SKIPPED with no `None` pollution; (f) conflict: `u_x.py` + `script.py` both target `u_x` → per-uniform
  wins (silently — precedence is intentional, not warned), forcing a reload order that would invert
  under naive insertion order (write `script.py` first); (g) `script_driven_uniforms` returns driven
  names (not `"script.py"`), partial
  before first tick; (h) partial: brain drives 2 of 3, the 3rd keeps its default; (i) a key driven frame
  N then omitted frame N+1 keeps its last value AND clears its error (no zombie). ADD one brain compile-
  error test: a raising `__init__` declares zero keys yet must surface an error under the sentinel (the
  gap the per-uniform compile tests don't transfer). The shared compile/arity/no-subclass tests cover the
  brain's compile path by construction (same `PythonBehavior`).
- **`tests/test_script_engine_gl.py`** — ADD a node-brain twin of `test_fresh_export_instance_renders_clean`:
  a 2-uniform brain reaches the GPU + a fresh export instance renders cold.
- **`ai_docs/features/041_stateful_script_engine.md`** — AMEND: decision 8 ("one file = one uniform = one
  class") is now FALSE for `script.py` (add a "see 044" note); decision 9 ("ScriptError shape UNCHANGED")
  gains the `(node_id, "script.py")` sentinel error-KEY shape; the Out-of-scope ordering bullet (59-61)
  gains the scoped 044 write-precedence exception; note that 044 is the SANCTIONED resolution of the
  "one object, many uniforms" intent (lines 55-56), NOT a reopening of the cross-SCRIPT shared-state
  trigger.
- **`ai_docs/conventions.md`** — the `## Design decisions` script bullet gains: a node-brain script
  (`scripts/script.py`) returns a dict driving many uniforms; per-uniform is the 1-entry case of the
  same value-map pipeline (`run` + `coerce_one`); filename dispatch (not return-type sniffing); conflict
  = explicit write order (brain first, `u_<name>.py` wins).
- **`ai_docs/roadmap.md`** — add the 044 row + rewrite the Active-context banner (044 = node-brain engine;
  042 UI now must cover both per-uniform AND brain, with the node-level error indicator).
- **`ai_docs/todo.md`** — file the deferred-stub-for-brain trigger (042's "Add script" must add a "New
  node-brain" template avoiding the `Any`-in-annotation trap); file the cross-NODE state trigger (the
  mini-game milestone). The cross-SCRIPT shared-state deferral (041) is UNCHANGED — note 044 did NOT fire it.

## Manual verification

> Display-less Pi — the agent runs the headless backbone; the visual confirm is the maintainer's (or a
> throwaway dogfood render, as in the 041 exploration). 044 has NO UI, so the headless backbone is the
> whole agent-side verification.

**Headless backbone (the agent runs these):**
1. `uv run pytest tests/test_script_engine.py` — the parametrized cardinality-agnostic invariants (both
   paths) + the node-brain-only cases a-i + the brain-compile-error case (decisions 1-11).
2. `uv run pytest tests/test_script_engine_gl.py` — a 2-uniform brain reaches the GPU + a fresh export
   instance renders cold (standalone EGL; skips if no GL).
3. `make check` — ruff + pyright 0 errors (confirm the `Behavior` Protocol `compute`→`run` rename
   doesn't strand a stale member; no new suppression beyond the 041 allowlist).
4. `make smoke` — the existing per-uniform scripted node still ticks clean for 200 frames (proves the
   per-uniform path survived becoming the 1-entry case). Optionally add a brain node to the smoke seed.
5. A throwaway dogfood render (the 041-exploration style): a node-brain spring pendulum driving
   `u_pos`/`u_speed`/`u_bounces` from ONE instance — the physics written ONCE — renders + animates, and
   its EXPORT starts cold. (Confirms the headline goal: no copy-paste, export-isolated.)

**`make run` hand-steps (handed to the maintainer — confirms the model works live, pre-UI):**
1. `make run`, pick a node. Hand-create `nodes/<id>/scripts/script.py`:
   ```python
   class Behavior(ScriptBehavior):
       def __init__(self) -> None:
           self.pos = 0.5; self.vel = 0.3
       def update(self, ctx: Ctx) -> dict:
           self.vel -= 0.9 * ctx.dt
           self.pos += self.vel * ctx.dt
           if self.pos < 0.0:
               self.pos = 0.0; self.vel = -self.vel * 0.8
           return {"u_y": self.pos, "u_speed": abs(self.vel)}
   ```
   → both `u_y` and `u_speed` animate from ONE bouncing-ball instance.
2. Add `nodes/<id>/scripts/u_speed.py` returning a constant → `u_speed` is now OVERRIDDEN by the
   per-uniform file (conflict rule, decision 7); `u_y` still driven by the brain.
3. Break the brain (raise in `update`) → both `u_y` and `u_speed` freeze together (behavior-level
   freeze, decision 6); no crash; a `ScriptError` under `(node, "script.py")`. Fix → resumes.
4. Export a video → the ball animates from a CLEAN start (fresh export instance, decision 9).

## Review sizing

**Proposed: mid (the default) — 1-2 pre-impl + 2-3 post-impl in parallel.** Rationale: it refactors the
core of an existing subsystem (the `compute` split, the `_tick_behaviors` reshape) and adds a second
path, but it crosses NO new module boundaries (the 025 headless ownership, the coercion hoist, the
export-isolation seam all stay), has NO UI surface (042), and the new-code estimate is ~60-90 LOC. The
risk concentrates in: the error-marker threading (decision 11 — brain tracebacks at the user line), the
export-isolation cardinality-from-key (decision 9 — the interleave test is the falsifier), the
unknown-key skip (decision 5 — no `None` pollution), and the conflict apply-order (decision 7 — must not
rely on dict-insertion order). The per-uniform path becoming the 1-entry case is a behavior-preservation
claim the parametrized tests + `make smoke` must prove.

- **Pre-impl: 1-2 agents.** (a) *correctness & design* — the `run`/`coerce_one` split genuinely keeps
  the per-uniform path byte-identical downstream of the fan-out; the marker threading keeps brain errors
  at the user line; the cardinality-from-key recovers brain fan-out in the export set without a flag; the
  conflict apply-order is deterministic (not insertion-order). (b) *verification & blast-radius* — the
  test matrix covers both paths with shared fixtures + the brain-only falsifiers (per-key vs behavior-
  level freeze, unknown-key skip, the interleave asserting the extracted value not the raw dict); the 041
  spec amendments are honest (what's superseded vs kept).
- **Post-impl: 2-3 agents, convergence loop if findings cluster.** *code correctness* (the run/coerce
  seam, the per-tick error catch never escaping, no instance leak, the fresh-export brain fan-out, the
  unknown-key None-pollution guard), *architecture & conventions* (the seam relocation is genuinely
  shared not duplicated, no imgui/glfw in the engine, filename dispatch, the Protocol rename), and a
  *spec-fidelity audit* (decisions 1-11 landed; per-uniform is provably the 1-entry case, not a parallel
  impl). At least one reviewer anchors to a non-self-authored artifact (the 041 engine source as the
  before-state; `tests/test_script_engine.py`'s export-isolation interleave as the fixture precedent).

## Open questions for the user

None — all six scoping decisions are plan-locked (see the resolution log). The conflict apply-order fork
is now resolved to TWO-PASS (decision 7). The only thing delegated to implementation is cosmetic naming
(the sentinel constant name, the exact `_resolved_pairs` parameter names).

## Plan-lock resolution log (2026-06-13)

Maintainer answers at scoping, folded into the decisions:

1. **Implement node-brain as a SECOND path PARALLEL to per-uniform, sharing the core natively** (not a
   replacement, not a bolt-on). A multi-agent research swarm (3 map → 3 design → 6 critique → 1
   synthesize, 13 agents) found the value-map seam: split `compute` into `run` + `coerce_one`, fan out
   to a (name, value) pair stream, per-uniform = the 1-entry case. Decisions 2, 4, 5.
2. **Base class → reuse `ScriptBehavior`** (no new base; the strongest native signal). Decision 2.
3. **File → `scripts/script.py`** (inside `scripts/`, keeps `reload`'s single-arg contract). Decision 1.
4. **Conflict → BOTH apply in a locked write order; brain writes first, `u_<name>.py` writes last and
   wins** (explicit > general; enables migrating a uniform out of the brain). The maintainer's framing
   ("can we just apply both in a fixed order?") is exactly write-precedence: at one slot the last writer
   wins, and at different slots both apply with no conflict. Decision 7.
5. **Dispatch by filename, sentinel key `"script.py"`, dict non-exhaustive, brain `stub_for` deferred to
   042** — all recommended defaults confirmed. Decisions 3, 6, 8; stub in Out-of-scope.
6. **Sequencing → 044 = the node-brain engine (headless); the UI (covering both paths + the node-level
   error indicator) folds into 042; copilot write-behavior for brains folds into 043.** 044 is a sibling
   of 041 (engine-level), numbered after the already-planned 042/043 to avoid renumbering them.

## Review history

**Pre-impl review round 1 (2026-06-13, 3 adversarial agents — correctness&design / verification&blast /
conventions&native-ness, each anchored to the real source + the 041 before-state).** All three independently
confirmed the architecture is GENUINELY NATIVE (per-uniform IS the 1-entry case; the fan-out is isolated to
`_resolved_pairs`; dispatch is filename-driven, not return-type sniffing). The findings were almost entirely
SPEC-PRECISION gaps, not design flaws — triaged, the real ones folded:
- **[real] `coerce_one`'s third arg collided with `uniform_shape_hint`'s `label`** (the GLSL-type string).
  FIXED decision 4: renamed `error_name`; `coerce_one` derives the GLSL label internally via `gl_type_label`.
- **[real] `_user_error_line` marker double-wrap risk** (passing an already-wrapped `<u:script.py>` would
  yield `<u:<u:script.py>>`). FIXED decision 11: the threaded value is the UNWRAPPED name; the fn keeps
  building `f"<u:{label}>"`; param renamed `marker_name`.
- **[real] conflict apply-order left an unresolved "claimed-set OR two-pass" fork** (all 3 flagged). FIXED
  decision 7: LOCKED to TWO-PASS (all brain pairs, then all per-uniform pairs overwrite), explicitly NOT
  reliant on dict-insertion order.
- **[real] wrong line citations + skip placement** (the unconditional `frozen` write is at engine.py:277-278,
  not 267; the skip must land BEFORE that write). FIXED decision 5 with the precise lines + before-the-write
  placement.
- **[real, minor] `Behavior` Protocol dual-error semantics + `last_driven` reset-persistence + script.py
  discovery is a SEPARATE branch (glob `u_*.py` doesn't match it) + impl-comment guidance at the 3 non-obvious
  seams.** FIXED in decisions 1/4/6/10 + the behavior.py/engine.py Files-touched bullets.
- **[REJECTED] one reviewer returned FAIL on "test pseudo-code not in the spec".** A false standard: the
  dev_flow spec recipe lists test CASES (prose), not code; tests are written at Implement. 041's spec did the
  same. The test matrix here names a falsifier per decision — that is the spec's job; the code is the impl's.
- **[REJECTED] one reviewer argued decision 7 fires 041's compute-order deferral.** Re-read 041 Out-of-scope
  line 60-61: the trigger is "a script depends on another's same-frame OUTPUT" — a READ dependency. 044 has
  no read (no `ctx.uniforms`); both files WRITE the slot, neither reads the other. Strengthened decision 7's
  argument to quote the exact "output" wording so a future reader won't re-trip on it.
All three reviewers' residuals are folded; no open redesign. Implementation-ready (the locked decisions now
carry the precise seams the impl + post-impl spec-fidelity audit will check against).

**Design provenance (2026-06-13, maintainer-directed multi-agent research swarm "node-brain-seams").** 13
agents: 3 readers mapped the engine into shared-core vs per-uniform-specific (with file:line); 3 designers
proposed seam cuts under distinct lenses (unify / compiled-leaf / absolute-minimum); each design was
adversarially critiqued on two axes (illusory-reuse + testability/spec-fidelity); a synthesizer merged the
survivors. The swarm caught three real holes the individual designs glossed, now folded as decisions
9/11/5: the error-marker hardcode (`behavior.py:34` + `engine.py:295` → brain tracebacks at line -1), the
export-isolation flag loss (`fresh_behaviors_for` rebuilds from `sources` with no flag → resolved by making
cardinality a pure function of the key), and the unknown-key write-instead-of-skip (`engine.py:267/278`
writes `None`). The designs' fancier "one shared `update_map` method" framing was DOWNGRADED by their own
critiques to interface-sharing not logic-sharing — the spec keeps the honest shared atoms (`run` +
`coerce_one` + the per-name freeze body) and synthesizes per-uniform's 1-entry map in `_resolved_pairs`.

**Post-impl dogfood hardening (2026-06-13, direct-engine workflow — 4 creative scenes + 3 edge sweeps on
the Pi's V3D, no copilot).** The engine was confirmed crash-free + state-isolated on every adversarial path;
the dogfood surfaced failure-VISIBILITY gaps (not correctness), all fixed in the follow-up commit:
- **Engine-owned brain guard (was a BUG).** The brain's per-key gate now routes through the SAME
  `_binding_reject` the per-uniform path uses at reload, so a brain key naming `u_time`/`u_aspect`/
  `u_resolution` (or an orphan/typo, or a sampler/block) is rejected at tick: a soft `(node_id, name)`
  `ScriptError` + skip, and it does NOT enter `last_driven` (so `script_driven_uniforms` no longer falsely
  claims ownership — the false claim would have made the copilot `set_uniform` reject refuse a real user set).
- **NaN/Inf frozen-as-data.** `coerce_one` rejects a non-finite coerced value (records a runtime
  `ScriptError` + freezes), instead of writing it to the GPU silently (a black frame + poisoned last-good).
  Shared atom → covers both paths.
- **Typo'd-key soft error.** A brain key naming no active scriptable uniform records a soft `(node_id, name)`
  error (not loguru-only) so 042's UI surfaces it; cleared zombie-free when the key stops being returned.
- **Conflict-freeze-fallback decision (variant B, maintainer-approved).** When a `u_<name>.py` that conflicts
  with a brain on one slot is BROKEN, the slot yields to the brain's live value (a broken override lets the
  base behavior show through) rather than freezing over the brain's write — making the outcome deterministic
  regardless of whether the per-uniform ever succeeded. The per-uniform's error is still surfaced. Threaded
  via a `brain_driven` set into the per-uniform pass's freeze. Recorded in `conventions.md`'s script bullet.
- **DX papercuts folded:** friendly `import` message (`__import__` shim), `chr`/`ord` in the script globals,
  an `Array`-of-nested-tuples flatten hint. **Accepted limitation (not a bug):** a coercion/non-dict error
  reports `ScriptError.line == -1` (the mismatch is detected outside the user's `update` frame; the message
  names the contract). 042 may tag the return-statement line if it proves worth it.
