# Feature 041 — stateful-class script engine (040 redesign)

> Status: IMPLEMENTED + post-impl-review CONVERGED (2026-06-13) — headless engine landed on `dev`;
> `make check` + 28 CPU + 4 GL tests + smoke + the dogfood determinism/export-isolation check all green
> on the Pi's V3D. Post-impl review: 3 agents x 2 rounds, a BLOCKER (GUI export paths un-bracketed) + 3
> bugs fixed, round 2 PASS. This SUPERSEDES the feature-040 engine contract
> (`out.set()` body-only scripts) before 040 ever shipped publicly (last release is v0.15.0, 040 is on
> `dev` unshipped — no migration burden). 041 redesigns the engine so a script is a **stateful class**
> the user finalizes (a base subclass with a user-implemented `update(self, ctx)` method that RETURNS a
> typed value), making per-frame STATE first-class. v1 is the engine ONLY, headless (like 040 was) —
> the in-app UI + `u_mouse` are feature **042**.

## Goal

Make a uniform's behavior script a **stateful object**, because state is the entire reason to compute a
uniform on the CPU: a stateless `sin(t)` can be written directly in the shader, so the only scripts
worth dropping to Python are the ones that carry state across frames — integrators, springs, easings,
finite-state machines, a pong ball's position. 040 shipped a stateless `out.set(...)` body that could
not naturally hold state (it re-`exec`'d a flat body each tick with nowhere to put `self`). 041
replaces it: a script is a user-finalized **class** subclassing an implicit base `ScriptBehavior`, with:

- `__init__(self)` — optional, holds per-instance STATE (`self.*`), persisting across frames.
- `update(self, ctx) -> <output>` — the user implements this; called once per tick; RETURNS the
  uniform's value as a typed result.

The user never writes the class from scratch: the engine generates a **type-aware stub** for the target
uniform (the right return type + a docstring documenting `ctx` + the output, pre-shaped). The engine
`exec`s the file VERBATIM (no rewrite — keeping 040's guarantee that an error's lineno points at the
user's real source), finds the `ScriptBehavior` subclass, instantiates it ONCE, and calls `.update(ctx)`
each tick. A bad script is error-as-data (freeze last-good + record a `ScriptError`), exactly as 040.

This also FOLDS IN two things 040 deferred to 042: per-frame STATE is now the core model (old-042's
shared-`ctx.state` machinery is deleted — instance state replaces it), and **export builds a FRESH
instance per render** so an exported integrator starts from a clean state (resolving the 040
`ctx.uniforms`-integrator export double-tick deferral by construction).

## Out of scope (each deferral carries a trigger)

- **In-app UI (the script-driven row indicator, the script chip, the editor affordance, the error
  surfaced in-app, the manual state-RESET button).** v1 scripts are hand-placed + hot-reloaded, errors
  log-only — same posture 040 shipped with. Trigger: feature **042** (the script UI + `u_mouse`).
- **`u_mouse` / real cursor in `ctx.mouse`.** `EngineContext` carries the clock only in v1
  (`t/dt/frame`). The mouse field + live wiring + the fixed-`(0.5,0.5)` export value are feature **042**.
  (v1 leaves `ctx` WITHOUT a mouse field until 042 — keeping the surface minimal; 042 adds it.)
- **Copilot write-behavior tool.** No copilot tool authors/edits scripts in v1; that's feature **043**,
  which must write the CLASS form (and `_capture_node`-equivalent the script file into the turn
  checkpoint — the `todo.md` rollback-capture deferral fires at 043, not here).
- **Scripting non-scalar/non-vector uniform kinds beyond what coercion handles.** v1 scripts target the
  kinds the engine can drive: scalar (`float`/`int`/`uint`), small vector (`vec2/3/4`), and the existing
  capped arrays (`float[N]`, `vecN[M]`, the `uint[N]` text array). Samplers, UBOs, matrices, and `bool`
  are NOT scriptable (matrices have no coercion path; `bool` is actively rejected by `is_number` — both
  pre-existing). Trigger: a concrete need (matrices → a coercion + an output-type design; `bool` → lift
  the `is_number` rejection deliberately).
- **A cross-uniform shared-state channel (one script writes, another reads).** Old-042's `ctx.state`
  shared dict is REMOVED — instance state (`self.*`) covers per-uniform state, which is the pong-ball
  case (the physics lives in ONE behavior). Trigger: a real workflow needs two scripts to share mutable
  state (then a shared channel returns as a deliberate design, with the read/write ordering invariant —
  NOT a `dict` smuggled back into `ctx`).
- **Cross-uniform compute order guarantees.** v1 ticks a node's bindings in a stable order (sorted by
  name) but makes NO promise that binding A's `update` runs before B's. Trigger: a script depends on
  another's same-frame output (needs the shared-state channel above + an explicit phase order).

## Design decisions (numbered; lock-in only)

> Maintainer scoping answers (2026-06-13) are folded in — see the *Plan-lock resolution log*.

1. **A script is a user-finalized class subclassing `ScriptBehavior`, with `update(self, ctx) ->
   <output>`.** The persisted file (`nodes/<id>/scripts/u_<name>.py`) literally contains a class
   definition:
   ```python
   class Behavior(ScriptBehavior):
       """<generated docstring: ctx fields + the expected return type>"""
       def __init__(self) -> None:
           self.phase = 0.0          # per-instance STATE — persists across frames

       def update(self, ctx: Ctx) -> Vec2:
           self.phase += ctx.dt
           return Vec2(0.4 * cos(self.phase), 0.4 * sin(self.phase))
   ```
   `ScriptBehavior`, `Ctx`, the output types (`Vec2/3/4`, the capped `Array`/`Text` — decision 3), and
   the math namespace (`sin`/`cos`/`clamp`/…) are ALL implicitly in the exec namespace — the user
   imports nothing. The engine `exec`s the file verbatim, then finds the FIRST `ScriptBehavior` subclass
   defined in the resulting namespace (by class name `Behavior` first, else any subclass — decision 8),
   instantiates it once (`__init__` with no args), and calls `.update(ctx)` each tick. **The class name
   is `Behavior` by convention** (the stub uses it); the resolver tolerates any subclass so a rename
   doesn't break it. `update` is REQUIRED; `__init__` is optional. Revisit if a second lifecycle method
   earns its place (e.g. an `on_reset()` distinct from `__init__` — only if the manual-reset affordance
   at 042 needs a hook beyond re-instantiation).

2. **Verbatim exec — NO rewrite (040's no-AST-surgery guarantee preserved).** Because the file IS a
   class definition (not a body fragment wrapped into one), the engine NEVER edits the user's source:
   `compile(file_text, "<u:u_name>", "exec")` then `exec` into a namespace seeded with the implicit
   names. A `SyntaxError` or a runtime exception's `lineno`/traceback already points at the user's real
   line — no wrapper, no line-remap (the 039 ghost stays dead). This is the SAME guarantee 040 made; it
   survives the class redesign precisely because there's still no source transform. A compile failure
   (bad class syntax, no `ScriptBehavior` subclass found, no `update` method) makes a `Behavior` that
   permanently freezes + carries a `ScriptError(kind="compile")` until the file changes. Revisit only if
   the no-sandbox/no-rewrite posture is ever reversed (it won't be for a personal IDE).

3. **Typed outputs correspond DIRECTLY to GLSL types; only the shaped/capped ones get a type — scalars
   are returned bare (resolves the output-type question, user-convenience-first).** What `update`
   returns:
   - **scalar** (`float`/`int`/`uint` uniform) → a bare Python number: `return 0.5`. No wrapper — zero
     ceremony for the common case.
   - **vector** (`vec2/3/4`) → `Vec2(x, y)` / `Vec3(x, y, z)` / `Vec4(x, y, z, w)` — the type names the
     GLSL shape, reads like the shader, and the stub already supplies it.
   - **capped array** (`float[N]`, `vecN[M]`) → `Array(values)` (the cap N is the uniform's; the type
     carries "this is the whole array" intent and validates length).
   - **text array** (`uint[N]` glyph uniform) → `Text("Hello")` (carries the char cap; string →
     codepoints, the existing `str_to_unicode` path).
   **Normalization contract (pin — the single most likely impl bug):** `coerce_uniform_value` validates
   a vector via `isinstance(value, list | tuple)` and an array via `all(is_number(v))` on a FLAT
   sequence. So the engine MUST hand coercion a plain `list|tuple|str|number`, NOT a bare custom object
   — a `Vec3` object that is not a `tuple` subclass FAILS the `isinstance` check → `None` → a spurious
   freeze. Therefore: `Vec2/3/4` subclass `tuple` (or the engine unwraps to `tuple(out)` before
   coercion); `Array` holds a FLAT sequence of numbers (a `vec2[3]` Array is `[x0,y0,x1,y1,x2,y2]`, NOT
   `[Vec2,Vec2,Vec2]` — coercion's `coerce_array` expects `n*dim` flat numbers); `Text` normalizes to
   the raw `str` (coercion's `str_to_unicode` branch). The engine calls
   `coerce_uniform_value(output.normalize(), uniform)` where `normalize()` yields exactly one of
   `number | list | tuple | str`. The output types are thin value-carriers (`Vec3` = a 3-tuple
   subclass; `Array`/`Text` wrap a flat sequence / str). The engine validates the normalized value
   against the actual uniform via the SAME `coerce_uniform_value` (decision 6) — a `Vec3` returned for a
   `vec2` uniform → a runtime `ScriptError` freeze (the type names intent; the uniform is the source of
   truth). A bare number for a vector uniform (or vice-versa) → the same clean shape error, NOT a murky
   GL crash. **No separate `Float`/`Int` type** — a scalar is just a number (decided: convenience). The
   scalar stub returns `0.0` (a float, not `0`) so a `float` uniform gets a float (decision 7). Revisit
   the type set when matrices land (a `Mat3` joins the family + its coercion path) or if `Array`/`Text`
   prove too thin.

4. **State lives in the instance; lifecycle = fresh instance on (re)load + fresh per export render + a
   manual reset (042).** The behavior instance is created:
   - **on first compile + on every hot-reload** (a `(path, mtime)` change recompiles → re-instantiates,
     so editing the script resets its state — predictable: a code change is a fresh start). On project
     load, the same — one instance per binding.
   - **fresh per EXPORT render** — the export entry builds a NEW behavior instance for the node it
     renders, so an exported integrator starts from `__init__`'s clean state regardless of the live
     instance's accumulated state. This resolves the 040 `ctx.uniforms`-integrator export double-tick
     deferral **via the export seam in decision 11** (NOT "by construction" — it requires the live and
     export tick paths to be SPLIT so export ticks a separate instance; today they are the SAME path —
     see decision 11). With the split, the live frame's state can no longer poison the export's frame-0
     start (separate instances). The LIVE path keeps its long-lived instance across frames (that's the
     point — state accumulates). **The proof is the live-then-export interleave test** (Manual
     verification): accumulate state on the live instance across N ticks, THEN export, and assert the
     export's frame-0 equals the clean-`__init__` baseline, NOT the live-accumulated value. A weaker
     "two exports reproducible" test would pass even if export shared the live instance — insufficient.
   - **manual reset** — re-instantiate a live binding on demand (re-run `__init__`) WITHOUT editing the
     file. The engine exposes `reset(node_id, name)`; the UI button is feature **042** (for pong-style
     "restart"). v1 ships the engine method; the affordance is 042.
   Determinism is now SHARPER than 040's scoped rule: an export render of a `ctx.t`-pure `update` is
   reproducible (fresh instance, deterministic `t`); a STATEFUL `update` (reads `self`) is
   path-dependent in LIVE (variable `dt`) but its EXPORT is reproducible across runs (fresh instance +
   fixed `dt=1/fps`). Live≠export for an integrator remains true by nature (variable vs fixed dt) —
   documented, not a bug. Revisit if a script needs state to PERSIST into an export from the live
   session (it shouldn't — export is a clean render).

5. **`EngineContext` carries the clock only in v1 (`t/dt/frame`); mouse is 042.** `Ctx` (the name in
   scope; `EngineContext` is the class) is the one read-only object `update` receives: `t` (elapsed s),
   `dt` (s since last frame), `frame` (int index). **The 040 `state` and `uniforms` fields are REMOVED**
   — instance state replaces `state`; an integrator reads its OWN previous value from `self`, not from a
   `ctx.uniforms` snapshot (cleaner: state is where state belongs). `mouse` is added by 042. Keeping
   `ctx` to three fields makes the stub docstring tiny and the surface honest. **COMPILE-BREAKING edits
   this forces (must all land in the impl commit, or `make check` reds):** (a) `EngineContext` drops
   `mouse`/`state`/`uniforms` (3 fields removed); (b) `ScriptEngine.state` attribute is DELETED (the
   phase-A scratchpad — no longer exists); (c) `project_session.py::_tick_node` drops the
   `state=self.script_engine.state` and `uniforms=dict(...)` kwargs from its `EngineContext(...)` call
   (it won't construct otherwise). Revisit per 042 (adds `mouse`) / a real need for light read-only meta
   (app/node name — trigger-gated, was explicitly deferred).

6. **Output validation REUSES `uniform_coerce` (no new coercion).** `update`'s return value is shaped
   against the uniform via the existing `coerce_uniform_value(value, uniform)` (the `uniform_coerce.py`
   leaf, 040 hoist). The typed outputs normalize to what coercion expects: `Vec3(r,g,b)` → the tuple
   `(r,g,b)` coercion validates against `dim==3`; a bare number → the scalar path; `Array`/`Text` → the
   array/text-codepoint paths. A `None` from coercion (shape mismatch) → a runtime `ScriptError` carrying
   `uniform_shape_hint`'s message, freeze last-good. The scriptable-kind GATE is the existing
   `_is_scriptable` predicate (scalar/vector/array, NOT sampler/UBO; matrices/`bool` fall outside what
   coercion handles and so are non-scriptable — honest, no false promise). Revisit when the scriptable
   set grows (matrices → both a coercion branch AND a `Mat*` output type, together).

7. **Type-aware stub generation lives on the engine (a node/uniform → a ready-to-edit class).** The
   engine produces the initial script text for a given uniform: the right `update` return type
   (`-> Vec2` for a vec2, bare number for a scalar, `-> Array`/`-> Text` for the capped kinds), a
   docstring documenting the `ctx` fields + the expected output, a minimal `__init__`/`update` body that
   returns a shape-valid default (so a freshly-created script compiles + runs immediately, showing the
   default value, not an error). `ScriptEngine.stub_for(uniform) -> str` is the public seam; 042's "Add
   script" writes it. v1 ships `stub_for` + a unit test asserting each kind's stub compiles, instantiates,
   and returns a coercion-valid value. Revisit the stub content when the snippet-library deferral fires.

8. **The `Behavior` subclass resolver (compile-time): find the user's class deterministically.** After
   `exec`, the engine resolves the behavior class: prefer a class literally named `Behavior` that
   subclasses `ScriptBehavior`; else the FIRST `ScriptBehavior` subclass defined in the namespace
   **excluding `ScriptBehavior` itself** (the base is in the exec globals, so `v is not ScriptBehavior`
   is required in the scan); if none, a compile `ScriptError("no ScriptBehavior subclass found — keep
   the `class Behavior(ScriptBehavior)` line")`. If the class has no `update` (didn't override the base
   — `cls.update is ScriptBehavior.update`), a compile error naming it. Instantiation (`cls()`) failing
   (a raising `__init__`) is a compile-time `ScriptError` too (the binding can't even start). **The
   export fresh-instance path (decision 11) instantiates too — a raising `__init__` there records a
   `ScriptError` and freezes the export uniform at its seeded default, NEVER crashing the encode.** This
   resolve+instantiate happens once per (path, mtime) live / once per export; the per-tick path only
   calls `.update`. Revisit if multiple behaviors per file is ever wanted (it isn't — one file = one
   uniform = one class).

9. **`ScriptError` shape UNCHANGED; the freeze-as-data contract UNCHANGED.** The 040 `ScriptError`
   (`uniform_name`, `kind: compile|runtime`, `message`, `line`) and the engine's `errors[(node_id,
   name)]` map are kept verbatim. A compile error (bad class / no subclass / no `update` / raising
   `__init__` / `SyntaxError`) freezes permanently until the file changes; a runtime error (`update`
   raises, output coercion returns None) is caught per-tick. EITHER way the uniform freezes at last-good
   and the frame continues — never raises into the loop. This is 040's decision 8, intact. Revisit only
   if a third error kind earns a UI distinction (042's call, not the engine's).

10. **Package shape: redesign `behavior.py` + `engine.py`; `context.py` slims; ADD `outputs.py`.**
    - `scripting/context.py` — `EngineContext` slimmed to `t/dt/frame` (drop `mouse`/`state`/`uniforms`;
      042 re-adds `mouse`). Aliased as `Ctx` in the exec namespace.
    - `scripting/outputs.py` (NEW leaf) — `Vec2`/`Vec3`/`Vec4` + `Array` + `Text` value types (thin,
      coercion-targeting normalizers). No GL, no engine import.
    - `scripting/behavior.py` — `ScriptBehavior` (the base the user subclasses: `update(self, ctx)`
      raising `NotImplementedError` by default), the `Behavior` Protocol (the engine's seam — still
      language-swappable), and `PythonBehavior` (compile the file verbatim, resolve the subclass,
      hold the live instance, `compute(ctx)` → `instance.update(ctx)` → coerce → record/freeze). The
      `UniformOut` sink is REMOVED (no `out.set`). The math namespace builder stays here.
    - `scripting/engine.py` — `ScriptEngine`: per-node registry, `(path, mtime)` cache (now caching a
      `PythonBehavior` that owns its instance), reload/resolve/orphan-warn, the per-tick `tick`
      (calls `behavior.compute(ctx)`), `reset(node_id, name)` (re-instantiate), `fresh_behavior_for`
      (export's clean instance), and `stub_for(uniform)`. `script_driven_uniforms` unchanged.
    - `scripting/errors.py` — `ScriptError` UNCHANGED.
    - `scripting/__init__.py` — export `EngineContext`/`Ctx`, `ScriptBehavior`, the output types,
      `Behavior`/`PythonBehavior`, `ScriptEngine`, `EngineNode`, `ScriptError`. Drop `UniformOut`.

11. **Export instance isolation — STRUCTURAL, inside `Node.render_media` (the gap that defeated 040, now
    bypass-proof).** The live tick and the export hook were the SAME path (`node.on_pre_render` → the LIVE
    instance), so "fresh per export" needs them SPLIT — and the split must fire on EVERY export without a
    caller opting in. **LANDED AS an injected `Node.export_isolation` factory that `render_media` enters
    itself**, so no export caller can bypass it:
    - `Node` gains `export_isolation: Callable[[], AbstractContextManager[None]]` (default
      `contextlib.nullcontext` — a bare Node / no scripts is a no-op). `render_media` wraps its WHOLE body
      in `with self.export_isolation():`. Since every export funnels through `render_media` (Render tab,
      Share scratch via `render_to`, all copilot render tools), the bracket fires once per export,
      structurally — a NEW export caller gets isolation for free (no `with` to forget). `Node` stays
      engine-free: it only enters an OPAQUE injected context manager (the same shape as `on_pre_render`
      being an opaque injected callback — the 025 boundary holds).
    - `ProjectSession._make_export_isolation(node_id)` builds the factory, injected at load beside
      `on_pre_render` (`_resolve_scripts`). Entering it swaps `node.on_pre_render` to tick a FRESH behavior
      set (`engine.fresh_behaviors_for(node_id)` — NEW instances independent of the live registry), then
      restores the live hook in `finally`. The export loop (`_render_video`/`_render_image`) fires
      `on_pre_render` per-frame, so inside the bracket it ticks the fresh set; the live `ui.py` path
      (outside any render_media) ticks the long-lived one.
    - **Why the injected factory, NOT a per-caller `with session.exporting(...)` (the second pivot):** the
      first impl wired a `session.exporting(node_id)` context manager that EACH export caller had to wrap —
      a per-call-site discipline a new caller could forget (a post-impl review caught the Render/Share tabs
      un-wrapped). Moving the bracket INTO `render_media` makes it impossible to bypass: the single funnel
      every export already passes through owns the isolation. (The even-earlier `render_node` funnel idea was
      rejected because the three callers don't share a signature; `render_media` IS the shared funnel they
      all already reach, so the bracket belongs there.)
    - A fresh instance whose `__init__` RAISES freezes (records a compile `ScriptError`), NEVER crashes the
      encode — the export tick path catches it (decision 8).
    - **`fresh_behaviors_for` recompiles from the live registry's CACHED source, NOT a fresh disk
      read** — avoids a mid-edit half-saved read during export, and reuses the `(path,mtime)`-captured
      body so the export compiles the same source the live preview last compiled.
    The LIVE `ui.py` tick path is unchanged — `session.tick(node_ids, ...)` ticks the long-lived registry.
    The harness `export_at` enters `node.export_isolation()` directly (it renders at a single chosen `t`,
    not a full `render_media` sequence). Revisit if the export loop needs to tick MULTIPLE nodes with shared
    state (it renders one node at a time — not v1).

12. **The `exec()` seam: no-sandbox posture kept, but `__builtins__` is NOT empty (the class statement
    needs it) — corrected from 040.** `compile()` + `exec()` of the user file, no sandbox (the locked
    posture for a personal IDE). **040's "`__builtins__` emptied" trick does NOT survive the class
    model** (empirically verified at pre-impl review): `class Behavior(ScriptBehavior):` compiles to a
    call to `__build_class__`, which lives in `__builtins__` — with `{}` builtins it raises
    `NameError: __build_class__ not found` on frame one; the class statement ALSO reads `__name__` from
    the module globals, so the exec globals must set `__name__` too. So the exec globals are:
    `{"__builtins__": {"__build_class__": __build_class__, ...the curated math vocab...}, "__name__":
    "<u:u_name>", "ScriptBehavior": ScriptBehavior, "Ctx": EngineContext, "Vec2": Vec2, ...}`. The
    curated vocabulary (sin/cos/clamp/Vec*/Array/Text) lives as top-level globals names (free-variable
    lookup inside `update`'s body falls through to module globals — verified). **SECOND-ORDER (pin —
    empirically verified at review): method annotations evaluate EAGERLY at class-def/exec time, because
    `from __future__ import annotations` is BANNED by convention.** So `def update(self, ctx: Ctx) ->
    Vec2:` does a live lookup of BOTH `Ctx` AND `Vec2` at exec time (before any tick) against the
    exec-GLOBALS dict — not `__builtins__`, not "available to the body." Therefore **EVERY type name any
    stub annotation can emit — `Ctx`, `Vec2`, `Vec3`, `Vec4`, `Array`, `Text` — MUST be a top-level key
    in the exec-globals dict**, all six, unconditionally (NOT a per-uniform subset matching the target
    type). A stub annotating `-> Array` while the globals seeded only `Vec*` would `NameError` on frame
    one → a freshly-generated stub becomes a permanent compile-freeze. (Seed the full output-type set in
    the globals regardless of which uniform the script targets.) The no-sandbox intent is unchanged (we
    expose a curated set, not the full builtins); what changes is that a few dunder builtins
    (`__build_class__` at minimum) MUST be present for `class` to work at all. 040 found this
    needed NO `# type: ignore` (ruff `S102` isn't in `select`, pyright basic is clean on the
    curated-namespace exec); re-verify on the class-resolve path. If a marker IS needed, it goes on the
    `conventions.md ## Known quirks` allowlist with the 040 rationale, in the same wave. **The 040
    conventions note ("the exec engine seam needed NO suppression") stays true; the "empty builtins"
    claim there must be corrected to "curated builtins carrying `__build_class__`."**

## Files touched (mapped to the dev_flow module map)

- **`shaderbox/scripting/behavior.py`** — REWRITE: remove `UniformOut`; add `ScriptBehavior` base; make
  `PythonBehavior` compile-the-file → resolve-the-subclass → hold-an-instance → `compute(ctx)` calls
  `instance.update(ctx)`, coerces the typed return, freezes/records on error. Keep the math namespace +
  the `Behavior` Protocol. Add `reset()` (re-instantiate).
- **`shaderbox/scripting/engine.py`** — `tick` now calls `behavior.compute(ctx)` (no `UniformOut`
  construction); the freeze/error/last-good logic stays. Add `reset(node_id, name)` (re-instantiate one
  live binding), `fresh_behaviors_for(node_id)` (compile the node's scripts into a NEW independent
  behavior set the export seam ticks — decision 11), and `stub_for(uniform) -> str` (decision 7).
  `_is_scriptable` → public `is_scriptable` (042's Add-script gate + the stub generator both need it).
  DELETE `ScriptEngine.state` (decision 5).
- **`shaderbox/scripting/context.py`** — slim `EngineContext` to `t/dt/frame`; the `Ctx` alias.
- **`shaderbox/scripting/outputs.py`** — NEW leaf: `Vec2`/`Vec3`/`Vec4`/`Array`/`Text`.
- **`shaderbox/scripting/__init__.py`** — update the public surface (drop `UniformOut`; add
  `ScriptBehavior` + outputs + `Ctx`).
- **`shaderbox/project_session.py`** — (a) `_tick_node` builds the slimmed `EngineContext(t,dt,frame)`
  (DROP the `state=`/`uniforms=` kwargs — compile-breaking per decision 5). (b) the EXPORT seam
  (decision 11): `begin_export(node_id)`/`end_export(node_id)` swap `node.on_pre_render` to a fresh-instance
  tick and restore it; `_make_pre_render` (the export closure) stays the export hook but now ticks the
  fresh set. (c) the live `tick(node_ids,...)` keeps calling `_tick_node` against the long-lived registry
  (LIVE path unchanged). `get_script_driven_uniforms` unchanged. (No UI getters here — those are 042.)
- **`shaderbox/scripting/engine.py`** — additionally: DELETE `ScriptEngine.state` (the phase-A
  scratchpad — compile-breaking per decision 5; `_tick_node` referenced it).
- **`shaderbox/core.py`** — UNCHANGED. `Node.render_media`/`_render_media_into`/`_render_image`/
  `_render_video` keep their shape; the `on_pre_render` SIGNATURE stays `(t,dt,frame)`. The export
  isolation bracket lives in the NEW `ProjectSession.render_node` funnel (decision 11), NOT in `core.py`
  (keeps `Node` engine-free, 025). Listed to confirm `core.py` does not change.
- **`shaderbox/project_session.py`** (export funnel) — NEW `render_node(node_id, details_or_preset, ...)`
  that brackets `node.render_media(...)` with `begin_export`/`end_export` (decision 11). `tabs/render.py`
  + `tabs/share_state.py::render_to` + `copilot/backend.py` render tools reroute through it (they call
  `Node.render_media` directly today). `share_state.render_to` (a free fn taking a bare `Node`) gains the
  session/node_id or moves under the funnel — impl picks; the constraint is all three export paths go
  through `render_node` so the fresh-instance bracket always fires.
- **`shaderbox/tabs/render.py` + `shaderbox/tabs/share_state.py` + `shaderbox/copilot/backend.py`** —
  reroute their `Node.render_media` / `render_to` export calls through `ProjectSession.render_node`
  (decision 11). Behavior-only (same render output); the only change is the export bracket now fires.
- **`scripts/smoke.py`** — the seeded scripted-uniform node (040, `u_wave.py` = `out.set(0.5 + 0.3 *
  sin(ctx.t))` on a `float u_wave`) migrates to the CLASS form. **PIN the exact body (smoke asserts only
  no-crash, so a shape-mismatch would silently freeze and pass — the body must be correct):**
  ```python
  class Behavior(ScriptBehavior):
      def update(self, ctx: Ctx) -> float:
          return 0.5 + 0.3 * sin(ctx.t)
  ```
  A bare-float scalar return for the `float u_wave` uniform. No new assert (correctness is the unit/GL
  tests; smoke owns "App-with-a-class-script-node runs 200 frames without crashing").
- **`scripts/dogfood/verify_script_engine.py`** — migrate to the class form. **PIN two bodies:** (1) a
  `ctx.t`-pure scalar (`return sin(ctx.t)`) for the determinism assertion (same `t` → same value, live
  and export); (2) a STATEFUL integrator (`__init__: self.v = 0.0`; `update: self.v += ctx.dt; return
  self.v % 1.0`) for the export-isolation assertion. **The export assertion is REQUIRED and must
  interleave (not just "two runs reproducible"):** run the LIVE `session.tick` loop to t≈5s so the live
  integrator accumulates, THEN render an EXPORT frame at t=0 via the export seam (decision 11 — NOT
  `harness.render_at`, which is the LIVE path), and assert that export frame-0's pixels match a
  COLD-START render (fresh instance), NOT the warmed-up live value. NOTE: the existing check probes two
  `t` values with hardcoded luma-diff thresholds + a "t=0 -> 0.5" comment tied to the OLD `0.5+0.45*sin`
  body — migrate the probe t-values + comments WITH the new body (don't leave a stale "-> 0.5" claim);
  prefer using the stateful integrator as the animate-across-t probe so determinism + isolation share one
  node.
- **`scripts/dogfood/harness.py`** — ADD an export-path render entry (the existing `render_at` ticks the
  LIVE engine via `session.tick`, so it CANNOT exercise the fresh-per-export instance). New
  `harness.export_at(t, node_id)` (or route a flag through `render_at`) that goes through the export seam
  (`begin_export`/`end_export` + the `_render_image`/`on_pre_render` fresh-instance path), so the
  determinism check can compare a LIVE render vs an EXPORT render of the same stateful node. Without this
  the export-isolation claim is unverifiable headless.
- **`tests/test_script_engine.py`** — REWRITE to the class model. The required cases, each pinned (a
  generic "each output type coerces" bullet under-tests `Text`/`Array` — name them separately):
  - **state accumulates**: a `self.v += ctx.dt` integrator; tick N times; assert the value grew.
  - **state RESETS on edit (VALUE, not object-identity)**: tick to accumulate, edit the file (mtime
    bump), tick once, assert the value dropped back to the `__init__` BASELINE (not just that the
    behavior object changed — the 040 identity-only cache test is too weak here).
  - **`reset()` clears state (VALUE)**: tick to accumulate, `reset(node_id,name)`, tick once, assert
    baseline.
  - **export isolation (INTERLEAVE)**: accumulate on the LIVE instance N ticks, then obtain the
    fresh export behavior set (`fresh_behaviors_for`) and tick it at frame 0, assert it equals the
    clean-`__init__` baseline and is NOT the live-accumulated value. (Pure-CPU — no GL needed.)
  - **`ctx.t`-pure determinism**: same `t` → same value across different `dt`.
  - **integrator divergence by design (re-ported from 040 to `self`)**: the old test read
    `ctx.uniforms`; now the integrator reads `self.prev`. Assert the same elapsed time reached via
    different `dt` step counts diverges (the live-vs-export-dt divergence, documented as expected).
  - **output types — SEPARATE tests**: bare scalar (float) ; `Vec2/3/4` (coerces to the dim-tuple,
    and is a `tuple` subclass so the `isinstance` check in coercion passes — decision 3) ; `Array`
    for `float[N]`/`vecN[M]` (flat sequence, length validated, a wrong length freezes) ; `Text("Hi")`
    for a `uint[N]` text uniform (→ codepoints via `str_to_unicode`, over-cap truncates).
  - **shape mismatch freezes + records**: e.g. `Vec3` returned for a `vec2` uniform → runtime
    `ScriptError`, last-good held.
  - **compile errors at the user line**: `SyntaxError` ; no `ScriptBehavior` subclass ; subclass
    missing `update` ; a raising `__init__` — each a compile `ScriptError` with the right `kind`/line.
  - **`(path,mtime)` cache**: no re-instantiate when mtime unchanged (object identity stable);
    fresh instance on change.
  - **`stub_for` per kind**: scalar / vec / `Array` / `Text` stub each compiles, instantiates, and
    `update` returns a coercion-valid value (so a freshly-created script runs, not errors).
  - perf sanity (not a gate).
- **`tests/test_script_engine_gl.py`** — update to the class form; assert a stateful `update`'s value
  reaches the GPU and changes across ticks; a fresh export instance renders clean.
- **`tests/test_script_driven_reject.py`** — UNCHANGED conceptually (the copilot `set_uniform` reject
  keys on `script_driven_uniforms`, which is unchanged) — but verify it doesn't reference `UniformOut`/
  `out.set`. (The reject itself is 040 decision 5, untouched.)
- **`ai_docs/features/040_uniform_script_engine.md`** — (a) add a top banner note: "engine CONTRACT
  superseded by 041 (stateful-class model); the `out.set()` body design never shipped publicly. The
  surrounding architecture (headless ProjectSession ownership, binding-by-filename, error-as-data,
  export-routes-through-the-tick, the coercion hoist) is RETAINED by 041." Don't delete 040 — it's the
  origin; 041 is the contract revision. (b) RETARGET the four in-body `feature 041`(=the editor/UI)
  references — in `## Out of scope` (the in-app UI bullet + the `set_uniform` indicator bullet + the
  type-change/orphan bullet) and Manual-verification ("surfaced in UI at 041") — to **042**, since the
  renumber moves the UI to 042. The `feature 043` (copilot write-behavior) refs stay.
- **`ai_docs/conventions.md`** — update the feature-040 `## Design decisions` bullet: the script value
  is returned by a stateful `ScriptBehavior.update(self, ctx)` (NOT `out.set`); state is first-class
  (instance state, no `ctx.state`); export ticks a fresh instance. Keep the rest (headless ownership,
  filename binding, error-as-data, the `Behavior`-backend seam). One canonical home — edit in place.
- **`ai_docs/roadmap.md`** — the 040 row gains a "(contract superseded by 041)" note; add the 041 row +
  rewrite the banner (next = 042 UI + mouse). Renumber the downstream plan (042 = UI+mouse, 043 = copilot
  write-behavior, unchanged number).
- **`ai_docs/todo.md`** — the 040 `ctx.uniforms`-integrator export double-tick deferral is RESOLVED by
  041 (fresh export instance) → DELETE it, but **only in the SAME commit as the export-isolation
  interleave test that PROVES it** (don't remove the landmine marker before the test defuses it — the
  deletion is premature otherwise). The 042 phase-A/`ctx.state` deferral is OBSOLETE (state is the
  instance now, no shared phase-A) → DELETE / rewrite to the residual (cross-uniform shared state,
  trigger-gated per Out-of-scope). The type-change/orphan-cleanup deferral keeps its trigger but now
  points at 042's UI (unchanged). The 043 rollback-capture deferral is unchanged.

## Manual verification

> Display-less Pi — `make run` is the maintainer's; the agent runs the headless backbone. 041 has NO
> UI, so the headless backbone is the WHOLE agent-side verification (the visual confirm is 042's).

**Headless backbone (the agent runs these):**
1. `uv run pytest tests/test_script_engine.py` — the class-model invariants: state accumulates,
   state-resets-on-edit + `reset()` (VALUE back to baseline), the export-isolation INTERLEAVE
   (accumulate-live → fresh-export → cold-baseline), output types coerce (separate `Vec*`/`Array`/`Text`/
   scalar), errors-as-data at the user line (incl. no-subclass / no-update / raising-`__init__`),
   `stub_for` per kind.
2. `uv run pytest tests/test_script_engine_gl.py` — a stateful `update` reaches the GPU + changes across
   ticks; a fresh export instance renders clean (standalone EGL; skips if no GL).
3. `make check` — ruff + pyright 0 errors (confirm no new suppression beyond the 040 allowlist —
   decision 12; the curated-builtins exec still needs none).
4. `make smoke` — the class-form scripted node (pinned body) ticks clean for 200 frames, no crash.
5. `scripts/dogfood/verify_script_engine.py` — exits 0: a `ctx.t`-pure export is deterministic AND the
   **export-isolation interleave** holds (a live-warmed integrator's EXPORT frame-0 matches a cold-start
   render via `harness.export_at`, NOT the warmed live value — proves the fresh-instance seam exists).

**`make run` hand-steps (handed to the maintainer — confirms the class model works live, pre-UI):**
1. `make run`, pick a node with `float u_x`. Hand-create `nodes/<id>/scripts/u_x.py`:
   ```python
   class Behavior(ScriptBehavior):
       def __init__(self) -> None:
           self.v = 0.0
       def update(self, ctx: Ctx) -> float:
           self.v += ctx.dt * 0.5     # an INTEGRATOR — only possible with state
           return self.v % 1.0
   ```
   → `u_x` ramps 0→1 repeatedly (the instance accumulates `self.v`). A stateless body could not do this.
2. Edit the script (change the rate) + save → hot-reload re-instantiates (state resets to 0, ramp
   restarts) — confirms (path,mtime) recompile makes a fresh instance.
3. Break the class (delete the `update` line) → `u_x` freezes; no crash; a compile `ScriptError` logged.
   Fix → resumes.
4. A `vec2` script returning `Vec2(cos(ctx.t)*0.4, sin(ctx.t)*0.4)` → the uniform orbits.
5. Export a video of the integrator → it animates the ramp from a CLEAN start (fresh export instance),
   independent of how long the live preview had been running.

## Review sizing

**Proposed: mid → upper end (treat as high-blast for post-impl).** Rationale: it REWRITES the core of
an existing subsystem (`behavior.py` + `engine.py` + the test suite) and changes the export tick path
(fresh-instance isolation) — more than an additive feature. It does NOT cross new module boundaries (the
025 headless ownership, the coercion hoist, the `on_pre_render` seam all stay) and has NO UI surface
(that's 042), so it's not the `ui.py`-split tier. The risk concentrates in: the verbatim-exec +
subclass-resolve correctness (errors point at the user line), the instance lifecycle (state resets on
edit, fresh on export), and the per-tick path not leaking instances.

- **Pre-implementation: 2 review agents.** (a) *correctness & design* — verbatim exec genuinely keeps
  errors pointed at the user's source (no wrapper), the subclass resolver is deterministic + its failure
  modes are clean compile errors, the output-type → coercion mapping is total (every scriptable kind has
  a return shape), instance lifecycle (reset-on-edit / fresh-on-export / manual reset) is coherent, the
  040 deferrals it folds (export double-tick, phase-A/state) are genuinely resolved not just renamed. (b)
  *verification & blast-radius* — does the test rewrite cover the stateful contract (accumulation, reset,
  export isolation), does `stub_for` have a per-kind test, is the smoke/dogfood migration to the class
  form real, is the 040→041 supersede note honest (what's kept vs revised).
- **Post-implementation: 3 review agents in parallel, convergence loop** (high-blast floor): *code
  correctness* (the exec/resolve/instantiate seam, the per-tick error catch never escaping, no instance
  leak in the `(path,mtime)` cache, the fresh-export-instance path, GL lifetime in the GL test),
  *architecture & conventions* (the `outputs.py` leaf boundary, no imgui/glfw in the engine, the ONE
  exec posture, the coercion reuse, the slimmed `ctx`), and a *spec-fidelity audit* (decisions 1–12
  landed). At least one reviewer anchors to a non-self-authored artifact (the 040 engine source as the
  before-state, `tests/test_uniform_seed_save.py` as the GL fixture precedent).

## Review history

**Pre-impl review round 1 (2026-06-13, 2 adversarial agents — correctness & design / verification &
blast).** Both PARTIAL; all findings ACCEPTED (no false positives), folded into the decisions above:
- **[BLOCKER, empirically verified] empty `__builtins__` breaks the `class` statement.** A class def
  calls `__build_class__` (in builtins) + reads `__name__` from globals; 040's `{}`-builtins trick fails
  on frame one. FIX: decision 12 rewritten — curated builtins MUST carry `__build_class__`; globals set
  `__name__`. The reviewer confirmed the rest of the mechanism IS sound once this lands (base-from-globals
  resolution, in-method name lookup, traceback lineno pointing at the user's source — all verified).
- **[BLOCKER] export isolation needs an INTERLEAVE test**, not "two runs reproducible" (which passes even
  if export shares the live instance). FIX: the test plan + dogfood now require accumulate-live → export →
  assert-cold-baseline. AND `harness.render_at` uses the LIVE path (`session.tick`), so it can't verify
  isolation — added `harness.py` (`export_at`) to Files-touched.
- **[MAJOR] the fresh-per-export seam doesn't exist in the "unchanged" path** (export hook = live
  `_tick_node` today — the same reality that defeated 040). FIX: decision 11 now names the concrete
  `begin_export`/`end_export` + `fresh_behaviors_for` seam; the "by construction" claim (decision 4) is
  downgraded to "via the decision-11 seam, proven by the interleave test"; the `core.py` row is
  conditional (unchanged iff a session-level export wrapper brackets all callers, else `render_media`
  brackets).
- **[MAJOR] removing `ctx.state`/`uniforms` is compile-breaking** — enumerated in decision 5 (the
  `EngineContext` fields, `ScriptEngine.state`, and `_tick_node`'s kwargs must all land together) and the
  integrator-divergence test re-ported to read `self.prev`.
- **[MAJOR] output types must normalize to `list|tuple|str|number` BEFORE coercion** (a bare `Vec3`
  object fails coercion's `isinstance(value, list|tuple)` check → spurious freeze — the single most
  likely impl bug). FIX: decision 3 pins the normalization contract (`Vec*` subclass `tuple`; `Array`
  flat; `Text`→str) + separate `Text`/`Array` tests.
- **[MAJOR] smoke/dogfood migration under-specified** (smoke asserts only no-crash, so a wrong body
  silently freezes + passes). FIX: pinned the exact class bodies in the smoke + dogfood rows.
- **[MINOR, accepted]** resolver must exclude `ScriptBehavior` itself (`v is not ScriptBehavior`); a
  raising `__init__` on the export path must freeze not crash the encode; scalar stub returns `0.0` not
  `0`; bool/matrix non-scriptability confirmed honest (coercion returns None → clean freeze); the
  `test_script_driven_reject.py` row is a no-op verify (already clean). All folded.
- **Verdict after fixes:** the one BLOCKER (builtins) is a real falsifier, now corrected; the rest were
  spec-precision gaps (an unspecified seam, under-specified tests) — no redesign.

**Pre-impl review round 2 (2026-06-13, the same 2 agents, against the patched spec).** Verification
reviewer: **PASS** (all 4 round-1 fixes confirmed adequate; the export-isolation interleave is a true
falsifier, pure-CPU-testable via `fresh_behaviors_for` + `_FakeNode`; the pinned smoke/dogfood bodies are
correct + unambiguous; every decision 1–12 contract now has a test). Correctness reviewer: PARTIAL with
findings — all ACCEPTED + folded (none required redesign):
- **[second-order, empirically verified] eager method-annotation evaluation.** `def update(self, ctx:
  Ctx) -> Vec2:` evaluates `Ctx` AND `Vec2` at class-def/exec time (no `from __future__ import
  annotations` — banned). FIX: decision 12 now pins that ALL six output-type names (`Ctx`/`Vec2`/`Vec3`/
  `Vec4`/`Array`/`Text`) MUST be top-level exec-globals keys unconditionally (not a per-uniform subset),
  else a stub annotating `-> Array` NameErrors → permanent compile-freeze.
- **[seam target] the three export callers converge on `Node.render_media`, NOT a session funnel** (both
  reviewers verified independently; `Node` has no `node_id` back-ref, `render_to` is a free fn on a bare
  `Node`). FIX: decision 11 now locks a NEW `ProjectSession.render_node(node_id, ...)` funnel that all
  three callers reroute through (the bracket has a real home; `core.py`/`Node` stay engine-free).
- **[precision] `fresh_behaviors_for` recompiles from the live registry's CACHED source**, not a fresh
  disk read (no mid-edit half-saved read during export) — pinned in decision 11.
- **[dogfood] the pinned `sin(ctx.t)` body shifts the probe t-values** the existing check hardcodes — the
  dogfood row now says migrate probe values + comments with the body.
- **[renumber collateral] the 040 spec's in-body `feature 041`(=editor) refs** must retarget to 042 — the
  040-spec Files-touched row now says so.
- Confirmed NOT-a-bug: the live `ui.py` tick + the export bracket do NOT double-tick the export (separate
  instances, single-threaded deferred render after the live tick) — the 040 double-tick is genuinely
  resolved, not reintroduced.
- **CONVERGED — design sound, ready for implementation.** Both reviewers' residuals are folded; no open
  redesign. (The verification reviewer's PASS + the correctness reviewer's findings-all-accepted, with
  no new failure CLASS, is the convergence stopping condition per `/review-agent-loop`.)

**Post-impl review round 1 (2026-06-13, 3 adversarial agents — code-correctness / architecture-conventions
/ spec-fidelity, run as a convergence loop).** All three independently converged on the same BLOCKER plus
distinct real findings; ALL ACCEPTED + fixed (no false positives):
- **[BLOCKER — all 3] live-app export paths not bracketed.** The `exporting()` bracket was wired only into
  the copilot render path; `tabs/render.py` (the Render button) + `tabs/share_state.py::render_to` (Share/
  sticker export) called `Node.render_media` bare — so a maintainer rendering a STATEFUL script from the
  GUI got the live-warmed instance, not a fresh one (the headline guarantee broken on the primary user
  path). Exactly the miss-a-caller risk the spec's `render_node` funnel existed to prevent; the
  context-manager pivot reintroduced it. FIXED: `tabs/render.py::_run_render` wraps `with
  app.session.exporting(node_id)`; `tabs/share.py::_render` gained an `app` param + wraps `with
  app.session.exporting(current_node.id)`. All three export paths now bracket.
- **[MAJOR — code] copilot passed the UNRESOLVED handle to `exporting`** (`node`, a short id / `""`) →
  silent no-op for "render current". FIXED: `render_image`/`render_video` pass the resolved `ui_node.id`
  (publish already used the resolved `node_id`).
- **[MAJOR — code] `_instantiate` didn't clear `self._error` on success** → a `reset()` recovering a
  once-failing `__init__` stayed frozen forever. FIXED: clear on the success path + regression test.
- **[MINOR — code] curated builtins omitted exception types** → a user `raise ValueError` became a
  misleading `NameError`. FIXED: common exceptions seeded + regression test.
- **[MAJOR — arch/spec] dev_flow Module-map `scripting/` + `uniform_coerce.py` bullets stale** (described
  the 040 `out.set`/`UniformOut` contract). FIXED to the 041 class model + `outputs.py` + the bracket.
- **[MINOR — doc] `errors.py` docstring "041's UI" → "042's UI"** (post-renumber). FIXED.
- All else CLEAN per the reviewers: convention compliance (no new suppression, full annotations, 025
  headless boundary, the `exporting` injection follows the `ShaderLibFileManager` idiom + correctly NOT
  on the `CopilotCapabilities` Protocol), DRY (single normalization/stub/math homes), exec-globals
  annotation coverage, the resolver, freeze/last-good + KeyError-safety, and the engine export-isolation
  seam (`fresh_behaviors_for` recompiles from cached source). NOTE the decision-11 mechanism landed as a
  `session.exporting()` context manager rather than the spec's `render_node` funnel — same invariant (all
  export callers bracketed), simpler seam (no funnel threaded through the `render_to` free fn).

**Post-impl review round 2 (2026-06-13, the same 3 agents, against the patched state).** Code-correctness:
**PASS** — all four round-1 fixes confirmed genuinely correct (the GUI brackets pass the FULL node id;
`ui_nodes` is keyed by full id; the `finally` restores the hook; `_instantiate` clears the error + the
engine pops the stale `errors` entry on the next successful tick; the exception set is adequate), and NO
new bug introduced (the `_render` `app`-param has one caller; the render-defer interaction is
single-threaded; live `node.render()` doesn't fire `on_pre_render` so no double-tick). Spec-fidelity:
**PASS** on code — the call-site audit confirms EVERY `render_media`/`render_to`/`render_for` site is
bracketed (Render tab, Share tab transitively via `render_to`, all 3 copilot paths, dogfood `export_at`);
decisions 3/4/5/8/11/12 all hold post-patch. Two doc-only residuals, both fixed in this wave: decision
11's body rewritten to the landed `exporting()` seam (was describing the abandoned `render_node` funnel);
the stale "26 CPU" count corrected to 28. **CONVERGED — no code blocker, no new failure class.**

**Post-impl structural follow-up (2026-06-13, maintainer-directed: "don't leave the crap in the code").**
The one accepted residual from round 2 — the per-call-site `with session.exporting(...)` bracket (a new
export caller could forget it) — was ELIMINATED by making isolation STRUCTURAL: `Node.export_isolation`
is now an injected factory `Node.render_media` enters around its whole body (decision 11, rewritten).
Removed: `ProjectSession.exporting`, the `CopilotBackend.exporting` constructor dep + its 3 `with` wraps,
and the `tabs/render.py` + `tabs/share.py` wraps (share.py's added `app` param reverted). The bracket now
fires once per export from the single funnel every caller already passes through — bypass-proof, no
per-caller discipline. New GL test `test_render_media_auto_enters_export_isolation` pins the guarantee
(a warmed live integrator + a real `render_media` call → the export entered the factory + ticked a fresh
instance). All green: `make check`, 28 CPU + 4 GL, smoke, dogfood interleave.

**Review-swarm wave (2026-06-13, maintainer-directed "ultracode" multi-agent audit of the whole scripting
feature).** A 6-lens swarm (bugs / simplify / DRY / perf / UX / gaps), each finding adversarially verified,
5 rounds to dry, 101 confirmed findings → a synthesis that separated real bugs from a large subjective/stale
tail. Acted on the do-now tier (the rest declined as taste / triggered-deferral / stale — recorded in the
synthesis). Fixes landed:
- **[bug, high] integer uniforms never reached the GPU.** `coerce_uniform_value` was gl_type-blind: a float
  returned for an `int`/`uint` scalar, `ivecN`/`uvecN`, or `int[N]` passed coercion, then moderngl raised
  on the write, `Node.render` swallowed it at DEBUG + popped the value EVERY frame — silently, no
  `ScriptError`. Shader math naturally yields floats, so it was pervasive. Fixed at the shared root
  (`uniform_coerce`): all integer GL types round to int (mirroring the existing text-uint branch); also
  fixes the copilot `set_uniform` path. `stub_for` now emits `-> int`/`return 0` for a scalar int/uint. GL
  test asserts RETENTION (value stays in `uniform_values`, not popped).
- **[bug, high] an unguarded `read_text()` in `reload` could crash the frame loop** (a half-saved / non-UTF8
  file mid-edit — `UnicodeDecodeError` is a `ValueError`). Guarded `(OSError, ValueError) → continue`.
- **[bug, medium] `_instantiate` didn't clear a stale error on success** (already fixed in the prior wave);
  **`drop_node` was defined but never called** (node delete leaked behaviors + stale errors, and
  trash-recover reattached stale state) → wired into `_delete_node_unguarded`; **post-load nodes
  (copilot-create / template / revert) never got their export hooks** → factored `_wire_node_hooks`, called
  from both `_resolve_scripts` and `reload_scripts` (idempotent, covers every insertion path); **the export
  tick wrote into the shared `errors` dict** (could clear/fabricate a live binding's error) → `tick_behaviors`
  now passes a throwaway errors sink (mirrors the `last_good` isolation).
- **[gap, high] the engine's own `super().__init__()` idiom NameError'd** (super + containers missing from the
  curated namespace) → added super/object/list/dict/tuple/set/sum/sorted/zip/map/filter/isinstance/all/any/
  print + `mix` (a `lerp` alias). **runtime errors recorded `line=-1`** though the traceback carries the
  `<u:name>` frame → `_user_error_line` walks the traceback for the deepest user frame (runtime + raising-init).
  **`def update(ctx)` (forgot self) threw a cryptic per-tick TypeError** → arity-validated at compile.
- **[ux/dry/doc] a script-driven uniform error said "does not match u_color" not "vec3"** → `gl_type_label`
  derives a real GLSL label; the shape hint now also names a bool return; the dead `name` param dropped;
  `is_text_array` factored (the verbatim-triplicated test); `EngineContext` is `frozen=True` (the docstring
  said read-only); `reset_script` syncs the engine's recorded error; a u_time/u_aspect/u_resolution script is
  rejected (engine-driven set passed to the engine); orphan warnings log once (transition) with a false-orphan
  gate during program invalidation, and an inactive uniform's binding is reclaimed.
- **NOT a bug (refuted):** the synthesis flagged `uint[N]` codepoint-list padding as silent corruption — but
  `uint[N]` IS the text glyph BUFFER, where partial-list padding is the correct, documented contract (matches
  the `str` path). Kept padding for the text kind; only non-text numeric arrays are exact-length.
- Added 18 regression/coverage tests (int coercion + retention, vecN[M] chunking, super, arity, non-UTF8
  reload, engine-driven reject, inactive reclaim + false-orphan, drop_node, export-error isolation, frozen
  ctx, error-line recovery, post-load-hook + drop_node in the dogfood). All green: `make check`, 46 CPU + 5
  GL + 17 copilot + smoke + dogfood.

**Convergence re-swarm (2026-06-13).** A second swarm (verify-fixes / regression / fresh-sweep lenses,
4 rounds to dry) — 101 findings collapsed to 10, verdict "functionally sound, regression-free; a hygiene
pass not a correctness rescue." It caught a MIS-FRAMED first-pass finding (a "make check fails" headline
was FALSE — the gate passes; it's a ruff version-pin skew, `uv` ruff 0.11.4 flags `RUF046` that the
pinned pre-commit 0.3.4 doesn't) and correctly de-rated most to wontfix/polish. Acted on:
- **RUF046 (M1):** dropped the redundant `int(round())` (single-arg `round()` already returns `int`) in
  `uniform_coerce` — now `uv` ruff + the gate agree. (The pin-bump itself is the `todo.md`-tracked
  maintainer call.)
- **S1:** `_tick_behaviors` early-outs `if not behaviors` — a scriptless node no longer builds the
  active-uniform dict every frame (verified: 0 `get_active_uniforms` calls on the tick path).
- **S2:** removed the dead `_make_live_pre_render` closure (the live path ticks via `session.tick`, NOT
  `on_pre_render` — every `on_pre_render` call site runs inside the export-isolation swap, so the live
  hook was never the active value) + its self-contradicting docstring; `_wire_node_hooks` now wires the
  isolation factory ONCE on first sight (sentinel-gated on `export_isolation is not nullcontext`), ending
  the per-frame closure re-allocation in the reload poll.
- F2 stub docstring gained a one-line "math is pre-loaded, no import" note; the warn-once dedup + the
  scriptless-tick early-out got regression tests.
- **Declined** (recorded as polish/triggered-deferral): the `import math` → cryptic-ImportError message
  (a deferred stub-content concern), the arity message wording for the extra-param case, NaN/inf int
  freeze, the `warned`-set file-deletion leak (node-scoped, session-bounded). None are correctness bugs.
- **CONVERGED.** `make check` clean (`uv` ruff now agrees), 47 CPU + 5 GL + smoke + dogfood green.

## Plan-lock resolution log (2026-06-13)

Maintainer answers at scoping, folded into the decisions:

1. **Script value → a typed RETURN from a stateful CLASS, not `out.set` into an injected sink.** The
   user finalizes a `ScriptBehavior` subclass with `__init__` (state) + `update(self, ctx)` (returns the
   typed value). State is the whole point of CPU scripting — stateless work belongs in the shader.
   Decisions 1, 2.
2. **The user is given a type-aware STUB** (adjusted to the uniform's type), not a blank file. The base +
   output types + `ctx` are implicitly in scope. Decisions 1, 7.
3. **State lifecycle → fresh on (re)load + fresh per export + a manual RESET affordance.** The manual
   reset is engine-method in 041, the UI button is 042. Decision 4.
4. **Sequencing → NEW 041 = the engine redesign (headless); 042 = the UI + `u_mouse`.** Downstream
   renumbers (old 042 state-scripts is ABSORBED — state is now core; old 043 copilot write-behavior
   stays as 043). 040 is superseded-in-contract, not deleted.
5. **Output types → only the shaped/capped GLSL kinds get a type (`Vec2/3/4`, `Array`, `Text`); scalars
   return bare** (`return 0.5`). Reasoned from user convenience. Decision 3.
6. **`ctx` → clock only (`t/dt/frame`) in v1**; `mouse` is 042; no `ctx.state`/`ctx.uniforms` (instance
   state replaces them). Decision 5.

Low-level naming/shape (the exact `Vec*`/`Array`/`Text` API, the `ScriptBehavior` base internals, the
stub wording, the subclass-resolver tie-breaks) is delegated to implementation.

## Open questions for the user

1. **The base class + output type names.** Proposed: `ScriptBehavior` (base), `Ctx` (the context alias
   the user annotates with), `Vec2`/`Vec3`/`Vec4`, `Array`, `Text`. OK, or do you want shorter (e.g.
   `Behavior` as the base name itself — but then the user's class can't also be `Behavior`; that's why
   the base is `ScriptBehavior` and the user's subclass is `Behavior`)? Flag any rename.
2. **Stub default body.** Proposed stub for a `vec2`:
   ```python
   class Behavior(ScriptBehavior):
       """Drive u_offset (vec2) each frame.
       ctx.t  elapsed seconds | ctx.dt  delta seconds | ctx.frame  frame index
       Return Vec2(x, y). Keep state on self (persists across frames).
       """
       def __init__(self) -> None:
           pass

       def update(self, ctx: Ctx) -> Vec2:
           return Vec2(0.0, 0.0)
   ```
   Good, or barer (drop the empty `__init__`, shorter docstring)?
3. **Manual `reset()` in 041 even though the UI button is 042?** It's a tiny engine method + a test; I'd
   include it in 041 so 042 only wires the button. Object if you'd rather it land with the UI.
