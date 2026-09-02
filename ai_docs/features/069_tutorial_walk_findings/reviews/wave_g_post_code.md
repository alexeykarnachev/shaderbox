# Wave G post-implementation code review — `928c231`

Role: code correctness (dev_flow step 6). Bugs, races, GL lifecycle, error handling,
the engine's routing under every input shape, persistence.

## Verdict

| Area | Verdict |
|---|---|
| Routing (`_tick_script` / `_active_by_pass`) | **FAIL** — finding 1 |
| `(pass, name)` keys end to end | **PASS** |
| Persistence (`StoppedKey`, salvage, `_load_ui_state`) | **PASS** |
| Mouse (`down` / prev, latch, `EXPORT_MOUSE`) | **PASS** |
| `RESET_FEEDBACK` / `has_feedback` | **PASS** |
| Shader-tab error strip | **PASS** |
| GL (`script_ready`, the GPU test's absolute pixel) | **PASS** |
| Tests | **PASS** — four falsifiers re-run red, restored, tree clean |
| Conventions | **PASS** |

`make gates` GREEN, judged by the exit code captured unpiped:
`MESA_GL_VERSION_OVERRIDE=4.6 MESA_GLSL_VERSION_OVERRIDE=460 xvfb-run -a make gates > g.log 2>&1`
→ `EXIT=0`, and the log's last lines read `test passed` / `smoke passed`, not skipped.

---

## Findings

### 1. A BROADCAST key is not held for a never-compiled pass — it errors, and the copilot reads the error as fact

The commit's own rule, stated three times (the module docstring, `_active_by_pass`'s
comment, `Pass.script_ready`'s comment, and the commit message): a pass that has never
ATTEMPTED a compile is HELD for the tick — no error, nothing written — "because
erroring would spray one row per pass on every first frame".

The pass-block phase implements it. The broadcast phase does not. `_active_by_pass`
OMITS a not-ready pass from the mapping, so in the broadcast loop:

```python
targets = [
    (pass_name, active)
    for pass_name, active in active_by_pass.items()
    if self._binding_reject(pass_name, name, active) is None
]
if not targets:
    errors[(document_id, "", name)] = ScriptError(
        name, "runtime", f"no pass declares '{name}' (orphan key)"
    )
```

"absent because it has not compiled yet" and "absent because no pass declares it" are
the same condition to this branch. The block loop distinguishes them
(`active = active_by_pass.get(pass_name); if active is None: continue`); the broadcast
loop cannot, because it never asks about a named pass.

**Evidence — real `Pass` objects, the app's own frame order (tick then render,
`ui.py::_tick_frame_state`), a two-pass document, the script `return {'u_wave': ctx.t}`:**

```
script_ready before any render: {'seed': False, 'out': False}
frame0 errors: {('n', '', 'u_wave'): "no pass declares 'u_wave' (orphan key)"}
frame0 driven: set()
frame0 soft_errors (STRIP ROWS): [('', 'u_wave', "no pass declares 'u_wave' (orphan key)")]
frame1 errors: {}
frame1 driven: {('seed', 'u_wave'), ('out', 'u_wave')}
frame1 soft_errors: []
```

That is precisely the "orphan error that clears a frame later" the hold exists to
prevent, reproduced on the one addressing form 069 introduced and the one the whole
design leans on ("case 1, the brush on `paint`, needs no pass block at all").

**The expensive half is `dry_run`, which never renders at all.** Same document, the
single pass compiled-but-never-drawn, three sample times:

```
COLD dry_run driven: set()
COLD dry_run orphan: [('', 'u_wave', "no pass declares 'u_wave' (orphan key)")]
COLD dry_run samples: [(0.0, {}), (0.5, {}), (1.0, {})]
```

`write_script_source` reloads the script and calls `dry_run` with no compile in
between, so a `write_script` on a document whose passes have not rendered yet hands
the agent three false facts at once: `driven: []` (the deliberate "loud no-op fact"),
an orphan error naming a uniform the shader does in fact declare, and
`values UNCHANGED across t (STATIC)` from empty samples. The agent then debugs a
script that is correct. 066 D1's no-compile-in-the-frame-loop constraint is not even
in force on this path — `dry_run` is a synchronous copilot call on the main thread,
the same context in which `_scriptable_uniforms_for` compiles every pass and its own
comment says that is "correct here and only here: both callers are user/agent actions".

**Fix:** give the broadcast phase the same three-way answer the block phase has.
Distinguish "no ready pass declares it" from "no pass declares it at all" by testing
the not-ready passes too: build the not-ready name set once per tick alongside
`active_by_pass`, and when `targets` is empty, HOLD (continue, no error) if any
not-ready pass exists, erroring only when every pass is ready and none declares the
key. Separately, have `dry_run` compile the document's passes before its loop (it is
already an agent-action path, not the frame loop) so a cold probe reports the real
driven set instead of an orphan.

### 2. A bare key on a pass whose compile FAILED reports "no pass declares it", naming neither the pass nor the compile failure

Same branch, different input. When the only pass declaring a uniform is
ready-but-empty (its compile was attempted and failed), the broadcast key produces
`no pass declares 'u_wave' (orphan key)` — permanently, for the life of the broken
source. The block path for the same situation says
`pass 'main' has no active uniform 'u_wave' (orphan key)`, which at least names the
pass whose shader tab carries the compile error.

**Evidence** — one pass with a deliberately broken shader, three frames:

```
frame0 ready=True soft=[('', 'u_wave', "no pass declares 'u_wave' (orphan key)")]
frame1 ready=True soft=[('', 'u_wave', "no pass declares 'u_wave' (orphan key)")]
frame2 ready=True soft=[('', 'u_wave', "no pass declares 'u_wave' (orphan key)")]
```

The row is also pass-free, so by the strip's own rule (`_script_errors_for_pass`
filters on `err_pass == pass_name`) it appears on the script tab and on NO shader
tab — the one tab that carries the actual cause. The user reads "no pass declares it"
on the script while the shader tab beside it says the shader will not compile, and
nothing connects the two.

**Fix:** when the broadcast finds no target but at least one ready pass has a compile
error, word the message against that (`no pass declares 'u_wave' — <pass> does not
compile`), or emit the row under that pass's key so it reaches its shader tab. The
fix for finding 1 naturally has the not-ready/failed distinction in hand.

---

## Non-findings, verified rather than assumed

**The routing matrix**, executed against the engine with a two-pass fake
(`paint` declares `u_a` + `u_b`, `composite` declares `u_a`), 14 cases. Every case
behaves as the design specifies, and nothing raises:

| case | written | error key | driven |
|---|---|---|---|
| bare, both declare | both passes | none | both pairs |
| bare, one declares | `paint` only | none | `(paint, u_b)` |
| bare, none declares | nothing | `("", u_z)` | none |
| block, declared | that pass only | none | that pair |
| block, undeclared there | nothing | `(composite, u_b)` | none |
| block, non-existent pass | nothing | `("", nope)`, lists real passes | none |
| bare + block on a pass that does NOT declare it | broadcast reaches `paint`, block errors on `composite` | `(composite, u_b)` | `(paint, u_b)` |
| block wins over bare on that pass | `paint=2.0`, `composite=1.0` | none | both |
| double nesting (dict under a block) | frozen at last-good | `(paint, u_a)`, "PASS BLOCK" | that pair (correctly: driven precedes coerce) |
| sampler in a block | nothing | `(paint, u_s)`, names the pass | none |
| sampler bare | nothing | `("", u_s)` | none |
| empty document | nothing | `("", u_a)` | none |

The broadcast-plus-block case is the one the brief flagged, and it is right: the
broadcast still reaches the passes that declare the uniform while the block's own pass
records its error. `driven` and `last_skipped` stay disjoint, so
`script_driven_uniforms` never claims a bad key.

**`(pass, name)` keys end to end.** Grepped every consumer of `stopped_uniforms`,
`is_uniform_stopped`, `set_uniform_stopped`, `set_document_all_stopped`,
`uniform_is_driven`, `get_script_driven_uniforms`, `script_driven_uniforms`,
`last_driven`, `last_skipped`, `all_stopped` across `shaderbox/`, `scripts/` and
`tests/`. Zero name-keyed survivors. `_drop_script` unpacks the pair into the
three-tuple as claimed. All four `copilot/backend.py` sites use `_driven_on`, and the
`set_uniform` gate asks the OUTPUT pair below the `target =` bind, which is correct
because `set_uniform` writes into `target.render_pass.uniform_values` and nothing else.

**Persistence.** `StoppedKey` is hashable and rejects mutation (`ValidationError`).
The `frozen=True` class-arg rationale is real, not folklore — pyright on a two-class
probe: `model_config = {"frozen": True}` gives `error: Set entry must be hashable /
Type "B" is not hashable (reportUnhashable)`; the class-arg form is clean. Salvage
executed: a stale `["u_x"]` drops to `[]` with `all_stopped: True` intact; a scalar
and a mixed list both drop to `[]`; the round trip dumps `[{"pass_name": …, "name": …}]`
and json-serializes. `_load_ui_state` is a verbatim extraction — diffed the pre- and
post-commit bodies line for line, the only change is `ui_state = ...` becoming
`return ...` and the two lines the loader kept. The roster in
`test_persistence_completeness.py` still lists `ui_models.py`, which is the module
that changed; no new store was added.

**Mouse.** `down` and `prev_*` are set inside the hit-test only when
`hit is not None and hit[2]`, and both `down` and `script_mouse_inside` are cleared
BEFORE either branch, so the empty-state branch (a closed or switched document) runs
no hit test and cannot leave `down` latched — and an exception later in
`_draw_app_panel` (which `ui.py` catches) cannot either, because the clear is the
first thing `_draw_document_image` does. `item_normalized_mouse` ANDs
`is_window_hovered(child_windows)`, so an open modal drops `inside` and hence `down`.
Re-entry: `prev` chains from `previous_mouse` only when `was_inside`, else restarts at
the current sample. `EXPORT_MOUSE = MouseState(0.5, 0.5, False, 0.5, 0.5)` —
`down=False`, prev == current. The headless probe path is the `EngineContext` default
(`field(default_factory=lambda: EXPORT_MOUSE)`), which `dry_run`'s loop uses since it
constructs `EngineContext(t=…, dt=…, frame=…)` with no `mouse`.

**`RESET_FEEDBACK`.** `CommandId.RESET_FEEDBACK` → `_chord(K.f6)` with `C.DOCUMENT`,
which is the CATEGORY (cheatsheet grouping); the scope defaults to `GLOBAL`, so it
fires under editor focus as intended, and `chord_needs_modifier` exempts F1-F12. Wired
in `App._command_handlers` to `reset_current_document_feedback`, which no-ops on a
missing document. The `Clear` ghost button sits inside
`if app.current_document_id in app.ui_documents:` and is further gated on
`document.has_feedback`, which reads `plan_passes(self.graph)[0].feedback` —
`plan_passes` is pure, non-raising, and already called per frame in
`widgets/pass_list.py`, so the per-frame cost is the established pattern.
Mid-iteration semantics executed on a real feedback document: after 10 accumulating
frames the pixel reads 130; `begin_frame(10)` then `reset_feedback()` then
`render()` (the app's actual order — dispatch runs after `_tick_frame_state`) gives 13,
one step from black, and the next frame 26. No double-swap, no stale history, and
`has_feedback` stays True after the clear. The export path's `reset_feedback` inside
`render_media`'s `export_isolation` bracket is untouched.

**Shader-tab error strip.** No script line number can leak into a shader's markers:
`_apply_markers` builds its fingerprint from `err.line >= 0 and err.path ==
current_path`, and `_script_errors_for_pass` emits rows carrying
`app.session.script_path_for(...)`, never `tab.path` — so on a shader tab those rows
are filtered out of the marker set by construction, not by an ordering that could
change. The click branch calls `open_script_for` before latching
`editor_jump_request`, and `_consume_jump` (called at the top of the draw) runs a
frame later against the now-current script tab, so the jump lands. Compile errors
sort first, which matters because `_MAX_ERROR_ROWS` caps the unexpanded strip.

**GL.** `script_ready` is a pure property over `self.program is not None or
bool(self.compile_unit.error_raw)` and compiles nothing — it is exactly the negation
of `get_active_uniforms`'s `if self.program is None and not self.compile_unit.error_raw`
guard. The GPU test's absolute-pixel assertion cannot pass on an unrendered canvas: an
unrendered `seed` reads 0, and both expectations (64 at t=0.25, 255 at t=1.0) fail
`abs(0 - expected) <= 2`. Falsified in both directions — routing the broadcast to the
LAST target (the true pre-069 output-only behaviour) produces
`the broadcast did not reach the NON-output pass at t=0.25 (read 0, expected ~64)`,
so the pixel half does the work the comment claims; routing to the FIRST target is
caught by the `uniform_values` half instead. Both mutations restored, tree verified
clean.

**Tests — four falsifiers re-run, each red, each restored with `git diff --quiet`
before anything else ran:**

1. *Routing matrix, broadcast branch dropped* (`targets[:1]`) — 2 failed:
   `test_the_routing_table[{'u_a': 1.0}…]` and
   `test_a_pass_block_beats_a_broadcast_on_that_pass` (`KeyError: 'u_a'` on the
   sibling pass).
2. *`coerce_one`'s dict rejection deleted* — `test_coerce_one_rejects_a_dict` red;
   the fallback message is `value does not match float — provide a number`, so the
   guard hardens the DIAGNOSTIC rather than the safety (`normalize_output` still
   rejects a dict). Worth knowing: the invariant is double-covered, so this test
   pins the message, not the dispatch.
3. *Salvage under the forbidden compat step* — added
   `{"pass_name": "main", "name": v} for a str v` before `drop_invalid`;
   `test_a_stale_string_stopped_set_drops_to_empty` red with
   `Left contains one more item: StoppedKey(pass_name='main', name='u_x')`. The
   no-migration rule is mechanically enforced.
4. *Orphan console line restored* (`logger.warning` + the loguru import) —
   `test_the_orphan_warning_no_longer_reaches_the_console` red with the captured
   sink line. The real-sink point is confirmed: the record carries
   `shaderbox.scripting.engine:_tick_script`, which stdlib `caplog` would not see.

Dry-run isolation independently executed: after a live tick warms the caches, a
`dry_run` leaves `uniform_values` byte-identical, `last_driven` / `last_skipped` /
`errors` unchanged, and reports the same pairs the live tick did.

**Conventions.** No new `noqa` / `type: ignore` / `pyright: ignore` / inline import /
`if TYPE_CHECKING` / `@staticmethod` in any changed file — the `# noqa: E402` block in
`scripts/dogfood/harness.py` is the pre-existing sanctioned env-setup pattern, and the
`@classmethod`s are alternate constructors. Every new `Any` sits on a heterogeneous
uniform value, the established shape. No history-narrating comments: the new comments
state what the code IS (the two-phase ordering, why `_drop_script` unpacks, why
`has_feedback` reads the plan). `scripting/keys.py` imports only pydantic, so
`ui_models` importing it opens no cycle and `api_doc`'s GL-free invariant is untouched
(that test asserts over `api_doc`'s own import list, which does not name `keys`). Docs
updated in the same commit: the conventions scripting entry, the `dev_flow` module map,
065 D12's supersession and 068 D7's lifted retraction.

---

## False trails

- `pass_name_of(panel_pass.source.path)` disagreeing with the engine's pass key: it
  cannot in the live app. `Document.load_from_dir` is the only construction site, and
  it keys `document.passes` by `pass_name_of(shader_path)` from the same glob. The bare
  `Document()` mismatch (key `"main"`, path `default.frag.glsl` → `"default"`) exists
  only inside the loader before the passes are replaced, and in tests.
- `play_stop_toggle(f"u_{pass_name}_{name}")` colliding across passes: the panel shows
  one pass at a time (`App.panel_pass`), so two colliding ids are never submitted in
  one frame.
- The `# Clickable toggle (F6)` comment in `tabs/code.py` next to the new F6 binding:
  that F6 is feature 047's finding id, not a key.
- `_write_one` writing `None` into `uniform_values` when a first-tick coercion fails
  with no last-good: pre-existing (`git show 928c231^` has the same fallback chain),
  and unreachable on a real `Pass`, whose `get_active_uniforms` seeds `uniform_values`
  before any consumer reads them.
- `ScriptStatus.driven_count` becoming a count of PAIRS (a broadcast to 3 passes counts
  3): it reaches no UI and no prompt — tests are its only readers.
- `dry_run` stashing `last_driven` but not `last_skipped`: executed, and the live
  `last_skipped` and its errors survive the probe unchanged, so no soft error is
  stranded.
- `e.error.pass_name = pass_name` mutating a shared `ScriptError` across a broadcast's
  targets: `coerce_one` raises a fresh `_RuntimeScriptError` with a fresh `ScriptError`
  per call, so each target's error is its own object.

---

## Coverage

Read end to end: `scripting/engine.py`, `scripting/behavior.py`, `scripting/keys.py`,
`scripting/errors.py`, `scripting/context.py`, `scripting/__init__.py`,
`scripting/api_doc.py`, and the full diff of the other 31 changed files, plus
`core.py::Pass.script_ready` / `get_active_uniforms`, `document.py::has_feedback` /
`reset_feedback` / `begin_frame` / `render_media`, `pass_graph.py::plan_passes`,
`ui.py::_draw_document_image` / `_draw_app_panel` / `_tick_frame_state`,
`tabs/code.py`'s two adapters + `_apply_markers` + `_consume_jump`,
`project_session.py`'s script cluster, `copilot/backend.py`'s five re-keyed sites, and
every changed test file. Not exercised: the interactive drag itself (no synthetic input
on this box) — the mouse verdict rests on reading the branch structure plus the
`item_normalized_mouse` contract, not on a driven cursor.

`git status --short` is empty apart from this review file; every probe file lives in
the scratchpad, and each of the six mutations (four falsifiers plus two GPU-test
direction checks) was restored and verified with `git diff --quiet` before the next
command ran.

---

# Round 2 (closure) — `a873ace`

Narrow closure round on the two findings round 1 returned FAIL for, plus the routing
matrix and the GPU-test `finish()`. W-D is being implemented concurrently, so every
probe ran against a tree materialized from the commit
(`git archive a873ace | tar -x`), verified by hash against
`git show a873ace:shaderbox/scripting/engine.py`
(`9623859806e0f6…`) and by each probe printing the engine `__file__` it actually
imported. Nothing was read from the dirty working tree.

## Verdicts

| Item | Verdict |
|---|---|
| Finding 1 — broadcast held for an uncompiled pass | **CLOSED** |
| Finding 1 — cold `dry_run` | **CLOSED** |
| Finding 2 — compile-failed pass attribution | **CLOSED** |
| 14-case routing matrix | **2 rows changed, both intended; 12 byte-identical** |
| `finish()` for the GPU-test flake | **CORRECT — diagnosis and primitive both confirmed by measurement** |

**Overall: PASS.**

## Finding 1a — frame-0 broadcast. CLOSED

Same reproduction as round 1: two passes, neither compiled, the app's own order
(tick then render), script `return {'u_wave': ctx.t}`.

```
engine from: …/scratchpad/a873ace/shaderbox/scripting/engine.py
script_ready before any render: {'seed': False, 'out': False}
frame0 errors: {}
frame0 driven: set()
frame0 soft_errors (STRIP ROWS): []
frame1 errors: {}
frame1 driven: {('out', 'u_wave'), ('seed', 'u_wave')}
frame1 soft_errors: []
```

Frame 0 is silent — no error, no strip row, nothing written — and frame 1 drives both
passes. Round 1 had `frame0 errors: {('n','','u_wave'): "no pass declares 'u_wave'"}`
with a strip row that cleared a frame later.

The hold is gated: mutating `if not_ready:` to `if False:` turns
`test_a_broadcast_is_held_for_a_not_yet_compiled_pass` red. Restored, verified against
`git show a873ace:…` with `diff -q`.

## Finding 1b — cold `dry_run`. CLOSED

Single never-rendered pass, three sample times:

```
script_ready before probe: {'main': False}
COLD dry_run driven:  {('main', 'u_wave')}
COLD dry_run orphan:  []
COLD dry_run per_key: []
COLD dry_run samples: [(0.0, {('main','u_wave'): 0.0}),
                       (0.5, {('main','u_wave'): 0.5}),
                       (1.0, {('main','u_wave'): 1.0})]
```

All three false facts are gone: the driven set is real, no orphan names a uniform the
shader declares, and the values move across t, so the agent reads ANIMATING rather than
STATIC. Round 1 returned `driven: set()`, an orphan row, and three empty sample dicts.

The pre-compile does not cost the probe its isolation. Warming the document the way the
live app does (two tick/render pairs) and then probing:

```
probe driven: {('main', 'u_wave')}
uniform_values UNCHANGED by probe: True  {'main': {'u_wave': 0.7}} -> {'main': {'u_wave': 0.7}}
last_skipped UNCHANGED: True
errors UNCHANGED: True
```

The byte-identical invariant holds. On a COLD document the pre-compile does seed
`uniform_values` from `{}` to `{'u_wave': 0.0}`, but that is `get_active_uniforms`'s
documented seeding (`core.py`: "Seeding rides the compile so every consumer keeps the
invariant that a returned uniform has a value"), not a probe write — the same thing the
first render would have done a moment later.

## Finding 2 — the compile-failed pass is named. CLOSED

Sole pass with a deliberately broken shader, three frames:

```
frame0 ready=True  soft=[]
frame1 ready=True  soft=[('main','u_wave',"no pass declares 'u_wave' (orphan key) — pass 'main' does not compile")]
frame2 ready=True  soft=[('main','u_wave',"no pass declares 'u_wave' (orphan key) — pass 'main' does not compile")]
error keys: [('n', 'main', 'u_wave')]
```

The row is keyed on the pass, so `_script_errors_for_pass`'s `err_pass == pass_name`
filter now puts it on `main`'s shader tab, beside the compile error that is the real
cause — the exact gap round 1 named. (Frame 0 is empty because the compile has not been
attempted until the first render; the hold covers it, correctly.)

The genuinely-homeless case is not over-attributed. A single pass that compiles cleanly
and simply does not declare the key:

```
soft: [('', 'u_wave', "no pass declares 'u_wave' (orphan key)")]  keys: [('n','','u_wave')]
```

still pass-free, as it must be. And with a broken pass beside a good one that does not
declare the key, once the broken pass is on the output chain and has attempted its
compile:

```
frame1..3 soft=[('zzz_broken','u_wave',"no pass declares 'u_wave' (orphan key) — pass 'zzz_broken' does not compile")]
```

Gated: replacing `broken = self._broken_pass_for(document)` with `broken = None` turns
`test_a_broadcast_names_the_broken_pass_that_would_have_declared_it` red. Restored and
diffed clean.

**`_broken_pass_for` does not reintroduce the 066 D1 hazard.** It calls
`get_active_uniforms()`, but behind `render_pass.script_ready and …`, which
short-circuits, and it is only reached after `if not_ready:` has already returned. Pinned
with a stub that raises if the engine ever pulls uniforms from a not-ready pass:

```
not-ready pass get_active_uniforms calls: 0 (must be 0)
errors: {} (must be empty: held)
after both ready, errors: {('n','broken','u_wave'): "… — pass 'broken' does not compile"}
```

Zero calls across three ticks; the attribution fires only once every pass is ready.

## Routing matrix — 2 of 14 rows changed, both intended

Re-ran the round-1 matrix unchanged against both trees (each materialized from its own
commit, each printing the engine file it loaded) and diffed. Only two rows differ, and
they are the two the fix targets:

- **case 9** (`{'paint': {'u_a': 1.0}, 'u_b': 5.0}`, `paint` not ready): the bare `u_b`
  was `("", 'u_b')` "no pass declares" with a strip row; now `errors = {}`,
  `last_skipped = set()`, held.
- **case 14** (every pass not ready, bare key): same change.

The other twelve are byte-identical, including the broadcast-plus-block interaction
(broadcast still reaches `paint` while `composite`'s block records its own error), the
double-nesting rejection, both sampler cases, the unknown-pass listing, and the
empty-document case. **case 10** (pass block on a compile-FAILED pass) is deliberately
unchanged at `pass 'paint' has no active uniform 'u_a'` — the block phase already named
the pass, so `_broken_pass_for`, which fires only on the broadcast branch, correctly does
not touch it.

## The GPU-test flake — `finish()` judged, not taken on faith

The commit's causal claim is that `texture.read()` raced the GPU and llvmpipe returned
the previous frame's mapping, and that the fingerprint is reading *exactly* the other
sample's value. Both halves reproduce. Rendering the same two samples 200 times with no
barrier, on the same f2 targets the fixed test uses:

```
seed canvas dtype: f2
  MISMATCH it=56 t=0.25 read=255 expected~64
  MISMATCH it=57 t=0.25 read=255 expected~64
  MISMATCH it=58 t=0.25 read=255 expected~64
no-finish mismatches over 400 reads:   22
with-finish mismatches over 400 reads:  0
```

22 of 400 reads wrong, every one returning precisely the *other* sample's value, never
an intermediate or arbitrary number — a stale mapping, exactly as claimed, and not
something a routing error could produce. `finish()` takes it to 0. The diagnosis is
measured, not guessed. (My reproduction fails in the mirror direction from the one the
commit describes — t=0.25 reading the t=1.0 value rather than the reverse — which is the
same mechanism reading whichever sample ran previously.)

**`finish()` vs `glFlush` is settled by measurement, not preference.** Swapping the
barrier for `glFlush()` over the same 400 reads:

```
no-finish mismatches over 400 reads:   35
with-flush  mismatches over 400 reads: 21
```

`glFlush` leaves 21 of 400 wrong; `glFinish` leaves 0. That is the specified difference —
`glFlush` only guarantees commands are *issued*, while `glFinish` blocks until they have
*completed*, which is what a host-side read-back requires. `finish()` is the correct
primitive here, not merely the heavier one. `moderngl.Context.finish()` is a direct
`glFinish` (`self.mglo.finish()`).

The test itself is stable in situ: `tests/test_script_engine_gl.py` run five times
consecutively under xvfb gives `7 passed` every time.

## Gates

`tests/test_script_engine.py`, `test_script_dry_run.py`, `test_script_engine_gl.py`,
`test_script_error_strip.py`, `test_ui_models.py`, `test_command_routing.py` on the live
tree: **113 passed**. Ruff clean and `ruff format --check` clean; pyright reports
**0 errors** on the whole tree and on `scripting/engine.py` alone.

`make gates` on the live repo is RED at `check`, and it is **not** this commit's doing:
pyright prints `0 errors, 7 warnings`, and the hook fails on "files were modified by this
hook" from the concurrent W-D work — `ai_docs/conventions.md` and
`tests/test_ui_prose_budget.py` appeared in `git status` mid-run.
`git diff --stat` over the five files `a873ace` owns is empty, so nothing the hook
rewrote belongs to this fix. Recorded rather than reported as green, since the gate's own
rule is to judge it by the exit code; a clean re-run belongs to whoever lands W-D.

## Late-round discipline

Nothing is filed here as a preference. The two items I checked hardest for a
false-positive both came back as correct-by-measurement rather than as taste: the
`finish` / `glFlush` choice (settled by 21-vs-0 wrong reads, not by "finish is safer"),
and `_broken_pass_for`'s `get_active_uniforms()` call, which looked like a reintroduced
066 D1 violation on reading and is provably zero calls on a not-ready pass. One probe
result that looked like a defect — a broken pass beside a good one producing no error at
all — was my own harness leaving the broken pass unwired from the graph, so it was never
drawn, never attempted a compile, and was correctly held forever. That is the hold
working, not a gap.

## Probe hygiene

Every probe lives in the scratchpad. Both engine mutations (the dropped `not_ready` hold,
the disabled attribution) were applied to the scratchpad copy only, never the repo, and
each was restored and verified with `diff -q` against `git show a873ace:…` before the
next command ran. `git status --short` in the repo shows no file I created or modified
apart from this review.
