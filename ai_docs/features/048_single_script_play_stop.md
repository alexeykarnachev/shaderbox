# 048 — Single-script model + per-uniform play/stop

A maintainer-walkthrough-driven redesign of the scripting feature (supersedes the per-uniform half of
041/044/047). Driven by a voice walkthrough (foo msg 1870, 2026-06-14) after testing the 047 Script
Showcase node. The maintainer found the two-script-kinds model (per-uniform `u_*.py` scripts vs the
node-brain `script.py`) **too confusing** — you can't tell, from one `</>` glyph, whether a uniform is
driven by its own per-uniform script or by the brain. The decision: collapse to ONE script per node
(the brain), and replace the create/activate/glyph affordances with a direct **play/stop** model.

Validated by an ultracode swarm (4 lenses + devil's-advocate + synthesis, 2026-06-14): all the
maintainer's concerns trace to real code; several refinements below come from that pass (the state
model, the node-stop-freezes-writes nuance, the yellow→blue collision, the explicit-import fallback,
the tab-label-not-disk-rename scope).

Two architectural moves anchor the wave:
- **One script per node.** Delete per-uniform scripts entirely (`u_<name>__<tag>.py` + the whole 047
  F14 `(name,type)`-tag binding, the copy-content selector, the two-pass brain/per-uniform conflict
  override). A node has at most one `scripts/script.py` (the brain) returning `dict[str, value]`. Private
  helper methods inside it compute individual uniforms in any order the user wants.
- **Play/stop replaces active/inactive.** A uniform the script returns PLAYS (script drives it); the user
  STOPS it to edit by hand; STOP auto-engages when the user grabs a playing uniform's widget. A
  whole-node play/stop freezes/resumes every uniform at once. The 047 `is_script_active`/`is_brain_active`
  model flags retire; the script auto-binds (its existence is the bind), and stop-state is node-scoped.

## Goal

Make scripting legible: ONE script per node, ONE at-a-glance state per uniform (playing vs manual), a
direct play/stop control instead of create→activate→glyph indirection, a script object whose available
types are VISIBLE (explicit imports, not magic injection), node-derived tab labels so two nodes' tabs are
distinguishable, and a canonical script-object reference in the stub itself. Plus the standalone
startup-blank-editor bug fix.

## Out of scope

- **The copilot write-behavior tool (043).** Still next on the roadmap after this wave. 048 makes its
  checkpoint-capture deferral SIMPLER (one known path `scripts/script.py`, no glob) — `todo.md` updated in
  the impl wave, the tool itself unbuilt.
- **A scripts LIBRARY / cross-node reuse.** 047's copy-content selector is DELETED with per-uniform
  scripts; cross-node script reuse is gone for now. **Trigger:** first time the maintainer wants to reuse a
  whole-node script across nodes/projects — build a script picker then (the `popups/lib_picker/` pattern),
  not the flat per-uniform-type dropdown 047 had.
- **Cross-uniform / cross-node shared state.** Still 041/044's deferrals (`todo.md`) — unchanged. One brain
  per node, `self.*` state, no shared channel.
- **An in-app scripting help panel.** The canonical reference is the stub docstring (decision 7); a separate
  doc surface is disproportionate for a solo tool. **Trigger:** the stub proves insufficient for a real user.
- **A runaway-brain kill switch.** A broken/expensive brain runs every frame even while node-stopped (stop
  freezes WRITES, not ticking — decision 7). Accepted for a solo tool. The off switch is **DELETING or
  EMPTYING `script.py`** (no `Behavior` subclass → the compile early-returns a `ScriptError` → no tick runs
  at all) — NOT an empty-dict `update` return (that still ticks every frame, it just writes nothing).
  **Trigger:** a brain's per-frame cost becomes a real pain in practice.

## One-time migration (impl-commit, not deferred)

Deleting per-uniform scripts orphans every existing `nodes/<id>/scripts/u_*.py` on disk — under the new
single-`script.py` discovery they match nothing and silently drive nothing. This affects the live dev
sandbox: the 047 Script Showcase node (`projects/dev/nodes/1d4f8a20-…000047/scripts/`) carries four
(`u_pulse__float.py`, `u_swirl__float.py`, `u_wave_offset__vec2.py`, `u_tint__vec3.py`). **Never silent.**

The impl commit ships a one-time migration as a **`ProjectSession.load` post-step scoped to
`self.paths.nodes_dir` ONLY** — NEVER in `load_node_from_dir`, which also runs on the read-only shipped
templates dir (`project_session.py` loads templates through the same function) and must not be mutated.
For each node dir with `u_*.py` files: concatenate their bodies as a COMMENTED block prepended to that
node's `scripts/script.py` (creating `script.py` if absent), then `detach_script` each `u_*.py` (trashed,
recoverable, reusing `detach_script`'s existing collision-rename). A loud per-node INFO log names what
moved. **Idempotency = "no `u_*.py` left on disk"** (the migration's own teardown is the marker — no
separate flag); re-running `load` finds no `u_*.py` and no-ops. The user manually merges the commented
bodies into `update`/helpers.

**Dev-sandbox sequencing (avoid a stale double-prepend):** both dev nodes currently carry BOTH a
`script.py` AND `u_*.py` files. The example node is rebuilt by hand for the new single-brain model in the
finalization sandbox-sync wave; that rebuild must DELETE the node's `u_*.py` files BEFORE the auto-migration
runs against the sandbox, so the migration never prepends a stale commented block into the hand-authored
`script.py`. The auto-migration is then verified by a real run on any *other* node that still carries
`u_*.py`, not just unit-tested.

Dropped model fields: `UIUniform.is_script_active` + `UINodeState.is_brain_active` change meaning and are
DROPPED (decision 8 of conventions persistence-evolution: an intentional reset drops the key).
`load_node_from_dir` already filters unknown keys + logs them (fail-soft) — so a 047 `node.json` loads
clean, the stale flags ignored. `is_brain_active=True` intent is harmless to lose (the brain auto-binds now).

## Design decisions (locked)

1. **One script per node — per-uniform scripts deleted entirely.** A node has at most one
   `nodes/<id>/scripts/script.py` (the brain), `update(self, ctx) -> dict[str, value]`. DELETE: the
   `u_<name>__<tag>.py` filename scheme, `_SCRIPT_GLOB`, `parse_script_filename`, `per_uniform_filename`,
   `uniform_tag`, `stub_for` (per-uniform stub), the per-uniform discovery loop in `engine.py::reload`, the
   `_resolved_pairs` 1-vs-N fan-out (the brain is always N), the two-pass brain-then-per-uniform write order
   + the `brain_driven` conflict-freeze-fallback in `_apply_behavior`, `same_type_scripts`,
   `copy_script_body`, `_draw_copy_content_selector`, the `uniform_script` tab kind. The engine collapses to:
   discover `script.py` → compile → tick → fan its dict into `(name, value)` pairs → coerce each → write.
   This explicitly **supersedes 047 F14** (the `(name,type)` tag + copy-content) and **044's per-uniform
   override conflict rule** (there is no per-uniform script left to override the brain). The `coerce_one`
   atom, the freeze-as-data store, the export-isolation seam, `EngineContext`, the typed outputs, and the
   per-key vs behavior-level freeze granularity all STAY (they were always brain-shared).

2. **The script auto-binds — `is_brain_active` retires.** With one script and an empty-dict default, "the
   file exists" IS the binding; there is no activate step. `engine.reload` binds `script.py` whenever it
   exists on disk (no `active_scripts` set, no `is_brain_active` flag). DELETE `is_brain_active` from
   `UINodeState` and the `_active_scripts_for`/`active_scripts` machinery 047 added. The engine's
   `reload(active_scripts=…)` param is removed (it existed only for the per-uniform/brain active flags).

3. **Play/stop state is node-scoped + name-keyed, NOT a per-`UIUniform` flag.** Add
   `UINodeState.stopped_uniforms: list[str]` (persisted) — the uniform NAMES the user has STOPPED (frozen for
   manual edit). Stored as `list[str]`, NOT `set[str]`: `UINode.save` serializes via `model_dump()` (default
   mode) → `json.dump`, which raises `TypeError` on a Python `set` (verified empirically against the current
   save path). `ProjectSession.tick` coerces it back to a `set` when building the per-frame `stopped`. A
   per-`UIUniform` flag would re-trip the **lazy-row law** (`conventions.md ## Design decisions` — a row born
   on first draw / a retyped uniform gets a fresh `UIUniform`, leaking the flag; the 047 ROOT-2 trap). A
   node-scoped name-keyed list survives retype (the name is stable) and is reachable before any row draws.
   STOP and "absent from the dict" are DIFFERENT states (decision 5).

4. **THREE uniform states, the distinction is load-bearing (decision per the swarm's central finding):**
   - **MANUAL (absent)** — the script's dict does NOT return this name. Fully hand-editable, value sticks,
     name uncolored, NO play/stop button (nothing to resume). This is the default for every non-scripted
     uniform.
   - **PLAYING** — the script returns this name AND it is not in `stopped_uniforms`. The engine writes it
     each tick; the name is colored (decision 9); the row shows a **STOP** button. The value widget stays
     editable but the tick overwrites it next frame (snap-back, the 047 F11 cue) UNLESS the edit auto-stops
     it (decision 6).
   - **STOPPED** — the script returns this name AND it IS in `stopped_uniforms`. The engine SKIPS the write
     (the brain still ticks; `self.*` advances; the name still enters `last_driven`); the value is
     hand-editable and STICKS; the row shows a **PLAY** button; the name is uncolored.
   The UI tells PLAYING/STOPPED (button present) from MANUAL (no button) via the engine's `last_driven` set —
   a name in `last_driven` is script-targeted (playing or stopped); a name absent from it is pure-manual.

5. **The engine learns STOP via a fresh per-frame `tick(stopped=…)` param — never cached across tick/draw.**
   `ProjectSession.tick` builds `stopped: set[str]` from `UINodeState.stopped_uniforms` ∪ (`all_stopped` ?
   every driven name : ∅) each frame (`_stopped_for`) and passes it to `ScriptEngine.tick`. In
   `ScriptEngine._tick_brain`'s write loop, a `name in stopped` SKIPS the `node.uniform_values[name] =
   coerced` write but STILL records the name in `driven`/`last_driven` (so the UI's play/stop button shows)
   and still advances `self.*` (the run already happened). The skip lands BETWEEN the `driven.add(name)` and
   the `coerce_one` call — `driven.add` already precedes the coerce, so a STOPPED uniform stays in
   `last_driven` and keeps its PLAY button. `last_good` is intentionally NOT updated while stopped (frozen
   at stop); a resume-coercion failure falls back to the pre-stop `last_good` (existing freeze-as-data
   behavior, accepted). The `stopped` param DEFAULTS to empty; `tick_export` (the EXPORT path)
   forwards NO stopped set, so an export plays every driven uniform deterministically (a STOPPED uniform in
   the live preview still renders the script value in an export). Mirrors the retired `active_scripts`
   plumbing — intent flows through a param, the engine never learns `UINodeState` (the headless boundary holds).

6. **STOP auto-engages when the user grabs a PLAYING uniform's widget (`is_item_activated()`).** In
   `draw_ui_uniform`, immediately after the value-PRODUCING branch (the `drag_*`/`color_edit`/`input_text`
   item) and BEFORE the trailing play/stop button, capture `imgui.is_item_activated()` (fires ONCE on grab —
   NOT `is_item_active()`/return-changed, which spam every drag frame). Gate the auto-stop on
   `state == PLAYING`: only a PLAYING uniform can auto-stop. This defuses the per-input-branch hazard — a
   `texture`/`buffer`/`auto` row's trailing item is not a settable scalar, but a texture is non-scriptable so
   it's never in `last_driven` → never PLAYING → never auto-stops; the gate makes the capture point safe across
   all branches. If `playing and is_item_activated()`, add the name to `stopped_uniforms` THIS frame so the
   manual edit applies and sticks. The user RESUMES via the row's PLAY button (removes it from
   `stopped_uniforms`) or node-play. This is the maintainer's explicit choice (they rejected drag-wins-while-held:
   a vec3 can't be set in one click). The footgun the devil's-advocate raised (silent sticky latch) is defused:
   STOP is now a VISIBLE, named, persisted state with an obvious PLAY button — the appearing button + the
   un-colored name ARE the cue the old invisible snap-back lacked.

7. **Whole-node play/stop freezes WRITES, the brain keeps TICKING.** Add `UINodeState.all_stopped: bool`.
   Node-STOP must NOT stop the brain ticking — if it did, `self.*` would freeze and node-PLAY would resume
   from stale state, not "resume from script" (the maintainer's words). So node-STOP adds every driven name
   to the per-frame `stopped` set (decision 5) while the brain still runs; node-PLAY clears `all_stopped`.
   A per-uniform stop coexists (the union). Node-play does NOT clear per-uniform `stopped_uniforms` (those
   are explicit user intent); it only clears the `all_stopped` blanket. (Open question 1 — default chosen.)

8. **Explicit imports in the stub, injection kept as a fallback.** The script object's available types must
   be VISIBLE (the maintainer: "никто не знает есть ли у меня Vec3"). `brain_stub_for` emits a real
   `from shaderbox.scripting import ScriptBehavior, Ctx, Vec2, Vec3, …` line. The emitted set is pinned:
   `ScriptBehavior` + `Ctx` ALWAYS, plus only the output types the node's uniforms reference (a vec3 uniform →
   `Vec3`, a text uniform → `Text`, …); emit only names that exist in `shaderbox/scripting/__init__.py`'s
   `__all__`. Verified empirically: this import RESOLVES inside the exec'd script — `behavior.py::_build_globals`
   binds `__builtins__` at MODULE scope (the module form carries `__import__`), and `__init__.py`'s `__all__`
   already exports `ScriptBehavior`/`Ctx`/`Vec2`/`Vec3`/`Vec4`/`Array`/`Text`/`MouseState` (all eight present —
   no export change needed). `_build_globals` KEEPS injecting the same names as a fallback, so a user who
   deletes the import line degrades to today's behavior instead of an opaque eager-annotation-eval
   compile-freeze; a malformed import line fail-soft-freezes via the existing `except Exception` in
   `PythonBehavior.__init__` (compile-error-as-data, not a crash). "Don't hide anything from the user, *насколько
   можно, если это возможно*" — the user's own "if possible" sanctions the safety net.

9. **Script-driven uniform names colored `STATE_INFO` (blue), NOT yellow.** The maintainer said "yellow",
   but yellow IS the default accent (`ACCENT_PRIMARY`) AND `STATE_WARN` AND `FAVS`/`SYN_TYPE` — a literal
   yellow name is indistinguishable from a warning / a yellow-accent user's accent, and a new yellow token
   fails the theme import-time invariant (`theme.py` — a fixed hue must differ from every accent + every
   STATE_*). Use `COLOR.STATE_INFO` (`blue_n`), which `_draw_auto_row` ALREADY uses for engine-driven
   uniform names — giving every "machine-owned" row (engine-driven + script-driven) a consistent semantic.
   Only PLAYING names are colored; STOPPED + MANUAL stay default. (Open question — blue chosen; surfaced.)

10. **Tab labels are NODE-DERIVED; on-disk filenames stay STABLE.** REJECT renaming disk files after the
    node name (the swarm was unanimous: `ui_name` is mutable, non-fs-safe, non-unique, can be empty; node
    identity keys on the immutable UUID dir; `_NODE_SHADER_BASENAME`/`script.py` are load-bearing across
    `EditorSession` path-keying, the mtime watcher, save/load, `_on_node_*`). Instead a `tab_label(app, tab)`
    helper derives the DISPLAY string: `"<ui_name> (shader)"` / `"<ui_name> (script)"` for node tabs,
    `"library - <path.stem>"` for lib tabs (ASCII hyphen — RUF001), falling back to a short id slice when
    `ui_name` is empty. The imgui tab `##id` stays keyed on the stable `tab.path`/index (NOT the label) so two
    briefly-identical labels can't merge tabs. The copy-path in `draw_chrome` stays the real path. This
    partially REVIVES the `_tab_label` semantic-alias branching 047 decision 8 deleted (noted in 047).

11. **The `node_script` tab kind is the only script kind; `EditorTab` loses `name`; `ScriptState` retires.**
    DELETE the `uniform_script` kind; rename `node_script` → `script` in `EditorTabKind` (`editor_types.py`).
    The `uniform`-addressing dimension (`EditorTab.name`, `open_script_for(node_id, name)`'s `name` param) is
    vestigial with one script per node — REMOVE it (don't leave it always-None). `_is_script_tab` /
    `_script_errors_for` / `open_script_for` simplify to the single brain. The consumers that key on the old
    shape MUST update: `tabs/code.py::_tab_has_error` + `_is_script_tab` + the `tab.kind in
    ("node_script","uniform_script")` check in `draw` (→ `kind == "script"`, drop the `tab.name` arg from
    `script_state_for`). RETIRE the `ScriptState` literal entirely (it described the per-uniform glyph's
    4-state with `absent`/`inactive` — gone with the glyph): drop it from `engine.py`, `scripting/__init__.py`
    (`__all__` + import), and its three importers (`ui_primitives.py`, `widgets/uniform.py`,
    `project_session.py`). A brain tab's error tint reads a plain bool (`get_brain_status(...).sentinel_error
    is not None`), not a `ScriptState`.

12. **The script-local editor bar is deleted; node play/stop lives in the node header.** DELETE
    `tabs/code.py::_draw_script_bar` + `_draw_copy_content_selector` + their call (the Activate/Deactivate
    button retires with `is_brain_active`; the copy selector retires with per-uniform scripts; the
    stale-retype warning retires with the tag scheme). The node header (`tabs/node.py`) gets TWO affordances
    where `_draw_brain_glyph` was one: (a) an **open-script glyph** (click opens/creates the `script.py` tab —
    navigation), and (b) the **node play/stop** toggle (execution). Header trailing cluster:
    `[name input] [resolution combo] [open-script] [node play/stop] [...]`. (Open question 1 — two buttons.)

13. **Per-uniform play/stop replaces the row's `</>` glyph 1:1.** DELETE `widgets/uniform.py::_draw_script_glyph`.
    At the same row trailing slot, draw a play/stop toggle ONLY for a script-targeted uniform (name in the
    engine's `last_driven`): a STOP button when PLAYING, a PLAY button when STOPPED. A MANUAL uniform (not in
    `last_driven`) shows NOTHING there. Use a `ui_primitives` play/stop primitive (draw-list triangle/bars, NOT
    a font glyph — the §1/§8 mixed-font-in-button caveat), tier ghost. Frozen mid-copilot-turn
    (`begin_disabled(copilot_turn_active)`).

14. **Script-driven widgets stay editable; the write-back skip follows the play/stop state (047 F11 kept,
    re-cast).** Keep the "value widget stays editable" behavior. The manual-write-back skip
    (`if new_value is not None and not owned`) now keys on PLAYING (engine owns + not stopped): a PLAYING
    uniform's manual edit triggers auto-stop (decision 6) so the same-frame write applies AND it becomes
    STOPPED (so the tick no longer overwrites). A STOPPED/MANUAL uniform's edit always applies + sticks.
    `is_uniform_script_owned` collapses to "name in `last_driven` and not stopped" (PLAYING).

15. **The startup blank-editor bug + dead-node-id guard ship as a STANDALONE commit BEFORE the redesign.**
    Root cause (swarm-unanimous): `app.py::_init` calls `session.load()`, which restores `current_node_id` by
    direct field assignment in `UIAppState.load_and_migrate` — NOT through `set_current_node_id` — so
    `_on_current_node_changed` (the only caller of `ensure_shader_tab` for the restored node) never fires;
    `editor_tabs` stays empty until the user switches nodes. Fix: after load/seed in `_init`, if
    `current_node_id` is a live node, call `ensure_shader_tab(current_node_id)` (idempotent — `_focus_or_add_tab`
    dedups by path). Do NOT reroute `load()` through `set_current_node_id` (it would fire a transition before
    exporters/bindings are wired). Dead-node-id guard (independent latent bug): if the restored
    `current_node_id` is NOT in `ui_nodes` and `ui_nodes` is non-empty, reselect a valid node via
    `set_current_node_id`. This commit lands FIRST so a trivial smoke-verifiable fix doesn't ride the
    high-blast redesign diff.

16. **Documentation lives in the stub docstring (canonical) + this spec (narrative).** `brain_stub_for`'s
    docstrings (`_UPDATE_DOC`/`_INIT_DOC` + the class doc) are the canonical, at-the-moment-of-use reference:
    the dict contract (name→value, absent=manual), `ctx.t/dt/frame/mouse`, the importable types (the explicit
    import line decision 8), `self.*` persistence, and the play/stop interaction ("a returned uniform PLAYS;
    STOP it to edit by hand; editing a playing uniform auto-STOPs it"). NO standalone `scripting_reference.md`
    (a third home guarantees drift — the one-canonical-home rule). The stub's commented examples show the
    node's available uniforms (discoverability) while `update` returns an empty dict by default (decision 17).

17. **Empty-dict default, commented examples.** `update` returns `{}` by default (the maintainer's explicit
    ask — a fresh script drives nothing, every uniform stays manual). The stub PRE-SEEDS the dict as
    COMMENTED example lines (one per scriptable uniform at a coercion-valid default) so the user sees what's
    available without a live-driving body. Uncomment + edit to drive. (Open question 2 — empty+commented chosen.)

## Files touched

- `shaderbox/scripting/engine.py` — DELETE per-uniform discovery/glob/`parse_script_filename`/
  `per_uniform_filename`/`uniform_tag`/`stub_for`/`_resolved_pairs` fan-out/two-pass override/`brain_driven`
  fallback; `reload` drops `active_scripts`, binds `script.py` by existence; `tick`/`_apply_behavior` gain a
  `stopped: set[str]` param skipping the WRITE (keeping `driven`/state-advance); rewrite `brain_stub_for`
  (explicit import line + empty-dict body + commented examples + the play/stop docstring).
- `shaderbox/scripting/behavior.py` — KEEP `_build_globals` injection (fallback for decision 8); update its
  comment to name the fallback role.
- `shaderbox/scripting/__init__.py` — export `ScriptBehavior` + `Ctx` (+ `MouseState`?) + the output types in
  `__all__` so the stub's real import line resolves.
- `shaderbox/project_session.py` — DELETE `same_type_scripts`/`copy_script_body`/`_active_scripts_for`/per-
  uniform branches of `script_state_for`/`uniform_pill_state`/`script_path_for`/`create_script`/`set_script_active`;
  collapse to the single brain; `tick` builds the `stopped` set (from `stopped_uniforms` ∪ `all_stopped`) and
  passes it; `is_uniform_script_owned`→PLAYING; new setters `set_uniform_stopped(node_id, name, stopped)` +
  `set_node_all_stopped(node_id, stopped)`; `is_uniform_playing`/`is_uniform_stopped`/`uniform_is_driven`
  (reads `last_driven`); `script_driven_uniforms` keeps working (brain-only now); `script_file_for` always
  `script.py`.
- `shaderbox/ui_models.py` — DROP `UIUniform.is_script_active` + `UINodeState.is_brain_active`; ADD
  `UINodeState.stopped_uniforms: set[str]` + `all_stopped: bool` (both defaulted, pydantic-serialized); the
  one-time `u_*.py`→`script.py` migration in `load_node_from_dir` (or a `ProjectSession.load` post-step).
- `shaderbox/widgets/uniform.py` — DELETE `_draw_script_glyph`; draw the play/stop primitive only for a
  driven uniform; auto-stop on `is_item_activated()` captured before the trailing button; name color
  `STATE_INFO` when PLAYING; write-back skip follows PLAYING.
- `shaderbox/tabs/node.py` — replace `_draw_brain_glyph` with the open-script glyph + node play/stop cluster.
- `shaderbox/tabs/code.py` — DELETE `_draw_script_bar`/`_draw_copy_content_selector` + the call;
  `_script_errors_for` keeps only the brain branch; route tab labels + `draw_chrome` through `tab_label`.
- `shaderbox/app.py` — decision 15 (startup fix + dead-node-id guard, SEPARATE commit); `open_script_for`
  loses `name`; `EditorTab` kind collapse plumbing; `set_script_active`/copy-content forwarders deleted; new
  play/stop forwarders.
- `shaderbox/editor_types.py` — `EditorTabKind` drop `uniform_script`, rename `node_script`→`script`; drop
  `EditorTab.name`.
- `shaderbox/ui_primitives.py` — new `play_stop_toggle(id_, playing, *, tooltip) -> bool` (draw-list, no font
  glyph); a `tab_label` helper if it lives here (else in `code.py`); DELETE `script_glyph`.
- `shaderbox/copilot/backend.py` — collapse the two-way `set_uniform` reject (per-uniform vs `script.py`) to
  one message; keep the reject firing (the brain still owns driven uniforms).
- `tests/test_script_engine.py` / `tests/test_script_engine_gl.py` / `scripts/dogfood/verify_script_engine.py`
  — rewrite the per-uniform tests to the single-brain + play/stop model; add the stopped-skip tick-canary;
  delete the tag/copy-content tests; keep the brain + freeze-as-data + export-isolation tests. DELETE
  `tests/test_script_filename_helpers.py` (the `parse_script_filename`/`per_uniform_filename`/
  `same_type_scripts` tests — all-deleted machinery) and `tests/test_uniform_pill_state.py` (the 4-state pill
  is gone). `tests/test_script_driven_reject.py` STAYS (it exercises only `script_driven_uniforms`, which
  survives brain-only). `scripts/dogfood/verify_script_engine.py` drops its `set_script_active` call.
- `scripts/smoke.py` — add `len(app.editor_tabs) > 0` after `_init` (decision-15 regression canary). The
  script-driven smoke seed gains an INTEGRATOR key (`self.v += ctx.dt` returned under a uniform name) so the
  stopped-skip canary is falsifiable: tick → assert the integrator value is written; add the name to the
  `stopped` set → tick → assert the value is FROZEN at the manual value AND the name is still in
  `last_driven` AND a SECOND (un-stopped) driven key still advanced. The `set_script_active` call at
  smoke.py:168-169 is deleted (the flag is gone).
- `ai_docs/features/044_node_brain_script.md` / `047_scripting_ux_refinement.md` — mark the superseded
  sections (044's per-uniform-override conflict rule; 047's F14 tag scheme + copy-content + the
  active-flag/`.disabled`-replacement decisions).
- `ai_docs/conventions.md` — the scripting Design-decision bullet: collapse to one-script-per-node + the
  play/stop state model + node-scoped `stopped_uniforms`; retire the 044 explicit-wins conflict clause +
  047's active-intent-on-model clause.
- `ai_docs/todo.md` — the 043 deferral's checkpoint capture is now one path (`scripts/script.py`), not a
  glob; retire the 044 cross-script-order entry's per-uniform framing where stale; retire the resolved 047
  type-change references.
- `ai_docs/roadmap.md` — banner rewrite + the 048 row; 047 row brief notes the partial supersession.
- `projects/dev/` — the Script Showcase node rebuilt for the single-brain model (Finalization) + sandbox sync.

## Manual verification (maintainer `make run` — headless can't judge these)

Per dev_flow step 7, each check falsifiable + names the consumer:

- **BUG #15 startup (verify FIRST, separate commit)** — launch on a project whose `current_node_id` is a
  saved node → the editor LEFT shows that node's shader code immediately, no switch needed. Falsifier: a
  blank editor on launch that fills only after switching nodes = the `ensure_shader_tab`-after-load wire is
  missing. Smoke also asserts `len(app.editor_tabs) > 0` after `_init`.
- **BUG #15 dead-node-id** — point `app_state.json` `current_node_id` at a non-existent id, launch → a valid
  node is selected + its code shows, not a permanent blank. Falsifier: blank editor with no recovery.
- **D1 one script** — a uniform row has NO `</>` glyph and NO per-uniform create affordance anywhere; only the
  node header opens THE script. Falsifier: any per-uniform script-create UI remains.
- **D4 three states** — (MANUAL) a uniform the script doesn't return: editable, sticks, no play/stop button,
  name default color. (PLAYING) a uniform the script returns: name BLUE, a STOP button, value snaps back on
  drag-release unless auto-stopped. (STOPPED) press STOP: name default, a PLAY button, value edits + STICKS,
  and the brain's OTHER uniforms keep playing (proving the brain still ticks). Falsifier: a MANUAL uniform
  shows a play/stop button; a STOPPED uniform's value snaps back; stopping one uniform freezes the others.
- **D5 stopped-skip ticks the brain** — STOP a uniform that the brain integrates (e.g. an accumulator);
  leave it stopped a while; PLAY it → it jumps to the value the brain reached WHILE stopped (state kept
  advancing), not the value it had at stop time. Falsifier: on PLAY it resumes from the stop-instant value =
  the brain stopped ticking (decision 5 violated).
- **D6 auto-stop on grab** — with a uniform PLAYING, grab+drag its slider → it STOPS (PLAY button appears,
  name un-colors) and the dragged value sticks. A single click that doesn't change the value also stops it
  (activation, not edit). Falsifier: the value snaps back after release (auto-stop didn't fire) OR it
  auto-stops every frame of a drag on a DIFFERENT already-stopped uniform (wrong predicate).
- **D7 node play/stop freezes writes, brain ticks** — node-STOP → every playing uniform freezes (all
  editable + sticking), but the brain's integrator state keeps advancing; node-PLAY → uniforms resume from
  the advanced state. Per-uniform stops set before node-stop remain stopped after node-play. Falsifier:
  node-PLAY resumes from stale state (brain stopped ticking) OR clears an explicit per-uniform stop.
- **D8 explicit imports + fallback** — a fresh `script.py` has a visible `from shaderbox.scripting import …`
  line naming the types; the script runs. DELETE the import line → it STILL runs (injection fallback), no
  compile-freeze. Falsifier: deleting the import line opaque-freezes the script.
- **D9 blue names** — a PLAYING uniform's name is BLUE (same hue as the engine-driven `u_time` row), not
  yellow/accent. Falsifier: the name is yellow (collides with accent) or a warning hue.
- **D10 tab labels** — two nodes' shader tabs read `"<NodeA name> (shader)"` / `"<NodeB name> (shader)"`,
  the script tab `"<name> (script)"`, a lib tab `"library - <file>"`. Rename a node → its tab label updates
  live; the tabs don't merge. Falsifier: both shader tabs read the same label, or renaming merges/breaks tabs.
- **D10 colliding-label selection** — rename NodeA so its label exactly matches NodeB's, then
  programmatically open NodeA's script tab (a node-select / glyph open) → the editor switches to it.
  Falsifier: the editor stays on the wrong tab = the `##tab{i}` id (or set_selected drive) regressed under
  colliding labels (the 047 ROOT-1 read-back trap re-broken by label-keying). The `##id` MUST stay keyed on
  the index / stable `tab.path`, never the mutable `tab_label`.
- **Export-isolation (D5)** — STOP a driven uniform in the live preview, then EXPORT the node → the exported
  render shows the SCRIPT value for that uniform (export forwards no `stopped` set), not the frozen manual
  value. Falsifier: the export freezes the stopped value = `tick_behaviors` leaked the live `stopped` set.
- **D16/D17 stub** — open a fresh node's script → an empty-dict `update` body, commented example lines naming
  the node's uniforms, the explicit import line, and a clean docstring documenting ctx + play/stop. Falsifier:
  the stub drives uniforms by default (non-empty dict) or hides the available types.
- **Migration** — `make run` against a project carrying old `u_*.py` files (the dev sandbox before rebuild)
  → each affected node's `script.py` gains a commented block with the old bodies + a log names them; the
  `u_*.py` files are trashed. Falsifier: the `u_*.py` bodies are silently lost OR still drive nothing-visible.
- **Copilot reject still fires** — `set_uniform` on a PLAYING uniform via the copilot is rejected with a
  one-line "driven by script.py, edit the script" message. Falsifier: the set_uniform silently no-ops or the
  message names a per-uniform file.
- **`make check` + `make smoke` green, AND smoke proves the stopped-skip wire** — smoke ticks a script-driven
  seed, adds a uniform to `stopped`, ticks again, asserts the value is the manual value (not the brain's), and
  asserts the brain's OTHER driven uniform still advanced. Falsifier: an unchanged-value pass with no stopped
  set proves nothing; the canary must go RED if the `stopped` skip is unwired.

## Open questions for the user (defaults chosen — change any if you disagree)

1. **Node play/stop + open-script = two separate header buttons?** DEFAULT: yes, two (open = navigation,
   play/stop = execution; conflating them is what made the old brain glyph ambiguous). If the header is too
   tight at `make run`, the open-script glyph moves into the editor and the header keeps node-play/stop + "...".
2. **Empty-dict default with commented examples (vs a pre-seeded live-driving stub)?** DEFAULT: empty dict +
   commented examples (your explicit ask — a fresh script drives nothing; uncomment to drive). Discoverability
   comes from the comments + the import line, not a live body.
3. **Script-driven name color: blue (`STATE_INFO`) instead of yellow?** DEFAULT: blue. Yellow is the default
   accent + the warning hue + favorites — a literal-yellow name is ambiguous and fails the theme invariant.
   Blue matches the existing engine-driven-uniform name color (consistent "machine-owned" semantic). One-token
   change to a different hue if you insist, but no collision-free saturated yellow exists.
4. **Auto-stop fires on widget GRAB (`is_item_activated`) — engages even on a click that doesn't change the
   value.** DEFAULT: grab (one-shot, no per-frame spam). The alternative (only on an actual value change) lets
   a click-to-focus NOT stop, but a vec3's first component change would stop before you set the rest — grab is
   the cleaner "I'm taking manual control" signal.
5. **Auto-migration concatenates old `u_*.py` bodies as a COMMENTED block into `script.py` (then trashes the
   files).** DEFAULT: yes — preserves your work for manual merge, never silent, recoverable from trash. The
   alternative (warn-and-leave the files) keeps them on disk driving nothing, which is the confusing state
   we're removing.

## Review history

### Validation swarm (2026-06-14, ultracode — 4 lenses + devil's-advocate + synthesis)

Confirmed every maintainer concern against real code (the startup bug root cause unanimous; the two-source
glyph ambiguity; the magic-injected namespace; indistinguishable tab labels). Refinements adopted into the
locked decisions above: the three-state play/stop model (MANUAL/PLAYING/STOPPED — "absent" ≠ "stopped");
node-scoped name-keyed stop state (not a per-`UIUniform` flag — sidesteps the 047 ROOT-2 lazy-row trap);
node-stop freezes WRITES not ticking (else resume is from stale state); auto-stop on `is_item_activated()`;
yellow→`STATE_INFO` blue (yellow is the default accent — hard collision); tab-label-only, never disk-rename;
explicit imports WITH the injection fallback. The devil's-advocate's "keep per-uniform scripts / smaller fix"
was weighed and overruled by the maintainer's explicit twice-stated "remove entirely / vacuum it" — but its
findings converted to hard requirements: the loud recoverable migration + the explicit 044/047 supersession.

### Pre-impl review (2026-06-14, 2 adversarial agents — correctness/design + verification/blast-radius + triage)

High-blast-radius (deletes a just-shipped mechanism, changes the persisted model, reverses two locked 047/044
decisions), so 2 reviewers + a triage pass; the verification reviewer anchored to the 047 post-impl ROOT
findings + the conventions lazy-row law + the user transcript (non-self-authored). 13 findings ACCEPTED and
folded above; 2 rejected as false positives (verified on disk):
- **R2-B1 (BLOCKER, dominant) — `set[str]` crashes `UINode.save`.** `model_dump()` (default mode) → `json.dump`
  raises `TypeError` on a Python set, even an empty one (verified empirically). → D3 stores `stopped_uniforms`
  as `list[str]`, coerced to a set per-frame in `tick`.
- **R1-SF-7 / R2-B2 (BLOCKER) — `set_script_active` blast radius + two unlisted test files.** Deleting the
  flag breaks smoke.py:168-169, the dogfood verify script, the app forwarder, the code.py bar. → Files-touched
  now DELETES `test_script_filename_helpers.py` + `test_uniform_pill_state.py` and enumerates the break sites;
  `test_script_driven_reject.py` STAYS (uses only the surviving `script_driven_uniforms`).
- Folded clarifications: the stopped-skip lands between `driven.add` (693) and `coerce_one` (695) so a STOPPED
  uniform keeps its PLAY button (D5); export forwards no `stopped` set (D5 + export-isolation canary); auto-stop
  gated on `state == PLAYING` to defuse the per-input-branch trailing-item hazard (D6); `ScriptState` retired
  with its 3 importers (D11); the migration pinned to `ProjectSession.load` over `nodes_dir` only, idempotent
  by "no `u_*.py` left", dev-sandbox sequenced (migration section); the smoke seed gains an integrator so the
  stopped-canary is falsifiable; D10 gains the colliding-label selection falsifier.
- **REJECTED (false positives, verified):** R1-BLOCKER-2 ("`__all__` omits `Text`") — `Text` IS in `__all__`
  (line 41); all 8 surface names resolve, the D8 import works (re-confirmed empirically by exec'ing a real
  `from shaderbox.scripting import …` script). R1-BLOCKER-1's "double-append on trash collision" — `detach_script`
  already collision-renames; only its dev-sandbox-sequencing half was real (folded).

Overall: implementable as-is-plus-the-13-edits; no fresh user sign-off required beyond the 5 open-question
defaults below. The architecture is internally consistent.

### Post-impl review (2026-06-14, 3 adversarial agents — code-correctness + architecture/conventions + spec-fidelity + triage)

High-blast-radius diff → 3 reviewers; the architecture reviewer anchored to conventions.md (the lazy-row
law) + the 047 post-impl ROOT findings (non-self-authored). Verdict: FAIL — two real BLOCKERs + cleanups,
all fixed; spec-fidelity confirmed 15/17 decisions landed clean (the 2 "partial" were prose-vs-impl reads,
not gaps). Findings (each verified against code before triage):
- **B1 (BLOCKER) — startup fix missed the empty-id case.** The `_init` guard `if current_node_id and …`
  left the editor blank when `current_node_id == ""` but the project HAS nodes (a load with no saved
  pointer) — which made `make smoke`'s decision-15 canary RED. Fix: the `elif` keys on `self.ui_nodes`
  alone (empty/stale id both fall through to selecting a live node).
- **B2 (BLOCKER) — a STOPPED uniform was clobbered on a per-key coercion failure.** `_apply_behavior`'s
  failure path wrote `frozen` (stale last-good) unconditionally; for a stopped uniform that overwrites the
  user's manual value. Fix: guard the freeze-write `if name not in stopped` (matching the success path);
  the error is still recorded + the name stays driven.
- **Cleanups (ACCEPTED):** deleted the now-dead `get_script_file_for` plumbing (project_session method +
  capability wire + backend ctor param + field + the test stub arg); added `tests/test_migration.py` (3
  falsifiable tests — body-survives / orphans-trashed / idempotent / noop); rebuilt the dev-sandbox node
  `4c6c34f4` via the real migration + re-saved all node.jsons clean; filed the pre-existing full-suite GL
  crash as a `todo.md` deferral.
- **REJECTED (false positives, verified):** "`last_good[name]=coerced` while stopped contradicts D5" — the
  success path correctly tracks the brain's advancing state (D5's "frozen at stop" is about the freeze
  TARGET on a resume-failure, not the bookkeeping); the `all_stopped` frame-1 flash (cosmetic, inherent);
  test `# type: ignore` markers (test-scoped, not the production allowlist).

### Convergence re-review (2026-06-14, 3 verifiers — B1 / B2 / cleanups+regression)

Re-spawned against the patched tree: all three PASS. B1 verified across all 3 cases (valid/empty/stale id)
with the smoke canary green live (exit 0); B2 traced line-by-line (stopped+bad-shape → manual value
survives, error recorded, name stays driven, freeze-as-data preserved for playing); cleanups confirmed zero
dangling refs + the migration wired into `load`. `make check` 0 errors; 461 tests green (the only RC=1 is
the pre-existing V3D multi-context teardown crash, stash-verified at baseline + filed in todo.md).

### Final polish swarm (2026-06-14, ultracode — 6 finder lenses → 3-skeptic verify → synthesis)

6 finders (correctness, ui/imgui, conventions, user-intent, deletion-completeness, tests/docs) → each
finding majority-verified by 3 skeptics (≥2/3 CONFIRMED + worth-fixing) → synthesis. 8 majority-confirmed
fixes, all surgical, applied + convergence-verified:
- **[MED] `create_script` listed engine-driven uniforms as stub examples** the engine silently drops →
  excluded `ENGINE_DRIVEN_UNIFORMS` from the stub's uniform list (the legibility gap 048 targets).
- **[MED] doc staleness:** `conventions.md` editor-tab bullet (`node_script`/`uniform_script` → the real
  `shader`/`script`/`lib` + node-derived labels); `todo.md` 043 deferral (`u_*.py` → `script.py`); the
  cross-shared-state NOTE's deleted brain-vs-`u_*.py` conflict clause.
- **[LOW] dead code:** `is_uniform_scriptable` (session + App forwarder), `is_uniform_playing` (session),
  the `App.detach_script` forwarder — all zero-caller, deleted (`ProjectSession.detach_script` kept; the
  migration calls it).
- **[LOW] smoke truth-in-comment:** the seed's 2nd key was `ctx.mouse.x` (frozen headless) while the
  comment claimed it "advances" → both keys are now real integrators + the canary asserts the un-stopped
  key ADVANCED while the stopped one froze (the spec-mandated sibling-advance check).
REFUTED (considered, not majority-worth): the pre-existing inert "Стишок" black-render sandbox drift; the
per-uniform play button being a no-op under `all_stopped` (deep edge, 1/3); stale-`stopped_uniforms`
pruning; the read-only-`script.py` migration brick; assorted comment-only nits. After the fixes: `make
check` 0 errors, 461 tests green, dogfood verify PASS, GL test 6/6 in isolation — COMMIT-READY.
