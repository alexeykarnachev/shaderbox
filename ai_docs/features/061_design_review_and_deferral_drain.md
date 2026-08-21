# 061 — design review & deferral drain

**Status:** done (2026-08-21)

## Goal

060 audited the module LAYOUT and returned a null result: the layout is sound. But it was framed as
a *rot hunt*, and it deferred seven items with the reasoning "real, but this wave had
better-evidenced work". This wave drains that register and sweeps the design level a layout audit
structurally cannot see: runtime contracts, concurrency, lifecycle, and persisted-model integrity.

Method: 4 sonnet lanes (brake falsifiability, node-handle resolvers, worker lifecycle, persisted
model), each anchored to a concrete in-repo artifact and required to demonstrate by execution.
**Every finding was re-verified first-hand before any code changed** — which mattered again: two
lanes corrected the deferral they were auditing, and one of my own fixes inherited a bug the lane
then found in it.

## The architectural answer: still no reorg

Independently re-derived, and it agrees with 060. `CopilotCapabilities` is 41 methods because the
product is 32 tools — the Protocol is 1:1 with the tool surface, and splitting it would produce
facades that all get injected into the same constructor. `App`'s 132 fields are genuinely
UI/glfw/imgui-bound state. No boundary moved in this wave.

What was wrong was never the shape of the modules. It was contracts that no test held to.

## Fixed

Ordered by consequence. Each fix carries a guard that was **mutation-tested** — reintroduce the bug
and the named test goes red.

1. **[CRITICAL] Saving with no live program wiped every tuned uniform value.** `UINode.save` rebuilds
   `node.json["uniforms"]` from `get_active_uniforms()`, which is empty whenever `node.program is
   None`. That state is ordinary, not exotic: `release_program()` nulls the program and returns
   WITHOUT recompiling (the recompile rides the next render), so an external shader edit picked up by
   the watcher, followed by a quit (`ui.py` calls `app.save()` on close), lands exactly there.
   Reproduced end-to-end on a shipped example: **8 tuned values lost, 14 cosmetic UI rows kept** —
   the priority exactly inverted. Now the existing on-disk block is carried forward.
   Guard: `tests/test_node_save_preserves_values.py`.
2. **[HIGH] One bad key in `app_state.json` cost the user every setting.** `UIAppState.load` caught
   `ValidationError` and returned `cls()`. A retired enum member — what any future refactor
   produces — or one wrong-typed value reset the node selection, fps, split, Telegram pack,
   keybindings and editor prefs together, and the app writes that back on quit. This is 060's
   credential-wipe class, one file over. The already-solved twin was in-tree: `IntegrationsStore`'s
   `_drop_unknown`, whose comment states the rule verbatim ("a retired field must cost the user that
   setting, never their tokens"). Generalised to `model_salvage.py` (`drop_unknown` + `drop_invalid`
   + `load_model`); BOTH stores now share one implementation. Guard: `tests/test_model_salvage.py`.
3. **[HIGH] Both exporter workers were `daemon=False`, breaking the stated teardown contract.**
   `conventions.md` says the worker "is **daemon**" because a survivor abandoned after a join timeout
   is re-joined by interpreter `_shutdown`. The copilot obeyed it; both exporters did not — and
   `copilot/config.py:98` even described them as doing "warn-and-leave", which is what they did NOT
   do. Measured: `release()` returns after its 1s join saying it abandoned the worker, then exit
   blocks the full 30s. Guard enumerates spawn sites from the AST, so a new worker defaults INTO the
   check: `tests/test_worker_daemon_contract.py`.
4. **[HIGH] A stale STOP killed the next worker and stranded the user's click.** Found by the
   lifecycle lane *in the `ExporterWorker` extraction this wave had just written* — the extraction
   faithfully preserved the bug. `stop()` whose join times out leaves its STOP in the queue; the next
   worker consumes that leftover and exits immediately, so the job the user just queued has no
   consumer and `in_flight` spins forever. Reproduced (`['upload', 'STOP']`, new worker dead) and
   fixed by stamping each STOP with the generation it was addressed to; a worker skips a STOP from an
   abandoned predecessor. After: `['upload', 'connect']`, worker alive.
5. **[MED] Exporter worker machinery was duplicated.** 060 deferred this and even noted "BOTH copies
   set `daemon=False`" without registering that as a contract violation. Extracted to
   `exporters/worker.py` (thread + lock + both queues + spawn + teardown + the push helpers);
   `youtube.py` no longer imports `queue` or `threading` at all. Telegram's asyncio task-cancel stays
   subclass-side, where it belongs. Its trigger ("next time a worker-thread bug is fixed in
   telegram.py") had fired.
6. **[MED] `push_event` dropped the events that clear `in_flight`.** The progress path is
   lossy-newest, which keeps the queue permanently FULL — precisely when `push_event` dropped. So the
   mitigation was the trigger, and a dropped `_ConnectEvent`/`_LinkEvent` left the UI spinning on a
   finished job. Events now evict a progress item rather than drop themselves.
7. **[MED] UI uniform rows outlived their uniforms.** `ui_uniforms` is keyed by a hash of name AND
   shape and populated lazily in the draw loop, so every rename and retype stranded its predecessor;
   the dict only grew (a demo: 3 rows -> 9 across six edits). 295 dead rows repo-wide, **including
   two shipped examples**. Pruned in `save` (the funnel every path reaches, including headless ones
   that never draw a row), never in the draw loop. Shipped examples + the dev sandbox were
   regenerated BY HAND through the normal load+save path. Guard: `tests/test_uniform_row_pruning.py`,
   which pins both sides of the bound (a clean example must lose nothing).
8. **[MED] Renaming a sampler orphaned its media file forever.** The unbind cleanup is keyed by the
   uniform's OWN name, so it can only visit names the shader still has — a renamed-away sampler was
   never looked at again. Demonstrated: three renames, three files, one reference. Same
   narrowed-domain shape as #7, fixed at the same funnel.
9. **[MED] Retired ids persisted forever.** `exporter_settings` was mutated in place rather than
   rebuilt, so a removed exporter's block survived — live proof in the tracked sandbox, which still
   carried `"x"` from commit `548a97c`. `key_bindings` had the same shape (inert at read time, but
   nothing ever removed it). Both now prune; the sandbox files were hand-fixed.
10. **[MED] A masked test — passed in the suite, failed alone.** `COPILOT_CONFIG` is a process-wide
    mutable singleton and loading ANY project pushes the persisted user limits onto it
    (`ProjectSession` -> `apply_limits`), with nothing restoring it. So
    `test_clean_streak_config_defaults_sane` asserted against whatever the last test left behind.
    Proven directly: `apply_limits(99)` moves the singleton to 99 while a fresh `CopilotConfig()`
    still reads 12. Fixed at both levels — the test asserts a fresh instance, and an autouse
    conftest fixture snapshots/restores the singleton around every test.
11. **[LOW] Node-dir basenames were re-spelled beside their own constants.** `"node.json"` had no
    constant at all (5 sites) and `sync_nodes_from_disk` re-spelled `"shader.frag.glsl"` — so a
    rename would break the typed loader loudly and the watcher's half-written-node guard SILENTLY
    (it would skip every dir forever, and a watcher reporting "nothing changed" looks exactly like a
    working watcher). Both now live in `paths.py` with `node_json_for`/`node_shader_for`.
    `ui_models.py` also stopped importing a private name from `core.py`.
    Guard: `tests/test_node_dir_layout.py`.
12. **[LOW] Persisted numerics were unbounded.** `global_target_fps` feeds `1.0 / target_fps` at two
    frame-loop sites; a `0` raises inside `update_and_draw`, which skips the `save()`/`release()`
    tail. The Settings slider clamps 30-240, but the MODEL accepted anything, so a hand-edited or
    corrupted file bypassed the UI's own contract. Bounds now live on the model — the funnel every
    loader passes through — and compose with #2 so an out-of-range value costs only itself.
13. **[LOW] Every JSON save omitted its trailing newline**, which is why every tracked `node.json`
    showed `\ No newline at end of file` and fought the repo's own `end-of-file-fixer` hook. Fixed at
    all three write sites.
14. **[LOW] A test helper's bare `except Exception`** turned a broken probe into a clean empty list —
    the same "a checker that narrows its own domain" family, in the test layer.

## Deferrals CORRECTED rather than drained

Two lanes overturned the register they were auditing. Both corrections are the finding:

- **Node-handle resolvers: OVERSTATED, no fix needed for safety.** 060 implied a hazard. An
  exhaustive probe (2,056,320 handle probes over every 2-node pair and both insertion orders, plus
  21,840,000 over 3-node configurations) found **zero** cases where the two resolvers select
  DIFFERENT nodes. The loose resolver is a strict SUPERSET of the strict one, so the only divergence
  is confirm-then-error, never a wrong deletion. The load-bearing detail 060 omitted: `node_tree`
  carries SHORT ids while the backend matches FULL ids, so the bidirectional test is required, not
  sloppy. Still fixed (the confirm-then-error UX is bad) by aligning the uniqueness rule; guarded by
  a cross-resolver equivalence test that pins the two predicates to each other.
- **Copilot brakes: 3 of 5, not 5.** `copilot_working_set_max_nodes` and
  `auto_revert_after_failed_edits` ALREADY have falsifiers
  (`test_copilot_context_bounds.py`, `test_content_editing.py`); those should never have been on the
  list. Of the remaining three, `bulk_gate_threshold` turned out **unreachable, not merely
  untested** — no tool declares `GatePolicy.BULK`, so the branch is dead in the live registry. Its
  guard now pins that fact and goes red the moment a tool adopts BULK.

## Deliberately NOT done

- **`exporters/telegram.py` worker-half split** (1283 -> ~780 + `telegram_api.py`). The worker
  machinery it shared with `youtube.py` is extracted, which was the coupling that mattered; the
  remaining split is a file-size argument. **Trigger:** unchanged.
- **`backend.py` at 2169 L / 29 injected callbacks.** Re-derived independently and the 060 verdict
  holds: one coherent state behind a Protocol that requires all methods on one object.
  **Trigger:** when a 5th distinct domain wants in.
- **Dependency bumps.** Not rot-fixing; each needs a manual render/copilot check. **Trigger:**
  unchanged.
- **`bulk_gate_threshold`'s fate.** Dead config with a live reader. Deleting it or giving a tool
  BULK is a maintainer call, not a drive-by — the guard makes either choice loud.

## Files touched

New: `shaderbox/exporters/worker.py`, `shaderbox/model_salvage.py`, and the guards
`tests/test_{worker_daemon_contract,node_dir_layout,node_save_preserves_values,model_salvage,uniform_row_pruning,retired_ids_pruned,brake_falsifiers}.py`.
Modified: `shaderbox/{app,core,integrations,paths,project_session,ui_models}.py`,
`shaderbox/exporters/{telegram,youtube}.py`, `shaderbox/copilot/tools/shader.py`,
`tests/{conftest,test_tool_registry,test_youtube_exporter,test_probe_clock_and_turn_end}.py`,
the shipped node examples + the `projects/` sandbox (hand-regenerated).

Suite: 675 -> **714 passing**, `make check` and `make smoke` green.
