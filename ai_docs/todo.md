# TODO — bugs & must-fix tech debt

ONLY bugs and obligatory technical debt — defects we are committed to fixing. **Not a feature
backlog, not a wishlist, not a quirks log.** A "nice to have", an optimization that isn't fixing a
defect, a future-feature's infrastructure, or a documented trade-off does NOT belong here — when we
reach that work we'll remember it; the durable knowledge lives in the feature spec / `conventions.md`
/ git history. If you're tempted to file a feature here "so it isn't lost", don't: delete it instead.

Each entry carries a **Trigger** — the concrete observable moment that demands picking it up. **Grep
this file by `Trigger` before starting work in an area.**

`[BUG]` = an observable defect (wrong output, crash, freeze, data loss, silent corruption).
`[DEBT]` = a structural weakness we must harden — a latent bug the current code only narrowly avoids,
or an invariant a future change can silently break.

What makes a good Trigger: it fires at a moment that *demands attention* —
- ❌ "before the next release" / "when we have time" / "eventually" — passes silently, never fires.
- ✅ "next time you edit `seed.py`" / "first user report of `<observable symptom>`" / "before
  plan-locking any feature that adds a mutating copilot tool".

If you can't name a moment that demands attention, the entry is wrong-shaped — fix it now or delete
it. When an entry resolves, delete it in the SAME commit as the fix (git history is authoritative —
no "Resolved YYYY-MM-DD" headers).

<!-- Shape (per entry):
     ## [BUG|DEBT] <short title>
     - **Trigger:** <a concrete observable moment — file/code touch, count threshold, user complaint with a measurable surface>
     - <context: what / why / where>
     State the CURRENT constraint in present tense, not a feature roll-call. No designs here — the fix
     sketch is one line; a real design lives in the feature spec / conventions.
-->

---

## [BUG] "Reset library to shipped" never removes a live-root file that left the shipped set

- **Trigger:** next time you touch `shader_lib/seed.py::reset_to_shipped`, or a user reports "reset
  did nothing / stale lib helpers persist after a factory reset".
- `reset_to_shipped` restores/overwrites every SHIPPED file but has no removal pass for a live-root
  `.glsl` that is NOT in the current shipped set — so a helper that moved (e.g. flat `noise.glsl` →
  `noise/fbm.glsl` when the toolbox shipped) leaves the OLD file lingering, giving DUPLICATE `SB_*`
  definitions and a reset that reports "0 restored" while the user still sees the stale lib.
  `sync_shipped_lib` already does manifest-guarded stale-removal (pristine-only); `reset_to_shipped`
  should mirror it. Only bit the maintainer's dev live-root (hand-authored pre-subdir flat files —
  cleaned by hand 2026-07-17); a fresh user install seeds only subdirs so it can't hit this yet, but
  any future shipped rename re-arms it. Fix sketch: after the restore loop, delete a live-root file
  absent from `seed` whose hash matches its manifest entry (never an edited/user-authored one).

## [VERIFY] Features 053+056 vision/copilot — LIVE-only UI checks, unverified on this box

- **Trigger:** next `make run` on a machine with a display. Do it before the next itch cut.
- The classifier, the look_for wire, the vision cache, and the forward-time contact sheet are all
  headless-gated in `tests/test_vision_probe.py`. NOT verifiable headless (imgui-ui §0): the Settings →
  Copilot → Vision badge RENDERING — set a known vision model → `supports vision`; a text-only id → `no
  image input`; a typo → `model not recognized`; pull the network → `couldn't verify`. Confirm the
  `status_slot` height never jitters as status changes and rapid model-field edits cause no daemon-thread
  re-kick storm. Also eyeball one real `probe_render(look_for=...)` on an animated node (the 3-frame strip
  read) against a live OpenRouter key.
- Feature 056 additions in the same run: the chat shows the attributed engine-look line ("the engine
  checked the render against your ask") only on turns where a vision read happened; a deflected
  publish (no credentials) renders a neutral "handed off" line, not a red failure; the Settings →
  Copilot row for `copilot_convergence_max_looks` renders and persists.

## [DEBT] Unbind/rebind leaves an orphaned `media/<uniform>.*` file on disk

- **Trigger:** next time you touch the `GL_SAMPLER_2D` branch of `ui_models.py::UINode.save`, or a
  feature reads a node's `media/` dir contents (not just `node.json`).
- Feature 052's `unbind_media` resets a sampler to the default; `save` then SKIPS it
  (`is_default_image`), so `node.json` no longer references the old `media/<uniform>.<ext>` — but the
  file itself is never deleted. Load is driven purely by `node.json` (`core.py`), so the stale file is
  correctly IGNORED (no wrong re-bind) — this is a disk-cleanliness leak, not a correctness bug. Note:
  `duplicate_node` copytrees the node dir, so it carries the orphan into the fork. Fix sketch: when
  `save` skips a default sampler, unlink a pre-existing `media/<uniform>.*` for that uniform.

## [DEBT] A killed harness turn loses the conversation while its disk edits persist

- **Trigger:** next time you touch `scripts/dogfood/harness.py`'s turn lifecycle, or a dogfood run
  resumes into a "the model doesn't remember its own visible edits" state.
- The one-process-per-turn harness `dump()`s (and persists the conversation) only AFTER
  `drive_until_idle` returns; an external kill (timeout, Ctrl-C) loses the in-flight turn's
  conversation while every tool edit already landed on disk — the next resume is half-restored
  (nodes changed, history unaware). The engine-side `turn_time_budget_s` (180s) makes external
  kills rare, but the window remains. Fix sketch: a SIGTERM handler (or incremental persist at
  each bridge drain) that saves the conversation before exit.

## [DEBT] No liveness indicator during hidden-reasoning bursts in the chat

- **Trigger:** first user complaint shaped "the copilot froze" during a turn that was actually a
  long reasoning burst; or the next feature touching `widgets/copilot_chat.py`'s streaming state.
- A reasoning-heavy iteration streams no visible deltas for up to ~a minute; the chat shows
  nothing alive, indistinguishable from a hang. The engine now bounds the turn
  (`turn_time_budget_s`), but the UI should still show "thinking… Ns" (the worker knows a stream
  is open with zero visible deltas). UI-only; no engine change.

## [BUG] full pytest suite corrupts the X connection mid-run (pre-existing, not 051)

- **Trigger:** any full `uv run pytest` on the dev box — reproduces today, every run.
- The app-fixture window churn plus the `pytest.mark.forked` GL module (`test_revert_executor`)
  corrupt the shared X connection on the WM-less `:1` display: the forked tests error
  ("Cannot detect window with OpenGL support"), and later the process dies with
  `XIO: fatal IO error` before printing a summary. Verified independent of feature 051 — the
  pre-051 tree crashes identically; every module passes when run alone. Fix sketch: process-isolate
  ALL app-fixture modules (forked/xdist) or run the suite under a dedicated Xvfb.
