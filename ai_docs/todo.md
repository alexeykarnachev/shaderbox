# TODO — bugs & must-fix tech debt

**FROZEN (maintainer call, 2026-07-27): DRAIN-ONLY.** No new entries, ever — a new defect gets
FIXED in the wave that finds it, or its knowledge goes to the feature spec / `conventions.md`.
This file only shrinks; the goal is zero entries.

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
- Liveness counter (todo-drain 2026-07-27): during a quiet stream stretch the live status line
  grows a ticking "waiting Ns" suffix after ~3s of silence, and it disappears when deltas resume.
- Dev-box only, same session: run `make test` (now MESA-override + one-context-recipe; the
  WSL-verified fix for the "GL modules segfault / every module passes alone" class — the explicit-
  EGL release in `test_script_engine_gl` poisoned the process's EGL display, fixed 2026-07-27,
  622/622 headless). If the dev box's full run STILL dies with `XIO: fatal IO error` on `:1`, the
  shared-X window-churn half is real there — then per-run `xvfb-run` on that box is the next move.
