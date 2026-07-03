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

## [VERIFY] Feature 053 vision — Settings badge is LIVE-only, unverified on this box

- **Trigger:** next `make run` on a machine with a display (the same session that finally UI-verifies
  052 + uploads v0.22.0). Do it before the next itch cut.
- The classifier, the look_for wire, the vision cache, and the forward-time contact sheet are all
  headless-gated in `tests/test_vision_probe.py`. NOT verifiable headless (imgui-ui §0): the Settings →
  Copilot → Vision badge RENDERING — set a known vision model → `supports vision`; a text-only id → `no
  image input`; a typo → `model not recognized`; pull the network → `couldn't verify`. Confirm the
  `status_slot` height never jitters as status changes and rapid model-field edits cause no daemon-thread
  re-kick storm. Also eyeball one real `probe_render(look_for=...)` on an animated node (the 3-frame strip
  read) against a live OpenRouter key.

## [DEBT] Unbind/rebind leaves an orphaned `media/<uniform>.*` file on disk

- **Trigger:** next time you touch the `GL_SAMPLER_2D` branch of `ui_models.py::UINode.save`, or a
  feature reads a node's `media/` dir contents (not just `node.json`).
- Feature 052's `unbind_media` resets a sampler to the default; `save` then SKIPS it
  (`is_default_image`), so `node.json` no longer references the old `media/<uniform>.<ext>` — but the
  file itself is never deleted. Load is driven purely by `node.json` (`core.py`), so the stale file is
  correctly IGNORED (no wrong re-bind) — this is a disk-cleanliness leak, not a correctness bug. Note:
  `duplicate_node` copytrees the node dir, so it carries the orphan into the fork. Fix sketch: when
  `save` skips a default sampler, unlink a pre-existing `media/<uniform>.*` for that uniform.
