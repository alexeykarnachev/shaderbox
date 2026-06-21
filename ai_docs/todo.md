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


_No open bugs or must-fix debt. (Resolved entries are deleted in their fixing commit — git log is the history.)_
