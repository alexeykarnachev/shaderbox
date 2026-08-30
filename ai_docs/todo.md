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
`[VERIFY]` = a shipped change whose check cannot run on this box (needs a display, real secrets, other
hardware) — it stays until someone runs it on a machine that can.

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

## [VERIFY] The 064 Steps panel, unseen on a display

- **Trigger:** next `make run` on a machine with a display. Do it before the next itch cut.
- The Steps section (`widgets/step_list.py`) was verified by driving the real app loop headless --
  it draws, a float step pins and tonemaps, a node switch clears the pin, a stale pin self-heals --
  but layout and aesthetics cannot be judged without a display and this box has no WM.
- Open the shipped "Render Steps" example and look at: whether a 56px thumbnail beside two dim
  caption lines reads as a row or as clutter at four steps; whether the two captions
  (`480x360 . f2 . linear` and `reads: sparks`) are worth two lines or want one; whether the
  selected row's accent border is legible against the panel; and whether the pinned-step preview
  needs a visible marker beyond the row's border, since a pinned intermediate can otherwise be
  misread as a broken shader.

## [VERIFY] Copilot live-only UI checks, unverified on this box

- **Trigger:** next `make run` on a machine with a display. Do it before the next itch cut.
- NOT verifiable headless (imgui-ui §0), all small: a deflected publish (no credentials) renders a
  neutral "handed off" line, not a red failure; the Settings → Copilot rows render after the 058
  vision-block removal (no orphaned gap; the "Turn time budget (s)" row present and persisting);
  the liveness counter — during a quiet stream stretch the live status line grows a ticking
  "waiting Ns" suffix after ~3s of silence and it disappears when deltas resume.
