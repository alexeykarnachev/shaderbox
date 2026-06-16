# 050 — Dogfood stress report (the four edit-churn/probe/turn-end fixes)

A TARGETED dogfood run (not a coverage run): every turn was engineered to provoke one of feature
050's four changed code paths in the REAL copilot engine, headless, against `gpt-5.1-codex-mini`.
The goal was to confirm the fixes hold end-to-end (not just in unit tests), and to try to BREAK them.

Run: `data-stress050` + `data-stress050b` (the hard-stop turn) — 2026-06-16 — gpt-5.1-codex-mini.
Total cost ~USD 0.038. Reference bug this wave fixes: the `gpt-5.3-codex` fire-tutorial spiral.

## Verdict

**All four findings confirmed fixed/working in the live pipeline.** The headline (Finding 1, the
comment-duplication spiral) is dead: the EXACT edit shape that produced the 1→2 duplication and the
16-edit spiral now stays at 1 copy, both on the create path AND the cleanup path. Finding 2 (churn
brake) fired both halves: the escalating soft fact made a well-behaved model self-limit, and a
threshold-lowered turn proved the hard force-end fires at the cap. Finding 3 (probe clock + tool) is a
clean win — the agent called the new `probe_render` at three chosen `t`s and reasoned about motion,
and every edit fact now stamps `t=0.0s` not wall-clock. Finding 4 (honest turn-end) is verified on the
hard-stop path: the force-ended turn told the user "I hit my own limit ... NOT a pause you asked for".

No regressions, no crashes, no GLError 1282, zero failed turns.

## Finding-by-finding evidence

### F1 — comment-duplication span-grow (the ROOT fix). CONFIRMED.
- **Create path (turns 1-3):** built a fire shader as `// Step 1/2/3` incremental `edit_shader`s, each
  `old_str` LEADING with a comment above the code (the exact bug-trigger). Final file: Step 1/2/3 each
  appear EXACTLY once. Pre-fix this same shape duplicated the leading comment on every edit.
- **Cleanup/convergence path (turn 4):** hand-injected 3 mismatched-indent `// Step 2` dupes (the
  spiral's mid-state), asked the agent to clean them. It collapsed 3→1 in a SINGLE `edit_shader`
  (1 tool call, 2 iterations) — the cleanup whose `old_str` leads with `// Step 1` and quotes all
  three dupes. Pre-fix this is the precise edit that re-added a copy and climbed to 16.
- Verified by reading the on-disk `.frag.glsl` after each turn, not just the model's claim.

### F2 — churn brake (soft escalating + hard force-end). CONFIRMED both halves.
- **Soft (turn 6):** asked for 10 separate tiny edits. The escalating per-file fact fired at edit 6
  (soft=6) with the exact text "6 clean edits to this file in a row...", and the model COMPLIED —
  stopped at 6 voluntarily: "the editor is warning that multiple rapid edits were made without you
  reviewing, so I paused there." The escalating fact growing each edit (`1 → 2 → ...`) is visible in
  the trace.
- **Hard (turn 10, soft=1/hard=3 override):** with a fresh conversation the model made edits without
  the self-imposed one-edit rut; the hard stop fired at edit 3 exactly (`WARNING: copilot clean-edit
  hard stop at 3 edits on one file`, `clean_streak_giveup: 1`), force-ending the turn at 3 not 5.
- `write_shader` reset is unit-tested; not separately provoked live (the model never chose write_shader
  mid-spree).

### F3 — probe clock t=0 + `probe_render` tool. CONFIRMED (clean win).
- `probe_render` is in the eager tool list (24 tools). Turn 5 asked "how does it look at a few moments?"
  → the agent called `probe_render` THREE times at `t=0`, `t=1.5`, `t=3.0`, got distinct correct facts
  (`render@t=1.5s: ink 88% | bbox ... y 0.02-1.00` vs `t=0.0s: ink 91% | y 0.00-0.92`), and synthesized
  a real motion comparison. The render-blind agent can now aim its probe — exactly the missing
  affordance.
- EVERY edit auto-probe stamped `render@t=0.0s` across all turns (was wall-clock `t=951s` in the
  reference bug). The clock is honest and stable.
- The probe is ungated + non-mutating: it fired silently dozens of times, never popped a gate.

### F4 — honest forced turn-end. CONFIRMED on the hard-stop path.
- The turn-10 hard-stop terminal message to the user (an `error`-role message): "[engine] I hit my own
  limit of 3 edits to one file in a turn (NOT a pause you asked for), so I stopped to keep from churning
  while you can't see the result. If more is needed, tell me to continue and I'll finish in one rewrite."
- It OWNS the stop ("I hit my own limit"), explicitly DENIES user-agency ("NOT a pause you asked for"),
  is ASCII (parenthetical replaced the em-dashes), and lists what DID apply. This is the exact fix for
  the "you're right to pause now" misattribution.
- The `max_iterations` + budget terminals share the same cause-aware nudge (unit-tested); not provoked
  live (no turn hit max_iterations this run).

## Per-render visual eyeball

- `UV_Mango_5372_2.png` (final fire shader, t=0): a vertical heat gradient — hot orange at the bottom
  fading to near-black at the top, with the horizontal turbulence/flicker visible as wavy intensity
  bands along the bottom edge. Matches the agent's description and its `render@t=0.0s: ink 91%` fact.
  The shader built through all the leading-comment edits is coherent and correct — the fix did not
  corrupt the artifact.

## Honesty / visual-blindness

No dishonesty observed. The agent's claims tracked its facts: in turn 5 it reported the per-`t` ink/bbox
numbers verbatim and drew correct conclusions; in turn 10 it listed the FLAT rgba values the auto-probe
reported. It never claimed a visual result beyond the facts. The probe_render facts are now correlatable
(t=0 default), which removes the reference bug's "facts uncorrelated with what the user sees" class.

## TODOs

### (a) improve the COPILOT / agent
- **The model develops a "one edit_shader per turn" belief mid-session** (turns 7-9): after the soft
  nudge fired once, gpt-5.1-codex-mini began refusing ALL multi-edit turns ("I can only make one
  edit_shader call per turn"), which is FALSE (the engine batches calls in order). This is a model
  misconception, but the soft-nudge wording ("make them in ONE write_shader") may reinforce it. Low
  priority (it errs toward caution), but worth watching: if a better model also over-generalizes the
  nudge into "one edit per turn", soften the wording. Not a code bug — parked, no trigger yet.

### (b) improve the DOGFOODING framework / harness / skill
- **`COPILOT_CONFIG` overrides must be set AFTER `DogfoodHarness.create()`**, not before:
  `ProjectSession.__init__` calls `integrations_store.copilot.apply_limits()`, which clobbers a
  pre-create override from the persisted integrations.json. Cost me one wasted turn (turn 10 first
  attempt silently ran with the default thresholds). **Filed into the `/dogfood` skill gotchas** so the
  next run threading a config override doesn't re-discover it.
- A cheap, well-behaved model CANNOT be naturally driven into a long clean-edit spree (it self-limits on
  the soft nudge, or invents a one-edit-per-turn rule) — so the hard-stop + max_iterations + F4 paths
  need a CONFIG override (low threshold) to provoke live. That's the sanctioned way; record it.

### (c) improve the LIBRARY
- None this run (the run touched no `SB_*` helpers — it was a change-stress run, not a library run).

## Note on tool coverage

Coverage was 2/15 reachable tools (edit_shader, probe_render) BY DESIGN — this was a change-stress run
targeting the four 050 paths, not a breadth run. The cold tools (grep/read_lib/set_uniform/switch_node/
create_node/render_image/...) are exercised by the standard scenario runs; provoking them here would
have diluted the stress signal. A future broad run should still cover them.
