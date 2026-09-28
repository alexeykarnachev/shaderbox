# 102 — The script-reporting contract

**The engine must never know something is wrong and say nothing.** Every defect here is an
instance of that, and the fix is two shared seams rather than seven patches.

Research, measurements and the full evidence: `ai_docs/features/101_instancing_research/01_spec.md`
(Part 1, items I1–I7). This spec carries the decisions and the plan; it does not repeat the
citations.

## Goal

Two mechanisms, both at the shared root:

1. **One rule for a script key that does not land, and it WARNS** (D-F). No branch per case.
2. **`Pass.render` returns what happened, and one place reads it** (D-G).

Everything below follows from those two.

## Design decisions

- **D1. A script key that does not land warns — to the logs AND the notifications.** One
  code path. A key naming a uniform no pass declares, a key naming a pass that does not
  exist, `@instances` reaching a pass with no `flat in`, and any sibling not yet enumerated
  are the SAME case and must not be distinguishable in the code. **Revisit if** a case
  appears whose correct behaviour is provably not a warning — in which case the rule is
  wrong, not the case, and the rule changes once for all of them.
- **D2. This reverses 079 D5's silence for every script key.** The old rule ("writing the
  script before the shader is normal authoring, so stay quiet") is what hid I1 and what let
  two rules disagree. The warning is the answer for the mid-edit case too. **Revisit if**
  the warning proves unusable in practice after D3 — not before.
- **D3. The warning is EDGE-triggered on the key's state changing, never level-triggered.**
  This path runs at frame rate; a warning per frame per key is a notification storm while an
  author types. A key that stops landing warns once; it warns again when it changes state.
  **This is unmeasured and is the wave's first risk** — measure the event rate before
  building the surface.
- **D4. `Pass.render` returns a per-frame outcome.** The five reporting defects share one
  cause: the draw judges the population and can only return, so nothing reads a verdict and
  `_instances_error` has nowhere to go. The outcome carries what happened —
  drew-N / fullscreen / empty / refused-why / not-instanced — and one surface reads it.
  **The honesty test:** the sibling call sites must be shown UNCHANGED except for the fix.
  If a trace or a reviewer cannot show that, the generalisation was speculation and the
  point fix is correct instead.
- **D5. Blend mode is per-pass, with the options people use.** Additive, alpha, opaque and
  the usual others, living beside `dtype` on the pass's target configuration, persisted, and
  surfaced in the graph AND the pass panel AND anywhere else a pass's configuration shows.
  Reverses feature 100 D7, whose rationale covers the glowing case and is silent about the
  rest of the range. **Revisit if** a mode needs per-frame switching, which the current
  per-pass model cannot express.
- **D6. `_instances_error` is deleted, not fixed.** It is dead state that also goes stale.
  D4 replaces it; keeping both would be two spellings of one fact.

## What this wave fixes

| id | Defect | Closed by |
|---|---|---|
| I1 | population to a pass with no `flat in` — silent forever | D1 |
| I2 | no population — draws fullscreen, no signal | D4 |
| I3 | zero entities — clears to black, no signal | D4 |
| I4 | `_instances_error` written and read by nothing, goes stale | D6 |
| I5 | failed recompile keeps old fields and old program | D4 |
| I6 | additive hardcoded; opaque sprite doubles, dark-on-light impossible | D5 |
| I7 | `f1` + additive saturates; 6 of 7 examples use `f1` | D5 (dissolved) |

## Blast radius — this is NOT an instancing change

D1/D2 touch every script key, so the wave reaches well past instancing:

- **The `skipped` / `driven` / `orphan_keys` split** was built around silence being a normal
  outcome. With every non-landing key warning, `orphan_keys` stops being a quiet category,
  and the copilot's `ScriptProbe` (which already reports orphans) changes shape with it —
  coordinate with 103, which consumes it.
- **`conventions.md ## Design decisions`** carries 079 D5 in the script-engine bullet and the
  reserved-vocabulary rule in the `@instances` bullet. Both are rewritten to the single rule
  IN THIS WAVE, or the docs state behaviour the code no longer has.
- **Tests pin the silence and therefore INVERT.** `tests/test_script_engine.py` and
  `tests/test_instances_routing.py` assert quiet outcomes for orphan keys — the routing
  file's own docstring calls it "deliberately SILENT (079 D5)". Each such assertion becomes
  a place the new rule must be SEEN to fire, which is where the new behaviour gets gated.

## Gates

Each must be **broken and watched to fail** before it is believed, and the commit says which
break was tried.

- Every non-landing key kind produces a warning: one test per kind, each reintroducing the
  silence and watching the gate fail. The kinds are enumerated FROM the code's own key
  taxonomy, not from a list written here — a checker that narrows its own domain is the
  failure mode this wave is most exposed to.
- The warning is edge-triggered: a key that fails to land for N frames warns once. Break it
  by making it level-triggered and watch the count exceed one.
- Each of the eight draw states reports its own outcome and no other. States (b) and (e) are
  the pair worth care — both leave the strip empty today and they are different pictures.
- A `dry_run` over an instanced document leaves `pending_instances` untouched (the 063
  isolation property, which 103's reporting path could silently break).
- Blend mode round-trips through `graph.json` and each mode is distinguishable in a rendered
  frame — opaque overlap must NOT double, which is the measurement that made I6 real.

## Out of scope

- Off-thread simulation and snapshot interpolation. **Trigger:** the maintainer asks, or the
  inline tick becomes the thing in the way.
- Copilot work (103) and discoverability (104), both of which gate on this.

## Open questions

None. D-A…D-G are settled in the research spec.
