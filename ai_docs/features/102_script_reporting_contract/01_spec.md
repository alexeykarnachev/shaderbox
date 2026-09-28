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

- **D1. A script key that does not land warns — one code path, with two stated exemptions.**
  A key naming a uniform no pass declares, a key naming a pass that does not exist,
  `@instances` reaching a pass that HAS compiled and declares no `flat in`, a sampler/block
  key, and any sibling not yet enumerated are the SAME case and must not be distinguishable
  in the code.

  **Two cases are exempt, and they are exempt because their correct behaviour is provably
  not a warning — not because they are special.** A review found both already in the repo,
  which means D1 cannot be stated without them:
  - **NOT-YET-COMPILED is not a failure to land.** `engine.py:948-952` holds a key for a
    pass that has never attempted a compile, documented in code as "HELD for this tick, no
    error, nothing written. The next tick recomputes." On frame one of every document every
    pass is uncompiled, so warning here fires on every document open. The same state is the
    second half of `engine.py:835-843`'s `if not fields:` — the comment there names both
    reasons ("No `flat in` yet, or the pass has not compiled") and only the FIRST is I1.
    **The wave must split that branch**; today it cannot tell them apart.
  - **ENGINE-OWNED keys are dropped silently** (`engine.py:892`, `:954`), pinned by
    `tests/test_script_engine.py::test_engine_owned_key_dropped_silently`. The engine owns
    the slot and a script cannot be expected to avoid naming it. That test does NOT invert.

  **Revisit if** a THIRD such case appears — at which point "warn unless the engine owns it
  or the pass has not compiled" is the wrong rule shape and wants re-deriving, rather than a
  third exemption.
- **D1a. The single path returns a named REASON, and that enum is the gate's domain.** There
  is no key-failure taxonomy in the code today — the kinds live in `_tick_script`'s control
  flow across seventeen sites, not in an enum, a `Literal` or a dict. So a gate "enumerated
  from the code's own taxonomy" cannot be written until the taxonomy exists. **Building it is
  part of this wave and comes first**: one enum of reasons, `get_args`-enumerable the way
  `TargetDtype`/`TARGET_DTYPES` already is (`pass_graph.py:44-45`). Without it the gate falls
  back to a hand-written list, which is the exact domain-narrowing this spec's gates section
  forbids.
- **D2. This reverses 079 D5's silence for every script key, and RELAXES the `@` rule.** The
  old silence ("writing the script before the shader is normal authoring, so stay quiet") is
  what hid I1 and what let two rules disagree. The warning is the answer for the mid-edit
  case too. Note the direction on each side, because they differ: for a plain key this is
  silence -> warning, and **for an `@` key it is HARD ERROR -> warning** (the
  reserved-vocabulary rule made an unrecognised `@` key fatal). A relaxation needs saying out
  loud; the argument for it is that one rule beats two, and that a warning still reaches the
  author, which the silence did not.
  **Revisit if** D3's measured event rate exceeds one warning per key per edit-burst in
  ordinary authoring — the threshold, not a feeling. Until D3 has measured, this decision has
  no fireable trigger and is provisional on that number.
- **D3. The warning is EDGE-triggered on the key's state changing, never level-triggered.**
  This path runs at frame rate; a warning per frame per key is a notification storm while an
  author types. "State" is the REASON from D1a, not a boolean — a key oscillating between two
  failure modes must warn on each transition, and a key stuck in one must warn once.
  **This is unmeasured and is the wave's first risk** — measure the event rate before
  building the surface.

  **And the edge trigger does not cover the bigger surface.** Every newly-warning key also
  becomes a soft error in `ScriptStatus.soft_errors` (`engine.py:333-350`), which
  `tabs/code.py:208-235` renders as an error-strip row on the script tab AND on each named
  pass's shader tab. That strip is LEVEL-triggered: it shows what is currently true, so an
  author mid-edit gets a standing wall of rows that no edge trigger suppresses. Decide the
  strip's behaviour in this wave — it is the usability risk, and the notification storm is
  the smaller half.
- **D3a. The warning leaves the engine through an injected callback, not by importing the UI.**
  `notifications.py` imports `imgui_bundle`; the script engine is headless core and
  `conventions.md` states it imports no imgui/glfw. The project's established seam for this is
  an injected `on_*` callback through `ProjectSession`. Naming it here so the builder does not
  discover the boundary by violating it.
- **D3b. `dry_run` must not inherit the edge-trigger state.** The probe re-ticks the same
  script for N frames (`engine.py:591-611`). If the edge-trigger state lives on the live
  `DocumentScripts`, a probe mutates state the 063 ruling promises is left byte-identical; if
  it lives per-call it must be threaded through. Pick one and gate it — the existing isolation
  gate covers `pending_instances`, not this, and this is what the wave newly adds.
- **D4. The outcome has THREE producers, not one, and is a named public seam.** The five
  reporting defects share one cause: the draw judges the population and can only return, so
  nothing reads a verdict and `_instances_error` has nowhere to go. But a review found the
  eight reachable states are not all decided in the same place, so a return value from
  `Pass.render` alone can express at most five of them:

  | producer | states it decides |
  |---|---|
  | `Pass.render` / `_upload_instances` | drew-N, fullscreen, empty, refused(why) |
  | `Pass.compile` | compile-failed (state f), drawing-stale-after-failed-recompile (state g) |
  | `scripting/engine.py` | population-to-a-pass-with-no-fields (state h) |

  So the outcome is a TYPE the three producers write, not a return type on one function.
  `refused` carries its reason, which is what keeps states (c) and (d) distinguishable —
  the research separated them deliberately and a single bucket loses that.

  **It is PUBLIC.** Three consumers read it: the surface D4a names, 103's `probe_render`
  (C9), and 104's per-pass live count (104 D3). An internal return value would have to be
  re-exposed in each. Its shape is therefore part of this wave's deliverable, and 104 D3's
  requirement — that a per-frame UI surface can read a live instance count off it — is a
  constraint on the shape, pulled forward here rather than discovered in the last wave.

  **The honesty test:** the sibling call sites must be shown UNCHANGED except for the fix. A
  review confirmed this is cheap to satisfy — `Pass.render` has exactly ONE production caller
  (`document.py:1002`), and the rest are tests. If a trace or a reviewer cannot show it, the
  generalisation was speculation and the point fix is correct instead.

  **Revisit if** a second draw shape lands, or if the outcome must carry per-iteration rather
  than per-frame state (see D4b).
- **D4a. Name the surface.** D4 is worthless until one is chosen: the research MEASURED that
  no surface in the app has any instancing vocabulary — not the error strip, not the uniform
  panel, not the graph, not the profiler, not logging. `Document._graph_errors`
  (`document.py:951`) is an existing per-frame diagnostic bag the UI already reads and is the
  shape to reuse. I2 and I3 are not closed until this is decided.
- **D4b. An iterated pass produces N outcomes per frame.** `document.py:998-1011` calls
  `render` once per iteration. Decide whether the last wins, whether they collect, or whether
  only iteration 0 reports — a "per-frame outcome" is ambiguous for `iterations > 1` and the
  builder would otherwise guess.
- **D5. Blend mode is per-pass, with the options people use.** Additive, alpha, opaque and
  the usual others, living beside `dtype` on the pass's target configuration, persisted, and
  surfaced in the graph AND the pass panel AND anywhere else a pass's configuration shows.
  **The vocabulary is decided HERE, not by the builder**: it becomes permanent on-disk
  vocabulary the moment it ships, and this project writes no migrations. Declare it as a
  `Literal` beside `TargetDtype` with a `get_args` tuple, so it is enumerable and a gate can
  walk it.

  **This reverses feature 100's decision 7, which was THREE claims — "Additive, no depth, no
  per-frame sort" — and only the first is reversed.** The other two were free BECAUSE additive
  is order-independent. Under alpha blending it is not: overlapping alpha sprites are
  order-dependent, so "no per-frame sort" stops being a free consequence and becomes a
  standing limitation of the alpha mode. **State that limitation where an author meets the
  mode**, rather than leaving 100's decision half-standing.

  **Three consequences a review found, none obvious:**
  - **A blend-only change would silently wipe a feedback pass's trail, by TWO paths.**
    `Document.set_pass_target` returns early only when `render_pass.target == target`
    (`document.py:672`) and otherwise calls `drop_feedback` unconditionally; separately
    `Pass.set_target` reallocates on any inequality (`core.py:287-299`) and bumps
    `target_generation`, which `_feedback_canvas` reads at `document.py:797` to drop the
    history a second way. **Both equality checks must compare only the allocation-relevant
    fields**, or a mode the user picks from a combo destroys their trail.
  - **Blend is applied only on the instanced path today** — a fullscreen pass returns at
    `core.py:634-636` before the blend block. Decide whether the per-pass control applies to
    fullscreen passes too (a behaviour change to every existing document) or whether the
    control is hidden for them. A control that shows and does nothing is the worse option.
  - **"Surfaced in the graph" is new capability, not a new field.** The graph node
    (`graph_canvas/adapter.py`, `pass_node`) carries no target properties at all — not dtype,
    not scale, not iterations. Size it accordingly.

  **Revisit if** a mode needs per-frame switching, which the per-pass model cannot express.
- **D5a. `conventions.md` gets a blend bullet in this wave.** It currently contains NO blend
  text — the `@instances` bullet carries the instancing record and says nothing about
  `ONE, ONE`. A new persisted per-pass control with no entry in the settled-decisions file is
  a decision that exists only in a feature spec.
- **D5b. The `graph.json` field is hand-fixed, never migrated.** A defaulted field round-trips
  and back-loads with no reader change, which is exactly the shape that tempts an implementer
  into a compat path. `projects/dev/` and the seven shipped
  `resources/document_examples/*/graph.json` are hand-edited (or regenerated through the
  normal load+save path) and `git add`-ed in the same wave. This is the project's only
  sanctioned fix.
- **D6. `_instances_error` is deleted, not fixed.** It is dead state that also goes stale
  (the refused branch returns at `core.py:661` before either assignment, so an earlier
  frame's message survives a later refusal). D4 replaces it; keeping both would be two
  spellings of one fact. **No revisit trigger, deliberately** — a deletion has no condition
  under which it returns. Gated by a grep for any reader after the delete, which is the
  cheapest defined-but-not-wired check available.

## What this wave fixes

| id | Defect | Closed by |
|---|---|---|
| I1 | population to a pass with no `flat in` — silent forever | D1 |
| I2 | no population — draws fullscreen, no signal | D4 |
| I3 | zero entities — clears to black, no signal | D4 |
| I4 | `_instances_error` written and read by nothing, goes stale | D6 |
| I5 | failed recompile keeps old fields and old program | D4 makes it VISIBLE; the stale fields survive whatever `render` reports, so closing it is a separate decision this wave must take |
| I6 | additive hardcoded; opaque sprite doubles, dark-on-light impossible | D5 |
| I7 | `f1` + additive saturates; 6 of 7 examples use `f1` | D5, PROBABLY — the research hedged and this spec must not un-hedge it. Gated: an `f1` target under each blend mode does not saturate. If the gate says otherwise, I7 is live and needs its own fix |

## Blast radius — this is NOT an instancing change

The full argument for why this reaches past instancing lives in
`101_instancing_research/01_spec.md ## What D-F costs beyond instancing`; what follows is
the enumerated work, MEASURED by grep rather than estimated.

**Nineteen `079 D5` sites across six files**, plus `ScriptProbe`'s class docstring:

| file | sites |
|---|---|
| `shaderbox/scripting/engine.py` | 6 (`:416`, `:616`, `:653`, `:808`, `:901`, `:969`) |
| `shaderbox/copilot/capabilities.py` | 1 (`:283`, `ScriptWriteResult`'s docstring) |
| `ai_docs/conventions.md` | 2 (`:616` the `@instances` bullet, `:664` the script-engine bullet) |
| `tests/test_script_engine.py` | 7 |
| `tests/test_script_dry_run.py` | 2 |
| `tests/test_instances_routing.py` | 1 (the module docstring) |

An earlier draft of this spec named two conventions bullets and two test files and called it
"the largest doc edit in the wave" — it undercounted by more than half, and
`tests/test_script_dry_run.py` was missing entirely.

- **`orphan_keys` does not change shape; it goes EMPTY.** `engine.py:622-626` builds it as
  exactly "seen_skipped with no error". Once every non-landing key carries an error, nothing
  qualifies. That invalidates three things THIS WAVE owns and must fix — `ScriptProbe`'s
  class docstring ("since 079 D5 is a normal authoring state and carries no error"),
  `ScriptWriteResult`'s docstring, and `tools/script.py`'s orphan rendering, which presents
  the case as a hint rather than a warning. An earlier draft said "coordinate with 103"; 103
  does not claim it, and a handoff to a spec that does not catch it is no handoff.
- **Tests pin the silence and therefore INVERT** — each becomes a place the new rule must be
  SEEN to fire, which is where the behaviour gets gated. **One does NOT invert:**
  `test_engine_owned_key_dropped_silently` guards D1's engine-owned exemption and must stay
  green unchanged.
- **`ScriptStatus.soft_errors` and the error strip** — see D3, the level-triggered surface.

## Gates

Each must be **broken and watched to fail** before it is believed, and the commit says which
break was tried.

- **Every non-landing key REASON produces a warning**, enumerated by `get_args` over D1a's
  enum — which is why D1a comes first. One case per member, each reintroducing the silence
  and watching the gate fail. The two D1 exemptions are asserted as exemptions, so a later
  widening has to argue with a test rather than with a paragraph.
- **The warning is edge-triggered**: tick a persistently failing script N times, assert the
  warn count is 1. Break it by making it level-triggered and watch the count reach N.
- **Each producer reports the states it decides** (D4's table): `Pass.render` for four,
  `Pass.compile` for two, the engine for one. Do NOT write one gate claiming all eight — a
  gate on `render`'s return can observe five at most, and would narrow its own domain exactly
  as the first gate forbids. States (b) and (e) are the pair worth most care: both leave the
  strip empty today and they are different pictures.
- **`_instances_error` has no reader after D6** — grep, and break it by reintroducing one.
- **Blend mode round-trips through `graph.json` and each mode is distinguishable in a rendered
  frame.** The fixture needs TWO OVERLAPPING entities, because over the unconditional black
  clear a single non-overlapping sprite renders identically under every mode — the operation
  is only observable in the overlap. Opaque overlap must NOT double, which is the measurement
  that made I6 real. Also extend `tests/test_graph_persistence.py`'s round-trip test, which
  enumerates fields BY HAND and would pass with the new field at its default.
- **An `f1` target does not saturate under each blend mode** — the gate I7's "dissolved"
  claim rests on and which an earlier draft dropped because of that very claim.
- **`dry_run` does not mutate the edge-trigger state** (D3b). NOT the `pending_instances`
  isolation gate — that one already exists at `tests/test_instances_routing.py:357-391`,
  already passes, and was verified in review by breaking it. It is a regression guard this
  wave inherits, not work it does.

## Files touched

Named because D4's honesty test requires showing the sibling call sites unchanged, which is
impossible against a spec that does not say what they are.

- `shaderbox/core.py` — `Pass.render`, `_upload_instances`, `compile`, `set_target`; delete
  `_instances_error`.
- `shaderbox/scripting/engine.py` — the D1a reason enum, the single warn path, the
  `if not fields:` split, `ScriptProbe`'s docstring, six `079 D5` comment sites.
- `shaderbox/pass_graph.py` — the blend `Literal` + `get_args` tuple on `TargetConfig`.
- `shaderbox/document.py` — the one production `Pass.render` call site (`:1002`), the
  per-iteration question (D4b), the outcome's surface (D4a, likely `_graph_errors`).
- `shaderbox/project_session.py` — the injected warn callback (D3a).
- `shaderbox/popups/pass_settings.py` — the blend control.
- `shaderbox/copilot/capabilities.py`, `shaderbox/copilot/tools/script.py` — the
  `orphan_keys` tier change this wave owns.
- `ai_docs/conventions.md` — two `079 D5` bullets rewritten, one blend bullet added (D5a).
- `projects/dev/**/graph.json` and `shaderbox/resources/document_examples/*/graph.json` —
  hand-fixed per D5b.
- Tests: `test_script_engine.py`, `test_script_dry_run.py`, `test_instances_routing.py`,
  `test_graph_persistence.py`.

## Out of scope

- Off-thread simulation and snapshot interpolation. **Trigger:** the maintainer asks, or the
  inline tick becomes the thing in the way.
- Copilot work (103) and discoverability (104), both of which gate on this.

## Open questions for the user

None blocking. D-A…D-G are settled in the research spec, and the decisions above close what
the review found open. Two are provisional on a measurement this wave takes rather than on a
maintainer answer: D2's trigger needs D3's event rate, and I7's "dissolved" needs its gate.
