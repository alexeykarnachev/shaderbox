# 033 — Copilot robustness wave (error-analysis driven)

**Status:** landed (C1-C8 + mega-review fixes, 2026-06-10 night); awaiting the verification
dogfood round. Review verdicts + triage: see `## Review round` below.

## Goal

Kill the failure classes the 032 dogfood rounds actually produced, with GENERIC mechanisms (no
per-bug overfit, no new tools — the tool count should shrink or stay, never grow). Evidence base:
3 mined experiments, ~28 compile errors + behavioral inventory (traces in
`scripts/dogfood/runs/data-{textfx,textfx2,round3}/copilot_traces/`, gitignored).

## The evidence (condensed)

| Class | Count | Exemplar |
|---|---|---|
| Line-edit range bookkeeping (`replace_lines`/`insert_after` new_text overlaps content surviving outside the range) | **19/20** compile errors in round 3 | `` `text_sdf' redeclared`` ×9; dangling tail / brace mismatch ×10; worst thrash: 7 consecutive broken iterations; recovery edits re-triggered the class twice |
| Hand-counted `uint[64]` initializers | 6 consecutive fails (exp 2) | `initializer of type uint[73] cannot be assigned to uint[64]` — then 69, 72, 65, 62, 62; zero diagnostic calls between tries; root: hardcoded const text instead of the documented `set_uniform("u_text", …)` path |
| Silent turns (user gets NO reply) | 3/10 turns in round 3 | T2: 8000 tok of hidden reasoning, zero actions; T6: 16-iteration thrash → max_iterations cutoff, no reply; exp-1: cutoff silently dropped a requirement (`u_noise_strength` never set) |
| Visual blindness (compile-clean, render broken) | recurring | spacing-blob; 5 blind re-edits oscillating 0.2→0.35→0.7→0.35→0.6; ring rendered as filled disc (`SB_glow` without `SB_op_onion` — while the SAME shader used onion correctly for an outline); noise scale 2.8 (flat) and 200 (white noise) vs the doc's "x4..x16" |
| Policy/doc bypasses | systemic | `u_wave_amplitude` tuned by SOURCE edits 6× (set_uniform rule ignored); told "check the doc" → zero `read_lib` calls; values changed beyond the ask (amplitude 0.25→0.4 smuggled) |
| Reply quality | recurring | replies read as commentary on the LAST TOOL action, not an answer to the user; prose/action mismatches ("half speed = 0.6" after setting 1.2 then 0.6 — quarter); "compile was clean" omitting 7 intermediate failures |
| Engine/harness bugs (ours) | 4 | set_uniform values LOST between harness turns (project never saved → forced re-sets burned the iteration budget); errored turns missing from `session_cost_usd`; lib rename strands nodes (resolver silently ignores unknown `SB_*` → cryptic driver error); dump echoes empty `last_render_path` |

Full inventories: the 2026-06-10 error-mining chat (3 agent reports); final-source audit found edit
sediment in every survivor shader (dead clamps, duplicate predicates, no-op guards narrated as
fixes).

## Design decisions (locked with the maintainer)

1. **One generic mechanism: enriched tool results.** Mutating shader tools append engine-computed
   FACTS to their result string. Two fact families ride the same seam:
   (a) on compile errors — structural hints: duplicate top-level declaration ("your new_text
   duplicates `text_sdf` declared at line N below the range"), brace-balance delta of the edit,
   array-initializer count ("you wrote 73 elements, need 64 — or declare `uint[]` unsized");
   (b) on clean compile — one line of render facts from a tiny probe render (e.g. 64x64): ink
   fraction, ink bbox, 3x3 luma grid. Gives the agent eyes for the
   did-my-change-take / off-screen / blank-render classes. Precedent: the slice-12 `edit_shader`
   whitespace near-miss hint — this generalizes that channel, not a per-bug patch.
2. **No new tools.** `replace_between` rejected; the `inspect_render` TOOL variant rejected
   (a lazy model wouldn't call it; tool count must not grow). The VLM-judge variant (a vision model
   critiques the render) is deferred to a HARNESS-side pilot — drive it in dogfood first, build
   into the engine only if it proves its worth.
3. **Force-checkpoint unstick.** N consecutive failed/broken edits to the same file → the engine
   restores the last clean state (030 checkpoint machinery) and tells the agent as a fact ("file
   restored to the last clean state; rewrite the whole block"). N lives in `CopilotConfig`
   (tunable), off switchable.
4. **Guaranteed final reply.** The last agent iteration runs WITHOUT tools and must produce text;
   it receives the turn ledger (what was changed/set/failed) as context. No turn may end silent
   (3/10 did in round 3). Reply framing fix in the same pass: the reply addresses the USER and
   their request — not a narration of the last tool call (maintainer-observed pattern).
5. **Reasoning off — verify, then enforce.** `openrouter.py` already sends
   `reasoning: {effort: "minimal"}`; trace T2 (8000 reasoning tokens, zero output) suggests
   codex-mini ignores it. Probe `reasoning_tokens` in usage; if ignored, escalate the param
   (`max_tokens` / `exclude` / `enabled: false` / `"low"`). Hold the `max_tokens_per_turn` bump
   unless the probe says reasoning is unavoidable.
6. **Agent-facing text compression.** System prompt + CONVENTIONS + library `///` docs: compress to
   terse keyword style, ZERO information loss (the maintainer reads the spec, the agent reads
   keywords). Less text > more text; new rules must displace old fat.
7. **Engine/harness fixes (all):** harness persists the project between turns (set_uniform values
   currently lost); errored-turn usage folds into `session_cost_usd`; the lib resolver ERRORS on an
   unknown `SB_*` identifier with a closest-name suggestion (instead of silently passing it to the
   driver); harness dump's `last_render_path` echo fixed; mission practice gains a final
   sweep-the-dead-code turn (skill edit).

## Review round (4 adversarial reviewers, triaged)

All four verdicts in: spec-fidelity, correctness, prompt info-preservation, live-behavior
(real EGL/V3D runs). Fixed from findings: resolver false-positives on non-function user `SB_*`
names (CRITICAL — call-shaped references only now, any user definition whitelisted) + 0-based
line convention + comment-safe line numbers (`strip_comments_keep_lines` in parser, shared);
`redefined` + multi-name blob support in compile hints; the V3D `0:LINE(COL):` driver format now
parses to exact lines (was a line-0 blob — `shader_errors.py`); silent-turn guarantee extended to
empty `stop`/`content_filter` finishes; cancel guards around the final reply; trace `turn_done`
ordering; cancelled-turn stats; `rsn=` in trace usage; force-restore honesty (live-program check
for the clean anchor, streak restarts when a clean file breaks, no-anchor hint, restore-with-
errors note); aspect-true probe canvas; create_node now returns hints/facts like the edit tools;
prompt restorations (current-node-in-working-set, map-content boundary + grep recipe, pack
auto-activation, Studio/button names, relative-tweak procedure); harness `last_render_path` =
newest file in renders/ (agent renders included).

Accepted deviations (documented, not bugs): force-restore uses a per-node `_last_clean` source
map instead of the 030 checkpoint store (turn-grained, wrong shape for mid-turn restore);
library `///` docs NOT compressed — round-3 evidence shows the agent succeedes BECAUSE of those
docs and the whole catalogue costs ~0.9k tok (maintainer may overrule); lib-edit streaks (errors
surface via consumer recompiles) out of scope — parked; `_broken_streak`/`_last_clean` keys
survive node deletion (uuid keys, harmless); GL-marshalled force-restore path covered by the
dogfood round rather than unit tests (repo convention for bridge paths).

## Files touched (planned)

- `shaderbox/copilot/backend.py` — enriched results (compile hints + render facts), probe render.
- `shaderbox/copilot/agent.py` — force-checkpoint unstick, reserved final no-tools iteration.
- `shaderbox/copilot/config.py` — `auto_revert_after_failed_edits` (N), probe-render knobs.
- `shaderbox/copilot/prompt.py` + `prompt_context.py` — compression pass + final-reply framing +
  text-via-set_uniform rule.
- `shaderbox/copilot/llm/openrouter.py` — reasoning param escalation (after probe).
- `shaderbox/copilot/session.py` / `state.py` — cost accounting for errored turns.
- `shaderbox/shader_lib/resolver.py` — unknown-`SB_*` resolve error + suggestion.
- `shaderbox/resources/shader_lib/**` — doc compression (regen via `scripts/gen_glyphs.py` where
  applicable).
- `scripts/dogfood/harness.py` — project save between turns, render-path echo.
- `.claude/skills/dogfood/SKILL.md` — sweep-turn practice.

## Out of scope (deferred, with triggers)

- **VLM render critique in the engine** — trigger: the harness-side pilot (driver feeds VLM
  verdicts back as user text) demonstrably catches what the scalar render facts miss.
- **`max_tokens_per_turn` bump 8k→16k** — trigger: the reasoning probe shows thinking can't be
  suppressed below ~4k for the default model.
- **Aesthetics guidance (palettes/composition)** — trigger: post-033 dogfood round shows renders
  correct-but-ugly with the maintainer wanting library-side color tooling (the MIT vendoring list
  from the 032 research).

## Manual verification

A fresh dogfood round (same mission class as round 3) after landing: expect (a) range-bookkeeping
errors recovered in ≤1 attempt thanks to hints, (b) zero silent turns, (c) the agent reacting to
render facts ("ink 0%" → self-diagnosed blank) without maintainer prompting, (d) reasoning tokens
near-zero in the trace.

## Open questions

1. Render-facts cost: probe render per mutating edit (~ms on V3D for non-text shaders, ~0.3s for
   text-stack at 64x64) — acceptable per-edit, or gate behind "compile clean AND source changed"?
2. Force-checkpoint N default (maintainer tunes): start at 4?
