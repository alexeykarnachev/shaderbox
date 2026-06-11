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

Post-fix verification (focused live round): all six checks PASS on the real V3D engine —
resolver clean on defines/consts/user-fns with exact agent-visible lines, `redefined` + blob
hints fire with correct lines, V3D error format parses per-line, aspect-true probe verified
load-bearing, suite + make check green, cutoff region single-terminal. Known minor gaps (info,
pre-existing): the analyzer's per-turn rollup keys on `turn_done` so truncated turns lack
cost rows; the max-iterations tail traces `turn_done` even when the final reply falls back to
an error event.

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

## Review cycle 2 (post-round-4) — triage

Anti-overfit audit confirmed the maintainer's worry at the TRIGGER level (architecture generic,
surfaces Pi/Mesa-locked). Fixed generically: redeclare hints now match the closed vendor wording
set (Mesa/glslang/NVIDIA/AMD); the initializer hint counts elements in the SOURCE (driver-
agnostic — count-less vendor messages still get numbers); EMPTY verdict replaced by self-
describing FLAT (color + max deviation — a fill is no longer reported as a blank, sub-threshold
changes are visible in the deviation); the qualifier set is now the full spec-closed list +
layout(...); brace hints локализуют (first unmatched brace / unclosed opener line); NEW
oscillation brake (per-node source-state hashes — an A->B->A edit gets a NOTE; round-4's 16-edit
clean flip-flop class); final-reply nudge demands NET state, no intentions-as-done; trust-the-
user rule is two-sided (facts contradicting a report -> say and ask); prompt teaches the
`render@t=` format. Corrected diagnosis: round-4 "phase drift" was wrong — harness probes are
time-frozen at t=0 (glfw uninitialized), the fact changes were genuine edit effects; the t-stamp
matters only in the live App. Accepted (documented): nudge-before-restore ordering is best-effort
(comment fixed); restore vs manual user edits / mid-batch edges (rollback snapshot bounds them).

## Manual verification — ROUND 4 RESULTS (countdown mission, 6 turns, $0.12)

(a) Range-error recovery with hints: PARTIAL-GOOD — the orphan-tail class reproduced live and
converged in 3 hinted attempts with the brace hint quantifying each step (round 3: 6-7 blind);
one `redeclared` case (`out vec4 fs_color`) got no hint — the qualifier gap fixed post-round.
(b) Zero silent turns: PASS — 6/6 turns ended with a reply, including a 16-iteration thrash turn
(round-3 analog ended silent). (c) Facts reactions: MIXED — facts present on every mutation
(create_node included) but orientation is invisible to them, and animation PHASE made facts drift
between edits (mistaken for edit effects — fixed post-round: facts now stamp the sample time).
(d) Reasoning: observability VERIFIED — creative iterations burn ~6.8k rsn even at effort=minimal
(routine edits 100-1000); per the spec's out-clause `max_tokens_per_turn` bumped 8k->12k.
Cost accounting verified live: dump session_cost == trace total to the cent.
DRIVER LESSON: the round's "upside-down" saga was a maintainer-side misread of the segment font
(arched П reads as U) — judge calibration matters, for humans and future VLM judges alike; the
agent cannot push back on a false visual report (it trusts the user per prompt).

## Manual verification — ROUND 5 RESULTS (library-surgery mission, 4 turns)

Scenario varied per the convergence loop: lib-file surgery instead of node editing. T1 extract a
ring-progress helper into `lib:draw/progress_arc.glsl` + refactor Countdown onto it: PASS. T2
create a Loader node + RENAME the helper to `SB_sd_arc_sweep`: PASS — rename consistent across
lib file + both consumers (grep-verified, zero stale references). T3 signature change (5th param
`start_angle`) broke both consumers; the final reply honestly reported the broken NET state
(cycle-2 nudge working as designed). T4 "continue": the cycle-2 location-enhanced brace hint
("first unmatched '}' on line 34") -> fixed in 1 attempt (round-3 analog: 6-7 blind). End state
both nodes clean, renders correct (Countdown digit + ring from 12 o'clock, Loader arc from 3
o'clock; verified visually pre-reboot, PNG artifacts lost to a /tmp wipe).
NEW findings for cycle 3: (a) `read_shader` on a `lib:` address returns "no such node(s)" — no
redirect to the lib-read path; (b) turn-3 prose claimed "Loader now uses the new signature"
while the file was NOT yet updated — prose/state mismatch class recurs (round 4 had an
intentions-as-done analog; the nudge fixed the FINAL reply, mid-turn prose still drifts).
OPS NOTE: round sandboxes lived in /tmp (harness mkdtemp default) — two Pi reboots wiped traces
mid-loop; future rounds must set SHADERBOX_DATA_DIR under $HOME.

## Review cycle 3 (post-round-5) — triage

Three reviewers (runtime-finding verification PASS / anti-overfit + convergence CONTINUE /
code regression sweep PARTIAL: 0 critical, 2 medium, 4 low). All cycle-2 fixes audited generic —
zero overfits. Fixed: (1) `read_shader` resolves `lib:` addresses into the working set — the
grep-origin docs already advertised them as read handles, the read side now honors the same
address space as edit_shader (tool count unchanged); the not-found error names the lib: path
too. (2) Mid-turn prose/state mismatch (round-4 intentions-as-done class at the unfixed sibling
surface — mid-turn text IS user-visible via streaming): one prompt rule — text alongside tool
calls is a PLAN, present/future tense. (3) render facts were ALPHA-BLIND — the flagship sticker
pattern (white glyph on transparency) reported as "FLAT white fill"; facts are now rgba (alpha
in ink test + verdict, alpha-weighted luma). (4) A torn no-tools final stream propagated out of
run_turn, dropping the turn's WHOLE cost accounting + the irreversible-action ledger: now caught,
the terminal AgentError keeps real summary + stats. (5) Oscillation brake had ZERO tests
(checklist premise was false) + three gaps: now seeded with the pre-edit state (the first
A->B->A is caught), a no-op edit never fires it, distance uses the NEAREST occurrence, and lib
edits get the same brake (round-5's mission shape was lib surgery — node-only was the missing
half). (6) Initializer count starts at the MATCHED paren, not the line's first (an earlier call
on the line yielded a false count). (7) Declaration-hint regex excludes statement keywords
(`return weight;` listed as a declaration). (8) Resolver whitelists SB_ function PROTOTYPES
(legal GLSL, was a compile-blocking false positive; call statements still flagged). +11 unit
tests. Accepted (documented): V3D `warning:` lines become ShaderErrors inside a failed compile —
consistent with the pre-existing `_MESA_ERROR_RE` design.
CONVERGENCE: CONTINUE (fixes above unvalidated by a fresh round; two mechanisms have zero
behavioral evidence). Round 6 = visual-tuning mission with a deliberately FALSE driver report
("still mirrored/too dim" against contradicting facts) + uniform-heavy relative tweaks — targets
two-sided trust and the oscillation brake at once. Declarable converged after it if: no terminal
failure, no NEW class, agent pushes back (or asks) on the false report.

## Manual verification — ROUND 6 RESULTS (false-report mission, 6 turns, ~$0.06) — CONVERGENCE

Scenario per cycle-3's plan: badge build -> relative tweaks -> deliberately FALSE "screen is
black" report -> revert demand -> ledger confrontation -> sweep + gated render_image. Sandbox in
`scripts/dogfood/runs/` (reboot-proof), lib seeded.
(a) ALPHA FACTS: VERIFIED live — the white-on-transparent badge reported `ink 42% | bbox ...`
(pre-fix this was the "FLAT white" lie). Facts tracked every tweak (glow down: ink 42->21%, luma
dimmed) and certified the sweep as behavior-preserving (identical facts before/after).
(b) TWO-SIDED TRUST: WORKED — on the false report the reply cited the facts ("15% ink, bbox
centered — the badge is still drawn"), asked the user to check their view, and STOPPED (3
iterations; round-3's analog was a 16-edit thrash). Partial: before replying it silently
rewrote the ring block (unrequested artifact drift, ink 21->15%) and the reply said "I checked
the source" — concealing the edit (it IS disclosed by the ledger + tool cards, so prose-only).
(c) OSCILLATION BRAKE: FIRED LIVE (first time) — T5's revert flailing re-applied an earlier
state, the NOTE landed ("a state it already had 2 edit(s) ago"), and the agent OBEYED: stopped
editing source, finished with one set_uniform + reply.
(d) Model-honesty residuals (all pass the better-model test; the ledger/facts channel already
carries the ground truth, so no harness gap): denied its own edits with three `replace_lines:
ok` ledger lines visible in history (admitted only when quoted back); claimed the restored code
"matches exactly" the post-T2 state while the facts disagreed (ink 18% vs 21%) and no history
view exists to verify exactness (no-undo is by design); changed u_ring_speed unrequested
mid-"revert". These are cheap-model attention failures, not missing feedback.
(e) Framework: analyze.py's usage regex predated `rsn=` (every token/cost read 0) — fixed; two
skill gotchas recorded (skill-runner substitutes `$N` literals with invocation args;
white-on-transparent renders eyeball as blank — composite onto dark first). Coverage 4/12 by
design (behavioral round; round 5 owned the lib/read surface — read_shader-on-lib: stays
code-reviewed only, GL-marshalled paths are the documented test deviation).

VERDICT: CONVERGED. (i) Zero terminal failures across rounds 4-6; (ii) round 6 surfaced no new
harness-fixable class; (iii) every residual is model-bound. Live behavioral evidence now exists
for every 033 mechanism: hints converge (R4: 3 attempts, R5: 1), force-restore + streak (R4),
facts reacted to (R4/R6), alpha facts (R6), zero silent turns (R4-R6), oscillation brake (R6),
two-sided trust (R6), lib surgery + rename + honest broken-state reply (R5), cost accounting
exact (R4).

## Manual verification (original criteria)

A fresh dogfood round (same mission class as round 3) after landing: expect (a) range-bookkeeping
errors recovered in ≤1 attempt thanks to hints, (b) zero silent turns, (c) the agent reacting to
render facts ("ink 0%" → self-diagnosed blank) without maintainer prompting, (d) reasoning tokens
near-zero in the trace.

## Open questions

1. Render-facts cost: probe render per mutating edit (~ms on V3D for non-text shaders, ~0.3s for
   text-stack at 64x64) — acceptable per-edit, or gate behind "compile clean AND source changed"?
2. Force-checkpoint N default (maintainer tunes): start at 4?
