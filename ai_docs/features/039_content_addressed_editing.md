# 039 — Content-addressed editing (kill the line/anchor edit tools)

Collapse the copilot's three source-edit tools to TWO content-addressed ones: `edit_shader`
(old_str/new_str substring replace — the one partial-edit tool) and a new `write_shader`
(whole-file rewrite/create). `replace_lines` (ranged + whole-file modes) and `insert_after` are
DELETED, together with the entire anchor-resolution machinery 036 introduced and 038 grew
(`_locate_anchor` / `_locate_line` / `_pick_nearest` / `_absorb_orphan_tail` / `_resolve_block_close`
/ `_range_straddles_blocks` / `_top_level_blocks` / brace-delta coherence / `near_line`).

> **STATUS: DONE (2026-06-12), uncommitted — post-impl-reviewed + dogfood gate PASSED
> (safety/robustness AND cost — the run-2 re-measure beat the baseline; see Manual verification).** Plan-lock: Q1 `write_shader` (defaulted, no objection); Q2 maintainer picked
> variant Б — the shrink-fact watches functions AND uniform/const declarations ("пускай видит всё").

## Why (the evidence)

An adversarial 8-dimension review swarm over f4ba206 (56 findings; triage artifact preserved below
in `## Appendix: review evidence`) found the anchor mechanism structurally unsound: any addressing
scheme where the model must REPRODUCE a location (a line number or a text quote) inherits model
imprecision, and the engine then either trusts it (silent mislocation) or second-guesses it (a
guard pile + false rejects). Concretely: 3 critical silent-corruption classes (stale-quote
mislocation below top level; comment-blind locators vs comment-aware guards; absorb first-balance
duplication), a 038 regression (whole-function delete via empty `new_text` — the documented
contract — is always falsely rejected), and a growing false-reject surface. Meanwhile:

- **Zero of the 56 findings touch `edit_shader`'s path.** Content addressing is structurally safe:
  the address IS the content, so you can only replace what you quoted verbatim (token-matched);
  a stale quote is a LOUD no-match, a duplicate is a LOUD multi-match. Silent mislocation is
  impossible by construction — the property the five guards tried (and failed) to retrofit.
- Forensic replay of all 12 real ranged calls (df_t4–t7): the models' preference for
  `replace_lines` is our own steering (descriptions + the comment-loss guard literally redirects
  into it); in t1–t3 models worked fine via `edit_shader` + whole-file.
- Cost of the kill: old_str re-sends the replaced block — ~+150–250 output tokens on a typical
  13–23-line function rewrite (~+10–20% of an edit's cost, cents at dogfood scale).

**Reversal note:** at 036 time a "delete ranged mode" proposal was refuted (failure looked like a
narrow +1-on-blank class, 6/8 calls correct) and ranged mode was kept + re-anchored. This feature
re-decides with strictly stronger evidence: the failure is not one coordinate class but the
addressing scheme itself; two waves of deterministic guards later the mechanism still silently
corrupts in reachable corners and falsely rejects documented behavior.

## Goal

One partial-edit tool whose failure modes are all loud, plus one whole-file tool with a
deterministic structure-loss fact. Delete the guard machinery and its test battery. Robustness
comes from the addressing construction, not from intent-guessing guards.

## Out of scope (each with its trigger)

- **Re-adding any line/range-addressed edit tool.** Trigger: measured output-token pain — a real
  trace where re-quoting large old_str blocks dominates turn cost (e.g. a >300-line file edited
  repeatedly) AND the model demonstrably fails to fall back to `write_shader`. Re-enter via a
  verified-addressing design (number+text checksum pair — see `## Appendix`), never via bare
  anchors.
- **The dogfood analyzer forensic gaps** (`todo.md` entry: coverage-count keying, 🔴 glyph, `rsn=`,
  resolved-model record, `--calls` mode). Its trigger ("next time analyze.py is edited") DOES fire
  on this feature's vocabulary touch — **consciously re-deferred, not absorbed**: 039's analyze
  edit is a 5-line tool-name-list change, not analyzer work, and absorbing a day of forensic
  scope into a tool-removal wave is exactly the blast-radius creep the dev flow warns about. The
  entry's trigger is reworded in the same commit to "next SUBSTANTIVE analyze.py work (parsing /
  coverage / report logic), not a vocabulary touch". (Pre-impl review F7 — conscious call.)
- **Render-facts honesty / VLM-judge** — unchanged, own todo entries.

## Design decisions (numbered — lock-in)

1. **`edit_shader` is THE partial-edit tool.** Mechanics unchanged (glsl_lex token-match,
   near-miss hint, multi-match reject, comment-loss guard, `replace_all`). Description rewritten:
   drop "BEST FOR a SMALL change" + the redirect to replace_lines/insert_after; it is now the
   default for ANY partial edit (replace a statement, a block, a whole function; insert by quoting
   the neighbor line into old_str and re-sending it + the new lines in new_str; delete by
   new_str="" — deletion is exact-by-quotation, which restores the whole-function-delete contract
   038 regressed). Multi-match → "add surrounding context" (unchanged).

2. **New `write_shader(new_text, target)` — whole-file rewrite/create.** Replaces replace_lines'
   whole-file mode AND insert_after's lib-file-creation role: a `lib:` target that doesn't exist
   is created here (`allow_create`), empty-file bootstrap included. `mutating=True, is_edit=True,
   eager=True, GatePolicy.NONE` — identical retry-cap/checkpoint posture to today's whole-file
   mode. Labels per the `ToolDefinition` convention: "Writing shader" / "Wrote shader". Batch
   guard (D9) applies: a whole-file rewrite composed from a pre-edit working set would silently
   revert a same-step prior edit, so the `_batch_mutated` check rejects it exactly as whole-file
   mode rejects today (persist itself marks the address — the caller doesn't).

3. **Deterministic shrink-fact on `write_shader`** (closes review finding robustness/F4: the
   escape hatch had zero guards). After an APPLIED rewrite — including applied-with-compile-errors
   (the mutation is real), but NOT when force-restore undid it (`restored_note` present — reporting
   removals that were rolled back would be false) — compare top-level NAME sets of (a) function
   definitions and (b) `uniform`/`const` declarations between old and new source. If the rewrite
   REMOVED any, the tool result appends one factual line:
   `note: this rewrite removed function(s): <names>; declaration(s): <names>`. A fact, not a hint
   (same philosophy as render-facts / compile hints); no reject, no model-facing instruction.
   A removed uniform is load-bearing beyond source text (it kills the user's live control + saved
   value), so a deliberate cleanup still deserves the factual line.
   **Extraction mechanics (pre-impl review finding 1 — the naive regex pass is wrong):** a new
   pure helper in `shader_lib/parser.py` (`top_level_names(text) -> (fn_names, decl_names)` or
   equivalent) that comment-strips via `strip_comments_keep_lines`, walks lines tracking depth via
   `advance_brace_depth`, and at depth 0 only: matches `FN_SIG_RE` per line (NOT whole-text
   `finditer` — that matches nested `else if (…) {` as a function), and matches declarations with
   a WIDENED array group (`[\s*\w*\s*]` — `DECL_SIG_RE`'s digits-only group misses
   `uniform uint u_text[MAX_TEXT_LEN];`, the flagship live-control case). `DECL_SIG_RE` itself is
   NOT widened (the lib index's semantics stay untouched). Known residue, documented in the
   helper: multi-line function signatures (non-K&R) are missed — best-effort fact, not a guard.

4. **Capabilities/backend surface.** `apply_line_edit` + `apply_anchored_edit` deleted from the
   `CopilotCapabilities` Protocol and `CopilotBackend`; a new `apply_full_rewrite(new_text,
   target) -> EditResult` carries decision 2 (resolve target with `allow_create` → batch guard →
   `_copilot_persist_target` — which already owns checkpoint capture, `invalidate_lib_consumers`,
   the oscillation note, force-restore, and `_batch_mutated` marking). The whole anchor helper
   block (`_pick_nearest` through `_resolve_anchored_edit`, incl. `_splice_lines`) dies;
   `EditResult.applied_span` (capabilities.py) dies. The shrink-fact rides a NEW sibling field
   `EditResult.rewrite_note`, APPENDED by `_applied_result` after the compile-result head — it
   must NOT ride `lib_note` (pre-impl review finding 6: `lib_note` REPLACES the head and persist
   already sets it on every lib write). `_out_of_range` in `tools/shader.py` dies.
   `parser.brace_counts` STAYS (compile hints use it); `parser.find_body_end` reverts to its one
   pre-038 consumer (the shader_lib index).

5. **Steering text updated in the same wave** (prompt + descriptions are part of the tool surface,
   not an afterthought). The full inventory (pre-impl review fold-in):
   - `prompt.py` EDITING block: two tools — `edit_shader` (any partial edit; quote enough context
     to be unique) and `write_shader` (whole file; default for full-function rewrites in
     small-to-medium files — keeps today's "<=150 lines just rewrite it whole" guidance). The
     "max ONE replace_lines/insert_after per file per step" rule becomes "max ONE write_shader per
     file per step"; the SHADER LIBRARY lib-creation bullet points at `write_shader`; the second
     rule site in USING TOOLS ("Line edits stay one per file per step.") reworded.
   - `edit_shader`'s comment-loss guard message: redirect to "copy the region verbatim INCLUDING
     its comment lines in old_str" (or write_shader for a big region) instead of the dead
     replace_lines.
   - `agent.py` `_COMPILE_THRASH_NUDGE` ("rewrite it in ONE edit (replace_lines — whole-file
     mode…)") → names `write_shader`. (The backend force-restore note is tool-agnostic — no change.)
   - `backend.py` `_copilot_resolve_lib_target`'s lib-missing reject ("use insert_after to create
     it") → names `write_shader`.
   - `edit_hints.py`: the module docstring + the redeclaration hint ("the edit range missed one
     copy; widen the range over it") + the orphan-tail hint ("an orphan tail survived below the
     edit range…") are reworded TOOL-AGNOSTIC (the underlying failure shapes — duplicate
     definition, unbalanced braces after an edit — still happen via edit_shader/write_shader; the
     hints survive, their range vocabulary dies).
   - `tools/base.py` `is_edit` comment (names the three tools).
   - `_READ_SHADER_DESC` / `_WORKING_SET_HEADER` keep their "line-numbered" wording — the numbers
     still serve compile-error correlation (conscious call; pre-impl review 9f).

6. **`agent.py` line-edit plumbing: DELETE.** Both pre-impl reviewers confirmed
   `_redact_stale_line_args` strips only line-location keys (`_LINE_ARG_KEYS`) and never
   `new_text`; `write_shader` has no location args, so the redaction carries zero weight post-039.
   `_LINE_EDIT_TOOLS`, `_LINE_ARG_KEYS`, `_redact_stale_line_args`, and its two tests die.

7. **CRLF normalization at the write seams.** Model-supplied source text gets `\r\n`→`\n`
   normalized once in `_copilot_persist_target` AND in `create_node`'s source path (which bypasses
   persist — pre-impl review finding 8); closes review finding locator/F8 for every model-text
   write at the shared roots.

8. **Tests migration.** Dies: the anchor/guard battery in the old `test_line_editing.py` (locators,
   absorb, block-close, straddle, coherence, span-echo, T5/T6/T7 anchor fixtures, near_line),
   `insert_after` tool tests, the two `_redact_stale_line_args` tests in `test_copilot_loop.py`.
   Lives on, re-pointed: retry-cap spiral test (drive via `edit_shader` no-match), batch-guard
   tests in `test_working_set.py` (apply_full_rewrite + the edit_shader exemption), `_caps.py` /
   `test_copilot_loop.py` fakes (drop the two callables, add `apply_full_rewrite`),
   `test_edit_messages.py` stubs, `test_conversation_persistence.py` StepRecord fixture names
   (cosmetic rename). NEW (closes review finding coverage/F1 — production code was never executed
   by tests; all on the REAL backend method via the `__get__`-stub idiom):
   - `apply_full_rewrite`: create-lib path (asserting `invalidate_lib_consumers` fires),
     batch-guard reject, shrink-fact (removed fn + removed uniform incl. a macro-sized array;
     NOT fired on force-restore), CRLF normalization, empty-file bootstrap;
   - `edit_shader` whole-function delete via new_str="" with an in-body comment (pins the
     `span_drops_comment` interaction);
   - a force-restore test on the real persist path (its only test touch today dies with
     `applied_span`);
   - **a dead-name invariant test**: no registered tool description, no `_SYSTEM_PROMPT` text, no
     guard/nudge/hint string (`_COMPILE_THRASH_NUDGE`, comment-loss message, `edit_hints` texts,
     lib-missing reject) contains `replace_lines` or `insert_after` (catches a missed rewording
     forever — pre-impl review F6);
   - tool-registry invariant updated for the new tool set;
   - `parser.top_level_names` unit tests (K&R fns, macro-sized arrays, inline-default uniforms,
     const, the documented multi-line-signature residue).

9. **Docs + scripts in the resolving commits.** `todo.md`: delete the `insert_after` anchor
   deferral (tool gone); REWORD the giveup-counter entry (its context claims `_absorb_orphan_tail`
   is the live fix — now "superseded by 039: the ranged mechanism was removed; the retry shapes it
   healed cannot occur") and the error-recovery entry's "boundary-checksum reject PASS" line
   (mechanism deleted); reword the analyzer-deferral trigger (see Out of scope).
   `conventions.md`: DELETE the five-guards `## Known quirks` entry; ADD a `## Design decisions`
   bullet: copilot partial edits are content-addressed ONLY — the line/anchor scheme (020·14 →
   036 → 038) was removed by 039 after the adversarial review; revisit per the Out-of-scope
   trigger. `roadmap.md`: 036 → `superseded` (brief points at 039); new 039 row; banner rewrite.
   `dev_flow.md` module map: "the 3 edit tools" → the two new ones. Dogfood surfaces (the manual
   gate drives them): `scripts/dogfood/scenarios/01_shape_gallery.md` + `02_logo_design.md`
   pass-criteria reworded off the dead tools; `.claude/skills/dogfood/SKILL.md` dead-tool mentions
   updated. `scripts/token_probe.py` tool list updated. **`scripts/dogfood/analyze.py` splits its
   vocabulary** (pre-impl review F1/F2 — the naive "keep old names" collides with the
   tool-registry drift-guard test): a CURRENT tool set that mirrors the live registry (the
   `test_tool_registry` invariant keeps pinning exactly this set) + a HISTORICAL superset (old
   names) used only for parsing old transcripts; the coverage DENOMINATOR is the current set, so
   post-039 runs can reach full coverage (historical re-analysis denominators shift — accepted).
   **Accepted cosmetic regression (conscious call):** historical conversations' turn-snippet hover
   renders raw `replace_lines`/`insert_after` via the `label_for` unknown-name fallback — no
   crash (persisted history is NL-only), no legacy-label map built.

## Files touched

- `shaderbox/copilot/tools/shader.py` — delete `replace_lines`/`insert_after` handlers + arg
  models + descs + `_out_of_range`; add `write_shader`; comment-loss message; `_applied_result`
  appends `rewrite_note`; registry defs.
- `shaderbox/copilot/backend.py` — delete the anchor helper block + `apply_line_edit` +
  `apply_anchored_edit`; add `apply_full_rewrite` (+ shrink-fact via the new parser helper); CRLF
  normalization in `_copilot_persist_target` + `create_node`; lib-missing message.
- `shaderbox/copilot/capabilities.py` — Protocol: two methods out, one in; `applied_span` out,
  `rewrite_note` in.
- `shaderbox/copilot/prompt.py` — EDITING block, USING TOOLS line, SHADER LIBRARY bullet.
- `shaderbox/copilot/agent.py` — `_COMPILE_THRASH_NUDGE`; delete the redaction trio (decision 6).
- `shaderbox/copilot/edit_hints.py` — tool-agnostic rewording (decision 5).
- `shaderbox/copilot/tools/base.py` — `is_edit` comment.
- `shaderbox/shader_lib/parser.py` — new `top_level_names` helper (decision 3).
- `scripts/token_probe.py`, `scripts/dogfood/analyze.py` (vocabulary split),
  `scripts/dogfood/scenarios/*.md`, `.claude/skills/dogfood/SKILL.md` — decision 9.
- `tests/` — per decision 8 (`test_line_editing.py` reborn as `test_content_editing.py`).
- `ai_docs/` — per decision 9.

## Manual verification

- `make check` green; full pytest green.
- **Baseline pinned BEFORE implementation** (pre-impl review F10 — the runs dir is gitignored and
  the skill documents wiping it): run `scripts/dogfood/analyze.py` over the existing df_t4–t7
  runs and record the per-run cost table in `## Appendix` below, so the post-039 comparison
  survives any cleanup.
- **The dogfood gate RAN (2026-06-12, 8 turns, codex-mini): PASSED on (a)-(d)** — edit_shader +
  write_shader used naturally; ONE loud no-match in the whole run (the fbm turn's final iteration before its budget cutoff; the rewrite itself had already applied clean) vs the
  baseline's 9 rejects + 2 giveups; the baseline-killer fbm rewrite landed FIRST TURN; the
  shrink-fact fired live ("removed function(s): hillsSilhouette, paletteSunset"). COST, two runs:
  the gate run drew micro-edit spirals ($0.1512/8 turns, ~2.4x baseline); a second run of the
  same expensive shapes (create Scene2D / rewrite palette / rewrite fbm) with cache telemetry
  cost **$0.0135/3 turns — cheaper than the baseline on every shape** (baseline ~$0.042 incl.
  its 2-giveup fbm saga), 68% prompt-cache hit, zero rejects. Verdict: the churn is model-
  stochastic and nudge-bounded, not a systematic cost of content addressing; brake deferred to
  `todo.md` ("edit-churn brake") with a third-run trigger. Reports:
  `ai_docs/features/039_dogfood_report_gate.md`.
- No `make run` needed (no UI surface).

## Open questions for the user

None — both resolved at plan-lock (see STATUS).

## Late additions (rode the wave after plan-lock)

- **Cache telemetry** (built during the cost investigation): `LLMUsage.cached_tokens` (llm/api.py)
  ← `prompt_tokens_details.cached_tokens` (llm/openrouter.py) → trace `usage:` line gains
  `cache=` (trace.py) → analyzer parses it (`_USAGE_RE` optional group) and reports a
  Cache-share line. Pinned by `test_trace_usage_line_round_trips_through_analyzer` +
  `test_cache_share_parsed_and_reported`. The openrouter inline SDK parse itself stays unpinned
  (faking the streaming client isn't worth it — None-safety probe-verified by the final swarm).
- **Final-swarm fixes:** the removed-names note filters against the new text (a restyled
  signature can't yield a false "removed" claim — a live false fact in the cost run found by the
  claims-forensics reviewer); lib writes get a deterministic brace-imbalance warning at the
  persist seam (the unbalanced-lib-write cell was fully dark — no standalone compile); lone-\r
  line endings normalize alongside \r\n at both seams.

## Review history

- **Final global swarm (7 finders + verifier layer, 2026-06-12, pre-commit): engine/conventions/
  hygiene PASS, robustness/tests/docs/claims PARTIAL → 30 confirmed findings, 5 refuted as
  late-round noise, all confirmed items fixed same-wave.** Highlights: a LIVE false shrink-fact
  ("removed main" on an Allman-styled rewrite) → the removed-names filter; the dark
  unbalanced-lib-write cell → the persist-seam brace warning; lone-\r seam gap; the gate report's
  "self-corrected next iteration" claim corrected against the raw trace (it was the budget-cutoff
  iteration); cache-telemetry test coverage added (round-trip + share line + mutation-killers for
  the balance gate, the batch-guard-before-persist property, and the write_shader arg order);
  todo/dev_flow/038-spec/skill doc-rot swept. Consciously accepted: multi-declarator/
  array-of-arrays misses in top_level_names (lib convention is one declarator per line);
  create_node's CRLF site untestable GL-free; the openrouter SDK parse unpinned.

- **Post-impl review (3 reviewers, 2026-06-12): pass / partial / partial → all findings fixed
  same-wave.** Correctness: PASS; its one Low (a brace-broken new_text hides later definitions
  from the depth-0 scan, making the note claim still-present functions removed) → fixed with a
  brace-balance gate on the note (the unbalanced case is already loud via compile error + brace
  hint). Spec-fidelity: caught the one MISSING mandated test — force-restore on the REAL persist
  path — now landed (`test_persist_force_restores_after_streak_on_real_path`); empty-file
  bootstrap is consciously MERGED into the create-lib test (apply_full_rewrite has no empty-file
  special case — create == bootstrap). Architecture: stale-docs catches (this STATUS line, the
  roadmap 020 row's live tool inventory, the 038 row's pointer, the todo giveup-entry History
  block, the conventions-bullet trigger gloss) → all fixed; the dead-name invariant widened from
  `copilot/` to all of `shaderbox/`; `test_line_editing.py` renamed `test_content_editing.py`.
  Accepted as-is: the dogfood SKILL's GLError bullet retro-attributes to write_shader (the class
  keys on the persist→render path); no test pins normalize-before-oscillation statement order.

- **Pre-impl review (2 reviewers, 2026-06-12): both PARTIAL → all findings folded.** Reviewer A
  (correctness/design): decision 3's extraction was unimplementable as specced (`DECL_SIG_RE` not
  MULTILINE; macro-sized arrays unmatched; whole-text FN scan false-positives on `else if`) →
  decision 3 now pins the per-line depth-0 helper + widened array group + documented residue;
  `edit_hints.py` added to scope; shrink-fact firing condition pinned (applied-even-with-errors,
  not on force-restore); `rewrite_note` sibling field pinned over `lib_note`; CRLF extended to
  `create_node`; spec's raw line-number citations replaced with symbols. Reviewer B
  (verification/blast-radius): analyze.py vocabulary split mandated (registry drift-guard vs
  historical parsing — the naive keep-old-names plan was self-contradictory); dead-name invariant
  test added; force-restore re-test added; 3 todo entries + dev_flow module map + dogfood
  scenarios/skill added to scope; baseline cost extraction pinned pre-impl; the historical-label
  cosmetic fallback accepted consciously; the analyzer-deferral trigger consciously re-deferred
  with a reworded trigger.

## Appendix: review evidence

- Source review: 8-dimension adversarial workflow over f4ba206 (56 findings, 8 verdicts:
  locator FAIL, the rest PARTIAL). Headline classes: C1 comment-blindness asymmetry (critical,
  first-hand confirmed), C2 structural guards blind below top level (critical, first-hand
  confirmed), C3 absorb first-balance duplication (high), C4 false rejects of legitimate edits +
  lying messages (high — incl. the whole-function-delete regression), C5–C9 medium/residual.
  Forensic replay: 12/12 real ranged calls resolve correctly post-038 — the holes are the next
  ring out, not the observed traces.
- The "verified addressing" alternative (number+text checksum pair, deterministic conflict
  adjudication) was designed and REJECTED in favor of this feature: it keeps two addressing
  schemes and a protocol change where content addressing needs zero new mechanism. Re-enter there
  if the Out-of-scope token-cost trigger ever fires.
- **Pinned pre-039 baseline (data-df1, 2026-06-12, codex-mini-class model, 10 turns):** total
  **$0.0623**, per-turn $0.0027–$0.0140, billed-in 336,349 tok, peak ctx 9,537→13,986. Tool mix:
  16× `replace_lines`, 1× `edit_shader`, 2× `create_node`. The dearest turns (6–9, $0.0047–$0.0140,
  two 🔴) are exactly the fbm anchor-retry spirals — 5 compile-error recoveries, 11 failed attempts
  total. Post-039 comparison target: the same rewrite-fbm/palette scenario shapes; success =
  comparable-or-lower per-turn cost with no anchor-reject retries (they become structurally
  impossible) and no new giveup class. Full table: analyzer md snapshot of `data-df1` (regenerable
  while the run dir survives; this summary is the durable copy).
