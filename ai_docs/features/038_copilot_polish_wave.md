# 038 — Copilot polish wave (filed-deferral sweep)

A batch of small, point-scoped fixes for filed `todo.md` deferrals in the copilot stack — collected
into one polish wave rather than spread across drive-by commits. Every item is a localized fix; none
changes the copilot's user-facing behavior contract beyond closing the specific footgun named. This
wave ALSO closes out three long-drifted features whose code already landed (034 / 025 / 032 — all
`partial` only because of an un-runnable headless `make run` gate, not unfinished work).

> **STATUS: IN PROGRESS.** Each item lands as its own green-gated commit (`make check`; `make smoke`
> for none of these — no UI/lifecycle code is touched). Items are independent; order is by ascending
> blast radius.

## Goal

Resolve the cheap, locally-scoped copilot deferrals that have accumulated in `todo.md`, deleting each
resolved entry in its resolving commit (per the docs-discipline rule). Close the three drifted
features (034/025/032) as `done`.

## Out of scope (each with its trigger)

- **The model-bound honesty halves** (render-facts honesty residual; reply-time honesty enforcement)
  — these need a better model to even reproduce or a VLM-judge; not a point fix. Trigger unchanged in
  `todo.md`.
- **The structural/large deferrals** (IPv4 pins, exporter decomposition, multi-file editor, V3D
  cell-cull, keyring migration, the Stop-during-stall per-delta cancel) — real features with their own
  triggers; left untouched.
- **`insert_after` anchor-text (item 6) is BUILT but stays dormant by evidence policy** — see
  decision 6: the deferral's own trigger is "a trace shows a misplaced insert", which has NOT fired.
  We do NOT add the anchor to `insert_after` speculatively. Item 6 is therefore RE-CONFIRMED-DEFERRED,
  not implemented (the spec records the decision; the `todo.md` entry stays, reworded only if needed).

## Design decisions (numbered — lock-in)

1. **delete_node empty/unresolvable-target precheck (todo: "delete-gate can prompt to confirm
   deleting node `?`").** Add a `delete_precheck(args)` to `tools/shader.py` wired as the
   `delete_node` definition's `precheck=`, mirroring the publish tools' precheck seam. It resolves the
   target through `caps.node_tree()` (the same GL-free prefix rule `_node_display` already uses) and
   returns a guided message when the id is empty or matches no node — so the loop's pre-gate guard
   (`registry.precheck`, agent.py) short-circuits BEFORE `build_gate`, and the user never confirms a
   `Delete node `?`` that then errors. A missing target routes around the gate + retry cap exactly
   like a publish cred-miss (it's not a convergence failure). Message names the project map as the fix
   ("no node `<arg>` to delete — pass an id from the project map"). The handler's own
   resolution-failure branch (backend `delete_node` returning `ok=False`) STAYS as the backstop for a
   race (node deleted between precheck and execute).

2. **dogfood `analyze.py` per-turn cost/tokens for error-terminal turns (todo: "drops tokens/cost for
   error-terminal turns").** `_close_section`'s `turn_done` branch is the ONLY place
   `cost_usd`/`billed_in_tokens`/`reply_tokens` get set; giveup/stream-error/truncated/incompatible
   terminals never emit `turn_done`, so those turns keep the `0.0`/`0` defaults and the rollup
   understates real spend. Fix: when a turn closes (next `turn_start` or EOF) WITHOUT having seen a
   `turn_done`, fall back to summing its collected `Iteration` rows — `cost_usd = sum(it.cost_usd)`,
   `reply_tokens = sum(it.out_tokens)`, `billed_in_tokens = sum(it.in_tokens)` — the same per-iteration
   `usage:` lines the turn already parsed. `peak_iter_in_tokens` is already iteration-derived, so it's
   correct for error turns today; only the three `turn_done`-keyed fields need the fallback. The
   fallback is a per-turn finalizer invoked from `_close_section` on the `turn_start` edge (and once at
   EOF) — guarded on `cost_usd == 0.0 and iterations` so a real `turn_done` value is never overwritten.
   NOT in scope (recorded so the deferral's tail isn't lost): the coverage-count / 🔴-glyph / `rsn=` /
   resolved-model / `--calls` forensic items stay in the `todo.md` entry (reworded to drop only the
   cost/tokens half).

3. **Unsealed checkpoint dirs leak on quit/kill mid-turn (todo: "unsealed checkpoint dirs leak").**
   `CheckpointStore._rehydrate` iterates `root` dirs and `TurnCheckpoint.load` returns `None` for an
   index-less dir — so a turn killed before `seal()` (no `checkpoint.json` written) leaves an orphan
   snapshot dir invisible to `prune_to`/`drop` forever. Fix: in `_rehydrate`, after the load loop,
   `shutil.rmtree(turn_dir, ignore_errors=True)` any sub-dir that produced no loadable checkpoint
   (index missing or corrupt). This runs at store init (session start), so a stale dir from a previous
   crashed run is swept on next launch. Safe: an index-less dir is BY CONSTRUCTION never referenced by
   any sealed checkpoint (the index IS the reference), and an in-flight turn of the CURRENT session
   hasn't created its dir yet at init time (`open()` runs at turn start, after `__init__`).

4. **`publish_youtube` mutates exporter shape on the worker thread (todo: "publish_youtube mutates
   exporter shape state on the worker thread").** `backend.py::publish_youtube` calls
   `exporter.set_shape(is_short)` + the `finally` restore directly on the worker thread; `set_shape`
   writes `_render_state.shape` which the main-thread Share tab reads — the one mutation in the file
   that skips `run_on_main`. Fix: wrap the `set_shape`/render/restore body in a `_bridge.run_on_main`
   closure (the same marshalling every other backend mutation uses), so the shape write happens on the
   main thread. The `_copilot_publish` call inside already does its own bridge round-trips; verify it
   composes (a `run_on_main` that itself calls `run_on_main` would deadlock) — so ONLY the two
   `set_shape` calls move to main (a pair of trivial `run_on_main(lambda: exporter.set_shape(x))`
   round-trips bracketing the existing publish), NOT the whole publish body. Behavior identical
   (benign-today becomes correct-by-construction).

5. **The ranged-`replace_lines` anchor-resolution bug CLASS (the real root, found by dogfood —
   supersedes the original "no-op CLEAN spree" framing).** The todo "no-op CLEAN spree" item was
   investigated by a 6-run dogfood + three ultracode audit workflows. The filed framing was wrong, and
   what the dogfood actually surfaced was a CLASS of five distinct defects in how a ranged
   `replace_lines` locates a block from its two TEXT anchors. ALL are fixed DETERMINISTICALLY in the
   engine (`backend.py::_resolve_anchored_edit` + helpers) — no prompt change, no model-facing hint.
   The unifying idiom is brace STRUCTURE of the source, not line-text matching, so the five guards
   compose cleanly (each gated to its own trigger, run in pipeline order). This is the same off-by-one
   class 036 killed for line NUMBERS, returning via the SEMANTICS of which text the model picks as an
   anchor: 036 guarantees the engine finds the NAMED line, not that the model named the RIGHT one.
   - **(A) orphan-brace absorb** — model anchors `last_line` on the block's last STATEMENT (or its
     opening brace) while `new_text` re-sends the whole block; the original block tail survives below
     the range and the splice doubles the `}` (`N '{' vs N+1 '}'` → giveup, or a retry spree — a real
     Grid run did 8 edits / $0.025). `_absorb_orphan_tail` extends the range forward to swallow the
     duplicated tail, ONLY when the tail is a strip-invariant SUFFIX of `new_text` (proven duplicate),
     `new_text` is itself brace-balanced, AND swallowing rebalances the splice. Verified live (T5): the
     model anchored `last_line="return mix(...)"` (penultimate), absorb extended span to the real `}`,
     compiled clean.
   - **(B) multi-line `last_line` rejected** — model quotes `"return total;\n}"` (statement + brace) to
     mean "the block ends here"; the wire schema wants ONE line, so `_locate_anchor` rejected it and
     the model looped 3× → giveup. Fix: a multi-line quote is matched as a CONTIGUOUS RUN against the
     source and resolved to its boundary line (first quoted line for `first_line`, last for
     `last_line`); a run not present contiguously still rejects (never a bare-boundary fallback — that
     could mislocate). `_locate_line` holds the old single-line path. Verified live (T6).
   - **(C) cross-block silent deletion** — a multi-line `last_line` run UNIQUE to a LATER block makes
     `end` jump there; a balanced `new_text` clean-compiles while a whole function between the anchors
     is silently deleted (worse than a compile error — no signal). `_range_straddles_blocks` rejects
     when the splice DROPS a top-level block (`_top_level_blocks` count falls); a structure-preserving
     whole-file resend (count unchanged) is NOT flagged. Found by the audit, reproduced by hand
     (brace-balanced, function gone), then guarded.
   - **(D) ambiguous bare `}` + near_line on the block START** — `last_line="}"` strip-matches EVERY
     closing brace; `near_line` pointed at the block's opening line, so `_pick_nearest` chose the
     PRIOR function's close (nearer) → `start > end` → reverse-order reject → giveup. `_resolve_block_close`
     brace-matches from `first_line`'s opener (`parser.find_body_end`, comment-stripped) to find the `}`
     that actually closes THAT block — skipping nested for/if braces and the prior block. A `near_line`
     that lands ON an inner `}` defers to the near path (so a partial inner-block edit keeps its
     intent). Verified live (T6 re-run applied first try, no giveup).
   - **(E) incoherent-range corruption** — the audit's worst residual: `first_line` on an opener,
     `last_line` on an INNER `}`, a PARTIAL `new_text` — the range opens braces `new_text` doesn't
     re-supply, dropping the signature with a (coincidentally) balanced, clean-compiling result that
     (C)'s top-level-count guard is blind to. A brace-delta COHERENCE check (`_brace_delta(range) !=
     _brace_delta(new_text)` → reject) catches it: a coherent block edit replaces a region with text of
     equal net brace delta. Synthetic (the model never emitted it in 6 runs) but a real silent-data-loss
     corner; rejecting beats applying a broken splice.
   - **Defect "B" (the filed "no-op spree") does NOT exist as oscillation.** 6 dogfood runs never once
     reproduced source RETURNING to a prior state. What looked like churn was either forward refactor
     (rename vars, add `const` — monotonic, visuals unchanged but source always new) or the orphan
     retries of (A). The filed `consecutive_identical_edits` brake is therefore NOT built — there is no
     observed oscillation to brake, and a render-facts-stagnation comparator is defeated by the live
     `render@t={t:.1f}s:` timestamp anyway. The residual "render-blind forward churn" is covered by the
     existing `max_clean_edit_streak` nudge (F08) and is NOT expensive on its own. `agent.py` is
     untouched by this wave. The five guards above were each adversarially verified (the audit caught
     real holes in intermediate versions — (A)'s missing balance guard, (C), (E) — every one reproduced
     by hand before fixing).

6. **`insert_after` anchor text — RE-CONFIRMED DEFERRED, not implemented.** The `_locate_anchor`/
   `apply_anchored_edit` seam (036) exists and porting it to `insert_after` (an `after_line` text
   anchor) is mechanically a small change. BUT the deferral's trigger is explicit: "a trace showing a
   misplaced insert" — and `insert_after` has ZERO observed failures. The repo convention ("a guard
   earns its place — don't add standing structure for a transient/unobserved lapse") says we do NOT
   build it speculatively. Decision: leave `insert_after` line-number-addressed; keep the `todo.md`
   entry as the live trigger. NOTE: `insert_after` shares Defect A's coordinate-fragility class but adds
   (never replaces), so it cannot leave an orphan tail — the absorb fix doesn't apply to it.

8. **Consolidate the comment-safe brace count (audit follow-up).** Two identical comment-safe brace
   counters existed — `backend._brace_delta` (the absorb's balance check) and `edit_hints`'
   `opens/closes` (the orphan-tail hint) — the repair-half and the detect-half of one concept. Lifted
   into one `parser.brace_counts(text) -> (opens, closes)` beside `advance_brace_depth`; both callers
   derive from it. The hand-count (not `glsl_lex`) is correct BY DESIGN: GLSL has no string literals, so
   a raw count over comment-stripped source is exact, and a `glsl_lex` PUNCT-filter was verified
   byte-equal across every edge case (nested block comments, `#define {`, hex `0x7B`) — a lexer would be
   strictly slower, never more correct. (`advance_brace_depth` is NOT a candidate: it's per-line, clamped,
   and comment-UNAWARE.)

7. **Delete dead `ToolDefinition.needs_gl` + `.category` fields (todo: "needs_gl + category are dead
   fields").** Verified zero readers: `grep -rn '\.needs_gl\b|\.category\b'` over `shaderbox/` returns
   nothing (thread-marshalling happens inside the backend methods, not gated on `needs_gl`; `category`
   was scaffold for a lazy-catalogue that the deferral keeps deferred). Both are pure documentation
   today. Fix: delete both fields from the `ToolDefinition` dataclass (`tools/base.py`) + their
   per-tool kwargs across all definitions (`tools/shader.py`, `tools/publish.py`, `tools/telegram.py`).
   `eager` STAYS (it IS read — `registry.eager_specs`). The lazy-tool-catalogue deferral (lever 2)
   notes it would re-introduce `category` if/when built — that's an additive change at that time, not a
   reason to keep a dead field now (per "delete dead code; git history is the scaffold"). The `todo.md`
   "needs_gl + category dead fields" entry is deleted; the lazy-catalogue entry keeps its own note,
   reworded to "re-add `category` when the catalogue lands" instead of "category goes live".

## Feature closeouts (no code — roadmap/doc only, in the sanitize commit)

- **034 ui_polish_wave_2 → done.** All 13 findings `[fixed]`, tested, F13 live-verified; the residual
  per-finding `make run` is an un-runnable-headless visual confirm the maintainer covers in daily use.
- **025 project_session_extraction → done.** C1–C4 landed green (4 commits), C6 doc-flip already done
  (conventions has the ProjectSession bullet, todo entries flipped). The `App.__init__` runtime path
  is type-checked + review-verified; the `make run` gate is un-runnable on the display-less Pi.
- **032 sdf_shader_library → done.** Cyrillic chat replies fixed (034 F11), seed-loading done
  (`shader_lib_seed_sync`), the Д glyph is an explicitly-parked maintainer-aesthetic non-issue ("пока
  так и оставим") — not a TODO. Nothing open.

## Files touched

- `shaderbox/copilot/tools/shader.py` — item 1 (`delete_precheck` + wire to `delete_node` def); item 7
  (drop `needs_gl=`/`category=` kwargs).
- `shaderbox/copilot/tools/publish.py`, `shaderbox/copilot/tools/telegram.py` — item 7 (drop kwargs).
- `shaderbox/copilot/tools/base.py` — item 7 (drop the two dataclass fields + their comments).
- `shaderbox/copilot/backend.py` — item 4 (`publish_youtube` set_shape → `run_on_main`); item 5 (the
  five anchor guards: `_locate_anchor`/`_locate_line`/`_pick_nearest` multi-line locator, `_brace_delta`
  + `_absorb_orphan_tail`, `_top_level_blocks` + `_range_straddles_blocks`, `_resolve_block_close`, +
  the brace-delta coherence check — all wired through `_resolve_anchored_edit`).
- `shaderbox/shader_lib/parser.py` — item 8 (`brace_counts` helper).
- `shaderbox/copilot/edit_hints.py` — item 8 (`compile_hints` uses `parser.brace_counts`).
- `shaderbox/copilot/checkpoint.py` — item 3 (`_rehydrate` sweeps index-less dirs).
- `scripts/dogfood/analyze.py` — item 2 (error-terminal cost/token/peak fallback).
- `tests/` — `test_line_editing` (the orphan-tail block: narrow / opening-brace / multi-function /
  nested / whitespace-divergent / unbalanced-guard / divergent-body safety), `test_tool_registry`
  (delete precheck), `test_dogfood_analyze` (error-turn cost+peak fallback).
- `ai_docs/todo.md` — delete items 1/2/3/4/7 entries; reword the giveup-counter entry (Defect B
  refuted, Defect A resolved); `ai_docs/roadmap.md` — flip 034/025/032 to done + banner;
  `ai_docs/conventions.md` — the orphan-brace absorb gains a `## Known quirks` entry (item 8 + the
  suffix+balance double-guard + whitespace-invariance caveat).

## Manual verification

- `make check` green after every commit (the whole wave is GL-free / headless-checkable except the
  un-runnable `make run` visual confirms the closeouts inherit).
- Item 2: re-run `scripts/dogfood/analyze.py` over an existing run that HAS an error-terminal turn
  (035 mega run: turns 12/13/18) and confirm the per-turn table no longer shows `$0.0000` for them and
  the run total rises toward the real ~$0.27.
- Items 1/3/4/5/7: unit-tested headless (no app run needed — none touch UI/draw).
- The three closeouts need no run (doc-only).

## Open questions for the user

1. **Item 5 framing** — accept the narrowed fix (catch byte-identical resubmitted edits) over the
   todo's literal "render-facts didn't move" brake (which the live timestamp defeats)? The narrowed
   fix is honest and catches the actual observed trace; the render-facts comparator would need a
   timestamp-stripped, animation-aware design that's a real feature, not a polish item.
2. **Item 6** — agree it stays deferred (no observed misplaced-insert trace), or do you want the
   anchor ported to `insert_after` now anyway as a consistency move with `replace_lines`?

## Review history

<!-- Pre/post-impl review findings + any design reversals land here. -->
