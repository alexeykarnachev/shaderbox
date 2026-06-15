# 050 — Copilot edit-churn + probe-render honesty wave

A correctness/quality wave on the copilot, sourced from ONE real maintainer session that went badly:
the "fire effect step-by-step tutorial" build on `openai/gpt-5.3-codex`, 2026-06-15 ~21:31–21:42.

**Reference trace (DO NOT DELETE — kept as the source of truth for this feature):**
`~/.local/share/shaderbox/copilot_traces/copilot_dev_2026-06-15_21-26-29_114795.transcript`
(23372 lines, 11 user turns, 39 llm round-trips). The duplicate-comment spiral is the single turn at
`21:33:38` ("yes, proceed"): tool_call lines 2850 → 11488 in the transcript, **16 `edit_shader` calls**,
`clean_streak_nudge` fired at iteration 5 (transcript line 5364) and was ignored for 10 more edits.
The maintainer's two pushback turns are at `21:38:56` ("what was this issue with these duplicate
comments?") and `21:39:29` ("but why did you do so many identical comments?").

This spec is the PLAN-DRAFT (dev_flow step 2). NOT YET PLAN-LOCKED — the maintainer has one more
concern to add before lock. Findings are filed; design decisions below are PROPOSED, not locked.

---

## Goal

Stop the copilot from (a) churning a single file with non-converging micro-edits while the user sees
nothing, and (b) reporting probe-render facts the user cannot correlate with what they actually see.
Both fired in the reference trace; both are engine bugs, not pure model lapses.

The wave addresses three findings, ordered by how much they damaged the session:

### Finding 1 — duplicate-comment spiral: the comment matcher + `comment_loss` guard create a non-converging edit loop (ROOT)

**Symptom (trace):** across calls 2–16 of the spiral turn, a single `// Step 2:` comment line piled
up to 4, then 5 copies with mismatched indentation (`    //…` vs `        //…`), each `edit_shader`
trying to clean it up, the count oscillating instead of converging. The model itself diagnosed it to
the user as "an insertion pattern got reapplied repeatedly" (transcript ~line 14619).

**Mechanism (grounded in source):**
- `apply_shader_edit` (`copilot/backend.py:1415`) matches in two tiers: `token_match`
  (`copilot/glsl_lex.py:170`) first, then `_comment_only_spans` (`backend.py:241`) as fallback.
- A **comment line lexes as trivia** — `glsl_lex` emits no tokens for it — so `token_match` returns
  `[]` for a comment-only old_str AND never sees comment lines when matching a code region. So a
  duplicated comment with distinct indentation is invisible to the ">1 match → ambiguous, refuse"
  safety: the code-token anchor matches once, the comment riding alongside it is silently spliced.
- The cleanup then routes through `_comment_only_spans`, which matches by `_ws_normalize`
  (`backend.py:199`) — horizontal-whitespace runs collapsed to one space. So `    // Step 2` and
  `        // Step 2` normalize identical: the model's mental count of "how many dupes, at what indent"
  desyncs from what the splice does, and each "fix" re-emits a copy in new_str. Net oscillation.
- **The `comment_loss` guard actively fights the cleanup:** `span_drops_comment`
  (`glsl_lex.py:157`) returns `matches=0, comment_loss=True` (`backend.py:1441`) for any edit whose
  matched span contains a comment the old_str does not reproduce — i.e. *deleting a duplicate comment
  is treated as accidental comment-loss and refused.* The guard meant to protect comments is what
  trapped the model trying to remove the ones it accidentally created.

**The class:** an `edit_shader` whose old_str/new_str differ only in comment/whitespace content can
both (a) silently duplicate (token matcher blind to comment trivia) and (b) be un-cleanable (the
comment_loss guard refuses the removal). The honest fix targets the matcher+guard interaction, not a
prompt rule.

### Finding 2 — `clean_streak_nudge` is advisory-only and fires once; the model blew past it

**Symptom (trace):** the nudge fired at edit 6 (iteration 5, `clean_edits_this_turn >=
max_clean_edit_streak=6`) and the model did **10 more edits** in the same turn.

**Mechanism (grounded in source):**
- `agent.py:817-826`: `clean_edits_this_turn` increments on every clean edit-tool call; at the
  threshold it appends `_CLEAN_STREAK_NUDGE` (`agent.py:52`) to ONE tool result and sets
  `clean_streak_nudge_sent = True` — **once per turn, soft text, no enforcement.**
- `config.max_clean_edit_streak = 6` (`config.py:32`; `0 = off`).
- A non-edit tool (read/grep) does NOT reset `clean_edits_this_turn` (the counter is cumulative per
  turn by design — `agent.py:814` comment) — so that half is already correct; the gap is purely that
  one soft nudge is too weak against a model mid-spiral.

**`todo.md` cross-reference — the trigger has FIRED.** The existing deferral
*"copilot edit-churn brake (039 follow-up) — churn shown STOCHASTIC, do not build yet"* names its own
trigger: *"a third dogfood run showing a >10-iteration single-file clean-edit spree (the 039 gate
run's turns 3/5/6/8 shape), OR a user report of the copilot 'making many tiny edits'."* This session
is a **maintainer report of exactly that**, with a 16-edit single-file spree. Its listed levers, in
order: a per-FILE clean-streak fact in the tool result ("N edits in a row on this file — finish in ONE
write_shader"), + a same-file batch steer in `_EDIT_SHADER_DESC`. This feature subsumes that deferral
— it must be deleted from `todo.md` in the resolving commit.

### Finding 3 — probe-render `t` is the live wall-clock, so the agent's facts are uncorrelated with what the user sees

**Symptom (trace):** every probe render is stamped `render@t=951.9s … 1047.4s` — the app had been
open ~16 minutes. When the user asked "what should I see now?", the agent's only visual signal was a
frame at an arbitrary `u_time` phase, disconnected from (a) the user's live preview (its own clock)
and (b) a t=0 export.

**Mechanism (grounded in source):** `_render_facts_for` (`backend.py:1507`) uses
`render_t = glfw.get_time() if t is None else t` (`backend.py:1525`). For a **shader** edit, `t` is
None → live wall-clock. (A **script** probe passes its own sample time via `_script_render_line`
`backend.py:1540`, so scripts are anchored; only the GLSL-edit facts float.)

**Relation to the known roadmap finding:** roadmap Active-context (the 043 dogfood note) already
records *"the agent has no 'render at t=N' affordance, so it can't see its own animation past the
export-pinned t=0."* This trace shows the complementary failure: for a live shader edit the probe is
pinned to wall-clock, NOT t=0 — so even the single frame it gets is unanchored. The two together say:
the agent needs a DEFINED, STABLE render clock for its facts (and ideally the ability to sample more
than one `t`).

---

## Out of scope

- **Multi-`t` render sampling for the agent (the "render at t=N" affordance).** Finding 3's full fix
  (let the agent request a render at a chosen `t`, or auto-sample a few `t`s and report motion) is a
  larger capability. This wave's finding-3 scope is only to make the SINGLE probe `t` *defined and
  stable* (proposed: t=0, matching export) so the fact is correlatable; the richer affordance stays
  deferred.
  - **Trigger (deferred):** a dogfood/maintainer trace where a one-frame fact at the fixed `t` still
    can't answer "is the animation right" because the interesting motion isn't at that `t`.
- **The model-bound render-facts-honesty half** (a cheap model claiming success against damning
  facts) — already a `todo.md` deferral, untouched here.
- **General copilot cost/latency (lazy tool catalogue, lever 2)** — separate `todo.md` deferral.

## Design decisions

PROPOSED (not locked — for plan-lock discussion). Numbered for reference.

1. **Comment duplication at the matcher (Finding 1a).** The token matcher must not let a comment ride
   silently alongside a code-token match when that produces a duplicate. Options to weigh at lock:
   (a) include comment trivia in the token-match key so a comment-bearing region matches ">1" when
   ambiguous; (b) detect at splice time that new_str would create back-to-back identical comment
   lines and refuse/dedupe. Decision deferred to lock.

2. **`comment_loss` guard vs. legitimate dedupe (Finding 1b).** The guard
   (`span_drops_comment`) is correct for its intent (don't silently delete a real comment) but must
   not refuse the removal of a DUPLICATE comment. Option: when the dropped comment(s) still appear
   elsewhere in the file post-splice (the multiset is non-empty only because of duplication), allow
   the edit. Decision deferred to lock.

3. **Churn brake escalation (Finding 2).** Beyond the one soft nudge, the proposed lever (from the
   `todo.md` deferral) is a per-FILE clean-streak FACT in the tool result ("N edits in a row on this
   file — finish in ONE write_shader") that repeats/escalates rather than firing once. Whether to add
   a HARD stop (force turn-end after K clean edits) is the open question — a hard stop risks
   truncating a legitimate long edit sequence. Decision deferred to lock.

4. **Probe-render clock (Finding 3).** Make the shader-edit probe `t` defined and stable. Proposed:
   default to **t=0** (matches the export clock the user ultimately renders), so the agent's fact and
   any future export agree. Decision deferred to lock (t=0 vs. "match the user's live preview clock"
   — the latter is unstable/unknowable headless, so t=0 is the robust default).

## Files touched

(Anticipated — confirm at impl.)
- `shaderbox/copilot/glsl_lex.py` — `token_match` / `span_drops_comment` (Findings 1a/1b).
- `shaderbox/copilot/backend.py` — `apply_shader_edit` matcher path (`_comment_only_spans`,
  `_splice`); `_render_facts_for` clock (Finding 3).
- `shaderbox/copilot/agent.py` — churn brake (Finding 2).
- `shaderbox/copilot/config.py` — any new churn-brake config (Finding 2).
- `shaderbox/copilot/tools/shader.py` — `_EDIT_SHADER_DESC` same-file batch steer (Finding 2), if taken.
- Tests under `tests/` — a regression test per finding (see Manual verification).

## Manual verification

Each check must fail for exactly one reason; name the falsifier.

- **Finding 1 (duplicate-comment spiral) — the headline regression.** A unit test that replays the
  trace's failure shape: start from a source with one comment line + a code line; apply the
  insert-then-cleanup edit sequence; assert the comment appears **exactly once** and the cleanup edit
  is NOT refused with `comment_loss`. Falsifier: revert the matcher/guard fix → the test goes red
  (either a duplicate survives, or the dedupe edit returns `matches=0, comment_loss=True`). Anchor:
  the actual old_str/new_str pairs are in the reference transcript (calls 2–6, lines 2850–5360).
- **Finding 2 (churn brake).** A test that drives N clean edits in one turn on the same file and
  asserts the escalating per-file fact appears (and, if a hard stop is chosen, that the turn ends).
  Falsifier: disable the brake → no fact / no stop.
- **Finding 3 (probe clock).** A test asserting `_render_facts_for(node)` (shader path, `t=None`)
  stamps the FIXED clock, not `glfw.get_time()`. Falsifier: revert → the stamp tracks wall-clock.
- **`make check` + `make smoke`** green (copilot edit path is exercised by smoke only indirectly; the
  unit tests above are the real gate).
- **No live App manual check needed** — these are headless-testable engine behaviors. A maintainer
  re-run of a similar "build me an effect step by step" session is the real-world confirm but not a
  gate.

## Open questions for the user

1. **The one more concern** you flagged to add before plan-lock — append it here.
2. Design decisions 1–4 are proposed, not locked: which shape for the matcher fix (Decision 1/2),
   and whether the churn brake gets a HARD stop or stays escalating-soft (Decision 3).
3. Finding 3: confirm t=0 is the right fixed clock (vs. some other anchor), and confirm the richer
   multi-`t` affordance stays out of scope for this wave.

## Review history

(empty — pre-impl review not yet run; this is the plan-draft.)
