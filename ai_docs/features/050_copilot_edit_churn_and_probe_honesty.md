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

This spec is PLAN-LOCKED (dev_flow step 2 complete). Findings are filed; design decisions below are
LOCKED (maintainer walkthrough 2026-06-16). **Finding 1's mechanism was wrong in three earlier drafts
and was re-derived from a real-code repro + an adversarial swarm — see Investigation history before
touching it.**

---

## Goal

Stop the copilot from (a) churning a single file with non-converging micro-edits while the user sees
nothing, (b) reporting probe-render facts the user cannot correlate with what they actually see, and
(c) misattributing an engine-forced turn-end to the user ("you're right to pause now"). All are engine
bugs, not pure model lapses.

The wave addresses four findings, ordered by how much they damaged the session (1-3 from the reference
trace; 4 from a 2026-06-16 maintainer report, folded in because the Finding-2 hard-stop routes through
the same forced-end machinery):

### Finding 1 — the comment-blind matcher DUPLICATES a leading comment, and the model+guard layer let it spiral to 16 (ROOT)

> **This finding was rewritten after the first three mechanism theories were each REFUTED against the
> real trace + a real-code repro (see Investigation history).** The version below is reproduced
> end-to-end with the actual `token_match` + `_splice` and verified by an adversarial swarm
> (2026-06-16). Earlier prose blamed "the model re-emitting the same insert," "the model wasn't sure
> the edit landed," and "the `comment_loss` guard trapping the cleanup" — ALL false. The real cause is
> a splice-side span-boundary bug the model neither caused nor can clean via the obvious idiom.

**Symptom (trace):** in the "yes, proceed" turn a single `// Step 2:` comment line piled from 1 copy
to 16 across the iterations, the model issuing a "collapse the dupes" `edit_shader` each round and the
count GROWING by one each time, until `max_iterations` cut the turn. Every edit returned `ok —
compiled clean`; no oscillation/force-restore guard ever fired (the state grew monotonically, never
A→B→A). 0 `write_shader` calls, 136 `edit_shader` calls in the turn.

**Mechanism (real-code repro, byte-confirmed):**
- `glsl_lex` skips comments as **trivia** (`_skip_trivia`) — a comment produces ZERO tokens.
  `token_match` returns spans as `[first_matched_token.start, last_matched_token.end]`
  (`glsl_lex.py`, the `spans.append((hay[i].start, hay[i + k - 1].end))` line).
- So when an edit's `old_str` **LEADS WITH a comment line** above code (the model quoting the comment
  as context for an insert/cleanup), the matched span **starts at the first CODE token — the leading
  comment is byte-EXCLUDED from the span.** `_splice` (`backend.py`) replaces only the code span and
  re-inserts `new_str` (which reproduces that same comment line), while the file's ORIGINAL copy of the
  comment — sitting on the line just above the span — is never touched. Net: **+1 identical comment
  line.** Reproduced 1→2 with the real functions on the verbatim trace edit.
- **The `span_drops_comment` / `comment_loss` guard does NOT fire and does NOT help** — it inspects
  comments INSIDE `src[start:end]`; the duplicated comment is ABOVE the span, so the guard sees nothing
  and passes. (My earlier "the guard trapped the cleanup" theory was the opposite of the truth: the
  guard is blind here, it never engaged.)

**Why it reaches 16 (the two co-causes — NOT a pure engine bug):** a single comment-blind splice is
**self-limiting** — it gives +1 then STOPS, because the original `old_str` no longer token-matches once
the file has the extra copy (proven in repro: the next identical edit is a no-op). Climbing 2→16
required the model to keep **re-quoting the grown block as `old_str` to "clean it up"**, each cleanup
*also* leading with the comment, each re-triggering the +1. So:
- **Engine = ROOT cause:** it manufactures a duplicate the model didn't author (its `new_str` had ONE
  copy) and can't remove via the obvious leading-comment `old_str`.
- **Model = necessary co-driver of the escalation:** it never escalated to `write_shader` (the prompt's
  own "rewrite the whole block in ONE edit" idiom would have ended it instantly), and pattern-fired
  cleanup edits with empty reasoning text each iteration.
- **Guard layer = the missing interdiction:** the oscillation guard catches A→B→A cycles only; there is
  NO monotonic-growth / "the thing you're removing is still there, the file only grew" detector — so a
  file-growing quirk runs unbraked to `max_iterations`.

**The class:** an `edit_shader` whose `old_str` leads with (or, symmetrically, trails) a comment the
`new_str` reproduces gets a code-only span, so the splice DUPLICATES that comment. The honest fix is at
the span boundary (make the splice cover exactly what `old_str` quoted, so it can't duplicate), backed
by the Finding-2 churn brake as the systemic resilience net for any future file-growing quirk.

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

### Finding 4 — a forced turn-end mis-attributes the engine cap to the USER ("you're right to pause now")

**Symptom (maintainer report, 2026-06-16):** when a turn force-ends on `max_iterations`, the copilot's
closing reply says things like *"you're right to pause now"* — it does not understand that an ENGINE
threshold stopped it, not the user. It frames an involuntary cap as if the human chose to pause.

**Mechanism (grounded in source):** the `max_iterations` exhaustion path (`agent.py:875-890`) calls
`stream_final_reply()` — one no-tools closing stream. That function appends ONE generic nudge,
`_FINAL_REPLY_NUDGE` (`agent.py:38`): *"[engine] Tool budget exhausted for this turn. Reply to the
USER now…"*. It is used for BOTH turn-end causes (max_iterations at `:890` AND the per-iteration
budget-truncation at `:616`) and names no specific cause beyond "budget exhausted" — neutral enough
that the model rationalizes the stop as a natural pause and ascribes the agency to the user. Contrast
the **edit-giveup** path (`agent.py:845-849`), which DOES inject an honest cause-note (*"I've stopped
to avoid looping"*) and reads correctly. So the giveup terminal is already cause-honest; the
`stream_final_reply` terminal is not — and the new churn HARD-stop (Finding 2) routes through the same
forced-end machinery, so it would inherit the same misattribution unless fixed here.

**The class:** any engine-forced turn-end (max_iterations, the Finding-2 hard churn-stop, budget
truncation) must tell the model — in the closing-reply nudge — the SPECIFIC engine cause, so its reply
owns the stop ("I hit my own iteration/edit limit and stopped") instead of crediting the user with a
pause it didn't make. This is the same actor-model corollary the copilot is built on (the agent must
not narrate a fact it doesn't hold): it doesn't hold "the user paused me," so it must not say it.

---

## Out of scope

- **Auto-sampling several `t`s + a motion summary (the richer probe).** Letting the agent request a
  render at ONE chosen `t` IS in scope (Decision 4 / the new `probe_render` tool). What stays deferred
  is the engine auto-sampling a *spread* of `t`s and summarizing motion ("the flame rises from t=0→2s")
  without the agent asking — that's a larger capability.
  - **Trigger (deferred):** a dogfood/maintainer trace where the agent, given an aimable single-`t`
    probe, still can't answer "is the animation right" because it would have to guess which `t` to look
    at — i.e. it needs the engine to survey motion for it.
- **The model-bound render-facts-honesty half** (a cheap model claiming success against damning
  facts) — already a `todo.md` deferral, untouched here.
- **General copilot cost/latency (lazy tool catalogue, lever 2)** — separate `todo.md` deferral. NOTE:
  the new `probe_render` read tool adds one tool to the eager catalogue; size it minimally (it rides
  the lazy-load deferral when that lands).
- **Two existing `todo.md` deferrals that `probe_render` becomes the BUILDING BLOCK for — kept, NOT
  resolved, cross-referenced (pre-impl review):** "copilot scripting follow-on (043): render_video
  pixel-facts" (a first/last-frame fact pair) and "sticker render is always t=0..N" (the agent can't
  see motion past t=0). `probe_render(node, t)` is the primitive a future pass builds those on, but this
  wave does NOT auto-attach facts to `render_video` nor add an export-window offset — so leave both
  entries, add a one-line cross-ref to `probe_render` in each at impl.

## Design decisions

LOCKED (maintainer walkthrough 2026-06-16). Numbered for reference.

**Framing for Finding 1 — fix the SPAN BOUNDARY, not the parser, not a dedupe-after-the-fact.**
`glsl_lex` correctly treats comments as trivia (a GLSL compiler strips them; a lexer emitting comment
tokens would be wrong, and every other consumer of `glsl_lex` relies on a clean tokenizer). The bug is
that the copilot's edit matcher reuses that compiler-grade lexer to compute a SPLICE span, and the span
`[first_code_token.start, last_code_token.end]` **silently excludes a comment that `old_str` quoted at
its leading (or trailing) edge** — so the splice can't replace what the model said it was replacing, and
duplicates it. Per the conventions "structural impossibility over guard-piles" law, the fix makes the
splice cover **exactly the text `old_str` quoted**, so duplication is impossible BY CONSTRUCTION — not a
dedupe pass that cleans up after a splice that already misbehaved. The parser stays untouched.

1. **Span boundary swallows the comment `old_str` quoted (Finding 1, the engine ROOT).** When `old_str`
   begins with comment line(s) above its first code token, GROW the matched span START backward to
   include those leading comment lines (symmetrically, grow the END forward over trailing comment lines
   `old_str` quotes after its last code token), so `_splice` REPLACES them instead of leaving the
   file's originals above/below the span. Mechanics, locked (swarm-verified — a naive "grow over any
   leading comment" prototype FAILED; these constraints are load-bearing):
   - Grow only over comment lines that `old_str` actually reproduces at that edge (match `old_str`'s
     leading/trailing comment lines against the immediately-adjacent source lines, RAW per-line) — never
     swallow a comment `old_str` didn't quote.
   - Feed the `span_drops_comment` / `comment_loss` guard the **GROWN** span — else a leading-comment
     *deletion* (an `old_str` that quotes the comment, a `new_str` that drops it) becomes a silent
     comment loss the guard must still catch.
   - **Re-validate match UNIQUENESS after growing** — growing the span backward over a comment that
     recurs elsewhere can change whether the region is unique; the `>1 match → ambiguous` check must run
     on the grown spans, not the pre-grow ones.
   - The fix is at `token_match`'s span computation (or a wrapper in `apply_shader_edit` between
     `token_match` and `_splice`) — NOT in `glsl_lex`'s tokenization (comments stay trivia for every
     other consumer).
   Verified by real-code repro: with the grown span, the trace's iter-2 seed edit produces ONE comment
   (not two), a same-shape cleanup edit is a clean no-op, AND a legitimate comment-rename (old label →
   new label) still lands.

   **Why this alone is NOT the whole fix (the swarm's load-bearing finding):** Decision 1 kills THIS
   duplication, but a single comment-blind splice was already self-limiting (+1 then a no-op); reaching
   16 needed the model to keep re-quoting the grown block AND the absence of any monotonic-growth /
   no-progress detector (the oscillation guard catches A→B→A only). So the churn brake (Decision 2) is
   not a Finding-2 nicety — it is the structural backstop that makes any future file-growing quirk
   non-catastrophic. Recorded here so a reviewer doesn't cut Decision 2 as redundant once the span fix
   lands.

2. **Churn brake — TWO thresholds, both config + settings (Finding 2).** Two config fields on
   `CopilotConfig` (`config.py`), each `0 = off`, both surfaced in the Settings UI:
   - **soft** (`clean_edit_soft_streak`, default = today's 6): drives an ESCALATING per-FILE fact in
     the tool result ("N edits in a row on this file — finish in ONE write_shader"), repeating/growing
     louder each subsequent clean edit instead of firing once. Replaces the one-shot
     `clean_streak_nudge_sent` latch.
   - **hard** (`clean_edit_hard_streak`, default TBD at impl, e.g. 12): force-ends the turn after that
     many clean edits **on one file** (per-FILE, matching the single-file spiral the trace showed — NOT
     a per-turn total). The agent must return to the user rather than churn deeper. **Reuse the existing
     `giveup` machinery** (`agent.py` ~830-873: set the giveup/break, yield `AgentError`/turn-done with
     `_build_turn_summary` + stats) — do NOT invent a parallel turn-end path. The forced end MUST leave
     a well-formed turn (summary + stats present, no orphaned `tool_call_id`, clean trace/checkpoint),
     same as the giveup path.
   - **`write_shader` interaction:** `write_shader` is `is_edit=True`
     (`shader.py`), so it currently counts toward the streak — but the soft fact STEERS the model to
     "finish in ONE write_shader," and that corrective write must not itself be the straw that trips the
     hard turn-end. LOCKED: a `write_shader` (whole-file rewrite — the convergence escape hatch) does
     NOT increment the hard counter and RESETS the per-file streak (it's the sanctioned way to converge,
     not churn). `edit_shader` (the micro-edit) is what the brake counts. A regression test pins this:
     soft fact fires → model does one `write_shader` → turn does NOT force-end on that write.
   Per-file counting for the hard brake (the cumulative-per-turn counter today is replaced by a
   per-file map); `clean_streak_nudge_sent` latch is removed.

3. **Probe-render clock + an aimable probe (Finding 3).** Two parts:
   - **Auto-probe default → t=0.0.** The shader-edit auto-probe (`_render_facts_for` with `t=None`)
     defaults to **0.0**, not `glfw.get_time()`. Stable across edits (no phantom time-drift reading as
     an edit effect) and correlates with the export clock the user renders. The live-preview clock was
     rejected — unstable/unknowable headless.
   - **New `probe_render` read tool (the aimable probe).** A standalone, NON-mutating, UN-gated tool —
     `probe_render(node, t=0.0)` → the same one-line facts string the edit path produces, reusing
     `_render_facts_for(node, t)`. It is the READ-side counterpart to `render_image` (NOT a duplicate:
     `render_image` is `GatePolicy.ALWAYS` + `mutating=True` + produces a full-size deliverable FILE
     and pops a confirm dialog every call — wrong for the agent glancing dozens of times a turn). This
     is to `render_image` what `read_shader` is to `write_shader`. Gives the render-blind agent BOTH a
     stable automatic fact AND a deliberate "look at moment t" without touching the gated publish path.

4. **Cause-aware forced turn-end (Finding 4).** Every engine-forced turn-end must tell the model the
   SPECIFIC cause in its closing-reply nudge, so the reply owns the stop instead of crediting the user.
   - The generic `_FINAL_REPLY_NUDGE` (`agent.py:38`) gains a CAUSE clause. Cleanest shape: make the
     nudge a small builder `_final_reply_nudge(cause: str)` (or pass a cause string into
     `stream_final_reply`) so the max_iterations terminal passes an honest cause — e.g. *"[engine] You
     reached the per-turn ITERATION limit (not a user pause) — the turn is ending. Tell the user you hit
     your own limit and stopped, state the net file state + what's left, ask if they want you to
     continue."* The wording must explicitly deny user-agency ("not a user pause").
   - The Finding-2 HARD churn-stop reuses the giveup machinery, which ALREADY injects an honest cause
     note (`agent.py:845`) — so its note must read the same way ("I hit my own N-edit limit on this file
     and stopped"), NOT "you paused me." One shared honesty bar across all forced-end terminals.
   - The budget-truncation terminal (`agent.py:616`) shares `stream_final_reply`; it gets its own honest
     cause ("[engine] per-turn token budget reached"). No terminal stays cause-blind.
   This is the actor-model corollary (the agent must not narrate a fact it doesn't hold): it does not
   hold "the user paused me," so it must not say it. NO new config — pure wording/plumbing.

## Files touched

(Anticipated — confirm at impl.)
- `shaderbox/copilot/glsl_lex.py` — `token_match` span-grow over `old_str`'s leading/trailing comment
  lines (Finding 1); the `span_drops_comment` guard must receive the GROWN span (no signature change to
  the guard itself, just the span it's fed).
- `shaderbox/copilot/backend.py` — `apply_shader_edit` matcher path: feed the GROWN spans to the guard +
  `_splice`, re-validate `>1 match` uniqueness POST-grow (Finding 1); `_render_facts_for` default clock
  → 0.0 (Finding 3); a `probe_render` capability method backing the new tool (Finding 3).
- `shaderbox/copilot/agent.py` — churn brake: per-file clean-edit map, escalating soft fact, hard
  turn-end via the existing `giveup` path; remove the `clean_streak_nudge_sent` latch (Finding 2). ALSO
  the cause-aware forced turn-end: `_FINAL_REPLY_NUDGE` → cause-parameterized builder, an honest cause
  at each of the three forced-end terminals (`:616` budget, `:845` giveup/hard-stop, `:890`
  max_iterations) (Finding 4).
- `shaderbox/copilot/config.py` — `clean_edit_soft_streak` + `clean_edit_hard_streak`; **`apply_user_limits`
  signature** (fixed-kwarg) updates for the renamed/added fields (Finding 2).
- **`shaderbox/exporters/integrations.py`** — `CopilotIntegration` field(s) (the persisted, `extra="forbid"`
  model) + `apply_limits` call; the rename means the dev box's `integrations.json` carries a stale
  `max_clean_edit_streak` key. **`integrations.json` lives at `app_data_dir()` (global, NOT git-tracked, NOT
  in `projects/dev/`)** — so the no-migration "hand-fix the sandbox" rule applies to the maintainer's LIVE
  box: hand-edit (or let it fail-soft reset) the stale key; nothing to `git add` (Finding 2 / review-blocker).
- `shaderbox/popups/settings.py` — surface the two churn thresholds in the `_COPILOT_LIMITS` table; a
  wiring test must prove the table-row→field binding (by-name string) actually reaches `COPILOT_CONFIG`
  and changes loop behavior — not just renders (Finding 2 / dev_flow "config never connected").
- `shaderbox/copilot/tools/shader.py` — `_EDIT_SHADER_DESC` same-file batch steer (Finding 2), if taken.
- `shaderbox/copilot/tools/inspect.py` (NEW, or `publish.py`) — the new `probe_render` `ToolDefinition`
  (`mutating=False`, `gate_policy=GatePolicy.NONE`, modeled on `read_shader`). LEANING toward its own
  `inspect.py` (it's a read/inspect tool, not publish); confirm at impl.
- `shaderbox/copilot/backend.py` / `capabilities.py` — a `probe_render(node: str, t: float)` capability
  method (resolve the node STRING → `Node` via `_copilot_render_target`, bridge `run_on_main`, then
  `_render_facts_for(node, t)`) + the `CopilotCapabilities` Protocol decl (Finding 3). NOTE: the
  capability takes a node string; `_render_facts_for` takes a `Node` — the resolution + bridge wrapper
  is the real new code, not a thin passthrough.
- `shaderbox/copilot/prompt.py` — teach the agent `probe_render` as a FREE read-only look (distinct from
  the gated heavy `render_image`) + the t=0 auto-anchor; reconcile with the existing `render@t=Xs`
  facts-format explainer (Finding 3).
- `scripts/dogfood/analyze.py` — add `probe_render` to `CANONICAL_TOOLS` (else `test_tool_registry.py`
  reds).
- `tests/_caps.py` — add `probe_render` to the `_FakeCaps`/`minimal_caps` Protocol fake (else every
  registry-building test fails to construct caps).
- Tests under `tests/` — a regression test per finding (see Manual verification), AND the existing
  `test_copilot_user_limits.py` (`_LIMIT_FIELDS` + defaults/floors/round-trip/zero-disables) +
  `test_copilot_loop.py::test_clean_edit_streak_nudges_once` are REWRITTEN (the field rename + the
  once→escalating change break them).

## Manual verification

Each check must fail for exactly one reason; name the falsifier.

- **Finding 1 (comment duplication) — the headline regression.** A unit test on the REAL functions
  (`token_match` + `_splice`, the exact repro the swarm ran): source with ONE `// Step 2` above a
  `float heat …` line; `old_str` LEADS with that comment, `new_str` reproduces the comment + inserts a
  block; assert the result has the comment **exactly once** (1, not 2). Falsifier: revert the span-grow
  → red (result has 2 copies). Anchor: the verbatim iter-2 edit from the reference transcript.
- **Finding 1 the spiral can't climb (the convergence half).** Replay the trace's *cleanup* edit on a
  2-copy file (old_str leads with the comment, quotes both, new_str collapses to one): assert the
  result has ONE copy and a re-run of the same edit is a clean no-op (matches→1 then 0, never growing).
  Falsifier: revert → the count climbs 2→3.
- **Finding 1 legitimate comment-edit still lands (the false-positive half).** A comment-RENAME edit
  (old_str leads with `// old label`, new_str leads with `// new label`) replaces the label exactly
  once (no leftover old label above the span, no duplicate). Falsifier: a span-grow that swallows a
  comment `old_str` didn't quote → the rename leaves a stray line → red.
- **Finding 1 guard still protects real loss + uniqueness re-checked (the missing halves).** (a) A
  leading-comment DELETION (old_str quotes the comment, new_str drops it) where the comment is UNIQUE is
  still refused `comment_loss` (the guard is fed the GROWN span). (b) Growing a span backward over a
  comment that recurs elsewhere is re-validated for uniqueness — a now-ambiguous match returns
  `>1 matches`, not a silent wrong-region splice. Falsifier (a): feed the guard the pre-grow span → the
  unique comment is silently lost. Falsifier (b): skip the post-grow uniqueness re-check → a wrong
  region is spliced.
- **Finding 2 (churn brake).** A test that drives N `edit_shader` calls in one turn on the same file
  and asserts (a) the escalating per-file fact appears + grows past the SOFT threshold, and (b) the
  turn force-ends at the HARD threshold via the `giveup` path (a well-formed turn-done with summary +
  stats, no orphaned `tool_call_id`). Falsifier: set both thresholds to 0 (off) → no fact, no stop.
- **Finding 2 `write_shader` escape hatch (the missing half).** A test that after the soft fact fires,
  ONE `write_shader` does NOT trip the hard turn-end and RESETS the per-file streak. Falsifier: count
  `write_shader` toward the hard counter → the corrective write force-ends the turn → red.
- **Finding 2 Settings wiring.** A test that setting the two thresholds through `CopilotIntegration(...)
  .apply_limits()` reaches `COPILOT_CONFIG` AND changes loop behavior (not just that the table renders)
  — the row→field binding is a by-name string, an easy silent miss. Falsifier: break the name binding →
  the config value doesn't change → red.
- **Finding 3 (probe clock + tool).** (a) A test asserting `_render_facts_for(node)` (shader path,
  `t=None`) stamps `t=0.0`. **Falsifier MUST monkeypatch `glfw.get_time` to return a NON-ZERO value
  (e.g. 951.9, the trace's clock)** — headless `glfw.get_time()` already returns 0.0, so without the
  monkeypatch the test false-greens (passes before AND after the fix). With the monkeypatch: unfixed
  stamps `t=951.9s`, fixed stamps `t=0.0s`. (Stub `node.render`/`render_facts` so no GL is needed —
  sidesteps the V3D cross-test flake.) (b) A test that `probe_render(node, t=2.5)` returns a facts line
  stamped `render@t=2.5s` and is non-mutating + un-gated (does NOT fire the render gate). Falsifier:
  route it through the gated `render_image` path → the gate fires.
- **Finding 3 explicit-`t` path survives the default flip (the missing half).** A test that
  `_render_facts_for(node, t=2.5)` STILL stamps `t=2.5s` after the default changes to 0.0 — so
  `_script_render_line`'s own-`t` probe is unbroken. Falsifier: make the flip clobber the explicit
  path → red.
- **Finding 4 (cause-aware forced turn-end).** A test that the final-reply request at each forced-end
  terminal carries a cause clause naming the ENGINE limit (the request `messages` sent to the closing
  stream contain the cause string + "not a user pause"), and that the giveup/hard-stop note reads
  "I … stopped" not a user-pause framing. Falsifier: revert to the generic `_FINAL_REPLY_NUDGE` → the
  cause string is absent → red. (Assertable on the built request `messages` — no live model needed.)
- **`make check` + `make smoke`** green. NOTE: `make smoke` runs NO agent turn — it does not exercise
  the edit / churn / probe path AT ALL (the copilot UI opens only as a no-key gate). The unit tests
  above are the SOLE gate; smoke green proves only that nothing else regressed.
- **No live App manual check needed** — these are headless-testable engine behaviors. A maintainer
  re-run of a similar "build me an effect step by step" session is the real-world confirm but not a
  gate.

## Open questions for the user

(All resolved at the 2026-06-16 walkthrough — see Design decisions. Scope is FOUR findings; Finding 4,
the "you're right to pause now" misattribution, was added by the maintainer. Finding 1 fix = SPAN-GROW
over `old_str`'s leading/trailing comment so the splice can't duplicate (NOT the earlier dedupe/guard-
relax — that was refuted); churn brake = two thresholds soft+hard, both in config + Settings, per-file;
probe clock = auto-probe default t=0.0 PLUS a new ungated `probe_render(node, t)` read tool; forced-end
honesty = cause-aware closing nudge at every forced-end terminal.)

One detail left for impl (not a blocker): the HARD threshold's default value (e.g. 12) and the
`probe_render` tool's final home/module (`tools/publish.py` vs. its own `tools/inspect.py`).

## Investigation history

**Finding 1 mechanism — THREE refuted theories before the real one (2026-06-16).** This is recorded so
the wasted-loop pattern isn't repeated. Each theory "sounded right" from reading source; none survived
the trace.
1. *"The model wasn't sure its edit landed, so it re-issued the same insert."* REFUTED — the tool result
   said `ok — compiled clean` plainly; no ambiguity. Fabricated premise.
2. *"The matcher silently duplicates because comments are trivia; the `comment_loss` guard then traps the
   cleanup."* REFUTED — the cleanup edit returned `ok: True` in the trace; the guard NEVER fired.
3. *"Model sloppiness — it re-emitted from a slightly-off memory."* REFUTED — the model's `new_str` had
   exactly ONE comment; it never authored the duplicate.

   The REAL mechanism (verified by a real-code repro importing `token_match` + `_splice`, then an
   adversarial 2-agent swarm): the matched span is `[first_code_token, last_code_token]`, so a comment
   `old_str` quotes at its LEADING edge is byte-excluded; `_splice` re-inserts `new_str`'s copy while the
   file's original survives above the span → **+1 per edit.** Self-limiting alone (next identical edit is
   a no-op); it climbed to 16 only because the model kept re-quoting the grown block to clean it (each
   cleanup re-triggering +1) AND no monotonic-growth detector exists. The swarm also proved a NAIVE
   span-grow FAILS — hence Decision 1's three load-bearing constraints (grow only over reproduced
   comments; feed the guard the grown span; re-validate uniqueness post-grow). Swarm agent IDs:
   `ab2b1c1ec2112a7d9` (repro-confirm), `ae040ddea6ee66116` (devil's-advocate, SOUND-BUT-INCOMPLETE →
   surfaced the model+guard co-causes now in the finding).

**Earlier pre-impl review (2026-06-16, 2 agents) — NOW PARTLY SUPERSEDED.** That review ran against the
*dedupe/guard-relax* version of Decision 1 (theory 2 above) and returned PASS-WITH-FIXES; its
Decision-1/2 findings are obsolete (the approach was replaced). What SURVIVES from it and is still folded
in: the config-rename 4-site ripple + broken `test_copilot_user_limits.py` / `test_clean_edit_streak_
nudges_once` (Decision 2); `probe_render` breaking `test_tool_registry.py` + needing `tests/_caps.py`
(Decision 3); the t=0 test false-green needing a non-zero `glfw.get_time` monkeypatch; the
explicit-`t`-survives-flip + Settings-wiring tests; the smoke-runs-no-agent-turn correction. The
re-derived Finding 1 should get a FRESH pre-impl review pass (its mechanism changed entirely).

**Finding 4 added by the maintainer (2026-06-16).** A forced turn-end ("you're right to pause now")
misattributing an engine cap to the user; folded in because it shares the Finding-2 hard-stop's
forced-end machinery (one honesty bar across all forced-end terminals). Grounded in source:
`_FINAL_REPLY_NUDGE` is the single cause-blind nudge used by the `max_iterations` and budget terminals;
the giveup terminal is already cause-honest.

**Implementation + dogfood (2026-06-16).** All four findings landed in one diff (F1 span-grow in
`glsl_lex.py`, F2 churn brake in `agent.py`/`config.py`/`integrations.py`/`settings.py`, F3 probe clock
+ the new `probe_render` tool in `tools/inspect.py`, F4 cause-aware nudges in `agent.py`). A targeted
dogfood (`ai_docs/features/050_dogfood_report_stress.md`) confirmed ALL FOUR live against
gpt-5.1-codex-mini: the exact spiral edit shape stays at 1 comment (create + cleanup paths), the soft
escalating fact made the model self-limit, a threshold-lowered turn fired the hard force-end at exactly
the cap, `probe_render` was called at 3 chosen `t`s, and the hard-stop reply owned the stop ("I hit my
own limit ... NOT a pause you asked for").

**Post-impl review (2026-06-16, 3 adversarial agents in parallel — F1-correctness anchored to the real
trace, F2/F4+blast-radius, F3+conventions).** All returned PARTIAL/PASS, no FAIL, no ship-blocker; each
ran real code (repros + `make check` + the test files). 8 findings, ALL real (zero false positives),
all addressed in one fix wave:
- F1 block-comment `/* */` edge not grown (MEDIUM): documented as a known limit (agents emit `//`
  only) + a pinning test, not full block-comment span logic.
- F2 per-file key splits on current-node aliasing (`""` vs node-id for the same node — MEDIUM): a
  delays-not-disables edge on an artificial alternating pattern the real model doesn't produce;
  documented in `_edit_target_key` rather than threading a resolver through `run_turn`.
- F3 clock test false-greened (headless glfw is also 0.0 — MEDIUM): rewritten to introspect the literal
  `0.0` default + assert no `glfw` reference (now fails on a wall-clock reintroduction).
- LOW: mislabeled guard test rewritten to actually make the guard fire; trailing-comment-grow test
  added; `assert last.summary` tautology replaced with a concrete-field assert; write-with-errors reset
  behavior documented; `_PROBE_RENDER_DESC` trimmed 605→405 chars.
Convergence: round 1 had zero false positives (productive, not converged-by-noise), all findings minor
or test-quality with the 3 core mechanisms PASSing; a second full round was skipped to avoid late-round
fabricated gaps (per `review-agent-loop`). Reviewer IDs: `abbcc4e29adcd104c` (F1), `ad2557e82a061bef9`
(F2/F4), `a89a58cce1aa116af` (F3/conventions). `make check` + `make smoke` green; suite passing.

Status: DONE (pending commit). The `todo.md` edit-churn-brake deferral is subsumed and must be deleted
in the resolving commit.
