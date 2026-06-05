# 020·28 — Prompt tier architecture: NL-only history + read-result collapse

The copilot's message HISTORY is misused: `_commit_turn` appends the FULL per-turn tail — every
assistant-with-tool_calls message and every tool result, including `read_shader`'s complete ~4,850-token
source listing — and never dedups or evicts it. Measured: the SAME shader source appeared **10×** in one
session's history; per-turn input averaged ~25k tokens with tails to 117k/215k/231k. Supersedes the
verbatim-tail-append of 020·23 D4 at the commit boundary.

This is the **structure decision** (the maintainer's ordering: structure first → use-case analysis informs
it → a later retrospective shader-representation pass). Derived from an intent-driven swarm analysis +
verification against the real code. **No backward compatibility** — old persisted conversations are reset,
not migrated.

## Goal

History carries **natural-language only**: user messages + one engine-derived NL turn-summary per committed
turn. NO tool_call/tool messages survive a turn. The full source the agent reads is **never persisted** —
it is live-refetchable, and a stale copy is worse than no copy. Prompt assembly moves from a flat
string-concat to a **block constructor** (named blocks, volatility-ordered) so a block can be refactored in
isolation.

## The load-bearing finding (intent → memory model)

Every intent's cross-turn STATE need classifies as: **(a) live-state** (re-fetched this turn — source,
uniforms, project map, integration status), **(b) past-action memory**, or **(c) the conversation**. The
analysis result: almost everything is (a); the only durable need is (c) the conversation PLUS the semantic
OUTCOME of past mutations carried as NL. **Zero** future-turn intents need a verbatim past tool RESULT —
that's needed only WITHIN a turn (the provider's tool-pairing protocol). So: **full source does not belong
in history at all.**

Four facts the NL turn-summary MUST preserve (each pins a real corpus failure moment):
1. **verb + every node NAME touched OR referenced this turn** — referent binding ("it", "that shader", "do
   the same to C"). NOT just the mutated target: a cross-shader "match B to A" must record A's name too
   (else next turn's "do the same to C" loses what "the same" referenced — recoverable via live re-read only
   if A is named). Sourced from the tool-call args (node/target/nodes) across the turn's `_RunLog`.
2. **the NEW value of a mutation** ("set u_text_spacing.y = 2.5 on Hello World") — the "not so much"
   dial-back. NEW value only, NOT old→new (`set_uniform` captures no baseline; the next turn re-reads live).
3. **the agent's stated ASSUMPTION** — the "no, center is bot-left" correction. Carried by keeping the
   agent's final reply (`text_buf`) verbatim — **but ONLY at the clean-done terminal**; at a cutoff / giveup
   / cancel, `text_buf` is the last (often tool-calling, empty) iteration's text and is NOT the reply, so
   the summary prose there comes from the branch's `note` / `error_text`, never stale `text_buf`.
4. **failures/retries + an irreversible-action LEDGER** — "continue" after a cutoff must not re-publish.
   **The ledger MUST carry the action's IDENTITY (node id, pack name, URL), which lives in the tool's
   `payload`, NOT its `msg`** (e.g. `publish_telegram`'s `msg` is just "published to Telegram" — the pack is
   in payload; `delete_node`'s `msg` has the name but the id is in payload). So `_RunLog.record` MUST also
   capture `payload`, and `summary_lines` reads identity from it. Wired into ALL FIVE terminal paths (done +
   length-cutoff + giveup + max_iterations + **cancel**). Irreversible-action lines (publish/delete) are
   kept verbatim + uncapped (safety); non-irreversible mutating lines are capped/compressed (verb+target,
   drop the `msg` prose) so a 50-tool-call turn can't bloat the ledger unboundedly.

## Out of scope

- **Within-turn read de-dup** (the scratchpad's only present-day job): the freshness guard forces a re-read
  before each edit, so a multi-iteration edit turn emits N full source copies *within* the single turn —
  the driver of the 117k–231k intra-turn peaks. A node-keyed, overwrite-by-key bucket would emit the source
  once. **Deferred** — it touches the live loop + `head_len` slicing; the commit-collapse win is independent
  of it. **Trigger:** when the within-turn peaks remain the top cost after this lands.
- **Reasoning-notes scratchpad bucket** — the proper home for the agent's live coordinate assumption (fact
  #3's real fix). **Trigger:** when CORRECTION/COORDINATE regresses because the assumption wasn't in
  `text_buf`, or when reasoning/CoT is implemented — it becomes the first PER_TURN bucket member.
- **Shader-representation tuning** (020·27 STRUCTURE block content/size — structural vs raw listing). The
  maintainer's explicitly-named LATER retrospective pass. **Trigger:** after this tier structure is stable.
- **UI color-coding** of message vs scratchpad vs tool-call in the chat. **Trigger:** when the chat UI is
  revisited. A `/imgui-ui` concern.
- **Cross-shader derived-edit memory.** Fact #1 records every node *named* in a turn, but the *content* the
  agent derived an edit from (e.g. "match B's grain to A" — A's specific source) is gone next turn (source
  never persisted). Recoverable: the agent re-reads A live IF A's name survived (fact #1 ensures it does).
  Acceptable — a live re-read is correct-by-construction. **Trigger:** if a trace shows the agent failing a
  "do the same to C" because it can't reconstruct what "the same" was even with A's name in hand.

## Design decisions

1. **History = NL only.** `_commit_turn` appends `user` + exactly one `assistant` NL turn-summary. Never a
   tool_call/tool message. The full tail is consumed only to DERIVE the summary, then discarded.

2. **Within a turn, the live loop is UNCHANGED.** `messages[head_len:]` still carries full assistant+tool
   pairs during the turn (the provider 400s on an orphaned tool_call_id). NL-only applies ONLY at the commit
   boundary. So `_turn_tail`'s orphan-drop + the within-turn protocol stay intact.

3. **The turn-summary is engine-derived, deterministic, no extra LLM call.** Built from the reply prose
   (verbatim `text_buf` at clean-done; the branch `note`/`error_text` at a cutoff/giveup/cancel — fact #3) +
   a structured ledger from `_RunLog` (mutating calls + the new value from `msg` + the identity from
   `payload` + failures — facts #1/#2/#4). A `TurnSummary` value object on the agent module; ALL FIVE
   terminal events (AgentTurnDone, length-cutoff AgentError, giveup AgentError, max_iterations AgentError,
   **AgentCancelled**) carry it instead of `messages: list[LLMMessage]`. `_RunLog.record` is extended to
   `(name, ok, msg, payload)`; the dead `executed_actions_note` + the never-read `AgentTurnDone.note` are
   removed (verified: no consumer reads `.note`).

4. **Block constructor (Q3).** `prompt.py` gets `Volatility` (STATIC < RARE < DIALOGUE < PER_TURN) and a
   frozen `PromptBlock(name, volatility, render: Callable[[], list[LLMMessage]])`. `build_prompt(blocks)`
   sorts by volatility, calls each render, drops `[]`-yielding blocks, flattens. `render -> list[LLMMessage]`
   (not str) so dialogue (many messages), a singleton (one), and an empty scratchpad (`[]`, vanishes) are
   ONE mechanism, no special case. `build_messages` composes `[static, project_context, dialogue, pending]`
   (the scratchpads slot is reserved, empty this phase — a PER_TURN member sorts below DIALOGUE
   automatically, which is the cache-correct physical placement AND the resolution of "scratchpads above
   dialogue": logical tier rank ≠ physical position).

5. **Cache: NL-only history makes the dialogue tier byte-stable turn-over-turn**, so the cacheable prefix
   grows monotonically instead of being shifted every turn by a fresh full-source tail. The within-turn ~6×
   cache discount (static+context prefix) is preserved. (Honest limit: the project-context tier carries
   `is_current`/`HAS ERRORS` marks that flip on tree mutations — pre-existing, not worsened.)

6. **Persistence: NL-only, no migration — the reset is ACTIVE, not incidental.** `_HistoryModel` drops its
   `tool_call_id` + `tool_calls` fields and `from_runtime`/`to_history` stop mapping them. Combined with the
   existing `extra="forbid"`, an old v4 store carrying those keys raises `ValidationError` → the existing
   `load_and_migrate` except-branch returns an empty store (the desired fail-soft). `_VERSION` bumps 4→5 so
   the rejection is intentional, not a side effect. **The dev sandbox `projects/dev/copilot/conversation.json`
   is a v4 store with tool messages — it MUST be archived + reset in the same wave** (`make smoke` loads it
   via `App._init`); the one-time fail-soft WARNING on first load is expected. `_split_turns` stays (it still
   groups user+summary into whole turns for the window trim) but its orphan-pairing concern + the
   `_MIN_KEPT_TURNS` comment citing tool-pairing are updated (no tool messages to orphan).

7. **`_trim_history` stays but rarely fires.** With NL-only history the per-turn footprint is ~one user +
   one summary, so history grows slowly. The window trim (keep last N turns under `max_input_tokens`)
   remains as the overflow valve for very long sessions; it now evicts whole NL turns, never source.

## Files touched

- **`copilot/prompt.py`** — add `Volatility`, `PromptBlock`, `build_prompt`; rewrite `build_messages` to
  compose named blocks. `_trim_history` simplified (NL-only; drop the orphan-pairing concern).
- **`copilot/agent.py`** — add `TurnSummary` value object + `_build_turn_summary(reply_prose, run_log,
  registry)`. ALL FIVE terminal yields carry `TurnSummary` instead of `_turn_tail(...)`. `_RunLog.record`
  extended to `(name, ok, msg, payload)`; new `summary_lines(registry)` reads value from `msg` + identity
  (id/pack/url) from `payload`, irreversible lines verbatim+uncapped, others capped. Remove `_turn_tail`,
  `executed_actions_note`, and `AgentTurnDone.note`. Terminal-event `messages` field → `summary` field.
- **`copilot/session.py`** — `_commit_turn(user_text, summary, error_text)` appends `user` + one NL assistant
  message built from `summary` (safe empty default for the bare-except AgentError fallback). Drop the tail
  loop + orphan handling. Read `.summary` off all terminal types (incl. AgentCancelled).
- **`copilot/persistence.py`** — drop `tool_call_id`/`tool_calls` from `_HistoryModel` +
  `from_runtime`/`to_history`; bump `_VERSION` 4→5; the existing except-branch fail-softs an old store to empty.
- **`copilot/prompt.py`** — `Volatility` + `PromptBlock` + `build_prompt`; rewrite `build_messages`; keep
  `_split_turns` (update its tool-pairing comment + the `_MIN_KEPT_TURNS` comment).
- **`projects/dev/copilot/conversation.json`** — archive + reset (it's a v4 store with tool messages that
  `make smoke` loads; same wave, `git add projects/dev`).
- **`tests/`** — new `test_turn_summary.py` (facts: tweak value lands; assumption via reply prose at done +
  via note at cutoff; irreversible ledger lands WITH pack/id from payload on a cutoff; failure note on
  giveup; a mid-batch CANCEL yields a sane partial ledger). new `test_prompt_blocks.py` (ordering, empty-drop,
  render→messages). **new 2-turn CONSUMPTION test** (the load-bearing invariant the reviewers flagged unverified):
  turn 1 mutates + commits; turn 2 runs with that history + a fake client asserted to RECEIVE the NL summary
  carrying the target name + new value in its `messages` — proves a summary is resolvable, not just that it
  was built. **Rewrite (not just `.summary`-rename):** `test_copilot_loop.py::test_terminal_carries_tool_trajectory_for_history`
  (its premise — the tail IS the trajectory — is the thing being removed) + `test_conversation_persistence.py`'s
  tool-paired round-trip fixture. Re-run `scripts/copilot_gate_check.py` (a run_turn consumer; verified safe
  but in-family). `test_edit_safety.py` / `test_cross_project_tools.py` / `test_line_editing.py` assert only
  on `AgentToolCard`/`AgentError.message` — unaffected.

## Manual verification

Headless:
- `test_turn_summary.py` — the four facts (above).
- `test_prompt_blocks.py` — `build_prompt` ordering + empty-drop + flatten.
- A loop test asserts that after a `read_shader`-heavy turn, `session.history` contains NO message with
  `tool_calls` or `role=="tool"`, and no `read_shader` source listing — only `user` + NL `assistant`.
- `make check` + `make smoke`.

Maintainer (live): run `make run`, do a multi-turn edit session (animate → "not so much" → "no, center is
bot-left" → publish), confirm the agent still resolves the dial-back, the correction, and doesn't re-publish
on a "continue". Then check a trace: history should be NL-only, and per-turn input tokens should drop sharply
vs the measured ~25k baseline on read-heavy turns.

## Open questions for the user

- **Q-summary-form:** the turn-summary is engine-built deterministic prose (verb + target + new-value +
  failures, from `_RunLog`) PLUS the agent's verbatim `text_buf` reply. Alternative: ask the MODEL to emit a
  one-line summary as its final reply (it already does, roughly). Lean: engine-built ledger + verbatim reply
  — deterministic, no extra tokens, correct-by-construction (a free-form model summary could omit the
  irreversible ledger). Confirm.

## Review history

**Swarm Verify phase:** all 3 candidate designs sound-with-fixes; the fatal "write-only scratchpad"
assumption caught + removed (no cross-turn current-code scratchpad this phase — verified: `build_messages`
is a single per-turn call, so a source scratchpad would render empty/write-only).

**Pre-impl review (3 agents: correctness/design + verification/blast-radius + spec-fidelity).** Verdicts
LAND_WITH_FIXES / LAND_WITH_FIXES / FAITHFUL_WITH_GAPS; no SHOULD_NOT_LAND. The core thesis ("NL-only history
is sufficient; zero future-turn intents need a verbatim past tool result") was independently verified TRUE
against the code (history is consumed only by build_messages→stream; no cross-turn read of tool semantics).
Findings folded in:
- **F1 (HIGH):** the irreversible ledger can't be built from `_RunLog.msg` for `publish_telegram`/`delete_node`
  (pack/id live in `payload`). → `_RunLog.record` captures `payload` (Decision 3 / fact #4).
- **F2 (HIGH):** `AgentCancelled` also carries a tail + commits on user-Stop. → converted too (5 paths, not 4).
- **F3/CRITICAL (both reviewers):** persistence "reset" is NOT automatic — `_HistoryModel`'s optional tool
  fields load an old store straight through. → actively drop the fields + bump `_VERSION` (Decision 6); the
  v4 dev-sandbox `conversation.json` is archived+reset in-wave.
- **Verification gap (HIGH, both):** nothing proved a summary is *resolvable* next turn (every test checks
  content in isolation; no test feeds history into a 2nd `run_turn`). → the 2-turn consumption test added.
- **F5 (MED):** `text_buf` is stale at cutoff/giveup → reply prose comes from `note`/`error_text` there (fact #3).
- **Ledger unbounded (MED):** → irreversible lines verbatim+uncapped, others capped (fact #4).
- **F4/F6 (LOW):** `executed_actions_note` + `AgentTurnDone.note` are dead → removed; `_MIN_KEPT_TURNS`
  comment corrected.
- **Cross-shader limitation (MED):** documented as an Out-of-scope deferral with a trigger.

Post-impl review pending after implementation.
