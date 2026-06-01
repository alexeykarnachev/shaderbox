# 020 · 12 — edit robustness (Slice-1 self-correction completion)

A focused robustness slice closing a real failure observed in the live app: the model sent an
`edit_shader` `old_str` whose **whitespace** didn't match the source (6-space indent vs the source's
4-space), the exact-substring match correctly failed, and the agent then **retried the same failing
edit 11 times** until it hit `max_iterations=12` — burning ~$0.03 and ending in a **silent dead-end**
(12 `edit_shader: failed` cards in the chat, no message, no signal the turn was over).

Diagnosed from the new copilot transcript (`copilot_dev_2026-06-01_12-07-46.transcript`): the model
fabricated the indentation rather than copying the exact bytes `get_current_shader` had just returned
that same turn. It's a model-fidelity miss, but three tooling gaps turned a one-line miss into a
12-iteration disaster — this slice fixes all three.

This is **finishing Slice 1's own contract**, not new scope: the self-correction cap was specified at
`11_capability_wave_spec.md §I2` (*"a soft per-edit retry budget (`config.max_edit_retries`, ~3)…
After N failed compile-fix attempts… the agent stops and surfaces the error rather than looping to the
hard ceiling"*) but never implemented — `max_edit_retries=3` is dead config (defined in `config.py`,
read nowhere).

---

## Goal

Three fixes (A/B/C), each independently valuable, all in the "what happens when an edit keeps failing"
cluster:

- **A — Near-miss feedback in `edit_shader`.** When `old_str` doesn't match, stop returning a bare
  "not found — re-read". Detect the common case (a unique region that matches *ignoring whitespace*)
  and return the **exact source bytes** the model should have used, so the next attempt is a COPY, not
  another guess. (Superseded as the primary whitespace fix by feature 020 · 13's token matcher, which
  makes the whitespace-divergent edit *succeed* at the match layer instead of needing a re-copy; A's
  hint is now the no-op fallback on a true 0-match. See `13_glsl_lexer.md` Design 4.)
- **B — Enforce `max_edit_retries` (§I2).** Track consecutive failed `edit_shader` calls in the agent
  loop; after `config.max_edit_retries` (3), STOP attempting edits this turn and end with a final
  explanation rather than looping to `max_iterations`. Wires the dead config.
- **C — Surface turn-end cutoffs in the chat.** On BOTH the edit-retry giveup (B) and the
  `max_iterations` ceiling, emit a user-facing message into `state.messages` so the chat shows a clear
  warning AND the user knows the turn ended. Fixes the silent dead-end. The model's own final context
  reflects the giveup so it explains rather than vanishes.

## Out of scope (each with a trigger)
- **Auto-normalizing the match (apply an edit on a whitespace-insensitive match).** Rejected: it mutates
  matching semantics and risks applying the WRONG edit when whitespace difference is meaningful. A is a
  *hint*, not a silent fuzzy-apply. **Trigger:** if echo-the-bytes (A) proves insufficient in practice
  (the model still can't copy), reconsider a gated fuzzy-apply with explicit confirmation.
- **A general fuzzy/diff matcher (Levenshtein, longest-common-substring ranking).** A targets the ONE
  observed failure mode (whitespace-only divergence) with a cheap normalized compare. **Trigger:** a
  second distinct near-miss mode shows up in a trace (e.g. the model drops a comment line).
- **Raising `max_iterations` or making the limits user-tunable.** The limits stay `CopilotConfig`
  constants (§I3). **Trigger:** a legitimate multi-edit task routinely needs >12 iterations.
- **Per-tool retry budgets beyond edit.** Only `edit_shader` has the retry-spiral problem (the read
  tools are idempotent). **Trigger:** a future mutating tool develops its own spiral.

## Design decisions
*(numbered, lock-in only)*

1. **A computes the hint App-side, where the source lives.** The match runs in
   `app.py::_copilot_apply_shader_edit` (`_on_main`, has `src`). On `matches == 0`, before returning,
   compute the near-miss: normalize both the source and `old_str` by collapsing runs of horizontal
   whitespace (and stripping per-line leading/trailing), search for a UNIQUE normalized match; if found,
   carry back the EXACT original source slice for that region. The tool layer (`shader.py`) puts it in
   the error string. No fuzzy *apply* — only a fuzzy *locate* for the hint.

2. **`EditResult` gains a `hint: str` field (leaf type, cycle-free).** `EditResult` is a frozen leaf
   dataclass in `capabilities.py` (Seam A — no App/imgui/moderngl types). A plain `str` hint (empty when
   none) respects that. The handler returns it appended to the existing `error: old_str not found…`
   message: `…copy an exact substring.\nClosest match (copy this EXACTLY):\n<bytes>`.

3. **B counts CONSECUTIVE failed edits, reset on any success or non-edit tool.** A loop-local counter
   (like `_RunLog`, §T2 — never on state). Increment on an `edit_shader` call returning `ok=False`;
   reset to 0 on a successful edit or any other tool. At `>= max_edit_retries`, break the loop with a
   giveup. This is "N failures on the editing effort", matching §I2's intent (the model is stuck on the
   edit), not "N total tool failures".

4. **The giveup + the max_iterations cutoff both surface via a dedicated cutoff path → `AgentError`.**
   `session._apply_event` already renders `AgentError` as a distinct error bubble in the chat and ends
   the turn (`_finish_turn`). So both cutoffs yield an `AgentError` with a plain-language message
   ("I couldn't apply that edit after 3 tries — the substring kept not matching. …" / "Stopped after 12
   steps without finishing."). This reuses the existing surface; no new event type, no `state.py`
   change. The message is also appended to `history` as the assistant's turn so the NEXT turn's context
   shows the agent gave up (it won't silently re-loop).

5. **A's hint is also written to the trace** (it already is — the full tool result is traced). No extra
   event; the enriched error string flows through the existing `tool_call` event.

## Files touched
- **`shaderbox/copilot/capabilities.py`:** add `hint: str = ""` to `EditResult`.
- **`shaderbox/app.py`** (`_copilot_apply_shader_edit`): on `matches == 0`, compute the
  whitespace-normalized unique near-match and populate `EditResult.hint` with the exact source slice.
- **`shaderbox/copilot/tools/shader.py`** (`edit_shader`): when `matches == 0` and a hint exists, append
  it to the error message.
- **`shaderbox/copilot/agent.py`** (`run_turn`): the consecutive-failed-edit counter (B); the giveup +
  max_iterations → `AgentError` cutoff path (C). `run_turn` stays history-PURE (reads only).
- **`shaderbox/copilot/session.py`** (`_run_one_turn`): folds the turn-ending `AgentError` message into
  the committed history (the worker owns history, not the loop) — so a giveup/cutoff carries its note as
  the assistant's last word and the next turn's context shows the agent stopped (C).
- **`shaderbox/copilot/config.py`:** no change (the field exists) — but it stops being dead.
- **Docs:** `roadmap.md` 020 row (note slice-1 self-correction completed); this spec; `§I2` status note.

## Manual verification
- `make check` + `make smoke` green.
- **A — recovery on whitespace miss:** drive a turn where the model edits, then force a whitespace
  divergence (or reproduce the original "showcase all uniforms" prompt). Confirm the FIRST failed edit
  now returns the exact-bytes hint, and the model's next attempt COPIES it and succeeds (does NOT spiral).
  (Read the live transcript — the hint must appear in the `edit_shader` error result.)
- **B — retry cap:** if the model still can't match after 3 edit attempts, confirm the loop STOPS at 3
  (not 12) — check the log shows ≤3 `edit_shader -> ok=False` then a giveup, not the run to
  `max_iterations`.
- **C — chat surfacing:** confirm a giveup AND a max_iterations cutoff each append a visible message to
  the chat (an error/warning bubble), so the user sees why it stopped and that the turn is over (Send
  re-enabled, no ambiguity). (UN-headless — maintainer drives the app.)
- **Targeted test:** a headless unit test in `tests/` that drives `run_turn` with a fake client that
  always emits a failing `edit_shader` tool call, asserting the loop stops at `max_edit_retries` and
  yields an `AgentError` (not 12 iterations).

## Open questions for the user
- **None blocking.** Defaults taken from §I2 (`max_edit_retries=3`) and the prior session's decisions
  (echo-exact-bytes for A, no auto-normalize). The maintainer already chose "implement clean robust
  solutions" — proceeding.
