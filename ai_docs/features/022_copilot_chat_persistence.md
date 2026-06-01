# 022 — copilot_chat_persistence

A copilot conversation that is **tied to its project and survives restart**. Today the chat lives only
in memory (`CopilotSession.state` + `.history`) and is dropped on project switch / app exit — reopening
ShaderBox shows an empty chat. This feature persists the conversation per-project, restores it on
project open, and archives it on clear.

Built on **feature 021's on-disk cut** (021 decision 9): durable-portable, read-back-by-the-app state
lives in the **project dir**; disposable machine-local debug output lives in `app_data_dir()`. The
conversation is durable project state — same class as `app_state.json` and `nodes/` — so it lives
**inside the project dir** and travels with it. (021's agentic trace is the opposite: debug ephemera,
central, not restored.)

Sibling to 020 (the copilot agent itself). `CopilotSession.history`'s comment already names this:
*"Slice 1 keeps it in memory; per-project persistence is a later slice."* This is that slice.

---

## Goal
- **One conversation per project, persisted in the project dir.** A `copilot/conversation.json` under
  the active project dir (alongside `app_state.json` / `nodes/` / `media/` / `trash/`). It travels with
  the project (copy the folder → the chat comes along), exactly like the nodes do.
- **Restore on project open.** Opening a project (startup or switch) loads its conversation into the
  live chat — both the **UI render messages** (what the user sees) and the **LLM history** (what the
  agent replays for context). Reopening the app lands the user back in their last conversation.
- **Save at the right moments.** Persist after each completed turn (the durable unit), and on project
  switch / app shutdown. Never mid-turn (a partial turn isn't a coherent restore point).
- **Archive on clear, don't delete.** When the user clears/resets the chat, the current conversation is
  archived to `copilot/archive/conversation_<stamp>.json` rather than destroyed — recoverable history,
  cheap because projects are few. A fresh empty `conversation.json` takes its place.
- **Versioned, migration-safe, fail-soft.** `conversation.json` carries a schema version and loads via
  a `load_and_migrate`-style path (mirroring `UIAppState`). A corrupt/incompatible file degrades to an
  empty chat with a WARNING (never a crash, never a lost project) — the same posture as a bad
  `app_state.json`.

## Out of scope (each with a trigger)
- **A conversation picker / history browser UI** (list past archived conversations, reload one).
  **Trigger:** the user wants to revisit an archived chat without hand-copying the file back.
- **Cross-project / global conversation search.** **Trigger:** first time "where did I ask about X"
  spans projects often enough to matter.
- **Trimming/summarizing long histories to fit the context window.** The agent loop already guards
  `max_input_tokens` (`CopilotConfig`); persistence stores the full history regardless. **Trigger:** a
  restored history routinely overflows the model's context and a summarize-on-load step is needed.
- **Persisting in-flight / streaming state.** Only completed turns persist (Goal). A turn interrupted by
  a crash is simply not in the saved file. **Trigger:** never (by design — partial turns aren't restore
  points).
- **Migrating the agentic trace into the project dir.** The trace stays central debug ephemera (021
  decision 9). **Trigger:** never (the cut is deliberate).

## Design decisions
*(numbered, lock-in only; open questions separate below)*

1. **Persist BOTH `state.messages` and `history` — they are not derivable from each other.**
   `ChatState.messages` is the UI render stream (roles `user`/`assistant`/`tool_status`/`error`/
   `pending_action` — what's drawn in the chat). `history: list[LLMMessage]` is the LLM replay context
   (role + content + tool-calls — what the agent re-sends). A `tool_status` UI card has no `LLMMessage`
   twin; an assistant `LLMMessage` with tool-calls renders as one bubble. So the file stores both
   arrays. On load, both are restored to their owners (state → main thread, history → worker). (Considered
   deriving one from the other — rejected: lossy both ways.)

2. **`conversation.json` schema, versioned like `UIAppState`.** A pydantic model
   (`ConversationStore`?) with `version: int`, `messages: list[...]` (the render messages), `history:
   list[...]` (the LLM messages), and `usage` (the `SessionUsage` rollup, so the cost/token counter
   restores too). Mirrors `app_state.py`'s `extra="forbid"` + `load_and_migrate` discipline. Lives in
   `shaderbox/copilot/persistence.py` (new) — NOT in `state.py` (which stays pure runtime dataclasses).

3. **The file lives at `project_dir/copilot/conversation.json`; path via a `paths.py`-style helper.**
   A `copilot/` subdir in the project dir groups the conversation + its archive. The path resolves from
   the App's active `project_dir` (App already owns project-dir path resolution — `nodes_dir`,
   `media_dir`, etc.). Add a `copilot_dir` / `conversation_file_path` property on App alongside those.

4. **Save trigger = turn completion + project-switch + shutdown; on the MAIN thread.** The worker owns
   `history`, the main thread owns `state`. A consistent snapshot of both exists only on the main thread
   *between* turns (when `in_flight` is False). So save fires from the main thread: in
   `pump_events`/`_finish_turn` when a turn completes, in `_init` before `reset_conversation`, and in
   `release`. The worker never writes the file. (This sidesteps the thread-ownership split cleanly — no
   lock needed, the save reads both arrays at a quiescent point.)

5. **Load trigger = project open, in `_init`, replacing the unconditional `reset_conversation()`.**
   Today `_init` (app.py, after exporter wiring) calls `self.copilot.reset_conversation()` to DROP the
   chat on switch. 022 changes this to: save the OUTGOING project's conversation (decision 4), then
   `reset_conversation()`, then LOAD the incoming project's `conversation.json` into the fresh session.
   A new `CopilotSession.load_conversation(store)` restores `state` + `history` + `usage` from the
   parsed model. (The existing `reset_conversation` stays — it's the clean-slate primitive load builds on.)

6. **Clear = archive-then-reset, exposed where the chat is cleared today.** Find the current "clear
   chat" affordance (if any) — `reset_conversation` is the in-process reset, but the USER-facing clear
   may not exist yet. If a clear button exists, it archives (`copilot/archive/conversation_<stamp>.json`)
   then resets + writes a fresh empty file. If no user-facing clear exists yet, 022 adds a minimal one
   (the archive-on-clear is a stated goal). Resolve the exact UI surface in impl per `/imgui-ui`.

7. **Fail-soft load, matching `app_state`.** A missing file → empty chat (first use of copilot in this
   project). A corrupt/version-incompatible file → empty chat + WARNING (021's leveling: file-only
   warning, not console spam, not a crash). Never block project open on a bad conversation file. The
   `LLMMessage` reconstruction must tolerate older shapes (tool-call fields added later) via the
   migration path.

8. **The trace (021) records a `conversation_loaded` event.** When a conversation restores, the agentic
   trace notes it (how many messages/turns restored) so a cold-read trace shows the agent didn't start
   from zero. Small, but it closes an observability gap (a restored history changes what the model sees).

## Files touched (as built)
- **New `shaderbox/copilot/persistence.py`** — `ConversationStore` pydantic model (+ `_MessageModel` /
  `_HistoryModel` / `_ToolCallModel` / `_UsageModel` sub-models) with `from_runtime` / `to_messages` /
  `to_history` / `to_usage` / `save` / `load_and_migrate` + `archive_conversation`. Mirrors
  `UIAppState`'s `extra="forbid"` + fail-soft discipline; the runtime↔persisted mapping lives HERE
  (state.py stays runtime-pure). `cast` narrows the loose on-disk `role: str` back to its Literal.
- **`shaderbox/copilot/session.py`:** `save_conversation(path)` (pump_events → snapshot) +
  `load_conversation(store)` (+ the `conversation_loaded` trace event, decision 8); the `_drop_turn`
  Event (set by `reset_conversation`/`release`, cleared per turn) guarding the `history.append` so a
  teardown-aborted turn doesn't commit while a user Stop still does; the defensive WARNING in
  `reset_conversation`.
- **`shaderbox/app.py`:** `copilot_dir` / `copilot_conversation_path` properties; `release()` saves the
  outgoing conversation (top of `_init` + shutdown, guarded on `hasattr` project_dir/copilot);
  `_init` loads the incoming conversation after `reset_conversation`; `copilot_clear_chat`
  (archive→reset→save-empty); `_conversation_stamp`.
- **`shaderbox/ui.py`** (`update_and_draw`): the per-turn save on the `in_flight` True→False transition
  (the durable-unit fix — see Review history).
- **`shaderbox/widgets/copilot_chat.py`** (`_draw_top_bar`): the `begin_disabled(in_flight)` Clear
  button → `copilot_clear_chat`.
- **`ai_docs/todo.md`:** the trace-bleed deferral re-scoped to the residual `_ensure_open` structural
  weakness (orphaned-append + observable bleed both closed).
- **Tests:** `tests/test_conversation_persistence.py` (round-trip, fail-soft, archive, session
  save/load, the `_drop_turn`-vs-Stop distinction).
- **Docs:** `roadmap.md` (row + banner).

## Manual verification
- `make check` clean; `make smoke` passes.
- **Restore across restart:** run the app, have a multi-turn copilot conversation (read + edit), quit,
  relaunch → the chat is back, both bubbles AND the agent's context (ask a follow-up that depends on
  prior history; confirm it remembers). (UN-headless — maintainer drives the app.)
- **Per-project isolation:** two projects, different chats; switch between them → each shows its own
  conversation, not the other's.
- **Travels with the folder:** copy a project dir elsewhere, open it → its chat comes along.
- **Archive on clear:** clear the chat → `copilot/archive/conversation_<stamp>.json` appears, the live
  chat is empty, a fresh `conversation.json` exists.
- **Fail-soft:** corrupt `conversation.json` (hand-edit to invalid) → project opens with an empty chat +
  a WARNING in the log file (not a crash, not console spam).
- **Cost counter restores:** the session usage (tokens/cost) shown in the chat reflects the restored
  total, not zero.

## Pre-impl code re-validation (after 020·12–15 + 021 landed)
A read-only sweep confirmed the spec against current code: `ChatState`/`Message` (plain dataclasses,
roles user/assistant/tool_status/error/pending_action — JSON-safe), `CopilotSession.history:
list[LLMMessage]` (worker-owned, persists across turns on the session, frozen dataclass — JSON-safe),
`SessionUsage` (plain, on `state.usage`), the `_init` → `reset_conversation()` call (still after exporter
wiring, guarded `if hasattr(self, "copilot")`), `_finish_turn`/`pump_events`/`release` (all present;
`_finish_turn` is the single turn-end chokepoint), and `UIAppState.load_and_migrate` (pydantic
`extra="forbid"` + try/except → defaults + WARNING) as the model to mirror. Two amendments:

- **A1 — the worker is ALREADY IDLE when `reset_conversation` runs (a real join is defensive, not the
  mechanism), but the orphaned-history append needs a real 2-line guard.** A second pre-impl review
  corrected the original A1: 020·15's `open_project` gate (`_copilot_busy_blocked` on
  `copilot_turn_active`) prevents a project switch while a turn is in flight, so by the time
  `reset_conversation` runs the worker has already emitted its terminal event, `pump_events` has
  flipped `in_flight` False, and the worker is back at `_turn_queue.get()`. So NO blocking
  join-to-idle primitive is needed (it would also be safe — `_events` is unbounded so the worker never
  deadlocks — but it's unnecessary and implies a freeze risk that doesn't exist). The REAL deferral fix
  is: `_cancel.set()` does NOT stop the orphaned append — `run_turn` returns *normally* on cancel
  (`yield AgentCancelled(); return`), so the two `self.history.append(...)` at the end of
  `_run_one_turn` fire unconditionally. **Guard those appends on `not self._cancel.is_set()`** — that is
  the orphaned-history half of the deferral. The trace-bleed half is closed by the worker-is-idle
  invariant (no `run_turn` holds the old `TraceLog` when `reset_conversation` swaps it); a
  permanent-closed flag on `TraceLog.close()` is optional residual hardening, noted not required. A
  defensive WARNING-if-`in_flight` at the top of `reset_conversation` documents the invariant.
- **A1-save-order — `pump_events()` THEN save THEN reset.** The terminal `AgentTurnDone`'s assistant
  bubble + `usage` land in `state` only when `_apply_event` drains it. Save must `pump_events()` first
  (idempotent — already drained in the UI case, but explicit so a future non-modal switch is correct),
  read `(state, history, usage)`, then `reset_conversation` zeroes them, then load incoming.
- **A2 — the user-facing clear button does NOT exist yet** (`copilot_chat.py::_draw_top_bar` has only
  Layout + Close). 022 adds it there (decision 6 / Q3), via `ghost_button`. **It MUST be
  `begin_disabled(state.in_flight)`** — otherwise clear-during-turn is the one path that bypasses the
  020·15 protection (orphaned append, mid-turn archive of an incomplete conversation).

## Review history
- **Plan-lock implicit** (maintainer authorized proceeding; Q1–Q3 already locked). Spec pre-dates
  020·12–15 + 021 — a read-only re-validation confirmed all 9 code dependencies (above).
- **Pre-impl threading review (1 agent, adversarial) → FIX-THEN-SHIP; corrected the original A1.** It
  found the mid-turn switch is UNREACHABLE through the UI (020·15's `open_project` gate), so the worker
  is always idle when `reset_conversation` runs — no blocking join needed (the original A1 over-built
  it). The real deferral fix is guarding the `history.append` (cancel returns normally, not via
  exception, so the append fires today). Plus: save must `pump_events()` first; the new clear button
  must be `begin_disabled(in_flight)`. All folded into A1 / A1-save-order / A2.
- **Implemented**, then **post-impl review (1 agent, adversarial) → FIX-THEN-SHIP; two real findings
  fixed:**
  1. **[BLOCKER] the per-completed-turn save (the headline durability promise) was never wired** — only
     switch/shutdown/clear saved, so a crash between switches lost every completed turn. FIXED: `ui.py`'s
     per-frame copilot reconcile now saves on the `in_flight` True→False transition (the turn-complete
     moment), so the completed turn is the durable unit as promised.
  2. **[SHOULD-FIX] the deferral guard conflated teardown-cancel with the user Stop button** — guarding
     the commit on `_cancel` meant a user Stop (which also sets `_cancel`) silently dropped the turn AND
     desynced `state.messages` (bubbles kept) from `history` (turn erased). FIXED: a dedicated
     `_drop_turn` Event, set ONLY by `reset_conversation`/`release` (teardown), never by `cancel_turn`
     (Stop). A stopped turn now commits its partial reply to this conversation; only a project-teardown
     abort drops it. Added a session-level unit test for the distinction (the deferral fix had no
     coverage before). The reviewer's [NIT] (restoring an unproducible `pending_action` role) is inert
     today — noted, not fixed. 154 tests green.

## Resolved questions
*(all locked by the maintainer)*
- **Q1 — archive unbounded.** Archives accumulate in `copilot/archive/` on clear; left uncapped
  (projects + clears are both rare, growth negligible). Revisit only if it ever bloats.
- **Q2 — lifetime-per-project cost restores.** Decision 2 stands: `usage` (tokens/cost) is persisted and
  restored, so the counter reflects the project's lifetime copilot cost, not a per-app-run total.
- **Q3 — a user-facing "clear chat" button IS in 022's scope** (confirmed needed). It archives the
  current conversation then resets to a fresh empty one (decision 6). Built via `ui_primitives.py` +
  `/imgui-ui`.
