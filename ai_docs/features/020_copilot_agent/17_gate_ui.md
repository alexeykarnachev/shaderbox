# 020 · 17 — Gate UI wave: agent-blocking confirmation + `delete_node`

> Wave 17 of feature 020 (copilot agent). Builds the `GateChannel` BODY (its TYPE + the
> blocking primitive landed in slice 1) and lands the first tool that needs it: `delete_node`.
> Parent spec: `11_capability_wave_spec.md §7` (the interactive-widget family) + `§2.2/§2.3`
> (the loop's gate branch) + `§F4` (the confirm policy). Read those §-refs alongside this.

## Goal

A destructive copilot tool call (this wave: `delete_node`) hands control back to the chat for a
**Yes/No confirmation rendered inline in the transcript**, blocking the agent loop until the user
answers:

- **Yes** → the agent silently continues its tool-call sequence. The resolved confirm card grows a
  **Recover** button that restores the just-deleted node from the project trash (decision 10).
- **No** → the declined result feeds back into the loop; the model emits its own closing remark
  ("you said no, so I'm cancelling…") and the turn ends naturally — decision (A), maintainer call.

The confirmation reuses the existing `GateChannel` (the structural mirror of `CopilotBridge`) — no
new transport, no duplicated round-trip machinery (maintainer: "make sure it's generalized, we have
a similar bridge system, don't duplicate code").

This unblocks the live trigger: a test-session user asked the agent to "remove the last 3" nodes and
it had no tool to do so.

## Out of scope (each with a trigger)

- **"Yes for all" / batched confirms** — DEFERRED. This wave gates **per destructive tool call**
  (maintainer, explicit): "delete the last 3" pops three sequential Yes/No prompts. A later UX wave
  adds option-list responses ("Yes", "Yes to all this turn", "Yes to all future", "No"). *Trigger:*
  maintainer finds the per-call repetition annoying in real use, OR the first tool whose natural unit
  is a batch (a multi-render). The `GateRequest.options: list[str]` field + `GateResponse.option`
  already exist for this — only the loop's interpretation + the UI's button row grow.
- **`delete_lib_file`** — DEFERRED. The spec pairs it with `delete_node` (`§F4`), but it has no live
  trigger (no user has hit the lib-delete gap). *Trigger:* first user/agent need to delete a lib file
  via the copilot. It rides the exact same gate machinery this wave builds — adding it later is a tool
  definition + an `App` closure, no gate changes.
- **CREDENTIAL gate kind** — DEFERRED. `GateKind.CREDENTIAL` (inline secret field, `§7.2`) stays a
  type-only stub. The only live need is CONFIRM. The OpenRouter key is already gated pre-turn by the
  `unconnected_gate` in `copilot_chat.py`; Telegram/YouTube keys have no copilot publish tool yet.
  *Trigger:* the publish-tools wave (`§7.2/§F5`, the lazy telegram/youtube tools land).
- **BULK gate policy** — the mechanism (`requires_gate` BULK branch, `bulk_gate_threshold`) is ALREADY
  built and stays built; no tool sets `GatePolicy.BULK` this wave. *Trigger:* a tool whose args carry
  a list whose length should auto-confirm (folds into the batched-confirm UX wave above).
- **Progress bar (`§8`)** — the non-blocking sibling of the gate family; separate wave. N/A here.

## Design decisions (locked)

1. **Decline continues the loop (decision A).** On `No`, the loop does BOTH:
   `ran.record(tc.name, False, "error: user declined")` AND
   `messages.append(_tool_message(tc.id, "error: user declined"))`, then `continue` — NEVER `return`/
   `break` mid-batch (B2: a round can carry multiple tool_calls; every `tool_call_id` needs a matching
   `tool` result or the next `client.stream` 400s). The model sees the declined result among its tool
   results and produces a closing comment, which lands as the assistant's final text. NOT a hard
   `AgentCancelled` return (that would stop silently — the maintainer wants the agent to *speak*).
   Faithful to `11 §2.2`'s gate-branch sketch. A Stop / window-close / shutdown DURING the gate wait still
   returns `AgentCancelled` (that's `resp.cancelled`, a different branch — decision 14 resolves its
   pending card).

2. **A declined mutating call does NOT count toward the edit-retry cap.** `run_turn`'s
   `consecutive_failed_edits` counter increments on a mutating tool that returned `ok=False`. A
   user-declined `delete_node` is exactly that shape, so three declined deletes in a row would trip
   `max_edit_retries` and surface the "couldn't apply that edit" giveup note instead of a clean
   comment. A decline is a *user choice*, not a convergence failure — treat it like the existing
   `stale` reject (`§15`): it resets the counter / is excluded from it. Implementation: the decline
   path sets a `declined` marker the loop reads with the same `not stale`-style guard, OR (simpler)
   the gate branch `continue`s BEFORE the counter logic and records the decline as non-counting. The
   counter only ever sees results from `registry.execute`, so routing the decline around `execute`
   (which the gate branch already does — a declined call never reaches `execute`) keeps it out of the
   counter for free. **Verify in implementation:** the `continue` after the decline lands before
   `consecutive_failed_edits` is touched.

3. **The gate is event-driven, not UI-polled.** `gate.ask()` (worker) emits an `AgentGateOpened`
   event onto the existing `_events` queue THEN blocks on its event slot. `pump_events` (main thread),
   on the `AgentGateOpened` case, calls `gate.take_pending()` to dequeue the `_GatePending` (keeping
   `_pending` and `_current` in lockstep — S2: otherwise `_pending` leaks one stale entry per gate)
   and materializes a `pending_action` `Message` in `state.messages` from the event's request. The UI
   renders the Yes/No widget from that Message; on click it calls `CopilotSession.answer_gate(...)` →
   `gate.answer(...)` (which fills `_current`'s slot + sets the event). This keeps the single-writer
   invariant (`§7.3`): only `pump_events` writes `state`; the worker's block/unblock goes entirely
   through the `GateChannel`'s own event+slot. The UI does NOT poll `gate.take_pending()` each frame —
   `take_pending` is called exactly once, by `pump_events` when it sees the `AgentGateOpened` event
   (so it is NOT dead code; it is the materialization dequeue).

4. **`GateChannel` and `CopilotBridge` stay separate types — the twinning is documented, not merged.**
   They are deliberate structural mirrors (both: worker pushes request + `threading.Event`, blocks on
   `wait`; the other side fills a slot + sets the event; `cancel_all(reusable=)` parallel). But they
   marshal *different payloads in opposite directions* — the bridge runs a GL closure on the main
   thread and returns its result; the gate carries a UI confirmation request out and a user decision
   back. A forced shared base class would abstract over a 3-line `wait`/`set` core while the request,
   response, and dequeue shapes all differ — the false-abstraction trap (`conventions.md`: a
   convention collision means the design is wrong; here there's no collision, just two uses of the same
   *pattern*). "Don't duplicate code" is satisfied by NOT re-implementing the round-trip a third time —
   the gate already exists and is reused by every future widget kind (`§7.1`: "built ONCE, reused by
   every widget kind"). Decision: keep two types; this wave touches neither's core (both are built),
   only wires the gate's already-built `ask`/`answer` into the loop + UI. Revisit if a THIRD
   blocking worker→main round-trip appears (then extract a shared `_BlockingRoundTrip[Req, Resp]`).

5. **`delete_node` marshals the full teardown through the bridge, reusing `App.delete_node`'s body.**
   The copilot closure `_copilot_delete_node` binds the RAW delete logic (node `.release()`, editor-
   session pop, current-node reselection, trash-move), NOT the public `App.delete_node` — which is
   guarded by `_copilot_busy_blocked` and would refuse the copilot's own mid-turn call (exactly as
   `_copilot_create_node` binds the raw create body, not the guarded `create_node_from_selected_template`).
   To avoid duplicating the teardown, extract the guard-free body of `App.delete_node` into a private
   `_delete_node_unguarded(node_id)` that both the public guarded method AND the copilot closure call.
   The closure wraps it in `self.copilot.bridge.run_on_main(...)` (GL teardown is main-thread-affine).

6. **`delete_node` is `mutating=True`, `needs_gl=True`, `gate_policy=ALWAYS`, `eager=True`,
   `category="shader"`.** `mutating` → it contributes to the "what I did" note + the gate. `needs_gl`
   → the closure marshals (node release is a GL op). `ALWAYS` → `requires_gate` returns True
   unconditionally (destructive, `§F4`). `eager` → in the turn-start `tools=` set (the catalog is
   small; no lazy loading yet).

7. **`delete_node` takes a `node` arg (short id from the project map), required — no implicit current
   node.** Deleting is destructive enough that "which node" must be explicit; an empty/omitted id is a
   tool error ("specify which node to delete — use a node id from the project map"), never a silent
   delete-current. This differs from the edit tools' "empty = current" convenience on purpose.

8. **The gate prompt is engine-built, not model-supplied.** `build_gate(tc, args)` constructs the
   `GateRequest.prompt` from the tool name + resolved args (e.g. `Delete node 'gradient'? This moves it
   to the project trash.`). The model does not write the confirmation text — the engine owns the
   destructive-action phrasing so it's accurate (the model could misdescribe what it's about to do).
   `build_gate` lives in `agent.py` (loop-private, `§2.3`); a small per-tool prompt template keyed by
   tool name, falling back to a generic "Run {tool} with {args}?".

9. **Send is gated while a gate is open.** The chat input is already gated on `in_flight`; `in_flight`
   stays True while the worker blocks on `gate.ask()` (the turn hasn't ended), so the Send box is
   already disabled — no extra flag. The Yes/No buttons are the only live affordance. Verify: the
   `pending_action` Message renders buttons even though the input is disabled (it's a separate widget
   in the transcript, not the input row).

10. **Recover button on an approved-delete card — durable, user-side, agent-invisible.** When a delete
    confirm resolves with Yes, the card keeps a **Recover** button that restores the node from trash.
    Three properties, all maintainer-locked:
    - **Survives app restart / project reopen.** The recover affordance is persisted in
      `conversation.json` (feature 022). `_MessageModel` gains an optional recover sub-model carrying
      the trash **dir-name** + node_id + node_name + `done` flag — NOT an absolute path (B3: the
      project dir is relocatable; an absolute `trash_path` breaks on copy/rename). `restore` rebuilds
      `self.trash_dir / dir_name` at click time from the LIVE project dir. Bump `_VERSION` to 2 for
      honest schema-versioning (no migration branch keys on it — it's cosmetic; `load_and_migrate`
      just does `cls(**data)`). An old file (no field) loads soft (field defaults to None → a plain
      history card). NOTE: downgrade is NOT soft — an older `extra="forbid"` build hitting a v2 file's
      unknown `recover` key discards the conversation to empty; that is the existing posture (same as
      every prior field add), neither caused nor fixed by this wave. So: forward-compatible (new build
      reads old files), not backward (old build loses new files) — acceptable, matches the project.
    - **Dies on chat-clear.** No special handling — `copilot_clear_chat` archives the whole
      conversation + resets `state.messages`, so the card (and its recover field) leaves the live chat
      with everything else. The archived JSON still holds it (recoverable-by-hand, like any cleared
      message), but the live chat shows nothing.
    - **Dies on the trash dir being gone.** The button's *presence* is driven by the message's recover
      field; its *liveness* is checked at click against `Path(trash_path).exists()`. If the trashed
      node dir was removed (manual trash-clear, etc.), Recover no-ops gracefully and the card flips to
      a terminal "no longer recoverable" state. (Cheap pre-check each frame is acceptable — a delete
      card is rare; or check only on click and flip the card then. Implementation picks the cleaner.)
    - **Agent-invisible.** Recover is a pure user-side undo on the main thread — NOT a gate response,
      NOT a worker action (the turn is long over by the time the user clicks). It does not touch
      `history`; the agent learns the node is back only via the next turn's live project-map re-read.
    - **One-shot.** After a successful recover the card flips to "Recovered" and the button is gone (the
      trash dir has moved back; a second restore would find nothing).

11. **Restore reuses the delete teardown's inverse, sharing one body.** `App.restore_node_from_trash
    (trash_name, node_id) -> bool` is the mirror of `_delete_node_unguarded`. Order (move FIRST, then
    load — so the loaded id is the dir-name, not the trashed `<id>_<ts>`):
    `src = self.trash_dir / trash_name`; if not `src.exists()` → return False (graceful no-op,
    trash-was-cleared). `dst = self.nodes_dir / node_id`; if `dst.exists()` → return False (collision
    guard — near-impossible with UUIDs, but `shutil.move` onto an existing dir nests it, so guard).
    `shutil.move(src, dst)` → `load_node_from_dir(dst)` → `ui_nodes[node_id] = node` →
    `set_current_node_id(node_id)` (restored node becomes visible) → return True. Main-thread +
    GL-context-live (a normal main-thread call from a button click — no bridge, the worker isn't
    involved). `_delete_node_unguarded` pops `self._copilot_read_revision.pop(node_id, None)` for
    hygiene (the freshness leak reviewer 1 flagged — benign but cheap to clear). `DeleteNodeResult`
    carries `(ok, error, deleted_name, node_id, trash_name)` so the recover card needs no parsing.

12. **The success reply tells the model the delete is trash-recoverable.** `delete_node`'s ok-reply is
    e.g. `deleted node 'gradient' — moved to the project trash (recoverable).` So the agent can pass
    that reassurance to the user in its own words. (User-confirmed.)

13. **The system prompt must be rewritten to PERMIT deletion (B1).** `prompt.py::_SYSTEM_PROMPT` is a
    static literal carrying "You cannot delete a node (the user does that from the node grid)." and
    "You have no undo" under THE SANDBOX. Both are now false and would suppress the tool. Edits:
    (a) add a WHAT-YOU-CAN-DO bullet: `delete_node(node)` deletes a node — it will PROMPT THE USER to
    confirm before it happens, and the delete is recoverable from the project trash; (b) remove "You
    cannot delete a node (the user does that from the node grid)."; (c) reconcile "You have no undo" —
    a delete IS reversible (trash), and edits revert by re-editing, so soften to "you have no general
    undo — to revert an edit, re-edit to the prior state; a delete can be recovered from trash by the
    user." The "cannot create/switch/delete PROJECTS" boundary stays (still true).

14. **A cancelled/errored turn resolves its open gate card (B4).** `_apply_event`'s `AgentCancelled`
    and `AgentError` cases must, in addition to their current work, flip the TRAILING `pending_action`
    Message whose `resolved == False` to a terminal resolved state (text → "cancelled" so the card
    reads as history, buttons gone). Without this, a Stop/window-close mid-gate leaves a card with live
    Yes/No buttons that call `gate.answer` on a `_current` that's already None (no-op, but visually a
    bug). The resolution target rule everywhere (answer + cancel): "the last `pending_action` Message
    with `resolved == False`" — there is only ever one open gate (`_current`), so the trailing
    unresolved card is unambiguous.

15. **Headless verification script (B5).** `scripts/copilot_gate_check.py` — a standalone script (the
    project's stated pattern for non-visual verification, like `smoke.py`) covering the two
    pure-function-testable locked decisions: (a) **persistence round-trip** — build a `ChatState` with
    a resolved-Yes delete `Message` carrying `RecoverInfo`, `ConversationStore.from_runtime(...).save()`
    → `load_and_migrate()` → `to_messages()`, assert the recover sub-model survives intact; also load a
    hand-written `_VERSION=1` JSON (no recover field) and assert it loads soft to a plain card.
    (b) **decline-counter** — a stub `GateChannel` subclass whose `ask` always returns
    `GateResponse(approved=False)`, a stub LLM client emitting three `delete_node` tool calls across
    iterations, drive `run_turn`, assert it terminates in `AgentTurnDone` (the agent's comment), NOT an
    `AgentError` giveup (proves a declined mutating call doesn't trip `max_edit_retries`). Run it in
    CI-free fashion (`uv run python scripts/copilot_gate_check.py`); wire a mention into the verify
    checklist. NOT added to `make smoke` (it's copilot-specific + needs no GL — a separate script keeps
    smoke's GL-frame focus clean).

## Files touched

- **`shaderbox/copilot/agent.py`** — the core wiring:
  - New `AgentGateOpened(request: GateRequest)` event in the `AgentEvent` union (carries the request so
    `pump_events` can build the Message; the worker emits it right before blocking).
  - `build_gate(tc, args) -> GateRequest` — engine-built CONFIRM prompt (decision 8).
  - The `§2.2` gate branch in `run_turn`'s tool loop: `if registry.requires_gate(tc.name, args,
    config): yield AgentGateOpened(req); resp = gate.ask(req); ...` with the `cancelled` →
    `AgentCancelled` and `not approved` → record-decline + `continue` branches. Replaces `_ = gate`.
  - Pass `config` into the gate check (already in scope as `config`).
  - The decline `continue` lands before the `consecutive_failed_edits` logic (decision 2).
- **`shaderbox/copilot/session.py`** — `_apply_event`: handle `AgentGateOpened` → append a
  `Message(role="pending_action", text=<the prompt>)` (carrying the recover field built from the
  request's tool + args when the gated tool is `delete_node`). The answer path: when the UI calls the
  new `CopilotSession.answer_gate(response)`, it forwards to `self.gate.answer(response)` and flips the
  pending Message's `resolved=True`. `cancel_turn` already calls `gate.cancel_all(reusable=True)` — a
  Stop mid-gate releases the wait (no change needed there; verify the pending Message is marked
  resolved/removed when the worker reports `AgentCancelled`).
- **`shaderbox/copilot/state.py`** — `Message` gains an optional recover field (e.g.
  `recover: RecoverInfo | None = None`, a small frozen dataclass `node_id: str`, `node_name: str`,
  `trash_path: str`, `done: bool` for the one-shot flip). Kept runtime-pure (no disk concern — the
  persisted mirror lives in `persistence.py`, per that module's stated split).
- **`shaderbox/copilot/persistence.py`** — `_MessageModel` gains the optional persisted recover field;
  `_VERSION` → 2; `from_runtime` / `to_messages` map it both ways. Fail-soft on an old file (absent →
  None). This is the only feature-022-schema touch this wave.
- **`shaderbox/widgets/copilot_chat.py`** — `_draw_message` gains a `pending_action` branch: render the
  prompt text + a Yes/No button row (via `ui_primitives` button tiers — `primary_button("Yes")` /
  `ghost_button("No")`, per the imgui-ui skill; NOT hand-rolled). On click, call
  `app.copilot.answer_gate(GateResponse(approved=..., option=...))`. Once `resolved`, render the
  resolved choice as a static "You chose: Yes/No" line with the buttons gone (transcript-as-history).
  When `resolved` + approved + `recover` present + `not recover.done`: render the **Recover** button
  (`ghost_button("Recover")`); on click call `app.recover_deleted_node(msg)` (resolves via
  `restore_node_from_trash`, flips `recover.done`). When `recover.done`: show "Recovered". The
  `_draw_message` signature currently takes `(role, text)`; it needs the whole `Message` (for
  `resolved` + `recover`) — widen it to take `msg: Message`. NOTE the streaming-text call site
  (`_draw_transcript` draws `state.streaming_text` with no Message): construct a transient
  `Message(role="assistant", text=state.streaming_text)` for that call (S2). Guard `msg.recover is not
  None` before any `.trash_name`/`.done` access (pyright Optional).
- **`shaderbox/copilot/tools/shader.py`** — the `delete_node` `ToolDefinition` + handler:
  `_DeleteNodeArgs(node: str)`, `_DELETE_NODE_DESC`, the handler calls `caps.delete_node(node)` and
  returns the result. `gate_policy=GatePolicy.ALWAYS`.
- **`shaderbox/copilot/capabilities.py`** — add `delete_node: Callable[[str], DeleteNodeResult]` to
  `CopilotCapabilities` + a `DeleteNodeResult` frozen value object: `ok: bool`, `error: str = ""`,
  `deleted_name: str = ""`, `node_id: str = ""`, `trash_name: str = ""` (the last two feed the recover
  card; `trash_name` is the trash dir-name, NOT an absolute path — B3). GL-free leaf type.
- **`shaderbox/app.py`** —
  - `_delete_node_unguarded(node_id) -> str` (extracted from `delete_node`'s body; returns the actual
    `trash_path` so the result + recover card can capture it). Public `delete_node` becomes the guard +
    a call to it.
  - `_copilot_delete_node(node) -> DeleteNodeResult` (resolve short id → bridge round-trip → unguarded
    delete, capturing `trash_path` + original `node_id`) + wire into `_build_copilot_capabilities()`.
    Resolve the short id via `_copilot_resolve_node_id`; a miss returns `DeleteNodeResult(ok=False,
    error=...)`.
  - `restore_node_from_trash(trash_name, node_id) -> bool` (decision 11) — main-thread inverse of the
    unguarded delete; False if `self.trash_dir / trash_name` no longer exists OR the dest collides
    (graceful no-op). `recover_deleted_node(msg)` is the UI-facing wrapper the button calls: guards
    `msg.recover is not None`, runs the restore, flips `msg.recover.done`, pushes a notification (and
    re-persists the conversation so the flipped `done` survives — call the same save path
    `copilot_clear_chat` uses).
- **`shaderbox/copilot/prompt.py`** — REQUIRED rewrite (B1, decision 13): the prompt is a STATIC
  literal (does NOT derive from `registry.describe()`). Add the `delete_node` WHAT-YOU-CAN-DO bullet
  (notes the confirm prompt + trash-recoverability), remove "You cannot delete a node (the user does
  that from the node grid).", soften "You have no undo".
- **`scripts/copilot_gate_check.py`** — NEW (B5, decision 15): the headless verification script
  (persistence round-trip + decline-counter via stub gate/client). Standalone `uv run python`-able;
  mirrors `scripts/smoke.py`'s standalone shape but needs no GL context.
- **`shaderbox/copilot/gate.py`** — NO core change (the channel is built). Update the stale slice-1
  comment ("the BODY ... lands with the interactive-widget wave (step 5)") to reflect that the body
  now exists. Also refresh `agent.py`'s "never triggered in slice 1" comment block + drop `_ = gate`.
- **`ai_docs/features/020_copilot_agent/11_capability_wave_spec.md`** — flip `delete_node`'s status
  from DEFERRED to landed where it's mentioned (`§16`-era "delete_node is DEFERRED" lines), if the
  edits there don't drift the parent spec's own scope. (Light touch — point to this wave's spec.)
- **`ai_docs/roadmap.md`** — flip the 020 row + rewrite the Active-context banner (gate-UI wave done;
  next = render/publish tools or the UX-polish wave).
- **`ai_docs/todo.md`** — no new deferral unless the implementation surfaces one; the out-of-scope
  items above are trigger-shaped and go here ONLY if they need a grep-by-trigger landmine (the
  batched-confirm + delete_lib_file ones are roadmap-backlog, not landmines — keep them in the
  roadmap/spec, not todo, unless a real foot-gun emerges).

## Verification

### Headless (agent-runnable — must pass before any manual check)

- `make check` — ruff + pyright, 0 errors.
- `make smoke` — 200 headless frames of `update_and_draw` (popup-mutex + `current_node_id`
  invariants); the chat is never opened here, but it catches any import/callback break the new code
  introduces in the always-run path.
- `uv run python scripts/copilot_gate_check.py` (B5/decision 15) — the persistence round-trip
  (recover-card survives save→load→to_messages; a v1 file loads soft) + the decline-counter (3 declined
  `delete_node` calls end in `AgentTurnDone`, not an `AgentError` giveup, AND the post-decline
  `messages` carry a matching `tool` result for every `tool_call_id` — B2's orphaned-tool_call guard).

### Manual (maintainer — `make run`; no agent screenshot path)

The gate is UI + threading; the headless scripts above catch the logic, but the visual widget + the
block/unblock timing + the live focus/lock behavior need a hand check:

1. **Yes path:** ask the copilot to delete a specific node. A Yes/No prompt appears inline in the
   transcript. Click **Yes** → the node disappears from the grid, the agent continues and reports the
   deletion. The editor stays frozen (read-only) through the whole turn including the gate wait. The
   resolved card shows "You chose: Yes" + a **Recover** button.
2. **No path:** ask to delete a node. Click **No** → the node stays, and the agent emits a comment
   ("you said no, so I'm cancelling" or similar — model-authored). The turn ends cleanly (no giveup
   note, no error bubble).
3. **Multi-delete (per-call gating):** ask to "delete nodes X, Y, Z". THREE sequential Yes/No prompts,
   one per node. Answering No to the 2nd: the agent stops there and comments (decision 2 verified — no
   spurious giveup from the declined call counting as a failed edit).
4. **Stop mid-gate:** open a delete prompt, click the chat's **Stop** button instead of Yes/No → the
   wait releases (`cancelled`), the turn ends as cancelled, the node is NOT deleted, the pending
   widget resolves/clears.
5. **Window-close mid-gate:** open a delete prompt, close the chat window → no hang on next turn / app
   exit (the `cancel_all` releases the worker). Re-open the chat: the conversation is intact.
6. **Project-switch mid-gate** (edge): the `open_project` gate refuses while `in_flight`, and
   `in_flight` is True during the gate wait — so a switch is already blocked. Verify the
   "locked while the assistant is working" notification fires if you try.
7. **Delete the current node:** confirm the current-node reselection still works (the grid picks a new
   current node; the editor shows it) — same as the existing grid-delete path.
8. **Recover (same session):** after a Yes-delete, click **Recover** → the node reappears in the grid
   and becomes current; the card flips to "Recovered" and the button is gone.
9. **Recover survives restart:** Yes-delete a node, do NOT recover, restart the app (or switch project
   and back). Scroll to the delete card → the **Recover** button is still there and still works.
10. **Recover dies on chat-clear:** Yes-delete a node, click the chat's **Clear** → the whole
    transcript (including the delete card) is gone; no orphaned Recover affordance.
11. **Recover dies on trash removal:** Yes-delete, manually clear the project `trash/` dir, then click
    **Recover** → it no-ops gracefully (notification "node no longer in trash"), card flips to a
    terminal non-recoverable state; no crash.

## Open questions for the user

1. **Resolved-gate rendering** — RESOLVED: static "You chose: Yes/No" history line, buttons gone; a
   Yes-delete card additionally carries the **Recover** button (decisions 1 + 10). The Recover affordance
   survives restart, dies on chat-clear + trash removal (maintainer-locked).
2. **`delete_node` reply on success** — RESOLVED (decision 12): the reply names the node and tells the
   model the delete is trash-recoverable, so the agent can reassure the user.
3. **Scope** — RESOLVED: `delete_node` only this wave; `delete_lib_file` + CREDENTIAL + batched-confirm
   deferred (each trigger-tagged in Out of scope).

(All open questions resolved — spec is plan-locked, pre-implementation review pending.)

## Review history

### Pre-implementation review (2 adversarial reviewers, parallel)

Both reviewers independently converged on the same top findings — strong signal they're real. Triage:

**BLOCKERs — folded into the spec (see updated decisions/files):**
- **B1 — the system prompt FORBIDS deletion.** `prompt.py::_SYSTEM_PROMPT` is a STATIC literal (not
  derived from `registry.describe()`, verified) carrying two now-false sentences under "THE SANDBOX":
  "You cannot delete a node (the user does that from the node grid)." and "You have no undo". A
  compliant model reads these and refuses every delete → the whole feature silently no-ops. **Fix
  (decision 13):** rewrite those lines — add `delete_node` to WHAT YOU CAN DO (noting it prompts the
  user to confirm + is trash-recoverable), drop "cannot delete a node", reconcile "no undo". This is a
  REQUIRED edit, not the conditional hedge the first draft had.
- **B2 — orphaned tool_call on a multi-call decline batch.** The assistant message carries ALL of a
  round's tool_calls before the per-call loop (`agent.py` appends `_assistant_message(text_buf, calls)`
  then iterates). If the model emits `[delete A, delete B]` in one round and the user declines A, the
  loop MUST still append a `_tool_message` for A (the declined one) AND continue the loop so B also
  gets a result — every `tool_call_id` needs a matching `tool` result or the next `client.stream`
  400s. The decline path's `continue` is inside `for tc in calls`, so it structurally completes the
  batch — but it must append `_tool_message(tc.id, "error: user declined")` for the declined call
  (NOT just `ran.record`). **Fix (decision 1 sharpened):** the decline branch does BOTH
  `ran.record(tc.name, False, "error: user declined")` AND
  `messages.append(_tool_message(tc.id, "error: user declined"))`, then `continue` — never `return`/
  `break` mid-batch.
- **B3 — absolute `trash_path` breaks on project copy/rename.** `trash_dir = project_dir / "trash"`;
  the project dir is relocatable (persistence.py travels-with-the-project). Persisting an ABSOLUTE
  `trash_path` defeats that. **Fix (decisions 10/11 sharpened):** persist only the trash **dir-name**
  (`<node_id>` or `<node_id>_<ts>`) + the node_id; reconstruct `self.trash_dir / name` at click time.
- **B4 — dangling live card after Stop / window-close mid-gate.** `_apply_event`'s `AgentCancelled` /
  `AgentError` cases don't touch `state.messages`, so an open `pending_action` card keeps live Yes/No
  buttons after the turn is cancelled. **Fix (decision 14):** those cases flip the trailing unresolved
  `pending_action` Message to a terminal "cancelled" resolved state.
- **B5 — zero headless coverage for two locked decisions.** `scripts/smoke.py` never opens the chat,
  so decisions 2 (decline-counter) + 10 (persistence round-trip) ship unverified, and both are
  pure-function testable. **Fix (decision 15):** add `scripts/copilot_gate_check.py` — a headless
  introspection that (a) round-trips a recover-card conversation through `ConversationStore`
  save→load→to_messages asserting no field loss + a v1-file loads soft, and (b) drives `run_turn` with
  a stub always-decline gate + a stub client emitting 3 `delete_node` calls, asserting the turn ends
  in `AgentTurnDone` (a comment), NOT an `AgentError` giveup.

**SHOULD-FIX — folded:** which-Message-to-resolve = "the trailing `pending_action` with `resolved==
False`" (decision 14); `_pending` queue drain owner = `pump_events` calls `take_pending()` when it
processes `AgentGateOpened`, keeping `_current`/`_pending` in lockstep (decision 3 sharpened — so
`take_pending` is NOT dead code, it's the materialization dequeue); `_draw_message` streaming call-site
(`copilot_chat.py` draws `streaming_text` with no Message) — construct a transient
`Message(role="assistant", text=streaming_text)` for that call; `DeleteNodeResult` MUST carry
`(ok, error, deleted_name, node_id, trash_name)` (reconciled — first draft listed only the first
three); `RecoverInfo` access guarded `is not None` for pyright; the `_VERSION` bump is cosmetic (no
migration branch keys on it) — KEEP it at 2 for honest schema-versioning but DROP the false "degrades
soft on downgrade" claim (an old `extra="forbid"` build discards an unknown-field file to empty — same
posture as today, neither caused nor fixed by the int).

**Rejected / downgraded (reviewer 1 self-corrected):** freshness-map-not-cleared = a benign leak
(full-id keyed, reset next turn), not corruption — still pop it for hygiene (decision 11); restore
dir-collision is near-impossible with UUIDs but the cheap guard goes in anyway (decision 11).

**No todo.md deferral's trigger fires this wave** (both reviewers confirmed: the `_ensure_open`,
node-grid, and cleartext-credentials deferrals are all un-fired). The Out-of-scope items here are
roadmap-backlog, not landmines — they stay in this spec, not todo.md.

### Post-impl maintainer-found fixes (live testing)

Three bugs surfaced only in the real app (the headless `run_turn` tests structurally couldn't reach
worker↔init lifecycle); each landed with a teeth-verified regression in `copilot_gate_check.py`:

- **Gate instant-cancelled every confirm.** `gate.ask()` short-circuits to `cancelled=True` when
  `_shutdown` is latched; `App._init`'s teardown `release()` latches it before first use, and — unlike
  the bridge — the gate had no `reopen()` re-armed in `enqueue_turn`. Cards resolved "(cancelled)" with
  no clickable buttons. Fix: `GateChannel.reopen()` mirroring `bridge.reopen()`, called in
  `enqueue_turn`. Generalized to `conventions.md ## Design decisions` (the latch-needs-reopen rule).
  Regression: check C.
- **A silent successful turn was flagged "model incompatible".** A reasoning model ran the tools then
  ended with `finish_reason=stop` + empty text (reasoning-only output); the empty-after-tool guard
  misread that as incompatibility. Fix: a native tool call already executed proves compatibility, so a
  recognized terminal reason (`stop`/`length`/`content_filter`) is never incompatibility. Regression:
  check D (`stop`).
- **`length`/`content_filter` were the missing half of the above.** A reasoning model can burn the
  whole `max_tokens_per_turn` budget → `length` + empty text → still falsely "incompatible". Fix:
  `length` gets an honest "ran out of budget — ask me to continue" note; `content_filter` ends clean.
  Regression: check D (`length` + `content_filter`).
