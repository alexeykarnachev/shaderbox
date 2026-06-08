# 020 · 30 — Copilot turn rollback (checkpoint + revert)

> **STATUS: PLAN-LOCKED + pre-impl review folded — ready to implement.** Mechanism, data model, and
> all four forks resolved (decisions 1-14). The pre-impl review (2 agents, anchored to the real code)
> caught two blockers that overturned the original disk-copy model — the corrected model is
> serialize-the-LIVE-node on capture + reload-and-replace on restore (decisions 1-2, 10-14). Next:
> implement in one coherent diff.

## The problem (why this exists)

The copilot's editing tools overwrite files in place with **zero capture of prior content** —
`backend.py::_copilot_persist_shader` does `node.source.path.write_text(new_text)` and the old
source is gone. All three editing tools (`edit_shader` / `replace_lines` / `insert_after`) funnel
through the one choke point `_copilot_persist_target`. There is no undo, on disk or in memory.

Observed (trace `copilot_dev_2026-06-08_17-05-43`): the agent edited the wrong node, compounded the
damage across two more turns, and finally "reverted" by reconstructing a generic shader **from
memory** (not the original). The user's work was destroyed with no way back. The lesson: the harness
must hold a real byte-snapshot taken BEFORE the turn mutates — never trust the agent to rebuild.

(The wrong-node TARGETING bug is SEPARATE — a prompt fix, tracked in `todo.md`. This feature is
purely RECOVERABILITY.)

## Goal
Per USER-TURN checkpointing: snapshot the pre-turn state of every node the copilot mutates that
turn, and give the user a per-user-message **Revert** button (near the copy icon) that restores that
state — files (source + uniforms + name, as one coherent unit) AND reconciles the conversation.

## Design decisions (resolved — grounded in the code)

1. **The unit of capture/restore is the NODE DIRECTORY, not a field.** `ui_models.save_ui_node`
   writes a node dir as one unit: `shader.frag.glsl` + `node.json` (which holds uniform VALUES,
   ui_state, name, sort key) + `media/` + `textures/`. So source and uniforms serialize TOGETHER —
   which dissolves the uniform↔file-coupling problem. **BUT (pre-impl review, blocker): disk is NOT
   the live state.** The running app holds a live `Node` (`source.text`, `uniform_values` dict, a
   compiled GL program) + an open `EditorSession`; and `set_uniform` writes ONLY the in-memory
   `uniform_values` dict, never `node.json` (`backend.py::set_uniform`). So a blind dir-copy captures
   a STALE `node.json` and a blind dir-restore is a no-op against the live value. The corrected model:
   - **CAPTURE = serialize the LIVE node, not copy the on-disk dir.** At first-touch, call
     `save_ui_node(live_node, into=<checkpoint_dir>)` so the snapshot reflects in-MEMORY state
     (the just-about-to-change uniform value, the pre-edit source). For a lib file (no node), copy its
     pre-edit bytes (disk IS the source of truth for a lib file — it has no in-memory Node).
   - **RESTORE = reload-and-replace, not a file overwrite.** Mirror `restore_node_from_trash`: copy
     the snapshot files over `nodes/<id>/`, then `load_node_from_dir` → REPLACE `ui_nodes[id]` with a
     fresh `Node` (rebuilds `uniform_values` + a fresh GL program), release the old Node's GL,
     re-sync any open editor session (`sync_editor_from_disk(id, restored_text)` — no-ops if no
     session), and re-point `current_node_id` if needed. Restore updates FOUR live surfaces: the
     `ui_nodes` entry, the GL program, `uniform_values`, and the editor session.
   So this is the SAME primitive as `delete_node`→trash / `restore_node_from_trash` ONLY for the
   create/delete cases; the EDIT and SET_UNIFORM cases are reload-and-replace — new code, not a reused
   move.
2. **Capture = tools log what they touch (the maintainer's call — a whole-project snapshot per turn
   is too large).** A turn-checkpoint container records, per touched node, a `save_ui_node` serialize
   of the LIVE node taken the FIRST time that node is mutated this turn (subsequent edits don't
   re-snapshot — pre-turn state is what matters). Driven from the mutation methods in `backend.py`:
   - SOURCE edits (all 3 editing tools) → ONE seam at `_copilot_persist_target` (it funnels node +
     lib edits; `tgt` identifies the target BEFORE the write). Auto-covers any future tool routing
     through it.
   - `set_uniform` → serialize the live node on first touch (the value is in-memory `uniform_values`).
   - `create_node` → record "this node is NEW this turn" (reverse = delete it → trash; no snapshot).
   - `delete_node` → already trashes the dir; record the trash_name (reverse = restore from trash).
   - `switch_node` → record the pre-switch current-node id (cheap; reverse = switch back).
   - LIB edits (`lib:` target) → snapshot the lib file's pre-edit bytes; reverse rewrites them AND
     re-invalidates consumer nodes via `_copilot_invalidate_lib_consumers` (the forward path does
     this — a byte-only rewrite leaves consumers compiled against the edited lib, stale).
   A NEW mutating tool MUST register with the container or its change escapes the net — the one
   fragility of log-what-you-touch, called out in `conventions.md` + the `todo.md` trigger.
3. **Conversation on revert = ANNOTATE, not truncate (the maintainer's later preference).** Revert
   restores the files and appends ONE system/notice message to BOTH the render messages and the LLM
   history ("Reverted the changes from turn <user-msg-excerpt>: nodes X, Y restored to their state
   before that turn."). The agent SEES it, so its history ("I edited X") stays coherent with disk
   (X is un-edited). This avoids the truncate-drift where history describes edits no longer on disk.
   (Truncate-and-refill-the-input was the rejected alternative — it loses the conversation and
   desyncs from the persisted history.)
4. **Persisted across restart; DELETED (not archived) on Clear.** Checkpoints live under the project
   (`project_dir/copilot/checkpoints/<turn_id>/`) and survive a restart like `conversation.json` —
   so the Revert buttons work after reopening the app. On **Clear**, the conversation is ARCHIVED
   (recoverable, per feature 022) but the checkpoints are **deleted outright** — they're throwaway
   recovery scratch, not history worth keeping, and a checkpoint with no live conversation to attach
   a Revert button to is dead weight. So `_clear`/`reset_conversation` does an `rmtree` of
   `copilot/checkpoints/`, NOT a move-to-archive. (Maintainer: "when the user clears up the session,
   no need to preserve the checkpoints.")
5. **A revert is itself a turn-shaped action, but does NOT create its own checkpoint.** Reverting is
   a restore, not a new mutating turn — it appends its notice but takes no new snapshot (you can't
   "un-revert"; the prior turns' checkpoints still stand if the user wants to go further back).
   Reverting turn N restores the state as of *before* turn N (later turns' edits on the same node are
   also undone, since the snapshot predates them — the honest meaning of "rewind to before here", per
   decision 7).

10. **Capture is BEST-EFFORT — a capture failure NEVER fails the edit (pre-impl review).** The
   capture runs on the hot mutation path (inside the bridge `_on_main`, BEFORE the write — it needs
   pre-edit state). If `save_ui_node`-into-snapshot raises (disk full, permission), it is caught +
   `logger.warning`'d + swallowed; the edit proceeds and returns its normal result. The container
   records that node as "not snapshotted" so a later Revert skips it with a clear "couldn't restore
   node Y (no snapshot)" notice. The line the maintainer drew: an UNcheckpointed edit is degraded
   (acceptable); an edit that FAILS because checkpointing failed is a regression (not acceptable).
11. **Restore reconciles against CURRENT disk reality, not an assumed in-place node (pre-impl
   review).** A snapshot is a full serialized node, so restore is well-defined even when the node was
   changed by a later turn: if the touched node's dir is ABSENT from `nodes/` at revert time (a later
   turn deleted it → it's in trash), restore RE-CREATES it from the snapshot (reappears in
   `ui_nodes`) rather than assuming an in-place overwrite. A create-then-(reverting-the-create) of a
   node a later turn already deleted is a no-op (guard `id in ui_nodes`). The confirm modal lists each
   node by its CURRENT name (which may have been renamed since the snapshot).
12. **Revert is gated on `in_flight` / `copilot_turn_active`, exactly like the Recover button (pre-impl
   review).** The revert glyph + its confirm are `begin_disabled` while a turn runs (mirror
   `copilot_chat.py`'s Recover gate). This is the invariant that makes the annotate-history (decision
   3) safe: the notice is appended to `history` ONLY while the worker is idle, so it can never land
   mid-turn or orphan a `tool_call_id` (the within-turn `messages` array with tool pairings is
   rebuilt from `history` each turn and discarded — Revert can't touch it). The notice is a plain NL
   user/assistant message (no tool role), consistent with the NL-only history rule; persist via
   `save_conversation` right after, exactly like `recover_deleted_node`.
13. **The checkpoint container is sealed + persisted where `save_conversation` already fires — in
   `ui.py` at the `copilot_turn_active` True→False transition, NOT a session.py "turn done" (pre-impl
   review).** Open + attach `checkpoint_id` to the user Message in `enqueue_turn` (main-thread, OK);
   capture writes happen inside the backend `_on_main` blocks (already main-thread — no new
   thread-safety surface, single-writer with `state`); seal + persist the checkpoint index beside the
   existing `save_conversation` call so a crash never loses it and it survives a project switch like
   `conversation.json`. Capture keys on the ACTIVE turn id (a getter the backend reads, mirroring
   `get_current_node_id`) so a deferred op (a `defer=True` render) from a prior turn can't leak into
   the next turn's container.
14. **Retention bound (pre-impl review).** Checkpoint snapshots are full per-node serializes (text
   only, per decision 9 — small, but non-zero), one per touched node per turn, inside the PORTABLE
   project dir. Cap retention: keep checkpoints only for turns whose user Message is still in the live
   conversation; when a turn's Message is pruned/compacted out (or on Clear, decision 4), its
   checkpoint dir is collectable. (No unbounded growth in a dir the user copies around.)

## The mutation surface (grounded in backend.py)
| Tool | Mutates | Checkpoint capture (first touch) | Reverse |
|---|---|---|---|
| `edit_shader`/`replace_lines`/`insert_after` (node) | live `Node.source.text` + disk source | `save_ui_node(live)` into snapshot | reload-and-replace from snapshot |
| `edit_shader`/`replace_lines`/`insert_after` (`lib:`) | lib file bytes | copy pre-edit bytes | rewrite bytes + invalidate consumers |
| `set_uniform` | in-memory `uniform_values` (NOT node.json) | `save_ui_node(live)` into snapshot | reload-and-replace from snapshot |
| `create_node` | new node in `ui_nodes` + dir | mark NEW (no snapshot) | `_delete_node_unguarded` → trash (guard `id in ui_nodes`) |
| `delete_node` | dir → trash | record trash_name | restore from trash |
| `switch_node` | `current_node_id` | record pre id | switch back |
| render / publish | external artifacts in `renders/` | NOT captured | NOT revertable (out of scope) |

## Out of scope
- The wrong-node TARGETING fix (prompt-level — `todo.md`).
- Reverting external artifacts (renders, Telegram/YouTube publishes) — irreversible by nature; the
  notice can MENTION an external action happened but can't undo it.
- Cross-node "do the same to C" derived-edit memory (separate `todo.md` deferral).
- A selective-revert modal (uncheck individual nodes) — dropped, not deferred (decision 8).
- True per-turn isolation (undo only this turn, keep later turns) — needs a diff/replay model (decision 7).

## Files touched (anticipated)
- `copilot/backend.py` — best-effort capture calls at `_copilot_persist_target` + `set_uniform` /
  `create_node` / `delete_node` / `switch_node` (each serializes the live node into the snapshot,
  decision 2); the active-turn-id getter the capture keys on (decision 13).
- `app.py` — the `restore_checkpoint(turn_id)` orchestration (App-side, mirrors
  `recover_deleted_node` / `restore_node_from_trash`): reload-and-replace each touched node
  (`load_node_from_dir` → replace `ui_nodes[id]` → release old GL → resync editor session →
  recompile → re-point `current_node_id`), lib-byte rewrite + `_copilot_invalidate_lib_consumers`,
  create-reverse via `_delete_node_unguarded` (guarded), the absent-node re-create path (decision 11).
- `copilot/session.py` — open the container in `enqueue_turn` (main-thread, before the worker),
  attach its id to the turn's user `Message`; the capture-into-container API the backend calls.
- `ui.py` — SEAL + persist the checkpoint index at the `copilot_turn_active` True→False transition,
  beside the existing `save_conversation` call (decision 13).
- `copilot/state.py` — the checkpoint container dataclass + a `checkpoint_id` (or `RevertInfo`) on
  `Message` (mirrors the existing `RecoverInfo` on the delete card).
- `copilot/persistence.py` — persist the checkpoint index in `ConversationStore` (versioned bump);
  the per-node serializes live on disk under `copilot/checkpoints/<turn_id>/`; `rmtree` on Clear
  (decision 4) + the retention prune (decision 14).
- `widgets/copilot_chat.py` — the Revert glyph on a user bubble (near the copy icon) + its confirm
  modal, gated on `in_flight` (decision 12).
- `ui_primitives.py` — a revert/undo glyph button (drawn, no font dep — like `copy_icon_button`).

## Locked decisions (from the plan-lock conversation)
6. **Revert affordance = a small corner glyph on the user bubble + a CONFIRM MODAL that explains the
   consequence before it fires.** The glyph mirrors `copy_icon_button` (drawn, no font dep). Clicking
   it opens a modal that spells out what reverting will do ("Restore <node names> to their state
   before this message? The copilot's edits since then will be undone.") with Confirm / Cancel — not
   a silent action. (Reuses the `modal_window` primitive + the drawn-glyph button pattern.)
7. **Revert meaning = "rewind to before this message" (the simple snapshot-restore).** Reverting a
   turn restores every node it touched to that node's pre-turn snapshot; if a later turn also edited
   the same node, that later edit on that node is undone too (the snapshot predates it). True
   per-turn isolation (undo ONLY this turn, keep later turns) is explicitly NOT built — it needs a
   diff/replay model, not snapshot-restore. (Maintainer confirmed the simple semantics.)
8. **No per-node selection — revert is all-or-nothing for the turn.** The selective-revert modal
   (uncheck individual nodes) is dropped, not deferred-with-intent: the confirm modal (decision 6)
   lists what WILL be restored, but offers no checkboxes. (Maintainer: "no need for nodes selection".)
9. **Snapshot scope = ONLY `shader.frag.glsl` + `node.json` per touched node (never `media/` /
   `textures/`).** Verified against `backend.py`: the copilot's render tools (`render_image` /
   `render_video`) write to a SEPARATE `<project>/renders/` output dir (external artifacts, out of
   scope — like a publish), NOT into the node dir. A node dir's `media/` / `textures/` are populated
   ONLY by media-binding (a uniform holding a `MediaWithTexture`/`Texture`), which the copilot has NO
   tool for (the `bind_media` deferral). So within a node dir the copilot only ever mutates the two
   text files — snapshotting just those is both small AND complete. A revert does NOT delete
   `renders/` artifacts the turn produced (the user asked for + gated them; the notice may MENTION
   them as kept). Re-verify this scope if `bind_media` ever lands (then a media-binding turn would
   need its `media/`/`textures/` captured too — a trigger for this feature).

## Manual verification (when built)
- **LIVE-surface desync (the #1 risk — assert in-memory, not just disk):** edit a node, SWITCH to it
  so its editor session is OPEN, Revert → the EDITOR pane shows the pre-turn source (not the edited
  text), the PREVIEW renders the pre-turn shader THIS frame (recompiled), and the uniform SLIDERS read
  the pre-turn values. Not just "shader.frag.glsl on disk is correct".
- Multi-edit turn across ≥2 nodes → Revert → both nodes' live state back to pre-turn; the conversation
  gains the NL notice and the agent's next turn sees coherent state.
- `set_uniform`-only turn → Revert restores the value (proves capture serialized LIVE `uniform_values`,
  not the stale on-disk `node.json`).
- `create_node` turn → Revert moves the created node to trash; it leaves `ui_nodes`.
- `delete_node` turn → Revert restores from trash.
- **Edit-then-delete across turns:** turn A edits node X, turn B deletes X, Revert A → X is RE-CREATED
  from the snapshot and reappears in `ui_nodes` (decision 11).
- `lib:`-edit turn → Revert → a node that calls the edited fn recompiles to the pre-turn behavior
  (consumer invalidation, decision 2).
- Restart the app mid-conversation → the Revert buttons still work (persistence, decision 13).
- Clear the chat → `copilot/checkpoints/` is gone (decision 4); the conversation is in `archive/`.
- Capture-failure path (simulate a write error) → the EDIT still succeeds; the Revert later reports
  "couldn't restore node Y (no snapshot)" rather than the edit having failed (decision 10).
