# 020 · 30 — Copilot turn rollback (checkpoint + revert)

> **STATUS: PLAN-LOCKED — ready for pre-implementation review.** Mechanism, data model, and all four
> forks resolved (decisions 1-9 below). Next: pre-impl review, then implement.

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
   writes a node dir as one atomic unit: `shader.frag.glsl` + `node.json` (which holds uniform
   VALUES, ui_state, name, sort key) + `media/` + `textures/`. So source and uniforms are already
   ONE file-system unit. This DISSOLVES the uniform↔file-coupling problem the maintainer raised:
   snapshot the dir, restore the dir, and source+uniforms+name come back coherent by construction —
   no "restore a uniform against new source" incoherence is possible. It also makes this the SAME
   primitive as the existing `delete_node`→trash / `restore_node_from_trash` (a directory move).
2. **Capture = tools log what they touch (the maintainer's call — a whole-project snapshot per turn
   is too large).** A turn-checkpoint container records, per touched node, a COPY of its dir taken
   the FIRST time that node is mutated this turn (subsequent edits to the same node don't re-snapshot
   — the pre-turn state is what matters). The capture is driven from the mutation methods in
   `backend.py`, NOT a per-field hook. The seams:
   - SOURCE edits (all 3 editing tools) → ONE seam at `_copilot_persist_target` (it already funnels
     node + lib edits; `tgt` identifies the target before the write). Capturing here auto-covers any
     future editing tool that routes through it.
   - `set_uniform` → snapshot the node dir on first touch (the value lives in `node.json`).
   - `create_node` → record "this node is NEW this turn" (reverse = delete it, no dir to snapshot).
   - `delete_node` → already trashes the dir; record the trash_name (reverse = restore from trash).
   - `switch_node` → record the pre-switch current-node id (cheap; reverse = switch back).
   - LIB-file edits (`lib:` target) → snapshot the lib file's pre-edit bytes (it lives outside any
     node dir, under the shader-lib root).
   A NEW mutating tool MUST register with the container or its change escapes the net — this is the
   one fragility of the log-what-you-touch model, called out in `conventions.md` + the `todo.md`
   trigger.
3. **Conversation on revert = ANNOTATE, not truncate (the maintainer's later preference).** Revert
   restores the files and appends ONE system/notice message to BOTH the render messages and the LLM
   history ("Reverted the changes from turn <user-msg-excerpt>: nodes X, Y restored to their state
   before that turn."). The agent SEES it, so its history ("I edited X") stays coherent with disk
   (X is un-edited). This avoids the truncate-drift where history describes edits no longer on disk.
   (Truncate-and-refill-the-input was the rejected alternative — it loses the conversation and
   desyncs from the persisted history.)
4. **Persisted, pruned with the conversation.** Checkpoints live under the project
   (`project_dir/copilot/checkpoints/<turn_id>/`), survive a restart like `conversation.json`, and
   are cleared/archived together with the conversation on Clear (a checkpoint with no conversation to
   attach to is dead weight). Mirrors the feature-022 persisted-conversation posture.
5. **A revert is itself a turn-shaped action, but does NOT create its own checkpoint.** Reverting is
   a restore, not a new mutating turn — it appends its notice but takes no new snapshot (you can't
   "un-revert" via this mechanism; the prior turns' checkpoints still stand if the user wants to go
   further back). Reverts apply newest-to-oldest semantics: reverting turn N restores the state as of
   *before* turn N (later turns' edits on the same node are also undone, since the snapshot predates
   them — this is the honest meaning of "undo everything from this turn onward on these nodes").
   **OPEN nuance** — see open question 2.

## The mutation surface (grounded in backend.py)
| Tool | Mutates | Checkpoint capture | Reverse |
|---|---|---|---|
| `edit_shader`/`replace_lines`/`insert_after` (node) | node dir | copy dir on first touch | restore dir |
| `edit_shader`/`replace_lines`/`insert_after` (`lib:`) | lib file | copy file bytes on first touch | rewrite bytes |
| `set_uniform` | `node.json` | copy dir on first touch | restore dir |
| `create_node` | new node dir exists | mark NEW | delete the node |
| `delete_node` | dir → trash | record trash_name | restore from trash |
| `switch_node` | current-node ptr | record pre id | switch back |
| render / publish | external artifacts | NOT captured | NOT revertable (out of scope) |

## Out of scope
- The wrong-node TARGETING fix (prompt-level — `todo.md`).
- Reverting external artifacts (renders, Telegram/YouTube publishes) — irreversible by nature; the
  notice can MENTION an external action happened but can't undo it.
- Cross-node "do the same to C" derived-edit memory (separate `todo.md` deferral).
- A selective-revert modal (uncheck individual nodes) — see open question 3; likely a v2.

## Files touched (anticipated)
- `copilot/backend.py` — the capture calls at `_copilot_persist_target` + `set_uniform` /
  `create_node` / `delete_node` / `switch_node`; a `restore_checkpoint(turn_id)` orchestration (or
  it lives App-side mirroring `recover_deleted_node`).
- `copilot/session.py` — open the container in `enqueue_turn` (main-thread, before the worker), seal
  it at turn done, attach its id to the turn's user `Message`.
- `copilot/state.py` — the checkpoint container dataclass + a `checkpoint_id` (or `RevertInfo`) on
  `Message` (mirrors the existing `RecoverInfo` on the delete card).
- `copilot/persistence.py` — persist the checkpoint index in `ConversationStore` (versioned bump);
  the dir copies live on disk under `copilot/checkpoints/`.
- `widgets/copilot_chat.py` — the Revert button on a user bubble (near the copy icon) + its confirm.
- `ui_primitives.py` — a revert/undo glyph button (drawn, no font dep — like `copy_icon_button`).
- `app.py` — the restore orchestration if it needs `App`-level node reload (mirrors
  `restore_node_from_trash` / `recover_deleted_node`).

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
- Multi-edit turn across ≥2 nodes → Revert → both nodes back to exact pre-turn source + uniforms;
  the conversation gains the notice and the agent's next turn sees coherent state.
- `set_uniform`-only turn → Revert restores the value.
- `create_node` turn → Revert deletes the created node.
- `delete_node` turn → Revert restores from trash.
- Restart the app mid-conversation → the Revert buttons still work (persistence).
- A media-bearing node edited by the copilot → Revert restores text, media untouched (per Q4).
