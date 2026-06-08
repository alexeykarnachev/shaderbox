# 020 · 30 — Copilot turn rollback (checkpoint + revert)

> **STATUS: SPEC STUB — not plan-locked.** Filed mid-brainstorm so the design isn't lost while a
> UI/UX refactor sweep runs first. The mechanism is decided in shape; several decisions below are
> still OPEN (marked). Do NOT implement until plan-locked with the maintainer.

## The problem (why this exists)

The copilot's editing tools overwrite files in place with **zero capture of prior content** —
`backend.py::_copilot_persist_shader` does `node.source.path.write_text(new_text)` and the old
source is gone. All three editing tools (`edit_shader` / `replace_lines` / `insert_after`) funnel
through the one choke point `_copilot_persist_target`. There is no undo, no snapshot, nothing on
disk or in memory.

Observed failure (trace `copilot_dev_2026-06-08_17-05-43`): the agent edited the WRONG node
(matched the word "sphere" in "project this onto a sphere" to a node NAMED "Raymarched Sphere"
instead of resolving "this" to the current node), then across the next two turns compounded the
damage — rewrote a second node "correcting" it, and finally "reverted" by **reconstructing a
generic shader from memory** (NOT the original). The user's real work was destroyed with no way
back. The agent-driven revert is the anti-pattern: the harness must hold a real byte-snapshot, not
trust the agent to rebuild.

(The wrong-node TARGETING bug is a SEPARATE concern — a prompt fix — tracked apart from this.
This feature is purely about RECOVERABILITY.)

## How Claude Code solves the analogous problem
Claude Code's Edit/Write don't keep their own undo store — they delegate to git + the editor undo
stack; the user reverts with `git checkout` / editor undo. The durable lesson: the safety net is a
checkpoint of file state taken BEFORE the tool mutates, restorable independently of the agent.
ShaderBox projects aren't git repos (a shipped user's project won't be), so we build the checkpoint
ourselves — exactly as `delete_node` already does (move dir → `project_dir/trash/` → a Recover
button → `restore_node_from_trash`). That delete→trash→Recover triad is the proven in-repo
precedent to mirror.

## Goal
Per USER-TURN checkpointing: capture the pre-turn state of everything the copilot mutates that turn,
and give the user a per-message **Revert** button (near the copy icon) that restores that state.

## Mechanism (decided in shape)
- **Capture = tools log what they touch** (maintainer's call — a whole-project snapshot per turn is
  too large a footprint). Each mutating operation, before it writes, records the target's pre-state
  into the turn's checkpoint container. The natural seam for SOURCE edits is the single choke point
  `_copilot_persist_target` (top of the method, `tgt.source` = pre-edit content, `tgt.ws_address` =
  stable key) — so all three editing tools are captured at ONE point, not three.
- **Checkpoint container** = a per-turn object holding `{ws_address → pre-edit bytes}` for files,
  plus whatever non-file state the turn touched (see OPEN below), plus the conversation length /
  message index at turn start (so a revert can truncate or annotate the chat).
- **Turn lifecycle** = the container opens at turn start (before the worker runs) and is sealed at
  turn done, then attached to that turn's user Message so the chat can render its Revert button.

## What to capture (the mutation surface — grounded in backend.py)
Every copilot operation that changes durable/visible state:
| Tool | Mutates | Capture |
|---|---|---|
| `edit_shader` / `replace_lines` / `insert_after` | `shader.frag.glsl` (node OR `lib:` file) | pre-edit source bytes @ `_copilot_persist_target` |
| `set_uniform` | `node.json` `uniform_values[name]` (in-memory until save) | **OPEN** — pre-value; coupled to file (see below) |
| `create_node` | a new node dir exists | reverse = delete the created node |
| `delete_node` | node dir → trash | reverse = restore from trash (already exists) |
| `switch_node` | current-node pointer | pre-switch current id (cheap) |
| renders / publishes | external artifacts (files in `renders/`, Telegram/YouTube) | **NOT revertable** — out of scope, external + irreversible |

## OPEN questions (must resolve before plan-lock)
1. **Restore scope — files only, or also uniforms / node.json / node existence?** The maintainer
   leaned "only files" for footprint, but flagged uniforms probably need it too. KEY COUPLING the
   maintainer surfaced: a uniform value sometimes can't be restored without restoring the file (a
   uniform that only exists because the edited source declares it — restoring the old value against
   new source, or vice versa, is incoherent). So uniform-restore may have to be ATTACHED to its
   node's file-restore as one unit, not independently selectable. Needs a concrete model.
2. **Conversation: truncate vs. annotate?** Two maintainer proposals:
   (a) **Truncate** — on revert, drop chat history up to & including that user message, and put the
       user's original text BACK into the input field (re-prompt from a clean state).
   (b) **Annotate (maintainer's later preference)** — keep the conversation, restore the files, and
       inject a SYSTEM message into history so the copilot SEES the revert and keeps context
       coherent (its history said "I edited X"; the file is now un-X'd — the system note reconciles
       that). Avoids the drift where history describes edits no longer on disk.
3. **Selective revert modal?** The maintainer wants a modal on Revert-click to UNCHECK individual
   changes (restore file A, keep file B). Interacts hard with #1's coupling: if uniforms are welded
   to their file, the modal's unit of selection is the node, not the field. Modal needed v1, or a
   v2 follow-on after the all-or-nothing revert proves out?
4. **Retention / persistence.** Do checkpoints survive a restart (persisted under the project like
   `conversation.json`, pruned with Clear/archive) or are they session-only in memory? Session-only
   desyncs from the already-PERSISTED conversation. Likely persisted; cap TBD.
5. **Storage location + format.** `project_dir/copilot/checkpoints/<turn_id>/...`? Bytes-per-touched-
   file. Reuse the `ConversationStore` versioned/fail-soft pattern.

## Out of scope (this feature)
- The wrong-node TARGETING fix (prompt-level — separate).
- Reverting external artifacts (renders, Telegram/YouTube publishes) — irreversible by nature.
- Cross-node "do the same to C" derived-edit memory (separate `todo.md` deferral).

## Files this will touch (anticipated, not locked)
`copilot/backend.py` (capture seam at `_copilot_persist_target` + the mutation methods),
`copilot/session.py` (turn lifecycle open/seal the container), `copilot/state.py` +
`copilot/persistence.py` (the checkpoint model + persistence, versioned), `widgets/copilot_chat.py`
(the Revert button + the selective-revert modal), `app.py` (the restore orchestration — mirrors
`recover_deleted_node` / `restore_node_from_trash`), `ui_primitives.py` (a revert glyph button).

## Manual verification (when built)
Drive a multi-edit turn across ≥2 nodes, Revert, confirm both files + (if in scope) uniforms return
to exact pre-turn bytes; confirm the conversation behaves per decision #2; confirm a selective
revert (if v1) keeps the unchecked node.
