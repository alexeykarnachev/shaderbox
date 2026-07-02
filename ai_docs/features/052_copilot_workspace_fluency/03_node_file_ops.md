# Slice 3 — Node & lib file operations

A bag of small, independent "manipulate the files of the workspace" tools. Each is its own task; do
them in any order. All are content/handle-addressed (a node id or a `lib:` address the model copies
from a read — never a raw path).

## Goal

Round out the file-manipulation surface a shader author expects: rename a node, fork a node to
experiment, set a node's canvas resolution, and manage lib files.

## Design decisions (lock-in)

1. **`rename_node(node, new_name)`** — set a node's display name (`UINodeState.ui_name` +
   persist `node.json`). `mutating=True`, `gate=NONE` (reversible: rename back). Node dir id is
   unchanged (the id is the stable handle; only the display name moves — same split the prompt already
   enforces: "call nodes by NAME, address by id"). Reverts cleanly (node.json is in checkpoint scope).

2. **`duplicate_node(node, new_name="", switch_to=false)`** — fork a node: copy its dir (source +
   `scripts/script.py` + `media/` + `textures/`) to a fresh uuid, load it as a new `ui_node`, compile,
   return the new id + errors (same return shape as `create_node`). `new_name` defaults to
   "<name> copy". `mutating=True`, `gate=NONE` (it CREATES, like create_node). The playground staple —
   "try a variant without losing this one". **Revert wiring:** like `create_node` (`backend.py:971`),
   `duplicate_node` MUST call `cp.mark_created(new_id)` so the reverse is a delete of the fork — name
   this call (a create with no `mark_created` escapes the revert net, `30_turn_rollback.md` decision 2).
   NOTE: the copy includes `media/` — a duplicated node's media is a fresh copy on disk (no shared
   reference), so binaries land in the new dir (acceptable — the user asked to fork).

3. **`set_canvas_size(node, width, height)`** — set a node's native canvas resolution
   (`node.canvas.set_size` + persist `canvas_size` in `node.json`). Clamp to sane bounds (reuse the
   UI's canvas-size limits if any; else a documented min/max). `mutating=True`, `gate=NONE`
   (reversible). Pairs with slice-1 canvas awareness — the working-set header shows the change took.
   Reverts cleanly (node.json in scope). This is what answers "make this 1080x1080" / "why is it
   blurry".

4. **`delete_lib_file(path)`** — the parked destructive lib op; its un-park trigger IS met (feature
   020 `17_gate_ui.md` parked it "until the publish-tools wave lands"; publish shipped). Move a
   `lib:<path>` file to the shader-lib trash (reuse `ShaderLibFileManager` delete-to-trash — same
   `.trash/` the picker uses). `mutating=True`, `gate_policy=ALWAYS` (destructive — matches
   `delete_node`). After delete, invalidate consumers (a node calling a now-gone `SB_*` recompiles with
   an error next step — honest feedback). Checkpoint: capture the pre-delete bytes for restore (reuses
   `snapshot_lib`/`_revert_lib_file`, `revert.py:172`). **Impl-verify:** `_revert_lib_file` re-creates
   via `resolve_copilot_path(rel)` on a now-absent path — confirm it resolves an absent target for the
   re-create case (the delete-then-revert falsifier below catches a miss).

5. **`rename_lib_file` — CUT (round-1 review).** Dropped, not deferred. Two reasons, both from review:
   (a) it is **derivable** from tools that exist — `write_shader` to the new `lib:` address +
   `delete_lib_file` the old — so it only adds a schema the model must be taught for something it can
   already compose (`conventions.md ## Speculative machinery`: a surface you must teach/maintain gets
   cut); (b) as locked it had NO revert path (`revert.py` has no lib-rename reverse) and NO falsifier —
   shipping it would be an actual half-feature, the very thing "no half-features" forbids. Cutting it is
   MORE aligned with the mandate than keeping it. (This reverses the earlier O-5 "IN" — see
   `99_synthesis.md` Review history.)

## Out of scope

- Moving a node between projects — the copilot is single-project (feature 020 §13 cross-project
  deferral stands). Trigger: a real multi-project workflow.
- Reordering nodes in the grid / node folders / tags — UI-organizational, not a copilot need. Trigger:
  a user asks the copilot to organize the grid.
- Editing a node's uniform CONTROL look (slider vs drag, ranges) — the prompt already states the
  copilot changes a control's VALUE or DECLARATION, not its look; unchanged here.

## Files touched

- `copilot/capabilities.py` — Protocol methods + result value objects (`rename` → reuse a small
  ok/error struct; `duplicate` → the `create_node` return shape; `set_canvas_size` → ok/error).
- `copilot/backend.py` — implementations (all main-thread: node mutation + persist + GL for canvas
  resize / duplicate compile).
- `copilot/tools/` — extend `tools/shader.py` (node-scoped ops) or a small `tools/node_ops.py`;
  `delete_lib_file` next to the lib tools. Lazy once slice 0 lands.
- `copilot/prompt.py` — a NODES/LIBRARY note extension (rename/duplicate/canvas; delete_lib_file is
  gated + irreversible).
- `copilot/backend.py` — `rename_node`/`set_canvas_size`/`duplicate_node` call `_capture_node`
  (duplicate also `cp.mark_created(new_id)`); `delete_lib_file` reuses `snapshot_lib`/`_revert_lib_file`
  (`revert.py`). No new checkpoint scope.
- `tests/_caps.py` — a `_FakeCaps` field + `minimal_caps` default for each new method (`rename_node`,
  `duplicate_node`, `set_canvas_size`, `delete_lib_file`) — else pyright Protocol conformance fails.

## Manual verification

- **rename:** rename a node → the map + working-set header show the new name; the id is unchanged;
  reverting the turn restores the old name (node.json in checkpoint scope).
- **duplicate:** duplicate a node with a bound texture + a script → the new node compiles, has its OWN
  `media/` copy (not a shared ref — assert two files on disk), and editing the copy does NOT touch the
  original (falsifier: edit copy, assert original source byte-identical). **Revert falsifier:**
  duplicate, then revert the turn → the fork is GONE (asserts `mark_created` was wired; without it the
  fork survives revert).
- **set_canvas_size:** set 256→1024 → the working-set header reads `canvas 1024x...`; a `render_image`
  at NATIVE returns the new size (codec-snap aside); revert restores 256.
- **delete_lib_file:** delete a lib file a node calls → confirm gate fires; on confirm, the consumer
  node recompiles with a missing-`SB_*` error next step; the file is in `.trash/`; **revert restores
  the file at its original path** (this exercises `_revert_lib_file`'s absent-target re-create — the
  falsifier for the impl-verify note in decision 4; a broken re-create leaves the file gone after
  revert).
