# Slice 4 — External import (bring a file in from the user's disk)

The literal "interact with the user's system" slice: pull an existing shader file the user has on
disk into the project as a node. Reuses slice 2's **user-file-pick primitive** — no new file-dialog
machinery, and the same corollary-1 guarantee (the model never types a path; the user picks).

## Goal

"Someone sent me this `.glsl`, add it" / "import this shader file" → the copilot opens the picker,
the user selects a `.frag.glsl`, and it becomes a compiled node in the project.

## Design decisions (lock-in)

1. **`import_node(switch_to=false)` — opens the picker for a `.glsl`, creates a node from it.**
   - Flow: `pick_user_file(("glsl",))` (the same GATE-family primitive as slice 2 — worker blocks on
     the file gate, UI polls `pfd` across live frames, abs path consumed engine-side) → on a pick, read
     the file text ENGINE-SIDE → `create_node(<basename stem>, <text>, "", switch_to)` (positional —
     `create_node` is positional-only with a required `template`; pass `template=""` for the default
     starter, `capabilities.py:315`) → `cp.mark_created(new_id)` (revert wiring) → return the new id +
     compile errors + the source basename.
   - `mutating=True` (creates), `gate_policy=NONE` (the file gate is the consent). The imported source
     compiles through the SAME path as `create_node`, so a broken import returns compile errors the
     copilot then fixes — no special-casing.
   - The model never authors the path; the abs path is consumed inside `import_node` on the main thread
     and never returned to the handler. The model gets back `imported -> node <id> (2 compile errors)`.

2. **Import is TEXT-only (a single `.glsl`), not a node DIR / bundle.** A shared `.frag.glsl` is the
   common case; importing a whole node dir (with `media/` + `node.json`) is a bundle format we don't
   have. If the file references a `sampler2D`, the import creates the node and the copilot then offers
   `bind_media` (slice 2) — the two slices compose. Trigger for a bundle importer: a real
   "export/import a node as a shareable package" need (its own feature, pairs with the parked 051
   examples project).

3. **Image/video import is already `bind_media` (slice 2)** — not a separate `import_media` tool.
   "Import an image" and "bind a texture" are the same act (pick a file → into the project). Keeping
   ONE picker-backed media tool avoids a redundant tool (skill §4).

## Out of scope

- Importing from a URL / clipboard — the picker is the file channel; a URL is a different (network)
  surface the copilot doesn't have. Trigger: a user pastes a shadertoy URL and wants it imported
  (that's a fetch+translate feature, not a file import).
- Exporting a node OUT to disk as a file — the deliverable path is render/publish; a "save this node
  as a .glsl file" is the inverse and low-demand. Trigger: a user asks to hand a node's source to
  someone outside ShaderBox.

## Files touched

- `copilot/capabilities.py` — `import_node` Protocol method (returns the `create_node` shape).
- `copilot/backend.py` — `import_node`: `pick_user_file(("glsl",))` → read → `create_node`.
- `copilot/tools/` — the `import_node` tool (in `tools/node_ops.py` or beside `create_node` in
  `tools/shader.py`). Lazy once slice 0 lands.
- `copilot/prompt.py` — one line in the NODES section (import opens a picker; the user selects the
  file; a broken import returns compile errors to fix).
- `tests/_caps.py` — a `_FakeCaps` field + `minimal_caps` default for `import_node` (and
  `pick_user_file` if slice 2 didn't already add it).
- `constants.py` — a `GLSL_EXTENSIONS` tuple for the picker filter (`.glsl`, `.frag`, `.frag.glsl`).

## Manual verification

- **Falsifier (import compiles):** import a known-good `.frag.glsl` → a new node appears, compiles
  clean, its source matches the file byte-for-byte (assert). Import a known-BROKEN one → the node is
  created AND the result carries the compile errors (not a silent success).
- **Falsifier (path never leaks):** same as slice 2 — grep trace + history for the absolute path;
  only the basename may appear.
- **Cancel:** dismiss the dialog → `user cancelled`, no node created.
- **Revert falsifier (mark_created wiring):** import a node, then revert the turn → the imported node
  is GONE. Without the `cp.mark_created(new_id)` call the node survives revert (the "defined but not
  connected" trap its `duplicate_node` sibling guards against).
- **Sampler compose:** import a shader that declares `sampler2D u_tex` → node created, working-set row
  shows `u_tex sampler2D <- (no media bound)`, and `bind_media` then wires it (the two slices compose).
