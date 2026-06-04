# 020 · 22 — Template library (the agent sees + reads + instantiates shipped templates)

The copilot gains a first-class TEMPLATE LIBRARY: the shipped node templates (UV Mango / Media Input /
Text Rendering, + future ones) are always in its prompt context WITH descriptions, so when a user says
"make a shader that renders text" the agent picks the Text Rendering template WITHOUT being told to.
The agent can READ + GREP a template via the EXISTING `read_shader`/`grep` tools (one implementation,
a `template:` address) and INSTANTIATE one via `create_node(template=...)`. The default starter becomes
just-another-template (no hardcoded special-case). The node-creator popup becomes a template-management
center where the user EDITS descriptions. Driven by the maintainer's voice note (tg foo id 1833) + a
deep research pass (`w8fg08xih`).

THE PRINCIPLE (maintainer intent): the template library IS the agent's shader entry point. REUSE the
ordinary-shader machinery for templates — ONE implementation, a different addressing interface, never a
parallel "read template" path. Descriptions serve BOTH the user (a name isn't enough) AND the agent (it
understands a template without reading code each time).

## Goal

1. **Always-in-context catalogue.** Templates (name + `template:<id>` handle + description) render in the
   prompt's warm prefix, parallel to the project map + lib catalogue. The agent picks one on intent.
2. **Read + grep templates via the EXISTING tools.** `read_shader(['template:5372'])` returns a
   `ShaderView` the same way it returns a node; `grep` surfaces `template:` origins. No parallel path.
3. **`create_node(template=...)`** instantiates a named template; the default starter is just the
   conventional default template (UV Mango by id), no special code path.
4. **Editable descriptions.** The node-creator popup shows + edits a template's description; edits
   persist to a user-writable sidecar (shipped resources are read-only in a bundle).

## Out of scope (each with a trigger)

- **Render/publish a template directly.** Templates are READ-ONLY; the agent must `create_node` from a
  template first, then render/publish the resulting node. **Trigger:** a user wants a one-shot "demo
  this template" without keeping the node.
- **Grep over description TEXT.** `grep` stays source-level (the agent already has descriptions in
  context). **Trigger:** a user expects "find the template about X" to match a description phrase.
- **Markdown descriptions.** Plain text only (simplest for prompt + caption + the agent-as-data read).
  **Trigger:** a description needs structure the popup caption can't show.
- **A "reset description to shipped default" button.** Deferred — the sidecar shadows the shipped value
  permanently (user-wins). **Trigger:** a shipped template's purpose changes materially and a stale user
  override misleads the agent.
- **File-watch invalidation of a shipped template edited on disk mid-session.** Templates load once at
  startup. **Trigger:** the dev loop needs live template-source reload.

## Design decisions (numbered, lock-in)

### D1 — Templates ARE nodes; the feature is wiring, not a new model
`ui_node_templates: dict[str, UINode]` are already fully-compiled `UINode`/`Node` objects, identical
shape to `ui_nodes`. `read_shader`'s per-node work (force-compile + `get_active_uniforms` +
`_to_error_infos` + `_number_lines`) is kind-agnostic. So: make templates ADDRESSABLE through the
existing resolve->read/grep path, make the prompt SEE them, make the starter just-a-template, give the
popup a description editor. NO new model, NO parallel read path.

### D0 — Reconcile the in-flight RED tree FIRST (atomic step 1)
The working tree has a half-started skeleton that does NOT import/construct/test (pre-impl B1-B4):
`TemplateEntry` is used in `app.py` but not imported (NameError on import); `template_catalog` is a
REQUIRED `CopilotCapabilities` field but unwired in `_build_copilot_capabilities` (TypeError at App
init); `minimal_caps` (`tests/_caps.py`) lacks it (suite won't collect); `create_node` is 4-tuple in
the capability comment but 3-arg in `_CreateNodeArgs` / the handler / `_copilot_create_node` / the
stub. **Step 1 lands the full vertical slice atomically** (import `TemplateEntry`; wire
`template_catalog=`; add it + the 4-arg `create_node` to `_caps.py`; thread `template` through
`_CreateNodeArgs` -> handler -> `_copilot_create_node`) so `make check` is 0-errors before anything
else. Until then EVERY flow (template or not) is dead at `App.__init__`.

### D2 — Addressing: a `template:<short_id>` prefix mirroring `lib:`
- The catalogue emits the PREFIXED handle `template:5372` (not the bare short id) so the agent copies a
  self-describing token — the exact reason `lib:` is prefixed. Short id = the dir-uuid 4-char prefix.
- A new internal `_copilot_resolve_source(handle) -> (kind, full_id|None)`: `template:` -> strip +
  `_copilot_resolve_template_id`, kind="template"; else `_copilot_resolve_node_id`, kind="node". (`lib:`
  is not a `read_shader` target — `read_lib` owns lib.)
- `read_shader` + `grep` branch on kind but build the SAME `ShaderView`/`GrepHit`. A template's
  `ShaderView.node_id` carries the prefixed `template:5372` so a follow-up read reuses the token.
- **The prefix self-rejects edits — via an EXPLICIT guard.** Add a branch in `_copilot_resolve_target`
  RIGHT AFTER the `if target.startswith("lib:")` check: `if target.startswith("template:"): return
  EditResult(unresolved=True, unresolved_reason="templates are read-only — create_node(template=...)
  from it first, then edit the node")`. NOT incidental non-resolution (today a `template:` target falls
  to `_copilot_resolve_node_id` -> a misleading "no node with id 'template:...'" — the agent loops). The
  explicit guard short-circuits before resolution, survives a future lenient-resolver refactor, and the
  message is ACTIONABLE. `unresolved` (not `stale`) so it counts toward the edit-retry cap.

### D3 — Read-path template branch (NO freshness stamp)
`_copilot_read_shaders` branches on kind: the node path is unchanged (stamps `_copilot_read_revision`);
the template path force-compiles (templates can have a stale `.program`), builds the SAME `ShaderView`,
but does NOT stamp freshness (no edit ever targets a template, so a stamp is dead state). `grep` scans
`ui_node_templates` after `ui_nodes`, emitting `GrepHit(origin="template:<short>", location="template
'<name>'")`. The `read_shader` missing-handle diff tolerates the prefixed short id.

### D4 — `create_node(template=...)` + starter generalization
- `_CreateNodeArgs` gains `template: str = ""` (a `template_catalog` handle, bare or prefixed; empty =
  the default starter). `source` non-empty still overrides the instantiated body.
- `_copilot_create_node` resolves `template` to its dir (empty -> the default-starter id, now a named
  default CONSTANT, not a special path) and `load_node_from_dir`s THAT dir instead of the hardcoded
  `_STARTER_TEMPLATE_ID`. The `CopilotCapabilities.create_node` signature becomes the 4-tuple
  `(name, source, template, switch_to)`.
- `_seed_starter_node` also resolves the default-starter id — but KEEPS its SOFT guard (warn + skip on a
  missing/corrupt default-template dir, NEVER raise). The agent-path `_copilot_create_node` RAISES on a
  missing template (the bridge turns it into a clean tool error); the seed-path must NOT — a broken
  install must degrade first-run to "no node", not crash `App.__init__`. So "one home for the default-
  starter ID" = a shared CONSTANT, NOT a shared raising resolver (pre-impl HIGH).
- Short-id uniqueness: template short ids must be UNIQUE across the shipped set (the catalog emits
  `tid[:4]`, the resolver prefix-matches — a 4-char collision would emit two identical handles AND make
  both unresolvable). Today the 3 uuids differ (`5372`/`73ea`/`f90f`); a test PINS uniqueness so a
  future template with a colliding prefix fails CI, not silently in chat.
- **Empty-template default (manual-review Q):** an empty `create_node` still seeds UV Mango (the
  conventional blank canvas) — the context catalogue STEERS the agent to pick a template on intent, but
  a blank create is not an error. (Research-recommended; matches "blank canvas still exists".)

### D5 — Description persistence: two-tier (shipped default + user sidecar)
- **Shipped default:** `description: str = ""` on `UINodeState` -> lives in each shipped `node.json`
  under `ui_state` (forward-compat, pydantic default, NO migration — the UINodeState additive-field
  precedent). Authored by the maintainer at build time.
- **User override:** a new `TemplateDescriptionsStore` (mirrors `ShaderLibFavoritesStore` byte-for-byte
  in posture — a proper leaf importing only `paths.app_data_dir`, NO App) at
  `app_data_dir()/template_descriptions.json`, a `{full_template_uuid: description}` map, load/save
  fail-soft, keyed by FULL uuid (stable; short ids are display-only). Cross-project (templates are
  global), NOT per-project. LOADED ONCE in `App.__init__` (beside `shader_lib_favorites`), NOT in
  `_init` (a global store must survive `open_project`, not reload per-project).
- **Non-ASCII:** the catalogue render sanitizes the description to ASCII (`sanitize_display`) — the
  prompt's `_sanitize` only strips control chars, and a description rides the warm prefix + the popup
  caption; a stray em-dash/emoji would tofu in both. (Plain-text-only is already out-of-scope.)
- **Lookup precedence (at the CONSUMPTION site, never by mutating the in-memory template):**
  `store.get(full_uuid)` if present else `ui_node.ui_state.description`. A user edit shadows the shipped
  default and survives app updates. (Manual-review Q: user-wins-forever — the dev's later shipped edit
  is never seen once a user edits; accepted, matches favorites/tags.)
- **Write timing:** ON CHANGE (each keystroke -> `store.set` + `save`), like `labeled_text_input` —
  the sidecar is always current, no flush-on-close race.

### D6 — Node-creator popup = template-management center
Below the preview grid (when a template is selected): render its description (via the lookup helper) as
read-only WRAPPED caption text (`push_text_wrap_pos(0.0)` — `imgui.text` clips) + an "Edit description"
affordance that toggles a small multiline (`labeled_multiline_input`) bound to an App-held `InlineInput`
(bind to the template DIR `Path`, not a bare uuid str — `InlineInput.target` is `Path | None`). Edits
apply on-change to the sidecar (D5). The toggle SWAPS the cell (no nested modal). **This popup is the
EVERYDAY "New node" flow, not a template-only path — the editor must not regress it** (pre-impl HIGH):
- **Enter collision:** the popup's `enter_create` (Enter -> create + close) MUST be suppressed while the
  description multiline is focused (track `is_item_focused()` after it / a sticky "input owns keys" flag,
  `/imgui-ui` §7.5) — else pressing Enter to add a newline creates a node + closes the modal.
- **Esc collision:** the modal auto-closes on Esc even with an input focused (`/imgui-ui` §7.5) — give
  the editor an explicit `x`/Cancel affordance; don't rely on Esc to just-close-the-editor.
- **Selection change while editing:** when the selected template changes, CLOSE/re-bind the editor — else
  it shows template A's text but saves to template B's uuid (data corruption). Add
  `reset_template_inline_state()` (mirror `reset_inline_state`), called on modal open + on selection change.
- **Layout:** fixed-height description SLOT (caption OR editor, `/imgui-ui` §2 "one slot not a stack") +
  the grid in a `begin_child` with a scrollbar so grid growth never pushes the editor off the 490x530
  modal. The fit is a maintainer make-run visual check (`/imgui-ui` §0 — can't screenshot).

## Files touched (anticipated)

- `ui_models.py` — `UINodeState.description`.
- `templates_descriptions.py` (new) — `TemplateDescriptionsStore`.
- `app.py` — `template_description(uuid)` helper; `_copilot_template_catalog` (prefixed handle + merged
  description); wire `template_catalog=` into `_build_copilot_capabilities`; `_copilot_resolve_source`;
  `_copilot_read_shaders` + `_copilot_grep` template branch; `_copilot_create_node` template arg +
  starter generalization; `_seed_starter_node` de-hardcode; the popup inline-input state.
- `copilot/capabilities.py` — `TemplateEntry.description`; the 4-tuple `create_node` signature.
- `copilot/context.py` — `CopilotContext.template_catalog` + `_render_template_catalog`.
- `copilot/prompt.py` — the TEMPLATE LIBRARY block + the `template:` ADDRESSING bullet + the read-only note.
- `copilot/tools/shader.py` — `_CreateNodeArgs.template`; the create handler; the missing-handle diff;
  the explicit `template:`-edit reject.
- `popups/node_creator.py` — the description view + edit affordance.
- the 3 shipped `node.json` — authored descriptions.
- tests — template read/grep/create + the sidecar store + the prefixed-handle round-trip.

## Manual verification (maintainer, in-app + the review checkpoints)

- Ask the copilot "make a shader that renders text" WITHOUT naming a template: it should pick + instantiate
  the Text Rendering template (it sees it in context).
- Ask it "is the text-rendering template SDF or texture-based?": it should `read_shader('template:...')`
  or `grep` and answer from the code.
- `create_node(template=...)` makes a node from the right template; an empty create still seeds UV Mango.
- Edit a template description in the node-creator popup: it persists across restart; the agent's next
  turn sees the new description.
- An edit tool targeting `template:...` is cleanly rejected with the ACTIONABLE read-only message.
- Each shipped template READS with NO compile errors (catches a broken-install / lib-drift / a malformed
  `node.json` that would make a template silently vanish from the grid + catalogue).
- A non-ASCII description renders without breaking the prompt or the popup caption (sanitized).
- The everyday "New node" popup flow is unbroken: Enter-in-description adds a newline (does NOT create);
  Esc/Cancel in the editor doesn't nuke the modal unexpectedly; switching templates mid-edit re-binds.
- (Headless: the catalogue in `build_context`, `read_shader('template:<id>')` -> ShaderView, a `template:`
  grep origin, `create_node(template=...)`, the sidecar store round-trip, template short-id uniqueness,
  all 3 templates load (count==3). The popup visual + the 490x530 fit are a maintainer make-run check.)

## Manual-review checkpoints (resolved with research defaults per the auto-approve; flag for the maintainer)

- **Empty-template create** defaults to UV Mango (not an error) — D4.
- **Description precedence** is user-wins-forever (no reset button this wave) — D5 / out-of-scope.
- **Templates are read-only** (no direct render/publish; no grep over description text) — out-of-scope.
- **Plain-text descriptions** — out-of-scope.
- **The 3 shipped descriptions are agent-drafted** (UV Mango / Media Input / Text Rendering) — review the
  wording; they encode product voice + what each template is FOR.

## Review history

- **Pre-impl review (2 agents — correctness/design + blast/UX):** confirmed the architecture (templates
  ARE nodes; wiring not a new model) and the additive posture (the `description` field, the sidecar,
  the read/grep branch, the edit-reject). Folded in: (D0) the in-flight RED tree (TemplateEntry import +
  template_catalog wiring + create_node 4-arity + the `_caps.py` stub) lands atomically in step 1 or the
  build is dead at App init; (D2) the `template:`-edit reject is an EXPLICIT guard with an actionable
  message, not incidental non-resolution; (D4) `_seed_starter_node` keeps its SOFT guard (a shared
  CONSTANT, not a shared raising resolver) + a short-id-uniqueness pin; (D5) the store loads in
  `__init__` not `_init`, and the catalogue render sanitizes non-ASCII; (D6) the popup must not regress
  the everyday create flow — Enter/Esc suppression while the description input is focused + re-bind on
  selection change + a fixed-height slot with a scrollable grid; verification gains clean-template-
  compiles + non-ASCII + short-id-uniqueness + the unbroken-create-flow checks.
