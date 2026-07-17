# 051 — Node examples (templates → examples rework + instant New Node + fire showcase)

Supersedes the parked `051_shipped_examples_project.md` draft (deleted in this wave). The seed-into-
writable-area design and the example↔lib-coupling blocker are both RESOLVED by re-audit — see
"Rejected designs" at the bottom.

## Goal

Make ShaderBox's first-stranger experience carry itself (HN-post prep): the app ships browsable,
polished **example nodes** the user can open, watch, copy, and dissect — while node creation becomes
one keystroke.

Three moves, one entity, no new mechanism:

1. **The "template" notion is renamed to "example", totally.** Same shipped read-only node-dirs under
   `shaderbox/resources/`, same instantiate-by-copy, same description sidecar, same copilot
   addressing — one vocabulary everywhere (UI, code, copilot prompt, tests). No dual naming survives.
2. **New Node is instant.** `Ctrl+N` / File → New node / the node-grid "+" cell clone the starter
   example (UV Mango) into the project immediately — the picker popup leaves the creation path.
3. **The node-creator popup becomes the Examples browser.** Same modal machinery, retitled and
   re-purposed: open it deliberately (menu bar / `Alt+E` / palette / first run), browse the
   animated previews + descriptions, "Open a copy" to instantiate one into the current project. The
   **fire showcase** (the `/shader-lab` timed-reveal node from `projects/_lab/fire/` — tracked in
   git despite the `_lab` gitignore rule, via a prior commit) ships as the first real showcase
   entry.

The lib-coupling problem that parked the old draft dissolves: examples are read from the installed
bundle at runtime, so an example and the shipped lib ALWAYS come from the same commit (lockstep by
construction — no version matrix, no pinning). The residual risk (maintainer edits the lib, forgets
the examples) is closed by two new tests, not by machinery.

## Out of scope

- **List-left / detail-right browser layout** (the game-engine-browser look). This iteration keeps
  the existing grid + description-slot layout — it already is a browser. Trigger: the example count
  outgrows the grid (≈ >9 entries), or the maintainer judges the grid too weak for the HN demo video.
- **More showcase examples.** Only fire ships now; each future showcase is a `/shader-lab` session
  graduating a node into `resources/node_examples/` + one `EXAMPLE_ORDER` entry (no code change).
  Trigger: maintainer schedules the next `/shader-lab` session.
- **Sections / tags / difficulty metadata in the browser.** Flat grid, name + description only.
  Trigger: the flat grid stops scanning well (same threshold as the layout rework).
- **User-configurable starter.** `STARTER_EXAMPLE_ID` stays a constant. Trigger: a user asks to
  change what Ctrl+N creates.
- **Per-project local shader lib** (the old draft's leaning). Rejected for this problem — lockstep +
  tests cover examples; a local lib would add a "which lib?" dimension to the picker, the editor
  tree, the watcher, the copilot reads. Trigger to revisit: **project portability** — a user shares a
  project dir and it breaks on the recipient's machine because their live lib diverged. That
  use-case earns vendoring on its own merits; examples never did.
- **Migration of any kind** (per `conventions.md`): the renamed `UIAppState` field and the renamed
  description-sidecar filename are hand-fixed in `projects/dev/` + the dev machine's
  `app_data_dir()`; no compat reader, no rename-shim. N/A beyond that.

## Design decisions (lock-in)

1. **Total rename, one wave: template → example.** `resources/node_templates/` →
   `resources/node_examples/` (git mv, same three node-dirs); `NODE_TEMPLATES_DIR` →
   `NODE_EXAMPLES_DIR`, `TEMPLATE_ORDER` → `EXAMPLE_ORDER`, `STARTER_TEMPLATE_ID` →
   `STARTER_EXAMPLE_ID` (`constants.py`); `ProjectSession.ui_node_templates` → `ui_node_examples`,
   `node_templates_dir` → `node_examples_dir`, `template_description` → `example_description`,
   `_order_templates` → `_order_examples` (+ the `App` forwarders);
   `UIAppState.selected_node_template_id` → `selected_example_id`.
   Copilot: `address.py` `TEMPLATE_PREFIX "template:"` → `EXAMPLE_PREFIX "example:"` (+ helper
   renames), `TemplateEntry` → `ExampleEntry`, `template_catalog` → `example_catalog`,
   `create_node(template=)` → `create_node(example=)`, prompt text speaks "example" (`prompt.py`,
   `prompt_context.py`). Rationale: two names for one notion is permanent confusion; the no-compat
   rule makes the rename free.
2. **`Ctrl+N` clones the starter instantly.** `CommandId.NEW_NODE`'s callback becomes
   `App.create_node_from_example(STARTER_EXAMPLE_ID)` (the generalized rename of
   `create_node_from_selected_template` — takes the example id as a param instead of reading
   `selected_node_template_id`). The palette "New node" two-step prompt machinery
   (`_palette_new_node_initial`/`_palette_new_node_subsequent`/`_palette_template_ids` AND the
   `if spec.id == CommandId.NEW_NODE:` branch in `_register_palette_commands`) is DELETED — the
   palette entry registers single-step like every other command. The node-grid's **"New node"
   button** (`widgets/node_grid.py` — there is no "+" cell) and the File-menu item route to the same
   callback. The existing copilot-busy guard on node creation is inherited unchanged.
3. **The browser is the same modal, re-purposed.** `popups/node_creator.py` → `popups/examples.py`;
   `PopupState.NODE_CREATOR` → `PopupState.EXAMPLES`; `App.open_node_creator` → `open_examples`.
   Layout unchanged this iteration (grid of `draw_node_preview_button` cells + the fixed description
   slot + action row); title "Examples"; primary action "Open a copy" (+ Enter), Cancel closes.
   Selection persists in `UIAppState.selected_example_id`. The `ui.py` render gate that animates the
   previews while the popup is open follows the enum rename.
   **Descriptions become read-only shipped facts** (maintainer-authored in each example's
   `node.json`): the description-editing subsystem is DELETED wholesale —
   `templates_descriptions.py` (`TemplateDescriptionsStore` + the `template_descriptions.json`
   sidecar), `App.template_desc_input` (`InlineInput`), the in-modal description editor +
   "Edit description" button, `App.set_template_description`, and the `conventions.md` "two-tier
   editable metadata on a shipped resource" bullet (this store was its only instance).
   `example_description(id)` reads the shipped `node.json` only. The copilot catalogue's description
   source follows. Also deleted for the same read-only reason: the `App.template_descriptions`
   forwarder + `ProjectSession.template_descriptions` field, the stale override-store comment on
   `UINodeState.description` (`ui_models.py`), and the **"Save as template"** context-menu action in
   `tabs/node.py` — a runtime WRITE into the shipped resources dir, structurally incompatible with
   read-only examples (example authoring is the repo-side `/shader-lab` promotion path).
4. **Three openers for the browser:** a new `CommandId.EXAMPLES` ("Examples", `Alt+E`,
   `CommandScope.TOOLS`, in palette + cheatsheet — mirrors Settings on `Alt+S`) wired to
   `open_examples`; a top-level **"Examples" menu-bar item** in `ui.py` (a direct-click bar item, not
   a dropdown — it's an action, not a menu); and the first-run auto-open (decision 6).
5. **Fire ships as the first showcase.** Copy
   `projects/_lab/fire/nodes/0b0d16bb-f014-4a85-b155-6be74c33eded/{node.json, shader.frag.glsl}`
   into `resources/node_examples/0b0d16bb-…/`; append its uuid to `EXAMPLE_ORDER` (last). Content
   polish at promotion: `ui_name` → `"Fire"`; rewrite the description for a stranger (the current
   one narrates lab-session mechanics — "v11", step timings).
   The other 10 lab nodes + `NOTES.md` never enter the resource. **Prerequisite promoted with it:
   `SB_fbm` exists ONLY in the desktop live lib root, not `resources/shader_lib/`** (verified — the
   one-way-seed divergence `conventions.md` warns about); copy **only the `SB_fbm` function body**
   into `resources/shader_lib/noise/` (its transitive deps `SB_value_noise`/`SB_hash21` ALREADY ship
   there — do NOT copy the live `noise.glsl` wholesale, that would duplicate `SB_value_noise`), and
   update the conventions NOTE that names `SB_fbm` as desktop-only. The fire shader's other refs
   (`SB_center_uv`/`SB_hash21`/`SB_sd_char`) all resolve from resources (verified by grep).
   `build.sh` needs nothing (the whole `shaderbox/` package ships; `node.json`/`*.glsl` match no
   FORBIDDEN pattern). The `ProjectSession` ctor params rename with everything else
   (`node_examples_dir=`/`starter_example_id=`/`example_order=`) — its TWO callers, `App.__init__`
   and `scripts/dogfood/harness.py` (which also renames its `seed_templates` param), update in the
   same wave; the harness is outside `make check`'s blast so it must be swept deliberately.
6. **First run auto-opens the browser.** `App.__init__` ALREADY computes
   `is_first_launch = project_dir is None and not self.project_dir_file_path.exists()` — reuse it:
   thread it into `_init` as a param (beside the existing starter-seed path) and have `_init` end
   with `open_examples()` when true. It must ride the threaded param, NOT be re-derived inside
   `_init` (by then the pointer file is already written — a re-derive always reads False). Since
   `is_first_launch` requires `project_dir is None`, smoke / the pytest `app` fixture / any harness
   (explicit dir) never auto-opens — no new flag, no new gate. A stranger's first launch lands on
   the moving gallery; every later launch (and every project switch, which also runs `_init`) is
   normal.
7. **Lib coupling = lockstep + two tests (the old draft's blocker, closed).**
   - `tests/test_examples_resolve.py::test_examples_resolve_clean` — every
     `resources/node_examples/**/shader.frag.glsl` resolves with ZERO `ResolveError`s against
     `ShaderLibIndex.build(SHADER_LIB_SEED_DIR)` (pure, GL-free). This is the wire that makes a
     lib-breaks-example regression a red test at dev time — and it's red TODAY until `SB_fbm` is
     promoted (decision 5), which proves the falsifier.
   - `tests/test_examples_resolve.py::test_shader_lib_api_lock` — a checked-in snapshot
     (`tests/shader_lib_api_lock.json`: `{name: signature}` from the seed lib) must EQUAL the live
     extraction; changing, removing, **or adding** a shipped `SB_*` signature fails until the
     snapshot is deliberately regenerated (the test's docstring says how). Turns silent API drift
     into a conscious act. Consequence for THIS wave: the snapshot is generated after the `SB_fbm`
     promotion, so it ships already containing the `SB_fbm` row. (`resolve_usage` +
     `ShaderLibIndex.build` are GL-free — both tests run headless; verified against the imports.)
   - `conventions.md` shader-lib bullet gains one line: shipped `SB_*` is **supersede-don't-mutate**
     (add a new name; never change semantics/signature of an existing one).
8. **The renamed persisted surfaces are hand-fixed, not migrated.** `projects/dev/app_state.json`
   (`selected_node_template_id` key) is hand-edited in the same wave; the dev machine's
   `template_descriptions.json` sidecar is hand-deleted (the store is gone). `extra='forbid'` +
   fail-soft covers any other stale file.

## Files touched

- `shaderbox/resources/node_examples/` — git mv from `node_templates/`; + the fire node-dir (new).
- `shaderbox/resources/shader_lib/noise/…` — `SB_fbm` (+ transitive deps) promoted from the live root.
- `shaderbox/constants.py` — renames (D1), `EXAMPLE_ORDER` + fire entry.
- `shaderbox/templates_descriptions.py` — DELETED (D3).
- `shaderbox/ui_models.py` — `UIAppState.selected_example_id`.
- `shaderbox/project_session.py` — renames; starter-seed comment follows.
- `shaderbox/app.py` — renames; `create_node_from_example(example_id)`; NEW_NODE callback swap;
  palette two-step machinery + the NEW_NODE branch in `_register_palette_commands` deleted;
  `open_examples`; new EXAMPLES command wiring; first-run auto-open threading (D6);
  `template_desc_input` + `set_template_description` + the `template_descriptions` forwarder
  deleted (D3).
- `shaderbox/tabs/node.py` — "Save as template" context-menu action DELETED (D3).
- `shaderbox/commands.py` — `CommandId.EXAMPLES` + spec row (`Alt+E`, TOOLS).
- `shaderbox/popups/examples.py` — git mv from `node_creator.py`; retitle, "Open a copy", renames,
  description editor removed. (`popups/__init__.py` is empty — no change; the dispatch lives in
  `ui.py`'s import + call.)
- `shaderbox/ui.py` — popup call + render gate + menu bar (File → New node stays; new top-level
  Examples bar item), `ui_node_examples` loop.
- `shaderbox/widgets/node_grid.py` — the "New node" button → instant create.
- `shaderbox/hotkeys.py` — only if the node-creator arrow/Enter nav names the popup (follow rename).
- `shaderbox/copilot/{address,capabilities,backend,prompt,prompt_context}.py`,
  `shaderbox/copilot/tools/shader.py` — the `example:` rename surface (D1).
- `scripts/smoke.py` — seed/fixture renames + a frame-0 `popup_state == CLOSED` assert (pins "no
  auto-open headless" as a real check, not a claim).
- `scripts/dogfood/harness.py` — ctor kwargs + `seed_templates` param rename (D5); `.claude/skills/
  dogfood/SKILL.md` prose follows (its `REPORT_TEMPLATE.md`/`--template` refs are a different
  concept — untouched).
- `tests/conftest.py`, `tests/_caps.py` — seed/fixture renames.
- `tests/test_template_library.py` → `tests/test_example_library.py` — the 3 description-sidecar/
  override tests DELETED with the store (D3); remainder renamed; count assertions 3 → 4.
- `tests/test_node_dir_sync.py` — `STARTER_TEMPLATE_ID`/`TEMPLATE_ORDER` import renames.
- `tests/test_examples_resolve.py` + `tests/shader_lib_api_lock.json` — new (D7).
- `tests/test_copilot_loop.py`, `tests/test_prompt_blocks.py` — fixture-string renames.
- `tests/test_cross_project_tools.py` — fixture renames incl. the
  `..._does_not_touch_starter_template` test NAME and its `test_template_library` cross-ref comment.
- `projects/dev/app_state.json` — hand-fix the renamed key (D8).
- `ai_docs/conventions.md` — popups-enum bullet member list; two-tier-metadata bullet DELETED (its
  only instance is gone); the `<kind>:` prefix bullet (`template:` → `example:`); shader-lib bullet:
  supersede-don't-mutate + the `SB_fbm` NOTE update.
- `.claude/skills/shader-lab/SKILL.md` — the promotion path: graduate a lab node →
  `resources/node_examples/` + `EXAMPLE_ORDER`.
- `ai_docs/roadmap.md` — 051 row + banner rewrite.
- `ai_docs/dev_flow.md` — module-map follow-through (examples browser entry, templates-descriptions
  entry deleted, resource-path rename).
- `scripts/dogfood/verify_script_engine.py` — the `seed_examples` kwarg rename (a harness caller
  outside `make check`'s blast).
- Deleted: `ai_docs/features/051_shipped_examples_project.md` (superseded by this spec).

## Manual verification

(Each step names its falsifier; consumer-side, per `dev_flow.md` step 7.)

1. **Instant create:** `make run` → `Ctrl+N`. A new UV-Mango node appears selected in the grid, NO
   popup. Same via File → New node, the grid's "New node" button, AND the palette's "New node"
   entry (now single-step — it must create, not dead-end). Falsifier: a picker popup appears, or no
   node is created.
2. **Browser:** menu-bar "Examples" (and `Alt+E`, and palette) opens the modal; EXACTLY 4 entries,
   ALL animating (a static cell = a dead preview); the cheatsheet shows the Examples row with
   `Alt+E`; selecting fire shows its description; "Open a copy" lands an editable fire node in the
   CURRENT project and closes the modal. Falsifier: no instantiation, wrong project, a dead
   preview, or a missing cheatsheet row.
3. **Fresh-install truth:** `make run-bundle` (throwaway data dir) → the Examples browser is OPEN on
   first frame (D6); open a copy of fire → the timed reveal plays with its step captions. Two
   SEPARATE falsifiers: a COMPILE error = a lib promotion gap (`SB_fbm` — though
   `test_examples_resolve_clean` catches this earlier and exactly); BLANK captions with a clean
   compile = a glyph-table problem (`glyph_tables.py` load or C6020 register overflow — the
   conventions uniform-array quirk), NOT the lib.
4. **Second launch is normal:** relaunch the same bundle data dir → no auto-open. Falsifier: browser
   opens every launch (the pointer-file gate isn't read).
5. **Tests as falsifiers:** `uv run pytest` green; then (throwaway) delete `SB_fbm` from
   `resources/shader_lib/` → `test_examples_resolve_clean` goes RED; change one signature in the
   seed lib → `test_shader_lib_api_lock` goes RED. Restore both.
6. **Copilot surface:** existing (renamed) `test_example_library.py` covers the `example:` catalogue
   / read / edit-reject / `create_node(example=)`; a quick live copilot turn ("create a node from
   the fire example") exercises the prompt rename end-to-end. Falsifier: the agent addresses
   `template:` (stale prompt text survived).
7. **`make check` + `make smoke` green.** Smoke gains an explicit frame-0
   `popup_state == PopupState.CLOSED` assert — the wire that proves the auto-open never fires for
   an explicit-dir App (without it, "smoke green" wouldn't verify this at all). Falsifier: an
   import error from a missed rename; the frame-0 assert; the popup-mutex assert.

## Open questions for the user

None — all three resolved at plan-lock (2026-07-17): chord `Alt+E` (mirrors Settings `Alt+S`);
fire ships as `"Fire"`; the description-editing subsystem is deleted (examples are strictly
read-only — decision 3).

## Review history

**Pre-impl round 1 (2026-07-17, two adversarial opus reviewers: correctness&design,
verification&blast-radius — both PARTIAL, all findings triaged into the spec above):**
- Real, folded in: the `tabs/node.py` "Save as template" write-into-resources action (unlisted +
  contradicted read-only examples → deleted, D3); `scripts/dogfood/harness.py` +
  `tests/test_node_dir_sync.py` missing from the rename sweep; the Alt+E/Ctrl+Shift+E spec
  inconsistency (Alt+E everywhere); "+ cell" → the grid's "New node" button; the 3 description-store
  tests deleted not renamed; D6 re-specified to thread the EXISTING `is_first_launch` into `_init`
  (a re-derive after the pointer write always reads False); `SB_fbm` promoted as a single function
  body (deps already ship — no `SB_value_noise` duplicate); the API-lock test fires on ADDs too
  (snapshot generated post-promotion); smoke gains the frame-0 CLOSED assert; manual steps
  tightened (palette single-step, cheatsheet row, exactly-4, the step-3 two-falsifier split);
  assorted Files-touched completeness (forwarders, `popups/__init__.py` dropped, ui_models comment,
  test-name renames).
- Rejected as false positive: "the fire shader may call a bare `SB_hash`" — primary grep shows
  `SB_hash21`, which ships in `resources/shader_lib/noise/value_noise.glsl`.
- Noted, no action: `projects/_lab/*/app_state.json` carry the stale selected-key — gitignored +
  fail-soft under `extra='forbid'`; the 1080×1920 fire preview renders letterboxed in the square
  grid cell and merely costs more pixels in smoke's popup-open frames.

**Post-impl round 1 (2026-07-17, three adversarial opus reviewers: code-correctness PARTIAL,
architecture/conventions/product-intent PASS, spec-fidelity PARTIAL — converged after triage, no
re-spawn needed):**
- Fixed: `scripts/dogfood/verify_script_engine.py` still passed the renamed `seed_templates=` kwarg
  (the exact outside-`make-check` class D5 warned about — both reviewers found it independently);
  "a example" grammar in the model-facing tool description + two comments; a toast on the
  missing-example guard in `create_node_from_example` (was log-only — a silent Ctrl+N dead end on a
  corrupted install); the roadmap row/banner + this Files-touched list completed at close-out.
- Accepted as intended: Enter on a keyboard-focused grid cell both selects and opens the example in
  one press (correct browser UX, flagged note-only).
- Premise correction: `projects/_lab/fire/` is TRACKED (committed pre-051 despite the `_lab`
  gitignore rule), not gitignored as this spec first claimed; un-tracking it is out of 051's scope.
- Intent audit (anchored to the maintainer's verbatim requirements) confirmed: one default state,
  one browsing surface, one create funnel, no duplicated entity/logic, strictly read-only examples.

## Rejected designs (recorded for the record, decided this session)

- **Seed-into-writable-area examples project** (the old 051 draft): the whole manifest/pristine/
  deleted contract existed because a seeded *project* is user-writable. Examples-as-shipped-resources
  are read-only by construction — instantiation is the only write, and the copy is user-owned. The
  entire apparatus (new package, `.seed_manifest.json` at project granularity,
  `reset_examples_to_shipped`) deleted from the plan.
- **Per-project local lib** — see Out of scope (trigger: project portability).
- **Flatten examples to self-contained GLSL** (splice lib functions into the shipped shader,
  shadowing pins them): survives lib evolution but stops demonstrating the auto-resolve library —
  the headline authoring feature. Kept as a documented escape hatch, not the default.
