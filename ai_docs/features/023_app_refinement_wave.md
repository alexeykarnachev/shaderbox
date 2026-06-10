<!-- Spec shape (pinned): Goal / Out of scope (each deferral with a Trigger) / Design decisions
     (numbered, lock-in only; open questions separate) / Files touched / Manual verification /
     Open questions. Cite SYMBOLS, never raw line numbers or file-length counts (conventions.md
     ## Code rules — both rot on every edit). The landed-reality / bug-fix story lives here + the
     commit message, NOT the roadmap row. -->

# Feature 023 — app.py refinement wave

## Goal

`app.py` has become the codebase's overgrown hub — the single largest module by a wide margin, and
roughly two-fifths of it is the copilot capability backend (the 49 `_copilot_*` methods bound by
`_build_copilot_capabilities`), a whole subsystem's worker-facing implementation that grew up inside
the orchestrator after the copilot stack (020·12→29) landed. This wave lifts that backend into its
own module where the rest of `copilot/` already lives, then folds in two cheap adjacent refinements.

It is **pure-shape except for one isolated, intentional bug fix**, delivered as **three focused,
bisect-clean commits** (the commit boundaries are deliberate — see Design decision 11):

- **C2 — extract `copilot/backend.py`.** All 49 `_copilot_*` methods (un-prefixed) + the
  `_CopilotEditTarget` dataclass + the copilot-only free helpers move into one `CopilotBackend`
  class. The centerpiece; zero behavior change.
- **C3 — collapse the popup booleans into a `PopupState` enum.** The four mutually-exclusive modal
  flags become one enum field; the "at most one open" mutex becomes structural. Zero behavior change.
- **C4 — fix the Esc handler.** `_handle_escape` consumes the Esc keypress but leaves the emoji and
  shader-lib pickers open. Under C3 this is a one-assignment fix. **The only behavior change in the
  wave.**

(There is no "C1": an earlier audit floated a separate `uniform_helpers.py`; it was dropped — see
Out of scope. The commit IDs keep the audit's C2/C3/C4 labels to avoid renumber churn against the
review trail.)

## Out of scope

- **A separate `uniform_helpers.py`.** The earlier audit listed six uniform helpers
  (`_format_uniforms`, `_coerce_uniform_value`, `_coerce_array`, `_uniform_type_label`, `_is_number`,
  `_set_uniform_shape_hint`) for extraction into a shared module on the claim they were "shared with
  the uniform UI". A grep disproved it: all six (and `_ENGINE_DRIVEN_UNIFORMS`) are **copilot-only** —
  zero callers outside the copilot span in `app.py`, zero callers elsewhere in the package. They have
  no independent reuse case, so they simply travel with `backend.py` in C2. **Trigger:** the first time
  one of them genuinely needs sharing with the uniform-widget layer (`widgets/uniform.py`) — then hoist
  that one to `util.py` (non-UI) or `ui_primitives.py` (UI), not a copilot-named module.
- **Splitting Telegram into its own collaborator/sub-module.** The ~11 telegram methods
  (`set_telegram_token` / `telegram_connect` / pack CRUD) fold into `CopilotBackend` alongside the core
  methods. The call graph proves them detachable (they touch only `exporter_registry` + the bridge,
  nothing on the node/edit/render spine), but detaching now buys zero cohesion — they are copilot
  capabilities, not a reusable export client. A one-line seam comment marks the fold point.
  **Trigger:** when Telegram becomes a standalone export client reused outside the copilot, OR when
  `backend.py` grows a second clearly-separable cluster and editing it feels unwieldy.
- **Further `app.py` / `ui.py` splits (node-CRUD, path-properties, the picker forwards).** The audit
  rated these net-negative (node-CRUD is a multi-domain lifecycle orchestrator; the `shader_lib_*`
  forwards are pure shims; the path properties are coupled to mutable App state). Out of this wave;
  the existing `todo.md [DEFERRAL] split ui.py / app.py further` continues to cover them. **Trigger:**
  unchanged from that deferral (editing `app.py` feels painful, OR a 4th tab module needs cross-cutting
  App ops).

## Design decisions

*(numbered, lock-in only; open questions separate below)*

### C2 — `CopilotBackend` extraction

1. **One file `copilot/backend.py`, one class `CopilotBackend`, the `ShaderLibFileManager` idiom.**
   Explicit dependencies + injected callbacks; the class **never imports `App`** (the no-`TYPE_CHECKING`
   rule forbids the cycle). It is stateless on project identity — every project-dependent read is a
   **getter** re-executed per call, never cached in `__init__`. Mirrors `shader_lib/file_ops.py`
   exactly so a reader sees the same shape.

2. **The constructor splits dependencies into five categories, each by its volatility.** Keyword-only:
   - **Direct ref** (stable, resolved lazily): `get_bridge: Callable[[], CopilotBridge]` — the bridge
     lives on the `CopilotSession`, constructed *after* the backend (it takes the caps the backend's
     methods bind into), so it can't be an eager ref; a getter resolves it at turn-time. (`notifications`
     is NOT a dep — no moved method uses it; `recover_deleted_node`, which does, stayed on App.)
   - **Frozen values** (project-independent, no side effect): `node_templates_dir: Path` (it is
     `RESOURCES_DIR / "node_templates"` — safe to freeze); `starter_template_id: str` (the default
     "UV Mango" template uuid `_copilot_create_node` falls back to — see decision 8a).
   - **Getters** (project-dependent / side-effecting / live — re-read every call): `get_renders_dir`,
     `get_ui_nodes`, `get_ui_node_templates`, `get_exporter_registry`, `get_shader_lib_index`,
     `get_shader_lib_files`, `get_current_node_id`, `get_is_cancelled`.
   - **App-action callbacks** (effects App owns/guards): `set_current_node_id`, `save_ui_node`,
     `sync_editor_from_disk`, `delete_node_unguarded`, `template_description`.
   - **Shared-state accessors** (working-set/batch state stays on App — see decision 3):
     `working_set_reader`, `working_set_add`, `batch_mutated_reader`, `batch_mutated_add`. (No
     `batch_begin` accessor — the batch-clear is bound ONLY into the capability, App-side; no moved
     method clears the set itself.)

   The getter-vs-frozen split is load-bearing: `open_project` reassigns the project dir, so a `Path`
   captured in `__init__` would stale-point. `get_renders_dir` **must** stay a getter; only the shipped,
   project-independent `node_templates_dir` is safe to freeze. No getter's *result* is cached in the
   constructor.

3. **Working-set ownership: App owns the state, the backend accesses it via callbacks.** This is
   forced, not chosen: `_build_copilot_capabilities` (which constructs the backend and captures its
   dependency closures) runs in `App.__init__` *before* `_copilot_working_set` / `_copilot_batch_mutated`
   are initialized today. Backend-owned state would demand two-phase construction. App-owned matches
   "app.py = state holder" and the `ShaderLibFileManager` posture. The backend reads/writes the live
   `list`/`set` through the injected accessors; dedup lives in **exactly one place** (App-side — the
   working-set is added to via a single accessor that owns the membership check; the backend never
   pre-checks, so there is no double-guard).

4. **The state initializers relocate above the backend construction.** Because of decision 3's ordering
   reality, `App.__init__` must initialize `_copilot_working_set` and `_copilot_batch_mutated`
   *before* the `CopilotSession(caps=self._build_copilot_capabilities())` block, so the captured
   closures see real attributes. This is a pure move of two assignments earlier in `__init__` — no
   logic change. It is the wave's one **invisible-in-the-copilot-diff** edit (the hazard lives in
   `__init__` ordering, not in `backend.py`), so it is called out here and in Manual verification.

5. **`_copilot_busy_blocked` stays on `App`.** It reads the `copilot_turn_active` lifecycle latch and
   gates five non-copilot lifecycle methods (`select_node`, the editor flush/save path, `open_project`,
   `delete_node`, `create_node_from_selected_template`). It is a lifecycle guard, not a copilot verb;
   moving it would invert control (lifecycle code calling into the backend).

6. **`_build_copilot_capabilities` stays on `App` as the seam factory.** It changes only its bind
   target — from `self._copilot_*` to `self.copilot_backend.*`. `CopilotCapabilities` remains the
   frozen dataclass of bound callables that `copilot/` imports as its only App-facing leaf; the backend
   is **not** a capabilities field, it is internal to App's copilot machinery. The `batch_begin`
   capability binds to App's `_copilot_batch_mutated.clear` (the canonical owner of the clear), the same
   accessor the backend receives — one clear, one owner.

7. **Telegram folds into `CopilotBackend`** (see Out of scope for the rationale + trigger). A one-line
   seam comment marks the detachable cluster head.

8. **Free-function dispositions** (verified by grep on the callers):
   - **Move to `backend.py`** (copilot-only): `_number_lines`, `_to_error_infos`, `_format_uniforms`,
     `_whitespace_near_match`, `_ws_normalize` (called only by `_whitespace_near_match`), `_splice`,
     `_coerce_uniform_value`, `_coerce_array`, `_uniform_type_label`, `_is_number`,
     `_set_uniform_shape_hint`, plus the `_ENGINE_DRIVEN_UNIFORMS` frozenset and the `_CopilotEditTarget`
     dataclass.
   - **Stay on `app.py`** (non-copilot callers): `_order_templates` (called by `_init`),
     `_conversation_stamp` (called by `copilot_clear_chat`).

   **8a. `_STARTER_TEMPLATE_ID` stays a module constant on `app.py` and is passed into the backend as
   the frozen `starter_template_id` ctor value.** It is used by *both* `_copilot_create_node` (moves to
   the backend) *and* `_seed_starter_node` (stays on App), and is imported by
   `tests/test_cross_project_tools.py`. Moving it to `backend.py` would force `_seed_starter_node` + the
   test to back-import from the backend (a cycle / wrong-direction dep). Keeping it on `app.py` and
   freezing it into the backend ctor (it's a shipped, project-independent uuid) keeps the dependency
   one-way, leaves the test's import unchanged, and avoids inventing a new leaf home. The backend reads
   `self._starter_template_id`, never the module constant. *(Feature 031 later moved the constant —
   with `TEMPLATE_ORDER`/`NODE_TEMPLATES_DIR` — to `constants.py`, the leaf home that didn't exist as
   an option here.)*

9. **Cycle-safety is a locked invariant: one-way `app → backend`.** `backend.py` imports only leaf
   modules verified (BFS over the real import graph) to carry zero transitive `app` import:
   `copilot.bridge`, `copilot.capabilities` (value types), `copilot.config` / `errors` / `glsl_lex` /
   `text_render`, `core` (`Node`), `ui_models` (`UINode`), `notifications`, `shader_lib.index`
   (`ShaderLibIndex`), `shader_lib.file_ops` (`ShaderLibFileManager`), `exporters.registry`
   (`ExporterRegistry`), `paths`, `render_preset`, `shader_errors`. No `if TYPE_CHECKING`, no
   `from __future__`, no forward-ref strings.

### C3 — `PopupState` enum

10. **Four modal booleans collapse into one `PopupState` field; the palette stays a separate bool.**
    `is_node_creator_open` / `is_settings_open` / `is_emoji_picker_open` / `is_shader_lib_picker_open`
    become `self.popup_state: PopupState` ∈ {`CLOSED`, `NODE_CREATOR`, `SETTINGS`, `EMOJI_PICKER`,
    `SHADER_LIB_PICKER`}. `is_palette_open` is **not** folded in — the command palette is a non-modal
    floating search box, deliberately excluded from the popup mutex, and coexists with any modal. The
    mutex becomes **structural**: each `open_*()` helper sets `popup_state = PopupState.X` (one field
    cannot hold two states — the old "set mine True, clear three siblings" dance is gone);
    `any_popup_open()` is `self.popup_state != PopupState.CLOSED`. Each popup's self-close sets
    `popup_state = CLOSED` (the `imgui.close_current_popup()` call is unchanged). C3 carries **zero**
    behavior change — Esc still leaks the two pickers until C4.

### C4 — Esc bug fix (the only behavior change)

11. **`_handle_escape` closes all modals with one assignment.** Today the handler runs whenever
    `escape_has_job()` is true (which includes the emoji + lib pickers, since both are in
    `any_popup_open()`), **consumes** the keypress, but clears only the node-creator / settings /
    palette flags — leaving the emoji and shader-lib pickers open (they close only via imgui's own modal
    auto-close). That contradicts the handler's stated contract ("Esc returns the app to its default
    state: close any popup"). C3 already migrates `_handle_escape` to `popup_state` **without** changing
    behavior (a literal translation of today's three clears — the two pickers still leak). C4 is the
    minimal delta on top: replace those clears with the single `popup_state = PopupState.CLOSED` (closes
    every modal; the leak is impossible). The `was_settings_open` capture (which decides whether to
    `apply_editor_settings()` on close) becomes `popup_state == PopupState.SETTINGS`, read **before** the
    reset. This is a real, intended behavior change — Esc now closes the emoji and lib pickers — and is
    the sole behavioral delta in the wave, isolated to C4.

12. **Three separate commits, green-gated.** C2 → C3 → C4 land as three commits, each
    `make check` + `make smoke`-green before the next, so a bisect points at exactly one kind of change.
    Bundling them into one squashed commit is explicitly rejected: it would erase the bisect that
    isolates the one behavior-changing line (C4) from the two wide pure-shape moves (C2/C3).

### Convention update (lands in C3)

`conventions.md ## Design decisions` — the **`popups/*.py` … open/closed state lives on `App` as
booleans** bullet is reworded to: state lives as a single `PopupState` enum field; each `open_*()`
sets the field; the single field **is** the mutex (the "at most one open" invariant is enforced by
construction, not by a clear-the-siblings convention); the non-modal command palette remains a
separate bool. (Same revisit trigger: a popup grows internal state that doesn't belong on `App`.)

## Files touched

### C2 — backend extraction
- **NEW `shaderbox/copilot/backend.py`** — the `CopilotBackend` class (all 49 methods, un-prefixed),
  the `_CopilotEditTarget` dataclass, the copilot-only free helpers + `_ENGINE_DRIVEN_UNIFORMS`
  (decision 8), the telegram fold-in with its seam comment.
- **`shaderbox/app.py`** — the 49 `_copilot_*` method definitions + the moved free functions removed;
  `_build_copilot_capabilities` rewired to construct `self.copilot_backend` and bind its methods;
  the two state initializers relocated above the `CopilotSession` construction (decision 4);
  `_copilot_busy_blocked`, `_order_templates`, `_conversation_stamp` retained.
- **`shaderbox/copilot/capabilities.py`** — verify **unchanged** (the seam's shape is stable).
- **`tests/test_uniform_arrays.py`** — repoint `_coerce_uniform_value` import to
  `shaderbox.copilot.backend`.
- **`tests/test_cross_project_tools.py`** — repoint `_coerce_uniform_value` + `_ENGINE_DRIVEN_UNIFORMS`
  imports to `shaderbox.copilot.backend` (`_STARTER_TEMPLATE_ID` stays from `shaderbox.app`); repoint the
  `_id_stub`-driven `_copilot_short_ids`/`_copilot_resolve_node_id` unit calls onto `CopilotBackend`
  (the stub now feeds `_get_ui_nodes`).
- **`tests/test_line_editing.py`** — repoint `_number_lines` import to `shaderbox.copilot.backend`.
- **`tests/test_template_library.py`** + **`tests/test_working_set.py`** — repoint the `app._copilot_*`
  call sites to `app.copilot_backend.*` (a consequence of the extraction: these exercise the backend
  through a live `App`).

### C3 — PopupState enum
- **`shaderbox/app.py`** — the `PopupState` enum + `popup_state` field; `any_popup_open()`,
  `escape_has_job()`, and the four `open_*()` methods rewired; the four modal bools removed
  (`is_palette_open` kept).
- **`shaderbox/ui.py`** — the one direct modal-flag read (`is_node_creator_open` in the template-grid
  render gate) repointed to `popup_state == PopupState.NODE_CREATOR`. (The other render-gates already
  go through `any_popup_open()`, which is unchanged behaviorally — so `tabs/code.py`,
  `widgets/node_grid.py`, `widgets/cheatsheet.py`, `widgets/copilot_chat.py` needed NO edit.)
- **`shaderbox/hotkeys.py`** — `_handle_escape` migrates its modal-flag reads/writes to `popup_state`
  in C3 (it reads `is_settings_open` + writes the modal flags, so it MUST move with the enum or C3's
  green-gate breaks). C3 keeps the SAME behavior — it still clears node-creator/settings/palette and
  still leaves the two pickers (a literal `popup_state` translation of today's three assignments). The
  Esc *behavior* change is C4 below.
- **`shaderbox/popups/emoji_picker.py`**, **`shaderbox/popups/settings.py`**,
  **`shaderbox/popups/node_creator.py`**, **`shaderbox/popups/lib_picker/__init__.py`** — each popup's
  self-close set to `popup_state = PopupState.CLOSED`.
- **`scripts/smoke.py`** — the popup-mutex invariant adapts from "≤1 of four bools true" to a
  `popup_state`-is-a-single-value check (the mutex now holds structurally; the assertion documents it).
- **`ai_docs/conventions.md`** — the popups bullet reword (see Convention update).

### C4 — Esc fix
- **`shaderbox/hotkeys.py`** — `_handle_escape` only: replace C3's three explicit modal-flag clears with
  the single `popup_state = PopupState.CLOSED` (now also closing the emoji + lib pickers — the behavior
  change). `was_settings_open` is the pre-reset `popup_state == PopupState.SETTINGS` capture. This is a
  ~2-line delta on top of C3's translation, isolating the one behavioral change to its own commit.

## Manual verification

### C2 behavior-preservation checklist (impl + review MUST confirm each)
- **Init-ordering relocation done:** the two state initializers sit above the backend construction in
  `App.__init__`; nothing touches the list/set between the old and new positions.
- **`get_renders_dir` stays a getter** (no `self._renders_dir = get_renders_dir()` in the ctor); a
  project switch retargets renders.
- **`node_templates_dir` frozen Path is fine** (project-independent, no side effect).
- **No getter result cached in the ctor** — `ui_nodes` / `exporter_registry` / `shader_lib_files` /
  `shader_lib_index` re-read every call.
- **All GL ops stay inside `bridge.run_on_main` `_on_main` closures** — visual-diff every closure; none
  leaks to the worker thread; `defer=True` + the custom render/publish timeouts move verbatim.
- **`sync_editor_from_disk` keys on the edit-target id** passed in, not a worker-side
  `get_current_node_id()` read.
- **`read_shaders` resolves `[] → current` INSIDE `_on_main`** (main-thread live value).
- **`batch_begin` capability + accessor both resolve to App's `_copilot_batch_mutated.clear`**; the
  backend never clears the set itself.
- **Working-set dedup lives in exactly one place** (the App-side `working_set_add` accessor).
- **`get_is_cancelled` read fresh per poll iteration** in the publish/telegram await loops.

### Per-commit gate
- `make check` (ruff + pyright, 0 errors) green after **each** of C2, C3, C4.
- `make smoke` green after **each** of C2, C3, C4 (it touches popup/lifecycle code — mandatory here).

### Maintainer copilot live-pass (after C2 + C3)
Exercise the full backend through the real app: edit a shader (`edit_shader` / `replace_lines`),
`set_uniform`, `create_node`, `delete_node` (+ confirm the working set prunes the id), `render_image` /
`render_video` (confirm `renders_dir` resolves to the live project), `publish_telegram` /
`publish_youtube` (if creds set), and telegram pack list/create/select/delete. All must work exactly as
before C2.

### C4 Esc behavior check (the intended change)
1. Open the **emoji picker** (don't select), press **Esc** → it closes. *(New — previously Esc was
   consumed but the picker stayed open.)*
2. Open the **shader-lib picker** (don't select), press **Esc** → it closes. *(New.)*
3. Open **settings**, press **Esc** → closes **and** editor settings apply (regression check —
   unchanged).
4. Open the **node creator**, press **Esc** → closes (regression check — unchanged).
5. **Esc with nothing open** → swallowed at the glfw layer, no in-frame handler runs (unchanged).

## Open questions for the user

None blocking — the two prior open questions are resolved inline and noted here for visibility:

1. **Working-set dedup placement** → App-side accessor (decision 3). The `working_set_add` callback owns
   the membership check; the backend does not pre-check. Simpler, matches "App owns state", no double
   guard.
2. **Commit granularity** → three separate green-gated commits (decision 12), not one squashed commit —
   to preserve the bisect that isolates C4's behavior change.
