# 002 — UI widgets + popups extraction (the first half of the ui.py split)

Status: **draft, NOT plan-locked — user sign-off pending**
Shape: mid-to-high-blast-radius feature flow (`ai_docs/dev_flow.md ## Feature flow`).

---

## Goal

Continue the `ui.py` split started by feature 001 (`tabs/share.py`). Extract the **widget
functions** and **the settings popup** from `ui.py` into new `shaderbox/widgets/` and
`shaderbox/popups/` subpackages. After this PR, `ui.py` should drop from 1508 → ~900-1000 lines
and the widget code is reusable across tabs.

**Pure refactor — no behavior change, no feature change.** Every imgui pixel should look and
behave identically before and after.

Resolves: partially advances `todo.md [DEFERRAL] split ui.py` (the easier half — widgets are
mostly leaf functions with model-in-model-out signatures; the tab modules they're used by are
left for feature 003).

## Out of scope

- **Tab modules (`tabs/node.py`, `tabs/render.py`).** Heavier coupling to `App` operations
  (`save_ui_node`, `nodes_dir`, `edit_current_node_fs_file`, etc.). Will become feature 003,
  where the right shape of "thread App state into tab modules" is the central design question.
  *Trigger:* after this feature lands; pattern is set by `tabs/share.py` already (feature 001).
- **`hotkeys.py` extraction.** The hotkey block in `update_and_draw` (~1580-1620) is mixed with
  the render-loop body; can't be cleanly extracted without touching loop semantics. *Trigger:*
  feature 004 (or after both tabs land, depending on what shape `update_and_draw` is in).
- **`project.py` extraction.** All the `@property`-style path methods + `save` / `open_project` /
  `delete_current_node` / `create_node_from_selected_template`. Touches lifecycle. *Trigger:*
  after `hotkeys.py` extraction.
- **`App` slimming itself.** `App` will still hold all the state references after this PR;
  what changes is that the imgui-drawing functions live elsewhere. Slimming the class to a thin
  orchestrator happens naturally as the remaining methods get extracted.
- **Re-tightening pyright.** Same status as feature 001 — `todo.md [DEFERRAL] re-tighten
  pyright` still applies. Don't add new pyright errors; don't try to fix pre-existing ones.
- **Any new widget functionality or imgui re-design.** No new buttons, no new UX. Pure move.

## Extraction targets (the concrete list)

From `ui.py` to new modules:

| Source method (current line range) | Target module | Notes |
|---|---|---|
| `draw_file_details` 1080-1115 (~35 lines) | `widgets/details.py` | Pure: takes `FileDetails`, returns `FileDetails`. Uses `_notifications`. |
| `draw_resolution_details` 1117-1159 (~42 lines) | `widgets/details.py` | Uses `ui_nodes`, `current_node_id`. |
| `draw_media_details` 1161-1215 (~55 lines) | `widgets/details.py` | Composes the two above. Uses `ui_nodes`. |
| `draw_selected_ui_uniform_settings` 810-888 (~78 lines) | `widgets/uniform.py` | Uses `ui_nodes`, `current_node_id`, `_ui_app_state`. |
| `draw_ui_uniform` 890-1026 (~137 lines) | `widgets/uniform.py` | Uses `ui_nodes`, `media_dir`, `_notifications`, `_ui_app_state`. Largest extraction; many uniform-input-type branches. |
| `draw_media_models` 731-769 (~38 lines) | `widgets/media_ops.py` | Uses `modelbox_info`, `_ui_app_state`, `media_dir`, `trash_dir`, `fetch_modelbox_info`, `_notifications`. |
| `draw_video_filters` 771-808 (~37 lines) | `widgets/media_ops.py` | Uses `current_node_ui_state_or_default`, `_notifications`. |
| `draw_node_preview_grid` 392-442 (~50 lines) | `widgets/node_grid.py` | Uses `ui_nodes`, `current_node_id`, `set_current_node_id`, `_active_popup_label`, `_NODE_CREATOR_POPUP_LABEL`. |
| `draw_node_creator` 444-479 (~35 lines) | `widgets/node_grid.py` | Uses `ui_node_templates`, `_ui_app_state`, `create_node_from_selected_template`. |
| `draw_settings` 481-572 (~91 lines) | `popups/settings.py` | Uses `_ui_app_state` only — cleanest extraction. |

Total: ~600 lines extracted from `ui.py`. Expected post-impl `ui.py` size: ~900 lines.

## Design decisions (locked — these need user sign-off before impl)

1. **`AppContext` dataclass** — bundle the most-common state refs the widgets need (`app_state`,
   `ui_nodes`, `notifications_push`, `media_dir`, `trash_dir`, `current_node_id`) into one
   dataclass owned by `App` and threaded into every widget call. Reasons:
   - The widget functions need 3-7 state refs each. Long argument lists are awkward.
   - A single `AppContext` parameter avoids the "next time I need a new ref I have to thread it
     through every call site" pain.
   - The pattern is the imgui equivalent of `App`'s self-references, but **read-only** — widgets
     can mutate via callbacks (`notifications_push`, `set_current_node_id`) or by writing back
     into `app_state` (a mutable pydantic model).
   - It's NOT a god-object — `App` keeps its non-render-thread state (worker locks, lifecycle
     refs, etc.) on itself; only the imgui-render-relevant refs go into `AppContext`.

   *Why locked:* the alternative (pass individual refs everywhere) creates 7-arg function
   signatures the next maintainer has to update on every change. The alternative (pass `App`
   itself) re-creates the god-class coupling we're trying to undo.

2. **`AppContext` is render-thread-only.** All widgets are imgui-draw functions; they run on
   the render thread. `AppContext` lives on `App` as `self._ctx` and is constructed in
   `App._init` (after `app_state` loads, after `media_dir` resolves). Widget functions take
   `ctx: AppContext` as their first non-self parameter. *Why locked:* matches the thread
   affinity rule from feature 001's `Exporter` ABC.

3. **Module layout:**
   - `shaderbox/widgets/__init__.py`
   - `shaderbox/widgets/details.py` — `draw_file_details`, `draw_resolution_details`,
     `draw_media_details`.
   - `shaderbox/widgets/uniform.py` — `draw_selected_ui_uniform_settings`, `draw_ui_uniform`.
   - `shaderbox/widgets/media_ops.py` — `draw_media_models`, `draw_video_filters`.
   - `shaderbox/widgets/node_grid.py` — `draw_node_preview_grid`, `draw_node_creator`.
   - `shaderbox/popups/__init__.py`
   - `shaderbox/popups/settings.py` — `draw_settings`.
   - `shaderbox/app_context.py` — `AppContext` dataclass.

   *Why locked:* one module per cohesive widget family. `media_ops` mixes ModelBox + video
   filter because both are "media transformation buttons" attached to texture uniforms; could
   split further if needed later.

4. **Each widget function is a free `def`, not a method.** Signature shape (consistent with
   `tabs/share.py`):
   ```python
   def draw_<widget>(ctx: AppContext, <model_in>) -> <model_out>:
       ...
   ```
   For widgets that don't mutate the input (`draw_node_preview_grid`), return type is `None`
   and the function calls callbacks (`ctx.set_current_node_id(new_id)`) for side effects.
   *Why locked:* matches the `tabs/*.py` convention. Free functions are easier to test, reuse,
   and reason about than methods on a god-class.

5. **`AppContext` exposes callbacks, not raw `App` methods.** Where a widget needs to call
   an `App` method (`set_current_node_id`, `notifications_push`, `fetch_modelbox_info`,
   `create_node_from_selected_template`, `edit_current_node_fs_file`, `open_current_node_dir`),
   those are stored as bound-method references in `AppContext`:
   ```python
   @dataclass
   class AppContext:
       app_state: UIAppState
       ui_nodes: dict[str, UINode]
       ui_node_templates: dict[str, UINode]
       media_dir: Path
       trash_dir: Path
       modelbox_info: dict[str, Any]
       notifications_push: Callable[[str, tuple[float, float, float]], None]
       set_current_node_id: Callable[[str], None]
       fetch_modelbox_info: Callable[[], None]
       create_node_from_selected_template: Callable[[], None]
       edit_current_node_fs_file: Callable[[], None]
       open_current_node_dir: Callable[[], None]
       save_ui_node: Callable[[UINode, Path | None, str | None], Path]
   ```
   *Why locked:* exposing `App` directly recreates the coupling. Exposing each method as a
   `Callable` keeps the widget's contract explicit.

6. **`current_node_id` is a derived value, not a context field.** Widgets that need it read
   `ctx.app_state.current_node_id` rather than having a separate field. *Why locked:* one
   source of truth; `App.current_node_id` is already just a `@property` accessor.

7. **No behavior changes during extraction.** Each method's body is moved verbatim (modulo
   `self.X` → `ctx.X` rewrites). If a method has a bug, it stays — fix in a follow-up. The
   only legitimate changes are: (a) the `self.` rewrites, (b) adding explicit type
   annotations where they were inferred via `self` (per `conventions.md`), (c) renaming the
   first parameter from `self` to nothing (free function).

8. **`App` continues to expose its `@property` paths** (`media_dir`, `trash_dir`,
   `nodes_dir`, etc.) for other (non-widget) callers; the widget extraction doesn't change
   these. `AppContext` gets snapshot copies of the resolved `Path` values at `_init` time.
   *Why locked:* paths don't change during a project session (they're project-bound). After
   `open_project` → `_init` re-runs, `AppContext` is rebuilt fresh.

## Files touched

**Created:**
- `shaderbox/app_context.py` — ~30 lines (dataclass + factory function building it from `App`).
- `shaderbox/widgets/__init__.py` — empty.
- `shaderbox/widgets/details.py` — ~140 lines (3 extracted methods + type annotations).
- `shaderbox/widgets/uniform.py` — ~220 lines (2 extracted methods).
- `shaderbox/widgets/media_ops.py` — ~80 lines (2 extracted methods).
- `shaderbox/widgets/node_grid.py` — ~90 lines (2 extracted methods).
- `shaderbox/popups/__init__.py` — empty.
- `shaderbox/popups/settings.py` — ~95 lines (1 extracted method).

**Modified:**
- `shaderbox/ui.py` — 10 methods removed (~600 lines); `App.__init__` constructs `self._ctx`;
  all internal call sites become `widgets.X.draw_Y(self._ctx, ...)`. The `draw_popup_if_opened`
  call site for the settings popup updates to call `popups.settings.draw(self._ctx)`.
  Expected: 1508 → ~900 lines.
- `shaderbox/tabs/share.py` — already uses an ad-hoc `Callable` for notifications_push from
  feature 001; the spec's Decision 5 generalizes this. Update `share_tab.draw()` to take
  `ctx: AppContext` instead of separate `current_node` + `notifications_push` args, OR leave
  as-is (feature 001 shipped with the per-arg shape; consistency would be nice but isn't
  load-bearing for THIS feature). Decision in spec: **leave `share.py` as-is**; harmonize when
  feature 003 lands the other tabs and the pattern stabilizes.
- `ai_docs/conventions.md` — `## Design decisions`: add a "**Widget functions live in
  `widgets/*.py`, take `AppContext`**" bullet. Revisit when 5+ widget modules exist and the
  shape stops fitting.
- `ai_docs/dev_flow.md ## Recipes` module map: add `widgets/` and `popups/` subpackage bullets;
  note `app_context.py`.
- `ai_docs/todo.md` — update `[DEFERRAL] split ui.py` to note widget+popup extraction landed;
  add follow-up trigger for `tabs/node.py` + `tabs/render.py` extraction.
- `ai_docs/worklog.md` — new entry on completion.

**Deleted:** none.

## Manual verification

Pure refactor — no behavior change, so verification is "the app still works the same."

1. **`make check`:** ruff fix + format clean; pyright still non-blocking with **0 errors** post-
   refactor (the goal is to not regress from feature 001's clean state). Any new error is a
   FAIL.
2. **Launch the app:** `uv run python ./shaderbox/ui.py` against `projects/dev/`. Verify no
   import errors, no startup errors, the window opens normally.
3. **Visual regression sweep:** click through every tab (Node, Render, Share) — every widget
   should look pixel-identical to before. Imgui layout is fragile to refactor; this is the
   real check.
4. **Settings popup:** Alt+S opens it; every field (FPS, text editor cmd, modelbox URL) reads
   and writes correctly. Apply changes, save (Ctrl+S), reopen — values persist.
5. **Node creator:** Ctrl+N opens the popup; select a template, create a node; verify the new
   node appears in the grid and is selected. Cancel button works.
6. **Node preview grid:** the existing node-preview grid renders all nodes with correct
   borders (green for selected, red for shader error); clicking a preview selects that node;
   keyboard arrow keys navigate.
7. **Uniform widgets:** select a node with multiple uniform input types (color, drag, texture,
   text, array). Verify every type renders the correct UI, and that editing each one updates
   the shader's render in real time.
8. **Media ops (ModelBox):** if a ModelBox server is running (`modelbox_url` in
   `app_state.json`), select a texture uniform → "Generate" button runs the model and the
   texture updates. (Skip step if no ModelBox server available; degrade gracefully —
   `fetch_modelbox_info` failure should still let the rest of the app work.)
9. **Video filters:** select a video texture uniform → temporal smoothing button runs ffmpeg
   and produces a new video uniform value.
10. **File / resolution / media details widgets:** in the Render tab, the resolution drag-ints
    and aspect-ratio buttons work; the file path picker opens the correct dialog; the media
    details panel shows resolution / FPS / duration correctly.
11. **Project switch:** Ctrl+O, pick a different project. Verify `_ctx` is rebuilt
    (`media_dir` / `trash_dir` / `ui_nodes` snapshot reflects the new project).

A real UX gap found at any step is a FAIL, not pass-with-caveat.

## Open questions for the user

1. **`AppContext` design (Decision 1 + 5):** the proposed shape bundles ~12 fields (state +
   callbacks). Is this the right abstraction, or would you prefer one of:
   - (a) Pass individual refs per function (no `AppContext` — every widget takes 3-7 args).
   - (b) Pass `App` itself (re-couples to the god-class but minimal refactor).
   - (c) The proposed `AppContext` dataclass (recommended — clean contract, explicit callbacks).
2. **`tabs/share.py` retrofit (Decision in Files touched):** should this feature also retrofit
   `tabs/share.py` to take `AppContext` (currently takes `state, registry, current_node,
   notifications_push` separately)? Recommendation: leave as-is, harmonize in feature 003.
3. **Scope of feature 002 vs 003:** the proposed split is "widgets + popups in 002, tabs in
   003". Alternative: do everything in one PR. The "everything" version is ~1500 lines diff
   (vs ~600 for widgets-only) and includes the harder `tabs/node.py` extraction (the uniform
   editor's tight coupling to the node tab). Recommendation: 002 = widgets, 003 = tabs.
4. **Should widget functions return updated models, or mutate in place?** Decision 4 above says
   "return updated model" (consistent with current style — most `draw_*_details` already do
   `details = details.model_copy(); ...; return details`). But for widgets that don't fit this
   shape (e.g. `draw_node_preview_grid` doesn't return anything), there's mild inconsistency.
   Recommendation: keep the existing per-widget shape; don't normalize.

## Review history

*(Populated by review agents during the feature flow.)*
