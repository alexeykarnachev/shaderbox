# 002 — UI widgets + popups extraction (the first half of the ui.py split)

Status: **implemented 2026-05-15 — multiple post-impl reversals; final state described at the bottom**
Shape: mid-to-high-blast-radius feature flow (`ai_docs/dev_flow.md ## Feature flow`).

> **Reading note — read the "Final state" section at the bottom first.** The body below was the
> plan-locked spec; it went through 3 post-impl reversals (AppContext deleted → popup classes
> flattened to free functions → `App` moved from `ui.py` to `app.py` + tab dispatch moved into
> `ui.py` as free functions + `tabs/node.py` + `tabs/render.py` extracted + `tabs/share.py`
> harmonized to take `app: App`). Every reversal is documented in Review history. The body
> still describes the plan-locked interim states (`AppContext`, popup classes, `_active_popup_label`
> translation table) — those are kept for archaeological value, not as current truth. Mechanical
> signature mappings to current state: `ctx.X` → `app.X`; `popup.open()` / `popup.is_open` →
> `app.open_node_creator()` / `app.is_node_creator_open` (and same for settings);
> `ctx.notifications_push(...)` → `app.notifications.push(...)`.

---

## Goal

Continue the `ui.py` split started by feature 001 (`tabs/share.py`). Extract the **widget
functions** and **the popup bodies** from `ui.py` into new `shaderbox/widgets/` and
`shaderbox/popups/` subpackages.

**Primary motivation:** unblock feature 003 (`tabs/node.py` + `tabs/render.py` extraction).
Those tab modules need to call widgets like `draw_media_details` and `draw_ui_uniform` without
crossing the `App` boundary. Without `widgets/`, feature 003 either imports `App` (reintroduces
the god-class coupling) or duplicates the widget code.

Secondary effect: `ui.py` drops from 1508 → ~900 lines.

**Pure refactor — no behavior change, no feature change.** Every imgui pixel should look and
behave identically before and after.

Resolves: partially advances `todo.md [DEFERRAL] split ui.py` (the easier half — leaf widgets
plus the two popups; the tab modules that wire them together are left for feature 003).

## Out of scope

- **Tab modules (`tabs/node.py`, `tabs/render.py`).** Heavier coupling to `App` operations
  (`save_ui_node`, `nodes_dir`, `edit_current_node_fs_file`, etc.). Will become feature 003,
  where the right shape of "thread App state into tab modules" is the central design question.
  *Trigger:* after this feature lands; pattern is set by `tabs/share.py` already (feature 001).
- **`tabs/share.py` retrofit to `AppContext`.** `tabs/share.py` currently takes
  `state, registry, current_node, notifications_push` separately. Harmonizing to `AppContext`
  is deferred to feature 003 — by then we'll have 3 tab modules and a real shape will emerge,
  rather than retrofitting one outlier mid-extraction. *Trigger:* feature 003.
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

From `ui.py` to new modules. The "Shape" column records each widget's chosen contract — see
Decision 4 for why these vary.

| Source method (current line range) | Target module | Shape | Notes |
|---|---|---|---|
| `draw_file_details` 1080-1115 (~35 lines) | `widgets/details.py` | `(ctx, details: FileDetails) -> FileDetails` | Edits a value object; returns updated copy. Notification via `ctx.notifications_push`. |
| `draw_resolution_details` 1117-1159 (~42 lines) | `widgets/details.py` | `(ctx, details: ResolutionDetails, *, is_changeable: bool) -> ResolutionDetails` | Same — value-object editor. |
| `draw_media_details` 1161-1215 (~55 lines) | `widgets/details.py` | `(ctx, details: MediaDetails, *, is_changeable: bool) -> MediaDetails` | Composes the two above. |
| `draw_selected_ui_uniform_settings` 810-888 (~78 lines) | `widgets/uniform.py` | `(ctx, ui_uniform: UIUniform) -> None` | Mutates `ui_node.node.uniform_values[name]` + `ui_node.ui_state` in place. No coherent return value. |
| `draw_ui_uniform` 890-1026 (~137 lines) | `widgets/uniform.py` | `(ctx, ui_uniform: UIUniform) -> None` | Same — mutates storage dict directly; performs `moderngl.Buffer.write` at the natural site. |
| `draw_media_models` 731-769 (~38 lines) | `widgets/media_ops.py` | `(ctx, current: T) -> T` (T = Image \| Video) | Value transformer. Caller does `if new is not current: try_to_release(current); store = new`. |
| `draw_video_filters` 771-808 (~37 lines) | `widgets/media_ops.py` | `(ctx, current: Video) -> Video` | Same. Reads node UI state via `ctx.get_current_node_ui_state_or_default()` (see Decision 5). |
| `draw_node_preview_grid` 392-442 (~50 lines) | `widgets/node_grid.py` | `(ctx, *, width: float, height: float) -> None` | Action-firer. Selection via `ctx.set_current_node_id`; popup open via `ctx.node_creator.open()`. Width/height typed as `float` to match the existing call site (`control_panel_width / 2.6` at `ui.py:1450`). |
| `draw_node_creator` 444-479 (~35 lines) | `popups/node_creator.py` (**class `NodeCreatorPopup`**) | `class.draw(ctx) -> None` | Popup. Owns `_is_open`; opened via `popup.open()`; closes itself. |
| `draw_settings` 481-572 (~91 lines) | `popups/settings.py` (**class `SettingsPopup`**) | `class.draw(ctx) -> None` | Popup. Same shape as `NodeCreatorPopup`. |

Total: ~600 lines extracted from `ui.py`. Expected post-impl `ui.py` size: ~900 lines.

## Design decisions (locked)

1. **Widgets take `app: App` directly. ~~`AppContext` dataclass~~ — REVERSED post-impl 2026-05-15.**
   Original spec proposed a 15-field `AppContext` dataclass; an investigation after implementation
   showed every claimed benefit was either illusory (render-thread enforcement: not enforced,
   just a docstring; test/audit boundary: no tests exist; bounded coupling: solo-project hygiene
   pyright already covers) or equally provided by passing `app` (stable signatures under field
   additions). The asymmetric renaming layer (`_ui_app_state` → `app_state`, `_notifications.push`
   → `notifications_push` callback, lambda wrapper for `current_node_ui_state_or_default`,
   `_node_creator_popup` → `node_creator`) existed *only* to bridge between `App` and `AppContext`,
   not for any external need. The existing precedent `tabs/share.py` explicitly opted out of
   context-bundling and works fine.

   **Final shape:** widgets and popups take `app: App` directly. Circular import handled by
   `if TYPE_CHECKING: from shaderbox.ui import App` (same trick that already worked for popup
   classes). The "god class" concern is addressed by *continuing to extract from `App` over time*,
   not by wrapping it.

   To support direct passing, three `App` attrs were made public: `_ui_app_state` → `app_state`,
   `_notifications` → `notifications`, `_node_creator_popup` / `_settings_popup` → drop underscore.
   `App.any_popup_open()` added as a one-line method.

2. **Widgets are render-thread-only.** Documented in `conventions.md ## Design decisions` — not
   enforced by types; same convention as feature 001's `Exporter` ABC. Widgets take
   `app: "App"` as the first parameter (string-forward-ref under `TYPE_CHECKING`).

3. **Module layout:**
   - `shaderbox/app_context.py` — `AppContext` dataclass.
   - `shaderbox/widgets/__init__.py`
   - `shaderbox/widgets/details.py` — `draw_file_details`, `draw_resolution_details`, `draw_media_details`.
   - `shaderbox/widgets/uniform.py` — `draw_selected_ui_uniform_settings`, `draw_ui_uniform`.
   - `shaderbox/widgets/media_ops.py` — `draw_media_models`, `draw_video_filters`.
   - `shaderbox/widgets/node_grid.py` — `draw_node_preview_grid`.
   - `shaderbox/popups/__init__.py`
   - `shaderbox/popups/node_creator.py` — `NodeCreatorPopup` class.
   - `shaderbox/popups/settings.py` — `SettingsPopup` class.

   One module per cohesive widget family. `media_ops` mixes ModelBox + video filter because
   both are "media transformation buttons" attached to texture uniforms. Could split further
   if a third widget joins.

4. **Each widget gets the shape that fits its job — no enforced uniform contract.** A widget is
   just a chunk of imgui-drawing code worth lifting out of `ui.py`; it is NOT an instance of a
   `Widget` protocol, an ABC, or a polymorphic interface. There is no `list[Widget]` consumer
   anywhere in the codebase, so uniformity buys nothing.

   The "Shape" column in the extraction-targets table records each widget's concrete contract:
   - **Value-object editors** (`draw_file_details`, `draw_resolution_details`,
     `draw_media_details`) take a pydantic model and return the updated copy. The body starts
     with `details = details.model_copy()`.
   - **Value transformers** (`draw_media_models`, `draw_video_filters`) take a value, return a
     possibly-different value. The caller compares identity (`if new is not current`) before
     calling `try_to_release(current)` and writing into storage — only the caller knows the
     storage location, so the widget can't safely do this itself.
   - **Action-firers / state-mutators** (`draw_selected_ui_uniform_settings`, `draw_ui_uniform`,
     `draw_node_preview_grid`) mutate in place via `ctx` callbacks (`ctx.set_current_node_id`,
     `ctx.notifications_push`) or by writing directly to the storage dict (`uniform_values`)
     when the widget's whole subject IS the dict-and-key.
   - **Popups** are classes (Decision 9).

   *Alternatives investigated and rejected (see Review history):*
   - **Always-return** with synthetic `Result` dataclasses: forces ~8 new dataclasses for one
     consumer each; `draw_settings` would unconditionally write 4 fields every frame.
   - **Always-mutate**: breaks `try_to_release` ordering at `ui.py:855-861, 1019-1021` (caller
     needs both old and new in scope); breaks the popup keep-open `bool` protocol enforced by
     `draw_popup_if_opened` at `ui.py:306-315` (which Decision 9 retires anyway, but at the
     cost of a different out-of-band channel).
   - **In-out `(changed, new_value)` tuples uniformly**: drops 3 callbacks from `AppContext`
     but adds a permanent `notification` slot tax to every notifying widget and still needs a
     compound dataclass for `draw_ui_uniform`'s 4-event return. Tuple width varies 1→4, so
     "uniform shape" is illusory.
   - **Reducer / event-bus (Elm/Redux)**: architectural over-engineering for imgui's
     immediate-mode model — ~15 sites do `state.X = imgui.foo(state.X)[1]` per frame, which
     either bloats the event sum type to 25+ variants of plumbing or escapes back to direct
     mutation, breaking the pattern.
   - **`Widget` ABC + classes everywhere**: no polymorphic consumer exists; deviates from the
     established `tabs/share.py` free-function precedent.

5. **~~`AppContext` exposes state + bound-method callbacks~~ — REVERSED.** See Decision 1.
   Widgets call `app.notifications.push(...)`, `app.set_current_node_id(...)`, etc. — direct
   attribute access on the `App` instance. The dataclass below is kept here only for historical
   context; it no longer exists in the code:

   ```python
   @dataclass
   class AppContext:
       """Thread affinity: render-thread only. Held as App._ctx.

       app_state, ui_nodes, ui_node_templates, modelbox_info are LIVE REFERENCES
       (mutation through them propagates back to App). media_dir / trash_dir are
       snapshot Path values, rebuilt by _init() on project switch.
       """
       app_state: UIAppState              # live reference
       ui_nodes: dict[str, UINode]        # live reference
       ui_node_templates: dict[str, UINode]  # live reference
       media_dir: Path                    # snapshot
       trash_dir: Path                    # snapshot
       modelbox_info: dict[str, Any]      # live reference; matches ui.py:110 exactly

       # callbacks
       notifications_push: Callable[[str, RGB], None]
       set_current_node_id: Callable[[str], None]
       fetch_modelbox_info: Callable[[], None]
       create_node_from_selected_template: Callable[[], None]
       edit_current_node_fs_file: Callable[[], None]
       open_current_node_dir: Callable[[], None]
       save_ui_node: Callable[[UINode, Path | None, str | None], Path]
       get_current_node_ui_state_or_default: Callable[[], UINodeState]

       # popup instances (Decision 9). Constructed by App.__init__ before _ctx.
       node_creator: NodeCreatorPopup
       settings: SettingsPopup

       def any_popup_open(self) -> bool:
           return self.node_creator.is_open or self.settings.is_open
   ```

   Notes on each field:
   - `modelbox_info: dict[str, Any]` matches the existing `App.modelbox_info` type at
     `ui.py:110` verbatim — no new pydantic model is introduced (this is a pure refactor).
   - `get_current_node_ui_state_or_default` is needed by `draw_video_filters` (ui.py:776).
     Exposed as a callback because it depends on `current_node_id` resolution which the widget
     shouldn't replicate.
   - `any_popup_open()` preserves the "at most one popup open" invariant that the single
     `_active_popup_label: str | None` field expressed by construction. Without it, the
     two-booleans design leaks the invariant. Used by the render-gate and hotkey-routing
     translations in Decision 11.

6. **`current_node_id` is read from `ctx.app_state.current_node_id`, not a separate field.**
   One source of truth; `App.current_node_id` is already a `@property` accessor.

7. **No behavior changes during extraction.** Each body is moved verbatim modulo `self.X` →
   `ctx.X` rewrites and the popup `bool`-return → `self._is_open = False` rewrites. Any other
   change (bug fixes, simplifications) is out of scope.

8. **`App`'s `@property` paths stay** (`media_dir`, `trash_dir`, `nodes_dir`, etc.) for
   non-widget callers. `AppContext` snapshots the resolved `Path` values at `_init` time. After
   `open_project` → `_init` re-runs, `AppContext` is rebuilt fresh.

9. **The two popups are classes, not functions.** `NodeCreatorPopup` and `SettingsPopup` each
   own their `_is_open` flag plus a public `is_open` property and a `close()` method,
   replacing the global `_active_popup_label` mechanism. Shape:

   ```python
   class NodeCreatorPopup:
       def __init__(self) -> None:
           self._is_open: bool = False

       @property
       def is_open(self) -> bool:
           return self._is_open

       def open(self) -> None:
           # NOTE: do NOT call imgui.open_popup() here. Callers may invoke open() from
           # the pre-frame hotkey block (ui.py:1310-1346, runs before imgui.new_frame() at
           # 1350). The actual imgui.open_popup() call happens inside draw(), guarded by
           # imgui.is_popup_open() — mirrors the current idempotent pattern at ui.py:1462-1465.
           self._is_open = True

       def close(self) -> None:
           self._is_open = False

       def draw(self, ctx: AppContext) -> None:
           if not self._is_open:
               return
           if not imgui.is_popup_open(_LABEL):
               imgui.open_popup(_LABEL)
           opened, _ = imgui.begin_popup_modal(_LABEL)
           if not opened:
               self._is_open = False
               return
           try:
               # ... popup body; mutates ctx.app_state directly;
               # calls ctx.create_node_from_selected_template() on Create;
               # calls self.close() (or self._is_open = False) to close
               ...
           finally:
               imgui.end_popup()
   ```

   Why classes here specifically: popups have a real piece of session-only state
   (`_is_open`) that is widget-internal, not project state. Encapsulating it in `self` is
   strictly cleaner than threading a label string through a global dispatcher. *Why not
   classes for the other 8 widgets:* they have no widget-internal state (every mutation
   either belongs to `app_state` and is persisted, or is fired via `ctx` callbacks), so a
   class is pure ceremony.

   *Critical imgui frame-stack note:* hotkey handlers run at `ui.py:1310-1346`, BEFORE
   `imgui.new_frame()` at line 1350. Calling `imgui.open_popup()` before `new_frame()` is
   undefined imgui behavior. The `open()` method therefore only flips `_is_open`; the
   `imgui.open_popup()` call is deferred to `draw()`, guarded by `imgui.is_popup_open()` for
   idempotence. This preserves the exact frame-ordering of the current code.

10. **`draw_popup_if_opened` (ui.py:306-315) is retired.** Only callers are the 2 known popup
    dispatches (`ui.py:1467, 1471` — confirmed by grep). Replaced by `popup.draw(ctx)` calls
    that internally guard on `self._is_open`.

11. **Every `_active_popup_label` read/write site is migrated.** The field is read/written in
    more places than the dispatcher. Full translation table — every site MUST be updated for
    the refactor to be behavior-preserving:

    | site | current code | post-refactor |
    |---|---|---|
    | `ui.py:112` (field def) | `self._active_popup_label: str \| None = None` | **deleted**; replaced by `self._ctx.node_creator` and `self._ctx.settings` instances |
    | `ui.py:1296` (render gate) | `if self._active_popup_label is None: ...` | `if not self._ctx.any_popup_open(): ...` |
    | `ui.py:1304` (render-templates gate) | `elif self._active_popup_label == self._NODE_CREATOR_POPUP_LABEL: ...` | `elif self._ctx.node_creator.is_open: ...` |
    | `ui.py:1323` (Ctrl+N hotkey) | `self._active_popup_label = self._NODE_CREATOR_POPUP_LABEL` | `self._ctx.node_creator.open()` |
    | `ui.py:1326` (Alt+S hotkey) | `self._active_popup_label = self._SETTINGS_POPUP_LABEL` | `self._ctx.settings.open()` |
    | `ui.py:1329-1331` (Esc handler) | `if self._active_popup_label is None: glfw.set_window_should_close(...); self._active_popup_label = None` | `if not self._ctx.any_popup_open(): glfw.set_window_should_close(...); self._ctx.node_creator.close(); self._ctx.settings.close()` |
    | `ui.py:1334-1338` (Left/Right when no popup) | `if not self._active_popup_label: ... select_next_current_node(...)` | `if not self._ctx.any_popup_open(): ... select_next_current_node(...)` |
    | `ui.py:1339-1346` (Left/Right/Enter inside node-creator) | `if self._active_popup_label == self._NODE_CREATOR_POPUP_LABEL: ...; self._active_popup_label = None` | `if self._ctx.node_creator.is_open: ...; self._ctx.node_creator.close()` |
    | `ui.py:1376` ("Settings" button in main menu) | `self._active_popup_label = self._SETTINGS_POPUP_LABEL` | `self._ctx.settings.open()` |
    | `ui.py:1462-1465` (centralized `imgui.open_popup` block) | `if _active_popup_label is not None and not imgui.is_popup_open(label): imgui.open_popup(label)` | **deleted**; logic now inside each `popup.draw()` |
    | `ui.py:1467-1473` (popup-body dispatch via `draw_popup_if_opened`) | `if self._active_popup_label == X: ... draw_popup_if_opened(label, draw_X)` | `self._ctx.node_creator.draw(self._ctx); self._ctx.settings.draw(self._ctx)` |
    | `ui.py:397` (popup-open from node-grid widget) | `_active_popup_label = self._NODE_CREATOR_POPUP_LABEL` | `ctx.node_creator.open()` (widget calls via `ctx`) |
    | `_NODE_CREATOR_POPUP_LABEL` / `_SETTINGS_POPUP_LABEL` class constants | currently on `App` | move to module-level constants in `popups/node_creator.py` / `popups/settings.py` |

    The `any_popup_open()` helper (Decision 5) preserves the "at most one popup open"
    invariant that the single-string discriminator expressed by construction. Without it, the
    two-booleans design would leak the invariant to every read site.

## Files touched

**Created:**
- `shaderbox/app_context.py` — ~35 lines (dataclass + factory function building it from `App`).
- `shaderbox/widgets/__init__.py` — empty.
- `shaderbox/widgets/details.py` — ~140 lines (3 extracted functions + types).
- `shaderbox/widgets/uniform.py` — ~220 lines (2 extracted functions).
- `shaderbox/widgets/media_ops.py` — ~80 lines (2 extracted functions).
- `shaderbox/widgets/node_grid.py` — ~55 lines (1 extracted function — node-creator moves to popups).
- `shaderbox/popups/__init__.py` — empty.
- `shaderbox/popups/node_creator.py` — ~55 lines (`NodeCreatorPopup` class).
- `shaderbox/popups/settings.py` — ~110 lines (`SettingsPopup` class).

**Modified:**
- `shaderbox/ui.py` — 10 method bodies removed (~600 lines); `App.__init__` constructs
  `self._ctx` (which holds the popup instances); call sites become `widgets.X.draw_Y(self._ctx, ...)`
  and `self._ctx.<popup>.draw(self._ctx)`. The `_active_popup_label` field and
  `draw_popup_if_opened` helper are deleted. Hotkey writes switch to `popup.open()`. Expected:
  1508 → ~900 lines.
- `shaderbox/tabs/share.py` — **unchanged.** Retrofit to `AppContext` deferred to feature 003.
- `ai_docs/conventions.md` — `## Design decisions`: add a "**Widgets are an organizational
  convention, not a contract**" bullet — widget functions live in `widgets/*.py`, take
  `AppContext`, and choose whatever return shape fits their job. Popups are classes. Revisit
  when 5+ widget modules exist and a pattern emerges.
- `ai_docs/dev_flow.md ## Recipes` module map: add `widgets/`, `popups/`, `app_context.py`.
- `ai_docs/todo.md` — update `[DEFERRAL] split ui.py` to note widget+popup extraction landed;
  add follow-up trigger for `tabs/node.py` + `tabs/render.py` extraction.
- `ai_docs/worklog.md` — new entry on completion.

**Deleted:** none.

## Manual verification

Pure refactor — no behavior change, so verification is "the app still works the same."

1. **`make check`:** ruff fix + format clean; pyright still non-blocking with **0 errors** post-
   refactor (no regression from feature 001's clean state). Any new error is a FAIL.
2. **Launch the app:** `uv run python ./shaderbox/ui.py` against `projects/dev/`. No import
   errors, no startup errors, window opens normally.
3. **Visual regression sweep:** click through every tab (Node, Render, Share). Every widget
   pixel-identical to before. Imgui layout is fragile to refactor; this is the real check.
4. **Settings popup:** Alt+S opens it; every field (FPS, text editor cmd, modelbox URL) reads
   and writes correctly. Apply, save (Ctrl+S), reopen — values persist.
5. **Node creator:** Ctrl+N opens the popup; select a template, create a node; verify the new
   node appears in the grid and is selected. Cancel button works.
6. **Node preview grid:** renders all nodes with correct borders (green for selected, red for
   shader error); clicking selects; arrow keys navigate.
7. **Uniform widgets:** select a node with multiple uniform input types (color, drag, texture,
   text, array, **buffer**). Every type renders correct UI; editing each updates the shader in
   real time. **Buffer-uniform check:** the "Randomize" button writes via `moderngl.Buffer.write`
   (ui.py:910-912 in the original code) — verify it still produces a visible shader change.
8. **Media ops (ModelBox):** if a ModelBox server is running, select a texture uniform →
   "Generate" runs the model and the texture updates. Skip if no server; degrade gracefully.
9. **Video filters:** select a video texture uniform → temporal smoothing runs ffmpeg and
   produces a new video uniform value.
10. **File / resolution / media details widgets:** in the Render tab, resolution drag-ints and
    aspect-ratio buttons work; file path picker opens the correct dialog; media details panel
    shows resolution / FPS / duration correctly.
11. **Project switch:** Ctrl+O, pick a different project. `_ctx` is rebuilt — `media_dir` /
    `trash_dir` snapshots reflect the new project; `app_state` / `ui_nodes` references point at
    the freshly-loaded state. Popup instances are owned by `App` (constructed once in
    `App.__init__`, *before* `_ctx`); they survive project switch with `_is_open = False`.

A real UX gap at any step is a FAIL, not pass-with-caveat.

## Open questions for the user

*(All resolved during plan-lock — none open.)*

- **Q1: `AppContext` design.** Resolved → (c) the dataclass.
- **Q2: `tabs/share.py` retrofit.** Resolved → leave; harmonize in feature 003.
- **Q3: Scope of 002 vs 003.** Resolved → 002 = widgets + popups; 003 = tabs.
- **Q4: Uniform widget return shape.** Resolved → no uniform shape; widgets are an
  organizational convention, not a contract. See Decision 4 + Review history for the full
  investigation.

## Review history

### Pre-lock investigation (2026-05-15)

User pushed back on the original "per-widget shape, follow your intuition" framing of Q4,
asking for a deeper investigation. Spawned 6 parallel investigations:

**Round 1 — three uniform-style candidates:**
- **Always-mutate**: works for 7/10 widgets, genuinely resists on 3. `draw_media_models` +
  `draw_video_filters` need return values for the caller's `try_to_release(old); store=new`
  swap (`ui.py:855-861`). `draw_node_creator` + `draw_settings` need the `bool` return for the
  popup-keep-open protocol enforced by `draw_popup_if_opened` (`ui.py:306-315`). Rejected.
- **Always-return**: works for 3/10 cleanly. Forces 8 synthetic compound dataclasses for one
  consumer each, plus ~25-40 LOC of caller-side `if r.X: state.X = r.X` dispatch.
  `draw_ui_uniform` would need a 4-slot dataclass for its multi-event return; `draw_settings`
  would return all 4 fields every frame (imgui's per-field changed-flag is currently
  discarded). Rejected.
- **Per-widget hybrid governed by a `model_copy()` trigger rule**: classification unambiguous
  for 9/10; matches imgui's own `(changed, value)` tuple convention for value-shaped widgets
  and `tabs/share.py`'s mutate-via-reference for state-shaped widgets. Survived round 1.

**Round 2 — three alternative re-designs:**
- **In-out `(changed, new_value)` tuples uniformly**: partial win. Drops 3 callbacks from
  `AppContext` (`notifications_push`, `set_current_node_id`,
  `create_node_from_selected_template`). Handles `try_to_release` cleanly (caller has both
  values in scope). Doesn't deliver shape-uniformity (tuple width varies 1→4); adds permanent
  notification-slot tax to non-notifying widgets; `draw_ui_uniform` still needs a 4-slot
  compound. Not a clean win, would force the same compound problem relocated to tuples.
- **Reducer / event-bus (Elm/Redux)**: architectural over-engineering. Imgui's immediate-mode
  reads (~15 sites doing `state.X = imgui.foo(state.X)[1]` per frame) have no discrete change
  boundary; modelling them as events either bloats the sum type to 25+ plumbing variants or
  escapes to direct mutation, breaking the invariant. No multi-consumer use case (no
  time-travel, no replay, no subscribers) justifies the structure. Rejected.
- **Class-based widgets uniformly**: only one piece of state genuinely wants encapsulation
  (`_active_popup_label`, session-only, single-widget per popup). Everything else either
  belongs to `app_state` (persisted, mandated by no-behavior-change) or is stateless. Adds
  `__init__` ceremony, deviates from `tabs/share.py` free-function precedent, doesn't collapse
  the 10-signature variance. Scoped variant adopted (Decision 9): classes for the 2 popups
  only, free functions for the 8 widgets.

**Convergent conclusion:** widgets are an organizational convention, not a polymorphic
contract. Each widget gets the shape that fits its job. The only common contract is
`AppContext`. The 2 popups become classes to encapsulate their `_is_open` state.

### Pre-impl review (2026-05-15)

Three parallel reviewers (spec fidelity / architecture / devil's advocate) ran on the
plan-locked spec. Convergent findings:

**BLOCKERS (all 3 agreed) — fixed in this revision:**

1. **`popup.open()` calling `imgui.open_popup()` directly is broken.** Hotkeys (`ui.py:1310-1346`)
   run BEFORE `imgui.new_frame()` at line 1350. `imgui.open_popup` from a pre-frame handler is
   undefined imgui behavior. **Fix applied:** `popup.open()` only flips `_is_open`; the
   `imgui.open_popup()` call happens inside `popup.draw()`, guarded by `imgui.is_popup_open()`
   for idempotence — mirrors the current centralized pattern at `ui.py:1462-1465`. See
   Decision 9.

2. **`_active_popup_label` has 5+ read sites the original spec didn't enumerate.** Reads at
   `ui.py:1296` (render-vs-templates gate), `1304` (templates render), `1328-1331` (Esc closes
   any popup), `1334-1346` (Left/Right/Enter routing), `1376` (Settings button), `1462-1465`
   (centralized open trigger), `1467-1473` (dispatcher). **Fix applied:** Decision 11 now
   contains the full translation table for every site. Added `ctx.any_popup_open()` helper to
   preserve the "at most one popup open" invariant that the single-string discriminator
   expressed by construction.

3. **`current_node_ui_state_or_default` missing from `AppContext`.** `draw_video_filters`
   (`ui.py:776`) reads `self.current_node_ui_state_or_default` and the original `AppContext`
   field list didn't expose it. **Fix applied:** added
   `get_current_node_ui_state_or_default: Callable[[], UINodeState]` to Decision 5.

**MAJORS (2+ reviewers agreed) — fixed:**

4. **`AppContext.app_state` aliasing ambiguity.** Decision 8's "snapshot paths" wording risked
   an implementer over-snapshotting `app_state`, which would break every
   `state.X = imgui.foo(state.X)[1]` write site. **Fix applied:** Decision 5's docstring now
   explicitly distinguishes live references (`app_state`, `ui_nodes`, `ui_node_templates`,
   `modelbox_info`) from snapshots (`media_dir`, `trash_dir`).

5. **`modelbox_info` type drift.** Original spec invented `ModelboxInfo | None` but the actual
   field at `ui.py:110` is `dict[str, Any]`. "Pure refactor" cannot introduce new types. **Fix
   applied:** Decision 5 now uses `dict[str, Any]` verbatim.

**MINORS — fixed:**

- `draw_node_preview_grid` width/height typed `float` (caller passes `control_panel_width / 2.6`
  at `ui.py:1450`).
- Goal section now leads with "unblocks feature 003," not the line-count drop.
- Decision 1 framing: honest "narrow god-context" rather than "NOT a god-object."
- Verification step 7 adds the buffer-uniform check.
- Verification step 11 clarifies popups are owned by `App`, not `_ctx` — survive project switch.

**Risks acknowledged (not blocked but noted for impl):**

- Tab modules left in `ui.py` still call private widget methods; every internal call site in
  `draw_render_tab` (`ui.py:1032`), `draw_node_tab` (`ui.py:716, 726`) etc. must update to
  `widgets.X.draw_Y(self._ctx, ...)`.
- The `_active_popup_label` translation table at Decision 11 is the highest-risk patch — easy
  to miss a site. Implementer should grep the final tree for any remaining
  `_active_popup_label` / `_NODE_CREATOR_POPUP_LABEL` / `_SETTINGS_POPUP_LABEL` references
  before declaring done.

**Survived all attacks (not changed):**

- The "no uniform contract for widgets" decision (Decision 4) — the 6-investigation Review
  history at Round 1/Round 2 holds up.
- The widgets-as-free-functions / popups-as-classes asymmetry — `_is_open` is the only
  widget-internal session state in the spec, justifying the asymmetry concretely.
- The deferral of `tabs/share.py` retrofit to feature 003.
- Module layout (`media_ops` mixing ModelBox + filters is defensible).

### Post-impl reversal: AppContext → app (2026-05-15)

User questioned whether `AppContext` provided real value vs. passing the `App` object directly.
Spawned an investigation. Findings:

- **Bounded coupling claim:** PARTIAL — the fence is real (15-field surface vs ~50 on `App`),
  but the threat (widget accidentally calling `delete_current_node()`) is solo-project hygiene
  pyright + grep already cover. `tabs/share.py` survives without this fence.
- **Render-thread-only boundary:** ILLUSORY — not enforced anywhere, just a docstring. AppContext
  carries `ui_nodes` which transitively reaches every moderngl object on `App`. No actual barrier.
- **Test/audit boundary:** ILLUSORY — zero test files in the repo. Mocking-friendliness is
  hypothetical.
- **Stable signatures under field additions:** REAL vs. individual-args alternative; ILLUSORY
  vs. passing `app` (same property — adding a method to `App` requires zero changes at call sites).
- **`any_popup_open()` invariant preservation:** ILLUSORY — the method just `or`s two booleans;
  same one-liner could live on `App` (now does).
- **Circular import decoupling:** PARTIAL — real concern, but the same `if TYPE_CHECKING: from
  shaderbox.ui import App` trick AppContext already used for popup classes solves it for widgets.

**Concrete costs that disappear with the reversal:**
- `app_context.py` — 47 lines, deleted.
- `_build_ctx` method — 19 lines in `ui.py`, deleted.
- `_ctx` rebuild dance in `_init`, deleted.
- Asymmetric naming layer existed *only* because AppContext renamed things: `_ui_app_state` →
  `app_state` (callback field), `_notifications.push` → `notifications_push` (callback),
  lambda wrapper for `current_node_ui_state_or_default`, `_node_creator_popup` → `node_creator`.

**What the reversal cost:**
- 3 App attrs made public: `_ui_app_state` → `app_state`, `_notifications` → `notifications`,
  `_node_creator_popup` / `_settings_popup` → drop underscore.
- `App.any_popup_open()` added (one line).
- Widgets/popups gain visibility into all of `App` — including `delete_current_node`, `release`,
  etc. This is the real trade-off, accepted because the surface boundary is better expressed
  by *continuing to extract from `App`* over time than by wrapping it.

The strongest signal: `tabs/share.py` (feature 001) explicitly opted out of context-bundling
and worked fine. The spec rationalized this as "harmonize in feature 003" — that admission was
the tell. The abstraction was built without waiting to see if it was needed.

**`ui.py` final size:** 1508 → 859 lines (was 879 with AppContext; saved 20 from `_build_ctx`
+ `_ctx` rebuild).

### Post-impl reversal 2: popup classes → free functions (2026-05-15)

User pushed back on a `TYPE_CHECKING` workaround needed to break a circular import between
`app.py` (post-AppContext-deletion attempt to move `App` there) and `popups/*.py`. Investigation
identified the popup classes as the cycle source: `App.__init__` instantiated
`NodeCreatorPopup()` / `SettingsPopup()`, the popups needed `App` for the type annotation on
`draw(self, app: App)`. Three structural fixes explored; chosen one: **flatten popup classes to
free functions**, move the only encapsulated state (`_is_open`) to plain booleans on `App`.

Result:
- `popups/node_creator.py` and `popups/settings.py` are now single-function modules:
  `draw_node_creator(app: App)`, `draw_settings(app: App)`. Early-return when closed; render
  modal body when open.
- `App` gains `is_node_creator_open: bool` and `is_settings_open: bool` fields, plus
  `open_node_creator()` / `open_settings()` helpers that set one True and clear the other (this
  preserves the "at most one popup open" invariant that the original `_active_popup_label: str | None`
  single-slot model expressed by construction — a pre-impl reviewer caught the regression
  before commit).
- `app.any_popup_open()` answers the render-gate question; `app.is_node_creator_open` and
  `app.is_settings_open` are read by hotkey routing and the render-templates branch.

User framing: **"App = runtime state, widgets/popups = pure logic, no state on the UI side."**

### Post-impl restructure: App → app.py + dispatch → ui.py + tabs extracted (2026-05-15)

The circular import that motivated the popup-flatten extended further: widgets ALSO need to
type-annotate `app: App`, and were also subject to the cycle even though they're free functions
(Python evaluates the annotation reference at module-load time when `app.py` is mid-loading).
Three structural options explored; chosen one: **move `App` to its own module**, and move all
imgui-dispatch out of `App` into `ui.py` as free functions, so `app.py` imports nothing from
widgets/popups/tabs/ui.

Result (final post-feature-002 layout):
- `shaderbox/app.py` (373 lines) — `App` class: state + lifecycle (`_init`, `save`, `release`,
  `open_project`, `delete_current_node`, `create_node_from_selected_template`,
  `fetch_modelbox_info`, `edit_current_node_fs_file`, `open_current_node_dir`, `save_ui_node`,
  `select_next_current_node`, `select_next_template`, `set_current_node_id`,
  `any_popup_open` / `open_node_creator` / `open_settings`, `current_node_id` /
  `current_node_ui_state_or_default` properties, `@property` paths). Public attrs:
  `notifications`, `app_state`, `ui_nodes`, `ui_node_templates`, `modelbox_info`,
  `exporter_registry`, `share_tab_state`, `font_14` / `font_18`, `preview_canvas`, `window`,
  `imgui_renderer`, `is_node_creator_open`, `is_settings_open`, `global_fps`, `frame_idx`,
  `app_start_time`.
- `shaderbox/ui.py` (294 lines) — orchestrator: `run(app)`, `update_and_draw(app)` (frame
  loop body), `_draw_node_settings(app)` (tab-bar dispatcher calling `node_tab.draw` /
  `render_tab.draw` / `share_tab.draw`), `main()`.
- `shaderbox/tabs/node.py` (146 lines) — `draw(app)`, node tab.
- `shaderbox/tabs/render.py` (59 lines) — `draw(app)`, render tab + `_draw_render_button`.
- `shaderbox/tabs/share.py` (175 lines) — `draw(app)`, `update(app)`, share tab. Now consistent
  with the other tabs (was the spec's deferred Q2; no longer deferred). Internal `TabState`
  dataclass lives in `tabs/share_state.py` (18 lines) so `app.py` can import it without
  cycling through `share.py`.
- `shaderbox/widgets/` (547 lines across 4 files), `shaderbox/popups/` (166 lines across 2 files)
  — all stateless, take `app: App` at module top.

Dependency graph (no cycles):
```
ui.py ─→ app.py
  │        ↑
  ├─→ tabs/{node,render,share}.py ─┤
  ├─→ widgets/*.py ─────────────────┤
  └─→ popups/*.py ──────────────────┘
app.py ─→ tabs/share_state.py (no cycle: share_state imports only media + exporters.base)
```

**`ui.py` final size:** 1508 → 294 lines (-1214). **`App` class extracted to `app.py`** (373
lines). **Tabs extracted in same wave**: `tabs/node.py` + `tabs/render.py` new; `tabs/share.py`
harmonized. Total imgui-drawing surface lives in `widgets/` + `popups/` + `tabs/` (~1110 lines)
+ `ui.py` orchestration (~290 lines) — separated from `App` data (`app.py`, 373 lines).
