# Feature 004 — `pyimgui` → `imgui-bundle` migration

## Goal

Replace the `pyimgui` (package: `imgui`) library with `imgui-bundle` (Dear ImGui Bundle) as the
Python binding for Dear ImGui. After this lands, ShaderBox uses imgui-bundle's actively-maintained
bindings (tracks current Dear ImGui releases vs. pyimgui frozen at Dear ImGui 1.82 / March 2021),
has complete `.pyi` type stubs (eliminating most `# type: ignore` suppressions per
`conventions.md ## Known quirks`), adopts the library's officially-documented context-manager
idiom for begin/end pairs, and swaps `crossfiledialog` for the imgui-bundle-bundled
`portable_file_dialogs` (drop-in, fewer external deps).

This is a **large, high-blast-radius feature**: 322 imgui calls across 15 files, every UI module
touched, dependency swap, integration-layer rewrite.

## Out of scope

- **`hello_imgui` high-level boot.** imgui-bundle ships a `hello_imgui.run(...)` boot helper that
  takes over the render loop. ShaderBox requires manual loop control (moderngl canvas updates +
  per-frame share-tab callbacks before `imgui.new_frame()`); the low-level GLFW backend
  (`imgui_bundle.python_backends.glfw_backend.GlfwRenderer`) is the correct integration.
  **Trigger to revisit**: never — unless a future feature requires multi-window or web/Pyodide
  support, both of which are `hello_imgui` strengths. (Source: imgui-bundle docs cited below.)
- **All bundled addons except `portable_file_dialogs`.** `imgui_md`, `implot`, `imspinner`,
  `imgui-knobs`, `imgui_node_editor`, `imgui_color_text_edit`, `immvision`, `im_cool_bar` —
  audited in spike round 2 (`ai_docs/features/004_*` review history below). No current need.
  Each has a documented bundled-addon adoption pattern if/when a use case arises.
- **`shaderbox/notifications.py` rewrite.** Custom in-house 44-line toast stack; no imgui-bundle
  native equivalent. Stays as-is.
- **GLSL shader-error rendering UX improvement.** Spike 2 noted markdown rendering could
  prettify shader errors, but ShaderBox's errors are GLSL-compiler raw output (terse, single-
  line). Out of scope. **Trigger to revisit**: first user complaint about unreadable shader
  errors.
- **`imgui-knobs` adoption for shader parameter UX.** Spike-deferred; would require uniform-
  widget refactor with new parameter-type metadata. **Trigger to revisit**: post-migration, if
  drag-float UX feels insufficient for shader parameter tweaking.

## Design decisions (locked)

Each decision cites the official source that confirmed it (per the
`CLAUDE.md ## Hard rules` "library docs as source of truth" rule).

### 1. Manual render loop preserved; low-level GLFW backend
`imgui_bundle.python_backends.glfw_backend.GlfwRenderer` is the integration class. Inherits from
`ProgrammablePipelineRenderer` (OpenGL 3.3+). Same boot pattern as pyimgui: `glfw.init()` →
`glfw.create_window()` → `imgui.create_context()` → `GlfwRenderer(window)`.
**Source**: https://github.com/pthom/imgui_bundle/tree/main/bindings/imgui_bundle/python_backends
(`glfw_backend.py`, `opengl_backend_programmable.py`).

### 2. Drop `contextlib.suppress(GLError)` around `imgui_renderer.render(...)`
`ProgrammablePipelineRenderer.render()` saves and restores all common GL state via
`get_common_gl_state()` / `restore_common_gl_state()`. The defensive suppression in
`ui.py:262-264` is unnecessary with imgui-bundle. Removing it surfaces real GL errors instead
of swallowing them.
**Source**:
https://raw.githubusercontent.com/pthom/imgui_bundle/main/bindings/imgui_bundle/python_backends/opengl_backend_programmable.py

### 3. Drag/Input kwarg renames (BREAKING)
For `drag_int`, `drag_float` (and their variants `drag_int2/3/4`, `drag_float2/3/4`):
- `min_value=` → `v_min=`
- `max_value=` → `v_max=`
- `change_speed=` → `v_speed=`

For `input_text`:
- Drop `buffer_length=` kwarg entirely. imgui-bundle auto-sizes the buffer.

**Source**: https://raw.githubusercontent.com/pthom/imgui_bundle/main/bindings/imgui_bundle/imgui/__init__.pyi
(definitive `.pyi` stub: `drag_int(label, v, v_speed=1.0, v_min=0, v_max=0, ...)`).

### 4. Color-argument shape changes — `color_edit3/4` AND `text_colored` (BREAKING)

**`color_edit3` / `color_edit4`**:
- pyimgui: `imgui.color_edit3("##name", *vals)` (unpack list)
- imgui-bundle: `imgui.color_edit3("##name", vals)` (pass list)

Return shape stays `(changed, vals)`.

**`text_colored`** (caught by convergence loop round 2 — NOT in original spike output):
- pyimgui: `imgui.text_colored(text, r, g, b, a=1.0)` — text first, color components unpacked
- imgui-bundle: `imgui.text_colored(col: ImVec4Like, fmt: str)` — **color first**, color as
  single `ImVec4Like` value (tuple `(r,g,b,a)` or `ImVec4`), NOT unpacked components

Both argument **order** and **shape** change. 19 call sites across **9 files**:
`popups/settings.py:41`, `widgets/node_grid.py:28`, `widgets/uniform.py:104`,
`widgets/details.py:42,48`, `tabs/render.py:45,55`, `tabs/share.py:32,67,68,116`,
`tabs/node.py:98`, `notifications.py:42`, `exporters/telegram.py:208,210,213,227,275,292`.

**Standard rewrite** (call-site has literal color tuple, e.g.
`*(0.5, 0.5, 0.5)`): `imgui.text_colored(text, *(r,g,b))` → `imgui.text_colored((r,g,b,1.0), text)`.
imgui-bundle requires 4-component color; add `1.0` alpha if pyimgui call was 3-component.

**Stored-tuple rewrite** (call-site unpacks a stored 3-tuple `color` variable — happens in
`notifications.py:42` where `color` is a `(r,g,b)` notification field, and similar in
`widgets/details.py:42,48` and `exporters/telegram.py:208,210,213,292`): rewrite at the
**call site only**, not at the storage. Replace `imgui.text_colored(text, *color)` with
`imgui.text_colored((*color, 1.0), text)`. Do NOT change `Notifications.push()` signature
or the stored tuple shape — keep storage as 3-component, append alpha at call site. This
keeps the migration localized to imgui-touching code per the "only change what's asked"
rule.

**Source**: imgui-bundle pyi stub —
`def text_colored(col: ImVec4Like, fmt: str) -> None` (verified by direct curl + grep).

### 5. Context-manager idiom adopted for begin/end pairs (officially recommended)
imgui-bundle's official demos use `with imgui_ctx.begin_*(...) as ...:` instead of the proxy
`.selected` / `.opened` attribute idiom. Per the `CLAUDE.md ## Hard rules` "adopt the
library's documented best-practice idiom even if it differs from the existing in-repo shape",
we migrate.

Affected pyimgui idioms → imgui-bundle replacements:
- `if imgui.begin_tab_item("X").selected: ... imgui.end_tab_item()` → `with imgui_ctx.begin_tab_item("X") as tab: if tab: ...`
- `if imgui.begin_popup_modal(_LABEL).opened: ... imgui.end_popup()` → context-manager form
- `if imgui.begin_tab_bar(...).opened: ... imgui.end_tab_bar()` → context-manager form
- `imgui.begin(...); ...; imgui.end()` → `with imgui_ctx.begin(...): ...`
- `imgui.begin_child(...); ...; imgui.end_child()` (or `with imgui.begin_child(...)`) → `with imgui_ctx.begin_child(...): ...`
- `imgui.begin_tooltip(); ...; imgui.end_tooltip()` → `with imgui_ctx.begin_tooltip(): ...`

**Source**:
https://raw.githubusercontent.com/pthom/imgui_bundle/main/bindings/imgui_bundle/demos_python/demos_immapp/demo_python_context_manager.py
(official demo demonstrating the pattern).

The proxy `.selected`/`.opened` idiom still works in imgui-bundle (backward compatible), but
the rule mandates adopting the documented idiom.

### 6. `crossfiledialog` → `imgui_bundle.portable_file_dialogs` swap
Same underlying C++ library (`samhocevar/portable-file-dialogs`); same platform support
(Linux/macOS/Windows native dialogs); same blocking behavior. Drops one external dependency.
Three call sites:
- `app.py::open_project` (folder picker) → `pfd.select_folder(title, default_path=start_dir)`
- `widgets/uniform.py::draw_ui_uniform` (texture file picker with filter) → `pfd.open_file(title, default_path=".", filters=[...])`
- `widgets/details.py::draw_file_details` (save path picker) → `pfd.save_file(title, default_path=".")`

**Source**: https://github.com/samhocevar/portable-file-dialogs (the underlying C++ library
that both crossfiledialog and imgui-bundle wrap; same Linux backend: zenity/kdialog/qarma).

Filter-format translation:
- crossfiledialog: `filter=["*.png", "*.jpg"]` (flat list of extensions)
- pfd: `filters=["Image Files", "*.png *.jpg"]` (alternating label/pattern pairs, space-
  separated within a pattern). Adapt the 1 call site in `widgets/uniform.py`.

### 7. Window flags BREAKING (enum form); modifier keys + font loading unchanged
- **Window flags**: imgui-bundle exposes `WindowFlags_(enum.IntFlag)` with snake_case members.
  pyimgui's UPPER_CASE module-level constants (`imgui.WINDOW_NO_COLLAPSE`,
  `imgui.WINDOW_ALWAYS_AUTO_RESIZE`, `imgui.WINDOW_NO_TITLE_BAR`, `imgui.WINDOW_NO_SCROLLBAR`,
  `imgui.WINDOW_NO_SCROLL_WITH_MOUSE`) do NOT exist in imgui-bundle. Rewrite every usage to
  `imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.always_auto_resize | ...`. The site is
  `ui.py:138-142` (the main window's `imgui.begin(flags=...)` call).
- **Modifier keys**: `imgui.get_io().key_ctrl`, `io.key_alt`, `io.key_shift` — same `IO`
  attribute surface, unchanged.
- **Font loading**: `fonts.add_font_from_file_ttf(path, size_pixels=..., glyph_ranges=...)` +
  `imgui_renderer.refresh_font_texture()` — identical API. `App.get_font()` migrates verbatim.

**Source**: imgui-bundle pyi stub
(https://raw.githubusercontent.com/pthom/imgui_bundle/main/bindings/imgui_bundle/imgui/__init__.pyi)
grep'd directly for `WindowFlags_` — confirms `class WindowFlags_(enum.IntFlag)` with snake_case
members; NO module-level `WINDOW_*` aliases. (Spike round 1 missed this; pre-impl review
caught it.)

### 8. Strip the imgui-related `# type: ignore` markers; fix underlying types
**Scope**: 20 `# type: ignore` markers currently in the codebase, but only **8 are imgui-
related** (pyimgui stub gaps). This feature strips those 8 (or fewer, after migration). The
other 12 are tied to moderngl / freetype / pydantic / internal polymorphism and are NOT in
scope of this feature — leave them alone.

The 8 imgui-related markers to strip:
- `shaderbox/ui_models.py:103,105` — `imgui.get_text_line_height_with_spacing()`
- `shaderbox/notifications.py:41` — `imgui.set_cursor_pos((x, current_y))`
- `shaderbox/ui.py:178` — `imgui.set_cursor_screen_pos((cursor_pos[0], cursor_pos[1] + ...))`
- `shaderbox/ui.py:234,238,242` — `imgui.begin_tab_item("X").selected` (handled by Decision 5
  context-manager idiom adoption — proxy access disappears entirely)
- `shaderbox/widgets/uniform.py:108` — `imgui.text(...)` context

For each: strip the ignore. If pyright surfaces an error, **prefer fixing the underlying type
issue** (better annotation, narrowed cast, structural refactor) over re-adding the ignore.
Only re-add if the underlying issue genuinely cannot be fixed (e.g. imgui-bundle stub bug
with no workaround) — and if a re-add is needed, that's a finding worth flagging in the
worklog.

**Escape clause**: if stripping an ignore forces a circular import (per the dev_flow
"cycle-from-types" signal), keep the ignore. Circular imports are a worse outcome.

**End-state target**: ≤2 imgui-related `# type: ignore` remaining (down from 8). The
`conventions.md ## Known quirks` "pyimgui stubs incomplete" bullet is rewritten or deleted
based on the actual residual count. The `CLAUDE.md ## Hard rules` "imgui.* exception" line is
also updated.

**Non-goal**: this feature is NOT a general type-debt sweep. The 12 non-imgui ignores stay
untouched. If post-migration audit suggests fixing them is cheap, file a separate
`[DEFERRAL]` with a concrete trigger.

### 9. Single feature branch, single coherent landing
Per locked staging decision: work on `feature/imgui-bundle-migration` (branched from master).
**Rationale for the branch override** (default repo rule is "work on current branch"): this
is a high-blast-radius refactor across 15 files touching every UI module; isolating it on a
dedicated branch keeps master runnable for the multi-day duration. Per the repo's "branch
only if the user explicitly asks" rule — the user explicitly asked.

Sub-commits during implementation are organized by logical group (boot → frame loop → flags &
inputs → widgets → popups → tabs → file dialogs → cleanup), each keeping `make smoke` green
where possible.

**Merge strategy**: at completion, **squash-merge** the branch to a single commit on master
("feature 004: migrate to imgui-bundle"). Sub-commits are implementation detail; the feature
is one atomic landing. If master accrues commits during the migration, **rebase** the feature
branch on master (`git rebase origin/master`) before final merge — handles conflicts naturally.

## Files touched

**Code:**
- `shaderbox/app.py` — boot (imports + `__init__` body), `get_font` font-loading.
- `shaderbox/ui.py` — frame loop (`update_and_draw`), `with imgui_ctx.begin` for main window
  and `_draw_node_settings` tab bar / tab items. Drop `contextlib.suppress(GLError)`. Drop
  `from OpenGL.GL import GLError` import.
- `shaderbox/hotkeys.py` — `imgui.get_io()` access (unchanged shape but re-verify io.key_ctrl
  semantics work via imgui-bundle's GLFW backend; should be no-change per spike).
- `shaderbox/widgets/details.py` — drag-int/drag-float kwarg renames; `save_file` swap to pfd.
- `shaderbox/widgets/media_ops.py` — drag-int/drag-float kwarg renames in `draw_video_filters`.
- `shaderbox/widgets/node_grid.py` — `with imgui_ctx.begin_child` migration.
- `shaderbox/widgets/uniform.py` — drag-int/drag-float kwarg renames; `color_edit3` unpack-arg
  fix; `crossfiledialog.open_file` → `pfd.open_file`. ~70+ call sites here (largest widget).
- `shaderbox/popups/node_creator.py` — `with imgui_ctx.begin_popup_modal` migration (drop
  `.opened` attribute access).
- `shaderbox/popups/settings.py` — `with imgui_ctx.begin_popup_modal` + `with
  imgui_ctx.begin_tooltip` migration; drag-int kwarg rename.
- `shaderbox/tabs/node.py` — `with imgui_ctx.begin_tab_bar` and `with imgui_ctx.begin_tab_item`
  migration (drop `.selected`); drag-int/drag-float renames.
- `shaderbox/tabs/render.py` — kwarg renames if any drag calls present.
- `shaderbox/tabs/share.py` — kwarg renames if any drag calls present.
- `shaderbox/notifications.py` — verify `imgui.set_cursor_pos` and `imgui.text_colored` calls
  are signature-compatible (per spike, yes).
- `shaderbox/ui_models.py` — `imgui.get_text_line_height_with_spacing` usage (per spike,
  signature-compatible; the two `# type: ignore` markers on lines 103,105 strip per Decision 8).
- `shaderbox/exporters/telegram.py` — imports `imgui` directly. Audit ALL imgui calls during
  impl (input_text fields, button, text_colored with unpacked-tuple color arg pattern that may
  need the same fix as `color_edit3` per Decision 4). Pre-impl reviewer 2 flagged this as a
  hidden surface; spec gives it explicit scope here.
- `scripts/smoke.py` — verify `glfw.window_hint(VISIBLE, FALSE)` still works before `App(...)`;
  expected no-change per spike.

**Out of files-touched (audited, confirmed no imgui calls):**
- `shaderbox/fonts.py` — freetype glyph-atlas for the in-shader text-rendering shader, NOT
  imgui font. Zero imgui imports. (Originally listed in the spec draft; reviewer 1 caught this.)
- `shaderbox/ui_utils.py` — no `imgui.*` calls (one moderngl-related `# type: ignore` here is
  NOT in scope per Decision 8).

**Config:**
- `pyproject.toml` — `uv remove imgui crossfiledialog` then `uv add imgui-bundle`. Verify
  Python 3.12 compatibility (imgui-bundle supports 3.10+).
- `uv.lock` — regenerates.

**Docs:**
- `CLAUDE.md` — the `# type: ignore` exception line (~line 44-46): rewrite from "pyimgui's
  stubs genuinely lack symbols" to reflect imgui-bundle's stub coverage. Possibly delete the
  exception entirely if post-migration audit shows zero residual suppressions.
- `ai_docs/conventions.md ## Known quirks` — "pyimgui's stubs are incomplete" bullet: rewrite
  or delete based on post-migration audit.
- `ai_docs/dev_flow.md ## Recipes` — module map: no structural change (all files keep same
  layout). Update prose if it references pyimgui-specific behavior.
- `ai_docs/todo.md` — no current deferrals affected; potentially file new ones for the
  deferred enhancements (imgui-knobs, markdown error rendering) with concrete triggers.
- `CLAUDE.md` — line 3 stack sentence: `**moderngl + glfw + pyimgui**` → `**moderngl + glfw +
  imgui-bundle**`. (Reviewer 1 caught: the original spec draft incorrectly pointed at README.md;
  README.md line 18 is unrelated content.)
- `README.md` — audit pass: confirm no other pyimgui/imgui references. Update if found.

## Manual verification

Full UX sweep (the diff touches every UI surface). Required, not skippable.

### Pre-impl validation (BEFORE writing migration code)

0a. **Capture smoke baseline**: on master (current pyimgui), run
   `make smoke 2>&1 | tee /tmp/smoke_baseline_pyimgui.log`. Confirm exit 0. Keep the log
   file — post-impl smoke output gets diffed against it to identify NEW errors (vs. errors
   that were always there but suppressed). **If `/tmp/smoke_baseline_pyimgui.log` is missing
   when step 2 runs (machine reboot, /tmp cleanup), re-capture it from master via
   `git stash && git checkout master && make smoke 2>&1 | tee /tmp/smoke_baseline_pyimgui.log
   && git checkout - && git stash pop`** before proceeding.

0b. **getattr pattern compatibility**: `widgets/uniform.py` uses
   `getattr(imgui, f"color_edit{N}")` and `getattr(imgui, f"drag_float{N}")` patterns. In the
   imgui-bundle pyi stub (https://raw.githubusercontent.com/pthom/imgui_bundle/main/bindings/imgui_bundle/imgui/__init__.pyi),
   confirm that `color_edit3`, `color_edit4`, `drag_float2`, `drag_float3`, `drag_float4`
   exist as module-level `def`s. If yes, getattr works unchanged. If they're nested somewhere
   else, impl must replace with explicit if/elif dispatch.

0c. **pfd filter syntax sanity-check**: write a 5-line standalone Python script:
   ```python
   from imgui_bundle import portable_file_dialogs as pfd
   result = pfd.open_file("test", "", ["Image", "*.png *.jpg"]).result()
   print(result)
   ```
   Run it. Confirm the dialog shows on Linux and the "Image" filter shows `*.png *.jpg`.
   pyimgui+crossfiledialog equivalent for reference:
   `crossfiledialog.open_file(title="test", filter=["*.png", "*.jpg"])`. If pfd's filter
   format is wrong, Decision 6 needs revision before impl touches `widgets/uniform.py`.

### Post-impl verification

1. **`make check`** — pyright/ruff clean. Per Decision 8, expect 8 imgui-related ignores
   stripped; for each pyright complaint, fix the underlying type rather than re-adding the
   ignore. Target end-state: ≤2 imgui-related suppressions (down from 8); 12 non-imgui
   markers untouched.

2. **`make smoke` + GLError diagnostic** (this is the gating step — if it fails, halt and
   diagnose before continuing).
   - Run `make smoke 2>&1 | tee /tmp/smoke_after_imgui_bundle.log`. Expect exit 0; 200
     frames; no exceptions.
   - `diff /tmp/smoke_baseline_pyimgui.log /tmp/smoke_after_imgui_bundle.log` to confirm no
     new error categories beyond the expected ModelBox-connection-refused line.
   - If smoke fails with a GLError (now that `contextlib.suppress(GLError)` is gone per
     Decision 2): (a) real ShaderBox moderngl bug → fix it; (b) real imgui-bundle renderer
     bug → diagnose against opengl_backend_programmable.py source, work around or pin to a
     known-good version; (c) benign edge case → document in worklog. **DO NOT re-add
     `suppress(GLError)` to silence it.** Re-adding is an automatic FAIL on this verification.

3. **Boot**: `uv run python ./shaderbox/ui.py` starts cleanly. App window appears. No console
   warnings about imgui-bundle deprecations.

4. **Settings popup** (Alt+S): opens, FPS drag works, text-editor input works, tooltip on `?`
   renders, Close button closes. Mutex with Node Creator still works.

5. **Node Creator popup** (Ctrl+N): opens, template grid renders, arrows navigate, Enter
   creates a node, Esc cancels.

6. **Hotkeys**: Ctrl+O, Ctrl+S, Ctrl+E, Ctrl+D, Ctrl+N, Alt+S, Esc, Left/Right arrows — every
   one fires its action exactly once per press (no double-fire, no missed presses).

7. **Texture uniform editor** (the largest UI surface, `widgets/uniform.py`): pick a node with
   image/video textures. Verify:
   - Image preview renders.
   - Video preview renders.
   - "Load" file dialog opens (now via `portable_file_dialogs`), file picker shows filter.
   - Drag-int/drag-float widgets for non-texture uniform types render and respond.
   - Color picker (color_edit3/4 via `getattr`) works with the new pass-list signature.
   - Array/text/buffer/auto input types still render.
   - Video filters (Smoothing / Window / Sigma / Apply##video_to_video_smoothing) work.

8. **Tab bar in node settings**: Node / Render / Share tabs select correctly via the new
   context-manager idiom.

9. **Telegram exporter UI** — if Telegram is configured (bot token / user id / sticker set
   name in settings), open Share tab → Telegram panel. Verify: (a) `input_text` fields for
   Bot token / User ID / Sticker set name render and accept input; (b) `text_colored` status
   display renders correct colors — auth state, error messages, info text (6 call sites in
   `exporters/telegram.py:208,210,213,227,275,292` migrated per Decision 4); (c) authenticate
   button doesn't crash; (d) if a sticker export succeeds, worker-thread + asyncio flow
   completes.

   **If Telegram is NOT configured**: skip the Telegram-specific subchecks; document in the
   manual-verification worklog entry that "Telegram exporter UI was not exercised in this
   verification — text_colored coverage validated via tabs/share.py + notifications.py
   instead." To trigger a notification for the `notifications.py:42` spot-check: open
   Settings (Alt+S), set Text editor cmd to `invalid-cmd-xyz`, close, then press Ctrl+E —
   the failed-editor-launch path calls `app.notifications.push(err, color=(1, 0, 0))` in
   `app.py:edit_current_node_fs_file` (verified via grep). Confirm the red toast renders.

10. **Project open** (Ctrl+O): file dialog opens (now pfd), folder selection loads a project.

11. **Node grid sidebar**: previews render, click selects a node.

12. **Project switching with state recovery**:
    a. In current project, open Settings (Alt+S), change **Global target FPS** from default
       (e.g. 60 → 45). Ctrl+S to save. Note the value.
    b. Ctrl+O → pick a DIFFERENT project folder (`projects/` has no second project by
       default — create a sibling folder `projects/scratch/` with an empty `nodes/`
       subfolder for this test, or pick any other folder). Verify the app loads with default
       state.
    c. Ctrl+O → return to the original project folder. Verify the FPS slider reads 45 (the
       value from step a). Verifies `UIAppState` round-trips correctly across the migration.
    d. If Telegram is configured: open Settings → Share tab → Telegram panel, verify token /
       sticker-set fields are restored from `app_state.json` (auth state resets per existing
       behavior, but config fields persist).

13. **Save/restart cycle**: Ctrl+S saves, exit, restart app, project loads with same state.

## Open questions for the user

1. **`hello_imgui` shortcut helpers** (theme presets, asset folder helper, log window, etc.):
   **LOCKED — no hello_imgui adoption in this feature.** The maintainer has a planned UI/UX
   refactor with custom themes coming next; `hello_imgui.apply_theme()` adoption will land
   there with a concrete theme design in hand. Adopting now would be premature scaffolding.

   **Sanitize-time action**: file a `[DEFERRAL] adopt hello_imgui theme utilities` entry in
   `todo.md` with trigger "when starting the UI/UX refactor with custom themes". imgui-bundle
   ships `hello_imgui.apply_theme(ImGuiTheme_.<name>)` (~15 named themes) and
   `hello_imgui.show_theme_tweak_gui` for live theme editing. Adoption cost at that point is
   ~5 LOC.

2. **`# type: ignore` audit aggressiveness**: **LOCKED — most aggressive form.** Strip every
   `# type: ignore` from the codebase during migration. Run `make check`. For each pyright
   complaint that surfaces, **prefer fixing the underlying type issue** (better type
   annotation, narrowed cast, structural refactor) over re-adding the ignore. Only re-add an
   ignore if the underlying issue genuinely cannot be fixed (e.g. third-party stub bug with
   no workaround). Goal: end-state has zero or near-zero `# type: ignore` markers in the
   codebase. The `conventions.md ## Known quirks` "pyimgui stubs incomplete" bullet likely
   becomes obsolete entirely.

3. **Sub-commit granularity on the feature branch**: **LOCKED — logical groups** (~6 commits:
   boot, kwarg renames, color_edit, context managers, pfd swap, doc updates). Each commit aims
   to keep `make smoke` green where possible. Smaller decision; the maintainer is indifferent.

## Review history

Drafted 2026-05-16 after two spike rounds against official imgui-bundle docs and source.

**Spike round 1** (call inventory + integration layer): resolved 5 UNCONFIRMED items against
the imgui-bundle `.pyi` stub, `python_backends/` source, official demos. All 5 resolved with
citations. Findings folded into Design decisions 1-7.

**Spike round 2** (bundled addons audit): evaluated `portable_file_dialogs`, `imgui_md`,
`implot`, `imspinner`, `imgui-knobs`, toggles, node-editor, color-text-edit, immvision,
im-cool-bar. Honest result: only `portable_file_dialogs` justifies adoption now; others either
have no current need or are post-migration enhancement track. Folded into Design decision 6 and
Out-of-scope section.

**Pre-impl review** (2026-05-16, 2 reviewers in parallel — correctness/design +
verification/blast-radius). Both verdicts: SHIP-WITH-EDITS. Key catches applied inline:

- **BLOCKER (R1)**: WindowFlags claim in Decision 7 was wrong. Direct grep against the
  imgui-bundle pyi confirmed `class WindowFlags_(enum.IntFlag)` with snake_case members; no
  module-level `WINDOW_*` aliases exist. Spike round 1 had guessed pattern-based; reviewer
  caught it. Decision 7 rewritten to reflect the breaking enum form. Lesson: the new
  `CLAUDE.md ## Hard rules` "library docs as source of truth" rule did its job exactly as
  intended — surfaced a wrong claim before impl.
- **BLOCKER (R1+R2)**: Files touched had `fonts.py` listed (no imgui imports) and was missing
  `exporters/telegram.py` (imports imgui directly with input_text + text_colored calls that
  may match Decision 4's pass-list pattern). Fixed. Verified: 15 files import imgui across
  `shaderbox/` (matches reviewer claim).
- **BLOCKER (R1+R2)**: Decision 8's "strip every # type: ignore" wording over-scoped. Direct
  grep found 20 markers total, only 8 imgui-related. Decision 8 narrowed to the 8 imgui
  markers (listed explicitly); other 12 (moderngl/freetype/pydantic) stay untouched. Escape
  clause added for circular-import edge cases.
- **MAJOR (R1)**: Doc reference was wrong: CLAUDE.md line 3 holds the stack sentence, not
  README.md line 3. Files-touched corrected.
- **MAJOR (R2)**: GLError-suppress removal could surface latent GL state collisions in
  `make smoke`. Added pre-impl baseline capture (step 0a) + post-impl diagnostic procedure
  (step 2a) with three resolution paths.
- **MAJOR (R2)**: Manual verification list missed exporter UI + project-switching with state
  recovery. Added explicit steps 9 (Telegram exporter UI) and 12 (project switching with
  state recovery).
- **MAJOR (R2)**: `getattr(imgui, f"color_edit{N}")` pattern in `widgets/uniform.py` not
  spike-verified. Added pre-impl validation step 0b.
- **MINOR (R1)**: Branch override (Decision 9) had no rationale. Added rationale sentence.
- **MINOR (R2)**: Merge strategy not specified. Added: squash-merge to master; rebase the
  branch on master on conflict.
- **MINOR (R2)**: pfd filter syntax untested. Added pre-impl validation step 0c.

No findings warranted DON'T-SHIP after edits applied. Both reviewers verified the 7
spike-cited URLs against the actual sources; 6/7 confirmed, 1 (WindowFlags) refuted —
spec corrected.

**Pre-impl review round 2 / convergence loop** (2026-05-16, per the new global "Review-agent
discipline" + "convergence loop" rules). Verdict-per-finding:
- Correctness reviewer (round 2): all 5 round-1 fixes PASS. 3 MINOR new findings, no
  blockers.
- Verification reviewer (round 2): all 7 round-1 fixes PASS. Adversarial sweep found 1
  BLOCKER + 3 MAJOR new findings.

**Convergence BLOCKER**: `imgui.text_colored` signature was MISSED by spike round 1 and
round 1's pre-impl review. Direct curl + grep against the imgui-bundle pyi revealed:
`def text_colored(col: ImVec4Like, fmt: str) -> None` — argument order swapped AND color
shape changed (pass `ImVec4Like` tuple, not unpacked components). 19 call sites across 10
files affected (popups/settings.py, widgets/node_grid.py, widgets/uniform.py,
widgets/details.py, tabs/render.py, tabs/share.py, tabs/node.py, notifications.py,
exporters/telegram.py). Decision 4 expanded to cover `text_colored` alongside `color_edit3/4`
with full call-site list and rewrite recipe.

**Convergence MAJORs applied**:
- Smoke baseline + diff (steps 0a + 2): explicit log-file paths so post-impl diff is real.
- Pre-impl validation 0c (pfd filter sanity-check): concrete runnable Python snippet, not
  prose.
- Step ordering: smoke + GLError diagnostic (step 2) now BEFORE boot (step 3) — failing
  smoke halts before further verification.
- Step 9 (Telegram UI): explicit fallback procedure when Telegram is unconfigured (spot-
  check `notifications.py` instead).
- Step 12 (project switching): concrete FPS canary value (60 → 45) so the test passes only
  for the right reason.

**Convergence outcome**: spec patched with the BLOCKER fix + 4 MAJOR refinements. Both
reviewers' findings captured to `/tmp/review_004_iter2_correctness.md` +
`/tmp/review_004_iter2_verification.md`. Next round (iter3) should converge to PASS/PASS
once it verifies the `text_colored` Decision 4 expansion and the rebase-on-conflict retest
note (round-2 MINOR).

**Pre-impl review round 3 / convergence iter3** (2026-05-16, single focused verification
reviewer). Verdict-per-iter2-patch:
- BLOCKER from iter2 (text_colored Decision 4 expansion): PASS, except spec said "10 files"
  while enumerated list and codebase grep show 9. Off-by-one corrected to "9 files".
- MAJOR (smoke baseline path): PASS.
- MAJOR (pfd runnable snippet): PASS.
- MAJOR (step ordering): PASS.
- MAJOR (step 9 fallback): PARTIAL → tightened with concrete trigger (Ctrl+E with invalid
  editor cmd triggers `notifications.push` red toast; grep-verified at `app.py:223`).
- MAJOR (step 12 canary): PASS.

**Iter3 adversarial sweep** caught 2 additional issues, both fixed:
- Stored 3-tuple `*color` unpack pattern (`notifications.py:42`, `widgets/details.py:42,48`,
  `exporters/telegram.py:208,210,213,292`): Decision 4 expanded with "Stored-tuple rewrite"
  case — rewrite at call site only via `(*color, 1.0)`, keep storage as 3-component (no
  Notifications class change). Honors "only change what's asked".
- Ephemeral `/tmp/smoke_baseline_pyimgui.log`: step 0a got a recovery procedure (re-capture
  from master via stash + checkout dance) if the file is missing when step 2 runs.

**Convergence verdict**: PASS. Loop closes. Spec ready to implement. Round 2 + 3 captured to
`/tmp/review_004_iter2_correctness.md`, `/tmp/review_004_iter2_verification.md`,
`/tmp/review_004_iter3.md`. Three rounds of review surfaced 1 BLOCKER spike-missed
(`text_colored` arg-swap), 1 BLOCKER spike-missed (WindowFlags enum form), 3 MAJOR
verification gaps, and 5 MINOR clarifications. None blocked plan-lock; all patched in spec.

**Post-impl review**: to be filled.
