# Feature 003 — ModelBox removal

## Goal

Remove all ModelBox integration from ShaderBox. The maintainer plans to replace it with
in-process PyTorch inference later; this feature wipes the slate (no compat shims, no
preserved API surface). After this lands, the codebase has no `modelbox.py` module, no
HTTP client for an external inference service, no `requests` dependency, no ModelBox UI in
the Settings popup, and no media-model dropdown on texture uniforms.

## Out of scope

- **In-process PyTorch inference design.** Future work; this spec doesn't anticipate its
  shape. **Trigger to revisit:** when starting the PyTorch-inference feature, re-read the
  pre-removal commit (`shaderbox/widgets/media_ops.py::draw_media_models`,
  `shaderbox/popups/settings.py` ModelBox block, `shaderbox/modelbox.py`) for prior-art
  UX cues — they're frozen in git history, not in the live tree.
- **`shaderbox/media.py`'s `Image` / `Video`.** Stay as-is. They are the in-tree media
  types and were only *used* by modelbox.py for typing; they aren't conceptually tied to
  ModelBox.
- **Historical feature specs (001/002) and the worklog entries that reference ModelBox.**
  Don't rewrite history; those documents are accurate snapshots of past decisions.

## Design decisions (locked)

1. **Clean delete, no shims.** `shaderbox/modelbox.py` is deleted outright; no
   re-export, no stub. The `[DEFERRAL] blocking HTTP in the render loop (ModelBox)` entry
   in `todo.md` is **deleted** (resolved by removal, not by unblocking).
2. **`requests` dependency dropped.** `modelbox.py` is the only `requests` user in the
   repo; remove via `uv remove requests`.
3. **`UIAppState` migration: silent pop of removed fields.** Existing `app_state.json`
   files in the wild contain `modelbox_url` and `media_model_idx`. Pydantic's
   `model_config = {"extra": "forbid"}` would reject the load. Extend
   `UIAppState.load_and_migrate` to `data.pop("modelbox_url", None)` and
   `data.pop("media_model_idx", None)` as a new (third) idempotent migration generation,
   placed before the final `cls(**data)` call (after the existing
   `share_provider_configs → exporter_settings` block). The fields are dropped from the
   model class definition in the same change. **Update the docstring**: "Two migration
   generations" → "Three migration generations", add line 3 ("Remove `modelbox_url` /
   `media_model_idx` — feature 003").

   **Post-migration save behavior (intentional, silent):** after `load_and_migrate`
   strips the fields, the next `App.save()` writes a clean `app_state.json` without
   them. No user-visible warning. ModelBox is gone; the fields shouldn't survive.
4. **`widgets/media_ops.py` keeps its name even though only `draw_video_filters` remains.**
   Rationale: future PyTorch inference probably re-introduces media-model operations and
   will live here again. Don't rename the file for one function then rename back later.
5. **No `media_ops.py` file deletion.** Per (4), keep the module shell — single function
   inside is acceptable; widgets are organizational only (per `conventions.md`).
6. **Settings popup section deleted, not hidden behind a flag.** The "ModelBox url" input,
   "Install from GitHub" button, and explanatory text block are removed wholesale.
7. **README + CLAUDE.md prose updated, not just the code refs.** ModelBox is a documented
   user-facing feature; user-facing docs lead, code lives downstream.
8. **Don't preserve `app.modelbox_info` field as dead state.** Delete from `App.__init__`
   along with the `fetch_modelbox_info` method and its `_init` call.

## Files touched

**Code (delete):**
- `shaderbox/modelbox.py` — entire file.

**Code (modify):**
- `shaderbox/app.py` — remove `from shaderbox import modelbox` import, the
  `self.modelbox_info: dict[str, Any] = {}` field assignment in `__init__`, the
  `self.fetch_modelbox_info()` call in `_init`, and the `fetch_modelbox_info` method
  entirely. Drop the now-unused `typing.Any` import only if nothing else uses it.
- `shaderbox/widgets/media_ops.py` — delete `draw_media_models` function. Imports to
  drop: `typing.TypeVar`, `shaderbox.modelbox`, `shaderbox.media.Image`, and the
  `T = TypeVar("T", Image, Video)` line. **Keep:** `shaderbox.media.Video` (used by
  `draw_video_filters`), `imgui`, `loguru`, `pathlib.Path`, `shaderbox.app.App`.
- `shaderbox/widgets/uniform.py` — replace
  `new_value = draw_media_models(app, current_value)  # type: ignore` (line ~75) with
  `new_value = current_value`. The surrounding video-filter chain stays:
  `if isinstance(new_value, Video): new_value = draw_video_filters(app, new_value)`
  (line ~76-77) remains active — only the media-model path is removed; temporal
  smoothing is preserved. Update the `from shaderbox.widgets.media_ops import
  draw_media_models, draw_video_filters` line to import only `draw_video_filters`.
- `shaderbox/popups/settings.py` — delete the ModelBox URL input block (lines **73-94**
  per reviewer-confirmed range: the "ModelBox url" input, "Install from GitHub" button,
  the explanatory tooltip text block, plus the trailing separator + spacing). Verify the
  exact range at impl time — surrounding FPS/editor sections must remain intact.
- `shaderbox/ui_models.py` — remove `modelbox_url: str = "http://localhost:8228/"` and
  `media_model_idx: int = 0` fields from `UIAppState`. Add a new migration block to
  `load_and_migrate` (after the existing `share_provider_configs` block):
  `data.pop("modelbox_url", None); data.pop("media_model_idx", None)`. Extend the
  docstring with a "(3)" generation entry.

**Config:**
- `pyproject.toml` — `uv remove requests` (or manually drop the entry under
  `[project.dependencies]`). `uv.lock` regenerates. Transitive deps (`urllib3`,
  `certifi`, `charset-normalizer`) are not referenced elsewhere in the repo — `uv sync`
  will clean them automatically.

**Docs:**
- `CLAUDE.md` — line 6 stack sentence: drop "optionally talks to an external 'ModelBox'
  HTTP service for AI image ops".
- `ai_docs/conventions.md` — delete the `modelbox.py imports Image / Video` known-quirks
  bullet (~line 80). In the "No `async` in the codebase" design decision (~line 61),
  **remove the entire parenthetical** "(Tracked: `todo.md [DEFERRAL] blocking HTTP in
  render loop` — narrowed to ModelBox after feature 001.)" — the deferral is resolved
  by this feature, and the design decision stands on its own without it.
- `ai_docs/todo.md` — DELETE `[DEFERRAL] blocking HTTP in the render loop (ModelBox)`
  entry entirely.
- `ai_docs/dev_flow.md` — module-map: remove the `modelbox.py` bullet. `### Run the app`
  recipe: remove "ModelBox is an optional external HTTP service" sentence. `### make smoke`
  recipe: remove the "noisy ModelBox connection errors when the optional service isn't
  running" parenthetical.
- `README.md` — line 18: remove "Integration with ModelBox for AI-powered image
  processing (depth maps, background removal)" bullet.

**Untouched (deliberately):**
- `ai_docs/features/001_exporter_refactor.md`, `ai_docs/features/002_ui_widgets_extraction.md`
- `ai_docs/worklog.md` historical entries (the new worklog entry covering this work goes
  on top per `/sanitize` step 5).
- `shaderbox/media.py` — `Image` / `Video` stay.

## Manual verification

1. `make check` — pyright/ruff clean.
2. `make smoke` — exit 0; 200 frames; output should no longer contain the
   "Failed to establish a new connection: [Errno 111] Connection refused" line during
   `App.__init__` (the call site is gone).
3. `uv run python shaderbox/ui.py`:
   - App starts.
   - Open Settings (Alt+S). The "ModelBox url" / "Install from GitHub" block is gone;
     the remaining settings (FPS, text editor cmd) render correctly.
   - Pick a texture uniform on a node (`projects/dev/` has one). The media-model
     dropdown ("Media model" + "Apply##media_model") is gone; for videos the
     "Smoothing" / "Window" / "Sigma" / "Apply##video_to_video_smoothing" section still
     renders.
   - Save (Ctrl+S), exit, restart, confirm app_state.json loads without error
     (proves migration works against the just-saved file).
4. **Spot-check: existing `app_state.json` with `modelbox_url` field still loads.**
   Add a temporary throwaway `modelbox_url: "test"` field to a copy of the dev
   `app_state.json`, load it via `App(project_dir=...)`, confirm no exception.
   (Can be skipped if the migration grep + read review covers it confidently.)

## Open questions for the user

None at draft time — the maintainer's instruction was unambiguous ("remove all of these
features completely"; pytorch inference is "later" and explicitly out of scope here). The
migration shape (silent pop) is the only judgment call and the spec locks it in as
decision (3) — happy to revisit if the user prefers an explicit "ModelBox config dropped"
notification on first load, but that's noise for a feature most users never configured.

## Review history

Drafted 2026-05-16 by main agent after user instruction.

**Pre-impl review (2026-05-16, 2 parallel reviewers — correctness/design + verification/
blast-radius).** Both verdicts: SHIP-WITH-EDITS. Convergent findings applied inline:
- (1) `load_and_migrate` docstring updated from "Two generations" → "Three", explicit
  placement of the new pop-block before `cls(**data)`.
- (2) `conventions.md` "narrowed to ModelBox" parenthetical removed entirely (deferral
  resolved by removal, not by unblock).
- (3) `widgets/media_ops.py` import-cleanup specified explicitly: drop `TypeVar`,
  `Image`, `shaderbox.modelbox`, the `T = TypeVar` line; keep `Video`.
- (4) `widgets/uniform.py:75` replacement specified to preserve the video-filter chain
  (`if isinstance(new_value, Video): draw_video_filters(...)` stays).
- (5) `settings.py` deletion range tightened to lines 73-94 (was ~73-90).
- (6) Pydantic post-migration save behavior documented (silent + intentional).
- (7) Transitive deps note added (`uv sync` cleans).
- (8) Manual verification step 2 made explicit about exit 0.

No findings warranted DON'T-SHIP or design changes. Both reviewers confirmed clean blast
radius — no hidden callers, `requests` is the only `requests` user, no PyTorch hints in
the current tree.

**Post-impl review:** to be filled.
