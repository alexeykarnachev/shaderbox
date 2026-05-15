# 001 — Exporter refactor

Status: **draft, plan-locked pending user sign-off**
Shape: high-blast-radius feature flow (`ai_docs/dev_flow.md ## Feature flow`).

---

## Goal

Replace the speculative `ShareProvider` / `ShareableMedia` / `ShareManager` layer in
`shaderbox/sharing.py` + `shaderbox/telegram_provider.py` with an `Exporter` abstraction whose
contract is honest about the real shape of "export rendered media to a target" — designed against
the contracts of YouTube Data API v3 and X API v2 (the planned future exporters), then validated
by porting the existing Telegram sticker exporter onto it.

**Only one concrete exporter ships in this PR (Telegram).** The abstractions must be **complete**
(every contract a future exporter will need — auth state machine, prepare/export split, progress
stream, credential storage, can_export) and **not exhaustive** (no resumable-session persistence,
no rate-limit DB, no shared list-of-existing-items widget — those land in a future PR when the
second concrete exporter forces them).

Resolves:
- `todo.md [DEFERRAL] two near-identical sticker models — also gates re-tightening pyright`
- `todo.md [DEFERRAL] blocking asyncio / blocking HTTP in the render loop` (folded in via the
  per-exporter worker-thread + mailbox pattern)
- Drops `|| true` from the pyright pre-commit hook (re-tightens `make check`).

Does NOT resolve:
- `todo.md [DEFERRAL] split ui.py (1778-line god-class)` — but lands the **first real `tabs/*.py`
  extraction** (`tabs/share.py`), so partially advances it.

## Out of scope

- **YouTube exporter implementation.** Spec is designed against YouTube's contract (resumable
  upload, OAuth2 loopback, chunked progress, transform-then-export) but the concrete `youtube.py`
  is a separate feature. *Trigger to land it: when the user starts implementing YouTube export.*
- **X / Twitter exporter.** Same. *Trigger: when the user starts implementing X export.*
- **Resumable upload session persistence across app restarts.** `ExportProgress.resume_token: bytes
  | None` exists in the protocol but Telegram doesn't use it; no sidecar storage implemented.
  *Trigger: when YouTube exporter lands.*
- **Persistent per-exporter rate-limit / retry state.** Same. *Trigger: when YouTube or X exporter
  lands.*
- **Shared "browse existing items" widget across exporters.** Explicit non-goal — each exporter
  owns its own panel; if a future common shape emerges between two real exporters, it gets
  extracted at that point.
- **Localization** of exporter-facing strings.
- **`ui.py` god-class split** beyond extracting `tabs/share.py`.

## Design decisions (locked)

1. **`Exporter` is an ABC, not a `typing.Protocol`.** Exporters are stateful (worker thread,
   mailbox queue, auth state machine, init/teardown lifecycle); nominal typing is the right fit.
   Matches existing precedent (`ShareProvider`, `MediaWithTexture`). *Why locked:* architecture
   reviewer flagged Protocol shines for value-shaped duck typing; this isn't that.

2. **`RenderedArtifact` is a pure value type — NO GL handles.** Fields: `path: Path`, `is_video:
   bool`, `duration: float`, `size: tuple[int, int]`, optional `preview_bytes: bytes | None`
   (CPU-side PNG/JPEG bytes for thumbnail, not a `moderngl.Texture`). The share-tab UI owns one
   render-thread `Canvas` and renders the live preview into it (mirroring `ui.py:1555`'s node
   preview). *Why locked:* GL contexts are thread-bound — passing a `moderngl.Texture` to a
   worker-thread exporter is a latent UAF. The original `selected_media.preview_canvas` per-item
   pattern is the bug-shape that motivates this rule.

3. **`Exporter.export()` is NOT one-shot — it returns a stream of `ExportProgress` events via a
   per-exporter `queue.Queue[ExportProgress]` (the "mailbox").** The exporter's worker thread
   pushes progress (`uploading 40%`, `processing`, `published url=...`); the imgui frame reads the
   mailbox snapshot once at the top of `draw_target_panel`. *Why locked:* YouTube has a
   post-upload "processing" wait; X has media-then-tweet; Telegram has add-then-set-thumb.
   Single-bool return forces every exporter to hide multi-step state. Also folds in the blocking-
   asyncio deferral cleanly.

4. **Two verbs, not one: `prepare()` THEN `export()`.**
   - `prepare(artifact: RenderedArtifact, settings: dict) -> RenderedArtifact`: may re-encode via
     ffmpeg to satisfy the target's format constraints (Telegram: ≤256KB webm/VP9 ≤3s ≤512px ≤30fps;
     YouTube: H.264 mp4; X: H.264 mp4 with duration cap). Synchronous, called on the worker thread
     before `export()`. Returns a new artifact pointing at the prepared file.
   - `export(artifact: RenderedArtifact, settings: dict) -> None`: pushes progress to the mailbox;
     terminal events are `ExportProgress.done` (with optional `url` or error). May internally be
     N API calls (chunked upload, media-then-tweet, etc.).
   - `can_export(artifact: RenderedArtifact) -> tuple[bool, str]`: pre-flight check on the
     **prepared** shape (i.e. called after `prepare()` to confirm constraints are met, or used to
     short-circuit before `prepare()` when the artifact is obviously incompatible).

   *Why locked:* reject-only `can_export` forces the user to re-render manually. Every exporter
   needs an ffmpeg transform step; making it implicit in `export()` hides where the slow step is.

5. **Auth is a first-class sub-protocol, not a `draw_config_ui` field.** Methods:
   - `auth_state() -> AuthState` (enum: `unconfigured / configured / authenticating / authed /
     error`)
   - `begin_auth() -> None` (exporter spawns its own loopback HTTP listener + `webbrowser.open(...)`
     for OAuth2; or just validates a bot token for Telegram). Non-blocking; updates `auth_state()`
     via mailbox.
   - `draw_config_ui(settings: dict) -> dict` draws non-secret config (sticker set name, video
     privacy default, etc.) and an "Authenticate" button that calls `begin_auth()`.

   *Why locked:* OAuth2 loopback is a multi-second browser round-trip; can't live inside one
   synchronous imgui frame. Telegram's bot-token-in-a-textbox is the easy case and the original
   design was misled by it.

6. **Exporter instances live for the app's lifetime, owned by `App`.** Created in
   `App.__init__`, settings rebound on project load (the old `set_active_provider`'s `media_list.clear()`
   becomes a no-op — there's no manager-owned list). Worker thread starts lazily on first export
   call and stops in `App.release()`. *Why locked:* per-project teardown of an authed session
   is wrong UX (re-auth every project switch); the project-bound bit is the per-exporter settings
   in `UIAppState`, which `load_and_migrate` already handles.

7. **Credentials stay where they live today — in `UIAppState` (the project's `app_state.json`).**
   Telegram bot token continues to live in `app_state.json`; no keyring, no new storage layer.
   *Why locked:* the JSON isn't checked in (`projects/` is gitignored), the user has been running
   it this way, and adding `keyring` would introduce a new dep + new failure modes (no daemon on
   headless Linux) for a problem this PR doesn't have. *Revisit trigger:* when an exporter brings
   credentials that actually demand secure storage (OAuth refresh tokens — i.e. YouTube/X) —
   that PR introduces keyring or equivalent.

8. **Each exporter owns its own per-target panel UI — no shared "browse existing items"
   widget.** YouTube's "my uploads", X's "my posts", and Telegram's "current sticker set" have
   superficially similar list+delete shapes but the schemas, pagination, and edit verbs differ
   enough that lifting them into a shared widget is a trap. Each exporter's `draw_target_panel`
   renders whatever imgui it wants. *Why locked:* domain reviewer confirmed all three diverge.

9. **Threading invariant: the worker thread MUST NOT call `moderngl.*`.** No `Texture`, `Canvas`,
   `Node`, `Image`, `Video` construction or access off the render thread. Enforced by design (the
   protocol's worker-side methods take `RenderedArtifact` which is GL-free); documented in
   `conventions.md ## Known quirks`. *Why locked:* `conventions.md ## Known quirks` already notes
   "a live moderngl context must exist before constructing Image/Video/Font/Canvas/Node — they
   call moderngl.get_context() lazily."

10. **Module layout:**
    - `shaderbox/exporters/base.py` — `Exporter` ABC, `RenderedArtifact`, `ExportProgress`,
      `AuthState`, `ExporterStatus` (the mailbox snapshot type).
    - `shaderbox/exporters/registry.py` — `ExporterRegistry` (replaces `ShareManager`). Owns
      exporter instances dict + active id. No `media_list` / `selected_media_index` / `refresh_media`.
    - `shaderbox/exporters/telegram.py` — `TelegramExporter(Exporter)`. Owns its sticker-grid
      panel internally (today's `draw_share_tab` Telegram-specific blocks move here).
    - `shaderbox/tabs/share.py` — the share-tab orchestration (exporter dropdown, dispatches
      `exporter.draw_config_ui()` / `exporter.draw_target_panel()`, owns the one preview
      `Canvas`). First real `tabs/*.py` module — sets the pattern.
    - `shaderbox/ui.py` — `draw_share_tab` deleted; `App` instantiates `ExporterRegistry` and
      `tabs.share.draw(app_state, registry, current_artifact)` is called from the tab dispatch.
      `App.release()` calls `registry.release()` (which stops worker threads).
    - `shaderbox/sharing.py` — **deleted.**
    - `shaderbox/telegram_provider.py` — **deleted** (logic moves to `exporters/telegram.py`).

11. **`RenderedArtifact` is exporter-domain, NOT a media primitive.** Lives in
    `exporters/base.py`, not `media.py`. `media.py` stays GL-bound (`Image`, `Video`, `MediaWithTexture`).
    *Why locked:* media primitives need a live GL context; artifacts cross thread boundaries.
    They're different abstraction layers.

12. **`UIMessage` (`ui_models.py:19`) is not deleted — it stays as a generic per-exporter log
    primitive** consumed by `ExporterStatus.message`. *Why locked:* it's not coupled to sharing;
    other places use it.

## Files touched

**Created:**
- `shaderbox/exporters/__init__.py`
- `shaderbox/exporters/base.py` — ~120 lines (ABC, dataclasses)
- `shaderbox/exporters/registry.py` — ~60 lines
- `shaderbox/exporters/telegram.py` — ~250 lines (port + sticker-grid panel + auth + prepare/export)
- `shaderbox/tabs/__init__.py`
- `shaderbox/tabs/share.py` — ~150 lines (orchestration + preview canvas)

**Modified:**
- `shaderbox/ui.py` — delete `draw_share_tab` (~280 lines); replace `_share_manager` with
  `_exporter_registry`; thread `current_artifact` into the share-tab call site; `release()`
  updated.
- `shaderbox/ui_models.py` — `UIAppState`: rename `share_provider_configs` →
  `exporter_settings` (keep migration shim); `active_share_provider` → `active_exporter_id`.
  Telegram bot token stays in `exporter_settings["telegram"]["bot_token"]` (same shape as today).
- `pyproject.toml` — no new runtime deps (keyring is NOT added; `google-api-python-client` NOT
  added — YouTube is out of scope). `python-telegram-bot` stays (see Open questions).
- `.pre-commit-config.yaml` — drop `|| true` from pyright hook.
- `CLAUDE.md` — no structural change expected.
- `ai_docs/conventions.md` — `## Known quirks`: update "ui.py has pre-existing type debt" entry
  (the share-tab hasattr-dispatch reason goes away); `## Design decisions`: add "Exporters: own
  thread, own panel, GL-free artifacts" bullet with revisit trigger.
- `ai_docs/dev_flow.md ## Recipes` module map: replace `sharing.py` + `telegram_provider.py`
  bullets with `exporters/` dir bullet; mention `tabs/share.py` as the first tabs module.
- `ai_docs/todo.md` — delete two resolved deferrals (sticker models, blocking asyncio); update
  `[DEFERRAL] split ui.py` to note tabs/share.py landed as the first extraction.

**Deleted:**
- `shaderbox/sharing.py`
- `shaderbox/telegram_provider.py`

## Manual verification

Telegram is the only exporter; verification requires bot token + user id + sticker set name
(maintainer-owned secrets — `dev_flow.md ## Recipes` confirms this can't be exercised without
configuration). Maintainer runs:

1. **Pre-condition:** existing project with `share_provider_configs.telegram.bot_token` set in
   `app_state.json`.
2. **Migration:** launch app post-upgrade. Expected: `share_provider_configs` key renamed to
   `exporter_settings`, `active_share_provider` to `active_exporter_id`; bot token stays in place;
   no UI regression.
3. **Auth:** Share tab → Telegram → "Authenticate" button → bot initializes; auth_state shows
   `authed`.
4. **Existing stickers:** sticker grid populates with current sticker set (move into Telegram
   panel — this is `TelegramExporter.draw_target_panel`, not host UI).
5. **Render → prepare → export (new sticker):** render a node, switch to share tab, "Add as
   sticker"; verify ffmpeg re-encode to ≤256KB webm/VP9 ≤3s ≤512px ≤30fps; verify upload
   completes; verify mailbox shows "preparing" → "uploading X%" → "done"; verify imgui frame
   stays responsive during upload (>30fps, no freezes — the blocking-asyncio fix).
6. **Replace sticker:** select existing sticker → re-render → "Replace"; verify replace flow.
7. **Delete sticker:** select existing → "Delete"; verify deletion.
8. **Project switch:** load a different project; verify exporter instance survives; verify the
   new project's `exporter_settings` is applied to the exporter (rebound, not torn down).
9. **`make check`:** clean run, pyright re-tightened (no `|| true`), zero errors.
10. **App shutdown:** verify no leaked worker threads (`App.release()` joins them).

A real UX gap found at any step is a FAIL, not pass-with-caveat.

## Resolved decisions (answered during plan-lock)

- **`python-telegram-bot`:** keep. Wrap async calls in a worker thread with its own asyncio loop.
- **Branch strategy:** stay on `master`, incremental commits.
- **Credentials storage:** stay in `app_state.json` (no keyring). See Design Decision 7.
- **Sticker grid live preview:** keep current behavior. When an existing sticker is selected in
  the share-tab grid, the current shader renders live into that sticker's preview slot
  (`ui.py:1559-1562` today — only the selected one, not all of them). Under the refactor, the
  Telegram exporter's `draw_target_panel` owns one preview `Canvas` and the render-thread driver
  in `tabs/share.py` calls `node.render(canvas=...)` into it each frame when the selected target
  has a preview slot.

## Open questions for the user

*(None — all resolved during plan-lock.)*

## Review history

*(Populated by review agents during the feature flow.)*
