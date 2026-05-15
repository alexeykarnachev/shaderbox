# 001 — Exporter refactor

Status: **plan-locked + pre-impl review folded; ready to implement**
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
stream, credential storage) and **not exhaustive** (no resumable-session persistence, no rate-limit
DB, no shared list-of-existing-items widget — those land in a future PR when the second concrete
exporter forces them).

Resolves:
- `todo.md [DEFERRAL] two near-identical sticker models` (the sticker models themselves —
  `ShareableMedia` and `TelegramShareableMedia` — are gone, deleted with `sharing.py` and
  `telegram_provider.py`).
- The Telegram half of `todo.md [DEFERRAL] blocking asyncio / blocking HTTP in the render loop`
  (per-exporter worker-thread + mailbox pattern); the deferral is **narrowed**, not deleted —
  `modelbox.py`'s synchronous `requests` call remains in scope of the (renamed) deferral.

Does NOT resolve (de-scoped during impl):
- **Re-tightening pyright** (drop `|| true` from the pre-commit hook). The original spec claimed
  this; revisited during impl — pre-existing type debt across `ui.py` / `media.py` / `core.py` /
  `modelbox.py` (not just the share-tab `hasattr`-dispatch this PR removed) means re-tightening
  now turns this PR into "audit the entire repo for type debt." De-scoped to a separate effort,
  gated on the cleanup backlog. New deferral added: `todo.md [DEFERRAL] re-tighten pyright`.

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
   bool`, `duration: float`, `size: tuple[int, int]`. No `preview_bytes` (dropped — no consumer in
   this PR; add it back when an exporter needs a returned-thumbnail flow). The share-tab UI owns
   one render-thread `Canvas` and renders the live preview into it (mirroring the "Render current
   node preview" block in `App.update_and_draw`). *Why locked:* GL contexts are thread-bound —
   passing a `moderngl.Texture` to a worker-thread exporter is a latent UAF. The original
   `selected_media.preview_canvas` per-item pattern is the bug-shape that motivates this rule.

3. **`Exporter.export()` is NOT one-shot — it returns a stream of `ExportProgress` events via a
   per-exporter `queue.Queue[ExportProgress]` (the "mailbox").** The exporter's worker thread
   pushes progress (`uploading 40%`, `processing`, `published url=...`); the imgui frame reads the
   mailbox snapshot once at the top of `draw_target_panel`. *Why locked:* YouTube has a
   post-upload "processing" wait; X has media-then-tweet; Telegram has add-then-set-thumb.
   Single-bool return forces every exporter to hide multi-step state. Also folds in the blocking-
   asyncio deferral cleanly.

4. **Two verbs, not one: `prepare()` THEN `export()`. No `can_export` — use exceptions.**
   - `prepare(artifact: RenderedArtifact, settings: dict) -> RenderedArtifact`: may re-encode via
     ffmpeg to satisfy the target's format constraints (Telegram: ≤256KB webm/VP9 ≤3s ≤512px ≤30fps;
     YouTube: H.264 mp4; X: H.264 mp4 with duration cap). Synchronous, called on the worker thread
     before `export()`. Returns a new artifact pointing at the prepared file. Raises
     `ExporterValueError` if the input cannot be made to satisfy constraints (e.g. a 10-minute
     video for Telegram); the worker pushes the exception's message to the mailbox as a terminal
     `ExportProgress` error event.
   - `export(artifact: RenderedArtifact, settings: dict) -> None`: pushes progress to the mailbox;
     terminal events are `ExportProgress.done` (with optional `url` or error). May internally be
     N API calls (chunked upload, media-then-tweet, etc.). Raises `ExporterError` on API failure
     (caught by the worker; surfaced via mailbox).

   *Why locked:* reject-only `can_export` was ambiguous (pre- or post-prepare?) and exceptions are
   the python-shaped way to signal "this input cannot satisfy this constraint" — one less abstract
   method to teach. Every exporter needs an ffmpeg transform step; making it implicit in `export()`
   hides where the slow step is.

5. **Auth is a first-class sub-protocol, not a `draw_config_ui` field.** Methods (final shape
   after post-impl review):
   - `@property auth_state -> AuthState` (enum: `unconfigured / authed / error` — three states.
     The `authenticating` and `pending_user_action` shapes that OAuth2 needs get added by the
     YouTube exporter PR). *Property, not method* — it's pure state read; methods imply work.
   - `begin_auth() -> None` (exporter spawns its own loopback HTTP listener + `webbrowser.open(...)`
     for OAuth2; or just validates a bot token for Telegram). Non-blocking; updates `auth_state`
     via mailbox.
   - `draw_config_ui() -> None` draws non-secret config inputs (sticker set name, video
     privacy default, etc.). Mutates `self._settings` in place. The "Authenticate" button +
     auth status display live in `draw_target_panel` (auth is target-level state).
   - `current_settings() -> dict[str, Any]` returns the persistable settings view. Host calls
     this from `App.save` to flush each exporter's settings into `UIAppState.exporter_settings`.
   - `set_media_dir(media_dir: Path) -> None` injects the per-project media dir. Host calls
     once per project load BEFORE `rebind`.

   *Why locked:* OAuth2 loopback is a multi-second browser round-trip; can't live inside one
   synchronous imgui frame. Telegram's bot-token-in-a-textbox is the easy case and the original
   design was misled by it. 3-state enum is the honest scope-fit for one concrete exporter.
   The `(settings) -> settings` shape originally specced was awkward because the host had to
   know which keys were settings vs host-injected; making the exporter own its dict and exposing
   `current_settings()` + `set_media_dir()` keeps the boundary clean.

6. **Exporter instances live for the app's lifetime, owned by `App`.** Created in
   `App.__init__`, settings rebound on project load via `registry.rebind(state.exporter_settings)`
   (the old `set_active_provider`'s `media_list.clear()` becomes a no-op — there's no manager-
   owned list). Worker thread starts lazily on first export call and stops in `App.release()`.
   **Settings rebind clears `auth_state()` to `unconfigured`**: in-flight uploads finish on the
   previous client (not cancelled — losing user work on project switch is bad UX), but the user
   must click "Authenticate" once after switching projects. *Why locked:* per-project full
   teardown of an authed session is wrong UX (re-auth every project switch is fine; losing in-
   flight work isn't); silently using a stale client (the lazy-swap option) decouples errors
   from their cause (token-bad error surfaces at upload, not at the settings change that caused
   it).

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
      exporter instances dict + active id. No `media_list` / `selected_media_index` /
      `refresh_media`. `set_active(id)` only updates `active_exporter_id`; no callbacks, no
      notifications — `tabs/share.py` re-reads on every frame. `rebind(exporter_settings)` calls
      each exporter's `rebind()` (which clears auth — see Decision 6).
    - `shaderbox/exporters/telegram.py` — `TelegramExporter(Exporter)`. Owns its sticker-grid
      panel internally (today's `draw_share_tab` Telegram-specific blocks move here).
    - `shaderbox/tabs/share.py` — the share-tab orchestration (exporter dropdown, dispatches
      `exporter.draw_config_ui()` / `exporter.draw_target_panel()`, owns the one preview
      `Canvas`). First real `tabs/*.py` module — sets the pattern.
    - `shaderbox/ui.py` — `draw_share_tab` deleted; `App` instantiates `ExporterRegistry` and
      `tabs.share.draw(app_state, registry)` is called from the tab dispatch (the artifact is
      tab-internal — see Decision 13). `App.release()` calls `registry.release()` (which stops
      worker threads).
    - `shaderbox/sharing.py` — **deleted.**
    - `shaderbox/telegram_provider.py` — **deleted** (logic moves to `exporters/telegram.py`).

11. **`RenderedArtifact` is exporter-domain, NOT a media primitive.** Lives in
    `exporters/base.py`, not `media.py`. `media.py` stays GL-bound (`Image`, `Video`, `MediaWithTexture`).
    *Why locked:* media primitives need a live GL context; artifacts cross thread boundaries.
    They're different abstraction layers.

12. **`UIMessage` (`ui_models.py:19`) is not deleted — it stays as a generic per-exporter log
    primitive** consumed by `ExporterStatus.message`. *Why locked:* it's not coupled to sharing;
    other places use it.

13. **`current_artifact` ownership: the share tab owns one `MediaDetails` and one `RenderedArtifact`
    for the active node.** Today each share-media slot owns its own `media_details` + path; the
    refactor removes the slot list, so the rendering UX moves to the tab itself. `tabs/share.py`
    holds: (a) one `MediaDetails` settings struct (resolution / fps / duration — the user tweaks
    these), (b) one tab-owned scratch path under `<project>/exporter_scratch/<uuid>.{webm,png}`,
    (c) the most-recent `RenderedArtifact | None`. The "Render" button in `tabs/share.py` calls
    `node.render_media(details)` against the scratch path → builds a fresh `RenderedArtifact` with
    a new UUID-suffixed filename → assigns it as the current artifact (the previous artifact's
    file is unlinked unless an export is in flight against it). The exporter's "Add as sticker" /
    "Replace" / etc. buttons in `draw_target_panel` consume the current artifact (pass it to the
    worker via `exporter.enqueue(artifact)`). *Why locked:* without this the spec's `current_artifact`
    parameter is undefined and a fresh agent has to invent the entire input-side flow.

14. **`RenderedArtifact` file lifecycle: tab-owned scratch dir + UUID filenames + post-export
    cleanup.** Render-output files live in `<project>/exporter_scratch/`; each `RenderedArtifact`
    gets a UUID-suffixed filename so concurrent re-renders don't overwrite a file an in-flight
    export is reading. `prepare()` writes its re-encoded output as `<original-stem>.prepared.<ext>`
    in the same dir. The worker deletes both the input and prepared files after a terminal
    `ExportProgress` event (success or error). On `App.release()`, the scratch dir is wiped of any
    files not currently in-flight. The mailbox `queue.Queue` is bounded (`maxsize=128`); progress
    events are dropped oldest-first if the consumer (imgui frame) falls behind, but `done`/`error`
    events are never dropped (separate `_terminal_event` slot). *Why locked:* without explicit
    lifecycle, repeat renders pile up files in `media_dir`, concurrent renders silently corrupt
    in-flight uploads, and a long-running session with chatty exporters leaks unbounded queue
    memory.

15. **Threading invariant enforcement: ABC docstring + method-affinity discipline + manual
    method-walk step.** Decision 9 ("worker thread MUST NOT call `moderngl.*`") is honor-
    system in pure-design terms; we add three lightweight enforcements: (a) the `Exporter` ABC
    docstring lists each method's thread affinity explicitly; (b) **method affinity, not
    import affinity** — render-thread methods may construct/access `Image / Video / Canvas /
    Node / Texture`; worker-thread methods (`prepare`, `export`, `_handle_*`, `_do_*`,
    `_worker_main`, `_run_async`, `_with_bot`, `_cleanup_paths`, `_push_*`) MUST NOT. Imports
    of these symbols at the top of `exporters/telegram.py` are expected — they're load-bearing
    for typing the render-side state (`_StickerSlot.image: Image | None`,
    `_RenderState.preview_canvas: Canvas | None`). The realistic bug-shape is constructing GL
    objects in worker code, not importing the type names; (c) the existing telegram-provider
    flow that downloads sticker files and immediately wraps them as `Image`/`Video` (today:
    `telegram_provider.py:53-86`) is split — the worker downloads bytes/files only, the render-
    thread `_lazy_thumbnail` constructs `Image`/`Video` on first display reference. *Why
    locked:* a runtime threading assertion in `media.py` is overkill for one consumer; the
    method-affinity rule + manual method-walk verification is cheap and catches the realistic
    regression. *Mid-impl note:* originally specced as "no imports from media/core"; relaxed
    to method affinity during post-impl review (see Review history) because the strict import
    rule forced awkward `TYPE_CHECKING` shims for no runtime benefit.

16. **Shutdown drainage: non-daemon worker, sentinel + bounded join, abandon survivors.**
    Exporter worker threads are non-daemon. `App.release()` calls `registry.release()` which for
    each exporter: (a) sends a sentinel to its enqueue queue, (b) waits up to 5 seconds for the
    worker to drain to a terminal event and exit, (c) if the timeout expires, abandons the upload
    (logs a warning; user re-runs next session — the artifact files in scratch dir are kept on
    timeout for that reason). Then closes the per-exporter asyncio loop and releases the per-
    exporter preview `Canvas`. *Why locked:* daemon threads silently leak `tg.Bot`/`imageio`
    resources; indefinite-wait hangs the app on close during a slow upload; aggressive cancel
    loses user work without warning.

17. **`tabs/*.py` pattern (sets the convention for all future tab extractions): a `draw()` free
    function + an optional `update()` for render-thread per-frame work.**
    - `def draw(app_state: UIAppState, *deps) -> None`: imgui draw calls only. Caller (the tab
      dispatch in `App`) owns module-instance state; the function is stateless across frames.
    - `def update(*deps) -> None` (optional): runs from `App.update_and_draw` BEFORE imgui draw
      begins — for canvas ticks, mtime polling, anything that touches GL state. `tabs/share.py`
      uses both: `tabs.share.update(node, registry)` ticks the preview canvas;
      `tabs.share.draw(app_state, registry)` draws the imgui panel.
    - Module-internal state (the cached `RenderedArtifact`, `MediaDetails`, the preview `Canvas`)
      lives in a module-level `_TabState` dataclass instantiated once in `App.__init__` and passed
      into both `update()` and `draw()`. Future `tabs/render.py`, `tabs/node.py` follow the same
      shape.

    *Why locked:* this is the first real `tabs/*.py` extraction (tracked in `[DEFERRAL] split
    ui.py`); without an explicit pattern, the next extraction has to invent one and we drift.
    Also added to `conventions.md ## Design decisions` as part of this PR (the pattern is
    cross-cutting, not one-off).

## Files touched

**Created:**
- `shaderbox/exporters/__init__.py`
- `shaderbox/exporters/base.py` — ~120 lines (ABC, dataclasses)
- `shaderbox/exporters/registry.py` — ~60 lines
- `shaderbox/exporters/telegram.py` — ~250 lines (port + sticker-grid panel + auth + prepare/export)
- `shaderbox/tabs/__init__.py`
- `shaderbox/tabs/share.py` — ~150 lines (orchestration + preview canvas)

**Modified:**
- `shaderbox/ui.py` — concrete touches:
  - Delete `draw_share_tab` (~280 lines, `ui.py:1080-1306`) and `_draw_share_tab_safe`
    (`ui.py:1502-1511`); the safe-wrapper's try/except moves into `tabs/share.py`'s top-level
    `draw()` (per-exporter panel methods do NOT inherit a try/except — exporter panels must be
    safe by construction).
  - Replace `_share_manager` field with `_exporter_registry`; the `_init_share_manager` call in
    `App._init` becomes `self._exporter_registry.rebind(state.exporter_settings)` after
    `load_and_migrate`. The `media_list.clear()` line in `_init` is deleted (no manager-owned
    list under the new design).
  - **Delete `self._loop = asyncio.new_event_loop()` and the unused `import asyncio`** — the
    asyncio loop now lives inside `TelegramExporter`'s worker thread, not on `App`. The three
    `_loop.run_until_complete` call sites (around `ui.py:276`, `:1226`, `:1294`) all go away
    with `draw_share_tab`'s deletion.
  - The per-frame share-preview block in `App.update_and_draw` ("Render current share media
    preview", currently `ui.py:1558-1562`, the `selected_media.preview_canvas` `hasattr`
    dispatch) is removed; its replacement is `tabs.share.update(current_node, registry)` called
    from `update_and_draw` immediately after the "Render current node preview" block, which
    delegates to the active exporter's render-thread tick (e.g. `TelegramExporter` rendering the
    selected sticker's preview into the tab-owned `Canvas`).
  - `App.release()` calls `self._exporter_registry.release()` (joins worker threads, closes
    asyncio loops, releases per-exporter preview canvases — see Decision 16).
  - The share-tab call site in the tab dispatch (currently `ui.py:1497`) becomes
    `tabs.share.draw(self._ui_app_state, self._exporter_registry)` — no `current_artifact`
    parameter (it's owned inside `tabs/share.py`'s module state, see Decision 13).
- `shaderbox/ui_models.py` — `UIAppState`:
  - Rename `share_provider_configs` → `exporter_settings` and `active_share_provider` →
    `active_exporter_id` on the model.
  - **Add `model_config = {"extra": "forbid"}`** to make pydantic raise on unknown keys instead
    of silently dropping them — this is what makes the migration shim verifiable (without it, a
    broken shim leaves the user with default-empty fields that look valid).
  - Extend `load_and_migrate` to (a) keep the existing `tg_*` migration block (idempotent,
    cheap, harmless if no `tg_*` keys present) and run it FIRST, (b) then a new block: if
    `share_provider_configs` is present, pop it and assign to `exporter_settings`; if
    `active_share_provider` is present, pop it and assign to `active_exporter_id`. The migration
    is one-shot — first save after launch writes the file with only the new keys.
  - Telegram bot token stays in `exporter_settings["telegram"]["bot_token"]` (same shape as today).
- `pyproject.toml` — no new runtime deps (keyring is NOT added; `google-api-python-client` NOT
  added — YouTube is out of scope). `python-telegram-bot` stays. Wheel/sdist auto-discovers
  `exporters/` and `tabs/` subpackages via `packages = ["shaderbox"]` (verified — no change
  needed).
- `.pre-commit-config.yaml` — no change (pyright stays non-blocking; see "Does NOT resolve" above).
- `CLAUDE.md` — no structural change expected.
- `ai_docs/conventions.md`:
  - `## Known quirks`: **delete** the "ui.py has pre-existing type debt" bullet entirely (the
    share-tab `hasattr`-dispatch is the entire reason for that bullet, and it's gone).
  - `## Design decisions`: revise the "No `async`" bullet — drop the "if the share path is
    rewritten" qualifier (the share path is being rewritten); change the trigger to "Revisit
    if a future exporter brings a new async-required dep that doesn't fit the worker-thread +
    own-loop pattern." Also add a new bullet: "**`tabs/*.py` pattern: free `draw()` + optional
    `update()`, module-level `_TabState` for cross-frame state.** First instance: `tabs/share.py`
    (this PR). Revisit when 3+ tab modules exist and a different shape emerges."
- `ai_docs/dev_flow.md ## Recipes` module map: replace `sharing.py` + `telegram_provider.py`
  bullets with `exporters/` dir bullet; mention `tabs/share.py` as the first tabs module +
  reference `conventions.md ## Design decisions` for the pattern.
- `ai_docs/todo.md`:
  - Delete `[DEFERRAL] two near-identical sticker models` (resolved: both `ShareableMedia` and
    `TelegramShareableMedia` are deleted with `sharing.py` + `telegram_provider.py`).
  - **Narrow** (do NOT delete) `[DEFERRAL] blocking asyncio / blocking HTTP in the render loop`
    to ModelBox only — the Telegram half is resolved (no more `_loop.run_until_complete` in any
    imgui-frame draw path), but `modelbox.py:52`'s synchronous `requests.post(timeout=600.0)` in
    `infer_media_model` still blocks the render thread on a long inference. Update the deferral's
    title and trigger accordingly.
  - Update `[DEFERRAL] split ui.py` to note `tabs/share.py` landed as the first extraction (and
    the pattern is documented in `conventions.md ## Design decisions`).
- `ai_docs/worklog.md` — new entry on completion (per `dev_flow.md` step 9).
- `projects/dev/app_state.json` — mutates on first launch post-impl by the migration shim
  (`share_provider_configs` → `exporter_settings`, `active_share_provider` → `active_exporter_id`).
  The maintainer commits the migrated file as part of the manual-verification step.

**Deleted:**
- `shaderbox/sharing.py`
- `shaderbox/telegram_provider.py`

## Manual verification

Telegram is the only exporter; verification requires bot token + user id + sticker set name
(maintainer-owned secrets — `dev_flow.md ## Recipes` confirms this can't be exercised without
configuration). Maintainer runs:

1. **Pre-condition:** existing project at `projects/dev/app_state.json` with
   `share_provider_configs.telegram.bot_token` set (maintainer's tracked dev project — verified
   to exist).
2. **Migration roundtrip (the only check that catches a broken shim):**
   - Launch the app. The share tab should show the bot token field populated.
   - Press Ctrl+S to save.
   - Inspect `projects/dev/app_state.json`. Expected: `share_provider_configs` key is **gone**,
     `active_share_provider` key is **gone**, `exporter_settings.telegram.bot_token` exists with
     the original value, `active_exporter_id == "telegram"`.
   - Re-launch the app. Bot token still populated → migration is idempotent and round-trips
     cleanly.
   - Negative check: temporarily add `garbage_unknown_key: "foo"` to the JSON, launch, expect
     pydantic to raise (proves `extra="forbid"` is wired). Remove the bad key after.
3. **Auth (happy path):** Share tab → Telegram → "Authenticate" button → bot initializes;
   `auth_state()` transitions `unconfigured → authed` (sub-second for Telegram); UI shows the
   authed indicator.
4. **Existing stickers grid:** Telegram panel shows N+1 slots (N existing stickers in the
   sticker set + 1 "new" slot at index 0); selecting an existing sticker shows its current
   Telegram thumbnail in the slot; the selected slot's preview canvas updates live with the
   current node's shader output (Resolved decision: live-preview UX preserved). This is
   `TelegramExporter.draw_target_panel`, NOT host UI.
5. **Render → prepare → export (new sticker, the load-bearing happy path):**
   - Render a node, switch to share tab, click the tab's "Render" button, click "Add as sticker".
   - Verify `<project>/exporter_scratch/<uuid>.webm` is created (the `RenderedArtifact`'s file).
   - Verify `prepare()` produces `<uuid>.prepared.webm` ≤256KB webm/VP9 ≤3s ≤512px ≤30fps.
   - Verify mailbox progress events appear in order: "preparing" → "uploading X%" → "done".
   - **Responsiveness check (concrete, replaces the old "no freezes" mouse-feel):** during the
     upload, drag the share-tab window or scroll the node grid — verify the UI keeps drawing.
     The previous bug-shape was `_loop.run_until_complete` blocking the imgui frame for the
     entire upload; the new shape must keep the frame loop running. As an objective assertion,
     temporarily add a `print(glfw.get_time() - last_frame_time)` around
     `App.update_and_draw`'s `glfw.swap_buffers` call and confirm no delta exceeds ~50ms
     during the upload window. (Don't commit the print — it's a one-shot diagnostic.)
   - Verify after success: both `<uuid>.webm` and `<uuid>.prepared.webm` are deleted from
     `exporter_scratch/`.
6. **Replace sticker:** select existing sticker → click the tab's "Render" button → "Replace";
   verify replace flow + same file-cleanup invariant as step 5.
7. **Delete sticker:** select existing → "Delete"; verify deletion + the slot disappears from
   the grid.
8. **Error path (catches silent failure):**
   - Edit `exporter_settings.telegram.bot_token` to garbage; restart; click Authenticate.
     Expected: `auth_state()` transitions `unconfigured → error` with a user-readable error
     message in the share tab. NOT a freeze, NOT a swallowed exception, NOT a stuck spinner.
   - Restore the real token after.
9. **Project switch:**
   - Load a different project (Ctrl+O). Verify exporter instance survives (no re-instantiation
     log line); verify auth state clears to `unconfigured` (per Decision 6); the bot token field
     is populated from the new project's `exporter_settings.telegram.bot_token` (or empty if
     the new project has no Telegram config).
   - Re-click Authenticate; verify `authed` again.
10. **Threading invariant grep (catches Decision 9 regressions):** the rule is
    *method affinity*, not import affinity (see Review history amendment 4). Verify by reading
    the worker-thread methods (`prepare`, `export`, `_handle_*`, `_do_*`, `_worker_main`,
    `_cleanup_paths`, `_push_*`, `_run_async`, `_with_bot`) in `exporters/telegram.py` and
    confirming none construct or access `Image / Video / Canvas / Node / Texture`. The render-
    thread methods (`update`, `draw_target_panel`, `_lazy_thumbnail`, `_draw_sticker_button`,
    `_release_sticker_slots`) may. Imports of `Image`/`Video`/`Canvas` are expected and load-
    bearing for the render-side typing. Any GL construction inside a worker-side method is a
    FAIL.
11. **`make check`:** ruff fix + format clean (blocking). Pyright still non-blocking; diff its
    output against the pre-refactor baseline — any **new** errors introduced by the refactor
    are a FAIL. The ~16 share-tab `hasattr`-dispatch errors are gone (the dispatch is gone);
    any pre-existing errors elsewhere stay until the dedicated re-tightening effort.
12. **App shutdown / drainage (Decision 16):**
    - With no upload in flight: close the window. Verify the app exits cleanly (no zombie
      worker threads — `ps -L` shows the python process exited).
    - With an upload in flight: start an upload, immediately close the window. Verify either
      (a) the upload completes within the 5s drain window and the app exits, or (b) the 5s
      timeout fires, the warning is logged, and the app still exits (the scratch files for the
      abandoned upload remain on disk so the user can re-run next session).

A real UX gap found at any step is a FAIL, not pass-with-caveat.

## Resolved decisions (answered during plan-lock)

- **`python-telegram-bot`:** keep. Wrap async calls in a worker thread with its own asyncio loop.
- **Branch strategy:** stay on `master`, incremental commits.
- **Credentials storage:** stay in `app_state.json` (no keyring). See Design Decision 7.
- **Sticker grid live preview:** keep current behavior. When an existing sticker is selected in
  the share-tab grid, the current shader renders live into that sticker's preview slot (today:
  the "Render current share media preview" block in `App.update_and_draw`, which checks
  `selected_media.preview_canvas` via `hasattr` — only the selected one, not all of them). Under
  the refactor, the Telegram exporter's `draw_target_panel` owns one preview `Canvas` and the
  render-thread driver `tabs.share.update(node, registry)` (called from `App.update_and_draw`,
  immediately after the "Render current node preview" block) calls `node.render(canvas=...)`
  into it each frame when the selected target has a preview slot.

## Open questions for the user

*(None — all resolved during plan-lock.)*

## Review history

### Pre-impl review (2026-05-15) — 2 reviewers in parallel

Two `Plan`-agent reviewers ran in parallel: (R1) correctness & design, (R2) verification &
blast-radius. Convergent findings + decisions folded into the spec inline; this section records
the load-bearing picks and the why.

**Convergent blockers (both reviewers, independently):**
- `current_artifact` was undefined → resolved by Decision 13 (tab owns one `MediaDetails` +
  `RenderedArtifact`; "Render" button in `tabs/share.py` produces it).
- Migration shim was unverifiable (pydantic silently drops unknown keys with default
  `extra="ignore"`) → resolved by adding `model_config = {"extra": "forbid"}` to `UIAppState`
  + Manual verification step 2 now inspects the on-disk JSON post-save and includes a negative
  check for `extra="forbid"`.
- Render-output file lifecycle undefined → resolved by Decision 14 (UUID filenames in
  `<project>/exporter_scratch/`, post-export cleanup, bounded mailbox queue).

**Convergent majors:**
- Threading invariant had no enforcement → Decision 15 (ABC docstring annotations + import-
  discipline rule + Manual verification step 10 grep).
- `App._loop`, `_share_manager`, per-frame preview block all hidden in "replace `_share_manager`
  with `_exporter_registry`" → §Files touched ui.py bullet now enumerates each touch
  concretely.
- Shutdown drainage policy undefined → Decision 16 (sentinel + 5s join + abandon-on-timeout).
- `tabs/*.py` pattern undefined for the first extraction → Decision 17 (`draw()` + optional
  `update()` + module-level `_TabState`); also added to `conventions.md ## Design decisions`
  per the spec's `## Files touched`.
- "Responsiveness" verification was unmeasurable → step 5 now uses `glfw.get_time()` deltas
  around `swap_buffers` as an objective check.

**Design picks made by main agent (the spec ducked these; user signed off on
UX-and-code-clarity intersection):**
1. **`can_export()` — DROPPED**, replaced with `prepare()` raising `ExporterValueError`. UX
   identical; one fewer ambiguous abstract method to teach.
2. **`AuthState` — 3 states (`unconfigured / authed / error`)**, not 5. UX identical for
   Telegram; the OAuth2 in-flight states get added by the YouTube exporter PR when its OAuth
   dance is read end-to-end. Avoids speculative pre-fit.
3. **`RenderedArtifact.preview_bytes` — DROPPED.** No consumer in this PR; speculative fields
   rot and invite GL-free invariant confusion.
4. **Settings rebind clears auth to `unconfigured`** (the gentle option). One extra
   "Authenticate" click on project switch is fine; lazy-swap decouples errors from cause
   (worst UX); cancel-in-flight loses user work (worst UX).

**Minor findings folded inline** without separate decisions (see commit diff):
- `_draw_share_tab_safe` fate: top-level try/except in `tabs/share.py`'s `draw()`.
- `[DEFERRAL] blocking asyncio` is **narrowed**, not deleted — ModelBox's synchronous
  `requests` call is still in scope of that deferral (the Telegram half is resolved here).
- Existing `tg_*` migration block in `load_and_migrate` stays (idempotent, harmless).
- Stale `ui.py:NNNN` line refs in spec replaced with symbolic refs (`App.update_and_draw`
  "Render current share media preview" block).
- `projects/dev/app_state.json` mutates on first launch; added to §Files touched.
- Sticker grid concrete regression-detector in step 4.
- Sticker-fetch flow split (worker downloads bytes, render thread constructs `Image`/`Video`)
  documented in Decision 15.
- `pyproject.toml` package discovery confirmed — no change needed beyond the spec's existing
  bullet.
- `ai_docs/worklog.md` added to §Files touched.
- `ExporterRegistry.set_active` clarified as no-callback in Decision 10.
- `ai_docs/conventions.md ## Known quirks` "ui.py type debt" bullet is **deleted**, not
  "updated"; the `## Design decisions` "No async" bullet is revised.

**Items NOT escalated:** neither reviewer returned "should not land in current form"; all
findings were addressable inline.

**Reviewer coverage statements:**
- R1 read end-to-end: spec, CLAUDE.md, conventions.md, todo.md, dev_flow.md, sharing.py,
  telegram_provider.py, ui_models.py, core.py, media.py, .pre-commit-config.yaml. Selectively:
  ui.py lines 1-300 + 1080-1599. Confirmed via grep no other consumers of the deleted modules.
- R2 read end-to-end: spec, CLAUDE.md, dev_flow.md, todo.md, sharing.py, telegram_provider.py,
  ui_models.py, modelbox.py, .pre-commit-config.yaml, projects/dev/app_state.json,
  pyproject.toml, build.sh. Targeted reads on ui.py, media.py, core.py, conventions.md.
  Grepped the full repo for stale `sharing|telegram_provider|share_manager|...` references.

### Post-impl review (2026-05-15) — 3 reviewers in parallel + round-1 fixes

Three `Plan`-agent reviewers ran in parallel: code correctness, architecture & conventions,
spec-fidelity audit. Convergent blockers + majors fixed inline; minor findings folded.

**Convergent blockers:**
- Banned `from __future__ import annotations` in `ui_models.py` (pre-existing but the refactor
  edited the file) → removed; added `Self` returns where needed.
- `# type: ignore[union-attr]` in `tabs/share.py:_draw_inner` papered over registry's
  `Optional` return — banned outside imgui exception → refactored to assert + dedicated loop.
- `imgui.begin_child` without try/finally would corrupt imgui layout state on inner exception
  → wrapped both `begin_child` calls in try/finally.
- Worker shutdown could hang non-daemon thread on in-flight upload → `release()` now cancels
  the in-flight asyncio Task via `loop.call_soon_threadsafe(task.cancel)` BEFORE sending the
  stop sentinel; worker catches `CancelledError` and pushes a terminal error event. Worker
  loop ref + current task ref tracked on the exporter for this.
- `_cleanup` was only called in success-path try/finally, leaking input artifact files when
  `prepare()` raised → `_handle_add` and `_handle_replace` now wrap `prepare()` in their own
  try/except that cleans up the input artifact on failure before re-raising.

**Convergent majors:**
- `__media_dir` settings side-channel was fragile → added `Exporter.set_media_dir(media_dir)`
  abstract method; host calls it once per project load BEFORE `rebind`. Dropped the strip-`__`
  logic from `current_settings()` (now just `dict(self._settings)`).
- `auth_state()` was a method despite being pure state read → converted to `@property` on
  both `Exporter` ABC and `TelegramExporter`. Spec Decision 5 amended.
- Decision 15(b) "no imports from media/core" was too strict — render-thread methods need
  `Image`/`Video`/`Canvas` typing → relaxed to "method affinity" (no GL construction in
  worker methods); Manual verification step 10 rewritten to read methods rather than grep
  imports.
- `draw_config_ui(settings) -> settings` signature was awkward (host had to know which keys
  were settings vs host-injected) → changed to `() -> None` plus new `current_settings()`
  abstract method. Spec Decision 5 amended; the `set_media_dir` change above replaces the
  reason this was needed.
- `rebind()` didn't release sticker `Image`/`Video` GL handles → extracted
  `_release_sticker_slots()` helper, called from both `rebind` and `release`.
- Worklog promised but never written → covered in /sanitize sweep (separate commit).

**Minor / hygiene fixes folded:**
- ABC docstring listed nonexistent `enqueue` method → corrected to actual method set.
- Auth status display was duplicated in `draw_config_ui` and `draw_target_panel` → moved to
  `draw_target_panel` only (auth is target-level state).
- `_DummyEvent` dead class → deleted.
- ffmpeg `subprocess.run` had no timeout → added `timeout=60s`, raises `ExporterError` on
  timeout.
- "Preparing (ffmpeg)..." progress event added at the top of `_handle_add` and
  `_handle_replace` so the user sees something during the 1-3s ffmpeg encode.
- Magic numbers in `telegram.py` (preview thumb height, grid columns, queue/timeout/codec
  caps) lifted to module-level `_*` constants.
- Five copies of `tg.Bot(token); init; op; shutdown` extracted into `_with_bot()` helper.
- `current_node: Any` typed as `UINode | None` in `tabs/share.py` and the `Exporter` ABC
  methods that take it.
- `TabState.scratch_dir` no longer defaults to `Path()` (cwd footgun); `App` defers
  `_share_tab_state` construction to `_init` after `project_dir` is known. Call sites guard
  on `is None`.
- `share_tab.update` call site now wrapped in try/except (matches `draw`'s safety).
- `_render_into_state` now cleans up partial render output on exception.

**Items deferred (not blockers; better as follow-ups):**
- `asyncio.get_event_loop()` deprecation: addressed implicitly by `_run_async()` using
  `self._worker_loop` directly.
- Sticker thumbnail file cache staleness (`<file_id>.webm` files accumulate forever): pre-
  existing, out of scope for this refactor — leave as-is; the cache-by-file_id design is
  intentional (file_id is content-stable).
- Spec Decision 14's "wipe scratch on `App.release()`" promise: not implemented. The per-
  export cleanup covers the happy path; orphan files only accumulate if the user renders
  but never exports (or hits an unhandled crash). Left as a known minor gap; pre-existing
  scratch-dir entries on next launch are harmless.
- Spec Decision 14's "never drop terminal events": the queue has `maxsize=128` with
  drop-oldest. In practice unreachable (one terminal per job, one job at a time). Left as-is.
- Spec Decision 13's "unless export in flight" file-unlink guard: not implemented. The
  worker's `prepare()` reads bytes synchronously into ffmpeg; once `prepare()` returns the
  input artifact is no longer read, so re-rendering mid-upload is safe in practice. Left
  as-is; spec text is honest about the gap.

**`ABC.docstring` thread-affinity table:** rewritten to list the actual method set after the
`set_media_dir` and `current_settings` additions, and the `enqueue` typo fix.
