# 012 — YouTube upload exporter (long-form + Shorts, bring-your-own-OAuth, private-only)

Status: **shipped v0.8.0 — implemented, manually verified end-to-end, reviews converged (2 spec + 2
post-impl rounds)**
Shape: high-blast-radius feature flow (`ai_docs/dev_flow.md ## Feature flow`) — adds the second concrete
exporter, a new global-integration credential block, two new MAIN deps (Google client libs), the
exporter's worker thread, Settings→Integrations config UI, and a Share-tab outlet panel.

> Note: this spec's `draw_target_panel` / layout sections describe the design as planned. The shipped
> layout simplified to **fixed-size preview (`SIZE.SHARE_PREVIEW_*`, taller than the controls) + controls
> stacked top-down**, no bottom-alignment/measuring (the alignment fight was abandoned). The share-panel
> UI is composed from shared `ui_primitives` (`preview_box`, `status_slot`, `labeled_*`,
> `unconnected_gate`, `connection_status`, `setup_steps`) — see `conventions.md ## Design decisions`.

---

## Context

ShaderBox already uploads to Telegram via the "bring your own credentials" model (paste a bot token,
Connect once, render → upload). The maintainer wants the same for **YouTube**, covering both
**long-form** video and **Shorts** (vertical 9:16, ≤60s). The whole OAuth + upload path was
prototyped end-to-end against a real account (channel `bashbarash`) before this spec — every claim
below is empirically confirmed, not assumed.

The intended outcome: a YouTube outlet that sits beside Telegram on the Share tab, with **all controls
in the ShaderBox UI** (title/description/tags/category/shape/duration), uploading the rendered video
as a **private** draft and handing the user a one-click deep link to flip it public in YouTube Studio.
"Minimum manual actions" is the explicit priority — the only unavoidable manual steps are Google's
one-time Cloud-project setup and the one-time "unverified app" consent click-through (Google's gate,
not ours).

### Verified facts this spec rests on (prototyped against a real account, not assumed)

| Claim | Verdict | Evidence |
|---|---|---|
| YouTube programmatic upload needs OAuth2 + a user-owned GCP project (no API-key path) | **Yes** | google-api-python-client #1196; the prototype |
| Same "bring your own credentials" model as Telegram works | **Yes** | prototype: pasted the user's `client_secret.json`, connected once |
| `InstalledAppFlow.from_client_config(cfg, scopes).run_local_server(port=0)` opens browser, blocks on loopback redirect, returns Credentials w/ refresh_token | **Yes** | prototype `auth` ran; token.json had `refresh_token` |
| Credentials persist via `creds.to_json()`, reload via `from_authorized_user_info(...)`, auto-refresh via `creds.refresh(Request())` — "connect once" | **Yes** | prototype force-expired the token; it auto-refreshed, no re-auth |
| `videos().insert(part="snippet,status", body=..., media_body=MediaFileUpload(path, chunksize=-1, resumable=True))` + `next_chunk()` loop returns a videoId | **Yes** | prototype uploaded `3zE9Lv3y_Uc` |
| Unverified app keeps uploads **private** (we request private; API sets private) | **Yes** | prototype: `privacyStatus: private` in the insert response |
| A Short = vertical 9:16 ≤60s with `#Shorts` in title/description; no separate API | **Yes** | YouTube docs; prototype uploaded a 9:16 clip tagged `#Shorts` |
| The Studio edit deep link `https://studio.youtube.com/video/{id}/edit` works | **Yes** | prototype returned + opened it |
| Google client libs are **synchronous** (no asyncio) | **Yes** | google-api-python-client is sync |
| The GCP console is now the "Google Auth Platform" (7-tab nav, 4-step accordion), not the old "OAuth consent screen" wizard | **Yes** | walked it in Playwright 2026-05-27 |
| Personal (non-Workspace) account → only **External** user type available; app starts in **Testing** (100-user cap) until "Publish app" | **Yes** | walked it — "Internal" was disabled |
| Adding a **test user** OR clicking **Publish app** clears the `Error 403: access_denied` | **Yes** | prototype hit the 403, fixed by adding self as test user |

## Goal

Add `YouTubeExporter` — the third concrete exporter — mirroring `TelegramExporter`'s shape but
**synchronous** (the Google client has no asyncio). One exporter, one Share-tab outlet, with an
in-panel **shape toggle** (long-form ↔ Short) that swaps the `RenderPreset`. Credentials are global
(in `integrations.json`), pasted once in Settings → Integrations. Uploads are private; the user gets a
copyable Studio link + a notification.

User flow:
1. **One-time GCP setup** (Google's gate, ~6 web steps — instructions rendered inline in Settings):
   create a project → enable YouTube Data API v3 → configure OAuth consent as **External** → add self
   as **test user** (or Publish app) → create a **Desktop** OAuth client → download `client_secret.json`.
2. **One-time in-app:** paste `client_secret.json` (or client_id + secret) into Settings → Integrations
   → YouTube → **Connect**. The browser opens the consent screen; on "Google hasn't verified this app"
   click **Advanced → Continue**; grant both scopes. Token (with refresh token) persists — connect once.
3. **Per upload:** on the Share tab's YouTube outlet — pick **shape** (long-form / Short), fill
   title/description/tags/category, set duration, **Render**, then **Upload**. The video lands
   **private** on the user's channel.
4. **Publish:** click the copyable **Studio edit link** in the panel (or in the notification) → set
   visibility to Public in Studio. (This one click is the user's; an unverified app cannot publish
   public via the API — by design, this is private-only.)

## Out of scope

- **Public auto-publish.** Unverified apps can only upload as private; making public requires Google's
  OAuth verification audit (+ likely CASA assessment) — disproportionate for a solo itch.io tool.
  *Trigger: the maintainer decides public auto-publish is worth a verification audit.*
- **Scheduled / playlist / thumbnail / caption uploads.** Just `videos.insert` with snippet+status.
  *Trigger: a user asks for scheduling, a custom thumbnail, or playlist targeting.*
- **Loop-offset / start-time render** (the t=0..N issue, `todo.md`). A long-form upload always renders
  from t=0. *Trigger: same as the existing todo entry — a user wants a window other than the first.*
- **Re-encode/transcode in `prepare()`.** YouTube accepts large MP4s natively; `prepare()` only
  validates (see Decision 6). *Trigger: YouTube rejects the native libx264 output for some node.*
- **Keyring / secure credential storage.** `client_secret` + `token_json` live in `integrations.json`
  in cleartext, same posture as Telegram's `bot_token`. *Trigger: a credential-storage hardening pass
  across all integrations (do Telegram + YouTube together).*
- **A dedicated re-connect / import-existing-credentials flow.** Not needed (decided 2026-05-27): the
  GCP project + API + consent screen are server-side and permanent, so recovery is cheap. Recovery
  paths, by case:
  - *Same machine, file deleted but already connected once* → nothing to do: `rebind()` restores AUTHED
    from the persisted `token_json`+`channel_id`, and uploads auto-refresh the token.
  - *New machine, or never connected + lost the JSON* → **create a fresh Desktop OAuth client**
    (console → Clients → + Create client → Desktop app → the creation dialog has Download JSON + shows
    the secret) and Load that. **Caveat (Google UX wart):** the new "Google Auth Platform" console does
    NOT let you re-download `client_secret.json` for an *existing* client — its detail page has no
    Download button, only Delete. So "re-download the same JSON" is NOT a path; creating a new client
    is. Same project/API/consent — only the client JSON differs, and the exporter doesn't care which
    client it is.
  - *Expired/revoked token* → Disconnect (keeps the client key) → Connect.
  A type-the-id+secret-pair fast path was considered and rejected (creating a fresh client is a few
  clicks; a second entry mode would re-clutter the panel). The in-app setup steps point at the Clients
  page where Create client lives. *Trigger: a user reports this is genuinely insufficient.*
- **Audio.** ShaderBox renders silent video; uploads have no audio track. *Trigger: ShaderBox gains
  audio.*

## Decisions

### Decision 1 — One exporter with a shape toggle (not two exporters)
**We decided:** a single `YouTubeExporter` whose Share-tab panel has a long-form ↔ Short toggle.
**Why:** same auth, same panel, same upload code — "the same paths, the same ui/ux, just different
outlets" (maintainer). Two exporters would duplicate the instance and split the "Connected as" status.
**Revisit if:** the two shapes diverge enough (different metadata, different upload verbs) that a shared
panel becomes a tangle of `if shape == ...`.

### Decision 2 — `render_preset()` reads shape from the exporter instance
**We decided:** the toggle state lives on `self._render_state.shape: Literal["long","short"]` — an
attribute of the exporter, NOT in `outlet.extra_state`.
**Why:** `tabs/share.py::_draw_outlet` calls `preset = exporter.render_preset()` (share.py:104) and
`render_preset()` takes no args, so the only state it can read is the exporter instance. `extra_state`
is unreachable from there. So shape MUST live on the exporter. The render callback `_do_render`
(share.py:119) closes over the `preset` captured at line 104, i.e. the shape as of frame-top.
**The stale-preset hazard is REAL, so we do NOT depend on it being safe** (this corrects an earlier
draft): within a single frame the user could flip the toggle (drawn at :141) and click Render (also at
:141), and `_do_render` would render with the *pre-flip* preset. We therefore never compare a "stamped
intended shape" against the toggle (those can disagree). Instead the compatibility gate (Decision 3)
reads the **artifact's actual rendered size**, which is ground truth regardless of any frame-ordering.
**Revisit if:** the exporter contract gains a way to pass the live preset into `draw_target_panel`
(then the closure capture goes away and the hazard with it).

### Decision 3 — Upload gate compares the ARTIFACT'S ACTUAL SIZE to the current toggle's resolved size
**We decided:** the Upload button is enabled only when the rendered artifact's actual dimensions match
what the **currently-selected** shape's preset would produce for this node. The exporter has the node
(passed to `draw_target_panel`) and its own `render_preset()`, so it computes the expected size with
`render_preset.resolve_dims(self.render_preset(), node.canvas.size)` and compares to
`artifact.size`. On mismatch: disable Upload, show a `COLOR.STATE_INFO` notice "Rendered as
{WxH} — re-render for {Short|Long-form}" with the Render button right there (one click to fix).
**Why this, not a stamped shape:** the artifact carries its true size (`share_state.py:113` builds
`RenderedArtifact.size` from the actual render). Comparing actual-vs-expected size is immune to the
stale-closure hazard (Decision 2) — it doesn't matter *which* preset the render used, only whether the
file on disk matches what the user now wants. An exact size-equality check (not an aspect heuristic) is
unambiguous even for a square node: Short resolves to 608×1088, long-form to the node's native size; if
those happen to coincide (a 608×1088 node) the file is valid for both and the gate correctly passes.
This realizes the maintainer's ask: "check compatibility of the rendered artifact with the picked
toggle; if mismatch, disable publishing and ask to re-render."
**`is_short` for the upload** is taken from the **current toggle**, and is only ever sent when the gate
passed (size matches) — so a `#Shorts`-tagged upload is always genuinely vertical. No `prepare()`
aspect check is needed (Decision 6 drops it — it would be dead code behind this gate).
**Revisit if:** a third shape is added, or auto-re-render-on-upload is later preferred over the gate.

### Decision 4 — Credentials are global (in `integrations.json`); `current_settings()` returns `{}`
**We decided:** add a `YouTubeIntegration` block to `IntegrationsStore`; the exporter returns `{}` from
`current_settings()` so nothing per-project persists (mirrors Telegram). No per-project YouTube pointer
exists (unlike `telegram_default_pack`), so **no `isinstance` branch is needed in `app.py`** beyond the
one `register()` call.
**Why:** OAuth creds are account-wide, not per-project — exactly the rule in `conventions.md ##
Design decisions` ("Integration credentials are GLOBAL").
**Revisit if:** a genuinely per-project YouTube setting appears (then follow the `set_default_pack`
isinstance pattern at `app.py:282-284`).

### Decision 5 — Synchronous worker thread (no asyncio loop); `in_flight` gates BOTH connect and upload
**We decided:** reuse Telegram's `_job_queue`/`_progress_queue`/`_ensure_worker`/`_enqueue`/`release`
skeleton but strip the asyncio loop, `_run_async`, and `_current_async_task`. `_worker_main` calls the
blocking Google client methods directly. **`_BUSY_KINDS = frozenset({"connect", "upload"})`** — BOTH
set `in_flight = True` on enqueue (corrects an earlier draft that gated only upload). This is load-bearing:
the worker queue is serial, so while a `connect` job blocks on `run_local_server` the worker can run
nothing else; if Upload weren't gated, a user could enqueue an upload that silently waits behind the
blocked connect. The Connect and Upload buttons are both disabled while `in_flight`.
**Connect feedback + cancellation:** `run_local_server(timeout=_CONNECT_TIMEOUT_SEC=180)` is finite, so
an abandoned browser flow unblocks the worker eventually. While a connect is in flight, `draw_config_ui`
shows "Waiting for authorization in your browser… (close the window to cancel)" — there is no in-app
hard-cancel of the blocked socket (the OS server only releases on redirect or timeout); the user cancels
by completing or abandoning the browser, and the 180s timeout backstops it. `release()` puts the stop
sentinel + `join(timeout=_DRAIN_TIMEOUT_SEC)`; if a connect is mid-block it won't exit in time — log the
known leak (mirrors Telegram's `release` leak log), do not hang the shutdown.
**Why:** the Google client is synchronous; blocking calls run fine off the render thread.
**Revisit if:** a future Google async client is adopted (then converge toward Telegram's shape).

### Decision 6 — `prepare()` validates non-empty video only; no transcode, no aspect check
**We decided:** `prepare()` asserts the artifact is a video and `path.exists()` with `st_size > 0`;
returns the **same** artifact unchanged. **No Short aspect/duration check** — that would be dead code
behind Decision 3's size gate (the UI never enqueues an upload whose artifact size mismatches the
toggle, and `is_short` is only sent when the gate passed).
**Why:** YouTube accepts large native libx264 MP4s — no Telegram-style 256KB/512px re-encode needed.
The size/shape correctness lives entirely in Decision 3's render-thread gate, which is the single source
of truth (no duplicated check that could drift).
**Revisit if:** YouTube rejects the native output (then add a light re-mux/transcode in prepare).

### Decision 7 — Container is `.mp4` (H.264/yuv420p); inline GCP setup instructions live in the config UI
**We decided:** both presets set `container=".mp4"` (→ `core.py::_render_video` uses libx264/yuv420p,
YouTube's preferred format). The ~6-step GCP setup is rendered as numbered `wrapped_caption` text +
copyable console links inside `draw_config_ui`, including the "unverified app → Advanced → Continue"
note and the "uploads are always private; publish in Studio" note.
**Why:** we can't shrink Google's setup, but we make the in-app side a single paste + Connect, with the
instructions where the user needs them.
**Revisit if:** Google changes the console flow (the instructions will need a refresh — they're prose,
expected to drift; keep them terse).

## Files

### New
- **`shaderbox/exporters/youtube.py`** — `YouTubeExporter(Exporter)`. `_Job`/event dataclasses,
  `_RenderState` (shape, title, description, tags, category_id, rendered_shape, last_studio_url, auth
  state, in_flight, last_progress, media_dir), the sync worker, all ABC methods, `draw_config_ui`,
  `draw_target_panel`, `render_preset`, `build_render_extras`. Mirrors `telegram.py` structure.
- **`shaderbox/youtube_util.py`** — pure, GL-free, network-free helpers (unit-testable):
  `parse_client_secret_json(raw) -> (client_id, client_secret)` (accept the `{"installed": {...}}`
  blob; raise `ExporterValueError` on a missing/malformed key or a `{"web": ...}` blob — instruct
  Desktop type); `build_client_config(id, secret) -> dict`; `studio_edit_url(video_id) -> str`;
  `decorate_short(title, description) -> (title, description)` (ensure `#Shorts`, no dup);
  `build_insert_body(title, description, tags, category_id, is_short) -> dict`
  (`privacyStatus="private"`, `selfDeclaredMadeForKids=False`); constants `YOUTUBE_SCOPES`,
  `DEFAULT_CATEGORY_ID="22"`, `SHORT_MAX_DURATION_SEC=60.0`, `SHORT_LONGEST_EDGE=1080`, `DEFAULT_FPS`.
- **`tests/test_youtube_util.py`** — pure-helper unit tests (no GL, no network).
- **`tests/test_youtube_exporter.py`** — mock-Google tests (patch `InstalledAppFlow.from_client_config`
  + `build()` + `videos().insert()`/`channels().list()`); drive `_handle_connect`/`_handle_upload`
  synchronously (no thread). `IntegrationsStore` round-trip with a `youtube` block.

### Modified
- **`pyproject.toml`** — add to MAIN `dependencies`: `google-api-python-client>=2.0.0`,
  `google-auth-oauthlib>=1.0.0` (google-auth, google-auth-httplib2 come transitively). Then `uv sync`.
- **`shaderbox/integrations.py`** — add `YouTubeIntegration(BaseModel)` (fields below) + a
  `youtube: YouTubeIntegration = YouTubeIntegration()` field on `IntegrationsStore` (next to `telegram`,
  line 37). `extra="forbid"`; adding a defaulted field is backward-compatible.
- **`shaderbox/app.py`** — import `YouTubeExporter`; after `register(TelegramExporter())` (line 119) add
  `self.exporter_registry.register(YouTubeExporter())`. No `isinstance` branch needed (Decision 4).

`YouTubeIntegration` fields (mirror `TelegramIntegration`, all defaulted, `extra="forbid"`):
```
client_id: str = ""
client_secret: str = ""
token_json: str = ""        # creds.to_json() — carries the refresh_token
channel_title: str = ""     # whoami display (youtube.readonly)
channel_id: str = ""        # the unambiguous "real Connect happened" signal
```

## Implementation outline

### Worker thread (sync)
- `_worker_main`: `while True: job = _job_queue.get(); if STOP: return; _handle_job(job)`. No asyncio.
  `daemon=False`, matching Telegram.
- `release()`: put `_STOP_SENTINEL`, `join(timeout=_DRAIN_TIMEOUT_SEC)`; if a `connect` is mid-block on
  `run_local_server` it won't exit in time — log the known leak (like Telegram), do NOT hang shutdown.
  `_CONNECT_TIMEOUT_SEC = 180` on `run_local_server(timeout=...)` backstops an abandoned flow.
- Job kinds: **`connect`** (blocking `run_local_server` + whoami) and **`upload`**. `disconnect` runs
  synchronously on the render thread (clear token + save, like Telegram).
- `_BUSY_KINDS = frozenset({"connect", "upload"})` drives the `in_flight` gate — BOTH disable the
  Connect and Upload buttons while busy (Decision 5; the serial worker can't run a second job behind a
  blocked connect). Mirrors telegram.py:868-875 but widened to include connect.
- Events on `_progress_queue`: `ExportProgress`; `_AuthEvent(state, message)`;
  `_ConnectEvent(token_json, channel_title, channel_id)`. **Token persistence happens in the WORKER**,
  not via the event: after a successful connect or a `creds.refresh()`, the worker writes
  `self._yt.token_json` (+ channel fields on connect) and calls `self._store.save()` directly, so a
  crash between refresh and the next frame cannot lose the token. **`IntegrationsStore.save()` must be
  serialized against concurrent callers** (round-2 finding): worker + render thread (Ctrl+S `app.save()`,
  `disconnect()`) can both call `save()`, and two interleaved `path.open("w")+json.dump` runs corrupt
  `integrations.json`. Fix at the source — a **module-level `threading.Lock` in `integrations.py`
  wrapped around the body of `save()`** — so ALL writers serialize regardless of thread (also hardens
  Telegram's existing same-shaped writes). Per-field assignment (`token_json = ...`) is a GIL-atomic
  pointer swap, so reads in `_is_connected`/`draw_config_ui` need no lock; only `save()` does. The
  `_ConnectEvent` then only carries values for `update()` to reflect into auth-state + the "Connected as"
  line; the disk write already happened.

### `connect` job
Build `client_config` from persisted id/secret → `InstalledAppFlow.from_client_config(cfg,
YOUTUBE_SCOPES)` → `creds = flow.run_local_server(port=0, timeout=_CONNECT_TIMEOUT_SEC)` (blocks) →
`build("youtube","v3", credentials=creds)` → `channels().list(part="snippet", mine=True)`. **If
`items` is empty → push `_AuthEvent(ERROR, "This Google account has no YouTube channel. Create one at
youtube.com, then Connect again.")` and stop** (reviewer finding: the prototype account had a channel;
a channel-less account must not crash on `items[0]`). Else take `items[0]` for `channel_title`/`id`,
persist creds in the worker (above), push `_ConnectEvent`. Error mapping (`_map_yt_error`):
`access_denied` → "Authorization was denied — grant access on the consent screen."; `invalid_client` /
`invalid_grant` → "Client credentials rejected — re-check the pasted client_secret (must be a Desktop
client)."; loopback timeout → "Browser authorization timed out — click Connect to retry."

### `upload` job
`prepare()` (Decision 6: video + non-empty) → reload `Credentials.from_authorized_user_info(
json.loads(token_json), YOUTUBE_SCOPES)`; if `expired and refresh_token`: `refresh(Request())`, then
**persist the refreshed `token_json` in the worker** (write + `store.save()` under the lock) → build
service → `build_insert_body(title, description, tags, category_id, is_short)` (`is_short` from the job,
set only when Decision 3's gate passed) → `MediaFileUpload(path, chunksize=-1, resumable=True)` →
`videos().insert(part="snippet,status", ...)` → loop `status, response = next_chunk()`, push
`ExportProgress("Uploading...", status.progress() if status else 0.5)` → on `response["id"]`: stash
`last_studio_url = studio_edit_url(id)`, push terminal `ExportProgress(message="Uploaded (private).
Open in Studio to publish.", fraction=1.0, is_terminal=True, url=studio_url)` (auto-surfaced to
notifications by share.py:28-40). Errors: `HttpError` — map `quotaExceeded` → "Daily YouTube upload
quota reached (~100/day) — try again tomorrow.", other 4xx → the API reason string; `RefreshError`
(revoked token) → `_AuthEvent(ERROR, "Connection expired — Reconnect in Settings.")` so the green
"Connected" doesn't lie (mirrors telegram.py:919-929). All → terminal error `ExportProgress`. The
`next_chunk` loop already gives `resumable=True` transient-retry; a persistent network failure surfaces
as a terminal error. `export()` ABC method raises "dispatched per job kind" (telegram.py:819-821).

### `rebind()` (project switch — reviewer finding)
Mirrors Telegram (telegram.py:210-222): restore auth purely from the just-injected store (no network) —
`auth_state = AUTHED if (token_json and channel_id) else UNCONFIGURED`; **clear all per-outlet render
state** (`title`, `description`, `tags`, `category_id`, `shape→"long"`, `last_studio_url`,
`last_progress`, `in_flight=False`) so Project A's metadata/artifact never bleeds into Project B. Auth
identity (`channel_title`/`channel_id` — they live in the global store, not `_RenderState`) is
unaffected. Credentials are global so they correctly persist across the switch (Decision 4).

### `render_preset()` (Decision 2)
```
shape == "short": RenderPreset(is_video=True, container=".mp4", fps=DEFAULT_FPS,
    duration_max=SHORT_MAX_DURATION_SEC, resolution_policy=FIXED_ASPECT, aspect=(9,16),
    longest_edge=SHORT_LONGEST_EDGE, fit=RENDER_AT_TARGET)   # -> 608x1088 (16-aligned: 1080→1088)
shape == "long": RenderPreset(is_video=True, container=".mp4", fps=DEFAULT_FPS,
    resolution_policy=FREE, fit=RENDER_AT_TARGET)            # node native size
```
(608×1088, not 608×1080: `_align(1080,16)=1088` — verified against render_preset.py `_align`. The Short
is still a valid 9:16-ish vertical; YouTube does not require an exact pixel size, only vertical ≤60s.)

### `draw_config_ui()` (Settings → Integrations)
Setup guidance is **plain text shown only while no client key is loaded** (`not self._yt.client_id`) —
no accordion nesting (maintainer: the YouTube->Setup double-accordion was unwanted). The numbered steps
use `_wrapped_step` (normal-color, wrapped — `imgui.text` clips) + the four console URLs via
`draw_copyable_text` + the "Advanced -> Continue" note + the `COLOR.STATE_INFO` private-only note. Once a
key is loaded the block disappears; it reappears only when the key is actually cleared (see
`clear_credentials`).
**Credential input is a file picker, not paste-first** (maintainer: paste is clunky; `client_secret.json`
is a file the user downloads — there is NO URL for it, Google generates it for *their* GCP project):
`primary_button("Load client_secret.json...")` -> `pfd_block(pfd.open_file(..., filters=["JSON","*.json"]))`
-> read + `parse_client_secret_json` via `_ingest_client_secret` (wrapped in `try/except
ExporterValueError`; a `{"web":...}`/malformed/unreadable file sets `paste_error` shown in red, never
crashes). `ghost_button("Paste instead")` reveals an `input_text_multiline` fallback. Both paths funnel
through `_ingest_client_secret` (one error surface). `Client loaded: ...{id[-28:]}` confirms a key.
**Two distinct clearing actions:** `danger_button("Disconnect")` (shown when connected) drops the OAuth
token but KEEPS the client key — one-click re-Connect, instructions stay hidden;
`danger_button("Clear credentials")` (shown with a key but not connected) wipes the client key too ->
instructions reappear (the "reveal again when the user clears" path). `primary_button("Connect")` shows
only with a key loaded + not connected, disabled while `in_flight`; while `in_flight` show "Waiting for
authorization in your browser... (close it to cancel)". Connected-as line: `COLOR.STATE_OK` "Connected as
{channel_title}" / WARN / ERROR + `auth_message`.
`_is_connected()` = `auth_state == AUTHED and bool(token_json and channel_id)`.

**Integration-panel consistency (cross-cutting):** Telegram and YouTube config UIs share the same shape
via primitives in `ui_primitives.py`, so they're consistent by construction, not hand-matched:
`setup_steps(steps)` (numbered ghost/dim wrapped steps, a `(text, url)` item renders a `draw_link`
below; both panels gate it on "credential unset" and hide it once set) and `connection_status(connected,
is_error, message, who)` (the one OK/ERROR/WARN color rule + "Connected as {who}" / "Not connected."
line). Telegram's instruction was reworded from one sentence into 4 steps; both hide their steps once the
credential is set. `draw_link(label, url)` (new primitive) is a clickable URL that **opens the browser
AND copies to clipboard** on click (prepends `https://` if schemeless) — used by `setup_steps`, the
Telegram pack link, and the YouTube Studio link. `draw_copyable_text` stays copy-only for file paths
(code/details tabs), where a browser-open is meaningless.

**Settings-modal sizing (cross-cutting fix):** the modal previously snapped back to `SETTINGS_W` on every
reopen / on return from a native file dialog — `set_next_window_size` used `Cond_.appearing`, which
re-applies on every appear and clobbered any manual resize (read as a blink/reset). Changed to
`Cond_.first_use_ever` (`popups/settings.py`); imgui's `.ini` is now rooted at `app_data_dir()/imgui.ini`
(`app.py` `io.set_ini_filename` — the default wrote a stray `imgui.ini` in the launch CWD). imgui then
persists the modal size + every other resizable window across sessions.

### `draw_target_panel()` (Share-tab outlet)
Not-connected gate first → `primary_button("Set up credentials")` → `open_settings` (via
`build_render_extras` `_OPEN_SETTINGS_KEY`, telegram.py:310-313). Connected:
- **Shape control:** two side-by-side `chip_button`s ("Long-form" / "Short"). `chip_button` gains a new
  `active: bool = False` param (round-2 finding — it currently has no selected state): when `active`, it
  fills with `COLOR.ACCENT_PRIMARY`/text `COLOR.BG_APP` (like `primary_button`) so the selected mode is
  glance-readable; the unselected chip uses the default `CHIP_BG`. Always-visible (a hidden combo is the
  wrong affordance for a binary mode). Click writes `self._render_state.shape`.
- **Metadata** in a `tree_node "Details"` (collapsed by default so the panel isn't a tall wall —
  reviewer finding): Title (`input_text`), Description (`input_text_multiline`), Tags (`input_text`,
  comma-split at upload), Category (`combo` over a small fixed id→label map, default "22" People & Blogs
  — NOT Music; reviewer caught the wrong default).
- **Duration:** `duration_drag("Duration", rc.duration, v_max, w)` — `v_max=60` for Short, generous for
  long. → `rc.set_duration`.
- **Preview:** `centered_image(rc.preview_texture_glo, rc.preview_size, ...)` in a bordered child sized
  to the shape's aspect (tall for Short, wide for long).
- **`button("Render")`** → `rc.render()`. (No shape stamping — Decision 3 reads the artifact size.)
- **Upload gate (Decision 3):** compute `expected = resolve_dims(self.render_preset(), node.canvas.size)`;
  `size_ok = artifact is not None and artifact.size == expected`. `primary_button("Upload")` disabled
  unless `connected and artifact_fresh and not in_flight and size_ok`. When the ONLY failing condition is
  `size_ok` (i.e. connected + fresh but size mismatches the current shape), draw a `COLOR.STATE_INFO`
  notice "Rendered as {artifact W}x{artifact H} — re-render for {Short|Long-form}" right above the
  disabled button (Render stays enabled). On click: enqueue `_Job(kind="upload", artifact, title,
  description, tags, category_id, is_short=(shape=="short"))`.
- **In-flight:** single-slot `progress_bar` + message (telegram.py:751-756).
- **Post-upload:** `draw_copyable_text(last_studio_url, copy_value=last_studio_url,
  color=COLOR.STATE_INFO)`.

Panel state (title/description/tags/category/shape/last_studio_url) lives on `_RenderState`, not
`extra_state` (shape MUST be on the exporter per Decision 2; co-locate the rest). `build_render_extras`
passes only `_OPEN_SETTINGS_KEY`. **No `rendered_shape` field** — the gate derives shape from
`artifact.size` (Decision 3), so nothing is stamped.

## Risks / things reviewers must scrutinize
1. **Decision 3 size gate** — confirm the Upload predicate is `artifact.size == resolve_dims(
   render_preset(), node.canvas.size)` (NOT a stamped shape), so it's immune to the Decision-2
   stale-closure hazard. Confirm `is_short` is only ever sent when this gate passed.
2. **`in_flight` covers connect** (`_BUSY_KINDS` includes both) — so Upload can't enqueue behind a
   blocked `run_local_server`. Trace that both buttons disable while busy.
3. **Token persisted in the WORKER** before the connect/refresh event is enqueued (not via `update()`)
   — so a crash between refresh and the next frame can't lose the token. Confirm the lock + `store.save()`.
4. **Empty channel list** — `channels().list()` `items==[]` → friendly `_AuthEvent` error, never
   `items[0]` IndexError.
5. **`{"web":...}` / malformed paste** — `parse_client_secret_json` raises `ExporterValueError`, caught
   in `draw_config_ui` and shown as red text, never crashes the frame.
6. **`rebind()` clears per-outlet state** on project switch (title/description/shape/last_studio_url/
   in_flight) while keeping global creds — no Project-A bleed into Project-B.
7. **Blocking `run_local_server` on shutdown** — finite `timeout=180`; `release()` logs the leak, never
   hangs. Same `daemon=False` as Telegram.
8. **Scopes constant** — `from_authorized_user_info(..., YOUTUBE_SCOPES)` uses the SAME scopes as
   connect; single constant in `youtube_util`.
9. **Cleartext `token_json`/`client_secret`** in `integrations.json` — same posture as Telegram's
   `bot_token`; flagged, no policy change (out-of-scope hardening pass would do both).
10. **IPv6/VPN egress** — Telegram needed an IPv4 bind (vpn-stack gotcha #4). The Google client uses
    `httplib2`/`requests`; the prototype worked under the maintainer's tunnel, but it's a manual-verify
    item (if uploads stall on the tunnel, a transport/IPv4 workaround may be needed).
11. **Thread affinity** — `MediaFileUpload` built in the worker from `artifact.path` (GL-free Path); no
    moderngl object crosses into the worker.
12. **No exporter-domain vocab in generic code** — `RenderControl.extras` stays opaque; no YouTube
    concept in `base.py`/`registry.py`/`share.py`/`app.py` beyond the one `register()` call.

## Verification
- `make check` (ruff + pyright, 0 errors) and `make smoke` (200 headless frames) must pass.
- `make test` — `test_youtube_util.py` (pure helpers) + `test_youtube_exporter.py` (mock-Google). The
  exporter tests MUST cover the error paths (not just happy path): empty channel list →error;
  `parse_client_secret_json` on `{"web":...}`/garbage →`ExporterValueError`; `creds.refresh()` raising
  `RefreshError` →`_AuthEvent(ERROR)`; the size gate (`artifact.size != expected` → Upload disabled);
  token persisted to the store on connect AND on refresh; `IntegrationsStore` round-trip with a
  `youtube` block; `current_settings()=={}`. Drive `_handle_connect`/`_handle_upload` synchronously (no
  thread). Patch `InstalledAppFlow.from_client_config`, `build()`, `videos().insert()`,
  `channels().list()`.
- **Manual end-to-end (maintainer, the real test):** fresh GCP project → paste client_secret →
  Connect → consent (Advanced → Continue) → render a Short → Upload → confirm `private` on the channel
  → click the Studio link → publish. Then flip to long-form, **confirm Upload is disabled with the
  re-render notice** until re-rendered → Upload long-form. Then quit + relaunch → confirm "Connected as
  {channel}" persists (no re-auth). (The dev box can't be screenshotted; visual checks are the
  maintainer's per the no-screenshot rule.)
- Review loop: spec reviewers → converge → implement → post-impl reviewers → converge → manual test.
