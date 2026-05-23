# 009 — Integrations rework (Telegram UX collapse + Settings→Integrations + emoji picker)

Status: **implemented + manually verified live + post-impl reviews converged (2026-05-23); basic flow works, UI-optimization refactor is the next wave**
Shape: high-blast-radius feature flow (`ai_docs/dev_flow.md ## Feature flow`) — touches global
Settings, the exporter contract, app_state migration, a new picker module + vendored resource, and
the Telegram exporter internals.

---

## Goal

Collapse the Telegram sticker UX from "remember + type three secrets (bot token, numeric user id,
sticker-set name)" down to **one paste + two taps**, and relocate integration credentials from the
per-tab share panel into the global **Settings → Integrations** section (laying a thin framework so
YouTube/X slot in later as visible-but-disabled stubs).

Concretely, the user flow becomes:
1. One-time: create a bot via @BotFather (no hosting — see Decision 1), paste the **bot token** into
   Settings → Integrations → Telegram.
2. Tap **Start** on their own bot in Telegram, then click **Connect** in ShaderBox → the app calls
   `getUpdates`, reads `from.id` + `from.username`, stores the user id. (Captures identity + proves
   the token works in one step.)
3. Pick or create a **pack** from a dropdown (model C — per-project default, overridable). Pack names
   are **derived** (`<slug>_by_<botusername>`) and **auto-created** on first upload — never typed,
   never remembered.
4. Render → pick an **emoji** from a Telegram-ordered grid picker → **Add as sticker**.

**No serverside anything** — the user owns their bot; ShaderBox calls the Telegram HTTP API via
outbound polling. **No new color-rendering risk** — the emoji picker is monochrome glyphs (verified
ceiling, see Decision 7); the chosen emoji uploads in full color.

### Verified facts this spec rests on (empirically confirmed during research, not assumed)

| Claim | Verdict | Evidence |
|---|---|---|
| Bot token alone can enumerate owned sticker sets | **No** — API has no list method | Bot API docs; PTB `_bot.py` has no `get_sticker_sets` |
| Bot needs hosting / a server | **No** — `getUpdates` polling, token = outbound HTTPS only | Bot API model |
| Sticker set is owned by the bot | **No** — owned by the `user_id` passed to `createNewStickerSet`; bot is editor only; user manages via @Stickers | Bot API `createNewStickerSet`; cross-checked aiogram/PTB |
| Set name has a required suffix | **Yes** — must end `_by_<bot_username>` (case-insensitive), begin with a letter, `[a-z0-9_]`, no consecutive `_`, len 1–64 | PTB `_bot.py:6818-6821` docstring; `constants.StickerLimit.MAX_NAME_AND_TITLE=64` |
| `user_id` discoverable without user action | **No** — only via `getUpdates` after the user messages the bot (Start) | Bot API. (The "user must have interacted with the bot" precondition for `createNewStickerSet` is a Telegram **server-side** rule, NOT stated in the PTB docstring — true of Telegram, evidence is the live API, not PTB source.) |
| Telegram exposes an allowed-emoji list to fetch | **No** — any Unicode emoji is accepted; order is the Unicode CLDR / `emoji-test.txt` order (same as the native picker) | Telegram stickers API; parsed `emoji-test.txt` (10 groups, 3773 fully-qualified emoji) |
| imgui-bundle (this build) renders **color** emoji | **No** — NotoColorEmoji + `LoadColor` produced blank glyphs in the glfw backend; only **monochrome** `NotoEmoji-Regular.ttf` renders | spike run by maintainer 2026-05-23 (screenshots) |
| imgui-bundle renders **monochrome** emoji glyphs | **Yes** — `NotoEmoji-Regular.ttf` standalone at `legacy_size` renders clean line-art | same spike |

## Out of scope

- **YouTube / X exporters.** Settings → Integrations renders them as **disabled stub rows**
  (greyed, "Coming soon") to validate the multi-integration layout, but no concrete exporter, no
  OAuth, no `youtube.py`/`x.py`. *Trigger to build: when the user starts a real YouTube/X export.*
- **Import an externally-created pack** (a pack made by hand in @Stickers / another tool). Minimal
  stub only: the data model carries a pack list so import *can* be added, and there is a single
  "Add existing pack name…" text input behind a small expander — but no validation/listing polish.
  *Trigger: when the user reports wanting to target a pack ShaderBox didn't create.*
- **Multi-emoji per sticker** (Telegram allows 1–20). One emoji per sticker at upload. *Trigger: a
  user asks to attach several emoji to one sticker.*
- **Color emoji rendering.** Ruled out by the spike (Decision 7). *Trigger: a future imgui-bundle
  bump exposes a working FreeType color path — re-run the spike then.*
- **Sticker-set thumbnail / title editing, set deletion, sticker reordering.** Keep today's verb
  set (add / replace / delete a sticker) plus the new create-set-on-demand. *Trigger: user asks.*
- **Keyring / secure credential storage.** Bot token stays in `app_state.json` (per feature 001
  Decision 7 — unchanged here; the token is now global, see Decision 3, but still plain JSON). The
  itch.io/solo-project threat model hasn't changed. *Trigger: an exporter brings OAuth refresh
  tokens.*
- **UI redesign.** Per the user's explicit instruction: rearrange minimally to make the feature work
  and robust; do NOT redesign the share tab or settings layout beyond what the feature needs.

## Design decisions (locked pending plan-lock)

1. **Model A — the user owns the bot; no serverside.** ShaderBox does not ship a bot token, runs no
   relay. The user creates a bot via @BotFather once (a chat, not hosting), pastes the token.
   Telegram delivers updates by *polling* (`getUpdates`), so the "bot" is just a token making
   outbound HTTPS calls when the app acts — nothing runs in the background, nothing is hosted.
   *Why locked:* user explicitly rejected serverside; this is also the only model with no
   shared-secret liability. *Rejected:* shared ShaderBox-owned bot (needs a relay server or ships an
   extractable token).

2. **Pack model C — global pack list + per-project default pointer, overridable.** The set of pack
   names ShaderBox knows lives **globally** (a pack belongs to a bot username, which is global). Each
   **project** stores a `default_pack` pointer; opening a project pre-selects it in the dropdown; the
   user can switch packs per upload, and switching updates the project's default. "+ New pack" prompts
   for a human title, derives the slug + suffix, and creates the set on first upload to it.
   *Why locked:* user chose C over A (plain picker) / B (rigid 1:1 project↔pack). *Storage shape:*
   pack list is global → lives in `exporter_settings["telegram"]["packs"]` (Telegram creds are now
   global, Decision 3); the per-project default lives on `UIAppState` (Decision 4).

3. **Telegram credentials (bot token, linked user id, linked username, pack list) are GLOBAL, not
   per-project.** They move out of the per-project `app_state.json` `exporter_settings` into a new
   **global** integrations store at `app_data_dir()/integrations.json` (rooted via `app_data_dir()`
   per `conventions.md` — never `platformdirs` directly). *Why:* you authenticate one bot once, not
   per shader project; today's per-project token forces re-entry on every project. *Why locked:*
   the user said "keys live in global Settings." *Migration:* on load, if a project's
   `exporter_settings["telegram"]` carries `bot_token`/`user_id`, lift them into the global store
   once (Decision 9). *Revisit trigger:* an integration whose creds are genuinely per-project.

4. **`UIAppState` (per-project) keeps only the per-project default pack pointer.** New field
   `telegram_default_pack: str = ""` (the derived set name). Everything else Telegram moves to the
   global store. `exporter_settings` / `active_exporter_id` stay (still used by the registry +
   non-Telegram future exporters). *Why locked:* keeps the per-project / global boundary clean and
   the migration one-directional.

5. **Set-name derivation is a pure function, single seam.**
   `derive_set_name(title: str, bot_username: str) -> str`. Lives in `shaderbox/telegram_util.py`
   (new, dependency-free — so the share tab / picker can derive without importing the exporter; also
   avoids any cycle). Algorithm, in order:
   a. slugify `title`: NFKD-normalize → drop non-ASCII → lowercase → replace any char not in
      `[a-z0-9_]` with `_` → collapse runs of `_` → strip leading/trailing `_`.
   b. strip leading non-letter chars (Telegram requires the name begin with a letter).
   c. **empty-slug fallback:** if the slug is now empty (title was all emoji/punctuation/digits),
      use the stem `"set"`.
   d. **clamp the STEM** (not the whole name) so that `len(stem) + len("_by_") + len(bot_username)
      <= MAX_NAME_AND_TITLE (64)`; the `_by_<bot_username>` suffix is fixed-length and must survive
      intact (clamping the whole string would corrupt the required suffix).
   e. append `_by_<bot_username.lower()>`.
   Pack "identity" stored is the **full derived set name**; the human title is stored alongside for
   display. *Why locked:* the suffix rule is a hard API constraint (PTB `_bot.py:6818-6821`); one
   function owns it. Edge cases (empty slug, clamp-the-stem-not-the-suffix) are pinned because each
   independently makes `createNewStickerSet` reject the name (pre-impl review R1 finding 7).

6. **Account linking via `getUpdates`, on the worker thread, surfaced via the existing mailbox.**
   Linking is a network round-trip; it follows the same worker-thread + progress-mailbox pattern as
   auth/add/replace today. A new job kind `"link"`:
   a. calls `get_me()` → captures the **bot** username (the `_by_<botusername>` source for Decision
      5) + bot id; persists the bot username into the global store.
   b. calls `get_updates(offset=-1, limit=1)` → **negative offset is required**: the default offset
      returns the *earliest* unconfirmed update, so "most recent" needs `offset=-1` (PTB
      `_bot.py:4542-4544`). (R2 finding B3 — the original "small limit" wording was API-incorrect.)
   c. reads `update.effective_user` (NOT `message.from_user`) — `effective_user` (PTB
      `_update.py:480`) does the message / edited_message / callback / channel-post triage and
      returns `None` for userless updates; pressing Start can arrive as `my_chat_member` too. (R2
      finding B1.)
   d. captures `effective_user.id` (the **user** id, for `createNewStickerSet` ownership) +
      `effective_user.username` (`Optional[str]` — may be `None`; display "Connected (id …)" when
      absent). Pushes a `_LinkEvent`.
   The render thread (Settings panel) shows "Open @<botusername> in Telegram, press Start, then click
   Connect"; on click → enqueue `"link"`. Terminal outcomes: success → "Connected as @<user>"
   (prominently, so a wrong-user capture is visible — R2 finding B2; no blocking confirm modal, this
   is a personal single-user bot, but the captured identity is shown and re-Connect is always
   available); no updates (`effective_user is None` / empty) → "No message received — open the bot,
   press Start, then Connect." *Why locked:* never block the imgui frame on network (feature 001
   Decision 3). *Offset policy:* `offset=-1` peeks the newest without confirming/advancing the queue
   (open-question 2 resolution, corrected for the earliest-first default).

7. **Emoji picker: monochrome glyph grid, Telegram/Unicode order, vendored `emoji-test.txt`.**
   - **Data:** vendor Unicode's `emoji-test.txt` (the canonical grouped+ordered source the native
     picker derives from) into `shaderbox/resources/emoji/emoji-test.txt`; parse at startup into
     groups → ordered emoji (char + name). ~0.5 MB, offline, **no new pip dep** (the `emoji` PyPI
     package was checked and does NOT carry group/order metadata — rejected).
   - **Grapheme integrity (R2 finding E):** an emoji is parsed and carried as the **whole
     fully-qualified codepoint sequence** (one Python `str`, possibly several scalars — ZWJ
     sequences, skin-tone modifiers, regional-indicator flags). It is passed whole to
     `emoji_list=[value]`; the picker grid and `pending_emoji` storage NEVER index it per-`char`.
     Only fully-qualified rows from `emoji-test.txt` are kept (the `; fully-qualified` lines).
   - **Render:** a new monochrome glyph font (`NotoEmoji-Regular.ttf`, already shipped inside
     imgui-bundle's `demos_assets`; we copy it into `shaderbox/resources/fonts/NotoEmoji/` so the
     build doesn't depend on the bundle's asset path). It is added to the atlas in `App.__init__`
     **alongside `font_12/14/18`** (app.py:101-103) — NOT lazily mid-frame (adding a font face during
     a frame is unsafe; on-demand glyph rasterization of an already-added face is fine). The picker
     pushes it with the **rasterized size** (`push_font(emoji_font, <size_pixels used at add time>)`,
     never `get_font_size()` — `conventions.md ## Known quirks`). Some ZWJ sequences will render as
     fallback boxes in the mono font; acceptable (the upload is still valid + color). Selected emoji
     uploads as the real codepoint → full color on the sticker.
   - **UI:** a popup (`popups/emoji_picker.py`) — a searchable (by name substring) scrollable grid
     with category separators in canonical order; clicking an emoji sets the pending sticker emoji
     and closes. Opened from the share panel's "Emoji: 🎨 [change]" control. Follows the popup
     convention (free `draw(app)`, open/close boolean on `App`, mutex via `open_*` — and the boolean
     is added to `any_popup_open()` AND the smoke mutex assert, Decision 16).
   - *Why locked:* color is impossible in this build (Decision 7 / spike); the monochrome grid is
     the verified-shippable ceiling and still gives the native order + grouping + search.

8. **Settings → Integrations holds only credentials/identity; pack management lives on the share
   tab.** (User decision, plan-lock 2026-05-23.) `popups/settings.py` gains an "Integrations"
   `separator_text` block where each registered exporter draws its **credential block** via
   `exporter.draw_config_ui()` — for Telegram: the bot token field, the **Connect** button, the
   "Connected as @<you>" status, and the BotFather setup guide. Non-Telegram exporters render as
   **disabled stub rows** (Decision 10). The **share tab** owns the pack **dropdown + "+ New pack" +
   the import-existing stub** (Decision 2/15) alongside render / emoji / add-replace-delete /
   progress — i.e. everything operational including pack lifecycle, because creating a pack happens
   in the moment of export. *Why locked:* user chose pack-management-on-share-tab over
   pack-management-in-Settings; Settings stays the one-time credential screen, the share tab is where
   you actually publish. *Editor-settings FPE caveat:* `popups/settings.py` applies editor settings
   on close only — Integrations writes are plain field edits (no editor), so they apply live; the
   close-only rule is editor-specific and unchanged.

9. **Migration: one-shot lift of per-project Telegram creds → global store; gen 5. Strict ordering
   in `App._init`.** **SUPERSEDED 2026-05-23** — `_lift_telegram_creds` was removed entirely (user:
   no backward-compat; the lifted-but-not-validated `user_id` caused a "stale state looks connected"
   bug). Creds now come only from a real Connect; `load_and_migrate` stays at gen 4. The decision is
   kept below for the review trail; ignore the lift mechanics.
   `UIAppState.load_and_migrate` cannot write the global file, so the lift runs in
   `App._init`, and the **order is load-bearing** (R1 finding 3):
   ```
   load_and_migrate(app_state)          # app.py:229 — reads project json
   store = IntegrationsStore.load()     # NEW — global creds
   _lift_telegram_creds(app_state, store)   # NEW — see below; persists store if it lifted
   registry.set_integrations(store)     # NEW — exporter now sees global creds
   registry.set_media_dir / rebind / set_active   # app.py:238-244, unchanged order
   ```
   `_lift_telegram_creds`: if `store.telegram.bot_token` is empty AND
   `app_state.exporter_settings.get("telegram", {})` carries `bot_token`, copy `bot_token` + `user_id`
   into `store.telegram`, seed `store.telegram.packs` + `app_state.telegram_default_pack` from the old
   `sticker_set_name` (if any), `store.save()`, and **set `auth_state = UNCONFIGURED`** (migrated creds
   are NOT re-validated — the user clicks Connect once after upgrade; R2 finding C). Idempotent: guarded
   on "global token empty."
   - **The strip** (removing the lifted keys from the project json) is handled by Decision 12's
     `current_settings()` contract: post-rework Telegram's `current_settings()` returns `{}` (creds
     are global now), so the next `App.save` writes `exporter_settings["telegram"] = {}` — the lift
     already ran and persisted to the global store before any save, so this is a clean strip, not a
     data-loss race. The lift reads the OLD project dict in `_init` *before* the registry rebinds and
     before any save overwrites it.
   - `load_and_migrate` doc bumps to gen 5; the gen-5 *note* says "Telegram creds lifted to global
     `integrations.json` (see `App._lift_telegram_creds`); `telegram_default_pack` seeded from old
     `sticker_set_name`." The lift code itself is NOT in `load_and_migrate` (it can't write the global
     file) — the doc points to where it lives.

10. **Registry gains a lightweight stub/disabled concept.** Add `Exporter.is_available -> bool`
    (default `True`) and `Exporter.unavailable_reason -> str` (default `""`). YouTube/X stubs are
    minimal `Exporter` subclasses returning `is_available = False`, `unavailable_reason = "Coming
    soon"`, and no-op/`raise NotImplementedError` worker methods (never called — the UI gates on
    `is_available`). Registered alongside Telegram so the Integrations section lists them greyed.
    *Why locked:* the user asked for generalization stubs; this is the minimal honest seam (one
    property pair) without speculative abstraction. *Rejected:* a separate "IntegrationDescriptor"
    registry — over-engineered for two greyed rows.

11. **Global integrations store is its own tiny pydantic model + file, mirroring app_state's
    load/save discipline.** New `shaderbox/integrations.py`: `IntegrationsStore(BaseModel)` with
    `extra="forbid"`, `telegram: TelegramIntegration` (`bot_token: str`, `user_id: str`,
    `user_username: str` [the linked human, may be empty], `bot_username: str` [from `get_me`, the
    suffix source], `packs: list[PackEntry]` where `PackEntry = {title: str, set_name: str}`), a
    `load()` classmethod reading `app_data_dir()/integrations.json` (defaults on missing/corrupt,
    logs — same shape as `UIAppState.load_and_migrate`), and `save()`.
    - **Import-cycle fix (R1 finding F — blocker):** `integrations.py` needs `app_data_dir()`, which
      today lives in `app.py:39`; but `app.py` eagerly imports `exporters.telegram`, which will import
      `integrations` → cycle, and the no-`TYPE_CHECKING`/no-inline-import rules forbid papering over
      it. **Resolution: extract `app_data_dir()` (and the `SHADERBOX_DATA_DIR` override logic) into a
      new dependency-free `shaderbox/paths.py`.** `app.py` re-exports / imports it from there;
      `integrations.py` imports `app_data_dir` from `paths.py`. `paths.py` imports only stdlib +
      `platformdirs`, so it's a clean leaf — no cycle. (`scripts/smoke.py` currently calls
      `user_data_dir("shaderbox")` directly for the project pointer; leave that as-is or point it at
      `paths` — cosmetic, not load-bearing.)
    *Why locked:* keeps global creds out of the per-project file (Decision 3) and reuses the proven
    robust-load pattern. The `TelegramExporter` is handed this store (not a raw dict). *Revisit:* when
    a second integration needs global creds (extend the model).

12. **`TelegramExporter` re-shaped to consume the global store; `current_settings()` returns `{}`.**
    (Resolves R1 finding C — the cred-contract blocker.) The exporter reads ALL creds (token, user
    id, bot username, pack list) from the injected `IntegrationsStore`, and **writes** changes back
    into that store object (the host persists it via `store.save()` on `App.save`). Concretely:
    - `set_integrations(store: IntegrationsStore) -> None` (render thread; called once after load +
      whenever the store is reloaded) — the exporter keeps a reference and reads/writes
      `store.telegram`.
    - `current_settings() -> dict` returns **`{}`** — Telegram persists nothing per-project. This is
      the single source of truth fix: `App.save` (app.py:358-361) still calls
      `exporter_settings["telegram"] = current_settings()` = `{}`, which harmlessly empties the
      stale project sub-dict (completing the migration strip, Decision 9). No double-write: creds
      live ONLY in `integrations.json`. The registry's per-project persistence becomes a no-op for
      Telegram by virtue of the empty dict — no special-casing in the registry needed.
    - `rebind(settings)` signature stays (registry contract) but for Telegram the dict is empty/
      ignored; rebind only resets `auth_state` + clears sticker slots (as today).
    - Per-project default pack: read/written via the project's `app_state.telegram_default_pack`,
      injected with `set_default_pack(set_name)` / read via `current_default_pack()` (host wires
      `app_state.telegram_default_pack` ↔ these in `_init` and `save`).
    Worker methods (`add`/`replace`/`delete`) handle "create the set if it doesn't exist" (Decision
    13). *Why locked:* the cred relocation (Decision 3) forces store-reads; returning `{}` from
    `current_settings()` is what makes the project json the non-authoritative copy and lets the
    migration strip itself. Threading affinity (worker MUST NOT touch GL) unchanged.

12b. **Add/create preconditions + pack reconciliation (R1 finding 6, R2 findings C/D).**
    - **Gate:** "Add as sticker" / "+ create" is enabled only when `auth_state == AUTHED` AND
      `store.telegram.user_id` is non-empty. Otherwise the share panel shows "Connect your Telegram
      account in Settings first." This replaces the old three-key `_is_configured` check (telegram.py:420)
      — `int(user_id)` (telegram.py:378) must never run on an empty id.
    - **Dangling default pointer:** if `app_state.telegram_default_pack` names a `set_name` not in
      `store.telegram.packs`, re-add it as `PackEntry(title=set_name, set_name=set_name)` (the orphan
      pointer carries no known title, so the set_name doubles as the display title — same fallback as
      the import stub, Decision 15). The API cannot enumerate packs, so an orphan pointer is the only
      evidence a real pack exists — dropping it would lose access. If empty → dropdown shows "no pack —
      + New pack."
    - **Slug collision on "+ New pack":** derive the set_name; if it already exists in `packs`, do
      NOT create a duplicate entry — select the existing pack and notify ("Pack already exists,
      selected it"). Two titles slugging to the same name alias intentionally.
    - **Stale-state error mapping:** Telegram "user not found / hasn't interacted / blocked" errors
      from `create_new_sticker_set` map to "Open your bot, press Start, and Connect again" rather
      than the raw API string (still logged verbatim).

13. **Add-to-pack auto-creates the set if absent.** The worker's add path: `get_sticker_set(name)` →
    if it raises a `Stickerset_invalid` error (set doesn't exist yet) → `create_new_sticker_set(
    user_id=..., name=..., title=..., stickers=[input_sticker])`; else `add_sticker_to_set(user_id,
    name, input_sticker)`. The `input_sticker` is built exactly as today —
    `tg.InputSticker(sticker=bytes, emoji_list=[chosen_emoji], format=tg.constants.StickerFormat.VIDEO)`
    (telegram.py:372-376) — and is reused for BOTH the create and add branches. **There is no
    `sticker_format=` kwarg on `create_new_sticker_set` in this PTB** (removed in 21.2, PTB
    `_bot.py:6813`); the format lives per-`InputSticker` (R2 finding A). *Why locked:* "auto-create
    on first upload" is the whole point of derived names (Decision 5). *Edge:* `createNewStickerSet`
    requires ≥1 initial sticker (PTB `MIN_INITIAL_STICKERS=1`), so the create path *is* the first add.

14. **Emoji selection state lives on the share `TabState`** (`pending_emoji: str = "🎨"`), default
    🎨. Add/replace jobs read it. The picker popup writes it. *Why locked:* matches the existing
    "tab state on `App`/`TabState`" convention (feature 001 Decision 17); not exporter-internal
    because the emoji is chosen in the share UI, not the Integrations config.

15. **Import-existing-pack is a minimal stub.** On the share tab, behind a small expander
    ("Add existing pack…"), a single text input takes a full set_name; on submit it's appended to
    `store.telegram.packs` (title = the set_name) with no validation beyond non-empty + derived-suffix
    sanity. No listing, no live verification. *Why locked:* user asked for "a minimalistic stub for
    later"; the data model already carries the list, so this is one input + one append. *Trigger to
    flesh out:* user reports targeting an externally-made pack and wanting validation.

16. **Popup mutex is generalized so the smoke test actually guards it (R1 finding G — blocker).**
    The new emoji-picker boolean (`is_emoji_picker_open`) joins `is_node_creator_open` /
    `is_settings_open`. `App.any_popup_open()` ORs all three; `App.open_emoji_picker()` (like the
    other `open_*`) sets its own flag and clears the siblings. **`scripts/smoke.py::_check_invariants`
    (smoke.py:21-22) currently hardcodes a 2-popup pairwise assert** — it is rewritten to a
    generalized "at most one popup open" check (`sum([is_node_creator_open, is_settings_open,
    is_emoji_picker_open]) <= 1`) so a missing emoji-picker mutex entry is actually caught. *Why
    locked:* without editing smoke, the spec would over-claim coverage — the test passes even if the
    new boolean is left out of the mutex.

## Files touched

**Created:**
- `shaderbox/paths.py` — `app_data_dir()` + `SHADERBOX_DATA_DIR` override, extracted from `app.py`
  so `integrations.py` can root its file without an import cycle (Decision 11). Dependency-free leaf
  (stdlib + `platformdirs`). (~15 lines)
- `shaderbox/telegram_util.py` — `derive_set_name(title, bot_username)` (Decision 5) +
  any small Telegram-domain pure helpers; dependency-free so the share tab can derive without
  importing the exporter. (~40 lines)
- `shaderbox/integrations.py` — `IntegrationsStore` + `TelegramIntegration` + `PackEntry` pydantic
  models; `load()`/`save()` against `paths.app_data_dir()/integrations.json`. (~90 lines)
- `shaderbox/exporters/stubs.py` — `YouTubeExporterStub`, `XExporterStub` (`is_available=False`).
  (~50 lines)
- `shaderbox/popups/emoji_picker.py` — `draw(app)` searchable monochrome emoji grid popup. (~120
  lines)
- `shaderbox/emoji_data.py` — parse `resources/emoji/emoji-test.txt` → `list[EmojiGroup]` (cached
  module-level); pure, no imgui. (~60 lines)
- `shaderbox/resources/emoji/emoji-test.txt` — vendored Unicode data (~0.5 MB).
- `shaderbox/resources/fonts/NotoEmoji/NotoEmoji-Regular.ttf` — vendored monochrome emoji font.

**Modified:**
- `shaderbox/exporters/base.py` — add `is_available`/`unavailable_reason` (default impls, NOT
  abstract — stubs override, Telegram inherits `True`); add `set_integrations(store)`,
  `set_default_pack(set_name)`, `current_default_pack()` abstract methods (explicit, not folded into
  settings — Decision 12). Update the thread-affinity docstring (add the new methods, all render-thread).
- `shaderbox/exporters/registry.py` — `register` the two stubs; expose iteration that the settings
  popup uses; pass the `IntegrationsStore` through on a new `set_integrations(store)` that fans out.
- `shaderbox/exporters/telegram.py` — biggest change: read creds from `IntegrationsStore` not the
  flat dict; `derive_set_name`; `"link"` job (`getUpdates` → user id/username); pack-aware add/
  replace/delete with auto-create (Decision 13); `draw_config_ui()` becomes the **Settings** cred
  block (token + Connect + linked-as status + BotFather guide); the share-side `draw_target_panel`
  becomes the **operational** panel (pack dropdown + "+ New pack" + import stub, sticker grid, emoji
  control, add/replace/delete, progress). The hardcoded `emoji_list=["🎨"]` becomes the chosen emoji.
- `shaderbox/popups/settings.py` — add the "Integrations" section: iterate the registry, draw each
  available exporter's `draw_config_ui()`, render stubs greyed with `unavailable_reason`.
- `shaderbox/tabs/share.py` — drop the exporter dropdown + config block + the two info lines (config
  moves to Settings); keep render controls + the active exporter's operational panel; add the
  "Emoji: <e> [change]" control wired to the picker popup. `_draw_render_panel` mostly unchanged.
- `shaderbox/tabs/share_state.py` — add `pending_emoji: str = "🎨"`.
- `shaderbox/ui_models.py` — `UIAppState`: add `telegram_default_pack: str = ""`; extend
  `load_and_migrate` doc to gen 5 (the *strip* of lifted keys is idempotent here; the *lift* itself
  is in `App._init`, Decision 9).
- `shaderbox/app.py` — **move `app_data_dir()` into `paths.py`** and import it from there (Decision
  11 cycle fix); load `IntegrationsStore` at init in the strict order of Decision 9
  (`set_integrations` after the lift, before `rebind`); `_lift_telegram_creds` helper (Decision 9);
  `is_emoji_picker_open` boolean + `open_emoji_picker()` mutex helper + `any_popup_open()` includes
  it; `save()` persists the global store (`store.save()`) and wires `telegram_default_pack`; add the
  emoji font in `__init__` alongside `font_12/14/18` (Decision 7) + a `get_emoji_font()` accessor (or
  store the handle directly). Wire `set_default_pack`/`current_default_pack` ↔
  `app_state.telegram_default_pack`.
- `shaderbox/ui.py` — dispatch `draw_emoji_picker(app)` alongside `draw_settings`; the share-tab
  `update`/`draw` calls unchanged in shape. (`ui.py` imports `app_data_dir` — repoint to `paths`.)
- `scripts/smoke.py` — rewrite `_check_invariants` popup assert to the generalized "≤1 popup open"
  form covering `is_emoji_picker_open` (Decision 16). (`user_data_dir` call may repoint to `paths` —
  cosmetic.)
- `shaderbox/theme.py` — emoji-picker tokens (grid cell size, columns) if not reusing existing.
- `pyproject.toml` — confirm `resources/emoji` + the font are packaged (the wheel globs
  `resources/**` — verify; add to package-data if not). No new runtime pip dep.
- `ai_docs/conventions.md` — `## Design decisions`: add the global-integrations-store decision
  (creds global, per-project default pointer) + the stub-exporter `is_available` seam; `## Known
  quirks`: add "imgui-bundle (this build) renders monochrome emoji only — color NotoColorEmoji
  produces blank glyphs; vendored `NotoEmoji-Regular.ttf` is the emoji font" (the spike result, so a
  future agent doesn't re-attempt color).
- `ai_docs/roadmap.md` — banner rewrite + new 009 row.
- `ai_docs/todo.md` — any new deferral (import-existing-pack, color-emoji-retrigger) with triggers.

**Deleted:** none.

**Mutated on first launch post-impl:** the maintainer's `projects/dev/app_state.json` (Telegram creds
stripped, `telegram_default_pack` possibly set) + a new `app_data_dir()/integrations.json`. Committed
as part of manual verification.

## Manual verification (maintainer, with a real bot — the agent CANNOT do this; headless box, no
window, live Telegram round-trip)

A real UX gap at any step is a FAIL, not pass-with-caveat.

0. **BotFather guide accuracy (R2 finding G):** the inline guide must reflect that `/newbot` is a
   **two-prompt** flow (choose a display name, THEN a username ending in `bot`), and that the user
   presses Start on **their own bot**, not on @BotFather. The bot username for the set-name suffix is
   captured automatically (`get_me`), so the guide need not mention it.
1. **Fresh bot creation guide:** Settings → Integrations → Telegram shows a short inline step list
   ("1. Message @BotFather → /newbot … 2. Paste token below 3. Open your bot, press Start 4. click
   Connect"). Readable without external docs.
2. **Token + Connect (happy path):** paste a valid token, open the bot in Telegram, press Start,
   click **Connect**. Expect: status flips to "Connected as @<you> ✓"; `auth_state` → AUTHED;
   `integrations.json` now has `bot_token`, `user_id`, `user_username`, `bot_username`. No freeze
   during the network call (drag the window — frames keep drawing).
3. **Connect with no Start:** valid token, but do NOT press Start first → click Connect → a clear
   "No message received — open @<bot> and press Start" terminal message. Not a freeze, not a silent
   stuck spinner.
4. **Bad token:** paste garbage → Connect → "error" status with a readable message; restore.
5. **Create pack + first sticker (the load-bearing path):** "+ New pack" → title "ShaderBox" →
   render a node → pick an emoji from the grid → **Add as sticker**. Expect: the set
   `shaderbox_by_<botusername>` is auto-created, the sticker lands with the chosen emoji, and it
   appears in the maintainer's **@Stickers** as a set the *user* owns. Progress events:
   preparing → uploading → done.
6. **Add second sticker to same pack; switch pack default:** add another → appears in the same set;
   "+ New pack" a second pack, add to it → project's default now points at the second; reopen the
   project → dropdown pre-selects the second (Decision 2).
7. **Replace + delete** an existing sticker (today's verbs) — unchanged behavior, plus the emoji of
   a replace is the currently-picked one.
8. **Emoji picker UX:** opens as a popup; categories in Telegram order (Smileys first); search by
   name narrows the grid; glyphs are monochrome but render (not boxes); selecting closes + updates
   the share control; the *uploaded* sticker shows the emoji in color in Telegram.
9. **Global creds survive project switch:** Ctrl+O another project → Telegram still "Connected as
   @<you>" (creds are global now, Decision 3) — NO re-Connect needed (contrast feature 001 which
   re-authed per project). The pack dropdown shows the global pack list; the default pre-selects per
   the new project's pointer (empty → first/none).
10. **Migration roundtrip:** start from a `projects/dev/app_state.json` that still has the OLD shape
    (`exporter_settings.telegram.bot_token` + `.user_id` + `.sticker_set_name`). Launch → creds
    lifted into `integrations.json`; `sticker_set_name` seeded as a pack + project default. Ctrl+S →
    project `app_state.json` no longer carries the token/user_id; `integrations.json` has them;
    relaunch → still Connected. Negative: a stray unknown key still trips nothing fatal (defaults +
    log).
11. **Stub rows:** Settings → Integrations shows YouTube + X greyed with "Coming soon"; not
    clickable; the share-tab exporter list does not offer them as active targets.
12. **`make check`** green (ruff + pyright, 0 new errors). **`make smoke`** still passes (popup mutex
    invariant must include the new emoji-picker boolean).
13. **Shutdown drainage** unchanged (feature 001 Decision 16) — close mid-upload still drains/abandons
    within 5s.

## Open questions for the user

*(All resolved at plan-lock 2026-05-23.)*
1. **Settings vs share split for the pack list.** RESOLVED: pack management on the **share tab**;
   Settings holds only creds/identity (Decision 8).
2. **`getUpdates` offset handling.** RESOLVED: read-only peek (no offset advance) — default (a).
3. **Emoji default** — RESOLVED: 🎨.

## Review history

### Pre-impl review (2026-05-23) — 2 adversarial reviewers in parallel

R1 (correctness/conventions, anchored to the hard rules + current code) and R2 (Telegram API +
user-flow, anchored to PTB source) ran in parallel. The agent verified the riskiest API claims
directly from source before accepting findings (`getUpdates offset=-1`, `effective_user`, the smoke
2-popup hardcode). **No false positives** — both reviewers were source-grounded; one R2 finding (a
blocking confirm-identity modal) was downgraded to "show @username prominently" (personal single-user
bot — a modal is friction). All findings folded into the decisions above:

**Blockers fixed inline:**
1. `app_data_dir` import cycle → extract `paths.py` (Decision 11).
2. `current_settings()` cred contract / double-write → returns `{}`; Telegram persists nothing
   per-project; project json becomes non-authoritative (Decision 12) — also completes the migration
   strip cleanly.
3. Migration lift ordering in `App._init` → pinned: `load_and_migrate` → `IntegrationsStore.load` →
   `_lift_telegram_creds` (persists + sets AUTH=UNCONFIGURED) → `set_integrations` → `rebind`
   (Decision 9).
4. `make smoke` mutex not generalized → rewrite `_check_invariants` to ≤1-popup; add `scripts/smoke.py`
   to Files-touched (Decision 16).
5. Link path missing bot-username capture → `"link"` job calls `get_me()` for `bot_username`
   (Decision 6a).
6. Link used `message.from` → use `Update.effective_user` (Decision 6c, PTB `_update.py:480`).
7. Link offset wrong → `get_updates(offset=-1, limit=1)` for newest (Decision 6b, PTB `_bot.py:4542`).

**Majors fixed inline:** dangling default-pack pointer fallback + slug-collision dedupe + AUTHED-and-
user_id gate + stale-error→re-Connect mapping (Decision 12b); migrated user_id reaches UNCONFIGURED
not AUTHED, re-Connect required (Decision 9).

**Minors fixed inline:** create-path `InputSticker(format=VIDEO)` explicit, no `sticker_format` kwarg
(Decision 13); emoji carried as whole grapheme, never per-char (Decision 7); emoji font added in
`__init__` + rasterized push_font size (Decision 7); empty-slug fallback + clamp-the-stem (Decision 5);
verified-facts evidence attribution corrected (the "user must have interacted" precondition is a
server-side rule, not in PTB source); BotFather guide is a two-prompt flow (verification step 0).

**Items NOT escalated:** neither reviewer returned "should not land"; all addressable inline. Single-
threaded worker already serializes racing adds (R1 withdrew that concern). `Stickerset_invalid`
substring sniff is pre-existing + reused (acceptable).

### Round 2 (convergence, 2026-05-23) — 1 reviewer

Verified all 7 round-1 blockers genuinely RESOLVED (not reworded), traced against real `app.py` /
`smoke.py` / PTB source. No new contradictions except two narrow items the round-1 fixes left, both
closed: (1) Decision 12b dangling-pointer re-add now pins `title=set_name`; (2) verification step 2
now names `user_username`/`bot_username` (was ambiguous `username`). Verdict: PASS. Ready to implement.

### Post-impl review (2026-05-23) — 2 reviewers + convergence round

R1 (spec-fidelity, anchored to the 17 decisions): PASS — every decision COVERED with an implementing
file:symbol. R2 (correctness/conventions): FAIL with real bugs. Round-2 convergence reviewer verified
all fixes landed: PASS.

**R2 blockers/bugs fixed (commit `6479b34`):**
- **CRITICAL — `in_flight` stuck True after Connect** bricked all sticker buttons (`begin_auth` →
  `_enqueue` set it True; link/auth events never cleared it). Fixed: `_apply_event` clears `in_flight`
  on `_LinkEvent` + `_AuthEvent`. Live-verified against the real bot (no-message error path + the flag
  clears).
- **HIGH — illegal `# type: ignore`** in `share.py` (suppressed a real `Optional` access, outside the
  sanctioned allowlist). Fixed: bind `[e for e in registry.all() if e.is_available]` (non-Optional).
- **HIGH — `make check` failed** on the vendored `emoji-test.txt` (trailing-whitespace hook would
  corrupt the canonical Unicode file). Fixed: excluded it in `.pre-commit-config.yaml`.
- **MEDIUM — delete-refresh cross-thread read** of `active_pack_set_name`. Fixed: `_handle_delete`
  takes `pack_set_name` from the job.
- **LOW — success-then-refresh-error toast**, dead `_AUTH_SENTINEL`, missing annotations. Fixed
  (`_safe_refresh` swallows post-success refresh errors; sentinel removed; annotations added).

No false positives this round (both reviewers source-grounded; the one downgrade — a blocking
confirm-identity modal — was rejected pre-impl as friction for a personal bot). Convergence: PASS,
ready for manual verification. `make check` + `make smoke` green.

### Live manual-verification debugging arc (2026-05-23)

Maintainer ran the real round-trip (bot `@ShaderBox_bot`, account `@kaktebyazvaly`); a series of
UX/correctness bugs surfaced and were fixed + live-verified against the real bot. Each landed as its
own commit on `dev`:
- Emoji rendered as "?" → drawn in the emoji font (`share.py` pushes `app.font_emoji`).
- Buttons clipped off-panel → full-width vertical layout in `draw_config_ui` + `_draw_pack_controls`
  + `_draw_progress`.
- Invalid pack names (`asdfsadf` with no `_by_<bot>` suffix) → root cause was acting before a real
  Connect; `_is_connected()` now requires `bot_username` (only a successful `get_me` sets it).
- **Dropped the credential migration entirely** (user: "no backward-compat") — no legacy lift; creds
  come only from a real Connect. Killed the "stale state looks connected" failure mode.
- Stuck "Working… 0%" → `in_flight` set by the background refresh that never cleared it; now only
  `add/replace/delete/delete_pack` jobs gate `in_flight`.
- Re-Connect every launch → connection (token + user_id + bot_username) persists to
  `integrations.json` the instant Connect succeeds; `rebind` restores AUTHED from it with no network
  call. Packs persist on create/delete too.
- Empty grid after restart → `set_default_pack` enqueues a refresh on launch.
- Bot DMs the pack link after each add (`_notify_user` via `send_message`; the user pressed Start so
  the bot may DM them). Link also shown copyable in-panel.
- Delete a sticker (no render needed — was wrongly gated behind an artifact) + delete a whole pack
  (armed confirm → `delete_sticker_set`). Dropped the dead per-frame preview canvas.

### Post-impl review wave 2 (2026-05-23) — 2 reviewers + convergence — PASS

Run against the full post-debugging state (everything above). R1 (correctness/threading), R2
(Telegram API/UX-flow). `make check` green, no convention violations, no corruption/crash/threading
bugs. Findings fixed inline + convergence-verified PASS:
- MEDIUM — stale/revoked token showed green "Connected" while emitting a raw error → `_with_bot`
  catches `InvalidToken`/`Forbidden` (around `initialize()` too) and flips `auth_state` to ERROR.
- MEDIUM — `_select_pack` rendered the previous pack's slots (wrong-pack Delete/Replace target for a
  frame) → `_clear_grid()` (slots + selected_index + last_progress) runs before the refresh; all
  pack-change paths route through it.
- LOW — `set_default_pack` orphan re-add now `save()`s; `_delete_pack` refreshes the newly-active
  pack; `Stickerset_invalid` match is case-insensitive (PTB capitalize() coupling).
Both reviewers' "selected_index out of range" / "GL off worker thread" / "in_flight stuck" concerns
were checked and found already-correct (drain-before-draw + n_slots==0 guard; render-thread-only GL;
terminal events on every path). `_current_async_task` unlocked-write + wrong-user `getUpdates`
capture noted as accepted low-risk for a single-user tool.
