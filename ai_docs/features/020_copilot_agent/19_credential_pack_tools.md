# 020·19 — Inline-credential gate + Telegram connect/pack-CRUD tools

The copilot can now render + publish (020·18), but the *setup* is a manual scavenger hunt: when
Telegram isn't connected the agent can only TELL the user to go to Settings, then to the Share tab to
pick a pack. This wave wires the **whole Telegram path conversationally**: the agent pastes the bot
token via an **inline secret-input gate** (the `GateKind.CREDENTIAL` stubbed since 020·17, now built),
auto-kicks the link flow, and does full **pack CRUD** (list / select / create / delete) — every
mutation behind a confirm. The token is **redacted in copilot history** (first 4-8 chars) so the full
secret never sits in the LLM context, the persisted conversation, or anywhere but the live
`IntegrationsStore`.

Cold-context: read `17_gate_ui.md` (the CONFIRM gate body this extends), `18_render_publish_tools.md`
(the precheck seam + the await-loop + the exporter-public-API pattern this mirrors), and
`11_capability_wave_spec.md §7.2/§F5` (the original CREDENTIAL-widget sketch). The maintainer
triggered this directly after live-testing 020·18: "wire the whole path, even secret key pasting … a
blocking interactive widget which accepts a key and sets it into settings … strip it to the first
4-8 chars … Telegram key here [not OpenRouter]." Pack CRUD: "list, select, create, delete, everything
the user can do … hide all the mutation tools under the confirmation gate."

---

## Goal

New copilot tools (all `category="telegram"`, eager):

- **`set_telegram_token`** — opens a **masked inline input** in the chat (a CREDENTIAL gate); the user
  pastes the BotFather token, hits Save; the token is set into `IntegrationsStore` + the connect/link
  flow auto-kicks; returns a **redacted** confirmation (`token set (1234…) — …`). The full token never
  reaches the LLM history / the persisted card / the trace.
- **`list_telegram_packs`** — read the user's saved packs (no gate, no await).
- **`select_telegram_pack(set_name)`** — make a pack the default + load its stickers (gated).
- **`create_telegram_pack(title)`** — register a new pack + select it (gated). (A Telegram pack is
  created lazily on first sticker upload, so this is a local register — the real creation rides
  `publish_telegram`.)
- **`delete_telegram_pack(set_name)`** — delete a pack from Telegram + drop it locally (gated; awaits
  the delete job's terminal progress).

After this wave the agent can drive: *"connect my Telegram bot"* → paste token inline → *"make a pack
called Foo"* → *"publish this shader to it"* — entirely in chat, with confirms on every mutation and
the secret never logged.

---

## Out of scope (each with a trigger)

- **Credential gate for OpenRouter / any non-Telegram secret.** The OpenRouter key drives the LLM
  inference itself — it's set manually before the chat can run (the existing pre-turn `unconnected_gate`
  in `copilot_chat.py`), so an in-chat widget for it is circular. This wave's CREDENTIAL gate is
  generic infrastructure but only `set_telegram_token` uses it. **Trigger:** a future integration with
  a pasteable secret the user would naturally provide mid-conversation.
- **YouTube connect via the credential gate.** YouTube uses a browser OAuth flow (no pasteable
  secret), so the inline-text widget doesn't fit; it stays a guided handoff (open Settings). **Trigger:**
  YouTube ever exposes a paste-a-token path.
- **Awaiting the sticker-grid refresh after select.** `select_telegram_pack` sets the default + enqueues
  a best-effort refresh (the grid populates in the Share tab); the copilot does NOT await the refresh
  (it carries no terminal progress). **Trigger:** the agent needs to *read back* a pack's sticker list
  conversationally (then add a `list_telegram_stickers` tool that awaits the `_StickerListEvent`).
- **Removing individual stickers / set-emoji via the copilot.** The maintainer's ask was pack-level
  CRUD; per-sticker ops stay UI-only. **Trigger:** a user asks the agent to remove one sticker.
- **A timeout-then-retry UX on the credential gate.** The gate waits indefinitely for Save/Cancel
  (like the CONFIRM gate). **Trigger:** a maintainer finds an abandoned credential card wedges a turn.

---

## Design decisions (numbered — lock-in only)

### The CREDENTIAL gate (generic infra)

1. **`gate_kind` is a NEW field on `ToolDefinition`, orthogonal to `gate_policy`.** `gate_policy`
   (NONE/BULK/ALWAYS) decides WHETHER to gate; `gate_kind` (CONFIRM/CREDENTIAL) decides WHICH widget.
   `set_telegram_token` is `gate_policy=ALWAYS` + `gate_kind=CREDENTIAL`; every other gated tool stays
   the default `gate_kind=CONFIRM`. A tool also carries `secret_field: str` — a **marker/label** (e.g.
   `"telegram_bot_token"`) for the gate request + a grep anchor, NOT a functional store key (the
   handler sets the store via its capability, Decision 5; the widget doesn't index `IntegrationsStore`
   by it). `build_gate` gains a `registry` param (it's pure today) so it can read the tool's
   `gate_kind`/`secret_field` via a new `registry.definition_for(name)`: a CREDENTIAL tool →
   `GateRequest(kind=CREDENTIAL, prompt=<from _GATE_PROMPTS>, secret_field=…)`; everything else →
   the existing `GateRequest(kind=CONFIRM, …)` branch BYTE-UNCHANGED (default `gate_kind=CONFIRM` falls
   through to today's code, so delete_node/publish are untouched). The call site
   (`req = build_gate(registry, tc.name, args)`) has `registry` in scope already.

2. **`GateResponse` gains a dedicated `secret: str` field — NOT a reuse of `option`.** A secret silently
   riding in a field named `option` is exactly how secrets leak (a future log line dumps `option`
   thinking it's "Yes/No"). An explicit `secret` field is self-documenting, greppable, and the type
   makes the redaction discipline auditable. `GateResponse` is frozen, so the UI constructs it once with
   the typed secret (`GateResponse(approved=True, secret=<typed>)`).

3. **`Message` gains `gate_kind: GateKind` + `gate_input: str` (UI-only).** The CREDENTIAL gate reuses
   `role="pending_action"` (so `_open_gate_card`/`_resolve_open_gate_card` find it — no new
   `MessageRole`). The card needs to know which widget to draw, so `pump_events` stamps the materialized
   `pending_action` Message with `gate_kind=ev.request.kind`. `gate_input` is the per-card input buffer
   the masked field writes to (a proper field, not a `hasattr`-monkeypatch). The CREDENTIAL widget draws
   ONLY when `not msg.resolved` (a resolved credential card renders just its redacted `text` echo, never
   re-instantiates the input). **Neither the typed secret nor `gate_input` is ever persisted** —
   `gate_input` is excluded from `_MessageModel`; `text` is the prompt (then a redacted echo), never the
   secret. `gate_kind` IS persisted (harmless, keeps the resolved-card model honest) — added to
   `_MessageModel` as a loose `str` (default `"confirm"`, like `role`, so future enum members survive an
   old build), which **bumps `ConversationStore._VERSION` 2 → 3** (an old file lacking `gate_kind` loads
   via the field default; fail-soft per feature 022).

4. **The masked widget reuses `ui_primitives.labeled_text_input(..., password=True)`.** Per the
   all-UI-through-`ui_primitives` rule, `_draw_pending_action`'s CREDENTIAL branch draws the existing
   password input (the same primitive the Share-tab Bot-token field uses) + a **Save** (primary) and
   **Cancel** (ghost) button. Save calls a new `CopilotSession.answer_gate_credential(secret)`; Cancel
   calls the existing `answer_gate(approved=False)` (a declined credential = the user backed out).

5. **The secret flows worker→back via a SEPARATE channel — never through `args`/trace/history.** The
   credential threading, end to end:
   - Worker hits the gate (`gate.ask`) for a CREDENTIAL tool → blocks.
   - UI draws the masked input; Save → `answer_gate_credential(secret)` → `gate.answer(GateResponse(
     approved=True, secret=secret))`. The resolved card text appends ONLY a redacted echo
     (`You provided: 1234…`), never the full secret → so the persisted card (Decision 3) is clean.
   - Worker unblocks with `resp.secret`. The agent loop, for a CREDENTIAL tool, calls
     `registry.execute(name, args, secret=resp.secret)` — a new optional `secret` param that `execute`
     forwards to a **credential-aware handler** (`CredentialToolHandler = Callable[[dict, str], …]`),
     NOT merged into `args`. So `args` (which the trace + the debug log print) NEVER contains the secret.
     **`execute` dispatches on `tool.gate_kind is GateKind.CREDENTIAL`** (the SAME field `build_gate`
     reads — no parallel discriminator): credential tool → `handler(args, secret)`; every other tool →
     the existing `handler(args)`. The credential handler's returned `payload` MUST be `None` or
     secret-free (payload is traced — a secret in payload would leak).
   - The handler sets the secret via its capability (`caps.set_telegram_token(secret) -> TelegramConnectResult`)
     and returns a **redacted** `msg` — that redacted `msg` is the ONLY thing that reaches LLM history
     (`messages.append(_tool_message(tc.id, msg))`) and the trace's `result=` field.
   - **Net:** the full secret exists only transiently in `GateResponse.secret`, the `gate_input` buffer,
     and inside `caps.set_telegram_token`'s call into `IntegrationsStore`. It is in zero log lines, zero
     trace events, zero history messages, zero persisted files (beyond the live `integrations.json`,
     which already holds it in cleartext — the existing posture, tracked in `todo.md`).

6. **Redaction helper, one home.** A `mask_secret(s: str) -> str` (e.g. `s[:6] + "…"` for len>6, else
   `"…"`) lives next to the gate/credential code; the handler `msg` and the card echo both use it. 6
   chars: a Telegram token's prefix is `<bot_id>:` (the bot id is public-ish, not the secret part), so
   showing ~6 lets the agent/user distinguish tokens without exposing the secret tail. (4-8 per the
   maintainer; 6 is the midpoint.)

### Connect / link (the auto-link step)

7. **`set_telegram_token` sets the token THEN auto-kicks `begin_auth()` + awaits a fresh result via a
   `LINKING` floor.** Per "token paste + auto-link". The await reuses the publish-await shape but polls
   `exporter.status().auth_state` (connect pushes typed `_AuthEvent`/`_LinkEvent`, NOT `ExportProgress`,
   so there's no `last_progress` terminal to poll). **The freshness problem the publish-await solves
   with an object-identity baseline does NOT transfer to an enum** — a stale `AUTHED`/`ERROR` from a
   prior attempt would make a level-poll (`until auth_state == AUTHED`) return instantly with the wrong
   result. **Fix: a transient floor.** Add `AuthState.LINKING`; `begin_auth()` sets
   `auth_state = LINKING` synchronously BEFORE enqueueing the link job. The connect-await then waits for
   `auth_state` to LEAVE `LINKING` (→ `AUTHED` = success, read `bot_username`; → `ERROR` = read
   `auth_message`), which can only happen when the worker's `_LinkEvent`/`_AuthEvent` is applied. The
   await loop, each iteration, runs ONE bridge op `run_on_main(lambda: (exporter.update(None),
   exporter.status())[1])` — the `update(None)` is MANDATORY (it drains the typed-event queue into
   `auth_state`; a poll that only reads `status()` never sees the transition and spins to timeout) —
   then checks `auth_state != LINKING`. Bounded by `telegram_connect_timeout_s` (~30s) and
   `is_cancelled()`-aware (same teardown safety as the publish-await). `AuthState.LINKING` is a new
   enum member; the UI's existing `== AUTHED` / `== ERROR` equality checks are unaffected (a card mid-
   link just isn't "connected" yet), and `is_connected()` (= `AUTHED and identity`) stays false during
   LINKING — correct.

8. **The link CANNOT be fully auto — the user MUST DM the bot first; the tool says so.** `_do_link`
   calls `bot.get_updates(offset=-1, limit=1)` ONCE — it reads the *most recent* message the user
   already sent the bot; it does NOT wait for one (Telegram won't let a bot DM a user who hasn't started
   it). So if the user hasn't opened the bot + pressed Start, the link returns "No message received".
   `set_telegram_token`'s result handles both: on AUTHED → `token set (1234…) and linked to @bot`; on
   the no-message error → a guided result: `token set (1234…). Now open @bot in Telegram, press Start
   (or send any message), then tell me to connect.` plus a separate **`telegram_connect`** tool the
   agent calls once the user confirms they've messaged the bot (re-runs `begin_auth` + the await). So
   the realistic flow: paste token → (agent: "go press Start on @bot") → user does it → "connect" →
   AUTHED.

### Pack CRUD

9. **Pack CRUD is render-side synchronous (one bridge op each); only delete awaits a network terminal.**
   Telegram has no "create pack" API — a pack is created on the first sticker upload — so
   `create_telegram_pack` just registers a `PackEntry` locally + selects it (the real creation rides
   `publish_telegram`). `select`/`set_default` are local state writes (+ a best-effort refresh, not
   awaited — Out of scope). `list` is a plain read of `self._tg.packs`. Only `delete_telegram_pack`
   has a real Telegram API call (`delete_sticker_set`) — it removes the pack locally immediately
   (matches the UI) AND awaits the delete job's terminal `ExportProgress` (the publish-await loop,
   reused) so the agent can report success/failure. All mutations (`select`/`create`/`delete`) are
   `gate_policy=ALWAYS` CONFIRM-gated; `list` is ungated.

10. **All pack-CRUD precheck on connection** (a guided handoff "Telegram isn't connected — set your
    token first") so an unconnected pack op never pops a confirm (the 020·18 precheck pattern).

### Exporter public API

11. **`TelegramExporter` gains public wrappers; the private bodies stay.** `_create_pack`/`_delete_pack`/
    `_select_pack` are UI-private today. Add public `create_pack(title)` / `delete_pack(set_name)` /
    `select_pack(set_name)` / `list_packs() -> list[PackEntry]` / `set_token(token)` (the token setter +
    `store.save()` the UI does inline). They delegate to the existing private bodies (no behavior change
    to the Share-tab path). `begin_auth` / `current_default_pack` / `is_connected` / `status` are already
    public (020·18). These are Telegram-concrete (reached via `isinstance` at the App call sites, like
    the existing `_copilot_telegram_has_default_pack`), NOT hoisted to the `Exporter` ABC (the
    generic-seam rule: only promote when ≥2 exporters share it — these are Telegram-only).

### Seams

12. **New capability fields + value types** (`capabilities.py`, all GL-free frozen): `set_telegram_token:
    Callable[[str], TelegramConnectResult]`, `telegram_connect: Callable[[], TelegramConnectResult]`,
    `list_telegram_packs: Callable[[], list[TelegramPackInfo]]`, `select_telegram_pack`/
    `create_telegram_pack`/`delete_telegram_pack: Callable[[str], TelegramOpResult]`, plus
    `telegram_token_set: Callable[[], bool]` (a precheck read). Value types `TelegramConnectResult(ok,
    error, bot_username)`, `TelegramOpResult(ok, error, set_name)`, `TelegramPackInfo(title, set_name)`.

13. **A new `tools/telegram.py` module** (parallel to `publish.py`): the `telegram_tools(caps)` factory
    — the 5 tool definitions + args models + handlers + prechecks. `set_telegram_token`'s handler is the
    credential-aware shape (`(args, secret)`); the rest are normal. `build_registry` becomes
    `[*shader_tools, *publish_tools, *telegram_tools]`.

14. **System prompt gains a TELEGRAM section** (`prompt.py`): describes the connect flow (set token →
    user presses Start on the bot → connect), the pack tools, and that the agent should walk the user
    through connecting rather than dumping them in Settings. Notes that pasting the token opens a secure
    input (the agent never sees the token itself).

---

## Files touched

- **`shaderbox/copilot/gate.py`** — `GateResponse` gains `secret: str = ""`. (`GateKind.CREDENTIAL` +
  `GateRequest.secret_field` already exist.)
- **`shaderbox/copilot/state.py`** — `Message` gains `gate_kind: GateKind` (default CONFIRM) + `gate_input:
  str = ""` (UI-only).
- **`shaderbox/copilot/tools/base.py`** — `ToolDefinition` gains `gate_kind: GateKind = GateKind.CONFIRM`
  + `secret_field: str = ""`; a `CredentialToolHandler` type.
- **`shaderbox/copilot/tools/registry.py`** — `definition_for(name)`; `execute` gains an optional
  `secret: str` param forwarded to a credential handler; `build_registry` adds `telegram_tools`.
- **`shaderbox/copilot/agent.py`** — `build_gate` dispatches on the tool's `gate_kind`; the gate branch,
  on a CREDENTIAL resolve, calls `execute(name, args, secret=resp.secret)`; a `set_telegram_token`
  `_GATE_PROMPTS` entry ("Paste your Telegram bot token below."); `mask_secret` lives here or in a small
  `redact.py` leaf.
- **`shaderbox/copilot/session.py`** — `answer_gate_credential(secret)`; `pump_events` stamps
  `gate_kind` onto the materialized card; the resolved-card echo is redacted for CREDENTIAL.
- **`shaderbox/copilot/persistence.py`** — `_MessageModel` gains `gate_kind: str = "confirm"`;
  `_VERSION` 2 → 3; `gate_input` is NOT serialized.
- **`shaderbox/exporters/base.py`** — `AuthState` gains a `LINKING` member (the connect-await floor,
  Decision 7).
- **`shaderbox/widgets/copilot_chat.py`** — `_draw_pending_action` dispatches on `msg.gate_kind`; a
  CREDENTIAL branch draws `labeled_text_input(password=True)` + Save/Cancel.
- **`shaderbox/copilot/capabilities.py`** — the new fields + value types (Decision 12).
- **`shaderbox/copilot/tools/telegram.py`** — NEW (Decision 13).
- **`shaderbox/copilot/prompt.py`** — the TELEGRAM section (Decision 14).
- **`shaderbox/exporters/telegram.py`** — the public wrappers (Decision 11); `begin_auth` sets
  `auth_state = LINKING` before enqueueing (Decision 7). The Share-tab UI is unaffected (a transient
  LINKING just reads as not-yet-connected, same as before the link returns).
- **`shaderbox/app.py`** — the `_copilot_*` closures (set-token/connect/pack-CRUD) + the connect-await +
  the delete-await; bind them in `_build_copilot_capabilities`.
- **`scripts/copilot_render_check.py`** (or a new `copilot_credential_check.py`) — a headless redaction
  test: a stub secret through the handler + the card echo never yields the full secret; the registry
  builds with the new tools + `set_telegram_token` is `gate_kind=CREDENTIAL`. *(Feature 031: the
  redaction checks now live in pytest — `tests/test_credential_redaction.py` — and the structural
  CREDENTIAL invariants in `tests/test_tool_registry.py`; both check scripts deleted.)*
- **`scripts/copilot_gate_check.py`** — the `_stub_caps` grows the new capability fields.
- docs: `roadmap.md` row/banner, `dev_flow.md` module map, `todo.md` (resolve the CREDENTIAL-widget
  deferral; the pack-CRUD deferral collapses).

---

## Manual verification

### Headless (the agent runs these)

- **H1 — registry + gate-kind.** `build_registry` includes the 5 telegram tools; `set_telegram_token`
  is `gate_kind=CREDENTIAL` + `secret_field="telegram_bot_token"`; the pack-mutation tools are
  `ALWAYS`+CONFIRM; `list` is ungated. `build_gate` for `set_telegram_token` returns a CREDENTIAL
  `GateRequest`.
- **H2 — redaction.** Drive `mask_secret` + the credential handler + `answer_gate_credential` with a
  fake token; assert the handler `msg`, the resolved-card `text`, and a `ConversationStore` round-trip
  of the card all contain ONLY the masked prefix, never the full token. (Pure logic, no network.)
- **H3 — prechecks.** Each pack tool's precheck hands off when not connected; `set_telegram_token` has
  no precheck (it's the entry point).
- `make smoke` still passes (App constructs with the new closures).

### Maintainer, live (`make run`)

1. Ask the copilot to connect Telegram with no token → the chat shows a **masked input** card; paste a
   real BotFather token, Save → the agent reports `token set (…)` and either links (if you'd messaged
   the bot) or tells you to press Start on @bot then say "connect".
2. Press Start on the bot, say "connect" → AUTHED, the agent confirms the linked username.
3. "list my packs" / "make a pack called Foo" (gate → created) / "publish this shader to it" (the
   020·18 publish, now to the agent-created pack) / "delete the Foo pack" (gate → deleted on Telegram).
4. Verify the conversation.json + the trace + the console log contain NO full token anywhere (grep the
   files for the token's tail).
5. Cancel a credential card mid-entry (the turn ends cleanly); Stop during the connect-await.

---

## Open questions for the user (maintainer, on return)

1. **Redaction length: I chose 6 chars** (`token[:6]…`) within your 4-8 range. OK, or prefer fewer?
2. **The two-step connect** (paste token → "press Start on the bot" → "connect") — unavoidable because
   Telegram won't let a bot DM a user who hasn't started it. The agent guides it. Acceptable?
3. **`delete_telegram_pack` deletes the pack on Telegram for real** (irreversible, gated). Confirm that's
   the intended scope (vs. only dropping it from ShaderBox's local list).

---

## Review history

### Pre-implementation review (2 adversarial agents: secret-leak audit + design/blast-radius)

**Round 1: 1 PARTIAL (leak audit), 1 PARTIAL (design).** The secret-leak audit traced the token's full
lifecycle and found **no leak path** — every point the secret touches is the live `IntegrationsStore`
(accepted cleartext posture) or a redacted echo; the 3 enabling invariants (mask in the card echo;
keep the secret out of `args`; the handler `payload` is secret-free) are all stated. The real findings,
all in the connect-await + seam specs, fixed before impl:
- **[BLOCKER] The connect-await had no freshness guard.** `auth_state` is an enum (no object identity
  like the publish-await's `last_progress` baseline), so a stale `AUTHED`/`ERROR` from a prior attempt
  made a level-poll return instantly with the wrong result. **Fixed:** Decision 7 adds an
  `AuthState.LINKING` floor — `begin_auth` sets it before enqueueing; the await waits for `auth_state`
  to LEAVE `LINKING`.
- **[BLOCKER] The connect-await must pump `exporter.update(None)` inside the bridge poll** — `auth_state`
  only transitions when `update()` drains the typed-event queue; a status-only poll spins to timeout.
  **Fixed:** Decision 7 makes the `update(None)` mandatory (mirrors the publish-await).
- **[MEDIUM] `execute`'s credential-dispatch discriminator was unspecified.** **Fixed:** Decision 5
  branches on `tool.gate_kind is GateKind.CREDENTIAL` (the same field driving `build_gate` — no parallel
  discriminator).
- **[MEDIUM] `build_gate` is pure today; reading `gate_kind` needs a `registry` param + a `_VERSION`
  bump for the persisted `gate_kind`.** **Fixed:** Decision 1 states the `build_gate(registry, …)`
  signature + the byte-unchanged CONFIRM fall-through; Decision 3 bumps `_VERSION` 2 → 3 and persists
  `gate_kind` as a loose `str`.
- **[LOW] `secret_field` was framed as a store key but is just a marker; the credential card must reuse
  `role="pending_action"` and draw the input only when unresolved.** **Fixed:** Decisions 1 + 3.
- **[REFUTED] delete-pack await would hang** — verified `_handle_delete_pack` DOES push a terminal
  `ExportProgress`, so the delete-await (reusing the publish-await) terminates correctly. Kept as-is.

Convention fidelity, blast radius, and the synthesis divergences (dedicated `GateResponse.secret` not
`option`-reuse; secret-to-handler not agent-sets-store) all PASSed both reviewers — the divergences are
the leak-prevention crux and hold up.

**Round 2 (re-spawn against the patched spec): PASS.** All fixes verified resolved against code: no
exhaustive `match` on `AuthState` (so `LINKING` is safe); `is_connected` stays false during LINKING;
`LINKING` is in-memory only (`rebind` recomputes it on launch, never persisted/stuck); the connect-await
mirrors the publish-await's `update(None)+status()`/`is_cancelled()`/deadline verbatim; the
`_VERSION` 2→3 + `gate_kind: str` default is the EXACT precedent 020·17 set when it added `recover`
(1→2); the `(dict)` vs `(dict, str)` handler union narrows cleanly via `cast` (the sanctioned pattern,
already used in `persistence.py` — NOT a `# type: ignore`). Impl note carried forward: use `cast` for
the handler-union, set `auth_state = LINKING` in `begin_auth` (the Share-tab Connect button reads it as
not-yet-connected, no regression). Converged.

### Post-implementation review (3 adversarial agents: secret-leak re-audit, code-correctness, architecture/spec-fidelity)

**All three converged PASS on code + security.** Outcomes:
- **[Secret-leak re-audit — PASS]** Traced the IMPLEMENTED secret lifecycle (gate_input buffer →
  `GateResponse.secret` → `execute(secret=)` → handler → `caps.set_telegram_token` → store): every point
  resolves to the live `IntegrationsStore` (accepted cleartext posture) or a `mask_secret` echo. No path
  reaches LLM history / `conversation.json` / the trace / a log line. H4 exercises the real leleak paths
  with tail-substring assertions + a `ConversationStore` round-trip and passes; an independent real-
  `TraceLog` write confirmed the secret is absent from the trace file too.
- **[Code-correctness — PASS]** The LINKING-floor connect-await is sound (a timeout leaves LINKING but
  self-heals on the next pumped `update()`; a `telegram_connect` retry re-sets the floor + re-enqueues —
  no stuck-LINKING). The delete-await baseline-by-identity is correct despite `_clear_grid` nulling
  `last_progress` between capture and the terminal. Credential dispatch, the byte-unchanged CONFIRM path,
  `AuthState.LINKING` (no exhaustive-match consumers), and the cancel/teardown safety all hold.
  **One LOW finding fixed:** `needs_start` was a fragile cross-file substring (`"no message"` had to keep
  matching the exporter's literal) — replaced with a shared `telegram.NEEDS_START_ERROR` constant
  imported by both sites (exact match, compile-coupled).
- **[Architecture/spec-fidelity — PARTIAL→addressed]** All 14 decisions LANDED; all conventions PASS
  (no suppressions — the handler union uses `cast`; the new exporter methods stay off the ABC, reached
  via `isinstance`; UI through `labeled_text_input`; no secret in any log). The PARTIAL was the stale
  docs (the `gate.py` "CREDENTIAL is a type-only stub" comment + the `todo.md`/`roadmap.md` deferrals) —
  the `gate.py` comment is fixed; the rest is the `/sanitize` closing wave (sequenced, not skipped).

No fabricated late-round findings; the loop converged.
