# 020 · 21 — First-class chat widgets (structured tool results + inline config panels)

The copilot's tool results become STRUCTURED OUTCOMES the UI renders as first-class chat widgets; the
agent reports the FACT and is TOLD which widget was shown — it never receives raw URLs/paths. Inline
config panels (YouTube connect) reuse the EXACT Settings widget set + a Cancel button. Driven by the
maintainer's voice notes (tg foo id 1830 + 1831) and a deep research pass (`wpggn6pt9`).

THE PRINCIPLE (maintainer, verbatim intent): tool results are structured outcomes, not text the agent
echoes. Buttons / links / config panels a tool surfaces are FIRST-CLASS CHAT ENTITIES rendered by the
engine from structured payload. The agent reports "done" and KNOWS a widget was shown (we tell it in
the result). Reduce entropy + surface — REUSE the existing payload channel + the existing
`draw_config_ui`, don't reinvent.

## Goal

1. **Structured result widgets.** A tool returns a terse fact in its model-facing message + a
   structured widget spec in `payload`. The chat renders the widget (e.g. an "Open in YouTube Studio"
   button); the raw URL/path NEVER enters LLM context. The result message TELLS the agent a widget was
   shown so its reply is coherent ("done — there's a button to open it").
2. **Inline config panels.** When an integration isn't connected, the agent surfaces the EXACT Settings
   setup widgets (`exporter.draw_config_ui()`) INLINE in the chat — a proactive setup guide, not a "go
   to Settings" deflection. Plus a **Cancel** button (new — absent in Settings) that aborts the panel;
   on cancel the agent resumes and explains it couldn't proceed.
3. **Open-only link/path buttons** (no auto-copy — `draw_link` is a reuse TRAP, it copies + forces
   https). A web URL opens the browser; a local render path opens the file manager.
4. **Prompt truth fix.** The YouTube "browser sign-in / go to Settings" misdescription is corrected to
   the real flow (the agent now surfaces the inline panel), and render/publish lines stop telling the
   model to echo the path/link (the widget carries it).

## Out of scope (each with a trigger)

- **Selectable chat prose + title-bar-only window drag.** DEFERRED — the maintainer is still deciding
  the mechanism (voice note 1831: "I'll think about how best to do it"). **Trigger:** the maintainer
  settles the selection approach. NOT this wave.
- **Result widgets on edit tools** (a "jump to edited line" button). Edit tools stay `widget=None`.
  **Trigger:** a maintainer asks for in-app-callback result widgets.
- **A widget-kind registry / factory.** A small literal-dispatch on `kind` is enough at 3 kinds.
  **Trigger:** the kind set grows past ~6.
- **Inline YouTube file-PICKER.** The inline panel is PASTE-ONLY (`pfd_block` is a blocking native
  dialog, awkward in the transcript); the file picker stays in Settings. **Trigger:** a maintainer
  wants the file picker inline.

## Design decisions (numbered, lock-in)

### D1 — Two orthogonal mechanisms: non-blocking result widgets vs blocking config gates
The research established these are DIFFERENT and must NOT be merged into one `GateKind`:
- **Result widget = NON-BLOCKING.** A tool succeeds, attaches a widget spec to its result; the agent
  does NOT wait for the user to click. Rides the existing `(ok, msg, payload)` -> `AgentToolCard` ->
  `tool_status` Message channel (which already hides payload from the model).
- **Config gate = BLOCKING.** The YouTube connect panel uses the existing `GateChannel` (the worker
  blocks on `ask()` until the user loads creds + the panel signals done, or Cancel). It's the
  non-secret twin of the CREDENTIAL gate.

### D2 — `ResultWidget` dataclass + one optional field (the spine)
- A frozen `ResultWidget` in `state.py` (next to `RecoverInfo`, the proven precedent), discriminated by
  a `Literal` kind: `kind: Literal["open_url", "open_path"]`, `label: str`, `target: str` (the url, or
  the ABSOLUTE render path — see D3 F1). (`config_panel` is NOT a result widget — it's a gate; see D4.)
- `Message` gains `result_widget: ResultWidget | None = None` (parallel to `recover`, NOT a reuse of
  `gate_kind`). `AgentToolCard` gains `widget: ResultWidget | None = None` (frozen, trailing-defaulted
  after `result` so the frozen dataclass stays backward-compat).
- Thread: tool builds the widget in `payload["widget"]` -> `agent.py` extracts it onto `AgentToolCard`
  -> `session._apply_event` `AgentToolCard` branch carries it onto the SAME `tool_status` Message it
  already builds (independent of the `_attach_recover` trailing-card scan — don't perturb it). The model
  still sees ONLY the terse `msg`.

### D3 — Producers: terse fact + "a widget was shown" (conditional) + structured widget
- `publish.py` result builders (`_publish_result`, `_render_result` — they return the full
  `(ok, msg, payload)` tuple) emit TERSE facts with NO url/path. CRITICAL (pre-impl F-leak): the `msg`
  IS the model-facing string (it lands in LLM history at `agent.py` `_tool_message`) — terse-ify THERE,
  not the session display layer, or the URL stays in context. `tests/test_result_widgets.py` pins that
  the output contains no `r.url`/`r.path`.
- The "a widget was shown" sentence is CONDITIONAL on the widget actually being built: only when
  `target` is non-empty. e.g. with a URL: `"published to YouTube. An 'Open in YouTube Studio' button is
  shown to the user — point them to it in words; you don't have the link."`; with an empty url (upload
  ok but Studio URL missing — a real race, `app.py` reads `prog.url or ""`): `"published to YouTube
  (the Studio link wasn't captured; tell the user to check YouTube Studio)."` and NO widget. So the
  agent never promises a button that isn't there.
- `render_image`/`render_video` -> `open_path` widget; `target` is the ABSOLUTE render path
  (`RenderResult.path` is already absolute — NOT relative, NOT re-anchored; accept-stale per D6).
  `publish_telegram`/`publish_youtube` -> `open_url` widget. A `grep` over the registry confirms no
  other tool embeds a url/path it should surface structurally.

### D4 — Inline config gate: reuse `draw_config_ui` + a Cancel button (blocking)
Resolved from pre-impl review (F2/F3/F4/F8 + reviewer-2 Q3):
- **A NEW `GateKind.CONFIG`** (a sibling of CONFIRM/CREDENTIAL), NOT a secret_field-dispatch on
  CREDENTIAL — a CONFIG gate has no secret string to return, and forking the CREDENTIAL branch would
  put the Telegram path's blast radius at risk. The CONFIG card renders the integration's EXISTING
  `draw_config_ui()` inline VERBATIM (the paste textarea is inside it; `_ingest_client_secret` stays
  PRIVATE — no public exposure needed). `build_gate` gets a CONFIG branch + a `_GATE_PROMPTS` entry.
- **No per-card `exporter.update(None)` pump.** The global `share_tab.update(app)` (`ui.py`, before
  `copilot.pump_events`) already drains the exporter's single-consumer `_progress_queue` every frame; a
  second drain inside the card would STEAL events. Caveat to note in-code: that global pump is guarded
  on a current node existing — fine, since publish requires one anyway.
- **Resolution contract (the load-bearing part):** the CONFIG gate resolves via the existing BOOLEAN
  `answer_gate`. Connect-click -> `answer_gate(approved=True)` -> the worker's `connect_youtube`
  capability ENQUEUES the OAuth job and RETURNS (does NOT await the up-to-180s `run_local_server`; the
  Telegram 30s await is too short to copy). The agent's follow-up turn re-checks `youtube_connected()`
  via precheck. Cancel-click -> `answer_gate(approved=False)` -> the decline handoff. So `GateResponse`
  needs NO new field, and the gate never holds the worker through OAuth.
- **The Cancel button** (NEW, not in Settings) lives in the CHAT card (adjacent to the reused panel),
  `##..._{idx}`-suffixed like every gate button. On cancel the worker resumes and the agent explains
  ("you cancelled — connect in Settings if you want; I can't publish to YouTube without credentials").
  The CONFIG card inherits `_resolve_open_gate_card` + `cancel_all` (Stop/reset/shutdown) for free,
  exactly like CONFIRM/CREDENTIAL — no new cancel-coordination code.
- ONE CONFIG-gated tool: `set_youtube_credentials` (`tools/youtube.py`). It needs NO separate
  `connect_youtube` tail tool + NO new App closure — UNLIKE Telegram (whose token-paste and
  Start-the-bot are two distinct steps needing two tools), the inline `draw_config_ui` panel IS the
  whole multi-step flow (paste -> Connect -> the exporter's own OAuth), and the gate auto-resolves
  when `exporter.is_connected()` flips. So it reuses the EXISTING `youtube_connected()` capability
  only. The CONFIG `gate_kind` keeps the tool OFF the CREDENTIAL execute-dispatch (a plain
  `ToolHandler`, not a `CredentialToolHandler`). Connect is disabled inside `draw_config_ui` while
  the exporter is `in_flight` (its own guard).
- Lifecycle (reviewer-2 Q3): a live CONFIG gate keeps the turn `in_flight=True`, so it inherits the
  `open_project` busy-block + the `save_conversation` not-in_flight guard for free (the panel never
  persists). A mid-OAuth shutdown abandons the exporter's OWN worker (pre-existing for Settings-Connect,
  not new); `gate.cancel_all` on release unblocks the copilot worker independently. A Stop mid-OAuth can
  leave a "phantom connect" (the browser flow completes after the agent gave up) — benign, accept-stale.

### D5 — Open-only actions + the renderer
- `copilot_chat.py` `_draw_message` tool_status branch: if `msg.result_widget`, dispatch on `.kind` to
  `ui_primitives` helpers — an OPEN-ONLY url button (`webbrowser.open`, NO clipboard) and an open-path
  button (a file-manager opener). DO NOT route through `draw_link` (it copies + forces https).
- **Fail-soft dispatch (F12/Q1):** `kind` is matched with an `else -> render nothing` branch (an
  unknown future kind on an older build must not crash `_draw_message`); a widget with an empty
  `target` renders no button (guard `if not target: return`).
- Path opener: REUSE `util.open_in_file_manager(path: Path)` — it EXISTS (cross-platform xdg-open /
  open / explorer, opens the parent dir for a file). No new util needed.
- Widgets render through the button tiers (`primary_button`/`ghost_button`), no hand-rolled styling.
- Button ids carry `##<kind>_{idx}` (like the gate buttons) so two transcript widgets never collide.

### D6 — Persistence v3 -> v4
- `_MessageModel` gains optional `result_widget: _ResultWidgetModel | None = None` (mirrors the
  `recover`/`gate_kind` defaulted-optional pattern, `extra="forbid"`); `_ResultWidgetModel` carries
  `kind: str` (LOOSE str, not Literal, so an unknown future kind round-trips), `label`, `target`.
  `_VERSION` 3 -> 4 (provenance only — `load_and_migrate` is fail-soft, not branch-on-version).
- A v3 file (no `result_widget`) loads as `None` (defaulted optional) — a regression test pins it.
  Forward-compat asymmetry (a v4 file on an older `extra="forbid"` build loses the WHOLE conversation)
  is pre-existing for every field added since v1 and ACCEPTED (single-user posture) — noted, not fixed.
- `open_url`/`open_path` widgets persist as-is (accept-stale: a deleted render / expired URL just fails
  on click — maintainer-default; an `open_path` click opens the parent dir even if the file is gone). A
  config gate is transient (a live `pending_action` that keeps the turn in_flight, so `save_conversation`
  can't run while it's open) — it never persists a frozen panel.

### D7 — Prompt truth fix
- Correct the YouTube line ("browser sign-in" -> the real paste-client_secret flow + "I surface the
  connect panel inline"). Soften the render/publish lines that tell the model to echo the path/link.
- Add the widget convention to the prompt: "When a tool says a widget/button was shown to the user,
  point them to it in plain words — never paste a raw URL or path (you don't have it)."

## Files touched

- `copilot/state.py` — `ResultWidget`; `Message.result_widget`.
- `copilot/agent.py` — `AgentToolCard.widget`; extract `payload["widget"]`.
- `copilot/session.py` — carry the widget onto the Message; terse tool_status line.
- `copilot/tools/publish.py` + `registry.py` — terse facts + widget specs (D3).
- `copilot/tools/youtube.py` (new) — the youtube connect tool pair (CONFIG-gated `set_youtube_credentials`
  + `connect_youtube`), mirroring `tools/telegram.py` (D4).
- `copilot/gate.py` — the new `GateKind.CONFIG` member (D4).
- `copilot/agent.py` — `build_gate` CONFIG branch + `_GATE_PROMPTS` entry (D4).
- (NO new caps/closures for YouTube — the CONFIG tool reuses the existing `youtube_connected()` cap;
  the inline `draw_config_ui` drives the exporter directly. NO `ingest_client_secret` exposure.)
- `widgets/copilot_chat.py` — result-widget dispatch + inline config panel + Cancel (D5/D4).
- `ui_primitives.py` — open-only url button + open-path button (D5; reuses `util.open_in_file_manager`).
- `copilot/persistence.py` — v3->v4 round-trip (D6).
- `copilot/prompt.py` — the truth fix + widget convention (D7).
- `tests/test_conversation_persistence.py` — v3-no-widget load + v4-each-kind round-trip + unknown-kind
  fail-soft (D6).

## Review history

- **Pre-impl review (2 agents — correctness/design + verification/blast):** corrected (F1) the render
  path is ABSOLUTE not relative (drop the trash_name-reanchor language; accept-stale); (F2/F3/F8)
  locked the gate contract — a NEW `GateKind.CONFIG` resolved via boolean `answer_gate`, Connect
  enqueues OAuth + returns (no 180s worker block), `_ingest_client_secret` stays private; (F4) no
  per-card pump (the global share-tab pump already drains, a second drain steals events); (Q4) the
  "a widget was shown" sentence + the widget are CONDITIONAL on a non-empty target (no lying about a
  button that isn't there); (F12/Q1) persist `kind` as loose str + fail-soft renderer dispatch; the
  URL-leak trap is in the PRODUCER strings (terse-ify those, not the display layer). All folded into
  D2-D6 above.

## Manual verification (maintainer, in-app)

- Publish to Telegram/YouTube: the chat shows a clickable "Open in ..." BUTTON (opens browser, does NOT
  copy); the agent's reply points at the button and contains NO raw URL.
- Render an image/video: a "Reveal render" button opens the file manager at the render.
- Ask to publish to YouTube while disconnected: the chat shows the SAME setup widgets as Settings
  (instruction + paste client_secret) inline, plus a Cancel button. Loading + Connect works from chat;
  Cancel resumes the agent, which explains it couldn't proceed.
- Reload a conversation: persisted result buttons reappear and still work (accept-stale).
- (Headless: the widget plumbing + dispatch + no-crash render of each card via the standalone GL+imgui
  driver + persistence round-trip tests; the focus/click behavior is a maintainer make-run check.)

- **Post-impl review (3 agents — correctness + spec-fidelity + blast-radius):** all 6 landed decisions
  confirmed correct + additive; the CONFIG-gate auto-resolve idempotence, the persistence fail-soft, and
  the lifecycle (Stop / project-switch / shutdown mid-panel) all verified safe. Two findings actioned:
  (1) the spec's D4 "MIRROR telegram with a connect_youtube tool" wording over-promised — the impl
  correctly needs only ONE tool (the inline panel is the whole flow); spec wording corrected to match.
  (2) added `tests/test_result_widgets.py` pinning the no-url/path-leak property the reviewers flagged
  as the likeliest silent failure. One reviewer's "stale->unresolved payload-key mismatch" concern was
  a FALSE alarm (verified: `_stale_result` returns `{"stale":True}`, `_unresolved_result` returns None —
  the intended 020·20 D4 split, covered by `test_stale_rejects_do_not_trip_retry_cap`).

## Open questions for the user

All resolved by the maintainer's voice notes + the research; nothing blocks planning:
- YouTube gate = blocking + Cancel (voice 1831). Inline = paste-only (research). Stale widgets =
  accept-stale (research default). Selectable text = DEFERRED (voice 1831). Edit-tool widgets = NO.
- Lock at impl (low-surface mechanical choices, not maintainer calls): CONFIG-as-new-GateKind vs
  CREDENTIAL-secret_field-dispatch; whether `open_url`/`open_path` collapse to one `kind` with a
  url-vs-path discriminator.
