# 020·18 — Render / Publish copilot tools (Tier 2)

The next copilot tool wave: give the agent the ability to **render** a node to an image/video
file and to **publish** that artifact externally (Telegram sticker pack, YouTube). Both are gated
(the user confirms every render and every publish); both reuse the gate machinery `020·17` built and
the render/exporter pipelines the Share tab already drives. This is "Tier 2" in the
`16 ## Out of scope` backlog ("the editing set is shipped and a render-or-publish scenario is the
next priority").

Cold-context note: read `16_cross_project_tools.md ## Out of scope` (the backlog entry that scopes
this), `17_gate_ui.md` (the gate this reuses), and `11_capability_wave_spec.md §3.1/§5/§F4/§F5`
(the maintainer's original sketch of these tools). This spec LOCKS the decisions those left open.

---

## Goal

Four new eager, always-gated tools on the copilot:

- **`render_image(node, width?, height?)`** — render the node's current frame to a PNG in the
  project's `renders/` dir; return the path + dimensions. GL → marshalled through the bridge.
- **`render_video(node, seconds, fps?, width?, height?)`** — render `seconds` of the node's
  animation to a `.webm` in `renders/`; return the path + duration + dimensions. GL → bridge.
- **`publish_telegram(node, emoji?)`** — render a Telegram-sticker-shaped `.webm` and add it to the
  user's **currently-selected default pack**; return the pack URL. Gated + credential-guarded.
- **`publish_youtube(node, title, description?, is_short?)`** — render a YouTube-shaped video and
  upload it (private); return the Studio URL. Gated + credential-guarded.

The differentiator vs. the Share tab: the agent can do this conversationally ("render this at 1080p",
"publish the gradient node to my sticker pack") without the user touching the Share UI — and it gets
a real result back (a path / a URL / a clear error), so it can react and report.

---

## Out of scope (each with a trigger)

- **Lazy catalogue-nav (`search_tools` / `list_tools` + `grow_specs_from_payload`).** `11 §4` makes
  publish tools *lazy* (discovered via `search_tools`), but that whole mechanism is itself deferred
  in `16 ## Out of scope` ("the lazy catalogue-nav … Later waves"). This wave ships all four tools
  **eager** (full schemas at turn start) — 13 eager tools (9 shader + 4 new) is fine for a single-model context.
  **Trigger:** `registry.eager_specs()` exceeds ~16 tools (the point where the turn-start `tools=`
  block is a non-trivial slice of the prompt-cache prefix), OR the first maintainer/log observation of
  the model picking a wrong tool attributable to catalogue size.
- **`GateKind.CREDENTIAL` inline secret widget.** Missing credentials use a **guided handoff** (a
  tool result telling the user to connect in Settings — Decision 7), NOT an inline secret field in
  the chat card. `GateKind.CREDENTIAL` stays a type-only stub (its trigger from `17` moves here).
  **Trigger:** the first maintainer/user report that leaving the chat to open Settings → Integrations
  mid-conversation to connect a publish target is friction worth removing, OR a publish target whose
  credential has no Settings UI to point the handoff at.
- **Telegram pack CRUD from the copilot** (create-pack / select-pack / delete-pack). `publish_telegram`
  targets the existing **default** pack (`set_default_pack`/`current_default_pack`), erroring with a
  guided handoff if none is set. **Trigger:** a user wants the agent to create a new pack
  conversationally (then it's a `telegram_create_pack` tool reusing `_create_pack`).
- **YouTube long-form metadata richness** (tags, category, playlist, scheduling). `publish_youtube`
  takes title + description + the short/long shape only; everything else uses the exporter defaults
  (category "22", no tags, private). **Trigger:** a user asks the agent to set tags/category/playlist.
- **Loop-offset / "which 3s" sticker window.** `render_video` always renders `t = 0 … seconds`
  (the existing `_render_video` limitation, already tracked in `todo.md ## sticker render is always
  t=0..N`). This wave does not add a start-offset. **Trigger:** unchanged — that `todo.md` entry's own.
- **Non-blocking render (chunked / progress-bar).** A render freezes the frame loop for its
  duration behind no modal yet (the modal is a UI-polish concern — Decision 9). `11 §13` already
  defers chunked render. **Trigger:** unchanged.
- **Multi-node composite render, render-at-arbitrary-`u_time`, format choice (jpg/exr).** Single
  node, current frame (image) / `t=0..seconds` (video), PNG/WebM only. **Trigger:** a concrete need.
- **`delete_lib_file`** (still deferred from `17`) and the docs tools (`read_doc`) — unrelated to
  render/publish, stay in their own backlog entries.

---

## Design decisions (numbered — lock-in only)

### Render

1. **Render output lands in a project `renders/` dir, NOT `exporter_scratch/`.** Scratch is
   transient export-staging the Share tab cleans per-export; a copilot render is a **user artifact
   the agent produced on request** — it must persist. `App` gains `renders_dir` (created lazily like
   `exporter_scratch`, `project_dir / "renders"`). Files are named `<node-name>_<short-id>_<n>.png` /
   `.webm` where `<n>` is a per-dir monotonic counter so repeated renders of the same node don't
   collide (the agent may render several variants). **The copilot never deletes a render** — these
   are the user's outputs; cleanup is the user's (a future "clear renders" affordance if wanted).

2. **Render marshals through the bridge with the longer `render_op_timeout_s`.** `core.Node.render_media`
   is GL-affine (main thread). The render closure runs as ONE `bridge.run_on_main(fn, timeout=…)` op —
   the frame loop is held for the whole encode (matches the Share tab's own blocking render; `11 §5`'s
   R3 decision). The bridge's default `bridge_op_timeout_s=5.0` would fire on any encode > 5s, so the
   render tools pass `render_op_timeout_s=60.0` (already a `CopilotConfig` field, comment says "Used
   via `bridge.run_on_main(fn, timeout=…)` by the render tools (a later slice)"). **This wave adds the
   `timeout` parameter to `CopilotBridge.run_on_main`** (currently hardcodes `bridge_op_timeout_s`) —
   a pre-planned stub being wired, not new scope.

3. **The render closure reuses `render_for`'s exact logic, not a parallel copy.** `tabs/share_state.render_for(node,
   preset, duration, scratch_dir)` already owns: path minting, the `RenderPreset.duration_max` cap,
   the render try/except + partial-file cleanup, and `RenderedArtifact` construction. The render
   closures build a `RenderPreset` from the tool args and call `render_for` into `renders_dir`. The
   only divergence from `render_for`'s defaults: a deterministic filename (Decision 1) instead of a
   `uuid4()`, so the App closure mints the path and `render_for`'s uuid path is bypassed — to avoid
   duplicating `render_for`, the App closure calls a small shared helper `render_to(node, preset,
   duration, out_path)` that `render_for` is refactored to delegate to (pure extraction, no behavior
   change). If the refactor's blast radius is non-trivial at impl time, fall back to the App closure
   constructing `MediaDetails` + calling `node.render_media` directly (the `render_for` body is ~25
   lines); decide at impl, record which in Review history.

4. **Render args: dimensions optional, defaulting to the node's canvas size.** `render_image(node,
   width?, height?)` and `render_video(node, seconds, fps?, width?, height?)`. Omitted width/height →
   the node's current canvas size (`node.canvas.texture.size`, read on the main thread inside the
   closure). `fps` defaults to `DEFAULT_FPS`. Dimensions are clamped by a pydantic `Field`
   (`ge=16, le=4096`) — a schema-level guard, not handler logic (`16`'s design note). `seconds` is
   **required** on `render_video` (the agent can't infer animation length — `11 §A`'s no-vision rule)
   and `Field(gt=0, le=60.0)` (the `render_op_timeout_s` ceiling; a longer encode would time out).
   **Diverges from `11 §3.1`'s sketch** (`render_image(node_id, out_path?)` / `render_video(node_id,
   out_path?, seconds?, fps?)`) in two ways, both deliberate: (a) NO `out_path` arg — Decision 1 owns
   the path (an agent-chosen path is a sandbox-escape footgun, and the agent has no filesystem per
   `11`'s sandbox rule); (b) `seconds` is REQUIRED, not optional (no sane default for "how long"). The
   `RenderResult` reports the ACTUAL rendered size, which `resolve_dims`/`_align`
   (`render_preset.py`) snaps UP to the codec alignment — so a requested 1000×1000 video may come back
   as 1008×1008; the agent quotes the returned size, and the system prompt notes the snap.

### Publish

5. **Publish is enqueue-and-await on the COPILOT WORKER, but EVERY exporter touch goes through the
   bridge — including the await poll.** The exporters publish asynchronously: `export()` is a stub
   that raises; the real upload is a `_Job` enqueued onto the exporter's OWN worker thread, with
   progress drained per-frame on the main thread into `exporter.status().last_progress`. A copilot
   tool handler is synchronous and must return a real result. The composition (each numbered step is
   a thread boundary):
   - **(a) Render** the artifact first (Decision 6) into `renders_dir` (Decision 1), through the
     bridge (`render_op_timeout_s`). One blocking main-thread op.
   - **(b) Enqueue + capture baseline** through ONE bridge op on the main thread: read
     `exporter.status().last_progress` (the pre-enqueue baseline identity), then call the exporter's
     **public** `publish(...)` (Decision 14 — NOT the private `_enqueue`). Return the baseline
     object's `id()` to the worker. (The whole step is on the main thread = the exporter's render
     thread, honoring `base.py`'s affinity contract.)
   - **(c) Await** on the copilot worker: a loop that, each iteration, (i) returns a cancelled
     `PublishResult` if `cancel_check()` is true (Decision 15), (ii) sleeps `publish_poll_interval_s`,
     (iii) does ONE bridge op `run_on_main(lambda: (exporter.update(), exporter.status())[1])` — this
     **drains the exporter's queue AND reads status on the main thread** (contract-clean; no worker
     ever touches `status()` directly, closing the `base.py` render-thread-only concern), and (iv)
     checks whether `last_progress` is a terminal object whose `id()` differs from the baseline (a
     stale terminal from a prior manual upload must not false-positive). On a new terminal, read
     `is_error` / `message` / `url` and return.

   Why this is correct and safe (each prior review finding addressed):
   - **Cancel reaches the loop (was the central gap).** The await checks `cancel_check()` every
     iteration AND each `run_on_main` raises `CopilotCancelled` once the bridge is hard-cancelled at
     `release()`. Stop (`cancel_turn`) sets `_cancel` but does NOT cancel the bridge, so the
     `cancel_check()` path is what catches Stop; teardown (`release`) cancels the bridge too, so the
     `run_on_main`-raises path catches shutdown. Both paths covered. (Decision 15 plumbs `cancel_check`.)
   - **The drain no longer depends on the share-tab.** Step (c)(iii) calls `exporter.update()` itself
     via the bridge, so progress advances even in a 0-node project or after the agent deleted the
     current node in the same turn — it does NOT rely on `share_tab.update(app)` (which is gated on
     `current_node_id in ui_nodes` in `ui.py::update_and_draw` — the prior draft's claim that the share drain was
     ungated was WRONG). The share-tab drain still runs when applicable; double-draining a queue is
     safe (get_nowait until empty), and the share tab's terminal-notification dedup
     (`outlet.notified_progress`) is unaffected.
   - **No deadlock.** The three steps are strictly sequential (render → enqueue → await); no step has
     the main thread waiting on the worker while the worker waits on the main. Each bridge op is the
     proven worker-blocks-main-runs-then-unblocks round-trip.
   - **Main loop stays responsive during the network upload.** Only the render encode (step a) freezes
     the frame loop; the await's per-iteration bridge ops are trivial (one `update()` + one status
     read), so the UI keeps painting while the exporter worker uploads.
   - **Every outcome terminates.** Both exporters push a terminal `ExportProgress` on success
     (`is_terminal=True` + `url`) AND on every failure path (`is_terminal=True, is_error=True` +
     message, from each `_handle_job`'s try/except). So the await never hangs silently.

   `publish_await_timeout_s` (new `CopilotConfig`, 300.0) bounds the worst case (a never-terminal
   stuck upload) → an honest "the upload is taking too long; check the Share tab" result.

6. **Publish auto-renders; it does not take a pre-rendered artifact.** The agent says "publish the
   gradient node to my pack" — the tool renders the node with the **exporter's own `render_preset()`**
   (which already encodes the Telegram 3s/512px/30fps caps and the YouTube short/long shape) and then
   publishes. Reusing `exporter.render_preset()` means the platform caps live where they already live
   (in the exporter), not duplicated in the tool. The render uses the same bridge path as Decision 2.

7. **Missing credentials = a guided handoff via a PRE-GATE PRECHECK (`11 §F5`).** A publish that can't
   run (no creds, or no Telegram default pack) must NOT pop a confirm dialog only to fail after the
   user clicks Yes. But the loop fires the gate (`requires_gate` → `gate.ask`) BEFORE it ever calls
   the handler (`agent.py` runs the gate at the top of the tool-call block, `execute()` at the
   bottom) — so a handler-side cred check can't run before the gate. The fix is a **pre-gate
   precheck** (Decision 15): the publish tool's `ToolDefinition` carries an optional
   `precheck(args) -> str | None`; the loop runs it BEFORE `requires_gate`, and a non-None return
   short-circuits to a tool message (the guided handoff) + `continue` — no gate, no `execute`, no
   counter. The precheck reads the connection/pack state through a GL-free capability
   (`telegram_connected` / `youtube_connected` / `telegram_has_default_pack`). Messages are
   actionable: *"Telegram isn't connected. Open Settings → Integrations → Telegram, connect your bot,
   then ask me again."* / *"No Telegram sticker pack is selected — pick or create one in the Share
   tab's Telegram panel first."* / YouTube analogously. This routes the cred-fail AROUND `execute`
   (like the gate-decline path), so it also can't false-trip the edit-retry cap (the finding that a
   `mutating` publish returning `ok=False` would increment `consecutive_failed_edits`). NOT a
   `GateKind.CREDENTIAL` inline widget (deferred, Out of scope).

### Gate + loop

8. **All four tools are `GatePolicy.ALWAYS` with engine-built confirm prompts; the LOOP gains a
   pre-gate precheck step (the only `agent.py` behavior change beyond the prompt templates).** Each
   tool gets a `_GATE_PROMPTS` template in `agent.py` (the engine owns the phrasing, `17` Decision).
   Render prompts warn about the freeze ("Render a 1920×1080 image? The app pauses while it
   encodes."); publish prompts warn it's external+live ("Publish to your Telegram pack? This uploads
   the sticker."). The CONFIRM gate **body** (`gate.ask` → `AgentGateOpened` → Yes/No → resolve) is
   reused unchanged — **`gate.py` does NOT change**. What changes in `agent.py`: (i) the precheck call
   before `requires_gate` (Decision 7/15), and (ii) the new `_GATE_PROMPTS` entries. A declined
   render/publish flows through the existing decline path (records + appends the tool message +
   continues so the model comments; does not count toward the edit-retry cap — `17` Decisions 1+2
   already handle this for any ALWAYS-gated tool).

9. **No "Rendering…" modal in this wave; the freeze is bare. This REVERSES the maintainer's dated R3
   decision (`11 §5`, "DECIDED — maintainer 2026-05-31") and needs sign-off.** `11 §5` mandates a
   two-phase "Rendering…" modal so the UI paints "rendering" before the frame loop freezes. That modal
   needs a paint-then-freeze two-phase commit and belongs in the copilot UI/UX polish wave that gates
   shipping. For now the render freeze is bare (the app visibly hangs for the encode, like a heavy
   Share-tab render). Because this overrides a *dated, explicitly-decided* maintainer point (not a
   fresh open question), it is flagged in Review history AND Open Question 4 asks for re-confirmation.
   Recorded as a `todo.md` deferral with a concrete trigger ("first maintainer/user report that the
   render freeze reads as a hang/crash"). `render_op_timeout_s=60` bounds the worst case.

### Seams

10. **New result value types `RenderResult` / `PublishResult` in `capabilities.py`** (frozen, GL-free,
    mirroring `DeleteNodeResult`): `RenderResult(ok, error, path, is_video, width, height, duration)`;
    `PublishResult(ok, error, url, kind)`. The tool handlers turn these into `(ok, msg, payload)`. The
    payload carries `path` (render) / `url` (publish) for a future chat affordance (an "open" button) —
    but **this wave adds no new chat UI** beyond what `_draw_message` already does for a tool card; the
    URL/path just appears in the agent's text reply. (A clickable card is UI-polish-wave scope.)

11. **Capabilities gain the closures the tools + precheck need**, all App-bound: `render_image`,
    `render_video`, `publish_telegram`, `publish_youtube` (the two render + two publish closures
    marshal through the bridge), plus the GL-free precheck reads `telegram_connected` /
    `youtube_connected` / `telegram_has_default_pack` (plain reads, no bridge — they back Decision 7's
    pre-gate precheck). All are bound in `_build_copilot_capabilities`; all respect the leaf-import
    rule (the new `RenderResult`/`PublishResult` are frozen GL-free value types).

12. **A new `tools/publish.py` module** (parallel to `tools/shader.py`) holds the four tool factories
    + their pydantic arg models + the `(ok, msg, payload)` handlers + the `precheck` callables.
    `build_registry` becomes `[*shader_tools(caps), *publish_tools(caps)]`. (Render + publish live
    together in `publish.py` since they share the gate + the external-artifact theme; splitting render
    into its own module is premature for four tools.)

13. **The system prompt gains a render/publish capability block.** A new "RENDER & PUBLISH" section in
    `_SYSTEM_PROMPT` (`prompt.py`) describes the four tools with their caveats: render freezes the UI
    briefly + costs a confirm; publish is EXTERNAL and IRREVERSIBLE (the post goes live) + needs the
    integration connected in Settings first; the agent cannot see the result so it describes what it
    rendered/published and gives the user the path/URL; rendered dimensions snap to the codec
    alignment. It also states the agent renders the CURRENT source (so it must land edits first) and
    cannot pick a render *time window* (always from t=0).

14. **Each exporter gains a small PUBLIC publish-trigger + connection API; this IS a (bounded)
    exporter change.** The publish path needs to enqueue a job and read connection state, but
    `_enqueue` and `_is_connected` are PRIVATE today (only `draw_target_panel`'s buttons call them).
    So the `Exporter` ABC (`base.py`) gains two public methods, implemented on both exporters by
    delegating to the existing private bodies (no behavior change to the Share-tab path):
    - `is_connected() -> bool` — wraps the existing `_is_connected` (`auth_state == AUTHED` AND the
      identity check). Backs `telegram_connected`/`youtube_connected`.
    - `publish(artifact, settings) -> None` — enqueues the upload job (Telegram `add` / YouTube
      `upload`) via the existing `_enqueue(_Job(...))`. The current abstract `export()` stub (which
      raises "dispatched per job kind") is REPLACED by this honest public entry; the per-job-kind
      `_handle_*` dispatch is unchanged. (Telegram also needs `current_default_pack()` exposed for
      `telegram_has_default_pack` — it already is, `telegram.py`.) The publish render's input artifact
      lands in `renders_dir` (Decision 1), so the exporter's `prepare()` re-encode + the `prepared`
      cleanup don't touch the original (matches the Share-tab contract — `_handle_add` deletes only
      the prepared file, not `job.artifact`).

15. **Cancel + precheck plumbing — the loop and the session grow two seams.**
    - **`cancel_check` into publish closures.** The publish-await (Decision 5c) must observe a Stop /
      teardown. `CopilotSession.is_cancelled() -> bool` (reads the CURRENT `self._cancel` — an
      indirection, because `reset_conversation` REPLACES the event object, so capturing it would go
      stale). The publish closures read `self.copilot.is_cancelled()` **at call time** inside the
      await loop — NOT a build-time capture (`_build_copilot_capabilities` runs as an argument to
      `CopilotSession(...)`, before `self.copilot` exists; the same call-time-read pattern
      `_copilot_delete_node` uses for `self.copilot.bridge`). (The render closures don't need it — a
      render is one bounded bridge op; teardown's `bridge.cancel_all()` already raises through it.)
      **Invariant note (load-bearing):** `is_cancelled()` catches a *project switch* correctly only
      because `reset_conversation` is unreachable while an await is live — both its callers are gated
      (`copilot_clear_chat` refuses while `state.in_flight`; `open_project`→`_init` runs
      `self.release()` first, which hard-cancels the bridge + joins the worker before
      `reset_conversation` runs). The `_cancel`-replace ordering in `reset_conversation` (set old,
      then replace with a fresh unset event) WOULD miss a cancel if an await ran concurrently — so a
      future refactor that makes `reset_conversation` reachable mid-turn must re-check this. The
      shutdown path (`release`) is independently safe: it sets `_cancel` on the live event AND
      `bridge.cancel_all()` (the await's next `run_on_main` raises `CopilotCancelled`, caught by
      `registry.execute`'s broad `except`).
    - **Pre-gate precheck in the loop.** `ToolDefinition` gains `precheck: Callable[[dict], str | None]
      | None = None`. `run_turn`, for each tool call, runs `precheck` (if present) BEFORE
      `requires_gate`; a non-None return is appended as the tool message + `continue`d (no gate, no
      `execute`, no counter). This is the ONLY new control path in the loop; everything else (gate,
      execute, decline, counter) is unchanged. The precheck is pure (GL-free reads) and idempotent.

---

## Files touched

- **`shaderbox/copilot/bridge.py`** — add `timeout: float | None = None` param to `run_on_main`
  (defaults to `bridge_op_timeout_s` when None). Pre-planned per the `render_op_timeout_s` comment.
- **`shaderbox/copilot/config.py`** — add `publish_await_timeout_s: float = 300.0` +
  `publish_poll_interval_s: float = 0.2` (`render_op_timeout_s` already exists).
- **`shaderbox/copilot/bridge.py`** (already above) — the `timeout` param.
- **`shaderbox/copilot/capabilities.py`** — add `RenderResult` / `PublishResult` value types + the
  closure fields (Decision 11): `render_image`, `render_video`, `publish_telegram`,
  `publish_youtube`, `telegram_connected`, `youtube_connected`, `telegram_has_default_pack`.
- **`shaderbox/copilot/tools/base.py`** — add the optional `precheck: Callable[[dict], str | None] |
  None = None` field on `ToolDefinition` (Decision 15).
- **`shaderbox/copilot/tools/publish.py`** — NEW. `publish_tools(caps)` factory: four `ToolDefinition`s
  (all `GatePolicy.ALWAYS`, `eager=True`, `category="publish"`; render tools `needs_gl=True` +
  `mutating=True`; publish tools `needs_gl=False` since the GL render is inside the closure, but
  `mutating=True` — they have external side-effects, and carry a `precheck`) + pydantic arg models +
  handlers + the precheck callables.
- **`shaderbox/copilot/tools/registry.py`** — `build_registry` adds `*publish_tools(caps)`.
- **`shaderbox/copilot/agent.py`** — the pre-gate `precheck` step in `run_turn` (Decision 7/15) + four
  `_GATE_PROMPTS` templates.
- **`shaderbox/copilot/session.py`** — `is_cancelled() -> bool` (reads the current `_cancel`).
- **`shaderbox/copilot/prompt.py`** — the RENDER & PUBLISH system-prompt block.
- **`shaderbox/exporters/base.py`** — add abstract `is_connected() -> bool` + replace the abstract
  `export()` stub with a public `publish(artifact, settings) -> None` (Decision 14); add
  `is_connected`/`publish` to the render-thread list in the class thread-affinity docstring.
- **`shaderbox/exporters/telegram.py` + `youtube.py`** — implement `is_connected()` (delegates to the
  existing `_is_connected`) + `publish()` (delegates to the existing `_enqueue(_Job(...))`). No
  behavior change to the Share-tab path. (Any caller of the old `export()` — there are none beyond the
  raising stub — is updated.)
- **`shaderbox/app.py`** — `renders_dir` property; the `_copilot_*` closures (`_copilot_render_image`,
  `_copilot_render_video`, `_copilot_publish_telegram`, `_copilot_publish_youtube`,
  `_copilot_telegram_connected`, `_copilot_youtube_connected`, `_copilot_telegram_has_default_pack`);
  bind them in `_build_copilot_capabilities` (the publish factory also gets `self.copilot.is_cancelled`).
- **`shaderbox/tabs/share_state.py`** — IF Decision 3's extraction is taken: factor `render_for`'s
  body into a `render_to(node, preset, duration, out_path) -> RenderedArtifact | None` the App closure
  also calls. (Pure extraction, no behavior change.)
- **`scripts/`** — a NEW headless check (or a `make smoke` extension) per the verification section: a
  registry-build + render-to-file introspection script (Decision: it CAN be headless — smoke already
  runs a real GL context).
- **`ai_docs/features/020_copilot_agent/11_capability_wave_spec.md`** — the `§16` reality block gets a
  pointer to this spec (sanitize sweep).
- **`ai_docs/todo.md`** — the "Rendering… modal" deferral (Decision 9) + the lazy-catalogue-nav
  deferral (Out of scope) + the CREDENTIAL-widget deferral (re-homed from `17`).
- **`ai_docs/roadmap.md`** — flip the 020 row's render/publish status; rewrite the Active-context
  banner ("what's next" → the UI/UX polish wave, now that render/publish landed).

No `core.py` change — render reuses `render_media`. The exporter change is bounded (Decision 14): two
public methods delegating to existing private bodies; the Share-tab path is untouched.

---

## Manual verification

### Headless (the agent runs these — within the sanctioned `dev_flow.md` envelope)

`make smoke` does NOT construct a turn or call the registry — it only runs `update_and_draw`, so it
catches a new-module import error ONLY if App-construction imports it (it does, transitively, via
`build_registry`). That is NOT enough. So this wave ADDS a focused headless introspection script
(`scripts/copilot_render_check.py` or similar; a real GL context like smoke). *(Feature 031: the
script bit-rotted with zero wiring; H1/H3-class checks were ported into pytest —
`tests/test_tool_registry.py` — and the script deleted.)*

- **H1 — registry builds.** `build_registry(caps)` returns 13 eager specs (9 shader + 4 new); each new tool's
  `args_model.model_json_schema()` builds (catches a pydantic `Field(ge=16, le=4096)` / required-arg
  typo); each new tool's `gate_policy is GatePolicy.ALWAYS`; the publish tools carry a non-None
  `precheck`.
- **H2 — render-to-file.** Construct a `Node`, build a `RenderPreset` from sample render args, call
  the `render_to` path (Decision 3) into a temp dir; assert the output file exists, is a PNG / WebM,
  and its dimensions match the requested (snapped) size. This pins the Decision-3 `render_for`→
  `render_to` refactor as behavior-preserving (a refactor on the Share-tab hot path with no test is
  exactly where a regression hides) AND the `renders_dir` filename-counter minting.
- **H3 — precheck logic.** With a stub caps reporting "not connected", each publish tool's `precheck`
  returns a non-None guided message; with "connected + pack set", it returns None. (No network.)
- **`make smoke`** still passes (App constructs, the frame loop runs with the new closures bound).

### Maintainer, live (`make run`) — the LLM/upload paths headless can't reach

Needs an OpenRouter key + a live model (an agent turn) and real Telegram/YouTube creds (a real
upload), so:

1. **render_image** — ask "render this node to an image". A Yes/No gate pops; on Yes the app briefly
   freezes then a PNG appears in `<project>/renders/`; the reply states the path + size. On No, the
   agent comments, no file.
2. **render_video** — "render a 2-second video of this node". Gate → freeze (longer) → a `.webm` in
   `renders/`; reply states path + duration. A `seconds: 90` request is rejected at the schema (the
   agent sees a validation error and self-corrects or reports the cap).
3. **render → live preview intact** — after a render completes, the live preview canvas still animates
   correctly (no frozen / wrong-size canvas — the render shares the main GL thread with the preview).
4. **publish_telegram, connected + default pack set** — "add this as a sticker to my pack". Gate →
   render → upload (UI stays RESPONSIVE during the network upload, only the render froze) → reply with
   the `t.me/addstickers/…` URL; the sticker really appears in the pack. Do it with the Share tab NOT
   the active sub-tab AND in a project where the current node was just deleted this turn (proves the
   await's own bridge-drain works without the share-tab drain — Decision 5c).
5. **publish_telegram, NOT connected / no pack** — disconnect Telegram (or clear the default pack),
   ask to publish. The agent relays the guided handoff (the precheck message) with **NO gate confirm**
   and **NO render** — verify the confirm dialog does NOT appear (this is the Decision-7 fix; the old
   broken design would pop a Yes/No then fail).
6. **publish_youtube** — analogous to 4 (connected) and 5 (not). The private video appears in Studio;
   the reply carries the Studio URL.
7. **cancel/teardown** — decline a render gate (agent comments, no file). Stop mid-render and
   mid-publish-await: the turn cancels cleanly, the worker unblocks (the await sees `is_cancelled()`),
   no hang. Close the app / switch projects mid-publish-await: clean shutdown, no 5s+ stall, no
   use-after-release (the await bails on cancel before touching the released exporter).

---

## Open questions for the user (maintainer, on return)

1. **Eager vs. lazy publish (Decision: eager).** `11 §4` wanted publish lazy. I shipped eager to
   avoid building the whole lazy-catalogue mechanism (itself deferred). OK, or do you want the lazy
   path built now as part of this wave?
2. **Guided-handoff vs. inline CREDENTIAL widget (Decision 7: handoff).** I deferred the inline secret
   field. If you'd rather the agent let the user paste a key right in the chat, that's the
   `GateKind.CREDENTIAL` build — a bigger UI surface. Confirm the handoff is enough for v1.
3. **Telegram default-pack only (Out of scope).** The agent adds to the selected default pack and
   can't create/pick a pack conversationally. Enough, or do you want `telegram_create_pack` now?
4. **No "Rendering…" modal — this REVERSES your dated R3 decision (`11 §5`, 2026-05-31).** You
   explicitly decided render gets a two-phase "Rendering…" modal. I deferred it to the UI/UX polish
   wave (the bare freeze behaves like a heavy Share-tab render). Re-confirm the deferral, or do you
   want the modal built here?

---

## Review history

### Pre-implementation review (2 adversarial agents, both anchored to code-on-disk + the `11`/`16`/`17` backlog)

Both reviewers returned **FAIL** on the first draft. The findings were real (verified against code,
not fabricated), and the first draft had two structurally-broken decisions. Triage + resolution:

- **[CRITICAL, both reviewers] The cred-check-before-gate (old Decision 7) was unimplementable.** The
  loop fires the gate (`requires_gate` → `gate.ask`) at the TOP of the tool-call block and only calls
  the handler via `execute()` at the BOTTOM (`agent.py`), so a handler-side cred check can't precede
  the gate — a missing-cred publish would pop a confirm dialog and only then fail. Cascade: a
  `mutating` publish returning `ok=False` for missing creds would also false-trip
  `consecutive_failed_edits`. **Resolved:** added a **pre-gate `precheck`** seam on `ToolDefinition`,
  run before `requires_gate`, short-circuiting to a guided-handoff tool message + `continue` (no gate,
  no execute, no counter). New Decisions 7 (rewritten) + 15.
- **[CRITICAL, reviewer 2] The publish-await's "cancel-aware loop" couldn't see `cancel`.** No
  capability/handler receives the cancel event; `reset_conversation` even REPLACES the `_cancel`
  object, so capturing it goes stale. STOP and shutdown mid-await would hang/stall + read a released
  exporter (use-after-release across threads). **Resolved:** `CopilotSession.is_cancelled()`
  indirection bound into the publish closures; the await checks it each iteration AND polls via the
  bridge (which raises at hard shutdown). New Decision 15 + rewritten Decision 5.
- **[HIGH, reviewer 2] The await's drain depended on `share_tab.update`, gated on `current_node_id in
  ui_nodes` in `ui.py::update_and_draw`, which the first draft wrongly called ungated).** Empty project or
  delete-then-publish-same-turn → drain never runs → await hangs to timeout. **Resolved:** Decision 5c
  now drains via the await's OWN bridge op (`run_on_main(exporter.update(); status())`), independent of
  the share tab.
- **[HIGH, both] "No exporter change" was false** — `_enqueue` and `_is_connected` are private.
  **Resolved:** Decision 14 adds public `is_connected()` + `publish()` to the `Exporter` ABC + both
  impls (delegating to the existing private bodies); the spec now lists the bounded exporter change
  honestly and drops the "no exporter change" claim.
- **[MEDIUM, reviewer 1] `status()` is documented render-thread-only (`base.py`).** Resolved by 5c
  routing the status read through the bridge (main thread), so no worker ever calls `status()` directly.
- **[HIGH, reviewer 2] `make smoke` does NOT build the registry** — the old verification step 7 was
  wrong. **Resolved:** added headless checks H1–H3 (registry-build + render-to-file + precheck logic),
  within the sanctioned `dev_flow.md` headless envelope.
- **[MEDIUM, reviewer 1] Decision 9 reverses a dated maintainer R3 decision** — now flagged loudly in
  Decision 9 + Open Question 4 (re-confirm required).
- **[MEDIUM, reviewer 2] A silent 4th divergence from `11 §3.1`** (dropped `out_path`, required
  `seconds`) — now flagged explicitly in Decision 4.
- **[MEDIUM, reviewer 2] Three vague deferral triggers** (lazy-nav, CREDENTIAL, modal) — rewritten to
  concrete observable triggers (tool-count threshold / first-report / specific symptom).
- **[LOW]** dim-alignment snap (now noted in Decision 4 + the prompt), the double terminal-notification
  (a copilot publish also toasts via the Share tab's `_surface_terminal_progress` — acknowledged,
  benign), and the loop-offset trigger technically firing on the `render_for` touch (re-affirmed
  deferred). Accepted as-is.

The three documented divergences from `11` (eager-not-lazy, guided-handoff-not-CREDENTIAL, no-modal)
stand — each defensible (the coupled mechanisms are themselves deferred / are UI-polish-wave scope) and
each surfaced as an Open Question for maintainer sign-off on return. Net effect of the revision: the
loop grows ONE new step (the precheck), the exporters grow TWO public methods, and the session grows
ONE accessor — all small, all listed in Files-touched. **Post-revision the spec is implementable as
written.**

**Round 2 (same 2 reviewers, re-spawned against the revised spec): both PASS.** All six round-1
findings verified RESOLVED against code (the precheck slots between the cancel-check and
`requires_gate` in `run_turn`; `is_cancelled()` survives the `_cancel` replacement via call-time read
and covers both Stop and shutdown; the await drains via its own bridge op; `export()` has no external
caller so the ABC swap strands nothing; H1/H3 run GL-free like `copilot_gate_check.py`; the
divergences are flagged). Round-2 raised only minor, real refinements, all folded in BEFORE impl:
(N1) reconciled the `publish` vs `enqueue_publish` name to `publish`; (N2) made the `cancel_check`
binding a call-time read, not a build-time capture (the `_build_copilot_capabilities`-runs-before-
`self.copilot`-exists trap — Decision 15 now says so explicitly); (N3) the `base.py` affinity
docstring update is now in Files-touched; (finding C) the `is_cancelled()` project-switch invariant
(release-joins-first + the `in_flight` gate) is recorded in Decision 15 so a refactor can't silently
reintroduce the miss. No fabricated late-round findings; the loop converged in 2 rounds.

### Post-implementation review (3 adversarial agents: code-correctness, architecture/conventions, spec-fidelity — convergence loop)

**Round 1: 1 FAIL (code), 1 PASS (arch), 1 PARTIAL (spec-fidelity).** The real findings, all fixed:
- **[BLOCKER, code] `render_image`/`render_video` width/height were INERT.** The closures built a
  `RenderPreset(target_w, target_h)` but left the default `resolution_policy=FREE` + `fit=SCALE_DISTORT`
  — `core.render_media` early-returns into the node's own canvas and never calls `resolve_dims`, so
  "render at 1080p" silently rendered at canvas size. **Fixed:** `App._copilot_render_preset` sets
  `FIXED_DIMS` + `RENDER_AT_TARGET`; the new headless H2 (a real GL context) asserts the output dims
  equal the requested (aligned) size — proven (canvas 1280×960, request snaps to 1328×1008).
- **[MEDIUM, code] `publish_youtube(is_short)` didn't drive the render shape** (it came from the
  Share-tab's `_render_state.shape`, independent of the arg) → a long render could upload flagged
  Short. **Fixed:** `YouTubeExporter.set_shape(is_short)` called before `render_preset()`; the prior
  shape is restored in a `finally` so a copilot publish doesn't silently flip the user's Share-tab chip.
- **[HIGH, arch] The pre-gate cred reads (`is_connected`/`render_preset`) run on the copilot worker
  without the bridge, contradicting `base.py`'s render-thread-only affinity doc.** **Fixed:** the doc
  gains an explicit "any thread (GL-free scalar reads)" tier for those two methods (they are pure
  GIL-atomic field reads — no GL, no queue mutation).
- **[LOW, code] baseline `id()` could false-negative on CPython id reuse; a poll-op timeout escaped.**
  **Fixed:** hold the baseline OBJECT and compare `prog is not baseline`; wrap the poll `run_on_main` in
  `except CopilotToolError: continue` (verified `CopilotCancelled` is a sibling, not a subclass — so a
  hard-shutdown cancel still propagates and isn't swallowed).
- **[PARTIAL, spec-fidelity] The spec's doc-touch obligations** (roadmap row/banner, `todo.md`
  deferrals, `11 §16` pointer) were unmet at review time — they are the `/sanitize` closing wave,
  done after review (sequenced, not skipped).

**Round 2 (re-spawn against the patched code): both code reviewers PASS.** All five round-1 fixes
verified resolved at the root; the F4 `except CopilotToolError` confirmed NOT to swallow
`CopilotCancelled` (sibling exceptions, checked in `errors.py`); H1/H2/H3 + the existing
`copilot_gate_check` + `make smoke` all green. Two minor UX items surfaced and hardened: a publish
with no current node now fails-fast in the precheck (a `has_current_node` cap) instead of gate-then-
fail; the YouTube shape is restored post-publish. The remaining notes (a copilot `render_video`
restarts the node's video uniforms in the live preview — pre-existing, shared verbatim with the
Share-tab render; YouTube long-form is a fixed 6s) are inherited/by-design, not regressions. Converged.
