# 020 · 20 — Copilot UI/UX polish wave

The ship gate for the whole 020 stack. Folds in the transcript-legibility gaps the audit flagged
(`wt1jrv2z4`), the glyph/`?` rendering bug the maintainer hit, the low-severity correctness footguns
batched as one wave, and the deferred siblings whose triggers have fired (the "Rendering…" modal, the
lazy tool-catalogue). The five must-fix correctness items (history divergence, the two edit-safety
guards, the array-uniform reject, the double-escape guard) already landed — this wave is the
state-model + render + behavior polish that sits on top.

## Goal

Make the chat transcript legible, honest, and renderable so the maintainer can ship the copilot:
- Per-tool progress + tool results reach the transcript (today AgentStatus is dropped and a tool card
  carries only `name: ok/failed` — render path / publish URL / error text never become a Message).
- Non-ASCII glyphs the font can't render stop showing as `?` (an LLM reply with an arrow `→`, our own
  `thinking…` ellipsis).
- The blocking render/publish shows a cue, not a frozen "thinking…".
- The batch of low-severity correctness footguns the audit confirmed is closed.

## Out of scope (each with a trigger)

- **bind_media tool** (load an image/video into a sampler uniform) — a scope decision, not a gap;
  set_uniform already tells the agent samplers aren't settable. **Trigger:** first real session where a
  user asks the copilot to bind media and the Settings-handoff reads as a wall.
- **undo_edit tool** (revert an edit_shader/replace_lines/insert_after) — only delete has a recover
  affordance today. **Trigger:** first session where a bad multi-edit can only be re-edited, not
  reverted, and that friction is reported.
- **A merged fallback font** (DejaVu/Noto for native arrows/math/non-Latin) — rejected this wave in
  favor of ASCII sanitization (D2). **Trigger:** sanitization visibly mangles a real non-Latin reply
  often enough to annoy, OR an imgui-bundle bump makes the dynamic-atlas merge path clean.

## Design decisions (numbered, lock-in)

### D1 — AgentStatus + tool-result legibility (the transcript state-model plumbing)
The polish-wave foundation: do the state plumbing first so the transcript visuals paint on a model
that can actually show this data.
- Add `ChatState.status: str` — the latest transient in-flight status line. `_apply_event` sets it on
  `AgentStatus` (replacing the `pass`), clears it on every terminal event. The transcript draws it in
  place of the bare `thinking…` caption while in_flight.
- `AgentToolCard` already carries `name`/`ok`/`payload`; extend it (or the Message it materializes) so
  a tool's RESULT line (render path, publish URL, the error text on a failed mutating tool) reaches a
  `tool_status` Message under the card. The result string already exists at the `tool_call` trace
  event — surface it to the card, not just the trace/LLM history. On the three AgentError exits
  (length cutoff, edit-giveup, max_iterations) a just-succeeded publish's URL must still be visible.

### D2 — Glyph rendering: sanitize copilot text to ASCII (render + history + a prompt nudge)
EMPIRICAL grounding (don't re-derive): imgui-bundle 1.92 uses the dynamic font atlas (no
`glyph_ranges` param; `App.get_font` loads on demand). `AnonymousPro-Regular.ttf` HAS em-dash /
ellipsis / smart quotes / Latin-1 accents (`is_glyph_in_font -> True`) but does NOT have arrows
(U+2192 `→` -> False). So the fix is NOT a glyph-range load — it's a transliteration, applied at
THREE points so the model is conditioned on what it actually rendered:
- A `sanitize_display(text) -> str` helper (new leaf `copilot/text_render.py` — NOT imgui/App-coupled)
  maps the common LLM offenders to ASCII: `→`/`←`/`↔` -> `->`/`<-`/`<->`, `—`/`–` -> `--`/`-`, `…` ->
  `...`, smart quotes -> `'`/`"`, `•` -> `*`, `·` -> `.`. Any char STILL outside the font's renderable
  set becomes a literal `?` (Q3: marks "something was here", bounded to truly-unmappable chars).
- **`sanitize_display` is idempotent and applied at THREE boundaries (pre-impl review pinned the exact
  sites to avoid a desync/tear trap):**
  - **History (worker):** sanitize `assistant_text` at the `_commit_turn` boundary in `session.py` —
    `content = sanitize_display(assistant_text) or error_text or None`. So the model's next turn is
    conditioned on the sanitized text it actually produced (it stops re-emitting `?`-ed glyphs).
  - **Rendered final Message (main):** sanitize where `streaming_text` becomes the assistant `Message`
    on `AgentTurnDone` in `session.py` (the materialize boundary).
  - **Live streaming preview + status line + `thinking…` (-> `...`):** sanitize at DRAW time in
    `copilot_chat.py`, NOT by mutating `state.streaming_text` in `_apply_event`. Deltas arrive
    token-by-token; a multi-byte glyph can be torn across two `AgentTextDelta`s, so per-delta sanitize
    would mis-map a half-arrived sequence. Sanitizing the full accumulated string at draw is tear-safe;
    `state.streaming_text` stays the raw truth (single-writer invariant intact), the widget renders a
    sanitized VIEW.
- **CONTENT-ONLY guard (lock-in):** `sanitize_display` touches assistant CONTENT text ONLY — never
  `tool_calls` / `tool_call_id` / `arguments` / a GLSL source payload. Today the persisted `history` is
  a flattened user/assistant text pair with no tool_calls, so pairing is safe; the guard is written so
  it survives a future history-shape change. NEVER route it through `_assistant_message` /
  `_tool_message` / `_parse_args` (mangling a `→` inside a JSON `arguments` or a `tool_call_id` would
  corrupt a tool call). Distinct from the existing control-char `_sanitize` in `prompt.py` (injection
  hygiene on `user_text`) — do not conflate.
- A ONE-SENTENCE prompt nudge (`prompt.py`, near the system prompt): tell the model to output ASCII-only
  symbols (no arrows / em-dashes / smart punctuation). Belt-and-suspenders with the sanitizer — the
  nudge reduces the work, the sanitizer guarantees the glyph never reaches the atlas.
- The substitution table is the single source of truth; a recurring missing-glyph gets added there.

### D3 — Blocking render: the two-phase "Rendering…" modal (Q1 RESOLVED — in this wave)
The `todo.md` "copilot render has no Rendering… modal" deferral, resolved as a two-phase
paint-then-freeze modal. **GROUNDED MECHANISM (pre-impl review corrected the naive framing):** the
worker thread BLOCKS inside `bridge.run_on_main` for the whole encode (`bridge.py` runs `op.fn()` to
completion inline in `drain()`), so there is NO worker-side two-phase commit. The deferral lives in
the MAIN-THREAD frame loop holding the render op back one frame:
- `MainThreadOp` gains a `defer: bool` (default False) marker; `run_on_main` takes `defer=True` for the
  render ops ONLY. Ordinary GL ops (`set_uniform`, edit recompiles) keep running same-frame — `ui.py`'s
  comment explicitly relies on same-frame execution for the recompile-this-frame behavior, so the drain
  must defer ONLY tagged render ops, never all ops.
- `drain()`: when it pulls a `defer` op the FIRST time it's seen, it sets an `App.copilot_rendering`
  busy flag and puts the op back (or holds it) WITHOUT running it, so that frame paints the modal; the
  NEXT frame's drain runs it and clears the flag. Exactly one frame of deferral (the worker's
  `op.done.wait` clock starts at enqueue, so a multi-frame hold would eat the `render_op_timeout_s=60`
  budget — keep it to one frame).
- The "Rendering…" modal draws from `App.copilot_rendering` in the main-window block (mirroring the
  existing `is_*_open` modal pattern). Decide explicitly whether it joins `any_popup_open()` (it
  suppresses the per-frame node preview render while open — acceptable for the freeze frame, but must
  be a conscious choice, not a default).
- Publish is NOT covered by the modal — it polls responsively (short bridge ops + sleep), so its cue is
  the D1 status line, not a freeze.

### D4 — The low-severity correctness footguns (batched)
All audit-confirmed, all small, all behavior-only:
- **Retry-cap scope:** `consecutive_failed_edits` keys on the three shader-edit tools, not all
  `is_mutating` — a failed render/publish must not trip the "edit kept not applying" giveup.
- **Unresolvable-target vs stale:** an unknown node-id / invalid lib path reject must NOT set
  `stale=True` (which exempts it from the retry cap); reserve `stale` for freshness rejects.
- **Empty-handle guard:** `_copilot_resolve_node_id("")` must return None (today `startswith("")`
  matches the sole node) for required-target tools (delete/switch), and/or `min_length=1` on the field.
- **Malformed-args cap:** a bad-args result for a mutating tool counts toward the edit-retry cap.
- **read_shader missing-report:** diff against the model's ORIGINAL handles, not the returned short
  ids (a full-id/long-prefix read succeeds yet is reported missing). Dedup resolved node-ids.

### D5 — Lazy tool-catalogue — DEFERRED to its own follow-up slice (Q2 RESOLVED)
NOT this wave. The `eager`/`category` fields already exist on `ToolDefinition`; the
`search_tools`/`list_tools` lazy path is a prompt-cache-prefix cost optimization, not a
legibility/correctness fix, so it doesn't gate shipping. Stays the `todo.md` deferral; built next.

## Files touched (anticipated)

- `copilot/state.py` — `ChatState.status`; a tool-result field on Message (D1).
- `copilot/session.py` — `_apply_event` AgentStatus/result plumbing (D1); sanitize assistant text into
  the rendered Message AND `history` before commit (D2).
- `copilot/agent.py` — retry-cap scope (D4), malformed-args cap (D4); the render two-phase staging seam
  feeds off the existing bridge (D3).
- `copilot/prompt.py` — the 1-sentence ASCII-only nudge (D2).
- `copilot/tools/shader.py` — read_shader missing-report + dedup (D4).
- `app.py` — empty-handle guard (D4), unresolvable-target-not-stale (D4); the two-phase render modal
  staging (D3, App owns the frame loop + the busy-modal state).
- `copilot/text_render.py` (new leaf) — `sanitize_display` (D2).
- `widgets/copilot_chat.py` — sanitize at render, status-line draw, tool-result line (D1/D2); the
  "Rendering…" modal draw (D3).
- (Already landed pre-wave: `glsl_lex.py` comment guard, `app.py` array reject, `agent.py`
  double-escape guard, `session.py` history-commit.)

## Manual verification

- Run the app, open chat, send a turn that calls a tool: the status line shows per-tool progress; the
  tool card shows the result (render path / publish URL); a failed tool shows its error.
- Ask the agent something whose reply contains an arrow / em-dash / ellipsis: it renders as ASCII, no
  `?` boxes.
- Trigger a render: a cue shows during the freeze, not a bare frozen "thinking…".
- (D4 cases are headlessly assertable — unit-test the resolve/retry/dedup logic.)

## Review history

- **Pre-impl review (1 agent, the two risky seams):** corrected D3's naive "frame N / frame N+1
  worker-side two-phase commit" — the worker BLOCKS in `bridge.run_on_main` the whole encode, so the
  deferral must live in the main-thread `drain()` holding a TAGGED render op back one frame (needs a
  `defer` marker on `MainThreadOp`; must NOT defer ordinary same-frame ops). Pinned D2's exact sanitize
  sites (history-commit + Message-materialize + draw-time preview, never per-delta) and the
  content-only guard (never through `_assistant_message`/`_tool_message`/`_parse_args`). Both folded
  into D2/D3 above.

## Open questions for the user

All resolved at plan-lock:
- **Q1 (RESOLVED):** the full two-phase paint-then-freeze "Rendering…" modal IS in this wave (D3).
- **Q2 (RESOLVED):** the lazy catalogue is DEFERRED to its own follow-up slice (D5).
- **Q3 (RESOLVED):** an unmappable char renders a literal `?`; PLUS a 1-sentence prompt nudge for
  ASCII-only output; PLUS the sanitized text is fed back into `history` (D2).
