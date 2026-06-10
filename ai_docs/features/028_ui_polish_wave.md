# 028 — UI/UX polish wave (findings log)

An app-wide UI/UX polish pass, driven by a live maintainer walkthrough, in the run-up to shipping the
copilot. The app's surfaces are built and work; this wave collects the **rough edges, padding/scroll
glitches, and layout QoL gaps** that only surface when you actually drive the app, and fixes them as a
dedicated wave before ship.

Sibling to `020_copilot_agent/24_ui_polish_wave.md` (the copilot-chat-only polish wave) and
`020_copilot_agent/20_ui_ux_polish.md` (the audit-driven one). This wave is the same **maintainer-
experience-driven, fix-as-we-go** shape but app-wide (the New Node modal, the copilot chat, and
whatever other surfaces the walkthrough turns up), not copilot-scoped.

> **STATUS: IN PROGRESS — fix-as-we-go.** The maintainer drives the app and hands findings one at a
> time (live audio). Simple ones are fixed immediately (gated by `make check` + `make smoke` + a
> maintainer live pass); larger ones get filed as a `todo.md` deferral or their own spec, not a blind
> fix. Each finding is logged below as it's handled, with its resolution. The maintainer says when a
> finding should be filed-not-fixed.

---

## Goal
- **Capture + fix** the UI/UX rough edges the maintainer hits driving the app live — the simple ones
  inline, the larger ones batched/filed.

## Out of scope
- Larger features surfaced during the walkthrough get their own spec / `todo.md` deferral, not a blind
  fix.

## Manual verification
- Each finding is verified by a maintainer `make run` pass (visual; the dev box can't be screenshotted
  by the agent — `/imgui-ui §0`). Headless `make smoke` + `make check` gate the structural correctness
  of each change first.

---

## Findings

<!-- Per-finding shape (append one block per finding, in arrival order — do NOT renumber or reorder):

### F<NN> — <short title>   [fixed | filed | deferred]
- **Where:** <UI surface>
- **Observed:** <what the maintainer saw, faithful to their words>
- **Resolution:** <what was done + commit/file, OR the deferral trigger if not fixed now>

Keep entries faithful. A simple fix gets one or two lines of resolution, not a saga. -->

### F01–F03 — New Node modal: padding / outer scrollbar / description placement   [fixed]
- **Where:** the New Node modal (`popups/node_creator.py`).
- **Observed (three coupled symptoms):** (F01) Create/Cancel flush against the modal's bottom border
  while every other edge had normal padding. (F02) TWO scrollbars — the inner shader-grid one is
  correct, but the modal ALSO grew an outer scrollbar that scrolled only a couple of px (and that few-px
  scroll is what pushed the buttons against the bottom). (F03) the description should sit directly under
  the shader grid, but on a wide modal the grid was short while its child still filled all the reserved
  height, so the description floated far below the last row with a dead gap.
- **Root cause (all three):** `_draw_body` sized the grid child as `avail − _DESC_SLOT_H − BTN_SM_H`,
  accounting for none of the `item_spacing.y` gaps between the three stacked blocks (grid → desc →
  action row); the column overflowed the content region by ~2×spacing → outer scrollbar (F02) + buttons
  shoved down (F01). And the grid child filling all remaining height left the description stranded (F03).
- **Resolution (maintainer-chosen shape — supersedes the original "shrink-to-content" sketch):** the
  modal is now a FIXED-SIZE, non-resizable window holding a fixed **3-column** grid capped at **5 rows**
  (the grid child scrolls past 5). `node_creator._grid_dims` computes the grid child height + modal
  client width from the live style (`THUMB_LG`, `item_spacing`, `window_padding`, `scrollbar_size` —
  DPI-correct), and `draw_node_creator` sizes the modal to exactly fit grid + desc slot + action row +
  the two inter-block gaps + padding. So the grid is exactly as tall as its rows (description follows
  directly under it, F03), nothing overflows (no outer scrollbar, F02), and the bottom padding is
  uniform (F01). Window flags: `no_resize | no_scrollbar | no_scroll_with_mouse`; `modal_window` gained
  a `flags` pass-through + a `fixed_size` arg (forces `Cond_.always` so a stale imgui.ini size can't
  persist). The width-derived `_grid_cols` column math is gone (columns are the constant `_GRID_COLS`).
  `scripts/smoke.py` now opens the modal for a stretch of frames so its draw path is exercised headlessly
  (it never opened on its own). **Geometry gotcha (second pass):** `set_next_window_size` sizes the WHOLE
  window, so `modal_h` must include the TITLE BAR (`== get_frame_height()`) on top of the window padding —
  the first cut omitted it, under-sizing the window by ~20px → the bottom button row clipped + residual
  content scroll. Verified via a headless geometry probe (content region now fits the body exactly,
  0px overflow) + `make check` + `make smoke` green; visual = maintainer `make run` pass.

### F04 — explicit-dir processes hijack the active-project pointer (created nodes "lost")   [fixed]
- **Where:** `App.__init__` / `App._init` (root); `scripts/smoke.py` + `tests/conftest.py` (callers).
- **Observed:** after running smoke OR the pytest suite, the next `make run` opened an empty project;
  a node created + worked on in that session was gone on the following launch. (Files never deleted;
  `projects/dev/nodes/` intact — the user was unknowingly working inside a throwaway tmp project the
  pointer had been redirected to.)
- **Cause:** `App._init` UNCONDITIONALLY rewrote `app_data_dir()/project_dir` to the project it loaded.
  Both `scripts/smoke.py` AND the pytest `app` fixture construct `App(project_dir=<tmp>)`, so EITHER
  clobbered the pointer to a tmp dir that's deleted on exit → next real launch read a dead pointer →
  fell back to an empty/default project. (First fix attempt patched only smoke; the pytest fixture has
  the same bug — running `pytest` re-clobbered it.)
- **Resolution (single root):** `App.__init__` now computes `persist_pointer = (project_dir is None)`
  and threads it to `_init`, which writes the pointer only when `persist_pointer` is True. A real launch
  (dir resolved from the saved pointer/default) persists; an explicit-dir test/smoke process does NOT.
  `open_project` keeps the default `persist_pointer=True` (a real user action SHOULD persist). The
  earlier smoke band-aid was removed (redundant — one canonical home). Pinned as a Design decision in
  `conventions.md`. Verified: pointer unchanged across both a pytest run and a headless App construction.

### F05 — copilot chat: Send button glued to the input's TOP when the input is grown   [fixed]
- **Where:** `widgets/copilot_chat.py` — the message input + trailing Send/Stop button.
- **Observed:** dragging the feed/input splitter to make the input taller left Send aligned to the
  input's TOP edge (their top borders coincided); it should bottom-align with the input.
- **Resolution:** after `same_line()`, push the button cursor down by `input_h − frame_height` so the
  button's bottom lines up with the input's bottom (the button submits after the move — the sanctioned
  SetCursorPos pattern, `/imgui-ui §4`).

### F07 — copilot chat: in-flight "thinking…"/status line not in a bubble   [fixed]
- **Where:** `widgets/copilot_chat.py` — the in-flight status caption in the feed.
- **Observed:** the `thinking…`/status text touched the chat's left edge, unlike the message bubbles
  (it was a bare `caption_text`, not wrapped like the tool/error lines).
- **Resolution:** subsumed by F06 — the live status now renders INSIDE the turn snippet's invisible
  `message_bubble(bordered=False)` (above the square bar), so it carries the bubble's left indent. The
  separate standalone status block in `_draw_transcript` was removed (the snippet is its single home).

### F06 — copilot chat: collapse the tool-status column into a compact turn snippet   [fixed — needs make run]
- **Where:** `copilot/state.py` (Message model), `copilot/session.py` (`pump_events` mapping),
  `copilot/persistence.py` (v8→v9), `widgets/copilot_chat.py` (render), `theme.py` (a token or two).
- **Observed/wanted:** the per-action service lines ("edited shader", "1 compile error", …) stack
  vertically and clutter the chat. Replace them, per turn, with ONE compact snippet: a square-segment
  bar (one square per step — green ok / red fail) + a self-updating status line; on turn end the line
  becomes a mini-stat (N tools · N tokens · $cost); hover → a per-step list + the turn token/cost total.

#### Data audit (what the pipeline already tracks)
- Per **step**: `AgentToolCard(name, ok, payload, widget, …)` is emitted per tool call → name +
  success/fail per step is available (the squares).
- The **status line** already exists (`AgentStatus` → `state.status`, self-updating).
- Per **turn**: summed `usage` (input/output tokens) + `cost_usd` + `total_tool_calls` (the mini-stat).
- **NOT available:** tokens/cost per *individual* tool call — usage is billed per LLM iteration and one
  iteration fires many tool calls. So the hover shows tool name + ok/fail per step and the turn-total
  tokens/cost (Decision: option A — no fabricated per-call numbers).

#### Design decisions (locked)
1. **One `turn_snippet` Message per turn**, not N `tool_status` Messages. New `MessageRole`
   member `turn_snippet`; new `Message` fields `steps: list[StepRecord]` (`name`, `ok`) +
   `snippet_stats: TurnStats | None` (this turn's own stats, set at turn end — distinct from the
   session's `last_turn`). `enqueue_turn` appends the empty snippet right after the user message; each
   `AgentToolCard` appends a `StepRecord`; `AgentTurnDone` writes `snippet_stats`.
2. **Render** (invisible bubble, left-indent matching the others): a row of small filled squares
   (`STATE_OK` / `STATE_ERROR`, drawn via the draw list — no font dep) wrapping at the bubble width;
   below it, while in-flight the live `state.status` line, else `N tools · N tokens · $cost`. Hover the
   bubble → tooltip: numbered `name ✓/✗` per step + the turn token/cost total.
3. **Result widgets stay visible.** Only the render/publish tools emit a `widget` (Reveal render / Open
   in Telegram|YouTube) — actionable, always-gated, rare. Those keep their own visible result line
   (a `tool_status` Message with the widget, as today); only the WIDGET-LESS noise (edit/compile/grep/
   read) folds into the square bar.
4. **Persistence v8→v9.** `_MessageModel` gains `steps` + `snippet_stats` (defaulted → old files load
   fail-soft); old persisted `tool_status` lines still render as plain text (unknown role = plain text).
- **Status:** IMPLEMENTED + tested. `state.StepRecord` + `turn_snippet` role + `Message.steps`/
  `snippet_stats`; `session.enqueue_turn` appends the snippet, `_apply_event` folds each `AgentToolCard`
  into it (widget cards keep a visible `tool_status` line) + `AgentTurnDone` writes the stats;
  `ui_primitives.step_squares` draws the bar; `copilot_chat._draw_turn_snippet` renders bar + status/
  stat line + hover breakdown; persistence v9 + `_stats_model`/`_stats_or_none` helpers (collapsed the
  duplicated `to_last_turn`). Tests: snippet round-trip + pre-v9 default (`test_conversation_persistence`)
  + the event-mapping (`test_copilot_loop::test_turn_snippet_collects_steps_not_status_lines`). 246
  tests + `make check` + `make smoke` green. Visual = maintainer `make run`.

### F08 — focus lost / "no element focused" after a modal closes over the chat   [fixed — live-verified]
- **Where:** `App` (focus model), `hotkeys.py::_handle_escape`, `ui.py` (per-frame reconcile).
- **Observed:** chat input focused → Ctrl+N (New Node) → Cancel or Esc → the chat cursor vanished.
  After Esc specifically, NOTHING was focused (the editor should always hold a focus stop). Re-clicking
  the input only flashed the caret one frame and wouldn't stick.
- **Cause (two bugs):** (1) `_handle_escape` set `editor_defocus_requested = True` UNCONDITIONALLY
  whenever Esc had a job — so Esc closing a POPUP also defocused the editor, leaving nothing focused.
  (2) Closing the modal (Cancel or Esc) never returned focus to the surface that had it before the
  modal opened.
- **Resolution (live-verified via temporary `[FOCUS]` trace logging, then stripped):** (1) `_handle_escape`
  now dismisses ONE thing most-modal-first (popup → palette → chat → editor caret); closing a popup no
  longer touches editor/chat focus. (2) The focus-restore. FIRST cut (a next-frame open/close edge
  detector in `reconcile_popup_focus`) was WRONG — the open edge read `copilot_focused` a frame too late,
  AFTER the popup had already clobbered it to False, so it snapshotted "nobody" and the restore no-op'd
  (the trace caught `snapshot=none` with the chat visibly focused). ROOT FIX: capture at popup-OPEN time,
  in the openers (`App._open_popup`, which the 4 `open_*` funnel through) — they run in `dispatch_commands`
  BEFORE any window draws, so `copilot_focused` still holds the true pre-popup value. `reconcile_popup_focus`
  now only RESTORES on the close edge. SIMPLIFIED (maintainer push): the `FocusOwner` enum + a dedicated
  snapshot method collapsed to ONE bool `_chat_focused_before_popup` — the editor case reads the already
  sticky `editor_was_ever_focused` directly (it survives the popup), so only the chat (whose
  `copilot_focused` isn't sticky) needs capturing. Live-confirmed: Ctrl+N → Esc returns the caret active.

### F09 — thinking indicator: kill the pop-in jitter + design the square language   [fixed]
- **Where:** `ui_primitives.step_squares`, `copilot_chat._snippet_squares`/`_draw_turn_snippet`.
- **Observed:** on send, "thinking" showed, THEN the square bar popped in and pushed "thinking" down —
  a layout jitter. Wanted: a blinking square shown immediately. Plus a design question — the agent may
  call ZERO tools, so what does the bar mean / show then?
- **Design (locked with the maintainer — gray head + blue answer):** the bar is present from frame 1, so
  it never changes height (no pop-in). Square language (`_snippet_squares`, unit-tested):
  resolved step = solid green `STATE_OK` (ok) / red `STATE_ERROR` (fail); the live/pending HEAD = a
  pulsing gray `FG_DIM` square (alpha breathes via `imgui.get_time()`, a triangle wave) shown the whole
  time the turn runs; a CLEANLY-finished turn caps with one solid info-blue `STATE_INFO` ANSWER square
  (so a zero-tool reply renders as a single blue square; a multi-tool turn ends `[green]…[blue]`); an
  error/cancel turn (no stats) adds no cap (bar only). `step_squares` was generalised to take a
  `list[(color, pulse)]` (caller owns the colour language; the primitive stays feature-agnostic) and now
  reserves a count-independent height (the no-jitter guarantee). Test:
  `test_copilot_loop::test_snippet_square_language` covers all five states. Visual = maintainer `make run`.

### F10 — answer-square color contrast + human-readable step names   [fixed]
- **Where:** `theme.py` (`STATE_INFO`), `copilot/state.py` (`tool_label`), `copilot_chat._snippet_tooltip`.
- **Observed:** the blue answer square (`STATE_INFO` = `blue_b` #83a598, a desaturated teal) read too
  close to the aqua-green ok square (`STATE_OK` = `aqua_b` #8ec07c). Also: the hover breakdown showed raw
  tool ids (`create_node`) instead of human verbs.
- **Resolution (color — global one-token change, brainstormed by a 3-agent panel + a CIEDE2000/colorblind
  analysis):** `STATE_INFO` → `blue_n` #458588 (saturated blue). Nearly doubles the ok-vs-answer
  separation (ΔE 16→29) and is darker, so the answer cap reads as a distinct kind of square. KEY finding:
  `STATE_OK` stays `aqua_b` (NOT switched to lime `green_b`) — lime collapses into the red fail square
  under deuteranopia (ΔE 9, below the confusion floor), so aqua is the colorblind-safe ok hue. The
  warm-accent alternative (yellow/orange answer) was rejected (collides with the yellow accent /
  STATE_WARN). Invariant holds; the change also improves the other 4 `STATE_INFO` sites (links, uniform
  jump, "copilot" name).
- **Resolution (names):** `_TOOL_VERBS` (past-tense: "Created node", "Edited shader") moved from
  `session.py` to `state.py` as `tool_label()` (one shared home); the hover breakdown uses it
  (`1. Read shader / 2. Created node / 3. Edited shader (failed)`). The live status line under the bar
  keeps its separate present-tense phrasing ("Compiling…"); the hover lists FINISHED steps so past tense
  is correct there.

### F11 — chat input left edge not aligned with the message bubbles   [fixed]
- **Where:** `widgets/copilot_chat._draw_transcript` — the input + Send row.
- **Observed:** the input field's left edge nearly touched the chat border, sitting `SPACE.SM` left of
  the message bubbles (which self-indent `SPACE.SM` via `message_bubble`).
- **Resolution:** wrap the input row in `imgui.indent(SPACE.SM)` / `unindent`, and `_send_button_offset`
  gained a `right_inset` arg (indent moves only the left edge) so Send ends `SPACE.SM` short of the
  window edge too — the input row's left AND right margins now match the bubbles'.

---

## Review history
<!-- Design disagreements resolved by the main agent (per dev_flow.md feature flow step 4/6) land here. -->
- **F01–F03 layout shape.** The initial sketch (auto-shrink the grid to content, keep the 490×530 box)
  left a vertical void with few templates. The maintainer chose instead a fixed, non-resizable modal
  with a fixed 3-col grid capped at 5 rows (scrollbar past 5) — recorded above as the landed shape.
- **F06 post-impl review (3 adversarial reviewers: correctness / user-intent / persistence-conventions).**
  - REAL (fixed): an error/cancel/torn/crashed-mid-save turn left the snippet rendering a frozen
    "thinking…" — the `st is None` branch didn't check `in_flight`. Fixed: the live status line now
    gates on `live` (in_flight + no stats); a finished-statless snippet shows only its squares (the
    error/cancel reason is already a separate message). Regression test
    `test_errored_turn_leaves_snippet_finished_not_live`.
  - FALSE POSITIVE (rejected): "the bar/bubble is empty during pure thinking" — while live, the status
    line renders inside the bubble and the squares grow as tool calls arrive; that's intended.
  - CONFIRMED constraint: per-call token/cost is genuinely unavailable (usage is per-LLM-iteration, not
    per tool call — reviewer verified against `agent.py`), so the hover shows tool name + ok/fail per
    step + the turn total (option A). Persistence/conventions reviewer: PASS (v8→v9 follows the
    established fail-soft default pattern; forward-refs resolved; dedup via `_stats_*` clean).
