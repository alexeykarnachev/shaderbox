# 020 · 24 — Copilot UI/UX polish wave (findings log)

A second, broader UI/UX pass over the copilot chat, driven by a live maintainer walkthrough. The basic
chat machinery is built (020·05 design → 020·20/21/23 polish); this wave collects the **rough edges,
gaps, and QoL opportunities** that only surface when you actually drive the app, then fixes them as a
dedicated wave before ship.

Sibling to `20_ui_ux_polish.md` (the first, audit-driven polish wave — that one closed the
transcript-legibility + glyph-render + correctness-footgun gaps). This one is **maintainer-experience
driven**: the findings come from real use, not an audit.

> **STATUS: IN PROGRESS — fix-as-we-go.** The maintainer drives the app and hands findings one at a
> time; simple ones are fixed immediately (gated by `make check` + `make smoke` + a maintainer live
> pass), bigger ones get filed for a later batch. Each finding is logged below as it's handled, with
> its resolution. Do NOT fix without an explicit go on the harder ones; the maintainer says when a
> finding should be filed-not-fixed.

---

## Goal
- **Capture + fix** the UI/UX rough edges, inconveniences, and QoL gaps the maintainer hits driving
  the copilot chat live — the simple ones inline, the larger ones batched.

## Out of scope
- Larger features that surface during the walkthrough get their own spec / deferral, not a blind fix.

---

## Findings

<!-- Per-finding shape (append one block per finding, in arrival order — do NOT renumber or reorder):

### F<NN> — <short title>   [fixed | filed | deferred]
- **Where:** <UI surface>
- **Observed:** <what the maintainer saw, faithful to their words>
- **Resolution:** <what was done + commit/file, OR the deferral trigger if not fixed now>

Keep entries faithful. A simple fix gets one or two lines of resolution, not a saga. -->

### F01 — copilot chat input not focused; nav outline on the input   [fixed]
- **Where:** the floating copilot chat window — the message text input.
- **Observed:** (a) on app start with the chat restored-open, and after pressing Enter to send, the
  input was not focused — had to mouse-click it to type. (b) After wiring auto-focus, an accent
  outline drew around the input (a programmatic focus shows the nav cursor; a mouse click doesn't),
  including a one-frame flash on open.
- **Resolution:** one-shot `copilot_focus_pending` latch drives both the window
  (`set_next_window_focus`) and the input (`set_keyboard_focus_here`), consumed once at the input
  draw — armed by `focus_copilot()` (Ctrl+J / bar button / startup-restore) and after send. The
  nav outline (steady-state + the NavInit-on-appear flash) is killed by `WindowFlags_.no_nav_inputs`
  on the chat window (the input stays typeable). The every-frame `set_keyboard_focus_here` and the
  `set_nav_cursor_visible(False)` band-aid were both tried and rejected as wrong — durable imgui
  lessons folded into `/imgui-ui` §7.5 + §8. Clicking the chat *body* (not the input) leaves the
  input unfocused — intended (the one-shot must not fight other clicks).

### F02 — header buttons blur together; no context indicator   [fixed]
- **Where:** the floating copilot chat window — the top bar (`Layout: corner   Clear  Close`).
- **Observed:** the three ghost buttons read as one undifferentiated strip — no separation, and the
  destructive `Clear` looked identical to the benign `Layout`/`Close` (misclick risk). Then: the
  `Layout: <name>` text took too much room (wanted a compact icon), and the freed space should hold a
  context-fill indicator.
- **Resolution (incremental):** `Clear` → `danger_button` (red) + the window-action cluster
  (`Clear`/`Close`) right-aligned, `Layout` left as a control. Corner/strip presets enlarged; FREE
  pos/size remembered across a preset round-trip (`copilot_free_rect`/`copilot_prev_layout`). The
  compact icon + the context indicator grew into a specced feature (025) — the `Layout:` text became
  a drawn box-in-frame icon, and two thin stacked usage bars show the previous turn's input/output
  tokens vs budget. Spec + flow: `25_context_fill_indicator.md`.

### F03 — chat input still editable while a turn runs   [fixed]
- **Where:** the chat message input, during an in-flight turn.
- **Observed:** the Send button hides while the agent works, but the text field stayed editable.
- **Resolution:** the input is wrapped in `begin_disabled(in_flight)` — frozen until the turn ends.

### F04 — Ctrl+W to cycle the chat layout   [fixed]
- **Where:** keyboard, anywhere the global command scope fires.
- **Resolution:** new `CommandId.CYCLE_COPILOT_LAYOUT` (Ctrl+W, was free) → `App.cycle_copilot_layout`
  → `CopilotLayout.next()` (the canonical cycle now lives on the enum; the widget's `_NEXT_LAYOUT`
  dict was retired in favour of it).

### F05 — long input overflowed horizontally   [fixed]
- **Where:** the chat input with a message wider than the field.
- **Observed:** a long message scrolled off the right edge instead of wrapping.
- **Resolution:** the input is now `input_text_multiline` with `InputTextFlags_.word_wrap` (1.92.801
  supports it — the prior "multiline can't wrap" limitation is gone, `/imgui-ui` §8). Enter still
  sends (`enter_returns_true`); Ctrl+Enter inserts a newline (`ctrl_enter_for_new_line`). The history
  child reserves the 2-line input height via the shared `_input_height()`.

### F06 — held-backspace repeat felt sluggish vs the OS   [fixed]
- **Where:** any text input (global imgui key-repeat).
- **Observed:** holding backspace deleted slower than the desktop's native key-repeat.
- **Resolution:** `io.key_repeat_delay`/`key_repeat_rate` set to the maintainer's X11 values
  (500ms delay, ~33/s) at startup — imgui's own globals, app-wide. NOT a live OS read (no portable
  API); these are also the standard X11/GNOME defaults, so a sensible shipped default too.

### F07 — focus jumps chat -> editor when the copilot manipulates a shader   [fixed]
- **Where:** the chat, while the copilot creates/switches the current node mid-turn.
- **Observed:** focus left the chat for the code editor; a keystroke meant for the chat then landed
  in the editor (the ROOT CAUSE of the stray `a` that broke a shader compile earlier this session).
- **Resolution:** `copilot_chat.draw()` re-asserts `set_next_window_focus()` EVERY frame while
  `state.in_flight`, not just on the `copilot_focus_pending` one-shot — it out-races the TextEditor's
  first-render focus grab (`/imgui-ui §8`) that fires when the new current-node editor session renders.

### F08 — lock all user input while the copilot works   [fixed]
- **Where:** the right app panel (uniform sliders, tab controls, share) during an in-flight turn.
- **Observed:** the panel stayed interactive — its inputs could race the values the worker reads.
- **Resolution:** the `app_panel` body is wrapped in `begin_disabled(copilot_turn_active)`
  (try/finally-balanced so an exception can't leave the disabled stack unbalanced). The editor
  already had its read-only lock; the chat input is disabled (F03); the chat's own Stop stays live
  (separate window). Node/project mutations were already `_copilot_busy_blocked`.

### F09 — active-region outline leaks through the chat window   [fixed]
- **Where:** any region's accent nav-outline vs the floating chat. (Reported on the node grid in the
  bottom-right panel — the actual culprit was the GRID outline, not the editor.)
- **Observed:** a region's outline drew on top of the opaque chat window.
- **Resolution (two parts):** (1) `active_region_outline()` drew on the FOREGROUND draw list
  (always-on-top), so it punched through the chat — fixed IN THE PRIMITIVE to draw on the child's OWN
  window draw list (inset so the clip can't cut it), z-ordering beneath later windows. (2) That alone
  left a SECOND bug: `active_region` is sticky (stays GRID when the chat takes focus), so the grid kept
  drawing its outline ALONGSIDE the chat's — two "active" windows. Fixed by encapsulating the whole
  outline policy in `App.region_outline_visible(region)` (active + no popup + chat-not-focused); the
  three region draw-sites (editor/panel/grid) call it and reference the copilot for the outline NOWHERE.
  The earlier `is_copilot_open`-spread-into-region-code was a layering violation, reverted. (3) The
  window-draw-list switch then clipped the CHAT's own outline under its title bar (the chat has one;
  the borderless regions don't). Fixed with an `active_region_outline(foreground=True)` flag: regions
  use the default (window list, z-orders under), the chat uses foreground (covers the title bar, and
  being topmost it leaks over nothing but a gated modal). NOTE (pre-existing, not touched): the
  `active_region` ASSIGNMENT gates still carry `not copilot_focused` — same smell, separate concern;
  left for a focused decoupling pass.

### F10 — editor outline stays after clicking the render canvas (decoupled from the dim)   [fixed]
- **Where:** the code editor's focus outline vs its unfocused-dim.
- **Observed:** clicking the shader render canvas dims the editor (it lost focus) but leaves its
  active-region outline drawn — two signals disagreeing.
- **Cause:** the dim keys on LIVE focus (`editor_focused`); the outline keyed on the STICKY
  `active_region`, which stays EDITOR because the render canvas (a plain `image`) isn't one of the
  three nav regions, so nothing moves `active_region` off EDITOR.
- **Resolution:** the editor is a focus-stop, not a sticky region — its outline now gates on
  `editor_focused or editor_focus_requested` (the latch avoids a one-frame flicker on a chord-move
  into the editor), matching the dim condition. Grid/panel keep the sticky `active_region` (they ARE
  nav regions). Editor: focused = bright + outlined; unfocused = dim + no outline, together.

### F11 — chat layout breaks (transcript overflows horizontally) while a turn runs   [fixed]
- **Where:** the chat transcript during an in-flight turn.
- **Observed:** after pressing Enter to send, the message text ran off the right edge (clipped
  mid-word); only while the copilot was working.
- **Cause:** regression from F05's multiline input. The in-flight branch sized the input
  `input_w = -1.0` (full window width) then `same_line()` + Stop — a full-width multiline box
  reserves its width as real content, so the trailing button pushed content past the window edge and
  forced the whole transcript to wrap at that too-wide extent. (The old single-line input tolerated
  `-1.0` — it just clipped its own text.)
- **Resolution:** the two states (idle vs in-flight) are now ONE layout — identical input box
  (`_send_button_offset()` width, `_input_height()`) + an identical trailing button slot
  (`BTN_SM_W`). The only state-dependent differences are the ones that should differ: the input is
  `begin_disabled` mid-turn, and the button is Send (primary) idle / Stop (ghost) working — same slot,
  same geometry, so the row can't shift between modes. (The earlier `-1.0`-vs-reserved width fork was
  the divergence that broke the layout.)

### F12 — Send button not flush to the input's right edge   [fixed]
- **Where:** the chat input's trailing Send/Stop button.
- **Observed:** a few-px gap between the button's right edge and the window border.
- **Cause:** `_send_button_offset` reserved `BTN_SM_W + SPACE.SM` (4) but `same_line()` inserts
  `item_spacing.x` (`SPACE.MD` = 8), so the button stopped short.
- **Resolution:** reserve the real `same_line` gap (`item_spacing.x`). Verified flush (0.00px).

### F13 — last-turn usage bars don't survive an app restart   [fixed]
- **Where:** the header usage bars after closing + reopening the app / project.
- **Observed:** the bars reset to empty on restart (the stats were lost).
- **Cause:** 025 Decision 12 made `last_turn_usage` transient (not persisted) by design — wrong call.
- **Resolution:** persist it in `ConversationStore` beside the session `usage` (`_VERSION` 5 -> 6;
  nullable `_UsageModel` + `to_last_turn_usage()`, restored in `load_conversation`). Old v5 files
  load fail-soft (missing field -> None -> empty bars, no crash). Clear still empties them. Spec
  Decision 12 + Files-touched + manual-verification updated. Round-trip + back-compat verified.

### F14 — chat polish batch: gap, focus-on-send, bubbles, names, copy icon   [fixed]
- **Where:** the copilot chat header gap + the message transcript.
- **Observed (maintainer):** (a) gauge sat flush against Clear — wanted a gap; (b) the input lost
  focus after pressing Enter to send; (c) the per-message under-text `Copy` button was clutter;
  (d) "you" was labelled but the copilot's messages had no name; (e) messages divided by bare
  separators read as boring.
- **Resolution:** (a) `gauge_w` reserves `SPACE.LG` before the cluster. (b) the input is
  `begin_disabled` for the whole turn so the on-send focus latch was lost mid-turn; re-arm
  `copilot_focus_pending` on the turn-done transition (`ui.py`) so it re-focuses once editable.
  (c-e) user/assistant messages now render as rounded BUBBLES (`ui_primitives.message_bubble` —
  bordered auto-height child; user = faint accent tint + "you" in accent, assistant = surface bg +
  "copilot" in blue `STATE_INFO`) with a drawn corner copy glyph (`copy_icon_button`, top-right,
  allow_overlap) replacing the text button; tool/error/pending lines use a `dummy` gap, not a rule.
  `BUBBLE_ROUNDING` token. Headless-verified no SetCursorPos assert across all message kinds.

### F15 — edit_shader comment-loss guard false-positive (trace-surfaced)   [fixed]
- **Where:** `edit_shader` / the comment-loss guard (`backend.apply_shader_edit` +
  `glsl_lex.span_has_comment`).
- **Observed:** the 2026-06-08 live session's only two `edit_shader` failures were BOTH this guard
  ("that region spans a comment your old_str doesn't reproduce"), each forcing a wasted
  `replace_lines` retry. In both, the model's `old_str` DID include the comment and was deliberately
  rewriting the block — a false-positive. (The agent's own end-of-session feedback flagged
  edit_shader brittleness; the trace pinned the real cause to the guard, not whitespace.)
- **Cause:** `token_match` is comment-invariant, so the guard checked only whether the matched SOURCE
  span had a comment — never whether `old_str` reproduced it. It couldn't tell an accidental silent
  drop from an intentional rewrite, so it blocked both.
- **Resolution:** `span_has_comment` → `span_drops_comment(src, s, e, old_str)`: fire only when the
  span has a comment that `old_str` does NOT reproduce (a genuine silent drop); allow a rewrite that
  quotes the comment. `glsl_lex.comments_in` (normalized, multiset-aware) backs it. The guard's error
  message was already accurate for the new condition. 5 unit tests (both directions). NOT a
  prompt-level fix — the agent was already phrasing the edit correctly; the false-positive was wholly
  in the tool.

### F16 — draggable feed/input splitter (input keeps height on window resize)   [fixed]
- **Where:** the chat transcript feed vs the message input.
- **Observed:** the input was a fixed 2 lines; the maintainer wanted to make it bigger manually, and
  resizing the whole chat window must NOT shrink/hide the input.
- **Resolution:** a feed/input splitter (the `ui.py` editor-splitter idiom — `invisible_button` +
  `is_item_active` + `mouse_delta.y` + the `resize_ns_cursor`; imgui has no built-in sibling-splitter).
  The INPUT keeps its stored height (`UIAppState.copilot_input_h`, persisted, clamped); the FEED takes
  the remainder so it flexes with the window and the input is never hidden. Drag down = bigger feed,
  up = bigger input. (First tried `ChildFlags_.resize_y` on the feed — WRONG: resize_y stores an
  absolute size that fights window-flex and squeezes the input; reverted. A stored-height splitter is
  the correct model.)

### F17 — chat feed opens scrolled to the top, not the bottom   [fixed]
- **Where:** the chat feed on launch / project load.
- **Observed:** a restored conversation opened scrolled to the top — had to scroll down each time.
  (The feed only auto-scrolled DURING an in-flight turn.)
- **Resolution:** `App.copilot_scroll_pending` one-shot — armed at init (covers first launch) +
  after `load_conversation` (project switch), consumed at the feed draw (`set_scroll_here_y(1.0)`).
  One-shot, so it never fights a manual scroll; only re-fires on a fresh conversation load.
