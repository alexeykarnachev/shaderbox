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
- **Resolution (root, no coupling):** `active_region_outline()` drew on the FOREGROUND draw list
  (always-on-top, ignores window stacking) → it punched through the later-drawn chat. Fixed IN THE
  PRIMITIVE: it now draws on the child's OWN window draw list (inset by the stroke width so the clip
  can't cut it), so it z-orders naturally beneath any floating window. No region code references the
  copilot for the outline — an earlier `is_copilot_open` gate at the three draw-sites was a layering
  violation (grid/editor/panel must not know the copilot exists) and was reverted. NOTE (pre-existing,
  not touched): the `active_region` ASSIGNMENT gates still carry `and not app.copilot_focused` — same
  smell, separate concern; left for a focused decoupling pass.
