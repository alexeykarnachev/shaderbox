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
