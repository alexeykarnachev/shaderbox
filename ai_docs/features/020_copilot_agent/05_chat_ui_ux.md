# 020 Copilot Agent — chat UI/UX (placement, rendering, interaction model)

> ## ⚠️ PLACEMENT REVERSED — the chat is a FLOATING WINDOW, not a 4th tab.
> This report recommended a 4th `Copilot` tab in the right-panel tab bar. The maintainer reversed
> that (post-review): the chat is a **floating top-level `imgui.begin` window** (corner/strip presets
> + free mode, movable, launched by an in-editor top-right button + `Ctrl+J`), NOT a tab. See
> `10_skeleton_plan.md §6` for the landed shape. The rest of this report (message rendering, wrapped
> text, streaming, tool-call transparency, the not-configured gate, the safety UX) stays valid — only
> the **placement** section (§0/§1's "4th tab") is superseded.
>
> One of the swarm's idea reports (see `00_grounding.md`). Angle: **the chat panel's UI/UX** —
> where it lives in the existing LEFT-editor / RIGHT-panel split, how a scrollable transcript
> renders in immediate-mode imgui, how streaming + tool-calls + errors draw, the input box, the
> not-configured gate, nav integration, and the agent-mutates-the-live-app safety UX.
> All claims grounded in the real source (`ui.py`, `ui_primitives.py`, `theme.py`, `commands.py`,
> `app.py`, `tabs/share.py`, `widgets/cheatsheet.py`) + the `imgui-ui` skill. NOT a spec.

---

## 0. Recommendation (read this first)

**Placement: a 4th tab in the right-panel tab bar — `Node / Render / Share / Copilot`.** Not a
floating window, not a new dock, not the editor split. Rationale below (§1) and the adversarial
counter in §8.1. The decisive facts:

- The right-panel tab bar (`_NODE_TABS` in `ui.py:461`, drawn inside `_draw_node_settings`'s
  `begin_tab_bar("node_settings_tabs")`) is the app's established home for "a mode of working with
  the current node." Adding a 4th entry is a 3-line change + a draw fn — the smallest possible diff
  that respects every existing convention (button tiers, active-region outline, the
  `FOCUS_TAB_*`/Ctrl+digit jump machinery already generalises to a 4th).
- **The user must SEE the shader editor (LEFT) + the live preview (the big image at the top of the
  RIGHT app panel, `_draw_app_panel` `ui.py:400`) WHILE chatting** — the agent edits the GLSL and the
  preview updates via the existing hot-reload path (`_reload_if_changed`, grounding §3). A tab in the
  bottom-half settings panel keeps BOTH visible: editor on the left, preview filling the top-right,
  chat in the settings tab below the preview. A modal or a full-height floating window would occlude
  one of them. **A modal that hides the preview is categorically wrong for an agent that edits live.**
- The 4th tab inherits the active-region focus model (§7) and the `Ctrl+3`-style tab jumps for free.

**MVP UI cut (ship this first):**

| Ship in MVP | Defer to gold |
|---|---|
| 4th tab `Copilot`, scrollable transcript child + fixed input row at the bottom | floating/pop-out window |
| user rows + assistant rows, distinguished by **color + a dim role caption** (no bubbles) | message bubbles / rounded fills |
| streaming text appended into the live assistant row (drained from the queue per frame) | token-fade-in animation |
| **plain status lines** for tool-calls ("editing shader…", "created node 'plasma'") via a recolored `caption_text` | rich tool-call *chips* with icons + jump-to-node affordance |
| a spinner/"thinking" line + a **Stop** `ghost_button` | per-message timestamps, token counts |
| code blocks rendered as **a mono-ish dim child with a Copy button** (NOT syntax-highlit) | a read-only `TextEditor` with GLSL highlighting per code block |
| `unconnected_gate`-style "set your Anthropic key" state (§6) | inline key entry in the tab (vs. punting to Settings) |
| `CommandId.TOGGLE_COPILOT` + a cheatsheet row + auto-scroll-to-bottom | @-referencing other nodes; per-action undo UI |
| multi-line input, **Ctrl+Enter sends** (Enter = newline) | GLSL-aware input autocomplete |

**Biggest imgui risk (the one to design around up front):** there is **no selectable/copyable rich
text and no wrapping `imgui.text`** in this build. `imgui.text`/`text_colored` *clip* long lines, they
don't wrap (`imgui-ui` §5) — an LLM answer is long prose, so **every assistant line MUST go through a
`wrapped_caption`-style `push_text_wrap_pos(0.0)` wrapper or it vanishes mid-word.** And "let me copy
that GLSL the agent wrote" has no native affordance — plain `imgui.text` can't be selected. The
proven escape hatch already in the repo is the `imgui_color_text_edit.TextEditor` (read-only) used by
the editor, OR `draw_copyable_text` for short snippets. This drives the code-block design (§2.4). See
§8.3 for the full risk ranking.

---

## 1. Placement — options compared

The fixed structure (from `ui.py`): one full-screen main window → a LEFT `code_editor` child
(`begin_child("code_editor", …)`, `ui.py:250`) + a `_SPLITTER_W` splitter + a RIGHT `app_panel`
child. Inside `app_panel`: the big preview image on top (`_draw_app_panel`), then a `control_panel`
child holding the node-preview grid on the left and `node_settings` (the `Node/Render/Share` tab bar)
on the right.

```
 main window (full screen, menu bar)
+----------------------------------------------------------------------+
| File  Edit  Library                                                  |  <- menu bar
+------------------------+---------------------------------------------+
|                        |  [ big live preview image ]      [FPS chip] |  <- _draw_app_panel top
|  code_editor (LEFT)    |                                             |
|  GLSL TextEditor       +----------------+----------------------------+
|  + error strip         | node grid      |  node_settings tab bar     |
|                        | (preview cells)| [Node][Render][Share]      |  <- _NODE_TABS
|                        |                |  ...tab body...             |
+------------------------+----------------+----------------------------+
        ^ splitter (drag to resize the split fraction)
```

### (a) 4th tab — RECOMMENDED

```
| node grid      |  node_settings tab bar                         |
| (preview cells)| [Node][Render][Share][Copilot]                 |
|                |  +------------------------------------------+   |
|                |  | transcript (scrollable child)            |   |
|                |  |  you ▸ make the bg pulse red             |   |
|                |  |  copilot ▸ I'll edit the fragment shader |   |
|                |  |    editing shader...                     |   |
|                |  |    set u_pulse = 0.5                     |   |
|                |  |  copilot ▸ done — the background now ... |   |
|                |  +------------------------------------------+   |
|                |  [ multi-line input .................. ][Send] |
+----------------+------------------------------------------------+
```

- **Pros:** zero new layout primitives; reuses `begin_tab_item`; preview + editor both stay visible;
  inherits the active-region outline (`active_region_outline()`) + `FOCUS_TAB_*` jumps; the agent's
  "did it work?" feedback (the preview re-rendering, the editor's error strip lighting up) is
  *right there* on screen while the chat sits below.
- **Cons:** the settings panel is the narrower half of the right panel (`control_panel_width / 2.6`
  is the GRID; settings get the rest). The transcript column is ~300-400px wide — tight for code
  blocks. Mitigation: code blocks scroll horizontally (their own child with
  `WindowFlags_.horizontal_scrollbar`) or the user widens the split via the existing splitter, or
  pops the panel out later (gold-plate). Also: the tab body height is bounded by
  `PANEL_CTRL_MINH` (600) — fine for a transcript with internal scroll.

### (b) Floating own-window overlay (like the cheatsheet)

```
 main window ............................
 ...........................  +---------------------+
 .. editor .. preview .....   | Copilot          [x]|  <- own imgui window,
 ...........................  | transcript          |     draggable, resizable
 ...........................  | ...                 |
 ...........................  | [input ........][>] |
 ...........................  +---------------------+
```

- **Pros:** resizable/movable independent of the split; can be parked over dead space; doesn't steal
  the settings tab; could be pinned while the user works in Node/Render/Share.
- **Cons:** the cheatsheet is drawn on the **foreground draw list** (`get_foreground_draw_list()`,
  `cheatsheet.py:56`) precisely because it's *non-interactive* — it has no input, no scroll, no
  focus. A chat is the opposite: it needs a focusable scroll region + a text input + a Stop button.
  That means a real top-level `imgui.begin(...)` window, which collides with the app's
  single-full-screen-window model (`_MAIN_WINDOW_FLAGS` + `no_nav_focus` exists specifically to keep
  Ctrl+Tab from cycling a *second* window — `ui.py:44-48`). A 2nd interactive top-level window
  re-opens that Ctrl+Tab-window-switcher problem and competes with the active-region nav model
  (feature 019). It floats OVER the preview or the editor — occluding exactly what the user needs to
  watch. **Rejected for MVP** (revisit as a pop-out in gold-plate, §8.1).

### (c) New collapsible bottom/right dock

```
+------------------------+---------------------------+
|  editor                |  preview + grid + tabs    |
+------------------------+---------------------------+
|  Copilot dock (full-width, collapsible)            |
+----------------------------------------------------+
```

- **Pros:** full window width = roomy transcript + code blocks.
- **Cons:** the app has no docking concept today (`docking_*` theme slots exist but are noted
  "unused today, harmless", `theme.py:496`). Introducing a 3rd horizontal band steals vertical space
  from BOTH the editor and the preview every frame it's open, and needs a 2nd splitter +
  collapse-state persistence + a 4th nav region. Largest diff of all the options for the least
  convention reuse. **Rejected.**

### (d) Replace/share the editor's LEFT split

- **Cons:** the editor is the thing the agent EDITS; the user wants to watch the diff land in the
  editor while chatting. Hiding the editor to chat defeats the feature. **Rejected outright.**

**Verdict: (a) 4th tab.** It's the only option that is (1) a small diff, (2) keeps editor + preview
both visible, (3) reuses the active-region + tab-jump machinery, (4) introduces no new top-level
window or dock concept. The floating window is the right *eventual* pop-out, but not the MVP.

---

## 2. Message rendering in immediate-mode imgui

The transcript is drawn fresh every frame from an in-memory list of message objects (another agent
owns the model + the worker→main queue; this section owns how a message **draws**). imgui has no
retained rich-text widget, so the transcript is **a vertical stack of plain rows inside one
scrollable `begin_child`**, re-emitted each frame.

### 2.1 The transcript container

```python
# inside the Copilot tab body
avail = imgui.get_content_region_avail()
input_h = ...  # reserved for the input row + Stop button, computed once
with imgui_ctx.begin_child("copilot_transcript",
                           size=imgui.ImVec2(0, avail.y - input_h),
                           window_flags=imgui.WindowFlags_.none):
    for msg in app.copilot.messages:
        _draw_message(app, msg)   # one row block per message
    _maybe_autoscroll()           # §3
```

The transcript child scrolls vertically (default). The input row sits **below** it, fixed height —
same "fixed slot so nothing below shifts" discipline as `status_slot` (`ui_primitives.py:260`) and
the no-jitter rule (`imgui-ui` §2: "a conditionally-appearing line shifts everything below it").

### 2.2 Roles distinguished by COLOR + a dim caption, NOT bubbles

imgui has no rounded-bubble primitive and hand-rolling one (draw-list rounded rects behind measured,
wrapped text) is a jitter/SetCursorPos minefield (`imgui-ui` §4). Ship the **flat, calm** form:

```
you ▸        make the background pulse red          <- FG_DIM caption "you ▸", FG_PRIMARY body
copilot ▸    I'll edit the fragment shader to add   <- ACCENT_PRIMARY caption "copilot ▸",
             a time-driven red pulse.                   FG_PRIMARY body, WRAPPED
```

- Role caption: a `caption_text("you", color=COLOR.FG_DIM)` / `caption_text("copilot",
  color=COLOR.ACCENT_PRIMARY)` on its own line (or inline via `same_line`), then the body.
- Body: **`wrapped_caption(text, color=COLOR.FG_PRIMARY)`** — this is the load-bearing call. Plain
  `imgui.text` clips (`imgui-ui` §5); the wrapper pushes `push_text_wrap_pos(0.0)` so prose folds at
  the transcript's right edge. **Every assistant/user body line goes through it.** A
  `transcript_body(text, color)` primitive added to `ui_primitives.py` is the right home (one wrap
  push, not re-pushed at each call site — same reasoning as `wrapped_caption` itself).
- Separation between messages: a `imgui.dummy((0, SPACE.MD))` gap, never a `separator` line (the
  "cascade of horizontal lines reads as noise" rule, `imgui-ui` §2). Whitespace separates zones.

### 2.3 Mapping the agent's event stream to rows

The reference agents emit a stream (cc-server `AgentLoop.run` yields `StatusEvent`/`TextEvent`;
ovelia handler returns `(ok, message_for_llm, payload_for_client)` — grounding §4). The UI doesn't
consume those directly; the worker translates them into appended **message-model entries** the main
thread drains and draws. The row TYPES this UI must render:

| Model entry | Source event | How it draws |
|---|---|---|
| user message | the input box | role caption + wrapped body |
| assistant text | `TextEvent` deltas | role caption + wrapped body, **grows in place** while streaming (§3) |
| tool-status | `StatusEvent` ("editing shader…", "rendering…") | a dim `caption_text` line, optionally a spinner glyph (§3) |
| tool-result | the `(ok, msg, payload)` triple | a status line recolored OK/ERROR + the payload's "created node X" affordance (§4) |
| error | a failed turn / worker exception | a `STATE_ERROR` wrapped line (§2.5) |
| code block | fenced ```glsl in assistant text | the code-block sub-renderer (§2.4) |

The key UI principle: **the model is a flat ordered list of typed entries; `_draw_message` switches
on the entry type.** Streaming just means the last assistant entry's `.text` keeps growing between
frames.

### 2.4 Code blocks (GLSL) — the copyability problem

The agent will quote GLSL. Two sub-cases:

- **Short inline snippet / a uniform value** (`u_pulse = 0.5`): a `draw_copyable_text(...)`
  (`ui_primitives.py:667`) — already click-to-copy, already themed, already in the repo. Use as-is.
- **A multi-line GLSL block:** plain `imgui.text` in a loop CANNOT be selected/copied (the imgui
  risk, §0). Options, MVP→gold:
  - **MVP:** a recessed child (`begin_child` with `child_bg = COLOR.BG_SURFACE`, like the error
    strip's `_draw_error_strip` `code.py:58`, + `WindowFlags_.horizontal_scrollbar` so long lines
    don't clip), each line a `imgui.text` in the smaller `font_12` (`small_caption` precedent,
    `ui_primitives.py:414`), **plus a single `[Copy]` `ghost_button` in the block's top-right** that
    copies the whole block via pyperclip (the `draw_copyable_text` clipboard path). No per-char
    selection, but the whole block is copyable — covers 95% of "paste this into my shader."
  - **Gold:** a read-only `imgui_color_text_edit.TextEditor` per block with GLSL highlighting —
    proven viable (the editor uses it) but heavyweight (a TextEditor per block, focus-grab quirks per
    `imgui-ui` §8: `render()` auto-grabs keyboard focus on first frame — would steal focus from the
    input every time a block first appears). **Defer** — the focus-grab is a real hazard.

### 2.5 Errors

Two error surfaces, both `STATE_ERROR`:

- **Agent/turn errors** (network failed, key invalid, max-iterations hit): a wrapped
  `STATE_ERROR`-colored body line in the transcript, mirroring `share.py`'s
  `app.notifications.push(..., COLOR.STATE_ERROR[:3])` + the in-panel `text_colored(STATE_ERROR, …)`
  fallback (`share.py:50-52`). Keep the LLM-facing detail OUT of the user line — show a friendly
  "the request failed (see logs)"; the raw exception goes to loguru (the `notifications.push` path
  already logs, `notifications.py:30`). This also matches the references' "generic error strings,
  never leak internals" rule (grounding §4.5).
- **Shader-compile errors the agent CAUSED:** these already surface in the editor's error strip
  (`_draw_error_strip`, `code.py:55`) via `compile_unit.errors` (grounding §3 — the agent's "did it
  work?" signal). The chat doesn't need to re-render them; instead the tool-result line can say
  "edited shader — 2 compile errors (see editor)" so the user's eye goes to the existing strip. This
  is the cleanest division: **the chat narrates, the editor shows the actual error.**

---

## 3. Streaming UX

The worker pushes text deltas to a queue; the frame loop drains it per frame and appends to the last
assistant entry's `.text` (another agent owns the queue; this owns the draw). Because the whole
transcript re-draws every frame anyway (immediate mode), a growing string "just works" — no special
incremental-render machinery. The UI concerns:

- **Auto-scroll:** while a turn is streaming AND the user hasn't manually scrolled up, keep the view
  pinned to the bottom. Pattern: after the message loop, if `imgui.get_scroll_y() >=
  imgui.get_scroll_max_y() - epsilon` (user was at/near bottom) **or** a "streaming just appended"
  flag is set, call `imgui.set_scroll_here_y(1.0)` (or `set_scroll_y(get_scroll_max_y())`). If the
  user scrolled up to read history, DON'T yank them down — detect "not at bottom" and suppress the
  auto-scroll until they return to bottom (a "stick to bottom" latch). This is the one piece of
  genuinely fiddly scroll logic; keep it in one helper.
- **Thinking/spinner state:** before the first token arrives (and between tool rounds), draw a
  "copilot is thinking…" dim caption. A spinner: imgui-bundle has no guaranteed spinner widget and an
  emoji spinner is risky (monochrome-emoji caveat, `imgui-ui` §8). **Safest:** an ASCII
  rotating-char (`|/-\\`) or a `...`-dots animation keyed off `imgui.get_time()`, drawn as a
  `caption_text`. No font dependency, no jitter (it's one fixed-width line in its own slot). This is
  also the "one slot, not a stack" discipline — the spinner line is replaced in place by the first
  assistant text, never stacked above it.
- **Stop button:** a `ghost_button("Stop")` (low-emphasis, §1 tier table) shown in/near the input
  row ONLY while a turn is in flight (mutually exclusive with the Send button: `if in_flight: Stop
  else: Send` — the "one slot, not a stack" rule, `imgui-ui` §2). Clicking sets a cancel flag the
  worker reads (an `AbortSignal`-style flag, like marginalia's `AbortSignal/timeout`, grounding §4).
  The reference agents already emit an `_executed_actions_note` on cutoff (cc-server, grounding §4) —
  the UI should still show whatever partial text + tool-results landed; a stopped turn isn't an
  error, so it gets a dim "stopped" caption, not `STATE_ERROR`.
- **Input disabled while in-flight?** No — keep it editable (let the user queue the next message) but
  gate *Send* behind "not in flight" so two turns can't run at once. (Open question §9 — or allow a
  follow-up that the worker queues.)

---

## 4. Tool-call transparency

The user must see what the agent DID — the references converge on this (grounding §4: cc-server
status templates; ovelia renders a card from the `payload_for_client`; marginalia per-tool status).
In ShaderBox the *real* feedback is the live preview + the editor, so the chat's job is a **concise
narration that ties the agent's actions to on-screen objects.**

### 4.1 MVP: inline status lines

Each tool call appends a dim status line into the transcript, in order, between assistant text:

```
copilot ▸ I'll make the background pulse red.
          editing shader (plasma.frag.glsl)
          set u_pulse = 0.5
          rendering preview
copilot ▸ Done — the background now pulses red at ~1 Hz.
```

- Drawn as `caption_text(line, color=COLOR.FG_DIM)`, indented under the assistant turn.
- Recolor on result: OK → a brief `COLOR.STATE_OK` tick text; error → `COLOR.STATE_ERROR`. Mirrors
  `share.py::_surface_terminal_progress` choosing `STATE_ERROR`/`STATE_OK` by `is_error`
  (`share.py:39`).
- The status STRING comes from the tool (cc-server's per-tool status templates / ovelia's
  `message_for_llm` — but the user-facing one should be the friendly variant; keep the LLM-facing
  string separate, grounding §4). **Name the affected entity** (the node name, the uniform name) —
  the references all do this ("Done: deleted X") so a later turn can answer "what did you do?" and so
  the user can see the object referenced.

### 4.2 Gold: tool-call chips + jump-to-object

Promote the status lines to **chips** (the repo already has `chip_button` / `pill_button`,
`ui_primitives.py:114/93`) when the action references a jumpable object:

```
[ ✎ edited plasma.frag.glsl ]   <- chip, click -> open that file in the editor (show_node_editor / open_shader_lib_file)
[ + created node "ripple" ]     <- chip, click -> select_node(id) (app.py:418) + highlight its grid cell
[ ▸ u_speed = 1.2 ]             <- chip, click -> FOCUS_TAB_NODE + scroll to that uniform's row
```

- These are the `payload_for_client` (ovelia's triple, grounding §4) rendered as an affordance: the
  payload carries the node_id / file path / uniform name; the chip's onclick calls the matching App
  verb. **This is the ShaderBox-specific win the grounding asks for** — "jump-to-the-node-it-created"
  is literally `app.select_node(node_id)` which already exists and already re-latches grid focus.
- Glyph caveat: `✎`/`▸`/`+` — `+` and `▸`/`>` are ASCII-safe; fancy glyphs risk the monochrome-emoji
  blank + the RUF001/002 ambiguous-unicode ruff trip (`imgui-ui` §9). **Use ASCII markers** (`>`,
  `+`, `~`) or draw a tiny icon with the draw list (the `close_cross_button` precedent draws its ✕
  from two `add_line`s, `ui_primitives.py:439`). Don't rely on glyph fonts for the chip icons.
- A turn-summary footer ("the agent made 3 changes") is optional gold — the inline lines already
  convey it; a summary only helps if a turn does many things. Defer.

### 4.3 The (ok, msg, payload) contract this implies

If the handler-return triple from ovelia is adopted (grounding §4), the UI consumes the `payload` to
decide chip vs. plain line: payload present + has a node_id/path → jumpable chip; payload absent →
plain status line; `ok=False` → `STATE_ERROR` recolor. This keeps the UI dumb (it renders whatever
the payload describes) and the tool layer the source of truth for "what happened."

---

## 5. Input box

```
+--------------------------------------------------+
| make the bg pulse red                            |   <- input_text_multiline, ~3 rows
|                                                  |
+--------------------------------------------------+
                                        [ Send ]       <- primary_button, content-width, right
```

- **Multi-line:** `imgui.input_text_multiline` (the repo's `labeled_multiline_input`,
  `ui_primitives.py:306`, wraps it). GLSL-aware highlighting in the *input* is NOT worth it (the
  input is mostly natural language; the agent writes the GLSL). Skip.
- **Send key — Ctrl+Enter, NOT Enter.** This is the critical collision call. The app uses Enter
  heavily: feature 019 nav, `enter_returns_true` inline inputs in modals (`imgui-ui` §7.5), and the
  rebinder/grid. A multi-line chat input where Enter sends would make the user unable to type a
  newline AND fight the nav model. **Ctrl+Enter sends; Enter inserts a newline.** This also matches
  the dev's own register (a coding tool). Detect Ctrl+Enter via `io.key_ctrl &&
  is_key_pressed(Key.enter)` while the input is focused (the pattern `code.py:183` uses for
  Ctrl+scroll), and suppress the global Enter-nav while the input owns focus (the "outer keyboard
  suppression" rule, `imgui-ui` §7.5 — track `is_item_focused()` right after the input).
- **Send button:** `primary_button("Send")` (the ONE CTA of the panel, §1 tier table), content-width,
  bottom-right. Disabled (`begin_disabled`) when the input is empty OR a turn is in flight; replaced
  by the `Stop` ghost_button while in flight (§3, one-slot rule).
- **Implicit context = the current node.** The agent's context snapshot (grounding §4.6: current node
  + its uniforms + compile errors + available lib functions) is built from `app.current_node_id` —
  the user doesn't have to attach it. Show this implicitly: a dim caption above the input, e.g.
  `caption_text("context: node 'plasma'")`, so the user knows what the agent sees.
- **@-reference another node (gold):** typing `@` could pop a node picker (the node names from
  `app.ui_nodes`) to inject another node into context. Defer — the implicit current-node context
  covers the common case; cross-node ops are rarer. When built, reuse the grid's node list, not a new
  picker.

---

## 6. The not-configured gate

Before an Anthropic key exists, the tab shows a setup state — **mirror `unconnected_gate`**
(`ui_primitives.py:338`), the exact primitive the exporters use when credentials aren't set:

```python
def _draw_copilot_unconfigured(app: App) -> None:
    unconnected_gate(
        not_connected_msg="Copilot is not set up.",
        hint="Add your Anthropic API key in Settings to enable the in-app coding assistant.",
        action_label="Open Settings",
        on_action=app.open_settings,
    )
```

- This is byte-for-byte the share-tab pattern: a `STATE_WARN` line + a dim hint + a `primary_button`
  that opens Settings (`share.py`/`base.py` exporters call it; the telegram/youtube setup uses
  `setup_steps` + `connection_status` too, `ui_primitives.py:367/383`). The copilot's setup block in
  Settings can reuse `setup_steps` (numbered steps + a copyable link to console.anthropic.com) and
  `connection_status` ("Key set." / "Not set.") for full consistency.
- **Where the key lives** is an open question (grounding §4: `integrations.json` cleartext alongside
  the Telegram token, vs. an env var). The UI is agnostic — it gates on a `copilot.is_configured`
  bool the same way exporters gate on `is_available`/connected. Whatever the storage, the *gate UI*
  is `unconnected_gate`.
- **MVP shortcut:** punt configuration to Settings (the button above). **Gold:** an inline key entry
  in the tab itself (`labeled_text_input(..., password=True)`, `ui_primitives.py:296` — already
  supports password mode). Settings-punt is fewer moving parts for MVP and matches every other
  integration.

---

## 7. Nav / command integration

The chat is the **4th active region** (§1). Concretely:

- **A toggle command:** add `CommandId.TOGGLE_COPILOT` to `commands.py` (the `CommandId` StrEnum,
  `commands.py:17`) with a default chord (e.g. `Ctrl+J` — `Ctrl+1/2/3` are taken by
  `FOCUS_TAB_NODE/RENDER/SHARE`; the natural extension is `FOCUS_TAB_COPILOT` = `Ctrl+4`). Wire the
  callback on `App` (the `command_callbacks` dict, `app.py:314`) to `focus_node_tab(NodeTab.COPILOT)`
  — which already sets the tab + `_set_region(ActiveRegion.PANEL)` (`app.py:390`). So **`Ctrl+4`
  jumps to the Copilot tab AND focuses the panel region** for free, exactly like the other tab jumps.
- **`NodeTab.COPILOT`** added to the `NodeTab` StrEnum (`commands.py:42`) + a `_NODE_TABS` row
  (`ui.py:461`): `("Copilot", NodeTab.COPILOT, copilot_tab.draw)`. The `begin_tab_item` loop +
  `set_selected` jump machinery (`ui.py:501-518`) generalises with no change.
- **Cheatsheet row:** automatic. The cheatsheet iterates `COMMAND_SPECS` (`cheatsheet.py:30`) and
  renders any command with a bound chord active in the current scope. Adding the `CommandSpec` for
  `FOCUS_TAB_COPILOT`/`TOGGLE_COPILOT` makes it appear with zero cheatsheet code.
- **Active-region focus model:** the chat lives INSIDE the PANEL region (the `node_settings` child).
  It does NOT need to be a 4th `ActiveRegion` in `_REGION_CYCLE` (`app.py:73`) — it's a tab within the
  existing PANEL region, so `Ctrl+Tab` cycling EDITOR→GRID→PANEL still works and the Copilot tab is
  just one PANEL view. **This is a feature, not a limitation:** the user `Ctrl+Tab`s to the panel,
  then the panel's tab bar (or `Ctrl+4`) picks Copilot. Adding a 4th nav region would mean reworking
  `_REGION_CYCLE` + the GRID/PANEL focus latches — unnecessary scope.
- **Where keyboard focus goes:** when the Copilot tab activates, focus should land on the **input
  box** so the user can type immediately. Use the one-shot-focus pattern (`needs_focus` flag →
  `set_keyboard_focus_here(0)` on the input's first draw, consumed after one frame — the exact rule
  from `imgui-ui` §7.5; don't call it every frame or it fights the transcript's scroll/selection).
  **Caveat (`imgui-ui` §8):** the `set_window_focus(name)` string overload SEGFAULTS — use
  `set_next_window_focus()` before the child + `set_keyboard_focus_here` for the input; never pass a
  name string. Also note the transcript child must NOT auto-grab focus (no TextEditor in MVP code
  blocks → no auto-focus-grab problem; that's a reason to defer the highlit-code-block gold feature,
  §2.4).
- **Esc behavior:** Esc should defocus the input / return to the default state, consistent with
  `_handle_escape` (`hotkeys.py:45`) which closes popups + drops editor focus. The chat isn't a popup
  (it's a tab), so Esc just defocuses the input (drop `set_keyboard_focus_here`); it must NOT cancel
  an in-flight turn (that's the explicit Stop button — Esc-cancels-a-turn is too easy to hit by
  accident). Gate the input's Esc the way modals gate theirs (`imgui-ui` §7.5).

---

## 8. Safety UX (the agent mutates the live app)

This is where the copilot diverges hardest from the exporters: **exporters never destructively mutate
user data** (grounding §4 — "Tools mutate a live visual app… not a database"). The copilot deletes
nodes, overwrites shader source, changes uniform values. The safety affordances, by blast radius:

### 8.1 Non-destructive mutations (edit shader, set uniform) — rely on the existing undo + revert-on-disk

- **Shader edits are already reversible.** The agent edits by writing the `.glsl` file (grounding §3
  — the "free lunch" hot-reload seam). The editor's `TextEditor` has its own undo history, and
  `_reload_if_changed` re-syncs it (`ui.py:88-119`). So "undo what the agent wrote" = the editor's
  normal Ctrl+Z once the reload lands. **BUT the grounding flags a real edge:** a disk-write while the
  user has *unsaved* edits would clobber their session (the reload only re-syncs if texts diverge,
  `ui.py:111`). **Safety UX:** before the agent overwrites a shader the user has unsaved edits in,
  the chat should warn ("you have unsaved changes in plasma.frag.glsl — the agent will overwrite
  them") and require a confirm, OR auto-save first. Surface this as a transcript warning line +
  inline `[Overwrite] [Cancel]` — don't silently clobber.
- **Uniform sets** are cheap to reverse (it's a dict write, grounding §3 gap (a)). MVP: no per-set
  undo UI; the value is visible in the Node tab and the user can drag it back. Gold: a per-action
  "undo this" affordance on the tool-result chip.

### 8.2 Destructive mutations (delete node) — confirm before, NOT undo after

- **A deleted node is the one genuinely destructive op.** The app already has the right primitive:
  `cell_delete_confirm` (`ui_primitives.py:449`) — the in-cell `Delete? [Yes][No]` wash used by the
  node grid, and the `danger_button` tier whose "prominence comes from a confirm step, NOT a filled
  red" (`imgui-ui` §1). **The agent should not delete a node without a confirm.** Design: a
  delete-node tool returns a *pending* action the chat renders as a confirm prompt in the transcript:

```
copilot ▸ I'll delete the unused node "test2".
          [ Delete "test2"? ]  [Yes]  [No]      <- danger affordance, agent waits
```

  `[Yes]` calls `app.delete_node(node_id)` (`app.py`, clean id-taking verb, grounding §3); `[No]`
  tells the agent it was declined (feeds back as a tool result). This mirrors the references' mutation
  discipline (ovelia: "mutation tools that render a card return EMPTY text; never claim a past-tense
  action unless the tool returned this turn", grounding §4) — the agent doesn't *say* "I deleted it"
  until the user confirms and the tool fires.
- **Why confirm-before, not undo-after:** there is no node-undelete in the app today, and building one
  is out of scope. A confirm gate is the boring, correct, minimal safety primitive — and it reuses
  `danger_button` + the confirm-step convention already in the repo.

### 8.3 The "agent did N things while I wasn't looking" problem

Because the agent runs off-thread and mutates over several tool rounds, the user might miss a change.
The transcript's inline status lines (§4) ARE the audit log — they persist in the scroll history. The
gold-plate "revert everything this turn did" is appealing but expensive (needs a per-turn mutation
journal + inverse ops); **defer it.** MVP safety = (1) confirm-before-delete, (2) warn-before-clobber
-unsaved-edits, (3) the persistent transcript log. That's proportional to the real risk.

---

## 9. Adversarial section

### 9.1 Floating window vs docked-in-panel — argue both, pick

**Strongest case for a SEPARATE floating window:** A coding assistant is a *persistent companion*,
not a per-node mode — the user wants it open while they work in Node OR Render OR Share, resizable
big enough for real code blocks, parked where they like. The tab forces it to share the cramped
settings half (`/2.6` grid + the rest) and makes it mutually exclusive with the other three tabs (you
can't watch the Render tab AND chat). A floating window is also the natural home for a wide transcript
+ multi-line GLSL blocks that the ~350px tab column squeezes.

**Strongest case for DOCKED-in-panel (the recommendation):** The app is deliberately a
**single-full-screen-window** design — `_MAIN_WINDOW_FLAGS` + `no_nav_focus` exist *specifically* to
kill imgui's Ctrl+Tab window-switcher so the app's own active-region cycle (feature 019) owns Ctrl+Tab
(`ui.py:44-48`). A 2nd interactive top-level window re-introduces exactly that conflict and competes
with the nav model the maintainer just built. The cheatsheet "floating window" precedent is NOT a
counter-example — it's foreground-draw-list, non-interactive, no focus, no input
(`cheatsheet.py:56`); a chat needs all three. And the floating window would occlude the preview or
editor — the very things the agent's edits must be watched in.

**Pick: docked tab for MVP; a pop-out toggle as gold.** The decisive asymmetry: the docked tab is a
~10-line addition that reuses everything and breaks nothing; the floating window is a new top-level
interactive window that fights a just-shipped nav model. Ship the cheap correct thing; revisit
pop-out once the chat content model is proven and IF the column proves too cramped in real use.

### 9.2 Strongest case for the SIMPLEST UI shipping first

**The argument FOR plain-input + plain-text-log, no chips/cards/streaming flourish:** Everything that
makes a chat *look* like a product — bubbles, chips, jump-to-node affordances, syntax-highlit code
blocks, token-fade streaming — is precisely the imgui-hostile stuff (no rich text, no bubbles, the
SetCursorPos/jitter traps, the TextEditor focus-grab, the monochrome-emoji caveat). Each is a round of
blind-iteration risk (`imgui-ui` §0 — can't screenshot, every visual call goes to the maintainer). The
*value* of the feature is the agent editing the shader and the preview updating — which happens on
the LEFT and TOP regardless of how pretty the chat is. A dead-simple transcript (`wrapped_caption`
rows) + a multi-line input + a Send button + the unconnected gate delivers 100% of the function and
0% of the imgui risk. Streaming is nearly free (immediate mode redraws the growing string anyway).
Tool transparency in MVP is just dim status lines — no chips needed to *understand* what happened.

**Conclusion:** the MVP cut in §0 IS this argument operationalized. The only "flourish" kept in MVP is
the stuff that's cheap AND functionally necessary: wrapped text (mandatory — else text vanishes),
color/caption role distinction (one line of code), streaming (free), status lines (a recolored
caption), the Stop button (a flag + a ghost_button), and confirm-before-delete (reuses
`cell_delete_confirm`/`danger_button`). Everything genuinely ornamental — bubbles, chips,
jump-to-node, highlit code blocks, pop-out, per-action undo, @-refs — is deferred. **Ship the boring
log; earn the chips later.**

### 9.3 The imgui-bundle limitation that most threatens the chat UI

Ranked:

1. **No wrapping `imgui.text` (clips, doesn't fold) + no selectable/copyable rich text**
   (`imgui-ui` §5). This is THE threat: an LLM answer is long prose and multi-line GLSL. Mitigation is
   known and cheap — `wrapped_caption`/`push_text_wrap_pos(0.0)` for prose (mandatory, used
   everywhere), `draw_copyable_text` + a `[Copy]` button for code (copy-whole-block, no per-char
   selection). **Design around it from line one** — it's not a blocker, but forgetting it makes text
   silently vanish mid-word.
2. **The editor-FPE-behind-modals deferral** (`code.py:135` — the `TextEditor.render()` FPEs while a
   popup is open, so it's simply not drawn then). Relevant IF a code block uses a `TextEditor` (gold,
   §2.4) AND a modal is ever open over the chat. Combined with the §8 `TextEditor.render()`
   first-frame focus-grab (`imgui-ui` §8), this is the reason to **keep MVP code blocks as plain
   `imgui.text` + Copy**, not a TextEditor. Avoid the whole class for MVP.
3. **Monochrome-emoji + ambiguous-unicode** (`imgui-ui` §8, §9): the tempting `✎ ▸ ⏵ 🔧` chip icons
   render blank (color-emoji blank in this build) or trip ruff RUF001/002. **Use ASCII markers** (`>`
   `+` `~`) or draw-list glyphs (the `close_cross_button` two-line-✕ precedent). Cheap to obey,
   annoying to discover late.
4. **The SetCursorPos assert + jitter** (`imgui-ui` §3, §4): a hazard ONLY if the chat uses absolute
   positioning (bubbles, overlay badges). The flat-row MVP uses pure normal flow (`same_line`,
   `dummy`) → immune. Another vote for the simple form.

---

## 10. Open questions (for the spec)

1. **Where does the Anthropic key live?** `integrations.json` cleartext (alongside the Telegram
   token, with the existing cleartext-secret deferral) vs. an env var. UI is agnostic (gates on a
   bool), but the spec must decide + the Settings setup block depends on it.
2. **Handler return shape:** is ovelia's `(ok, message_for_llm, payload_for_client)` triple adopted?
   The chip-vs-line + jump-to-object UI (§4.2/4.3) depends on the `payload` carrying node_id/path.
   If only cc-server's bare string is adopted, MVP status lines still work but gold chips lose their
   structured source.
3. **Does an in-flight turn block new input, or queue a follow-up?** §3 leans "input stays editable,
   Send gated" — confirm the desired model.
4. **Confirm-before-delete granularity:** per-delete confirm (§8.2) vs. a session "let the agent
   delete freely" toggle for power users. MVP = always confirm; revisit.
5. **Transcript persistence across restarts?** The exporters don't persist op history. Does the chat
   transcript survive an app restart (a saved conversation log under `app_data_dir()`), or is it
   session-only? Affects whether a `UIAppState` field is added (with the migration discipline,
   grounding §3) or it's pure in-memory.
6. **Tab-column width:** is ~350px enough in real use, or does this force the pop-out (§9.1) sooner
   than "gold"? A maintainer `make run` eyeball once a stub transcript exists answers it — can't
   screenshot, so this is a hand-check (`imgui-ui` §0).
7. **Does the Copilot tab need to render every frame even when not visible** (so streaming continues
   updating the model while the user is on the Render tab)? The model update is independent of the
   draw (the queue drains in the frame loop regardless), so the transcript is correct when re-shown —
   but confirm the auto-scroll latch survives a tab switch.
