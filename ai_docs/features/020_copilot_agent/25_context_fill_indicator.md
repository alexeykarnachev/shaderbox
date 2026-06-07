# 020 · 25 — Copilot usage bars + compact layout toggle

The copilot chat header is `[Layout: corner    Clear  Close]`. Two changes, driven by a live
maintainer walkthrough (sibling to the 024 polish wave):

1. **Compact the layout toggle** — replace the `Layout: corner` text button with a small DRAWN icon
   depicting the current layout, freeing header space.
2. **Add two usage bars** in the freed space — the previous turn's real **input** and **output**
   token totals, as two thin stacked progress bars, each against its own budget; one shared hover
   tooltip with the detailed numbers.

> **Size: small→mid.** A few files, two new UI primitives, one per-turn usage snapshot on `ChatState`.
> A small `agent.py` change is needed (capture iteration-0 input tokens — see Decision 2), corrected
> after pre-impl review. Full feature flow scaled down (2 pre-impl reviewers done; 1-2 post-impl).

---

## Goal

- A glanceable readout of how heavy the **previous turn** was: a thin **input** bar (context size
  sent) over a thin **output** bar (reply size produced), each filled against its own real budget.
  Full input bar = "context is large, consider clearing the conversation."
- Real API token counts (`LLMUsage` from the stream) — no estimation.
- A **prototype** tooltip: plain textual detail (exact in/out tokens, budgets, cost). Graphics/visual
  polish deferred — this wave just wires the data + a clean seam so proper visuals drop in later.
- A compact layout toggle: a drawn box-in-frame icon (no icon font / emoji — this imgui-bundle build
  renders only monochrome/blank emoji, `/imgui-ui §5/§8`), cycling the three layouts as today.

## Out of scope

- **Per-block / segmented context breakdown, word counts, "which section costs what".** Explicitly
  dropped in favour of two plain per-turn usage bars (in/out). The earlier segmented-bar idea is
  superseded — revisit only if the maintainer later wants a composition breakdown.
- **Visual polish of the bars + tooltip.** This wave prototypes: two monochrome bars + a text
  tooltip. The encapsulation (Decision 7) is the deliverable that lets proper graphics land later
  without touching the data path.
- **Session-total vs per-turn toggle.** The bars show the LAST turn only. The running session total
  already lives in `ChatState.usage` (shown elsewhere); not duplicated here.

---

## Design decisions (locked)

1. **Two sub-features, sequenced.** Layout icon first (pure UI, no plumbing), then the usage bars
   (one small state field). Independent — either can land alone.

2. **Input bar = iteration-0 input tokens (real context size); output bar = summed output tokens.**
   CORRECTED after pre-impl review (both reviewers caught it). `AgentTurnDone.usage` carries the
   `_UsageRollup` SUM across iterations — and `input_tokens` is summed PER REQUEST, but each iteration
   re-sends the growing context, so the sum is ~N× the real context on a tool turn (it pegs the input
   bar near-full on any multi-iteration turn even with a tiny conversation). The real context size is
   the **first iteration's** `input_tokens` (= system + project_context + history + user + working set,
   before any within-turn tool churn). So:
   - **Input** = the first `LLMDone.usage.input_tokens` of the turn (captured once, in `run_turn`).
   - **Output** = the summed `output_tokens` (the rollup is correct here — total tokens the model
     produced across the turn).
   - **Cost** = the summed `cost_usd` (rollup, correct).
   This needs a small `agent.py` change: capture `first_input_tokens` on the first `LLMDone`, and carry
   it on the terminal event alongside the existing summed usage. (Reviewer option (b).)

3. **Per-turn snapshot stored on `ChatState`, distinct from the session total.** Add
   `last_turn_usage: LLMUsage | None = None` to `ChatState`, where this `LLMUsage` is built in
   `run_turn` with `input_tokens = first_input_tokens` (NOT the rollup sum) and `output_tokens` /
   `cost_usd` = the rollup. Carry it on `AgentTurnDone` (next to the existing `usage`, which stays the
   summed value for the session accumulator). `session._apply_event`'s handler keeps
   `self.state.usage.add(ev.usage)` (session total, summed) AND adds
   `self.state.last_turn_usage = ev.last_turn_usage`. `AgentError`/`AgentCancelled` carry none → bars
   keep the last completed turn's value (a cancelled turn has no meaningful context size).

4. **Each bar against its own real budget.** Input fraction = `first_input_tokens /
   COPILOT_CONFIG.max_input_tokens` (150k — the trim ceiling, the real per-request input budget).
   Output fraction = `output_tokens / COPILOT_CONFIG.max_tokens_per_turn` (8k — the per-reply output
   cap). Both clamped [0, 1]. The CALLER computes these two fractions; the primitive receives only
   fractions (Decision 7).

5. **Two thin monochrome bars, stacked, one color each — collision-proof pair.** Input bar =
   `COLOR.ACCENT_PRIMARY`; output bar = `COLOR.SELECT` (purple — a FIXED hue the theme invariant
   guarantees differs from EVERY accent preset, so the two bars can never merge to one color; the
   earlier `STATE_INFO` choice collides with `ACCENT_PRIMARY` under the "blue" accent). Track =
   `BG_FRAME`, frame = `BORDER`. No segments, no gradients (prototype). No invented palette entries.

6. **One shared hover tooltip, textual (prototype), OWNED BY THE PRIMITIVE.** The caller passes a
   preformatted tooltip string; the primitive owns the rect + `is_item_hovered` + `set_tooltip`, so
   the hover affordance lives in ONE place (Decision 7). Sample text (left-aligned, no space-padded
   columns — `set_tooltip` is not guaranteed monospace):
   ```
   Last turn
   input 8872 / 150000 tok
   output 71 / 8000 tok
   cost $0.0111
   ```
   Plain text now; richer graphics deferred (Out of scope).

7. **Encapsulation seam (the real deliverable for "proper visuals later") — primitive takes SCALARS,
   not feature types.** `ui_primitives` is feature-agnostic (`/imgui-ui §6`), so it must NOT import
   `LLMUsage` or know token budgets. Shape:
   `usage_bars(id_: str, fractions: tuple[float, float], tooltip: str, width: float) -> None`
   — draws both stacked bars (input over output) from the two pre-clamped fractions, owns the
   full-row-height hit rect + the hover tooltip. The caller (`copilot_chat`) computes the fractions
   from `LLMUsage` + the `COPILOT_CONFIG` budgets and formats `tooltip`. ALL geometry, color, bar
   count, AND the hover card live inside the helper → swapping the prototype for richer graphics is a
   one-function change, call site untouched. The empty state is the same call with `fractions=(0, 0)`
   and an empty-state tooltip — NOT a separate caller branch (Decision 9).

8. **Bar geometry (so two stacked bars fit a one-row header).** Add `SIZE.USAGE_BAR_H` (~6) and
   `SIZE.USAGE_BARS_W` (~`CHIP_W`) tokens. The helper claims a hit rect of `(width,
   imgui.get_frame_height())` via an `invisible_button` (a full-row-height item — so no `set_cursor`
   past the last item, no SetCursorPos assert, `/imgui-ui §4`), captures the entry origin, and draws
   the two bars + inter-bar gap (`2*USAGE_BAR_H + SPACE.XS`) **vertically centered** within that
   frame-height rect via the draw list. The header stays exactly ONE `get_frame_height()` tall — the
   bars never make it taller. Jitter-free (fixed sizes, draw-list, `/imgui-ui §3`).

9. **Layout icon glyph (drawn, 3 variants) — primitive takes a variant, not `CopilotLayout`.**
   `layout_icon_button(id_: str, variant: int, side: float) -> bool` (variant 0=corner / 1=strip /
   2=free), keeping the primitive feature-blind (`/imgui-ui §6`); the caller maps `CopilotLayout` →
   variant. Side ≈ `SIZE.BTN_SM_H`. Uses the existing `ghost_button` tier for the rect+click (NOT a
   hand-rolled `push_style_color`), then overlays via draw list an outer `add_rect` frame (`BORDER`,
   = the editor area) + one `add_rect_filled` sub-rect (`FG_SECONDARY`): CORNER = small rect
   bottom-right; STRIP = wide short rect along the bottom; FREE = centered rect. Caller cycles
   `_NEXT_LAYOUT` on click + sets the `Layout: <name>` tooltip. NOT disabled mid-turn.

10. **First-render / empty state.** Before any turn (`last_turn_usage is None`) the caller calls
    `usage_bars` with `fractions=(0, 0)` + an empty-state tooltip ("No turn yet.") — same primitive,
    no separate branch (Decision 7). Keeps the header layout stable so the first turn introduces no
    layout jump. **Clear (mid-session)** resets `ChatState` (`reset_conversation` builds a fresh
    `ChatState()`), so `last_turn_usage` returns to None and the bars empty — expected; verify it.

11. **Minimum width + overlap guard (per Q2).** The bars hold a minimum width (`SIZE.USAGE_BARS_W`,
    ~`CHIP_W`); the caller clamps the freed middle width to that floor. Because `_draw_top_bar`
    right-aligns the Clear/Close cluster via `same_line(avail.x - cluster_w)`, clamping alone does NOT
    prevent overlap in a hand-narrowed FREE window. Guard: set the chat window's min size (in
    `draw`/`_apply_layout`) to `>= icon_w + USAGE_BARS_W + cluster_w + spacings` so `avail.x` can
    never drop below what the row needs. (FREE is the only resizable layout; CORNER/STRIP force their
    size.) Verify no overlap at the min size.

12. **Not persisted.** `last_turn_usage` is transient `ChatState` (like `status`/`streaming_text`),
    refreshed each turn. `persistence.py` / `ConversationStore` / `_VERSION` UNTOUCHED. (The session
    total `ChatState.usage` IS already persisted — unchanged.) After a project reload the bars show
    empty until the first turn completes.

---

## Files touched

- `shaderbox/copilot/agent.py` — capture `first_input_tokens` on the first `LLMDone`; build the
  per-turn `LLMUsage` (input = first, output/cost = rollup) and carry it on `AgentTurnDone`
  (new field beside the existing summed `usage`).
- `shaderbox/copilot/state.py` — `ChatState.last_turn_usage: LLMUsage | None`.
- `shaderbox/copilot/session.py` — set `self.state.last_turn_usage = ev.last_turn_usage` in the
  `AgentTurnDone` handler (beside the existing session `usage.add`).
- `shaderbox/theme.py` — `SIZE.USAGE_BAR_H`, `SIZE.USAGE_BARS_W` tokens.
- `shaderbox/ui_primitives.py` — `layout_icon_button` (variant-based), `usage_bars` (fractions +
  tooltip, scalar-only — no `LLMUsage` import).
- `shaderbox/widgets/copilot_chat.py` — `_draw_top_bar` rewrite (icon + bars); compute fractions +
  format tooltip from `LLMUsage` + `COPILOT_CONFIG`; PRESERVE the F01 focus path and the danger-tier
  `Clear` + its `begin_disabled(in_flight)` gate verbatim. Possibly a window min-size in `draw`/
  `_apply_layout` (Decision 11).
- **NOT touched:** `prompt.py`, `persistence.py`, `config.py`.

---

## Manual verification (maintainer, live — no agent screenshot, `/imgui-ui §0`)

1. Open the copilot → header shows a drawn box-in-frame icon, not `Layout: corner`. Hover → tooltip
   reads the current layout name.
2. Click the icon → cycles corner → bottom_strip → free → corner; the glyph's sub-rect moves
   (bottom-right / wide bottom / centered) and the panel actually relayouts. Confirm NOT disabled
   mid-turn.
3. Before any turn → two empty tracks in the freed space; hover → "No turn yet."
4. **Single-iteration turn** (a plain question, no tools) → input bar fills to ~context/150k, output
   bar to ~reply/8k (distinct colors); hover EITHER bar → one tooltip with exact input/output/cost.
5. **CRITICAL — multi-iteration tool turn on a NEAR-EMPTY conversation** (e.g. "read shader X then
   edit it" as the first message) → the input bar must read SMALL (real context size), NOT near-full.
   This is the Decision-2 correctness check: if the input bar pegs high on a short conversation, the
   summed-vs-first-iteration bug regressed. Cross-check the input number against the FIRST
   `llm_response` usage line in the transcript (not the summed `turn_done` line).
6. **Bar geometry** → the two bars are vertically centered on the same row as the icon + Clear/Close;
   the header is NOT taller than one button row; no top-aligned float.
7. Press Stop / force an error mid-turn → bars keep the last completed turn's values (no blank).
8. **Clear** (mid-session, after some turns) → bars go empty (expected — `reset_conversation`).
9. Resize + switch layouts → bars and the Clear/Close cluster never overlap (verify at the FREE
   window's min size), no jitter; bars stay visible at the min width.
10. Reload the project → bars empty (per-turn usage not persisted); session cost line unaffected;
    Clear/Close still align.
11. `make check` clean (0 pyright errors), `make smoke` green.

---

## Review history

**Pre-impl review (2026-06-07, 2 adversarial reviewers, both anchored to the real code).** Both
independently raised one BLOCKER and several should-fixes; all accepted, spec revised before lock:
- **BLOCKER — summed input tokens ≠ context size.** `AgentTurnDone.usage.input_tokens` is the
  `_UsageRollup` SUM across iterations; each iteration re-sends the growing context, so the sum is
  ~N× the real context and pegs the input bar near-full on any tool turn. FIX: input bar uses the
  FIRST iteration's `input_tokens` (real context size); output/cost stay summed. Requires a small
  `agent.py` change — the "no agent.py change" framing was wrong (Decisions 2/3/4 rewritten,
  Files-touched updated, verification step 5 added to catch a regression).
- **Layering leak** — `usage_bars`/`layout_icon_button` must not import `LLMUsage`/`CopilotLayout`
  (feature types into feature-agnostic `ui_primitives`, `/imgui-ui §6`). FIX: primitives take scalars
  (fractions + tooltip; variant int) — Decisions 7/9.
- **Tooltip seam** — caller-drawn tooltip splits visuals across two files, defeating "swap visuals in
  one place". FIX: primitive owns the hover tooltip (string passed in) — Decisions 6/7.
- **Stacked-bars geometry under-specified** — added `SIZE.USAGE_BAR_H`/`USAGE_BARS_W`, the
  centered-in-frame-height draw, and the "header stays one row tall" invariant (Decision 8).
- **Color collision** — `STATE_INFO` == `ACCENT_PRIMARY` under the blue accent; switched output bar
  to `SELECT` (theme-invariant-protected) (Decision 5).
- **Overlap math** — clamp-to-min isn't an anti-overlap guarantee; added a window min-size guard
  (Decision 11).
- **NIT** — Clear empties the bars mid-session; documented + verified (Decisions 10, step 8).

## Open questions for the user

Resolved at plan-lock (2026-06-07):
1. **What the bars show** → previous turn's real **first-iteration** input tokens + summed output
   tokens, two thin stacked monochrome bars, one shared text tooltip (Decisions 2/5/6).
2. **Budgets** → each bar ÷ its own cap (input ÷ 150k, output ÷ 8k) (Decision 4).
3. **Encapsulation** → all visuals (bars + hover tooltip) inside `usage_bars`, which takes only
   scalars; call site passes fractions + a tooltip string, so richer graphics land later without
   touching the data path (Decisions 6/7).
4. **Narrow header / persistence** → min width + window-min-size overlap guard (Decision 11); not
   persisted (Decision 12).
