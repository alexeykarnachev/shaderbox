# 034 — UI/UX polish wave 2 (findings log)

The second app-wide UI/UX polish pass, driven by a live maintainer walkthrough — same
maintainer-experience-driven, fix-as-we-go shape as `028_ui_polish_wave.md`. The maintainer drives
the app on the desktop and reports issues one at a time; simple ones are fixed immediately, larger
ones get filed as a `todo.md` deferral or their own spec.

> **STATUS: IN PROGRESS — fix-as-we-go.** Each finding is logged below as it's handled, with its
> resolution. Fixes are gated by `make check` + `make smoke` (when UI/lifecycle code is touched) +
> a maintainer live pass. The maintainer says when a finding should be filed-not-fixed.

---

## Goal
- **Capture + fix** the UI/UX rough edges the maintainer hits driving the app live — the simple ones
  inline, the larger ones batched/filed.

## Out of scope
- Larger features surfaced during the walkthrough get their own spec / `todo.md` deferral, not a
  blind fix.

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

### F01 — delete gate asks with the node id + a noisy trash tail   [fixed — needs make run]
- **Where:** the copilot delete-node confirm card (`tools/shader.py` gate_prompt).
- **Observed (voice, tg msg 1860):** the card says `Delete node '2bd7'? It moves to the project
  trash (recoverable).` — should ask with the node NAME, and shorter: just delete yes/no, no
  trash/recover explanation.
- **Resolution:** `gate_prompt` resolves id -> name via `caps.node_tree()` (read_shader's prefix
  rule; GL-free, worker-safe) -> `` Delete node `Blank`? `` — the backticked name renders as a
  code chip via F02. Fallback: unknown id shows raw. Test:
  `test_tool_registry::test_delete_gate_prompt_shows_node_name`.

### F02 — chat renders `**bold**` as literal asterisks; names/ids could be code chips   [fixed — needs make run]
- **Where:** `ui_primitives.py` (new `markdown_text` + `parse_markdown_lines`),
  `widgets/copilot_chat.py` (assistant bubbles + gate prompts), `app.py` (`font_14_bold`).
- **Observed:** the agent's reply prose shows raw markdown (`**Blank**`); ids sit in plain quotes
  when they could render as code.
- **Resolution:** markdown-lite renderer — `**bold**` (AnonymousPro-Bold, ships in resources),
  backtick spans as `CHIP_BG` chips (one wrap unit), ``` fences as dim code lines; manual
  word-wrap for styled lines, native-wrap fast path for plain text; unmatched/torn markers stay
  literal (stream-safe; bold requires non-whitespace edges per CommonMark, so `2 ** 3` stays
  literal). Applied to assistant bubbles + pending_action prompts; user bubbles stay raw. Tests:
  `test_markdown_lite.py`; draw path checked headless 4 frames (/imgui-ui §0).

### F03 — "You chose: Yes" blends into the gate prompt text   [fixed — needs make run]
- **Where:** `copilot/state.py` (`Message.gate_outcome`), `copilot/session.py` (gate resolution),
  `copilot/persistence.py` (v9 -> v10), `widgets/copilot_chat.py` (render).
- **Observed:** the user's choice was appended into the card's text and reads as one undifferentiated
  blob; wanted a visually distinct (success-colored) answer.
- **Resolution:** structured `gate_outcome` token ("Yes"/"No"/"cancelled"/masked credential echo)
  replaces the text append; rendered under the prompt — `You chose: Yes` in `STATE_OK`, No/cancelled
  dim, credential echo dim. Persistence v10 with defaulted field — pre-v10 cards (outcome baked in
  text) render as before. Tests: `test_conversation_persistence::test_gate_outcome_round_trip_and_pre_v10_default`.

### F04 — replace_lines: the model keeps guessing the block-end line number   [fixed — needs make run]
- **Where:** `copilot/tools/shader.py` (`_ReplaceLinesArgs` + handler), `copilot/backend.py`
  (`apply_line_edit`, `_range_check_error`), `copilot/capabilities.py` (protocol),
  `copilot/prompt.py` (EDITING policy).
- **Observed (trace `12-20-49`, + the maintainer's "19 of 20 compile errors" voice stat):** the
  recurring orphan-tail compile-error class — `replace_lines` under-covers the block because the
  model derives `end_line` from a stale prior instead of transcribing it from the working set.
  Root cause: models address code reliably by CONTENT, unreliably by COORDINATES; the end-line is
  a low-salience bare `}` with zero redundancy, re-derived (stale) after every rewrite.
- **Resolution (tool reshape, no new tool):** (a) WHOLE-FILE mode — omit the range and `new_text`
  replaces the entire file (the dominant rewrite case carries no coordinates at all; sentinel
  start==end==0 through the capability); (b) ranged replaces now REQUIRE `first_line`/`last_line` —
  verbatim quotes of the boundary lines, checked stripped against the real content BEFORE applying;
  a mismatch rejects with the actual lines named (one-step correction, nothing applied). Prompt
  steers: "rewriting main() or most of a file -> OMIT the range". Tests:
  `test_line_editing.py` (whole-file, checksum pass/mismatch/missing, partial range).

### F05 — snippet hover: collapse identical consecutive tool calls   [fixed — needs make run]
- **Where:** `widgets/copilot_chat.py` (the step-grouping now lives in `_collapsed_steps`;
  renamed from `_snippet_tooltip` when F09 redesigned the popup).
- **Observed:** hovering the turn-snippet squares lists every step — 10 identical "Edited shader"
  rows is noise.
- **Resolution:** consecutive steps with the same (tool, outcome) collapse to one `4-13. Edited
  shader` row; an outcome flip or different tool breaks the run. Test:
  `test_copilot_loop::test_snippet_tooltip_collapses_identical_consecutive_steps`.

### F06 — stopped turn's snippet re-animates when the next turn runs   [fixed — needs make run]
- **Where:** `widgets/copilot_chat.py` (`_draw_transcript`/`_draw_turn_snippet`).
- **Observed (screenshot):** stop a generation, edit the message, resend — TWO "Editing shader"
  in-progress snippets pulse at once. A cancelled turn's snippet keeps `snippet_stats=None`
  forever, and `live` was derived from the GLOBAL `in_flight`, so any later turn re-animates it.
- **Resolution:** only the TRAILING turn_snippet may render live; earlier stats-less snippets
  stay bar-only.

### F07 — comment-only edits were structurally impossible (edit_shader)   [fixed — needs make run]
- **Where:** `copilot/backend.py` (`_comment_only_spans` + `apply_shader_edit`).
- **Observed (trace `14-24-54`, two edit_giveups):** "remove the 3-lined header comments" — every
  attempt failed with "old_str not found" even when copying the hint's exact bytes. Root cause:
  the token matcher lexes comments as TRIVIA, so a comment-only old_str produces zero tokens and
  can never match; the error message misled instead of explaining.
- **Resolution:** an old_str that lexes to zero tokens falls back to whitespace-normalized TEXT
  matching (the `_ws_normalize` machinery the near-miss hint already uses), all occurrences, so
  comment edits + `replace_all` banner-stripping just work. Tests: `test_edit_hints.py` (5 cases).

### F08 — render-blind aesthetic spree: 16 clean edits / $0.51 in one turn   [fixed — needs make run]
- **Where:** `copilot/agent.py` (`_CLEAN_STREAK_NUDGE`), `copilot/config.py`
  (`max_clean_edit_streak`).
- **Observed (trace `14-24-54`, "the bricks are still too flat"):** 16 consecutive clean
  `edit_shader` lighting tweaks in one turn — the model can't see the render, nothing brakes
  individually-clean edits (the caps count failures only), the user watches $0.51 burn.
- **Resolution:** a one-time per-turn nudge after `max_clean_edit_streak` (6, config-tunable)
  cumulative clean source edits: "the user has not seen these — stop, summarize, let them look."
  Symmetric to `compile_thrash_nudge`; broken-compile turns stay exempt (fixing comes first).
  Test: `test_copilot_loop::test_clean_edit_streak_nudges_once`.

### F09 — snippet stats tooltip redesigned (colors + token in/out)   [fixed — needs make run]
- **Where:** `widgets/copilot_chat.py` (`_draw_snippet_tooltip`/`_collapsed_steps`),
  `ui_primitives.py`.
- **Observed:** the hover popup was a plain text dump; total cost only, no token split.
- **Resolution:** custom-drawn tooltip — per-step rows carry the square bar's color language
  (ok/fail swatch + dim range + human verb, failures in `STATE_ERROR`), then a stats block:
  `context N tok in` / `reply N tok out` / cost in `STATE_WARN`. Headless-drawn 3 frames.

### F10 — Alt+S dead while the chat input is focused   [fixed — needs make run]
- **Where:** `commands.py::route_flag` (+ chord param), `hotkeys.py`.
- **Observed:** Ctrl+N works from the chat input; Alt+S does nothing.
- **Resolution:** imgui routes only Ctrl-chords through an active text input (it owns all
  keyboard keys). A GLOBAL Alt-chord can never type a character, so it now routes
  `route_always`. Test: `test_command_routing.py`.

### F11 — Cyrillic renders as `?` in the chat feed   [fixed — needs make run]
- **Where:** `copilot/sanitize.py`, `copilot/prompt.py`.
- **Observed:** Cyrillic shows in the input field but mangles to `?` in messages (the D2 ASCII
  sanitizer was the lossy step — the todo deferral's trigger fired).
- **Resolution:** the Cyrillic block (U+0400-U+04FF) passes through `sanitize_display` (the font
  carries it); the prompt's "plain ASCII replies" rule became "reply in the USER'S language,
  ASCII punctuation". Todo entry deleted. Tests: `test_sanitize_display.py`.

### F12 — copilot agent limits exposed in Settings with help hints   [fixed — needs make run]
- **Where:** `copilot/config.py` (unfrozen + `apply_user_limits`), `exporters/integrations.py`
  (`CopilotIntegration` limit fields + `apply_limits`), `popups/settings.py` (`_COPILOT_LIMITS`
  rows), `ui_primitives.py` (`help_marker`), `project_session.py` (startup apply),
  `copilot/agent.py` (0=off guards for both nudges).
- **Observed/wanted:** the queued banner item — token caps + retry budgets + nudge thresholds
  were frozen constants; expose them to the user with explanations.
- **Resolution:** 7 limits (context/reply caps, max steps, failed-edit giveup, both nudge
  thresholds, auto-restore) persist on `CopilotIntegration` (defaults sourced from
  `CopilotConfig` — single source of truth), edited in Settings -> Copilot as labeled int rows
  with `(?)` hover hints, applied live onto the shared `COPILOT_CONFIG` (floors guard a
  hand-edited json; 0 = off for nudges/restore). Tests: `test_copilot_user_limits.py`.

### F13 — click can't re-focus the chat input after an Esc defocus   [fixed — needs make run]
- **Where:** `app.py::_install_escape_filter` (the real fix); two earlier wrong fixes reverted
  (an orphan-activation latch in `copilot_chat.py`; an Esc->editor handoff in `hotkeys.py` that
  broke the designed Esc ladder and was rolled back same-day).
- **Observed:** chat input focused -> Esc (chat defocused, by design) -> click the input: the
  caret flashes ~2 frames and dies; every subsequent click too.
- **Root cause (found via imgui's own ActiveId/io debug log + reading imgui 1.92.8 source):**
  the glfw Esc filter swallowed ALL jobless Esc events — including the RELEASE that follows a
  forwarded press (the press's job, defocusing the chat, is gone by release time). imgui's
  Escape stayed logically held forever; any InputText activated afterwards self-cancelled on
  the key-repeat ticks (`is_cancel` -> silent `clear_active_id`), killing the caret ~2-3 frames
  after activation. Headless repros all passed because synthetic key events bypassed the filter.
- **Resolution:** the filter gates only PRESS/REPEAT; a RELEASE always passes (releasing an
  already-up key is a no-op). Esc semantics unchanged.

---

## Review history
<!-- Design disagreements resolved by the main agent (per dev_flow.md feature flow step 4/6) land here. -->
