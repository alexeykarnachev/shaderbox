# 071 W-B — Code panel: cursor follow, Ctrl+Tab (#3 #9 + D8)

Parent: `01_spec.md § W-B`. Size: small (three package files, one new test module), so no
pre-impl reviewer; the open question the spec left (was the Ctrl+Tab gate bypassed?) was
settled by a repro before any edit, and one post-impl reviewer judges the diff.

## The repro that settled #9b

Driving the real app headless (`update_and_draw` frames with imgui input injected through
`io.add_mouse_*` / `io.add_key_event`): two tabs open, a click on the app panel right of the
editor (`editor_focused` reads False), then Ctrl+Tab. The active tab moved from 1 to 0, and a
stack trace on `set_active_tab` showed `App.cycle_code_tab` as the caller. So the command
itself dispatched with the editor unfocused. Cause: `commands.py` declared the spec with
`C.EDITOR`, which is `CommandCategory.EDITOR`, the cheatsheet group; `CommandSpec.scope`
defaults to `GLOBAL`, and only `CLOSE_CODE_TAB` sets `scope=CommandScope.EDITOR` explicitly.
The ledger's "by the code the chord cannot fire" (#9) read the category as the scope. Under D8
the chord is global on purpose, so the spec now says `scope=CommandScope.GLOBAL` in so many
words, and the behaviour change is in the handler.

## What landed

- **Cursor follow after the layout** (#3). `tabs/code.py::layout_following_cursor(editor, size,
  px_per_em, rows, last_cursor)`, imgui-free: lay out, then if the cursor moved and sits outside
  `[first, first + rows)`, `scroll_to_line` against this frame's layout and lay out again. The
  draw path calls it once per frame and records the cursor only when `rows > 0`, as before.
  The second layout runs only on the frames a follow fires.
- **Tabs in display order** (#9a). The tab item id is the tab's PATH (`##{path}`), not the list
  index: with an index id every moved tab became a new tab to imgui, appended at the end. Inside
  the bar, after the item loop, `_display_order` reads imgui's own order back
  (`imgui.internal.get_current_tab_bar`, `tab_bar_find_tab_by_order`, `tab_bar_get_tab_name`;
  verified: the name carries the `##` suffix) and `_apply_display_order` permutes
  `app.editor_tabs`, keeping the active tab's identity, so cycle, close and every index-based
  verb address what the eye sees. Skipped on a frame that closes a tab or on which imgui's list
  and the model disagree (a tab's first frame).
- **Ctrl+Tab focus** (D8). `App.cycle_code_tab`: not focused → `editor_focus_requested = True`
  and return (the same latch the glyph-open and document-select paths use); focused → cycle.
  "Focused" is the code-panel child focus, which already routes keys and drives the
  unfocused dimming.

## Tests (`tests/test_code_panel.py`)

- The follow, against the vendored editor with the library-drawn status row: ten lines, five
  text rows, `G` then `o`; the caret's row is inside the text rows. Falsifier: move the follow
  back in front of the layout (the row reads 5 of 5). And an idle caret is left alone after a
  wheel scroll.
- The permutation helper on a bare tab list: `[2, 0, 1]` reorders and keeps the active tab;
  identity is a no-op.
- The frame-loop test from the repro: click the app panel, Ctrl+Tab leaves the tab and focuses
  the editor, a second Ctrl+Tab cycles; and the order read-back reported `[0, 1]` once two tabs
  existed (proof the names resolve to model indices through imgui's state).

## Manual verification (the maintainer, in the app)

1. `G` then `o` on a file longer than the panel: the new line sits above the status bar with
   the caret visible. Falsifier: the caret is under the status row until you type.
2. Drag the second tab to the front; Ctrl+Tab (editor focused) cycles in the new order.
3. Click a slider in the Document tab, press Ctrl+Tab once: the editor takes focus (dimming
   lifts) and the tab stays. Press again: it cycles.
