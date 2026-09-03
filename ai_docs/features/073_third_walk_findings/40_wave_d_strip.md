# 073 W-D — Pass strip: live fill, next / previous pass (#5 #7 + D3 D4)

Parent: `01_spec.md § W-D`. Host-only; landed second.

## What landed

- **The dormant tint is gone.** `COLOR.STALE_TINT` is deleted; `preview_cell` blits every
  image white-tinted. A dormant tile keeps its dim footer, dim chips and the corner tick.
- **A live tile carries a fill.** `_draw_pass_tile` passes
  `bg_color=fade(COLOR.ACCENT_PRIMARY, COLOR.ACCENT_TINT_ALPHA)` (`0.32`, the active tab's own tint, shared by one token after the maintainer's first look at `0.12`) for every pass in
  the output's chain, which `preview_cell` sets as the cell's `child_bg`: the ground under the
  picture, the footer and the chips. Computed at draw time because `apply_theme` swaps the
  accent at runtime. The output's accent border and error red are as they were.
- **`strip_order`** moved from the widget to `pass_graph.py` (GL-free, reachable from `App`
  without a widget import), with `step_in_order` beside it (wrap at both ends, first tile when
  the current is unknown, None on empty).
- **One funnel for picking a pass.** `App.pick_pass(document_id, name, focus_editor)` is what
  a tile click was (shader tab to the front, then `set_output_pass`); the tile click and the
  add-pass commit both call it with `focus_editor=False`. The widget's `open_pass` callback
  parameter is gone.
- **`CommandId.NEXT_PASS` / `PREV_PASS`**, `Alt+Right` / `Alt+Left`, category Document:
  `App.step_output_pass(±1)` walks `strip_order` from the output and calls `pick_pass` with
  `focus_editor=self.editor_focused`, the frame's focus state, so the editor keeps focus when
  it had it and is left alone when it did not. The chords reach the registry while the editor
  is focused because the vim keymap leaves every Alt chord unbound.

## Pinned by tests

`tests/test_pass_navigation.py`: wrap at both ends; unknown current lands on the first tile;
through the real command callbacks on a two-pass document, the tab summon receives the focus
flag the frame had, and the output follows. `tests/test_pass_verbs.py` follows the moved
`strip_order` and the funnel.

## Manual verification (the maintainer, in the app)

1. On the cascades example: dormant tiles show their picture untouched plus the tick; live
   tiles carry the fill; the output keeps its accent border. The fill is the active tab's tint.
2. Click into the editor, `Alt+Right`: the output moves one tile right (wrapping at the end),
   the tab follows, the caret is still yours. `Alt+Left` walks back.
3. Click the strip (editor unfocused), `Alt+Right`: the output moves and the editor is NOT
   focused.
4. `add pass` still opens the new pass and its gear.
