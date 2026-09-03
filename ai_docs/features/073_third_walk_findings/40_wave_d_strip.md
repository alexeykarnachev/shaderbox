# 073 W-D — Pass strip: live fill, next / previous pass (#5 #7 + D3 D4)

Parent: `01_spec.md § W-D`. Host-only; landed second.

## What landed

- **The dormant tint is gone.** `COLOR.STALE_TINT` is deleted; `preview_cell` blits every
  image white-tinted. A dormant tile keeps its dim footer, dim chips and the corner tick.
- **No fill either.** An accent fill at `0.12`, then at the active tab's `0.32`, were both
  rejected at the display, and an HTML picker over the palette found nothing better. Landed
  shape: a live tile's name is bold (`app.font_14_bold`) in `FG_TITLE`, a dormant tile's name
  is `FG_DORMANT` (`bg_4`) in the regular face; `preview_cell` gained `footer_font` /
  `footer_color` for it. The output's accent border and error red are as they were.
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

1. On the cascades example: every tile shows its picture untouched; a live tile's name is
   bold and bright, a dormant one's darker with the tick; the output keeps its accent border.
2. Click into the editor, `Alt+Right`: the output moves one tile right (wrapping at the end),
   the tab follows, the caret is still yours. `Alt+Left` walks back.
3. Click the strip (editor unfocused), `Alt+Right`: the output moves and the editor is NOT
   focused.
4. `add pass` still opens the new pass and its gear.
