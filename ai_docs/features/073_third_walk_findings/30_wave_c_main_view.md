# 073 W-C — Main view: channel view and Reset placement (#4 #8 + D2 D5)

Parent: `01_spec.md § W-C`. Host-only; landed first.

## What landed

- **`ChannelView`** (`ui_regions.py`): `COLOR` / `COLOR_ALPHA` / `ALPHA`, with
  `next_channel_view` (the cycle, wrapping) and `CHANNEL_VIEW_LABELS` (the chip's words).
  Persisted as `UIAppState.channel_view`, default `COLOR`, so a user who never touches it sees
  exactly the frame they saw before.
- **`CommandId.CYCLE_CHANNEL_VIEW`**, `Alt+V`, category View; the handler is
  `App.cycle_channel_view`.
- **The draw** (`ui.py::_draw_document_image`): the backdrop is the quiet checker in Color and
  the loud one (`COLOR.CHECKER_LIGHT_LOUD` / `CHECKER_DARK_LOUD`, two mid greys) in
  Color+Alpha; in Alpha the shown texture is `app.alpha_view.render(output)`, a one-quad blit
  writing `vec4(a, a, a, 1)` into its own canvas (`alpha_view.py::AlphaView`), so the output
  texture that feedback reads and exports sample is never touched. The chip over the preview's
  top-left names the state and cycles on click; the FPS chip keeps the top-right.
- **Reset** left the preview and sits after `Render all` in the documents grid
  (`widgets/document_grid.py`), same handler, same `F6`, same tooltip.

## Pinned by tests

`tests/test_channel_view.py`: the cycle visits every state and wraps; every state has a
label within the control budget; the default is Color and the choice survives a save/load;
the command reaches the handler; the Alpha blit of a known RGBA texture is white where alpha
is 1 and black where it is 0, and leaves the source bytes as they were.

## Manual verification (the maintainer, in the app)

1. Launch: the viewer looks as it did (quiet checker, no chip state change).
2. `Alt+V` three times: Color -> Color+Alpha -> Alpha -> Color; the chip's word follows.
3. On the cascades example in Color+Alpha the transparent regions show the loud checker; in
   Alpha they are black and the lit regions white; the strip's tiles and an export are unchanged.
4. Reset is after `Render all`; `F6` still resets; nothing sits over the preview's top-left
   but the chip.
