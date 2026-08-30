# 064 — The authoring surface: implementation spec

**Status: PLAN-LOCKED** (maintainer: "let's build already something, we will iterate and re-design
later if we need... write generalizable, robust, flexible code with clear separation of concerns").
Anchor: `00_scenario.md` R1-R10. Engine: `03_engine_spec.md` (landed). Judged proposals:
`design_round/`.

The engine ships and works; steps are invisible in the app. This is the surface that shows them.

## What ships

A **Steps section** in the Node panel listing each step with a live thumbnail, its resolved size and
format, and what it reads — plus a **view pin** that retargets the big preview to any step so an
intermediate can be looked at. Read-only over the chain's structure: the shader stays the only place
a step is declared or configured.

## The shape, and why

**S1. A hybrid of proposals B and D, taking each one's strength.** B ranked first on requirement
fidelity and grain; D ranked first on R9 (viewing intermediates), which is the primary debugging
tool for this effect class. B's rows carry the facts, D's view-pin carries the looking. Both authors
independently said their surface could sit over a text seam unchanged, which is exactly what the
landed engine is.

**S2. The surface is READ-ONLY over structure. The shader is the single source of truth.** No chips
that write size/format/filter, no drag-to-reorder, no add/delete-step buttons. The engine derives
steps and their order from the source by introspection, so a control that edits them would need a
write-back path into GLSL text — a second author of the same fact, and the desync class the whole
rider design exists to avoid. Editing a step means editing the shader, which is one click away.

Revisit trigger: the maintainer tunes a step's `scale`/format by hand often enough to want a widget,
AND a write-back into the declaration line is designed rather than assumed.

**S3. One state field, one primitive, one widget module.** The whole feature is `viewed_step` on
`UINodeState` plus `widgets/step_list.py::draw_step_list(app)`. That matches the locked layering
(`widgets/*.py` are free functions taking `app`), so nothing new is invented and the section can be
deleted in one commit if the design changes.

**S4. The view pin is transient UI state, not node state.** It resets on node switch and is not
persisted. Reloading a project into "showing cascade level 4" would be a confusing state to inherit,
and a pinned intermediate that survives a restart reads as a broken shader.

**S5. An HDR view transform lands with the surface, on the preview path.** Measured: the MAIN
preview is already fine (it draws the node's `f1` canvas, which the user's `main()` has tonemapped).
A float STEP target displayed raw is not — a cascade level holding 7.0 renders pure white, and every
step worth debugging is float. R7 and R9 are only jointly satisfiable with it, and no judged proposal
served it. Reinhard plus sRGB, applied only when the viewed texture is float, so an `f1` step and the
final composite are byte-identical to today.

**S6. Everything routes through the existing primitives.** `preview_cell` for the thumbnail (it
already carries selection border, footer, stale mark and a click target), `small_caption` for the
header, `ghost_button` for actions, `caption_text` for the dim facts line, `theme.py` tokens for
every colour and size. No new colour literals, no hand-rolled `push_style_color`.

## Requirement coverage

| # | Served by | Not served (and why) |
|---|---|---|
| R1 | one row per step | — |
| R2 | the row's resolved pixel size, shown | editing it (S2) |
| R3 | the row's `reads:` line, naming its sources | — |
| R4 | `reads:` naming non-adjacent steps IS the branch | no spatial view; B's judge called the 3-way merge "interrogated, not seen" and that stands |
| R5 | a `feedback` mark on a self-reading row | — |
| R6 | a `persist` mark, and the reset action | — |
| R7 | the format shown per row + the HDR transform (S5) | editing it (S2) |
| R8 | filter/wrap shown per row | editing it (S2) |
| **R9** | **the view pin: click a row's thumbnail, the big preview shows that step** | — |
| R10 | already served by the engine's uniform union; the rows make ownership legible by naming which step each uniform belongs to | — |

## Files touched

- `shaderbox/widgets/step_list.py` (new) — `draw_step_list(app)`, the whole section.
- `shaderbox/ui_models.py` — `UINodeState.viewed_step: str` (transient, excluded from the save).
- `shaderbox/tabs/node.py` — one call, sited after `_draw_entry_points`.
- `shaderbox/ui.py` — the preview reads the viewed step's texture; the HDR transform.
- `shaderbox/core.py` — a small read-only accessor for a step's resolved facts, so the widget does
  not reach into `_step_targets`.
- Tests: the accessor, the transform, and the pin's reset-on-switch.

## Verification

1. A step-free node shows no Steps section at all. Falsifier: an empty header appears.
2. A 4-step node lists 4 rows in evaluation order with correct sizes and formats.
3. Clicking a row's thumbnail retargets the big preview; clicking it again returns to the output.
4. A float step target displays tonemapped, not white.
5. The final composite and an `f1` step are unchanged by the transform.
6. Switching nodes clears the pin.
7. The section is disabled mid-copilot-turn, like every other panel control.
