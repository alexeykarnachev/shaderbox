# 064 — The authoring surface: implementation spec

**Status: PLAN-LOCKED** (maintainer: "let's build already something, we will iterate and re-design
later if we need... write generalizable, robust, flexible code with clear separation of concerns").
Anchor: `00_scenario.md` R1-R10. Engine: `03_engine_spec.md` (landed). Judged proposals:
`design_round/`.

The engine ships and works; steps are invisible in the app. This is the surface that shows them.

## What ships

A **Steps section** in the Node panel listing each step with a live thumbnail, its resolved size and
what it reads, plus combos editing that step's target — and a **view pin** that retargets the big
preview to any step so an intermediate can be looked at. The shader stays the only place a step is
DECLARED; the panel is where its target is configured.

## The shape, and why

**S1. A hybrid of proposals B and D, taking each one's strength.** B ranked first on requirement
fidelity and grain; D ranked first on R9 (viewing intermediates), which is the primary debugging
tool for this effect class. B's rows carry the facts, D's view-pin carries the looking. Both authors
independently said their surface could sit over a text seam unchanged, which is exactly what the
landed engine is.

**S2. The panel edits a step's TARGET; the shader owns the chain's STRUCTURE.** Size, format,
filter and edge behaviour are per-step node state (`UINodeState.step_configs`), so the rows carry
real combos and an edit recompiles the node and persists to `node.json`. What the panel does NOT
do is write GLSL: no add/delete-step buttons, no drag-to-reorder, no renaming. Which steps exist
and what each reads is introspected from the source, and a control that edited those would need a
write-back into shader text — a second author of the same fact.

Two kinds of fact, two homes, neither able to contradict the other: a step cannot exist in the
config without existing in the code, because the config is keyed off what the compiler reports, and
a config naming a step the shader dropped is simply unused.

**S3. One state field, one primitive, one widget module.** The whole feature is `viewed_step` on
`App` (transient, per S4) plus `widgets/step_list.py::draw_step_list(app)`. That matches the locked layering
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
| R2 | the row's `size` combo (full / 1/2 / 1/4 / 1/8 / 1/16 / 1/32), and the resolved pixel size beside it | — |
| R3 | the row's `reads:` line, naming its sources | — |
| R4 | `reads:` naming non-adjacent steps IS the branch | no spatial view; B's judge called the 3-way merge "interrogated, not seen" and that stands |
| R5 | a `feedback` mark on a self-reading row | — |
| R6 | the row's `on edit` combo (keep / clear), which is `persist` labelled by its effect | — |
| R7 | the row's `format` combo (f1/f2/f4) + the HDR transform (S5) | — |
| R8 | the row's `filter` and `edge` combos | — |
| **R9** | **the view pin: click a row's thumbnail, the big preview shows that step** | — |
| R10 | served by the engine's uniform union: a uniform declared in any step gets a control and persists | attribution — the panel does NOT name which step owns a uniform. Deliberate: a uniform of the same name in two steps is ONE row driving both (`03_engine_spec.md` D4), so there is no single owner to name |

## Files touched

- `shaderbox/widgets/step_list.py` (new) — `draw_step_list(app)`, the whole section.
- `shaderbox/ui_models.py` — `UINodeState.step_configs: dict[str, StepConfig]` (saved), and
  applying them on load. The view pin is transient and lives on `App`, not here (S4).
- `shaderbox/step_preview.py` (new) — the float-target tonemap (S5).
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
