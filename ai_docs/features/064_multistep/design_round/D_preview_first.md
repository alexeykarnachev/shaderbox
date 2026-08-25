# Proposal D — "The Contact Sheet" (bias: PREVIEW-FIRST / IMAGE-CENTRIC)

> Produced by a design agent anchored to `00_scenario.md` R1-R10, the reference RC demo, the
> primary-source survey, and the real UI code. Verbatim agent deliverable, saved for the judging round.

## 1. The core idea

A node's steps appear as a **horizontal strip of live thumbnails** under the main preview — one
picture per step, each rendering its own actual output every frame, in execution order left to
right. **Clicking a thumbnail makes the big preview show that step** (the reference demo's "Stage To
Render", `01_reference.md:99`) and simultaneously scopes the panel below: that step's shader, that
step's uniform sliders, that step's target settings. **Structure — who reads whom — is shown by
lighting up the source thumbnails whenever a step is focused, and edited by dragging a source
thumbnail onto a named input slot rendered beneath the focused step's picture**, so a connection is
always made between two pictures visible at the same time.

Working name for the strip: the **contact sheet** — the photographer's sheet of every frame on the
roll, which is exactly what it is.

## 2. Two verified facts this rests on

- **The big preview submits no interactive imgui item.** `imgui.image_with_bg` at `ui.py:504-511`;
  the comment at `ui.py:530-531` says so verbatim. The script mouse feed uses
  `item_normalized_mouse(img_min, img_max)` (`ui_primitives.py:403-428`), which hit-tests an
  EXPLICIT screen rect ANDed with `is_window_hovered(child_windows)`.
- **But interactive widgets CAN be drawn on top of the preview, and already are.** `fps_overlay`
  (`ui_primitives.py:1055-1096`, called at `ui.py:538-544`) does `set_cursor_screen_pos` then a real
  `chip_button`, and unfolds a real `begin_child`.

**Consequence:** the design does not need the image clickable. Every affordance over the preview is a
real imgui widget positioned like `fps_overlay`. Where image-body clicks ARE wanted (the convenience
drop route), one `invisible_button` sized to the image rect suffices — with the fps chip submitted
after it under `set_next_item_allow_overlap`, the pattern `preview_cell` already uses for its
delete-X (`ui_primitives.py:1021,1025`). `item_normalized_mouse` keeps working because it hit-tests a
rect, not "the last item".

## 3. The panel

```
|  +-------------------------------------------------------+   [ 60 FPS ]        |
|  |            BIG PREVIEW - shows the FOCUSED STEP         |                     |
|  |  [< out ]  <- breadcrumb chip, top-left, over the image |                     |
|  |                        [ merge4 ] 128x128 f2 lin/clamp  |  <- step badge      |
|  +-------------------------------------------------------+                      |
|  CONTACT SHEET  (new: widgets/contact_sheet.py)                                  |
|  | [scene][casc0][casc1][casc2][casc3]|[c4][c5][c6][c7]|[bright][b/2][b/4]...  | |
|  |   ^glow   ^dim    ^dim     ^dim     ^FOCUSED (accent border)                | |
+---------------------------------------------------------------------------------+
| | node preview grid | | STEP  merge4      [rename] [x]                        |  |
| | UNCHANGED         | | Reads: [casc5 thumb][scene thumb]  <- INPUT SLOTS     |  |
| | one cell per NODE | |         u_upper      u_scene       (drop targets)     |  |
| |                   | | Target: size [1/4 v] format [f2 v] filter [lin v]     |  |
| |                   | |         wrap [clamp v]  [ ] persistent                |  |
| |                   | | u_interval [=====O====] 0.42   <- THIS step only      |  |
```

Every element maps to an existing primitive: the sheet is a direct sibling of `node_grid.py:36`
(a `begin_child`, a loop, `preview_cell` per entry); the big preview is ONE substitution
(`node.canvas.texture` -> `focused_step_texture`); the breadcrumb and badge are the `fps_overlay`
technique; the uniform rows are `draw_ui_uniform` unchanged, only the SET it iterates changes.

**Nothing in the node grid changes** — one cell per NODE. Two levels, two surfaces: nodes in the
grid, steps in the sheet.

## 4. How structure becomes visible — the crux, in three layers

**Layer A — Focus lighting (passive, always on, zero clicks).** Focusing `merge4` snaps exactly its
two sources (`casc5`, `scene`) to full brightness with an accent border; everything else fades to
45%. **The answer to "what does step 7 read?" is a glance, not a query.** Holding `Alt` inverts the
lighting to show CONSUMERS. Two directions, no extra pixels. Implementable today: `preview_cell`
already takes `border_color`; adding a `tint` param to its `dl.add_image` is a one-argument change
and `fade()` already exists in the theme module.

*Limitation:* lighting shows the SET of sources, not which uniform each binds to. That is Layer B.

**Layer B — Input slots (the editable form of the wiring).** Under the focused step, a row of small
thumbnails, one per sampler uniform the step's shader declares, labelled with the uniform name. An
unbound sampler shows an "empty" placeholder — the visible form of "this shader declares an input
nobody filled". The slots come from **introspection, not declaration**: `get_active_uniforms()`
(`core.py:243`) already returns them and `node.py:67-68` already filters by `GL_SAMPLER_2D`. Same
call, new consumer. **The slot row cannot drift from the shader** — delete the uniform line and the
slot disappears on next compile. Labels reuse `uniform_name_label`, so click-to-jump and the
code<->panel hover bridge work for free.

**Layer C — Drag a picture onto a picture.** Three binding routes: (1) drag from the sheet onto a
slot (`preview_cell`'s existing `selectable` is a legal drag source); (2) **click an empty slot ->
the sheet enters pick mode**, dimming illegal sources; click a lit thumbnail — the keyboard/trackpad
path, primary and not an accessibility afterthought; (3) drop onto the big preview's body to bind the
first empty slot (the one place needing the new invisible_button; a convenience, not the primary
route).

**Self-binding is a first-class drop, not a special mode.** Dropping a step onto its own slot binds
it to itself; the slot renders with a **↻ corner glyph** and the caption `u_prev (last frame)`. This
is the survey's single strongest convergence ("if ShaderBox adopts one idea from this survey, this is
the one") and literally KodeLife's rule. **The user never sees two buffers; they see one picture with
a ↻ on it.**

**Order stays legible** because the sheet is laid out in execution order left-to-right with **thin
vertical dividers between dependency generations** — steps that could run in parallel share a band:

```
[scene] | [casc7]...[casc0] | [bright][b/2][b/4][b/8] | [trail] | [smoke] | [combine]
```

Order is spatial, dependencies are lighting, bindings are slots. Nothing is left to a wire.

**The honest cost, stated once:** with focus lighting you see the sources of ONE step at a time.
There is no single picture of the whole graph.

## 5. Requirement coverage

| # | UI element | User action |
|---|---|---|
| R1 | contact sheet, one `preview_cell` per step | `[+]` cell at the right end appends a step + opens its editor tab; right-click -> duplicate/rename/delete (delete reuses the armed-confirm wash); drag to reorder |
| R2 | `size [1/4 v]` combo (`full / 1/2 / 1/4 / 1/8 / custom`) | pick; default `full` (ratio-of-output, the survey's convergence). The thumbnail's scale visibly changes — **the setting is confirmed by picture**; exact numbers ride in the badge |
| R3 | **input slot row** — one slot per declared sampler, so N inputs is N slots, **not a fixed 4** (unlike `iChannel0..3`) | declare a second sampler -> a second slot appears on next compile; drag sources in |
| R4 | generation dividers + focus lighting; the bloom band sits in its own group | branching is created implicitly by binding to something other than the previous step. **No explicit "make a branch" verb** — branching is a consequence of binding, which is right |
| R5 | self-drop onto own slot -> ↻ `u_prev (last frame)` | self is NOT dimmed in pick mode. Double-buffering never surfaced |
| R6 | `[ ] persistent` checkbox + **clock/frost chips on the thumbnail** + `[reset]` in the cell menu | §7 |
| R7 | `format [f2 v]` (`f1/f2/f4`) | **`f2` is the DEFAULT for a new step, not `f1`** — 8-bit is measured fatal, so the safe value is the default and `f1` is the opt-in. An **`HDR` corner chip** appears when any texel exceeds 1.0, so a clipping value is visible rather than silently white |
| R8 | `filter [lin v]` `wrap [clamp v]` | defaults `linear`/`clamp` — fixes moderngl's `repeat_x/y = True` trap the survey records |
| R9 | **click a sheet thumbnail -> the big preview shows that step.** The design's centre | click, or arrow-key along the sheet (`preview_cell(nav_flatten=True)` already gives a keyboard ring). `Esc`/breadcrumb returns. The reference demo's "Stage To Render", except every stage, always, with no control to find |
| R10 | uniform list filtered to the focused step's program | focus, drag. **This fixes the ergonomics blocker by construction**: with one program per step, `u_interval` is genuinely active in `merge4`'s program and needs no decoy + no-op multiply. Ambiguity is impossible — same name in two steps is two rows under two focuses |

Rough edge, honestly: a uniform wanted across SEVERAL steps (a global `u_exposure`) must be set per
step or driven from the node script. No "node-level uniform" concept is added — scope with no
requirement behind it.

## 6. Scale at 15 steps

`THUMB_SM = 90`, cell ~90x108 with footer. 15 cells ~1425px; app panel ~1150px on a 1920 window with
a 40% editor split. So **two rows of 12+4, ~224px total height** against the ~600px
`PANEL_CTRL_MINH`. Fits and works.

Three user-selectable sizes, because 15 is not the ceiling: **Large** (150px, 3 rows, ~460px — 2-5
step effects), **Medium** (90px, 2 rows, ~224px — the scenario's default above 6 steps), **Compact**
(44px + name on hover, 1 row, ~54px — 20+, navigating not reading). The toggle is
`layout_icon_button` (`ui_primitives.py:784`), which already takes a `variant: int` built for exactly
this.

**Where it genuinely stops working: past ~30 steps** — Compact becomes a navigation bar with no
informational content, the failure of the whole premise. Answer: **collapsible bands** (the
generation dividers already group the sheet; a band collapses to `[casc x8 v]` showing its last
member plus a count). Flagged as "the honest extension point, not a proven one".

**The real scaling cost is not pixels, it's GPU work — and there is none.** Every step already
renders its own target every frame as part of the chain; the thumbnail is `dl.add_image` of a texture
that exists regardless. **No extra render per thumbnail.** Two caveats not papered over: a `1/8` step
is a 64x64 texture stretched into a 90px cell (blurry — the badge tells you it's small, not the
picture); and `preview_cell` draws with a fixed white tint, so an `f2` target above 1.0 needs the
tonemap-for-view + HDR chip, **genuinely new code in the primitive**.

## 7. R6 — and the angle unique to this design

**A thumbnail of a simulation is a readout of accumulated history, and a stale one is actively
misleading.** Three states, three marks (the `fps_overlay` chip technique):

| State | Mark | Meaning |
|---|---|---|
| **Warm** | `clock 4:12` chip | accumulating, and this is how long |
| **Cold** | `clock 0:00` + dimmed picture | just cleared or just loaded. **The picture is real but means nothing yet** — a smoke sim at t=0 is black, indistinguishable from a broken step without the chip |
| **Stale** | **frost chip** + 55% tint | the node isn't ticking (unselected with `is_render_all_nodes` off, `ui.py:200-203`). **The picture is a photograph of the past** |

**The Stale mark is the one that earns its keep, and it is not R6-specific** — it applies to every
thumbnail. Without it the sheet's core promise ("these are live pictures") is false exactly when it
matters most. *"If one thing from this section ships, it is the frost chip."*

- **Save/reload -> start cold, and say so.** Persistent targets are never serialized; `persistent` is
  a boolean in `node.json`, contents live only in VRAM. Resuming would mean reviving the
  never-executed raw-`Texture` branch AND dumping megabytes of transient float state per save. The
  clock chip makes the restart a stated fact rather than a mystery, and it dodges the whole class of
  "saved state captured at a different resolution/format than the shader now wants".
- **Copilot revert -> structure reverts, accumulation resets, and the user is told.**
  `CheckpointStore` snapshots a full serialize of the live node, so it restores declarations,
  bindings, target settings, uniform values — and CANNOT restore accumulated pixels. So revert clears
  every persistent target deterministically; `RevertResult` gains `reset_sims: list[str]` and the
  notice says *"...and reset 1 persistent step (smoke)"*. Deterministic-clear beats leave-it-running
  because **a reverted shader reading a buffer accumulated by the un-reverted shader is a state that
  never legitimately existed** — precisely the silent-corruption failure the script route is being
  abandoned for.
- **Export -> warm N frames, with N visible and its cost shown.** Exports funnel through
  `export_isolation()` (`core.py:546-551`), sited once "so no export caller can bypass it", and the
  script is deliberately re-instantiated fresh. Persistent targets must obey the same discipline or
  the same node exported twice differs by how long you'd been staring at it. `Warm-up: [120] frames
  (+2.0 s at 60 fps)`, **default 0**, shown only when the node has a persistent step — because 120
  frames of a 15-step chain is real wall-clock time and a user who doesn't know will think the export
  hung. **Escape hatch, deliberate:** a `[use live state]` toggle that captures the buffer as it
  stands — what someone who spent ten minutes growing a smoke plume actually wants — explicitly
  labelled non-reproducible. An opt-out from determinism taken knowingly, not a default that quietly
  breaks it.

## 8. Naming

A node contains **steps**; the strip is the **contact sheet**.

| Concept | Word | Why |
|---|---|---|
| one draw inside a node | **step** | the anchor doc's own noun throughout — zero translation, every requirement stays greppable |
| the strip | **contact sheet** | photographic, image-first; says "pictures" the way "graph" says "wires" |
| a step's output | **the step's picture** (UI) / **target** (settings row) | "picture" is what you point at; "target" is where technical nouns belong |
| a sampler reading another step | **input slot** | a slot is a hole you put a picture in — matches the gesture |
| a slot bound to its own step | **↻ last frame** | not "feedback", not "ping-pong": describes what the user GETS, hides the mechanism |
| a target surviving frames | **persistent** | ISF's own word; no invention needed |

Rejected: *pass* (jargon, and `064_multipass` was already retracted for framing the problem as a seam
question); *layer* (collides with Photoshop compositing — steps aren't composited); *stage* (reads as
a phase of a FIXED pipeline rather than an authored unit); *pipeline* (implies a straight line,
contradicting R4).

## 9. What this makes HARD

1. **There is no whole-graph picture.** Focus lighting answers "what does THIS step read" beautifully
   and "what does the whole chain look like" not at all. Understanding an unfamiliar 15-step node
   means clicking through ~15 steps holding the topology in your head. **The design's central
   sacrifice, with no mitigation the agent believes in** — a static overview diagram would be a
   second, non-image surface undercutting the premise.
2. **Diamond dependencies read badly.** If `combine` reads `casc0` AND `bloom3`, and `bloom3` also
   reads `casc0`, focusing `combine` lights two cells and says nothing about the edge between them.
   The user must integrate two views to see one shape.
3. **Drag-to-bind is fragile at Compact size** — a 44px drop target on a wrapped strip has a real
   miss-rate. Mitigated by click-to-pick being primary.
4. **Reorder-by-drag collides with bind-by-drag** — same gesture, same cell. Resolved by target
   (drop in the sheet = reorder, drop on a slot = bind), "but the first drag a user makes will
   sometimes do the wrong thing".
5. **Renaming a step** must rename `steps/<name>.frag.glsl` and re-anchor the open tab. Ordinary
   work, but a place a bug will live.
6. **The preview's mouse feed becomes ambiguous.** When the preview shows `casc4` rather than the
   output, what does feature 042's `ctx.mouse` mean? It keeps meaning the same normalized rect — but
   the user is painting into a picture that isn't the one whose result they see. **No clean fix**
   beyond a subtle tint on the breadcrumb.
7. **Vertical budget.** 224px of sheet plus a 600px minimum control panel squeezes the preview on a
   laptop — so **small screens get the least informative variant, the wrong way round**.
8. **Thumbnails lie when the node isn't rendering** — general truth of the surface, which is why the
   frost chip is the section's must-ship.

## 10. Self-verdict

R1-R10 all **COVERED**, "but coverage is not equal confidence". Least confident, in order:

1. **R4's legibility, not its mechanism.** Branching WORKS; "the branch is visible because the bands
   differ" is the weakest visual claim in the document, and items 1-2 of §9 name the same wound.
   **"If this design is wrong, that's where."**
2. **The drag-gesture collision** — resolved by drop target, not drawn, "and the first drag will
   sometimes surprise".
3. **The vertical budget** — arithmetic done on paper, not in a running window. Compact may end up
   mandatory on a laptop, inverting the intended defaults.
4. **The 30-step ceiling** — collapsible bands are sketched, not validated.
5. **`preview_cell`'s HDR display path** — reusing the primitive unchanged is the design's cheapest
   claim, but `dl.add_image` with a fixed white tint genuinely cannot show an `f2` target above 1.0.
   Real new code inside the primitive, costed as small without proof.

**Flagged as adjacent but outside brief:** `preview_cell` acquiring a `tint` param and an HDR-aware
image path serves the node grid too, so it may deserve its own commit before any of this.

## 11. False trails

- **A wiring diagram / node graph inside the node.** Rejected on the survey's finding (graph tools
  put the chain in a spatial patch with NO authored text form; adopting that means a graph editor
  plus an inter-node document model, "a different product, not a feature") and because a diagram is a
  WIRING picture, not an OUTPUT picture, competing for the same panel space as the images.
- **Thumbnails nested inside the node grid's cells.** `preview_cell` is a `begin_child` whose
  docstring says the child exists precisely so "the overlays' absolute cursor moves can't perturb the
  parent (no jitter / SetCursorPos assert)". **Nesting cells inside cells re-enters exactly the
  layout hazard that comment records** — and would cram 15 steps into a 150px cell.
- **A "Stage To Render" dropdown copying the reference demo.** The literal evidence for the bias, but
  the demo's control is a numeric `0-3` selector requiring you to know which number is which stage.
  **The worst image-centric design: it makes you NAME a picture in order to SEE it.** The sheet
  inverts that.
- **A `#pragma`/comment micro-syntax in the shader source.** Genuinely strong (config adjacent to
  declaration, cannot drift, no manifest). Rejected FOR THE UI LAYER because it makes structure
  invisible in the very surface this brief asks to make it visible. **Explicitly noted as not
  exclusive:** the slot row and target combos could be a VIEW OVER such a syntax, and if the backend
  phase picks that seam this design sits on top unchanged.
- **Making the big preview a fully interactive item.** Rejected as blast radius — it changes
  hover/focus semantics for feature 042's script mouse contract. Takes the minimum instead: one
  `invisible_button` used solely as a drop target, and only for the convenience route the design
  works without.
- **Auto-arranging the sheet by topological sort on every change.** Cells would jump under the cursor
  mid-edit. Order is user-controlled by drag; dividers are computed and only ever redraw dividers,
  never move cells.
- **Persisting simulation buffers to disk on save.** Revives a demonstrably-never-executed code path,
  writes megabytes of transient state per save, and buys a feature nobody asked for over a cold
  restart plus an honest clock chip.
