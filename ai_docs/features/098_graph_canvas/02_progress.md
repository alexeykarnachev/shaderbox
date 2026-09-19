# 098 — what landed

Three commits: `0d3a860` the three layers and their gates, `7a2c7fa` the app's
graph tab drawn by the library, `db68cfb` the imgui canvas retired.

## The shape

```
libgraph_canvas.so  (vendored, resources/graph_canvas/)
  -> graph_canvas/ffi.py       the C ABI. No imgui, no moderngl, no shaderbox model.
  -> graph_canvas/render.py    the two streams -> an FBO. Generic with ffi.py.
  -> graph_canvas/adapter.py   Document -> nodes; events -> pass names.
  -> graph_canvas/panel.py     the imgui seam: pointer in, texture out.
  -> widgets/pass_graph.py     which passes are packed, and what an event means.
```

The first two are the liftable pair and a test walks their imports to keep
them that way.

## Measured

- `widgets/pass_graph.py` 1537 -> 423 lines; `widgets/graph_state.py` 296 ->
  94. (The retirement commit reached 368 and 116; restoring the node menus and
  the Group prompt, which were real verbs, put some back.)
- The real app, driven headless to the graph tab on the six-pass
  radiance-cascade document, draws six node cards with their live rendered
  pictures, labelled ports and wires following the wiring.

## Defects this found, and where each is pinned

Each was reintroduced and the gate watched to fail before being restored.

| What | Why it was invisible | Pinned by |
|---|---|---|
| `_SHAPE_FORMAT` carried a padding token for padding that does not exist (the eight fields tile the 120 bytes exactly: 4+4+4+4+4+2+4+4 = 30, not 26) | Instance 0 always reads correctly, so a one-instance draw looks perfect; at one viewport the rest drew shifted and at another they vanished | `test_the_shape_format_describes_the_struct_exactly`, `test_every_instance_reads_its_own_row`, and a parametrized draw over three viewports |
| A run's base instance written as a `{n}x` pad in the format string joins the STRIDE | Instance 1 lands n bytes further on again; the last reads past the buffer and segfaults | `test_a_run_offset_draws_that_run_and_not_the_head_of_the_stream` |
| The glyph atlas and a node preview shared texture unit 0 | A preview is sampled as glyph coverage by the next text run | `test_the_glyph_atlas_and_a_preview_do_not_share_a_texture_unit` |
| `Gesture.NONE` numbered 1 instead of `1 << 31` | It aliases `DRAG`, so a node that refuses everything accepts drags | `test_the_refusal_mask_is_a_veto_and_not_another_gesture` |
| A producer's output slot is its input count, not 0 | A wire lands on the wrong pin rather than raising | `test_a_wire_lands_on_the_port_that_declares_it` |

Two the layout proof caught at load, before they could draw anything:
`Pin_Fill` has three members (0 is "unset" at the wire, not a member), and a
struct that gains a field is refused rather than read at stale offsets.

## Upstream

Two library defects found and fixed by its author this wave: a shape run's
`first` needed documenting as an instance index with no base-instance
parameter in most APIs (`aa5d597`), and `FFI_Result.flags` bit 0 missed a
node-body hover, so a host gating on it starts a node drag and a background
marquee on the same press (`157e258`). Both vendored.

A host has to read the Odin source wherever the README states semantics but
not VALUES — bit positions, enum bases, byte offsets, and where a pin's grab
area actually sits (it differs from where `hover_attribute` reports, which is
why the gesture tests probe for a point that produces the event rather than
computing one).

## Aiming at a pin

The grab area and the hover-reporting area are different rectangles, so for a
while the tests found a pin by pressing candidate points until one produced
the event. `gc_pin_point` was added upstream on the strength of that finding
and the probes are gone -- `Canvas.pin_point(node, attribute, output=)`.

Its readers are the gesture tests, and nothing in the widget calls it: the
canvas draws its own pins and the library hit-tests them, so the host never
needs the point. It was asked for to make a gesture AIMABLE, which is a
test's problem and a keyboard path's, and it stays for those.

One correction to what this feature reported upstream: `pin_fill`'s values
WERE documented. The binding read them correctly; the bug was deriving the
enum-count check from `len(PinFill)`, which counts the wire encoding's four
names against an enum of three. `gc_enum_count` caught it at load on the first
run. The generalisation "the README states semantics and not values" holds for
the other findings and not for that one.

## The gate that proved connectivity and not identity

`pin_point` shipped with its attribute index untested: every call in the suite
passed 0, so hardcoding the index to 0 inside it left 30 tests green. Found by
the library's author hitting the same hole in his own gate and saying so.

Widening the fixture is necessary and not sufficient — with three pins and the
gesture aimed at the third, an off-by-one still connects, because a wire only
has to start somewhere on the right node. Two checks now decide it: walking
every pin and requiring distinct points in pushed order (which an ignored
index collapses to one and a shift collapses to n-1), and an end-to-end drop
onto the THIRD sampler of a three-input consumer that requires `u_z` back by
name. Both were broken and watched to fail.

The single-input fixture could see neither: aim, resolve and report all agree
on index 0 whatever the code does with it.

## What a frame costs

Measured, and gated because it rots silently: **one textured run per PREVIEW,
not per distinct image.** A run is cut where the bound texture changes and a
node's preview sits among its own geometry, so two nodes sharing a texture are
never adjacent and their runs never merge -- six distinct names and six copies
of one both give 15 runs and 6 textured. Budgeting per distinct image
under-counts by the sharing factor.

Scaling is linear: **n textured runs for n passes**, and `2n+3` runs in total
(5/1 at one, 15/6 at six, 43/20 at twenty, 83/40 at forty).

The total carries a precondition the textured count does not: `2n+3` assumes
an ATLAS is loaded. Without one nothing interleaves and the same scene is
`2n+1` -- measured, and gated, because a headless budget that forgets the
atlas is two short at every size and looks like a scene that costs less.
shaderbox always loads one. Nothing is culled either, so an off-screen node
still costs its runs and a host cannot budget on the visible count.

Both cases must use a FRESH handle. Measuring them against one reports the
first case's textures in the second -- the result points into the handle's own
storage. That misread cost a minute here and a minute upstream, independently,
and the gate carries a break for it.

## What imgui still owns here

The window, the tab row, the context menus and the Group prompt. Moving those
is the next surface, not this one.

## An engine uniform is a CONTROL, and the kind is what shows

The library picks a row's background from its ATTRIBUTE KIND -- input,
output, control, both -- in `attribute_role_color`. So a control is drawn in
the neutral control tone where an input is green, measured at
(0.33, 0.34, 0.42) against (0.29, 0.45, 0.35), and that is what makes a
builtin read as a builtin.

The first attempt packed them as inputs with a refusing pin, on the belief
that a control could not be tinted at all. That belief was half right and the
conclusion was wrong: an attribute has no LABEL colour, so the pin is the only
tintable thing on a row -- but the row's own fill comes from the kind for
free, and it is the fill a reader actually sees. Packing them as inputs made
`u_aspect` and `u_resolution` look like free green ports, which the maintainer
caught immediately.

Also measured while chasing it: a HOLLOW pin draws its tone shaded by -0.45,
so a muted token arrives at 55% strength and reads as another dim port. Cored
keeps a centre at full strength. It does not matter here, since a control
carries no pin, but it is the reason the intermediate attempt still looked
wrong after the colour was "applied".

## Engine uniforms show their live values

A control row carries a read-only widget and the number the engine wrote this
frame, read from `Pass.uniform_values` after the render. The widget is chosen
by component count: a `LABEL` draws component 0 and nothing else, so a vec2
would show half of itself, and a read-only `DRAG` draws one field per
component while taking no pointer.

`FFI_Attribute.value` was declared and never read until ABI 3 -- reported
upstream with the measurement (a Label with a value gave the same glyph count
as one without, at every zoom, so it was not the LOD gate).

**The colour needed a theme export, which is what it got.** A row's
background comes from `attribute_role_color` reading the THEME, text colour
comes from three fixed theme roles, and at ABI 3 no export set the theme --
`gc_new` installed the default and nothing wrote it again. `pin_color` was
the only host-settable colour, and a control has no pin. Reported upstream as
the root cause and answered in ABI 4 with `FFI_Frame.theme` plus
`gc_default_theme` / `gc_get_theme`.

The palette is INHERITED and overridden, never built: a `Theme()` from zero
sets sixteen shading scalars to 0 and flattens every chamfer and shadow, and
restating them here would be the copy-that-drifts the library avoided by
making its own struct the boundary type. `theme_from` replaces the eleven
colour fields shaderbox has an opinion about and takes the rest from
`default_theme()`.

## Parity notes

- **Panning is the HOST's.** The library zooms itself from the wheel and uses
  `view_moving` only to suppress hover, so a host that sets the flag and never
  moves the view has no pan at all -- which is what shipped, because every
  gesture test drove wires and nodes and none dragged the canvas. Middle-drag,
  Alt+left-drag, and now a plain left-drag the library did not claim: the old
  canvas reserved that last one for the rubber band, and with no rubber band
  here it would otherwise do nothing.
- Ghosts are `fade` + `dashed` + `accepts = NONE`, so a press falls through.
- The rubber band, the snap guides and the mid-curve unwire badge are not
  re-expressed: the library has no primitive for them. A wire is removed by
  grabbing its input, which the library reports on the PRESS.
- **The scope decides the picture, not just the tab label.** The switchover
  packed the flat graph every frame: the tab row relabelled itself and the
  canvas never changed, so grouping a pass did nothing visible and the
  feature was absent while the row claimed it. `scoped_view` answers it now
  -- at the root a group is ONE box carrying its boundary ports, and inside a
  group's tab the outside neighbours are ghosts.
- **A box is a node standing for several passes**, which is where its
  awkwardness lives: it has no position of its own (it is drawn at its
  members' corner, so a drag is applied as a DELTA against a frozen anchor),
  its input slots belong to its members (so a wire dropped on it reaches the
  pass that holds the read), and its output dots name members rather than the
  group. Hence `CanvasNode.owners` and `.outputs` -- the two mappings every
  event is resolved through.
- Hover and selection are keyed by NODE KEY (`p:<pass>`, `b:<group>`,
  `g:in:` / `g:out:<pass>`) and not by pass name: at the root a group's box is
  one node and no pass carries its name, and a pass that both feeds and reads
  a group is drawn twice.
