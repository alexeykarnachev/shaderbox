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

- `widgets/pass_graph.py` 1537 -> 368 lines; `widgets/graph_state.py` 296 ->
  116; net −2067 lines including tests.
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

## What imgui still owns here

The window, the tab row, the context menus and the Group prompt. Moving those
is the next surface, not this one.

## Parity notes

- Ghosts are `fade` + `dashed` + `accepts = NONE`, so a press falls through.
- The rubber band, the snap guides and the mid-curve unwire badge are not
  re-expressed: the library has no primitive for them. A wire is removed by
  grabbing its input, which the library reports on the PRESS.
- Group boxes still open into their own tab through the box menu; the root
  scope does not yet draw a group as one collapsed box.
