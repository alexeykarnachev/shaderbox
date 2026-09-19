# 098 — graph_canvas

Replace the hand-drawn imgui node canvas with the `graph_canvas` library: an Odin
node-canvas renderer loaded over a C ABI, drawn by moderngl into an FBO the panel
presents. The library owns the look and the pointer gestures; shaderbox keeps the
model, the verbs and every document-shaped rule.

This is the first step of a larger intent the maintainer has stated: shaderbox
stops drawing its own UI in imgui. The graph is the first surface to move because
it is the one already drawn by hand on a draw list, so it is the surface where
imgui contributes least and costs most.

## Goal

The Document tab's graph view is drawn by `graph_canvas` and driven by its events,
with no imgui draw-list call in the graph's own painting. Every gesture that works
today still works, routed through the same `App` verbs.

Feature parity is the bar, not a redesign. A gesture whose shape the library does
not offer is listed under *Out of scope* with what replaces it, not silently
dropped.

## Design decisions

1. **The library is vendored as a built `.so`, exactly as feature 067 vendored
   `libeditor.so`.** `shaderbox/resources/graph_canvas/libgraph_canvas.so` plus its
   `atlas.png` / `atlas.json`, copied from `~/src/graph_canvas` by a documented
   refresh step. The build allowlist already copies `shaderbox/resources`, and
   `build.sh` already strips the sibling `libeditor.so` per platform — the same
   line grows to cover this one. No git submodule: 067 set the precedent and the
   source repo is the maintainer's own, on the same machine.

2. **The binding is a leaf ctypes module** — `shaderbox/graph_canvas/ffi.py`, no
   imgui, no moderngl, mirroring `shaderbox/editor/ffi.py`. It declares the
   structs, proves the layout at load through `gc_sizeof` / `gc_offsetof` /
   `gc_field_name` / `gc_enum_count` / `gc_enum_name`, and refuses to load on a
   mismatch rather than reading garbage at plausible offsets.

3. **The renderer is a moderngl module** — `shaderbox/graph_canvas/render.py`,
   mirroring `shaderbox/editor/render.py`: one shared program pair + atlas per GL
   context, one panel per drawn graph owning an FBO texture. The library's own
   `resources/shaders/*.glsl` are vendored and used unmodified; they compile in
   moderngl as-is (verified — see *Evidence*).

4. **The panel presents through `imgui.image`, as the editor already does.** imgui
   keeps the window, the tab row, the context menus and the popups for this
   feature; only the canvas interior stops being imgui-drawn. Moving the menus is
   a later surface, not this one — the goal is one surface fully migrated, not
   every surface half-migrated.

5. **`shaderbox/pass_graph.py` and the pure half of `widgets/graph_state.py` are
   kept and reused unchanged where they still apply.** The model, the planner,
   `group_boundary`, the cycle rules and the `App` verbs are not part of the
   renderer swap. What goes is the drawing and the imgui hit-testing.

6. **Identity crosses by `id`, never by index.** Every node carries a stable `u64`
   derived from the pass name, and every edge one derived from (consumer,
   sampler), so an event names a pass rather than a position in this frame's
   array. The library states this is what the field is for.

7. **The generic half is a package, not a shaderbox module.** `shaderbox/graph_canvas/`
   holds `ffi.py` and `render.py` with no import of any shaderbox model type —
   they take plain arrays and a GL context. The shaderbox-specific half (turning a
   `Document` into nodes, turning events into `App` verb calls) lives beside it in
   `adapter.py` and imports both. That split is what makes the pair liftable into
   another project, and it is checked by a gate rather than promised in prose.

8. **The pointer is fed from imgui's io, not from glfw directly.** imgui already
   owns the window's input and reports whether the canvas region is hovered; the
   panel converts that to the library's pointer flags. Going around imgui for input
   while imgui still owns the window is how a gesture ends up delivered twice.

## Out of scope

Each with the trigger that brings it back.

- **Moving the context menus, the tab row and the group-name prompt off imgui.**
  They stay imgui popups this feature. *Trigger:* the next UI surface migrated off
  imgui, or the library gaining its own menu primitive.
- **Ghost nodes, the unwire ✕ badge, snap guides and the rubber band** are host
  concerns the library does not draw. They are re-expressed with what it does
  offer (a faded dashed node is `fade` + `dashed`; a ghost is a node with
  `accepts` refusing everything) or listed as a parity gap in `02_progress.md`
  with a measurement. *Trigger:* a gap that has no expression at all halts and
  goes to the maintainer.
- **The strip view** (`widgets/pass_list.py`) is untouched.
- **Node-body widgets.** The library can draw a drag/slider/colour row inside a
  node; shaderbox's uniform rows stay in the Uniforms tab this feature. *Trigger:*
  the panel-density question 094 left unanswered being asked again.

## Files touched

New: `shaderbox/graph_canvas/__init__.py`, `ffi.py`, `render.py`, `adapter.py`,
`panel.py` (the imgui seam: the pointer rule, the canvas state, the framing);
`shaderbox/resources/graph_canvas/` (the `.so`, the atlas pair, the shaders);
`tests/test_graph_canvas_ffi.py`, `tests/test_graph_canvas_render.py`,
`tests/test_graph_canvas_adapter.py`, `tests/test_graph_canvas_panel.py`,
`tests/test_graph_canvas_gestures.py`.

Changed: `shaderbox/widgets/pass_graph.py` (the drawing and hit-testing half
replaced by the panel), `shaderbox/widgets/graph_state.py` (imgui-shaped state
retired where the library now owns it; the scope's own picture -- boxes and
ghosts -- resolved here), `shaderbox/app.py` (the renderer and the per-document
canvases, released on a project switch), `build.sh` (the per-platform `.so`
strip), `Makefile` (the layering gate), `ai_docs/roadmap.md`,
`ai_docs/conventions.md`.

Deleted: the imgui draw-list painting in `widgets/pass_graph.py` — the
primitives and the parts of the interaction loop the library now answers.

## Evidence already gathered

- The library loads from shaderbox's own interpreter: the ABI version matches
  on both sides (proven every run by `test_the_library_loads_and_its_layout_is_proven`,
  which reads it rather than restating it), the atlas loads, and
  `gc_atlas_distance_range` returns 8.0. It began at ABI 2 and reached 4 during
  the integration, as the gaps below were filled.
- Both reference shaders compile unmodified under moderngl 3.3 core, and a
  four-node shaderbox-shaped graph renders correctly — nodes, pins, bezier wires,
  MTSDF text, grid.
- A shape run can start at a non-zero `first`, so the renderer must honour the run
  offset; moderngl has no base-instance parameter, so the offset is applied by
  binding the attribute arrays at a byte offset. Reported upstream, and gated —
  `test_a_shape_run_resumes_after_a_glyph_run_at_a_nonzero_offset` asserts the case
  exists rather than restating a count, which moves with the scene and the build.

## Open questions for the user

None blocking. Two the maintainer may want to answer as it lands:

- Whether the graph panel should keep imgui's own context menus long-term, or
  whether this feature should already prove a non-imgui menu (currently: keep).
- Whether node sizing follows the library's layout or shaderbox's current one
  (currently: the library's, since fighting its layout defeats the adoption).
