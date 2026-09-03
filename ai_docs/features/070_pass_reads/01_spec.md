# 070 — The graph view of the pass strip (stub)

Status: **spec** -- a stub. The full spec is written in the next session, after a brainstorm the
maintainer opens with "I'm not sure how we'll do this". This file carries what is already fixed
and what is open, so that session starts from the decisions rather than from scratch.

## Fixed before this feature opened

- **The linear strip stays the default view and must read on its own** (071 D2). The graph view
  is an addition, and the strip is the thing it must not make worse.
- **Direction, from `069/01_spec.md § Out of scope`** (069 #19 option A, the maintainer's
  preferred order): small square thumbnails as nodes, laid out by evaluation order (the plan gives
  the topological order; rank = longest path from a source), edges as arrows from source pass to
  consuming pass labelled by the uniform (069 D9: a sampler `u_<pass>` reads that pass by name, so
  most labels are the pass name), feedback (a self-input, `u_prev`) as a loop mark on the node.
  Wiring stays edited in the gear; the view is read-only at first.
- **Draw with `draw_list` lines and beziers plus arrowheads on a plain child, not
  `imgui_node_editor`** (its own canvas and coordinate space, zoom and pan, and the hard-assert
  rule in `conventions.md ## Known quirks` are more machinery than a read-only picture of a few
  nodes needs). Recorded in 069 W-D's out-of-scope note; reconfirm at the brainstorm.
- **What a node knows** already exists: `Document.effective_graph()` and `pass_graph.plan_passes`
  give order, reads and the feedback set; `Document.sampler_source` (071 W-D) answers "what does
  this sampler read"; the strip's tile (`ui_primitives.preview_cell`) is the blit to reuse; the
  dormant tint (`COLOR.STALE_TINT`) is the dormancy cue.

## Open, for the brainstorm

- Tree or graph: a DAG can fan in (composite reads cascade and paint); does a tree reading with
  duplicated leaves read better than a true graph for the documents ShaderBox actually holds
  (two to eight passes)?
- Where it lives: a mode of the Passes strip (toggle), a tab of its own, or a popup like the gear.
- What a click does: the strip's one click sets the output pass; the view's click should agree.
- Layout at the panel's width: a left-to-right rank layout wraps badly on a narrow panel; a
  top-to-bottom one costs height the panel does not have.
- Whether the view stays read-only or the edge becomes editable (drag from a node to a sampler),
  and if so how that meets the gear's closed-set combo.

## Files it will touch (expected)

`shaderbox/widgets/pass_list.py` (or a sibling `pass_graph_view.py`), `shaderbox/ui_primitives.py`,
`shaderbox/theme.py`, `shaderbox/pass_graph.py` (rank/longest-path if the planner does not already
expose it), tests, the tutorial's strip screenshots if the view replaces the strip anywhere in it.
