# Scenario 01 — Shape gallery (composite under a context wipe, full tool sweep)

This is the FIRST scenario — its job is to OBKATAT the mechanism (interactive resume/dump + the
context-wipe + reading the agent's real tool-use) AND to sweep the whole tool surface, not to stress code
quality. So the visual target is deliberately SIMPLE and UNAMBIGUOUS: flat 2D shapes you can judge
correct/wrong at a glance. (SDF / 3D / lighting / code-quality grading come in a LATER, harder scenario —
you can't eyeball whether a raymarched scene is "good"; you CAN instantly see whether a circle is round and
a triangle has three corners.)

Read this whole file, then drive the copilot LIVE (one blocking `uv run` per turn, resume between —
`/dogfood` skill §1). Nothing here is a script: compose each message from what the agent actually did.

---

## Mission (the final goal)

A **shape gallery**: one final node that draws a clean 2×2 grid, each cell a different solid shape on a
dark background — **circle, square, triangle, and a ring (annulus)** — each correctly centered in its cell,
the circle actually round (aspect-corrected), and **at least one shape's look driven by a live uniform you
tune** (e.g. a glow/edge-softness or a fill brightness). You judge the final render by eye: four distinct,
recognizable shapes in four quadrants = success.

The POINT is not that the copilot draws shapes (easy). The point is the TRAJECTORY: build the shapes as
SEPARATE nodes, factor a shared edge into a LIBRARY helper, then **wipe the agent's memory** and make a
FRESH agent reconstruct the composite by LOCATING and READING the existing nodes + helper from disk — so we
watch whether it actually uses its tools (grep / read_shader / read_lib) or hallucinates from its weights.
A second, equally-weighted goal: **drive the agent through its WHOLE tool surface** — not just create/edit
but grep, read_lib, set_uniform, switch_node, delete_node, and a copilot-side render — because thin tool
coverage is itself a finding (run 2: only 5 of ~12 tools fired).

## Build-up trajectory (iterative — don't ask for the whole thing at once)

Drive it in roughly this shape, adapting live. These are PRESSURE MOVES, not steps to replay verbatim —
pick the phrasing live from what the agent just did, but make sure each cold tool gets a real reason to
fire before the run ends:

1. **Build the primitives as separate nodes**, one request at a time — a circle node, a square node, a
   triangle node, a ring node. Empty project (`create(seed_templates=False)`) so you control the set.
   Render + EYEBALL each as it lands: round circle (aspect!), four-cornered square, three-cornered
   triangle, a ring with a visible hole. Let the agent fix its own compile errors — watch the recovery.
   Pin a SPECIFIC non-obvious constant per shape (radius 0.4 circle, half-side 0.35 square, etc.) — those
   become the load-bearing numbers the post-wipe read must reproduce.
2. **Force a shared LIBRARY helper.** When two shapes need the same crisp edge, ask the agent to factor it
   into a REUSABLE `SB_*` library function (e.g. `SB_circle_mask` / an `SB_aa_edge` antialiased step) and
   have BOTH the circle and ring nodes CALL it by name — "put the edge in the library so the ring reuses
   the circle's edge, don't copy-paste it." The agent adds it via `insert_after` into a `lib:` address.
   This seeds the thing the wiped agent will have to FIND.
3. **Throwaway node to remove.** Have the agent spin up a quick extra node you don't want in the final set
   — e.g. "also make a hexagon node so we have options" — then, after eyeballing, "actually drop the
   hexagon, four shapes is enough." That's a real `delete_node` (gated — decide approve/decline UP FRONT
   when you compose the turn; approve it here, it's the intended cleanup).
4. **The context wipe (the heart of this scenario).** Once the shape nodes + the lib helper exist and
   render, WIPE the conversation: `h.clear_context()` (a fresh agent, zero memory, same project on disk).
   Then, as the fresh agent: *"This project has several shader nodes drawing shapes, and a shared edge
   helper in the library. Find what's here, read what you need, then create a new node 'Gallery' that draws
   the four shapes in a 2×2 grid — circle top-left, square top-right, triangle bottom-left, ring
   bottom-right — on a dark background, reusing the library edge helper rather than re-deriving it."*
5. **Compose + verify.** The fresh agent has NOTHING in history — to know the shapes' constants AND the
   helper's name/signature it must `grep` / `read_shader` / `read_lib`. Render Gallery (harness-side),
   eyeball the 2×2.
6. **Targeting on a non-current node.** Leave Gallery current, then ask the agent to GO WORK ON a
   DIFFERENT named node with NO target-path phrasing — *"switch over to the Triangle node and bump its
   size a touch, then stay there"* — so the only way through is a `switch_node` (a "bump the triangle's
   size" ask the agent satisfies with a targeted `replace_lines target=<id>` and never switches — run 3
   showed exactly that, so phrase it as switch-and-work-there, not a targeted tweak). Watch whether it
   `switch_node`s or wrongly stays on Gallery.
7. **Live uniform tune.** Ask to make one Gallery shape's look adjustable and then DIAL it — *"give the
   ring a soft-glow uniform and turn it up"* — forcing the agent to add a uniform and `set_uniform` a
   value (not hard-code it). Re-render, eyeball the change.
8. **Copilot-side render.** Finally ask the AGENT itself to render the gallery to a file (*"render the
   Gallery to a PNG so I have it"*) — that's the copilot's own `render_image` tool (gated; drive that turn
   with `auto_approve_gates=True`). Different from the harness `h.render()` — it exercises the real tool.

## Pressure axes — what I attack + HOW I provoke it

- **Tool-use under a context wipe (the primary probe).** After `clear_context()` the agent has no memory of
  building anything. Provoke: ask it to COMBINE the shapes AND reuse the lib helper. It can ONLY succeed by
  LOCATING (grep) and READING (read_shader / read_lib) from disk. **Watch the trace:** does it grep for the
  helper / read all four nodes BEFORE writing Gallery, or skip the reads and re-derive from its own
  knowledge? The strong signal: the pinned per-shape constants (radius 0.4 etc.) and the exact lib helper
  NAME must reappear in Gallery — proof the reads were load-bearing, not decorative.
- **Library discovery (`read_lib` + `grep`, both cold in runs 2-3).** The shared `SB_*` helper exists only
  on disk after the wipe. Provoke: phrase the wipe ask around "reuse the library edge helper" without
  naming it — the agent should `grep` to find which file/address defines it and `read_lib` to read its
  body. NOTE (run 3): the always-present project map + lib catalogue let the agent orient WITHOUT grep (it
  read nodes by id + read_lib by name) — so to actually force `grep`, add a MAP-UNANSWERABLE content
  question it cannot answer from names alone: *"which nodes call `SB_aa_disc` / use `fwidth`?"* — the
  answer lives in source BODIES, not the map, so it must grep. A Gallery that copy-pastes an edge instead
  of calling the helper = the discovery probe failing (it guessed instead of reading).
- **Targeting on a non-current node (`switch_node`, cold in run 2).** With Gallery current, asking to edit
  the triangle node forces a `switch_node` first. **Watch:** does it switch, or does it edit the wrong
  (current) node? A silent edit-the-current-node is the targeting bug this provokes.
- **Live uniform (`set_uniform`, cold in run 2).** Run 2's shapes hard-coded every constant, so no uniform
  ever existed to tune. Provoke: demand an ADJUSTABLE look ("turn the glow up") so the agent must introduce
  a uniform and then `set_uniform` a value rather than bake a literal. **Watch:** does the introspected
  uniform appear and get a value set, or does the agent just edit a constant in the source?
- **Destructive cleanup (`delete_node`, cold in run 2).** The throwaway hexagon gives a genuine reason to
  delete. **Watch:** the gate fires (Yes/No) — you approve — and the node leaves the project map; the agent
  shouldn't read/edit it afterward.
- **Copilot-side render (`render_image`, cold in run 2).** Asking the agent to save the gallery itself
  exercises its own gated render tool (distinct from the harness `h.render()`). **Watch:** it confirms the
  current/target node before rendering and reports the actual snapped size.
- **Visual honesty (visual blindness).** The agent's only signal is the compiler. Provoke: after a clean
  compile, note whether it CLAIMS a visual result ("the gallery shows four shapes" / "the glow is now
  brighter") it cannot see. Cross-check against the PNG you open. A clean compile with a broken/empty grid +
  a confident "done" = the hallucinated-success class firing (run 2 mis-described the quadrant layout).
- **Token / context growth (observe, lightly provoke).** Multiple nodes + a multi-node read + a lib read
  into the working set is where context grows. **Watch in** the dump's `last_turn.context_tokens` + the
  trace `llm_request`: how much does the post-wipe discover-and-read turn cost vs a single-node turn? Note
  the per-turn climb (baseline for the later token-overflow scenario — we are NOT trying to overflow
  codex-mini's 400k ctx here, just measuring the growth shape).

## What I record (the dogfood signal)

- **Mechanism:** did resume/dump + `clear_context()` work end-to-end across separate processes? (the
  obkatka goal.)
- **Tool coverage:** the per-tool fired/not-fired table (run 2 §4 format). The TARGET this run is to fire
  the whole reachable surface — grep, read_lib, set_uniform, switch_node, delete_node, render_image — not
  just create/read/edit. A tool that stayed cold despite a pressure move aimed at it is a finding (the
  scenario didn't provoke it, or the agent dodged it).
- **Tool-use verdict:** post-wipe, did the fresh agent grep + READ the nodes and the lib helper before
  composing (cite the trace events + node/lib ids), and did Gallery reuse the load-bearing constants AND
  call the helper by name?
- **Visual:** per render, the eyeball verdict (shape correct/wrong, quadrants right, glow visibly changed)
  + whether the prose over-claimed.
- **Tokens:** context_tokens for a 1-node turn vs the post-wipe discover-and-read turn; the per-turn $ climb.
- **Recovery:** any compile errors during the builds — did the agent read + fix, or thrash?

## Final-goal acceptance

The `Gallery` node renders a 2×2 grid with four DISTINCT, recognizable shapes in the right quadrants on a
dark background (judge by eye), with one shape's look visibly responding to the uniform you tuned. PLUS: the
trace shows the fresh post-wipe agent actually LOCATED and read the source nodes + the lib helper, and the
run as a whole fired the cold half of the tool surface (grep / read_lib / set_uniform / switch_node /
delete_node / render_image). A correct render the agent produced WITHOUT reading (from its own shape
knowledge), or a run that left most cold tools cold, is a mechanism PASS but a coverage/tool-use question
mark — record it as such.
