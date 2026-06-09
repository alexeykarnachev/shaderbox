# Scenario: Circle, square, morph (multi-file read + aggregate)

**Probes:** the core multi-file capability (the maintainer's canonical example). Create two distinct
shaders, then ask the copilot to combine/morph them — this forces it to read BOTH nodes into the working
set, reason across them, and produce a third result. Verifies multi-node read + the human-eyeball shape
judgment end-to-end.

**Setup:** `h = DogfoodHarness.create(seed_templates=False)` (empty — build from scratch).

---

Step 1 — create a circle
- User: "Create a node 'Circle' that draws a filled white circle centered on screen, on a black
  background. Account for u_aspect so it stays round."
- Expect: create_node → compile (fix any errors in a follow-up).
- `h.render()` → open the PNG. Human check: is it a round white circle (not an ellipse — did it use
  u_aspect)?

Step 2 — create a square
- User: "Create a node 'Square' that draws a filled white square centered on screen, black background."
- Expect: create_node → compile.
- `h.render()` → open. Human check: a clean white square.

Step 2.5 — make the read CONTENT load-bearing (the real multi-file probe)
- The plain "merge circle + square" is solvable from the model's own knowledge, so it proves only that
  read_shader FIRES, not that the read content MATTERS (run-1 verified: the agent DID read both, but wrote
  a generic squircle). To force the content to matter, give Circle + Square DIFFERENT specific constants
  the morph must reuse VERBATIM. After creating Circle and Square, do:
  - User: "Edit Circle so its radius is exactly 0.73 (a named const RADIUS = 0.73), and edit Square so its
    half-size is exactly 0.61 (a named const HALF = 0.61). Keep them compiling."
- Now the morph below is UNSOLVABLE without reading both — the agent can't guess 0.73 / 0.61.

Step 3 — the morph (the multi-file probe, content-load-bearing)
- User: "Create a new node 'Rounded' that's a squircle whose circle part uses Circle's EXACT radius and
  whose square part uses Square's EXACT half-size. Read both shaders first to get those two numbers."
- Expect (the key check): the trace shows **read_shader with BOTH node ids** (Circle + Square) entering the
  working set, THEN a create_node/edit whose source contains BOTH 0.73 AND 0.61. Confirm both reads happened
  AND both constants appear in Rounded — THAT proves the read content was load-bearing (not just that the
  read tool fired).
- `h.render()` (switch to Rounded first if needed) → open. Human check: is it a rounded square — corners
  visibly rounded but flatter sides than a circle? If it's just a circle, or just a square, the morph
  didn't actually aggregate the two. AND grep the Rounded source for 0.73 + 0.61 — if either is missing /
  wrong, the agent did NOT actually use the read content (the multi-file read was cosmetic).

Step 4 — a parametric tweak (uniform, not source)
- User: "Add a uniform to control the corner roundness, and set it to halfway."
- Expect: edit_shader (add `uniform float u_round;` + use it) → set_uniform(u_round, 0.5) OR the agent
  explains it must be a uniform.
- `h.render()` → open. Human check: does the roundness look mid-way? Try `h`-driving another set_uniform
  value and re-render to see it change (proves the uniform is live).

**What to record:** did both source shaders actually enter the working set (multi-file read), and did the
aggregate visually combine them? This is the closest thing to "is the copilot useful for real authoring".
