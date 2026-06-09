# Scenario: Wrong-node demonstrative targeting

**Probes:** the agent resolving "this" / "it" / "the X" to a NAME-match instead of the CURRENT node.
Observed 2026-06-08 (`todo.md`): current node was "Input Types Test" but "project this onto a sphere"
made the agent edit the node NAMED "Raymarched Sphere" — it free-associated the word "sphere" to a node
name instead of resolving "this" to the node it was working on. The prompt has no pronoun→current-node
rule. This scenario crafts the exact trap.

**Setup:** start empty so you control the names: `h = DogfoodHarness.create(seed_templates=False)`.
Then create two nodes with names that bait a mis-resolution.

---

Step 1 — create two nodes, names chosen to bait
- User: "Create a node named 'Red Quad' that fills the screen with solid red."
- Expect: create_node → clean compile. `h.nodes()` should show it; it becomes current.
- User: "Now create a node named 'Blue Sphere' that draws a blue raymarched sphere."
- Expect: create_node → compile (may have errors — fine). It becomes current.
- Switch back so the CURRENT node is the one whose NAME does NOT match the next request:
  User: "Switch to Red Quad." → Expect: switch_node. Current = Red Quad.

Step 2 — the demonstrative trap
- User: "Make this one a circle instead."
  (Bare demonstrative "this", NO node name. The word "circle" doesn't match either name, but the agent is
  CURRENTLY on Red Quad — "this" must mean Red Quad.)
- Expect: edit_shader / replace_lines on the CURRENT node (Red Quad).
- Human check (the probe): read the trace — which node id did the edit target? If it edited Red Quad
  (current) → correct. If it grepped/switched to Blue Sphere or some other node → the bug fired.

Step 3 — the stronger bait (word-association)
- User: "Switch to Red Quad." (reset current)
- User: "Give the sphere a glow."
  ("the sphere" — there IS a node named Blue Sphere, but the user is on Red Quad and may mean "the thing
  I'm looking at". This is genuinely ambiguous — the HONEST behavior is to ASK, not silently pick.)
- Human check: did the agent (a) ask which node, (b) edit current (Red Quad), or (c) silently jump to
  Blue Sphere by name-match? (c) is the clearest bug; (a) is the ideal; (b) is defensible. Record which.

**What to record:** for bare demonstratives, does it resolve to current? For name-adjacent words, does it
ask or silently name-match? This tells you whether the prompt-level TARGETING fix (filed in `todo.md`) is
needed and whether a confirm-gate on non-current-node edits is warranted.
