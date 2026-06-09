# Scenario: Agent-level error recovery (the biggest untested gap)

**Probes:** the single most important copilot behavior for real-user robustness — does the agent READ a
failure (a compile error, a bad edit, a malformed call) and CORRECT itself, or does it loop to
`max_iterations`? The 2026-06-09 dogfood run NEVER tested this: every failure the agent saw was the
transient GL quirk (a blind byte-identical retry, since fixed) or one designed `u_time` reject. So
"the copilot recovers from its own mistakes" is UNPROVEN. This scenario forces real agent-level failures.

**Setup:** `h = DogfoodHarness.create()`. Current = UV Mango.

---

## Part A — broken-compile recovery (the compile-error read→fix loop)

Step A1 — force a compile error, then see if it self-corrects
- User: "Rewrite the shader to output `vec3 c = u_missing * vs_uv;` as the color, where u_missing is a
  uniform you forgot to declare."
  (This compiles with an "undeclared identifier u_missing" error — the agent must READ that error and add
  the declaration.)
- `h.drive_until_idle()` and watch: does the agent (a) get the compile error in the tool result, (b) read
  it, (c) add `uniform vec3 u_missing;` and recompile clean? Count the iterations.
- Human check: read the trace. Did it converge (compiled clean within a few edits) or thrash? If it kept
  re-submitting edits that re-introduce the error, that's the broken-compile gap.

Step A2 — the harder thrash (overlapping nested errors)
- User: "Make an animated plasma using several nested sin() of u_time, u_aspect, and uv — elaborate, and
  use a helper function SB_wave you define inline." (Many places to drop a semicolon / forget a
  declaration → repeated applies-with-errors.)
- Human check: max consecutive applies-with-errors; does it hit the `max_iterations` cutoff ("I stopped
  after N steps")? Inspect the trace for `edit_giveup` / `turn_done cutoff=max_iterations` events.

## Part B — bad-edit recovery (old_str mismatch)

Step B1 — the agent's own stale edit
- User: "Change the blue channel computation to use 0.25 instead of 0.5." (A normal edit.)
- Then immediately, WITHOUT the agent re-reading: User: "Now change that same line to use 0.1."
  (The working-set rebuild should give it the CURRENT source — but watch whether its `old_str` matches.)
- Human check: did any `edit_shader` return "old_str not found"? If so, did the agent re-read + retry
  correctly (the working-set block has the live source), or give up? The token matcher (`glsl_lex`) should
  make whitespace-divergent old_str succeed — verify it does.

## Part C — bad target recovery (typo'd node id)

Step C1 — an invalid node id
- User: "Edit node 'zzzz' to be solid green." (No such node — the agent should either correct to a real
  node from the project map, or report it can't find it.)
- Human check: does the agent (a) recognize 'zzzz' isn't in the project map and ask / pick the current
  node, or (b) blindly call edit with target='zzzz' and get an "unknown node" error then recover? Either
  is acceptable IF it recovers; a loop or a hallucinated success is the bug.

**What to record:** for EACH failure class — does the agent READ the error and CONVERGE, or loop to
max_iterations / give up / hallucinate success? This is the robustness signal the whole dogfood exists
for. Cite the `edit_giveup` / `max_iterations` / `consecutive_failed_edits` trace events (the harness has
no assertions — you read them by eye).
