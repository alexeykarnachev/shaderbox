# Scenario: Broken-compile edit thrash

**Probes:** the missing circuit-breaker. An edit that APPLIES but compiles WITH errors returns ok=True
(`tools/shader.py::_applied_result`), so `consecutive_failed_edits` resets every step — the edit-giveup
cap never engages on a model that keeps producing applies-but-broken edits. Observed 2026-06-05
(`todo.md`): a turn ran 8 consecutive applies-with-errors, bounded only by `max_iterations` (one of
headroom). This scenario tries to induce that thrash and watch whether the agent self-recovers or loops.

**Setup:** `h = DogfoodHarness.create()`. Current = UV Mango.

---

Step 1 — ask for something fiddly that invites repeated broken edits
- User: "Rewrite this shader to render an animated plasma effect using several nested sin() calls of
  u_time, u_aspect, and the uv — make it elaborate."
  (Elaborate GLSL with engine uniforms that must be DECLARED — a common place the model forgets a
  declaration or a semicolon and has to re-edit.)
- Expect: read_shader → one or more edits → compile. Watch for the FIRST compile-with-errors.

Step 2 — observe the recovery loop
- `h.drive_until_idle()` and watch the printed events. Count the edit attempts.
- Human check (the probe): when an edit applies but compiles with errors, does the agent:
  - (a) read the error, fix it in ONE clean follow-up edit, compile clean — healthy; OR
  - (b) keep producing edits that each apply but re-introduce or shift the error, several in a row?
- If (b): how many consecutive applies-with-errors? Does the turn hit the `max_iterations` cutoff (the
  chat shows "I stopped after N steps")? That's the thrash with no circuit-breaker — the `todo.md`
  deferral firing.

Step 3 — force the failure mode if step 1 compiled cleanly
- If the agent nailed it first try, push harder: User: "Now also add a second color layer that uses a
  function `SB_hash3` from the library and blends it in." (A library call the empty dogfood lib doesn't
  have → the agent may try to declare/use it and thrash.)
- Human check: same — count consecutive broken compiles, watch for the max_iterations cutoff.

**What to record:** the max consecutive applies-with-errors you can induce, and whether the turn
self-recovers or hits max_iterations. If it routinely thrashes ≥4-5 in a row, the separate
`consecutive_compile_failures` counter + nudge (the `todo.md` honest fix) is worth building.
