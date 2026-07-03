# 054 — Copilot visual craft & robustness (faithful execution)

Make the copilot a **robust, competent visual craftsman that faithfully executes the user's visual
instruction** — whatever the style. The metric is **instruction-fidelity + baseline drawing competence**,
NOT realism. Ask for photoreal → photoreal; ask for cartoon → clean cel; ask for abstract → abstract. We
already give the copilot the tools (vision 053, the compiler, the whole engine, scripts, the tool set);
this feature gives it the **smarts** to use them well: when to reach for physics vs a script vs pure GLSL,
how to build so the result actually matches the ask, and a loop that converges the render onto the ask
instead of declaring done blind.

Why now: the one-shot US-flag dogfood (2026-07-03) exposed that the copilot is architected as a code-
CORRECTNESS editor, not a visual-CONVERGENCE craftsman. Asked for a clean waving flag with 50 stars, it
emitted one blob of fbm-noise "ripple", dropped the stars entirely, shipped a dark muddy frame, and
declared "cinematic done" — a total instruction-fidelity failure, and the exact fakes the shader labs
already learned to reject. The labs exist to work out the PROCESS of building visual effects well; that
knowledge was never transferred into the copilot (its prompt is ~100% tooling/mechanics, ~0% craft; the
lab lessons live only in the `shader-lab` skill, which drives *Claude in a lab*, not the in-app copilot).

The realistic flag is just the current tuning fixture — the capabilities here are style-agnostic.

## The diagnosis (four systemic gaps, fixed as a system)

The step-by-step flag reached lab quality ONLY because the user closed the loop each step (rendered,
judged, pushed). One-shot removes that and there is no internal replacement. A visual craftsman needs four
things the copilot lacks:

1. **SEE (aimed).** 053 gave it eyes, but the turn-end auto-look fires with an EMPTY `look_for` → a bland
   "structured, fine" that can't tell whether the result matches the ask.
2. **CONVERGE.** Nothing drives iteration toward the ask — it emits one pass, compiles, declares done.
3. **CRAFT + TOOL-SELECTION.** No craft knowledge and no idea when to use physics / a script / pure GLSL,
   so it reaches for the one lazy default (fbm noise) whatever the ask.
4. **ROBUSTNESS.** A big effect blows the per-turn token budget in one giant `write_shader` (turn 1 landed
   NOTHING) and hard sub-requirements (the 50 stars) silently drop.

## Goal (what changes)

1. **A `VISUAL CRAFT` block in the system prompt** (STATIC/cached tier) distilling the labs into
   style-agnostic + tool-selection craft the copilot applies ITSELF in code (NO prebuilt `SB_*`
   primitives — it derives the real model). Contents: match-the-asked-look (real→real model, stylized→
   clean cel, never a lazy in-between); pick tool by effect (per-pixel→GLSL, physics-sim/stateful→script
   pushing an array uniform, verify the sim numerically first); baseline quality ALWAYS (tonemap before
   output, ~1px AA of every edge, dither banding, drama=contrast+saturation+structured-texture-not-
   mottle); depth&light when the look has form (shadows>AO>normals, grazing light, finite-diff normal ÷
   sample-step, metal=reflection); motion when animated (emerges from feedback not an imposed sine,
   incommensurate rates + domain scroll, animate the silhouette); build-in-stages + SEE + iterate-until-
   it-matches. Plus ≤4 embedded short formulas (finite-diff normal, ACES, domain-warp, aeroelastic force)
   so a weak model doesn't mis-derive them from memory.
2. **Strengthen scripting-for-physics** in the existing SCRIPTING section: the current when-to-script is
   purely *mechanical* (pulse→script). Add the *craft* case: a PHYSICS SIM or stateful heavy compute
   (cloth/Verlet, particles) belongs in a `script.py` that steps state and pushes it to the shader as an
   array uniform; verify it numerically before rendering.
3. **Aim the vision loop at the ask (053 slice B follow-up).** The turn-end auto-look carries the turn's
   INTENT (derived from the user's ask) as `look_for` instead of `""`, so the eye critiques against the
   goal ("stripes muddy, no stars visible, dark"). And `_VISION_SYSTEM` gains a LEGIBILITY dimension (is
   the main subject clearly readable, or muddy/dark/washed-out/low-contrast?) so gross quality failures
   are reported, not glossed.
4. **Iterate-until-it-matches discipline** (prompt + the existing auto-look re-open): a visual task is NOT
   done at first clean compile; when the aimed eye reports the ask unmet, keep working. This reuses slice
   B's re-open-one-iteration machinery — the aimed critique is what makes that iteration productive.

## Out of scope (each with a trigger)

- **Prebuilt `SB_*` craft primitives (cloth/sky/tonemap helpers).** Maintainer-rejected — the copilot must
  derive the real model itself; a prompt teaches the technique, a library would do the thinking for it.
  **Trigger:** never, unless the maintainer reverses this.
- **A hard convergence LOOP (re-look + re-prompt until the eye is satisfied or a cap).** v1 uses the
  single slice-B re-open + the iterate-discipline prompt; a multi-round auto-loop risks a weak model
  spinning on fakes and burning budget. **Trigger:** a dogfood where the aimed single re-open isn't enough
  and the model stops one iteration short of a fixable result.
- **Reference-image targets (feed the user a picture, "match this").** A real TARGET representation is a
  bigger feature; v1's target is the text ask carried into `look_for`. **Trigger:** users want to supply a
  reference to match.
- **Raising `max_tokens_per_turn`.** The fix is build-in-stages (a robust habit), not a bigger budget that
  just moves the ceiling. **Trigger:** stages still can't land a legitimately huge single function.

## Design decisions (locked)

1. **Craft block goes in the STATIC system-prompt tier** (cached prefix, billed once per ~5-min TTL, ~4×
   cheaper on hits) — a fixed policy block, not per-turn. Cost ≈ +300-400 cached tokens; justified by the
   quality lift. Ordered least-volatile so it never busts the cache.
2. **Craft is STYLE-AGNOSTIC first, style-aware second.** The block leads with "deliver the look they
   asked for" and "baseline quality always" (universal), then the real-model / lighting / motion craft as
   *when-the-look-calls-for-it*, not "always be realistic." Instruction-fidelity is the frame.
3. **The aimed auto-look's `look_for` is derived from the turn's user_text** (the ask), truncated to a
   bounded length. run_turn already holds `user_text`; no new plumbing. Empty/whitespace ask → keep the
   generic baseline look (don't fabricate an intent).
4. **`_VISION_SYSTEM` legibility line is CORRECTNESS-adjacent, not beauty.** "Is the subject clearly
   readable vs muddy/dark/washed-out" is legibility (allowed); it must NOT drift into "is it pretty"
   (beauty stays the user's eye) — the wording stays observational.
5. **No new tool, no new agent-loop control flow beyond slice B.** The iterate-until-match is a prompt
   discipline + the existing aimed re-open; build-in-stages is prompt guidance. Keeps the change to
   prompt text + one `look_for` argument + one vision-prompt line.

## Files touched

- `shaderbox/copilot/prompt.py` — the new `VISUAL CRAFT` block in `_SYSTEM_PROMPT`; the scripting-for-
  physics addition; the iterate-until-it-matches + build-in-stages lines.
- `shaderbox/copilot/agent.py` — the turn-end auto-look passes a `look_for` derived from `user_text`
  (currently `""`); possibly a small helper to distill the ask.
- `shaderbox/copilot/llm/openrouter.py` — `_VISION_SYSTEM` gains the legibility dimension.
- Tests: `tests/test_vision_auto_look.py` — assert the auto-look now passes a non-empty `look_for` carrying
  the ask (extend the existing spy). A prompt-content smoke check that the craft block + physics-script
  guidance are present (cheap guard against a silent revert).

## Manual verification

**Deterministic (headless):** the auto-look `look_for` carries the ask (spy asserts the injected
`probe_render` got a `look_for` containing words from the user_text); `_VISION_SYSTEM` contains the
legibility clause; the prompt contains the craft block (marker-string test).

**Behavioral (dogfood — the real gate):** re-run the ONE-SHOT ambitious flag ("beautiful realistic waving
flag, hero shot") with NO step-by-step hand-holding, same cheap model. Compare to the pre-054 baseline
(dark mush, no stars, declared cinematic). Success = a measurable jump in instruction-fidelity: the stars
are present, the stripes readable, the frame tonemapped (not dark-muddy), and the copilot does NOT over-
claim (the aimed eye reports remaining gaps and it either fixes or honestly states them). Also spot-check
a DIFFERENT style ask (e.g. "flat cartoon flag") to confirm the craft block didn't hard-wire realism —
it should go clean/flat, not add fake lighting. Photorealism is NOT the pass bar; faithful, competent
execution of the ask is.

## Open questions for the user

None blocking — framing locked (instruction-fidelity + robustness + capability, realism incidental).
Proceeding to pre-implementation review on plan-lock.

## Review history

**Dogfood v1 (2026-07-03, same one-shot flag, cheap `codex-mini`):** a NIGHT-AND-DAY jump vs the pre-054
baseline. Before: dark muddy fbm-mush, no stars, declared "cinematic" (mean rgb 70,73,89). After: a bright,
properly-lit flag with real cloth folds (height-field + lighting normal from the craft block), soft
shading, a clean sky gradient — 6 tool calls across 8 iterations (build-in-stages behaviour), a compile
error self-fixed. mean rgb jumped to 140,150,170. The aimed auto-look WORKED: its vision line reported
`readability: main subject clear but lacks detail in stars` — the new legibility dimension + intent-carried
`look_for` caught the exact gap the empty look_for missed. **Remaining gap (the out-of-scope trigger now
fired):** the model got "lacks stars", did ONE edit, still didn't land the canton+stars, and OVER-CLAIMED
"blue canton with 50 stars" anyway. So craft-transfer + aimed sight landed; **convergence ENFORCEMENT** is
the next slice — a bounded loop that won't let the turn end (or the model claim the ask met) while the
aimed eye reports it unmet. Pulling the deferred hard-loop in, per its trigger.
