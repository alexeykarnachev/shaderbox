# Scenario 02 — Logo / hero-image design (iterative art direction with a human in the loop)

Drive the copilot to design a polished BRAND IMAGE (a logo, a hero shot, an app icon) from an empty
project, refining it over many turns the way a real art-director session goes: render, eyeball, give
direction, render again. This is the scenario that exercises the copilot as a CREATIVE collaborator,
not a coder fixing a bug — and it is the harshest test of the copilot's one true weakness, **visual
blindness**, because the whole task is judged by eye and the agent cannot see.

It is NOT tied to one logo. The shape, palette, subject, and concept are open every run — pick a fresh
brief each time (a neon wordmark; an emblem + wordmark; a minimal monogram; an animated icon; a
glassy badge). What stays fixed is the PROCESS and the weak spots it attacks. Drive it LIVE, one
blocking `uv run` per turn, composing each message from the render you just opened — never a
pre-scripted sequence (the `/dogfood` skill §1 has the mechanics + the key footgun).

The human (you) is the art director AND the only pair of eyes. Open every render with Read, judge it
honestly, and — this is the scenario's defining move — **send each candidate to the maintainer with
`SendUserFile` and let them steer the direction** (this run was driven exactly that way: the maintainer
picked "mark + wordmark", killed a reflection wave, asked for a CRT effect, rejected glass haze). The
copilot proposes pixels; the human picks the path.

## Mission (the final goal)

A finished, show-able brand image rendered from a single Logo node: a clear focal subject, a readable
wordmark if the brief has text, a deliberate palette, and at least one "fidelity" pass of detail
(glow, accents, a post-effect like scanlines/vignette/fisheye) — judged good by the maintainer's eye,
not by you alone. The artifact is the PNG; save the accepted one to `docs/branding/` at a real
resolution (render at 1024 via `h.render(size=1024)`).

## Build-up trajectory (adapt live — this is art direction, not a recipe)

Roughly this arc, but let the maintainer's feedback redirect it at any point:

1. **Block in the subject.** One open request ("design a logo for ShaderBox, a GLSL playground — bold,
   readable, on a dark background; use the SB_* library, don't hand-roll glyphs"). Render, eyeball,
   send it. Expect it crude: clipped text, wrong proportions, a muddy palette.
2. **Fix the structure before the paint.** Composition, sizing, readability — the load-bearing fixes.
   This is where the agent flails most (see pressure axes). Give it ONE concrete defect per turn.
3. **Lock the palette.** A separate pass once structure is right. Drifts easily — the agent will wash a
   saturated color out to pastel over a few edits and call it unchanged.
4. **Add fidelity detail.** Accents, an underline, corner marks, a post-effect (scanlines / vignette /
   barrel-distortion "CRT" look). One coherent idea at a time; a big multi-feature turn invites thrash.
5. **The three-way fork (optional, strong).** Clone the clean node into 2-3 independent project copies
   (`cp -r` the proj dir, delete its `copilot/` for a fresh conversation), give each an open
   "improve it your way" prompt with a DIFFERENT slant (more alive / more premium / more character),
   run them in PARALLEL (background `uv run`), render all, and present the set for the maintainer to
   pick. Surfaces the agent's taste — and its tendency to overdo a free brief.
6. **Final polish + save.** Strip what the maintainer rejected, render at 1024, save to `docs/branding/`.

## Pressure axes — what this attacks + HOW

- **Visual blindness (THE probe).** The agent's only signal is the compiler + the render-facts line
  (`ink %`, `bbox`, a coarse `luma` grid). None of that encodes *readability*, *symmetry*, or *beauty*.
  So the agent will confidently "improve" a wordmark into an illegible blob and report it "crisp and
  readable" (happened: a shrink turn merged all 9 letters into one smear; the agent claimed success).
  **Watch:** does the prose over-claim a look it can't see? Cross-check the PNG. The honest signal: when
  the same render-facts line repeats across N edits, NOTHING changed — the agent should notice and
  usually doesn't. Feeding back the facts verbatim ("luma is identical across your last 3 edits, so the
  image didn't change") is what unsticks it. This is the live demonstration of the `todo.md` VLM-judge /
  render-facts-honesty deferral — every run here is another datum for it.
- **Precise-diagnosis dependency.** The agent is GOOD when handed an exact defect ("the SDF sign is
  wrong: length(p)*sign(p.y) is not negative-inside" / "letter advance is (1+spacing.x)*0.5*char_height,
  so raise spacing.x to 1.8") and lands it first try. It is BAD on vague asks ("make it nicer") — it
  crutches uniforms blind. **Read the library helper's doc/signature yourself** and give the agent the
  real math; don't make it guess. The art direction quality you put in is the ceiling on what comes out.
- **Palette drift.** A saturated color, edited a few times through glow-mix code, decays to washed-out
  pastel. **Watch:** ask for "bright saturated cyan vec3(0.2,0.95,1.0) + a warm glow halo" with literal
  values, not adjectives, and check the render actually pops.
- **Free-brief over-reach.** Given "improve it however you like," codex-mini tends to PILE ON (a glass
  sheen that buried the focal wave, a frosted haze nobody asked for). The strongest single-element
  changes win; the kitchen-sink turns lose. The fork step (5) makes this visible.
- **No-op CLEAN spree + giveup brake (cost trap).** On a multi-edit turn the agent can issue the SAME
  successful edit several times, or thrash whole-file rewrites that re-introduce its own brace errors,
  until the `EDIT UNDONE` circuit-breaker fires (6 broken edits → restore last clean). It DOES recover,
  but a single turn can burn 200k+ cumulative input and $0.03. **Watch the trace** for repeated
  identical `replace_lines` and orphan-`}` rewrites; note the cost. (Another datum for the giveup-counter
  deferral — the no-op-clean case escapes the brake because each edit is ok=True.)
- **Animation is invisible on a still (harness limit, not an agent bug).** Ask for "make it alive"
  (flicker, a highlight traveling the wave, particles) and the agent writes real `u_time` animation —
  but `h.render()` samples t=0, so the still looks unchanged. **Don't judge animation from one frame:**
  render a few times across t (or note it as un-eyeballable here and hand it to a live `make run`). This
  is a dogfooding-framework gap to log, not a copilot failure.
- **set_uniform doesn't persist (resume trap).** A `set_uniform` value is in-memory only until a project
  save; a fresh resume process renders the source's inline default, not the tuned value. So LOCK accepted
  tweaks into the source with `edit_shader` (change the `uniform ... = <default>;` literal), not
  `set_uniform`, or the look reverts next turn. (The agent will reach for `set_uniform` to tune live —
  fine for exploring, but persist the winner.)

## Tool coverage (secondary here)

This scenario naturally hits create_node / read_shader / edit_shader / replace_lines (ranged + whole-file)
/ insert_after / set_uniform / read_lib + grep (to find SB_sd_text / SB_sd_box / SB_op_round / SB_fill /
SB_glow / SB_sd_segment signatures). It does NOT exercise delete_node / switch_node / publish unless you
add a reason (a throwaway variant node to delete, a second logo node to switch between). Coverage is not
the point of this scenario — the visual-blindness + art-direction loop is. If you want a coverage sweep,
run 01.

## What I record (the dogfood signal)

- **Visual honesty:** per render, did the prose over-claim vs the PNG you opened? The standout cases.
- **Diagnosis leverage:** which turns landed first-try (you gave exact math) vs flailed (vague ask)?
- **Cost spikes:** the turns that thrashed (identical edits, EDIT UNDONE) + their cumulative tokens / $.
- **Palette/structure drift:** did a value the agent claimed "unchanged" actually move, or vice versa?
- **The maintainer's path:** which directions they picked/killed (the human-in-the-loop is the judge).
- **Framework gaps:** anything the still-render couldn't show (animation, live tuning) → a `todo.md` note.

## Final-goal acceptance

The maintainer accepts a render as the logo/hero, and it's saved to `docs/branding/` at 1024. PLUS the
run produced fresh visual-blindness / cost / framework data for the report. A beautiful logo the agent
reached only because YOU spoon-fed every pixel-level fix is still a PASS for the artifact but a flag for
the copilot (it means the unguided creative loop is weak) — record it as such.
