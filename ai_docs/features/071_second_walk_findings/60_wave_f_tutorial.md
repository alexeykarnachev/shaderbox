# 071 W-F — The tutorial: cuts, the prose bar, the drawing chapter (#4 #8 + D1 D6 D7)

Parent: `01_spec.md § W-F`. The maintainer's words: "let's completely skip this scripting stuff
from the beginning of the tutorial, let's not disturb the radiance cascade narrative with these
brushes and strokes. But at the end of the tutorial you should describe the whole mechanism and
help the user to implement a proper drawing with memory, with strokes, with clearing, with all
this stuff" (D1); "no need for this 'Verifying it' section ... The matching visual is enough to
verify" (D6); "Don't use ... this 'bla bla bla, but not bla bla' construction" (D7).

## What landed (`tutorial_body.html`, then `build_tutorial.py` regenerated `tutorial.html`)

- **Cut.** The lede's mouse clause; the whole "Paint it with the mouse" subsection under step 1
  (uniforms, capsule, script, the "nothing to clear" note); the "Things to try" item on wiring
  `u_prev`; the "Verifying it" section (the oracle numbers, the mutation-test paragraph, the
  two measuring-wrong callouts), its contents entry, and the intro's `oracle.py` clause. Steps
  1-6 are the six shipped passes and nothing else. `oracle.py` stays where it is: the example's
  test and the image build run on it.
- **Rewritten under D7.** Every "X, not Y" / "rather than" site in what remains: the step-2
  heading (contents and section) reads "every solid texel's own position"; "coordinates, not colors"
  is "coordinates"; "not just the solid ones" is "the empty ones included"; "derived from the
  canvas, not from the run count" is "derived from the canvas"; the engine-index note (the
  maintainer's example) is "The engine hands over the run index. `u_pass_iteration` says which
  run this is, and the halving offset is one line in the shader."; "rather than the size you
  happen to be using" is gone. Zero contrast constructions remain in the body, measured with
  whitespace collapsed.
- **Added.** A closing chapter, "Draw into it", after "Things to try", listed in the contents
  under "Along the way". Paste-on-top on the finished example, in four steps each with its
  mechanism: a pass that remembers (`u_prev` declared, the fallthrough copying the previous
  frame, the name rule and the two-texture swap explained); the brush (three uniforms and the
  capsule stamp, why a segment from `prev` makes a continuous stroke); the script (`Alt+R`,
  the `Brush` behavior, the pass block by name, `ctx.mouse`, the re-entry reset); wiping it
  (`F6` / Reset restarts histories, clock, script and bound videos, document-wide on purpose);
  and why the example ships without it (a fixed export cursor with the button up; state starts
  from black). The lede now ends "At the end, a canvas you draw into."
- **Gate** (D7). `tests/test_tutorial_build.py::test_the_prose_says_what_a_thing_is` counts
  `, not ` / `— not` / `-- not` / `rather than` in the body's prose (code blocks excluded) and
  in every Help section, and fails above two; the allowance exists for the rare sentence where
  the excluded thing is the point, and the fix for a failure is a rewrite, never a larger
  allowance.

The existing tutorial gates all pass on the new body: every hand-written fragment names only
uniforms the example or the brush declares (`u_prev` is declared by `jfa` and `cascade`), every
script instruction carries `Alt+R`, no sentence instructs adding a script, every chord quoted
(`Ctrl+Shift+N`, `Alt+P`, `Alt+R`, `F6`) is in `COMMAND_SPECS`, and the committed
`tutorial.html` is a fresh build. `tests/test_prose_spelling.py` gained an exclusion for itself:
once tracked it flagged its own regex and docstring.

## Review history

**Post-impl prose audit (one reviewer, opus, anchored to the maintainer's words).** Verdict FIX,
taken in a fix-up: the step-2 heading's rewrite ("each texel's nearest solid") stated step 3's
result, which the body's own produces-lines contradict -- now "every solid texel's own position";
one "X, not Y" survived at the cascade's merge note because the line wrapped between the comma
and "not", where the gate's regex could not see it -- the sentence is rewritten and the gate
collapses whitespace before matching (a mutation of three added instances fails it); the
"What you built" table still called `paint` "the drawable scene", the one D1 leak a grep does
not find -- now "the scene to light"; the stamp snippet's `p` is named as the `vec2 p = vs_uv`
the pass opens with. Verified by the auditor by running the chapter headless on the shipped
example: `u_prev` wires itself with no gear step, a drag's stroke is continuous and persists after
release, it lights the scene (`composite` reads bright beside it), F6 wipes it; every anchor
pairs with an id, no oracle word survives, every pass step has an image and a one-line
"produces", and the script, `Alt+R`, the pass block and the mouse fields all match the engine.

## Manual verification (the maintainer, in the app, with the tutorial beside it)

1. Steps 1-6 mention no mouse, no script, no oracle; there is no Verifying section.
2. The prose reads as statements; no "X, not Y" left that the eye catches.
3. The last chapter, followed literally on the shipped example: after "A pass that remembers"
   nothing visibly changes; after the brush and the script, holding the left button draws a
   glowing stroke that stays and lights the scene; a fast drag is continuous; F6 wipes it and
   the warm light restarts at its t=0 position.
