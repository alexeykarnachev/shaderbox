# 096 reproductions

Standalone scripts, one per defect, each verified against the tree at the time the findings were
written. Run them before fixing anything and again after — a description is not evidence.

    uv run python ai_docs/features/096_canvas_ownership/repro/<script>.py

Each builds its own standalone GL context, so none needs the app or a display. They create many
contexts between them: run them in a fresh process rather than after a long session, or they
segfault for reasons that have nothing to do with the code (see the spec's note on this).

| Script | Defect | What a BROKEN tree prints | What a FIXED tree prints |
|---|---|---|---|
| `f6_frozen_export.py` | F6, the frozen export | `iterations=1 -> [10, 10, 10, 10, 10, 10]` and `iterations=2 -> [23, 23, 23, 23, 23, 23]` | a climbing sequence for both |
| `f2_promote_scaled.py` | F2, promotion strands a scaled pass | `helper AFTER promotion: (128, 128)` for a 256x256 document, `BUG: True` | `(256, 256)`, `BUG: False` |
| `f1_dtype_clamp.py` | F1, the dtype default mismatch | `f1: max R = 255` against `f2: max R = 3.0` | unchanged — this one shows WHY the mismatch matters, not the defect itself |

All three print their FIXED column as of 096. `f2_promote_scaled.py` promotes through
`Document.set_output_pass`, which is where the resize lives and what every production caller
takes; assigning `graph.with_output(...)` by hand bypasses the verb and still strands the pass,
so it is not a reproduction of anything a click can reach.

`f6_frozen_export.py` decodes the written video rather than reading a canvas. That distinction is
load-bearing: reading the canvas within one frame shows a correct value and is what led one
investigation agent to report this defect as sound. Judge it on the decoded frames.

`f1_dtype_clamp.py` is a property demonstration, not a failing case: it establishes that writing
above 1.0 into an `f1` target clamps while `f2` keeps the headroom, which is what makes the
export branch's bare `Canvas(...)` a real loss rather than a cosmetic one. The defect itself is
read off the two sibling branches in `document.py`.
