# 096 — Canvas ownership: the log

What actually happened, wave by wave, appended as each lands. A log, not a plan — the plan is
`01_spec.md` and the evidence is `00_findings.md`. **On resume, read this file and
`git log --oneline` first**: they are the truth about where the work stopped.

Each entry carries its done-condition (written BEFORE the wave started, as a checkable
statement), the verification result, what was ruled out and why, and any surprise worth the next
reader's time.

## Before anything: establish a real baseline

Run `make gates > /tmp/g.log 2>&1; echo $?` on the untouched tree and read the exit code. It was
GREEN when this spec was finished — check, test and smoke all passed — so a red baseline means
something changed, not that red is normal here.

One environmental note, since it cost an hour to diagnose and will look alarming: **the suite
cannot run while the machine's monitor is switched off.** With no connected output, `App.__init__`
calls `glfw.get_video_mode` on a NULL monitor and the process segfaults inside glfw — no
exception, just `Fatal Python error: Segmentation fault` in `tests/conftest.py`'s `app` fixture,
plus `node down: Not properly terminated` and an xdist `KeyError` when run in parallel. The
`repro/` scripts keep working throughout, because they build a standalone context and never
construct an `App`.

If the gate dies that way, check whether the display is on before touching any code. Guarding
that call is a real (small) improvement and it is NOT part of this feature; do not let it pull
the night off course.

## This is an unattended run

Nobody is watching. `01_spec.md` has a "Running this unattended" section and it is the contract:
carry on to the end, never ask, commit and push each wave, never weaken a test to get green, and
write down every choice made along the way. The one legitimate early exit is a red baseline gate
with an empty source diff — the box cannot run the suite, so nothing the night produced could be
trusted.

Append to this file BEFORE starting the next wave, not at the end. A log written only at the end
does not exist when it is needed.

## W-0 invariant checker — LANDED

done-condition (written in advance): a test helper asserts, for every canvas reachable from a
Document, that the live canvas matches what its graph entry implies and that a history matches
its live canvas on size, dtype, filter and wrap; driven over `filter_linear=False`, `wrap=True`,
`dtype="f4"`, `scale=0.5`; and it FAILS on the untouched tree for F2 and for the
`resample_canvas` filter/wrap mutation. No file under `shaderbox/` is modified by this wave.

The failure requirement is the point: a W-0 that passes everywhere checks nothing. Prove it red
before writing a single line of W-1.

**Result: red where it had to be red, clean on a healthy document.** `tests/canvas_invariants.py`
holds `canvas_violations(document)`, which compares every pass's live canvas against what its
graph entry implies -- output sized to the document, everyone else to its own scale -- and every
history against its live canvas, on size, dtype, filter and wrap. `NON_DEFAULT` is the corner it
is driven over: `scale=0.5`, `dtype="f4"`, `filter_linear=False`, `wrap=True`, each the opposite
of its default.

Three probes, run before any production line changed:

- a two-pass document after `set_canvas_size` -- clean, so the checker is not failing everywhere;
- F2, promoting the scaled `helper` to output -- `canvas (128, 128) but graph implies (256, 256)`;
- `resample_canvas` with `filter=` and `wrap=` deleted, the mutation the findings say the whole
  suite passes -- `(9729, 9729), False` against the implied `(9728, 9728), True`.

A fourth probe covered the history arm: a history built with the wrong `wrap` beside a correct
live canvas is named. The output pass carrying `scale=0.5` reports CLEAN, which is the output
exemption asserting itself rather than a gap.

No file under `shaderbox/` was touched. Returned `assert_canvases_agree` alongside the list
form -- the list is what lets a report name every violation at once, since these defects arrive
in groups.
