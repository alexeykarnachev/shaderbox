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

## W-1 F6, the frozen export — LANDED

done-condition (written in advance): `repro/f6_frozen_export.py` prints a CLIMBING sequence at
both `iterations=1` and `iterations=2`, judged on the decoded video rather than on a canvas read,
and `make gates` is green.

**Reproduced first, on this tree**, since every finding was verified at a commit now behind us:
`[10, 10, 10, 10, 10, 10]` and `[23, 23, 23, 23, 23, 23]` — frozen at both counts, exactly as
`00_findings.md` records. After the fix: `[10, 23, 38, 50, 62, 75]` and
`[23, 50, 75, 100, 127, 151]`.

The fix is the one the spec decided. `draw_into` is gone: every iteration draws into
`render_pass.canvas`, and after the loop an external `canvas` receives the output pass's picture
through a blit. `Document` already held a `CanvasResampler` for the resize path, so the blit is
that same tool — factored out of `resample_canvas` into `_blit_into(source, target)` so the two
callers share one home rather than each building a resampler.

`Pass.render`'s `canvas=` argument is no longer passed from this loop at all: it was constant
`None` after the change, and a constant argument reads as a choice that is still being made.
`render`'s own docstring said `canvas` "overrides the OUTPUT pass's target"; that is no longer
the mechanism, so it now says it RECEIVES the output's picture.

Ruff reformatted `tests/canvas_invariants.py` after the W-0 commit (a hook autofix, no behavior);
it is folded in here.

## W-2 F1 and F9, the default mismatch — LANDED, with (c) REVERTED

done-condition (written in advance): the export's fit branch allocates a canvas carrying the
output pass's dtype, filter and wrap; a pass file with no graph entry is born with the same
target the graph is backfilled with; and `make gates` is green.

(a) and (b) landed. **(c), changing `Canvas`'s own dtype default from `f1` to `f2`, is reverted**
— it is a real defect in the change rather than a stale expectation, and the spec's rule is that
a failing test means the change is wrong.

**What (c) broke: 9 tests, and not by naming the default.** Four of them read a texture with a raw
`texture.read()` and a uint8 reshape instead of going through `texture_to_rgba8`. An `f2` texture
returns twice the bytes, so those reads fail outright (`cannot reshape array of size 16 into shape
(1,2,4)`) or silently return garbage — a pass that wrote pure red read back as `(0, 60, 0, 0)`.
`01_spec.md` warns of exactly this trap, and the blast radius is wider than the spec assumed when
it scoped (c) as "any remaining bare `Canvas(...)` becomes f2": it changes what every raw-byte
reader in the tree sees. Two further tests argue for `f1` BY NAME with a recorded rationale, so
(c) also reverses a decision rather than filling a gap.

Doing (c) properly means first moving those readers onto `texture_to_rgba8`, which is a wider
change than this feature scoped. Left for the maintainer's call; the defects that SHIP are (a)
and (b), and both are fixed.

Worth recording: the old rationale for `f1` — "the whole export path reads it as 8-bit" — is no
longer accurate. `texture_to_rgba8` is dtype-driven and tonemaps a float target. So the
constraint holding (c) back is the raw-byte READERS in the tests, not the export path.

**(a) measured rather than read off the diff**, on the branch Telegram and the shared shapes take
(`FitPolicy.RENDER_AT_TARGET`), against an `f2` output pass:

    before: export fit-branch canvas dtype = f1   (output pass is f2)
    after:  export fit-branch canvas dtype = f2

Gate green: 2716 passed, 4 skipped, 37.6s total (check 8.2s, test 26.0s, smoke 2.6s).
