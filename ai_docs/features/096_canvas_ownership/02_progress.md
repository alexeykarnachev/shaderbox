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

## Interlude: the gate could hang without limit

Not a wave — an obstacle hit while running W-2, fixed because it makes every later wave's
verdict trustworthy.

**The symptom.** A `make gates` run sat for 12 minutes and had to be killed. One xdist worker was
`<defunct>` and the other seven spun at ~2% CPU waiting for a node that would never report:
`[gw3] node down: Not properly terminated`. Nothing bounded it — the run would have waited
forever.

**It is a hang, not slowness.** Timed separately, the gate is check 8.2s + test 26.0s (2716
tests) + smoke 2.6s = 37.6s, and the slowest single test is 6.0s. An early guess that W-2's
dtype change had made the suite slower was wrong and is recorded here because it was offered
before it was measured.

**Not reproduced, and not claimed to be understood.** Ten full-suite runs came back clean at
~25s, including one under deliberate GL-context pressure (24 standalone contexts held by another
process). `test_gl_lifetime_guards.py` remains the documented suspect — its module `gl` fixture
`return`s instead of yielding, so its context lives to process exit, and its last test then adds
a `stale_default_context` plus a full `app` on top — but that module passes 13/13 when run alone,
repeatedly, so the trigger is still unnamed.

**What is fixed is the failure MODE, which is what cost the time.** Two halves, and neither works
without the other:

- `timeout = 60` / `timeout_method = "thread"` in `pyproject.toml` (pytest-timeout, new dev dep).
  Ten times the slowest real test, so it can only fire on a hang. Alone it is NOT enough: it
  turns the hang into a hard worker death, and xdist then waits on the dead node exactly as
  before — measured, a full-suite run with a planted hang still stalled past 300s.
- `--max-worker-restart=0` on `make test`. This is the half that makes a dead worker a FAILURE
  the controller reports immediately rather than a node it waits on.

**Broken on purpose before being believed**, per the rule that an unbroken gate is a wish. A
planted `time.sleep(600)` test:

    before          stalled past 300s, killed from outside
    timeout only    stalled past 300s -- `node down: Not properly terminated`
    both halves     `make gates` RED in 84s, the hung test named, 2714 others still reported

The clean tree is unchanged at 37.6s green.

## W-3 F2, F7, the sizing rule at five sites — LANDED. F10 had already dissolved.

done-condition (written in advance): `repro/f2_promote_scaled.py` prints `BUG: False`, a search
for `target_size(` in `shaderbox/document.py` finds one caller, and `make gates` is green.

All three met. `grep -c "target_size(" shaderbox/document.py` returns 1, and that one is inside
`canvas_size_for` itself.

**(a) The sizing function.** `Document.canvas_size_for(name)` — the document's own size for the
output, `entry.target.target_size(canvas_size)` for everyone else, `PassEntry()` for a name the
graph does not hold, and `output_pass` handled as the `str | None` it is. The four in-document
copies (both loops in `set_canvas_size`, `_seed_feedback`, `render`'s fix-up) call it. A dead
`output` local in `_seed_feedback` went with them.

**(b) The modal's fifth copy**, which was the WRONG one — it applied `scale` with no output
exemption, so an output pass carrying a stored scale displayed a size it did not have. It now
reads `canvas_size if is_output else target.target_size(canvas_size)`, reusing the model's own
method rather than re-deriving the arithmetic.

**(c) F2, promotion — and TWO more doors found while closing it.** `Document.set_output_pass`
sets the output and resizes it to full; `project_session.set_output_pass` routes through it. The
probe that assigns `graph.with_output(...)` directly still shows the old size, which is the fix
working as designed rather than a gap — but that prompted a search for other ways the output
changes, and two were real:

- **deleting the output pass** promotes an arbitrary survivor through `_graph_without`, which
  may carry a scale. Verified broken: `helper after: (128, 128)` under a 256x256 document,
  `BUG: True`.
- **an import** can hand the output role to a scaled pass (`plan.becomes_output`).

Both are the same defect by another door, so `conform_output_canvas` was split out of
`set_output_pass` — it resizes whatever is CURRENTLY the output — and both paths call it.
Re-verified through the delete path: `(256, 256)`, `BUG: False`. `load_from_dir` was checked and
needs nothing: every pass is born at full `canvas_size`, so a scaled output loads correct.

**(d) F7, the clamp.** `__init__` now applies `clamp_canvas_size` as `set_canvas_size` does, so
the comment calling them "the field's only two writers" with both normalizing is true.
`Document(canvas_size=(8, 8)).canvas_size` is `(16, 16)`.

**(e) F10 no longer exists — W-1 dissolved it.** The findings file names its root as F6's: "a
draw aimed at an external canvas leaves the pass's own canvas unwritten". W-1 made every
iteration draw into the pass's own canvas, so after a foreign render the live canvas holds the
NEWER frame and `newest_frame` names it correctly. Traced: `foreign: live=101 hist=76
newest=101`. An edit was drafted, measured to fix nothing, and reverted — no production change
here.

**One test's subject genuinely moved** (the spec's one legitimate edit). The F7 clamp broke four
tests in `test_feedback_persistence.py`, whose fixture document is `(8, 8)` — BELOW the 16px
floor. `set_canvas_size((8,8))` already returned `(16,16)` before this wave, so the fixture was
only stable because `__init__` skipped the clamp. Lifted to `(32, 32)` / `_SCALED = (16, 16)`.
A fifth test then corrupted the stored size to a literal `[16, 16]` to force a mismatch, which
had become the CORRECT size; it now doubles `_SCALED` so it disagrees whatever the fixture is.
Falsified afterwards: making it agree again turns that test red, so it still discriminates.
