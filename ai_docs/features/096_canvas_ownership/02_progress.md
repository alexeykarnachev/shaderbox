# 096 — Canvas ownership: the log

What actually happened, wave by wave, appended as each lands. A log, not a plan — the plan is
`01_spec.md` and the evidence is `00_findings.md`. **On resume, read this file and
`git log --oneline` first**: they are the truth about where the work stopped.

Each entry carries its done-condition (written BEFORE the wave started, as a checkable
statement), the verification result, what was ruled out and why, and any surprise worth the next
reader's time.

## Before anything: establish a real baseline

The investigation session ended unable to run the suite — every pytest process segfaulted at GL
context creation, including pure-logic modules, after six agents and dozens of probe scripts had
each built standalone contexts. `git diff HEAD -- shaderbox/ tests/` was empty throughout, so the
tree was provably unchanged from the last green gate.

So the first action of the executing session is `make gates` on the untouched tree:

    make gates > /tmp/g.log 2>&1; echo $?

- **exit 0** — good, proceed.
- **red with an empty `git diff HEAD -- shaderbox/ tests/`** — the box, not the repo. Do not debug
  it as a regression, and do not start a wave until the baseline is green, or every later result
  is unreadable.

**The state this spec was written in, so it is recognizable.** At the end of the investigation
session the box was in exactly this condition, and the shape is specific:

- `make gates` red at `test`, with `node down: Not properly terminated` and an xdist
  `INTERNALERROR ... KeyError: <WorkerController gwN>`, only tens of tests run out of ~2720;
- running serially segfaults at `tests/conftest.py`'s `app` fixture, which builds a glfw window;
- yet the three scripts in `repro/` run fine, because each builds ONE standalone context and no
  window.

That combination means the display/driver is out of resources for new GL windows, not that the
code is broken — verified by checking out an older commit and seeing the same failure, and by
`git diff HEAD` being empty. The repro scripts working while the suite dies is the tell.

**What to do:** try once more after a few minutes. If it persists, write that into this file and
STOP — it is the one legitimate early exit. A night spent on a tree whose gate cannot run
produces nothing anyone can trust, and the waves are all still here tomorrow.

## This is an unattended run

Nobody is watching. `01_spec.md` has a "Running this unattended" section and it is the contract:
carry on to the end, never ask, commit and push each wave, never weaken a test to get green, and
write down every choice made along the way. The one legitimate early exit is a red baseline gate
with an empty source diff — the box cannot run the suite, so nothing the night produced could be
trusted.

Append to this file BEFORE starting the next wave, not at the end. A log written only at the end
does not exist when it is needed.

## W-0 invariant checker — NOT STARTED

done-condition (written in advance): a test helper asserts, for every canvas reachable from a
Document, that the live canvas matches what its graph entry implies and that a history matches
its live canvas on size, dtype, filter and wrap; driven over `filter_linear=False`, `wrap=True`,
`dtype="f4"`, `scale=0.5`; and it FAILS on the untouched tree for F2 and for the
`resample_canvas` filter/wrap mutation. No file under `shaderbox/` is modified by this wave.

The failure requirement is the point: a W-0 that passes everywhere checks nothing. Prove it red
before writing a single line of W-1.
