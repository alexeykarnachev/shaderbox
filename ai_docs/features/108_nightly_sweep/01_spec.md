# 108 — Nightly sweep: the runtime layer

## Status

The presence scan is DONE; what it found is in `progress.md`. No wave has landed.
Each wave appends to that file as it completes; on resume, read it and `git log` before
this spec — they are the truth about where the work stopped, where this is only the plan.

## Goal

**107 swept the gate layer and closed it.** Its four mutation slices broke roughly 190
things and found the checks strong; its structural questions — misfiling, dependency
direction, missing seams, comment history-narration — all came back ABSENT; its live-fact
wave closed empty because every flagged figure turned out to be frozen history.

So this sweep does not re-ask those questions. A category swept twice returns almost
nothing the third time, and this scan confirmed it: **dead symbols, duplicated blocks,
state-reset leaks, oversized files and test-suite padding all came back ABSENT or
LEAVE-WHOLE**, each with the evidence recorded below so a later session does not re-run them.

The yield sits in the layer neither sweep has looked at: **what the code does at runtime
when something goes wrong.** Three findings carry the night, and they share a shape —
*a failure that resolves to a plausible-looking answer instead of announcing itself.*

> **A miss resolves to something that looks like a result: a GL object dropped rather than
> released, a failed user action reported only to a log, a malformed key answered with a
> parent's value.**

One finding sits outside that shape and is worth more than its size, because it is the
one class of defect that reproduces itself: **a committed skill instructs a session to do
what the project's rules forbid.**

## What is present

Illustrative, ONE example each, verified by the main session against the artifact. **This
is the survey, not the inventory** — W-0 produces that, and the numbers here are not the
work order.

### Survey (measurement — re-measure it, do not trust it)

- **A GL object dropped without release.** `core.py::_upload_instances` sets `self.vao = None`
  when an instance buffer outgrows its reserve, while the two other sites that retire a
  live VAO (`invalidate`, `compile`) both call `.release()` first. The user path is an
  instanced pass whose population grows past its doubled reserve — ordinary during
  particle-count tuning.
- **A failed user action that only reaches a log.** `ui_primitives.py::draw_copyable_text`
  returns `False` when the clipboard has no backend, and its docstring says the caller
  decides whether to surface it. Two callers do; `popups/lib_picker/preview.py` discards
  the return, so a click to copy a path does nothing visible on a box without `xclip`.
- **A rule spelled twice with inverted comparisons.** "A document keeps at least one pass"
  is `len(document.passes) > 1` at `widgets/pass_list.py` (disabling the menu item) and
  `len(document.passes) == 1` at `project_session.py` (refusing the call). They agree
  today; the UI half's own comment leans on the disabled state meaning unreachable.
- **A skill instructing a forbidden act.** `.claude/skills/sanitize/SKILL.md` tells a
  session to add a `todo.md` entry, and repeats it in its report-table template, while
  `CLAUDE.md`, `dev_flow.md` and `todo.md`'s own header each say no new entries, ever.
  `todo.md` is currently empty, so following the skill reopens a file drained to zero.
- **A roadmap row claiming deferred work that shipped.** Row 020's `partial` clause lists
  `delete_lib_file` and `bind_media` as parked; both are live, gated copilot tools. Two
  items in the same clause (`undo_edit`, semantic editing) are genuinely absent.

### Proposal (argument — attack this hardest)

The survey above is measurement and gets re-measured. What follows is reasoning and is
where this spec can be wrong by thinking alone.

**The claim:** these findings are one class, not five unrelated defects, and the class is
*a failure that produces a well-formed answer.* A released VAO and an unreleased one look
identical at the next frame; a copy that silently did nothing looks like a copy that
worked; a capture name with a trailing space resolves to its parent's colour, which is a
real colour. Each is invisible precisely because the failure path returns something of the
right *shape*.

**Why that matters for the fix:** the remedy for this class is never "handle the error" —
it is *make the wrong answer impossible to represent*. The theme resolver's five rounds
are the evidence: four attempts handled a malformed key better, and the fifth made a
malformed key raise. The repo's own gate rule says the same thing one level up — remove
the duplicate if you can, gate it if you cannot, name the gap if you can do neither.

**Where this argument is weakest, stated so a reviewer can attack it:** the
delete-pass duplication and the skill contradiction do not really fit the class — the
first is ordinary drift risk with both halves currently correct, and the second is a
documentation defect, not a runtime one. They are in this sweep because they are cheap and
real, not because the class explains them. A wave that stretches the class to cover them
is over-fitting, and the honest framing is "three of five".

## The constraints, verified here rather than assumed

- **Verification is `make gates`** (check → test → smoke, stopping at the first failure,
  one exit code). Baseline run for this sweep: exit 0, smoke RAN rather than skipping.
  The target already warns on a non-tty stdout that a pipe reports the PIPE's status.
- **`make gates` autofixes and then reports failure** — pre-commit's formatter modifies
  files and exits non-zero for having done so, so a first run is "something changed" and
  only a second run is a verdict. Commit, then re-run.
- **A mutation run needs `PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider`.** A stale
  `__pycache__` makes a caught mutation look like a vacuous gate. `make test` sets this;
  a bare `uv run pytest` does not.
- **The package is installed editable, so a git worktree is NOT an isolation escape** —
  `uv run` inside a worktree imports the MAIN checkout. A mutation goes to a copy and its
  restore is verified in the one tree.
- **Never run two mutating agents against one working tree.** 107 recorded this failure
  twice in one night, the second time at four times the scale: a concurrent reader cannot
  tell a sibling's probe from a defect. Mutating work in this sweep is SERIAL.
- **Gates naming a source path** are found with `grep -rn 'Path("shaderbox/' tests/`.
  `tests/test_project_management.py` AST-walks `shaderbox/app.py` by literal path.
- **`vulture` is NOT installed** and pyright's config does not enable unused-symbol
  reporting; `pytest-cov` is not a dependency. A dead-code question is grep-and-reference
  based, and `ruff F401,F841` is clean.

## The waves

One wave, one commit, `make gates` green at each. Ordered so measurement precedes change,
and so the waves whose correctness an oracle decides run before the ones judgement decides.

- **W-0 — Inventory. NO CHANGES.** Enumerate the three runtime classes across the whole
  package, by RUNNING rather than reading:
  - every `ctx.<alloc>` site paired against its release, on the path where the object is
    REPLACED rather than where it is first made;
  - every helper that returns a failure sentinel, paired against each caller, asking
    whether the caller distinguishes it from success;
  - every string-keyed resolver, probed with the inputs that defeated the theme resolver:
    trailing space, trailing newline, doubled separator, empty string, leading separator, a
    name that does not exist, and a name that is a PREFIX of a real one.

  This wave sizes the rest, so nothing after it can be scoped first.
- **W-1 — The dropped GL object.** Release what is dropped, and gate it by the break that
  is currently silent. The existing `tests/test_gl_lifetime_guards.py` is the home.
- **W-2 — The silent failure.** Give each discarded failure sentinel a surface, through the
  notification seam the project already has. A sentinel whose caller provably cannot act on
  it is left alone and SAID SO — a wave that surfaces everything is noise, and a gate that
  cries wolf gets suppressed.
- **W-3 — The malformed key.** For each resolver W-0 found answering a malformed key with a
  plausible value, make the malformed key raise. Where raising is wrong because the caller
  legitimately passes unvalidated user input, the resolver returns an explicit
  not-found that the caller must handle, and the gate pins that it does.
- **W-4 — The skill that instructs a violation.** Correct `sanitize`'s two clauses to the
  drain-only framing every other doc uses. This is the only wave whose defect reproduces
  itself on every invocation, so it lands early despite being small.
- **W-5 — The stale roadmap claim.** Row 020's `partial` clause names two shipped tools as
  deferred. Correct the clause to the two items genuinely absent. **A status corrected, not
  a status deleted** — the row is frozen history about what a feature did, and the
  live-fact rule does not reach it.
- **W-6 — The repeated rule.** Give "a document keeps at least one pass" one spelling that
  both the menu item and the guard ask. Behaviour-preserving by construction; if a test
  expectation changes, the refactor is wrong and gets reverted.

## How correctness is decided

`make gates`, judged by its exit code captured unpiped:
`make gates > /tmp/g.log 2>&1; echo $?`. A skipped smoke reports as **skipped**, which is
not a pass.

**A changed test expectation in a structural wave is a defect in the refactor, not a test
to update.** The tests are the only evidence behaviour was preserved; weakening one to get
a green destroys what the green meant. The single legitimate edit is a test file moving
with the code it covers.

**A new or edited gate is done when the thing it guards has been broken, the gate has named
it, and the break has been restored — and the commit says which break was tried.** A gate
is the one kind of code whose correctness a passing run cannot show.

## Cold start — for the session that executes this

**Measure first.** Every specific in the survey above is an example, not a work order. Run
W-0 and work from what it returns; if W-0's inventory and this spec disagree, the inventory
wins and the spec is the thing that was wrong.

Work in wave order. W-4 and W-5 are doc-only and can land first to get them out of the way;
W-1 through W-3 are the substance. Append to `progress.md` after each wave, BEFORE starting
the next — a file written only at the end does not exist when it is needed. Write each
wave's done-condition as a checkable statement before starting it.

**Settled, and NOT to be re-opened as a question by any review agent:**

- **The three big files stay whole.** `copilot/backend.py`, `app.py` and `ui_primitives.py`
  were each examined for a split and each declined, on evidence rather than taste:
  `conventions.md` records a maintainer decision that `app.py` is not to be split further
  without a fresh pain signal; `tests/test_ui_kit_is_importable_alone.py` actively enforces
  that `ui_primitives.py` is ONE flat module with a minimal import closure, having already
  been trimmed from nine; and `backend.py`'s section comments are topic labels over one
  class sharing one injected-callback bundle and one brake-state machine, so following them
  would fracture shared state rather than isolate it. **A wave proposing any of these three
  splits has to first explain how the gate and the recorded decision are wrong.**
- **The test suite's weight is earned.** Its size comes from breadth across shipped
  subsystems, not from padding. No area is retired in this sweep.
- **`todo.md` stays empty.** Nothing in this sweep files an entry there. A defect found is
  fixed in this wave, or its knowledge goes to a spec or `conventions.md`.
- **No migration or backward-compatibility code.** If a change reshapes an on-disk format,
  the `projects/dev/` files are fixed BY HAND and committed in the same wave.

## The coverage claim

One line per scan. A scan with no such line is treated as not having run.

- `scanned: except-handler bodies and the notification seam across shaderbox/ (all subpackages); not scanned: the narrower non-Exception handlers individually, and threaded exporter queue logic beyond the except line.`
- `scanned: ctx.texture/framebuffer/program/buffer/vertex_array/renderbuffer allocation and reassignment sites across core.py, document.py, media.py, graph_canvas/render.py, editor/render.py, channel_blit.py, project_session.py, app.py; not scanned: exporters/, copilot/, tabs/, graph_canvas/ffi.py, tests/.`
- `scanned: private-attribute chains, cross-module underscore use, and moderngl/imgui/glfw imports in the seven declared-pure modules across tabs/, widgets/, popups/, ui.py; not scanned: exporters/, editor/, copilot/, and shader_lib internals beyond parser.py.`
- `scanned: dict-typed keyed stores and their pop/clear/reassign sites against the document-delete, project-switch, document-switch and pass-delete teardown chains across app.py, project_session.py, ui_models.py, document.py, tabs/code.py, render_plan.py; not scanned: copilot/ internal keyed state and editor/ FFI caches.`
- `scanned: sizes and limits, eligibility predicates, reserved-name sets, path construction and pydantic defaults across all shaderbox/ modules and their test call sites; not scanned: dogfood/, scripts/, and non-Python resources.`
- `scanned: enum and Literal definitions, if/elif dispatch over six enums, a sample of try/except sites, and two default-kwarg call-site audits across shaderbox/ and its subpackages; not scanned: the remaining is-None guard sites, most default kwargs, and the ffi IntEnum dispatch families.`
- `scanned: the six committed project skills in full against CLAUDE.md, dev_flow.md's documentation-discipline and closing-out sections, conventions.md's code rules, and todo.md's frozen header; not scanned: conventions.md's design-decision and known-quirk bullets walked one by one against each skill.`
- `scanned: every roadmap row marked partial, against the code that would show its remaining item present or absent; not scanned: the per-slice specs under 020 beyond a grep for the four named items.`
- `scanned: commit bodies for the last 120 commits, grouped by defect shape; not scanned: nothing in that range — all 120 bodies read.`

## What looks wrong and is correct

Anything below that a wave proposes to "clean up" has to explain how the thing it exists
for still works.

- **`document.py::resample_canvas` allocates before releasing.** The old canvas must stay
  readable during the blit, so release-first would be the bug. Its docstring says so.
- **`core.py::compile`'s explicit buffer release is redundant and stays.** Dropping the last
  Python reference would also free the names, measured at the same live count over twenty
  recompiles. The comment says the release stays because it does not depend on when
  collection runs, and that its twin in `invalidate` is the one that genuinely leaks.
- **`except ... : continue` inside enumeration loops** over uniforms, files or documents is
  skipping one bad item to keep processing the rest, not hiding an error.
- **`uniform_coerce.py` and `scripting/` naming `moderngl` types** is not a layering breach.
  GL-free means needing no live context, not never naming a GL type; the Module map's own
  description of `uniform_coerce.py` names `moderngl.Uniform`.
- **`app.shader_lib_files.*` reached directly from the lib picker** is sanctioned explicitly
  by the Module map — there is deliberately no `App` forwarding facade.
- **`commands.py` importing imgui** is sanctioned by its own Module map entry.
- **`copilot/backend.py`'s `len(document.passes) < 2`** is not a third spelling of the
  delete-pass rule: it chooses the compact single-pass prompt view over the graph listing.
- **`theme.py::apply_theme`'s `accent` parameter** has one caller that never passes it, and
  three of its four `AccentName` alternatives are unconstructible from any UI. Its docstring
  calls it a runtime accent swap. **This is a seam with no caller, which is a design question
  and is NOT decided unattended** — a wave may report it, and may not delete it.
- **`ui_models.py::save`'s orphan-sweep `keep` flag**, carried over from 107: no observable
  effect under any fixture, because an earlier per-uniform unbind resolves every case.
  Whether it is load-bearing is a design question and stays open deliberately.
