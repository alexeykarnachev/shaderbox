# 095 — Structural sweep: the log

What actually happened, wave by wave, appended as each lands. A log, not a plan — the plan is
`01_spec.md`. **On resume, read this file and `git log --oneline` first**: they are the truth
about where the work stopped.

Each entry carries its done-condition (written before the wave started), the verification
result, what was ruled out and why, and any surprise worth the next reader's time.

## W-0 inventory — DONE (no code changed)

done-condition (written in advance): every kind of Python symbol enumerated across every
directory named in the spec's wave list, sorted into the SAFE / CAREFUL / RISKY tiers, with the
enumerating command recorded here so the next session can re-run it rather than trust the
result. No file in `shaderbox/`, `scripts/` or `tests/` modified by this wave.

verification: `make gates` green before and after (nothing changed).

**How to re-run the enumeration** — no dead-code tool is a project dependency, so it runs
through `uv run --with`:

    uv run --with vulture vulture shaderbox/ scripts/ \
      --exclude 'scripts/dogfood/runs/*,shaderbox/resources/editor/abi_probe.py' \
      --min-confidence 60

Re-run this rather than trusting the tiers below; the tool's raw output is mostly false
positives and the sorting is the work.

### The false-positive classes, named so they are not re-investigated

The tool cannot see any of these, and each accounts for a large share of its raw output. A
later wave that re-runs the scan should discard them in the same way:

- **Writes to third-party library objects.** Assignments to imgui style fields
  (`window_rounding`, `cell_padding`, `grab_min_size`, …), moderngl texture settings
  (`repeat_x`, `repeat_y`), and GL blend state configure the library; they are not this repo's
  symbols. This is the single largest class in the raw output.
- **Pydantic validators.** `_id_validator`, `_reset_out_of_range_values` and
  `_reject_unnamed_pass` are `@model_validator` methods, invoked by pydantic at
  validate time.
- **Python protocol hooks.** `__dir__` and `__getattr__` in `scripts/dogfood/__init__.py`.
- **The editor FFI binding surface.** `shaderbox/editor/ffi.py` is a binding to the vendored
  editor: its methods are the product, and an unused one is an unbound capability rather than
  dead code. Every candidate the tool reported there falls under this.
- **Symbols exercised only by tests.** `get_current_session`, `sealed_ids`, `build_messages`
  and `graph_errors` are each called from the suite. Live surface.

### SAFE — unreferenced internal symbols

- `shaderbox/app.py::delete_current_document` — a one-line wrapper around `delete_document`
  with no caller. The command table routes `DELETE_DOCUMENT` to
  `delete_current_document_confirmed` instead, and `tests/test_menus.py` names this wrapper as
  the falsifier (the thing that would be wrong to call), which is what identifies it as the
  superseded half of a pair rather than an unused entry point.

### CAREFUL — assigned but never read; confirmed by search before removal

- `shaderbox/theme.py::COLOR.ACCENT_ALPHA` — declared as a token and assigned by `set_accent`,
  read nowhere. Confirm with a search that excludes its own declaration and assignment lines;
  the accent system's other two tokens (`ACCENT_PRIMARY`, `ACCENT_ACTIVE`) are read normally,
  so this is the one member of that trio with no consumer.
- `shaderbox/media.py::_frame_period` — computed in `__init__` from `fps`, never read. The
  neighboring `_fps` and `_n_frames` are read; this one is not.

### RISKY — leave, or ask

- `shaderbox/ui_primitives.py::segmented_choice` — see the spec's do-not-change section. The
  module is an enumerable UI vocabulary with a test walking `vars(ui_primitives)`, so removing
  an unreferenced widget is a judgement about the vocabulary, not a cleanup.
- `scripts/dogfood/judge.py` — its whole public surface reports as unused. The module docstring
  states its contract ("NUMBERS OUT, never a verdict") and it is the measurement toolkit for a
  maintainer-run workflow, consumed ad hoc rather than imported. A product, not dead code.

### Coverage

scanned: functions, methods, properties, attributes and module-level variables across
`shaderbox/` (every subpackage) and `scripts/`, via the command above plus per-candidate
reference searches; each candidate reported below was confirmed by its own search rather than
by the tool's confidence score.

**This claim was falsified once and the miss is recorded below** (the three dead `SIZE` tokens in
`theme.py`): the class-attribute tokens inside `theme.py`'s size bag were reported by the command
above and never triaged, so "module-level variables" over-stated what the triage actually
covered. The tokens were deleted in the follow-up; the lesson is that a coverage line is a claim
to be attacked, not a summary to be trusted.

not scanned: enum members, type aliases and whole-module deadness — the tool does not report
them and they were not enumerated separately; `tests/` as a target (scanned only as a
reference source, so a dead test helper would not appear); `shaderbox/resources/`; non-Python
assets. A later wave wanting those must enumerate them from the language's constructs.

## W-R rot removal — DONE

done-condition (written in advance): no doc in the harness states a fact an unrelated commit can
silently falsify; each one found either deleted or replaced by the command that produces it; the
stale `todo.md` pointer resolved; `make gates` green.

verification: green (check, test, smoke).

Two items, both in `ai_docs/dev_flow.md`. The pyright status line asserted a current error count;
the gate already enforces it, so the doc now states the mechanism instead of the state. The
shader-library entry described seeding "until the load mechanism lands", pointing at `todo.md` —
the mechanism landed and `todo.md` has drained, so the entry now names `shader_lib/seed.py`.
Checked before writing it: `sync_shipped_lib` is imported by `app.py` and runs before the first
lib index builds.

**ruled out, do not re-raise:**
- Three "currently / at the moment" hits in `dev_flow.md` are ordinary prose ("at the moment the
  information is lost", "at the moment you author", "as it currently is"), not status claims.
- The 1343-glyph count in `conventions.md` is anchored to a commit ("As of `e7db554`") and
  describes a baked artifact. Frozen history, stays.
- Code comments carry no live facts. Searched for current-state phrasing and for count-shaped
  comments across `shaderbox/`, `tests/` and `scripts/`; nothing. The repo's own comment
  discipline is holding, so this wave had no code half.

surprise: the wave was far smaller than the spec's survey implied. The presence scan reported
live facts as a live category, which is true, but the harness turned out to carry two rather
than a class worth sweeping — the feature specs' numbers are nearly all correctly frozen
before/after measurements.

## W-F the gl_ctx fixture — DONE

done-condition (written in advance): `gl_ctx` defined once; the EGL comment preserved verbatim in
its new home; no module's `xdist_group` changed (**this clause was vacuous** — verified afterwards
that none of the five modules ever carried a marker, so it could not have failed; a done-condition
that cannot fail is not one); `grep -rln "poisons the process's EGL display"
tests/` returns only the shared definition plus any deliberate variant; `make gates` green.

verification: green (check, test, smoke).

Five of the six copies were byte-identical or differed only in an extra comment; they now use the
shared module-scoped fixture in `tests/conftest.py`. The comment recording the EGL segfault moved
with it, expanded into a docstring that also states why the scope is per-module (the modules run
in their own xdist processes, so a session scope would share a context across files partitioned
apart on purpose). Ruff removed the imports the deletions orphaned.

**`tests/test_profiling.py` keeps its own copy, deliberately.** Its fixture binds a
`simple_framebuffer((512, 512))` after creating the context, which the shared one does not. The
module measures GPU timing spans and its `_burn` helper clears that framebuffer as the workload it
means to time, so the surface it draws against is part of what it tests. Caught by hashing each
fixture body rather than reading them — the "similar shape is not shared meaning" case.

Being exact about the evidence: collapsing it does NOT turn the module red. An adversarial review
falsified that — it deleted the private fixture, let the module fall through to the shared one, and
the tests passed alone and in a three-module run, because `gl.clear()` against a standalone
context's zero-size default framebuffer silently no-ops rather than erroring. So the carve-out
rests on what the module is timing, not on a failure the suite can show. A later session must not
read the green as permission to collapse it.

**falsifier attempted, and it did not fire.** The comment says an explicit `backend="egl"` context
poisons the process's EGL display and segfaults the next module. Reintroducing that backend and
running three GL modules in one process passed here. That does NOT disprove the comment — it
states the failure is module-order-dependent and it was recorded on a display-less box, while this
machine has a real display. The comment stays as written; a later session should not read this
note as license to weaken it. What IS verified: the consolidated modules still run together in one
process green, which is the scenario the fixture exists to survive.

## W-D deletion — DONE

done-condition (written in advance): each tier removed as its own batch with `make gates` green
between batches; every CAREFUL candidate proved dead by search before removal; nothing removed
from a documented extension contract.

verification: green after the SAFE batch, green after the CAREFUL batch.

**Removed (SAFE):** `App.delete_current_document`, a one-line wrapper with no caller — the
command routes to the confirming variant instead.

**Removed (CAREFUL):** `Video._frame_period` in `media.py`, computed in `__init__` and never
read. Proved dead first: no dynamic access, no field iteration, and its siblings `_fps` /
`_n_frames` are read normally, so the class does use the neighbors it keeps.

**`COLOR.ACCENT_ALPHA` was reclassified CAREFUL -> RISKY and NOT removed.** The search says
nothing reads it, and that is true but misleading: it is one of three tokens in the accent
system, `_ACCENTS` stores them as triples, `set_accent` unpacks all three for runtime accent
swapping, and `theme.py`'s module docstring documents `ACCENT_*` as the swappable role group with
"Adding a theme = new `_P` + `_ACCENTS` + role mapping". It is a documented extension contract,
so removing it means editing every accent preset and breaking the trio's symmetry — a large diff
from a small finding, which is the signal for solving the wrong problem. Left alone. Do not
re-raise it as dead code; a future reader should decide it as a theme-API question.

## W-C comments — DONE (one line removed; the rest were false positives)

done-condition (written in advance): every restating comment and verbatim duplicate removed, every
attempt-narration compressed to its live reason with each measurement and named failure kept, and
no comment touched that states why. `make gates` green.

verification: green.

**Removed:** one line, `# Process hotkeys` above `process_hotkeys(app)` in `ui.py`. Every other
divider in that frame function carries a real explanation; this one restated its call.

**The search that found it**, worth re-running rather than trusting a reading: match a short
comment whose words are a subset of the identifiers on the next line. It returned seven
candidates across `shaderbox/`, six of which were section dividers.

**ruled out, do not re-raise:**
- The `constants.py` labels (`# File extensions`, `# Default video settings`, …) and the
  equivalents in `theme.py` and `commands.py` are section dividers grouping a block in a long flat
  table. They navigate, they do not restate. Deleting them makes those files harder to scan.
- The imgui-font-atlas comments above each `pytestmark` are NOT verbatim duplicates: two
  phrasings, one or two lines, each directly above the `xdist_group` it explains. A local
  one-line rationale beats a pointer to a canonical copy for a reader who opens one file.
- Both attempt-narration comments STAY, in full. `test_script_engine.py` names the wrong rule that
  was tried ("asked whether the key was REWRITTEN this tick") and the symptom it produced (a row
  oscillating frame by frame), which is what makes the falsifier on the next line meaningful.
  `test_motion_verdict.py` names the failure the same way ("used to double the first line's
  indent", so the copilot wrote a syntax error onto the rescue path) and explains why the test
  parses instead of pattern-matching. These are named failures, not history.

surprise: the comment wave was the one the spec was most cautious about and it produced a
one-line diff. The presence scan's three comment categories were all over-called — the repo's
comment convention is working, and the measurable restatement rate across `shaderbox/` is
effectively zero once section dividers are excluded.

## The enum-member gap, probed after the waves

W-0's coverage line declared enum members unscanned. A follow-up scan tried to close it and
could not, which is the useful result: a naive "is `Cls.MEMBER` referenced anywhere" walk over
`shaderbox/` reports dozens of members as unreferenced, and the ones checked were all aliased or
binding surface — `commands.py` does `C = CommandCategory` and then writes `C.FILE`, and the
`editor/ffi.py` enums are the vendored editor's binding surface, where an unused member is an
unbound capability.

So the gap stands, and a later wave wanting to close it needs a resolver that follows aliases
rather than a text scan. Recorded so the next session does not repeat the naive version and
report its output as an inventory.

## Adversarial review — one defect, fixed

An opus review was run against the landed diff, anchored to the code and to its own test runs
rather than to this file or the spec. It ran `make gates` itself (green, smoke passed rather than
skipped), re-ran the inventory command, and probed dynamic reach independently.

**Sound:** all three deletions are unreachable — no `__getattr__`/`__getattribute__` exists
anywhere in `shaderbox/`, the copilot tool surface is a static name-to-handler list behind a
protocol, and the scripting engine installs a custom `__import__` that refuses every `shaderbox`
path except `shaderbox.scripting`, so a user script cannot reach `App` at all. The fixture
consolidation preserves one context per module (`--setup-plan` shows one setup per module, not a
shared one), and dropping the per-fixture MESA `setdefault` calls is safe because `conftest.py`
sets both at import, before any test module is collected.

**Defect found — a coverage gap, since fixed.** Three module-level tokens in `theme.py`
(`CANVAS_FIELD_W`, `ASPECT_FIELD_W`, `CANVAS_PRESETS_W`) are reported by the recorded inventory
command, have no Python reader and no dynamic reader, and appeared in no tier. Their readers in
`tabs/document.py` were removed by an earlier feature. Deleted, with the comments that described
only them; `make gates` green after.

They were kept out of a RISKY "094 may re-consume them" row on purpose: 094 has a mock and no
spec, and a feature sizes its own tokens when it lands. Dead constants held against a speculative
consumer are the inverse of a deleted abstraction that was organizing something.

**A correction to this log, not to the code:** the `test_profiling` carve-out note claimed
collapsing that fixture "would have changed what that module runs against". The review falsified
that and the entry above now states the real reason. The carve-out stands; its stated evidence
was wrong.

**False trails, do not re-litigate:** dynamic reach to `App` methods; an `xdist_group` regression
(none of the five consolidated modules ever carried a marker, so that done-condition was
vacuously satisfiable); the MESA override removal; `ACCENT_ALPHA` as an undeclared miss (it was
triaged and deliberately kept); and the whole set of imgui-style, moderngl-attribute and
`model_validator` candidates.

## Closing two of W-0's declared gaps

W-0 listed `tests/` as a target and whole-module deadness among the things it did not scan. Both
were closed after the first review, using AST walks rather than the tool:

- **Whole-module deadness: none.** Every `.py` under `shaderbox/` is named somewhere outside
  itself.
- **Dead private test helpers: two, removed.** `_census` and `_select_row_for_test` in
  `tests/test_project_management.py`, each defined once and called nowhere. The second was also an
  inline-import wrapper around `popups.projects::_select_row`, a shape the repo's own code rules
  forbid outside the two sanctioned lazy seams.
- **The whole theme-token class, re-checked.** After the three-token fix, an AST walk over every
  class-attribute token in `theme.py` finds no member with fewer than two references, and the same
  walk over every module-level constant in `shaderbox/` returns one hit, `editor/ffi.py`'s
  `PRIM_STRIDE` — the byte size of the `Prim` struct, sitting beside it as part of the FFI ABI, so
  binding surface rather than dead code.

What remains unscanned from the original list: enum members (see the note above on why a naive
walk cannot close it), type aliases, `shaderbox/resources/`, and non-Python assets.

## Review round two — one defect, plus a pre-existing contradiction the move exposed

A second opus review ran against the patched state, anchored to its own AST walks and test runs.

**Defect it found, and fixed: a falsifier pointing at a symbol W-D had deleted.**
`tests/test_menus.py`'s guard on the menu bar's destructive verb told a later maintainer to point
`DELETE_DOCUMENT`'s callback back at `App.delete_current_document`. That wrapper was dead as CODE
— its only non-definition reference in the repo was this prose — but the prose was an instruction,
so following it would raise `AttributeError` instead of producing the unconfirmed delete it
promises. The docstring now names the wrapper's body, `delete_document(self.current_document_id)`;
the test body is unchanged. Verified here rather than taken on trust: repointing the callback that
way fails with "the bar's Delete document trashed with no confirm", and restoring it goes green.

**The lesson worth carrying: a symbol with no caller can still be load-bearing as a NAME.** A
deletion wave should search prose for the symbol it removes, not only code — a falsifier, a
comment or a doc that names it is a reference the compiler cannot see.

**A pre-existing contradiction the fixture move exposed, now corrected.** Three GL test modules
opened with "on the display-less dev box use the EGL backend + the MESA version overrides". That
text predates this sweep (`git show e24ca4f~1` has it verbatim) and was already wrong — the
fixture beneath it used the default backend. With the fixture now in `conftest.py`, whose
docstring explains that an explicit EGL context poisons the process's EGL display, a reader
following those module docstrings would do the thing that causes the segfault. They now point at
the shared fixture instead.

**False trails from round two, do not re-litigate:** all 145 theme-bag members (the sixteen with
no external reader each have an internal one — the `SYN_*` colors feed the kind tables, the size
tokens feed `apply_theme`); `resources/__init__.py` and `abi_probe.py` as apparent orphans (the
first is reached through `importlib.resources`, the second is vendored and parsed by a test);
`_NAV_ONLY_FOCUSABLE` in `test_region_system_is_gone.py` (deliberate, self-documenting,
pre-existing); and the `make test` MESA paragraph in `dev_flow.md`, still accurate.
