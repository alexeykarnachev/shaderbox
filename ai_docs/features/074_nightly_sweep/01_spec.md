# 074 — the nightly sweep: structural pass over the repo

An unattended pass over the repo's SHAPE — dead code, duplicated case-lists, misfiling,
dependency direction, file size, comments — not over its behaviour. The survey below records
which KINDS of problem are present, with one example each. **It is not the work order.** The
first wave enumerates the real inventory and that is what the later waves act on.

## Status

Updated as each wave lands, so a session resuming after a crash learns from this file how far
the night got. Nothing has landed yet.

| Wave | State | Commit |
|---|---|---|
| W-0 inventory | not started | — |
| W-1 dead code | not started | — |
| W-2 case-list duplication | not started | — |
| W-3 comment duplicates | not started | — |
| W-4 ui.py draw code (recommend first) | not started | — |
| W-5 sanitize | not started | — |

## Goal

Structural only; there is no behaviour track (see § How correctness is decided for why). After
this sweep: no symbol survives that nothing reaches; a list of cases is enumerated in one place
rather than re-spelled beside its own registry; no comment paragraph appears twice; and the
question of whether the frame-loop orchestrator should still hold canvas-drawing code has been
answered with a recommendation rather than assumed either way.

## What is present

**Illustrative, not exhaustive. One example per category, found by a presence scan.** Every
count and every named symbol here is a SEED, not the inventory. W-0 replaces this section's
specifics with a measured list; if W-0 finds only what is named here, W-0 was run wrong.

### Survey (measurement — re-measure, do not trust)

- **Dead code: PRESENT, sparse.** Kinds found: dead module-level constants. Example:
  `DEFAULT_IMAGE_FILE_PATH` in `shaderbox/constants.py`, whose two siblings
  (`DEFAULT_VS_FILE_PATH`, `DEFAULT_FS_FILE_PATH`) are both live. Ruled out by the scan as NOT
  dead, each checked individually: pydantic `@model_validator` hooks, pytest fixtures, the
  dogfood harness's interactive API, imgui style attributes read dynamically. `ruff check
  --select F401,F811,F841 .` passes clean, so unused imports and locals are already absent.
  - `scanned: module-level functions, methods, classes, pydantic fields, module constants,
    private helpers, whole-module import graph across shaderbox/ and its subpackages, scripts/,
    tests/; not scanned: enum members individually, type aliases individually, projects/.`
- **Duplicated case-lists: PRESENT.** The higher-value duplication shape (the same set of cases
  written in two places). Example: `ToolDefinition.is_edit` is the canonical flag, read through
  `ToolRegistry.is_edit_tool`, yet `shaderbox/copilot/agent.py` re-spells the same universe as
  literal frozensets `_SCRIPT_EDIT_TOOLS` and `_WRITE_TOOLS`. A new edit tool registered with
  `is_edit=True` is picked up by the registry everywhere and silently missed by those two sets.
  Verified at the source, not relayed. Note `_RENDER_AUTHORING_TOOLS` in the same file is NOT
  this defect: it answers a different question and is correctly its own list.
  - `scanned: enum definitions and their usages, the copilot tool registry against hand-written
    tool-name sets, shape/preset case tables, across shaderbox/, scripts/, tests/; not scanned:
    general repeated-block duplication outside the case-table shape.`
- **Duplicated comment text: PRESENT, rare.** Two instances. Example: the same two-line comment
  about `root / rel` escaping the root appears twice in `shaderbox/shader_lib/seed.py`, above two
  near-identical unlink loops — a copy-paste that carried its comment along. Verified directly.
  - `scanned: every .py under shaderbox/ and subpackages, comment-line grep plus a full read of
    all 4+-line comment blocks, plus an exact-text duplicate detector; not scanned: tests/,
    scripts/, non-.py files.`
- **Comments restating the code: PRESENT but rare** (one clear instance found in a full read of
  the package's multi-line comment blocks). **Comments narrating a sequence of attempts:
  ABSENT** — none found; the repo's own ban on the bug-we-hit story is holding in practice.
- **Misfiling: ABSENT.** No symbol found whose subject conflicts with its file. Many modules
  carry an explicit "leaf module, imports only X" docstring a reviewer can check mechanically.
- **Dependency direction: the layered core is CLEAN.** `theme.py` imports nothing from the
  package; `ui_primitives.py` imports only `theme`; `core.py`, `pass_graph.py`, `document.py`
  reference the UI only inside comments. Verified directly. One contestable finding at a higher
  boundary, recorded as a proposal below rather than as a measurement.
- **File sizes.** Largest by line count at the commit below: `copilot/backend.py`, then the
  vendored `resources/editor/abi_probe.py`, then `app.py`; `ui_primitives.py` and
  `exporters/telegram.py` follow. `app.py` carries four section banners; `backend.py` has none.
  `abi_probe.py` is mostly a declarative signature table, not logic.
  - `scanned: line counts of every .py across shaderbox/, scripts/, tests/, and the internal
    section structure of the three largest; not scanned: structure of anything outside those.`

### Proposal (argument — attack this hardest in review)

- **`ui.py` holds draw code the documented layering assigns one layer down.** `conventions.md`
  calls `ui.py` a thin orchestrator owning `run` + `update_and_draw`, with `widgets`/`popups`/
  `tabs` as the pure draw functions. In fact `ui.py` defines `_draw_document_image`,
  `_draw_menu_bar`, `_draw_copilot_bar`, `_draw_splitter`, `_draw_canvas_backdrop`,
  `_draw_app_panel` and `_draw_document_settings`; `_draw_document_image` alone does layout
  math, issues imgui draw calls, hit-tests the canvas and mutates `app.script_mouse`. A reader
  following the documented layering would look in `tabs/document.py` — which exists — and miss
  it. This is a layer-collapse of RESPONSIBILITY, not an illegal import; `ui.py` correctly sits
  above the draw layer. **It is a recommendation, not a settled plan** — see W-4, and see the
  standing constraint against splitting without a pain signal.

## Constraints verified in this session

Each of these was run or read directly, not inferred:

- **Moving a module is mechanically safe.** `shaderbox/__init__.py` is empty, so there is no
  package re-export surface: every module is imported by its own path, and a move rewrites
  importers and nothing else. The one `from shaderbox import render_job` imports a MODULE, not
  an attribute, so it behaves the same way.
- **`make gates` reports failure honestly.** Nothing inside it is piped; each stage redirects to
  a log and its status is read before anything touches that log; a skipped smoke is reported as
  skipped rather than scored as passed. It already prints a warning when stdout is not a
  terminal, telling the caller that a pipe's `$?` is the pipe's. No first-wave change is needed
  to make the gate honest — it already is.
- **Gates and tests that name specific file paths.** These break the moment a file moves and
  must be updated in the SAME commit as the move, then re-verified by breaking what they guard:
  `.pre-commit-config.yaml` (excludes `shaderbox/resources/editor/abi_probe.py`), `pyproject.toml`
  (ruff `extend-exclude` and pyright `exclude`, same file), and in tests —
  `test_ui_prose_budget.py` (the largest concentration: a dict keyed on literal
  `("shaderbox/….py", "function")` pairs), `test_prose_spelling.py`, `test_worker_daemon_contract.py`
  (asserts `shaderbox/copilot/session.py` and `shaderbox/exporters/worker.py` appear),
  `test_region_system_is_gone.py`, `test_document_dir_layout.py`, `test_keymap_disjoint.py`,
  `test_lib_index.py`, and several document-example paths in `test_default_wiring.py`,
  `test_lazy_compile.py`, `test_raw_texture_round_trip.py`, `test_document_save_preserves_values.py`,
  `test_video_frame_stepping.py`, `test_gl_lifetime_guards.py`, `test_uniform_row_pruning.py`.
  **`test_ui_prose_budget.py` is the one to watch**: it names both a file and a function, so
  renaming a draw function breaks it even with no file move.
- **The oracle situation.** There is no golden-output oracle for rendering. `scripts/dogfood/judge.py`
  validates its own measurement primitives against synthetic images with answers known by
  construction — it decides whether the probes are correct, never whether a rendered frame looks
  right. The one byte-for-byte oracle is the vendored ABI probe, which decides only that the
  vendored artifact matches upstream, not that this repo's use of it is correct.
- **The visual blind spot, and no monitor overnight.** This box cannot screenshot the running
  app, and headless imgui reports focus, nav and post-`end_child` geometry differently from the
  real window. A headless pass on any of those is not verification. Overnight the machine has no
  display at all, so the GUI smoke skips (exit 87, gate still green) and the suite is the only
  net — see § How correctness is decided. Anything that must be judged by eye is left for the
  maintainer, not asserted by the sweep.

## The waves

One wave, one commit, `make gates` green before and after each. No commit mixes two waves.
Each wave is independently revertable so a failure can be bisected to it.

### W-0 — Inventory (measurement, no changes)

Enumerate every dead symbol of every kind across every package, from the language's own
constructs rather than from one regex. Kinds to enumerate explicitly, because the presence scan
did NOT enumerate the last two: module-level functions, methods, classes, pydantic and dataclass
fields, **enum members one by one**, module-level constants, private helpers, **type aliases one
by one**, and whole modules never imported. Directories: all of `shaderbox/` including every
subpackage, plus `scripts/` and `tests/`.

Run the tooling first (`uv run ruff check --select F401,F811,F841 .`, and vulture via
`uv run --with vulture vulture shaderbox scripts tests`), then grep for what the tools cannot
see. Record the tool output in the progress file.

Classify every candidate into the three tiers below BEFORE removing anything. This wave's output
sizes the rest of the night; nothing after it can be scoped until it has run. Its done-condition:
a written list, per tier, with the evidence for each, committed as `00_inventory.md`.

**A symbol is not kept alive by an old save.** The repo ships no backward-compatibility or
migration code, by standing rule: nothing pre-release ships to users, and the only data is the
dev sandbox. So "an existing `document.json` might still carry this field" is NOT a reason to
keep a field, and a removal that makes an old file unloadable is acceptable — the fix is to
hand-edit the affected files under `projects/dev/` (or regenerate them through the normal
load-and-save path) and `git add projects/dev` in the SAME commit. Never write a compat shim, an
old-format reader, or a migration step; if one seems necessary, that is the signal to stop and
report rather than to build it.

**False positives — these are NOT dead, do not remove them:** anything reached dynamically (a
name built as a string, a `getattr`, a registry lookup); pytest fixtures and test functions;
pydantic fields populated from JSON; enum members serialized to or from disk; a function stored
in a dict of handlers or passed as a callback; anything referenced from a `.glsl`, `.json` or
markdown file rather than from Python; and the dogfood harness's interactive API, which is driven
by hand rather than called from the package.

### W-1 — Deletion, by risk tier

| Tier | What | Rule |
|---|---|---|
| SAFE | unreferenced private helpers, dead module constants, unused dependencies | remove |
| CAREFUL | anything reachable dynamically — a registry name, a hook, a fixture by convention | prove dead before removing |
| RISKY | any exported surface a consumer outside this repo could use | leave it, or ask |

Remove one tier and one category at a time, gates between batches, so a failure names its batch.

### W-2 — The duplicated case-list

Derive `agent.py`'s edit-tool partitions from the registry that already owns the flag, rather
than re-spelling them. Leave `_RENDER_AUTHORING_TOOLS` alone; it answers a different question.
Done-condition: adding a tool with `is_edit=True` requires editing one place, shown by a test.

### W-3 — Duplicated comment text

Delete the verbatim duplicates only. **Do not run a comment sweep to a density target.** A long
comment explaining WHY, naming a failure it prevents, or quoting a measurement is the convention
working — keep it. Attempt-narration is already absent, so there is nothing to compress; if a
block that looks like history turns up anyway, keep every measurement and every named failure,
and when in doubt leave it. The one restating comment found may go with it.

### W-4 — `ui.py`'s draw code (RECOMMEND FIRST, do not assume)

Do not start by moving anything. First produce a recommendation, in the progress file, that
answers: which of `ui.py`'s draw functions have a natural home in `tabs/document.py`; what the
import churn would be; and whether the result is more readable than the status quo for someone
who reads `update_and_draw` top to bottom. Then apply it only if the answer is clearly yes.
**The repo's own rule wins over tidiness here** (see below). If the recommendation is "leave
it", that is a successful wave: write down why and move on.

If it does proceed: pure moves, no edits to bodies, import churn in the same commit and named in
the message; update every gate naming an old path in that same commit; then **re-verify each
gate you touched by breaking what it guards on purpose and confirming it still fails.**

### W-5 — Sanitize

The repo's closing sweep, per its own documented flow: walk the todo and conventions files, fix
stale references, update the roadmap banner and row, run the cold-context check.

## How correctness is decided

**The command is `make gates`.** Read its exit code, captured unpiped:
`make gates > /tmp/g.log 2>&1; echo $?`. A pipe reports the pipe's status, not the gate's; the
target itself warns about this when stdout is not a terminal.

**Overnight there is no monitor, so the GUI smoke skips every run.** This needs no action: the
smoke detects the missing display itself, exits 87, and `gates` prints `smoke SKIPPED` and stays
green. Read the gate's summary line rather than only its exit code, and note in the progress
file which stages ran.

**Coverage is not meaningfully weaker for this sweep.** The pytest suite drives real GL headless
— on the order of thirty test modules use moderngl or glfw directly, including a dedicated set
of GL-lifetime guards — so context handling, program compilation and resource lifetime are all
still covered. What the smoke uniquely adds is opening a real window on hardware GL and running
the frame loop in it. The remaining gap is therefore narrow, and the waves as scoped (deleting
symbols nothing reaches, deriving a case-list from the registry that owns it, comment text, and
a pure move only if W-4 recommends one) do not touch window creation or the frame loop.

The rule that keeps it that way: **a wave whose done-condition can only be checked by looking at
the running app does not run unattended** — report it for the morning instead.

**A changed test expectation in a structural wave is a defect in the refactor, not a test to
update.** The tests are the only evidence that behaviour was preserved; weakening one to get a
green destroys the meaning of the green. The single legitimate edit is a test file moving with
the code it covers, or a path string updated in the same commit as the move that changed it —
and that path update must be re-verified by breaking the guarded thing on purpose.

**There is no behaviour track.** No oracle decides rendering correctness here, and the visual
half cannot be checked on this machine, so unattended behaviour work is the thing not to do.
Anything behavioural that surfaces goes to the maintainer as a question, not into a commit.

## What looks wrong and is correct

Anything below will look like a finding and must NOT be changed. A wave proposing to clean one
up carries the burden of explaining how the thing it exists for still works.

- **`app.py` and `copilot/backend.py` are large on purpose.** `conventions.md` records that
  prior extractions already lifted the copilot backend and the headless project core out of
  `App`, that what remains is genuinely UI-bound, and — quoting it — *"Don't split further
  without a fresh pain signal (lost search-and-replace, unclear blast radius) — the remaining
  candidates are net-negative today."* Line count is explicitly NOT the trigger; the repo's own
  rule asks for a qualitative pain signal instead. No wave splits these on size.
- **`shaderbox/resources/editor/` is vendored upstream bytes.** `abi_probe.py` is compared
  byte-for-byte against upstream by `tests/test_editor_ffi.py`, and is excluded from ruff and
  pyright on purpose. Do not reformat, split, lint or tidy anything in that directory.
- **`shaderbox/glsl_docs.py` is generated** by `scripts/gen_glsl_docs.py` and says so. Do not
  hand-edit it; regenerate instead. The same holds for `shaderbox/glyph_tables.py`.
- **The dogfood harness's unused-looking methods** are an interactive API driven by hand.
- **Pydantic validators, pytest fixtures and imgui style attributes** look unreferenced to a
  static scan and are not.
- **`_RENDER_AUTHORING_TOOLS` beside the edit-tool sets** is not part of the W-2 duplication.
- **Long WHY comments quoting a measurement or naming a prevented failure** are the convention
  working, not clutter.
- **A dead constant does not mean its ASSET is dead** — and the reverse. Worked example, checked
  while writing this spec: `DEFAULT_IMAGE_FILE_PATH` has exactly one occurrence in the repo (its
  own definition) and is safe to remove, but `ai_docs/` still describes a default-image mechanism
  in the present tense, and the asset `shaderbox/resources/textures/default.jpeg` exists on disk.
  Nothing in the package reads that file today. Removing the CONSTANT is a W-1 SAFE removal;
  removing the ASSET is not, until the docs describing it are reconciled — a doc claiming a live
  mechanism is either stale (fix the doc) or the mechanism regressed (a defect, not a cleanup).
  **Deleting a resource file is out of scope for this sweep**; report it and leave it.

## Cold start

Read this file, then `00_progress.md` beside it, then `git log --oneline -15`. The progress file
and the git log are the truth about where the work stopped; this spec is only the plan.

**Measure before acting.** Every specific in § What is present is a seed from a shallow scan and
must be **re-measured**: the counts, the named symbols, the file sizes. W-0 exists precisely
because this spec's specifics cannot be trusted as an inventory. If a wave finds only what this
spec named, the wave was run wrong.

Order: W-0 first, always, since it sizes everything else. Then W-1, W-2, W-3 in any order (they
do not touch each other), then W-4's recommendation, then W-5. Append to the progress file after
each wave, before starting the next — including what was ruled out and why, or a resumed session
re-litigates a decision already settled.

Settled already, as CONSTRAINTS rather than open questions:

- No behaviour work. Structural only.
- **Breaking old saves and old projects is allowed.** No backward compatibility, no migration
  code, no old-format readers — the maintainer confirmed this explicitly for this sweep, and it
  is the repo's standing rule. A reshaped on-disk format is fixed by hand-editing
  `projects/dev/` in the same commit, never by a compat path.
- No file split on size alone; the repo's pain-signal rule governs (§ What looks wrong).
- No comment sweep to a density target; attempt-narration is already absent.
- Misfiling was scanned and found absent — there is no relocation wave beyond W-4's question.
- The vendored editor directory and the generated modules are off limits.
- Write each wave's done-condition, as a checkable statement, before starting that wave.
- No monitor overnight: the GUI smoke skips every run, so a green gate proves check and tests
  only. Nothing whose correctness needs the running app may land unattended.
- Report against the milestone, not against effort. If a wave turns out bigger than this spec
  assumed, say so rather than silently narrowing it.

## Coverage claims

Collected here so a narrow scan is visible as narrow. Each is the scanning agent's own claim,
recorded verbatim in § What is present beside its category. The two gaps this spec knows about:
**enum members and type aliases were not individually enumerated** by the dead-code scan, and
comment scanning did not cover `tests/` or `scripts/`. W-0 closes the first; the second is
accepted, since comments outside the package are not a target of this sweep.
