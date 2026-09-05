# 080 — nightly structural sweep

A large unattended pass over the repo's SHAPE: dead code, repetition, misfiling, wrong
dependency directions, file sizes, missing seams, comments narrating history. Structural
only — nothing here changes what the program does.

## Status

Nothing has landed yet. This spec is the plan; `00_progress.md` beside it is the record of
what actually happened, appended after every wave. **On resume, read `00_progress.md` and
`git log` first** — they are the truth about where the work stopped, where this file is only
the plan.

## Goal

One track only: **structural**, and after review it is a SMALL night. Leave the codebase with
less dead weight and behaviour bit-identical.

The honest headline: this repo has very little structural debt. A previous sweep drained most of
what existed, and an adversarial review of this spec's own first draft closed three of the five
waves it proposed — each had opened a question that was already settled, with the reason recorded
either in an earlier sweep or in the docstring of the file itself. What remains is a handful of
dead symbols and one piece of unused machinery. **A sweep that reports a large haul here would be
reporting a wrong measurement**, so the plan is deliberately short.

There is deliberately NO behaviour track. The repo's behaviour oracle is the dogfood harness,
which decides *copilot* behaviour against a human judge — it does not decide whether a
refactor preserved semantics, and it costs model spend per run. Unattended behaviour work
without an oracle is the thing not to do; behaviour questions found during the night get
written down in the progress log and handed to the maintainer, not fixed.

## What is present

The survey below is **measurement** — a presence scan, re-measured by W-0 before anything acts
on it. The proposal (the waves) is **argument** and is what a review round should attack
hardest.

Every example is ILLUSTRATIVE, not an inventory. How many of each kind exist is unknown until
the inventory wave runs. A spec that enumerated its dead symbols would become a spec to delete
exactly those, when the instruction is to find every one.

### Dead code — PRESENT

Example: `TOOLS_BLOCK` at `shaderbox/copilot/context_breakdown.py:25`. Its sibling
`EXCHANGE_BLOCK` two lines above is passed to `_measure` and asserted on in
`tests/test_context_breakdown.py`; `TOOLS_BLOCK` appears nowhere else in the repo — the
`tools=` part is measured through the separate `tools_chars` / `tools_est_tokens` fields
instead. Verified by `grep -rnw TOOLS_BLOCK shaderbox scripts dogfood tests`.

A second shape confirmed by hand: `COLOR.SYN_PREPROC` (`shaderbox/theme.py:190`) is a syntax
color from before feature 067 vendored the editor. `editor_palette()` maps nine syntax slots
(`SYNTAX_1`..`SYNTAX_9`) and `SYN_PREPROC` is not among them; the intel color table does not
name it either, and nothing in the repo reaches a `COLOR` attribute by `getattr`. It carries
its dataclass default and no reader.

**A previous sweep's inventory did not fully drain, and its commit message says otherwise.**
`ai_docs/features/074_nightly_sweep/00_inventory.md` lists
`TelegramExporter.bot_token_present` as REMOVED (tier SAFE) at commit `1b159f1`, whose message
states the method was removed. The method is at `shaderbox/exporters/telegram.py:262` today,
and `git show 1b159f1 -- shaderbox/exporters/telegram.py` shows that commit's diff for that
path is the two lines of a DIFFERENT edit — the method was never touched. The other nine
symbols in that inventory are genuinely gone (`DEFAULT_IMAGE_FILE_PATH`, `ProjectPaths.copilot_dir`,
`ExporterStatus.in_flight`, the `AgentToolCard` fields all verified absent). That table's ten
rows are nine removals plus one pointer into its KEPT section, so eight of nine landed and one
survived a removal its own commit claimed to have made.

Two consequences for this sweep. The inventory wave must **re-verify that earlier sweep's
"Removed" table entry by entry** rather than trusting it, and the deletion wave's done-condition
must be a grep over the landed tree rather than a commit message.

### Repetition — PRESENT as a shape, ABSENT as work

The case-list shape is real: adding a fourth document tab needs edits to `DocumentTab`
(`shaderbox/ui_regions.py`), a `CommandId` member and a `CommandSpec` row
(`shaderbox/commands.py`), a dispatch lambda (`shaderbox/app.py`) and `_NODE_TABS`
(`shaderbox/ui.py`), and nothing fails if a site is missed.

**But the split is deliberate and documented at the source.** `ui_regions.py`'s own docstring
says `DocumentTab` is a leaf on purpose: it is a plain name with no imgui in it, `ui_models.py`
persists the active tab, and keeping the enum beside the command table would drag imgui into the
headless model layer — `commands.py` builds `K = imgui.Key` at module scope, so importing it
loads the library. Hoisting the case list is precisely the coupling that docstring exists to
prevent.

Block-level duplication was probed and is essentially absent: the result-shaping paths are
already hoisted behind `_render_result` / `_publish_result`.

**No repetition wave.** The one candidate is a documented layering decision, not a defect.

### Misfiling — ABSENT; the two candidates were both misreadings

Both examples the presence scan produced were refuted on inspection, and an earlier sweep had
already scanned this dimension and found it absent.

`IntelCache` (`shaderbox/intel/document.py`) looked misfiled because the filename collides with
the top-level `shaderbox/document.py`. It does not: the module's own docstring says the cache is
keyed on an editor HANDLE and exists so a handle recreated for the same path is fed again. The
file is named for the document BUFFER whose intel it caches, and `conventions.md` names it in
that role. Moving it would make a documented decision stale to fix a name collision that is not
one.

`shaderbox/watch.py` imports `App` and is consumed only by `shaderbox/ui.py`, which reads as
app-tier glue in the top-level namespace. But `dev_flow.md`'s module map names it there with both
its functions, so the placement is recorded rather than accidental.

**No relocation wave.** An earlier sweep scanned this dimension and recorded it absent, noting
that many modules carry an explicit "leaf module, imports only X" docstring a reviewer can check
mechanically. Nothing here rebuts that finding.

### Dependency direction — the survey's own layering sketch was wrong

An import-graph scan reported upward edges (`theme.py` → `editor.ffi`, `ui_models.py` →
`copilot.state`, `watch.py` → `app`), then concluded in its own words that the modules it had
labelled leaves are actually central hub modules, and that the labels — not the edges — were
the error. `theme.py` importing `editor.ffi` exists so `editor_palette()` can return a dict
keyed by the editor's own `Slot` enum, which is the correct direction for that function.

**Recorded as a false lead, not as a wave.** No dependency-direction work is planned. If W-0
turns up a genuine inversion, it needs the maintainer, because reshaping the hub modules is a
design change, not a sweep.

### File sizes — no split warranted

The four largest hand-written files were read (`copilot/backend.py`, `app.py`,
`ui_primitives.py`, `exporters/telegram.py`). Each is one cohesive concern that happens to be
long: `backend.py` is the copilot's whole API surface as many small methods with no dividers;
`app.py`'s two section comments mark forwarder boilerplate over an already-extracted session
object, not a second concern; `ui_primitives.py`'s button/labelled-field blocks are internal
signposting within one shared-primitives job. Splitting any of them would invent a seam rather
than follow one.

### Comments — the convention is holding; nothing in scope

Restating-the-code comments are ABSENT: of the comment lines sampled, none restated the statement
below them. An earlier sweep reached the same verdict and recorded that its one candidate was a
false positive — `ui.py`'s `# Process hotkeys` is a section-banner label, part of the `# ----`
pattern that delimits the phases of `update_and_draw` and is the file's only navigation aid.
`conventions.md` sanctions those banners explicitly.

The verbatim-duplication scan found one block: an identical three-line rationale repeated across
five GL test fixtures, explaining that an explicit `backend="egl"` context released there poisons
the process's EGL display and segfaults the next module's first compile. **That block stays.** It
quotes a measured segfault and names the failure it prevents, which is the convention working;
an earlier sweep considered it and kept it for exactly that reason. Re-raising it here was a
re-litigation of a settled decision, and it is settled the same way.

One attempt-chronology exists at `tests/test_editor_ffi.py:1141`, narrating a superseded claim
before giving the current reason. It is one comment in a test, and compressing it buys nothing
that justifies an unattended edit to a file whose subject is a silent-failure ABI contract.

**No comment wave.**

### Speculative generality — PRESENT

Example: `apply_theme` (`shaderbox/theme.py`) takes `accent` / `density` / `rounding`, each
backed by a multi-value `Literal` and a preset table — 24 reachable combinations. The one real
call site (`shaderbox/app.py:217`) passes no arguments at all, and no test exercises the
function. The variation machinery serves a runtime theme-switcher that does not exist.

## Constraints, verified at `1439739`

- **Python has no directory/module coupling.** A move is a rename plus an import rewrite at
  every call site; there is no package boundary to negotiate. The tree carries about a
  thousand intra-package import statements, so a move's import churn belongs in the same
  commit as the move, stated in the message.
- **`make gates` is honest.** Run at `1439739`: exit 0, all three stages green. It runs check →
  test → smoke, stops at the first failure, pipes nothing, and warns on non-tty stdout that a
  piped `$?` is the pipe's. A skip for want of a display reports as *skipped*, which is not a
  pass. Judge it by the exit code captured unpiped.
- **`make check` type-checks `shaderbox`, `scripts`, `.claude/skills` and `dogfood`** via
  `[tool.pyright] include`, with `resources/editor/abi_probe.py`, `scripts/dogfood/runs` and
  three named scripts excluded. The pyright hook passes NO path argument on purpose: a path on
  the command line overrides the config's include list and silently un-widens the checked set.
  A wave must not add one.
- **`tests/` is lint-checked but NOT type-checked.** ruff runs over the whole repo minus its
  `extend-exclude`; pyright's `include` list does not name `tests`. So a rename that breaks a
  test's type contract surfaces only when the suite RUNS, never at check time — which is why
  `make gates` and not `make check` is this sweep's verification command.
- **Checks that name file paths break on a move.** `tests/test_generated_artifacts.py:27` pins
  `shaderbox/glyph_tables.py`; `tests/test_prose_spelling.py:74` pins `shaderbox/ui.py`;
  `tests/test_worker_daemon_contract.py:49-50` pin `shaderbox/copilot/session.py` and
  `shaderbox/exporters/worker.py`; `tests/test_ui_prose_budget.py` pins roughly forty
  `(file, function)` pairs across `ui_primitives.py`, `tabs/`, `popups/`, `widgets/` and
  `exporters/`. `build.sh` names `scripts/README.md` and the launcher
  scripts in its bundle allowlist; the `Makefile` and `.github/workflows/ci.yml` both name
  `scripts/smoke.py`; `scripts/gen_glyphs.py` and `scripts/gen_glsl_docs.py` each name the
  generated file they write, and `tests/test_generated_artifacts.py` pins the same pair.
  `tests/test_editor_ffi.py` reads the vendored `abi_probe.py` by path and SKIPS when it is
  absent — a move there goes quietly green rather than red, so it needs the re-break most. **Any move touching these updates them in
  the same commit, and then re-breaks the check on purpose to confirm it still fails.**
- **No behaviour oracle for refactors.** `scripts/dogfood/judge.py` states its own contract in
  its docstring: numbers out, never a verdict — the judgement stays with the human reading the
  report. Its blind spots are recorded in `ai_docs/features/057_dogfood_axes_and_scenarios/`:
  frame-pair sampling is pace-blind, and event-level outcomes are unobservable. It costs model
  spend per run and cannot tell you a refactor preserved semantics. For this sweep the oracle
  is `make gates` and nothing else.

## The waves

**One wave, one commit, `make gates` green at each, each independently revertable.** No commit
mixes two waves. Commit on `dev`; this repo has no per-feature branches and the push is the
maintainer's own, so push as each wave lands.

### W-0 — inventory. No changes.

Enumerate every dead symbol of every kind Python has, across every directory. This wave's
output SIZES the rest of the night, so nothing after it can be scoped before it runs.

The kinds, enumerated from the language rather than from another language's list: module-level
functions, classes, methods, module constants, enum members individually, dataclass and
pydantic fields individually, type aliases, private helpers, whole modules. The directories:
`shaderbox/` and every subpackage, `scripts/`, `dogfood/`, `tests/`.

Method: `uv run ruff check --select F401,F811,F841 .` for unused imports and locals (currently
clean), `uvx vulture` as a candidate generator, and grep from the language's own constructs for
what tools cannot see — write-only fields, per-member enum use, dynamic reach. **Grep including
`tests/`**: a scan that omits the test tree reports live symbols as dead. That error was made
during this spec's own scan, which called `pass_graph.DTYPES` dead when three test modules
import it.

**Re-verify the earlier sweep's "Removed" table entry by entry** — one of its rows survived,
as recorded above.

**False-positive classes; a static scan cannot see these and must not report them dead:**
pydantic `model_config`, `@model_validator` and `@field_validator` methods; imgui style
attributes assigned by name in `app.py` and `theme.py`; `editor/ffi.py`'s unused methods, flag
members and `PRIM_STRIDE` (a deliberate ctypes mirror of a vendored C ABI — completeness against
the ABI is the point, and two tests in `tests/test_editor_ffi.py` hold it there);
`PassGraph.version` and `ConversationStore.version` (round-trip-only by design); pytest fixtures
and test functions; entry points; names reached as strings — `popups/settings.py`'s
`TELEGRAM_TOKEN` and `YOUTUBE_CLIENT` are matched by string literal from the two exporters, with
a comment at each site saying so; re-exports in `__init__.py`; anything under
`shaderbox/resources/editor/abi_probe.py`; the dogfood harness's hand-driven interactive API.

Output: `00_inventory.md` beside this spec, with a per-symbol evidence row, a KEPT table, and a
coverage claim per scan in the form `scanned: <kinds> across <dirs>; not scanned: <the rest>`.

Done-condition — a written coverage claim is not enough, because a claim is written the same way
over a scan that worked and one that quietly narrowed:
- every kind above scanned, each with a coverage claim naming what it did NOT cover;
- every row of the earlier sweep's "Removed" table carrying a present/absent verdict from a grep
  against the current tree;
- **the scan falsified**: pick a symbol known to be live only through the test tree
  (`pass_graph.DTYPES` is the one this sweep already found) and confirm the scan does NOT report
  it dead. A scan that cannot be shown to spare a known-live symbol has not been shown to work.
  An earlier sweep's module-graph pass used a grep recipe that under-detected
  `from pkg import a, b, c` and would have written a confident claim over the miss.

### W-1 — deletion, by risk tier, one tier and one category at a time

| Tier | What | Rule |
|---|---|---|
| SAFE | unreferenced private helpers, unused internal symbols, unused deps | remove |
| CAREFUL | anything dynamically reachable — a string-built name, a plugin hook, a fixture collected by convention | prove it dead before removing |
| RISKY | public API, exported surface, a documented accessor with no internal caller | leave it, or ask |

Verify between batches so a failure names the batch. Removing a field from a shipped on-disk
model is a format change, not a deletion — those go to the maintainer.

**Follow the cascade.** 074 found that removing one field made a parameter dead; a removal that
orphans its producer is unfinished until the producer goes too, or until the producer's survival
is written down as deliberate.

Done-condition:
- `grep -rnw <symbol>` over the landed tree returns nothing for every symbol the inventory tiered
  SAFE;
- **the string-reach check run separately**, because a word-boundary grep structurally cannot see
  a name reached as a string: for each removed symbol, grep the whole tree for its name in
  quotes, and for a removed FIELD check it against the JSON keys under `projects/`. A field
  removed from a model that round-trips to disk narrows the save file silently — the salvage path
  drops unknown keys by design, so the gate stays green either way;
- `make gates` green;
- no test ASSERTION changed. A test may lose a constructor argument for a field that no longer
  exists; what it asserts must be identical. Check by reading the diff, not by the suite passing.

### W-2 — the speculative generality in `apply_theme`

`apply_theme` (`shaderbox/theme.py`) takes `accent` / `density` / `rounding`, each backed by a
multi-value `Literal` and a preset table. Its one call site (`shaderbox/app.py:217`) passes none
of them, and no test exercises the function or any accent.

**This wave is DANGEROUS in a specific way, and that is why it is written out at length.** The
accent presets are not only `apply_theme`'s parameter values. `theme.py` runs an IMPORT-TIME
assertion built from `_ACCENTS`: a fixed hue (`SELECT`, the state colors) may not equal any
accent preset's primary, or two cues merge under some accent. `conventions.md` records that
invariant as a design decision — a new theme supplies its own palette, presets and role mapping,
and the assertion is what holds a new one honest.

Deleting the accent machinery empties that assertion's domain. **An assertion over an empty set
passes.** So the naive removal leaves a green `make gates`, a green import, and a
theme-portability invariant that no longer checks anything — the checker that quietly narrows its
own domain, which this repo names as its most expensive bug family.

Therefore, in order:

1. Remove only what has no second job. `density` and `rounding` drive nothing but their own
   branches inside `apply_theme`; they and their `Literal` aliases go.
2. **`accent` and `_ACCENTS` stay** unless the invariant is preserved by construction. The
   presets ARE the assertion's domain; the parameter is the smaller half.
3. If a later judgement removes them anyway, the assertion must be rewritten first to enumerate
   the fixed hues against each other rather than against a table that no longer exists — and
   then broken on purpose to prove it still fires.

Done-condition, stated so a vacuous outcome fails it:
- `density` and `rounding` no longer appear in `shaderbox/`, verified by grep over the landed
  tree.
- `_accent_primaries` is still built from a NON-EMPTY `_ACCENTS`, verified by asserting its
  length in a Python one-liner run against the landed tree.
- **The invariant was re-broken and observed to fire**: set `COLOR.SELECT` to an accent
  preset's primary, confirm the import raises, restore it. The commit message names that break.
- `make gates` green.

If step 2's judgement call turns out to need a design decision, the wave stops and the question
goes to the maintainer rather than being resolved unattended.

## What looks wrong and is correct

A wave proposing to change any of these must first explain how the thing it exists for still
works.

- **`shaderbox/editor/ffi.py`'s unused half.** A ctypes mirror of a vendored C ABI. Deleting an
  unused flag member is SILENT — only the signature table is gate-compared — and would make the
  binding lie about the library's surface.
- **`PassGraph.version`, `ConversationStore.version`.** A schema version's whole job is to be
  there before a reader needs it. `ConversationStore.version` is written into every saved
  `conversation.json` on disk.
- **`ScriptError.uniform_name`, `PromptBlock.name`, `CompletionProvider.name`.** Write-only by
  grep, kept deliberately by 074: the first carries a diagnostic and is passed positionally at
  many sites; the other two are the identity column of a hand-written registry, and
  `conventions.md` specifies a prompt tier as a *named* block, so that field IS the name.
- **`popups/settings.py`'s `SettingsField` members.** Matched by string literal from
  `exporters/telegram.py` and `exporters/youtube.py`, with a comment at each site recording the
  coupling on purpose.
- **`GraphError.message` and `LookupPopup.word`.** Read from tests only, which is still reached.
  A grep restricted to `shaderbox/` reports both dead; the inventory's rule that `tests/` counts
  is what spares them.
- **`_ACCENTS` and the accent presets.** They are the domain of an import-time theme-portability
  assertion, not merely `apply_theme`'s parameter values. Emptying them leaves the assertion
  passing over nothing.
- **The `# ----` section banners** and their labels. `conventions.md` sanctions them, and an
  earlier sweep recorded that deleting them would destroy the only navigation aid in the longest
  draw function.
- **The five-site EGL fixture comment in `tests/`.** It quotes a measured segfault and names the
  failure it prevents. Kept by an earlier sweep for that reason; still kept.
- **The two sanctioned lazy imports** inside function bodies. Deliberate SDK-deferral seams; the
  repo's import-at-top rule names them as its only exceptions.
- **`shaderbox/theme.py` importing `shaderbox/editor/ffi.py`.** `editor_palette()` returns a dict
  keyed by the editor's own `Slot` enum; the import is that function's subject.

## How correctness is decided

`make gates`, exit code captured unpiped:

    make gates > /tmp/g.log 2>&1; echo $?

Nothing else counts. A green from a pipe is the pipe's status; a skipped smoke is not a pass.

**What the gate cannot see, per wave.** A green is evidence the change ran, not that it worked,
and each wave has a specific blind spot:

- **W-0** changes no code, so no gate applies at all. Its failure mode is a wrong inventory, and
  `make gates` is structurally silent about a document. Its falsifier is the defence.
- **W-1**: a field removed from a model that round-trips to disk. Nothing asserts that
  `projects/dev/`'s JSON survives, and the salvage path drops unknown keys by design, so the save
  file narrows silently behind a green gate. Hence the separate string-reach and JSON-key check.
- **W-1 and W-2 both**: modules using the app fixture SKIP without a display. Overnight on a
  display-less box an unknown fraction of the suite does not run, and the gate stays green
  through it. Confirm the smoke stage reported *passed* rather than *skipped* before believing a
  wave.
- **W-2**: the import-time theme assertion passes over an empty domain. No test references
  `apply_theme`, `_ACCENTS` or any accent name, so the gate is green through the whole
  regression. The re-break in its done-condition is the only thing that catches it.

**A changed test expectation in a structural wave is a defect in the refactor, not a test to
update.** The tests are the only evidence behaviour was preserved, so weakening one to get a
green destroys the thing the green was supposed to mean. The single legitimate edit is a test
file moving with the code it covers, or a call site moving because a symbol it constructs was
removed — the ASSERTIONS stay identical. If a refactor makes a test fail, the refactor is wrong:
revert it and record why in the progress log.

## Cold start

Read `00_progress.md` and `git log` before this file — they say where the work actually stopped.

**Measure before acting.** The survey above records that a KIND of problem is present and names
one example of each. It is not the work order. W-0 produces the inventory, from the current tree,
before any wave acts. Re-measure every number in this file: all of them were taken at `1439739`
and are a snapshot, and the test count and file sizes will have moved.

Settled, as CONSTRAINTS rather than options — do not re-open these, and pass them to any review
or implementation agent as fixed premises:

- **Structural only. No behaviour track.** There is no oracle for refactor correctness here, and
  the dogfood harness does not provide one. Behaviour questions get written down and handed over.
- **No dependency-direction wave.** The scan's layering labels were wrong, not the edges.
- **No file splits.** The four largest files were read and are cohesive.
- **No comment-density target.** Only verbatim duplicates and the one attempt-chronology.
- **No backward-compatibility or migration code, ever, unless the maintainer asks.** If a wave
  reshapes an on-disk model, the fix is to hand-edit `projects/dev/` and `git add` it in the same
  wave. A proposal containing a migration path is the signal to delete the proposal.
- **Commit on `dev`.** Never `master`, no per-feature branches. Never leave `projects/dev/`
  unstaged.
- **No relocation, comment or repetition wave.** All three were opened by the presence scan and
  closed on inspection: the misfiling candidates were misreadings of two documented placements,
  the comment block is a kept measurement, and the case list is a deliberate layering split its
  own docstring defends. An earlier sweep had already ruled out relocation and comments. Do not
  re-open them; that re-opening is itself the defect this constraint records.
- **Order: W-0 first, always** — it is measurement. Then W-1, then W-2. A wave that turns out
  bigger than this spec assumed gets SAID so in the progress log, not silently narrowed.

Write each wave's done-condition into `00_progress.md` BEFORE starting it, and append the result
after it lands, before starting the next. Record what was ruled out and why, or a resumed session
re-litigates a decision already made properly.

## Coverage claims — this spec's own scans, at `1439739`

These describe the PRESENCE scan, whose job was to find whether each kind of problem exists.
The inventory wave redoes the dead-code half properly and writes its own claims.

- `scanned: module-level functions, classes and methods across shaderbox/, scripts/, dogfood/,
  tests/; not scanned: nested and local functions, lambdas, dynamically-assigned attributes.`
- `scanned: module-level UPPER_SNAKE constants across shaderbox/, scripts/, dogfood/, plus a
  sample of enum members; not scanned: member-by-member sweep of every enum class,
  field-by-field sweep of every dataclass and pydantic model.`
- `scanned: repeated string literals by frequency, and case-list cross-referencing over
  copilot/tools/, tabs/, exporters/, scripting/, commands.py, app.py, ui.py, ui_regions.py; not
  scanned: scripting/engine.py and behavior.py line by line, tabs/code.py beyond grep, the
  exporter bodies beyond their interface shape, tests/.`
- `scanned: top-level def/class symbols in every shaderbox module against their importers; not
  scanned: symbol bodies, nested functions, ui_primitives.py and theme.py (declared shared-primitive homes).`
- `scanned: intra-package imports in every shaderbox/ module; not scanned: resources/, imports
  inside docstrings.`
- `scanned: section structure of the four largest hand-written files, read in full; not scanned:
  copilot/agent.py, tabs/code.py, project_session.py, scripting/engine.py, the dogfood scripts.`
- `scanned: comment-line sample across the main packages, verbatim-duplication frequency over
  shaderbox/ + scripts/ + tests/, narrative-marker grep over the whole tree; not scanned: the
  bulk of comment lines individually, resources/.`
- `scanned: ABC/Protocol classes, registry call sites, multi-default-parameter functions and
  kwargs passthroughs against every call site; not scanned: config/settings dataclasses,
  per-tool registry entries, exporter option flags.`
