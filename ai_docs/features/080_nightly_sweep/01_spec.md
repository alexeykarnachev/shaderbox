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

One track only: **structural**. Leave the codebase with less dead weight and nothing filed
where a reader would not look for it, with behaviour bit-identical.

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
(`SYNTAX_1`..`SYNTAX_9`) and `SYN_PREPROC` is not among them, so the token is written by
nothing and read by nothing.

**A previous sweep's inventory did not fully drain, and its commit message says otherwise.**
`ai_docs/features/074_nightly_sweep/00_inventory.md` lists
`TelegramExporter.bot_token_present` as REMOVED (tier SAFE) at commit `1b159f1`, whose message
states the method was removed. The method is at `shaderbox/exporters/telegram.py:262` today,
and `git show 1b159f1 -- shaderbox/exporters/telegram.py` shows that commit's diff for that
path is the two lines of a DIFFERENT edit — the method was never touched. The other nine
symbols in that inventory are genuinely gone (`DEFAULT_IMAGE_FILE_PATH`, `ProjectPaths.copilot_dir`,
`ExporterStatus.in_flight`, the `AgentToolCard` fields all verified absent). So: one symbol
survived a removal its own commit claimed to have made.

Two consequences for this sweep. The inventory wave must **re-verify that earlier sweep's
"Removed" table entry by entry** rather than trusting it, and the deletion wave's done-condition
must be a grep over the landed tree rather than a commit message.

### Repetition — PRESENT, in the case-list shape only

The valuable shape here is a list of cases enumerated at several sites. Example: adding a
fourth document tab requires editing `DocumentTab` (`shaderbox/ui_regions.py`), a `CommandId`
member and a `CommandSpec` registration (`shaderbox/commands.py`), a dispatch lambda
(`shaderbox/app.py`), and `_NODE_TABS` (`shaderbox/ui.py`) — and nothing fails if a site is
missed; the tab is simply unreachable by keybinding.

Block-level duplication was probed and is essentially absent: the result-shaping paths are
already hoisted behind `_render_result` / `_publish_result`.

### Misfiling — PRESENT

Example: `IntelCache` lives in `shaderbox/intel/document.py` but caches a GLSL syntax index
keyed by editor buffer — nothing to do with `Document`. Its siblings `GlslIndex`,
`GlslContext` and `build_glsl_index` all live in `shaderbox/intel/index.py`, which is where a
reader would look, and the filename collides with the unrelated top-level `shaderbox/document.py`.

A second shape: `shaderbox/watch.py` imports `App` and is consumed only by `shaderbox/ui.py`,
so it is app-tier glue sitting in the top-level module namespace.

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

### Comments — the convention is holding; one narrow defect

Restating-the-code comments are ABSENT: of the comment lines sampled, none restated the
statement below them. What is present is verbatim duplication — an identical three-line
GL-fixture rationale block repeated across five test modules
(`tests/test_pass_hot_reload.py:42`, `test_graph_persistence.py:54`, `test_pass_render.py:53`,
`test_document_graph.py:75`, `test_script_engine_gl.py:43`) — and one genuine
attempt-chronology at `tests/test_editor_ffi.py:1141`, which narrates a superseded claim
before giving the current reason.

**Comment work is narrow by construction.** Deleting duplicated text and compressing that one
chronology is in scope. Compressing long explanatory blocks is NOT: a block naming the failure
it prevents or quoting a measurement is the convention working, and is often the only
surviving record of something a real defect cost to learn. When in doubt, leave it.

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

Done-condition: every kind above has been scanned with a written coverage claim, and every row
of the earlier sweep's "Removed" table carries a present/absent verdict from a grep run against
the current tree.

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

Done-condition: `grep -rnw <symbol>` over the landed tree returns only removals for every symbol
the inventory tiered SAFE; `make gates` green; no test assertion changed.

### W-2 — relocation: move what is filed where a reader would not look

Pure moves, no edits to bodies. Import churn in the same commit, said so in the message. Every
path-naming check above updated in the same commit **and then re-broken on purpose to confirm it
still fails** — a check edited to accommodate a move is the classic way a gate goes quietly
vacuous while still reporting green.

Done-condition: `make gates` green; each moved symbol resolves from its new home; each edited
check demonstrated to still fail when the thing it guards is broken, with the break named in the
commit message.

### W-3 — comments: the verbatim repeats

Delete the verbatim repeats; compress the one attempt-chronology to its current reason. Every
measurement and every named failure survives. Nothing else in the comment layer is touched.

Done-condition: the repeated block appears once; `make gates` green; no comment that names a
failure or quotes a measurement was shortened.

### W-4 — the case lists, only where a case list is real

Hoist a list of cases enumerated at several sites behind one name, so adding a case is one edit.
**Clarity beats brevity**: do not collapse distinct concerns, do not trade an explicit form for a
compact one, and do not delete an abstraction that was organising something. Fewer lines is not
the goal. **Similar shape is not shared meaning** — two blocks that answer different questions
stay apart, because hoisting them couples two things that must change independently.

If the case-list work turns out to need a design decision, it stops and goes to the maintainer.

Done-condition: adding a hypothetical new case is demonstrably one edit; `make gates` green.

### W-5 — speculative generality

Remove machinery whose caller does not exist, `apply_theme`'s unused variation being the
candidate. **Name the caller that wants a seam, or delete the seam.** A seam with no caller is
itself a thing to find and delete, not to add.

Done-condition: no parameter remains that every call site passes identically; `make gates` green.

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
- **The two sanctioned lazy imports** inside function bodies. Deliberate SDK-deferral seams; the
  repo's import-at-top rule names them as its only exceptions.
- **`shaderbox/theme.py` importing `shaderbox/editor/ffi.py`.** `editor_palette()` returns a dict
  keyed by the editor's own `Slot` enum; the import is that function's subject.

## How correctness is decided

`make gates`, exit code captured unpiped:

    make gates > /tmp/g.log 2>&1; echo $?

Nothing else counts. A green from a pipe is the pipe's status; a skipped smoke is not a pass.

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
- **Order: W-0 first, always** — it is measurement. Then W-1 through W-5. A wave that turns out
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
