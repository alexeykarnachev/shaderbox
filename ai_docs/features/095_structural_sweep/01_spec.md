# 095 — Structural sweep

A repo-wide pass over the codebase's SHAPE rather than its behavior: dead code, duplication,
misfiling, live facts in the docs, and comment hygiene. No user-visible change is in scope —
every wave here moves or deletes code without altering what the app does.

This spec records that a KIND of problem is present and names one example of each. **It is not
the work order.** The inventory wave (W-0) produces the list the later waves act on, by
measuring the tree as it stands rather than by trusting anything written here. A count in this
document would be a count from a shallow scan, and a shallow count is how a sweep stops early.

## Status

**The sweep is COMPLETE.** Every wave below landed, each as its own commit with `make gates`
green before and after, and two adversarial review rounds followed. `02_progress.md` beside this
file is the record of what happened — read it, not this document, for where the work stopped and
what was deliberately left alone.

The waves ran W-0, W-R, W-F, W-D, W-C, then four follow-up commits closing gaps the reviews
found. List them with:

    git log --oneline --grep '^095'

Nothing here is a pending instruction. Read this spec for WHY the sweep was scoped the way it
was and, above all, for the "What looks wrong and is correct" section — that list is what a
later sweep is most likely to re-raise.

## Goal

One track only, structural. There is no behavior work in this sweep and none is to be
invented: the repo's safety net cannot judge whether a rendered frame looks correct (see
"How correctness is decided"), so unattended behavior work has no oracle to decide it.

The deliverable is a tree where: every symbol that survives is reachable, the one duplicated
test fixture has a single home, no document states a fact a later commit can silently falsify,
and the shared UI vocabulary's unreferenced members have been judged rather than assumed dead.

## What is present

Each entry is **one illustrative example**, found by a presence scan. None is the full extent
of its category — W-0 measures that.

### Survey (measurement — re-measure, do not trust)

- **Dead code: present.** `shaderbox/ui_primitives.py::segmented_choice` — a complete,
  docstringed widget whose name appears nowhere else in the tree. Verified directly:
  `grep -rn "segmented_choice" --include=*.py .` returns only its own `def` line.
  See the do-not-change section before deleting it.
- **Duplication of a case list: present.** `SymbolKind` (`shaderbox/intel/symbols.py`) is
  re-enumerated as `_KIND_RANK` there and as `_KIND_COLOR` / `_KIND_SLOT` in `theme.py`.
  **Already gated** — see the do-not-change section.
- **Duplication of a block: present, in the test suite.** The `gl_ctx` fixture is defined
  byte-identically in several test modules, comment and body alike, with no shared home in
  `tests/conftest.py`. Enumerate with:
  `grep -rln "poisons the process's EGL display" tests/`
  A second cluster shares the imgui-font-atlas comment:
  `grep -rln "font atlas is process-global" tests/`
- **Live facts in the harness: present.** `ai_docs/dev_flow.md` states the repo "is currently
  at 0 pyright errors" — a status any commit can falsify. Enumerate the class with a search
  for timings, counts, sizes, line-number citations and "currently/as of" phrasing across
  `CLAUDE.md`, `ai_docs/`, `Makefile` and code comments.
- **A stale doc-to-doc pointer: present.** `ai_docs/dev_flow.md` describes the shader library's
  live root as "seeded by copy until the load mechanism lands — `todo.md`". The mechanism
  landed (`shaderbox/shader_lib/seed.py` exists) and `todo.md` is drained to zero entries
  (`grep -c "^## " ai_docs/todo.md`).
- **Restating comments: present but rare**, and clustered in flat constant files such as
  `shaderbox/constants.py` (`# File extensions` above a list of file extensions). Outside
  those, sampled comments state a real invariant. Treat a large find here as a signal the
  scan is wrong, not that the repo is.
- **Attempt-narration comments: present, rare**, and confined to the test suite, each a single
  embedded clause rather than a paragraph — e.g. `tests/test_script_engine.py` carries "A first
  attempt at clearing the zombie row asked whether the key was REWRITTEN this tick…" beside a
  live falsifier statement.
- **Missing seams: absent.** The one natural extension point, `shaderbox/exporters/base.py`,
  already has two live implementers. Do not add a seam in this sweep.
- **Docs-vs-code drift: absent** beyond the one stale pointer above. Paths, symbols and make
  targets cited in `CLAUDE.md` and `dev_flow.md` resolve; the roadmap banner matches the log.

### Proposal (argument — attack this hardest)

Unlike the survey above, nothing here has been measured into a defect. Each is a claim that
something *should* move, and a reviewer should try to defeat it before any wave acts on it.

- **Claimed misfiling: `widgets/graph_state.py` and `tabs/share_state.py` hold state, not draw
  code, yet live in draw-layer packages; `app.py` imports both.** A presence scan called this a
  dependency-direction violation. **That framing is already partly disproven**: `graph_state.py`
  imports `ui_primitives.InlineInput`, so it is UI-adjacent state, not pure model state, and it
  sits correctly beside the widget that owns it. `share_state.py` imports no imgui at all, so
  the claim is live only for that one. The burden is on a wave that wants to move it: state the
  reader who would look in the new place, and account for the import churn (a module move in
  Python rewrites every importing site).
- **Claimed split candidates.** `copilot/backend.py` carries six feature-tagged section banners
  and is the one large file with real seams. `app.py` has banners over part of its span and
  none over the tab/session stretch — a split there would invent a seam. `ui_primitives.py` has
  two banners and a long unbannered tail. **Recommend, do not assume**: a file read top to
  bottom may be worth more whole, and no split is authorized by this spec.

## The constraints (verified in this session, not assumed)

- **A module move rewrites every importing site.** Python resolves `from shaderbox.x import y`
  by path, so relocation churns imports. Measure the fan-in before proposing one:
  `grep -rn "^from shaderbox\|^import shaderbox" shaderbox/ --include=*.py`
  The heavily-imported modules are `theme`, `ui_primitives`, `app` and `paths`.
- **`make gates` reports failure honestly.** The target ends in `exit $status`, sets `status`
  only from a captured `$?` per sub-step, pipes nothing, and prints a warning when stdout is
  not a terminal. Judge it by the exit code captured unpiped.
- **A skipped smoke is not a pass.** `make gates` reports it as `skipped` on a display-less box.
- **Several tests name specific source paths and break on a move.** They must be updated in the
  same commit as any relocation, and each updated gate must then be re-broken on purpose to
  confirm it still fires. Enumerate them with:
  `grep -rn 'Path(__file__)\|"shaderbox/' tests/*.py`
  Known members of this set include the tests covering confirm popups, project management, the
  keymap disjointness check, generated glyph artifacts and the worker-daemon contract.
- **The test suite is process-partitioned by `xdist_group`.** Modules driving real frames each
  carry their own group because the imgui font atlas is process-global. A fixture moved into
  `tests/conftest.py` must not disturb those groupings:
  `grep -rn "xdist_group" tests/*.py`
- **No dead-code tool is installed.** `vulture` is not a dependency of this repo. W-0 either
  adds one as a dev dependency or enumerates from Python's own constructs — and either way
  covers the kinds a reference scan misses.

## How correctness is decided

`make gates` — check, then test, then smoke, stopping at the first failure, one exit code.
Run it unpiped and read the status before anything else:

    make gates > /tmp/g.log 2>&1; echo $?

**A changed test expectation in a structural wave is a defect in the refactor, not a test to
update.** The tests are the only evidence behavior was preserved; weakening one to get a green
destroys what the green meant. The single legitimate edit is a test moving with the code it
covers, or a test whose *subject* legitimately changed name — never an assertion relaxed.

**The safety net's blind spot:** nothing in the suite judges whether a rendered frame looks
visually correct. The render tests assert determinism, dimensions and file placement; the
shader-library lock file pins signatures, not semantics; the glyph gate compares a committed
artifact against its generator, so a wrong generator agrees with itself. Any claim about visual
output needs the running app and a human eye, which is why this sweep stays structural.

## The waves

One wave, one commit, `make gates` green before and after each, each independently revertable.
No commit mixes two waves.

- **W-0 — Inventory. No changes.** Enumerate every dead symbol of every kind Python has —
  functions, methods, classes, pydantic/dataclass fields, enum members, module constants,
  type aliases, private helpers, whole modules — across every directory including
  `copilot/`, `widgets/`, `popups/`, `tabs/`, `exporters/`, `editor/`, `scripting/`,
  `shader_lib/`, `intel/`, `scripts/` and `tests/`. Sort the result into the three risk tiers
  below. This wave sizes every wave after it, so nothing downstream is scoped until it lands.
  Its output goes in `02_progress.md`.
- **W-R — Rot removal.** Delete the live facts. Each one either goes, or becomes the command
  that produces it. Frozen history stays: a recorded before/after of work already done, a dated
  decision, a superseded-feature note — none of those can drift, because they describe a past
  state rather than the current one. The test is whether an unrelated commit next week makes
  the statement wrong. Includes the stale `todo.md` pointer in `dev_flow.md`.
  This wave gets no gate, deliberately: a checker for live facts needs the number it guards and
  so rots in step with it. Deletion is the fix that stays fixed.
- **W-F — The test fixture with no single home.** Give `gl_ctx` one home in `tests/conftest.py`, keeping
  the comment that records the EGL segfault verbatim — it is the only surviving record of that
  failure. Same for the font-atlas cluster if it consolidates as cleanly. Respect the
  `xdist_group` partitioning. Verify no module's grouping changed.
- **W-D — Deletion, by tier, one tier per batch, verifying between batches.**
  **SAFE**: unreferenced private helpers and internal functions — remove.
  **CAREFUL**: anything reachable dynamically — a registry entry, a name built as a string, a
  fixture collected by convention, an enum member used only as a dict key — prove dead first.
  **RISKY**: the shared UI vocabulary's public surface, a documented accessor with no internal
  caller — leave it, or ask. A pydantic field is persistence surface: removing one changes the
  on-disk format and is not a deletion decision.
- **W-C — Comments.** Delete the restating one-liners in flat constant files and any verbatim
  duplicated paragraph. Compress an attempt-narration to its live reason, keeping every
  measurement and every named failure. **Not a density target.** A long comment stating why,
  naming a failure it prevents, or quoting a measurement is the convention working and stays
  untouched. If in doubt, leave it.

Relocation and file-splitting are deliberately **not** waves here. Both were surveyed, neither
produced a defect that survives scrutiny, and both are expensive in a language where a move
rewrites every call site. A later sweep can revisit them with a named reader who would look in
the new place.

## What looks wrong and is correct

A wave proposing to change any of these must first explain how the thing it exists for still
works.

- **`segmented_choice` has no caller, and that is not sufficient reason to delete it.**
  `ui_primitives.py` is the repo's shared UI vocabulary, and `tests/test_button_tiers.py`
  iterates `vars(ui_primitives)` — the module is treated as an enumerable surface with a
  contract, not as a private implementation. Deleting an unreferenced widget there is a
  judgement about the vocabulary, not a dead-code cleanup. Decide it deliberately.
- **The `SymbolKind` triple-enumeration is deliberate and already gated.** The three dicts are
  apart because rank is a completion concern and color/slot are an imgui-layer concern —
  hoisting them would couple two layers the repo separated on purpose. A test walks the full
  enum domain and asserts all three tables answer for every member, so a kind added without all
  three fails before a frame draws. Leave it.
- **`test_ui_prose_budget.py`'s exemption tables cannot rot silently.** A presence scan claimed
  a stale path key there would pass vacuously. That is false, and was tested: adding a key
  naming a non-existent file turns
  `test_every_unmeasurable_entry_still_names_a_real_site` red. The tables are self-invalidating
  by design. Do not "fix" them.
- **The measured constants in code comments are the convention working.** A comment such as the
  token-ratio note in `shaderbox/copilot/prompt.py`, which states what was measured and over
  what sample, is a record of something that cost real effort to learn. It is not a live fact to
  delete. A live fact asserts the repo's current state; a measurement states its own provenance.
- **`todo.md` being empty is the intended end state**, not a missing file. It is frozen
  drain-only and has drained.
- **The high comment density is the house style.** Do not treat comment volume as a defect.

## Cold start

Read this section first if you are resuming, compacted, or new to this work.

1. **Read `02_progress.md` beside this file, and `git log --oneline`, before anything else.**
   They are the truth about where the work stopped; this spec is only the plan.
2. **Re-measure before acting.** Every example in "What is present" is illustrative. Run the
   enumerating command yourself. A number in this document would be wrong by the time you read
   it, which is why there are none.
3. **Work in wave order**: W-0, then W-R, then W-F, W-D, W-C. W-0 is measurement and comes
   first always.
4. **Write each wave's done-condition before starting it**, as a checkable statement — which
   command must be green, what must no longer appear in which search — and append the result to
   `02_progress.md` before starting the next wave.
5. **Verify with `make gates`, unpiped, reading the exit code.** Green before and after each
   wave.

These are settled and are CONSTRAINTS, not options to reconsider:

- The sweep is **structural only**. No behavior work; there is no oracle for it here.
- **No relocation and no file split** in this sweep. Both were surveyed and rejected above.
- **No migration or compatibility code, ever**, per the repo's own standing rule. If a change
  reshapes an on-disk format, change it and fix `projects/dev/` by hand in the same wave.
- **A failing test is a wrong refactor**, never a test to relax.
- Commits land on `dev`. One wave per commit.
- The three "looks wrong and is correct" entries above are decided. Do not re-litigate them;
  a wave that wants to touch one must clear the burden stated there.

## Coverage claim

The presence scan behind this spec covered:

scanned: module-level functions, methods, classes, private helpers and module constants across
`shaderbox/` and its subpackages, `scripts/` and `tests/`, via reference search and manual
cross-check; internal import lines repo-wide for direction; enum-keyed dispatch tables; comment
text across all Python sources; `CLAUDE.md`, `ai_docs/roadmap.md`, `ai_docs/todo.md`,
`ai_docs/dev_flow.md`, `Makefile`, and `ai_docs/features/` by search; the Makefile gate chain
and the test suite's path-naming and oracle structure.

not scanned: pydantic model fields, enum members, type aliases and whole-module deadness (W-0
must cover all four — the scan that produced this spec did not); full bodies of files outside
the three largest; `shaderbox/resources/` content; `ai_docs/design/` and `ai_docs/itch/`;
`.glsl` sources and other non-Python assets; and every feature spec body line-by-line against
the code (searched, not read).
