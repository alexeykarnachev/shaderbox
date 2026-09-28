# 107 — Nightly sweep: the gate layer

## Status

The presence scan is DONE; what it found is in `progress.md`. No wave has landed.
Each wave appends to that file as it completes; on resume, read it and `git log` before
this spec — they are the truth about where the work stopped, where this is only the plan.

## Goal

**The code's SHAPE came back clean. The gate layer did not.** The structural questions --
misfiling, dependency direction, missing seams, comment history-narration -- returned
ABSENT with false trails recorded. So this sweep is not a
tidying pass over the source — it is a pass over the things that are supposed to catch
defects and do not.

One class dominates and it is worth more than the rest combined:

> **A check that asserts on SOURCE TEXT, or reads an import-time copy, rather than driving
> the runtime object — so it reports green while the mechanism underneath is missing.**

## What is present

Illustrative, ONE example each. This is not the inventory; W-0 produces that.

- **Vacuous checks.** `intel/document.py::IntelCache.index_for` — replacing
  `return entry.index, False` with `return build(), False` defeats the cache entirely
  while still reporting `changed=False`, and `test_the_cache_entry_belongs_to_its_handle`
  stays green. The test reads only the boolean, never the index it returned.
- **Dead code.** `theme.py::_muted` has no call sites. It went dead when feature 106
  deleted the four `GRAPH_PORT_*` tokens that were its only consumers, and the deletion
  left the helper behind. Same shape: `tests/test_graph_tab.py::_fitted_window`.
- **A decision made twice.** "Is this pass instanced" = truthy `entity_fields`, factored
  into `widgets/pass_graph.py::instanced_pass_keys` — whose own docstring says it exists
  because an inline version was once deleted and the whole suite stayed green — and then
  re-spelled inline at `popups/pass_settings.py:140` and `tabs/uniforms.py:61`. All three
  AGREE today, so this is drift risk rather than a live defect.
- **Live facts.** `dev_flow.md`'s Module map is a ~800-line closed-list inventory citing
  dozens of symbols and paths as current-tense claims. Spot-checked entries still resolve;
  nothing gates the section against the code it describes.

## The constraints, verified here rather than assumed

- **Verification is `make gates`** (check -> test -> smoke, stopping at the first failure,
  one exit code). It already warns on a non-tty stdout that a pipe reports the PIPE's
  status, so no wave needs to add that.
- **`make gates` autofixes and then reports failure.** pre-commit's formatter modifies
  files and exits non-zero for having done so, so a first run is "something changed" and
  only a second run is a verdict — and the exit code of that second run describes the
  REPAIRED tree, not necessarily what is staged. Commit, then re-run.
- **A mutation run needs `PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider`.** A stale
  `__pycache__` makes a caught mutation look like a vacuous gate. The Makefile sets this;
  a bare `uv run pytest` does not, and this repo has been misled by it.
- **Gates naming a source path** are found with
  `grep -rn 'Path("shaderbox/' tests/`. Those break on a file move and must be updated in
  the same commit as one.
- **`vulture` is NOT installed** and pyright's config does not enable unused-symbol
  reporting, so a dead-code wave is grep-and-reference based. `ruff F401,F841` is clean.

## The waves

One wave, one commit, `make gates` green at each. Ordered so measurement precedes change.

- **W-0 — Inventory. NO CHANGES.** Enumerate every instance of the vacuous-check class,
  by BREAKING rather than reading: for each candidate, keep the names and strings and
  destroy what the code decides, run the covering test, restore, verify with
  `git diff --quiet`. This wave sizes the rest, so nothing after it can be scoped first.

  **Ask what previous sweeps already exhausted** before spending the night on dead symbols:
  the roadmap and git log answer it in a minute, and a category swept twice returns almost
  nothing the third time. The yield here is expected to sit in what reference-scanning
  cannot see — checks that pass whether or not they work.
- **W-1 — The vacuous checks.** Strengthen each so the break that defeated it now fails.
  A check whose RULE turns out not to be worth holding is deleted rather than padded, and
  its rule is either restated where a human will read it or dropped deliberately.
- **W-2 — Removal of what is no longer reached.** Remove what W-0 confirmed, by risk tier: unreferenced private
  helpers first, then anything reachable dynamically only after proving it dead, and leave
  a public or documented surface alone.
- **W-3 — The repeated decision.** Point the two inline sites at `instanced_pass_keys`.
  Behaviour-preserving by construction; if a test expectation changes, the refactor is
  wrong and gets reverted.
- **W-R — Live facts.** Delete the facts that go stale on their own. A stale number found
  mid-wave is DELETED, never corrected — correcting it buys one commit and re-arms the
  trap. Frozen history stays.

## How correctness is decided

`make gates`, exit code read unpiped, green before and after every wave.

**A changed test expectation in a structural wave is a defect in the refactor, not a test
to update.** The tests are the only evidence behaviour was preserved; weakening one to get
a green destroys what the green was supposed to mean. The single legitimate edit is a test
moving with the code it covers.

## Coverage claims

From the presence scans, verbatim in the form each returned:

- `scanned: functions, classes, enum members, imports (F401), locals (F841), dependencies —
  across shaderbox/, tests/, scripts/, dogfood/; not scanned: per-method usage,
  dataclass/pydantic fields individually, type aliases, whole-module reachability`
- `scanned: the entity_fields/instanced predicate, script-tab identity, reserved-uniform
  partitioning, non-landing script keys, canvas-size and timeout defaults; not scanned:
  copilot prompt-tier and tool-schema contracts, graph_canvas ctypes ABI contracts, the
  export and Telegram paths, imgui widget-tier duplication`
- `scanned: tests/test_intel_cache.py (1 break confirmed vacuous), a grep survey of
  predicate-style assertions across tests/, Makefile gate structure; not scanned: the
  rest of the test files' internals, module-level asserts in shaderbox/, pre-commit
  ignore patterns`
- `scanned: CLAUDE.md, conventions.md, dev_flow.md, roadmap.md, todo.md, README.md, and
  code comments repo-wide; not scanned: the ~40 feature specs individually, BUILDING.md`
- `scanned: shaderbox/ top level, scripting/, theme/syntax_colors, popups/, exporters/,
  intel/, editor/ import edges, full wc -l of shaderbox+tests; not scanned: copilot/
  internals, graph_canvas/ffi.py, abi_probe.py, tabs/, widgets/, glsl_docs.py contents`
- `scanned: the recent commit window, subjects and full bodies (git log --format='%h %s%n%b');
  not scanned: diffs, commits before that window, merged trees of merge commits`

## What looks wrong and is CORRECT

A wave proposing to change any of these must first explain how the thing it exists for
still works.

- **`inspect.getsource` in three test files is fine.** `test_button_tiers.py` and
  `test_modal_chrome.py` parse it into an AST and ask a structural question ("does this
  function call X"), which text search cannot fake. `test_probe_clock_and_turn_end.py`
  pairs its source check with a real behavioural assertion. **A blanket ban would flag
  correct code** — the dangerous shape is narrower: a POSITIVE presence assertion on
  Python source standing in for behaviour. Absence assertions are fine too; a deleted
  thing has no runtime to drive.
- **Assertions on GENERATED TEXT are the right thing to assert.** `test_copilot_passes.py`
  and `test_working_set.py` check what the copilot actually receives. That is the
  consumer's view, not a source search.
- **`scripting/engine.py` mentioning imgui and glfw in COMMENTS** is not a layering
  violation; the import sweep is clean and those lines explain the injected-callback seam.
- **`intel/document.py` importing the editor** is a legitimate consumer relationship.
  No convention bans that direction.
- **`document.py::as_canvas_size`** reads like a `graph_canvas` symbol misfiled by name;
  it is the persisted render-canvas size field and is correctly placed.
- **`exporters/` is not missing a plugin seam** — `Exporter(ABC)` plus `ExporterRegistry`
  already exist and are injected end to end.
- **The big files are not automatically wrong.** See
  `find shaderbox tests -name '*.py' | xargs wc -l | sort -rn | head -20`. The largest are
  the copilot backend, a vendored ABI probe and a GLSL documentation table. One file per
  concept is defensible; judge, do not split on principle.

## Cold start

**Re-measure before acting. Every example in "What is present" is illustrative and may
have moved, been fixed, or multiplied since this was written — re-derive the inventory
from the tree rather than working from this list.** A spec that lists defects becomes a
spec to fix the listed ones, when the instruction is to find every one.

W-0 enumerates by BREAKING, not by reading — a check's name says what its
author intended and only a mutation says what it catches.

Settled already, as CONSTRAINTS rather than options:

1. The vacuous-check class is the night's subject. The structural questions came back
   ABSENT and are not re-opened.
2. A blanket ban on `inspect.getsource` is REJECTED, for the reason above.
3. A stale fact is deleted, not corrected.
4. A test expectation that changes during a structural wave means the refactor is wrong.
5. `PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider` on every mutation run.
