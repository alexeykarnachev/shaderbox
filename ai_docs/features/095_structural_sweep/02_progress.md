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
its new home; no module's `xdist_group` changed; `grep -rln "poisons the process's EGL display"
tests/` returns only the shared definition plus any deliberate variant; `make gates` green.

verification: green (check, test, smoke).

Five of the six copies were byte-identical or differed only in an extra comment; they now use the
shared module-scoped fixture in `tests/conftest.py`. The comment recording the EGL segfault moved
with it, expanded into a docstring that also states why the scope is per-module (the modules run
in their own xdist processes, so a session scope would share a context across files partitioned
apart on purpose). Ruff removed the imports the deletions orphaned.

**`tests/test_profiling.py` keeps its own copy, deliberately.** Its fixture binds a
`simple_framebuffer` after creating the context — different behavior, not duplication. Collapsing
it into the shared one would have changed what that module runs against. This is the
"similar shape is not shared meaning" case, and it was caught by hashing each fixture body rather
than by reading them.

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
