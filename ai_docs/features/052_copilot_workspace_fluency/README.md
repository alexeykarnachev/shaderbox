# Feature 052 — Copilot workspace fluency (umbrella)

<!-- Umbrella spec. PLAN-LOCKED (2026-07-02) — all open questions resolved to the full/solid option
     (99_synthesis.md ## Locked resolutions). The maintainer slices this into per-slice tasks; each
     slice below is a buildable unit with its own doc. Next per dev_flow.md: pre-implementation review. -->

## What this is

The copilot today is fluent at GLSL logic, scripts, scalar/vector uniforms, and publishing — but it
is **blind to textures**, cannot touch **files** beyond the two text files of a node, and has **no way
to bring anything in from the user's disk**. This umbrella closes that: it makes the copilot literate
in the ShaderBox *workspace* (assets, node files, canvas) and gives it ONE safe, user-in-the-loop
way to interact with the user's filesystem.

The whole design hangs off the copilot actor-model (`.claude/skills/copilot-llm-agent-design`):
- **Corollary 2 (blind outside its stream):** if a texture / canvas size / binding is not serialized
  into the working set, it does not exist for the model. → the *awareness* slice.
- **Corollary 1 (copies, never synthesizes):** a filesystem path is exactly what the model must NOT
  type. → every "bring a file in" op routes through the **user's own OS file picker**; the model
  triggers the dialog, the user chooses, the model never sees or authors a path. Same shape as the
  credential gate (`GateKind.CREDENTIAL`).

## The slices

| # | Slice | Doc | New tools | Depends on |
|---|---|---|---|---|
| 0 | **Lazy tool catalogue** (the parked D5 lever) — pays for the new tool surface | `05_lazy_tool_catalogue.md` | — | — |
| 1 | **Awareness** — samplers + bindings + canvas + control ranges in the working set (read-only, no new tools) | `01_awareness.md` | — | — |
| 2 | **Media literacy** — `bind_media` (opens the user file picker, gate-family); `unbind_media` = reset to default | `02_media_literacy.md` | `bind_media`, `unbind_media` | 0, 1 |
| 3 | **Node file ops** — rename / duplicate / canvas size + lib-file delete | `03_node_file_ops.md` | `rename_node`, `duplicate_node`, `set_canvas_size`, `delete_lib_file` | 0 |
| 4 | **External import** — bring a `.glsl` / node in from disk via the picker | `04_external_import.md` | `import_node` | 0, 2 (shares the file-pick primitive) |

`00_grounding.md` = the current-surface audit + on-disk reality + the two hard constraints every slice
inherits (all-eager tools today; checkpoint scope-9 names `bind_media` as a re-verify trigger).

## Build progress

Implementation order was REVERSED from the spec's recommendation after a fresh look: **capability
slices first, infra (slice 0) last**. Rationale: slice 0 mutates the core turn-loop (highest-risk
file) and its payoff scales with the number of lazy tools, so it's best landed once against the full
lazy set; the new tools ship eager meanwhile (spec-sanctioned — "land slice 0 first, OR accept the
per-turn token tax"). Contained capability slices deliver value at lower risk.

- [x] **Slice 1 — awareness** — sampler bindings + canvas in the working set; `ui_models.save`
  skips a default sampler (app-wide, round-trips). `make check` clean; tests green (save-skip
  round-trip, awareness rows, no-path-leak).
- [x] **Slice 3 — node file ops** — `rename_node` / `set_canvas_size` / `duplicate_node` /
  `delete_lib_file` (ALWAYS-gated, trash-recoverable, checkpoint-restore) all LANDED (backend + tools
  + `_caps` + `CANONICAL_TOOLS`). Real methods verified in-env via standalone-context stubs; app-fixture
  tests run in CI. `make check` clean.
- [x] **Slice 2 — media literacy** — `bind_media` (own `GateKind.FILE` slot: worker blocks in
  `ask_file`, `ui.py::_pump_file_gate` opens `pfd` non-blocking across live frames, binds on main via
  `bind_picked_media`, answers path-free) + `unbind_media` (reset to default). The abs path lives only
  in the poll + `bind_picked_media(path)`, never crosses to the worker/model. Verified in-env: gate
  slot roundtrip/cancel/independence, corollary-1 no-path-leak (sentinel-dir assert), bind/unbind
  behavior — 86 passed, `make check` clean. **Known edge:** if the user leaves the native dialog open
  after a turn Stop, a new turn's file gate waits until they close it (native dialogs aren't
  programmatically closable); the late pick is dropped safely.
- [x] **Slice 4 — external import** — `import_node` reuses the FILE gate (`file_action="import_node"`,
  `file_kinds=("glsl",)`); the poll reads the file on main via `import_picked_node` → the shared
  `_create_node_on_main` (extracted from `create_node`), answers a path-free `NodeImportResult`. A
  broken import still creates the node + returns compile errors. Verified in-env (creates a compiling
  node, sentinel-dir absent from result + msg). `make check` clean.
- [ ] **Slice 0 — lazy tool catalogue** (`load_tools` + demote the new + 7 integration tools) — LAST.

## Slicing guidance (for turning this into tasks)

- **Slice 0 and 1 are independent** and can land first, in either order — 1 is pure read enrichment
  (zero risk), 0 is pure infra (unblocks the tool budget).
- **Slice 2 is the headline** and the reason the feature exists; it introduces the reusable
  **user-file-pick primitive** that slice 4 reuses.
- **Slice 3 is a bag of small independent tools** — each is a task on its own; do them in any order.
- Don't ship all 7 tools eager: land slice 0 first, or accept the per-turn token tax (quantified in
  `05_lazy_tool_catalogue.md`).

## Status

PLAN-LOCKED (2026-07-02); **pre-impl review CONVERGED after 3 rounds** (code-anchored adversarial
reviewers; rounds 1-2 refuted 4 built-on premises across my own drafts+fixes, round 3 clean — see
`99_synthesis.md` Review history). No code written yet. **Next: the maintainer slices the lanes into
tasks** (each slice re-reviewed at its own impl per `dev_flow.md`).

## Tool surface added (7 new tools, all lazy)

`bind_media`, `unbind_media`, `rename_node`, `duplicate_node`, `set_canvas_size`, `delete_lib_file`,
`import_node` — plus the eager `load_tools` meta-tool (slice 0) and the read-only awareness enrichment
(slice 1, no tool). (`rename_lib_file` was CUT in round-1 review — derivable + revert-less; see
`99_synthesis.md` Review history.)
