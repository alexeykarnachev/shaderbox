# 051 — Shipped examples project (fire showcase)

> **STATUS: DRAFT — NOT plan-locked, do NOT implement yet.** Parked behind an unresolved
> architectural problem (the example↔shader-library coupling — see `todo.md` "[BLOCKER]
> example/project shader-library coupling"). The maintainer reframed this mid-draft (2026-06-17): the
> examples feature is bigger than first scoped — it wants a first-class **Examples picker** (a main-bar
> button → modal: project list on the left, description + embedded preview/video on the right, like a
> game-engine example browser, for onboarding) AND it cannot ship until the local-vs-global library
> question is answered (an example pinned to today's `SB_*` lib silently breaks when the lib evolves).
> The seed/bundle grounding below is still valid and worth keeping; treat it as research, not a locked
> plan. Revisit "in a couple of days" per the maintainer. The open questions + the new picker scope are
> NOT yet decided. The flame showcase node stays in the gitignored `projects/_lab/fire/` until then.

## Goal

Ship one read-only **examples project** with the app — always present on disk after launch, restorable if the user deletes or corrupts it, mirroring the existing shipped-shader-lib seed mechanism (`shaderbox/shader_lib/seed.py`) at **project/node-dir granularity** instead of per-`.glsl`-file granularity. Its sole entry is the polished step-by-step **fire showcase** node built this session (the 17s timed-reveal "fire (timed reveal)" node, currently at the gitignored `projects/_lab/fire/nodes/0b0d16bb-f014-4a85-b155-6be74c33eded/`). The `/shader-lab` skill gains a canonical, on-disk, inspectable reference effect to point sessions at (it has none today — `.claude/skills/shader-lab/SKILL.md` only flags "promote a node dir out of `projects/_lab/`" as an open follow-up).

The examples project must:
- be copied from `shaderbox/resources/example_projects/<slug>/` into the user's writable area at startup (so it can be opened + inspected like any project),
- track shipped content updates while pristine, never clobber a node the user edited, and stay gone if the user deleted it (the lib-seed intent contract, at node granularity),
- be reachable from the existing project-open surface (`App.open_project`) without inventing a parallel UI mechanism.

## Out of scope

- **Multiple example effects.** Only `fire` ships now. Trigger: a second showcase node graduates out of a `/shader-lab` session and the maintainer asks to add it — drop its `<slug>/` dir under `shaderbox/resources/example_projects/` and the same seed walks it; no code change.
- **A dedicated "Examples" menu / gallery UI.** The smallest real surface is: the seeded project folder is openable through the existing folder dialog (`App.open_project`), and a one-line `/shader-lab` reference. Trigger: the maintainer reports the open-folder dialog is too obscure to discover the examples in a real fresh-install test → add an explicit "Open examples" menu entry pointing at the seeded examples root.
- **A "Reset examples to shipped" UI button.** The shader-lib has `reset_to_shipped` wired to a factory-reset affordance; the examples project gets the equivalent function (`reset_examples_to_shipped`) but NO button this iteration — deletion-triggered reseed (delete the project, relaunch) is the primary restore path. Trigger: a user edits the example, wants the pristine copy back without deleting the whole folder, and asks for it.
- **Auto-opening the examples project at startup / making it the default project.** First launch keeps seeding the empty `default` project with a starter node. The examples project is seeded-but-not-active. Trigger: the maintainer decides a fresh install should land *in* the fire showcase instead of an empty default.
- **Backward-compat / migration of any kind.** Per `conventions.md ## Design decisions`, there is no old examples format to migrate from. N/A.
- **Shipping the exploration nodes (v01–v11).** `projects/_lab/fire/nodes/` holds 11 node dirs; only the final showcase (`0b0d16bb-…`) ships. The other 10 + `NOTES.md` are NOT copied into the shipped resource. N/A (never enters scope).

## Design decisions (lock-in)

1. **Shipped location: `shaderbox/resources/example_projects/<slug>/`**, mirroring `shaderbox/resources/shader_lib/`. The fire slug ships exactly: `example_projects/fire/app_state.json` + `example_projects/fire/nodes/0b0d16bb-…/{node.json, shader.frag.glsl}`. Add `EXAMPLE_PROJECTS_SEED_DIR = RESOURCES_DIR / "example_projects"` in `shaderbox/constants.py` next to `SHADER_LIB_SEED_DIR`. Because the whole `shaderbox/` package is copied by `build.sh::stage_common` (`cp -r shaderbox`), anything under `shaderbox/resources/` ships automatically (decision 8).

2. **Seed into the user's WRITABLE area, NOT opened read-only in place.** The app re-saves project state on the way out (`App._init` → `session.load`; node.json/app_state drift every run). Opening the read-only bundle tree directly would make the app write back into a read-only install path and a half-mutated examples folder that can't be restored — a footgun. So the seed COPIES the shipped tree into a writable examples root (decision 4), and a deleted/edited copy is the restorable unit. Matches the lib seed: canonical content ships read-only under `resources/`, the live copy is user-writable.

3. **User-intent contract at PROJECT + node-dir granularity** (the per-`.glsl` contract of `_sync_shipped_lib`, lifted to dirs): a manifest maps each shipped *node-dir rel-path* (e.g. `fire/nodes/<uuid>`) and the project `app_state.json` rel-path to a content hash of the shipped version it came from.
   - **New install / new shipped node** (target missing, rel-path NOT in manifest) → seed it.
   - **User deleted it** (target missing, rel-path IN manifest) → stays deleted, never resurrected.
   - **Pristine** (on-disk hash == manifest hash, and shipped changed) → follow the shipped update.
   - **Edited** (on-disk hash != manifest hash) → never touched; logged.
   - A node-dir is hashed as the hash of its files' bytes combined by sorted rel-path (the dir is the atom — node.json + shader.frag.glsl + any scripts/ together), so editing the shader OR the metadata marks the whole node "edited". The fire node has no `scripts/` today, but the hash walk includes scripts/ for future nodes.

4. **Live examples root: `app_data_dir() / "example_projects"`**, each project a subdir (`app_data_dir()/example_projects/fire/`). Add `paths.py::example_projects_root() -> Path` mirroring `shader_lib_root()` — `mkdir(parents=True, exist_ok=True)`, rooted at `app_data_dir()` so it honors `SHADERBOX_DATA_DIR`. It is a SIBLING of `app_data_dir()/projects` (the default user projects root), NOT inside it — keeping seeded examples out of the user's own projects listing and giving the deletion-stays-deleted manifest a clean home. Each seeded example IS a self-contained relocatable project dir.

5. **Manifest: `.seed_manifest.json` at the examples-root** (`app_data_dir()/example_projects/.seed_manifest.json`), same filename and `{rel-path: sha1}` JSON shape as the lib seed (`json.dumps(..., indent=2, sort_keys=True)`). The leading dot keeps it out of any project glob. Keys are POSIX rel-paths from the examples-root to each node-dir and each project-level file (`fire/app_state.json`, `fire/nodes/<uuid>`). The same corrupt-manifest guard as `_sync_shipped_lib` (reject an absolute or `..` key before unlinking) applies — a stale node-dir is removed only if it still matches its recorded hash.

6. **New module `shaderbox/example_projects/seed.py`** (a new package dir `shaderbox/example_projects/` with `__init__.py` + `seed.py`), exposing `sync_shipped_examples(seed_dir, root) -> int` and `reset_examples_to_shipped(seed_dir, root) -> tuple[int, int]`, structurally mirroring `sync_shipped_lib` / `reset_to_shipped`. Imports only stdlib + loguru (no `App`, no imgui) — same posture as `shader_lib/seed.py`. A sibling package rather than overloading `shader_lib/seed.py` because the lib seed globs `*.glsl` flat; the examples seed walks dirs and hashes node-dirs as atoms — a parallel *implementation* of the same *contract*, not a parallel *mechanism* (the contract is identical and that is what the constraint protects).

7. **Wire-up: one call in `App.__init__`, right after the existing `sync_shipped_lib` line.** Add `sync_shipped_examples(EXAMPLE_PROJECTS_SEED_DIR, example_projects_root())` immediately after `sync_shipped_lib(...)`. Runs once at startup before any project loads, fail-soft on `OSError` exactly like `sync_shipped_lib`. No `ProjectSession` change — the examples project is a folder the user opens via the unchanged `App.open_project`.

8. **Bundle: NO new `build.sh` line.** `stage_common` does `cp -r shaderbox "$stage/"`, so `shaderbox/resources/example_projects/**` and the new `shaderbox/example_projects/` package ship automatically. The shipped example's files (`app_state.json`, `node.json`, `shader.frag.glsl`) match NO `FORBIDDEN_NAMES` and NO `FORBIDDEN_PATHS`. Only the single showcase node dir + a minimal `app_state.json` are copied into `shaderbox/resources/example_projects/fire/` — no v01–v11 nodes, no `NOTES.md`, no `copilot/`/`exporter_scratch/`/`renders/`/`trash/`.

9. **The glyph dependency rides the EXISTING shader-lib seed, not this one.** The showcase shader calls `SB_sd_char` / `SB_fbm` / `SB_hash21` / `SB_center_uv` — lib functions auto-injected by `ShaderLibIndex` (no `#include` directive). `SB_sd_char` lives in `shaderbox/resources/shader_lib/text/glyphs.glsl` (regenerated this session with the math-symbol glyphs the captions use), which already ships via `sync_shipped_lib`. The examples seed copies an opaque node-dir; the shader compiles against whatever lib index is live — no dependency edge to manage here. The node's `node.json`/`shader.frag.glsl` carry no absolute paths (location-independent). **Consequence to verify:** the showcase renders correctly only if the shipped `glyphs.glsl` is also current — but that is the lib seed's job, already wired.

## Open questions (for the user)

1. **Slug naming + display.** The user opens a folder named `fire` under `example_projects/`. Is `fire` the intended user-facing name, or should the shipped `app_state.json` carry a friendlier title? (No project-title field is surfaced in the open dialog today — the folder name IS the identity.)
2. **Node-dir hash atom — include `scripts/`?** Decision 3 includes `scripts/` so a future scripted example marks "edited" when its script changes. The fire node has none. Confirm vs. hashing only node.json + shader.frag.glsl.
3. **Ship `reset_examples_to_shipped` now (function only, no UI) or defer entirely?** Listed as shipped-but-unwired to mirror `reset_to_shipped`. Drop it from "Files touched" if you'd rather not carry unused code until there's a button.
4. **`/shader-lab` skill edit in this feature or separate?** The reference is a one-line pointer to the seeded + shipped examples paths.

## Files touched

- `shaderbox/example_projects/__init__.py` (new) — package marker.
- `shaderbox/example_projects/seed.py` (new) — `sync_shipped_examples` + `reset_examples_to_shipped` + node-dir-atom hashing + manifest read/write, mirroring `shaderbox/shader_lib/seed.py`.
- `shaderbox/resources/example_projects/fire/app_state.json` (new) — minimal shipped project state, `current_node_id` = the showcase UUID.
- `shaderbox/resources/example_projects/fire/nodes/0b0d16bb-…/node.json` (new) — copied from the lab node, no absolute paths.
- `shaderbox/resources/example_projects/fire/nodes/0b0d16bb-…/shader.frag.glsl` (new) — copied from the lab node.
- `shaderbox/constants.py` — add `EXAMPLE_PROJECTS_SEED_DIR`.
- `shaderbox/paths.py` — add `example_projects_root() -> Path`.
- `shaderbox/app.py` — import the seed fn + constant + root accessor; add one call after the `sync_shipped_lib` line.
- `.claude/skills/shader-lab/SKILL.md` (pending open question 4) — add the canonical-reference pointer.
- `ai_docs/conventions.md` — one-line Design-decision: shipped example projects mirror the shader-lib seed at node-dir granularity.
- `ai_docs/roadmap.md` — add a 051 row.

(NOT touched: `build.sh` — decision 8.)

## Manual verification

Each check drives the CONSUMER (a launched app / a fresh data dir), not the seed function in isolation. Use `SHADERBOX_DATA_DIR=$(mktemp -d)` for a throwaway fresh-install root.

1. **Fresh install seeds the examples project.** Empty `SHADERBOX_DATA_DIR`, launch (`SHADERBOX_DATA_DIR=/tmp/sbX timeout 12 uv run python ./shaderbox/ui.py`), exit. Assert `/tmp/sbX/example_projects/fire/nodes/0b0d16bb-…/shader.frag.glsl` exists and `.seed_manifest.json` lists `fire/nodes/0b0d16bb-…`. **Falsifier:** manifest/node absent → seed never ran or wrong root.

2. **Deleted example STAYS deleted across relaunch.** After check 1, `rm -rf .../example_projects/fire`, relaunch, exit. Assert `fire/` still absent. **Falsifier:** `fire/` reappears → seed treats "missing + in manifest" as "new install".

3. **Deleting the manifest TOO triggers a reseed.** After check 2, also `rm .../example_projects/.seed_manifest.json`, relaunch. Assert `fire/` restored. **Falsifier:** stays gone → the missing-and-unknown branch isn't seeding.

4. **An edited node is NOT clobbered.** After check 1, append a comment to the live `shader.frag.glsl`, relaunch, exit. Assert the appended line survives. **Falsifier:** reverts to shipped → seed overwrote an edited node-dir.

5. **A pristine node tracks a shipped update.** After check 1, edit the bundle copy `shaderbox/resources/example_projects/fire/nodes/0b0d16bb-…/node.json` (bump a uniform default), relaunch same `SHADERBOX_DATA_DIR`, exit. Assert the live copy reflects the change AND the manifest hash advanced. **Falsifier:** live copy stays old → pristine-tracks-update broken. (Revert the resource edit after.)

6. **The seeded project opens and the fire renders.** `make run`, File → Open the `example_projects/fire` folder, confirm by eye the timed reveal plays and the step captions render (proves the glyph dependency resolved via the shader-lib seed). **Falsifier:** captions blank → shipped `glyphs.glsl` stale or lib seed didn't run.

7. **Bundle ships the examples, no dev leak.** `make run-bundle` (rebuild, throwaway data dir). Assert the gate passes (`✓ Bundle verified clean`) AND the unzipped tree contains `shaderbox/resources/example_projects/fire/nodes/0b0d16bb-…/shader.frag.glsl` but NOT any v01–v11 UUID, `NOTES.md`, or `copilot/`. **Falsifier:** gate aborts OR an exploration node is staged.

8. **`make smoke` + `make check` green.** **Falsifier:** import error from the new package, or a pyright error → malformed wire-up.

## Unresolved core problem (blocks the whole feature)

**Example ↔ shader-library coupling.** Every example shader depends on `SB_*` lib functions
(the fire showcase calls `SB_sd_char`/`SB_fbm`/`SB_hash21`/`SB_center_uv`). The library is *not
frozen* — the maintainer keeps evolving it (this session alone added math glyphs to `glyphs.glsl`).
So a shipped example pinned to today's lib silently breaks when a future lib release changes/removes a
function it used, and hand-maintaining example↔lib compatibility forever is untenable.

Maintainer's leaning (NOT decided, NOT designed): support a **per-project (local) library** so each
project carries the lib version it was authored against, decoupling it from the evolving global lib.
Open and explicitly unsolved: where the local lib lives, what copies from where to where (global→
project? on project create? on example seed?), how the resolver chooses local-vs-global, and — the
hard constraint — doing it **robustly + ergonomically** (the user never thinks about local vs global)
**without complicating the code** (the maintainer flagged it's already getting tangled). This is the
real blocker; the examples feature waits on it. Filed as a standalone `todo.md` BLOCKER because it
affects ALL projects, not just examples.

## Examples picker (new scope, added 2026-06-17 — superset of the seed work above)

The examples shouldn't just be a folder you open via the file dialog — they should be a **first-class
onboarding feature**: a main-bar **"Examples"** button opening a **modal** — a project **list on the
left**, a **description panel on the right** (text + an embedded preview/video of the effect), pick a
project from the list to load it. Modeled on game-engine example-project browsers. This is a real UI
feature on top of the seed plumbing; spec it properly when the coupling problem is resolved.

## Review history

Drafted 2026-06-16 (seed mechanics, grounded in `shader_lib/seed.py`). Reframed 2026-06-17: the
maintainer halted implementation — the feature is parked behind the library-coupling blocker and
re-scoped to include the Examples picker. NOT plan-locked. The skill-harvest task (generalized
shader-dev lessons) was split out and done separately, NOT as part of this feature.
