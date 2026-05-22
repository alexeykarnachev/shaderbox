# Feature 007 — release-pipeline hardening

## Goal

Close the "silently ship something broken / unversioned" gaps in the itch.io delivery path,
surfaced by the 2026-05-22 three-agent review. Concretely:

1. **A real release flow** — `make release VERSION=x.y.z` bumps `pyproject.toml`, commits, and
   `git tag`s, so every itch build maps to a version and a commit (today everything is `0.1.0`,
   never bumped, no tag).
2. **A build gate** — `build.sh` refuses to build unless `make check` + `make smoke` pass AND the
   git tree is clean (`--allow-dirty` overrides). Today `build.sh` bundles the working tree with no
   test gate, so a dirty/broken tree can become the shipped bundle.
3. **Consistent archive format** — both platforms ship `.zip` (today: Windows `.zip`, Linux
   `.tar.gz`); this matches the existing itch Linux-download label, costs nothing, and removes the
   drift the pipeline never validated.
4. **A Windows-build path that's clear from the repo** — a GitHub Actions `windows-latest` smoke
   job + a `docs`/README section so the maintainer can point a Windows coding-agent at the repo and
   verify the build there. (The maintainer has a Windows box; this prepares it, doesn't replace the
   manual test.)

## Out of scope

- **Auto-deriving the version from git tags** (`git describe` / dynamic version). Decided: manual
  semver bump. Trigger to revisit: if manual bumps are repeatedly forgotten before a release.
- **Code-signing the Windows launcher** (SmartScreen). Has a concrete fire-able trigger, so it
  lives in `todo.md [DEFERRAL]`, not here. The bundled README's "More info → Run anyway" note is
  the current mitigation (bundle-UX edits, this session).
- **Auto-posting itch devlogs from the pipeline.** Trigger: when releases get frequent enough that
  manual devlog posts get skipped. A `CHANGELOG.md` is in scope only if it falls out cheaply (it
  doesn't here — deferred).
- **`butler --if-changed` / verify-after-push / dropping the redundant `$?` check** in
  `upload-itch.sh` — cosmetic, low-value; not worth the diff this round.
- **Whether `smoke.py` actually runs on a headless Windows CI runner** — this is an open risk
  (glfw needs a GL context; GitHub windows-latest runners may lack one). The CI job is written to
  *attempt* `uv sync` + `make smoke`; if smoke can't run headless on Windows, the job degrades to
  `uv sync` + an import check (still catches the most likely Linux-built-Windows-bundle failure:
  a missing/uninstallable dependency). See Design decision 5.

## Design decisions

1. **Semver policy (manual).** Canonical home: `conventions.md ## Design decisions` (the
   major/minor/patch bump rule + revisit trigger live there — always-loaded, future-constraining).
   The mechanics this feature adds: the version lives in `pyproject.toml` and `make release
   VERSION=x.y.z` is the only sanctioned bump path (not auto-derived).

2. **`make release VERSION=x.y.z`** does, in order, aborting on any failure:
   - Validate `VERSION` matches `^[0-9]+\.[0-9]+\.[0-9]+$` (semver core; no pre-release suffixes for
     now — keeps the validation honest rather than permissive).
   - Refuse if the git tree is dirty (release must be reproducible from a commit).
   - Refuse if tag `v$VERSION` already exists.
   - Rewrite the `version = "..."` line in `pyproject.toml` with an **anchored** `^version = `
     match (the existing `upload-itch.sh::get_version` idiom — `target-version` / `requires-python`
     don't match an anchored `^version`, so it's unambiguous; verified one matching line).
   - `uv lock` is NOT needed (version bump of the project itself doesn't change the dep graph) — but
     run `make check` after the edit to catch a malformed toml. NOTE: `make check` only guards TOML
     *syntax* (e.g. an unbalanced quote from a bad sed); the regex pre-validation is what guards the
     version *value*. If `make check` fails after the rewrite, **roll the edit back**
     (`git checkout -- pyproject.toml`) before exiting so a failed release leaves no half-edit on
     disk.
   - Commit (`release: vX.Y.Z`) and `git tag vX.Y.Z`.
   - Print the next step (`./build.sh` then `./upload-itch.sh`); does NOT build or push (those stay
     separate, maintainer-triggered, so a tag can be cut without an immediate upload).
   Implemented as a `Makefile` target shelling to a small inline script (kept in the Makefile, not a
   separate `scripts/release.sh`, to keep the release surface in one obvious place — revisit if it
   outgrows ~25 lines). **Makefile-recipe footgun:** use `.ONESHELL:` so the multi-step recipe runs
   in one shell with `set -e` propagating across steps; escape shell `$` as `$$`; this is the
   highest-probability implementation bug for a multi-line release recipe.

3. **Build gate in `build.sh`** (added at the top, after `cd "$ROOT"`):
   - Parse a `--allow-dirty` flag (the ONLY flag; anything else is unchanged positional behavior).
   - Unless `--allow-dirty`: abort if `git status --porcelain` is non-empty, naming the dirty files.
   - Always: run `make check` and `make smoke`; abort the build on either failure. (`make smoke`
     needs a real GL context — it runs fine on the Linux dev box where builds happen; if a future
     headless build host lacks GL this becomes a documented limitation, not silent.)
   - The existing `verify_clean` allowlist gate stays exactly as-is (it's solid).

4. **Archive format → `.zip` for both.** Replace the Linux `tar -czf …tar.gz` with `zip -rq
   …linux.zip`, mirroring the Windows branch. Every reference to flip (full list — the implementer
   greps `tar.gz` to confirm zero remain):
   - `build.sh` — the `tar -czf` line + its success echo.
   - `upload-itch.sh` — **4 sites**: `check_distributions`, the `main` `print_status` label, the
     `main` `upload_to_itch` call (Linux), and the `--dry-run` echo.
   - `ai_docs/dev_flow.md ## Recipes > Build / ship to itch.io`.
   - `ai_docs/itch_page_copy.md` — **both** the Quick-start body line ("Windows .zip / Linux
     .tar.gz") **and** the metadata note.
   The itch page's existing Linux label already says `.zip`, so this REMOVES drift, not adds it.

5. **Windows CI** — `.github/workflows/ci.yml`, a `windows-latest` job: checkout → install `uv`
   (`astral-sh/setup-uv@v5` — **pinned**, no `@main`/floating) → `uv sync` → attempt smoke (via
   `uv run python scripts/smoke.py`; `make` may be absent on the runner, so call the script
   directly, not the Make target). `continue-on-error: true` goes on the **smoke step**, NOT the job
   — a failed `uv sync` (uninstallable dep on Windows) is the high-value signal and must red the
   job; a failed headless-smoke is logged but tolerated (the GL-context uncertainty). A short
   comment on that step points at this decision so a fresh agent reading green-CI-with-skipped-smoke
   doesn't rediscover it as a bug. A second `ubuntu-latest` job runs literally `make check` (make is
   present on ubuntu-latest; pre-commit pulls hook envs over the network on first run — caching is
   optional, not required) so CI also guards the thing that already works. **Cycle-from-types-style
   anticipation:** smoke may simply not run headless on Windows CI — the job degrades legibly rather
   than being retro-fitted when it fails. A `todo.md [DEFERRAL]` records the soft-fail with a
   trigger ("next time you touch `ci.yml` or smoke fails to run headless on Windows").

6. **Windows-build clarity from the repo.** A short top-level `BUILDING.md` (NOT shipped in the
   bundle — it's never copied into the stage, but add it to `FORBIDDEN_NAMES` as defense-in-depth so
   the clean-bundle gate's "asserted not assumed" promise holds for a future leak) describing: the
   build is normally Linux→both-platforms; to verify on a real Windows box, clone, `uv sync`, run
   `scripts\run.bat` (or `uv run python ./shaderbox/ui.py`), and exercise the manual-verification
   checklist. This is what the maintainer points the Windows coding-agent at. **Also add `.github`
   to `build.sh`'s `FORBIDDEN_PATHS`** — a new top-level dir the bundle never copies, but the
   defense-in-depth gate should still name it (matching the `ai_docs`/`.claude` entries).

## Files touched

- `Makefile` — new `release` target (+ `.PHONY`). ~20 lines.
- `build.sh` — `--allow-dirty` parse, dirty-tree guard, `make check`+`make smoke` gate (top);
  Linux archive `tar.gz`→`zip` (bottom); echo text. Add `BUILDING.md` to `FORBIDDEN_NAMES`.
- `upload-itch.sh` — Linux file path/label `tar.gz`→`zip` (`check_distributions`, `main`,
  `--dry-run`).
- `pyproject.toml` — no manual edit now; `make release` rewrites the `version` line. (Listed because
  the release flow's contract is "this file's version line is the bump target".)
- `.github/workflows/ci.yml` — NEW. Windows smoke + Ubuntu check jobs.
- `BUILDING.md` — NEW (repo root, not bundled). Windows-verification instructions.
- `conventions.md ## Design decisions` — one new bullet: the semver bump policy + revisit trigger.
- `ai_docs/dev_flow.md ## Recipes > Build / ship to itch.io` — update the archive names (`.zip`
  both) + mention `make release` as the version-bump entry point and the build gate.
- `ai_docs/itch_page_copy.md` — drop the now-resolved `.tar.gz`-vs-`.zip` note (both are `.zip`).
- `ai_docs/roadmap.md` — feature 007 row + banner rewrite (release flow exists; next = cut a
  version + ship). Done-step, not impl.

## Manual verification

- **`make release` happy path (on a CLEAN tree).** `make release VERSION=9.9.9` → `pyproject.toml`
  shows `9.9.9`, a `release: v9.9.9` commit + tag `v9.9.9` exist. Confirm it ran `make check` but
  **NOT** smoke (no glfw window opened) and did **NOT** build or push (`dist/` mtime unchanged, no
  build/butler output). **Undo** (clean tree only — `reset` anchored to the tag, never `HEAD~1`):
  `git reset --hard v9.9.9~1 && git tag -d v9.9.9`. ⚠️ `reset --hard` discards uncommitted work — do
  this test only with a clean tree, which the forward path already required.
- **`make release` rejection paths.** `make release VERSION=bad` aborts (regex). Re-running an
  existing version aborts (tag exists). On a dirty tree, aborts before editing `pyproject.toml`
  (no half-edit left on disk).
- **Build gate is NOT a no-op (the highest-value test).** Introduce a deliberate syntax error in a
  `.py` file, run `./build.sh`, confirm it **aborts before `dist/` is produced** (proves the gate
  honors `make check`'s exit code); revert the error.
- `./build.sh` on a clean tree → runs check+smoke, produces `dist/shaderbox-windows.zip` +
  `dist/shaderbox-linux.zip`, both verify-clean. On a dirty tree → aborts naming the files;
  `./build.sh --allow-dirty` proceeds.
- Extract `dist/shaderbox-linux.zip` → top-level `shaderbox-build-linux/` with `run.sh` executable,
  no dev files. `unzip -l` shows no `ai_docs/`, `.git`, `BUILDING.md`, `Makefile`, `.github`.
- `./upload-itch.sh --dry-run` → references `shaderbox-linux.zip`, not `.tar.gz`; `grep -r tar.gz`
  across `build.sh upload-itch.sh ai_docs/` returns only the unrelated design-handoff `.gitignore`
  line.
- CI: push the branch, confirm the Ubuntu job runs literally `make check` and passes; read the
  Windows job log to see whether `uv sync` succeeded (hard gate) and whether smoke ran or degraded
  (soft) — record the smoke-on-Windows outcome in the `todo.md` deferral / roadmap banner.

## Open questions for the user

(None blocking — decisions 1-6 reflect the choices already given: manual semver, gate + `--allow-dirty`,
`.zip` both, prepare-Windows-CI-and-docs-don't-replace-manual-test. Surfaced at plan-lock for sign-off.)

## Review history

**Pre-impl, 2 parallel reviewers (2026-05-22), both PARTIAL, no blockers.** Findings folded in:
- Build gate could be a silent no-op → added the negative "break a file, confirm build aborts" test.
- `make release` undo used `reset --hard HEAD~1` (eats uncommitted work) → anchored to `v9.9.9~1`
  with a clean-tree caveat.
- Makefile multi-step recipe `$$`-escaping / `.ONESHELL` flagged as the likeliest impl bug → noted
  in decision 2.
- No rollback of the version edit on post-rewrite `make check` failure → `git checkout -- pyproject.toml`.
- `itch_page_copy.md` has TWO `.tar.gz` refs (body + note), not one → both listed in decision 4.
- `.github/` not in `FORBIDDEN_PATHS` → added (defense-in-depth).
- `setup-uv` unpinned → pinned `@v5`; `continue-on-error` must be step-scoped not job-scoped.
- smoke-on-Windows soft-fail could be rediscovered → ci.yml comment + `todo.md` deferral.
- No verification `make release` skips build/push/smoke, nor CI-is-literally-`make check` → added.
- Rejected (false positive): bundled `scripts/README.md` "stale archive ref" — it names no archive
  format (verified clean).

**Post-impl, 3 parallel reviewers (2026-05-22): spec-fidelity PASS, correctness PASS, conventions
PARTIAL.** Three real findings fixed: (1) `zip -rq` appends to a stale `/tmp/*.zip` from an
interrupted prior run → `rm -f` before each zip in `build.sh`; (2) the semver bump rule was
paraphrase-copied in this spec + `conventions.md` → conventions.md is now the single canonical home,
decision 1 points to it; (3) the `ci.yml` soft-fail comment pointed at this spec → repointed to the
live `todo.md` deferral. Skipped: semver regex accepts leading zeros (cosmetic, hand-typed).
