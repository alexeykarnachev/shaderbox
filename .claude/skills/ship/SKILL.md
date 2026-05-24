---
name: ship
description: "The release-to-itch.io procedure — the single canonical home for shipping ShaderBox. Walks: sanitize → commit/push everything on dev → promote dev→master → auto-pick the semver bump → make release (tag) → gated build → butler upload to both channels → optional store-page sync → land back on a clean dev. The agent auto-picks the version (no VERSION arg needed). Triggers: 'ship', 'ship it', 'publish', 'release', 'cut a release', 'push a build', 'release to itch', 'new build', 'зарелизь', 'выкати', 'опубликуй'. DO NOT run any of this until the developer explicitly asks to ship/release/publish — it tags, pushes, and publishes a public build."
user_invocable: true
---

<command-name>ship</command-name>

The maintainer-triggered release flow that takes the current `dev` line to a live itch.io build and
leaves the tree clean on `dev`. This skill is the **single source of truth for the ship procedure** —
`BUILDING.md`, `ai_docs/dev_flow.md`, and `ai_docs/conventions.md` point here for the *how*; they keep
only the *why* (branch-model rationale, clean-bundle invariant, semver policy).

**NEVER run this unprompted.** It cuts a tag, pushes `master`, and publishes a public itch build —
all outward-facing and hard to reverse. Execute only when the developer explicitly says ship / release
/ publish (or runs `/ship`). When they do, that is your authorization for the whole sequence including
the itch upload; you do NOT need to re-ask before each push/upload. The one place you still stop for
review is the store-page Save (step 8) — publishing page copy stays assisted.

## The model (read once)

- **Develop on `dev`, ship from `master`.** `dev` accumulates freestyle commits with no version bump;
  `master` only advances at ship time and is the line released to itch. (Rationale lives in
  `dev_flow.md ## Branch model` — don't restate it elsewhere.)
- **`make release` cuts the tag only** — it bumps `pyproject.toml`, commits, tags `v<x.y.z>`, and runs
  `make check` (rolling back the version edit on failure). It does NOT build or push. Build + upload
  are deliberately separate steps so a tag can exist without an immediate publish.
- **The bundle is a source distribution** (`shaderbox/` + `uv.lock`; the user's machine runs `uv sync`
  + `uv run` via the launcher). `build.sh` is **gated**: it re-runs `make check` + `make smoke`,
  refuses a dirty tree, and aborts if any forbidden file leaks into the staged bundle (clean-bundle
  invariant — full allowlist in `dev_flow.md ### Build / ship to itch.io`).

## The sequence

Run top to bottom. Stop and report if any step fails — never paper over a failed gate.

### 1. Sanitize first
Run `/sanitize` (or its steps) so docs, roadmap banner, and todo are current BEFORE the version is
stamped. The release commit should ship already-swept docs, not a half-swept tree.

### 2. Confirm starting state
```
git rev-parse --abbrev-ref HEAD          # must be dev (if not, stop and ask)
git status --porcelain                   # see what's uncommitted
git log --oneline master..dev            # everything unshipped since last ship
```

### 3. Commit + push everything on `dev`
Stage all intended changes, commit with a single-line ASCII message (per `CLAUDE.md ## Commits`), push:
```
git add -A
git commit -m "<concise subject>"
git push origin dev
```
If the work spans multiple logical units, prefer separate commits; one is fine for a single feature.

### 4. Auto-pick the semver bump
Read `git diff master..dev` (everything since the last ship) and pick the bump yourself — the
developer does not supply a VERSION. Apply the major/minor/patch decision rule from its canonical
home (`conventions.md ## Design decisions` "Release versioning") and pick the **highest** tier any
unshipped change triggers. State the chosen version + the one-line reason
in your status before running the release. Current version is in `pyproject.toml`; last tag is
`git tag --sort=-v:refname | head -1`.

### 5. Promote `dev` → `master`
```
git checkout master
git merge --ff-only dev          # dev should be a strict descendant; if not, stop and investigate
```

### 6. Cut the release on `master`
```
make release VERSION=<x.y.z>     # bumps pyproject, commits "release: vX.Y.Z", tags vX.Y.Z, runs make check
```
**Footgun:** `make release` runs `make check`, which is `pre-commit run --all-files` — pre-commit
hooks (ruff-format, end-of-files) may *modify* files and exit non-zero on the first run after a change,
which rolls back the version edit. If that happens: the hooks already fixed the tree, so commit the
fixups (`git commit -am "style: ..."`) and re-run `make release`. To avoid it, ensure `make check` is
green with a clean tree BEFORE step 6.

### 7. Build, push master + tag, upload to itch
```
./build.sh                       # gated; produces dist/shaderbox-{windows,linux}.zip
git push origin master
git push origin v<x.y.z>         # plain `git push` does NOT push tags
yes | ./upload-itch.sh           # butler push to both channels; `yes |` clears the script's y/N confirm
```
Verify the upload registered (channels processing is async):
```
butler status where-is-your-keyboard/shaderbox    # both channels should show the new version
```
`build.sh --allow-dirty` skips only the dirty-tree guard, never the check/smoke gate — don't use it to
push past a real failure.

### 8. Store page (optional, last, review-gated)
Only if the page copy/tags/screenshots need to change for this release. The store page's authoring
surface is `ai_docs/itch/page.yaml`; itch has no write API, so the page is edited via an agent-driven
Playwright session. Full procedure + the hard-won footguns (the `<textarea>` vs redactor trap, the
Selectize tag API, stale snapshots) live in `dev_flow.md ### Sync the itch.io page` — follow it there.
**Stop before Save and show the staged form for review** — never auto-submit a public page edit.
If nothing on the page changed, skip this step and note it in the ship report.

### 9. Land back on a clean `dev`
Keep `dev` == `master` so they don't drift (the release + any fixup commits are on `master`):
```
git checkout dev
git merge --ff-only master
git push origin dev
```
Then sync the roadmap banner to the shipped state (rewrite, don't append; `dev_flow.md ### The
Active-context banner`), commit + push that doc update to both branches.

### 10. Final verification
```
git rev-parse --short dev master origin/dev origin/master   # all four identical
git status --porcelain                                       # clean (dist/ is gitignored)
make check && make test                                      # green
```

## Ship report (paste at the end)

```
SHIP:
| Step | Outcome |
|---|---|
| sanitize        | <ran / skipped — why> |
| commit+push dev | <sha + subject> |
| version pick    | <vX.Y.Z — tier + one-line reason> |
| promote→master  | <ff / merge> |
| release+tag     | <release sha + tag> |
| build           | <gated pass; both zips> |
| push            | <master + tag pushed> |
| itch upload     | <both channels live at vX.Y.Z, per butler status> |
| store page      | <synced / skipped — page.yaml unchanged> |
| land on dev     | <dev==master==origin; tree clean; check+test green> |
```

## What NOT to do
- Don't run any step before the developer explicitly asks to ship.
- Don't auto-submit the itch store page (step 8 stops for review).
- Don't `git push` expecting tags to follow — push the tag explicitly.
- Don't `--allow-dirty` past a real test failure.
- Don't restate the branch-model rationale, the clean-bundle allowlist, or the semver policy here —
  point at their canonical homes (`dev_flow.md`, `conventions.md`).
